//! TFC/TDF v3 task: native two-head spectra, centered STFT, UVR zero-padding and rectangular OLA.
use crate::{
    report::{StemReport, Timings},
    spectral,
};
use ancha_audio::{
    Audio,
    decode::{DecodeOptions, decode},
    dsp::Stft,
};
use ancha_models::mdx23c::{self, Config, Mdx23c};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Clone, Debug)]
pub struct Options {
    pub input: PathBuf,
    pub model: PathBuf,
    pub output: PathBuf,
    pub decode: DecodeOptions,
    /// Must equal hop*(frames-1), with frames surviving all encoder scales.
    pub chunk_samples: Option<usize>,
    pub overlap: Option<usize>,
    pub conv_gemm: bool,
    pub optimized: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    pub ancha_version: String,
    pub platform: String,
    pub backend: String,
    pub device: String,
    pub precision: String,
    pub dsp_backend: String,
    pub model_id: String,
    pub weights_sha256: String,
    pub input_path: String,
    pub input_pcm_sha256: String,
    pub start_seconds: f64,
    pub duration_requested_seconds: Option<f64>,
    pub audio_seconds: f64,
    pub sample_rate: u32,
    pub samples_per_channel: usize,
    pub channels: usize,
    pub profile: String,
    pub effective_config: Config,
    pub chunk_samples: usize,
    pub step_samples: usize,
    pub overlap: usize,
    pub conv_strategy: String,
    pub optimized: bool,
    pub tensor_count: usize,
    pub chunks: usize,
    pub model_forwards: usize,
    pub skipped_unused_tail_chunks: usize,
    pub build_features: Vec<String>,
    pub cpu_thread_environment: std::collections::BTreeMap<String, String>,
    pub stems: Vec<StemReport>,
    pub timings: Timings,
    pub rtf: f64,
}

/// Python's modulo semantics, full left/right zero context and constant overlap divisor.
#[derive(Clone, Debug)]
pub struct OverlapPlan {
    pub chunk: usize,
    pub step: usize,
    pub border: usize,
    pub pad: usize,
    pub length: usize,
    pub padded: usize,
    pub chunks: usize,
}
impl OverlapPlan {
    pub fn new(length: usize, chunk: usize, overlap: usize) -> Result<Self> {
        ensure!(
            length > 0 && chunk > 0 && (1..=16).contains(&overlap) && overlap <= chunk,
            "invalid MDX23C overlap plan"
        );
        let step = chunk / overlap;
        let (left, right) = (length % step, chunk % step);
        let remainder = if left >= right {
            left - right
        } else {
            step - (right - left)
        };
        let pad = step - remainder;
        let border = chunk - step;
        let padded = length
            .checked_add(pad)
            .and_then(|n| border.checked_mul(2).and_then(|b| n.checked_add(b)))
            .ok_or_else(|| anyhow::anyhow!("MDX23C padded length overflow"))?;
        ensure!(padded >= chunk, "insufficient MDX23C padding");
        let chunks = (padded - chunk) / step + 1;
        Ok(Self {
            chunk,
            step,
            border,
            pad,
            length,
            padded,
            chunks,
        })
    }
    pub fn intersects(&self, start: usize) -> bool {
        start < self.border + self.length && start + self.chunk > self.border
    }
    pub fn chunk(&self, input: &Audio, start: usize) -> Vec<Vec<f32>> {
        input
            .planes
            .iter()
            .map(|plane| {
                (0..self.chunk)
                    .map(|i| {
                        let pos = start + i;
                        if pos >= self.border && pos < self.border + self.length {
                            plane[pos - self.border]
                        } else {
                            0.
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

pub fn separate<B: Backend>(
    o: &Options,
    device: &B::Device,
    backend: &str,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(usize, usize),
) -> Result<Report> {
    let total = Instant::now();
    crate::device::install_guard();
    ensure!(!o.output.exists(), "output already exists");
    ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
    let manifest = mdx23c::read_manifest(&o.model)?;
    let mut config = manifest.config.clone();
    if let Some(chunk) = o.chunk_samples {
        ensure!(
            chunk <= 44100 * 60,
            "MDX23C context exceeds the 60-second limit"
        );
        ensure!(
            chunk.is_multiple_of(config.hop),
            "MDX23C chunk must equal hop*(frames-1)"
        );
        config.frames = chunk / config.hop + 1;
    }
    if let Some(overlap) = o.overlap {
        config.overlap = overlap;
    }
    config.validate()?;
    let mut timings = Timings::default();
    let timer = Instant::now();
    let original = decode(&o.input, o.decode)?.stereo()?;
    timings.decode_seconds = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let input = ancha_audio::resample(original, config.sample_rate)?;
    timings.resample_seconds = timer.elapsed().as_secs_f64();
    ensure!(
        input.samples() <= o.decode.max_samples_per_channel,
        "resampled PCM exceeds host sample budget"
    );
    let mut hash = Sha256::new();
    for x in input.planes.iter().flatten() {
        hash.update(x.to_le_bytes());
    }
    let input_pcm_sha256 = format!("{:x}", hash.finalize());
    let timer = Instant::now();
    let model = Mdx23c::<B>::load(&o.model, &manifest, device)?;
    B::sync(device).map_err(|e| anyhow::anyhow!("device synchronization: {e:?}"))?;
    crate::device::check()?;
    timings.model_load_seconds = timer.elapsed().as_secs_f64();
    let plan = OverlapPlan::new(input.samples(), config.chunk_samples(), config.overlap)?;
    let all_starts: Vec<usize> = (0..plan.chunks).map(|i| i * plan.step).collect();
    let starts: Vec<usize> = all_starts
        .iter()
        .copied()
        .filter(|&s| !o.optimized || plan.intersects(s))
        .collect();
    let mut output = vec![vec![vec![0.; input.samples()]; 2]; config.stems.len()];
    let mut stft = Stft::new(config.n_fft, config.hop)?;
    progress(0, starts.len());
    for (index, &start) in starts.iter().enumerate() {
        ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
        let chunk = plan.chunk(&input, start);
        let timer = Instant::now();
        let (packed, frames) = spectral::pack(&mut stft, &chunk, config.bins, 0)?;
        ensure!(frames == config.frames, "MDX23C STFT frame count mismatch");
        timings.stft_seconds += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let x =
            Tensor::<B, 4>::from_data(TensorData::new(packed, [1, 4, config.bins, frames]), device);
        let result: Vec<f32> = model
            .forward(
                x,
                mdx23c::Options {
                    conv_gemm: o.conv_gemm,
                    optimized: o.optimized,
                },
                cancelled,
            )?
            .into_data()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("MDX23C readback: {e:?}"))?;
        crate::device::check()?;
        ensure!(
            result.iter().all(|x| x.is_finite()),
            "non-finite MDX23C output"
        );
        let elapsed = timer.elapsed().as_secs_f64();
        timings.model_seconds += elapsed;
        timings.model_call_seconds.push(elapsed);
        let stride = 4 * config.bins * frames;
        for (stem, planes) in output.iter_mut().enumerate() {
            let timer = Instant::now();
            let waves = spectral::unpack(
                &mut stft,
                &result[stem * stride..(stem + 1) * stride],
                config.bins,
                frames,
                plan.chunk,
            )?;
            timings.istft_seconds += timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            let begin = start.max(plan.border);
            let end = (start + plan.chunk).min(plan.border + input.samples());
            for (sum, wave) in planes.iter_mut().zip(waves) {
                for pos in begin..end {
                    sum[pos - plan.border] += wave[pos - start];
                }
            }
            timings.overlap_seconds += timer.elapsed().as_secs_f64();
        }
        progress(index + 1, starts.len());
    }
    for x in output.iter_mut().flatten().flatten() {
        *x /= config.overlap as f32;
    }
    ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
    let parent = o
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".ancha-output-")
        .tempdir_in(parent)?;
    let timer = Instant::now();
    let mut stems = Vec::new();
    for (name, planes) in config.stems.iter().zip(output) {
        let audio = Audio {
            sample_rate: config.sample_rate,
            planes,
        };
        ancha_audio::write_wav(&stage.path().join(format!("{name}.wav")), &audio)?;
        let peak = audio
            .planes
            .iter()
            .flatten()
            .map(|s| s.abs())
            .fold(0f32, f32::max);
        let rms = (audio
            .planes
            .iter()
            .flatten()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
            / (2 * audio.samples()) as f64)
            .sqrt();
        stems.push(StemReport {
            name: name.clone(),
            origin: "predicted".into(),
            peak,
            rms,
        });
    }
    timings.write_seconds = timer.elapsed().as_secs_f64();
    timings.total_seconds = total.elapsed().as_secs_f64();
    let report = Report {
        schema_version: 1,
        ancha_version: env!("CARGO_PKG_VERSION").into(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        backend: backend.into(),
        device: B::name(device),
        precision: "f32".into(),
        dsp_backend: "cpu-realfft".into(),
        model_id: manifest.model_id,
        weights_sha256: manifest.weights_sha256,
        input_path: o.input.display().to_string(),
        input_pcm_sha256,
        start_seconds: o.decode.start_seconds,
        duration_requested_seconds: o.decode.duration_seconds,
        audio_seconds: input.seconds(),
        sample_rate: config.sample_rate,
        samples_per_channel: input.samples(),
        channels: 2,
        profile: if o.chunk_samples.is_some() || o.overlap.is_some() {
            "custom-context"
        } else {
            "native-context"
        }
        .into(),
        chunk_samples: plan.chunk,
        step_samples: plan.step,
        overlap: config.overlap,
        effective_config: config,
        conv_strategy: if o.conv_gemm { "gemm" } else { "backend" }.into(),
        optimized: o.optimized,
        tensor_count: model.tensor_count,
        chunks: starts.len(),
        model_forwards: starts.len(),
        skipped_unused_tail_chunks: all_starts.len() - starts.len(),
        build_features: [
            ("wgpu", cfg!(feature = "wgpu")),
            ("cuda", cfg!(feature = "cuda")),
            ("onnx", cfg!(feature = "onnx")),
            ("convert", cfg!(feature = "convert")),
            ("cpu-opt", cfg!(feature = "cpu-opt")),
            ("accelerate", cfg!(feature = "accelerate")),
        ]
        .into_iter()
        .filter(|(_, v)| *v)
        .map(|(k, _)| k.into())
        .collect(),
        cpu_thread_environment: [
            "RAYON_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MATMUL_NUM_THREADS",
        ]
        .into_iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| (k.into(), v)))
        .collect(),
        rtf: timings.total_seconds / input.seconds(),
        timings,
        stems,
    };
    std::fs::write(
        stage.path().join("run.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
    std::fs::rename(stage.path(), &o.output)?;
    Ok(report)
}
