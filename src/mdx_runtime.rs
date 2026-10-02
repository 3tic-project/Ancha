//! Classic MDX DSP and task orchestration, matching pinned UVR demix semantics.
use crate::{
    report::{StemReport, Timings},
    runtime::residual_audio,
};
use ancha_audio::{
    Audio,
    decode::{DecodeOptions, decode},
    dsp::{Spectrum, SpectrumComplex, Stft},
};
use ancha_models::mdx::{Mdx, MdxConfig};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Debug, Clone)]
pub struct MdxOptions {
    pub input: PathBuf,
    pub model: PathBuf,
    pub output: PathBuf,
    pub decode: DecodeOptions,
    pub overlap: Option<f64>,
    pub denoise: bool,
    pub optimized: bool,
    pub batch_size: usize,
    /// Ungrouped convolutions as patch gather + GEMM (GPU default).
    pub conv_gemm: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct MdxReport {
    pub schema_version: u32,
    pub ancha_version: String,
    pub platform: String,
    pub backend: String,
    pub device: String,
    pub precision: String,
    pub dsp_backend: String,
    pub build_features: Vec<String>,
    pub cpu_thread_environment: std::collections::BTreeMap<String, String>,
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
    pub task: String,
    pub profile: String,
    pub effective_config: MdxConfig,
    pub chunk_samples: usize,
    pub step_samples: usize,
    pub overlap_fraction: Option<f64>,
    pub denoise: bool,
    pub optimized: bool,
    pub batch_size: usize,
    /// `gemm` (patch gather + GEMM) or `backend` (Burn conv2d).
    #[serde(default)]
    pub conv_strategy: String,
    pub graph_nodes: usize,
    pub folded_bn: usize,
    pub chunks: usize,
    pub skipped_unused_tail_chunks: usize,
    pub model_forwards: usize,
    pub stems: Vec<StemReport>,
    pub residual_reconstruction_max_abs: f32,
    pub timings: Timings,
    pub rtf: f64,
}

/// UVR packs [left.real,left.imag,right.real,right.imag] in NCHW.
pub fn pack(stft: &mut Stft, planes: &[Vec<f32>], bins: usize) -> Result<(Vec<f32>, usize)> {
    ensure!(planes.len() == 2, "MDX requires stereo");
    let mut packed = Vec::new();
    let mut frames = 0;
    for plane in planes {
        let spectrum = stft.forward(plane)?;
        ensure!(bins <= spectrum.bins, "MDX bins exceed FFT");
        frames = spectrum.frames;
        for ri in 0..2 {
            for f in 0..bins {
                for t in 0..frames {
                    let z = spectrum.data[t * spectrum.bins + f];
                    packed.push(if f < 3 {
                        0.
                    } else if ri == 0 {
                        z.re
                    } else {
                        z.im
                    });
                }
            }
        }
    }
    Ok((packed, frames))
}
pub fn unpack(
    stft: &mut Stft,
    packed: &[f32],
    bins: usize,
    frames: usize,
    length: usize,
) -> Result<Vec<Vec<f32>>> {
    ensure!(
        packed.len() == 4 * bins * frames && packed.iter().all(|v| v.is_finite()),
        "invalid MDX output"
    );
    let full = stft.n_fft / 2 + 1;
    let mut planes = Vec::new();
    for channel in 0..2 {
        let mut data = vec![SpectrumComplex::new(0., 0.); full * frames];
        for t in 0..frames {
            for f in 0..bins {
                data[t * full + f] = SpectrumComplex::new(
                    packed[(channel * 2 * bins + f) * frames + t],
                    packed[((channel * 2 + 1) * bins + f) * frames + t],
                );
            }
        }
        planes.push(stft.inverse(
            &Spectrum {
                frames,
                bins: full,
                data,
            },
            length,
            false,
        )?);
    }
    Ok(planes)
}
pub fn separate_mdx<B: Backend>(
    o: &MdxOptions,
    d: &B::Device,
    backend: &str,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(usize, usize),
) -> Result<MdxReport> {
    let total = Instant::now();
    ensure!(!o.output.exists(), "output already exists");
    ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
    ensure!(
        (1..=4).contains(&o.batch_size),
        "MDX batch size must be between 1 and 4"
    );
    if let Some(overlap) = o.overlap {
        ensure!(
            overlap.is_finite() && (0.0..=0.95).contains(&overlap),
            "MDX overlap must be between 0 and 0.95"
        );
    }
    let mut timings = Timings::default();
    let timer = Instant::now();
    let model = Mdx::<B>::load(&o.model, o.optimized, d)?;
    B::sync(d).map_err(|e| anyhow::anyhow!("device synchronization: {e:?}"))?;
    timings.model_load_seconds = timer.elapsed().as_secs_f64();
    let c = &model.config;
    let timer = Instant::now();
    let original = decode(&o.input, o.decode)?.stereo()?;
    timings.decode_seconds = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let input = ancha_audio::resample(original, c.sample_rate)?;
    timings.resample_seconds = timer.elapsed().as_secs_f64();
    ensure!(
        input.samples() <= o.decode.max_samples_per_channel,
        "resampled PCM exceeds host sample budget"
    );
    let mut hash = Sha256::new();
    for sample in input.planes.iter().flatten() {
        hash.update(sample.to_le_bytes());
    }
    let pcm = format!("{:x}", hash.finalize());
    let chunk = c.chunk_samples();
    let trim = c.n_fft / 2;
    let gen_size = chunk - c.n_fft;
    let pad = gen_size + trim - input.samples() % gen_size;
    let mixture_len = trim + input.samples() + pad;
    let step = o
        .overlap
        .map_or(gen_size, |v| ((1. - v) * chunk as f64) as usize);
    ensure!(step > 0, "zero MDX step");
    let all: Vec<_> = (0..mixture_len).step_by(step).collect();
    let starts: Vec<_> = all
        .iter()
        .copied()
        .filter(|&s| !o.optimized || s < trim + input.samples())
        .collect();
    let skipped = all.len() - starts.len();
    progress(0, starts.len());
    let mut stft = Stft::new(c.n_fft, c.hop)?;
    let mut result = vec![vec![0f32; input.samples()]; 2];
    let mut divisor = vec![0f32; input.samples()];
    let mut forwards = 0;
    let packed_size = 4 * c.bins * c.frames;
    for (batch_index, batch) in starts.chunks(o.batch_size).enumerate() {
        ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
        let timer = Instant::now();
        let mut packed = Vec::with_capacity(batch.len() * packed_size);
        for &start in batch {
            let actual = chunk.min(mixture_len - start);
            let planes: Vec<Vec<_>> = input
                .planes
                .iter()
                .map(|p| {
                    (0..chunk)
                        .map(|i| {
                            let pos = start + i;
                            if i < actual && pos >= trim && pos < trim + input.samples() {
                                p[pos - trim]
                            } else {
                                0.
                            }
                        })
                        .collect()
                })
                .collect();
            let (spectrum, frames) = pack(&mut stft, &planes, c.bins)?;
            ensure!(frames == c.frames, "MDX frame mismatch");
            packed.extend(spectrum);
        }
        timings.stft_seconds += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let x = Tensor::<B, 4>::from_data(
            TensorData::new(packed, [batch.len(), 4, c.bins, c.frames]),
            d,
        );
        let y = if o.denoise {
            let negative = model
                .graph
                .forward(x.clone().neg(), o.conv_gemm, cancelled)?;
            let positive = model.graph.forward(x, o.conv_gemm, cancelled)?;
            forwards += 2;
            (positive - negative) * 0.5
        } else {
            forwards += 1;
            model.graph.forward(x, o.conv_gemm, cancelled)?
        };
        ensure!(
            y.dims() == [batch.len(), 4, c.bins, c.frames],
            "MDX output shape mismatch"
        );
        let flat = y.into_data().to_vec::<f32>()?;
        let elapsed = timer.elapsed().as_secs_f64();
        timings.model_seconds += elapsed;
        timings.model_call_seconds.push(elapsed);
        for (local_batch, &start) in batch.iter().enumerate() {
            let timer = Instant::now();
            let audio = unpack(
                &mut stft,
                &flat[local_batch * packed_size..(local_batch + 1) * packed_size],
                c.bins,
                c.frames,
                chunk,
            )?;
            timings.istft_seconds += timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            let actual = chunk.min(mixture_len - start);
            let first = start.max(trim);
            let end = (start + actual).min(trim + input.samples());
            for pos in first..end {
                let local = pos - start;
                let weight = if o.overlap == Some(0.) {
                    1.
                } else {
                    (0.5 - 0.5 * (std::f64::consts::TAU * local as f64 / (actual - 1) as f64).cos())
                        as f32
                };
                let out = pos - trim;
                divisor[out] += weight;
                for channel in 0..2 {
                    result[channel][out] += audio[channel][local] * weight;
                }
            }
            timings.overlap_seconds += timer.elapsed().as_secs_f64();
            progress(batch_index * o.batch_size + local_batch + 1, starts.len());
        }
    }
    for (i, &denom) in divisor.iter().enumerate() {
        ensure!(
            denom > 0.,
            "MDX OLA uncovered sample {i}; overlap setting is invalid"
        );
        for plane in &mut result {
            plane[i] = plane[i] / denom * c.compensate;
        }
    }
    let predicted = Audio {
        sample_rate: c.sample_rate,
        planes: result,
    };
    let residual = residual_audio(&input, &predicted)?;
    let reconstruction = input
        .planes
        .iter()
        .flatten()
        .zip(predicted.planes.iter().flatten())
        .zip(residual.planes.iter().flatten())
        .map(|((&x, &p), &r)| (x - (p + r)).abs())
        .fold(0f32, f32::max);
    ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
    let parent = o
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".ancha-output-")
        .tempdir_in(parent)?;
    let timer = Instant::now();
    let mut stems = Vec::new();
    for (name, origin, audio) in [
        (&c.predicted, "predicted", &predicted),
        (&c.residual, "residual", &residual),
    ] {
        ancha_audio::write_wav(&stage.path().join(format!("{name}.wav")), audio)?;
        let peak = audio
            .planes
            .iter()
            .flatten()
            .map(|v| v.abs())
            .fold(0f32, f32::max);
        let rms = (audio
            .planes
            .iter()
            .flatten()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            / (2 * audio.samples()) as f64)
            .sqrt();
        stems.push(StemReport {
            name: name.clone(),
            origin: origin.into(),
            peak,
            rms,
        });
    }
    timings.write_seconds = timer.elapsed().as_secs_f64();
    timings.total_seconds = total.elapsed().as_secs_f64();
    let report = MdxReport {
        schema_version: 1,
        ancha_version: env!("CARGO_PKG_VERSION").into(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        backend: backend.into(),
        device: format!("{d:?}"),
        precision: "f32".into(),
        dsp_backend: "realfft-rubato".into(),
        build_features: [
            ("onnx", cfg!(feature = "onnx")),
            ("wgpu", cfg!(feature = "wgpu")),
            ("accelerate", cfg!(feature = "accelerate")),
            ("convert", cfg!(feature = "convert")),
            ("simd", cfg!(feature = "simd")),
            ("cpu-opt", cfg!(feature = "cpu-opt")),
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
        .filter_map(|key| std::env::var(key).ok().map(|value| (key.into(), value)))
        .collect(),
        model_id: c.model_id.clone(),
        weights_sha256: c.weights_sha256.clone(),
        input_path: o.input.display().to_string(),
        input_pcm_sha256: pcm,
        start_seconds: o.decode.start_seconds,
        duration_requested_seconds: o.decode.duration_seconds,
        audio_seconds: input.seconds(),
        sample_rate: c.sample_rate,
        samples_per_channel: input.samples(),
        channels: 2,
        task: c.task.clone(),
        profile: if o.denoise {
            "uvr-denoise"
        } else {
            "uvr-single-pass"
        }
        .into(),
        effective_config: c.clone(),
        chunk_samples: chunk,
        step_samples: step,
        overlap_fraction: o.overlap,
        denoise: o.denoise,
        optimized: o.optimized,
        batch_size: o.batch_size,
        conv_strategy: if o.conv_gemm { "gemm" } else { "backend" }.into(),
        graph_nodes: model.graph.nodes,
        folded_bn: model.graph.folded_bn,
        chunks: starts.len(),
        skipped_unused_tail_chunks: skipped,
        model_forwards: forwards,
        stems,
        residual_reconstruction_max_abs: reconstruction,
        rtf: timings.total_seconds / input.seconds(),
        timings,
    };
    std::fs::write(
        stage.path().join("run.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    ensure!(!o.output.exists(), "output appeared during inference");
    std::fs::rename(stage.path(), &o.output)?;
    Ok(report)
}
