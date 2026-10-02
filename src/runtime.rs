//! Separation task orchestration.
use crate::report::{RunReport, StemReport, Timings};
use ancha_audio::{
    Audio,
    chunk::{Accumulator, ChunkPlan},
    decode::{DecodeOptions, decode},
    dsp::{Spectrum, Stft, reflect_index},
};
use ancha_models::{network::Roformer, roformer::AttentionPlan, weights::read_manifest};
use anyhow::{Context, Result, ensure};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Debug, Clone)]
pub struct SeparateOptions {
    pub input: PathBuf,
    pub model: PathBuf,
    pub output: PathBuf,
    pub decode: DecodeOptions,
    pub chunk_samples: Option<usize>,
    pub overlap: Option<usize>,
    pub attention: AttentionPlan,
    pub max_score_mib: usize,
}

/// Blocking inference task. Cancellation is checked between chunks, layers and heads.
/// Progress callbacks receive `(completed_chunks, total_chunks)`.
pub fn separate<B: Backend>(
    options: &SeparateOptions,
    device: &B::Device,
    backend: &str,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(usize, usize),
) -> Result<RunReport> {
    let total_start = Instant::now();
    crate::device::install_guard();
    ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
    ensure!(
        !options.output.exists(),
        "output already exists: {}; choose a new directory",
        options.output.display()
    );
    options.attention.validate()?;
    let mut manifest = read_manifest(&options.model)?;
    let profile = if options.chunk_samples.is_some() || options.overlap.is_some() {
        "custom-context"
    } else {
        "native-context"
    };
    if let Some(samples) = options.chunk_samples {
        manifest.config.chunk_samples = samples;
    }
    if let Some(overlap) = options.overlap {
        manifest.config.overlap = overlap;
    }
    manifest.validate()?;
    let c = &manifest.config;
    let frames = c.chunk_samples / c.hop + 1;
    let attention = options.attention;
    let bands = c.bands.len();
    let (group, query, score_bytes) = attention.worker_tiles(bands, frames, c.heads);
    let time_tiles = (group, query);
    let (group, query, frequency_score_bytes) = attention.worker_tiles(frames, bands, c.heads);
    let frequency_tiles = (group, query);
    let concurrent = (score_bytes * attention.per_worker(bands).0)
        .max(frequency_score_bytes * attention.per_worker(frames).0);
    ensure!(
        concurrent <= options.max_score_mib.saturating_mul(1024 * 1024),
        "concurrent attention scores need {:.1} MiB, above --max-score-mib; reduce tiles or --host-threads",
        concurrent as f64 / 1048576.0
    );
    let mut timings = Timings::default();
    let timer = Instant::now();
    let original = decode(&options.input, options.decode)?.stereo()?;
    timings.decode_seconds = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let input = ancha_audio::resample(original, c.sample_rate)?;
    timings.resample_seconds = timer.elapsed().as_secs_f64();
    ensure!(
        input.samples() <= options.decode.max_samples_per_channel,
        "resampled PCM exceeds host sample budget"
    );
    let mut hash = Sha256::new();
    for sample in input.planes.iter().flatten() {
        hash.update(sample.to_le_bytes());
    }
    let pcm_digest = format!("{:x}", hash.finalize());
    let timer = Instant::now();
    let model = Roformer::<B>::load(&options.model, &manifest, device)?;
    B::sync(device).map_err(|e| anyhow::anyhow!("device synchronization: {e:?}"))?;
    crate::device::check()?;
    timings.model_load_seconds = timer.elapsed().as_secs_f64();
    let plan = ChunkPlan::new(c.chunk_samples, c.overlap)?;
    let border = c.chunk_samples - plan.step;
    let border = if input.samples() > 2 * border {
        border
    } else {
        0
    };
    let length = input.samples() + 2 * border;
    let starts = plan.starts(length);
    progress(0, starts.len());
    let mut stft = Stft::new(c.n_fft, c.hop)?;
    let mut accumulators = (0..c.stems.len())
        .map(|_| Accumulator::new(2, length))
        .collect::<Result<Vec<_>>>()?;
    for (index, &start) in starts.iter().enumerate() {
        ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
        let remaining = (length - start).min(c.chunk_samples);
        let chunk: Vec<Vec<f32>> = input
            .planes
            .iter()
            .map(|plane| {
                (0..c.chunk_samples)
                    .map(|i| {
                        if i >= remaining && remaining <= c.chunk_samples / 2 {
                            return 0.0;
                        }
                        let local = if i < remaining {
                            i
                        } else {
                            reflect_index(i as isize, remaining)
                        };
                        let source = reflect_index(
                            (start + local) as isize - border as isize,
                            input.samples(),
                        );
                        plane[source]
                    })
                    .collect()
            })
            .collect();
        let timer = Instant::now();
        let spectra = chunk
            .iter()
            .map(|p| stft.forward(p))
            .collect::<Result<Vec<_>>>()?;
        let bins = c.n_fft / 2 + 1;
        let frames = spectra[0].frames;
        let rows = bins * 2;
        // (frame, bin·2+channel, re/im), the band-split feature order.
        let mut packed = vec![0f32; frames * rows * 2];
        for (channel, spectrum) in spectra.iter().enumerate() {
            for t in 0..frames {
                for f in 0..bins {
                    let v = spectrum.data[t * bins + f];
                    let base = (t * rows + f * 2 + channel) * 2;
                    packed[base] = v.re;
                    packed[base + 1] = v.im;
                }
            }
        }
        timings.stft_seconds += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let spectrum =
            Tensor::<B, 2>::from_data(TensorData::new(packed, [frames, rows * 2]), device);
        let masks: Vec<f32> = model
            .forward(spectrum, options.attention, cancelled)?
            .into_data()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("read model output: {e:?}"))?;
        crate::device::check()?;
        ensure!(
            masks.iter().all(|v| v.is_finite()),
            "non-finite model output"
        );
        let elapsed = timer.elapsed().as_secs_f64();
        timings.model_seconds += elapsed;
        timings.model_call_seconds.push(elapsed);
        let timer = Instant::now();
        let mut outputs = Vec::new();
        for stem in 0..c.stems.len() {
            let mut planes = Vec::new();
            for (channel, source) in spectra.iter().enumerate() {
                let mut data = Vec::with_capacity(frames * bins);
                for t in 0..frames {
                    for f in 0..bins {
                        // Masks are [stem, row, re/im, frame].
                        let base = ((stem * rows + f * 2 + channel) * 2) * frames + t;
                        let mask = real_complex(masks[base], masks[base + frames]);
                        data.push(source.data[t * bins + f] * mask);
                    }
                }
                planes.push(stft.inverse(
                    &Spectrum { frames, bins, data },
                    c.chunk_samples,
                    c.zero_dc,
                )?);
            }
            outputs.push(planes);
        }
        timings.istft_seconds += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        for (acc, planes) in accumulators.iter_mut().zip(&outputs) {
            acc.add(plan, start, planes)?;
        }
        timings.overlap_seconds += timer.elapsed().as_secs_f64();
        progress(index + 1, starts.len());
    }
    let mut stems = Vec::<(String, String, Audio)>::new();
    for (name, acc) in c.stems.iter().zip(accumulators) {
        let planes = acc
            .finish()?
            .into_iter()
            .map(|p| p[border..border + input.samples()].to_vec())
            .collect();
        stems.push((
            name.clone(),
            "predicted".into(),
            Audio {
                sample_rate: c.sample_rate,
                planes,
            },
        ));
    }
    let reconstruction = if stems.len() == 1 {
        let residual = residual_audio(&input, &stems[0].2)?;
        let error = reconstruction_error(&input, &stems[0].2, &residual);
        let name = if stems[0].0 == "vocals" {
            "instrumental"
        } else {
            "vocals"
        };
        stems.push((name.into(), "residual".into(), residual));
        Some(error)
    } else {
        None
    };
    ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
    let parent = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".ancha-output-")
        .tempdir_in(parent)?;
    let timer = Instant::now();
    let mut reports = Vec::new();
    for (name, origin, audio) in &stems {
        ancha_audio::write_wav(&stage.path().join(format!("{name}.wav")), audio)?;
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
        reports.push(StemReport {
            name: name.clone(),
            origin: origin.clone(),
            peak,
            rms,
        });
    }
    timings.write_seconds = timer.elapsed().as_secs_f64();
    timings.total_seconds = total_start.elapsed().as_secs_f64();
    let report = RunReport {
        schema_version: 1,
        ancha_version: env!("CARGO_PKG_VERSION").into(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        backend: backend.into(),
        device: B::name(device),
        precision: "f32".into(),
        dsp_backend: "cpu".into(),
        model_id: manifest.model_id.clone(),
        weights_sha256: manifest.weights_sha256.clone(),
        input_path: options.input.display().to_string(),
        input_pcm_sha256: pcm_digest,
        start_seconds: options.decode.start_seconds,
        duration_requested_seconds: options.decode.duration_seconds,
        audio_seconds: input.seconds(),
        sample_rate: c.sample_rate,
        samples_per_channel: input.samples(),
        channels: 2,
        tensor_count: model.tensor_count,
        chunks: starts.len(),
        profile: profile.into(),
        effective_config: c.clone(),
        query_tile: time_tiles.1,
        group_tile: time_tiles.0,
        frequency_query_tile: frequency_tiles.1,
        frequency_group_tile: frequency_tiles.0,
        attention_tiling: if attention.query_tile.is_none() || attention.group_tile.is_none() {
            "auto"
        } else {
            "manual"
        }
        .into(),
        host_threads: attention.host_threads,
        linear_layout: if options.attention.batched_linear {
            "batched"
        } else {
            "flattened"
        }
        .into(),
        build_features: [
            ("wgpu", cfg!(feature = "wgpu")),
            ("cuda", cfg!(feature = "cuda")),
            ("accelerate", cfg!(feature = "accelerate")),
            ("convert", cfg!(feature = "convert")),
            ("onnx", cfg!(feature = "onnx")),
            ("simd", cfg!(feature = "simd")),
            ("cpu-opt", cfg!(feature = "cpu-opt")),
        ]
        .into_iter()
        .filter(|(_, enabled)| *enabled)
        .map(|(name, _)| name.into())
        .collect(),
        cpu_thread_environment: [
            "RAYON_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MATMUL_NUM_THREADS",
        ]
        .into_iter()
        .filter_map(|key| std::env::var(key).ok().map(|value| (key.into(), value)))
        .collect(),
        estimated_time_attention_score_bytes: score_bytes,
        estimated_frequency_attention_score_bytes: frequency_score_bytes,
        residual_reconstruction_max_abs: reconstruction,
        stems: reports,
        rtf: timings.total_seconds / input.seconds(),
        timings,
    };
    std::fs::write(
        stage.path().join("run.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    ensure!(!options.output.exists(), "output appeared during inference");
    std::fs::rename(stage.path(), &options.output).context("publish separation outputs")?;
    Ok(report)
}

fn real_complex(re: f32, im: f32) -> ancha_audio::dsp::SpectrumComplex {
    ancha_audio::dsp::SpectrumComplex::new(re, im)
}

pub fn residual_audio(input: &Audio, prediction: &Audio) -> Result<Audio> {
    input.validate()?;
    prediction.validate()?;
    ensure!(
        input.sample_rate == prediction.sample_rate
            && input.planes.len() == prediction.planes.len()
            && input.samples() == prediction.samples(),
        "residual audio shape mismatch"
    );
    Ok(Audio {
        sample_rate: input.sample_rate,
        planes: input
            .planes
            .iter()
            .zip(&prediction.planes)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x - y).collect())
            .collect(),
    })
}

fn reconstruction_error(input: &Audio, prediction: &Audio, residual: &Audio) -> f32 {
    input
        .planes
        .iter()
        .flatten()
        .zip(prediction.planes.iter().flatten())
        .zip(residual.planes.iter().flatten())
        .map(|((x, y), r)| (x - (y + r)).abs())
        .fold(0f32, f32::max)
}
