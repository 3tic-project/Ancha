use ancha::{
    backend::{self, BackendKind},
    benchmark::attention_benchmark,
    runtime::SeparateOptions,
};
use ancha_audio::decode::DecodeOptions;
use ancha_models::{roformer::AttentionPlan, weights::read_manifest};
use anyhow::{Result, ensure};
use burn::{
    backend::NdArray,
    tensor::{Tensor, backend::Backend},
};
use burn_flex::Flex;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Parser)]
#[command(
    version,
    about = "Offline native Rust RoFormer and classic MDX audio separation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Check the selected backend using an actual tensor operation.
    Doctor {
        #[arg(long, value_enum, default_value = "cpu")]
        backend: BackendKind,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Inspect and validate a model package manifest.
    Inspect { model: PathBuf },
    /// Separate audio into float32 stems. Accepts a RoFormer package or classic MDX ONNX.
    Separate(SeparateArgs),
    /// Compare the original scalar reference with tiled GEMM attention.
    Bench {
        #[arg(long, default_value_t = 256)]
        tokens: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long, default_value_t = 128)]
        query_tile: usize,
        #[arg(long, default_value_t = 4)]
        group_tile: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Convert a local PyTorch checkpoint into a versioned F32 model package.
    #[cfg(feature = "convert")]
    Convert {
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "leap-xe-voc")]
        preset: String,
        #[arg(long)]
        config: Option<PathBuf>,
    },
}
#[derive(Args)]
struct SeparateArgs {
    input: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, short = 'o')]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: BackendKind,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value_t = 0.0)]
    start: f64,
    #[arg(long)]
    duration: Option<f64>,
    /// Explicit context override; changes the model quality profile.
    #[arg(long)]
    chunk_samples: Option<usize>,
    #[arg(long)]
    overlap: Option<usize>,
    /// Time/frequency attention query tile. Omit both tiles to size them from --max-score-mib.
    #[arg(long)]
    query_tile: Option<usize>,
    /// Attention group tile (bands or frames per score tensor).
    #[arg(long)]
    group_tile: Option<usize>,
    /// RoFormer projection layout: `auto` folds rows into one GEMM on CUDA, where it
    /// measured faster, and keeps batched projections on other backends.
    #[arg(long, value_enum, default_value = "auto")]
    linear_layout: LinearLayout,
    /// Same as `--linear-layout flattened`.
    #[arg(long, conflicts_with = "linear_layout")]
    flatten_linear: bool,
    /// RoFormer: host threads over independent attention groups (CPU default: all cores; 1 = off).
    #[arg(long)]
    host_threads: Option<usize>,
    /// Limit for concurrently materialized attention scores, also the automatic tile budget;
    /// not total GPU memory.
    #[arg(long, default_value_t = 512)]
    max_score_mib: usize,
    /// Host PCM limit at both source and output rates (1 hour at 44.1 kHz by default).
    #[arg(long, default_value_t = 3600)]
    max_seconds: usize,
    /// MDX only: run both x and -x as in UVR denoise mode.
    #[arg(long)]
    mdx_denoise: bool,
    /// MDX only: fractional overlap (0..=0.95); omitted uses UVR default step.
    #[arg(long)]
    mdx_overlap: Option<f64>,
    /// Disable MDX BN precomputation/folding and unused-tail pruning for comparison.
    #[arg(long)]
    mdx_no_optimize: bool,
    /// MDX only: fixed-shape chunks per forward (1..=4). Increases device memory.
    #[arg(long, default_value_t = 1)]
    mdx_batch_size: usize,
    /// MDX / HyperACE convolution: `auto` uses patch-gather GEMM for MDX on GPUs and for the
    /// HyperACE SegmModel on CUDA, Burn conv2d otherwise.
    #[arg(long, value_enum, default_value = "auto")]
    conv_strategy: ConvStrategy,
}
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ConvStrategy {
    Auto,
    Gemm,
    Backend,
}
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LinearLayout {
    Auto,
    Batched,
    Flattened,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ancha: {error:#}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect { model } => {
            if model.extension().is_some_and(|v| v == "onnx") {
                #[cfg(feature = "onnx")]
                {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&ancha_models::mdx::MdxConfig::identify(
                            &model
                        )?)?
                    );
                    return Ok(());
                }
                #[cfg(not(feature = "onnx"))]
                anyhow::bail!("ONNX support is not built; use --features onnx");
            }
            println!("{}", serde_json::to_string_pretty(&read_manifest(&model)?)?)
        }
        Command::Doctor { backend, device } => match backend {
            BackendKind::Cpu => doctor::<Flex>(&Default::default())?,
            BackendKind::Ndarray => doctor::<NdArray<f32>>(&Default::default())?,
            BackendKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                doctor::<burn::backend::Wgpu>(&burn::backend::wgpu::WgpuDevice::DiscreteGpu(
                    device,
                ))?;
                #[cfg(not(feature = "wgpu"))]
                {
                    let _ = device;
                    anyhow::bail!(
                        "WGPU support is not built; use cargo build --release --features wgpu"
                    );
                }
            }
            BackendKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    log_cuda_device(device, "doctor")?;
                    doctor::<burn::backend::Cuda>(&backend::cuda_device(device, "doctor")?)?;
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = device;
                    anyhow::bail!(
                        "CUDA support is not built; use cargo build --release --features cuda"
                    );
                }
            }
        },
        Command::Separate(args) => {
            ensure!(args.max_seconds > 0, "max-seconds must be positive");
            let cancelled = Arc::new(AtomicBool::new(false));
            let flag = cancelled.clone();
            ctrlc::set_handler(move || flag.store(true, std::sync::atomic::Ordering::Relaxed))?;
            if args.model.extension().is_some_and(|v| v == "onnx") {
                #[cfg(feature = "onnx")]
                return run_mdx(args, &cancelled);
                #[cfg(not(feature = "onnx"))]
                anyhow::bail!("ONNX support is not built; use --features onnx");
            }
            ensure!(
                !args.mdx_denoise
                    && args.mdx_overlap.is_none()
                    && !args.mdx_no_optimize
                    && args.mdx_batch_size == 1,
                "MDX flags require an ONNX model"
            );
            let kind = args.backend;
            let mut attention = kind.attention_plan();
            attention.query_tile = args.query_tile;
            attention.group_tile = args.group_tile;
            attention.score_budget = args
                .max_score_mib
                .checked_mul(1 << 20)
                .ok_or_else(|| anyhow::anyhow!("score limit overflow"))?;
            match (args.flatten_linear, args.linear_layout) {
                (true, _) | (_, LinearLayout::Flattened) => attention.batched_linear = false,
                (_, LinearLayout::Batched) => attention.batched_linear = true,
                (_, LinearLayout::Auto) => {}
            }
            if let Some(threads) = args.host_threads {
                attention.host_threads = threads;
            }
            match args.conv_strategy {
                ConvStrategy::Auto => {}
                ConvStrategy::Gemm => attention.conv_gemm = true,
                ConvStrategy::Backend => attention.conv_gemm = false,
            }
            let options = SeparateOptions {
                input: args.input,
                model: args.model,
                output: args.output,
                decode: DecodeOptions {
                    start_seconds: args.start,
                    duration_seconds: args.duration,
                    max_samples_per_channel: args
                        .max_seconds
                        .checked_mul(44_100)
                        .ok_or_else(|| anyhow::anyhow!("host sample limit overflow"))?,
                },
                chunk_samples: args.chunk_samples,
                overlap: args.overlap,
                attention,
                max_score_mib: args.max_score_mib,
            };
            #[cfg(feature = "cuda")]
            if kind == BackendKind::Cuda {
                log_cuda_device(args.device, &backend::cache_scope(&options.model))?;
            }
            let progress = |n, total| eprintln!("separated chunk {n}/{total}");
            let report = backend::separate(kind, args.device, &options, &cancelled, progress)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Bench {
            tokens,
            iterations,
            query_tile,
            group_tile,
            output,
        } => {
            let json = serde_json::to_string_pretty(&attention_benchmark(
                tokens,
                iterations,
                AttentionPlan {
                    query_tile: Some(query_tile),
                    group_tile: Some(group_tile),
                    batched_linear: false,
                    ..AttentionPlan::default()
                },
            )?)?;
            if let Some(path) = output {
                ensure!(!path.exists(), "benchmark report already exists");
                std::fs::write(path, &json)?;
            }
            println!("{json}");
        }
        #[cfg(feature = "convert")]
        Command::Convert {
            checkpoint,
            output,
            preset,
            config,
        } => {
            use ancha_models::config::ModelConfig;
            let (config, url, license) = if let Some(path) = config {
                (
                    serde_json::from_slice::<ModelConfig>(&std::fs::read(path)?)?,
                    if preset == "deux" { "https://huggingface.co/becruily/mel-band-roformer-deux/tree/2da74427d682a3df47a774378fc24d7a1a0cdaad" } else { "user-supplied checkpoint" }.into(),
                    if preset == "deux" { "CC-BY-NC-4.0" } else { "see checkpoint author" }.into(),
                )
            } else {
                match preset.as_str() {
                "leap-xe-voc" | "leap-xe-inst" => (ModelConfig::leap_xe(preset=="leap-xe-inst"),
                    "https://huggingface.co/pcunwa/BS-Roformer-Leap/tree/4e47d6662ae82eaa8b4ac4329fe66099a843b48e".into(),"not specified by checkpoint author".into()),
                "hyperace-v2-voc" | "hyperace-v2-inst" => (ModelConfig::hyperace_v2(preset=="hyperace-v2-inst"),
                    "https://huggingface.co/pcunwa/BS-Roformer-HyperACE/tree/5b1f8283125d5e4a3614d0e3635a636e09c84059".into(),"not specified by checkpoint author".into()),
                _ => anyhow::bail!("unknown preset; use leap-xe-voc / leap-xe-inst / hyperace-v2-voc / hyperace-v2-inst, or supply --config"),
            }
            };
            let m = ancha_models::convert::convert_checkpoint(
                &checkpoint,
                &output,
                &preset,
                config,
                url,
                license,
            )?;
            // CPU construction checks every key/shape before declaring conversion valid.
            let check = ancha_models::network::Roformer::<NdArray<f32>>::load(
                &output,
                &m,
                &Default::default(),
            );
            if let Err(error) = check {
                std::fs::remove_dir_all(&output)?;
                return Err(
                    error.context("converted package failed strict architecture validation")
                );
            }
            println!("{}", serde_json::to_string_pretty(&m)?);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn log_cuda_device(index: usize, scope: &str) -> Result<()> {
    let info = ancha::cuda::prepare(index, scope)?;
    eprintln!(
        "CUDA device {index}: {} (sm_{}, {:.1} GiB)",
        info.name,
        info.compute_capability,
        info.total_memory_bytes as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

fn doctor<B: Backend>(device: &B::Device) -> Result<()> {
    let x = Tensor::<B, 2>::from_floats([[1., 2.], [3., 4.]], device);
    let actual: Vec<f32> = x
        .clone()
        .matmul(x)
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("backend tensor readback: {e:?}"))?;
    ensure!(actual == [7., 10., 15., 22.], "backend matmul failed");
    println!(
        "{}",
        serde_json::json!({"status":"passed","backend":B::name(device),"precision":"f32","platform":format!("{}-{}",std::env::consts::OS,std::env::consts::ARCH),"scope":"2x2 matmul and readback; not full-model certification"})
    );
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_mdx(args: SeparateArgs, cancelled: &AtomicBool) -> Result<()> {
    use ancha::mdx_runtime::MdxOptions;
    ensure!(
        args.chunk_samples.is_none()
            && args.overlap.is_none()
            && !args.flatten_linear
            && args.linear_layout == LinearLayout::Auto
            && args.host_threads.is_none()
            && args.query_tile.is_none()
            && args.group_tile.is_none(),
        "RoFormer context/attention flags do not apply to MDX; use --mdx-overlap"
    );
    let options = MdxOptions {
        input: args.input,
        model: args.model,
        output: args.output,
        decode: DecodeOptions {
            start_seconds: args.start,
            duration_seconds: args.duration,
            max_samples_per_channel: args
                .max_seconds
                .checked_mul(44100)
                .ok_or_else(|| anyhow::anyhow!("host limit overflow"))?,
        },
        overlap: args.mdx_overlap,
        denoise: args.mdx_denoise,
        optimized: !args.mdx_no_optimize,
        batch_size: args.mdx_batch_size,
        conv_gemm: match args.conv_strategy {
            ConvStrategy::Auto => args.backend.mdx_conv_gemm(),
            ConvStrategy::Gemm => true,
            ConvStrategy::Backend => false,
        },
    };
    #[cfg(feature = "cuda")]
    if args.backend == BackendKind::Cuda {
        log_cuda_device(args.device, &backend::cache_scope(&options.model))?;
    }
    let progress = |n, total| eprintln!("separated MDX chunk {n}/{total}");
    let report = backend::separate_mdx(args.backend, args.device, &options, cancelled, progress)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
