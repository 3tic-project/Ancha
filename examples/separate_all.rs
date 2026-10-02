//! Separate one song with every supported model on one backend and record performance.
//!
//! ```bash
//! cargo run --release --features cuda --example separate_all -- --backend cuda
//! cargo run --release --example separate_all -- --backend wgpu --duration 30 --only deux
//! ```
//!
//! Each model writes its stems and run.json to `<output>/<model>/`. `<output>/summary.json`
//! holds host / device information and per-model timings, RTF and per-chunk call times; it is
//! rewritten after every model, so an interrupted run keeps its finished rows.
use ancha::{
    backend::{self, BackendKind},
    mdx_runtime::MdxOptions,
    runtime::SeparateOptions,
};
use ancha_audio::decode::DecodeOptions;
use anyhow::{Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Package directory or ONNX file under `--models`, and what it predicts.
const MODELS: [(&str, &str); 8] = [
    (
        "leap-xe-voc",
        "BS-RoFormer Leap Xe: vocals, residual instrumental",
    ),
    ("deux", "Mel-Band RoFormer Deux: vocals and instrumental"),
    (
        "hyperace-v2-voc",
        "HyperACE v2: vocals, residual instrumental",
    ),
    (
        "hyperace-v2-inst",
        "HyperACE v2: instrumental, residual vocals",
    ),
    (
        "UVR_MDXNET_9482.onnx",
        "MDX 9482: all vocals, residual instrumental",
    ),
    (
        "UVR_MDXNET_KARA.onnx",
        "MDX KARA: lead vocals, residual karaoke mix",
    ),
    (
        "UVR_MDXNET_KARA_2.onnx",
        "MDX KARA 2: karaoke mix, residual lead vocals",
    ),
    (
        "UVR-MDX-NET-Inst_HQ_2.onnx",
        "MDX Inst HQ 2: instrumental, residual all vocals",
    ),
];

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    backend: BackendKind,
    /// GPU ordinal for wgpu / cuda.
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value = "NO_TRACK/test_file/ReoNa - Amore.mp3")]
    input: PathBuf,
    #[arg(long, default_value = "NO_TRACK/models")]
    models: PathBuf,
    /// New directory; default NO_TRACK/runs/examples/separate-all-<backend>-<unix time>.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Comma-separated subset of the model names above.
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
    #[arg(long, default_value_t = 0.0)]
    start: f64,
    #[arg(long)]
    duration: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kind = args.backend;
    ensure!(
        kind.is_built(),
        "{} is not built into this binary; rebuild with --features {}",
        kind.label(),
        kind.label()
    );
    for name in &args.only {
        ensure!(
            MODELS.iter().any(|(model, _)| model == name),
            "unknown model {name}"
        );
    }
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let root = args.output.clone().unwrap_or_else(|| {
        format!(
            "NO_TRACK/runs/examples/separate-all-{}-{stamp}",
            kind.label()
        )
        .into()
    });
    ensure!(!root.exists(), "output already exists: {}", root.display());
    std::fs::create_dir_all(&root)?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = cancelled.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed))?;
    // One PTX cache partition for this mixed workload; the first prepare in a process wins.
    #[cfg(feature = "cuda")]
    if kind == BackendKind::Cuda {
        ancha::cuda::prepare(args.device, "separate_all")?;
    }
    let decode = DecodeOptions {
        start_seconds: args.start,
        duration_seconds: args.duration,
        ..DecodeOptions::default()
    };
    let mut summary = json!({
        "scope": "one process, models run serially on one backend with their native contexts and \
                  the backend defaults; timings from each run.json, wall time measured around the call",
        "ancha_version": env!("CARGO_PKG_VERSION"),
        "build_features": build_features(),
        "backend": kind,
        "device": device_info(kind, args.device),
        "host": host_info(),
        "input": {
            "path": args.input,
            "bytes": std::fs::metadata(&args.input).map(|m| m.len()).ok(),
            "start_seconds": args.start,
            "duration_seconds": args.duration,
        },
        "models": [],
    });
    let started = Instant::now();
    let mut failed = 0;
    for (name, description) in MODELS {
        if !args.only.is_empty() && !args.only.iter().any(|n| n == name) {
            continue;
        }
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let model = args.models.join(name);
        eprintln!("== {name}: {description}");
        let entry = if !model.exists() {
            json!({"name": name, "description": description, "status": "skipped",
                   "reason": format!("missing {}", model.display())})
        } else {
            let output = root.join(name.trim_end_matches(".onnx"));
            let timer = Instant::now();
            let result = run(
                kind,
                args.device,
                &model,
                &args.input,
                &output,
                decode,
                &cancelled,
            );
            let wall = timer.elapsed().as_secs_f64();
            match result {
                Ok(report) => record(name, description, wall, &report, kind, args.device),
                Err(error) => {
                    failed += 1;
                    json!({"name": name, "description": description, "status": "failed",
                           "error": format!("{error:#}"), "wall_seconds": wall})
                }
            }
        };
        eprintln!("   {}", row(&entry));
        summary["models"]
            .as_array_mut()
            .expect("models array")
            .push(entry);
        summary["wall_seconds"] = json!(started.elapsed().as_secs_f64());
        summary["process_peak_rss_mib"] = json!(peak_rss_mib());
        std::fs::write(
            root.join("summary.json"),
            serde_json::to_vec_pretty(&summary)?,
        )?;
    }
    println!(
        "| model | audio | chunks | load | model | first / steady call | total | RTF |\n|---|---:|---:|---:|---:|---:|---:|---:|"
    );
    for entry in summary["models"].as_array().expect("models array") {
        println!("{}", row(entry));
    }
    println!(
        "wall {:.1} s; summary: {}",
        started.elapsed().as_secs_f64(),
        root.join("summary.json").display()
    );
    ensure!(failed == 0, "{failed} model(s) failed; see summary.json");
    ensure!(!cancelled.load(Ordering::Relaxed), "cancelled");
    Ok(())
}

/// Run one model with the backend defaults and return its run.json as JSON.
fn run(
    kind: BackendKind,
    device: usize,
    model: &Path,
    input: &Path,
    output: &Path,
    decode: DecodeOptions,
    cancelled: &AtomicBool,
) -> Result<Value> {
    let progress = |done, total| eprintln!("   chunk {done}/{total}");
    if model.extension().is_some_and(|e| e == "onnx") {
        let options = MdxOptions {
            input: input.into(),
            model: model.into(),
            output: output.into(),
            decode,
            overlap: None,
            denoise: false,
            optimized: true,
            batch_size: 1,
            conv_gemm: kind.mdx_conv_gemm(),
        };
        let report = backend::separate_mdx(kind, device, &options, cancelled, progress)?;
        Ok(serde_json::to_value(report)?)
    } else {
        let attention = kind.attention_plan();
        let options = SeparateOptions {
            input: input.into(),
            model: model.into(),
            output: output.into(),
            decode,
            chunk_samples: None,
            overlap: None,
            max_score_mib: attention.score_budget >> 20,
            attention,
        };
        let report = backend::separate(kind, device, &options, cancelled, progress)?;
        Ok(serde_json::to_value(report)?)
    }
}

fn record(
    name: &str,
    description: &str,
    wall: f64,
    report: &Value,
    kind: BackendKind,
    device: usize,
) -> Value {
    let timings = &report["timings"];
    let calls: Vec<f64> = timings["model_call_seconds"]
        .as_array()
        .map(|c| c.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    let mut rest = calls.get(1..).unwrap_or_default().to_vec();
    rest.sort_by(f64::total_cmp);
    let audio = report["audio_seconds"].as_f64().unwrap_or(f64::NAN);
    let mdx = report.get("model_forwards").is_some();
    let config = &report["effective_config"];
    let tiles = [
        "group_tile",
        "query_tile",
        "frequency_group_tile",
        "frequency_query_tile",
    ]
    .map(|k| report.get(k).cloned().unwrap_or(Value::Null));
    json!({
        "name": name,
        "description": description,
        "status": "ok",
        "model_id": report["model_id"],
        "family": if mdx { json!("classic-mdx") } else { config["family"].clone() },
        "profile": report["profile"],
        "audio_seconds": audio,
        "chunks": report["chunks"],
        "model_forwards": report.get("model_forwards"),
        "context": if mdx {
            json!({"chunk_samples": report["chunk_samples"], "step_samples": report["step_samples"]})
        } else {
            json!({"chunk_samples": config["chunk_samples"], "overlap": config["overlap"]})
        },
        "settings": {
            "linear_layout": report.get("linear_layout"),
            "conv_strategy": report.get("conv_strategy"),
            "host_threads": report.get("host_threads"),
            "tiles": tiles,
        },
        "timings": timings,
        "first_call_seconds": calls.first(),
        "steady_call_seconds": rest.get(rest.len() / 2),
        "slowest_steady_call_seconds": rest.last(),
        "model_rtf": timings["model_seconds"].as_f64().map(|m| m / audio),
        "rtf": report["rtf"],
        "wall_seconds": wall,
        "device_memory_in_use_mib": device_memory_mib(kind, device),
        "stems": report["stems"],
    })
}

/// One Markdown table row (also used for progress lines).
fn row(e: &Value) -> String {
    if e["status"] != "ok" {
        return format!(
            "| {} | {} | | | | | | |",
            e["name"].as_str().unwrap_or("?"),
            e["status"]
        );
    }
    let t = &e["timings"];
    let f = |v: &Value| v.as_f64().map_or("-".into(), |x| format!("{x:.2}"));
    format!(
        "| {} | {} s | {} | {} s | {} s | {} / {} s | {} s | {} |",
        e["name"].as_str().unwrap_or("?"),
        f(&e["audio_seconds"]),
        e["chunks"],
        f(&t["model_load_seconds"]),
        f(&t["model_seconds"]),
        f(&e["first_call_seconds"]),
        f(&e["steady_call_seconds"]),
        f(&t["total_seconds"]),
        e["rtf"].as_f64().map_or("-".into(), |x| format!("{x:.3}")),
    )
}

fn build_features() -> Vec<&'static str> {
    [
        ("wgpu", cfg!(feature = "wgpu")),
        ("cuda", cfg!(feature = "cuda")),
        ("onnx", cfg!(feature = "onnx")),
        ("cpu-opt", cfg!(feature = "cpu-opt")),
        ("accelerate", cfg!(feature = "accelerate")),
        ("convert", cfg!(feature = "convert")),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect()
}

fn device_info(kind: BackendKind, device: usize) -> Value {
    match kind {
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => match ancha::cuda::probe(device) {
            Ok(info) => json!({
                "ordinal": device,
                "name": info.name,
                "compute_capability": info.compute_capability,
                "memory_gib": info.total_memory_bytes as f64 / (1u64 << 30) as f64,
            }),
            Err(error) => json!({"ordinal": device, "error": format!("{error:#}")}),
        },
        BackendKind::Wgpu => json!({"ordinal": device, "selection": "WgpuDevice::DiscreteGpu"}),
        _ => json!({"host_threads": kind.attention_plan().host_threads}),
    }
}

fn device_memory_mib(kind: BackendKind, device: usize) -> Option<f64> {
    #[cfg(feature = "cuda")]
    if kind == BackendKind::Cuda {
        return ancha::cuda::memory_info(device)
            .ok()
            .map(|(free, total)| (total - free) as f64 / (1u64 << 20) as f64);
    }
    let _ = (kind, device);
    None
}

/// First `key:` value of a Linux /proc file.
fn proc_field(path: &str, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().find(|l| l.starts_with(key))?;
    Some(line.split_once(':')?.1.trim().to_string())
}

fn host_info() -> Value {
    json!({
        "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "cpu": proc_field("/proc/cpuinfo", "model name"),
        "logical_cores": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "memory": proc_field("/proc/meminfo", "MemTotal"),
    })
}

fn peak_rss_mib() -> Option<f64> {
    let kib: f64 = proc_field("/proc/self/status", "VmHWM")?
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kib / 1024.0)
}
