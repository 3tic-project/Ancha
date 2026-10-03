//! Separate one file with one model, on a backend chosen at start-up.
//!
//! ```bash
//! cargo run --release --example separate -- 'audio/ReoNa - Amore.mp3' \
//!     models/leap-xe-voc outputs/example-leap --backend cuda --duration 30
//! # NVIDIA: build with `--features cuda` and pass `--backend cuda`.
//! ```
//!
//! A directory is a RoFormer or MDX23C package; a `.onnx` file is one of the
//! registered classic MDX models (9482, KARA, KARA 2, Inst HQ 2).
use ancha::{
    backend::{self, BackendKind},
    mdx_runtime::MdxOptions,
    report::StemReport,
    runtime::SeparateOptions,
};
use ancha_audio::decode::DecodeOptions;
use anyhow::Result;
use clap::Parser;
use std::{path::PathBuf, sync::atomic::AtomicBool};

#[derive(Parser)]
struct Args {
    /// WAV / FLAC / MP3 input.
    input: PathBuf,
    /// RoFormer / MDX23C package directory or classic MDX `.onnx` file.
    model: PathBuf,
    /// Directory for the stems and run.json. A previous result there is replaced.
    output: PathBuf,
    #[arg(long, value_enum)]
    backend: BackendKind,
    /// GPU ordinal for wgpu / cuda.
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value_t = 0.0)]
    start: f64,
    #[arg(long)]
    duration: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kind = args.backend;
    let decode = DecodeOptions {
        start_seconds: args.start,
        duration_seconds: args.duration,
        ..DecodeOptions::default()
    };
    let cancelled = AtomicBool::new(false);
    let progress = |done, total| eprintln!("chunk {done}/{total}");
    let (model_id, stems, total, rtf) = if args.model.extension().is_some_and(|e| e == "onnx") {
        // FFT size, stems and compensation come from the registered model, not the options.
        let options = MdxOptions {
            input: args.input,
            model: args.model,
            output: args.output,
            decode,
            overlap: None,
            denoise: false,
            optimized: true,
            batch_size: 1,
            conv_gemm: kind.mdx_conv_gemm(),
        };
        let r = backend::separate_mdx(kind, args.device, &options, &cancelled, progress)?;
        (r.model_id, r.stems, r.timings.total_seconds, r.rtf)
    } else if ancha_models::mdx23c::is_package(&args.model)? {
        let options = ancha::mdx23c_runtime::Options {
            input: args.input,
            model: args.model,
            output: args.output,
            decode,
            chunk_samples: None,
            overlap: None,
            conv_gemm: kind.mdx_conv_gemm(),
            optimized: true,
        };
        let r = backend::separate_mdx23c(kind, args.device, &options, &cancelled, progress)?;
        (r.model_id, r.stems, r.timings.total_seconds, r.rtf)
    } else {
        // `None` keeps the native chunk / overlap of the package manifest.
        let attention = kind.attention_plan();
        let options = SeparateOptions {
            input: args.input,
            model: args.model,
            output: args.output,
            decode,
            chunk_samples: None,
            overlap: None,
            max_score_mib: attention.score_budget >> 20,
            attention,
        };
        let r = backend::separate(kind, args.device, &options, &cancelled, progress)?;
        (r.model_id, r.stems, r.timings.total_seconds, r.rtf)
    };
    println!("{model_id} on {}: {total:.2} s, RTF {rtf:.3}", kind.label());
    for StemReport {
        name,
        origin,
        peak,
        rms,
    } in stems
    {
        println!("  {name}.wav ({origin}) peak {peak:.4} rms {rms:.5}");
    }
    Ok(())
}
