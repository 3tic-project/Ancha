//! Reproducible execution reports.
use ancha_models::config::ModelConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Timings {
    pub model_load_seconds: f64,
    pub decode_seconds: f64,
    pub resample_seconds: f64,
    pub stft_seconds: f64,
    pub model_seconds: f64,
    pub istft_seconds: f64,
    pub overlap_seconds: f64,
    pub write_seconds: f64,
    pub total_seconds: f64,
    /// Wall time of each model call (chunk or MDX batch), including upload and readback.
    /// The first GPU call also pays kernel compilation and autotuning.
    #[serde(default)]
    pub model_call_seconds: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StemReport {
    pub name: String,
    pub origin: String,
    pub peak: f32,
    pub rms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunReport {
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
    pub tensor_count: usize,
    pub chunks: usize,
    pub profile: String,
    pub effective_config: ModelConfig,
    /// Effective time-axis tiles (groups = bands, tokens = frames).
    pub query_tile: usize,
    pub group_tile: usize,
    /// Effective frequency-axis tiles (groups = frames, tokens = bands).
    #[serde(default)]
    pub frequency_query_tile: usize,
    #[serde(default)]
    pub frequency_group_tile: usize,
    /// `auto` sizes tiles from the score budget; `manual` uses explicit flags.
    #[serde(default)]
    pub attention_tiling: String,
    /// Host threads over independent attention groups (CPU backends).
    #[serde(default)]
    pub host_threads: usize,
    pub linear_layout: String,
    pub build_features: Vec<String>,
    #[serde(default)]
    pub cpu_thread_environment: std::collections::BTreeMap<String, String>,
    pub estimated_time_attention_score_bytes: usize,
    #[serde(default)]
    pub estimated_frequency_attention_score_bytes: usize,
    pub residual_reconstruction_max_abs: Option<f32>,
    pub stems: Vec<StemReport>,
    pub timings: Timings,
    /// Total wall time (including loading and writing) / audio duration.
    pub rtf: f64,
}
