use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    BsRoformer,
    MelBandRoformer,
}

/// Versioned forward contract. All non-persistent band constants are explicit.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub family: Family,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff_mult: usize,
    pub mask_depth: usize,
    pub mask_expansion: usize,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub chunk_samples: usize,
    pub overlap: usize,
    pub zero_dc: bool,
    pub stems: Vec<String>,
    /// FFT-bin indices in each band, before stereo/complex expansion.
    pub bands: Vec<Vec<usize>>,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.dim > 0 && self.depth > 0 && self.depth <= 64 && self.dim <= 4096,
            "invalid model dimensions"
        );
        ensure!(
            self.heads > 0
                && self.heads <= 64
                && self.head_dim > 0
                && self.head_dim.is_multiple_of(2)
                && self.head_dim <= 256,
            "invalid attention dimensions"
        );
        ensure!(
            self.ff_mult > 0
                && self.ff_mult <= 16
                && self.mask_depth > 0
                && self.mask_depth <= 8
                && self.mask_expansion > 0
                && self.mask_expansion <= 16,
            "invalid MLP dimensions"
        );
        ensure!(
            self.sample_rate == 44_100 && self.n_fft == 2048 && (1..=1024).contains(&self.hop),
            "schema 1 requires 44100 Hz, FFT 2048 and hop <= 1024"
        );
        ensure!(
            self.chunk_samples > self.n_fft / 2
                && self.chunk_samples <= 44_100 * 60
                && self.overlap > 0
                && self.overlap <= 16,
            "invalid chunk settings"
        );
        ensure!(
            !self.stems.is_empty()
                && self.stems.len() <= 2
                && self
                    .stems
                    .iter()
                    .all(|s| s == "vocals" || s == "instrumental"),
            "invalid stem labels"
        );
        ensure!(
            self.stems.len() == 1 || self.stems[0] != self.stems[1],
            "duplicate stems"
        );
        ensure!((2..=256).contains(&self.bands.len()), "invalid band count");
        let bins = self.n_fft / 2 + 1;
        let mut coverage = vec![0; bins];
        for band in &self.bands {
            ensure!(
                !band.is_empty() && band.windows(2).all(|w| w[0] < w[1]),
                "bands must be nonempty and strictly ordered"
            );
            for &f in band {
                ensure!(f < bins, "band index out of range");
                coverage[f] += 1;
            }
        }
        ensure!(
            coverage.iter().all(|&n| n > 0),
            "bands leave uncovered FFT bins"
        );
        if self.family == Family::BsRoformer {
            ensure!(
                coverage.iter().all(|&n| n == 1),
                "BS bands must partition the frequency axis"
            );
            ensure!(
                self.bands.iter().flatten().copied().eq(0..bins),
                "BS bands must be contiguous and in frequency order"
            );
        }
        Ok(())
    }

    pub fn leap_xe(instrumental: bool) -> Self {
        let counts: Vec<usize> = std::iter::repeat_n(2, 24)
            .chain(std::iter::repeat_n(4, 36))
            .chain(std::iter::repeat_n(12, 16))
            .chain(std::iter::repeat_n(24, 8))
            .chain(std::iter::repeat_n(48, 4))
            .chain([128, 129])
            .collect();
        let mut offset = 0;
        let bands = counts
            .into_iter()
            .map(|n| {
                let b = (offset..offset + n).collect();
                offset += n;
                b
            })
            .collect();
        Self {
            family: Family::BsRoformer,
            dim: 256,
            depth: 16,
            heads: 8,
            head_dim: 64,
            ff_mult: 4,
            mask_depth: 2,
            mask_expansion: 4,
            sample_rate: 44_100,
            n_fft: 2048,
            hop: 512,
            chunk_samples: 881_559,
            overlap: 2,
            zero_dc: true,
            stems: vec![
                if instrumental {
                    "instrumental"
                } else {
                    "vocals"
                }
                .into(),
            ],
            bands,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub model_id: String,
    pub weights_sha256: String,
    pub checkpoint_sha256: String,
    pub source_url: String,
    pub forward_revision: String,
    pub weight_license: String,
    pub config: ModelConfig,
}

impl Manifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported model package schema");
        ensure!(!self.model_id.is_empty(), "model ID is empty");
        for digest in [&self.weights_sha256, &self.checkpoint_sha256] {
            ensure!(
                digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid SHA256"
            );
        }
        self.config.validate()
    }
}
