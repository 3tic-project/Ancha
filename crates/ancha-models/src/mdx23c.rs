//! Native TFC/TDF v3 spectral network. STFT and UVR overlap-add live in the runtime.
use crate::{
    config::Manifest,
    spatial::{InstanceNorm, conv_transpose2d_gemm, conv2d_gemm},
    weights::{self, Weights},
};
use anyhow::{Context, Result, ensure};
use burn::{
    nn::Linear,
    tensor::{
        Tensor,
        activation::gelu,
        backend::Backend,
        module::{conv_transpose2d, conv2d},
        ops::{ConvOptions, ConvTransposeOptions},
    },
};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub const REFERENCE_REVISION: &str =
    "UVR/5517e0cf0d1acd16a1618eeedec596957523f9e1;lib_v5/tfc_tdf_v3.py";
pub const HUB_REVISION: &str = "f3bb9a312519f4404dde996ef1054ec30353c46f";
pub const CHECKPOINT_SHA256: &str =
    "7d960d8e40a458120412c1bd807e013d2dbca7b959cc9da2bbcb0eb203d1daea";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Family {
    #[serde(rename = "mdx23c")]
    Mdx23c,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub family: Family,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub bins: usize,
    pub frames: usize,
    pub overlap: usize,
    pub subbands: usize,
    pub channels: usize,
    pub growth: usize,
    pub scales: usize,
    pub blocks_per_scale: usize,
    pub bottleneck_factor: usize,
    pub scale: [usize; 2],
    pub stems: Vec<String>,
}
impl Config {
    pub fn inst_voc_hq2() -> Self {
        Self {
            family: Family::Mdx23c,
            sample_rate: 44100,
            n_fft: 8192,
            hop: 1024,
            bins: 4096,
            frames: 256,
            overlap: 8,
            subbands: 4,
            channels: 128,
            growth: 128,
            scales: 5,
            blocks_per_scale: 2,
            bottleneck_factor: 4,
            scale: [2, 2],
            stems: vec!["vocals".into(), "instrumental".into()],
        }
    }
    pub fn chunk_samples(&self) -> usize {
        self.hop * (self.frames - 1)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_rate == 44100
                && self.n_fft >= 32
                && self.n_fft <= 8192
                && self.n_fft.is_multiple_of(2)
                && self.hop > 0
                && self.hop <= self.n_fft / 2,
            "MDX23C requires stereo 44100 Hz and a valid centered STFT"
        );
        ensure!(
            (1..=6).contains(&self.scales) && self.scale == [2, 2],
            "MDX23C supports 1..=6 scales with 2x2 down/up sampling"
        );
        let factor = 1 << self.scales;
        ensure!(
            (1..=16).contains(&self.subbands)
                && self.bins > 0
                && self.bins <= self.n_fft / 2 + 1
                && self.bins.is_multiple_of(self.subbands * factor)
                && self.frames >= factor
                && self.frames <= 2048
                && self.frames.is_multiple_of(factor),
            "MDX23C frequency/time dimensions must survive all encoder scales"
        );
        ensure!(
            (1..=1024).contains(&self.channels)
                && (1..=1024).contains(&self.growth)
                && self.channels + self.scales * self.growth <= 2048
                && (1..=4).contains(&self.blocks_per_scale)
                && (1..=16).contains(&self.bottleneck_factor),
            "invalid MDX23C network size"
        );
        let bottom = self.bins / self.subbands / factor;
        ensure!(
            bottom >= self.bottleneck_factor && (self.frames / factor) * bottom > 1,
            "MDX23C bottleneck is too small for Linear or InstanceNorm"
        );
        ensure!(
            self.chunk_samples() > self.n_fft / 2
                && self.chunk_samples() <= 44100 * 60
                && (1..=16).contains(&self.overlap),
            "invalid MDX23C context"
        );
        ensure!(
            self.stems == ["vocals", "instrumental"],
            "MDX23C schema 1 requires two native heads in vocals/instrumental order"
        );
        Ok(())
    }
}
pub type Package = Manifest<Config>;
impl Manifest<Config> {
    pub fn validate(&self) -> Result<()> {
        self.validate_metadata()?;
        self.config.validate()
    }
}
pub fn read_manifest(path: &Path) -> Result<Package> {
    let m: Package = serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
    m.validate()?;
    Ok(m)
}
/// Reads only the discriminator; each architecture then validates its complete manifest.
pub fn is_package(path: &Path) -> Result<bool> {
    let p = path.join("manifest.json");
    let value: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&p).with_context(|| format!("read {}", p.display()))?,
    )?;
    Ok(value
        .get("config")
        .and_then(|c| c.get("family"))
        .and_then(|f| f.as_str())
        == Some("mdx23c"))
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub conv_gemm: bool,
    /// CUDA custom TDF GEMM; generic backends retain their validated batched linear path.
    pub optimized: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            conv_gemm: false,
            optimized: true,
        }
    }
}

struct Conv<B: Backend> {
    weight: Tensor<B, 4>,
    stride: [usize; 2],
    padding: [usize; 2],
}
impl<B: Backend> Conv<B> {
    fn load(
        w: &mut Weights<'_>,
        p: &str,
        channels: [usize; 2],
        k: usize,
        stride: [usize; 2],
        padding: [usize; 2],
        d: &B::Device,
    ) -> Result<Self> {
        let [ci, co] = channels;
        Ok(Self {
            weight: w.tensor(&format!("{p}.weight"), [co, ci, k, k], d)?,
            stride,
            padding,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        if gemm
            && let Some(y) = conv2d_gemm(
                x.clone(),
                self.weight.clone(),
                None,
                self.stride,
                self.padding,
            )
        {
            return y;
        }
        conv2d(
            x,
            self.weight.clone(),
            None,
            ConvOptions::new(self.stride, self.padding, [1, 1], 1),
        )
    }
}
struct Preact<B: Backend> {
    norm: InstanceNorm<B>,
}
impl<B: Backend> Preact<B> {
    fn load(w: &mut Weights<'_>, p: &str, c: usize, d: &B::Device) -> Result<Self> {
        Ok(Self {
            norm: InstanceNorm::load_with_epsilon(w, p, c, 1e-5, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Keep the reference spatial reductions; speculative layout changes are not enabled here.
        gelu(self.norm.forward(x))
    }
}
struct Block<B: Backend> {
    first_norm: Preact<B>,
    first_conv: Conv<B>,
    tdf_norm1: Preact<B>,
    tdf1: Linear<B>,
    tdf_norm2: Preact<B>,
    tdf2: Linear<B>,
    second_norm: Preact<B>,
    second_conv: Conv<B>,
    shortcut: Conv<B>,
}
fn tdf<B: Backend>(linear: &Linear<B>, x: Tensor<B, 4>, optimized: bool) -> Tensor<B, 4> {
    if !optimized || !crate::fused::available::<B>() {
        return linear.forward(x);
    }
    let [b, c, t, f] = x.dims();
    let weight = linear.weight.val();
    let out = weight.dims()[1];
    let flat = x.clone().reshape([b * c * t, f]);
    match crate::fused::linear(&flat, &weight, crate::fused::Epilogue::default()) {
        Some(y) => y.reshape([b, c, t, out]),
        None => linear.forward(x),
    }
}
impl<B: Backend> Block<B> {
    fn load(
        w: &mut Weights<'_>,
        p: &str,
        ci: usize,
        co: usize,
        f: usize,
        bn: usize,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            first_norm: Preact::load(w, &format!("{p}.tfc1.0"), ci, d)?,
            first_conv: Conv::load(w, &format!("{p}.tfc1.2"), [ci, co], 3, [1, 1], [1, 1], d)?,
            tdf_norm1: Preact::load(w, &format!("{p}.tdf.0"), co, d)?,
            tdf1: w.linear(&format!("{p}.tdf.2"), f, f / bn, false, d)?,
            tdf_norm2: Preact::load(w, &format!("{p}.tdf.3"), co, d)?,
            tdf2: w.linear(&format!("{p}.tdf.5"), f / bn, f, false, d)?,
            second_norm: Preact::load(w, &format!("{p}.tfc2.0"), co, d)?,
            second_conv: Conv::load(w, &format!("{p}.tfc2.2"), [co, co], 3, [1, 1], [1, 1], d)?,
            shortcut: Conv::load(w, &format!("{p}.shortcut"), [ci, co], 1, [1, 1], [0, 0], d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, o: Options) -> Tensor<B, 4> {
        let skip = self.shortcut.forward(x.clone(), o.conv_gemm);
        let x = self
            .first_conv
            .forward(self.first_norm.forward(x), o.conv_gemm);
        let y = tdf(&self.tdf1, self.tdf_norm1.forward(x.clone()), o.optimized);
        let y = tdf(&self.tdf2, self.tdf_norm2.forward(y), o.optimized);
        self.second_conv
            .forward(self.second_norm.forward(x + y), o.conv_gemm)
            + skip
    }
}
struct Blocks<B: Backend>(Vec<Block<B>>);
impl<B: Backend> Blocks<B> {
    fn load(
        w: &mut Weights<'_>,
        p: &str,
        ci: usize,
        co: usize,
        f: usize,
        config: &Config,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self(
            (0..config.blocks_per_scale)
                .map(|i| {
                    Block::load(
                        w,
                        &format!("{p}.blocks.{i}"),
                        if i == 0 { ci } else { co },
                        co,
                        f,
                        config.bottleneck_factor,
                        d,
                    )
                })
                .collect::<Result<_>>()?,
        ))
    }
    fn forward(
        &self,
        mut x: Tensor<B, 4>,
        o: Options,
        cancelled: &AtomicBool,
    ) -> Result<Tensor<B, 4>> {
        for block in &self.0 {
            ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
            x = block.forward(x, o);
        }
        Ok(x)
    }
}
struct Down<B: Backend> {
    blocks: Blocks<B>,
    norm: Preact<B>,
    conv: Conv<B>,
}
struct Up<B: Backend> {
    norm: Preact<B>,
    weight: Tensor<B, 4>,
    blocks: Blocks<B>,
}
impl<B: Backend> Up<B> {
    fn forward(
        &self,
        x: Tensor<B, 4>,
        skip: Tensor<B, 4>,
        o: Options,
        cancelled: &AtomicBool,
    ) -> Result<Tensor<B, 4>> {
        let x = self.norm.forward(x);
        let x = if o.conv_gemm {
            conv_transpose2d_gemm(x, self.weight.clone()).expect("checked non-overlapping upscale")
        } else {
            conv_transpose2d(
                x,
                self.weight.clone(),
                None,
                ConvTransposeOptions::new([2, 2], [0, 0], [0, 0], [1, 1], 1),
            )
        };
        ensure!(
            x.dims()[2..] == skip.dims()[2..],
            "MDX23C decoder skip shape mismatch"
        );
        self.blocks
            .forward(Tensor::cat(vec![x, skip], 1), o, cancelled)
    }
}
pub struct Mdx23c<B: Backend> {
    first: Conv<B>,
    down: Vec<Down<B>>,
    bottom: Blocks<B>,
    up: Vec<Up<B>>,
    final_first: Conv<B>,
    final_last: Conv<B>,
    pub config: Config,
    pub tensor_count: usize,
}
impl<B: Backend> Mdx23c<B> {
    pub fn load(path: &Path, manifest: &Package, d: &B::Device) -> Result<Self> {
        manifest.validate()?;
        let bytes = std::fs::read(path.join("model.safetensors"))?;
        weights::verified(&bytes, &manifest.weights_sha256, || {
            Self::build(&bytes, &manifest.config, d)
        })
    }
    fn build(bytes: &[u8], config: &Config, d: &B::Device) -> Result<Self> {
        let mut w = Weights::new(bytes)?;
        let input = 4 * config.subbands;
        let first = Conv::load(
            &mut w,
            "first_conv",
            [input, config.channels],
            1,
            [1, 1],
            [0, 0],
            d,
        )?;
        let (mut c, mut f) = (config.channels, config.bins / config.subbands);
        let mut down = Vec::new();
        for i in 0..config.scales {
            let p = format!("encoder_blocks.{i}");
            down.push(Down {
                blocks: Blocks::load(&mut w, &format!("{p}.tfc_tdf"), c, c, f, config, d)?,
                norm: Preact::load(&mut w, &format!("{p}.downscale.conv.0"), c, d)?,
                conv: Conv::load(
                    &mut w,
                    &format!("{p}.downscale.conv.2"),
                    [c, c + config.growth],
                    2,
                    [2, 2],
                    [0, 0],
                    d,
                )?,
            });
            f /= 2;
            c += config.growth;
        }
        let bottom = Blocks::load(&mut w, "bottleneck_block", c, c, f, config, d)?;
        let mut up = Vec::new();
        for i in 0..config.scales {
            let p = format!("decoder_blocks.{i}");
            let co = c - config.growth;
            f *= 2;
            up.push(Up {
                norm: Preact::load(&mut w, &format!("{p}.upscale.conv.0"), c, d)?,
                weight: w.tensor(&format!("{p}.upscale.conv.2.weight"), [c, co, 2, 2], d)?,
                blocks: Blocks::load(&mut w, &format!("{p}.tfc_tdf"), co * 2, co, f, config, d)?,
            });
            c = co;
        }
        let final_first = Conv::load(&mut w, "final_conv.0", [c + input, c], 1, [1, 1], [0, 0], d)?;
        let final_last = Conv::load(
            &mut w,
            "final_conv.2",
            [c, config.stems.len() * input],
            1,
            [1, 1],
            [0, 0],
            d,
        )?;
        let tensor_count = w.finish()?;
        Ok(Self {
            first,
            down,
            bottom,
            up,
            final_first,
            final_last,
            config: config.clone(),
            tensor_count,
        })
    }
    /// `[batch, L.re/L.im/R.re/R.im, bins, frames]` -> `[batch, stems*4, bins, frames]`.
    /// MDX23C retains DC and the first three bins and predicts spectra, not masks.
    pub fn forward(
        &self,
        spectrum: Tensor<B, 4>,
        o: Options,
        cancelled: &AtomicBool,
    ) -> Result<Tensor<B, 4>> {
        ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
        let [b, c, f, t] = spectrum.dims();
        let cfg = &self.config;
        let factor = 1 << cfg.scales;
        ensure!(
            b > 0 && b <= 4 && c == 4 && f == cfg.bins && t >= factor && t.is_multiple_of(factor),
            "invalid MDX23C spectrum shape"
        );
        let mix = spectrum.reshape([b, 4 * cfg.subbands, f / cfg.subbands, t]);
        let first = self.first.forward(mix.clone(), o.conv_gemm);
        let mut x = first.clone().swap_dims(2, 3);
        let mut skips = Vec::with_capacity(self.down.len());
        for down in &self.down {
            x = down.blocks.forward(x, o, cancelled)?;
            skips.push(x.clone());
            x = down.conv.forward(down.norm.forward(x), o.conv_gemm);
        }
        x = self.bottom.forward(x, o, cancelled)?;
        for up in &self.up {
            x = up.forward(x, skips.pop().expect("encoder skip"), o, cancelled)?;
        }
        let x = x.swap_dims(2, 3) * first;
        let x = self
            .final_first
            .forward(Tensor::cat(vec![mix, x], 1), o.conv_gemm);
        let x = self.final_last.forward(gelu(x), o.conv_gemm);
        Ok(x.reshape([b, cfg.stems.len() * 4, f, t]))
    }
}
