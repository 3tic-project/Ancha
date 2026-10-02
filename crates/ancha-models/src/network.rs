use crate::{
    config::{Family, Manifest, ModelConfig},
    roformer::{AttentionPlan, SourceNorm, Transformer},
    weights::{self, Weights},
};
use anyhow::{Result, ensure};
use burn::{
    nn::Linear,
    tensor::{
        IndexingUpdateOp, Int, Tensor, TensorData,
        activation::{sigmoid, tanh},
        backend::Backend,
    },
};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

struct Band<B: Backend> {
    norm: SourceNorm<B>,
    projection: Linear<B>,
}
struct MaskMlp<B: Backend> {
    layers: Vec<Linear<B>>,
}
struct Axis<B: Backend> {
    time: Transformer<B>,
    frequency: Transformer<B>,
}

pub struct Roformer<B: Backend> {
    bands: Vec<Band<B>>,
    axes: Vec<Axis<B>>,
    final_norm: Option<SourceNorm<B>>,
    masks: Vec<Vec<MaskMlp<B>>>,
    widths: Vec<usize>,
    config: ModelConfig,
    indices: Tensor<B, 1, Int>,
    coverage: Tensor<B, 5>,
    pub tensor_count: usize,
}

impl<B: Backend> Roformer<B> {
    pub fn load(package: &Path, manifest: &Manifest, device: &B::Device) -> Result<Self> {
        manifest.validate()?;
        let path = package.join("model.safetensors");
        ensure!(
            weights::sha256_file(&path)? == manifest.weights_sha256,
            "model checksum mismatch"
        );
        let bytes = std::fs::read(path)?;
        let mut w = Weights::new(&bytes)?;
        let c = &manifest.config;
        let widths: Vec<_> = c.bands.iter().map(|b| b.len() * 4).collect();
        let bands = widths
            .iter()
            .enumerate()
            .map(|(i, &width)| {
                Ok(Band {
                    norm: SourceNorm::load(
                        &mut w,
                        &format!("band_split.to_features.{i}.0"),
                        width,
                        device,
                    )?,
                    projection: w.linear(
                        &format!("band_split.to_features.{i}.1"),
                        width,
                        c.dim,
                        true,
                        device,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let normalized = c.family == Family::MelBandRoformer;
        let axes = (0..c.depth)
            .map(|i| {
                Ok(Axis {
                    time: Transformer::load(
                        &mut w,
                        &format!("layers.{i}.0"),
                        c,
                        normalized,
                        device,
                    )?,
                    frequency: Transformer::load(
                        &mut w,
                        &format!("layers.{i}.1"),
                        c,
                        normalized,
                        device,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let final_norm = if normalized {
            None
        } else {
            Some(SourceNorm::load(&mut w, "final_norm", c.dim, device)?)
        };
        let mut masks = Vec::new();
        for stem in 0..c.stems.len() {
            let mut mask_bands = Vec::new();
            for (i, &width) in widths.iter().enumerate() {
                let mut sizes = vec![c.dim];
                let hidden_layers = if c.family == Family::BsRoformer {
                    c.mask_depth - 1
                } else {
                    c.mask_depth
                };
                sizes.extend(std::iter::repeat_n(c.dim * c.mask_expansion, hidden_layers));
                sizes.push(width * 2);
                let layers = sizes
                    .windows(2)
                    .enumerate()
                    .map(|(j, sz)| {
                        w.linear(
                            &format!("mask_estimators.{stem}.to_freqs.{i}.0.{}", j * 2),
                            sz[0],
                            sz[1],
                            true,
                            device,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                mask_bands.push(MaskMlp { layers });
            }
            masks.push(mask_bands);
        }
        let tensor_count = w.finish()?;
        let bins = c.n_fft / 2 + 1;
        let indices: Vec<i64> = c
            .bands
            .iter()
            .flatten()
            .flat_map(|&f| [2 * f as i64, 2 * f as i64 + 1])
            .collect();
        let n_indices = indices.len();
        let mut coverage = vec![0f32; bins * 2];
        for &i in &indices {
            coverage[i as usize] += 1.0;
        }
        Ok(Self {
            bands,
            axes,
            final_norm,
            masks,
            widths,
            config: c.clone(),
            tensor_count,
            indices: Tensor::from_data(TensorData::new(indices, [n_indices]), device),
            coverage: Tensor::<B, 1>::from_floats(coverage.as_slice(), device).reshape([
                1,
                1,
                bins * 2,
                1,
                1,
            ]),
        })
    }

    /// `[1, FFT_bins*2, frames, RI]` -> `[1, stems, FFT_bins*2, frames, RI]`.
    pub fn forward(
        &self,
        stft: Tensor<B, 4>,
        plan: AttentionPlan,
        cancelled: &AtomicBool,
    ) -> Result<Tensor<B, 5>> {
        plan.validate()?;
        let [batch, rows, t, ri] = stft.dims();
        ensure!(
            batch == 1 && rows == (self.config.n_fft / 2 + 1) * 2 && ri == 2 && t > 0,
            "invalid model spectrum shape"
        );
        let total = self.widths.iter().sum::<usize>();
        let z = stft
            .clone()
            .select(1, self.indices.clone())
            .permute([0, 2, 1, 3])
            .reshape([1, t, total]);
        let mut band_features = Vec::new();
        let mut offset = 0;
        for (band, &width) in self.bands.iter().zip(&self.widths) {
            let value = band
                .projection
                .forward(band.norm.forward(z.clone().narrow(2, offset, width)));
            band_features.push(value.reshape([1, t, 1, self.config.dim]));
            offset += width;
        }
        let mut x = Tensor::cat(band_features, 2);
        let nb = self.bands.len();
        let dim = self.config.dim;
        for axis in &self.axes {
            ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
            let z = x.permute([0, 2, 1, 3]).reshape([nb, t, dim]);
            let z = axis
                .time
                .forward(z, plan)
                .reshape([1, nb, t, dim])
                .permute([0, 2, 1, 3])
                .reshape([t, nb, dim]);
            x = axis.frequency.forward(z, plan).reshape([1, t, nb, dim]);
        }
        if let Some(norm) = &self.final_norm {
            x = norm.forward(x);
        }
        let mut stem_masks = Vec::new();
        for head in &self.masks {
            let mut band_masks = Vec::new();
            for (i, mlp) in head.iter().enumerate() {
                ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
                let mut z = x.clone().narrow(2, i, 1).reshape([1, t, dim]);
                for (j, linear) in mlp.layers.iter().enumerate() {
                    z = linear.forward(z);
                    if j + 1 < mlp.layers.len() {
                        z = tanh(z);
                    }
                }
                let width = self.widths[i];
                let value = z
                    .clone()
                    .narrow(2, 0, width)
                    .mul(sigmoid(z.narrow(2, width, width)));
                band_masks.push(value);
            }
            stem_masks.push(Tensor::cat(band_masks, 2).reshape([1, 1, t, total]));
        }
        let ns = self.masks.len();
        let nr = total / 2;
        let masks = Tensor::cat(stem_masks, 1)
            .reshape([1, ns, t, nr, 2])
            .permute([0, 1, 3, 2, 4]);
        let masks = if self.config.family == Family::BsRoformer {
            masks
        } else {
            let indices = self
                .indices
                .clone()
                .reshape([1, 1, nr, 1, 1])
                .expand([1, ns, nr, t, 2]);
            Tensor::<B, 5>::zeros([1, ns, rows, t, 2], &stft.device())
                .scatter(2, indices, masks, IndexingUpdateOp::Add)
                .div(self.coverage.clone())
        };
        let source = stft.reshape([1, 1, rows, t, 2]);
        let ar = source.clone().narrow(4, 0, 1);
        let ai = source.narrow(4, 1, 1);
        let br = masks.clone().narrow(4, 0, 1);
        let bi = masks.narrow(4, 1, 1);
        let real = ar.clone().mul(br.clone()).sub(ai.clone().mul(bi.clone()));
        let imag = ar.mul(bi).add(ai.mul(br));
        Ok(Tensor::cat(vec![real, imag], 4))
    }
}
