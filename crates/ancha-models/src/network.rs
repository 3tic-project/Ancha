use crate::{
    config::{Family, Manifest, ModelConfig},
    hyperace::Segm,
    roformer::{AttentionPlan, AxisRope, SourceNorm, Transformer, dense, unit_norm},
    weights::{self, Weights},
};
use anyhow::{Context, Result, ensure};
use burn::tensor::{
    IndexingUpdateOp, Int, Tensor, TensorData,
    activation::{sigmoid, tanh},
    backend::Backend,
};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

/// Batched per-band `W·x + b` in feature-major layout: `[n, out, in]·[n, in, t]`.
/// The `[n, out, 1]` bias broadcasts along frames only.
struct BandAffine<B: Backend> {
    weight: Tensor<B, 3>,
    bias: Tensor<B, 3>,
}
impl<B: Backend> BandAffine<B> {
    fn new(weight: Vec<f32>, bias: Vec<f32>, shape: [usize; 3], device: &B::Device) -> Self {
        let [n, out, _] = shape;
        Self {
            weight: Tensor::from_data(TensorData::new(weight, shape), device),
            bias: Tensor::from_data(TensorData::new(bias, [n, out, 1]), device),
        }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.weight.clone().matmul(x) + self.bias.clone()
    }
}

/// Bands sharing one input width, consecutive in group order. Each group runs as one
/// batched GEMM instead of one small launch per band.
struct BandGroup<B: Backend> {
    width: usize,
    /// First band position in group order.
    start: usize,
    count: usize,
    /// First column of the group in the gathered band features.
    column: usize,
    /// Band projections with sqrt(width)·gamma folded into the input columns.
    projection: BandAffine<B>,
}
struct MaskHead<B: Backend> {
    hidden: Vec<BandAffine<B>>,
    /// Per width group: GLU value and gate halves of the last projection.
    output: Vec<(BandAffine<B>, BandAffine<B>)>,
}
struct Axis<B: Backend> {
    time: Transformer<B>,
    frequency: Transformer<B>,
}

pub struct Roformer<B: Backend> {
    groups: Vec<BandGroup<B>>,
    axes: Vec<Axis<B>>,
    final_norm: Option<SourceNorm<B>>,
    masks: Vec<MaskHead<B>>,
    segmentation: Option<Segm<B>>,
    config: ModelConfig,
    /// Packed spectrum columns `(row, re/im)` of every band, in group order.
    columns: Tensor<B, 1, Int>,
    /// (group -> band order, band -> group order); `None` when groups follow band order.
    reorder: Option<(Tensor<B, 1, Int>, Tensor<B, 1, Int>)>,
    /// Spectrum row of each mask row (group order) and per-row band coverage `[rows, 1]`.
    /// Used for overlapping Mel bands and any non-identity band order.
    overlap: Option<(Tensor<B, 1, Int>, Tensor<B, 2>)>,
    /// Unique rotary frequency sets; per axis layer `(time, frequency)` set index.
    rope_freqs: Vec<Vec<f32>>,
    rope_index: Vec<(usize, usize)>,
    pub tensor_count: usize,
}

impl<B: Backend> Roformer<B> {
    pub fn load(package: &Path, manifest: &Manifest, device: &B::Device) -> Result<Self> {
        manifest.validate()?;
        let path = package.join("model.safetensors");
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        weights::verified(&bytes, &manifest.weights_sha256, || {
            Self::build(&bytes, manifest, device)
        })
    }

    fn build(bytes: &[u8], manifest: &Manifest, device: &B::Device) -> Result<Self> {
        let mut w = Weights::new(bytes)?;
        let c = &manifest.config;
        let nb = c.bands.len();
        let mut width_groups: Vec<(usize, Vec<usize>)> = Vec::new();
        for (band, bins) in c.bands.iter().enumerate() {
            let width = bins.len() * 4;
            match width_groups.iter_mut().find(|(w, _)| *w == width) {
                Some((_, bands)) => bands.push(band),
                None => width_groups.push((width, vec![band])),
            }
        }
        let order: Vec<usize> = width_groups.iter().flat_map(|(_, b)| b.clone()).collect();
        let identity = order.iter().enumerate().all(|(i, &b)| i == b);
        ensure!(
            identity || c.family != Family::HyperaceV2,
            "HyperACE spatial masks require band-ordered width groups"
        );
        let mut groups = Vec::new();
        let (mut start, mut column) = (0, 0);
        for (width, bands) in &width_groups {
            let (width, count) = (*width, bands.len());
            let mut weight = Vec::with_capacity(count * c.dim * width);
            let mut bias = Vec::with_capacity(count * c.dim);
            for &band in bands {
                let p = format!("band_split.to_features.{band}");
                let scale = SourceNorm::<B>::scale(&mut w, &format!("{p}.0"), width)?;
                let rows = w.take(&format!("{p}.1.weight"), &[c.dim, width])?;
                for row in rows.chunks_exact(width) {
                    weight.extend(row.iter().zip(&scale).map(|(v, s)| v * s));
                }
                bias.extend(w.take(&format!("{p}.1.bias"), &[c.dim])?);
            }
            groups.push(BandGroup {
                width,
                start,
                count,
                column,
                projection: BandAffine::new(weight, bias, [count, c.dim, width], device),
            });
            start += count;
            column += count * width;
        }
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
        // BS depth counts all Linears; Mel depth counts hidden layers only.
        let hidden_layers = if normalized {
            c.mask_depth
        } else {
            c.mask_depth - 1
        };
        let mut sizes = vec![c.dim];
        sizes.extend(std::iter::repeat_n(c.dim * c.mask_expansion, hidden_layers));
        let last = sizes.len() - 1;
        let input = sizes[last];
        let mut masks = Vec::new();
        for stem in 0..c.stems.len() {
            let key = |band: usize, layer: usize| {
                format!("mask_estimators.{stem}.to_freqs.{band}.0.{}", layer * 2)
            };
            let mut hidden = Vec::new();
            for (layer, io) in sizes.windows(2).enumerate() {
                let (i, o) = (io[0], io[1]);
                let mut weight = Vec::with_capacity(nb * o * i);
                let mut bias = Vec::with_capacity(nb * o);
                for &band in &order {
                    weight.extend(w.take(&format!("{}.weight", key(band, layer)), &[o, i])?);
                    bias.extend(w.take(&format!("{}.bias", key(band, layer)), &[o])?);
                }
                hidden.push(BandAffine::new(weight, bias, [nb, o, i], device));
            }
            let mut output = Vec::new();
            for (width, bands) in &width_groups {
                let (width, count) = (*width, bands.len());
                let (mut vw, mut vb, mut gw, mut gb) = (vec![], vec![], vec![], vec![]);
                for &band in bands {
                    let weight =
                        w.take(&format!("{}.weight", key(band, last)), &[2 * width, input])?;
                    let bias = w.take(&format!("{}.bias", key(band, last)), &[2 * width])?;
                    let (value, gate) = weight.split_at(width * input);
                    vw.extend_from_slice(value);
                    gw.extend_from_slice(gate);
                    vb.extend_from_slice(&bias[..width]);
                    gb.extend_from_slice(&bias[width..]);
                }
                let shape = [count, width, input];
                output.push((
                    BandAffine::new(vw, vb, shape, device),
                    BandAffine::new(gw, gb, shape, device),
                ));
            }
            masks.push(MaskHead { hidden, output });
        }
        let segmentation = if c.family == Family::HyperaceV2 {
            Some(Segm::load(&mut w, "mask_estimators.0.segm", device)?)
        } else {
            None
        };
        let tensor_count = w.finish()?;
        let int = |v: Vec<i64>| {
            let n = v.len();
            Tensor::<B, 1, Int>::from_data(TensorData::new(v, [n]), device)
        };
        let columns: Vec<i64> = order
            .iter()
            .flat_map(|&band| &c.bands[band])
            .flat_map(|&f| (0..4).map(move |k| (4 * f + k) as i64))
            .collect();
        let reorder = (!identity).then(|| {
            let mut position = vec![0i64; nb];
            for (i, &band) in order.iter().enumerate() {
                position[band] = i as i64;
            }
            (
                int(position),
                int(order.iter().map(|&b| b as i64).collect()),
            )
        });
        let overlap = (normalized || !identity).then(|| {
            let rows = (c.n_fft / 2 + 1) * 2;
            let targets: Vec<i64> = order
                .iter()
                .flat_map(|&band| &c.bands[band])
                .flat_map(|&f| [2 * f as i64, 2 * f as i64 + 1])
                .collect();
            let mut coverage = vec![0f32; rows];
            for &row in &targets {
                coverage[row as usize] += 1.0;
            }
            (
                int(targets),
                Tensor::from_data(TensorData::new(coverage, [rows, 1]), device),
            )
        });
        let mut rope_freqs: Vec<Vec<f32>> = Vec::new();
        let mut slot = |freqs: &Vec<f32>| match rope_freqs.iter().position(|f| f == freqs) {
            Some(i) => i,
            None => {
                rope_freqs.push(freqs.clone());
                rope_freqs.len() - 1
            }
        };
        let rope_index = axes
            .iter()
            .map(|a| {
                (
                    slot(&a.time.attention.freqs),
                    slot(&a.frequency.attention.freqs),
                )
            })
            .collect();
        Ok(Self {
            groups,
            axes,
            final_norm,
            masks,
            segmentation,
            config: c.clone(),
            columns: int(columns),
            reorder,
            overlap,
            rope_freqs,
            rope_index,
            tensor_count,
        })
    }

    fn rope_tables(
        &self,
        tokens: usize,
        frequency: bool,
        device: &B::Device,
    ) -> Vec<Option<AxisRope<B>>> {
        let used: Vec<usize> = self
            .rope_index
            .iter()
            .map(|&(t, f)| if frequency { f } else { t })
            .collect();
        self.rope_freqs
            .iter()
            .enumerate()
            .map(|(i, freqs)| {
                used.contains(&i)
                    .then(|| AxisRope::new(freqs, tokens, self.config.heads, device))
            })
            .collect()
    }

    /// `[frames, rows*2]` packed as (frame, bin·2+channel, re/im)
    /// -> complex masks `[stems, rows, re/im, frames]`, rows = FFT bins × 2.
    pub fn forward(
        &self,
        spectrum: Tensor<B, 2>,
        plan: AttentionPlan,
        cancelled: &AtomicBool,
    ) -> Result<Tensor<B, 4>> {
        plan.validate()?;
        let [t, columns] = spectrum.dims();
        let bins = self.config.n_fft / 2 + 1;
        let rows = bins * 2;
        ensure!(t > 0 && columns == rows * 2, "invalid model spectrum shape");
        let device = spectrum.device();
        let nb = self.config.bands.len();
        let z = spectrum.select(1, self.columns.clone());
        let mut features = Vec::with_capacity(self.groups.len());
        for g in &self.groups {
            let zg = z
                .clone()
                .narrow(1, g.column, g.count * g.width)
                .reshape([t, g.count, g.width]);
            features.push(g.projection.forward(unit_norm(zg).permute([1, 2, 0])));
        }
        let x = Tensor::cat(features, 0);
        let x = match &self.reorder {
            Some((to_band, _)) => x.select(0, to_band.clone()),
            None => x,
        };
        let mut x = dense(x.swap_dims(1, 2));
        let time_rope = self.rope_tables(t, false, &device);
        let frequency_rope = self.rope_tables(nb, true, &device);
        for (axis, &(ti, fi)) in self.axes.iter().zip(&self.rope_index) {
            ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
            let rope = time_rope[ti].as_ref().expect("time rotary table");
            let y = dense(axis.time.forward(x, rope, plan).swap_dims(0, 1));
            let rope = frequency_rope[fi].as_ref().expect("frequency rotary table");
            x = dense(axis.frequency.forward(y, rope, plan).swap_dims(0, 1));
        }
        if let Some(norm) = &self.final_norm {
            x = norm.forward(x);
        }
        let spatial = match &self.segmentation {
            Some(segm) => {
                ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
                let s = segm.forward(
                    x.clone().permute([2, 1, 0]).unsqueeze_dim(0),
                    plan.conv_gemm,
                    cancelled,
                )?;
                // Channels are (stereo, re/im); rows follow (bin, stereo).
                Some(
                    s.reshape([2, 2, t, bins])
                        .permute([3, 0, 1, 2])
                        .reshape([rows, 2, t]),
                )
            }
            None => None,
        };
        let x = match &self.reorder {
            Some((_, to_group)) => x.select(0, to_group.clone()),
            None => x,
        };
        let features = x.swap_dims(1, 2);
        let mut stems = Vec::with_capacity(self.masks.len());
        for head in &self.masks {
            ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
            let mut h = features.clone();
            for layer in &head.hidden {
                h = tanh(layer.forward(h));
            }
            let mut parts = Vec::with_capacity(self.groups.len());
            for (g, (value, gate)) in self.groups.iter().zip(&head.output) {
                let hg = h.clone().narrow(0, g.start, g.count);
                let mask = value.forward(hg.clone()) * sigmoid(gate.forward(hg));
                parts.push(mask.reshape([g.count * g.width / 2, 2, t]));
            }
            let mut mask = Tensor::cat(parts, 0);
            if let Some(spatial) = &spatial {
                mask = mask + spatial.clone();
            }
            let mask = match &self.overlap {
                Some((targets, coverage)) => {
                    let r = mask.dims()[0];
                    Tensor::<B, 2>::zeros([rows, 2 * t], &device)
                        .select_assign(
                            0,
                            targets.clone(),
                            mask.reshape([r, 2 * t]),
                            IndexingUpdateOp::Add,
                        )
                        .div(coverage.clone())
                        .reshape([rows, 2, t])
                }
                None => mask,
            };
            stems.push(mask.unsqueeze_dim(0));
        }
        Ok(Tensor::cat(stems, 0))
    }
}
