use crate::{
    config::ModelConfig,
    weights::{Weights, folded_linear},
};
use anyhow::{Result, ensure};
use burn::{
    nn::Linear,
    tensor::{
        Int, Tensor, TensorData,
        activation::{gelu, sigmoid, softmax},
        backend::Backend,
    },
};

/// `x / max(||x||₂, 1e-12)` over the last axis: F.normalize without the learned scale.
pub fn unit_norm<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let denominator = x
        .clone()
        .mul(x.clone())
        .sum_dim(D - 1)
        .sqrt()
        .clamp_min(1e-12);
    x.div(denominator)
}

pub struct SourceNorm<B: Backend> {
    /// `sqrt(dim) * gamma`, precomputed once.
    scale: Tensor<B, 1>,
}
impl<B: Backend> SourceNorm<B> {
    pub fn load(
        weights: &mut Weights<'_>,
        key: &str,
        dim: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let scale = Self::scale(weights, key, dim)?;
        Ok(Self {
            scale: Tensor::from_data(TensorData::new(scale, [dim]), device),
        })
    }
    /// Host `sqrt(dim) * gamma`, for folding into the following projection.
    pub fn scale(weights: &mut Weights<'_>, key: &str, dim: usize) -> Result<Vec<f32>> {
        let root = (dim as f32).sqrt();
        Ok(weights
            .take(&format!("{key}.gamma"), &[dim])?
            .into_iter()
            .map(|g| g * root)
            .collect())
    }
    /// sqrt(dim)*gamma*x / max(sqrt(sum(x²)), 1e-12), matching F.normalize.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        unit_norm(x).mul(self.scale.clone().unsqueeze())
    }
}

/// Copy a strided view into a dense tensor once, before several consumers read it.
/// CubeCL keeps same-shape reshapes as views, so this round-trips through 1-D.
pub fn dense<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dims = x.dims();
    x.reshape([dims.iter().product::<usize>()]).reshape(dims)
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionPlan {
    /// Explicit tiles; `None` picks the largest exact tile within `score_budget`.
    pub query_tile: Option<usize>,
    pub group_tile: Option<usize>,
    /// Bytes allowed for one materialized score tile.
    pub score_budget: usize,
    /// Preserve batched projections; alternative GEMM folding is an ablation.
    pub batched_linear: bool,
    /// CPU only: run independent sequence groups of each axis transformer on this many
    /// host threads. Burn CPU element-wise kernels are single-threaded; 1 disables.
    pub host_threads: usize,
    /// HyperACE: ungrouped SegmModel convolutions as patch gather + GEMM.
    pub conv_gemm: bool,
    /// Single-pass attention kernel where the backend has one ([`crate::fused`]); scores
    /// are not materialized and the tiles above are unused.
    pub fused_attention: bool,
    /// Transformer projections through the hand-written GEMM ([`crate::fused::linear`]) where
    /// the backend and shape allow; rows are always folded then.
    pub custom_gemm: bool,
}
impl AttentionPlan {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.query_tile != Some(0)
                && self.group_tile != Some(0)
                && self.score_budget > 0
                && self.host_threads > 0,
            "attention tiles, score budget and host threads must be positive"
        );
        Ok(())
    }
    /// Effective `(group_tile, query_tile)` for `groups` sequences of `tokens`.
    /// Queries stay whole when one group fits; softmax always covers every key.
    pub fn tiles(self, groups: usize, tokens: usize, heads: usize) -> (usize, usize) {
        let row = heads * tokens * 4;
        let query = self
            .query_tile
            .unwrap_or_else(|| (self.score_budget / row).max(1))
            .clamp(1, tokens);
        let group = self
            .group_tile
            .unwrap_or_else(|| (self.score_budget / (row * query)).max(1))
            .clamp(1, groups);
        (group, query)
    }
    pub fn score_bytes(self, groups: usize, tokens: usize, heads: usize) -> usize {
        let (group, query) = self.tiles(groups, tokens, heads);
        group * heads * query * tokens * 4
    }
    /// Worker count over `groups` and the plan each worker runs; concurrent workers
    /// split the score budget, so total score memory stays within it.
    pub fn per_worker(self, groups: usize) -> (usize, Self) {
        let parts = self.host_threads.clamp(1, groups.max(1));
        let plan = Self {
            score_budget: (self.score_budget / parts).max(1),
            ..self
        };
        (parts, plan)
    }
    /// Effective per-worker `(group_tile, query_tile)` and bytes of one score tile.
    pub fn worker_tiles(self, groups: usize, tokens: usize, heads: usize) -> (usize, usize, usize) {
        let (parts, plan) = self.per_worker(groups);
        let groups = groups.div_ceil(parts);
        let (group, query) = plan.tiles(groups, tokens, heads);
        (group, query, plan.score_bytes(groups, tokens, heads))
    }
}
impl Default for AttentionPlan {
    fn default() -> Self {
        Self {
            query_tile: Some(128),
            group_tile: Some(4),
            score_budget: 512 << 20,
            batched_linear: true,
            host_threads: 1,
            conv_gemm: false,
            fused_attention: false,
            custom_gemm: false,
        }
    }
}

/// Interleaved-pair RoPE tables tiled over heads for `tokens` positions:
/// `rope(x) = x·cos + swap_pairs(x)·sin±`, the pair sign folded into `sin±`.
/// `scale` multiplies both tables (RoPE is linear), carrying the query 1/sqrt(d).
pub struct RopeTables<B: Backend> {
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
    swap: Tensor<B, 1, Int>,
}
impl<B: Backend> RopeTables<B> {
    pub fn new(freqs: &[f32], tokens: usize, heads: usize, scale: f32, device: &B::Device) -> Self {
        let width = heads * freqs.len() * 2;
        let mut cos = Vec::with_capacity(tokens * width);
        let mut sin = Vec::with_capacity(tokens * width);
        for t in 0..tokens {
            for _ in 0..heads {
                for &f in freqs {
                    let (s, c) = (t as f32 * f).sin_cos();
                    cos.extend([c * scale, c * scale]);
                    sin.extend([-s * scale, s * scale]);
                }
            }
        }
        let swap: Vec<i64> = (0..width as i64).map(|i| i ^ 1).collect();
        Self {
            cos: Tensor::from_data(TensorData::new(cos, [1, tokens * width]), device),
            sin: Tensor::from_data(TensorData::new(sin, [1, tokens * width]), device),
            swap: Tensor::from_data(TensorData::new(swap, [width]), device),
        }
    }
    /// `[groups, tokens, heads*head_dim]` rotated in place of the original layout.
    pub fn apply(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [groups, tokens, width] = x.dims();
        let swapped = x
            .clone()
            .reshape([groups * tokens, width])
            .select(1, self.swap.clone())
            .reshape([groups, tokens * width]);
        (x.reshape([groups, tokens * width]).mul(self.cos.clone()) + swapped.mul(self.sin.clone()))
            .reshape([groups, tokens, width])
    }
}

pub struct Attention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    gates: Linear<B>,
    output: Linear<B>,
    /// Host rotary frequencies; tables are built once per forward and axis.
    pub freqs: Vec<f32>,
    heads: usize,
    head_dim: usize,
}
impl<B: Backend> Attention<B> {
    fn load(w: &mut Weights<'_>, p: &str, c: &ModelConfig, device: &B::Device) -> Result<Self> {
        let inner = c.heads * c.head_dim;
        // The norm feeds only these projections, so sqrt(dim)·gamma folds into their input rows.
        let norm = SourceNorm::<B>::scale(w, &format!("{p}.norm"), c.dim)?;
        let qkv = w.take(&format!("{p}.to_qkv.weight"), &[3 * inner, c.dim])?;
        let gate_weight = w.take(&format!("{p}.to_gates.weight"), &[c.heads, c.dim])?;
        let gate_bias = w.take(&format!("{p}.to_gates.bias"), &[c.heads])?;
        Ok(Self {
            q: folded_linear(&qkv, c.dim, 0..inner, &norm, 1.0, None, device),
            k: folded_linear(&qkv, c.dim, inner..2 * inner, &norm, 1.0, None, device),
            v: folded_linear(&qkv, c.dim, 2 * inner..3 * inner, &norm, 1.0, None, device),
            gates: folded_linear(
                &gate_weight,
                c.dim,
                0..c.heads,
                &norm,
                1.0,
                Some(&gate_bias),
                device,
            ),
            output: w.linear(&format!("{p}.to_out.0"), inner, c.dim, false, device)?,
            freqs: w.take(&format!("{p}.rotary_embed.freqs"), &[c.head_dim / 2])?,
            heads: c.heads,
            head_dim: c.head_dim,
        })
    }

    fn forward(&self, x: Tensor<B, 3>, rope: &AxisRope<B>, plan: AttentionPlan) -> Tensor<B, 3> {
        let [groups, tokens, _] = x.dims();
        let (h, d) = (self.heads, self.head_dim);
        let x = unit_norm(x);
        let split = |t: Tensor<B, 3>| t.reshape([groups, tokens, h, d]);
        let q = split(rope.query.apply(linear3(&self.q, x.clone(), plan)));
        let k = split(rope.key.apply(linear3(&self.k, x.clone(), plan)));
        let v = split(linear3(&self.v, x.clone(), plan));
        let out = token_major_attention(q, k, v, plan);
        let gates = sigmoid(linear3(&self.gates, x, plan)).reshape([groups, tokens, h, 1]);
        let out = out.mul(gates).reshape([groups, tokens, h * d]);
        linear3(&self.output, out, plan)
    }
}

/// Query and key rotary tables of one axis; the query table carries 1/sqrt(d).
pub struct AxisRope<B: Backend> {
    pub query: RopeTables<B>,
    pub key: RopeTables<B>,
}
impl<B: Backend> AxisRope<B> {
    pub fn new(freqs: &[f32], tokens: usize, heads: usize, device: &B::Device) -> Self {
        let scale = ((freqs.len() * 2) as f32).sqrt().recip();
        Self {
            query: RopeTables::new(freqs, tokens, heads, scale, device),
            key: RopeTables::new(freqs, tokens, heads, 1.0, device),
        }
    }
}

/// Bound scores by groups and queries; each softmax still sees ALL keys.
/// This is exact global attention, independent of the audio chunk profile.
pub fn exact_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    plan: AttentionPlan,
) -> Tensor<B, 4> {
    let d = q.dims()[3];
    prescaled_attention(q.mul_scalar((d as f32).sqrt().recip()), k, v, plan)
}

/// [`prescaled_attention`] over `[groups, tokens, heads, d]` inputs, returning that layout.
fn token_major_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    plan: AttentionPlan,
) -> Tensor<B, 4> {
    if plan.fused_attention
        && let Some(out) = crate::fused::attention(&q, &k, &v)
    {
        return out;
    }
    let heads_major = |t: Tensor<B, 4>| dense(t.swap_dims(1, 2));
    prescaled_attention(heads_major(q), heads_major(k), heads_major(v), plan).swap_dims(1, 2)
}

/// `softmax(q·kᵀ)·v` with the 1/sqrt(d) scale already applied to `q`.
fn prescaled_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    plan: AttentionPlan,
) -> Tensor<B, 4> {
    let [groups, heads, tokens, _] = q.dims();
    let (group_tile, query_tile) = plan.tiles(groups, tokens, heads);
    let mut group_outputs = Vec::new();
    for g in (0..groups).step_by(group_tile) {
        let ng = group_tile.min(groups - g);
        let kg = k.clone().narrow(0, g, ng).swap_dims(2, 3);
        let vg = v.clone().narrow(0, g, ng);
        let qg = q.clone().narrow(0, g, ng);
        let mut query_outputs = Vec::new();
        for i in (0..tokens).step_by(query_tile) {
            let nq = query_tile.min(tokens - i);
            let scores = qg.clone().narrow(2, i, nq).matmul(kg.clone());
            query_outputs.push(softmax(scores, 3).matmul(vg.clone()));
        }
        group_outputs.push(cat(query_outputs, 2));
    }
    cat(group_outputs, 0)
}

/// `Tensor::cat` that skips the copy for a single part.
fn cat<B: Backend, const D: usize>(mut parts: Vec<Tensor<B, D>>, dim: usize) -> Tensor<B, D> {
    if parts.len() == 1 {
        parts.pop().expect("one part")
    } else {
        Tensor::cat(parts, dim)
    }
}

pub struct Transformer<B: Backend> {
    pub attention: Attention<B>,
    ff_in: Linear<B>,
    ff_out: Linear<B>,
    output_norm: Option<SourceNorm<B>>,
}
impl<B: Backend> Transformer<B> {
    pub fn load(
        w: &mut Weights<'_>,
        p: &str,
        c: &ModelConfig,
        normalized: bool,
        device: &B::Device,
    ) -> Result<Self> {
        let hidden = c.dim * c.ff_mult;
        let norm = SourceNorm::<B>::scale(w, &format!("{p}.layers.0.1.net.0"), c.dim)?;
        let ff_in = format!("{p}.layers.0.1.net.1");
        let weight = w.take(&format!("{ff_in}.weight"), &[hidden, c.dim])?;
        let bias = w.take(&format!("{ff_in}.bias"), &[hidden])?;
        Ok(Self {
            attention: Attention::load(w, &format!("{p}.layers.0.0"), c, device)?,
            ff_in: folded_linear(&weight, c.dim, 0..hidden, &norm, 1.0, Some(&bias), device),
            ff_out: w.linear(
                &format!("{p}.layers.0.1.net.4"),
                hidden,
                c.dim,
                true,
                device,
            )?,
            output_norm: if normalized {
                Some(SourceNorm::load(w, &format!("{p}.norm"), c.dim, device)?)
            } else {
                None
            },
        })
    }
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &AxisRope<B>,
        plan: AttentionPlan,
    ) -> Tensor<B, 3> {
        let groups = x.dims()[0];
        let (parts, plan) = plan.per_worker(groups);
        if parts <= 1 {
            return self.forward_groups(x, rope, plan);
        }
        // Every op is independent across groups (attention runs within one group),
        // so contiguous group slices run concurrently with identical arithmetic.
        let size = groups.div_ceil(parts);
        let outputs = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..groups)
                .step_by(size)
                .map(|g| {
                    let part = x.clone().narrow(0, g, size.min(groups - g));
                    scope.spawn(move || self.forward_groups(part, rope, plan))
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("transformer worker panicked"))
                .collect::<Vec<_>>()
        });
        Tensor::cat(outputs, 0)
    }
    fn forward_groups(
        &self,
        x: Tensor<B, 3>,
        rope: &AxisRope<B>,
        plan: AttentionPlan,
    ) -> Tensor<B, 3> {
        let x = self.attention.forward(x.clone(), rope, plan) + x;
        let hidden = gelu(linear3(&self.ff_in, unit_norm(x.clone()), plan));
        let y = linear3(&self.ff_out, hidden, plan) + x;
        if let Some(norm) = &self.output_norm {
            norm.forward(y)
        } else {
            y
        }
    }
}

/// Independent rows share one weight matrix. Folding them removes thousands of
/// small batched GEMMs without changing the attention or model context.
pub fn linear3<B: Backend>(
    linear: &Linear<B>,
    input: Tensor<B, 3>,
    plan: AttentionPlan,
) -> Tensor<B, 3> {
    let [groups, tokens, dim] = input.dims();
    if plan.custom_gemm {
        let bias = linear.bias.as_ref().map(|b| b.val());
        let rows = input.clone().reshape([groups * tokens, dim]);
        if let Some(output) = crate::fused::linear(&rows, &linear.weight.val(), bias.as_ref()) {
            let width = output.dims()[1];
            return output.reshape([groups, tokens, width]);
        }
    }
    if plan.batched_linear {
        return linear.forward(input);
    }
    let output = linear.forward(input.reshape([groups * tokens, dim]));
    let width = output.dims()[1];
    output.reshape([groups, tokens, width])
}
