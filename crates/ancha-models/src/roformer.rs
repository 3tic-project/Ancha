use crate::{config::ModelConfig, weights::Weights};
use anyhow::{Result, ensure};
use burn::{
    nn::Linear,
    tensor::{
        Tensor,
        activation::{gelu, sigmoid, softmax},
        backend::Backend,
    },
};

pub struct SourceNorm<B: Backend> {
    gamma: Tensor<B, 1>,
    dim: usize,
}
impl<B: Backend> SourceNorm<B> {
    pub fn load(
        weights: &mut Weights<'_>,
        key: &str,
        dim: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            gamma: weights.tensor(&format!("{key}.gamma"), [dim], device)?,
            dim,
        })
    }
    /// sqrt(dim)*gamma*x / max(sqrt(sum(x²)), 1e-12), matching F.normalize.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let denominator = x
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(1e-12);
        x.div(denominator)
            .mul_scalar((self.dim as f32).sqrt())
            .mul(self.gamma.clone().unsqueeze())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionPlan {
    pub query_tile: usize,
    pub group_tile: usize,
    /// Preserve batched projections; alternative GEMM folding is an ablation.
    pub batched_linear: bool,
}
impl AttentionPlan {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.query_tile > 0 && self.group_tile > 0,
            "attention tile sizes must be positive"
        );
        Ok(())
    }
    pub fn score_bytes(self, groups: usize, tokens: usize, heads: usize) -> usize {
        self.group_tile.min(groups) * heads * self.query_tile.min(tokens) * tokens * 4
    }
}
impl Default for AttentionPlan {
    fn default() -> Self {
        Self {
            query_tile: 128,
            group_tile: 4,
            batched_linear: true,
        }
    }
}

pub struct Rotary<B: Backend> {
    freqs: Tensor<B, 1>,
}
impl<B: Backend> Rotary<B> {
    pub fn load(
        w: &mut Weights<'_>,
        key: &str,
        head_dim: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            freqs: w.tensor(&format!("{key}.freqs"), [head_dim / 2], device)?,
        })
    }
    fn tables(&self, n: usize, d: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let pos = Tensor::<B, 1>::from_floats(
            (0..n).map(|i| i as f32).collect::<Vec<_>>().as_slice(),
            &self.freqs.device(),
        )
        .reshape([n, 1]);
        let angle = pos
            .matmul(self.freqs.clone().reshape([1, d / 2]))
            .reshape([n, d / 2, 1]);
        let angle = Tensor::cat(vec![angle.clone(), angle], 2).reshape([1, 1, n, d]);
        (angle.clone().cos(), angle.sin())
    }
    fn apply(x: Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, n, d] = x.dims();
        let pair = x.clone().reshape([b, h, n, d / 2, 2]);
        let rotated = Tensor::cat(
            vec![pair.clone().narrow(4, 1, 1).neg(), pair.narrow(4, 0, 1)],
            4,
        )
        .reshape([b, h, n, d]);
        x.mul(cos.clone()) + rotated.mul(sin.clone())
    }
}

pub struct Attention<B: Backend> {
    norm: SourceNorm<B>,
    qkv: Linear<B>,
    gates: Linear<B>,
    output: Linear<B>,
    rotary: Rotary<B>,
    heads: usize,
    head_dim: usize,
}
impl<B: Backend> Attention<B> {
    fn load(w: &mut Weights<'_>, p: &str, c: &ModelConfig, device: &B::Device) -> Result<Self> {
        let inner = c.heads * c.head_dim;
        Ok(Self {
            norm: SourceNorm::load(w, &format!("{p}.norm"), c.dim, device)?,
            qkv: w.linear(&format!("{p}.to_qkv"), c.dim, 3 * inner, false, device)?,
            gates: w.linear(&format!("{p}.to_gates"), c.dim, c.heads, true, device)?,
            output: w.linear(&format!("{p}.to_out.0"), inner, c.dim, false, device)?,
            rotary: Rotary::load(w, &format!("{p}.rotary_embed"), c.head_dim, device)?,
            heads: c.heads,
            head_dim: c.head_dim,
        })
    }

    fn forward(&self, x: Tensor<B, 3>, plan: AttentionPlan) -> Tensor<B, 3> {
        let [groups, tokens, _] = x.dims();
        let x = self.norm.forward(x);
        let qkv = linear3(&self.qkv, x.clone(), plan)
            .reshape([groups, tokens, 3, self.heads, self.head_dim])
            .permute([2, 0, 3, 1, 4]);
        let shape = [groups, self.heads, tokens, self.head_dim];
        let q = qkv.clone().narrow(0, 0, 1).reshape(shape);
        let k = qkv.clone().narrow(0, 1, 1).reshape(shape);
        let v = qkv.narrow(0, 2, 1).reshape(shape);
        let (cos, sin) = self.rotary.tables(tokens, self.head_dim);
        let q = Rotary::apply(q, &cos, &sin);
        let k = Rotary::apply(k, &cos, &sin);
        let out = exact_attention(q, k, v, plan);
        let gates = sigmoid(linear3(&self.gates, x, plan))
            .permute([0, 2, 1])
            .reshape([groups, self.heads, tokens, 1]);
        let out = out.mul(gates).permute([0, 2, 1, 3]).reshape([
            groups,
            tokens,
            self.heads * self.head_dim,
        ]);
        linear3(&self.output, out, plan)
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
    let [groups, _, tokens, d] = q.dims();
    let scale = (d as f32).sqrt().recip();
    let mut group_outputs = Vec::new();
    for g in (0..groups).step_by(plan.group_tile) {
        let ng = plan.group_tile.min(groups - g);
        let kg = k.clone().narrow(0, g, ng).swap_dims(2, 3);
        let vg = v.clone().narrow(0, g, ng);
        let qg = q.clone().narrow(0, g, ng);
        let mut query_outputs = Vec::new();
        for i in (0..tokens).step_by(plan.query_tile) {
            let nq = plan.query_tile.min(tokens - i);
            let scores = qg
                .clone()
                .narrow(2, i, nq)
                .matmul(kg.clone())
                .mul_scalar(scale);
            query_outputs.push(softmax(scores, 3).matmul(vg.clone()));
        }
        group_outputs.push(Tensor::cat(query_outputs, 2));
    }
    Tensor::cat(group_outputs, 0)
}

pub struct Transformer<B: Backend> {
    attention: Attention<B>,
    ff_norm: SourceNorm<B>,
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
        Ok(Self {
            attention: Attention::load(w, &format!("{p}.layers.0.0"), c, device)?,
            ff_norm: SourceNorm::load(w, &format!("{p}.layers.0.1.net.0"), c.dim, device)?,
            ff_in: w.linear(
                &format!("{p}.layers.0.1.net.1"),
                c.dim,
                c.dim * c.ff_mult,
                true,
                device,
            )?,
            ff_out: w.linear(
                &format!("{p}.layers.0.1.net.4"),
                c.dim * c.ff_mult,
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
    pub fn forward(&self, x: Tensor<B, 3>, plan: AttentionPlan) -> Tensor<B, 3> {
        let x = self.attention.forward(x.clone(), plan) + x;
        let hidden = gelu(linear3(&self.ff_in, self.ff_norm.forward(x.clone()), plan));
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
    if plan.batched_linear {
        return linear.forward(input);
    }
    let [groups, tokens, dim] = input.dims();
    let output = linear.forward(input.reshape([groups * tokens, dim]));
    let width = output.dims()[1];
    output.reshape([groups, tokens, width])
}
