//! PyTorch-compatible spatial primitives shared by the HyperACE adapter.
use crate::weights::Weights;
use anyhow::Result;
use burn::tensor::{
    Int, Tensor, TensorData,
    activation::sigmoid,
    backend::Backend,
    module::conv2d,
    ops::{ConvOptions, PadMode},
};

pub(crate) fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * sigmoid(x)
}

/// Ungrouped, undilated 2-D convolution as a patch gather plus one GEMM.
/// Without cooperative-matrix units CubeCL runs direct convolution, several times
/// slower than its tuned GEMM; the arithmetic is the same sum of products. On CUDA the
/// implicit-GEMM kernel in [`crate::fused::conv2d`] gathers patches on the fly instead.
/// Returns `None` when the weight and input disagree so callers keep the backend path.
pub fn conv2d_gemm<B: Backend>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Option<Tensor<B, 4>> {
    let [b, c, h, w] = x.dims();
    let [o, ci, kh, kw] = weight.dims();
    if c != ci || h + 2 * padding[0] < kh || w + 2 * padding[1] < kw {
        return None;
    }
    if let Some(y) = crate::fused::conv2d(&x, &weight, bias.as_ref(), stride, padding) {
        return Some(y);
    }
    if b > 1 {
        // One patch matrix at a time: a 3×3 layer expands its input 9×, and several
        // batch items at once exceeded 8 GB devices in practice.
        let items = (0..b)
            .map(|i| {
                let item = x.clone().narrow(0, i, 1);
                conv2d_gemm(item, weight.clone(), bias.clone(), stride, padding)
            })
            .collect::<Option<Vec<_>>>()?;
        return Some(Tensor::cat(items, 0));
    }
    let k = c * kh * kw;
    let x = if padding == [0, 0] {
        x
    } else {
        let [ph, pw] = padding;
        x.pad([(ph, ph), (pw, pw)], PadMode::Constant(0.0))
    };
    // Sliding windows are strided views; the reshape materializes the patch matrix once,
    // in (channel, ky, kx) order to match the flattened `[out, in, kh, kw]` weight.
    let windows: Tensor<B, 6> = x.unfold::<5, _>(2, kh, stride[0]).unfold(3, kw, stride[1]);
    let [_, _, oh, ow, _, _] = windows.dims();
    let patches = windows.permute([0, 1, 4, 5, 2, 3]).reshape([b, k, oh * ow]);
    let y = weight.reshape([1, o, k]).matmul(patches);
    let y = match bias {
        Some(bias) => y + bias.reshape([1, o, 1]),
        None => y,
    };
    Some(y.reshape([b, o, oh, ow]))
}

pub struct InstanceNorm<B: Backend> {
    gamma: Tensor<B, 4>,
    beta: Tensor<B, 4>,
    epsilon: f32,
}
impl<B: Backend> InstanceNorm<B> {
    pub(crate) fn load(w: &mut Weights<'_>, p: &str, c: usize, d: &B::Device) -> Result<Self> {
        Self::load_with_epsilon(w, p, c, 1e-8, d)
    }
    pub(crate) fn load_with_epsilon(
        w: &mut Weights<'_>,
        p: &str,
        c: usize,
        epsilon: f32,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            gamma: w
                .tensor::<B, 1>(&format!("{p}.weight"), [c], d)?
                .reshape([1, c, 1, 1]),
            beta: w
                .tensor::<B, 1>(&format!("{p}.bias"), [c], d)?
                .reshape([1, c, 1, 1]),
            epsilon,
        })
    }
    pub fn from_affine(gamma: Tensor<B, 1>, beta: Tensor<B, 1>) -> Self {
        Self::from_affine_with_epsilon(gamma, beta, 1e-8)
    }
    pub fn from_affine_with_epsilon(gamma: Tensor<B, 1>, beta: Tensor<B, 1>, epsilon: f32) -> Self {
        let c = gamma.dims()[0];
        Self {
            gamma: gamma.reshape([1, c, 1, 1]),
            beta: beta.reshape([1, c, 1, 1]),
            epsilon,
        }
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mean = x.clone().mean_dim(2).mean_dim(3);
        let centered = x - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2).mean_dim(3);
        centered / variance.add_scalar(self.epsilon).sqrt() * self.gamma.clone() + self.beta.clone()
    }
    /// One reduction axis over H*W. MDX23C uses PyTorch's biased spatial variance, eps=1e-5.
    pub fn forward_flattened(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let x = x.reshape([b, c, h * w]);
        let centered = x.clone() - x.mean_dim(2);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        (centered / variance.add_scalar(self.epsilon).sqrt()
            * self.gamma.clone().reshape([1, c, 1])
            + self.beta.clone().reshape([1, c, 1]))
        .reshape([b, c, h, w])
    }
}

/// ConvTranspose2d with kernel=stride, zero padding, dilation=1 and groups=1.
/// Each input position produces a disjoint spatial block, so one GEMM plus a shuffle is exact.
pub fn conv_transpose2d_gemm<B: Backend>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    let [b, ci, h, w] = x.dims();
    let [wi, co, kh, kw] = weight.dims();
    if ci != wi || kh == 0 || kw == 0 {
        return None;
    }
    let rows = x.permute([0, 2, 3, 1]).reshape([b * h * w, ci]);
    let weight = weight.reshape([ci, co * kh * kw]);
    let y = crate::fused::linear(&rows, &weight, crate::fused::Epilogue::default())
        .unwrap_or_else(|| rows.matmul(weight));
    Some(
        y.reshape([b, h, w, co, kh, kw])
            .permute([0, 3, 1, 4, 2, 5])
            .reshape([b, co, h * kh, w * kw]),
    )
}

pub(crate) struct Conv<B: Backend> {
    weight: Tensor<B, 4>,
    stride: [usize; 2],
    padding: [usize; 2],
    groups: usize,
}
impl<B: Backend> Conv<B> {
    pub(crate) fn load(
        w: &mut Weights<'_>,
        p: &str,
        channels: [usize; 2],
        kernel: usize,
        stride: [usize; 2],
        groups: usize,
        d: &B::Device,
    ) -> Result<Self> {
        let [input, output] = channels;
        Ok(Self {
            weight: w.tensor(
                &format!("{p}.weight"),
                [output, input / groups, kernel, kernel],
                d,
            )?,
            stride,
            padding: [kernel / 2; 2],
            groups,
        })
    }
    pub(crate) fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        let gemm = (gemm && self.groups == 1)
            .then(|| {
                conv2d_gemm(
                    x.clone(),
                    self.weight.clone(),
                    None,
                    self.stride,
                    self.padding,
                )
            })
            .flatten();
        gemm.unwrap_or_else(|| {
            conv2d(
                x,
                self.weight.clone(),
                None,
                ConvOptions::new(self.stride, self.padding, [1, 1], self.groups),
            )
        })
    }
}

/// Separable half-pixel bilinear resize, align_corners=false, clamped edges.
/// Burn backends differ in interpolation conventions; explicit indices avoid that ambiguity.
pub fn resize<B: Backend>(x: Tensor<B, 4>, size: [usize; 2]) -> Tensor<B, 4> {
    let mut x = x;
    for (axis, &target) in [2, 3].iter().zip(&size) {
        let source = x.dims()[*axis];
        if source == target {
            continue;
        }
        let mut lower = Vec::with_capacity(target);
        let mut upper = Vec::with_capacity(target);
        let mut weights = Vec::with_capacity(target);
        for i in 0..target {
            let pos = ((i as f64 + 0.5) * source as f64 / target as f64 - 0.5).max(0.0);
            let lo = (pos.floor() as usize).min(source - 1);
            lower.push(lo as i64);
            upper.push((lo + 1).min(source - 1) as i64);
            weights.push((pos - lo as f64) as f32);
        }
        let device = x.device();
        let low = Tensor::<B, 1, Int>::from_data(TensorData::new(lower, [target]), &device);
        let high = Tensor::<B, 1, Int>::from_data(TensorData::new(upper, [target]), &device);
        let mut shape = [1; 4];
        shape[*axis] = target;
        let weight = Tensor::<B, 1>::from_floats(weights.as_slice(), &device).reshape(shape);
        let a = x.clone().select(*axis, low);
        let b = x.select(*axis, high);
        x = a.clone() + (b - a) * weight;
    }
    x
}

/// Frequency-only shuffle: source channel order is (output_channel, phase).
pub fn frequency_shuffle<B: Backend>(x: Tensor<B, 4>, scale: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    x.reshape([b, c / scale, scale, h, w])
        .permute([0, 1, 3, 4, 2])
        .reshape([b, c / scale, h, w * scale])
}
