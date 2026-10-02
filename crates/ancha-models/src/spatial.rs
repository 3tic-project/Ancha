//! PyTorch-compatible spatial primitives shared by the HyperACE adapter.
use crate::weights::Weights;
use anyhow::Result;
use burn::tensor::{
    Int, Tensor, TensorData, activation::sigmoid, backend::Backend, module::conv2d,
    ops::ConvOptions,
};

pub(crate) fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * sigmoid(x)
}

pub struct InstanceNorm<B: Backend> {
    gamma: Tensor<B, 4>,
    beta: Tensor<B, 4>,
}
impl<B: Backend> InstanceNorm<B> {
    pub(crate) fn load(w: &mut Weights<'_>, p: &str, c: usize, d: &B::Device) -> Result<Self> {
        Ok(Self {
            gamma: w
                .tensor::<B, 1>(&format!("{p}.weight"), [c], d)?
                .reshape([1, c, 1, 1]),
            beta: w
                .tensor::<B, 1>(&format!("{p}.bias"), [c], d)?
                .reshape([1, c, 1, 1]),
        })
    }
    pub fn from_affine(gamma: Tensor<B, 1>, beta: Tensor<B, 1>) -> Self {
        let c = gamma.dims()[0];
        Self {
            gamma: gamma.reshape([1, c, 1, 1]),
            beta: beta.reshape([1, c, 1, 1]),
        }
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mean = x.clone().mean_dim(2).mean_dim(3);
        let centered = x - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2).mean_dim(3);
        centered / variance.add_scalar(1e-8).sqrt() * self.gamma.clone() + self.beta.clone()
    }
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
    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        conv2d(
            x,
            self.weight.clone(),
            None,
            ConvOptions::new(self.stride, self.padding, [1, 1], self.groups),
        )
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
