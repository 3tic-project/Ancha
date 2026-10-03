//! Hand-written CubeCL kernels for the CUDA fusion backend behind backend-generic entry points.
//! Model code stays generic over `B: Backend`; each entry point returns `None` unless `B` is
//! `burn::backend::Cuda` and the shapes are supported, so callers keep their Burn implementation
//! as the fallback. The kernels are not dispatched on WGPU: CubeCL's WGPU target lacks the
//! shared-memory vector reinterpretation the GEMM relies on, and the output there was wrong.
#[cfg(feature = "cuda")]
mod attention;
#[cfg(feature = "cuda")]
mod conv;
#[cfg(feature = "cuda")]
mod gemm;
#[cfg(feature = "cuda")]
mod norm;

use burn::tensor::{Tensor, backend::Backend};

/// Head width the fused attention kernel is specialised for.
pub const ATTENTION_HEAD_DIM: usize = 64;

/// `softmax(q·kᵀ)·v` per `(group, head)` over `[groups, tokens, heads, head_dim]` tensors, with
/// `q` already scaled; the output keeps that layout. Scores are never materialized.
pub fn attention<B: Backend>(
    q: &Tensor<B, 4>,
    k: &Tensor<B, 4>,
    v: &Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    if q.dims()[3] != ATTENTION_HEAD_DIM || k.dims() != q.dims() || v.dims() != q.dims() {
        return None;
    }
    #[cfg(feature = "cuda")]
    {
        let [groups, tokens, heads, width] = q.dims();
        let flat = |t: &Tensor<B, 4>| t.clone().reshape([groups, tokens, heads * width]);
        cube::dispatch(
            cube::Kernel::Attention { heads },
            &[&flat(q), &flat(k), &flat(v)],
        )
    }
    #[cfg(not(feature = "cuda"))]
    None
}

/// The RoFormer attention core between the projections, from one packed projection
/// `[groups, tokens, columns]` holding (already rotated) q, k, v (`heads·64` columns each, from
/// column 0) and the per-head gate logits after them: `softmax(q·kᵀ)·v · sigmoid(gate)`.
/// Returns `[groups, tokens, heads, 64]`.
pub fn gated_attention<B: Backend>(packed: &Tensor<B, 3>, heads: usize) -> Option<Tensor<B, 4>> {
    if packed.dims()[2] < heads * (3 * ATTENTION_HEAD_DIM + 1) {
        return None;
    }
    #[cfg(feature = "cuda")]
    return cube::dispatch(cube::Kernel::GatedAttention { heads }, &[packed]);
    #[cfg(not(feature = "cuda"))]
    None
}

/// Work fused into the output pass of [`linear`], applied in field order.
pub struct Epilogue<'a, B: Backend> {
    /// `[m]`.
    pub bias: Option<&'a Tensor<B, 1>>,
    /// erf-GELU.
    pub gelu: bool,
    /// Interleaved rotary embedding: `[4, tokens, 64]` tables (query cos / signed sin carrying
    /// 1/sqrt(d), key cos / signed sin) and the query width `w`; columns `[0, w)` and `[w, 2w)`
    /// of row `r` are rotated as queries and keys of token `r % tokens`.
    pub rotary: Option<(&'a Tensor<B, 3>, usize)>,
    /// `[n, m]` added last.
    pub residual: Option<&'a Tensor<B, 2>>,
}
impl<B: Backend> Default for Epilogue<'_, B> {
    fn default() -> Self {
        Self {
            bias: None,
            gelu: false,
            rotary: None,
            residual: None,
        }
    }
}

/// Whether the 128-wide GEMM tile covers `[n, k]·[k, m]` with no padding: `k % 8 == 0` and
/// `m % 128 == 0`. Bare products with a narrower `m` still run; see [`linear`].
pub fn linear_supported(k: usize, m: usize) -> bool {
    k > 0 && k.is_multiple_of(8) && m > 0 && m.is_multiple_of(128)
}

fn epilogue_empty<B: Backend>(epilogue: &Epilogue<'_, B>) -> bool {
    epilogue.bias.is_none()
        && !epilogue.gelu
        && epilogue.rotary.is_none()
        && epilogue.residual.is_none()
}

/// `x [n, k] · weight [k, m]` in Burn's `Linear` layout, then the [`Epilogue`].
///
/// The kernel tile is 128 columns. `m` must be a multiple of 128 when an epilogue is present.
/// A bare product (`k % 8 == 0`, any positive `m`) zero-pads the weight to that tile and
/// drops the padding, so MDX23C's narrower TDF layers stay on this kernel instead of a
/// generic matmul. Returns `None` when the shape or epilogue does not fit.
pub fn linear<B: Backend>(
    x: &Tensor<B, 2>,
    weight: &Tensor<B, 2>,
    epilogue: Epilogue<'_, B>,
) -> Option<Tensor<B, 2>> {
    let [n, k] = x.dims();
    let [rows, m] = weight.dims();
    if rows != k || k == 0 || !k.is_multiple_of(8) || m == 0 {
        return None;
    }
    // Padding would move epilogue columns; those paths keep the aligned tile.
    if !m.is_multiple_of(128) && !epilogue_empty(&epilogue) {
        return None;
    }
    if let Some((table, width)) = epilogue.rotary {
        let [four, tokens, dim] = table.dims();
        if four != 4
            || dim != ATTENTION_HEAD_DIM
            || n % tokens != 0
            || width % 64 != 0
            || 2 * width > m
        {
            return None;
        }
    }
    if epilogue.residual.is_some_and(|r| r.dims() != [n, m]) {
        return None;
    }
    #[cfg(feature = "cuda")]
    {
        let column_pad = (128 - m % 128) % 128;
        let padded;
        let weight = if column_pad == 0 {
            weight
        } else {
            let device = weight.device();
            padded = Tensor::cat(
                vec![
                    weight.clone(),
                    Tensor::<B, 2>::zeros([k, column_pad], &device),
                ],
                1,
            );
            &padded
        };
        // Every operand as a matrix so they share one rank.
        let bias = epilogue.bias.map(|b| b.clone().unsqueeze::<2>());
        let rotary = epilogue.rotary.map(|(table, width)| {
            let [_, tokens, dim] = table.dims();
            (table.clone().reshape([4 * tokens, dim]), tokens, width)
        });
        let mut inputs = vec![x, weight];
        inputs.extend(&bias);
        inputs.extend(epilogue.residual);
        inputs.extend(rotary.as_ref().map(|(table, ..)| table));
        let kernel = cube::Kernel::Linear {
            bias: bias.is_some(),
            gelu: epilogue.gelu,
            residual: epilogue.residual.is_some(),
            rotary: rotary.as_ref().map(|(_, tokens, width)| (*tokens, *width)),
        };
        let out = cube::dispatch(kernel, &inputs)?;
        if column_pad == 0 {
            Some(out)
        } else {
            Some(out.narrow(1, 0, m))
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (n, epilogue);
        None
    }
}

/// `conv2d(x, weight, bias)` for NCHW `x`, `weight [o, c, kh, kw]`, `groups == 1` and no
/// dilation, gathering patches while loading instead of materializing them; `None` unless
/// `c·kh·kw % 8 == 0`.
pub fn conv2d<B: Backend>(
    x: &Tensor<B, 4>,
    weight: &Tensor<B, 4>,
    bias: Option<&Tensor<B, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Option<Tensor<B, 4>> {
    let [_, channels, height, width] = x.dims();
    let [out_channels, c, kh, kw] = weight.dims();
    let depth = c * kh * kw;
    if c != channels
        || depth == 0
        || depth % 8 != 0
        || height + 2 * padding[0] < kh
        || width + 2 * padding[1] < kw
        || stride.contains(&0)
    {
        return None;
    }
    #[cfg(feature = "cuda")]
    {
        // Every operand as rank 4: the weight flattened to [o, depth, 1, 1], the bias to [o, 1, 1, 1].
        let weight = weight.clone().reshape([out_channels, depth, 1, 1]);
        let bias = bias.map(|b| b.clone().reshape([out_channels, 1, 1, 1]));
        let mut inputs = vec![x, &weight];
        inputs.extend(&bias);
        let kernel = cube::Kernel::Conv {
            kernel: [kh, kw],
            stride,
            padding,
            bias: bias.is_some(),
        };
        cube::dispatch(kernel, &inputs)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (out_channels, bias);
        None
    }
}

/// NCHW instance norm, `gamma` / `beta` shaped `[1, c, 1, 1]`. `None` unless `B` is CUDA.
/// Variance is the biased spatial mean of squares, with `eps` inside the square root.
pub fn instance_norm<B: Backend>(
    x: &Tensor<B, 4>,
    gamma: &Tensor<B, 4>,
    beta: &Tensor<B, 4>,
    eps: f32,
) -> Option<Tensor<B, 4>> {
    let [n, c, h, w] = x.dims();
    if n == 0
        || c == 0
        || h == 0
        || w == 0
        || gamma.dims() != [1, c, 1, 1]
        || beta.dims() != [1, c, 1, 1]
        || !eps.is_finite()
    {
        return None;
    }
    #[cfg(feature = "cuda")]
    {
        cube::dispatch(cube::Kernel::Norm { eps }, &[x, gamma, beta])
    }
    #[cfg(not(feature = "cuda"))]
    None
}

/// Whether backend `B` has the hand-written kernels (shape limits aside).
pub fn available<B: Backend>() -> bool {
    #[cfg(feature = "cuda")]
    return std::any::TypeId::of::<B>() == std::any::TypeId::of::<burn::backend::Cuda>();
    #[cfg(not(feature = "cuda"))]
    false
}

#[cfg(feature = "cuda")]
mod cube {
    use burn::tensor::{Shape, Tensor, TensorPrimitive, backend::Backend, ops::FloatTensor};
    use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};
    use std::{any::Any, marker::PhantomData};

    #[derive(Clone, Copy, Debug)]
    pub enum Kernel {
        /// `[q, k, v]` as `[groups, tokens, heads·64]` → `[groups, tokens, heads, 64]`.
        Attention { heads: usize },
        /// `[packed]`, see [`super::gated_attention`].
        GatedAttention { heads: usize },
        /// `[x, w, bias?, residual?, rotary table?]` → `[n, m]`; rotary is `(tokens, width)`.
        Linear {
            bias: bool,
            gelu: bool,
            residual: bool,
            rotary: Option<(usize, usize)>,
        },
        /// `[x, weight [o, depth, 1, 1], bias [o, 1, 1, 1]?]` → `[b, o, oh, ow]`.
        Conv {
            kernel: [usize; 2],
            stride: [usize; 2],
            padding: [usize; 2],
            bias: bool,
        },
        /// `[x, gamma, beta]` NCHW → normalized `x`. `eps` is inside the root.
        Norm { eps: f32 },
    }
    impl Kernel {
        fn id(self) -> &'static str {
            match self {
                Self::Attention { .. } => "ancha_flash_attention",
                Self::GatedAttention { .. } => "ancha_gated_attention",
                Self::Linear { .. } => "ancha_gemm",
                Self::Conv { .. } => "ancha_conv2d",
                Self::Norm { .. } => "ancha_instance_norm",
            }
        }
        fn output_shape(self, inputs: &[Shape]) -> Shape {
            match self {
                Self::Attention { heads } | Self::GatedAttention { heads } => {
                    [inputs[0][0], inputs[0][1], heads, super::ATTENTION_HEAD_DIM].into()
                }
                Self::Linear { .. } => [inputs[0][0], inputs[1][1]].into(),
                Self::Conv {
                    kernel,
                    stride,
                    padding,
                    ..
                } => {
                    let out =
                        |i: usize| (inputs[0][2 + i] + 2 * padding[i] - kernel[i]) / stride[i] + 1;
                    [inputs[0][0], inputs[1][0], out(0), out(1)].into()
                }
                Self::Norm { .. } => inputs[0].clone(),
            }
        }
    }

    /// Backends that launch the kernels on their own float primitives.
    pub trait Kernels: Backend {
        fn launch(kernel: Kernel, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self>;
    }

    impl<R: CubeRuntime, BT: BoolElement> Kernels for CubeBackend<R, f32, i32, BT> {
        fn launch(kernel: Kernel, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
            use super::attention::{Columns, launch};
            let mut inputs = inputs.into_iter();
            let mut next = || inputs.next().expect("kernel input");
            match kernel {
                Kernel::Attention { heads } => {
                    let columns = Columns {
                        q: 0,
                        k: 0,
                        v: 0,
                        gate: None,
                    };
                    launch([next(), next(), next()], heads, columns)
                }
                Kernel::GatedAttention { heads } => {
                    let packed = next();
                    let inner = heads * super::ATTENTION_HEAD_DIM;
                    let columns = Columns {
                        q: 0,
                        k: inner,
                        v: 2 * inner,
                        gate: Some(3 * inner),
                    };
                    launch([packed.clone(), packed.clone(), packed], heads, columns)
                }
                Kernel::Linear {
                    bias,
                    gelu,
                    residual,
                    rotary,
                } => {
                    let (x, w) = (next(), next());
                    let operands = super::gemm::Operands {
                        bias: bias.then(&mut next),
                        gelu,
                        residual: residual.then(&mut next),
                        rotary: rotary.map(|(tokens, width)| (next(), tokens, width)),
                    };
                    super::gemm::launch(x, w, operands)
                }
                Kernel::Conv {
                    kernel,
                    stride,
                    padding,
                    bias,
                } => {
                    let (x, w) = (next(), next());
                    super::conv::launch(x, w, bias.then(&mut next), kernel, stride, padding)
                }
                Kernel::Norm { eps } => {
                    let (x, gamma, beta) = (next(), next(), next());
                    super::norm::launch(x, gamma, beta, eps)
                }
            }
        }
    }

    #[derive(Debug)]
    struct CustomOp<B> {
        kernel: Kernel,
        desc: CustomOpIr,
        backend: PhantomData<B>,
    }

    impl<B: FusionBackend + Kernels> Operation<B::FusionRuntime> for CustomOp<B> {
        fn execute(
            &self,
            handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
        ) {
            let inputs = self
                .desc
                .inputs
                .iter()
                .map(|t| handles.get_float_tensor::<B>(t))
                .collect();
            let out = B::launch(self.kernel, inputs);
            handles.register_float_tensor::<B>(&self.desc.outputs[0].id, out);
        }
    }

    /// Queued on the fusion stream like any other operation; fused blocks end before it.
    impl<B: FusionBackend + Kernels> Kernels for Fusion<B> {
        fn launch(kernel: Kernel, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
            let client = inputs[0].client.clone();
            let streams = OperationStreams::with_inputs(inputs.iter());
            let shapes: Vec<Shape> = inputs.iter().map(|t| t.shape.clone()).collect();
            let out = TensorIr::uninit(
                client.create_empty_handle(),
                kernel.output_shape(&shapes),
                inputs[0].dtype,
            );
            let inputs: Vec<TensorIr> = inputs.into_iter().map(|t| t.into_ir()).collect();
            let desc = CustomOpIr::new(kernel.id(), &inputs, &[out]);
            let op = CustomOp::<B> {
                kernel,
                desc: desc.clone(),
                backend: PhantomData,
            };
            client
                .register(streams, OperationIr::Custom(desc), op)
                .pop()
                .expect("custom kernel output")
        }
    }

    /// Runs `kernel` when `B` is the CUDA fusion backend.
    pub fn dispatch<B: Backend, const D: usize, const O: usize>(
        kernel: Kernel,
        inputs: &[&Tensor<B, D>],
    ) -> Option<Tensor<B, O>> {
        on::<B, burn::backend::Cuda, D, O>(kernel, inputs)
    }

    fn on<B: Backend, C: Kernels, const D: usize, const O: usize>(
        kernel: Kernel,
        inputs: &[&Tensor<B, D>],
    ) -> Option<Tensor<B, O>> {
        let inputs = inputs
            .iter()
            .map(|&t| {
                (t as &dyn Any)
                    .downcast_ref::<Tensor<C, D>>()
                    .map(|t| t.clone().into_primitive().tensor())
            })
            .collect::<Option<Vec<_>>>()?;
        let out = Tensor::<C, O>::from_primitive(TensorPrimitive::Float(C::launch(kernel, inputs)));
        (Box::new(out) as Box<dyn Any>)
            .downcast::<Tensor<B, O>>()
            .ok()
            .map(|t| *t)
    }
}
