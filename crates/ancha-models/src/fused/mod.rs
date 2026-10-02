//! Hand-written CubeCL kernels for the CUDA fusion backend behind backend-generic entry points.
//! Model code stays generic over `B: Backend`; each entry point returns `None` unless `B` is
//! `burn::backend::Cuda` and the shapes are supported, so callers keep their Burn implementation
//! as the fallback. The kernels are not dispatched on WGPU: CubeCL's WGPU target lacks the
//! shared-memory vector reinterpretation the GEMM relies on, and the output there was wrong.
#[cfg(feature = "cuda")]
mod attention;
#[cfg(feature = "cuda")]
mod gemm;

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

/// Whether [`linear`] handles `[n, k]·[k, m]`: `k % 8 == 0` and `m % 128 == 0`.
pub fn linear_supported(k: usize, m: usize) -> bool {
    k > 0 && k.is_multiple_of(8) && m > 0 && m.is_multiple_of(128)
}

/// `x [n, k] · weight [k, m]` in Burn's `Linear` layout, then the [`Epilogue`]; `None` unless
/// [`linear_supported`] and the epilogue operands fit.
pub fn linear<B: Backend>(
    x: &Tensor<B, 2>,
    weight: &Tensor<B, 2>,
    epilogue: Epilogue<'_, B>,
) -> Option<Tensor<B, 2>> {
    let [n, k] = x.dims();
    let [rows, m] = weight.dims();
    if rows != k || !linear_supported(k, m) {
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
        cube::dispatch(kernel, &inputs)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = epilogue;
        None
    }
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
    }
    impl Kernel {
        fn id(self) -> &'static str {
            match self {
                Self::Attention { .. } => "ancha_flash_attention",
                Self::GatedAttention { .. } => "ancha_gated_attention",
                Self::Linear { .. } => "ancha_gemm",
            }
        }
        fn output_shape(self, inputs: &[Shape]) -> Shape {
            match self {
                Self::Attention { heads } | Self::GatedAttention { heads } => {
                    [inputs[0][0], inputs[0][1], heads, super::ATTENTION_HEAD_DIM].into()
                }
                Self::Linear { .. } => [inputs[0][0], inputs[1][1]].into(),
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
