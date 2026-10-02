//! Hand-written CubeCL kernels behind backend-generic entry points. Model code stays generic over
//! `B: Backend`; each entry point returns `None` unless `B` is a fusion GPU backend with the kernel
//! and the shapes are supported, so callers keep their Burn implementation as the fallback.
#[cfg(feature = "cube")]
mod attention;
#[cfg(feature = "cube")]
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
    #[cfg(feature = "cube")]
    return cube::dispatch(cube::Kernel::Attention, &[q, k, v]);
    #[cfg(not(feature = "cube"))]
    None
}

/// `x [n, k] · weight [k, m] (+ bias [m])` in Burn's `Linear` layout; `None` unless
/// `k % 8 == 0` and `m % 128 == 0`.
pub fn linear<B: Backend>(
    x: &Tensor<B, 2>,
    weight: &Tensor<B, 2>,
    bias: Option<&Tensor<B, 1>>,
) -> Option<Tensor<B, 2>> {
    let [_, k] = x.dims();
    let [rows, m] = weight.dims();
    if rows != k || k == 0 || k % 8 != 0 || m == 0 || m % 128 != 0 {
        return None;
    }
    #[cfg(feature = "cube")]
    {
        // As [1, m] so every operand has the same rank.
        let bias = bias.map(|b| b.clone().unsqueeze::<2>());
        match &bias {
            Some(b) => cube::dispatch(cube::Kernel::LinearBias, &[x, weight, b]),
            None => cube::dispatch(cube::Kernel::Linear, &[x, weight]),
        }
    }
    #[cfg(not(feature = "cube"))]
    {
        let _ = bias;
        None
    }
}

/// Whether backend `B` has the hand-written kernels (shape limits aside).
pub fn available<B: Backend>() -> bool {
    #[cfg(feature = "cube")]
    {
        let id = std::any::TypeId::of::<B>();
        #[cfg(feature = "cuda")]
        if id == std::any::TypeId::of::<burn::backend::Cuda>() {
            return true;
        }
        #[cfg(feature = "wgpu")]
        if id == std::any::TypeId::of::<burn::backend::Wgpu>() {
            return true;
        }
    }
    false
}

#[cfg(feature = "cube")]
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
        /// `[q, k, v]` → attention output shaped like `q`.
        Attention,
        /// `[x, w]` → `x·w`.
        Linear,
        /// `[x, w, bias [1, m]]` → `x·w + bias`.
        LinearBias,
    }
    impl Kernel {
        fn id(self) -> &'static str {
            match self {
                Self::Attention => "ancha_flash_attention",
                Self::Linear => "ancha_gemm",
                Self::LinearBias => "ancha_gemm_bias",
            }
        }
        fn output_shape(self, inputs: &[Shape]) -> Shape {
            match self {
                Self::Attention => inputs[0].clone(),
                Self::Linear | Self::LinearBias => [inputs[0][0], inputs[1][1]].into(),
            }
        }
    }

    /// Backends that launch the kernels on their own float primitives.
    pub trait Kernels: Backend {
        fn launch(kernel: Kernel, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self>;
    }

    impl<R: CubeRuntime, BT: BoolElement> Kernels for CubeBackend<R, f32, i32, BT> {
        fn launch(kernel: Kernel, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
            let mut inputs = inputs.into_iter();
            let mut next = || inputs.next().expect("kernel input");
            match kernel {
                Kernel::Attention => super::attention::launch(next(), next(), next()),
                Kernel::Linear => super::gemm::launch(next(), next(), None),
                Kernel::LinearBias => super::gemm::launch(next(), next(), Some(next())),
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

    /// Runs `kernel` when `B` is one of the backends built with the kernels.
    pub fn dispatch<B: Backend, const D: usize>(
        kernel: Kernel,
        inputs: &[&Tensor<B, D>],
    ) -> Option<Tensor<B, D>> {
        #[cfg(feature = "cuda")]
        if let Some(out) = on::<B, burn::backend::Cuda, D>(kernel, inputs) {
            return Some(out);
        }
        #[cfg(feature = "wgpu")]
        if let Some(out) = on::<B, burn::backend::Wgpu, D>(kernel, inputs) {
            return Some(out);
        }
        None
    }

    fn on<B: Backend, C: Kernels, const D: usize>(
        kernel: Kernel,
        inputs: &[&Tensor<B, D>],
    ) -> Option<Tensor<B, D>> {
        let inputs = inputs
            .iter()
            .map(|&t| {
                (t as &dyn Any)
                    .downcast_ref::<Tensor<C, D>>()
                    .map(|t| t.clone().into_primitive().tensor())
            })
            .collect::<Option<Vec<_>>>()?;
        let out = Tensor::<C, D>::from_primitive(TensorPrimitive::Float(C::launch(kernel, inputs)));
        (Box::new(out) as Box<dyn Any>)
            .downcast::<Tensor<B, D>>()
            .ok()
            .map(|t| *t)
    }
}
