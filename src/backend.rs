//! Backend chosen at start-up: one binary runs on CPU, WGPU or CUDA. The per-backend defaults
//! are the settings measured in docs/speed-optimization.md and docs/cuda.md.
use crate::{
    report::RunReport,
    runtime::{self, SeparateOptions},
};
use ancha_models::roformer::AttentionPlan;
use anyhow::Result;
use burn::backend::NdArray;
use burn_flex::Flex;
use std::sync::atomic::AtomicBool;

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// Burn Flex CPU backend: gemm matmul / im2col convolution, strided views.
    Cpu,
    /// Legacy Burn NdArray CPU backend, kept for ablation and comparison.
    Ndarray,
    /// WGPU through CubeCL (Metal / Vulkan / DX12); `wgpu` feature, on by default.
    Wgpu,
    /// NVIDIA CUDA through CubeCL (NVRTC kernels, fusion, autotune); `cuda` feature.
    Cuda,
}

impl BackendKind {
    /// `backend` field of run.json.
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu-flex",
            Self::Ndarray => "cpu-ndarray",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
        }
    }
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Wgpu | Self::Cuda)
    }
    /// Whether this binary was compiled with the backend.
    pub fn is_built(self) -> bool {
        match self {
            Self::Cpu | Self::Ndarray => true,
            Self::Wgpu => cfg!(feature = "wgpu"),
            Self::Cuda => cfg!(feature = "cuda"),
        }
    }
    /// RoFormer defaults: automatic tiles within 512 MiB of scores; all host threads on CPU,
    /// one on GPUs; on CUDA folded projections and GEMM convolution for HyperACE.
    pub fn attention_plan(self) -> AttentionPlan {
        AttentionPlan {
            query_tile: None,
            group_tile: None,
            score_budget: 512 << 20,
            batched_linear: self != Self::Cuda,
            host_threads: if self.is_gpu() {
                1
            } else {
                std::thread::available_parallelism().map_or(1, |n| n.get())
            },
            conv_gemm: self == Self::Cuda,
        }
    }
    /// MDX: patch-gather GEMM convolution on GPUs, Burn conv2d (im2col) on CPU.
    pub fn mdx_conv_gemm(self) -> bool {
        self.is_gpu()
    }
}

#[cfg(not(all(feature = "wgpu", feature = "cuda")))]
fn not_built(kind: BackendKind) -> anyhow::Error {
    anyhow::anyhow!(
        "{} support is not built; rebuild with --features {}",
        kind.label(),
        kind.label()
    )
}

/// PTX cache partition used for a model: its file or directory name.
pub fn cache_scope(model: &std::path::Path) -> String {
    model
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// Separate with a RoFormer package. `device` is the GPU ordinal and is ignored on CPU.
pub fn separate(
    kind: BackendKind,
    device: usize,
    options: &SeparateOptions,
    cancelled: &AtomicBool,
    progress: impl FnMut(usize, usize),
) -> Result<RunReport> {
    let label = kind.label();
    match kind {
        BackendKind::Cpu => {
            runtime::separate::<Flex>(options, &Default::default(), label, cancelled, progress)
        }
        BackendKind::Ndarray => runtime::separate::<NdArray<f32>>(
            options,
            &Default::default(),
            label,
            cancelled,
            progress,
        ),
        BackendKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                runtime::separate::<burn::backend::Wgpu>(
                    options,
                    &burn::backend::wgpu::WgpuDevice::DiscreteGpu(device),
                    label,
                    cancelled,
                    progress,
                )
            }
            #[cfg(not(feature = "wgpu"))]
            {
                let _ = device;
                Err(not_built(kind))
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                runtime::separate::<burn::backend::Cuda>(
                    options,
                    &cuda_device(device, &cache_scope(&options.model))?,
                    label,
                    cancelled,
                    progress,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device;
                Err(not_built(kind))
            }
        }
    }
}

/// Separate with a registered classic MDX ONNX model.
#[cfg(feature = "onnx")]
pub fn separate_mdx(
    kind: BackendKind,
    device: usize,
    options: &crate::mdx_runtime::MdxOptions,
    cancelled: &AtomicBool,
    progress: impl FnMut(usize, usize),
) -> Result<crate::mdx_runtime::MdxReport> {
    use crate::mdx_runtime::separate_mdx as run;
    let label = kind.label();
    match kind {
        BackendKind::Cpu => run::<Flex>(options, &Default::default(), label, cancelled, progress),
        BackendKind::Ndarray => {
            run::<NdArray<f32>>(options, &Default::default(), label, cancelled, progress)
        }
        BackendKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                run::<burn::backend::Wgpu>(
                    options,
                    &burn::backend::wgpu::WgpuDevice::DiscreteGpu(device),
                    label,
                    cancelled,
                    progress,
                )
            }
            #[cfg(not(feature = "wgpu"))]
            {
                let _ = device;
                Err(not_built(kind))
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                run::<burn::backend::Cuda>(
                    options,
                    &cuda_device(device, &cache_scope(&options.model))?,
                    label,
                    cancelled,
                    progress,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device;
                Err(not_built(kind))
            }
        }
    }
}

/// Checked CUDA device with the PTX cache set up; see [`crate::cuda::prepare`].
#[cfg(feature = "cuda")]
pub fn cuda_device(index: usize, scope: &str) -> Result<burn::backend::cuda::CudaDevice> {
    crate::cuda::prepare(index, scope)?;
    Ok(burn::backend::cuda::CudaDevice::new(index))
}
