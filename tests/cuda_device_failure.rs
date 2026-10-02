//! Hardware test (own binary, the failure is sticky): CUDA out-of-memory must surface.
#![cfg(feature = "cuda")]
use burn::{
    backend::{Cuda, cuda::CudaDevice},
    tensor::Tensor,
};

#[test]
fn cuda_allocation_failure_is_reported_instead_of_stale_results() {
    ancha::device::install_guard();
    let info = ancha::cuda::prepare(0, "contract-tests").unwrap();
    let device = CudaDevice::new(0);
    let small = Tensor::<Cuda, 1>::ones([1024], &device).sum().into_scalar();
    assert_eq!(small, 1024.0);
    ancha::device::check().unwrap();
    // One buffer larger than the whole device cannot be allocated on any GPU. Depending
    // on the op the caller sees stale data or a panic; the guard must report both.
    let elements = info.total_memory_bytes / 4 + (1 << 28);
    let _ = std::panic::catch_unwind(|| {
        Tensor::<Cuda, 1>::ones([elements], &device)
            .sum()
            .into_scalar()
    });
    let error = ancha::device::check().unwrap_err().to_string();
    assert!(error.contains("GPU device thread failed"), "{error}");
}
