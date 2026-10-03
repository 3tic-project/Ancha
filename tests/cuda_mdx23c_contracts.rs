//! Compile-only in CI; execute on NVIDIA hardware to accept the new MDX23C CUDA path.
#![cfg(feature = "cuda")]
use ancha_models::{mdx23c::Mdx23c, spatial::conv_transpose2d_gemm};
use burn::{
    backend::{Cuda, cuda::CudaDevice},
    tensor::{Tensor, TensorData},
};
use burn_flex::Flex;
use std::sync::atomic::AtomicBool;
#[path = "../crates/ancha-models/tests/support/mdx23c.rs"]
mod support;

#[test]
fn cuda_two_head_forward_matches_independent_pytorch_fixture() {
    ancha::cuda::prepare(0, "mdx23c-contracts").unwrap();
    let device = CudaDevice::new(0);
    let dir = tempfile::tempdir().unwrap();
    let manifest = support::package(dir.path());
    let model = Mdx23c::<Cuda>::load(dir.path(), &manifest, &device).unwrap();
    let fixture = support::fixture();
    for conv_gemm in [false, true] {
        for optimized in [false, true] {
            let output = model
                .forward(
                    Tensor::from_data(
                        TensorData::new(support::input(), fixture.input_shape),
                        &device,
                    ),
                    ancha_models::mdx23c::Options {
                        conv_gemm,
                        optimized,
                    },
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(output.dims(), fixture.output_shape);
            let actual: Vec<f32> = output.into_data().to_vec().unwrap();
            support::assert_close(&actual, &fixture.expected, 1e-6);
            let max = actual
                .iter()
                .zip(&fixture.expected)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(max < 1e-6, "CUDA MDX23C golden: {max}");
        }
    }
    ancha::device::check().unwrap();
}

#[test]
fn cuda_disjoint_transpose_gemm_matches_flex_with_custom_gemm_supported_shapes() {
    ancha::cuda::prepare(0, "mdx23c-contracts").unwrap();
    let device = CudaDevice::new(0);
    // GEMM [30,16] * [16,128] exercises the CUDA custom kernel, including a partial row tile.
    let x: Vec<f32> = (0..2 * 16 * 3 * 5)
        .map(|i| (i as f32 * 0.17).sin())
        .collect();
    let w: Vec<f32> = (0..16 * 32 * 2 * 2)
        .map(|i| (i as f32 * 0.071).cos() * 0.03)
        .collect();
    let expected: Vec<f32> = conv_transpose2d_gemm(
        Tensor::<Flex, 4>::from_data(
            TensorData::new(x.clone(), [2, 16, 3, 5]),
            &Default::default(),
        ),
        Tensor::from_data(
            TensorData::new(w.clone(), [16, 32, 2, 2]),
            &Default::default(),
        ),
    )
    .unwrap()
    .into_data()
    .to_vec()
    .unwrap();
    let actual: Vec<f32> = conv_transpose2d_gemm(
        Tensor::<Cuda, 4>::from_data(TensorData::new(x, [2, 16, 3, 5]), &device),
        Tensor::from_data(TensorData::new(w, [16, 32, 2, 2]), &device),
    )
    .unwrap()
    .into_data()
    .to_vec()
    .unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(x, y)| (x - y).abs() < 1e-5)
    );
    ancha::device::check().unwrap();
}
