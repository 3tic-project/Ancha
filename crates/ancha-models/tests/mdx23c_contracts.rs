use ancha_models::{
    mdx23c::{Config, Mdx23c, Options},
    spatial::{InstanceNorm, conv_transpose2d_gemm},
};
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use std::sync::atomic::AtomicBool;
#[path = "support/mdx23c.rs"]
mod support;
type B = NdArray<f32>;

#[test]
fn synthetic_two_scale_two_head_forward_matches_independent_pytorch_golden() {
    let temp = tempfile::tempdir().unwrap();
    let m = support::package(temp.path());
    let model = Mdx23c::<B>::load(temp.path(), &m, &Default::default()).unwrap();
    assert_eq!(model.tensor_count, 80);
    let fixture = support::fixture();
    for o in [
        Options {
            conv_gemm: false,
            optimized: false,
        },
        Options {
            conv_gemm: true,
            optimized: true,
        },
        Options {
            conv_gemm: false,
            optimized: true,
        },
    ] {
        let y = model
            .forward(
                Tensor::from_data(
                    TensorData::new(support::input(), fixture.input_shape),
                    &Default::default(),
                ),
                o,
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(y.dims(), fixture.output_shape);
        let actual: Vec<f32> = y.into_data().to_vec().unwrap();
        support::assert_close(&actual, &fixture.expected, 1e-8);
        let max = actual
            .iter()
            .zip(&fixture.expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 1e-5, "full MDX23C differs from PyTorch {o:?}: {max}");
    }
    let mut corrupt = m.clone();
    corrupt.weights_sha256 = "0".repeat(64);
    assert!(Mdx23c::<B>::load(temp.path(), &corrupt, &Default::default()).is_err());
    assert!(
        model
            .forward(
                Tensor::zeros([1, 4, 32, 7], &Default::default()),
                Options::default(),
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert!(
        model
            .forward(
                Tensor::zeros([1, 4, 32, 8], &Default::default()),
                Options::default(),
                &AtomicBool::new(true)
            )
            .is_err()
    );
}

#[test]
fn nonoverlapping_transpose_convolution_keeps_batch_channel_and_spatial_phases() {
    for (b, ci, co, h, w, kh, kw) in [(2, 3, 2, 3, 5, 2, 2), (1, 2, 3, 2, 4, 2, 3)] {
        let x: Vec<f32> = (0..b * ci * h * w)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
        let weight: Vec<f32> = (0..ci * co * kh * kw)
            .map(|i| (i as f32 * 0.07).cos() * 0.1)
            .collect();
        let actual: Vec<f32> = conv_transpose2d_gemm(
            Tensor::<B, 4>::from_data(
                TensorData::new(x.clone(), [b, ci, h, w]),
                &Default::default(),
            ),
            Tensor::from_data(
                TensorData::new(weight.clone(), [ci, co, kh, kw]),
                &Default::default(),
            ),
        )
        .unwrap()
        .into_data()
        .to_vec()
        .unwrap();
        let mut expected = vec![0f32; b * co * h * kh * w * kw];
        for batch in 0..b {
            for out in 0..co {
                for y in 0..h {
                    for z in 0..w {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let at = ((batch * co + out) * (h * kh) + y * kh + ky) * (w * kw)
                                    + z * kw
                                    + kx;
                                for input in 0..ci {
                                    expected[at] += x[((batch * ci + input) * h + y) * w + z]
                                        * weight[((input * co + out) * kh + ky) * kw + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }
}

#[test]
fn mdx_instance_norm_uses_epsilon_1e5_and_biased_variance_per_batch_channel() {
    let values: Vec<f32> = (0..24).map(|i| 1e-4 * (i as f32).sin()).collect();
    let norm = InstanceNorm::<B>::from_affine_with_epsilon(
        Tensor::from_floats([1.2, -0.7], &Default::default()),
        Tensor::from_floats([0.01, -0.03], &Default::default()),
        1e-5,
    );
    let actual: Vec<f32> = norm
        .forward_flattened(Tensor::from_data(
            TensorData::new(values.clone(), [2, 2, 2, 3]),
            &Default::default(),
        ))
        .into_data()
        .to_vec()
        .unwrap();
    for (group, x) in values.chunks_exact(6).enumerate() {
        let mean = x.iter().sum::<f32>() / 6.;
        let var = x.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / 6.;
        let c = group % 2;
        for (i, &value) in x.iter().enumerate() {
            let expected = (value - mean) / (var + 1e-5).sqrt() * [1.2, -0.7][c] + [0.01, -0.03][c];
            assert!((actual[group * 6 + i] - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn mdx23c_manifest_rejects_shapes_that_cannot_survive_all_scales() {
    let native = Config::inst_voc_hq2();
    native.validate().unwrap();
    assert_eq!(native.chunk_samples(), 261120);
    for change in 0..7 {
        let mut c = native.clone();
        match change {
            0 => c.frames = 255,
            1 => c.bins = 4095,
            2 => c.subbands = 0,
            3 => c.overlap = 0,
            4 => c.scales = 0,
            5 => c.stems.swap(0, 1),
            _ => c.bottleneck_factor = 0,
        }
        assert!(c.validate().is_err());
    }
}
