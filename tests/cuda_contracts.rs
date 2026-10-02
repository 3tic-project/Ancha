//! Hardware contracts: need an NVIDIA GPU and `--features cuda`; CI only compiles them.
#![cfg(feature = "cuda")]
use ancha::runtime::{SeparateOptions, separate};
use ancha_audio::{
    Audio,
    decode::{DecodeOptions, decode},
};
use ancha_models::{config::Family, roformer::AttentionPlan, spatial::conv2d_gemm};
use burn::{
    backend::{Cuda, cuda::CudaDevice},
    tensor::{Tensor, TensorData, backend::Backend, module::conv2d, ops::ConvOptions},
};
use burn_flex::Flex;
use std::{path::Path, sync::atomic::AtomicBool};
mod support;

fn device() -> CudaDevice {
    let info = ancha::cuda::prepare(0, "contract-tests").unwrap();
    assert!(info.compute_capability >= 50 && !info.name.is_empty());
    CudaDevice::new(0)
}

#[test]
fn cuda_probe_rejects_missing_device_without_panicking() {
    let error = ancha::cuda::probe(4096).unwrap_err().to_string();
    assert!(error.contains("not found"), "{error}");
}

fn max_difference(a: &Path, b: &Path) -> f32 {
    let a = decode(a, DecodeOptions::default()).unwrap();
    let b = decode(b, DecodeOptions::default()).unwrap();
    assert_eq!(a.samples(), b.samples());
    a.planes
        .iter()
        .flatten()
        .zip(b.planes.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn cuda_separation_matches_flex_cpu_for_both_families() {
    for family in [Family::BsRoformer, Family::MelBandRoformer] {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("model");
        support::tiny_package(&model, family);
        let input = temp.path().join("mono.wav");
        let signal = (0..5003).map(|i| (i as f32 * 0.017).sin() * 0.3).collect();
        ancha_audio::write_wav(
            &input,
            &Audio {
                sample_rate: 44100,
                planes: vec![signal],
            },
        )
        .unwrap();
        let cpu = SeparateOptions {
            input,
            model,
            output: temp.path().join("cpu"),
            decode: DecodeOptions::default(),
            chunk_samples: None,
            overlap: None,
            attention: AttentionPlan {
                query_tile: None,
                group_tile: None,
                ..AttentionPlan::default()
            },
            max_score_mib: 1,
        };
        let gpu = SeparateOptions {
            output: temp.path().join("cuda"),
            ..cpu.clone()
        };
        separate::<Flex>(
            &cpu,
            &Default::default(),
            "cpu-flex",
            &AtomicBool::new(false),
            |_, _| {},
        )
        .unwrap();
        let report =
            separate::<Cuda>(&gpu, &device(), "cuda", &AtomicBool::new(false), |_, _| {}).unwrap();
        assert_eq!(report.backend, "cuda");
        assert!(report.build_features.contains(&"cuda".to_string()));
        for name in ["vocals", "instrumental"] {
            let file = format!("{name}.wav");
            let max = max_difference(&cpu.output.join(&file), &gpu.output.join(&file));
            assert!(max < 1e-5, "CUDA differs from Flex on {name}: {max}");
        }
    }
}

#[test]
fn cuda_gemm_convolution_matches_backend_convolution() {
    type B = Cuda;
    let d = device();
    let values = |n: usize, s: f32| {
        (0..n)
            .map(|i| (i as f32 * 0.37 + s).sin())
            .collect::<Vec<_>>()
    };
    for (ci, co, k, s, p, h, w) in [(3, 4, 3, 1, 1, 5, 7), (2, 3, 2, 2, 0, 6, 9)] {
        let x = Tensor::<B, 4>::from_data(
            TensorData::new(values(2 * ci * h * w, 0.3), [2, ci, h, w]),
            &d,
        );
        let weight = Tensor::<B, 4>::from_data(
            TensorData::new(values(co * ci * k * k, 1.1), [co, ci, k, k]),
            &d,
        );
        let bias = Tensor::<B, 1>::from_data(TensorData::new(values(co, 2.0), [co]), &d);
        let expected: Vec<f32> = conv2d(
            x.clone(),
            weight.clone(),
            Some(bias.clone()),
            ConvOptions::new([s, s], [p, p], [1, 1], 1),
        )
        .into_data()
        .to_vec()
        .unwrap();
        let actual: Vec<f32> = conv2d_gemm(x, weight, Some(bias), [s, s], [p, p])
            .unwrap()
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
    }
    B::sync(&d).unwrap();
}
