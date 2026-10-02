//! Hardware contracts: need an NVIDIA GPU and `--features cuda`; CI only compiles them.
#![cfg(feature = "cuda")]
use ancha::runtime::{SeparateOptions, separate};
use ancha_audio::{
    Audio,
    decode::{DecodeOptions, decode},
};
use ancha_models::{
    config::Family,
    fused,
    roformer::{AttentionPlan, AxisRope, exact_attention},
    spatial::conv2d_gemm,
};
use burn::{
    backend::{Cuda, cuda::CudaDevice},
    tensor::{
        Tensor, TensorData,
        activation::{gelu, sigmoid},
        backend::Backend,
        module::conv2d,
        ops::ConvOptions,
    },
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
fn cuda_implicit_gemm_convolution_matches_backend_convolution() {
    let d = device();
    // 1×1, padded 3×3, strided 2×2; 2-row and 4-row channel tiles with partial tiles,
    // position counts off the 128 / 4 grid, two batch items.
    for (batch, ci, co, k, s, p, h, w, bias) in [
        (1, 8, 16, 1, 1, 0, 9, 13, false),
        (2, 16, 48, 3, 1, 1, 11, 23, true),
        (1, 8, 130, 3, 1, 1, 7, 37, true),
        (2, 48, 40, 2, 2, 0, 10, 31, false),
    ] {
        let x = wave([batch, ci, h, w], 0.3, &d);
        let weight = wave([co, ci, k, k], 1.1, &d).mul_scalar(0.2);
        let b = wave([co], 2.0, &d);
        let expected = conv2d(
            x.clone(),
            weight.clone(),
            bias.then(|| b.clone()),
            ConvOptions::new([s, s], [p, p], [1, 1], 1),
        );
        let actual = fused::conv2d(&x, &weight, bias.then_some(&b), [s, s], [p, p])
            .expect("implicit GEMM convolution");
        assert_eq!(actual.dims(), expected.dims());
        let max = max_abs(expected, actual);
        assert!(
            max < 1e-5,
            "conv [{batch}, {ci}, {h}, {w}] -> {co}, k{k} s{s} p{p}: {max}"
        );
    }
    let x = wave([1, 3, 5, 5], 0.0, &d);
    assert!(fused::conv2d(&x, &wave([4, 3, 3, 3], 0.0, &d), None, [1, 1], [1, 1]).is_none());
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

fn wave<const D: usize>(shape: [usize; D], shift: f32, device: &CudaDevice) -> Tensor<Cuda, D> {
    let n = shape.iter().product();
    let values = (0..n)
        .map(|i| ((i as f32 * 0.731 + shift).sin() * 1.7).tanh())
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn max_abs<const D: usize>(a: Tensor<Cuda, D>, b: Tensor<Cuda, D>) -> f32 {
    (a - b).abs().max().into_scalar()
}

#[test]
fn cuda_fused_attention_matches_tiled_attention() {
    let d = device();
    // Partial query / key tiles, a single token, and a strided (narrowed) layout.
    for (groups, tokens, heads, strided) in [
        (2, 1, 1, false),
        (3, 63, 2, false),
        (2, 65, 1, true),
        (1, 130, 3, false),
    ] {
        let width = if strided { 128 } else { 64 };
        let input = |shift| wave([groups, tokens, heads, width], shift, &d).narrow(3, 0, 64);
        let (q, k, v) = (input(0.1).mul_scalar(0.4), input(1.3), input(2.9));
        let plan = AttentionPlan {
            query_tile: Some(7),
            group_tile: Some(1),
            ..AttentionPlan::default()
        };
        // exact_attention applies 1/sqrt(64) to q; the fused kernel expects it pre-applied.
        let expected = exact_attention(
            q.clone().swap_dims(1, 2),
            k.clone().swap_dims(1, 2),
            v.clone().swap_dims(1, 2),
            plan,
        )
        .swap_dims(1, 2);
        let actual =
            fused::attention(&q.clone().mul_scalar(0.125), &k, &v).expect("fused attention");
        let max = max_abs(expected.clone(), actual);
        assert!(
            max < 1e-5,
            "fused attention [{groups}, {tokens}, {heads}]: {max}"
        );
        // The packed layout of the RoFormer block: q | k | v | per-head gates | padding.
        let flat = |t: Tensor<Cuda, 4>| t.reshape([groups, tokens, heads * 64]);
        let gates = wave([groups, tokens, heads], 4.4, &d).mul_scalar(3.0);
        let packed = Tensor::cat(
            vec![
                flat(q.mul_scalar(0.125)),
                flat(k),
                flat(v),
                gates.clone(),
                Tensor::zeros([groups, tokens, 5], &d),
            ],
            2,
        );
        let gated = expected * sigmoid(gates).unsqueeze_dim::<4>(3);
        let actual = fused::gated_attention(&packed, heads).expect("gated attention");
        let max = max_abs(gated, actual);
        assert!(
            max < 1e-5,
            "gated attention [{groups}, {tokens}, {heads}]: {max}"
        );
    }
    let narrow = wave([1, 8, 1, 32], 0.0, &d);
    assert!(fused::attention(&narrow, &narrow, &narrow).is_none());
}

#[test]
fn cuda_gemm_epilogues_match_burn_ops() {
    let d = device();
    let (groups, tokens, k, heads) = (3, 70, 64, 2);
    let (rows, width) = (groups * tokens, heads * 64);
    let m = 3 * width;
    let x = wave([rows, k], 0.3, &d);
    let w = wave([k, m], 1.1, &d).mul_scalar(0.1);
    let b = wave([m], 2.2, &d);
    let r = wave([rows, m], 3.3, &d);
    let projected = || x.clone().matmul(w.clone()) + b.clone().unsqueeze::<2>();

    let expected = gelu(projected()) + r.clone();
    let epilogue = fused::Epilogue {
        bias: Some(&b),
        gelu: true,
        residual: Some(&r),
        ..fused::Epilogue::default()
    };
    let max = max_abs(
        expected,
        fused::linear(&x, &w, epilogue).expect("GELU / residual"),
    );
    assert!(max < 1e-5, "bias + GELU + residual epilogue: {max}");

    let freqs: Vec<f32> = (0..32).map(|i| 10000f32.powf(-(i as f32) / 32.0)).collect();
    let rope = AxisRope::<Cuda>::new(&freqs, tokens, heads, &d);
    let y = projected().reshape([groups, tokens, m]);
    let expected = Tensor::cat(
        vec![
            rope.query.apply(y.clone().narrow(2, 0, width)),
            rope.key.apply(y.clone().narrow(2, width, width)),
            y.narrow(2, 2 * width, width),
        ],
        2,
    )
    .reshape([rows, m]);
    let table = rope.packed.as_ref().expect("rotary table on CUDA");
    let epilogue = fused::Epilogue {
        bias: Some(&b),
        rotary: Some((table, width)),
        ..fused::Epilogue::default()
    };
    let max = max_abs(expected, fused::linear(&x, &w, epilogue).expect("rotary"));
    assert!(max < 1e-5, "rotary epilogue: {max}");
}

#[test]
fn cuda_custom_gemm_matches_burn_matmul() {
    let d = device();
    for (rows, k, m, bias, strided) in [
        (1, 8, 128, true, false),
        (127, 256, 256, false, false),
        (129, 64, 384, true, true),
        (300, 1024, 128, true, false),
    ] {
        let width = if strided { k + 16 } else { k };
        let x = wave([rows, width], 0.2, &d).narrow(1, 0, k);
        let w = wave([k, m], 1.7, &d).mul_scalar(0.05);
        let b = wave([m], 3.1, &d);
        let mut expected = x.clone().matmul(w.clone());
        if bias {
            expected = expected + b.clone().unsqueeze::<2>();
        }
        let actual = fused::linear(
            &x,
            &w,
            fused::Epilogue {
                bias: bias.then_some(&b),
                ..fused::Epilogue::default()
            },
        )
        .expect("custom GEMM");
        let max = max_abs(expected, actual);
        assert!(max < 1e-5, "custom GEMM [{rows}, {k}]·[{k}, {m}]: {max}");
    }
    let x = wave([4, 16], 0.0, &d);
    let none = fused::Epilogue::default;
    assert!(fused::linear(&x, &wave([16, 8], 0.0, &d), none()).is_none());
    assert!(fused::linear(&wave([4, 12], 0.0, &d), &wave([12, 128], 0.0, &d), none()).is_none());
}

#[test]
fn cuda_custom_kernels_match_flex_cpu_on_a_full_model() {
    for family in [Family::BsRoformer, Family::MelBandRoformer] {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("model");
        support::kernel_package(&model, family);
        let input = temp.path().join("mono.wav");
        let signal = (0..9001).map(|i| (i as f32 * 0.013).sin() * 0.3).collect();
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
            max_score_mib: 16,
        };
        let gpu = SeparateOptions {
            output: temp.path().join("cuda"),
            attention: AttentionPlan {
                fused_attention: true,
                custom_gemm: true,
                ..cpu.attention
            },
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
        assert_eq!(report.attention_kernel, "fused");
        assert_eq!(report.gemm_kernel, "custom");
        assert_eq!(report.estimated_time_attention_score_bytes, 0);
        for name in ["vocals", "instrumental"] {
            let file = format!("{name}.wav");
            let max = max_difference(&cpu.output.join(&file), &gpu.output.join(&file));
            assert!(max < 1e-5, "CUDA kernels differ from Flex on {name}: {max}");
        }
    }
}
