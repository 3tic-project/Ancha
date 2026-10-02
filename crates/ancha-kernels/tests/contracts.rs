use ancha_kernels::{
    attention::{self, Shape},
    cache::*,
    fusion::*,
};

fn values(n: usize, shift: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + shift) as f32 * 0.13).sin())
        .collect()
}

#[test]
fn online_softmax_matches_dense_for_rectangles_and_partial_tiles() {
    for &(nq, nk, d, dv) in &[(1, 1, 1, 1), (19, 23, 7, 5), (33, 37, 64, 64)] {
        let s = Shape {
            groups: 2,
            queries: nq,
            keys: nk,
            head_dim: d,
            value_dim: dv,
        };
        let q = values(2 * nq * d, 1);
        let k = values(2 * nk * d, 2);
        let v = values(2 * nk * dv, 3);
        let a = attention::dense(&q, &k, &v, s).unwrap();
        for &(qt, kt) in &[(1, 1), (8, 7), (64, 128)] {
            let b = attention::tiled(&q, &k, &v, s, qt, kt).unwrap();
            assert!(a.iter().zip(b).all(|(x, y)| (*x - y).abs() < 2e-5));
        }
    }
}

#[test]
fn attention_rejects_empty_nonfinite_and_zero_tiles() {
    let s = Shape {
        groups: 1,
        queries: 1,
        keys: 1,
        head_dim: 1,
        value_dim: 1,
    };
    assert!(attention::tiled(&[1.], &[1.], &[1.], s, 0, 1).is_err());
    assert!(attention::dense(&[f32::NAN], &[1.], &[1.], s).is_err());
    assert!(attention::dense(&[], &[1.], &[1.], s).is_err());
}

#[test]
fn norm_zero_and_small_values_keep_source_epsilon_semantics() {
    let (sum, y) = residual_l2norm(&[0., 0., 1e-14, 0.], &[0.; 4], &[1., 1.], 2, 2, 1e-12).unwrap();
    assert_eq!(&sum[..2], &[0., 0.]);
    assert_eq!(&y[..2], &[0., 0.]);
    assert!((y[2] - 0.01 * 2f32.sqrt()).abs() < 1e-6);
}

#[test]
fn pack_keeps_gate_bias_and_fold_is_columnwise() {
    let (w, b) = pack_qkv_gate(&[1., 2., 3., 4., 5., 6.], &[7., 8.], &[0.5], 2).unwrap();
    assert_eq!(b, vec![0., 0., 0., 0.5]);
    let f = fold_norm_scale(&w, &[2., 3.]).unwrap();
    assert!((f[6] - 7. * 2. * 2f32.sqrt()).abs() < 1e-5);
    assert!((f[7] - 8. * 3. * 2f32.sqrt()).abs() < 1e-5);
}

#[test]
fn complex_multiplication_keeps_phase() {
    assert_eq!(
        complex_mask_interleaved(&[1., 2., 3., -1.], &[4., 5., 0., 1.]).unwrap(),
        vec![-6., 13., 1., 3.]
    );
}

fn frame(start: u64) -> FrameContext {
    FrameContext {
        source_digest: "source".into(),
        frontend_digest: "hann-gain-resampler".into(),
        projection_digest: "weights".into(),
        start_sample: start,
        chunk_samples: 573300,
        hop: 441,
        n_fft: 2048,
    }
}
#[test]
fn frontend_cache_checks_grid_support_and_weight_identity() {
    let a = frame(0);
    let b = frame(286650);
    assert!(can_reuse_first_pre_rope(&a, 653, &b, 3));
    assert!(!can_reuse_first_pre_rope(&a, 650, &b, 0)); // boundary reflected frame
    let mut c = b.clone();
    c.start_sample += 1;
    assert!(!can_reuse_first_pre_rope(&a, 653, &c, 3));
    c = b.clone();
    c.projection_digest = "different checkpoint".into();
    assert!(!can_reuse_first_pre_rope(&a, 653, &c, 3));
    assert_eq!(phase_cycle(440779, 512), Some(512));
    assert_eq!(phase_cycle(240000, 512), Some(4));
    assert_eq!(phase_cycle(286650, 441), Some(1));
}

#[test]
fn deep_kv_requires_identical_context() {
    let a = ChunkKey {
        input_digest: "pcm".into(),
        model_digest: "m".into(),
        forward_digest: "f".into(),
        frontend_digest: "p".into(),
        start_sample: 0,
        chunk_samples: 32,
        precision: "f32".into(),
        backend_plan: "cpu".into(),
    };
    assert!(can_reuse_deep_kv(&a, &a));
    let mut b = a.clone();
    b.start_sample = 16;
    assert!(!can_reuse_deep_kv(&a, &b));
    b = a.clone();
    b.model_digest = "other".into();
    assert!(!can_reuse_full_chunk(&a, &b));
}
