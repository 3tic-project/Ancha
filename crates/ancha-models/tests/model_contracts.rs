use ancha_kernels::attention::{Shape, dense};
use ancha_models::{
    config::{Manifest, ModelConfig},
    roformer::{AttentionPlan, SourceNorm, exact_attention},
    weights::Weights,
};
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
type B = NdArray<f32>;

#[test]
fn tiled_attention_matches_independent_scalar_reference_with_partial_tiles() {
    let device = Default::default();
    for n in [1, 19, 35] {
        let groups = 3;
        let heads = 2;
        let dim = 8;
        let values = |s: f32| {
            (0..groups * heads * n * dim)
                .map(|i| (i as f32 * 0.31 + s).sin())
                .collect::<Vec<_>>()
        };
        let q = values(1.);
        let k = values(2.);
        let v = values(3.);
        let expected = dense(
            &q,
            &k,
            &v,
            Shape {
                groups: groups * heads,
                queries: n,
                keys: n,
                head_dim: dim,
                value_dim: dim,
            },
        )
        .unwrap();
        for (qt, gt) in [(1, 1), (7, 2), (128, 4)] {
            let tensor = |values: Vec<f32>| {
                Tensor::<B, 4>::from_data(TensorData::new(values, [groups, heads, n, dim]), &device)
            };
            let actual: Vec<f32> = exact_attention(
                tensor(q.clone()),
                tensor(k.clone()),
                tensor(v.clone()),
                AttentionPlan {
                    query_tile: Some(qt),
                    group_tile: Some(gt),
                    batched_linear: false,
                    ..AttentionPlan::default()
                },
            )
            .into_data()
            .to_vec()
            .unwrap();
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).abs() < 2e-5)
            );
        }
    }
}

#[test]
fn source_norm_has_f_normalize_epsilon_semantics() {
    use safetensors::{
        serialize,
        tensor::{Dtype, TensorView},
    };
    let bytes: Vec<u8> = [1f32, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    let view = TensorView::new(Dtype::F32, vec![2], &bytes).unwrap();
    let data = serialize([("norm.gamma", view)], None).unwrap();
    let mut weights = Weights::new(&data).unwrap();
    let norm = SourceNorm::<B>::load(&mut weights, "norm", 2, &Default::default()).unwrap();
    weights.finish().unwrap();
    let x = Tensor::<B, 2>::from_floats([[0., 0.], [1e-14, 0.], [3., 4.]], &Default::default());
    let y: Vec<f32> = norm.forward(x).into_data().to_vec().unwrap();
    assert_eq!(&y[..2], &[0., 0.]);
    assert!((y[2] - 0.01 * 2f32.sqrt()).abs() < 1e-6);
    assert!((y[4] - 3. / 5. * 2f32.sqrt()).abs() < 1e-6);
    assert!((y[5] - 4. / 5. * 2. * 2f32.sqrt()).abs() < 1e-6);
}

#[test]
fn manifests_reject_unknown_schema_holes_duplicate_stems_and_invalid_digests() {
    let config = ModelConfig::leap_xe(false);
    config.validate().unwrap();
    assert_eq!(config.bands.len(), 90);
    let mut bad = config.clone();
    bad.bands[0].clear();
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.bands[0].push(1);
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.stems = vec!["vocals".into(), "vocals".into()];
    assert!(bad.validate().is_err());
    let mut m = Manifest {
        schema_version: 1,
        model_id: "leap-xe-voc".into(),
        weights_sha256: "0".repeat(64),
        checkpoint_sha256: "0".repeat(64),
        source_url: "fixture".into(),
        forward_revision: "fixture".into(),
        weight_license: "fixture".into(),
        config,
    };
    m.validate().unwrap();
    m.schema_version = 2;
    assert!(m.validate().is_err());
    m.schema_version = 1;
    m.weights_sha256 = "bad".into();
    assert!(m.validate().is_err());
    let json = serde_json::to_value(&m).unwrap();
    let mut json = json.as_object().unwrap().clone();
    json.insert("unknown".into(), true.into());
    assert!(serde_json::from_value::<Manifest>(json.into()).is_err());
}

#[test]
fn strict_loader_rejects_missing_wrong_shape_nonfinite_and_unused_tensors() {
    use safetensors::{
        serialize,
        tensor::{Dtype, TensorView},
    };
    let bytes: Vec<_> = [f32::NAN, 0.]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let data = serialize(
        [("a", TensorView::new(Dtype::F32, vec![2], &bytes).unwrap())],
        None,
    )
    .unwrap();
    let mut weights = Weights::new(&data).unwrap();
    assert!(
        weights
            .tensor::<B, 1>("missing", [2], &Default::default())
            .is_err()
    );
    assert!(
        weights
            .tensor::<B, 1>("a", [3], &Default::default())
            .is_err()
    );
    assert!(
        weights
            .tensor::<B, 1>("a", [2], &Default::default())
            .is_err()
    );
    assert!(weights.finish().is_err());
}
