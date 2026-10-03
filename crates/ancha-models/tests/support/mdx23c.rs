use ancha_models::{
    mdx23c::{Config, Package},
    weights::sha256_file,
};
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

#[derive(Deserialize)]
pub struct Weight {
    pub name: String,
    pub shape: Vec<usize>,
}
#[derive(Deserialize)]
pub struct Fixture {
    pub config: Config,
    pub weights: Vec<Weight>,
    pub input_shape: [usize; 4],
    pub output_shape: [usize; 4],
    pub expected: Vec<f32>,
}
pub fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/mdx23c-tiny.json")).unwrap()
}
pub fn input() -> Vec<f32> {
    (0..fixture().input_shape.iter().product())
        .map(|i: usize| (0.3 * ((i + 1) as f64 * 0.071).sin()) as f32)
        .collect()
}
/// A relative gate also rejects an all-zero result for this deliberately low-amplitude fixture.
pub fn assert_close(actual: &[f32], expected: &[f32], max_relative_mse: f64) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|x| x.is_finite()));
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>();
    let power = expected.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    assert!(
        power > 0. && error < power * max_relative_mse,
        "golden relative MSE {} exceeds {max_relative_mse}",
        error / power
    );
}
pub fn package(path: &Path) -> Package {
    let f = fixture();
    std::fs::create_dir_all(path).unwrap();
    let owned: Vec<_> = f
        .weights
        .iter()
        .map(|w| {
            let seed = w.name.bytes().map(|b| b as usize).sum::<usize>() % 97;
            let bytes: Vec<u8> = (0..w.shape.iter().product())
                .flat_map(|i: usize| {
                    let phase = (i + 1) as f64 * 0.17 + seed as f64 * 0.11;
                    let value = if w.shape.len() == 1 && w.name.ends_with("weight") {
                        1. + 0.03 * phase.sin()
                    } else {
                        0.02 * phase.sin()
                    };
                    (value as f32).to_le_bytes()
                })
                .collect();
            (w.name.clone(), w.shape.clone(), bytes)
        })
        .collect();
    let views: HashMap<_, _> = owned
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
            )
        })
        .collect();
    let weights = path.join("model.safetensors");
    serialize_to_file(views, None, &weights).unwrap();
    let m = Package {
        schema_version: 1,
        model_id: "synthetic-mdx23c".into(),
        config: f.config,
        weights_sha256: sha256_file(&weights).unwrap(),
        checkpoint_sha256: "0".repeat(64),
        source_url: "original synthetic contract".into(),
        forward_revision: ancha_models::mdx23c::REFERENCE_REVISION.into(),
        weight_license: "MIT synthetic data".into(),
    };
    m.validate().unwrap();
    std::fs::write(
        path.join("manifest.json"),
        serde_json::to_vec_pretty(&m).unwrap(),
    )
    .unwrap();
    m
}
