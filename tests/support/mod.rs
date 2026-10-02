use ancha_models::{
    config::{Family, Manifest, ModelConfig},
    weights::sha256_file,
};
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use std::{collections::BTreeMap, path::Path};

/// Original synthetic weights, created in memory: no external model/audio in CI.
pub fn tiny_package(directory: &Path, family: Family) -> Manifest {
    let mut c = ModelConfig::leap_xe(false);
    c.family = family;
    c.dim = 4;
    c.depth = 1;
    c.heads = 1;
    c.head_dim = 4;
    c.ff_mult = 1;
    // The same public depth=2 means 2 Linears for BS and 3 for Mel.
    c.mask_depth = 2;
    c.mask_expansion = 1;
    c.chunk_samples = 4096;
    c.overlap = 2;
    c.bands = if family == Family::BsRoformer {
        vec![(0..512).collect(), (512..1025).collect()]
    } else {
        vec![(0..800).collect(), (500..1025).collect()]
    };
    if family == Family::MelBandRoformer {
        c.stems = vec!["vocals".into(), "instrumental".into()];
    }
    let mut tensors = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    let mut insert = |key: String, shape: Vec<usize>| {
        let values: Vec<f32> = (0..shape.iter().product())
            .map(|i| {
                if key.ends_with("gamma") {
                    1.0
                } else if key.ends_with("freqs") {
                    if i == 0 { 1.0 } else { 0.01 }
                } else {
                    ((i % 17) as f32 - 8.0) * 0.001
                }
            })
            .collect();
        tensors.insert(
            key,
            (shape, values.iter().flat_map(|x| x.to_le_bytes()).collect()),
        );
    };
    for (band, bins) in c.bands.iter().enumerate() {
        let width = bins.len() * 4;
        let p = format!("band_split.to_features.{band}");
        insert(format!("{p}.0.gamma"), vec![width]);
        insert(format!("{p}.1.weight"), vec![4, width]);
        insert(format!("{p}.1.bias"), vec![4]);
        for stem in 0..c.stems.len() {
            let p = format!("mask_estimators.{stem}.to_freqs.{band}.0");
            insert(format!("{p}.0.weight"), vec![4, 4]);
            insert(format!("{p}.0.bias"), vec![4]);
            let last = if family == Family::MelBandRoformer {
                insert(format!("{p}.2.weight"), vec![4, 4]);
                insert(format!("{p}.2.bias"), vec![4]);
                4
            } else {
                2
            };
            insert(format!("{p}.{last}.weight"), vec![width * 2, 4]);
            insert(format!("{p}.{last}.bias"), vec![width * 2]);
        }
    }
    for axis in 0..2 {
        let p = format!("layers.0.{axis}");
        insert(format!("{p}.layers.0.0.norm.gamma"), vec![4]);
        insert(format!("{p}.layers.0.0.to_qkv.weight"), vec![12, 4]);
        insert(format!("{p}.layers.0.0.to_gates.weight"), vec![1, 4]);
        insert(format!("{p}.layers.0.0.to_gates.bias"), vec![1]);
        insert(format!("{p}.layers.0.0.to_out.0.weight"), vec![4, 4]);
        insert(format!("{p}.layers.0.0.rotary_embed.freqs"), vec![2]);
        insert(format!("{p}.layers.0.1.net.0.gamma"), vec![4]);
        for j in [1, 4] {
            insert(format!("{p}.layers.0.1.net.{j}.weight"), vec![4, 4]);
            insert(format!("{p}.layers.0.1.net.{j}.bias"), vec![4]);
        }
        if family == Family::MelBandRoformer {
            insert(format!("{p}.norm.gamma"), vec![4]);
        }
    }
    if family == Family::BsRoformer {
        insert("final_norm.gamma".into(), vec![4]);
    }
    let views: BTreeMap<_, _> = tensors
        .iter()
        .map(|(key, (shape, bytes))| {
            (
                key.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    std::fs::create_dir_all(directory).unwrap();
    let weights = directory.join("model.safetensors");
    serialize_to_file(&views, None, &weights).unwrap();
    let m = Manifest {
        schema_version: 1,
        model_id: "synthetic-ci-fixture".into(),
        weights_sha256: sha256_file(&weights).unwrap(),
        checkpoint_sha256: "0".repeat(64),
        source_url: "original synthetic tensors".into(),
        forward_revision: "synthetic schema-1".into(),
        weight_license: "MIT".into(),
        config: c,
    };
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&m).unwrap(),
    )
    .unwrap();
    m
}
