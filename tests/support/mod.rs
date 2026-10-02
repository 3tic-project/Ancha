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
    package(directory, family, |_| {})
}

/// [`tiny_package`] wide enough for the GPU kernels: 64-wide heads and 128-multiple projections.
#[allow(dead_code)]
pub fn kernel_package(directory: &Path, family: Family) -> Manifest {
    package(directory, family, |c| {
        c.dim = 128;
        c.heads = 2;
        c.head_dim = 64;
        c.ff_mult = 4;
    })
}

fn package(directory: &Path, family: Family, adjust: impl FnOnce(&mut ModelConfig)) -> Manifest {
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
    // Equal widths are not adjacent, exercising grouped band GEMMs and reordering.
    c.bands = if family == Family::BsRoformer {
        vec![
            (0..300).collect(),
            (300..700).collect(),
            (700..1000).collect(),
            (1000..1025).collect(),
        ]
    } else {
        vec![
            (0..400).collect(),
            (300..900).collect(),
            (625..1025).collect(),
        ]
    };
    if family == Family::MelBandRoformer {
        c.stems = vec!["vocals".into(), "instrumental".into()];
    }
    adjust(&mut c);
    let (dim, inner) = (c.dim, c.heads * c.head_dim);
    let (hidden, mask) = (c.dim * c.ff_mult, c.dim * c.mask_expansion);
    let mut tensors = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    let mut insert = |key: String, shape: Vec<usize>| {
        let values: Vec<f32> = (0..shape.iter().product())
            .map(|i| {
                if key.ends_with("gamma") {
                    0.75 + (i % 3) as f32 * 0.25
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
        insert(format!("{p}.1.weight"), vec![dim, width]);
        insert(format!("{p}.1.bias"), vec![dim]);
        for stem in 0..c.stems.len() {
            let p = format!("mask_estimators.{stem}.to_freqs.{band}.0");
            insert(format!("{p}.0.weight"), vec![mask, dim]);
            insert(format!("{p}.0.bias"), vec![mask]);
            let last = if family == Family::MelBandRoformer {
                insert(format!("{p}.2.weight"), vec![mask, mask]);
                insert(format!("{p}.2.bias"), vec![mask]);
                4
            } else {
                2
            };
            insert(format!("{p}.{last}.weight"), vec![width * 2, mask]);
            insert(format!("{p}.{last}.bias"), vec![width * 2]);
        }
    }
    for axis in 0..2 {
        let p = format!("layers.0.{axis}");
        insert(format!("{p}.layers.0.0.norm.gamma"), vec![dim]);
        insert(
            format!("{p}.layers.0.0.to_qkv.weight"),
            vec![3 * inner, dim],
        );
        insert(
            format!("{p}.layers.0.0.to_gates.weight"),
            vec![c.heads, dim],
        );
        insert(format!("{p}.layers.0.0.to_gates.bias"), vec![c.heads]);
        insert(format!("{p}.layers.0.0.to_out.0.weight"), vec![dim, inner]);
        insert(
            format!("{p}.layers.0.0.rotary_embed.freqs"),
            vec![c.head_dim / 2],
        );
        insert(format!("{p}.layers.0.1.net.0.gamma"), vec![dim]);
        insert(format!("{p}.layers.0.1.net.1.weight"), vec![hidden, dim]);
        insert(format!("{p}.layers.0.1.net.1.bias"), vec![hidden]);
        insert(format!("{p}.layers.0.1.net.4.weight"), vec![dim, hidden]);
        insert(format!("{p}.layers.0.1.net.4.bias"), vec![dim]);
        if family == Family::MelBandRoformer {
            insert(format!("{p}.norm.gamma"), vec![dim]);
        }
    }
    if family == Family::BsRoformer {
        insert("final_norm.gamma".into(), vec![dim]);
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
