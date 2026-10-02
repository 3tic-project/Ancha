//! One-time, Python-free checkpoint conversion. Adapted from eplyt's MIT converter;
//! original revision and license are preserved in THIRD_PARTY_NOTICES.md.
use crate::{
    config::{Manifest, ModelConfig},
    weights::sha256_file,
};
use anyhow::{Context, Result, bail, ensure};
use burn::tensor::TensorData;
use burn_store::pytorch::PytorchReader;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use std::{collections::HashMap, path::Path};

pub fn convert_checkpoint(
    source: &Path,
    output: &Path,
    model_id: &str,
    config: ModelConfig,
    source_url: String,
    license: String,
) -> Result<Manifest> {
    config.validate()?;
    ensure!(
        !output.exists(),
        "model package already exists: {}",
        output.display()
    );
    let reader = open_reader(source)?;
    let mut owned = Vec::<(String, TensorData)>::new();
    for name in reader.keys() {
        let data = reader
            .get(&name)
            .context("checkpoint tensor disappeared")?
            .to_data()
            .map_err(|e| anyhow::anyhow!("read tensor {name}: {e:?}"))?;
        let data = data.convert::<f32>();
        ensure!(
            data.to_vec::<f32>()?.iter().all(|x| x.is_finite()),
            "non-finite tensor {name}"
        );
        owned.push((name, data));
    }
    ensure!(!owned.is_empty(), "checkpoint has no tensors");
    let views: HashMap<_, _> = owned
        .iter()
        .map(|(name, data)| {
            Ok((
                name.clone(),
                TensorView::new(Dtype::F32, data.shape.to_vec(), data.as_bytes())?,
            ))
        })
        .collect::<Result<_>>()?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    // Conversion is local and atomic: publish only a complete validated package.
    let stage = parent.join(format!(".ancha-convert-{}", std::process::id()));
    ensure!(!stage.exists(), "conversion staging path already exists");
    std::fs::create_dir(&stage)?;
    let result = (|| {
        let weights = stage.join("model.safetensors");
        serialize_to_file(&views, None, &weights)?;
        let manifest = Manifest {
            schema_version: 1,
            model_id: model_id.into(),
            config: config.clone(),
            weights_sha256: sha256_file(&weights)?,
            checkpoint_sha256: sha256_file(source)?,
            source_url,
            weight_license: license,
            forward_revision: if config.family == crate::config::Family::HyperaceV2 {
                "HyperACE/5b1f8283125d5e4a3614d0e3635a636e09c84059;bs_roformer.py=48571e20d70ea8f245cffc6afbfa279f62042e7ba16fbaa3fe43dd2cbc25e1db".into()
            } else {
                "MSST/84b1eac0887756b4f1a9d7a1ff49105939749ed2".into()
            },
        };
        manifest.validate()?;
        std::fs::write(
            stage.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        std::fs::rename(&stage, output)?;
        Ok(manifest)
    })();
    if stage.exists() {
        let _ = std::fs::remove_dir_all(stage);
    }
    result
}

fn open_reader(path: &Path) -> Result<PytorchReader> {
    if let Ok(reader) = PytorchReader::new(path)
        && !reader.is_empty()
    {
        return Ok(reader);
    }
    for key in ["state_dict", "model", "model_state_dict"] {
        if let Ok(reader) = PytorchReader::with_top_level_key(path, key)
            && !reader.is_empty()
        {
            return Ok(reader);
        }
    }
    bail!("no supported tensor state_dict in {}", path.display())
}
