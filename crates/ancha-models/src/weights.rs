use crate::config::Manifest;
use anyhow::{Context, Result, ensure};
use burn::{
    module::Param,
    nn::Linear,
    tensor::{Tensor, TensorData, backend::Backend},
};
use safetensors::{SafeTensors, tensor::Dtype};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs::File, io::Read, path::Path};

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn read_manifest(package: &Path) -> Result<Manifest> {
    let path = package.join("manifest.json");
    let m: Manifest = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )?;
    m.validate()?;
    Ok(m)
}

/// Weight storage is released once model construction finishes. No random initialization.
pub struct Weights<'a> {
    tensors: SafeTensors<'a>,
    used: HashSet<String>,
}

impl<'a> Weights<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        Ok(Self {
            tensors: SafeTensors::deserialize(bytes)?,
            used: HashSet::new(),
        })
    }

    pub fn tensor<B: Backend, const D: usize>(
        &mut self,
        name: &str,
        shape: [usize; D],
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let view = self
            .tensors
            .tensor(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            view.shape() == shape,
            "{name}: expected {shape:?}, got {:?}",
            view.shape()
        );
        ensure!(
            view.dtype() == Dtype::F32,
            "{name}: schema 1 requires F32 weights, got {:?}",
            view.dtype()
        );
        let data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4-byte tensor element")))
            .collect();
        ensure!(
            data.iter().all(|x| x.is_finite()),
            "{name}: non-finite weight"
        );
        self.used.insert(name.into());
        Ok(Tensor::from_data(TensorData::new(data, shape), device))
    }

    pub fn linear<B: Backend>(
        &mut self,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
        device: &B::Device,
    ) -> Result<Linear<B>> {
        let weight = self
            .tensor::<B, 2>(&format!("{name}.weight"), [output, input], device)?
            .transpose();
        let bias = if bias {
            Some(Param::from_tensor(self.tensor(
                &format!("{name}.bias"),
                [output],
                device,
            )?))
        } else {
            None
        };
        Ok(Linear {
            weight: Param::from_tensor(weight),
            bias,
        })
    }

    pub fn finish(self) -> Result<usize> {
        let unused: Vec<_> = self
            .tensors
            .names()
            .into_iter()
            .filter(|n| !self.used.contains(*n))
            .take(8)
            .collect();
        ensure!(
            unused.is_empty(),
            "unrecognized checkpoint tensors: {unused:?}"
        );
        Ok(self.used.len())
    }
}
