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

/// Run `build` over `bytes` while their SHA-256 is computed on another thread, and
/// return its result only if the digest matches. On CPUs without SHA extensions hashing
/// an 830 MB package takes about 5 s, longer than parsing and uploading it.
pub fn verified<T>(bytes: &[u8], expected: &str, build: impl FnOnce() -> Result<T>) -> Result<T> {
    let (digest, built) = std::thread::scope(|scope| {
        let hasher = scope.spawn(|| format!("{:x}", Sha256::digest(bytes)));
        let built = build();
        (
            hasher.join().expect("weight hashing thread panicked"),
            built,
        )
    });
    ensure!(digest == expected, "model checksum mismatch");
    built
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
        let data = self.take(name, &shape)?;
        Ok(Tensor::from_data(TensorData::new(data, shape), device))
    }

    /// Checked host copy, for load-time folding before upload.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
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
        Ok(data)
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

/// Fold exact load-time scales into a PyTorch `[out, in]` matrix and return
/// selected output rows as a Burn `[in, out]` Linear:
/// `W'[i][o] = W[o][i] * input_scale[i] * output_scale`, `b' = b * output_scale`.
pub fn folded_linear<B: Backend>(
    weight: &[f32],
    input: usize,
    rows: std::ops::Range<usize>,
    input_scale: &[f32],
    output_scale: f32,
    bias: Option<&[f32]>,
    device: &B::Device,
) -> Linear<B> {
    debug_assert_eq!(input_scale.len(), input);
    let output = rows.len();
    let mut data = vec![0f32; input * output];
    for (o, row) in rows.clone().enumerate() {
        for i in 0..input {
            data[i * output + o] = weight[row * input + i] * input_scale[i] * output_scale;
        }
    }
    Linear {
        weight: Param::from_tensor(Tensor::from_data(
            TensorData::new(data, [input, output]),
            device,
        )),
        bias: bias.map(|b| {
            let b: Vec<f32> = b[rows].iter().map(|v| v * output_scale).collect();
            Param::from_tensor(Tensor::from_data(TensorData::new(b, [output]), device))
        }),
    }
}
