//! Strict classic MDX ONNX executor. Runs supported graphs directly in Burn;
//! neither ONNX Runtime nor Python is linked into the application.
use crate::{spatial::conv2d_gemm, weights::sha256_file};
use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{
    Tensor, TensorData,
    activation::relu,
    backend::Backend,
    module::{conv_transpose2d, conv2d},
    ops::{ConvOptions, ConvTransposeOptions},
};
use onnx_rs::ast::{DataType, Node, OpType};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MdxConfig {
    pub model_id: String,
    pub weights_sha256: String,
    pub task: String,
    pub predicted: String,
    pub residual: String,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub bins: usize,
    pub frames: usize,
    pub compensate: f32,
    pub source_revision: String,
}
impl MdxConfig {
    pub fn identify(path: &Path) -> Result<Self> {
        let digest = sha256_file(path)?;
        let (name, task, predicted, residual, fft, bins, compensate) = match digest.as_str() {
            "f4f365207c56deb115bceedff3ad8fe98a751c745f9e370cecec6226b8b47184" => (
                "mdx-9482",
                "all_vocals",
                "all_vocals",
                "instrumental",
                6144,
                2048,
                1.035,
            ),
            "e3167c87333a48548413e972a286bf40bf5694001d2853861eb1435953f02d63" => (
                "mdx-kara",
                "lead_vocals",
                "lead_vocals",
                "karaoke_mix",
                6144,
                2048,
                1.035,
            ),
            "bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4" => (
                "mdx-kara-2",
                "lead_vocals",
                "karaoke_mix",
                "lead_vocals",
                5120,
                2048,
                1.065,
            ),
            "197f8ab296df850f961e68c595f6649acb7d9e621b5600b460f3458967299112" => (
                "mdx-inst-hq-2",
                "all_vocals",
                "instrumental",
                "all_vocals",
                6144,
                3072,
                1.033,
            ),
            _ => bail!(
                "unregistered ONNX checksum {digest}; refusing to guess FFT or stem semantics"
            ),
        };
        Ok(Self {
            model_id: name.into(),
            weights_sha256: digest,
            task: task.into(),
            predicted: predicted.into(),
            residual: residual.into(),
            sample_rate: 44100,
            n_fft: fft,
            hop: 1024,
            bins,
            frames: 256,
            compensate,
            source_revision: "UVR/5517e0cf0d1acd16a1618eeedec596957523f9e1".into(),
        })
    }
    pub fn chunk_samples(&self) -> usize {
        self.hop * (self.frames - 1)
    }
}
struct Raw {
    shape: [usize; 4],
    data: Vec<f32>,
}
enum Op {
    Conv {
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
        transpose: bool,
        padding_out: [usize; 2],
    },
    Relu,
    Transpose([usize; 4]),
    MatMul,
    Add,
    Mul,
    Bn {
        epsilon: f32,
    },
}
struct Instruction<B: Backend> {
    op: Op,
    inputs: Vec<usize>,
    output: usize,
    affine: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
}
pub struct Graph<B: Backend> {
    constants: Vec<Option<Tensor<B, 4>>>,
    instructions: Vec<Instruction<B>>,
    uses: Vec<usize>,
    input: usize,
    output: usize,
    pub folded_bn: usize,
    pub nodes: usize,
}
fn ints(n: &Node<'_>, key: &str, default: &[i64]) -> Vec<i64> {
    n.attribute
        .iter()
        .find(|a| a.name == key)
        .map_or_else(|| default.to_vec(), |a| a.ints.clone())
}
fn int(n: &Node<'_>, key: &str, default: i64) -> i64 {
    n.attribute
        .iter()
        .find(|a| a.name == key)
        .map_or(default, |a| a.i)
}
fn float(n: &Node<'_>, key: &str, default: f32) -> f32 {
    n.attribute
        .iter()
        .find(|a| a.name == key)
        .map_or(default, |a| a.f)
}
fn pair(v: Vec<i64>, positive: bool) -> Result<[usize; 2]> {
    ensure!(
        v.len() == 2 && v.iter().all(|&v| if positive { v > 0 } else { v >= 0 }),
        "invalid 2D convolution attribute"
    );
    Ok([v[0] as usize, v[1] as usize])
}
impl<B: Backend> Graph<B> {
    pub fn from_bytes(bytes: &[u8], optimized: bool, d: &B::Device) -> Result<Self> {
        let m = onnx_rs::parse(bytes).context("parse ONNX")?;
        ensure!(
            m.opset_import.len() == 1
                && m.opset_import[0].domain.is_empty()
                && m.opset_import[0].version == 13,
            "classic MDX executor requires ONNX opset 13"
        );
        let g = m.graph.context("missing ONNX graph")?;
        ensure!(
            g.input.len() == 1 && g.output.len() == 1 && g.sparse_initializer.is_empty(),
            "expected one input and one output, no sparse initializers"
        );
        let mut raw = HashMap::<String, Raw>::new();
        for t in &g.initializer {
            ensure!(
                t.data_type() == DataType::Float
                    && t.dims().len() <= 4
                    && t.dims().iter().all(|&v| v > 0),
                "unsupported initializer {}",
                t.name()
            );
            let mut shape = [1; 4];
            for (i, &v) in t.dims().iter().enumerate() {
                shape[4 - t.dims().len() + i] = v as usize;
            }
            let data = t
                .as_f32()
                .context("external or invalid F32 initializer")?
                .into_owned();
            ensure!(
                data.len() == shape.iter().product::<usize>() && data.iter().all(|x| x.is_finite()),
                "invalid initializer {}",
                t.name()
            );
            ensure!(
                raw.insert(t.name().into(), Raw { shape, data }).is_none(),
                "duplicate initializer"
            );
        }
        let mut consumers = HashMap::<&str, usize>::new();
        for n in &g.node {
            for &s in &n.input {
                *consumers.entry(s).or_default() += 1;
            }
        }
        let mut skipped = HashSet::new();
        let mut redirected = HashMap::new();
        let mut folded_bn = 0;
        if optimized {
            for (i, n) in g.node.iter().enumerate() {
                if n.op_type != OpType::ConvTranspose
                    || n.output.len() != 1
                    || consumers.get(n.output[0]) != Some(&1)
                    || g.output[0].name == n.output[0]
                {
                    continue;
                }
                let Some((j, bn)) = g.node.iter().enumerate().find(|(_, v)| {
                    v.op_type == OpType::BatchNormalization && v.input.first() == n.output.first()
                }) else {
                    continue;
                };
                ensure!(
                    bn.input.len() == 5 && bn.output.len() == 1 && int(bn, "training_mode", 0) == 0,
                    "unsupported BN"
                );
                ensure!(
                    n.input.len() == 3 && int(n, "group", 1) == 1,
                    "folding expects ungrouped transposed convolution with bias"
                );
                let gamma = &raw[bn.input[1]].data;
                let beta = &raw[bn.input[2]].data;
                let mean = &raw[bn.input[3]].data;
                let var = &raw[bn.input[4]].data;
                ensure!(
                    gamma.len() == beta.len()
                        && gamma.len() == mean.len()
                        && gamma.len() == var.len()
                        && var.iter().all(|&v| v >= 0.),
                    "invalid BN statistics"
                );
                let eps = float(bn, "epsilon", 1e-5);
                ensure!(eps > 0. && eps.is_finite(), "invalid BN epsilon");
                let scale: Vec<_> = gamma
                    .iter()
                    .zip(var)
                    .map(|(&g, &v)| g / (v + eps).sqrt())
                    .collect();
                let shift: Vec<_> = beta
                    .iter()
                    .zip(mean)
                    .zip(&scale)
                    .map(|((&b, &m), &s)| b - m * s)
                    .collect();
                // Clone folded weights: shared initializer consumers retain the original values.
                let weight = &raw[n.input[1]];
                let mut data = weight.data.clone();
                let [ci, co, kh, kw] = weight.shape;
                ensure!(co == scale.len(), "BN / ConvTranspose channel mismatch");
                for input in 0..ci {
                    for (out, &s) in scale.iter().enumerate() {
                        for k in 0..kh * kw {
                            data[(input * co + out) * kh * kw + k] *= s;
                        }
                    }
                }
                let bias = &raw[n.input[2]];
                ensure!(bias.data.len() == co, "invalid transpose bias");
                let bd: Vec<_> = bias
                    .data
                    .iter()
                    .zip(&scale)
                    .zip(&shift)
                    .map(|((&b, &s), &a)| b * s + a)
                    .collect();
                let wn = format!("__ancha_fold_{i}_weight");
                let bn_name = format!("__ancha_fold_{i}_bias");
                raw.insert(
                    wn,
                    Raw {
                        shape: [ci, co, kh, kw],
                        data,
                    },
                );
                raw.insert(
                    bn_name,
                    Raw {
                        shape: [1, 1, 1, co],
                        data: bd,
                    },
                );
                skipped.insert(j);
                redirected.insert(i, bn.output[0]);
                folded_bn += 1;
            }
        }
        let mut ids = HashMap::<String, usize>::new();
        let mut constants = Vec::new();
        // Sort names so schedules and tests remain deterministic.
        let mut names: Vec<_> = raw.keys().cloned().collect();
        names.sort();
        for name in names {
            let v = &raw[&name];
            ids.insert(name, constants.len());
            constants.push(Some(Tensor::from_data(
                TensorData::new(v.data.clone(), v.shape),
                d,
            )));
        }
        let input = constants.len();
        ids.insert(g.input[0].name.into(), input);
        constants.push(None);
        let mut instructions = Vec::new();
        for (i, n) in g.node.iter().enumerate() {
            if skipped.contains(&i) {
                continue;
            }
            ensure!(
                n.domain.is_empty() && n.output.len() == 1,
                "unsupported node domain or multi-output"
            );
            let allowed: &[&str] = match n.op_type {
                OpType::Conv | OpType::ConvTranspose => &[
                    "dilations",
                    "group",
                    "kernel_shape",
                    "pads",
                    "strides",
                    "output_padding",
                ],
                OpType::BatchNormalization => &["epsilon", "momentum", "training_mode"],
                OpType::Transpose => &["perm"],
                _ => &[],
            };
            ensure!(
                n.attribute.iter().all(|a| allowed.contains(&a.name)),
                "unsupported attribute in {}",
                n.name
            );
            let mut inputs = n
                .input
                .iter()
                .map(|s| {
                    ids.get(*s)
                        .copied()
                        .with_context(|| format!("unknown input {s}"))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut affine = None;
            let op = match n.op_type {
                OpType::Conv | OpType::ConvTranspose => {
                    ensure!(
                        inputs.len() == 2 || inputs.len() == 3,
                        "invalid conv inputs"
                    );
                    let pads = ints(n, "pads", &[0, 0, 0, 0]);
                    ensure!(
                        pads.len() == 4 && pads[0] == pads[2] && pads[1] == pads[3],
                        "asymmetric convolution unsupported"
                    );
                    let groups = int(n, "group", 1);
                    ensure!(groups > 0, "invalid groups");
                    if redirected.contains_key(&i) {
                        inputs[1] = ids[&format!("__ancha_fold_{i}_weight")];
                        inputs[2] = ids[&format!("__ancha_fold_{i}_bias")];
                    }
                    Op::Conv {
                        stride: pair(ints(n, "strides", &[1, 1]), true)?,
                        padding: pair(pads[..2].to_vec(), false)?,
                        dilation: pair(ints(n, "dilations", &[1, 1]), true)?,
                        groups: groups as usize,
                        transpose: n.op_type == OpType::ConvTranspose,
                        padding_out: pair(ints(n, "output_padding", &[0, 0]), false)?,
                    }
                }
                OpType::Relu => {
                    ensure!(inputs.len() == 1, "invalid Relu inputs");
                    Op::Relu
                }
                OpType::Transpose => {
                    ensure!(inputs.len() == 1, "invalid transpose inputs");
                    let p = ints(n, "perm", &[3, 2, 1, 0]);
                    ensure!(
                        p.len() == 4
                            && p.iter().copied().collect::<HashSet<_>>()
                                == HashSet::from([0, 1, 2, 3]),
                        "invalid permutation"
                    );
                    Op::Transpose([p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize])
                }
                OpType::MatMul | OpType::Add | OpType::Mul => {
                    ensure!(inputs.len() == 2, "invalid binary inputs");
                    match n.op_type {
                        OpType::MatMul => Op::MatMul,
                        OpType::Add => Op::Add,
                        _ => Op::Mul,
                    }
                }
                OpType::BatchNormalization => {
                    ensure!(
                        inputs.len() == 5 && int(n, "training_mode", 0) == 0,
                        "only eval BN supported"
                    );
                    let eps = float(n, "epsilon", 1e-5);
                    ensure!(eps.is_finite() && eps > 0., "invalid BN epsilon");
                    let gamma = &raw[n.input[1]].data;
                    let beta = &raw[n.input[2]].data;
                    let mean = &raw[n.input[3]].data;
                    let var = &raw[n.input[4]].data;
                    ensure!(
                        gamma.len() == beta.len()
                            && gamma.len() == mean.len()
                            && gamma.len() == var.len()
                            && var.iter().all(|&v| v >= 0.),
                        "invalid BN statistics"
                    );
                    if optimized {
                        let scale: Vec<_> = gamma
                            .iter()
                            .zip(var)
                            .map(|(&g, &v)| g / (v + eps).sqrt())
                            .collect();
                        let shift: Vec<_> = beta
                            .iter()
                            .zip(mean)
                            .zip(&scale)
                            .map(|((&b, &m), &s)| b - m * s)
                            .collect();
                        let c = scale.len();
                        affine = Some((
                            Tensor::<B, 1>::from_floats(scale.as_slice(), d).reshape([1, c, 1, 1]),
                            Tensor::<B, 1>::from_floats(shift.as_slice(), d).reshape([1, c, 1, 1]),
                        ));
                    }
                    Op::Bn { epsilon: eps }
                }
                _ => bail!("unsupported ONNX operation {:?} at {}", n.op_type, n.name),
            };
            let output = constants.len();
            let name = redirected.get(&i).copied().unwrap_or(n.output[0]);
            ensure!(
                ids.insert(name.into(), output).is_none(),
                "duplicate ONNX value"
            );
            constants.push(None);
            instructions.push(Instruction {
                op,
                inputs,
                output,
                affine,
            });
        }
        let output = *ids.get(g.output[0].name).context("graph output missing")?;
        let mut uses = vec![0; constants.len()];
        for n in &instructions {
            for &input in &n.inputs {
                uses[input] += 1;
            }
        }
        uses[output] += 1;
        Ok(Self {
            constants,
            instructions,
            uses,
            input,
            output,
            folded_bn,
            nodes: g.node.len(),
        })
    }
    /// `conv_gemm` runs ungrouped convolutions as patch gather + GEMM (exact; for GPUs
    /// whose backend convolution falls back to direct kernels).
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        conv_gemm: bool,
        cancelled: &AtomicBool,
    ) -> Result<Tensor<B, 4>> {
        let mut values = self.constants.clone();
        values[self.input] = Some(input);
        let mut uses = self.uses.clone();
        for n in &self.instructions {
            ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
            let v: Vec<_> = n
                .inputs
                .iter()
                .map(|&i| values[i].as_ref().expect("validated schedule").clone())
                .collect();
            let y = match &n.op {
                Op::Conv {
                    stride,
                    padding,
                    dilation,
                    groups,
                    transpose,
                    padding_out,
                } => {
                    let bias = if v.len() == 3 {
                        let len: usize = v[2].dims().iter().product();
                        Some(v[2].clone().reshape([len]))
                    } else {
                        None
                    };
                    if *transpose {
                        conv_transpose2d(
                            v[0].clone(),
                            v[1].clone(),
                            bias,
                            ConvTransposeOptions::new(
                                *stride,
                                *padding,
                                *padding_out,
                                *dilation,
                                *groups,
                            ),
                        )
                    } else {
                        let gemm = (conv_gemm && *groups == 1 && *dilation == [1, 1])
                            .then(|| {
                                conv2d_gemm(
                                    v[0].clone(),
                                    v[1].clone(),
                                    bias.clone(),
                                    *stride,
                                    *padding,
                                )
                            })
                            .flatten();
                        gemm.unwrap_or_else(|| {
                            conv2d(
                                v[0].clone(),
                                v[1].clone(),
                                bias,
                                ConvOptions::new(*stride, *padding, *dilation, *groups),
                            )
                        })
                    }
                }
                Op::Relu => relu(v[0].clone()),
                Op::Transpose(p) => v[0].clone().permute(*p),
                Op::MatMul => v[0].clone().matmul(v[1].clone()),
                Op::Add => v[0].clone() + v[1].clone(),
                Op::Mul => v[0].clone() * v[1].clone(),
                Op::Bn { epsilon, .. } => {
                    if let Some((scale, shift)) = &n.affine {
                        v[0].clone() * scale.clone() + shift.clone()
                    } else {
                        let c: usize = v[1].dims().iter().product();
                        let gamma = v[1].clone().reshape([1, c, 1, 1]);
                        let beta = v[2].clone().reshape([1, c, 1, 1]);
                        let mean = v[3].clone().reshape([1, c, 1, 1]);
                        let var = v[4].clone().reshape([1, c, 1, 1]);
                        (v[0].clone() - mean) / var.add_scalar(*epsilon).sqrt() * gamma + beta
                    }
                }
            };
            for &i in &n.inputs {
                uses[i] -= 1;
                if uses[i] == 0 {
                    values[i] = None;
                }
            }
            values[n.output] = Some(y);
        }
        values[self.output].take().context("missing graph result")
    }
}

pub struct Mdx<B: Backend> {
    pub config: MdxConfig,
    pub graph: Graph<B>,
}
impl<B: Backend> Mdx<B> {
    pub fn load(path: &Path, optimized: bool, d: &B::Device) -> Result<Self> {
        let config = MdxConfig::identify(path)?;
        let bytes = std::fs::read(path)?;
        let m = onnx_rs::parse(&bytes)?;
        let g = m.graph.as_ref().context("missing graph")?;
        for info in g.input.iter().chain(&g.output) {
            let tensor = match info.r#type.as_ref().and_then(|t| t.value.as_ref()) {
                Some(onnx_rs::ast::TypeValue::Tensor(t)) => t,
                _ => bail!("invalid graph IO type"),
            };
            ensure!(tensor.elem_type == DataType::Float, "expected F32 graph IO");
            let shape = &tensor.shape.as_ref().context("missing graph IO shape")?.dim;
            ensure!(shape.len() == 4, "expected NCHW graph IO");
            for (dim, expected) in shape[1..].iter().zip([4, config.bins, config.frames]) {
                ensure!(
                    dim.value == onnx_rs::ast::Dimension::Value(expected as i64),
                    "MDX graph dimension disagrees with registered contract"
                );
            }
        }
        let graph = Graph::from_bytes(&bytes, optimized, d)?;
        Ok(Self { config, graph })
    }
}
