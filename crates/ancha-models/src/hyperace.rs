//! Complete v2 SegmModel. Architecture follows the author's pinned forward;
//! no learned branch is dropped, and InstanceNorm is never folded into Conv.
use crate::{
    spatial::{Conv, InstanceNorm, frequency_shuffle, resize, silu},
    weights::Weights,
};
use anyhow::{Result, ensure};
use burn::{
    nn::Linear,
    tensor::{Tensor, activation::softmax, backend::Backend},
};
use std::sync::atomic::{AtomicBool, Ordering};

struct Cn<B: Backend> {
    conv: Conv<B>,
    norm: InstanceNorm<B>,
}
impl<B: Backend> Cn<B> {
    fn load(w: &mut Weights<'_>, p: &str, ci: usize, co: usize, d: &B::Device) -> Result<Self> {
        Ok(Self {
            conv: Conv::load(w, &format!("{p}.conv"), [ci, co], 1, [1, 1], 1, d)?,
            norm: InstanceNorm::load(w, &format!("{p}.bn"), co, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        silu(self.norm.forward(self.conv.forward(x, gemm)))
    }
}
struct Ds<B: Backend> {
    dw: Conv<B>,
    pw: Conv<B>,
    norm: InstanceNorm<B>,
}
impl<B: Backend> Ds<B> {
    fn load(
        w: &mut Weights<'_>,
        p: &str,
        ci: usize,
        co: usize,
        stride: [usize; 2],
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            dw: Conv::load(w, &format!("{p}.dwconv"), [ci, ci], 3, stride, ci, d)?,
            pw: Conv::load(w, &format!("{p}.pwconv"), [ci, co], 1, [1, 1], 1, d)?,
            norm: InstanceNorm::load(w, &format!("{p}.bn"), co, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        silu(
            self.norm
                .forward(self.pw.forward(self.dw.forward(x, gemm), gemm)),
        )
    }
}
struct C3<B: Backend> {
    a: Cn<B>,
    b: Cn<B>,
    out: Cn<B>,
    blocks: Vec<(Ds<B>, Ds<B>)>,
}
impl<B: Backend> C3<B> {
    fn load(w: &mut Weights<'_>, p: &str, c: usize, n: usize, d: &B::Device) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..n {
            blocks.push((
                Ds::load(w, &format!("{p}.m.{i}.dsconv1"), c, c, [1, 1], d)?,
                Ds::load(w, &format!("{p}.m.{i}.dsconv2"), c, c, [1, 1], d)?,
            ));
        }
        Ok(Self {
            a: Cn::load(w, &format!("{p}.cv1"), c, c, d)?,
            b: Cn::load(w, &format!("{p}.cv2"), c, c, d)?,
            out: Cn::load(w, &format!("{p}.cv3"), 2 * c, c, d)?,
            blocks,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        let mut a = self.a.forward(x.clone(), gemm);
        for (first, second) in &self.blocks {
            a = a.clone() + second.forward(first.forward(a, gemm), gemm);
        }
        self.out
            .forward(Tensor::cat(vec![a, self.b.forward(x, gemm)], 1), gemm)
    }
}
struct C32<B: Backend> {
    a: Cn<B>,
    mid: C3<B>,
    out: Cn<B>,
}
impl<B: Backend> C32<B> {
    fn load(
        w: &mut Weights<'_>,
        p: &str,
        ci: usize,
        co: usize,
        n: usize,
        d: &B::Device,
    ) -> Result<Self> {
        let c = co / 2;
        Ok(Self {
            a: Cn::load(w, &format!("{p}.cv1"), ci, c, d)?,
            mid: C3::load(w, &format!("{p}.m"), c, n, d)?,
            out: Cn::load(w, &format!("{p}.cv2"), c, co, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        self.out
            .forward(self.mid.forward(self.a.forward(x, gemm), gemm), gemm)
    }
}
struct Graph<B: Backend> {
    a: Cn<B>,
    b: Cn<B>,
    out: Cn<B>,
    proto: Tensor<B, 3>,
    context: Linear<B>,
    query: Linear<B>,
    edge: Linear<B>,
    vertex: Linear<B>,
}
impl<B: Backend> Graph<B> {
    fn load(w: &mut Weights<'_>, p: &str, d: &B::Device) -> Result<Self> {
        let g = format!("{p}.ahc.adaptive_hyperedge_gen");
        Ok(Self {
            a: Cn::load(w, &format!("{p}.cv1"), 256, 256, d)?,
            b: Cn::load(w, &format!("{p}.cv2"), 256, 256, d)?,
            out: Cn::load(w, &format!("{p}.cv3"), 512, 256, d)?,
            proto: w
                .tensor::<B, 2>(&format!("{g}.global_proto"), [32, 256], d)?
                .reshape([1, 32, 256]),
            context: w.linear(&format!("{g}.context_mapper"), 512, 8192, false, d)?,
            query: w.linear(&format!("{g}.query_proj"), 256, 256, false, d)?,
            edge: w.linear(&format!("{p}.ahc.hypergraph_conv.W_e"), 256, 256, false, d)?,
            vertex: w.linear(&format!("{p}.ahc.hypergraph_conv.W_v"), 256, 256, false, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        let lateral = self.a.forward(x.clone(), gemm);
        let z = self.b.forward(x, gemm);
        let [b, c, h, w] = z.dims();
        let n = h * w;
        let z = z.reshape([b, c, n]).swap_dims(1, 2);
        let ctx = Tensor::cat(vec![z.clone().mean_dim(1), z.clone().max_dim(1)], 2);
        let proto = (self.proto.clone() + self.context.forward(ctx).reshape([b, 32, 256]))
            .reshape([b, 32, 8, 32])
            .permute([0, 2, 3, 1]);
        let query = self
            .query
            .forward(z.clone())
            .reshape([b, n, 8, 32])
            .permute([0, 2, 1, 3]);
        let incidence = softmax(
            (query.matmul(proto) * (32f32).powf(-0.5))
                .mean_dim(1)
                .reshape([b, n, 32])
                .swap_dims(1, 2),
            2,
        );
        let edges = silu(self.edge.forward(incidence.clone().matmul(z.clone())));
        let z = z + silu(self.vertex.forward(incidence.swap_dims(1, 2).matmul(edges)));
        self.out.forward(
            Tensor::cat(vec![z.swap_dims(1, 2).reshape([b, c, h, w]), lateral], 1),
            gemm,
        )
    }
}
struct Tfc<B: Backend> {
    n1: InstanceNorm<B>,
    c1: Conv<B>,
    nt1: InstanceNorm<B>,
    l1: Linear<B>,
    nt2: InstanceNorm<B>,
    l2: Linear<B>,
    n2: InstanceNorm<B>,
    c2: Conv<B>,
    skip: Conv<B>,
}
impl<B: Backend> Tfc<B> {
    fn load(w: &mut Weights<'_>, p: &str, c: usize, f: usize, d: &B::Device) -> Result<Self> {
        Ok(Self {
            n1: InstanceNorm::load(w, &format!("{p}.tfc1.0"), c, d)?,
            c1: Conv::load(w, &format!("{p}.tfc1.2"), [c, c], 3, [1, 1], 1, d)?,
            nt1: InstanceNorm::load(w, &format!("{p}.tdf.0"), c, d)?,
            l1: w.linear(&format!("{p}.tdf.2"), f, f / 4, false, d)?,
            nt2: InstanceNorm::load(w, &format!("{p}.tdf.3"), c, d)?,
            l2: w.linear(&format!("{p}.tdf.5"), f / 4, f, false, d)?,
            n2: InstanceNorm::load(w, &format!("{p}.tfc2.0"), c, d)?,
            c2: Conv::load(w, &format!("{p}.tfc2.2"), [c, c], 3, [1, 1], 1, d)?,
            skip: Conv::load(w, &format!("{p}.shortcut"), [c, c], 1, [1, 1], 1, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        let skip = self.skip.forward(x.clone(), gemm);
        let x = self.c1.forward(silu(self.n1.forward(x)), gemm);
        let t = self.l1.forward(silu(self.nt1.forward(x.clone())));
        let t = self.l2.forward(silu(self.nt2.forward(t)));
        self.c2.forward(silu(self.n2.forward(x + t)), gemm) + skip
    }
}
struct Up<B: Backend> {
    conv: Ds<B>,
    tdf: Vec<Tfc<B>>,
}
impl<B: Backend> Up<B> {
    fn load(
        w: &mut Weights<'_>,
        p: &str,
        ci: usize,
        co: usize,
        f: usize,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: Ds::load(w, &format!("{p}.conv"), ci, co * 2, [1, 1], d)?,
            tdf: (0..2)
                .map(|i| Tfc::load(w, &format!("{p}.out_conv.blocks.{i}"), co, f, d))
                .collect::<Result<_>>()?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, gemm: bool) -> Tensor<B, 4> {
        let mut x = frequency_shuffle(self.conv.forward(x, gemm), 2);
        for block in &self.tdf {
            x = block.forward(x, gemm);
        }
        x
    }
}

pub(crate) struct Segm<B: Backend> {
    stem: Ds<B>,
    encoder: Vec<(Ds<B>, C32<B>)>,
    fuse: Cn<B>,
    high: Vec<Graph<B>>,
    high_fuse: Cn<B>,
    low: C3<B>,
    final_fuse: Cn<B>,
    skips: Vec<Cn<B>>,
    h_maps: Vec<Cn<B>>,
    gammas: Vec<Tensor<B, 4>>,
    decoder: Vec<C32<B>>,
    final_decoder: C32<B>,
    up: Vec<Up<B>>,
    final_conv: Conv<B>,
}
impl<B: Backend> Segm<B> {
    pub(crate) fn load(w: &mut Weights<'_>, p: &str, d: &B::Device) -> Result<Self> {
        let channels = [256, 384, 512, 768];
        let mut encoder = Vec::new();
        let mut input = 64;
        for (i, &c) in channels.iter().enumerate() {
            let path = format!("{p}.backbone.p{}", i + 2);
            encoder.push((
                Ds::load(
                    w,
                    &format!("{path}.0"),
                    input,
                    c,
                    if i < 2 { [2, 1] } else { [2, 2] },
                    d,
                )?,
                C32::load(
                    w,
                    &format!("{path}.1"),
                    c,
                    c,
                    if i == 0 || i == 3 { 2 } else { 4 },
                    d,
                )?,
            ));
            input = c;
        }
        let mut skips = Vec::new();
        let mut h_maps = Vec::new();
        let mut gammas = Vec::new();
        let mut decoder = Vec::new();
        for (i, &c) in channels.iter().enumerate() {
            let j = i + 2;
            skips.push(Cn::load(w, &format!("{p}.decoder.skip_p{j}"), c, c, d)?);
            h_maps.push(Cn::load(w, &format!("{p}.decoder.h_to_d{j}"), 512, c, d)?);
            gammas.push(w.tensor(&format!("{p}.decoder.fusion_d{j}.gamma"), [1, c, 1, 1], d)?);
            if i > 0 {
                decoder.push(C32::load(
                    w,
                    &format!("{p}.decoder.up_d{j}"),
                    c,
                    channels[i - 1],
                    1,
                    d,
                )?);
            }
        }
        let hp = format!("{p}.hyperace");
        let mut up = Vec::new();
        let mut ci = 256;
        for i in 0..4 {
            up.push(Up::load(
                w,
                &format!("{p}.upsample_head.block{}", i + 1),
                ci,
                ci / 2,
                62 * (1 << (i + 1)),
                d,
            )?);
            ci /= 2;
        }
        Ok(Self {
            stem: Ds::load(w, &format!("{p}.backbone.stem"), 256, 64, [2, 1], d)?,
            encoder,
            fuse: Cn::load(w, &format!("{hp}.fuse_conv"), 1920, 512, d)?,
            high: (0..2)
                .map(|i| Graph::load(w, &format!("{hp}.high_order_branch.{i}"), d))
                .collect::<Result<_>>()?,
            high_fuse: Cn::load(w, &format!("{hp}.high_order_fuse"), 512, 256, d)?,
            low: C3::load(w, &format!("{hp}.low_order_branch.0"), 128, 1, d)?,
            final_fuse: Cn::load(w, &format!("{hp}.final_fuse"), 512, 512, d)?,
            skips,
            h_maps,
            gammas,
            decoder,
            final_decoder: C32::load(w, &format!("{p}.decoder.final_d2"), 256, 256, 1, d)?,
            up,
            final_conv: Conv::load(
                w,
                &format!("{p}.upsample_head.final_conv"),
                [16, 4],
                3,
                [1, 1],
                1,
                d,
            )?,
        })
    }
    pub(crate) fn forward(
        &self,
        x: Tensor<B, 4>,
        gemm: bool,
        cancelled: &AtomicBool,
    ) -> Result<Tensor<B, 4>> {
        let height = x.dims()[2];
        let mut x = self.stem.forward(x, gemm);
        let mut enc = Vec::new();
        for (ds, c3) in &self.encoder {
            ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
            x = c3.forward(ds.forward(x, gemm), gemm);
            enc.push(x.clone());
        }
        let size = [enc[2].dims()[2], enc[2].dims()[3]];
        let fused = self.fuse.forward(
            Tensor::cat(enc.iter().map(|x| resize(x.clone(), size)).collect(), 1),
            gemm,
        );
        let high = fused.clone().narrow(1, 0, 256);
        let low = fused.clone().narrow(1, 256, 128);
        let skip = fused.narrow(1, 384, 128);
        let high = self.high_fuse.forward(
            Tensor::cat(
                self.high
                    .iter()
                    .map(|m| m.forward(high.clone(), gemm))
                    .collect(),
                1,
            ),
            gemm,
        );
        let h = self.final_fuse.forward(
            Tensor::cat(vec![high, self.low.forward(low, gemm), skip], 1),
            gemm,
        );
        let mut decoded = self.skips[3].forward(enc[3].clone(), gemm);
        for i in (0..4).rev() {
            let size = [decoded.dims()[2], decoded.dims()[3]];
            decoded = decoded
                + self.gammas[i].clone() * self.h_maps[i].forward(resize(h.clone(), size), gemm);
            if i > 0 {
                let size = [enc[i - 1].dims()[2], enc[i - 1].dims()[3]];
                decoded = self.decoder[i - 1].forward(resize(decoded, size), gemm)
                    + self.skips[i - 1].forward(enc[i - 1].clone(), gemm);
            }
        }
        decoded = self.final_decoder.forward(decoded, gemm);
        let width = decoded.dims()[3];
        decoded = resize(decoded, [height, width]);
        for up in &self.up {
            ensure!(!cancelled.load(Ordering::Relaxed), "task cancelled");
            decoded = up.forward(decoded, gemm);
        }
        Ok(self
            .final_conv
            .forward(resize(decoded, [height, 1025]), gemm))
    }
}
