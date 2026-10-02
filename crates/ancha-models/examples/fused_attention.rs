//! Fused attention vs the tiled materialized path on CUDA: max difference and timing.
//! `cargo run --release -p ancha-models --features cuda --example fused_attention -- 90 1722`
use ancha_models::{
    fused,
    roformer::{AttentionPlan, dense, exact_attention},
};
use burn::{
    backend::{Cuda, cuda::CudaDevice},
    tensor::{Distribution, Tensor, backend::Backend},
};
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| {
            a.parse()
                .expect("usage: fused_attention GROUPS TOKENS [ITERATIONS]")
        })
        .collect();
    let (groups, tokens) = (
        args.first().copied().unwrap_or(90),
        args.get(1).copied().unwrap_or(1722),
    );
    let iterations = args.get(2).copied().unwrap_or(5);
    let (heads, dim) = (8, 64);
    let device = CudaDevice::new(0);
    let input = || {
        Tensor::<Cuda, 4>::random(
            [groups, tokens, heads, dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        )
    };
    let (q, k, v) = (input(), input(), input());
    let plan = AttentionPlan {
        query_tile: None,
        group_tile: None,
        ..AttentionPlan::default()
    };
    let scale = (dim as f32).sqrt().recip();
    // The materialized path as the model runs it: dense [groups, heads, tokens, dim] inputs.
    let heads_major = |t: &Tensor<Cuda, 4>| dense(t.clone().swap_dims(1, 2));
    let (qh, kh, vh) = (heads_major(&q), heads_major(&k), heads_major(&v));
    let reference = || exact_attention(qh.clone(), kh.clone(), vh.clone(), plan);
    let qs = q.clone().mul_scalar(scale);
    let fused = || fused::attention(&qs, &k, &v).expect("fused kernel");
    let difference = (reference().swap_dims(1, 2) - fused())
        .abs()
        .max()
        .into_scalar();
    let time = |f: &dyn Fn() -> Tensor<Cuda, 4>| {
        let mut samples: Vec<f64> = (0..iterations)
            .map(|_| {
                <Cuda as Backend>::sync(&device).unwrap();
                let start = Instant::now();
                let out = f();
                <Cuda as Backend>::sync(&device).unwrap();
                let elapsed = start.elapsed().as_secs_f64();
                drop(out);
                elapsed
            })
            .collect();
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    let flops = 4.0 * (groups * heads) as f64 * (tokens * tokens * dim) as f64;
    let materialized = time(&reference);
    let single = time(&fused);
    println!(
        "groups {groups} tokens {tokens}: max |diff| {difference:.3e}; materialized {materialized:.4} s, fused {single:.4} s ({:.2} TFLOPS), speed-up {:.2}x",
        flops / single / 1e12,
        materialized / single
    );
}
