//! Custom GEMM vs Burn matmul (+ bias) on CUDA at the RoFormer projection shapes.
//! `cargo run --release -p ancha-models --features cuda --example fused_linear -- 154980`
use ancha_models::fused;
use burn::{
    backend::{Cuda, cuda::CudaDevice},
    tensor::{Distribution, Tensor, backend::Backend},
};
use std::time::Instant;

fn main() {
    let rows: usize = std::env::args().nth(1).map_or(154_980, |a| {
        a.parse().expect("usage: fused_linear ROWS [ITERATIONS]")
    });
    let iterations: usize = std::env::args()
        .nth(2)
        .map_or(5, |a| a.parse().expect("iterations"));
    let device = CudaDevice::new(0);
    let time = |f: &dyn Fn() -> Tensor<Cuda, 2>| {
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
    for (k, m) in [(256, 512), (256, 1024), (512, 256), (1024, 256), (256, 256)] {
        let x = Tensor::<Cuda, 2>::random([rows, k], Distribution::Normal(0.0, 1.0), &device);
        let w = Tensor::<Cuda, 2>::random([k, m], Distribution::Normal(0.0, 0.05), &device);
        let b = Tensor::<Cuda, 1>::random([m], Distribution::Normal(0.0, 1.0), &device);
        let burn = || x.clone().matmul(w.clone()) + b.clone().unsqueeze::<2>();
        let custom = || {
            let epilogue = fused::Epilogue {
                bias: Some(&b),
                ..fused::Epilogue::default()
            };
            fused::linear(&x, &w, epilogue).expect("custom GEMM")
        };
        let difference = (burn() - custom()).abs().max().into_scalar();
        let flops = 2.0 * (rows * k * m) as f64;
        let (reference, single) = (time(&burn), time(&custom));
        println!(
            "[{rows}, {k}]·[{k}, {m}]: max |diff| {difference:.3e}; burn {reference:.4} s ({:.2} TFLOPS), custom {single:.4} s ({:.2} TFLOPS), speed-up {:.2}x",
            flops / reference / 1e12,
            flops / single / 1e12,
            reference / single
        );
    }
}
