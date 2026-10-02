use ancha_kernels::attention::{self, Shape};
use ancha_models::roformer::{AttentionPlan, exact_attention};
use anyhow::{Result, ensure};
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData, backend::Backend},
};
use std::time::Instant;

/// Synthetic operator ablation: inherited scalar reference vs BLAS tiled global attention.
pub fn attention_benchmark(
    tokens: usize,
    iterations: usize,
    plan: AttentionPlan,
) -> Result<serde_json::Value> {
    ensure!(
        (1..=2048).contains(&tokens) && iterations > 0 && iterations <= 100,
        "invalid benchmark size or iterations"
    );
    plan.validate()?;
    type B = NdArray<f32>;
    let device = Default::default();
    let groups = 4;
    let heads = 2;
    let dim = 64;
    let values = |shift: f32| {
        (0..groups * heads * tokens * dim)
            .map(|i| ((i as f32 * 0.137) + shift).sin() * 0.4)
            .collect::<Vec<_>>()
    };
    let q = values(1.0);
    let k = values(2.0);
    let v = values(3.0);
    let shape = Shape {
        groups: groups * heads,
        queries: tokens,
        keys: tokens,
        head_dim: dim,
        value_dim: dim,
    };
    let tq = Tensor::<B, 4>::from_data(
        TensorData::new(q.clone(), [groups, heads, tokens, dim]),
        &device,
    );
    let tk = Tensor::<B, 4>::from_data(
        TensorData::new(k.clone(), [groups, heads, tokens, dim]),
        &device,
    );
    let tv = Tensor::<B, 4>::from_data(
        TensorData::new(v.clone(), [groups, heads, tokens, dim]),
        &device,
    );
    let optimized = || {
        exact_attention(tq.clone(), tk.clone(), tv.clone(), plan)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("benchmark output: {e:?}"))
    };
    let reference = attention::dense(&q, &k, &v, shape).map_err(anyhow::Error::msg)?;
    let result = optimized()?; // warmup excluded
    let max_abs = reference
        .iter()
        .zip(&result)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ensure!(
        max_abs < 2e-5,
        "attention benchmark numerical mismatch {max_abs}"
    );
    let mut scalar = Vec::new();
    let mut tiled = Vec::new();
    for _ in 0..iterations {
        let t = Instant::now();
        std::hint::black_box(attention::dense(&q, &k, &v, shape).map_err(anyhow::Error::msg)?);
        scalar.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        std::hint::black_box(optimized()?);
        tiled.push(t.elapsed().as_secs_f64());
    }
    scalar.sort_by(f64::total_cmp);
    tiled.sort_by(f64::total_cmp);
    let baseline = scalar[iterations / 2];
    let optimized = tiled[iterations / 2];
    Ok(
        serde_json::json!({"scope":"synthetic attention operator; not full-model speedup", "backend":B::name(&device),
        "tokens":tokens,"groups":groups,"heads":heads,"dim":dim,"iterations":iterations,
        "query_tile":plan.query_tile,"group_tile":plan.group_tile,"max_abs":max_abs,
        "scalar_median_seconds":baseline,"tiled_median_seconds":optimized,"speedup":baseline/optimized,
        "scalar_scores_bytes":groups*heads*tokens*tokens*4,"tiled_scores_bytes":plan.score_bytes(groups,tokens,heads),
        "scalar_samples_seconds":scalar,"tiled_samples_seconds":tiled}),
    )
}
