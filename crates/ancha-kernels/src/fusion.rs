/// Fused CPU traversal of residual add + the source's F.normalize-based RMS operation.
/// Both residual and normalized outputs are retained, as required by the residual path.
pub fn residual_l2norm(
    x: &[f32],
    residual: &[f32],
    gamma: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
    if rows == 0 || dim == 0 || eps <= 0.0 || !eps.is_finite() {
        return Err("invalid dimensions/epsilon");
    }
    let n = rows.checked_mul(dim).ok_or("size overflow")?;
    if x.len() != n || residual.len() != n || gamma.len() != dim {
        return Err("length mismatch");
    }
    if x.iter()
        .chain(residual)
        .chain(gamma)
        .any(|x| !x.is_finite())
    {
        return Err("non-finite input");
    }
    let mut sum = vec![0.0; n];
    let mut norm = vec![0.0; n];
    for r in 0..rows {
        let mut sq = 0.0;
        for d in 0..dim {
            let y = x[r * dim + d] + residual[r * dim + d];
            sum[r * dim + d] = y;
            sq += y * y;
        }
        if !sq.is_finite() {
            return Err("norm overflow");
        }
        let scale = (dim as f32).sqrt() / sq.sqrt().max(eps);
        for d in 0..dim {
            norm[r * dim + d] = sum[r * dim + d] * scale * gamma[d];
        }
    }
    if norm.iter().any(|x| !x.is_finite()) {
        return Err("output overflow");
    }
    Ok((sum, norm))
}

/// Prepack QKV and gate weights in [out,in] order; QKV bias is zero, gate bias is kept.
/// This concatenates FOUR projections because upstream QKV is already packed.
pub fn pack_qkv_gate(
    qkv: &[f32],
    gate: &[f32],
    gate_bias: &[f32],
    dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
    if dim == 0
        || !qkv.len().is_multiple_of(dim)
        || gate.len() != gate_bias.len().checked_mul(dim).ok_or("size overflow")?
    {
        return Err("invalid projection shape");
    }
    let mut w = qkv.to_vec();
    w.extend_from_slice(gate);
    let mut b = vec![0.0; qkv.len() / dim];
    b.extend_from_slice(gate_bias);
    Ok((w, b))
}

/// Fold static sqrt(dim)*gamma into columns of a following linear weight.
/// The input-dependent L2 denominator MUST still be computed at runtime.
pub fn fold_norm_scale(weights: &[f32], gamma: &[f32]) -> Result<Vec<f32>, &'static str> {
    let dim = gamma.len();
    if dim == 0 || !weights.len().is_multiple_of(dim) {
        return Err("invalid projection shape");
    }
    Ok(weights
        .iter()
        .enumerate()
        .map(|(i, w)| *w * gamma[i % dim] * (dim as f32).sqrt())
        .collect())
}

pub fn complex_mask_interleaved(x: &[f32], mask: &[f32]) -> Result<Vec<f32>, &'static str> {
    if x.len() != mask.len() || !x.len().is_multiple_of(2) {
        return Err("complex layout mismatch");
    }
    let mut y = vec![0.0; x.len()];
    for i in (0..x.len()).step_by(2) {
        y[i] = x[i] * mask[i] - x[i + 1] * mask[i + 1];
        y[i + 1] = x[i] * mask[i + 1] + x[i + 1] * mask[i];
    }
    Ok(y)
}
