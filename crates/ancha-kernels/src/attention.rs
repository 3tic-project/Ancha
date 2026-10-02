#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub groups: usize,
    pub queries: usize,
    pub keys: usize,
    pub head_dim: usize,
    pub value_dim: usize,
}

impl Shape {
    fn sizes(self) -> Result<[usize; 4], &'static str> {
        if [
            self.groups,
            self.queries,
            self.keys,
            self.head_dim,
            self.value_dim,
        ]
        .contains(&0)
        {
            return Err("all dimensions must be positive");
        }
        let product = |a: usize, b: usize, c: usize| {
            a.checked_mul(b)
                .and_then(|x| x.checked_mul(c))
                .ok_or("size overflow")
        };
        Ok([
            product(self.groups, self.queries, self.head_dim)?,
            product(self.groups, self.keys, self.head_dim)?,
            product(self.groups, self.keys, self.value_dim)?,
            product(self.groups, self.queries, self.value_dim)?,
        ])
    }
    fn validate(self, q: &[f32], k: &[f32], v: &[f32]) -> Result<usize, &'static str> {
        let [qs, ks, vs, os] = self.sizes()?;
        if (q.len(), k.len(), v.len()) != (qs, ks, vs) {
            return Err("tensor length mismatch");
        }
        if q.iter().chain(k).chain(v).any(|x| !x.is_finite()) {
            return Err("non-finite input");
        }
        Ok(os)
    }
}

fn score(q: &[f32], k: &[f32], s: Shape, g: usize, i: usize, j: usize) -> f32 {
    let qi = (g * s.queries + i) * s.head_dim;
    let kj = (g * s.keys + j) * s.head_dim;
    let mut dot = 0.0;
    for d in 0..s.head_dim {
        dot += q[qi + d] * k[kj + d];
    }
    dot / (s.head_dim as f32).sqrt()
}

/// Full scores reference, non-causal, no mask/dropout. Layout is [group, token, dim].
pub fn dense(q: &[f32], k: &[f32], v: &[f32], s: Shape) -> Result<Vec<f32>, &'static str> {
    let out_len = s.validate(q, k, v)?;
    let count = s
        .groups
        .checked_mul(s.queries)
        .and_then(|x| x.checked_mul(s.keys))
        .ok_or("size overflow")?;
    let mut scores = vec![0.0; count];
    let mut out = vec![0.0; out_len];
    for g in 0..s.groups {
        for i in 0..s.queries {
            let row = (g * s.queries + i) * s.keys;
            for j in 0..s.keys {
                scores[row + j] = score(q, k, s, g, i, j);
            }
            let max = scores[row..row + s.keys]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() {
                return Err("score overflow");
            }
            let mut denom = 0.0;
            for j in 0..s.keys {
                scores[row + j] = (scores[row + j] - max).exp();
                denom += scores[row + j];
            }
            for j in 0..s.keys {
                let p = scores[row + j] / denom;
                for d in 0..s.value_dim {
                    out[(g * s.queries + i) * s.value_dim + d] +=
                        p * v[(g * s.keys + j) * s.value_dim + d];
                }
            }
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("output overflow");
    }
    Ok(out)
}

/// Global exact attention via online softmax. Floating-point operation order differs.
/// Scores workspace O(query_tile * key_tile), not O(groups * queries * keys).
/// Q/K/V and output still occupy their complete linear-size buffers.
pub fn tiled(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: Shape,
    query_tile: usize,
    key_tile: usize,
) -> Result<Vec<f32>, &'static str> {
    let out_len = s.validate(q, k, v)?;
    if query_tile == 0 || key_tile == 0 {
        return Err("tiles must be positive");
    }
    let qt = query_tile.min(s.queries);
    let kt = key_tile.min(s.keys);
    let mut scores = vec![0.0; qt.checked_mul(kt).ok_or("tile overflow")?];
    let mut numerator = vec![0.0; qt.checked_mul(s.value_dim).ok_or("tile overflow")?];
    let mut maxima = vec![f32::NEG_INFINITY; qt];
    let mut denoms = vec![0.0; qt];
    let mut out = vec![0.0; out_len];
    for g in 0..s.groups {
        for q0 in (0..s.queries).step_by(qt) {
            let nq = qt.min(s.queries - q0);
            numerator.fill(0.0);
            maxima.fill(f32::NEG_INFINITY);
            denoms.fill(0.0);
            for k0 in (0..s.keys).step_by(kt) {
                let nk = kt.min(s.keys - k0);
                for i in 0..nq {
                    let mut tile_max = f32::NEG_INFINITY;
                    for j in 0..nk {
                        let value = score(q, k, s, g, q0 + i, k0 + j);
                        if !value.is_finite() {
                            return Err("score overflow");
                        }
                        scores[i * kt + j] = value;
                        tile_max = tile_max.max(value);
                    }
                    let new_max = maxima[i].max(tile_max);
                    let alpha = (maxima[i] - new_max).exp();
                    denoms[i] *= alpha;
                    for d in 0..s.value_dim {
                        numerator[i * s.value_dim + d] *= alpha;
                    }
                    for j in 0..nk {
                        let p = (scores[i * kt + j] - new_max).exp();
                        denoms[i] += p;
                        for d in 0..s.value_dim {
                            numerator[i * s.value_dim + d] +=
                                p * v[(g * s.keys + k0 + j) * s.value_dim + d];
                        }
                    }
                    maxima[i] = new_max;
                }
            }
            for i in 0..nq {
                for d in 0..s.value_dim {
                    out[(g * s.queries + q0 + i) * s.value_dim + d] =
                        numerator[i * s.value_dim + d] / denoms[i];
                }
            }
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("output overflow");
    }
    Ok(out)
}
