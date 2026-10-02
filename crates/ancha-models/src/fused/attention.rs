//! Single-pass exact attention for one `(group, head)` per cube: a 64-query tile streams over
//! 64-key tiles with an online softmax, so scores never leave registers / shared memory.
//! 128 units as 8×16; unit `(tx, ty)` owns query rows `4ty..4ty+4`, score columns `tx + 8j`
//! and output channel chunks `tx`, `tx + 8`: 4×8 register tiles read with 12 vector loads per
//! 128 FMAs. Shared memory is 3×16 KiB (Pascal's 48 KiB limit, two cubes per SM, so up to
//! 255 registers per unit); the key buffer is reused for the transposed probabilities. Both are
//! XOR-swizzled on 16-byte chunks so the vector reads and writes are bank-conflict free; the
//! next key / value tile is loaded into registers while the current one is processed.
//!
//! Queries, keys and values are column ranges of `[groups, tokens, columns]` tensors (head `h`
//! at `offset + 64h`), so one packed projection output feeds the kernel without copies.
//! The gated variant scales each output row by `sigmoid(gate)` read from the same tensor.
use burn_cubecl::{
    CubeRuntime, kernel::into_contiguous, ops::numeric::empty_device_contiguous_dtype,
    tensor::CubeTensor,
};
use cubecl::prelude::*;

use super::ATTENTION_HEAD_DIM as HEAD_DIM;
const TILE: usize = 64;
/// Finite stand-in for -inf: `exp(MASKED - m)` is exactly 0 and never produces NaN.
const MASKED: f32 = -1.0e30;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn flash_attention<F: Float>(
    q: &Tensor<Vector<F, Const<4>>>,
    k: &Tensor<Vector<F, Const<4>>>,
    v: &Tensor<Vector<F, Const<4>>>,
    gates: &Tensor<F>,
    out: &mut Tensor<Vector<F, Const<4>>>,
    q_col: u32,
    k_col: u32,
    v_col: u32,
    gate_col: u32,
    #[comptime] gated: bool,
) {
    let tokens = q.shape(1);
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let unit = ty * 8 + tx;
    let head = CUBE_POS_Y as usize;
    let group = CUBE_POS_Z as usize;
    let first = CUBE_POS_X as usize * 64;
    // Tile loads: unit copies chunk `unit % 16` of rows `unit / 16 + 8it`.
    let chunk = unit % 16;
    let row_in = unit / 16;
    // Strides and column offsets are in scalars, indices in 4-wide vectors.
    let q_base = (group * q.stride(0) + q_col as usize) / 4 + head * 16 + chunk;
    let k_base = (group * k.stride(0) + k_col as usize) / 4 + head * 16 + chunk;
    let v_base = (group * v.stride(0) + v_col as usize) / 4 + head * 16 + chunk;
    let o_base = (group * out.stride(0) + head * out.stride(2)) / 4 + tx;
    let q_row = q.stride(1) / 4;
    let k_row = k.stride(1) / 4;
    let v_row = v.stride(1) / 4;
    let o_row = out.stride(1) / 4;

    let mut qs = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
    let mut ks = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
    let mut vs = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
    let zero = Vector::<F, Const<4>>::new(F::new(0.0));

    #[unroll]
    for it in 0..8usize {
        let row = it * 8 + row_in;
        let t = first + row;
        let mut value = zero;
        if t < tokens {
            value = q[q_base + t * q_row];
        }
        qs[row * 16 + chunk] = value;
    }

    // Key / value rows `8it + row_in` of the next tile, staged in registers so their global
    // latency overlaps the current tile's arithmetic.
    let mut key_next = Array::<Vector<F, Const<4>>>::new(8usize);
    let mut value_next = Array::<Vector<F, Const<4>>>::new(8usize);
    #[unroll]
    for it in 0..8usize {
        let t = it * 8 + row_in;
        key_next[it] = zero;
        value_next[it] = zero;
        if t < tokens {
            key_next[it] = k[k_base + t * k_row];
            value_next[it] = v[v_base + t * v_row];
        }
    }

    // Output rows `4ty + i`, channel chunks `tx` and `tx + 8`.
    let mut acc = Array::<Vector<F, Const<4>>>::new(8usize);
    let mut m = Array::<F>::new(4usize);
    let mut l = Array::<F>::new(4usize);
    // Scores of rows `4ty + i` and keys `tx + 8j` at `8i + j`.
    let mut s = Array::<F>::new(32usize);
    #[unroll]
    for i in 0..4usize {
        acc[2 * i] = zero;
        acc[2 * i + 1] = zero;
        m[i] = F::new(MASKED);
        l[i] = F::new(0.0);
    }

    let key_tiles = tokens.div_ceil(64);
    for tile in 0..key_tiles {
        let start = tile * 64;
        // The previous tile's probability and value reads are complete.
        sync_cube();
        #[unroll]
        for it in 0..8usize {
            let row = it * 8 + row_in;
            // Key row `r` keeps chunk `c` at `c ^ (r % 8)`; here r % 8 == row_in.
            ks[row * 16 + (chunk ^ row_in)] = key_next[it];
            vs[row * 16 + chunk] = value_next[it];
        }
        sync_cube();
        let next = start + 64;
        #[unroll]
        for it in 0..8usize {
            let t = next + it * 8 + row_in;
            key_next[it] = zero;
            value_next[it] = zero;
            if t < tokens {
                key_next[it] = k[k_base + t * k_row];
                value_next[it] = v[v_base + t * v_row];
            }
        }

        #[unroll]
        for e in 0..32usize {
            s[e] = F::new(0.0);
        }
        for d4 in 0..16usize {
            let mut kv = Array::<Vector<F, Const<4>>>::new(8usize);
            #[unroll]
            for j in 0..8usize {
                // Key `tx + 8j` has row % 8 == tx.
                kv[j] = ks[(tx + 8 * j) * 16 + (d4 ^ tx)];
            }
            #[unroll]
            for i in 0..4usize {
                let a = qs[(ty * 4 + i) * 16 + d4];
                #[unroll]
                for j in 0..8usize {
                    let b = kv[j];
                    s[i * 8 + j] =
                        s[i * 8 + j] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
                }
            }
        }

        // Online softmax; the 8 units of a row are 8 consecutive plane lanes.
        #[unroll]
        for i in 0..4usize {
            let mut row_max = F::new(MASKED);
            #[unroll]
            for j in 0..8usize {
                if start + tx + 8 * j >= tokens {
                    s[i * 8 + j] = F::new(MASKED);
                }
                row_max = max(row_max, s[i * 8 + j]);
            }
            row_max = max(row_max, plane_shuffle_xor(row_max, 1));
            row_max = max(row_max, plane_shuffle_xor(row_max, 2));
            row_max = max(row_max, plane_shuffle_xor(row_max, 4));
            let next = max(m[i], row_max);
            let scale = F::exp(m[i] - next);
            m[i] = next;
            let mut sum = F::new(0.0);
            #[unroll]
            for j in 0..8usize {
                let p = F::exp(s[i * 8 + j] - next);
                s[i * 8 + j] = p;
                sum += p;
            }
            l[i] = l[i] * scale + sum;
            acc[2 * i] = acc[2 * i] * Vector::new(scale);
            acc[2 * i + 1] = acc[2 * i + 1] * Vector::new(scale);
        }

        // All score reads of the key buffer are complete; reuse it for Pᵀ, whose row `c`
        // keeps row chunk `r` at `r ^ (c % 16)`: conflict-free writes, broadcast reads.
        sync_cube();
        #[unroll]
        for j in 0..8usize {
            let mut p = zero;
            p[0] = s[j];
            p[1] = s[8 + j];
            p[2] = s[16 + j];
            p[3] = s[24 + j];
            let c = tx + 8 * j;
            ks[c * 16 + (ty ^ (c % 16))] = p;
        }
        sync_cube();

        for c in 0..64usize {
            let p = ks[c * 16 + (ty ^ (c % 16))];
            let low = vs[c * 16 + tx];
            let high = vs[c * 16 + 8 + tx];
            #[unroll]
            for i in 0..4usize {
                let weight = Vector::new(p[i]);
                acc[2 * i] = acc[2 * i] + low * weight;
                acc[2 * i + 1] = acc[2 * i + 1] + high * weight;
            }
        }
    }

    #[unroll]
    for i in 0..4usize {
        let mut total = l[i];
        total += plane_shuffle_xor(total, 1);
        total += plane_shuffle_xor(total, 2);
        total += plane_shuffle_xor(total, 4);
        let t = first + ty * 4 + i;
        if t < tokens {
            let mut scale = F::new(1.0) / total;
            if gated {
                let gate =
                    gates[group * gates.stride(0) + t * gates.stride(1) + gate_col as usize + head];
                scale /= F::new(1.0) + F::exp(F::new(0.0) - gate);
            }
            let weight = Vector::new(scale);
            out[o_base + t * o_row] = acc[2 * i] * weight;
            out[o_base + t * o_row + 8] = acc[2 * i + 1] * weight;
        }
    }
}

/// Last axis dense and outer strides in whole 4-wide vectors, as the kernel reads them.
fn vector_ready<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    let strides = tensor.meta.strides();
    if strides[2] == 1 && strides[..2].iter().all(|s| s % 4 == 0) {
        tensor
    } else {
        into_contiguous(tensor)
    }
}

/// Column offsets (scalars) of q, k, v and, when gated, of the per-head gate logits in `q`.
pub struct Columns {
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub gate: Option<usize>,
}

/// `softmax(q·kᵀ)·v` per `(group, head)`; `q`, `k`, `v` are `[groups, tokens, columns]` f32 with
/// head `h` at `offset + 64h`, `q` pre-scaled. With a gate column, output rows are scaled by
/// `sigmoid(q[.., gate + h])`. Returns `[groups, tokens, heads, 64]`.
pub fn launch<R: CubeRuntime>(
    [q, k, v]: [CubeTensor<R>; 3],
    heads: usize,
    columns: Columns,
) -> CubeTensor<R> {
    let shape = q.meta.shape().clone();
    let (groups, tokens) = (shape[0], shape[1]);
    let fits = |t: &CubeTensor<R>, offset: usize, width: usize| {
        let s = t.meta.shape();
        s.num_dims() == 3 && s[0] == groups && s[1] == tokens && offset + width <= s[2]
    };
    let span = heads * HEAD_DIM;
    assert!(
        [columns.q, columns.k, columns.v].iter().all(|c| c % 4 == 0)
            && fits(&q, columns.q, span)
            && fits(&k, columns.k, span)
            && fits(&v, columns.v, span)
            && columns.gate.is_none_or(|gate| fits(&q, gate, heads)),
        "fused attention: q / k / v column ranges do not fit [{groups}, {tokens}, ..]"
    );
    let (q, k, v) = (vector_ready(q), vector_ready(k), vector_ready(v));
    let out = empty_device_contiguous_dtype(
        q.client.clone(),
        q.device.clone(),
        [groups, tokens, heads, HEAD_DIM].into(),
        q.dtype,
    );
    let client = q.client.clone();
    let count = CubeCount::Static(tokens.div_ceil(TILE) as u32, heads as u32, groups as u32);
    // SAFETY: tokens bound every row index; the column ranges were checked to fit above.
    unsafe {
        flash_attention::launch_unchecked::<f32, R>(
            &client,
            count,
            CubeDim::new_2d(8, 16),
            q.clone().into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            q.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            columns.q as u32,
            columns.k as u32,
            columns.v as u32,
            columns.gate.unwrap_or(0) as u32,
            columns.gate.is_some(),
        );
    }
    out
}
