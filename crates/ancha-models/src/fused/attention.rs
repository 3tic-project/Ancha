//! Single-pass exact attention for one `(group, head)` per cube: a 64-query tile streams over
//! 64-key tiles with an online softmax, so scores never leave registers / shared memory.
//! 256 units as 16×16; unit `(tx, ty)` owns query rows `4ty..4ty+4`, score columns
//! `tx + 16j` and output channels `4tx..4tx+4`. Shared memory is 3×16 KiB (Pascal's 48 KiB
//! limit); the key buffer is reused for the transposed probabilities, and both are
//! XOR-swizzled on 16-byte chunks so the vector reads and writes are bank-conflict free.
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
fn flash_attention<F: Float>(
    q: &Tensor<Vector<F, Const<4>>>,
    k: &Tensor<Vector<F, Const<4>>>,
    v: &Tensor<Vector<F, Const<4>>>,
    out: &mut Tensor<Vector<F, Const<4>>>,
) {
    let tokens = q.shape(1);
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let head = CUBE_POS_Y as usize;
    let group = CUBE_POS_Z as usize;
    let first = CUBE_POS_X as usize * 64;
    // Strides are in scalars, indices in 4-wide vectors.
    let q_base = (group * q.stride(0) + head * q.stride(2)) / 4;
    let k_base = (group * k.stride(0) + head * k.stride(2)) / 4;
    let v_base = (group * v.stride(0) + head * v.stride(2)) / 4;
    let o_base = (group * out.stride(0) + head * out.stride(2)) / 4;
    let q_row = q.stride(1) / 4;
    let k_row = k.stride(1) / 4;
    let v_row = v.stride(1) / 4;
    let o_row = out.stride(1) / 4;

    let mut qs = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
    let mut ks = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
    let mut vs = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
    let zero = Vector::<F, Const<4>>::new(F::new(0.0));

    #[unroll]
    for it in 0..4usize {
        let row = it * 16 + ty;
        let t = first + row;
        let mut value = zero;
        if t < tokens {
            value = q[q_base + t * q_row + tx];
        }
        qs[row * 16 + tx] = value;
    }

    let mut acc = Array::<Vector<F, Const<4>>>::new(4usize);
    let mut m = Array::<F>::new(4usize);
    let mut l = Array::<F>::new(4usize);
    let mut s = Array::<F>::new(16usize);
    #[unroll]
    for i in 0..4usize {
        acc[i] = zero;
        m[i] = F::new(MASKED);
        l[i] = F::new(0.0);
    }

    let key_tiles = tokens.div_ceil(64);
    for tile in 0..key_tiles {
        let start = tile * 64;
        // The previous tile's probability and value reads are complete.
        sync_cube();
        #[unroll]
        for it in 0..4usize {
            let row = it * 16 + ty;
            let t = start + row;
            let mut key = zero;
            let mut value = zero;
            if t < tokens {
                key = k[k_base + t * k_row + tx];
                value = v[v_base + t * v_row + tx];
            }
            // Key row `row` keeps chunk `c` at `c ^ (row % 16)`; here row % 16 == ty.
            ks[row * 16 + (tx ^ ty)] = key;
            vs[row * 16 + tx] = value;
        }
        sync_cube();

        #[unroll]
        for e in 0..16usize {
            s[e] = F::new(0.0);
        }
        for d4 in 0..16usize {
            let mut qv = Array::<Vector<F, Const<4>>>::new(4usize);
            let mut kv = Array::<Vector<F, Const<4>>>::new(4usize);
            #[unroll]
            for i in 0..4usize {
                qv[i] = qs[(ty * 4 + i) * 16 + d4];
            }
            #[unroll]
            for j in 0..4usize {
                kv[j] = ks[(tx + 16 * j) * 16 + (d4 ^ tx)];
            }
            #[unroll]
            for i in 0..4usize {
                #[unroll]
                for j in 0..4usize {
                    let a = qv[i];
                    let b = kv[j];
                    s[i * 4 + j] =
                        s[i * 4 + j] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
                }
            }
        }

        // Online softmax; the 16 units of a row are 16 consecutive plane lanes.
        #[unroll]
        for i in 0..4usize {
            let mut row_max = F::new(MASKED);
            #[unroll]
            for j in 0..4usize {
                if start + tx + 16 * j >= tokens {
                    s[i * 4 + j] = F::new(MASKED);
                }
                row_max = max(row_max, s[i * 4 + j]);
            }
            row_max = max(row_max, plane_shuffle_xor(row_max, 1));
            row_max = max(row_max, plane_shuffle_xor(row_max, 2));
            row_max = max(row_max, plane_shuffle_xor(row_max, 4));
            row_max = max(row_max, plane_shuffle_xor(row_max, 8));
            let next = max(m[i], row_max);
            let scale = F::exp(m[i] - next);
            m[i] = next;
            let mut sum = F::new(0.0);
            #[unroll]
            for j in 0..4usize {
                let p = F::exp(s[i * 4 + j] - next);
                s[i * 4 + j] = p;
                sum += p;
            }
            l[i] = l[i] * scale + sum;
            acc[i] = acc[i] * Vector::new(scale);
        }

        // All score reads of the key buffer are complete; reuse it for Pᵀ.
        sync_cube();
        #[unroll]
        for j in 0..4usize {
            let mut p = zero;
            p[0] = s[j];
            p[1] = s[4 + j];
            p[2] = s[8 + j];
            p[3] = s[12 + j];
            // Pᵀ row `c` keeps row chunk `r` at `r ^ (c % 16)`; here c % 16 == tx.
            ks[(tx + 16 * j) * 16 + (ty ^ tx)] = p;
        }
        sync_cube();

        for c in 0..64usize {
            let p = ks[c * 16 + (ty ^ (c % 16))];
            let value = vs[c * 16 + tx];
            #[unroll]
            for i in 0..4usize {
                acc[i] = acc[i] + value * Vector::new(p[i]);
            }
        }
    }

    #[unroll]
    for i in 0..4usize {
        let mut total = l[i];
        total += plane_shuffle_xor(total, 1);
        total += plane_shuffle_xor(total, 2);
        total += plane_shuffle_xor(total, 4);
        total += plane_shuffle_xor(total, 8);
        let t = first + ty * 4 + i;
        if t < tokens {
            out[o_base + t * o_row + tx] = acc[i] * Vector::new(F::new(1.0) / total);
        }
    }
}

/// Last axis dense and outer strides in whole 4-wide vectors, as the kernel reads them.
fn vector_ready<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    let strides = tensor.meta.strides();
    if strides[3] == 1 && strides[..3].iter().all(|s| s % 4 == 0) {
        tensor
    } else {
        into_contiguous(tensor)
    }
}

/// `softmax(q·kᵀ)·v` over `[groups, tokens, heads, 64]` f32 tensors, `q` pre-scaled;
/// the output has the same layout.
pub fn launch<R: CubeRuntime>(
    q: CubeTensor<R>,
    k: CubeTensor<R>,
    v: CubeTensor<R>,
) -> CubeTensor<R> {
    let shape = q.meta.shape().clone();
    let (groups, tokens, heads) = (shape[0], shape[1], shape[2]);
    assert!(
        shape[3] == HEAD_DIM && *k.meta.shape() == shape && *v.meta.shape() == shape,
        "fused attention expects equal [groups, tokens, heads, {HEAD_DIM}] inputs"
    );
    let (q, k, v) = (vector_ready(q), vector_ready(k), vector_ready(v));
    let out = empty_device_contiguous_dtype(q.client.clone(), q.device.clone(), shape, q.dtype);
    let client = q.client.clone();
    let count = CubeCount::Static(tokens.div_ceil(TILE) as u32, heads as u32, groups as u32);
    // SAFETY: every index is bounded by `tokens` and the [.., heads, 64] shape checked above.
    unsafe {
        flash_attention::launch_unchecked::<f32, R>(
            &client,
            count,
            CubeDim::new_2d(16, 16),
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            out.clone().into_tensor_arg(),
        );
    }
    out
}
