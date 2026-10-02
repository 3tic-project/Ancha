//! `C = A·B` for f32 row-major `A [n, k]`, `B [k, m]`: 128×128 output tile per cube,
//! k in steps of 8, double-buffered shared memory filled through registers so one barrier per
//! step suffices. 256 units as 16×16; unit `(tx, ty)` owns rows `{4ty, 64 + 4ty} + 0..4` and
//! columns `{4tx, 64 + 4tx} + 0..4`, an 8×8 register tile read with 16 vector loads per
//! 256 FMAs; rows sharing a vector load are broadcast, columns are contiguous.
//!
//! The epilogue optionally adds a bias, applies erf-GELU, rotates the first `2·rotary`
//! columns (queries, then keys) with interleaved RoPE for token `row % tokens`, and adds a
//! residual, so these never make a separate pass over the output.
use burn_cubecl::{
    CubeRuntime, kernel::into_contiguous, ops::numeric::empty_device_contiguous_dtype,
    tensor::CubeTensor,
};
use cubecl::prelude::*;

const BLOCK: usize = 128;
const STEP: usize = 8;

/// Compile-time epilogue selection.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct Epilogue {
    pub bias: bool,
    pub gelu: bool,
    pub rotary: bool,
    pub residual: bool,
}

/// `x·cos + swap_pairs(x)·sin` on one 4-wide chunk; the pair sign is folded into `sin`.
#[cube]
fn rotate<F: Float>(
    x: Vector<F, Const<4>>,
    cos: Vector<F, Const<4>>,
    sin: Vector<F, Const<4>>,
) -> Vector<F, Const<4>> {
    let mut swapped = x;
    swapped[0] = x[1];
    swapped[1] = x[0];
    swapped[2] = x[3];
    swapped[3] = x[2];
    x * cos + swapped * sin
}

/// Bias, GELU, rotary embedding and residual for the 4 outputs at `(row, col4)`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn finish<F: Float>(
    value: Vector<F, Const<4>>,
    row: usize,
    col4: usize,
    bias: &Tensor<Vector<F, Const<4>>>,
    residual: &Tensor<Vector<F, Const<4>>>,
    rope: &Tensor<Vector<F, Const<4>>>,
    tokens: usize,
    rotary4: usize,
    #[comptime] epilogue: Epilogue,
) -> Vector<F, Const<4>> {
    let mut v = value;
    if epilogue.bias {
        v += bias[col4];
    }
    if epilogue.gelu {
        let half = Vector::new(F::new(0.5));
        let one = Vector::new(F::new(1.0));
        v = v
            * half
            * (one + Vector::erf(v * Vector::new(F::new(core::f32::consts::FRAC_1_SQRT_2))));
    }
    if epilogue.rotary && col4 < 2 * rotary4 {
        // Tables `[4, tokens, 64]`: query cos / sin, then key cos / sin.
        let key = col4 >= rotary4;
        let mut table = 0usize;
        if key {
            table = 2;
        }
        let r = (table * tokens + row % tokens) * 16 + col4 % 16;
        v = rotate(v, rope[r], rope[r + tokens * 16]);
    }
    if epilogue.residual {
        v += residual[row * (residual.stride(0) / 4) + col4];
    }
    v
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gemm<F: Float>(
    a: &Tensor<Vector<F, Const<4>>>,
    b: &Tensor<Vector<F, Const<4>>>,
    bias: &Tensor<Vector<F, Const<4>>>,
    residual: &Tensor<Vector<F, Const<4>>>,
    rope: &Tensor<Vector<F, Const<4>>>,
    c: &mut Tensor<Vector<F, Const<4>>>,
    tokens: u32,
    rotary: u32,
    #[comptime] epilogue: Epilogue,
) {
    let rows = a.shape(0);
    let depth = a.shape(1);
    let a_row = a.stride(0) / 4;
    let b_row = b.stride(0) / 4;
    let c_row = c.stride(0) / 4;
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let unit = ty * 16 + tx;
    let row0 = CUBE_POS_Y as usize * 128;
    let col4 = CUBE_POS_X as usize * 32;

    // Per buffer: Aᵀ as [8 k][128 rows] scalars (read as 4-row vectors), B as [8 k][32 column
    // vectors]. Aᵀ keeps only two A vectors live per k, holding the kernel within 128
    // registers (two cubes per SM).
    let mut a_tile = SharedMemory::<F>::new(2048usize);
    let mut b_tile = SharedMemory::<Vector<F, Const<4>>>::new(512usize);
    let zero = Vector::<F, Const<4>>::new(F::new(0.0));

    // Global → register staging: one A and one B vector per unit and step.
    let a_load_row = unit / 2;
    let a_load_k = unit % 2;
    let b_load_k = unit / 32;
    let b_load_col = unit % 32;
    let a_valid = row0 + a_load_row < rows;
    let a_source = (row0 + a_load_row) * a_row + a_load_k;
    let b_source = b_load_k * b_row + col4 + b_load_col;
    let a_store = a_load_k * 512 + a_load_row;

    let mut a_next = zero;
    if a_valid {
        a_next = a[a_source];
    }
    let mut b_next = b[b_source];
    #[unroll]
    for u in 0..4usize {
        a_tile[a_store + u * 128] = a_next[u];
    }
    b_tile[unit] = b_next;
    sync_cube();

    let mut acc = Array::<Vector<F, Const<4>>>::new(16usize);
    #[unroll]
    for e in 0..16usize {
        acc[e] = zero;
    }

    let steps = depth / 8;
    for step in 0..steps {
        let current = (step % 2) * 256;
        let more = step + 1 < steps;
        if more {
            let k4 = (step + 1) * 2;
            if a_valid {
                a_next = a[a_source + k4];
            }
            b_next = b[b_source + (step + 1) * 8 * b_row];
        }

        let a_view = a_tile
            .slice(current * 4, current * 4 + 1024)
            .into_vectorized()
            .with_vector_size::<Const<4>>();
        #[unroll]
        for k in 0..8usize {
            let a0 = a_view[k * 32 + ty];
            let a1 = a_view[k * 32 + 16 + ty];
            let b0 = b_tile[current + k * 32 + tx];
            let b1 = b_tile[current + k * 32 + 16 + tx];
            #[unroll]
            for i in 0..4usize {
                let x = Vector::new(a0[i]);
                acc[i * 2] = acc[i * 2] + x * b0;
                acc[i * 2 + 1] = acc[i * 2 + 1] + x * b1;
                let y = Vector::new(a1[i]);
                acc[8 + i * 2] = acc[8 + i * 2] + y * b0;
                acc[8 + i * 2 + 1] = acc[8 + i * 2 + 1] + y * b1;
            }
        }

        if more {
            let next = 256 - current;
            #[unroll]
            for u in 0..4usize {
                a_tile[next * 4 + a_store + u * 128] = a_next[u];
            }
            b_tile[next + unit] = b_next;
        }
        // Next buffer written / current buffer free before the following step.
        sync_cube();
    }

    let tokens = tokens as usize;
    let rotary4 = rotary as usize / 4;
    #[unroll]
    for i in 0..8usize {
        let row = row0 + (i / 4) * 64 + ty * 4 + i % 4;
        if row < rows {
            #[unroll]
            for half in 0..2usize {
                let col = col4 + half * 16 + tx;
                c[row * c_row + col] = finish(
                    acc[i * 2 + half],
                    row,
                    col,
                    bias,
                    residual,
                    rope,
                    tokens,
                    rotary4,
                    epilogue,
                );
            }
        }
    }
}

/// Shapes the kernel handles: `k` a multiple of 8, `m` of 128.
pub fn supported(k: usize, m: usize) -> bool {
    k > 0 && k.is_multiple_of(STEP) && m > 0 && m.is_multiple_of(BLOCK)
}

fn vector_rows<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    let strides = tensor.meta.strides();
    if strides[strides.len() - 1] == 1 && strides[0].is_multiple_of(4) {
        tensor
    } else {
        into_contiguous(tensor)
    }
}

/// Optional epilogue operands of [`launch`].
#[derive(Default)]
pub struct Operands<R: CubeRuntime> {
    /// `[1, m]` (any shape with `m` contiguous values).
    pub bias: Option<CubeTensor<R>>,
    pub gelu: bool,
    /// Rotary tables `[4, tokens, 64]` (query cos / sin, key cos / sin), the token count and
    /// the query width; columns `[0, w)` are rotated with the query tables, `[w, 2w)` with the
    /// key tables.
    pub rotary: Option<(CubeTensor<R>, usize, usize)>,
    /// `[n, m]` added last.
    pub residual: Option<CubeTensor<R>>,
}

/// `a [n, k] · b [k, m]` in f32 with the [`Operands`] epilogue; see [`supported`].
pub fn launch<R: CubeRuntime>(
    a: CubeTensor<R>,
    b: CubeTensor<R>,
    operands: Operands<R>,
) -> CubeTensor<R> {
    let (n, k) = (a.meta.shape()[0], a.meta.shape()[1]);
    let m = b.meta.shape()[1];
    assert!(
        b.meta.shape()[0] == k && supported(k, m),
        "custom GEMM expects [n, k]·[k, m] with k % {STEP} == 0 and m % {BLOCK} == 0"
    );
    if let Some((table, tokens, width)) = &operands.rotary {
        assert!(
            table.meta.num_elements() == 4 * tokens * 64
                && n.is_multiple_of(*tokens)
                && width.is_multiple_of(64)
                && 2 * width <= m,
            "rotary epilogue expects 4·tokens·64 table values, whole token rows and 2·width <= m"
        );
    }
    if let Some(residual) = &operands.residual {
        assert_eq!(**residual.meta.shape(), [n, m], "residual shape");
    }
    let (a, b) = (vector_rows(a), vector_rows(b));
    let out =
        empty_device_contiguous_dtype(a.client.clone(), a.device.clone(), [n, m].into(), a.dtype);
    let client = a.client.clone();
    let epilogue = Epilogue {
        bias: operands.bias.is_some(),
        gelu: operands.gelu,
        rotary: operands.rotary.is_some(),
        residual: operands.residual.is_some(),
    };
    let (tokens, width) = operands
        .rotary
        .as_ref()
        .map_or((1, 0), |(_, tokens, width)| (*tokens, *width));
    // Unused slots bind any valid f32 tensor.
    let bias = operands
        .bias
        .map(into_contiguous)
        .unwrap_or_else(|| b.clone());
    let residual = operands
        .residual
        .map(vector_rows)
        .unwrap_or_else(|| b.clone());
    let rope = operands
        .rotary
        .map(|(table, ..)| into_contiguous(table))
        .unwrap_or_else(|| b.clone());
    let count = CubeCount::Static((m / BLOCK) as u32, n.div_ceil(BLOCK) as u32, 1);
    // SAFETY: rows are bounded by `n`, columns and depth by the shape checks above.
    unsafe {
        gemm::launch_unchecked::<f32, R>(
            &client,
            count,
            CubeDim::new_2d(16, 16),
            a.into_tensor_arg(),
            b.into_tensor_arg(),
            bias.into_tensor_arg(),
            residual.into_tensor_arg(),
            rope.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            tokens as u32,
            width as u32,
            epilogue,
        );
    }
    out
}
