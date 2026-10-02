//! `C = A·B (+ bias)` for f32 row-major `A [n, k]`, `B [k, m]`: 128×128 output tile per cube,
//! k in steps of 8, double-buffered shared memory filled through registers so one barrier per
//! step suffices. 256 units as 16×16; unit `(tx, ty)` owns rows `{4ty, 64 + 4ty} + 0..4` and
//! columns `{4tx, 64 + 4tx} + 0..4`, an 8×8 register tile read with 16 vector loads per
//! 256 FMAs; rows sharing a vector load are broadcast, columns are contiguous.
use burn_cubecl::{
    CubeRuntime, kernel::into_contiguous, ops::numeric::empty_device_contiguous_dtype,
    tensor::CubeTensor,
};
use cubecl::prelude::*;

const BLOCK: usize = 128;
const STEP: usize = 8;

#[cube(launch_unchecked)]
fn gemm<F: Float>(
    a: &Tensor<Vector<F, Const<4>>>,
    b: &Tensor<Vector<F, Const<4>>>,
    bias: &Tensor<Vector<F, Const<4>>>,
    c: &mut Tensor<Vector<F, Const<4>>>,
    #[comptime] has_bias: bool,
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

    // Per buffer: A as [128 rows][2 vectors of k], B as [8 k][32 vectors of columns].
    let mut a_tile = SharedMemory::<Vector<F, Const<4>>>::new(512usize);
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

    let mut a_next = zero;
    if a_valid {
        a_next = a[a_source];
    }
    let mut b_next = b[b_source];
    a_tile[unit] = a_next;
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

        #[unroll]
        for k4 in 0..2usize {
            let mut av = Array::<Vector<F, Const<4>>>::new(8usize);
            #[unroll]
            for i in 0..4usize {
                av[i] = a_tile[current + (ty * 4 + i) * 2 + k4];
                av[4 + i] = a_tile[current + (64 + ty * 4 + i) * 2 + k4];
            }
            #[unroll]
            for u in 0..4usize {
                let k = k4 * 4 + u;
                let b0 = b_tile[current + k * 32 + tx];
                let b1 = b_tile[current + k * 32 + 16 + tx];
                #[unroll]
                for i in 0..8usize {
                    let x = Vector::new(av[i][u]);
                    acc[i * 2] = acc[i * 2] + x * b0;
                    acc[i * 2 + 1] = acc[i * 2 + 1] + x * b1;
                }
            }
        }

        if more {
            let next = 256 - current;
            a_tile[next + unit] = a_next;
            b_tile[next + unit] = b_next;
        }
        // Next buffer written / current buffer free before the following step.
        sync_cube();
    }

    let mut bias0 = zero;
    let mut bias1 = zero;
    if has_bias {
        bias0 = bias[col4 + tx];
        bias1 = bias[col4 + 16 + tx];
    }
    #[unroll]
    for i in 0..8usize {
        let row = row0 + (i / 4) * 64 + ty * 4 + i % 4;
        if row < rows {
            c[row * c_row + col4 + tx] = acc[i * 2] + bias0;
            c[row * c_row + col4 + 16 + tx] = acc[i * 2 + 1] + bias1;
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

/// `a [n, k] · b [k, m] + bias [m]` in f32; see [`supported`].
pub fn launch<R: CubeRuntime>(
    a: CubeTensor<R>,
    b: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    let (n, k) = (a.meta.shape()[0], a.meta.shape()[1]);
    let m = b.meta.shape()[1];
    assert!(
        b.meta.shape()[0] == k && supported(k, m),
        "custom GEMM expects [n, k]·[k, m] with k % {STEP} == 0 and m % {BLOCK} == 0"
    );
    let (a, b) = (vector_rows(a), vector_rows(b));
    let out =
        empty_device_contiguous_dtype(a.client.clone(), a.device.clone(), [n, m].into(), a.dtype);
    let client = a.client.clone();
    let has_bias = bias.is_some();
    // Unused when `has_bias` is false; any valid f32 tensor binds the slot.
    let bias = bias.map(into_contiguous).unwrap_or_else(|| b.clone());
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
            out.clone().into_tensor_arg(),
            has_bias,
        );
    }
    out
}
