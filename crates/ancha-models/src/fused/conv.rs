//! Implicit-GEMM convolution `y[o, p] = Σ_k W[o, k] · patch[k, p]` for NCHW f32 tensors with
//! `groups == 1`: patches are gathered from the input while loading each tile, never stored.
//! A cube computes `16·R` output channels × 128 positions, `k` in steps of 8 with
//! double-buffered shared memory filled through registers. 256 units as 16×16; unit `(tx, ty)`
//! owns channels `R·ty..R·ty+R` and positions `{4tx, 64 + 4tx} + 0..4`, with `R` of 2, 3 or 4
//! picked so the channel count fills whole cubes where possible.
use burn_cubecl::{
    CubeRuntime, kernel::into_contiguous, ops::numeric::empty_device_contiguous_dtype,
    tensor::CubeTensor,
};
use cubecl::prelude::*;

const POSITIONS: usize = 128;

/// Kernel size, stride and padding; compile-time, so the patch index arithmetic divides by
/// constants. Models use a handful of geometries.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Geometry {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
}

#[cube(launch_unchecked)]
fn conv2d<F: Float>(
    x: &Tensor<F>,
    weight: &Tensor<Vector<F, Const<4>>>,
    bias: &Tensor<F>,
    y: &mut Tensor<F>,
    #[comptime] geometry: Geometry,
    #[comptime] rows: usize,
    #[comptime] has_bias: bool,
) {
    let channels_in = x.shape(1);
    let height = x.shape(2);
    let width = x.shape(3);
    let channels_out = y.shape(1);
    let out_w = y.shape(3);
    let positions = y.shape(2) * out_w;
    let depth = weight.shape(1);
    let plane = height * width;
    let block_rows = 16 * rows;

    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let unit = ty * 16 + tx;
    let p0 = CUBE_POS_X as usize * 128;
    let o0 = CUBE_POS_Y as usize * block_rows;
    let batch = CUBE_POS_Z as usize;
    let x_base = batch * x.stride(0);
    let y_base = batch * y.stride(0);

    // Per buffer: Wᵀ as [8 k][16R channels] scalars, patches as [8 k][32 position vectors].
    let mut a_tile = SharedMemory::<F>::new(2 * 8 * block_rows);
    let mut b_tile = SharedMemory::<Vector<F, Const<4>>>::new(512usize);
    let zero = Vector::<F, Const<4>>::new(F::new(0.0));

    // Weight staging: units below 2·16R load one 4-wide k chunk of one channel.
    let a_row = unit / 2;
    let a_k4 = unit % 2;
    let a_loads = unit < 2 * block_rows;
    let a_valid = a_loads && o0 + a_row < channels_out;
    let a_source = (o0 + a_row) * (weight.stride(0) / 4) + a_k4;
    let a_store = a_k4 * 4 * block_rows + a_row;

    // Patch staging: unit gathers k row `unit / 32` at positions `p0 + 4(unit % 32) + 0..4`.
    let b_k = unit / 32;
    let b_col = unit % 32;
    let mut origin_h = Array::<usize>::new(4usize);
    let mut origin_w = Array::<usize>::new(4usize);
    let mut inside = Array::<bool>::new(4usize);
    #[unroll]
    for u in 0..4usize {
        let p = p0 + b_col * 4 + u;
        inside[u] = p < positions;
        // Offset by the padding so coordinates stay unsigned: input row = origin - pad.
        origin_h[u] = (p / out_w) * geometry.stride_h;
        origin_w[u] = (p % out_w) * geometry.stride_w;
    }

    let mut a_next = zero;
    if a_valid {
        a_next = weight[a_source];
    }
    let mut b_next = zero;
    #[unroll]
    for u in 0..4usize {
        b_next[u] = gather(
            x,
            x_base,
            b_k,
            origin_h[u],
            origin_w[u],
            inside[u],
            channels_in,
            plane,
            height,
            width,
            geometry,
        );
    }
    if a_loads {
        #[unroll]
        for u in 0..4usize {
            a_tile[a_store + u * block_rows] = a_next[u];
        }
    }
    b_tile[unit] = b_next;
    sync_cube();

    let mut acc = Array::<Vector<F, Const<4>>>::new(2 * rows);
    #[unroll]
    for e in 0..2 * rows {
        acc[e] = zero;
    }

    let steps = depth / 8;
    for step in 0..steps {
        let current = step % 2;
        let more = step + 1 < steps;
        if more {
            if a_valid {
                a_next = weight[a_source + (step + 1) * 2];
            }
            #[unroll]
            for u in 0..4usize {
                b_next[u] = gather(
                    x,
                    x_base,
                    (step + 1) * 8 + b_k,
                    origin_h[u],
                    origin_w[u],
                    inside[u],
                    channels_in,
                    plane,
                    height,
                    width,
                    geometry,
                );
            }
        }

        let a_offset = current * 8 * block_rows;
        let b_offset = current * 256;
        #[unroll]
        for k in 0..8usize {
            let b0 = b_tile[b_offset + k * 32 + tx];
            let b1 = b_tile[b_offset + k * 32 + 16 + tx];
            #[unroll]
            for i in 0..rows {
                let a = Vector::new(a_tile[a_offset + k * block_rows + ty * rows + i]);
                acc[2 * i] = acc[2 * i] + a * b0;
                acc[2 * i + 1] = acc[2 * i + 1] + a * b1;
            }
        }

        if more {
            let next = 1 - current;
            if a_loads {
                #[unroll]
                for u in 0..4usize {
                    a_tile[next * 8 * block_rows + a_store + u * block_rows] = a_next[u];
                }
            }
            b_tile[next * 256 + unit] = b_next;
        }
        // Next buffer written / current buffer free before the following step.
        sync_cube();
    }

    #[unroll]
    for i in 0..rows {
        let o = o0 + ty * rows + i;
        if o < channels_out {
            let mut offset = F::new(0.0);
            if has_bias {
                offset = bias[o];
            }
            #[unroll]
            for half in 0..2usize {
                #[unroll]
                for u in 0..4usize {
                    let p = p0 + half * 64 + tx * 4 + u;
                    if p < positions {
                        y[y_base + o * positions + p] = acc[2 * i + half][u] + offset;
                    }
                }
            }
        }
    }
}

/// Patch element `(k, p)`: input `[channel, origin_h + ky - pad_h, origin_w + kx - pad_w]`,
/// zero outside the input or past the last position.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gather<F: Float>(
    x: &Tensor<F>,
    x_base: usize,
    k: usize,
    origin_h: usize,
    origin_w: usize,
    inside: bool,
    channels_in: usize,
    plane: usize,
    height: usize,
    width: usize,
    #[comptime] geometry: Geometry,
) -> F {
    let window = geometry.kernel_h * geometry.kernel_w;
    let channel = k / window;
    let tap = k % window;
    let row = origin_h + tap / geometry.kernel_w;
    let col = origin_w + tap % geometry.kernel_w;
    let (pad_h, pad_w) = (geometry.pad_h, geometry.pad_w);
    let mut value = F::new(0.0);
    if inside
        && channel < channels_in
        && row >= pad_h
        && row - pad_h < height
        && col >= pad_w
        && col - pad_w < width
    {
        value = x[x_base + channel * plane + (row - pad_h) * width + col - pad_w];
    }
    value
}

/// Whether [`launch`] handles `weight [o, c, kh, kw]`: the flattened depth `c·kh·kw` must be
/// a multiple of 8.
pub fn supported(depth: usize) -> bool {
    depth > 0 && depth.is_multiple_of(8)
}

/// `conv2d(x, weight, bias)` with `groups == 1` and no dilation; `x` NCHW, `weight`
/// `[o, c·kh·kw]` (the flattened `[o, c, kh, kw]`), output `[b, o, oh, ow]`.
#[allow(clippy::too_many_arguments)]
pub fn launch<R: CubeRuntime>(
    x: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> CubeTensor<R> {
    let shape = x.meta.shape().clone();
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    let (out_channels, depth) = (weight.meta.shape()[0], weight.meta.shape()[1]);
    assert!(
        depth == channels * kernel[0] * kernel[1] && supported(depth),
        "implicit GEMM convolution expects [o, c·kh·kw] weights with c·kh·kw % 8 == 0"
    );
    let out_h = (height + 2 * padding[0] - kernel[0]) / stride[0] + 1;
    let out_w = (width + 2 * padding[1] - kernel[1]) / stride[1] + 1;
    let (x, weight) = (into_contiguous(x), into_contiguous(weight));
    let out = empty_device_contiguous_dtype(
        x.client.clone(),
        x.device.clone(),
        [batch, out_channels, out_h, out_w].into(),
        x.dtype,
    );
    // Channels per unit: whole cubes where the channel count allows (MDX widths are multiples
    // of 48), two for narrow layers.
    let rows = if out_channels.is_multiple_of(64) {
        4
    } else if out_channels.is_multiple_of(48) {
        3
    } else if out_channels <= 32 {
        2
    } else {
        4
    };
    let client = x.client.clone();
    let has_bias = bias.is_some();
    // Unused when `has_bias` is false; any valid f32 tensor binds the slot.
    let bias = bias.map(into_contiguous).unwrap_or_else(|| x.clone());
    let count = CubeCount::Static(
        (out_h * out_w).div_ceil(POSITIONS) as u32,
        out_channels.div_ceil(16 * rows) as u32,
        batch as u32,
    );
    let geometry = Geometry {
        kernel_h: kernel[0],
        kernel_w: kernel[1],
        stride_h: stride[0],
        stride_w: stride[1],
        pad_h: padding[0],
        pad_w: padding[1],
    };
    // SAFETY: every gather is bounds-checked against the input; rows / positions against the
    // output shape computed above.
    unsafe {
        conv2d::launch_unchecked::<f32, R>(
            &client,
            count,
            CubeDim::new_2d(16, 16),
            x.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            geometry,
            rows,
            has_bias,
        );
    }
    out
}
