//! Instance norm over NCHW spatial axes: one cube per `(batch, channel)`, 256 threads,
//! two-pass mean and biased variance. Matches `mean((x - mean)²)` with `eps` inside the
//! square root, which is what MDX23C (1e-5) and HyperACE (1e-8) require.
use burn_cubecl::{
    CubeRuntime, kernel::into_contiguous, ops::numeric::empty_device_contiguous_dtype,
    tensor::CubeTensor,
};
use cubecl::prelude::*;

const THREADS: usize = 256;

#[cube]
fn block_sum<F: Float>(value: F, tid: usize, shared: &mut SharedMemory<F>) -> F {
    shared[tid] = value;
    sync_cube();
    if tid < 128 {
        shared[tid] = shared[tid] + shared[tid + 128];
    }
    sync_cube();
    if tid < 64 {
        shared[tid] = shared[tid] + shared[tid + 64];
    }
    sync_cube();
    if tid < 32 {
        shared[tid] = shared[tid] + shared[tid + 32];
    }
    sync_cube();
    if tid < 16 {
        shared[tid] = shared[tid] + shared[tid + 16];
    }
    sync_cube();
    if tid < 8 {
        shared[tid] = shared[tid] + shared[tid + 8];
    }
    sync_cube();
    if tid < 4 {
        shared[tid] = shared[tid] + shared[tid + 4];
    }
    sync_cube();
    if tid < 2 {
        shared[tid] = shared[tid] + shared[tid + 2];
    }
    sync_cube();
    if tid < 1 {
        shared[tid] = shared[tid] + shared[tid + 1];
    }
    sync_cube();
    shared[0]
}

#[cube(launch_unchecked)]
fn instance_norm<F: Float>(
    x: &Tensor<F>,
    gamma: &Tensor<F>,
    beta: &Tensor<F>,
    y: &mut Tensor<F>,
    eps: f32,
) {
    let height = x.shape(2);
    let width = x.shape(3);
    let spatial = height * width;
    let channels = x.shape(1);
    let nc = CUBE_POS_X as usize;
    let batch = nc / channels;
    let channel = nc % channels;
    let base = (batch * channels + channel) * spatial;
    let tid = UNIT_POS_X as usize + UNIT_POS_Y as usize * 16;
    let mut shared = SharedMemory::<F>::new(THREADS);

    let mut sum = F::new(0.0);
    let mut index = tid;
    while index < spatial {
        sum += x[base + index];
        index += THREADS;
    }
    let count = F::cast_from(spatial as u32);
    let mean = block_sum(sum, tid, &mut shared) / count;

    let mut sumsq = F::new(0.0);
    index = tid;
    while index < spatial {
        let delta = x[base + index] - mean;
        sumsq += delta * delta;
        index += THREADS;
    }
    let inv = (block_sum(sumsq, tid, &mut shared) / count + F::cast_from(eps)).sqrt();
    let scale = gamma[channel] / inv;
    let shift = beta[channel] - mean * scale;

    index = tid;
    while index < spatial {
        y[base + index] = x[base + index] * scale + shift;
        index += THREADS;
    }
}

/// `instance_norm(x)` for contiguous NCHW `x`, affine `[1, c, 1, 1]`.
pub fn launch<R: CubeRuntime>(
    x: CubeTensor<R>,
    gamma: CubeTensor<R>,
    beta: CubeTensor<R>,
    eps: f32,
) -> CubeTensor<R> {
    let shape = x.meta.shape().clone();
    let x = into_contiguous(x);
    let gamma = into_contiguous(gamma);
    let beta = into_contiguous(beta);
    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), shape, x.dtype);
    let client = x.client.clone();
    let cubes = (x.meta.shape()[0] * x.meta.shape()[1]) as u32;
    // SAFETY: one cube per (n, c); threads only touch that plane, and the reduction
    // synchronizes before reading another thread's partial sum.
    unsafe {
        instance_norm::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_2d(16, 16),
            x.into_tensor_arg(),
            gamma.into_tensor_arg(),
            beta.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            eps,
        );
    }
    out
}
