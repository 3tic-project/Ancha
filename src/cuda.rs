//! CUDA device preflight and the persistent NVRTC kernel cache.
use anyhow::{Context, Result, anyhow, ensure};
use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
use cudarc::driver::{result, sys::CUdevice_attribute};
use std::sync::OnceLock;

/// A CUDA device checked through the driver API.
#[derive(Debug, Clone)]
pub struct CudaInfo {
    pub index: usize,
    pub name: String,
    /// Compute capability as `major * 10 + minor`, e.g. 61 for sm_61.
    pub compute_capability: i32,
}

/// Validate the device ordinal before CubeCL starts its server, which panics on
/// a missing driver or an invalid ordinal instead of returning an error.
pub fn probe(index: usize) -> Result<CudaInfo> {
    std::panic::catch_unwind(result::init)
        .map_err(|_| anyhow!("CUDA driver library (libcuda) could not be loaded"))?
        .context("CUDA driver initialization")?;
    let count = result::device::get_count().context("CUDA device count")?;
    ensure!(
        index < count.max(0) as usize,
        "CUDA device {index} not found; {count} visible"
    );
    let device = result::device::get(index as i32).context("CUDA device handle")?;
    let name = result::device::get_name(device).context("CUDA device name")?;
    // SAFETY: `device` is a valid handle from `device::get`; attributes are read-only queries.
    let (major, minor) = unsafe {
        use CUdevice_attribute::*;
        (
            result::device::get_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            result::device::get_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        )
    };
    Ok(CudaInfo {
        index,
        name,
        compute_capability: major.context("CUDA compute capability")? * 10
            + minor.context("CUDA compute capability")?,
    })
}

/// Probe the device and, once per process, persist NVRTC PTX per architecture next to
/// the autotune cache. Without it every process recompiles each kernel on its first
/// model call. Call before any CUDA tensor operation; a `[compilation] cache` set in
/// cubecl.toml / Burn.toml is kept as is.
pub fn prepare(index: usize) -> Result<CudaInfo> {
    let info = probe(index)?;
    static CACHE_ARCH: OnceLock<i32> = OnceLock::new();
    let arch = *CACHE_ARCH.get_or_init(|| {
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        if config.compilation.cache.is_none() {
            let root = CacheConfig::Target
                .root()
                .join(format!("ptx-sm{}", info.compute_capability));
            config.compilation.cache = Some(CacheConfig::File(root));
        }
        CubeClRuntimeConfig::set(config);
        info.compute_capability
    });
    ensure!(
        arch == info.compute_capability,
        "this process caches sm_{arch} kernels; run sm_{} devices in a separate process",
        info.compute_capability
    );
    Ok(info)
}
