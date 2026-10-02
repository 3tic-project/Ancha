//! Cache correctness policy. Metadata equality is necessary; source IDs must be verified digests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkKey {
    pub input_digest: String,
    pub model_digest: String,
    pub forward_digest: String,
    pub frontend_digest: String,
    pub start_sample: u64,
    pub chunk_samples: usize,
    pub precision: String,
    pub backend_plan: String,
}

pub fn can_reuse_full_chunk(a: &ChunkKey, b: &ChunkKey) -> bool {
    a == b
}

/// A new bidirectional context invalidates deep K/V. Whole-context equality is required.
pub fn can_reuse_deep_kv(a: &ChunkKey, b: &ChunkKey) -> bool {
    can_reuse_full_chunk(a, b)
}

#[derive(Clone, Debug)]
pub struct FrameContext {
    pub source_digest: String,
    // Includes gain, resampler, channel mapping, window and preprocessing values.
    pub frontend_digest: String,
    // Exact first projection/bandsplit weights, not just the model architecture name.
    pub projection_digest: String,
    pub start_sample: u64,
    pub chunk_samples: usize,
    pub hop: usize,
    pub n_fft: usize,
}

fn interior_center(c: &FrameContext, frame: usize) -> Option<u64> {
    if c.hop == 0 || c.n_fft == 0 || !c.n_fft.is_multiple_of(2) {
        return None;
    }
    let local = frame.checked_mul(c.hop)?;
    if local < c.n_fft / 2 || local.checked_add(c.n_fft / 2)? > c.chunk_samples {
        return None;
    }
    c.start_sample.checked_add(u64::try_from(local).ok()?)
}

/// Only STFT / pointwise frontend / FIRST pre-RoPE projection may use this admission.
/// It is deliberately not permission to reuse post-attention hidden states or rotated K.
pub fn can_reuse_first_pre_rope(a: &FrameContext, fa: usize, b: &FrameContext, fb: usize) -> bool {
    if a.source_digest != b.source_digest
        || a.frontend_digest != b.frontend_digest
        || a.projection_digest != b.projection_digest
        || a.hop != b.hop
        || a.n_fft != b.n_fft
    {
        return false;
    }
    match (interior_center(a, fa), interior_center(b, fb)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

pub fn phase_cycle(step: usize, hop: usize) -> Option<usize> {
    if hop == 0 {
        return None;
    }
    let (mut a, mut b) = (step, hop);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Some(hop / a)
}

/// Graphs cache executable work, not inference answers. Input CONTENT is intentionally absent.
/// Actual device addresses and ownership must be kept by a graph slot; they are not modeled here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphPlanKey {
    pub model_digest: String,
    pub kernel_plan_digest: String,
    pub device_identity: String,
    pub precision: String,
    pub batch: usize,
    pub tokens: usize,
    pub bands: usize,
    pub layout: String,
}
