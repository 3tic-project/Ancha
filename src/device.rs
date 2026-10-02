//! Guard against GPU device failures that CubeCL does not report to the caller.
use anyhow::{Result, bail};
use std::sync::{Mutex, Once};

static FAILURE: Mutex<Option<String>> = Mutex::new(None);

/// CubeCL serves each device from `DSU-*` / `DSD-*` threads. In CubeCL 0.10 an allocation
/// failure there (e.g. out of device memory) panics on that thread only: later reads
/// return stale buffers and `Backend::sync` still succeeds. Chain a panic hook that
/// records such panics. Idempotent; the previous hook still runs.
pub fn install_guard() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let thread = std::thread::current();
            if thread
                .name()
                .is_some_and(|n| n.starts_with("DSU-") || n.starts_with("DSD-"))
            {
                FAILURE
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert_with(|| info.to_string());
            }
            previous(info);
        }));
    });
}

/// Fails once any device thread has panicked; every later result of the process is
/// untrusted, so the state is not cleared.
pub fn check() -> Result<()> {
    if let Some(message) = FAILURE.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        bail!(
            "GPU device thread failed, results are invalid (often out of device memory; \
             lower --max-score-mib or --mdx-batch-size): {message}"
        );
    }
    Ok(())
}
