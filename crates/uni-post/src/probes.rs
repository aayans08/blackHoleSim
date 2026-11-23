//! GPU probes of uni-post's WGSL against its CPU twins, run by `universe --selftest` (Task B6;
//! B0 stub).
//!
//! B6 adds: the histogram and its reduction vs `meter_histogram` + `target_ev` on a synthetic
//! 512² HDR (±0.05 EV), the resolve pass vs `resolve_exposure` for 16 settings (±1e-6
//! relative), and bloom on an impulse (±1 %). M0's `agx` check stays in the app's selftest.

use uni_gfx::probe::ProbeResult;
use uni_gfx::{GpuContext, ShaderLib};

/// Every uni-post probe, registering its shaders in `lib`. B0 stub: none yet.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    let _ = gpu;
    crate::register_shaders(lib);
    Vec::new()
}
