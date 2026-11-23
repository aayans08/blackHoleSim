//! GPU probes of uni-rel's WGSL against its CPU twins, run by `universe --selftest` (Task B7;
//! B0 stub).
//!
//! B7 adds the environment-sampling probe (exit direction → cube texel for 32 rays against a
//! CPU lookup through `uni_gfx::env::cube_dir`'s convention, the write-then-sample check A0a
//! deferred). M0's Binet check stays in the app's selftest (`selftest.rs`, ruling B.10).

use uni_gfx::probe::ProbeResult;
use uni_gfx::{GpuContext, ShaderLib};

/// Every uni-rel probe, registering its shaders in `lib`. B0 stub: none yet.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    let _ = gpu;
    crate::register_shaders(lib);
    Vec::new()
}
