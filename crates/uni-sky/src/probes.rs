//! GPU probes of uni-sky's WGSL against its CPU twins, run by `universe --selftest` (Task B4;
//! B0 stub).
//!
//! B4 adds: apparent magnitude and direction for 64 stars (±1e-3 mag, ±1e-6 rad) and the PSF
//! (erf coverage, halo weight, footprint radius; review m8). B5's band probes are
//! `band::band_probes`, called below.

use uni_gfx::probe::ProbeResult;
use uni_gfx::{GpuContext, ShaderLib};

/// Every uni-sky probe, registering its shaders in `lib`.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    crate::register_shaders(lib);
    let mut results = Vec::new();
    results.extend(crate::band::band_probes(gpu, lib));
    results
}
