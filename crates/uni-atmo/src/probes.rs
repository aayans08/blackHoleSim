//! GPU probes of uni-atmo's WGSL against its CPU twins, run by `universe --selftest` (Task B3;
//! B0 stub).
//!
//! B3 adds: the transmittance LUT vs [`crate::transmittance`] (64 samples ±1e-3), the
//! multiple-scattering LUT (16, ±2 %), the irradiance LUT vs [`crate::sky_irradiance`] (16,
//! ±2 %), `atmo_sun_transmittance` and `atmo_sky_irradiance` through the ground-lighting group
//! (32 each), and [`crate::segment_radiance`] (32 rays, ±2 %).

use uni_gfx::probe::ProbeResult;
use uni_gfx::{GpuContext, ShaderLib};

/// Every uni-atmo probe, registering its shaders in `lib`. B0 stub: none yet.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    let _ = gpu;
    crate::register_shaders(lib);
    Vec::new()
}
