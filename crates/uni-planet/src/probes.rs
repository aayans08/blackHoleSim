//! GPU probes of uni-planet's WGSL against its CPU twins, run by `universe --selftest` (Task B1;
//! B0 stub).
//!
//! B1 adds: `ray_ellipsoid` (256 rays, including 1 m altitude), `equirect_uv` (256),
//! `sun_visible_fraction` (256, B0's twin in `planet/eclipse.wgsl`), `reflectance` (64),
//! `ring_transmittance` (64), `bodies_disk_flux` and `bc_matches_fallback`. B2's probes are its
//! own modules' `star_probes` and `ring_probes`, called below.

use uni_gfx::probe::ProbeResult;
use uni_gfx::{GpuContext, ShaderLib};

/// Every uni-planet probe, registering its shaders in `lib`.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    crate::register_shaders(lib);
    let mut results = Vec::new();
    results.extend(crate::star_surface::star_probes(gpu, lib));
    results.extend(crate::rings::ring_probes(gpu, lib));
    results
}
