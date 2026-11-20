//! uni-rel: gravitational lensing. M0: brute-force Schwarzschild (Binet RK4) ground-truth tracer.

pub mod binet;
pub mod tracer;

pub use tracer::{
    BEYOND_REACH_WARNING, LensCoverage, LensTracer, TRACE_REACH, inside_horizon, lens_coverage,
};

/// `ShaderLib` path of the Binet integrator (`binet_trace`, `static_observer_ray`).
pub const BINET_WGSL: &str = "rel/binet.wgsl";
/// `ShaderLib` path of the M0 debug sky (`debug_sky`).
pub const DEBUG_SKY_WGSL: &str = "rel/debug_sky.wgsl";
/// `ShaderLib` path of the lens pass (`cs_trace`, `cs_resolve`).
pub const LENS_TRACE_WGSL: &str = "rel/lens_trace.wgsl";

/// Registers this crate's WGSL files in `lib` (they import `common/camera.wgsl`, so `lib`
/// should come from [`uni_gfx::ShaderLib::with_common`]). [`LensTracer::new`] calls this; the
/// selftest calls it to compose its own kernel around `binet_trace`.
pub fn register_shaders(lib: &mut uni_gfx::ShaderLib) {
    lib.add(BINET_WGSL, include_str!("../shaders/binet.wgsl"));
    lib.add(DEBUG_SKY_WGSL, include_str!("../shaders/debug_sky.wgsl"));
    lib.add(LENS_TRACE_WGSL, include_str!("../shaders/lens_trace.wgsl"));
}
