//! uni-rel: gravitational lensing. M0: brute-force Schwarzschild (Binet RK4) ground-truth tracer,
//! and the debug-grid background it lenses. M1 (Task B7): the lens environment is the Milky Way
//! band's cubemap ([`LensEnv::Band`]), with the M0 debug grid kept as an opt-in environment
//! ([`LensEnv::DebugGrid`], ruling 14) for the shadow regression bookmark.
//!
//! Task B0 changed the constructors and `debug_background_draws` to take the environment and
//! added the [`probes`] stub; B7 owns this crate's files.

pub mod background;
pub mod binet;
pub mod probes;
pub mod tracer;

pub use background::{DebugSkyBackground, debug_background_draws, lens_labels};
pub use probes::gpu_probes;
pub use tracer::{
    BEYOND_REACH_WARNING, DEFAULT_RESOLUTION_SCALE, LensCoverage, LensScale, LensTracer,
    TRACE_REACH, inside_horizon, lens_coverage,
};

/// What the lens pass shows through its escaped rays, and what the background draws around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LensEnv {
    /// The environment cubemap (the Milky Way band, `uni_sky::BandCubemap::env_map`): physical
    /// sky light at star exposure.
    Band,
    /// The M0 debug grid (ruling 14): display-referred, so the app runs Manual exposure while it
    /// is selected, and the HUD shows "Lens background: M0 debug grid".
    DebugGrid,
}

/// `ShaderLib` path of the Binet integrator (`binet_trace`, `static_observer_ray`).
pub const BINET_WGSL: &str = "rel/binet.wgsl";
/// `ShaderLib` path of the M0 debug sky (`debug_sky`, `pixel_footprint`).
pub const DEBUG_SKY_WGSL: &str = "rel/debug_sky.wgsl";
/// `ShaderLib` path of the M0 debug-grid background pass (`cs_background`).
pub const DEBUG_BACKGROUND_WGSL: &str = "rel/debug_background.wgsl";
/// `ShaderLib` path of the lens pass (`cs_trace`, `cs_resolve`).
pub const LENS_TRACE_WGSL: &str = "rel/lens_trace.wgsl";

/// Registers this crate's WGSL files in `lib` (they import `common/camera.wgsl`, so `lib`
/// should come from [`uni_gfx::ShaderLib::with_common`]). [`LensTracer::new`] and
/// [`DebugSkyBackground::new`] call this; the selftest calls it to compose its own kernel
/// around `binet_trace`.
pub fn register_shaders(lib: &mut uni_gfx::ShaderLib) {
    lib.add(BINET_WGSL, include_str!("../shaders/binet.wgsl"));
    lib.add(DEBUG_SKY_WGSL, include_str!("../shaders/debug_sky.wgsl"));
    lib.add(
        DEBUG_BACKGROUND_WGSL,
        include_str!("../shaders/debug_background.wgsl"),
    );
    lib.add(LENS_TRACE_WGSL, include_str!("../shaders/lens_trace.wgsl"));
}
