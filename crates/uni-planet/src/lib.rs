//! uni-planet: analytic bodies and star surfaces (M1 Wave B); cube-sphere quadtree terrain,
//! oceans and night lights in M2.
//!
//! Task B0 wrote this file, the stubs and three shared twins that B1 and B2 both use, so neither
//! waits on the other (review I9):
//! - [`capmesh`]: [`cap_mesh`] / [`cap_indices`], the camera-facing circumscribed proxy mesh;
//! - [`eclipse`]: [`sun_visible_fraction`] and its WGSL twin `planet/eclipse.wgsl`;
//! - [`lod`]: [`lod_exposure_mix`], the point → disk exposure blend.
//!
//! B1 owns [`body`], [`material`] and [`probes`]; B2 owns [`star_surface`] and [`rings`]. Only B0
//! and the lead edit this file (module list, re-exports, shader registration).
//!
//! M1 Wave C: Task C0 added [`comet`] (C1's comets, pass "comets": nucleus, coma, dust and ion
//! tails) and its shader `planet/comet.wgsl`, and the `comet_probes` call in [`probes`].

pub mod body;
pub mod capmesh;
pub mod comet;
pub mod eclipse;
pub mod lod;
pub mod material;
pub mod probes;
pub mod rings;
pub mod star_surface;

pub use body::{
    BodyRenderer, BodyScene, equirect_uv, ray_ellipsoid, ring_transmittance,
    venus_cloud_longitude_offset_deg,
};
pub use capmesh::{CAP_RINGS, CAP_SEGMENTS, cap_indices, cap_mesh};
pub use comet::coma::coma_surface_brightness;
pub use comet::nucleus::nucleus_mesh;
pub use comet::tails::{ion_tail_dir, tail_schedule};
pub use comet::{CometRenderer, CometScene, MAX_TAILED_COMETS, TAIL_REFRESH_WALL_S};
pub use eclipse::sun_visible_fraction;
pub use lod::lod_exposure_mix;
pub use probes::gpu_probes;
pub use rings::{RingRenderer, planet_shadow_on_ring, ring_phase};
pub use star_surface::{StarSurfaceRenderer, granulation, limb_darkening, limb_darkening_mean};

// Dependencies the stubs do not use yet (B0 adds every Wave B manifest entry, review I8). These
// lines keep `cargo::unused_dependencies` quiet until B1/B2 use them; delete each one then.
use bytemuck as _;
use pollster as _;
use uni_physics as _;

/// `ShaderLib` path of the bodies pass (B1).
pub const BODY_WGSL: &str = "planet/body.wgsl";
/// `ShaderLib` path of the star-surface pass (B2).
pub const STAR_SURFACE_WGSL: &str = "planet/star_surface.wgsl";
/// `ShaderLib` path of the ring pass (B2).
pub const RINGS_WGSL: &str = "planet/rings.wgsl";
/// `ShaderLib` path of `sun_visible_fraction`, the WGSL twin of [`sun_visible_fraction`] (B0).
pub const ECLIPSE_WGSL: &str = "planet/eclipse.wgsl";
/// `ShaderLib` path of the comet pass (C1).
pub const COMET_WGSL: &str = "planet/comet.wgsl";

/// Registers every uni-planet WGSL file in `lib` (which should come from
/// [`uni_gfx::ShaderLib::with_common`]), and uni-atmo's, because `planet/body.wgsl` imports
/// `atmo/ground_lighting.wgsl`.
pub fn register_shaders(lib: &mut uni_gfx::ShaderLib) {
    uni_atmo::register_shaders(lib);
    lib.add(ECLIPSE_WGSL, include_str!("../shaders/eclipse.wgsl"));
    lib.add(BODY_WGSL, include_str!("../shaders/body.wgsl"));
    lib.add(
        STAR_SURFACE_WGSL,
        include_str!("../shaders/star_surface.wgsl"),
    );
    lib.add(RINGS_WGSL, include_str!("../shaders/rings.wgsl"));
    lib.add(COMET_WGSL, include_str!("../shaders/comet.wgsl"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_gfx::{ShaderLib, validate_wgsl};

    /// Every uni-planet WGSL file composes (cross-crate imports included) and validates (Global
    /// Constraints: every WGSL file is validated by a naga test in the crate that owns it).
    #[test]
    fn wave_b_planet_shaders_validate() {
        let mut lib = ShaderLib::with_common();
        register_shaders(&mut lib);
        for path in [
            ECLIPSE_WGSL,
            BODY_WGSL,
            STAR_SURFACE_WGSL,
            RINGS_WGSL,
            COMET_WGSL,
        ] {
            let src = lib.compose(path).unwrap();
            if let Err(e) = validate_wgsl(&src) {
                panic!("{path}: {e}\n{src}");
            }
        }
    }
}
