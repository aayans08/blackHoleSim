//! uni-sky: everything drawn at infinity or as point sources. M0: PSF star splats with Planck
//! colours ([`StarField`], the BSC5 fallback). M1 Wave B: AT-HYG stars in 3-D ([`StarField3`],
//! B4), point sprites ([`PointSprites`], B4), the Milky Way band and the environment cubemap
//! ([`BandCubemap`], [`BandDraw`], B5).
//!
//! [`psf`] holds the CPU twins of the shaders' PSF maths. Task B0 wrote the Wave B module
//! stubs; B4 owns [`stars3d`], [`sectors`], [`points`] and [`probes`], B5 owns [`band`]. Only B0
//! and the lead edit this file (module list, re-exports, shader registration).
//!
//! M1 Wave C: Task C0 added [`asteroids`] and [`belt`] (C2's real asteroids and procedural belt,
//! pass "asteroids") with the shader `sky/asteroids.wgsl`, and the `asteroid_probes` call in
//! [`probes`].
//!
//! M1 final fix wave (F3, final review render I2): [`world_points`], the light-adapting point
//! lights (the Sun beyond ~3 AU) drawn as world light by the star-surface pass, with the shader
//! `sky/world_points.wgsl`.

pub mod asteroids;
pub mod band;
pub mod belt;
pub mod points;
pub mod probes;
pub mod psf;
pub mod sectors;
pub mod stars;
pub mod stars3d;
pub mod world_points;

pub use asteroids::{
    AsteroidField, AsteroidInput, BELT_HIGHLIGHT_GAIN, belt_highlight_v, kepler_f32, split_f64,
};
pub use band::{BandCubemap, BandDraw, BandInput, band_texel};
pub use belt::{BELT_KUIPER_COUNT, BELT_MAIN_COUNT, BeltModel};
pub use points::{PointSprite, PointSprites};
pub use probes::gpu_probes;
pub use psf::psf_weights;
pub use sectors::sector_of;
pub use stars::{DEFAULT_STAR_EV100, STARS_SHADER, StarField, StarInstance, star_instance};
pub use stars3d::{
    STARS3D_MIN_RADIUS_PX, StarField3, StarInput, star_apparent_mag, visible_star_count,
    vivid_color,
};
pub use world_points::{
    ADAPTING_LUX, WorldPoint, WorldPointDrawer, is_adapting_light_sprite, light_adapts_eye,
};

// A dependency the library does not use yet (B0 added every Wave B manifest entry, review I8);
// its tests do. This line keeps `cargo::unused_dependencies` quiet; delete it when the library
// uses it. (Task C6 removed `uni_physics`'s, which C2's asteroids now use.)
use uni_sim as _;

/// `ShaderLib` path of the 3-D star pass (B4).
pub const STARS3D_WGSL: &str = "sky/stars3d.wgsl";
/// `ShaderLib` path of the point-sprite pass (B4).
pub const POINTS_WGSL: &str = "sky/points.wgsl";
/// `ShaderLib` path of the band-update compute pass (B5).
pub const BAND_UPDATE_WGSL: &str = "sky/band_update.wgsl";
/// `ShaderLib` path of the band draw pass (B5).
pub const BAND_DRAW_WGSL: &str = "sky/band_draw.wgsl";
/// `ShaderLib` path of the asteroid pass (C2).
pub const ASTEROIDS_WGSL: &str = "sky/asteroids.wgsl";

/// Registers every uni-sky WGSL file in `lib` (which should come from
/// [`uni_gfx::ShaderLib::with_common`]): M0's `stars.wgsl`, the Wave B passes, and uni-procgen's
/// `procgen/galaxy.wgsl`, which `band_update.wgsl` imports (ruling A.6).
pub fn register_shaders(lib: &mut uni_gfx::ShaderLib) {
    stars::register_shaders(lib);
    lib.add(
        uni_procgen::galaxy::GALAXY_WGSL_PATH,
        uni_procgen::galaxy::GALAXY_WGSL_SRC,
    );
    lib.add(STARS3D_WGSL, include_str!("../shaders/stars3d.wgsl"));
    lib.add(POINTS_WGSL, include_str!("../shaders/points.wgsl"));
    lib.add(
        BAND_UPDATE_WGSL,
        include_str!("../shaders/band_update.wgsl"),
    );
    lib.add(BAND_DRAW_WGSL, include_str!("../shaders/band_draw.wgsl"));
    lib.add(ASTEROIDS_WGSL, include_str!("../shaders/asteroids.wgsl"));
    lib.add(
        world_points::WORLD_POINTS_WGSL,
        include_str!("../shaders/world_points.wgsl"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec3, Vec3};
    use uni_gfx::{ShaderLib, validate_wgsl};

    /// Every Wave B uni-sky WGSL file composes (the galaxy import included) and validates (Global
    /// Constraints: every WGSL file is validated by a naga test in the crate that owns it).
    #[test]
    fn wave_b_sky_shaders_validate() {
        let mut lib = ShaderLib::with_common();
        register_shaders(&mut lib);
        for path in [
            STARS_SHADER,
            STARS3D_WGSL,
            POINTS_WGSL,
            BAND_UPDATE_WGSL,
            BAND_DRAW_WGSL,
            ASTEROIDS_WGSL,
            world_points::WORLD_POINTS_WGSL,
        ] {
            let src = lib.compose(path).unwrap();
            if let Err(e) = validate_wgsl(&src) {
                panic!("{path}: {e}\n{src}");
            }
        }
        // The vivid-colour twin lives in the star shader (carry-over 1).
        let m = validate_wgsl(&lib.compose(STARS3D_WGSL).unwrap()).unwrap();
        assert!(
            m.functions
                .iter()
                .any(|(_, f)| f.name.as_deref() == Some("vivid_color"))
        );
    }

    /// Carry-over 1 (owner feedback): vivid colours are on by default, no twinkle in space, and
    /// the boost keeps every star's Rec. 709 luminance, so the photometry stays physical.
    #[test]
    fn star_input_defaults_and_vivid_color_keeps_luminance() {
        let input = StarInput::default();
        assert!(input.vivid_colours);
        assert_eq!(input.scintillation, 0.0);
        assert_eq!(input.camera_pc, DVec3::ZERO);
        assert!(input.suppressed.is_empty());
        // ITU-R BT.709-6 item 3.2 luminance weights.
        let y = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        for t in [2500.0, 3500.0, 5772.0, 9940.0, 30_000.0] {
            let c = uni_core::blackbody_linear_srgb(t);
            let c = c / y(c);
            let v = vivid_color(c);
            assert!((y(v) - 1.0).abs() < 1e-5, "{t} K: {c} -> {v}");
            assert!(v.min_element() >= 0.0, "{t} K: {v}");
        }
        assert_eq!(STARS3D_MIN_RADIUS_PX, 2.0);
    }
}
