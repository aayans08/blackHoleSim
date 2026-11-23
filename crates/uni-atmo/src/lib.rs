//! uni-atmo: planetary atmospheres (M1 Task B3): Hillaire 2020 transmittance, multiple-scattering
//! and ground-irradiance LUTs with Bruneton 2017's (r, μ) parameterisation, a per-pixel march for
//! the view, and the ground-lighting group the bodies pass (uni-planet, B1) binds at group 2.
//! Cloud shells in M2.
//!
//! Task B0 wrote this file, the module stubs and the stub WGSL (valid, neutral) so B1 can import
//! `atmo/ground_lighting.wgsl` from day one. B3 implements every module; only B0 and the lead edit
//! this file (module list, re-exports, shader registration).
//!
//! - [`params`]: [`AtmosphereParams`], the four presets, [`camera_in_daylight`],
//!   [`scintillation_strength`].
//! - [`lut`]: [`AtmosphereSet`] (LUTs per atmosphere, built once) and the LUT CPU twins.
//! - [`ground`]: [`GroundLighting`], the group-2 bind group of the bodies pass.
//! - [`feature`]: [`AtmosphereFeature`] (pass "atmosphere"), [`AtmoScene`], [`segment_radiance`].
//! - [`probes`]: [`gpu_probes`] for `universe --selftest`.

pub mod feature;
pub mod ground;
pub mod lut;
pub mod params;
pub mod probes;

pub use feature::{AtmoDraw, AtmoScene, AtmosphereFeature, segment_radiance};
pub use ground::{AtmoGround, GroundLighting};
pub use lut::{AtmosphereSet, multiscatter, sky_irradiance, transmittance};
pub use params::{
    AtmosphereParams, VENUS_CLOUD_TOP_M, camera_in_daylight, earth, mars, preset_for,
    scintillation_strength, titan, venus,
};
pub use probes::gpu_probes;

// A dependency the stubs do not use yet (B0 adds every Wave B manifest entry, review I8). This
// line keeps `cargo::unused_dependencies` quiet until B3 uses it; B3 deletes it then.
use uni_core as _;

/// `ShaderLib` path of the ground-lighting functions and their group-2 bindings
/// (`atmo_sun_transmittance`, `atmo_sky_irradiance`). The bodies pass (B1) imports it.
pub const GROUND_LIGHTING_WGSL: &str = "atmo/ground_lighting.wgsl";
/// `ShaderLib` path of the transmittance LUT pass (256 × 64, Rgba16Float).
pub const TRANSMITTANCE_LUT_WGSL: &str = "atmo/transmittance_lut.wgsl";
/// `ShaderLib` path of the multiple-scattering LUT pass (32 × 32, Rgba16Float).
pub const MULTISCATTER_LUT_WGSL: &str = "atmo/multiscatter_lut.wgsl";
/// `ShaderLib` path of the ground-irradiance LUT pass (64 × 16, Rgba16Float).
pub const IRRADIANCE_LUT_WGSL: &str = "atmo/irradiance_lut.wgsl";
/// `ShaderLib` path of the view march shared by the three draw variants.
pub const ATMOSPHERE_COMMON_WGSL: &str = "atmo/atmosphere_common.wgsl";
/// `ShaderLib` path of the dual-source variant's colour pipeline (needs
/// `wgpu::Features::DUAL_SOURCE_BLENDING`; validate with `validate_wgsl_with`).
pub const ATMOSPHERE_DUAL_WGSL: &str = "atmo/atmosphere_dual.wgsl";
/// `ShaderLib` path of the grey-transmittance variant (one pipeline, two targets).
pub const ATMOSPHERE_GREY_WGSL: &str = "atmo/atmosphere_grey.wgsl";
/// `ShaderLib` path of the dual-source variant's `lens_mask.g` pipeline.
pub const ATMOSPHERE_MASK_WGSL: &str = "atmo/atmosphere_mask.wgsl";

/// Registers every uni-atmo WGSL file in `lib` (which should come from
/// [`uni_gfx::ShaderLib::with_common`]). `uni_planet::register_shaders` calls it, because
/// `planet/body.wgsl` imports [`GROUND_LIGHTING_WGSL`].
pub fn register_shaders(lib: &mut uni_gfx::ShaderLib) {
    lib.add(
        GROUND_LIGHTING_WGSL,
        include_str!("../shaders/ground_lighting.wgsl"),
    );
    lib.add(
        TRANSMITTANCE_LUT_WGSL,
        include_str!("../shaders/transmittance_lut.wgsl"),
    );
    lib.add(
        MULTISCATTER_LUT_WGSL,
        include_str!("../shaders/multiscatter_lut.wgsl"),
    );
    lib.add(
        IRRADIANCE_LUT_WGSL,
        include_str!("../shaders/irradiance_lut.wgsl"),
    );
    lib.add(
        ATMOSPHERE_COMMON_WGSL,
        include_str!("../shaders/atmosphere_common.wgsl"),
    );
    lib.add(
        ATMOSPHERE_DUAL_WGSL,
        include_str!("../shaders/atmosphere_dual.wgsl"),
    );
    lib.add(
        ATMOSPHERE_GREY_WGSL,
        include_str!("../shaders/atmosphere_grey.wgsl"),
    );
    lib.add(
        ATMOSPHERE_MASK_WGSL,
        include_str!("../shaders/atmosphere_mask.wgsl"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_gfx::{ShaderLib, validate_wgsl, validate_wgsl_with};
    use wgpu::naga::valid::Capabilities as Caps;

    /// Every uni-atmo WGSL file composes and validates (Global Constraints: every WGSL file is
    /// validated by a naga test in the crate that owns it). The dual-source file needs
    /// `DUAL_SOURCE_BLENDING` (review I7); the others validate with m0.1's capabilities.
    #[test]
    fn wave_b_atmo_shaders_validate() {
        let mut lib = ShaderLib::with_common();
        register_shaders(&mut lib);
        for path in [
            GROUND_LIGHTING_WGSL,
            TRANSMITTANCE_LUT_WGSL,
            MULTISCATTER_LUT_WGSL,
            IRRADIANCE_LUT_WGSL,
            ATMOSPHERE_COMMON_WGSL,
            ATMOSPHERE_GREY_WGSL,
            ATMOSPHERE_MASK_WGSL,
        ] {
            let src = lib.compose(path).unwrap();
            if let Err(e) = validate_wgsl(&src) {
                panic!("{path}: {e}\n{src}");
            }
        }
        let src = lib.compose(ATMOSPHERE_DUAL_WGSL).unwrap();
        assert!(
            src.starts_with("enable dual_source_blending;\n"),
            "compose hoists the directive: {src}"
        );
        let caps = Caps::default() | Caps::SHADER_FLOAT16 | Caps::DUAL_SOURCE_BLENDING;
        if let Err(e) = validate_wgsl_with(&src, caps) {
            panic!("{ATMOSPHERE_DUAL_WGSL}: {e}\n{src}");
        }
    }

    /// The ground-lighting contract B1 builds on (brief "Interfaces"): the two functions exist,
    /// and group 2 holds the uniform, the two LUTs and the sampler at bindings 0–3.
    #[test]
    fn ground_lighting_contract() {
        let mut lib = ShaderLib::with_common();
        register_shaders(&mut lib);
        let m = validate_wgsl(&lib.compose(GROUND_LIGHTING_WGSL).unwrap()).unwrap();
        for name in ["atmo_sun_transmittance", "atmo_sky_irradiance"] {
            assert!(
                m.functions
                    .iter()
                    .any(|(_, f)| f.name.as_deref() == Some(name)),
                "missing fn {name}"
            );
        }
        let binding = |name: &str| {
            m.global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some(name))
                .and_then(|(_, v)| v.binding.clone())
                .map(|b| (b.group, b.binding))
        };
        assert_eq!(binding("atmo_ground"), Some((2, 0)));
        assert_eq!(binding("atmo_transmittance_lut"), Some((2, 1)));
        assert_eq!(binding("atmo_irradiance_lut"), Some((2, 2)));
        assert_eq!(binding("atmo_sampler"), Some((2, 3)));
    }
}
