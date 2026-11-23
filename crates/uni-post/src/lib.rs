//! uni-post: post-processing of the pre-exposed HDR frame. M0: upscale to the output, AgX
//! tonemap and sRGB dither. M1 Wave B (B6): histogram auto-exposure with the dual star exposure
//! ([`exposure`], [`histogram`]) and bloom ([`bloom`]).
//!
//! Task B0 wrote the Wave B module stubs; B6 owns [`exposure`], [`histogram`], [`bloom`] and
//! [`probes`]. Only B0 and the lead edit this file (module list, re-exports, shader
//! registration).

pub mod agx;
pub mod bloom;
pub mod exposure;
pub mod histogram;
pub mod probes;
pub mod tonemap;

pub use agx::agx;
pub use bloom::{BLOOM_INPUT_MAX, Bloom, DEFAULT_BLOOM_STRENGTH};
pub use exposure::{
    EV100_MAX, EV100_MIN, ExposureMode, ExposureReadback, ExposureResolve, ExposureSettings,
    NIGHT_EV100, adapt, auto_exposure, resolve_exposure,
};
pub use histogram::{
    ExposureMeter, HIGHLIGHT_PERCENTILE, HIGHLIGHT_PRE, METER_FULL_LIT_PX, METER_MIN_CD_M2,
    METER_MIN_LIT_PX, Meter, meter_histogram, target_ev,
};
pub use probes::gpu_probes;
pub use tonemap::{TONEMAP_SHADER, TonemapFeature};

/// `ShaderLib` path of the exposure resolve pass (B6). It imports only `common/exposure.wgsl`.
pub const EXPOSURE_WGSL: &str = "post/exposure.wgsl";
/// `ShaderLib` path of the meter pass (B6).
pub const HISTOGRAM_WGSL: &str = "post/histogram.wgsl";
/// `ShaderLib` path of the bloom passes (B6).
pub const BLOOM_WGSL: &str = "post/bloom.wgsl";

/// Registers every uni-post WGSL file in `lib` (which should come from
/// [`uni_gfx::ShaderLib::with_common`]): M0's tonemap ([`TONEMAP_SHADER`]) and the Wave B
/// passes.
pub fn register_shaders(lib: &mut uni_gfx::ShaderLib) {
    tonemap::register_shaders(lib);
    lib.add(EXPOSURE_WGSL, include_str!("../shaders/exposure.wgsl"));
    lib.add(HISTOGRAM_WGSL, include_str!("../shaders/histogram.wgsl"));
    lib.add(BLOOM_WGSL, include_str!("../shaders/bloom.wgsl"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_gfx::{Exposure, ShaderLib, validate_wgsl};

    /// Every uni-post WGSL file composes and validates (Global Constraints: every WGSL file is
    /// validated by a naga test in the crate that owns it), and the resolve pass imports no
    /// group-0 binding (review m14).
    #[test]
    fn wave_b_post_shaders_validate() {
        let mut lib = ShaderLib::with_common();
        register_shaders(&mut lib);
        for path in [TONEMAP_SHADER, EXPOSURE_WGSL, HISTOGRAM_WGSL, BLOOM_WGSL] {
            let src = lib.compose(path).unwrap();
            if let Err(e) = validate_wgsl(&src) {
                panic!("{path}: {e}\n{src}");
            }
        }
        let m = validate_wgsl(&lib.compose(EXPOSURE_WGSL).unwrap()).unwrap();
        assert!(
            m.global_variables
                .iter()
                .filter_map(|(_, v)| v.binding.as_ref())
                .all(|b| b.group != 0),
            "the resolve pass must not bind group 0"
        );
    }

    /// The eye-adapted condition of the M1 frame contract: eye_adapted && !daylight (review
    /// m12, m13).
    #[test]
    fn eye_adapted_in_effect_follows_daylight() {
        let mut s = ExposureSettings {
            mode: ExposureMode::Auto,
            manual: Exposure {
                scene_ev100: 0.0,
                star_ev100: NIGHT_EV100,
            },
            compensation_ev: 0.0,
            eye_adapted: true,
            eye_star_ev100: NIGHT_EV100,
            daylight: false,
            adapt_tau_s: [0.6, 1.2],
            settle: false,
        };
        assert!(s.eye_adapted_in_effect());
        s.daylight = true;
        assert!(!s.eye_adapted_in_effect());
        s.eye_adapted = false;
        s.daylight = false;
        assert!(!s.eye_adapted_in_effect());
        // The plan's constants (B6 rules; the app checks NIGHT_EV100 and the limits, B9).
        assert_eq!((EV100_MIN, EV100_MAX, NIGHT_EV100), (-30.0, 40.0, -4.0));
    }
}
