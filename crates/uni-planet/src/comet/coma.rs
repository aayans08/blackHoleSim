//! The coma (Task C1).
//!
//! **Light.** The total luminous intensity of a comet from SBDB's magnitude law, I = E·Δ² with
//! E = `star_illuminance_lux(comet_total_v(M1, K1, r, Δ))` ([`comet_total_intensity_cd`]).
//! The coma carries that total minus the nucleus's own reflected light, floored at 0 (lead ruling
//! "comet double light": the app's point keeps the nucleus flux only, so the comet's light is not
//! counted twice; [`coma_intensity_cd`]). A comet without both M1 and K1 has no coma.
//!
//! **Profile.** Surface brightness S(ρ) = K/ρ · e^(−ρ/ℓ) [cd/m²] at projected distance ρ from
//! the nucleus, with ℓ = 10⁵ km · r_h² (photodissociation scale, Haser 1957, Bull. Acad. R. Sci.
//! Belg. 43, 740; the r_h² scaling as in Combi, Harris & Smyth 2004, *Comets II*, 523; the plan's
//! values), normalised so ∫S dA = I: K = I/(2πℓ) ([`coma_surface_brightness`]). On screen the
//! cusp at ρ = 0 is softened over one pixel and a coma smaller than a pixel is drawn pixel-sized,
//! both keeping ∫S dA = I exactly ([`coma_drawn_brightness`]). The profile is the sky-plane
//! projection seen from outside the coma; from inside it, it is approximate.
//!
//! **Colour.** Half C₂ Swan-band green (the 516 nm chromaticity, gamut-mapped into Rec. 709),
//! half sunlight, by luminance (approximate; card note `uni_ui::theme::NOTE_TAILS` covers the
//! comet's brightness model). A camera-facing quad per comet, additive, world exposure
//! (`scene_pre`), depth-tested, no depth write. The coma is optically thin: it does not dim the
//! sky behind it, so it writes nothing to `lens_mask`.

use glam::Vec3;
use uni_core::{AU, TEFF_SUN_K, blackbody_linear_srgb, star_illuminance_lux};
use uni_physics::photometry::comet_total_v;
use uni_sim::snapshot::CometView;

use super::line_colour;
use super::nucleus::nucleus_illuminance_lux;

/// The coma's scale length at r_h = 1 AU [m]: 10⁵ km (plan C1; Haser 1957, Combi et al. 2004).
pub const COMA_SCALE_AT_1AU_M: f64 = 1.0e8;

/// The coma quad reaches this many scale lengths from the nucleus: the light beyond it is
/// e^(−10) = 4.5 × 10⁻⁵ of the coma's (∫S dA within ρ is I·(1 − e^(−ρ/ℓ))).
pub const COMA_EXTENT_SCALES: f64 = 10.0;

/// CIE 1931 2° colour-matching functions x̄, ȳ, z̄ at 516 nm, the C₂ Swan (0,0) band head
/// (CIE 2018, "CIE 1931 colour-matching functions, 2 degree observer", data table, DOI
/// 10.25039/CIE.DS.xvudnb9b; read 2026-11-23).
pub const C2_SWAN_CMF: [f64; 3] = [0.034_814_85, 0.629_345_6, 0.103_904_8];

/// The coma's scale length ℓ [m] at heliocentric distance `r_au`: 10⁵ km · r_h².
pub fn coma_scale_m(r_au: f64) -> f64 {
    COMA_SCALE_AT_1AU_M * r_au * r_au
}

/// Coma surface brightness [cd/m²] at projected radius `rho_m` [m] for a coma of total luminous
/// intensity `intensity_cd` [cd] and scale length `scale_m` [m] (module docs). CPU twin of the
/// shader's; probed on the GPU.
pub fn coma_surface_brightness(rho_m: f64, intensity_cd: f64, scale_m: f64) -> f64 {
    intensity_cd / (std::f64::consts::TAU * scale_m) / rho_m * (-rho_m / scale_m).exp()
}

/// The surface brightness the pass draws [cd/m²]: [`coma_surface_brightness`] softened over
/// `pixel_m` (the radius of a disk with one pixel's area at the nucleus's distance [m]).
///
/// With a = `pixel_m`, ℓ′ = max(ℓ, a) and u = √(ρ² + a²) it is I·e^(a/ℓ′)/(2πℓ′) · e^(−u/ℓ′)/u:
/// substituting u (u du = ρ dρ) gives ∫S dA = I exactly. Beyond a few pixels from the nucleus of
/// a resolved coma (ℓ ≫ a) it is S(ρ); at the nucleus it is ≈ 2K/a, the exact profile's mean
/// over a disk of radius a, so the cusp neither aliases nor loses light; a coma smaller than a
/// pixel (ℓ < a) becomes a soft blob a few pixels wide carrying the same light.
pub fn coma_drawn_brightness(rho_m: f64, intensity_cd: f64, scale_m: f64, pixel_m: f64) -> f64 {
    let l = scale_m.max(pixel_m);
    let u = (rho_m * rho_m + pixel_m * pixel_m).sqrt();
    coma_surface_brightness(u, intensity_cd * (pixel_m / l).exp(), l)
}

/// Total (nucleus + coma) luminous intensity [cd] of a comet with SBDB's M1, K1 at heliocentric
/// `r_au` and camera distance `delta_au`: E·Δ² (module docs). Independent of Δ.
pub fn comet_total_intensity_cd(m1: f64, k1: f64, r_au: f64, delta_au: f64) -> f64 {
    let delta_m = delta_au * AU;
    star_illuminance_lux(comet_total_v(m1, k1, r_au, delta_au)) * delta_m * delta_m
}

/// The coma's luminous intensity [cd]: the total minus the nucleus's reflected light, floored at
/// 0; 0 without SBDB's M1 and K1 (module docs).
pub fn coma_intensity_cd(view: &CometView) -> f64 {
    let (Some(m1), Some(k1)) = (view.m1, view.k1) else {
        return 0.0;
    };
    let d = view.nucleus_rel_m.length();
    if !(d > 0.0 && d.is_finite() && view.r_au > 0.0) {
        return 0.0;
    }
    let total_lux = star_illuminance_lux(comet_total_v(m1, k1, view.r_au, view.delta_au));
    let coma_lux = (total_lux - nucleus_illuminance_lux(view)).max(0.0);
    coma_lux * d * d
}

/// The coma's colour, linear Rec. 709 with luminance 1: half C₂ Swan green, half sunlight.
pub fn coma_colour() -> Vec3 {
    0.5 * line_colour(C2_SWAN_CMF) + 0.5 * blackbody_linear_srgb(TEFF_SUN_K)
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::comet::luminance;
    use crate::comet::test_support::static_view;

    /// 2π ∫₀^R S(ρ) ρ dρ by the midpoint rule in ρ (the integrand 2πρS is finite at 0).
    fn encircled(s: impl Fn(f64) -> f64, r_max: f64, n: usize) -> f64 {
        let h = r_max / n as f64;
        (0..n)
            .map(|k| {
                let rho = (k as f64 + 0.5) * h;
                2.0 * std::f64::consts::PI * rho * s(rho) * h
            })
            .sum()
    }

    /// ∫S dA out to 20ℓ is I within 1 % (plan C1), for several intensities and r_h; the drawn
    /// (softened) profile also integrates to I, for a resolved coma (pixel ≪ ℓ) and for a coma
    /// smaller than a pixel.
    #[test]
    fn coma_profile_integrates_to_intensity() {
        for (i, r_au) in [(3.0e15, 1.3), (2.4e17, 0.5), (7.0e11, 4.9)] {
            let l = coma_scale_m(r_au);
            assert!((l / (1.0e8 * r_au * r_au) - 1.0).abs() < 1e-15);
            let total = encircled(|rho| coma_surface_brightness(rho, i, l), 20.0 * l, 200_000);
            assert!((total / i - 1.0).abs() < 0.01, "r_h {r_au}: ∫S dA = {total} of {i}");
            for pixel in [1e-3 * l, 0.3 * l, 10.0 * l] {
                let reach = 30.0 * l.max(pixel);
                let drawn = encircled(
                    |rho| coma_drawn_brightness(rho, i, l, pixel),
                    reach,
                    400_000,
                );
                assert!(
                    (drawn / i - 1.0).abs() < 0.01,
                    "pixel {pixel} m, ℓ {l} m: drawn ∫S dA = {drawn} of {i}"
                );
            }
        }
        // Far from the nucleus the drawn profile is the exact one, scaled by e^(a/ℓ) ≈ 1 + a/ℓ:
        // the light the softening takes from the cusp's central pixel.
        let (l, a) = (1.0e8, 1.0e4);
        let (exact, drawn) = (
            coma_surface_brightness(3.0 * l, 1e15, l),
            coma_drawn_brightness(3.0 * l, 1e15, l, a),
        );
        assert!((drawn / exact - 1.0).abs() < 2.0 * a / l, "{drawn} vs {exact}");
    }

    /// The total intensity follows SBDB's K1 law: from r_h 1 to 2 AU at fixed Δ it scales by
    /// 10^(−0.4·K1·log₁₀2) (±1e-9, plan C1), and it does not depend on Δ.
    #[test]
    fn coma_follows_k1_law() {
        for (m1, k1) in [(5.5, 10.0), (8.0, 25.0), (12.0, 4.0), (6.0, 0.0)] {
            let ratio = comet_total_intensity_cd(m1, k1, 2.0, 0.8)
                / comet_total_intensity_cd(m1, k1, 1.0, 0.8);
            let want = 10f64.powf(-0.4 * k1 * 2f64.log10());
            assert!((ratio / want - 1.0).abs() < 1e-9, "K1 {k1}: {ratio} vs {want}");
            let near = comet_total_intensity_cd(m1, k1, 1.5, 0.05);
            let far = comet_total_intensity_cd(m1, k1, 1.5, 3.0);
            assert!((near / far - 1.0).abs() < 1e-12);
        }
        // M1 = 0 at r = Δ = 1 AU: 2.54e-6 lx at 1 AU (Allen 1973's zero point).
        let i = comet_total_intensity_cd(0.0, 10.0, 1.0, 1.0);
        assert!((i / (star_illuminance_lux(0.0) * AU * AU) - 1.0).abs() < 1e-12);
    }

    /// Lead ruling "comet double light": the coma carries SBDB's total minus the nucleus's own
    /// light (p_V 0.04, assumed radius), floored at 0; none without M1/K1.
    #[test]
    fn coma_carries_total_minus_nucleus() {
        let sun_rel = DVec3::new(-AU, 0.0, 0.0);
        let rel = DVec3::new(0.3 * AU, 0.4 * AU, 0.0);
        let view = static_view(3, rel, sun_rel, Some(9.0), Some(12.0));
        let total = star_illuminance_lux(view.total_v.unwrap_or(f64::NAN));
        let nucleus = nucleus_illuminance_lux(&view);
        assert!(nucleus > 0.0 && nucleus < total);
        let want = (total - nucleus) * rel.length_squared();
        assert!((coma_intensity_cd(&view) / want - 1.0).abs() < 1e-12);
        // A nucleus brighter than SBDB's total (a faint M1, a huge nucleus) leaves no coma.
        let mut huge = static_view(4, rel, sun_rel, Some(30.0), Some(12.0));
        huge.nucleus_radius_m = 5.0e4;
        assert_eq!(coma_intensity_cd(&huge), 0.0);
        let no_law = static_view(5, rel, sun_rel, None, None);
        assert_eq!(coma_intensity_cd(&no_law), 0.0);
    }

    /// The colour has luminance 1 and no negative channel; its C₂ half is the 516 nm colour
    /// moved toward white just far enough to enter the gamut (one channel exactly 0), so it is
    /// green-dominant.
    #[test]
    fn coma_colour_is_half_c2_half_sunlight() {
        let c2 = line_colour(C2_SWAN_CMF);
        assert!((luminance(c2) - 1.0).abs() < 1e-5, "{c2}");
        assert!(c2.min_element() >= 0.0 && c2.min_element() < 1e-6, "{c2}");
        assert!(c2.y > c2.x && c2.y > c2.z, "{c2}");
        let c = coma_colour();
        assert!((luminance(c) - 1.0).abs() < 1e-4, "{c}");
        assert!(c.min_element() > 0.0 && c.y > c.x && c.y > c.z, "{c}");
        let sun = uni_core::blackbody_linear_srgb(uni_core::TEFF_SUN_K);
        assert!((c - (0.5 * c2 + 0.5 * sun)).abs().max_element() < 1e-6);
    }
}
