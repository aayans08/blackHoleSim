//! Point-spread-function weights, the CPU twin of `shaders/stars.wgsl` (M0 Task 8).
//!
//! A star's image is a PSF made of two parts. The core is a Gaussian with σ = [`CORE_SIGMA_PX`].
//! Stars brighter than [`HALO_MIN_LUX`] also get a faint, wide glare halo, a Moffat profile that
//! carries [`HALO_ENERGY_FRACTION`] of the energy. The core is integrated exactly over each pixel's
//! square, with erf differences, so the weights over a footprint sum to the star's energy wherever
//! its centre lies. That is what keeps sub-pixel stars from flickering or looking pixelated in
//! motion (docs/research/2026-11-20-rendering.md §2, "Star point sprites").
//!
//! Coordinates are framebuffer pixels: x to the right, y down, and pixel k spans [k, k + 1].
//! Every function here has a WGSL twin of the same name in `stars.wgsl` (`erf` there is
//! `psf_erf`), and `shader_constants_match_cpu` keeps the constants equal.
//!
//! "Display-normalised" values are pixel values after the tonemap's exposure: what AgX
//! (uni-post) receives, where 0.18 is mid-grey and 2^−12.47393 = 0.18·2^−10 ≈ 1.76e-4 is black.

use std::f32::consts::{PI, SQRT_2, TAU};

use glam::Vec2;

/// σ of the Gaussian PSF core [px] (brief: 0.65; research §2 recommends 0.6–0.8).
pub const CORE_SIGMA_PX: f32 = 0.65;

/// Share of a bright star's energy in its glare halo. The core keeps the rest, so the total is
/// unchanged. The glare idea follows Spencer et al. 1995, "Physically-Based Glare Effects for
/// Digital Images" (SIGGRAPH '95), at the small strength of a camera or eye scatter tail.
pub const HALO_ENERGY_FRACTION: f32 = 0.02;
/// Stars with illuminance above this [lux] get the halo: about V < 1.0, since
/// `uni_core::star_illuminance_lux(1.0)` = 1.01e-6 lx.
pub const HALO_MIN_LUX: f32 = 1.0e-6;
/// Moffat core width α [px]. The halo's FWHM, 2α·sqrt(2^(1/β) − 1) = 4.5 px, is three times the
/// core's (2.355σ = 1.5 px), and its wings extend further for brighter stars.
pub const HALO_ALPHA_PX: f32 = 4.0;
/// Moffat exponent β (brief: 2.5). Wings fall as r^(−2β) = r^−5.
pub const HALO_BETA: f32 = 2.5;

/// Smallest footprint (quad) radius [px]. 3 px is 4.6σ, which keeps the core's energy to within
/// 1e-5.
pub const MIN_RADIUS_PX: f32 = 3.0;
/// Largest footprint radius [px]. It bounds the fill cost of the brightest stars.
pub const MAX_RADIUS_PX: f32 = 64.0;
/// Display-normalised value treated as invisible when sizing a footprint. uni-post's AgX clamps
/// log2 of its input at −12.47393 (Sobotka's AgX log2 range, Task 5), so its black point is
/// 2^−12.47393 = 0.18·2^−10 ≈ 1.76e-4. This cutoff is 2.55 stops below that, so cutting the PSF
/// there leaves no visible edge; it is conservative, and footprints are somewhat larger than
/// the black point alone needs.
pub const VISIBILITY_CUTOFF: f32 = 3.0e-5;

/// erf(x), by Abramowitz & Stegun 1964, *Handbook of Mathematical Functions*, eq. 7.1.26
/// (after Hastings 1955), |error| ≤ 1.5e-7. WGSL has no erf, so the shader's `psf_erf` uses the
/// same formula.
#[allow(clippy::excessive_precision)] // A&S's published digits, rounded to f32 by the compiler.
pub fn erf(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A1: f32 = 0.254_829_592;
    const A2: f32 = -0.284_496_736;
    const A3: f32 = 1.421_413_741;
    const A4: f32 = -1.453_152_027;
    const A5: f32 = 1.061_405_429;
    let a = x.abs();
    let t = 1.0 / (1.0 + P * a);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    let y = 1.0 - poly * (-a * a).exp();
    // erf is odd. `if` rather than `signum`, because signum(0.0) = 1 in Rust but 0 in WGSL.
    if x >= 0.0 { y } else { -y }
}

/// Fraction of a unit-energy 1-D Gaussian (σ = `sigma_px`, centred at 0) that falls in
/// [`edge_px`, `edge_px` + 1]: ½·[erf((e + 1)/(σ√2)) − erf(e/(σ√2))].
///
/// Neighbouring pixels share an edge, so the sum over a row telescopes to the mass between the
/// row's outer edges. That holds whatever erf's error is, which is why the energy is conserved
/// at any sub-pixel position. Clamped at 0 so rounding far out in the tail never gives a negative
/// weight. `sigma_px` must be positive.
pub fn pixel_coverage(edge_px: f32, sigma_px: f32) -> f32 {
    let k = 1.0 / (SQRT_2 * sigma_px);
    (0.5 * (erf((edge_px + 1.0) * k) - erf(edge_px * k))).max(0.0)
}

/// Per-pixel weights of the Gaussian PSF core around a star, CPU twin of the shader's
/// `pixel_coverage(edge.x) * pixel_coverage(edge.y)`.
///
/// `sub_px` is the star's centre inside its own pixel, measured from that pixel's top-left
/// corner (each component in [0, 1)). The result is the (2r + 1)² grid of pixels at offsets
/// dx, dy ∈ [−r, r] from that pixel, row-major with dy as the row (y down): index
/// (dy + r)·(2r + 1) + (dx + r). A negative `radius_px` is treated as 0.
pub fn psf_weights(sub_px: Vec2, sigma_px: f32, radius_px: i32) -> Vec<f32> {
    let r = radius_px.max(0);
    let coverage = |offset: i32, sub: f32| pixel_coverage(offset as f32 - sub, sigma_px);
    let row: Vec<f32> = (-r..=r).map(|dx| coverage(dx, sub_px.x)).collect();
    (-r..=r)
        .flat_map(|dy| {
            let wy = coverage(dy, sub_px.y);
            row.iter().map(move |&wx| wx * wy)
        })
        .collect()
}

/// Share of the energy in the glare halo for a star of illuminance `lux`:
/// [`HALO_ENERGY_FRACTION`] above [`HALO_MIN_LUX`], otherwise 0.
pub fn halo_fraction(lux: f32) -> f32 {
    if lux > HALO_MIN_LUX {
        HALO_ENERGY_FRACTION
    } else {
        0.0
    }
}

/// Moffat profile with unit energy over the plane [1/px²]:
/// I(r) = (β − 1)/(πα²)·(1 + r²/α²)^(−β). Moffat 1969, "A Theoretical Investigation of Focal
/// Stellar Images in the Photographic Emulsion", A&A 3, 455.
pub fn moffat(r_px: f32) -> f32 {
    let a2 = HALO_ALPHA_PX * HALO_ALPHA_PX;
    (HALO_BETA - 1.0) / (PI * a2) * (1.0 + r_px * r_px / a2).powf(-HALO_BETA)
}

/// Halo weight of a pixel whose centre is `r_px` from the star, for a footprint of `radius_px`.
///
/// The Moffat profile is lowered by its value at the radius R, so it reaches 0 continuously
/// there, and renormalised to unit energy inside R. Its energy inside R is
/// 1 − (1 + R²/α²)^(1−β), and lowering it removes πR²·I(R). Sampling a profile that falls
/// continuously to 0 at pixel centres sums to 1 within 1% for R ≥ 3 px; a hard cut at R would
/// be off by up to 8%.
pub fn halo_weight(r_px: f32, radius_px: f32) -> f32 {
    // Negated so that a NaN distance also gives 0.
    if !(r_px < radius_px) {
        return 0.0;
    }
    let a2 = HALO_ALPHA_PX * HALO_ALPHA_PX;
    let edge = moffat(radius_px);
    let inside = 1.0
        - (1.0 + radius_px * radius_px / a2).powf(1.0 - HALO_BETA)
        - PI * radius_px * radius_px * edge;
    (moffat(r_px) - edge) / inside
}

/// Footprint (quad) radius [px] for a star with display-normalised total energy `energy` and halo
/// share `halo_fraction`: the distance where both the core and the halo have fallen below
/// [`VISIBILITY_CUTOFF`], clamped to [[`MIN_RADIUS_PX`], [`MAX_RADIUS_PX`]].
///
/// The core's peak is (1 − h)·E/(2πσ²) and it falls to the cutoff at σ·sqrt(2·ln(peak/cutoff)).
/// The halo's peak is h·E·I(0) and it falls to the cutoff at α·sqrt((peak/cutoff)^(1/β) − 1).
/// A NaN energy gives the minimum radius.
pub fn footprint_radius_px(energy: f32, halo_fraction: f32) -> f32 {
    let mut r = MIN_RADIUS_PX;
    let core_peak = (1.0 - halo_fraction) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
    if core_peak > VISIBILITY_CUTOFF {
        let r_core = CORE_SIGMA_PX * (2.0 * (core_peak / VISIBILITY_CUTOFF).ln()).sqrt();
        r = r.max(r_core);
    }
    let halo_peak = halo_fraction * energy * moffat(0.0);
    if halo_peak > VISIBILITY_CUTOFF {
        let q = (halo_peak / VISIBILITY_CUTOFF).powf(1.0 / HALO_BETA);
        r = r.max(HALO_ALPHA_PX * (q - 1.0).sqrt());
    }
    r.min(MAX_RADIUS_PX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psf_conserves_energy_any_subpixel() {
        for (x, y) in [(0.0, 0.0), (0.5, 0.5), (0.13, 0.91), (0.99, 0.01)] {
            let s: f32 = psf_weights(Vec2::new(x, y), 0.65, 3).iter().sum();
            assert!((s - 1.0).abs() < 2e-3, "sum {s} at ({x}, {y})");
        }
    }

    #[test]
    fn psf_peak_moves_with_subpixel() {
        let r = 3;
        let n = (2 * r + 1) as usize;
        let at = |w: &[f32], dx: i32, dy: i32| w[(dy + r) as usize * n + (dx + r) as usize];
        let centred = psf_weights(Vec2::splat(0.5), CORE_SIGMA_PX, r);
        let off = psf_weights(Vec2::splat(0.9), CORE_SIGMA_PX, r);
        assert!(at(&centred, 0, 0) > at(&off, 0, 0));
        // A centred star is symmetric; one near the right edge leans right (and down).
        assert!((at(&centred, -1, 0) - at(&centred, 1, 0)).abs() < 1e-6);
        assert!(at(&off, 1, 0) > at(&off, -1, 0));
        assert!(at(&off, 0, 1) > at(&off, 0, -1));
        assert!(off.iter().chain(&centred).all(|&w| w >= 0.0));
    }

    #[test]
    fn erf_matches_tabulated_values() {
        // erf to 10 digits (Abramowitz & Stegun 1964, Table 7.1). 7.1.26 is good to 1.5e-7;
        // the rest of the tolerance is f32 rounding.
        for (x, want) in [
            (0.1, 0.112_462_916_0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (1.5, 0.966_105_146_5),
            (2.0, 0.995_322_265_0),
            (3.0, 0.999_977_909_5),
        ] {
            assert!((erf(x) - want).abs() < 3e-7, "erf({x}) = {}", erf(x));
            assert_eq!(erf(-x), -erf(x));
        }
    }

    const SUB_PX: [(f32, f32); 5] = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.13, 0.91),
        (0.99, 0.01),
        (0.3, 0.7),
    ];

    /// Sum of `halo_weight` over every pixel centre, for a star at `sub` inside pixel (0, 0).
    fn halo_sum(sub: Vec2, radius: f32) -> f32 {
        let n = radius.ceil() as i32 + 1;
        let mut sum = 0.0;
        for dy in -n..=n {
            for dx in -n..=n {
                let centre = Vec2::new(dx as f32 + 0.5, dy as f32 + 0.5);
                sum += halo_weight(centre.distance(sub), radius);
            }
        }
        sum
    }

    #[test]
    fn moffat_has_unit_energy() {
        // ∫ 2πr·I(r) dr over [0, ∞) = 1, by the midpoint rule out to 2000 px.
        let dr = 0.01;
        let e: f64 = (0..200_000)
            .map(|i| {
                let r = (i as f64 + 0.5) * dr;
                std::f64::consts::TAU * r * f64::from(moffat(r as f32)) * dr
            })
            .sum();
        assert!((e - 1.0).abs() < 1e-3, "{e}");
    }

    #[test]
    fn halo_conserves_energy_any_subpixel() {
        for radius in [MIN_RADIUS_PX, 5.0, 8.0, 20.0, MAX_RADIUS_PX] {
            for (x, y) in SUB_PX {
                let s = halo_sum(Vec2::new(x, y), radius);
                assert!((s - 1.0).abs() < 1.5e-2, "R {radius} ({x}, {y}): {s}");
            }
            // The halo ends at the footprint radius.
            assert_eq!(halo_weight(radius, radius), 0.0);
            assert_eq!(halo_weight(radius + 1.0, radius), 0.0);
        }
    }

    #[test]
    fn core_plus_halo_conserves_energy() {
        // The whole PSF of a bright star (core and halo together) sums to 1 within 0.1%.
        for radius in [MIN_RADIUS_PX, 6.0, 30.0] {
            for (x, y) in SUB_PX {
                let sub = Vec2::new(x, y);
                let core: f32 = psf_weights(sub, CORE_SIGMA_PX, radius.ceil() as i32)
                    .iter()
                    .sum();
                let h = HALO_ENERGY_FRACTION;
                let total = (1.0 - h) * core + h * halo_sum(sub, radius);
                let err = (total - 1.0).abs();
                assert!(err < 1e-3, "R {radius} ({x}, {y}): {total}");
            }
        }
    }

    #[test]
    fn halo_only_for_stars_brighter_than_1e_minus_6_lux() {
        assert_eq!(halo_fraction(2e-6), HALO_ENERGY_FRACTION);
        assert_eq!(halo_fraction(1e-6), 0.0);
        assert_eq!(halo_fraction(1e-9), 0.0);
        assert_eq!(halo_fraction(f32::NAN), 0.0);
    }

    #[test]
    fn footprint_radius_grows_with_energy_within_bounds() {
        let h = HALO_ENERGY_FRACTION;
        assert_eq!(footprint_radius_px(0.0, 0.0), MIN_RADIUS_PX);
        assert_eq!(footprint_radius_px(f32::NAN, h), MIN_RADIUS_PX);
        assert_eq!(footprint_radius_px(1e12, h), MAX_RADIUS_PX);
        for share in [0.0, h] {
            let mut last = 0.0;
            let mut energy = 1e-6;
            while energy < 1e9 {
                let r = footprint_radius_px(energy, share);
                assert!((MIN_RADIUS_PX..=MAX_RADIUS_PX).contains(&r));
                assert!(r >= last, "radius shrank at energy {energy}, halo {share}");
                last = r;
                // Past the radius, neither the core nor the halo is above the cutoff.
                let core_peak = (1.0 - share) * energy / (TAU * CORE_SIGMA_PX.powi(2));
                let core = core_peak * (-0.5 * (r / CORE_SIGMA_PX).powi(2)).exp();
                let halo = share * energy * moffat(r);
                if r < MAX_RADIUS_PX {
                    assert!(core <= VISIBILITY_CUTOFF * 1.001, "core {core} at R {r}");
                    assert!(halo <= VISIBILITY_CUTOFF * 1.001, "halo {halo} at R {r}");
                }
                energy *= 2.0;
            }
        }
        // Bright stars get a wider footprint than the 3 px minimum.
        assert!(footprint_radius_px(10.0, h) > 2.0 * MIN_RADIUS_PX);
    }
}
