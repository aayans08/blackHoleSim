//! Partial eclipses of the Sun's disk (Task B0, shared by B1 and B2): the CPU twin of
//! `sun_visible_fraction` in `shaders/eclipse.wgsl` (ShaderLib path [`crate::ECLIPSE_WGSL`]).
//! B1's probe compares the two on the GPU.

use std::f32::consts::PI;

/// Fraction of the Sun's disk visible past one occluder, both seen from the shaded point as
/// uniform disks: angular radii `sun_ang_r` and `occ_ang_r` [rad], centre separation `sep` [rad]
/// (its sign does not matter). 1 when the disks are apart or just touch, 0 when the occluder
/// covers the Sun, 1 − (r_o/r_s)² when it lies inside the Sun's disk (annular).
///
/// The covered area of a partial overlap is the lens of two circles, the sum of two circular
/// segments (Weisstein, "Circle-Circle Intersection", MathWorld):
/// A = r_s²·(α − sin α cos α) + r_o²·(β − sin β cos β), where α and β are the half-angles the
/// chord subtends at each centre, cos α = (d² + r_s² − r_o²)/(2 d r_s) and
/// cos β = (d² + r_o² − r_s²)/(2 d r_o). The result is 1 − A/(π r_s²), worked in units of r_s.
///
/// Approximations: the disks are planar (for the Sun's 0.27° from Earth, treating small
/// spherical caps as planar disks errs by about r² ≈ 2e-5 relative), and uniform (no limb
/// darkening, B1 brief "circle–circle overlap"; a limb-darkened Sun would weight a covered limb
/// less). Inputs are angles ≥ 0; a point Sun (`sun_ang_r` ≤ 0) is either hidden (its centre
/// behind the occluder) or not, and an occluder of radius ≤ 0 hides nothing.
///
/// f32 throughout, in the same order as the WGSL twin, so the two agree to f32 rounding.
pub fn sun_visible_fraction(sun_ang_r: f32, occ_ang_r: f32, sep: f32) -> f32 {
    let d = sep.abs();
    if !(occ_ang_r > 0.0) {
        return 1.0;
    }
    if !(sun_ang_r > 0.0) {
        return if d < occ_ang_r { 0.0 } else { 1.0 };
    }
    // Apart or touching; covering; inside (annular). Compared in absolute angles, so the
    // boundary cases are exact.
    if d >= sun_ang_r + occ_ang_r {
        return 1.0;
    }
    if occ_ang_r >= sun_ang_r + d {
        return 0.0;
    }
    let rho = occ_ang_r / sun_ang_r;
    if d + occ_ang_r <= sun_ang_r {
        return 1.0 - rho * rho;
    }
    // Partial: d > |r_s − r_o| ≥ 0 here.
    let delta = d / sun_ang_r;
    let cos_a = ((delta * delta + 1.0 - rho * rho) / (2.0 * delta)).clamp(-1.0, 1.0);
    let cos_b = ((delta * delta + rho * rho - 1.0) / (2.0 * delta * rho)).clamp(-1.0, 1.0);
    let seg_a = cos_a.acos() - cos_a * (1.0 - cos_a * cos_a).sqrt();
    let seg_b = cos_b.acos() - cos_b * (1.0 - cos_b * cos_b).sqrt();
    (1.0 - (seg_a + rho * rho * seg_b) / PI).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_gfx::{ShaderLib, validate_wgsl};
    use wgpu::naga;

    /// The Sun's angular radius from 1 AU [rad]: R☉/AU = 6.957e8 m / 1.495978707e11 m (IAU 2015
    /// Resolution B3 nominal solar radius; IAU 2012 Resolution B2 astronomical unit).
    const SUN_FROM_1AU: f32 = 4.650_467e-3;

    /// Visible fraction by numerical integration in f64: the covered area is
    /// ∫ 2·min(√(r_s² − x²), √(r_o² − (x − d)²)) dx over the x where both disks have a chord
    /// (both centres on the x axis), by the midpoint rule.
    fn numeric_visible(rs: f64, ro: f64, d: f64) -> f64 {
        let (lo, hi) = ((-rs).max(d - ro), rs.min(d + ro));
        if hi <= lo {
            return 1.0;
        }
        let n = 200_000;
        let dx = (hi - lo) / f64::from(n);
        let covered: f64 = (0..n)
            .map(|i| {
                let x = lo + (f64::from(i) + 0.5) * dx;
                let a = (rs * rs - x * x).max(0.0).sqrt();
                let b = (ro * ro - (x - d) * (x - d)).max(0.0).sqrt();
                2.0 * a.min(b) * dx
            })
            .sum();
        1.0 - covered / (std::f64::consts::PI * rs * rs)
    }

    #[test]
    fn eclipse_fraction_cases() {
        let rs = SUN_FROM_1AU;
        // Disjoint, and exactly touching: nothing covered.
        assert_eq!(sun_visible_fraction(rs, 0.5 * rs, 2.0 * rs), 1.0);
        assert_eq!(sun_visible_fraction(rs, 0.5 * rs, 1.5 * rs), 1.0);
        assert_eq!(sun_visible_fraction(rs, 0.3 * rs, 10.0), 1.0);
        // The occluder covers the whole disk (total eclipse), also when exactly as large.
        assert_eq!(sun_visible_fraction(rs, 2.0 * rs, 0.5 * rs), 0.0);
        assert_eq!(sun_visible_fraction(rs, 1.0e3 * rs, 0.0), 0.0);
        assert_eq!(sun_visible_fraction(rs, rs, 0.0), 0.0);
        // Centred occluder of half the radius: 1 − 1/4.
        assert!((sun_visible_fraction(rs, 0.5 * rs, 0.0) - 0.75).abs() < 1e-6);
        // Annular, off centre but inside: 1 − (r_o/r_s)².
        let annular = sun_visible_fraction(rs, 0.4 * rs, 0.5 * rs);
        assert!((annular - 0.84).abs() < 1e-6, "{annular}");
        // Only the distance matters, not the sign of `sep`.
        assert_eq!(
            sun_visible_fraction(rs, 0.7 * rs, -0.9 * rs),
            sun_visible_fraction(rs, 0.7 * rs, 0.9 * rs)
        );
        // A point Sun is either hidden or not; no occluder hides nothing.
        assert_eq!(sun_visible_fraction(0.0, rs, 0.5 * rs), 0.0);
        assert_eq!(sun_visible_fraction(0.0, rs, 1.5 * rs), 1.0);
        assert_eq!(sun_visible_fraction(rs, 0.0, 0.0), 1.0);

        // Partial overlaps against the numerical integral, ±1e-4, over occluder sizes from a
        // tenth of the Sun to three times it (the Moon is 0.97–1.07 × from the Earth) and
        // separations across the whole partial range, including near-tangent ones.
        let mut worst = 0.0_f64;
        for ratio in [0.1, 0.5, 0.97, 1.0, 1.07, 3.0] {
            let ro = ratio * rs;
            let (d_min, d_max) = ((ro - rs).abs(), ro + rs);
            for k in 0..=20 {
                let d = d_min + (d_max - d_min) * k as f32 / 20.0;
                let got = sun_visible_fraction(rs, ro, d);
                let want = numeric_visible(f64::from(rs), f64::from(ro), f64::from(d));
                assert!((0.0..=1.0).contains(&got), "ratio {ratio}, d {d}: {got}");
                let err = (f64::from(got) - want).abs();
                worst = worst.max(err);
                assert!(
                    err < 1e-4,
                    "ratio {ratio}, d/r_s {}: {got} vs {want}",
                    d / rs
                );
            }
        }
        // Monotonic: moving the occluder away never hides more of the Sun.
        let mut prev = 0.0_f32;
        for k in 0..=400 {
            let d = 2.0 * rs * k as f32 / 400.0;
            let v = sun_visible_fraction(rs, 0.97 * rs, d);
            assert!(v >= prev - 1e-6, "d/r_s {}: {v} < {prev}", d / rs);
            prev = v;
        }
        eprintln!("eclipse_fraction_cases: worst |f32 − numeric| = {worst:.2e}");
    }

    /// The WGSL twin `planet/eclipse.wgsl` validates, has the CPU twin's signature
    /// (f32, f32, f32) -> f32 and binds nothing (any pass can import it).
    #[test]
    fn shared_twin_shaders_validate() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let src = lib.compose(crate::ECLIPSE_WGSL).unwrap();
        let m = validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let (_, f) = m
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("sun_visible_fraction"))
            .expect("fn sun_visible_fraction in planet/eclipse.wgsl");
        let f32_ty = |h: naga::Handle<naga::Type>| {
            m.types[h].inner == naga::TypeInner::Scalar(naga::Scalar::F32)
        };
        assert_eq!(f.arguments.len(), 3);
        assert!(f.arguments.iter().all(|a| f32_ty(a.ty)));
        assert!(f.result.as_ref().is_some_and(|r| f32_ty(r.ty)));
        assert!(m.global_variables.iter().all(|(_, v)| v.binding.is_none()));
    }
}
