//! Schwarzschild null geodesics in Binet/Cartesian form, units of M, hole at the origin (M0 Task 9).
//!
//! Units: G = c = M = 1, so the horizon is at r = 2 (r_s = 2M, Schwarzschild 1916). The ray
//! moves as a particle in flat Cartesian 3-space, with Schwarzschild r, θ, φ read as Euclidean
//! coordinates, under
//!
//! ```text
//! d²x/dλ² = −3·h²·x / r⁵,   h = |x × dx/dλ| (conserved)
//! ```
//!
//! This is the Binet orbit equation u'' + u = 3M·u² (u = 1/r, ' = d/dφ; the Schwarzschild
//! null-geodesic orbit equation, Chandrasekhar 1983, *The Mathematical Theory of Black Holes*,
//! ch. 3; physics research §1.1) written in Cartesian form; it reproduces the null geodesic's
//! spatial path exactly. λ is the affine parameter: r'' = h²/r³ − 3h²/r⁴ and
//! r'² = E² − (1 − 2/r)·h²/r² match the Schwarzschild radial equation, so `v = dx/dλ` is never
//! renormalised (that would change h) and b = h/E.
//!
//! The reference integrator and step rule are `deflect()` in `docs/research/m0-golden-values.py`;
//! `shaders/binet.wgsl` is the GPU twin of [`trace`] with the same RK4 and step rule.

use glam::DVec3;

/// Capture margin ε: a ray is captured once r < 2(1 + ε) (as in m0-golden-values.py).
pub const CAPTURE_EPS: f64 = 1e-6;
/// Capture radius 2(1 + ε) [M]: the horizon r = 2M (Schwarzschild 1916) plus the margin.
pub const CAPTURE_RADIUS: f64 = 2.0 * (1.0 + CAPTURE_EPS);
/// Critical impact parameter b_c = 3√3 M, the shadow edge (photon sphere r = 3M; Darwin 1959).
pub const CRITICAL_IMPACT: f64 = 5.196_152_422_706_632;

/// Start of the ray in [`deflection_for_impact`]: x = −1e4 M, as in m0-golden-values.py.
const DEFLECTION_START_X: f64 = -1e4;
/// Step budget for [`deflection_for_impact`]: ample for the ~5000 steps a 2e4 M path needs.
const DEFLECTION_MAX_STEPS: u32 = 1_000_000;

/// How a traced ray ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RayEnd {
    /// Reached r < 2(1 + ε): the ray came from the hole (black).
    Captured,
    /// Left r_far moving outwards; `dir` is the unit direction of dx/dλ there.
    Escaped { dir: DVec3 },
    /// Neither within the step budget.
    MaxSteps,
}

/// Affine step dλ = clamp(0.02·r, 1e-3, 5.0) [M]: proportional to r (physics research §1.3),
/// floored near the hole and capped in the far field. `binet.wgsl` uses the same rule.
pub(crate) fn step_size(r: f64) -> f64 {
    (0.02 * r).clamp(1e-3, 5.0)
}

/// a = −3·h²·x/r⁵ with h = |x × v| taken from the current state, as in the reference
/// `deflect()`. Written as (h²/r²)/r³ so the f32 twin does not overflow at large r.
fn accel(x: DVec3, v: DVec3) -> DVec3 {
    let r2 = x.length_squared();
    let r = r2.sqrt();
    let h2 = x.cross(v).length_squared();
    x * (-3.0 * (h2 / r2) / (r2 * r))
}

/// One classical 4th-order Runge–Kutta step of (x, v) by dλ = `dl`.
fn rk4_step(x: DVec3, v: DVec3, dl: f64) -> (DVec3, DVec3) {
    let k1x = v;
    let k1v = accel(x, v);
    let k2x = v + k1v * (0.5 * dl);
    let k2v = accel(x + k1x * (0.5 * dl), k2x);
    let k3x = v + k2v * (0.5 * dl);
    let k3v = accel(x + k2x * (0.5 * dl), k3x);
    let k4x = v + k3v * dl;
    let k4v = accel(x + k3x * dl, k4x);
    (
        x + (k1x + 2.0 * k2x + 2.0 * k3x + k4x) * (dl / 6.0),
        v + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) * (dl / 6.0),
    )
}

/// Integrates the ray from `x0` with tangent `v0 = dx/dλ` (hole at the origin, units of M).
///
/// Before each step: captured if r < 2(1 + ε); escaped if r > `r_far` and the ray is moving
/// outwards (x·v > 0). After `max_steps` steps without either, [`RayEnd::MaxSteps`]. A start
/// inside the capture radius (including the centre) is captured at once.
pub fn trace(x0: DVec3, v0: DVec3, r_far: f64, max_steps: u32) -> RayEnd {
    let (mut x, mut v) = (x0, v0);
    for i in 0..=max_steps {
        let r = x.length();
        if r < CAPTURE_RADIUS {
            return RayEnd::Captured;
        }
        if r > r_far && x.dot(v) > 0.0 {
            return RayEnd::Escaped { dir: v.normalize() };
        }
        if i == max_steps {
            break;
        }
        (x, v) = rk4_step(x, v, step_size(r));
    }
    RayEnd::MaxSteps
}

/// Total deflection angle [rad] of a ray sent along +x from x = −1e4 M with impact parameter
/// `b` [M], measured when it leaves r = 1e4 M (the reference `deflect()`). In this form the
/// transverse pull falls off as b³/r⁵, so cutting the path at D = 1e4 M drops ≈ 1.5·b³/D⁴ of
/// the asymptotic angle (1.5e-7 rad at b = 1000 M). `None` if the ray is captured
/// (b < b_c = 3√3), or (not expected) if it does not finish within the step budget.
pub fn deflection_for_impact(b: f64) -> Option<f64> {
    let x0 = DVec3::new(DEFLECTION_START_X, b, 0.0);
    match trace(x0, DVec3::X, -DEFLECTION_START_X, DEFLECTION_MAX_STEPS) {
        RayEnd::Escaped { dir } => Some(f64::atan2(-dir.y, dir.x)),
        RayEnd::Captured | RayEnd::MaxSteps => None,
    }
}

/// Initial tangent v = dx/dλ, with E = 1, for a ray leaving a static observer at `cam` in the
/// unit local direction `local_dir` (its length is ignored).
///
/// The radial part is kept and the tangential part is divided by √(1 − 2/r), which gives
/// b = h/E = r·sin δ/√(1 − 2/r), δ being the angle from the inward radial direction
/// (static-observer aberration; Synge 1966, MNRAS 131, 463; CLAUDE.md §6).
/// `None` if r ≤ 2 (no static observer exists on or inside the horizon) or the input is
/// degenerate (non-finite, or a zero direction).
pub fn static_observer_ray(cam: DVec3, local_dir: DVec3) -> Option<DVec3> {
    let r = cam.length();
    if !(r > 2.0 && r.is_finite()) {
        return None;
    }
    let d = local_dir.try_normalize()?;
    let n = cam / r;
    let radial = n * d.dot(n);
    Some(radial + (d - radial) / (1.0 - 2.0 / r).sqrt())
}

/// Angular radius [rad] of the shadow seen by a static observer at `r_cam` [M], measured from
/// the direction to the hole: sin α = (3√3/r)·√(1 − 2/r) (Synge 1966, MNRAS 131, 463).
///
/// Inside the photon sphere (2 < r < 3) the shadow covers more than a hemisphere, so α is the
/// obtuse solution π − asin(…); at r = 3 it is exactly π/2. On or inside the horizon (r ≤ 2)
/// the whole sky is dark: π.
///
/// Evaluated as α = atan2(√(27(r − 2)), (r − 3)·√(r + 6)), using
/// cos²α = 1 − 27(r − 2)/r³ = (r − 3)²(r + 6)/r³ with the sign of r − 3. This is the same
/// angle as the asin form, picks the obtuse branch by itself and stays accurate near r = 3,
/// where asin loses half its digits.
pub fn shadow_angular_radius(r_cam: f64) -> f64 {
    if !(r_cam > 2.0) {
        return std::f64::consts::PI;
    }
    let sin_part = (27.0 * (r_cam - 2.0)).sqrt();
    let cos_part = (r_cam - 3.0) * (r_cam + 6.0).sqrt();
    sin_part.atan2(cos_part)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binet;
    use std::f64::consts::{FRAC_PI_2, PI};

    // Golden values: docs/research/m0-golden-values.py (deflect(), b in M, x0 = −1e4).
    #[test]
    fn deflection_b10m() {
        assert!((binet::deflection_for_impact(10.0).unwrap() - 0.59040).abs() < 2e-3);
    } // b = 5 r_s
    #[test]
    fn deflection_b20m() {
        assert!((binet::deflection_for_impact(20.0).unwrap() - 0.23614).abs() < 2e-3);
    } // b = 10 r_s
    #[test]
    fn deflection_b100m() {
        assert!((binet::deflection_for_impact(100.0).unwrap() - 0.04122).abs() < 1e-3);
    } // b = 50 r_s
    #[test]
    fn critical_impact() {
        assert!(binet::deflection_for_impact(5.19).is_none());
        assert!(binet::deflection_for_impact(5.21).is_some());
    } // b_c = 3√3 = 5.196152
    #[test]
    fn shadow_at_30m() {
        assert!((binet::shadow_angular_radius(30.0).to_degrees() - 9.6327).abs() < 1e-3);
    }
    #[test]
    fn static_observer_ray_inside_horizon_is_none() {
        assert!(binet::static_observer_ray(DVec3::new(1.5, 0.0, 0.0), DVec3::X).is_none());
    } // Review Focus 2
    #[test]
    fn trace_from_inside_is_captured() {
        assert_eq!(
            binet::trace(DVec3::new(1.0, 0.0, 0.0), DVec3::Y, 1e3, 100),
            binet::RayEnd::Captured
        );
    } // Review Focus 2

    // Review Focus 2: exactly on the horizon and at the centre are also "inside".
    #[test]
    fn horizon_and_centre_are_inside() {
        assert!(static_observer_ray(DVec3::new(0.0, 2.0, 0.0), DVec3::X).is_none());
        assert!(static_observer_ray(DVec3::ZERO, DVec3::X).is_none());
        assert_eq!(trace(DVec3::ZERO, DVec3::X, 1e3, 100), RayEnd::Captured);
        assert_eq!(shadow_angular_radius(2.0), PI);
        assert_eq!(shadow_angular_radius(0.0), PI);
    }

    /// Weak-field series α = 4(M/b) + (15π/4)(M/b)² + (128/3)(M/b)³ + O((M/b)⁴)
    /// (Keeton & Petters 2005, Phys. Rev. D 72, 104006, Schwarzschild coefficients). The
    /// x = ±1e4 M cut-off costs ≈ 1.5e-7 rad at b = 1000 M (see `deflection_for_impact`).
    #[test]
    fn deflection_matches_weak_field_series() {
        for b in [300.0, 1000.0] {
            let u: f64 = 1.0 / b;
            let series = 4.0 * u + 15.0 * PI / 4.0 * u * u + 128.0 / 3.0 * u.powi(3);
            let got = deflection_for_impact(b).unwrap();
            assert!((got - series).abs() < 2e-7, "b = {b}: {got} vs {series}");
        }
    }

    /// The static observer's shadow edge (Synge 1966) separates captured from escaping rays.
    #[test]
    fn shadow_edge_matches_trace() {
        let r = 30.0;
        let cam = DVec3::new(r, 0.0, 0.0);
        let alpha = shadow_angular_radius(r);
        // δ is measured from the inward radial direction (−x), turning towards +y.
        let ray = |delta: f64| {
            let local = DVec3::new(-delta.cos(), delta.sin(), 0.0);
            trace(cam, static_observer_ray(cam, local).unwrap(), 1e3, 2000)
        };
        assert_eq!(ray(alpha * (1.0 - 2e-3)), RayEnd::Captured);
        assert!(matches!(ray(alpha * (1.0 + 2e-3)), RayEnd::Escaped { .. }));
    }

    #[test]
    fn shadow_inside_photon_sphere_exceeds_a_hemisphere() {
        assert!((shadow_angular_radius(3.0) - FRAC_PI_2).abs() < 1e-12);
        let a = shadow_angular_radius(2.5);
        assert!(a > FRAC_PI_2 && a < PI, "{a}");
        // Monotonic: the shadow shrinks as the observer moves out.
        let mut last = PI;
        for i in 0..200 {
            let r = 2.0 + 1e-3 + 0.25 * f64::from(i);
            let a = shadow_angular_radius(r);
            assert!(a <= last, "r = {r}");
            last = a;
        }
    }

    /// b = h/E = r·sin δ / √(1 − 2/r) for a static camera (CLAUDE.md §6): only the tangential
    /// part is scaled, and E = 1.
    #[test]
    fn static_observer_ray_scales_only_the_tangential_part() {
        let r = 30.0;
        let cam = DVec3::new(0.0, 0.0, r);
        let delta = 0.3_f64; // from the inward direction (−z) towards +x
        let local = DVec3::new(delta.sin(), 0.0, -delta.cos());
        let v = static_observer_ray(cam, local * 7.0).unwrap(); // length of local_dir is ignored
        let lapse = (1.0 - 2.0 / r).sqrt();
        assert!(
            (v.z + delta.cos()).abs() < 1e-15,
            "radial part changed: {v}"
        );
        assert!((v.x - delta.sin() / lapse).abs() < 1e-15, "{v}");
        let b = cam.cross(v).length();
        assert!((b - r * delta.sin() / lapse).abs() < 1e-12);
        // E² = (dr/dλ)² + (1 − 2/r)·h²/r² = 1 (Schwarzschild radial null equation).
        let e2 = (v.dot(cam) / r).powi(2) + (1.0 - 2.0 / r) * b * b / (r * r);
        assert!((e2 - 1.0).abs() < 1e-14, "{e2}");
    }

    #[test]
    fn radial_rays_go_straight() {
        let cam = DVec3::new(0.0, -40.0, 0.0);
        let out = static_observer_ray(cam, DVec3::NEG_Y).unwrap();
        match trace(cam, out, 1e3, 2000) {
            RayEnd::Escaped { dir } => assert!((dir - DVec3::NEG_Y).length() < 1e-12, "{dir}"),
            other => panic!("{other:?}"),
        }
        let inward = static_observer_ray(cam, DVec3::Y).unwrap();
        assert_eq!(trace(cam, inward, 1e3, 2000), RayEnd::Captured);
    }

    #[test]
    fn step_budget_is_reported() {
        // A ray far outside r_far moving inwards cannot escape in 3 steps.
        let end = trace(DVec3::new(-500.0, 10.0, 0.0), DVec3::X, 100.0, 3);
        assert_eq!(end, RayEnd::MaxSteps);
    }

    /// h = |x × v| and the null condition E² = (dr/dλ)² + (1 − 2/r)·h²/r² are constants of
    /// the motion; RK4 at this step rule keeps both through a strong-field pass.
    #[test]
    fn integrator_conserves_h_and_energy() {
        let invariants = |x: DVec3, v: DVec3| {
            let r = x.length();
            let h = x.cross(v).length();
            let e2 = (v.dot(x) / r).powi(2) + (1.0 - 2.0 / r) * h * h / (r * r);
            (h, e2)
        };
        let (mut x, mut v) = (DVec3::new(-200.0, 8.0, 3.0), DVec3::X);
        let (h0, e0) = invariants(x, v);
        let mut r_min = f64::INFINITY;
        for _ in 0..1500 {
            (x, v) = rk4_step(x, v, step_size(x.length()));
            r_min = r_min.min(x.length());
        }
        assert!(
            r_min < 10.0,
            "the test ray should pass close to the hole: {r_min}"
        );
        let (h1, e1) = invariants(x, v);
        assert!((h1 / h0 - 1.0).abs() < 1e-7, "h drift {}", h1 / h0 - 1.0);
        assert!((e1 / e0 - 1.0).abs() < 1e-7, "E² drift {}", e1 / e0 - 1.0);
    }

    #[test]
    fn step_rule() {
        assert_eq!(step_size(1.0), 0.02);
        assert_eq!(step_size(1e-3), 1e-3);
        assert_eq!(step_size(1e6), 5.0);
    }
}
