//! "Exposure on paper" with the CPU twins (plan, "Exposure on paper"; M1 Task B9 Step 5): the
//! world light of a view, pixel by pixel, through B6's meter rule.
//!
//! As the plan's table works it: resolved bodies are Lambert or lunar-Lambert with ρ from p_V
//! (A3, `rho_from_geometric_albedo`), lit by the Sun at the World's distance, with A3's
//! Mallama & Hilton phase correction per pixel (clamped at 10, as B1 does); the Sun is its mean
//! disk luminance I/(πR²) × B2's limb darkening at 550 nm; Saturn's rings are B2's I/F twin
//! (`uni_planet::rings::ring_i_over_f`) over the committed ring profile, in the planet's shadow
//! (`planet_shadow_on_ring`) where it falls. Atmospheres are left out, except Earth's (ruling
//! W-B1/B3 I4): its ground and clouds are the maps' disk composite ρ = 0.302 (A8), scaled by B3's
//! disk reflectance with the atmosphere over the bare Lambert disk at the same phase
//! (`uni_atmo::feature::disk_reflectance`), so the sky's contribution is spread over the disk.
//! Points (unresolved bodies) are sky light at star exposure and are not metered.
//!
//! The luminances go through B6's CPU meter (`Histogram::from_luminances(...).meter`,
//! `target_ev`), every pixel of the frame.

use glam::{DQuat, DVec2, DVec3, UVec2};
use uni_core::CameraView;
#[cfg(test)]
use uni_physics::photometry::{Brdf, disk_phase_function};
use uni_physics::photometry::{
    lambert_phase, planet_phase_correction, reflectance, rho_from_geometric_albedo,
};
use uni_post::histogram::Histogram;
use uni_sim::snapshot::{BodyView, Snapshot};

use crate::measure::{Cam, hit_body, hit_ellipsoid};
use crate::world::World;

/// Pixels sampled per side: every `STRIDE`-th pixel in x and y, weighted `STRIDE²` (1: every
/// pixel, so thin crescents are counted exactly).
const STRIDE: u32 = 1;
/// Earth's map composite ρ, ground and clouds (ruling W-B1: A8's physical calibration).
pub const EARTH_COMPOSITE_RHO: f64 = 0.302;
/// NAIF ids.
const EARTH: i32 = 399;
const VENUS: i32 = 299;
const SATURN: i32 = 699;

/// One view on paper.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Paper {
    /// Pixels at or above the meter's 10⁻³ cd/m², at the frame's size.
    pub lit_px: u32,
    /// B6's metered target [EV100] (`NIGHT_EV100` with too few lit pixels).
    pub target_ev: f32,
    pub ev_avg: Option<f32>,
    pub ev_hi: Option<f32>,
    /// The brightest and the 5th-percentile lit pixel, pre-exposed at `target_ev`.
    pub brightest_pre: f32,
    pub p5_pre: f32,
}

/// The luminance [cd/m²] of world light along unit `dir` from the camera (0 for sky).
pub fn world_luminance(
    snap: &Snapshot,
    dir: DVec3,
    rings: Option<&uni_data::RingProfile>,
    earth_k: f64,
) -> f64 {
    // The nearest resolved body or star surface along the ray.
    let mut best: Option<(f64, f64)> = None; // (distance, luminance)
    let mut offer = |t: f64, l: f64| {
        if best.is_none_or(|(b, _)| t < b) {
            best = Some((t, l));
        }
    };
    for s in &snap.star_surfaces {
        if let Some(t) = hit_ellipsoid(-s.rel_pos_m, dir, DQuat::IDENTITY, DVec3::splat(s.radius_m))
        {
            let n = (dir * t - s.rel_pos_m).normalize();
            let mu = (-n.dot(dir)) as f32;
            let ld = uni_planet::limb_darkening(mu).y / uni_planet::limb_darkening_mean().y;
            let mean = s.luminous_intensity_cd / (std::f64::consts::PI * s.radius_m * s.radius_m);
            offer(t, mean * f64::from(ld));
        }
    }
    for b in &snap.bodies {
        let drawn = drawn(b);
        if let Some(t) = hit_body(&drawn, dir) {
            offer(t, body_luminance(&drawn, dir, t, earth_k));
        }
    }
    let mut l = best.map_or(0.0, |(_, l)| l);
    // Saturn's rings, in front of whatever is behind them.
    if let (Some(profile), Some(s)) = (rings, snap.bodies.iter().find(|b| b.naif == Some(SATURN)))
        && let Some(ring) = s.ring
    {
        let n = ring.normal;
        let denom = dir.dot(n);
        if denom.abs() > 1e-12 {
            let t = s.rel_pos_m.dot(n) / denom;
            let q = dir * t - s.rel_pos_m;
            let r = q.length();
            let in_front = best.is_none_or(|(b, _)| t < b);
            if t > 0.0 && in_front && r >= profile.inner_m && r <= profile.outer_m {
                let sun = s.sun_rel_m.normalize();
                let alpha = uni_planet::body::ring_profile_opacity(profile, r as f32);
                let k = ((r - profile.inner_m) / (profile.outer_m - profile.inner_m)
                    * profile.samples.len() as f64) as usize;
                let c = profile.samples[k.min(profile.samples.len() - 1)];
                let rho = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                let mu0 = sun.dot(n).abs() as f32;
                let mu = denom.abs() as f32;
                let lit_face = (sun.dot(n) > 0.0) == (-dir.dot(n) > 0.0);
                let cos_scatter = (-sun.dot(-dir)) as f32;
                let inv = s.orient.conjugate();
                let shadow = uni_planet::planet_shadow_on_ring(inv * q, inv * sun, s.radii_m);
                let i_f =
                    uni_planet::rings::ring_i_over_f(rho, alpha, mu0, mu, cos_scatter, lit_face)
                        * shadow;
                let tau = -(1.0 - f64::from(alpha).min(0.9999)).ln();
                let t_view = (-tau / f64::from(mu.max(1e-3))).exp();
                l = f64::from(i_f) * s.sun_illuminance_lux / std::f64::consts::PI + t_view * l;
            }
        }
    }
    l
}

/// The ellipsoid the bodies pass draws: Venus's cloud deck, 70 km above its solid radius (B1).
fn drawn(b: &BodyView) -> BodyView {
    let mut d = b.clone();
    if b.naif == Some(VENUS) {
        d.radii_m += DVec3::splat(uni_atmo::VENUS_CLOUD_TOP_M);
    }
    d
}

/// The luminance [cd/m²] of body `b` hit at distance `t` along `dir`.
fn body_luminance(b: &BodyView, dir: DVec3, t: f64, earth_k: f64) -> f64 {
    let x = dir * t - b.rel_pos_m; // surface point relative to the centre, root axes
    let inv = b.orient.conjugate();
    let xb = inv * x;
    let normal = (b.orient * (xb / (b.radii_m * b.radii_m))).normalize();
    let sun = b.sun_rel_m.normalize();
    let mu0 = normal.dot((b.sun_rel_m - x).normalize());
    let mu = -normal.dot(dir);
    let phase = crate::measure::angle(sun, -b.rel_pos_m);
    let (rho, k) = if b.naif == Some(EARTH) {
        (EARTH_COMPOSITE_RHO, earth_k)
    } else {
        (rho_from_geometric_albedo(b.brdf, b.geometric_albedo), 1.0)
    };
    let corr = b
        .naif
        .map_or(1.0, |n| planet_phase_correction(n, b.brdf, phase))
        .min(uni_sim::snapshot::MAX_PHASE_CORRECTION);
    b.sun_illuminance_lux / std::f64::consts::PI * rho * reflectance(b.brdf, mu0, mu) * corr * k
}

/// Earth's atmosphere factor at `phase` [rad]: B3's disk reflectance of the composite under the
/// atmosphere over the bare Lambert disk's, (2/3)ρΦ_L(α), in luminance (ruling W-B1/B3 I4).
pub fn earth_atmosphere_factor(phase: f64) -> f64 {
    let Some(params) = uni_atmo::preset_for(EARTH) else {
        return 1.0;
    };
    let rho = EARTH_COMPOSITE_RHO as f32;
    let with = uni_atmo::feature::disk_reflectance(&params, glam::Vec3::splat(rho), phase, 32, 16);
    let with = f64::from(crate::scene::luminance(with));
    let bare = 2.0 / 3.0 * EARTH_COMPOSITE_RHO * lambert_phase(phase);
    if bare > 0.0 { with / bare } else { 1.0 }
}

/// The view `view` of `world` at `size` pixels on paper (module docs).
pub fn on_paper(
    world: &World,
    view: &CameraView,
    size: UVec2,
    rings: Option<&uni_data::RingProfile>,
) -> Paper {
    let snap = world.sim.snapshot(view, size);
    let cam = Cam::new(view, size);
    let earth_k = snap
        .bodies
        .iter()
        .find(|b| b.naif == Some(EARTH))
        .map_or(1.0, |b| {
            earth_atmosphere_factor(crate::measure::phase_rad(b))
        });
    let w = STRIDE * STRIDE;
    let mut lums = Vec::new();
    for y in (0..size.y).step_by(STRIDE as usize) {
        for x in (0..size.x).step_by(STRIDE as usize) {
            let p = DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5);
            let l = world_luminance(&snap, cam.ray(p), rings, earth_k);
            if l > 0.0 {
                for _ in 0..w {
                    lums.push(l as f32);
                }
            }
        }
    }
    let total = size.x * size.y;
    let meter = Histogram::from_luminances(lums.iter().copied()).meter(total);
    let target = uni_post::target_ev(meter, 0.0);
    let pre = uni_gfx::exposure_scale(target);
    let mut lit: Vec<f32> = lums
        .into_iter()
        .filter(|l| *l >= uni_post::METER_MIN_CD_M2)
        .collect();
    lit.sort_by(f32::total_cmp);
    let p5 = lit.get(lit.len() / 20).copied().unwrap_or(0.0);
    Paper {
        lit_px: lit.len() as u32,
        target_ev: target,
        ev_avg: meter.map(|m| m.ev_avg),
        ev_hi: meter.map(|m| m.ev_hi),
        brightest_pre: lit.last().copied().unwrap_or(0.0) * pre,
        p5_pre: p5 * pre,
    }
}

/// The p_V point flux [lx] of body `b` (A4's point path: ρ·w·E·(R/d)²·Φ(α)·C(α)).
#[cfg(test)]
pub fn point_lux(b: &BodyView) -> f64 {
    let phase = crate::measure::phase_rad(b);
    let weight = match b.brdf {
        Brdf::Lambert => 2.0 / 3.0,
        Brdf::LunarLambert { l } => l + 2.0 * (1.0 - l) / 3.0,
    };
    let rho = rho_from_geometric_albedo(b.brdf, b.geometric_albedo);
    let corr = b
        .naif
        .map_or(1.0, |n| planet_phase_correction(n, b.brdf, phase))
        .min(uni_sim::snapshot::MAX_PHASE_CORRECTION);
    let ratio = b.mean_radius_m / b.rel_pos_m.length();
    rho * weight * b.sun_illuminance_lux * ratio * ratio * disk_phase_function(b.brdf, phase) * corr
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare Lambert sphere on paper integrates to the point path's flux (the per-pixel model
    /// and A4's point path agree; Review Focus 4), seen from 118 radii: the point path assumes a
    /// distant observer (from 12 radii the disk sum runs 2 % over it, a finite-distance effect).
    #[test]
    fn paper_disk_flux_matches_the_point_path() {
        let mut world = World::bare(uni_core::Epoch::J2000);
        world.set_epoch(uni_core::Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap());
        let mars = world.position_of("Mars").unwrap();
        let sun = world
            .position_of("Sun")
            .unwrap()
            .delta_from(mars)
            .normalize();
        let (e1, _) = {
            let h = DVec3::Z - sun * sun.z;
            (h.normalize(), ())
        };
        let c = sun * 0.5 + e1 * 0.75_f64.sqrt(); // phase 60°
        let cam_pos = mars.offset(c * 4.0e8);
        let fwd = -c;
        let right = fwd.cross(DVec3::Z).normalize();
        let up = right.cross(fwd);
        let view = CameraView {
            pos: cam_pos,
            orient: DQuat::from_mat3(&glam::DMat3::from_cols(right, up, -fwd)),
            fov_y_rad: 0.04,
            velocity_mps: DVec3::ZERO,
        };
        let size = UVec2::new(640, 480);
        let snap = world.sim.snapshot(&view, size);
        let b = snap.bodies.iter().find(|b| b.naif == Some(499)).unwrap();
        let cam = Cam::new(&view, size);
        let mut lux = 0.0;
        for p in cam.pixels() {
            lux += world_luminance(&snap, cam.ray(p), None, 1.0) * cam.pixel_solid_angle(p);
        }
        let point = point_lux(b);
        assert!((lux / point - 1.0).abs() < 0.02, "{lux} vs {point}");
    }
}
