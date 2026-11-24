//! "Exposure on paper" with the CPU twins (plan, "Exposure on paper"; M1 Task B9 Step 5): the
//! world light of a view, pixel by pixel, through B6's meter rule.
//!
//! As the plan's table works it: resolved bodies are Lambert or lunar-Lambert with ρ from p_V
//! (A3, `rho_from_geometric_albedo`), lit by the Sun at the World's distance, with A3's
//! Mallama & Hilton phase correction per pixel (clamped at 10, as B1 does); the Sun is its mean
//! disk luminance I/(πR²) × B2's limb darkening at 550 nm; Saturn's rings are B2's I/F twin
//! (`uni_planet::rings::ring_i_over_f`) over the committed ring profile, in the planet's shadow
//! (`planet_shadow_on_ring`) where it falls. Atmospheres are left out, except Earth's and
//! Venus's:
//! - Earth (ruling W-B1/B3 I4): its ground and clouds are the maps' disk composite ρ = 0.302
//!   (A8), scaled by B3's disk reflectance with the atmosphere over the bare Lambert disk at the
//!   same phase (`uni_atmo::feature::disk_reflectance`), so the sky's contribution is spread
//!   over the disk.
//! - Venus (lead ruling "row 13 Venus haze", task B9g), pixel by pixel as the bodies and
//!   atmosphere passes draw it ([`VenusAtmo`]): the cloud deck at the map's ρ̄ 0.95 under A3's C,
//!   lit through B3's haze (`ground_lighting`: sun transmittance and skylight), seen through it
//!   (`segment_radiance`: view transmittance), plus the haze's in-scatter, also above the limb
//!   and past the terminator. Without it the paper put `venus-crescent` at EV 18.93 against the
//!   capture's 17.25: at phase 155° the deck is seen and lit at grazing angles through τ ≈ 0.36
//!   of haze, and the haze's dim twilight glow beyond the hard crescent (≥ 10⁻³ cd/m², lit to
//!   the meter) fills the 50th–95th percentile band the meter averages.
//!
//! Comets (Task C6, rows 20–21): each listed comet's coma is C1's drawn Haser profile
//! (`uni_planet::comet::coma::coma_drawn_brightness` with `coma_intensity_cd`, softened over a
//! pixel at the nucleus), added to the pixel's world light. The dust tails are the comet pass's
//! CPU twin (`uni_planet::dust_tail_luminance`, lead ruling C1L-b): the same comets, splats, cap
//! and floor as a capture's first frame, without the visibility cull and occlusion. The ion
//! tail (≈ 0.02 × the coma's light along a 10⁵ km ribbon, C1L fix round 2) and the km-sized
//! nucleus are left out.
//!
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
/// Venus's cloud deck reflectance as the bodies pass draws it: the map's disk average as stored
/// (`data/baked/maps/index.ron`, `venus_cloud_tops`: "disk average as stored 0.9500"; ruling
/// W-B3 keeps it), here uniform and grey (the map's pattern has a 1.9 % disk rms, stored
/// neutral).
pub const VENUS_DECK_RHO: f64 = 0.95;
/// First μ_s of [`VenusAtmo`]'s skylight table: below it the haze above the deck is in the
/// planet's shadow (sunlight grazes the shell top at μ_s = −√(1 − (r_b/r_t)²) = −0.139).
const SKY_MU_MIN: f64 = -0.25;
/// Step of the skylight table in μ_s (the GPU's irradiance LUT has 64 columns over [−1, 1]).
const SKY_MU_STEP: f64 = 1.0 / 32.0;
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

/// B3's atmosphere over Venus's cloud deck, for the paper (module docs): the preset and the
/// deck's skylight per lux of sunlight, tabulated in μ_s from [`SKY_MU_MIN`] to 1 (the CPU
/// twin `uni_atmo::sky_irradiance`, 256 view marches each).
pub struct VenusAtmo {
    params: uni_atmo::AtmosphereParams,
    sky: Vec<DVec3>,
    /// The Sun's colour, luminance 1 (a CIE integration: computed once, review minor 2).
    colour: DVec3,
}

impl VenusAtmo {
    /// The shared table, built on first use (about 40 × 256 marches).
    pub fn get() -> Option<&'static VenusAtmo> {
        static ATMO: std::sync::OnceLock<Option<VenusAtmo>> = std::sync::OnceLock::new();
        ATMO.get_or_init(|| {
            let params = uni_atmo::preset_for(VENUS)?;
            let r = params.bottom_radius_m;
            let n = ((1.0 - SKY_MU_MIN) / SKY_MU_STEP).round() as usize + 1;
            let sky = (0..n)
                .map(|k| {
                    let mu = SKY_MU_MIN + k as f64 * SKY_MU_STEP;
                    uni_atmo::sky_irradiance(&params, r, mu).as_dvec3()
                })
                .collect();
            let c = uni_core::blackbody_linear_srgb(uni_core::TEFF_SUN_K).as_dvec3();
            let colour = c / f64::from(crate::scene::luminance(c.as_vec3()));
            Some(VenusAtmo {
                params,
                sky,
                colour,
            })
        })
        .as_ref()
    }

    /// The deck's (the preset's bottom) and the haze shell's top radius [m].
    pub fn radii_m(&self) -> (f64, f64) {
        (self.params.bottom_radius_m, self.params.top_radius_m)
    }

    /// Skylight on the deck per lux of sunlight at μ_s, linear in the table; 0 below it.
    fn sky_at(&self, mu_s: f64) -> DVec3 {
        let x = (mu_s - SKY_MU_MIN) / SKY_MU_STEP;
        if !(x >= 0.0) {
            return DVec3::ZERO;
        }
        let last = self.sky.len() - 1;
        let k = (x.floor() as usize).min(last);
        let f = (x - k as f64).min(1.0);
        let next = self.sky[(k + 1).min(last)];
        self.sky[k] * (1.0 - f) + next * f
    }

    /// The luminance [cd/m²] along unit `dir` from the camera through Venus's haze (body `b`,
    /// drawn at the deck, the preset's bottom radius) and the distance to what it reaches (the
    /// deck, else the shell's nearest point), or `None` where the ray misses the shell. As the
    /// passes draw it: the deck ρ·E·C·(μ₀·T_sun + skylight)/π (body.wgsl, A3's C, B3's ground
    /// lighting at r = the deck, μ_s about the normal) times the view transmittance, plus the
    /// in-scatter × E (atmosphere_common.wgsl: no C), each with the Sun's colour.
    pub fn luminance(&self, b: &BodyView, dir: DVec3) -> Option<(f64, f64)> {
        self.ray(b, dir).map(|r| (r.luminance, r.dist_m))
    }

    /// [`VenusAtmo::luminance`] with the sky's share: what the band and stars behind the ray
    /// keep, the view transmittance's Rec. 709 luminance (the passes multiply sky light by
    /// 1 − g, g = 1 − luminance(T)), 0 where the ray meets the deck (nothing is drawn behind a
    /// body).
    pub fn ray(&self, b: &BodyView, dir: DVec3) -> Option<HazeRay> {
        let p = &self.params;
        let ro = -b.rel_pos_m;
        let top = hit_ellipsoid(ro, dir, DQuat::IDENTITY, DVec3::splat(p.top_radius_m))?;
        let deck = hit_ellipsoid(ro, dir, DQuat::IDENTITY, DVec3::splat(p.bottom_radius_m));
        let sun = b.sun_rel_m.normalize();
        let (insc, trans) =
            uni_atmo::segment_radiance(p, ro, dir, deck.unwrap_or(f64::INFINITY), sun);
        let colour = self.colour;
        let e = b.sun_illuminance_lux;
        let mut l = insc.as_dvec3() * colour * e;
        if let Some(t) = deck {
            let x = ro + dir * t;
            let n = x.normalize();
            let mu0 = n.dot((b.sun_rel_m - x).normalize());
            let sun_t = uni_atmo::transmittance(p, p.bottom_radius_m, mu0).as_dvec3();
            let corr = planet_phase_correction(VENUS, b.brdf, crate::measure::phase_rad(b))
                .min(uni_sim::snapshot::MAX_PHASE_CORRECTION);
            let refl = reflectance(b.brdf, mu0, (-n.dot(dir)).max(1.0e-4));
            let ground = colour
                * (VENUS_DECK_RHO * e * corr / std::f64::consts::PI)
                * (sun_t * refl + self.sky_at(mu0));
            l += trans.as_dvec3() * ground;
        }
        let lum = |c: DVec3| 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
        Some(HazeRay {
            luminance: lum(l),
            dist_m: deck.unwrap_or(top),
            sky_t: if deck.is_some() {
                0.0
            } else {
                lum(trans.as_dvec3())
            },
        })
    }
}

/// One ray through Venus's haze ([`VenusAtmo::ray`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HazeRay {
    /// World light along the ray [cd/m²] (Rec. 709 luminance).
    pub luminance: f64,
    /// Distance to the deck, or to the shell where the ray misses the deck [m].
    pub dist_m: f64,
    /// The fraction of the sky behind that the passes keep (0 on the deck).
    pub sky_t: f64,
}

/// The luminance [cd/m²] of world light along unit `dir` from the camera (0 for sky). With
/// `venus`, Venus is drawn with B3's haze ([`VenusAtmo::luminance`]); without, as a bare
/// Lambert deck with ρ from p_V (the plan's row).
/// The comae of `snap.comets` along the ray `dir` [cd/m²] (module docs): at the ray's
/// perpendicular distance from each nucleus in front of the camera, with one pixel
/// (`pixel_angle` [rad] across) of softening at the nucleus's distance.
pub fn coma_luminance(snap: &Snapshot, dir: DVec3, pixel_angle: f64) -> f64 {
    snap.comets
        .iter()
        .filter(|c| c.nucleus_rel_m.dot(dir) > 0.0)
        .map(|c| {
            let d = c.nucleus_rel_m.length();
            let rho = c.nucleus_rel_m.cross(dir).length();
            // The radius of a disk with one pixel's area at the nucleus (C1's softening).
            let a = d * pixel_angle / std::f64::consts::PI.sqrt();
            uni_planet::comet::coma::coma_drawn_brightness(
                rho,
                uni_planet::comet::coma::coma_intensity_cd(c),
                uni_planet::comet::coma::coma_scale_m(c.r_au),
                a,
            )
        })
        .filter(|l| l.is_finite())
        .sum()
}

pub fn world_luminance(
    snap: &Snapshot,
    dir: DVec3,
    rings: Option<&uni_data::RingProfile>,
    earth_k: f64,
    venus: Option<&VenusAtmo>,
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
        if b.naif == Some(VENUS)
            && let Some(atmo) = venus
        {
            if let Some((l, t)) = atmo.luminance(b, dir) {
                offer(t, l);
            }
            continue;
        }
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
    on_paper_with(world, view, size, rings, true)
}

/// [`on_paper`], with Venus's haze (`venus_haze`, the model the passes draw) or as the plan's
/// bare Lambert deck (for comparison: the "row 13 Venus" finding).
pub fn on_paper_with(
    world: &World,
    view: &CameraView,
    size: UVec2,
    rings: Option<&uni_data::RingProfile>,
    venus_haze: bool,
) -> Paper {
    on_paper_model(world, view, size, rings, venus_haze, HDR_SUBNORMALS)
}

/// [`on_paper_with`] with the `hdr` target's f16 subnormals flushed or kept (for comparison:
/// B9g fix round 1).
pub fn on_paper_model(
    world: &World,
    view: &CameraView,
    size: UVec2,
    rings: Option<&uni_data::RingProfile>,
    venus_haze: bool,
    subnormals: Subnormals,
) -> Paper {
    let snap = world.sim.snapshot(view, size);
    let venus = if venus_haze && snap.bodies.iter().any(|b| b.naif == Some(VENUS)) {
        VenusAtmo::get()
    } else {
        None
    };
    let cam = Cam::new(view, size);
    let earth_k = snap
        .bodies
        .iter()
        .find(|b| b.naif == Some(EARTH))
        .map_or(1.0, |b| {
            earth_atmosphere_factor(crate::measure::phase_rad(b))
        });
    // The dust tails at every pixel (lead ruling C1L-b).
    let dust = uni_planet::dust_tail_luminance(&snap.comets, world.epoch(), view, size);
    let w = STRIDE * STRIDE;
    let mut lums = Vec::new();
    for y in (0..size.y).step_by(STRIDE as usize) {
        for x in (0..size.x).step_by(STRIDE as usize) {
            let p = DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5);
            let ray = cam.ray(p);
            let l = world_luminance(&snap, ray, rings, earth_k, venus)
                + coma_luminance(&snap, ray, cam.pixel_angle())
                + dust.at(x, y);
            if l > 0.0 {
                for _ in 0..w {
                    lums.push(l as f32);
                }
            }
        }
    }
    let total = size.x * size.y;
    let (meter, target, pre) = settled_meter(&lums, total, subnormals);
    let mut lit: Vec<f32> = lums
        .into_iter()
        .map(|l| through_f16(l, pre, subnormals))
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

/// f16's smallest subnormal, 2⁻²⁴, and smallest normal, 2⁻¹⁴ (IEEE 754-2008 binary16).
const F16_MIN_SUBNORMAL: f64 = 5.960_464_477_539_063e-8;
const F16_MIN_NORMAL: f64 = 6.103_515_625e-5;
/// Most fixed-point steps of [`settled_meter`] (it settles in 2–3).
const SETTLE_STEPS: usize = 8;

/// How the `hdr` target (Rgba16Float) stores a value below f16's smallest normal, 2⁻¹⁴.
///
/// Metal does not promise to keep f16 subnormals when it stores a render target (the contract
/// `uni_atmo::feature::STAR_SHARE_MIN` and uni-planet's twin are built on). B9g fix round 1
/// (review minor 1) computed both at `venus-crescent`, whose dim haze glow lies in that range:
/// - flushed: EV 18.32, 3 568 lit px;
/// - kept: EV 17.00, 5 436 lit px;
/// - the capture (6cd81e9, the meter on the GPU): EV 17.25, 5 180 lit px.
///
/// The capture's meter saw 1 600 lit pixels more than a flushing target could hold (every
/// pixel between 0.011 and 11 cd/m² at EV 17.25 is an f16 subnormal), so on the M3 the `hdr`
/// target keeps them: the paper models [`Subnormals::Keep`]. The other contract (never rely on
/// a stored subnormal, `STAR_SHARE_MIN`) stays right for code that must work on any GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subnormals {
    /// Stored as 0 (the comparison in `intent_13_venus_crescent`).
    #[cfg_attr(not(test), allow(dead_code))]
    Flush,
    /// Kept as multiples of 2⁻²⁴ (round to nearest), the IEEE 754 binary16 behaviour.
    Keep,
}

/// The paper's model of the `hdr` target (see [`Subnormals`]).
pub const HDR_SUBNORMALS: Subnormals = Subnormals::Keep;

/// Luminance `l` [cd/m²] as the meter reads it back from the `hdr` target, which stores
/// l·`pre` in f16: below 2⁻¹⁴ a value is flushed to 0 ([`Subnormals::Flush`]) or kept as a
/// multiple of 2⁻²⁴ ([`Subnormals::Keep`]). At EV 17 the meter then sees nothing below
/// 11 cd/m² (flushed) or 0.011 cd/m² (kept), against its own 10⁻³ lit threshold, which is why
/// Venus's dimmest haze pixels drop out of the capture's meter (task B9g). The normal range's
/// 2⁻¹¹ relative rounding is left out (it moves a pixel by 0.0007 EV).
pub fn through_f16(l: f32, pre: f32, subnormals: Subnormals) -> f32 {
    let x = f64::from(l) * f64::from(pre);
    let q = match subnormals {
        _ if x >= F16_MIN_NORMAL => x,
        Subnormals::Flush => 0.0,
        Subnormals::Keep => (x / F16_MIN_SUBNORMAL).round() * F16_MIN_SUBNORMAL,
    };
    (q / f64::from(pre)) as f32
}

/// The meter's reading once auto exposure has settled: the frame is stored at the pre-exposure
/// of the EV it meters ([`through_f16`]), so the EV is the fixed point of meter → EV → storage,
/// found by iteration from the unquantised reading (what the capture's settle loop does on the
/// GPU). Returns the meter, its target [EV100] and that target's pre-exposure.
fn settled_meter(
    lums: &[f32],
    total: u32,
    subnormals: Subnormals,
) -> (Option<uni_post::histogram::Meter>, f32, f32) {
    let mut meter = Histogram::from_luminances(lums.iter().copied()).meter(total);
    let mut ev = uni_post::target_ev(meter, 0.0);
    let dimmest = lums.iter().copied().fold(f32::INFINITY, f32::min);
    for _ in 0..SETTLE_STEPS {
        let pre = uni_gfx::exposure_scale(ev);
        if f64::from(dimmest) * f64::from(pre) >= F16_MIN_NORMAL {
            // Every pixel is stored as a normal f16: nothing to quantise.
            break;
        }
        meter = Histogram::from_luminances(lums.iter().map(|&l| through_f16(l, pre, subnormals)))
            .meter(total);
        let next = uni_post::target_ev(meter, 0.0);
        let settled = (next - ev).abs() < 1.0e-3;
        ev = next;
        if settled {
            break;
        }
    }
    (meter, ev, uni_gfx::exposure_scale(ev))
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
            lux += world_luminance(&snap, cam.ray(p), None, 1.0, None) * cam.pixel_solid_angle(p);
        }
        let point = point_lux(b);
        assert!((lux / point - 1.0).abs() < 0.02, "{lux} vs {point}");
    }
}
