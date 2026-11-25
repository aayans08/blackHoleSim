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
//! Points (unresolved bodies) are sky light at star exposure and are not metered, except a light
//! that light-adapts the eye (final review render I2: the Sun inside ~356 AU, at ≥
//! `uni_sky::ADAPTING_LUX`): below 1.5 px the star-surface pass draws it as world light before
//! the meter, the star pass's PSF with the eye's glare at the scene exposure
//! (`uni_planet::world_points`, `uni_sky::world_points::world_splat`), and so does the paper
//! ([`world_point_glare`]), on every pixel no resolved body covers. Its footprint depends on the
//! exposure (the clamped halo's kept light and taper), so the paper iterates the meter to the
//! fixed point. In the LOD band its disk is × the cross-fade's f (`uni_planet::light_disk_fade`)
//! and the point carries 1 − f. Atmospheres and rings over the glare are left out.
//!
//! The luminances go through B6's CPU meter (`Histogram::from_luminances(...).meter`,
//! `target_ev`), every pixel of the frame.

use glam::{DQuat, DVec2, DVec3, UVec2};
use uni_core::CameraView;
use uni_gfx::LightView;
#[cfg(test)]
use uni_physics::photometry::{Brdf, disk_phase_function};
use uni_physics::photometry::{
    lambert_phase, planet_phase_correction, reflectance, rho_from_geometric_albedo,
};
use uni_post::histogram::Histogram;
use uni_sim::snapshot::{BodyView, CometView, Snapshot};

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

/// The comae of `comets` along the ray `dir` [cd/m²] (module docs): at the ray's
/// perpendicular distance from each nucleus in front of the camera, with one pixel
/// (`pixel_angle` [rad] across) of softening at the nucleus's distance, every comet at every
/// distance. The reference [`coma_raster`] is checked against; the paper itself uses the raster.
#[cfg(test)]
pub fn coma_luminance(comets: &[CometView], dir: DVec3, pixel_angle: f64) -> f64 {
    comets
        .iter()
        .filter(|c| c.nucleus_rel_m.dot(dir) > 0.0)
        .map(|c| coma_term(c, dir, pixel_angle))
        .filter(|l| l.is_finite())
        .sum()
}

/// One comet's coma along the ray `dir` [cd/m²] ([`coma_luminance`]'s term).
fn coma_term(c: &CometView, dir: DVec3, pixel_angle: f64) -> f64 {
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
}

/// One coma the paper adds ([`visible_comae`]): its comet (an index into the list), the reach
/// [m] (perpendicular distance from the nucleus) inside which it is added, and its screen
/// bounding box [x0, x1) × [y0, y1) [px].
struct PaperComa {
    index: usize,
    reach_m: f64,
    bbox: [u32; 4],
}

/// The comae of `comets` the comet pass can draw on a `size` frame seen with `cam` (C1L fix
/// round 2, perf; lead request "paper comet cull"), with the pass's own rules, so the paper and
/// the GPU agree:
/// - **Which comets** (`plan_frame`): active, r_h < `COMET_ACTIVE_AU` with a finite total V,
///   with coma light.
/// - **How far** (`vs_coma`, `fs_coma`): within the coma quad, `COMA_EXTENT_SCALES` scale lengths
///   (max(ℓ, a)), shrunk by the glow cull `coma_visible_half_m` at the brightest exposure the
///   app can meter, EV100 −30 (`scene::EV100_LIMITS`), so nothing the pass could draw at any
///   exposure is dropped.
/// - **What is dropped:** light beyond 10 scale lengths (the pass draws none; e^−10 of the
///   coma's light) and comae whose brightest point is below `GLOW_CULL_PRE` at EV100 −30
///   (2 × 10⁻¹⁶ cd/m²). Before C1L fix round 2 the paper added every comet's coma at every pixel,
///   which made the intent tests very slow; the meter counts only pixels ≥ 10⁻³ cd/m², so its EVs
///   do not move (`paper_comet_cull_keeps_the_meter`).
/// - **Where:** none when the reach lies wholly behind the camera; else the bounding box of the
///   cone of rays within the reach of the nucleus (sampled at
///   32 points, widened by 1/cos(π/32) and 2 px); the whole frame when the camera is within 1.01 ×
///   the reach of the nucleus or the cone reaches behind it.
fn visible_comae(comets: &[CometView], cam: &Cam, size: UVec2) -> Vec<PaperComa> {
    use uni_planet::comet::coma::{
        COMA_EXTENT_SCALES, coma_intensity_cd, coma_scale_m, coma_visible_half_m,
    };
    let pre_max = f64::from(uni_gfx::exposure_scale(*crate::scene::EV100_LIMITS.start()));
    let pixel_angle = cam.pixel_angle();
    let whole = [0, size.x, 0, size.y];
    let mut out = Vec::new();
    for (index, c) in comets.iter().enumerate() {
        if !(c.r_au < uni_sim::snapshot::COMET_ACTIVE_AU && c.total_v.is_some_and(f64::is_finite)) {
            continue;
        }
        let intensity = coma_intensity_cd(c);
        let d = c.nucleus_rel_m.length();
        if !(intensity > 0.0 && intensity.is_finite() && d > 0.0 && d.is_finite()) {
            continue;
        }
        let scale = coma_scale_m(c.r_au);
        let a = d * pixel_angle / std::f64::consts::PI.sqrt();
        let half = COMA_EXTENT_SCALES * scale.max(a);
        let reach_m = coma_visible_half_m(intensity, scale, a, half, pre_max);
        // Nothing drawn, or all of it behind the camera (the pass's `sphere_in_view` culls it).
        if !(reach_m > 0.0) || c.nucleus_rel_m.dot(cam.forward()) < -reach_m {
            continue;
        }
        let bbox = if reach_m >= d / 1.01 {
            whole
        } else {
            // The cone's edge on a plane facing the camera at the nucleus: radius
            // reach/√(1 − (reach/d)²), widened for the 32-gon.
            let n = c.nucleus_rel_m / d;
            let e1 = n.any_orthonormal_vector();
            let e2 = n.cross(e1);
            let r = reach_m
                / (1.0 - (reach_m / d).powi(2)).sqrt()
                / (std::f64::consts::PI / 32.0).cos();
            let pts: Option<Vec<DVec2>> = (0..32)
                .map(|k| {
                    let t = std::f64::consts::TAU * f64::from(k) / 32.0;
                    cam.project(c.nucleus_rel_m + r * (t.cos() * e1 + t.sin() * e2))
                })
                .collect();
            match pts {
                None => whole,
                Some(p) => {
                    let lo = p.iter().copied().fold(DVec2::INFINITY, DVec2::min) - 2.0;
                    let hi = p.iter().copied().fold(DVec2::NEG_INFINITY, DVec2::max) + 2.0;
                    let clamp = |v: f64, max: u32| v.clamp(0.0, f64::from(max)) as u32;
                    [
                        clamp(lo.x.floor(), size.x),
                        clamp(hi.x.ceil(), size.x),
                        clamp(lo.y.floor(), size.y),
                        clamp(hi.y.ceil(), size.y),
                    ]
                }
            }
        };
        if bbox[0] < bbox[1] && bbox[2] < bbox[3] {
            out.push(PaperComa {
                index,
                reach_m,
                bbox,
            });
        }
    }
    out
}

/// The comae on a `size` frame seen with `cam` [cd/m²], row-major from the top-left pixel,
/// sampled at pixel centres as the paper's loop does: [`coma_luminance`]'s terms, added in the
/// same order, but only for [`visible_comae`] inside their reach and bounding box (C1L fix
/// round 2, perf).
pub fn coma_raster(comets: &[CometView], cam: &Cam, size: UVec2) -> Vec<f64> {
    let (w, h) = (size.x as usize, size.y as usize);
    let mut out = vec![0.0_f64; w * h];
    let pixel_angle = cam.pixel_angle();
    for coma in visible_comae(comets, cam, size) {
        let c = &comets[coma.index];
        let [x0, x1, y0, y1] = coma.bbox;
        for y in y0..y1 {
            for x in x0..x1 {
                let ray = cam.ray(DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5));
                if !(c.nucleus_rel_m.dot(ray) > 0.0)
                    || c.nucleus_rel_m.cross(ray).length() > coma.reach_m
                {
                    continue;
                }
                let l = coma_term(c, ray, pixel_angle);
                if l.is_finite() {
                    out[y as usize * w + x as usize] += l;
                }
            }
        }
    }
    out
}

/// The luminance [cd/m²] of world light along unit `dir` from the camera (0 for sky). With
/// `venus`, Venus is drawn with B3's haze ([`VenusAtmo::luminance`]); without, as a bare
/// Lambert deck with ρ from p_V (the plan's row).
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
    let lights = snapshot_lights(snap);
    for s in &snap.star_surfaces {
        if let Some(t) = hit_ellipsoid(-s.rel_pos_m, dir, DQuat::IDENTITY, DVec3::splat(s.radius_m))
        {
            let n = (dir * t - s.rel_pos_m).normalize();
            let mu = (-n.dot(dir)) as f32;
            let ld = uni_planet::limb_darkening(mu).y / uni_planet::limb_darkening_mean().y;
            let mean = s.luminous_intensity_cd / (std::f64::consts::PI * s.radius_m * s.radius_m);
            // A light-adapting light's disk in the point → disk cross-fade (module docs).
            let fade = uni_planet::light_disk_fade(s, &lights).map_or(1.0, f64::from);
            offer(t, mean * f64::from(ld) * fade);
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

/// The frame's lights as the app builds them (`scene::frame_ctx`: the snapshot's Sun).
pub fn snapshot_lights(snap: &Snapshot) -> Vec<LightView> {
    snap.sun
        .iter()
        .map(|s| LightView {
            rel_pos_m: s.rel_pos_m,
            radius_m: s.radius_m,
            luminous_intensity_cd: s.intensity_cd,
            color: s.color,
        })
        .collect()
}

/// Whether a resolved body or star surface lies along unit `dir` nearer than `depth_m`: the
/// passes drawn after the `sun` pass's world-light point cover it there.
fn covered(snap: &Snapshot, dir: DVec3, depth_m: f64) -> bool {
    snap.bodies
        .iter()
        .any(|b| hit_body(&drawn(b), dir).is_some_and(|t| t < depth_m))
        || snap.star_surfaces.iter().any(|s| {
            hit_ellipsoid(-s.rel_pos_m, dir, DQuat::IDENTITY, DVec3::splat(s.radius_m))
                .is_some_and(|t| t < depth_m)
        })
}

/// The world luminance [cd/m²] the light-adapting lights' points (`uni_planet::world_points`) add
/// to each pixel of a `size` frame of `view` written at scene exposure `pre`, row-major: each
/// point's splat (`uni_sky::world_points::world_splat`, the `sun` pass's maths) on every pixel of
/// its footprint that no resolved body or star surface covers. Empty when there is none.
pub fn world_point_glare(
    snap: &Snapshot,
    view: &CameraView,
    size: UVec2,
    cam: &Cam,
    pre: f32,
) -> Vec<f64> {
    let lights = snapshot_lights(snap);
    let scene = uni_planet::BodyScene {
        stars: snap.star_surfaces.clone(),
        ..Default::default()
    };
    let points = uni_planet::world_points(&scene, &lights, view.fov_y_rad, f64::from(size.y));
    if points.is_empty() {
        return Vec::new();
    }
    let (w, h) = (size.x as usize, size.y as usize);
    let mut out = vec![0.0; w * h];
    let view_rot = crate::scene::view_rot(view.orient);
    let aspect = size.x as f32 / size.y as f32;
    let proj = uni_gfx::reversed_z_infinite_proj(
        view.fov_y_rad as f32,
        aspect,
        crate::scene::NEAR_PLANE_M,
    );
    let fsize = glam::Vec2::new(size.x as f32, size.y as f32);
    for p in &points {
        let dir = p.rel_pos_m.normalize();
        let Some(splat) = uni_sky::world_points::world_splat(
            view_rot,
            proj,
            fsize,
            view.fov_y_rad as f32,
            dir.as_vec3(),
            p.illuminance_lux as f32,
            pre,
        ) else {
            continue;
        };
        let r = splat.shape.radius_px + 1.0;
        let lo = (splat.centre_px - glam::Vec2::splat(r))
            .floor()
            .max(glam::Vec2::ZERO);
        let hi = (splat.centre_px + glam::Vec2::splat(r))
            .ceil()
            .min(fsize - glam::Vec2::ONE);
        for y in lo.y as u32..=hi.y as u32 {
            for x in lo.x as u32..=hi.x as u32 {
                let l = splat.luminance_at(glam::Vec2::new(x as f32, y as f32));
                if !(l > 0.0) {
                    continue;
                }
                let ray = cam.ray(DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5));
                if !covered(snap, ray, p.depth_m) {
                    out[y as usize * w + x as usize] += f64::from(l);
                }
            }
        }
    }
    out
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
    // The comae and the dust tails at every pixel (lead ruling C1L-b; the comae culled as the
    // comet pass culls them, C1L fix round 2).
    let comae = coma_raster(&snap.comets, &cam, size);
    let dust = uni_planet::dust_tail_luminance(&snap.comets, world.epoch(), view, size);
    let w = STRIDE * STRIDE;
    // Every sampled pixel's world light without the light-adapting points (index, luminance).
    let mut base = Vec::new();
    for y in (0..size.y).step_by(STRIDE as usize) {
        for x in (0..size.x).step_by(STRIDE as usize) {
            let p = DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5);
            let ray = cam.ray(p);
            let i = y as usize * size.x as usize + x as usize;
            let l = world_luminance(&snap, ray, rings, earth_k, venus) + comae[i] + dust.at(x, y);
            base.push((i, l));
        }
    }
    let lums_at = |glare: &[f64]| -> Vec<f32> {
        let mut lums = Vec::new();
        for &(i, l) in &base {
            let l = l + glare.get(i).copied().unwrap_or(0.0);
            if l > 0.0 {
                for _ in 0..w {
                    lums.push(l as f32);
                }
            }
        }
        lums
    };
    let total = size.x * size.y;
    let mut lums = lums_at(&[]);
    let (mut meter, mut target, mut pre) = settled_meter(&lums, total, subnormals);
    // A light-adapting point's glare is drawn at the scene exposure, and its footprint depends on
    // it: iterate to the fixed point (module docs; it settles in 2–4 steps).
    for _ in 0..SETTLE_STEPS {
        let glare = world_point_glare(&snap, view, size, &cam, pre);
        if glare.is_empty() {
            break;
        }
        lums = lums_at(&glare);
        let previous = target;
        (meter, target, pre) = settled_meter(&lums, total, subnormals);
        if (target - previous).abs() < 1.0e-3 {
            break;
        }
    }
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

    /// C1L fix round 2 (perf): the culled coma raster against the old every-comet sum
    /// ([`coma_luminance`]) on a 192 × 108 frame with a bright comet near the centre, a faint
    /// one far off, one behind the camera, one beyond 5 AU and one without a magnitude law. The
    /// raster differs only by what the comet pass never draws (beyond 10 scale lengths, or a
    /// coma beyond 5 AU or below the glow cull at EV −30), bounded per pixel by the dropped
    /// comets' largest terms; every pixel ≥ 10⁻³ cd/m² keeps its f32 value, and the meter's EV is
    /// identical.
    #[test]
    fn paper_comet_cull_keeps_the_meter() {
        use uni_core::{AU, Epoch, UniversePos};
        use uni_physics::smallbody::conic_from_perihelion;
        let comet = |id: u32, rel: DVec3, r_au: f64, law: Option<(f64, f64)>| {
            let delta_au = rel.length() / AU;
            CometView {
                id,
                name: format!("comet {id}"),
                nucleus_rel_m: rel,
                nucleus_radius_m: 2_000.0,
                radius_assumed: true,
                spkid: 2_000_000 + id as i32,
                sun_rel_m: DVec3::new(-r_au * AU, 0.0, 0.0),
                helio_vel_mps: DVec3::ZERO,
                r_au,
                delta_au,
                total_v: law
                    .map(|(m1, k1)| uni_physics::photometry::comet_total_v(m1, k1, r_au, delta_au)),
                m1: law.map(|l| l.0),
                k1: law.map(|l| l.1),
                conic: conic_from_perihelion(r_au, 0.0, 0.0, 0.0, 0.0, 2_451_545.0),
                conic_epoch: Epoch::J2000,
                proj_radius_px: 0.0,
            }
        };
        let comets = vec![
            comet(
                1,
                DVec3::new(0.002, 0.001, -0.05) * AU,
                0.6,
                Some((5.0, 10.0)),
            ),
            comet(2, DVec3::new(0.4, -0.2, -3.0) * AU, 4.0, Some((14.0, 10.0))),
            comet(3, DVec3::new(0.0, 0.0, 0.1) * AU, 1.0, Some((5.0, 10.0))),
            comet(4, DVec3::new(-0.2, 0.1, -2.0) * AU, 5.5, Some((14.0, 10.0))),
            comet(5, DVec3::new(0.1, 0.1, -1.0) * AU, 1.0, None),
        ];
        let view = CameraView {
            pos: UniversePos::default(),
            orient: DQuat::IDENTITY,
            fov_y_rad: 20f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let size = UVec2::new(192, 108);
        let cam = Cam::new(&view, size);
        let kept = visible_comae(&comets, &cam, size);
        let kept_ids: Vec<u32> = kept.iter().map(|k| comets[k.index].id).collect();
        assert_eq!(kept_ids, vec![1, 2], "{kept_ids:?}");
        // The most a dropped term can be at any pixel: a kept comet beyond its reach, or a
        // dropped comet in front of the camera anywhere (its term at its own nucleus); one
        // behind the camera adds nothing to any pixel in either sum.
        let pixel_angle = cam.pixel_angle();
        let mut bound = 0.0_f64;
        for (i, c) in comets.iter().enumerate() {
            if !(c.nucleus_rel_m.dot(cam.forward()) > 0.0) {
                continue;
            }
            let at_nucleus = coma_term(c, c.nucleus_rel_m.normalize(), pixel_angle);
            let edge = kept.iter().find(|k| k.index == i).map_or(at_nucleus, |k| {
                let d = c.nucleus_rel_m.length();
                let a = d * pixel_angle / std::f64::consts::PI.sqrt();
                uni_planet::comet::coma::coma_drawn_brightness(
                    k.reach_m,
                    uni_planet::comet::coma::coma_intensity_cd(c),
                    uni_planet::comet::coma::coma_scale_m(c.r_au),
                    a,
                )
            });
            if edge.is_finite() {
                bound = bound.max(edge);
            }
        }
        let raster = coma_raster(&comets, &cam, size);
        let (mut reference, mut culled) = (Vec::new(), Vec::new());
        let mut worst = 0.0_f64;
        for y in 0..size.y {
            for x in 0..size.x {
                let ray = cam.ray(DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5));
                let r = coma_luminance(&comets, ray, pixel_angle);
                let c = raster[(y * size.x + x) as usize];
                worst = worst.max((r - c).abs());
                if r >= f64::from(uni_post::METER_MIN_CD_M2) {
                    assert_eq!(r as f32, c as f32, "({x}, {y}): {r:e} vs {c:e}");
                }
                if r > 0.0 {
                    reference.push(r as f32);
                }
                if c > 0.0 {
                    culled.push(c as f32);
                }
            }
        }
        let lit = reference
            .iter()
            .filter(|&&l| l >= uni_post::METER_MIN_CD_M2)
            .count();
        println!(
            "comae kept {kept_ids:?}; largest dropped term {bound:.2e} cd/m², largest difference \
             {worst:.2e}; {lit} lit px"
        );
        assert!(
            worst <= bound * (1.0 + 1e-9) && bound < 1e-5,
            "{worst:e} vs {bound:e}"
        );
        assert!(lit > 100, "{lit}");
        let total = size.x * size.y;
        let ev = |l: &[f32]| {
            uni_post::target_ev(
                Histogram::from_luminances(l.iter().copied()).meter(total),
                0.0,
            )
        };
        assert_eq!(ev(&reference), ev(&culled));
    }

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
