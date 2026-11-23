//! The per-frame view of the world that the renderer draws (M1 Task A4). Positions are
//! camera-relative in f64 [m], ICRS axes.
//!
//! Each Solar System body is in `bodies` when it projects to at least [`POINT_LOD_PX`], else
//! in `points` (when on screen), never both (Review Focus 4). The Sun and the catalogue stars
//! within 0.1 pc go to `star_surfaces` or `points` the same way, drawn once from f64
//! positions, and every near star's catalogue index goes to `suppressed_stars`, so the star
//! pass skips it. An unnamed catalogue star becomes a registry object at the [`World::tick`]
//! after the first snapshot that finds it near (review I1); in that one snapshot it is a point
//! without an id at any size, since [`StarSurfaceView::id`] names an object. Points and picks
//! off screen are left out.

use glam::{DQuat, DVec2, DVec3, UVec2, Vec2, Vec3};
use uni_core::{
    AU, CameraView, FrameId, PC, TEFF_SUN_K, blackbody_linear_srgb, star_illuminance_lux,
};
use uni_physics::photometry::{
    Brdf, comet_total_v, disk_phase_function, planet_phase_correction, rho_from_geometric_albedo,
};

use crate::registry::{ObjectId, ObjectKind, Source};
use crate::world::{SATURN, World, luminous_intensity_cd, sun_intensity_cd};

/// Below this projected radius [px] a body is drawn as a point.
pub const POINT_LOD_PX: f64 = 1.5;
/// Between [`POINT_LOD_PX`] and this projected radius [px] a resolved body blends its exposure
/// from star exposure to scene exposure.
pub const BLEND_LOD_PX: f64 = 4.0;

/// Points whose centre is up to this far [px] outside the viewport are kept: the largest
/// footprint a point sprite has (`uni_sky::psf::MAX_RADIUS_PX`, 64 px).
const POINT_MARGIN_PX: f64 = 64.0;
/// At most this many named stars are pickable per frame, the brightest on screen (plan A4).
const MAX_STAR_PICKS: usize = 2000;
/// Upper clamp on Mallama & Hilton's phase correction, in the point path as in the disk path
/// (lead ruling after A3, so brightness stays continuous where a body resolves). Public so the
/// bodies pass (B1) clamps its per-pixel C with the same constant (B0 ruling A.1).
pub const MAX_PHASE_CORRECTION: f64 = 10.0;
/// Siblings and moons that may eclipse a body, after its parent (plan A4: up to 3).
const MAX_OTHER_OCCLUDERS: usize = 3;

/// Projected radius [px] of a sphere of `radius_m` at `dist_m`:
/// (h/2)·tan(asin(R/d))/tan(fov/2); R ≥ d → +∞ (review m5).
///
/// A sphere subtends asin(R/d) and the projection is rectilinear, so its silhouette's radius on
/// the image plane is tan(asin(R/d)) = (R/d)/√(1 − (R/d)²), in units of the half-height's
/// tan(fov/2). The same formula as the bodies pass's WGSL.
pub fn proj_radius_px(radius_m: f64, dist_m: f64, fov_y_rad: f64, height_px: f64) -> f64 {
    if !(dist_m > radius_m) {
        return f64::INFINITY;
    }
    let s = radius_m / dist_m;
    0.5 * height_px * (s / (1.0 - s * s).sqrt()) / (0.5 * fov_y_rad).tan()
}

/// The Sun as a light source.
#[derive(Clone, Debug)]
pub struct SunLight {
    pub rel_pos_m: DVec3,
    pub radius_m: f64,
    /// Luminous intensity [cd].
    pub intensity_cd: f64,
    pub color: Vec3,
}

/// A sphere that can eclipse a body.
#[derive(Clone, Copy, Debug)]
pub struct Occluder {
    pub rel_pos_m: DVec3,
    pub radius_m: f64,
}

/// A planet's ring plane.
#[derive(Clone, Copy, Debug)]
pub struct RingView {
    pub inner_m: f64,
    pub outer_m: f64,
    /// Unit normal, ICRS axes.
    pub normal: DVec3,
}

/// A body drawn as a resolved disk.
#[derive(Clone, Debug)]
pub struct BodyView {
    pub id: ObjectId,
    pub naif: Option<i32>,
    pub rel_pos_m: DVec3,
    /// Body-fixed → ICRS rotation.
    pub orient: DQuat,
    pub radii_m: DVec3,
    pub proj_radius_px: f64,
    /// The Sun relative to the body [m].
    pub sun_rel_m: DVec3,
    pub sun_illuminance_lux: f64,
    pub occluders: Vec<Occluder>,
    pub ring: Option<RingView>,
    pub has_atmosphere: bool,
    /// Reflectance model the body's albedo was solved for (Lambert for the atmosphere bodies
    /// and the giants, lunar-Lambert L = 0.9 for the airless ones; A8's map calibration). The
    /// same model the point path ([`reflected_lux`]) uses, so the point ↔ disk handover is
    /// continuous (B0 ruling A.1).
    pub brdf: Brdf,
    /// V geometric albedo p_V: the fact sheet's, or the registry's assumed default for a body
    /// without one (`uni_sim::{DEFAULT_MOON_ALBEDO, COMET_ALBEDO, ASSUMED_ASTEROID_ALBEDO}`).
    pub geometric_albedo: f64,
    /// Volumetric mean radius (a·b·c)^⅓ [m], the radius behind the point path's flux.
    pub mean_radius_m: f64,
}

/// A star drawn as a resolved surface (the Sun, or a catalogue star nearer than 0.1 pc).
#[derive(Clone, Debug)]
pub struct StarSurfaceView {
    pub id: ObjectId,
    pub catalog_index: Option<u32>,
    pub rel_pos_m: DVec3,
    pub orient: DQuat,
    pub radius_m: f64,
    pub teff_k: f64,
    pub luminous_intensity_cd: f64,
    pub proj_radius_px: f64,
}

/// An unresolved source drawn as a point.
#[derive(Clone, Debug)]
pub struct PointView {
    pub id: Option<ObjectId>,
    pub rel_pos_m: DVec3,
    /// Illuminance at the camera [lx].
    pub illuminance_lux: f64,
    pub color: Vec3,
}

/// A clickable object on screen.
#[derive(Clone, Copy, Debug)]
pub struct PickItem {
    pub id: ObjectId,
    /// Position in internal pixels.
    pub screen_px: Vec2,
    pub dist_m: f64,
    pub priority: u8,
}

/// Everything the renderer and the HUD need from the world this frame.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub sun: Option<SunLight>,
    pub bodies: Vec<BodyView>,
    pub star_surfaces: Vec<StarSurfaceView>,
    pub points: Vec<PointView>,
    pub pick: Vec<PickItem>,
    /// Catalogue indices the star pass must skip (drawn as surfaces or points instead; ≤ 8).
    pub suppressed_stars: Vec<u32>,
    /// The nearest surface and the distance to it [m].
    pub nearest_surface: Option<(ObjectId, f64)>,
    pub dominant_frame: FrameId,
}

/// The camera's projection into internal pixels (origin top left, y down; pixel i spans
/// [i, i + 1), as `pixel_dir` in `common/camera.wgsl`).
pub(crate) struct Projector {
    to_camera: DQuat,
    tan_half_y: f64,
    aspect: f64,
    size: DVec2,
}

impl Projector {
    pub(crate) fn new(view: &CameraView, viewport: UVec2) -> Self {
        let size = viewport.max(UVec2::ONE).as_dvec2();
        Projector {
            to_camera: view.orient.normalize().conjugate(),
            tan_half_y: (0.5 * view.fov_y_rad).tan(),
            aspect: size.x / size.y,
            size,
        }
    }

    /// Where a point at `rel` [m] from the camera lands [px]; `None` behind the camera.
    pub(crate) fn screen(&self, rel: DVec3) -> Option<DVec2> {
        let c = self.to_camera * rel;
        if !(c.z < 0.0) {
            return None;
        }
        let ndc = DVec2::new(c.x / self.aspect, c.y) / (-c.z * self.tan_half_y);
        Some(DVec2::new(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * self.size)
    }

    /// `p` [px] within `margin_px` of the viewport.
    pub(crate) fn on_screen(&self, p: DVec2, margin_px: f64) -> bool {
        p.x >= -margin_px
            && p.y >= -margin_px
            && p.x <= self.size.x + margin_px
            && p.y <= self.size.y + margin_px
    }
}

/// Pick priority, higher first (C3: bodies and the Sun > comets > named stars and exoplanet
/// hosts > asteroids; TestBH-1 counts as a body).
fn pick_priority(kind: ObjectKind) -> u8 {
    match kind {
        ObjectKind::Sun
        | ObjectKind::Planet
        | ObjectKind::DwarfPlanet
        | ObjectKind::Moon
        | ObjectKind::BlackHole => 3,
        ObjectKind::Comet => 2,
        ObjectKind::Star | ObjectKind::ExoHost => 1,
        ObjectKind::Asteroid => 0,
    }
}

/// Angle between two vectors [rad], accurate near 0 and π.
fn angle(a: DVec3, b: DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b))
}

/// Builds the snapshot (see [`World::snapshot`]).
pub(crate) fn build(world: &World, view: &CameraView, viewport: UVec2) -> Snapshot {
    let proj = Projector::new(view, viewport);
    let height = f64::from(viewport.y.max(1));
    let fov = view.fov_y_rad;
    let cam = view.pos;
    let sun_pos = world.sun.map(|s| world.cached[s as usize]);
    let sun_color = blackbody_linear_srgb(TEFF_SUN_K);
    let sun_cd = sun_intensity_cd();
    let near = world.near_stars(&cam);
    let mut snap = Snapshot {
        dominant_frame: world.dominant_frame(&cam),
        nearest_surface: world.nearest_surface_among(&cam, &near),
        ..Snapshot::default()
    };
    // The comets drawn, so ticks re-osculate the stale ones first (review r2 P1).
    let mut shown_comets = Vec::new();
    let pick = |snap: &mut Snapshot, id: ObjectId, rel: DVec3, kind: ObjectKind| {
        if let Some(p) = proj.screen(rel).filter(|&p| proj.on_screen(p, 0.0)) {
            snap.pick.push(PickItem {
                id,
                screen_px: p.as_vec2(),
                dist_m: rel.length(),
                priority: pick_priority(kind),
            });
        }
    };

    for k in 0..world.cached.len() {
        let id = k as ObjectId;
        let info = &world.objects[k];
        let rel = world.cached[k].delta_from(cam);
        if !rel.is_finite() {
            continue;
        }
        pick(&mut snap, id, rel, info.kind);
        let d = rel.length();
        let px = proj_radius_px(info.radii_m.max_element(), d, fov, height);
        let on_screen = || proj.screen(rel).is_some_and(|p| proj.on_screen(p, POINT_MARGIN_PX));
        match world.physical[k].source {
            Source::BlackHole(_) | Source::Star => {}
            Source::Ephem if Some(id) == world.sun => {
                if px >= POINT_LOD_PX {
                    snap.sun_surface(world, id, rel, px, sun_cd);
                } else if on_screen() {
                    snap.points.push(PointView {
                        id: Some(id),
                        rel_pos_m: rel,
                        illuminance_lux: sun_cd / (d * d),
                        color: sun_color,
                    });
                }
            }
            source => {
                let sun_rel = sun_pos.map_or(DVec3::ZERO, |s| s.delta_from(world.cached[k]));
                let lux = if px >= POINT_LOD_PX {
                    snap.bodies.push(body_view(world, id, rel, sun_rel, px, &cam, sun_cd));
                    f64::INFINITY
                } else if on_screen() {
                    let lux = reflected_lux(world, id, rel, sun_rel, sun_cd);
                    snap.points.push(PointView {
                        id: Some(id),
                        rel_pos_m: rel,
                        illuminance_lux: lux,
                        color: sun_color,
                    });
                    lux
                } else {
                    continue;
                };
                if let Source::Comet(c) = source {
                    shown_comets.push((c, lux));
                }
            }
        }
    }
    world.note_shown_comets(shown_comets);
    if let (Some(sun), Some(sun_pos)) = (world.sun, sun_pos) {
        snap.sun = Some(SunLight {
            rel_pos_m: sun_pos.delta_from(cam),
            radius_m: world.objects[sun as usize].radii_m.x,
            intensity_cd: sun_cd,
            color: sun_color,
        });
    }

    // Catalogue stars within 0.1 pc: drawn here instead of by the star pass. An unnamed one
    // not registered yet (no object id until the next tick) is a point even where it would
    // resolve (module doc).
    for &(index, id) in &near {
        let star = world.star_data()[index as usize];
        let rel = world.star_position(index).delta_from(cam);
        let d = rel.length();
        let radius = world.star_radius_m(index);
        let px = proj_radius_px(radius, d, fov, height);
        let teff = world.star_teff_k(index);
        let intensity = luminous_intensity_cd(f64::from(star.abs_mag_v));
        match id {
            Some(id) if px >= POINT_LOD_PX => snap.star_surfaces.push(StarSurfaceView {
                id,
                catalog_index: Some(index),
                rel_pos_m: rel,
                orient: DQuat::IDENTITY,
                radius_m: radius,
                teff_k: teff,
                luminous_intensity_cd: intensity,
                proj_radius_px: px,
            }),
            _ if proj.screen(rel).is_some_and(|p| proj.on_screen(p, POINT_MARGIN_PX)) => {
                snap.points.push(PointView {
                    id,
                    rel_pos_m: rel,
                    illuminance_lux: intensity / (d * d),
                    color: blackbody_linear_srgb(teff),
                });
            }
            _ => {}
        }
        snap.suppressed_stars.push(index);
    }

    // Exoplanet hosts on screen, and the brightest named stars on screen.
    let mut named = Vec::new();
    for k in world.cached.len()..world.objects.len() {
        let info = &world.objects[k];
        let Some(index) = info.catalog_index else {
            continue;
        };
        let rel = world.star_position(index).delta_from(cam);
        match info.kind {
            ObjectKind::ExoHost => pick(&mut snap, k as ObjectId, rel, info.kind),
            ObjectKind::Star => {
                if proj.screen(rel).is_some_and(|p| proj.on_screen(p, 0.0)) {
                    let d_pc = rel.length() / PC;
                    let abs_mag = f64::from(world.star_data()[index as usize].abs_mag_v);
                    named.push((abs_mag + 5.0 * (d_pc / 10.0).log10(), k as ObjectId, rel));
                }
            }
            _ => {}
        }
    }
    if named.len() > MAX_STAR_PICKS {
        named.select_nth_unstable_by(MAX_STAR_PICKS, |a, b| a.0.total_cmp(&b.0));
        named.truncate(MAX_STAR_PICKS);
    }
    named.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for (_, id, rel) in named {
        pick(&mut snap, id, rel, ObjectKind::Star);
    }
    snap
}

impl Snapshot {
    fn sun_surface(&mut self, world: &World, id: ObjectId, rel: DVec3, px: f64, sun_cd: f64) {
        self.star_surfaces.push(StarSurfaceView {
            id,
            catalog_index: None,
            rel_pos_m: rel,
            orient: world.orientation(id),
            radius_m: world.objects[id as usize].radii_m.x,
            teff_k: TEFF_SUN_K,
            luminous_intensity_cd: sun_cd,
            proj_radius_px: px,
        });
    }
}

/// A resolved body: `rel` camera → body, `sun_rel` body → Sun [m].
fn body_view(
    world: &World,
    id: ObjectId,
    rel: DVec3,
    sun_rel: DVec3,
    px: f64,
    cam: &uni_core::UniversePos,
    sun_cd: f64,
) -> BodyView {
    let info = &world.objects[id as usize];
    let physical = &world.physical[id as usize];
    let orient = world.orientation(id);
    let r_sun = sun_rel.length();
    let ring = (info.naif == Some(SATURN)).then(|| RingView {
        inner_m: world.ring_m.0,
        outer_m: world.ring_m.1,
        normal: orient * DVec3::Z,
    });
    BodyView {
        id,
        naif: info.naif,
        rel_pos_m: rel,
        orient,
        radii_m: info.radii_m,
        proj_radius_px: px,
        sun_rel_m: sun_rel,
        sun_illuminance_lux: if r_sun > 0.0 { sun_cd / (r_sun * r_sun) } else { 0.0 },
        occluders: occluders(world, id, sun_rel, cam),
        ring,
        has_atmosphere: physical.has_atmosphere,
        brdf: physical.brdf,
        geometric_albedo: physical.geometric_albedo,
        mean_radius_m: physical.mean_radius_m,
    }
}

/// Spheres that can eclipse body `id`: its parent (a planet, not the Sun) and up to three
/// siblings (moons of the same planet) or moons of its own, those that can come between some
/// point of the body and some point of the Sun's disk: seen from the body's centre, the
/// occluder must be nearer than the Sun and within its angular radius + the Sun's + the
/// body's own extent (as seen at the occluder's distance) of the Sun's direction. The others
/// are ordered by how deep inside that limit they are. Heliocentric siblings (planets for a
/// planet) are not considered.
fn occluders(
    world: &World,
    id: ObjectId,
    sun_rel: DVec3,
    cam: &uni_core::UniversePos,
) -> Vec<Occluder> {
    let info = &world.objects[id as usize];
    let r_sun = sun_rel.length();
    let Some(sun) = world.sun else {
        return Vec::new();
    };
    if world.physical[id as usize].source != Source::Ephem || r_sun == 0.0 {
        return Vec::new();
    }
    let body_pos = world.cached[id as usize];
    let body_r = info.radii_m.max_element();
    let sun_angle = (world.objects[sun as usize].radii_m.x / r_sun).min(1.0).asin();
    // How far inside the eclipse limit occluder `o` is [rad]; `None` if it cannot eclipse.
    let margin = |o: ObjectId| -> Option<f64> {
        let w = world.cached[o as usize].delta_from(body_pos);
        let d = w.length();
        let r_o = world.objects[o as usize].radii_m.max_element();
        if d <= r_o + body_r {
            return Some(f64::INFINITY);
        }
        if d >= r_sun {
            return None;
        }
        let limit = (r_o / d).asin() + sun_angle + (body_r / d).atan();
        let m = limit - angle(sun_rel, w);
        (m > 0.0).then_some(m)
    };
    let parent = info.parent.filter(|&p| p != sun);
    let mut others: Vec<(f64, ObjectId)> = world
        .ephem_order
        .iter()
        .copied()
        .filter(|&o| o != id && Some(o) != parent)
        .filter(|&o| {
            let p = world.objects[o as usize].parent;
            p == Some(id) || (parent.is_some() && p == parent)
        })
        .filter_map(|o| Some((margin(o)?, o)))
        .collect();
    others.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let sphere = |o: ObjectId| Occluder {
        rel_pos_m: world.cached[o as usize].delta_from(*cam),
        radius_m: world.physical[o as usize].mean_radius_m,
    };
    parent
        .filter(|&p| margin(p).is_some())
        .into_iter()
        .chain(others.into_iter().take(MAX_OTHER_OCCLUDERS).map(|(_, o)| o))
        .map(sphere)
        .collect()
}

/// Illuminance [lx] at the camera from an unresolved Solar System body: `rel` camera → body,
/// `sun_rel` body → Sun [m].
///
/// A body lit at normal incidence by E☉ sends ρ·[L + 2(1 − L)/3]·E☉·(R/d)²·Φ(α), the
/// closed form of `photometry::disk_illuminance_lux` (`disk_phase_function`'s doc; lead ruling
/// after A3: never the 65 536-sample integral per frame), times Mallama & Hilton's C(α)
/// for the planets, clamped to ≤ 10 as in the disk path (lead ruling). C holds its value
/// beyond each law's range; for Earth and Mars that hold starts at 170°, A3's choice accepted
/// by the lead. ρ comes from p_V (`rho_from_geometric_albedo`), R is the volumetric mean
/// radius. A comet with SBDB's M1, K1 uses its total (nucleus + coma) magnitude instead.
pub(crate) fn reflected_lux(
    world: &World,
    id: ObjectId,
    rel: DVec3,
    sun_rel: DVec3,
    sun_cd: f64,
) -> f64 {
    let (d, r_sun) = (rel.length(), sun_rel.length());
    if !(d > 0.0 && r_sun > 0.0) {
        return 0.0;
    }
    if let Some((m1, k1)) = world.comet_magnitude(id) {
        return star_illuminance_lux(comet_total_v(m1, k1, r_sun / AU, d / AU));
    }
    let physical = &world.physical[id as usize];
    let brdf = physical.brdf;
    let phase = angle(sun_rel, -rel);
    let weight = match brdf {
        Brdf::Lambert => 2.0 / 3.0,
        Brdf::LunarLambert { l } => l + 2.0 * (1.0 - l) / 3.0,
    };
    let rho = rho_from_geometric_albedo(brdf, physical.geometric_albedo);
    let correction = world.objects[id as usize]
        .naif
        .map_or(1.0, |n| planet_phase_correction(n, brdf, phase))
        .min(MAX_PHASE_CORRECTION);
    let ratio = physical.mean_radius_m / d;
    rho * weight * (sun_cd / (r_sun * r_sun)) * ratio * ratio
        * disk_phase_function(brdf, phase)
        * correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proj_radius_uses_asin_tan() {
        // R/d = 0.5 subtends asin 0.5 = 30°; at fov 60° that is the frame's half-height.
        let fov = 60f64.to_radians();
        let p = proj_radius_px(1.0, 2.0, fov, 1080.0);
        assert!((p - 540.0).abs() < 1e-9, "{p}");
        // The m0-era atan(R/d) form gave 540·tan(26.57°)/tan(30°) = 467.65 px here.
        assert!((p - 540.0 * 0.5 / (fov / 2.0).tan()).abs() > 70.0);
        assert_eq!(proj_radius_px(2.0, 2.0, fov, 1080.0), f64::INFINITY);
        assert_eq!(proj_radius_px(3.0, 2.0, fov, 1080.0), f64::INFINITY);
        assert_eq!(proj_radius_px(1.0, 0.0, fov, 1080.0), f64::INFINITY);
        // Far away it tends to (h/2)·(R/d)/tan(fov/2).
        let far = proj_radius_px(1.0, 1.0e6, fov, 1080.0);
        assert!((far / (540.0e-6 / (fov / 2.0).tan()) - 1.0).abs() < 1e-11, "{far}");
    }
}
