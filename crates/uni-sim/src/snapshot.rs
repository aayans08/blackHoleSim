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
    AU, CameraView, FrameId, PC, TEFF_SUN_K, UniversePos, blackbody_linear_srgb,
    star_illuminance_lux,
};
use uni_physics::photometry::{
    Brdf, comet_total_v, disk_phase_function, planet_phase_correction, rho_from_geometric_albedo,
};

use crate::registry::{ObjectId, ObjectKind, Source};
use crate::starindex::star_pos_from;
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

/// The Sun's colour (linear sRGB, luminance 1), computed once: `blackbody_linear_srgb` integrates
/// the CIE tables, a fixed cost every snapshot paid (B9g profile).
fn sun_color() -> Vec3 {
    static SUN_COLOR: std::sync::OnceLock<Vec3> = std::sync::OnceLock::new();
    *SUN_COLOR.get_or_init(|| blackbody_linear_srgb(TEFF_SUN_K))
}

/// [`proj_radius_px`] with tan(fov/2) computed once per snapshot: the same operations in the
/// same order, so the same result to the bit.
fn proj_radius_px_tan(radius_m: f64, dist_m: f64, tan_half_fov: f64, height_px: f64) -> f64 {
    if !(dist_m > radius_m) {
        return f64::INFINITY;
    }
    let s = radius_m / dist_m;
    0.5 * height_px * (s / (1.0 - s * s).sqrt()) / tan_half_fov
}

/// A conservative test that lets the named-star loop skip stars far off screen without the
/// exact `UniversePos` arithmetic (B9g, `sky-orion-3d`: 7 866 named stars per snapshot, 822 of
/// them on screen). A star is skipped only when its direction, from plain f64 metres, is more
/// than the cone through the frame's corners plus [`StarCull::MARGIN_RAD`] from the view axis,
/// and that f64 shortcut is good to [`StarCull::MAX_ERROR_RAD`] for it. Every other star takes
/// the exact path, so the snapshot is the one the exact path alone gives
/// (`build_matches_the_reference`).
struct StarCull {
    /// The camera's position from the SSB [m], in plain f64 (rounded far from the SSB).
    cam_m: DVec3,
    /// The view axis (the camera's −Z).
    forward: DVec3,
    /// cos(cone + margin); −2 (never skip) when cone + margin reaches π.
    cos_limit: f64,
}

impl StarCull {
    /// Added to the corner cone [rad]: ten times the largest shortcut error allowed, and far
    /// more than the exact path's own rounding at the frame's edge.
    const MARGIN_RAD: f64 = 1.0e-6;
    /// The largest angular error [rad] of the f64 shortcut for which a star may be skipped.
    const MAX_ERROR_RAD: f64 = 1.0e-7;

    fn new(view: &CameraView, viewport: UVec2) -> Self {
        let size = viewport.max(UVec2::ONE).as_dvec2();
        let tan_half_y = (0.5 * view.fov_y_rad).tan();
        // On screen (`Projector::screen`, `on_screen` with no margin) means |x| ≤ aspect·tan·(−z)
        // and |y| ≤ tan·(−z) in camera space, so the angle from the axis is at most this.
        let cone = (tan_half_y * (1.0 + (size.x / size.y).powi(2)).sqrt()).atan();
        let limit = cone + Self::MARGIN_RAD;
        StarCull {
            cam_m: view.pos.delta_from(UniversePos::default()),
            forward: view.orient.normalize() * DVec3::NEG_Z,
            cos_limit: if limit < std::f64::consts::PI {
                limit.cos()
            } else {
                -2.0
            },
        }
    }

    /// Whether the star at `pos_m` [m from the SSB] is certainly off screen.
    fn off_screen(&self, pos_m: DVec3) -> bool {
        let rel = pos_m - self.cam_m;
        let d = rel.length();
        // The shortcut's error [m]: a few ulps of the larger absolute position (`cam_m` and
        // `pos_m` each rounded once, the difference once more), seen from distance d.
        let error_m = 8.0 * f64::EPSILON * (pos_m.length() + self.cam_m.length());
        d.is_finite()
            && error_m <= Self::MAX_ERROR_RAD * d
            && rel.dot(self.forward) < self.cos_limit * d
    }
}

/// An object whose direction is more than asin(this) behind the camera's plane has no screen
/// position: far above the rounding of `rel · forward` and of `Projector::screen`'s rotation
/// (a few ulps of |rel|), so the shortcut never disagrees with `Projector::screen`.
const BEHIND_COS: f64 = 1.0e-9;

/// Builds the snapshot (see [`World::snapshot`]).
///
/// B9g (Wave B gate, snapshot CPU at `sky-orion-3d`) made it cheaper without changing what it
/// returns: each object's screen position is computed once for its pick and its point; tan(fov/2),
/// the Sun's colour and the stars' epoch offset once per snapshot; the body → Sun vector only for
/// what is drawn; objects clearly behind the camera skip the rotation into camera space; and
/// named stars and exoplanet hosts far off screen are skipped by a conservative cone test
/// ([`StarCull`]). `build_matches_the_reference` compares it with the previous code bit for bit.
pub(crate) fn build(world: &World, view: &CameraView, viewport: UVec2) -> Snapshot {
    let proj = Projector::new(view, viewport);
    let height = f64::from(viewport.y.max(1));
    let tan_half_fov = (0.5 * view.fov_y_rad).tan();
    let cam = view.pos;
    let sun_pos = world.sun.map(|s| world.cached[s as usize]);
    let sun_color = sun_color();
    let sun_cd = sun_intensity_cd();
    let cull = StarCull::new(view, viewport);
    let near = world.near_stars(&cam);
    let mut snap = Snapshot {
        dominant_frame: world.dominant_frame(&cam),
        nearest_surface: world.nearest_surface_among(&cam, &near),
        ..Snapshot::default()
    };
    // The comets drawn, so ticks re-osculate the stale ones first (review r2 P1).
    let mut shown_comets = Vec::new();
    // A pick for the object at `rel`, whose screen position is `screen` (`proj.screen(rel)`).
    let pick = |snap: &mut Snapshot,
                id: ObjectId,
                rel: DVec3,
                screen: Option<DVec2>,
                kind: ObjectKind| {
        if let Some(p) = screen.filter(|&p| proj.on_screen(p, 0.0)) {
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
        let d = rel.length();
        // Clearly behind the camera (|rel| ≫ the rounding of either test): no screen position,
        // which `Projector::screen` would also find, at the cost of a full rotation.
        let screen = if rel.dot(cull.forward) < -BEHIND_COS * d {
            None
        } else {
            proj.screen(rel)
        };
        pick(&mut snap, id, rel, screen, info.kind);
        let px = proj_radius_px_tan(info.radii_m.max_element(), d, tan_half_fov, height);
        let on_screen = || screen.is_some_and(|p| proj.on_screen(p, POINT_MARGIN_PX));
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
                let sun_rel = || sun_pos.map_or(DVec3::ZERO, |s| s.delta_from(world.cached[k]));
                let lux = if px >= POINT_LOD_PX {
                    let body = body_view(world, id, rel, sun_rel(), px, &cam, sun_cd);
                    snap.bodies.push(body);
                    f64::INFINITY
                } else if on_screen() {
                    let lux = reflected_lux(world, id, rel, sun_rel(), sun_cd);
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
        let px = proj_radius_px_tan(radius, d, tan_half_fov, height);
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

    // Exoplanet hosts on screen, and the brightest named stars on screen, in id order (the star
    // objects are the ids after the cached ones: `World::star_rows`). A star's position is
    // `World::star_position`'s, from the decoded motion, with the epoch offset taken once.
    let years = world.star_years();
    let mut named = Vec::new();
    for row in &world.star_rows {
        let pos_m = row.motion.map(|(p, v)| star_pos_from(p, v, years) * PC);
        let position = || pos_m.map_or(UniversePos::default(), UniversePos::from_meters);
        match row.kind {
            // A pick needs the star on screen, so the cull applies to both kinds.
            _ if pos_m.is_some_and(|p| cull.off_screen(p)) => {}
            ObjectKind::ExoHost => {
                let rel = position().delta_from(cam);
                pick(&mut snap, row.id, rel, proj.screen(rel), row.kind);
            }
            ObjectKind::Star => {
                let rel = position().delta_from(cam);
                let screen = proj.screen(rel);
                if screen.is_some_and(|p| proj.on_screen(p, 0.0)) {
                    let d_pc = rel.length() / PC;
                    let abs_mag = f64::from(world.star_data()[row.index as usize].abs_mag_v);
                    named.push((abs_mag + 5.0 * (d_pc / 10.0).log10(), row.id, rel, screen));
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
    for (_, id, rel, screen) in named {
        pick(&mut snap, id, rel, screen, ObjectKind::Star);
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
    use crate::starindex::star_pos_pc;

    /// The snapshot as `build` made it before B9g's speed-up: the reference for
    /// `build_matches_the_reference`. A verbatim copy of the old code, kept to prove the
    /// speed-up changed nothing: remove it (and that test) after the Wave B gate, rather than
    /// mirror later changes to `build` here (review B9g r0 minor 5).
    pub(crate) fn build_reference(world: &World, view: &CameraView, viewport: UVec2) -> Snapshot {
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

    /// B9g: the faster `build` returns exactly what the previous code did (every f64 to the
    /// bit: `Debug` prints the shortest string that round-trips), over the baked data, from
    /// views that exercise each branch: bodies resolved and as points, the Sun as a surface and
    /// as a point, near stars as surfaces, the named-star cull with wide, narrow and
    /// panoramic fields (`sky-orion-3d` among them), and a camera far beyond the Galaxy.
    #[test]
    fn build_matches_the_reference() {
        use crate::world::load_world_data;
        let data = load_world_data(&uni_data::data_dir().join("baked"));
        let epoch = uni_core::Epoch::parse_iso_utc("2026-11-20T00:00:00Z").expect("date");
        let world = World::new(data, epoch);
        let at = |name: &str| world.position(world.find(name).expect(name));
        let look = |pos: UniversePos, dir: DVec3, fov_deg: f64, roll: f64| CameraView {
            pos,
            orient: DQuat::from_rotation_arc(DVec3::NEG_Z, dir.normalize())
                * DQuat::from_rotation_z(roll),
            fov_y_rad: fov_deg.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let (earth, sun, saturn) = (at("Earth"), at("Sun"), at("Saturn"));
        let radec = |ra: f64, dec: f64| {
            let (ra, dec) = (ra.to_radians(), dec.to_radians());
            DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
        };
        let orion = radec(83.8, -1.2);
        let gc = radec(266.405, -28.936);
        let alpha_cen = at("Rigil Kentaurus");
        let views = [
            // sky-orion-3d and sky-galactic-center (bookmarks/m1.ron #15, #16).
            look(sun.offset(orion * 100.0 * AU), orion, 60.0, 0.0),
            look(sun.offset(gc * 100.0 * AU), gc, 60.0, 0.3),
            // Earth and the Moon from 60 000 km; Saturn and its moons; the Sun close up.
            look(
                earth.offset(DVec3::new(6.0e7, 0.0, 0.0)),
                DVec3::NEG_X,
                60.0,
                0.0,
            ),
            look(saturn.offset(DVec3::new(0.0, 1.5e9, 3.0e8)), -DVec3::Y, 20.0, 1.0),
            look(sun.offset(DVec3::new(3.0e9, 0.0, 0.0)), DVec3::NEG_X, 30.0, 0.0),
            // Near α Cen (stars within 0.1 pc), a narrow and a panoramic field.
            look(alpha_cen.offset(DVec3::new(1.0e12, 0.0, 0.0)), DVec3::NEG_X, 20.0, 0.0),
            look(earth, orion, 1.0, 0.0),
            look(earth, gc, 170.0, 2.0),
            // From 1 Mpc: every star far from the camera's f64 shortcut's comfort.
            look(
                UniversePos::from_meters(DVec3::new(3.0857e22, 1.0e21, -2.0e21)),
                -DVec3::X,
                60.0,
                0.0,
            ),
        ];
        for (i, view) in views.iter().enumerate() {
            for viewport in [UVec2::new(1920, 1080), UVec2::new(640, 1280)] {
                let fast = build(&world, view, viewport);
                let reference = build_reference(&world, view, viewport);
                assert_eq!(
                    format!("{fast:?}"),
                    format!("{reference:?}"),
                    "view {i} at {viewport}"
                );
            }
        }
        // The cull is exercised: from the Orion view it skips 2 624 of the 3 321 named stars
        // (79 %; the frame's corner cone, 49.7° from the axis, holds 18 % of the sky, but the
        // named stars crowd toward Orion and the Milky Way).
        let orion_view = &views[0];
        let cull = StarCull::new(orion_view, UVec2::new(1920, 1080));
        let years = world.star_years();
        let (mut named, mut skipped) = (0, 0);
        for k in world.cached.len()..world.objects().len() {
            let info = &world.objects()[k];
            if info.kind != ObjectKind::Star {
                continue;
            }
            let Some(s) = info.catalog_index.and_then(|i| world.star_data().get(i as usize))
            else {
                continue;
            };
            named += 1;
            skipped += usize::from(cull.off_screen(star_pos_pc(s, years) * PC));
        }
        assert!(skipped * 10 > named * 7, "{skipped} of {named} skipped");
    }

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
