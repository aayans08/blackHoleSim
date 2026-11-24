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
//!
//! Comets (Task C0, lead rulings A.2): a comet is never in `bodies`. Its nucleus is a point
//! below [`POINT_LOD_PX`] (when on screen, as any body), and the comet is in `comets`, which the
//! comet pass (C1) draws, when it is active (heliocentric distance below [`COMET_ACTIVE_AU`]) or
//! its nucleus is resolved, on screen or not (the comet pass culls its own coma and tails).
//! Every comet in `comets` counts as drawn for the comets' catch-up order and
//! `HonestyFlag::CometsCatchingUp`.
//!
//! `suppressed_asteroids` lists the asteroid record indices that uni-sim draws itself (the
//! registry asteroids, plus SBDB's Pluto row, which the ephemeris draws), so the asteroid pass
//! (C2) skips them; it is the same list every frame ([`World::suppressed_asteroid_records`]).
//!
//! `destinations` are the objects that carry a marker and can be aimed at (lead ruling B.3, for
//! C3's aim and C5's markers; amended by the coordinator's ruling at Task C6: soft and
//! uncluttered), on screen or off it: always the Sun, the planets, the dominant planet's major
//! moons ([`MAJOR_MOONS`]; from Earth, the Moon) and TestBH-1 (B.3's, which the amendment leaves
//! as it was); any object that is selected or the goto target; and dwarf planets, comets,
//! asteroids and exoplanet hosts only as one of those or as one of the
//! [`MAX_BRIGHT_DESTINATIONS`] brightest of them all together on screen (C5's overlay drops one
//! of those whose label would overlap another's).

use glam::{DQuat, DVec2, DVec3, UVec2, Vec2, Vec3};
use uni_core::{
    AU, CameraView, Epoch, FrameId, PC, TEFF_SUN_K, UniversePos, blackbody_linear_srgb,
    star_illuminance_lux,
};
use uni_physics::kepler::Conic;
use uni_physics::photometry::{
    Brdf, comet_total_v, disk_phase_function, planet_phase_correction, rho_from_geometric_albedo,
};
use uni_physics::smallbody::heliocentric_state_icrs;

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
/// A comet nearer the Sun than this [au] is active: it has a coma and tails (plan C1, "Only
/// comets with r_h < 5 AU get a coma and tails"), so it is in [`Snapshot::comets`].
pub const COMET_ACTIVE_AU: f64 = 5.0;
/// Dwarf planets, comets, asteroids and exoplanet hosts that are neither selected nor the goto
/// target are destinations only as one of this many brightest of them all together on screen
/// (coordinator ruling at Task C6, amending lead ruling B.3's six per kind).
pub const MAX_BRIGHT_DESTINATIONS: usize = 3;
/// NAIF ids of the major moons, those with their own NASA NSSDCA fact sheet (the moons of A9's
/// `body_facts.ron`, from <https://nssdc.gsfc.nasa.gov/planetary/factsheet/>): the Moon, Phobos,
/// Deimos, the four Galileans, Saturn's eight (Mimas … Iapetus), Uranus's five, Triton and
/// Charon. The dominant planet's are always destinations (lead ruling B.3).
pub const MAJOR_MOONS: [i32; 22] = [
    301, 401, 402, 501, 502, 503, 504, 601, 602, 603, 604, 605, 606, 607, 608, 701, 702, 703, 704,
    705, 801, 901,
];

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

/// A comet the comet pass draws (Task C0; see the module docs for which comets are listed).
#[derive(Clone, Debug)]
pub struct CometView {
    pub id: ObjectId,
    /// Registry name ([`crate::registry::ObjectInfo::name`]).
    pub name: String,
    /// The nucleus relative to the camera [m], ICRS axes.
    pub nucleus_rel_m: DVec3,
    /// Nucleus radius [m]: half SBDB's diameter, else `crate::ASSUMED_RADIUS_M`.
    pub nucleus_radius_m: f64,
    /// SBDB gives no diameter, so the radius is the assumed 2 km (card: "size assumed").
    pub radius_assumed: bool,
    /// SBDB primary SPK-ID (the same number as the comet's `ObjectInfo::naif`, which for a comet
    /// is not a NAIF id: never look comets up in NAIF-keyed tables).
    pub spkid: i32,
    /// The Sun relative to the nucleus [m] (Sun − nucleus), ICRS axes.
    pub sun_rel_m: DVec3,
    /// Heliocentric velocity [m/s], ICRS axes, from the same conic as the position.
    pub helio_vel_mps: DVec3,
    /// Heliocentric distance r_h [au].
    pub r_au: f64,
    /// Distance from the camera Δ [au].
    pub delta_au: f64,
    /// Total (nucleus + coma) V from the camera, m = M1 + 5 log₁₀Δ + K1 log₁₀r (SBDB's law,
    /// `photometry::comet_total_v`); `None` unless SBDB gives both M1 and K1.
    pub total_v: Option<f64>,
    pub m1: Option<f64>,
    pub k1: Option<f64>,
    /// The osculating conic the comet moves on now: heliocentric, J2000 ECLIPTIC (SBDB's frame;
    /// lead ruling A.1), so states come from `uni_physics::smallbody::heliocentric_state_icrs`
    /// and `dust::finson_probstein` takes it as is.
    pub conic: Conic,
    /// The conic's osculation epoch (the card's "two-body from epoch" note).
    pub conic_epoch: Epoch,
    /// Projected radius of the nucleus [internal px] (`proj_radius_px` of its radius): between
    /// `POINT_LOD_PX` and `BLEND_LOD_PX` the nucleus is a world writer in the LOD band
    /// (`pre_mix` exposure and `lens_mask.a`, as a body).
    pub proj_radius_px: f64,
}

/// An object that carries a marker and can be aimed at (lead ruling B.3; see the module docs
/// for the rule), on screen or not.
#[derive(Clone, Copy, Debug)]
pub struct Destination {
    pub id: ObjectId,
    pub kind: ObjectKind,
    /// Relative to the camera [m], ICRS axes.
    pub rel_pos_m: DVec3,
    pub dist_m: f64,
    /// Where its centre projects [internal px], also outside the viewport; `None` behind the
    /// camera.
    pub screen_px: Option<Vec2>,
    /// Projected radius [internal px] of its largest radius.
    pub proj_radius_px: f64,
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
    /// The comets the comet pass draws, brightest `total_v` first (comets without one last), ties
    /// by id (Task C0; module docs).
    pub comets: Vec<CometView>,
    /// Asteroid record indices (into `World::asteroid_records`) that uni-sim draws as bodies or
    /// points (the registry asteroids) or that the ephemeris draws (SBDB's Pluto row), sorted:
    /// the asteroid pass skips them (review I12). The same list every frame.
    pub suppressed_asteroids: Vec<u32>,
    /// The objects that carry a marker, in id order (lead ruling B.3; module docs).
    pub destinations: Vec<Destination>,
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

    /// Whether a sphere of `radius` [m] at `rel` [m] from the camera may reach into the view:
    /// the sphere-vs-frustum plane test (e.g. Akenine-Möller, Haines & Hoffman, *Real-Time
    /// Rendering*, 4th ed., §22.14.1) against the four side planes through the eye and the
    /// camera plane (no near or far plane: the projection is infinite). Exact on every side, off
    /// axis too, where the on-axis pixel radius undercounts the limb's reach by up to 1/cos²
    /// (Task C6, C0 review minor); conservative only near a corner, where a sphere outside the
    /// frame but within `radius` of two planes passes.
    pub(crate) fn sphere_in_frustum(&self, rel: DVec3, radius: f64) -> bool {
        let c = self.to_camera * rel;
        if !(c.is_finite() && radius >= 0.0) {
            return false;
        }
        // Inside: |x| ≤ a·t·(−z) and |y| ≤ t·(−z); each plane's outward signed distance is
        // (±x + a·t·z)/√(1 + (a·t)²), (±y + t·z)/√(1 + t²).
        let (tx, ty) = (self.aspect * self.tan_half_y, self.tan_half_y);
        let (nx, ny) = ((1.0 + tx * tx).sqrt(), (1.0 + ty * ty).sqrt());
        c.z < radius
            && (c.x + tx * c.z) / nx <= radius
            && (-c.x + tx * c.z) / nx <= radius
            && (c.y + ty * c.z) / ny <= radius
            && (-c.y + ty * c.z) / ny <= radius
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
/// the exact path, so the snapshot is the one the exact path alone gives (B9g proved it bit for
/// bit against the previous code; `star_cull_skips_most_named_stars_and_none_on_screen` keeps
/// the cull's own check).
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

/// The objects that may become destinations by brightness (lead ruling B.3, amended at C6):
/// (illuminance at the camera [lx], +∞ for a resolved one; the destination).
type BrightCandidates = Vec<(f64, Destination)>;

/// Keeps the [`MAX_BRIGHT_DESTINATIONS`] brightest of `candidates` (ties by id).
fn brightest(mut candidates: BrightCandidates) -> impl Iterator<Item = Destination> {
    let order = |a: &(f64, Destination), b: &(f64, Destination)| {
        b.0.total_cmp(&a.0).then(a.1.id.cmp(&b.1.id))
    };
    if candidates.len() > MAX_BRIGHT_DESTINATIONS {
        candidates.select_nth_unstable_by(MAX_BRIGHT_DESTINATIONS, order);
        candidates.truncate(MAX_BRIGHT_DESTINATIONS);
    }
    candidates.into_iter().map(|(_, d)| d)
}

/// The registry body whose moons are the dominant planet's: the body of `frame` if it is a
/// planet or dwarf planet, its parent if it is a moon; `None` for the Sun's frame or ROOT.
fn dominant_planet(world: &World, frame: FrameId) -> Option<ObjectId> {
    let id = world
        .ephem_order
        .iter()
        .copied()
        .find(|&id| world.objects[id as usize].frame == Some(frame))?;
    let info = &world.objects[id as usize];
    match info.kind {
        ObjectKind::Planet | ObjectKind::DwarfPlanet => Some(id),
        ObjectKind::Moon => info.parent,
        _ => None,
    }
}

/// Builds the snapshot (see [`World::snapshot`]).
///
/// B9g (Wave B gate, snapshot CPU at `sky-orion-3d`) made it cheaper without changing what it
/// returns: each object's screen position is computed once for its pick and its point; tan(fov/2),
/// the Sun's colour and the stars' epoch offset once per snapshot; the body → Sun vector only for
/// what is drawn; objects clearly behind the camera skip the rotation into camera space; and
/// named stars and exoplanet hosts far off screen are skipped by a conservative cone test
/// ([`StarCull`]). Task C0 added the comets, the suppressed asteroid records and the
/// destinations (module docs).
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
        suppressed_asteroids: world.suppressed_asteroid_records().to_vec(),
        ..Snapshot::default()
    };
    let home = dominant_planet(world, snap.dominant_frame);
    // Selected or the goto target: always a destination.
    let chosen = |id: ObjectId| world.selected == Some(id) || world.goto_target == Some(id);
    // Comets, asteroids and exoplanet hosts on screen that may be destinations by brightness.
    let (mut bright_comets, mut bright_asteroids, mut bright_hosts) = (
        BrightCandidates::new(),
        BrightCandidates::new(),
        BrightCandidates::new(),
    );
    // The comets drawn, so ticks re-osculate the stale ones first (review r2 P1).
    let mut shown_comets = Vec::new();
    // A pick for the object at `rel`, whose screen position is `screen` (`proj.screen(rel)`).
    let pick =
        |snap: &mut Snapshot, id: ObjectId, rel: DVec3, screen: Option<DVec2>, kind: ObjectKind| {
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
        let sun_rel = || sun_pos.map_or(DVec3::ZERO, |s| s.delta_from(world.cached[k]));
        // Illuminance at the camera of what this frame draws of the object [lx] (+∞ resolved);
        // `None` when it draws nothing of it.
        let mut drawn_lux = None;
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
            // Never in `bodies` (lead ruling A.2a): the comet pass draws a resolved nucleus.
            // The point carries the nucleus alone (lead ruling "comet double light", Task C6):
            // the comet pass draws the coma and tails, which hold the rest of SBDB's total
            // magnitude. The comet's prominence (destinations by brightness, catch-up order)
            // stays its total magnitude ([`comet_prominence_lux`]).
            Source::Comet(c) => {
                let resolved = px >= POINT_LOD_PX;
                if resolved {
                    drawn_lux = Some(f64::INFINITY);
                } else if on_screen() {
                    snap.points.push(PointView {
                        id: Some(id),
                        rel_pos_m: rel,
                        illuminance_lux: reflected_lux(world, id, rel, sun_rel(), sun_cd),
                        color: sun_color,
                    });
                    drawn_lux = Some(comet_prominence_lux(world, id, rel, sun_rel(), sun_cd));
                }
                let active = sun_pos.is_some() && sun_rel().length() < COMET_ACTIVE_AU * AU;
                if resolved || active {
                    snap.comets
                        .push(comet_view(world, id, c, rel, sun_rel(), px));
                    // Every listed comet counts as drawn (carry-over 8).
                    let lux = drawn_lux
                        .unwrap_or_else(|| comet_prominence_lux(world, id, rel, sun_rel(), sun_cd));
                    shown_comets.push((c, lux));
                } else if let Some(lux) = drawn_lux {
                    shown_comets.push((c, lux));
                }
            }
            Source::Ephem | Source::Asteroid(_) => {
                if px >= POINT_LOD_PX {
                    let body = body_view(world, id, rel, sun_rel(), px, &cam, sun_cd);
                    snap.bodies.push(body);
                    drawn_lux = Some(f64::INFINITY);
                } else if on_screen() {
                    let lux = reflected_lux(world, id, rel, sun_rel(), sun_cd);
                    snap.points.push(PointView {
                        id: Some(id),
                        rel_pos_m: rel,
                        illuminance_lux: lux,
                        color: sun_color,
                    });
                    drawn_lux = Some(lux);
                }
            }
        }

        // Destinations (lead ruling B.3); built only for the objects that may become one.
        let dest = || Destination {
            id,
            kind: info.kind,
            rel_pos_m: rel,
            dist_m: d,
            screen_px: screen.map(|p| p.as_vec2()),
            proj_radius_px: px,
        };
        let always = match info.kind {
            ObjectKind::Sun | ObjectKind::Planet | ObjectKind::BlackHole => true,
            ObjectKind::DwarfPlanet => false,
            ObjectKind::Moon => {
                home.is_some()
                    && info.parent == home
                    && info.naif.is_some_and(|n| MAJOR_MOONS.contains(&n))
            }
            ObjectKind::Comet | ObjectKind::Asteroid | ObjectKind::Star | ObjectKind::ExoHost => {
                false
            }
        };
        if always || chosen(id) {
            snap.destinations.push(dest());
        } else if let Some(lux) = drawn_lux
            && screen.is_some_and(|p| proj.on_screen(p, 0.0))
        {
            match info.kind {
                ObjectKind::Comet => bright_comets.push((lux, dest())),
                ObjectKind::Asteroid | ObjectKind::DwarfPlanet => {
                    bright_asteroids.push((lux, dest()));
                }
                _ => {}
            }
        }
    }
    world.note_shown_comets(shown_comets);
    snap.comets.sort_by(|a, b| {
        match (a.total_v, b.total_v) {
            (Some(x), Some(y)) => x.total_cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
        .then(a.id.cmp(&b.id))
    });
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
            _ if proj
                .screen(rel)
                .is_some_and(|p| proj.on_screen(p, POINT_MARGIN_PX)) =>
            {
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
    let star_destination = |id: ObjectId, kind: ObjectKind, rel: DVec3, screen: Option<DVec2>| {
        let d = rel.length();
        let radius = world.objects[id as usize].radii_m.max_element();
        Destination {
            id,
            kind,
            rel_pos_m: rel,
            dist_m: d,
            screen_px: screen.map(|p| p.as_vec2()),
            proj_radius_px: proj_radius_px_tan(radius, d, tan_half_fov, height),
        }
    };
    for row in &world.star_rows {
        let pos_m = row.motion.map(|(p, v)| star_pos_from(p, v, years) * PC);
        let position = || pos_m.map_or(UniversePos::default(), UniversePos::from_meters);
        // A chosen star is a destination wherever it is (added after this loop).
        let apparent_v = |rel: DVec3| {
            let abs_mag = f64::from(world.star_data()[row.index as usize].abs_mag_v);
            abs_mag + 5.0 * (rel.length() / PC / 10.0).log10()
        };
        match row.kind {
            // A pick needs the star on screen, so the cull applies to both kinds.
            _ if pos_m.is_some_and(|p| cull.off_screen(p)) => {}
            ObjectKind::ExoHost => {
                let rel = position().delta_from(cam);
                let screen = proj.screen(rel);
                pick(&mut snap, row.id, rel, screen, row.kind);
                if !chosen(row.id) && screen.is_some_and(|p| proj.on_screen(p, 0.0)) {
                    let lux = star_illuminance_lux(apparent_v(rel));
                    bright_hosts.push((lux, star_destination(row.id, row.kind, rel, screen)));
                }
            }
            ObjectKind::Star => {
                let rel = position().delta_from(cam);
                let screen = proj.screen(rel);
                if screen.is_some_and(|p| proj.on_screen(p, 0.0)) {
                    named.push((apparent_v(rel), row.id, rel, screen));
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

    // A chosen star or exoplanet host is a destination on screen or off it.
    let mut chosen_stars: Vec<ObjectId> = [world.selected, world.goto_target]
        .into_iter()
        .flatten()
        .filter(|&id| (id as usize) >= world.cached.len() && (id as usize) < world.objects.len())
        .collect();
    chosen_stars.dedup();
    for id in chosen_stars {
        let rel = world.position(id).delta_from(cam);
        if rel.is_finite() {
            let kind = world.objects[id as usize].kind;
            snap.destinations
                .push(star_destination(id, kind, rel, proj.screen(rel)));
        }
    }
    // One pool for every "other" kind (coordinator ruling at Task C6).
    bright_comets.extend(bright_asteroids);
    bright_comets.extend(bright_hosts);
    snap.destinations.extend(brightest(bright_comets));
    snap.destinations.sort_by_key(|d| d.id);
    snap
}

/// The comet pass's view of comet `id` (comet orbit `c`): `rel` camera → nucleus, `sun_rel`
/// nucleus → Sun [m], `px` the nucleus's projected radius.
fn comet_view(
    world: &World,
    id: ObjectId,
    c: usize,
    rel: DVec3,
    sun_rel: DVec3,
    px: f64,
) -> CometView {
    let (record, conic, conic_epoch) = world.comet_parts(c);
    let helio_vel_mps = heliocentric_state_icrs(&conic, world.epoch).1;
    let r_au = sun_rel.length() / AU;
    let delta_au = rel.length() / AU;
    let total_v = record
        .m1
        .zip(record.k1)
        .map(|(m1, k1)| comet_total_v(m1, k1, r_au, delta_au));
    CometView {
        id,
        name: world.objects[id as usize].name.clone(),
        nucleus_rel_m: rel,
        nucleus_radius_m: world.physical[id as usize].mean_radius_m,
        radius_assumed: !record.diameter_km.is_some_and(|d| d.is_finite() && d > 0.0),
        spkid: record.spkid,
        sun_rel_m: sun_rel,
        helio_vel_mps,
        r_au,
        delta_au,
        total_v,
        m1: record.m1,
        k1: record.k1,
        conic,
        conic_epoch,
        proj_radius_px: px,
    }
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
        sun_illuminance_lux: if r_sun > 0.0 {
            sun_cd / (r_sun * r_sun)
        } else {
            0.0
        },
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
    let sun_angle = (world.objects[sun as usize].radii_m.x / r_sun)
        .min(1.0)
        .asin();
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
/// radius.
///
/// A comet is its nucleus alone (lead ruling "comet double light", Task C6): a p_V 0.04
/// lunar-Lambert (L = 0.9) body of its (assumed) radius with no planetary phase correction, the
/// model of `uni_planet::comet::nucleus::nucleus_illuminance_lux` (uni-app's
/// `comet_point_is_the_nucleus_alone` pins the two equal). The comet pass draws the coma and
/// tails, which carry the rest of SBDB's total magnitude ([`comet_prominence_lux`]).
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
    let physical = &world.physical[id as usize];
    let brdf = physical.brdf;
    let phase = angle(sun_rel, -rel);
    let weight = match brdf {
        Brdf::Lambert => 2.0 / 3.0,
        Brdf::LunarLambert { l } => l + 2.0 * (1.0 - l) / 3.0,
    };
    let rho = rho_from_geometric_albedo(brdf, physical.geometric_albedo);
    // A comet's `naif` holds its SBDB SPK-ID (ruling C.9), never a planet's id: no correction.
    let is_comet = matches!(physical.source, Source::Comet(_));
    let correction = world.objects[id as usize]
        .naif
        .filter(|_| !is_comet)
        .map_or(1.0, |n| planet_phase_correction(n, brdf, phase))
        .min(MAX_PHASE_CORRECTION);
    let ratio = physical.mean_radius_m / d;
    rho * weight
        * (sun_cd / (r_sun * r_sun))
        * ratio
        * ratio
        * disk_phase_function(brdf, phase)
        * correction
}

/// A comet's prominence [lx] at the camera: its total (nucleus + coma) magnitude from SBDB's
/// M1, K1 (`photometry::comet_total_v`), or its nucleus ([`reflected_lux`]) without them. It
/// ranks comets (destinations by brightness, the catch-up order); the drawn point is the
/// nucleus alone.
pub(crate) fn comet_prominence_lux(
    world: &World,
    id: ObjectId,
    rel: DVec3,
    sun_rel: DVec3,
    sun_cd: f64,
) -> f64 {
    let (d, r_sun) = (rel.length(), sun_rel.length());
    match world.comet_magnitude(id) {
        Some((m1, k1)) if d > 0.0 && r_sun > 0.0 => {
            star_illuminance_lux(comet_total_v(m1, k1, r_sun / AU, d / AU))
        }
        _ => reflected_lux(world, id, rel, sun_rel, sun_cd),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::starindex::star_pos_pc;

    /// B9g's named-star cull (carry-over 1 of Task C0 kept this assertion when it removed the
    /// one-off `build_reference` comparison, as its doc asked after the Wave B gate): from the
    /// `sky-orion-3d` view it skips most of the named stars (2 624 of the 3 321 at B9g: 79 %; the
    /// frame's corner cone, 49.7° from the axis, holds 18 % of the sky, but the named stars crowd
    /// toward Orion and the Milky Way), and it never skips a star that is on screen.
    #[test]
    fn star_cull_skips_most_named_stars_and_none_on_screen() {
        use crate::world::load_world_data;
        let data = load_world_data(&uni_data::data_dir().join("baked"));
        let epoch = uni_core::Epoch::parse_iso_utc("2026-11-20T00:00:00Z").expect("date");
        let world = World::new(data, epoch);
        let sun = world.position(world.find("Sun").expect("Sun"));
        let (ra, dec) = (83.8_f64.to_radians(), (-1.2_f64).to_radians());
        let orion = DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
        let view = CameraView {
            pos: sun.offset(orion * 100.0 * AU),
            orient: DQuat::from_rotation_arc(DVec3::NEG_Z, orion),
            fov_y_rad: 60f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let viewport = UVec2::new(1920, 1080);
        let cull = StarCull::new(&view, viewport);
        let proj = Projector::new(&view, viewport);
        let years = world.star_years();
        let (mut named, mut skipped) = (0, 0);
        for k in world.cached.len()..world.objects().len() {
            let info = &world.objects()[k];
            if info.kind != ObjectKind::Star {
                continue;
            }
            let Some(index) = info.catalog_index else {
                continue;
            };
            let Some(s) = world.star_data().get(index as usize) else {
                continue;
            };
            named += 1;
            if cull.off_screen(star_pos_pc(s, years) * PC) {
                skipped += 1;
                let rel = world.star_position(index).delta_from(view.pos);
                assert!(
                    !proj.screen(rel).is_some_and(|p| proj.on_screen(p, 0.0)),
                    "{} is on screen but culled",
                    info.name
                );
            }
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
        assert!(
            (far / (540.0e-6 / (fov / 2.0).tan()) - 1.0).abs() < 1e-11,
            "{far}"
        );
    }
}
