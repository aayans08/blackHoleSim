//! The per-frame view of the world that the renderer draws (M1 Task A4). Positions are
//! camera-relative in f64 [m], ICRS axes. A0b stub; A4 implements.

use glam::{DQuat, DVec3, Vec2, Vec3};
use uni_core::FrameId;

use crate::registry::ObjectId;

/// Below this projected radius [px] a body is drawn as a point.
pub const POINT_LOD_PX: f64 = 1.5;
/// Between [`POINT_LOD_PX`] and this projected radius [px] a resolved body blends its exposure
/// from star exposure to scene exposure.
pub const BLEND_LOD_PX: f64 = 4.0;

/// Projected radius [px] of a sphere of `radius_m` at `dist_m`:
/// (h/2)·tan(asin(R/d))/tan(fov/2); R ≥ d → +∞ (review m5).
pub fn proj_radius_px(_radius_m: f64, _dist_m: f64, _fov_y_rad: f64, _height_px: f64) -> f64 {
    todo!("A4")
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
