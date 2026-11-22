//! Every named object the world knows (M1 Task A4).

use glam::DVec3;
use uni_core::FrameId;

/// Index of an object in the world's registry.
pub type ObjectId = u32;

/// What kind of object an entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    Sun,
    Planet,
    DwarfPlanet,
    Moon,
    Asteroid,
    Comet,
    Star,
    ExoHost,
    BlackHole,
}

/// One registry entry.
#[derive(Clone, Debug)]
pub struct ObjectInfo {
    pub id: ObjectId,
    /// Unique case-insensitively.
    pub name: String,
    pub kind: ObjectKind,
    /// NAIF id, for Solar System bodies.
    pub naif: Option<i32>,
    /// Index into the star data, for stars and exoplanet hosts.
    pub catalog_index: Option<u32>,
    pub parent: Option<ObjectId>,
    /// Triaxial radii [m], a ≥ b ≥ c.
    pub radii_m: DVec3,
    /// GM [m³/s²].
    pub gm_m3s2: f64,
    /// The object's own frame, if it has one.
    pub frame: Option<FrameId>,
    /// Hill radius about its parent [m].
    pub hill_radius_m: f64,
}
