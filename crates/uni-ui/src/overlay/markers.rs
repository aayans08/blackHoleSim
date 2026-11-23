//! Destination markers (Task C5; Task C0 wrote the types).
//!
//! One marker per `Snapshot::destinations` entry that is on screen: a glyph per kind, the name
//! and a mono distance; grey, cyan when selected; no ring over a disk larger than
//! `theme::MARKER_MAX_DISK_PT` (the label only); the aim ring while the aim fills.

use glam::Vec2;
use uni_sim::registry::ObjectId;

/// What a marker stands for (its glyph). The Sun is a `Star`, dwarf planets are `Planet`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    Planet,
    Moon,
    Comet,
    Asteroid,
    /// An exoplanet host: ring and centre dot.
    ExoHost,
    Star,
    BlackHole,
}

/// One marker, in output points.
#[derive(Clone, Debug)]
pub struct Marker {
    /// The object it marks.
    pub id: ObjectId,
    pub pt: Vec2,
    /// `None` where the label would overlap another.
    pub label: Option<String>,
    pub kind: MarkerKind,
    pub selected: bool,
    /// The mono sub-line (distance).
    pub sub: Option<String>,
    /// The aim ring's fill, 0 ..= 1 (0: no ring).
    pub aim_fill: f32,
    /// The goto's destination.
    pub target: bool,
    /// Angle from the view axis [rad] (markers far off axis are fainter).
    pub off_axis_rad: f32,
}
