//! The world: data loading, the registry, the frame tree and per-tick cached states (M1 Task
//! A4). A0b stub; A4 implements.

use std::path::Path;

use glam::{DVec3, UVec2};
use uni_core::{CameraView, Epoch, FrameId, FrameTree, Lens, UniversePos};
use uni_data::{AsteroidRecord, BodyFacts, Comet, ExoSystem, SatRecord, Star3, StarNames};

use crate::honesty::HonestyFlag;
use crate::registry::{ObjectId, ObjectInfo};
use crate::snapshot::Snapshot;
use crate::time::TimeController;

/// Every baked data file the world uses; a missing file leaves its field empty and adds one
/// warning.
pub struct WorldData {
    pub stars: Vec<Star3>,
    pub star_names: Option<StarNames>,
    pub sats: Vec<SatRecord>,
    pub comets: Vec<Comet>,
    pub asteroids: Vec<AsteroidRecord>,
    pub asteroid_names: Vec<String>,
    pub exo: Vec<ExoSystem>,
    pub facts: Vec<BodyFacts>,
    pub warnings: Vec<String>,
}

/// Loads every baked file from `dir`. Never panics; each missing file gives one warning.
pub fn load_world_data(_dir: &Path) -> WorldData {
    todo!("A4")
}

/// The simulated Solar System and sky.
pub struct World {
    pub time: TimeController,
    /// Placeholder for the private state; A4 replaces it.
    _private: (),
}

impl World {
    pub fn new(_data: WorldData, _epoch: Epoch) -> Self {
        todo!("A4")
    }

    /// Advances time by `dt_real_s` of wall time and refreshes the cached states.
    pub fn tick(&mut self, _dt_real_s: f64) {
        todo!("A4")
    }

    pub fn epoch(&self) -> Epoch {
        todo!("A4")
    }

    pub fn frames(&self) -> &FrameTree {
        todo!("A4")
    }

    pub fn objects(&self) -> &[ObjectInfo] {
        todo!("A4")
    }

    /// Case-insensitive lookup by name.
    pub fn find(&self, _name: &str) -> Option<ObjectId> {
        todo!("A4")
    }

    pub fn position(&self, _id: ObjectId) -> UniversePos {
        todo!("A4")
    }

    /// Velocity [m/s], root axes.
    pub fn velocity(&self, _id: ObjectId) -> DVec3 {
        todo!("A4")
    }

    /// The deepest frame whose Hill sphere contains `pos`.
    pub fn dominant_frame(&self, _pos: &UniversePos) -> FrameId {
        todo!("A4")
    }

    /// The nearest body surface and the distance to it [m].
    pub fn nearest_surface(&self, _pos: &UniversePos) -> Option<(ObjectId, f64)> {
        todo!("A4")
    }

    /// What to draw from `view` at internal size `viewport` [px].
    pub fn snapshot(&self, _view: &CameraView, _viewport: UVec2) -> Snapshot {
        todo!("A4")
    }

    pub fn honesty(&self, _view: &CameraView) -> Vec<HonestyFlag> {
        todo!("A4")
    }

    pub fn lenses(&self) -> &[Lens] {
        todo!("A4")
    }

    /// AT-HYG stars, then archive-only exoplanet hosts (review I11).
    pub fn star_data(&self) -> &[Star3] {
        todo!("A4")
    }

    pub fn exo_systems(&self) -> &[ExoSystem] {
        todo!("A4")
    }
}
