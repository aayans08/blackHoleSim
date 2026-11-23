//! uni-sim: world state and time controller from M1; streaming/LOD scheduler and job system as
//! the universe grows (M2+).
//!
//! M1 Task A4: [`time`] (warp and pause),
//! [`registry`] (every named object), [`world`] (frames, cached states, data loading),
//! [`snapshot`] (what the renderer draws this frame), [`honesty`] (world-state labels) and
//! [`starindex`] (nearest catalogue stars).

pub mod honesty;
pub mod registry;
pub mod snapshot;
pub mod starindex;
pub mod time;
pub mod world;

// The assumed defaults B1 must use for its own p_V / radius fallbacks (one source of truth).
pub use world::{ASSUMED_ASTEROID_ALBEDO, ASSUMED_RADIUS_M, COMET_ALBEDO, DEFAULT_MOON_ALBEDO};
