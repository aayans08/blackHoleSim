//! uni-sim: world state and time controller from M1; streaming/LOD scheduler and job system as
//! the universe grows (M2+).
//!
//! M1 Task A4 (A0b stubs; bodies `todo!("A4")` until it lands): [`time`] (warp and pause),
//! [`registry`] (every named object), [`world`] (frames, cached states, data loading),
//! [`snapshot`] (what the renderer draws this frame), [`honesty`] (world-state labels) and
//! [`starindex`] (nearest catalogue stars).

pub mod honesty;
pub mod registry;
pub mod snapshot;
pub mod starindex;
pub mod time;
pub mod world;

// A0b pre-adds every Wave A manifest entry (review I8), so a crate can be declared before the task
// that uses it lands. These lines keep `cargo::unused_dependencies` quiet until then; delete each
// one once its crate is used.
use ron as _;
use serde as _;
use uni_physics as _;
