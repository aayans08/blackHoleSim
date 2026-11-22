//! uni-procgen: procedural Milky Way. M1 Task A5 (A0b stubs; bodies `todo!("A5")` until it
//! lands): the galaxy model v0 ([`galaxy`], with its WGSL twin `shaders/galaxy.wgsl`) and the
//! local luminosity function ([`lf`]). Sector octree and stellar populations in M4; planetary
//! systems in M5.

pub mod galaxy;
pub mod lf;

// A0b pre-adds every Wave A manifest entry (review I8), so a crate can be declared before the task
// that uses it lands. This line keeps `cargo::unused_dependencies` quiet until then; delete it
// once `uni_core` is used.
use uni_core as _;
