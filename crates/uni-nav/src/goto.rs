//! The animated goto (Task C3; Task C0 stub).
//!
//! Lead ruling on the C0 context (owner point 2), after the owner-approved preview
//! (`72-nav.js`):
//! - **Turn first** to face the target, with the quintic ease x³(6x² − 15x + 10) over
//!   [`turn_duration_s`] = 0.40 + 0.55·turn/π s (it replaces the plan's
//!   `GOTO_TURN_RATE_DEG_S`).
//! - **Then travel** for [`GOTO_DURATION_S`] = 4.5 s from any distance: ln(d − d_stop) moves along
//!   the distance share of [`goto_profile`], whose speed share is
//!   smoothstep(0, [`GOTO_ACCEL_FRAC`], x)·(1 − smoothstep(1 − [`GOTO_DECEL_FRAC`], 1, x)) and whose
//!   distance share is its integral, normalised (accelerate, cruise, decelerate, with parallax).
//!   Short hops (< 10 × d_stop) use the same profile. The whole goto takes
//!   `turn_duration_s(turn) + GOTO_DURATION_S`, independent of the distance
//!   (`goto_duration_independent_of_distance`, review m20).
//! - **Arrival** when d − d_stop < [`ARRIVAL_FRAC`]·d_stop: free flight, `DropOut::Arrived`.
//! - **Stand-off** ([`standoff_distance`], plan C3): bodies 4 × their largest radius, or 1.5 × the
//!   atmosphere top if larger; the Sun and stars 1 AU × √(L/L☉) but at least 10 R★; comets
//!   10⁵ km; asteroids 10 × their radius (at least 50 km); exoplanet hosts as stars; TestBH-1
//!   100 GM/c².

use uni_sim::registry::ObjectInfo;
use uni_sim::world::World;

/// Travel time of a goto after its turn [s], independent of the distance (lead ruling, owner
/// point 2).
pub const GOTO_DURATION_S: f64 = 4.5;
/// Share of the travel spent accelerating (the preview's smoothstep(0, 0.30, x)).
pub const GOTO_ACCEL_FRAC: f64 = 0.30;
/// Share of the travel spent decelerating (the preview's 1 − smoothstep(0.64, 1, x)).
pub const GOTO_DECEL_FRAC: f64 = 0.36;
/// The turn's fixed part [s] (0.40 + 0.55·turn/π).
pub const GOTO_TURN_BASE_S: f64 = 0.40;
/// The turn's part per π of turn [s].
pub const GOTO_TURN_PER_PI_S: f64 = 0.55;
/// A goto has arrived when the distance left to the stand-off point is below this share of the
/// stand-off distance (plan C3).
pub const ARRIVAL_FRAC: f64 = 1.0e-3;

/// Duration [s] of the turn through `turn_rad` (0 ..= π) before the travel: 0.40 + 0.55·turn/π.
pub fn turn_duration_s(turn_rad: f64) -> f64 {
    let _ = turn_rad;
    todo!("C3")
}

/// The quintic ease x³(6x² − 15x + 10) of the turn, for x in 0 ..= 1.
pub fn quintic_ease(x: f64) -> f64 {
    let _ = x;
    todo!("C3")
}

/// The travel profile at x = elapsed ÷ [`GOTO_DURATION_S`] (0 ..= 1): (distance share 0 ..= 1,
/// speed share 0 ..= 1). See the module docs.
pub fn goto_profile(x: f64) -> (f64, f64) {
    let _ = x;
    todo!("C3")
}

/// The distance [m] from `info`'s centre at which a goto stops (module docs).
pub fn standoff_distance(info: &ObjectInfo, world: &World) -> f64 {
    let _ = (info, world);
    todo!("C3")
}
