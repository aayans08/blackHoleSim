//! The dust and ion tails (Task C1; Task C0 stub).
//!
//! - **Dust:** `uni_physics::dust::finson_probstein` on `CometView::conic` (J2000 ecliptic, as it
//!   expects), β ∈ [0.001, 0.9] × age ≤ 60 d, drawn as soft Gaussian sprites; total tail light
//!   0.5 × the coma's (v0; card note `theme::NOTE_TAILS`).
//! - **Ion:** straight along û = normalize(400 km/s · r̂_h − v_comet) (Brandt 1968; ruling 8),
//!   0.3 AU / r_h long (at most 1 AU), 10⁵ km wide, 0.2 × the coma's light (v0).
//! - **Schedule:** tails for at most `MAX_TAILED_COMETS` comets (the brightest `total_v` on
//!   screen), each recomputed when the epoch has moved more than 1 h of sim time since its last
//!   refresh but never more often than every `TAIL_REFRESH_WALL_S` of wall time, round-robin
//!   within ≤ 1 ms of CPU per frame. The refresh times are kept by `ObjectId`, because
//!   `Snapshot::comets` changes order and membership from frame to frame (ruling C.8).

use std::collections::HashMap;

use glam::DVec3;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::CometView;

/// Direction of the ion tail (unit, ICRS) from the Sun → nucleus geometry: `sun_rel` is the Sun
/// relative to the nucleus [m] (`CometView::sun_rel_m`), `helio_vel` the comet's heliocentric
/// velocity [m/s]. Anti-sunward, aberrated by the 400 km/s solar wind (module docs).
pub fn ion_tail_dir(sun_rel: DVec3, helio_vel: DVec3) -> DVec3 {
    let _ = (sun_rel, helio_vel);
    todo!("C1")
}

/// Which of `comets` (indices into it) get a tail refresh this frame at wall time `wall_s` [s],
/// given each comet's last refresh `last_refresh` [wall s] by id (module docs).
pub fn tail_schedule(
    comets: &[CometView],
    last_refresh: &HashMap<ObjectId, f64>,
    wall_s: f64,
) -> Vec<usize> {
    let _ = (comets, last_refresh, wall_s);
    todo!("C1")
}
