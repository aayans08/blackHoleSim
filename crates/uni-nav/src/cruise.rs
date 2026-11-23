//! Cruise (Task C3; Task C0 stub; design §5.6), labelled `CRUISE · FTL · NON-PHYSICAL · GRAVITY
//! IGNORED` (`uni_ui::theme::CRUISE_LABEL`).
//!
//! v = 10^cruise_log10 s⁻¹ × d_surface (at least [`MIN_CRUISE_SPEED_MPS`]) along the camera's −Z;
//! the wheel moves `cruise_log10` within [`CRUISE_LOG10_RANGE`]. Cruise drops out to free flight
//! at [`DROPOUT_SPEED_MPS`] on crossing an `ObjectInfo::atmosphere_top_m`, within
//! [`NEAR_STAR_DROPOUT_RADII`] photosphere radii of the Sun or a catalogue star, or within
//! [`BLACK_HOLE_DROPOUT_RS`] Schwarzschild radii of a black hole; one frame's step never crosses
//! a drop-out boundary.
//!
//! How a frame's step is bounded (each bound only shortens it):
//! - **Drop-out spheres.** The step ends on the first boundary sphere the ray enters; that is the
//!   drop-out. A camera already inside a sphere may cruise outward, but a step with any inward
//!   component drops out at once without moving (it is "within" the boundary).
//! - **Surfaces** (Review Focus 2: travel modes stop outside). At most [`SURFACE_STEP_FRAC`] of
//!   d_surface, the distance to the nearest surface, which no surface is nearer than; so the
//!   camera only ever approaches an airless surface, never reaches it. This binds only when
//!   10^cruise_log10 × dt > ½ (at the fastest setting, frames longer than 0.05 s).
//! - **Unseen stars.** At most [`MAX_CRUISE_STEP_M`] (0.05 pc), half the radius within which
//!   uni-sim finds catalogue stars for the snapshot (0.1 pc). Any star whose drop-out sphere a
//!   step could reach is therefore in the snapshot (as a surface or a point), so its sphere is
//!   tested. A near star the snapshot shows without an id (registered at the next tick) limits
//!   the step to half the distance to it. The cap binds only far from every surface (d_surface
//!   above 0.3 pc at 60 fps and the fastest setting; 30 pc at the default).
//!
//! Spheres are tested where their centres are now; their motion relative to the camera's anchor
//! during the coming tick is not swept. The camera rides the dominant frame (an atmosphere's or
//! the Sun's own body, TestBH-1's frame), so that motion is zero for every boundary but the
//! frameless catalogue stars, which move a few tens of km/s (far more under time warp); a star
//! that sweeps over the camera still drops cruise out on the next frame that moves inward.

use glam::DVec3;
use uni_core::{C, PC, UniversePos};
use uni_sim::registry::{ObjectId, ObjectKind};
use uni_sim::snapshot::Snapshot;
use uni_sim::world::World;

use crate::camera::FlyCamera;
use crate::nav::DropOut;

/// Range of `Navigator::cruise_log10` [log10 of 1/s] (plan C3).
pub const CRUISE_LOG10_RANGE: (f64, f64) = (-3.0, 1.0);
/// `Navigator::cruise_log10` of a new navigator: a tenth of the distance per second.
pub const CRUISE_LOG10_DEFAULT: f64 = -1.0;
/// Cruise never goes slower than this [m/s] (plan C3).
pub const MIN_CRUISE_SPEED_MPS: f64 = 1.0;
/// Speed [m/s] the camera is left at when cruise drops out (plan C3: 1 km/s).
pub const DROPOUT_SPEED_MPS: f64 = 1.0e3;
/// Cruise drops out within this many photosphere radii of the Sun or a catalogue star.
pub const NEAR_STAR_DROPOUT_RADII: f64 = 10.0;
/// Cruise drops out within this many Schwarzschild radii of a black hole.
pub const BLACK_HOLE_DROPOUT_RS: f64 = 100.0;
/// Largest share of the distance to the nearest surface one frame's step may cover (module
/// docs).
pub const SURFACE_STEP_FRAC: f64 = 0.5;
/// Radius [m] within which uni-sim finds the catalogue stars it draws as surfaces or points
/// (`uni_sim::world`'s `NEAR_STAR_PC` = 0.1 pc, plan A4).
const NEAR_STAR_SEARCH_M: f64 = 0.1 * PC;
/// Longest step [m] of one frame (module docs): half of `NEAR_STAR_SEARCH_M`.
pub const MAX_CRUISE_STEP_M: f64 = 0.5 * NEAR_STAR_SEARCH_M;

/// Cruise speed [m/s] at `d_surface_m` from the nearest surface: 10^cruise_log10 × d_surface,
/// at least [`MIN_CRUISE_SPEED_MPS`] (so also on or inside a surface, or with no number).
pub fn cruise_speed(d_surface_m: f64, cruise_log10: f64) -> f64 {
    let v = 10_f64.powf(cruise_log10) * d_surface_m;
    if v > MIN_CRUISE_SPEED_MPS {
        v
    } else {
        MIN_CRUISE_SPEED_MPS
    }
}

/// What one frame of cruise did.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CruiseStep {
    /// The camera's speed this frame [m/s]: the step ÷ dt, or the cruise speed when dt is 0.
    pub(crate) speed_mps: f64,
    pub(crate) dropout: Option<DropOut>,
}

/// A drop-out boundary relative to the camera.
struct Sphere {
    /// Camera − centre [m].
    rel: DVec3,
    radius: f64,
    reason: DropOut,
}

/// Moves `cam` one frame of `dt` seconds along its −Z at the cruise speed, bounded as the
/// module docs say; reports a drop-out (the caller switches to free flight).
pub(crate) fn advance(
    cam: &mut FlyCamera,
    world: &World,
    snap: &Snapshot,
    cruise_log10: f64,
    dt: f64,
) -> CruiseStep {
    let pos = cam.view.pos;
    let nearest = world.nearest_surface(&pos);
    let d_surface = nearest.map_or(f64::NAN, |(_, altitude)| altitude);
    let speed = cruise_speed(d_surface, cruise_log10);
    let dir = (cam.view.orient * DVec3::NEG_Z).normalize();

    let mut len = (speed * dt).min(MAX_CRUISE_STEP_M);
    if d_surface > 0.0 {
        len = len.min(SURFACE_STEP_FRAC * d_surface);
    }
    // Near stars the snapshot found but the registry has not named yet (module docs).
    for point in snap.points.iter().filter(|p| p.id.is_none()) {
        let d = point.rel_pos_m.length();
        if d.is_finite() {
            len = len.min(0.5 * d);
        }
    }
    let len = len.max(0.0);

    let mut step = len;
    let mut dropout = None;
    for sphere in spheres(world, snap, nearest.map(|(id, _)| id), &pos) {
        if let Some(t) = first_contact(sphere.rel, dir, sphere.radius, step) {
            step = t;
            dropout = Some(sphere.reason);
        }
    }
    cam.translate(dir * step);
    let speed_mps = if dt > 0.0 { step / dt } else { speed };
    CruiseStep { speed_mps, dropout }
}

/// Every drop-out boundary that one step from `pos` could reach: the atmosphere tops, the Sun,
/// the black holes, and the catalogue stars the snapshot shows near the camera (as surfaces or
/// as points with an id) or that `nearest` names.
fn spheres(
    world: &World,
    snap: &Snapshot,
    nearest: Option<ObjectId>,
    pos: &UniversePos,
) -> Vec<Sphere> {
    let objects = world.objects();
    let mut out = Vec::new();
    let mut push = |id: ObjectId, radius: f64, reason: DropOut| {
        out.push(Sphere {
            rel: pos.delta_from(world.position(id)),
            radius,
            reason,
        });
    };
    for info in objects {
        if let Some(top) = info.atmosphere_top_m {
            push(info.id, top, DropOut::Atmosphere);
        }
        match info.kind {
            ObjectKind::Sun => {
                push(info.id, NEAR_STAR_DROPOUT_RADII * info.radii_m.x, DropOut::NearStar);
            }
            // r_s = 2GM/c² (Schwarzschild 1916).
            ObjectKind::BlackHole => {
                let r_s = 2.0 * info.gm_m3s2 / (C * C);
                push(info.id, BLACK_HOLE_DROPOUT_RS * r_s, DropOut::BlackHole);
            }
            _ => {}
        }
    }
    let mut stars: Vec<ObjectId> = snap
        .star_surfaces
        .iter()
        .map(|s| s.id)
        .chain(snap.points.iter().filter_map(|p| p.id))
        .chain(nearest)
        .filter(|&id| {
            objects
                .get(id as usize)
                .is_some_and(|o| matches!(o.kind, ObjectKind::Star | ObjectKind::ExoHost))
        })
        .collect();
    stars.sort_unstable();
    stars.dedup();
    for id in stars {
        let radius = objects[id as usize].radii_m.x;
        push(id, NEAR_STAR_DROPOUT_RADII * radius, DropOut::NearStar);
    }
    out
}

/// Distance along the unit ray `dir` from `rel` (camera − centre) at which it enters the sphere
/// of `radius`, if that is within `max_len`. From inside (or on) the sphere: 0 when the ray has
/// any inward component, else `None`.
fn first_contact(rel: DVec3, dir: DVec3, radius: f64, max_len: f64) -> Option<f64> {
    if !(rel.is_finite() && radius > 0.0) {
        return None;
    }
    let b = rel.dot(dir);
    if rel.length() <= radius {
        return (b < 0.0).then_some(0.0);
    }
    if b >= 0.0 {
        return None;
    }
    // The ray's closest approach to the centre, squared: |rel − b·dir|², which keeps its
    // precision when |rel| ≫ radius (unlike |rel|² − b²).
    let miss2 = (rel - dir * b).length_squared();
    let r2 = radius * radius;
    if miss2 > r2 {
        return None;
    }
    let t = (-b - (r2 - miss2).sqrt()).max(0.0);
    (t <= max_len).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cruise_speed_scales_with_distance() {
        assert!((cruise_speed(1e6, 0.0) - 1e6).abs() < 1e-6);
        assert!((cruise_speed(1e6, -1.0) - 1e5).abs() < 1e-6);
        assert!((cruise_speed(3e11, 1.0) - 3e12).abs() < 1.0);
        // Never below MIN_CRUISE_SPEED_MPS: on, inside or near a surface, or with no number.
        assert_eq!(cruise_speed(0.5, 0.0), MIN_CRUISE_SPEED_MPS);
        assert_eq!(cruise_speed(-10.0, 0.0), MIN_CRUISE_SPEED_MPS);
        assert_eq!(cruise_speed(f64::NAN, 0.0), MIN_CRUISE_SPEED_MPS);
    }
}
