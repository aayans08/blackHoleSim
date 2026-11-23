//! Clicking on objects (Task C3; Task C0 stub).
//!
//! The nearest `Snapshot::pick` item within the pick radius of the cursor, ties by priority
//! (bodies and the Sun > comets > named stars and exoplanet hosts > asteroids; TestBH-1 counts as
//! a body: `PickItem::priority`), then by distance. The app converts the cursor from window
//! pixels (`InputState::cursor_px`, C3) to internal pixels (× internal ÷ output).

use glam::Vec2;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::PickItem;

/// The pick radius [px] at 1080 internal lines (plan C3); [`pick_radius_px`] scales it.
pub const PICK_RADIUS_PX_AT_1080: f32 = 12.0;

/// The pick radius [internal px] at an internal height of `internal_lines`.
pub fn pick_radius_px(internal_lines: u32) -> f32 {
    let _ = internal_lines;
    todo!("C3")
}

/// The item of `items` nearest `cursor_px` within `radius_px`, by priority on ties (module docs).
pub fn nearest_pick(items: &[PickItem], cursor_px: Vec2, radius_px: f32) -> Option<ObjectId> {
    let _ = (items, cursor_px, radius_px);
    todo!("C3")
}
