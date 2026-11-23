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

/// Internal height [lines] at which the pick radius is [`PICK_RADIUS_PX_AT_1080`].
const REFERENCE_LINES: f32 = 1080.0;

/// The pick radius [internal px] at an internal height of `internal_lines`: 12 px at 1080
/// lines, in proportion otherwise, so it covers the same share of the screen at any size.
pub fn pick_radius_px(internal_lines: u32) -> f32 {
    PICK_RADIUS_PX_AT_1080 * internal_lines as f32 / REFERENCE_LINES
}

/// The item of `items` within `radius_px` of `cursor_px` (on the radius counts) with the
/// highest `priority`; among equal priorities the nearest to the cursor, then the nearest to the
/// camera (`dist_m`). `None` when no item is that close or the cursor is not a number.
pub fn nearest_pick(items: &[PickItem], cursor_px: Vec2, radius_px: f32) -> Option<ObjectId> {
    if !cursor_px.is_finite() {
        return None;
    }
    items
        .iter()
        .filter_map(|item| {
            let px = item.screen_px.distance(cursor_px);
            (px <= radius_px).then_some((item, px))
        })
        .min_by(|(a, pa), (b, pb)| {
            b.priority
                .cmp(&a.priority)
                .then(pa.total_cmp(pb))
                .then(a.dist_m.total_cmp(&b.dist_m))
        })
        .map(|(item, _)| item.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Priorities as `uni_sim::snapshot` gives them: bodies 3, comets 2, stars 1, asteroids 0.
    const BODY: u8 = 3;
    const STAR: u8 = 1;

    fn item(id: ObjectId, dx: f32, dy: f32, dist_m: f64, priority: u8) -> PickItem {
        PickItem {
            id,
            screen_px: CURSOR + Vec2::new(dx, dy),
            dist_m,
            priority,
        }
    }

    const CURSOR: Vec2 = Vec2::new(640.0, 360.0);

    #[test]
    fn pick_nearest_with_priority() {
        let r = pick_radius_px(1080);
        // A planet 8 px and a named star 5 px from the cursor: the planet.
        let planet_and_star = [item(1, 8.0, 0.0, 1e11, BODY), item(2, 0.0, 5.0, 1e17, STAR)];
        assert_eq!(nearest_pick(&planet_and_star, CURSOR, r), Some(1));
        // Two stars: the nearer to the cursor.
        let two_stars = [item(3, 7.0, 0.0, 1e16, STAR), item(4, -3.0, 0.0, 1e18, STAR)];
        assert_eq!(nearest_pick(&two_stars, CURSOR, r), Some(4));
        // Same priority, same screen distance: the nearer to the camera.
        let twins = [item(5, 4.0, 0.0, 2e16, STAR), item(6, -4.0, 0.0, 1e16, STAR)];
        assert_eq!(nearest_pick(&twins, CURSOR, r), Some(6));
        // A body outside the radius loses to a star inside it.
        let far_body = [item(7, 0.0, 13.0, 1e11, BODY), item(8, 0.0, 11.0, 1e17, STAR)];
        assert_eq!(nearest_pick(&far_body, CURSOR, r), Some(8));
    }

    #[test]
    fn pick_none_beyond_radius() {
        assert_eq!(pick_radius_px(1080), PICK_RADIUS_PX_AT_1080);
        let items = [item(1, 13.0, 0.0, 1e11, BODY), item(2, 0.0, -20.0, 1e11, BODY)];
        assert_eq!(nearest_pick(&items, CURSOR, pick_radius_px(1080)), None);
        // The radius scales with the internal height: 13 px is inside at 2160 lines.
        assert_eq!(pick_radius_px(2160), 24.0);
        assert_eq!(nearest_pick(&items, CURSOR, pick_radius_px(2160)), Some(1));
        // On the radius counts as within it.
        assert_eq!(nearest_pick(&[item(3, 12.0, 0.0, 1.0, STAR)], CURSOR, 12.0), Some(3));
        // No items, or a cursor that is not a number: nothing.
        assert_eq!(nearest_pick(&[], CURSOR, 12.0), None);
        assert_eq!(nearest_pick(&items, Vec2::NAN, 1e9), None);
    }
}
