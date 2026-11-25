//! Clicking on objects (Task C3; Task C0 stub; final review, app I2: disks and occlusion).
//!
//! What a click at the cursor picks ([`pick_object`], in internal pixels; the app converts the
//! cursor from window pixels, `InputState::cursor_px` × internal ÷ output):
//! - **Disks.** Every resolved object of the snapshot is a solid ([`pick_disks`]): a body
//!   (`Snapshot::bodies`) its ellipsoid (radii and orientation), a star surface
//!   (`Snapshot::star_surfaces`) and a resolved comet nucleus (`Snapshot::comets` at
//!   `POINT_LOD_PX` or more) a sphere. The cursor's ray ([`PickView::ray`]) is intersected
//!   with each, so a body is picked anywhere on its disk, also when its centre is off screen
//!   (it then has no pick item at all), and the nearest hit along the ray wins: an object
//!   behind a disk is never picked there.
//! - **Points.** The `Snapshot::pick` items of the objects that are not resolved keep the pick
//!   radius ([`pick_radius_px`]: 12 px at 1080 lines), unless a nearer disk covers the item's
//!   own pixel (it is hidden there: an occulted moon, a star behind a planet).
//! - With a disk under the cursor, a point within the radius that is drawn over that disk (in
//!   front of it at its own pixel, as a moon in transit) wins over the disk; any other point
//!   loses to it.
//! - With no disk under the cursor, the candidates are the points within the radius and the
//!   disks whose projected edge is within the radius of the cursor (a near miss; the gap from
//!   the cursor to the edge of the disk of the body's smallest radius, so an oblate body's
//!   halo is never wider than the radius), ranked by priority (bodies and the Sun > comets >
//!   named stars and exoplanet hosts > asteroids; TestBH-1 counts as a body:
//!   `PickItem::priority`), then by the gap, then by the distance to the camera.
//!
//! Approximations: rings, atmospheres and comae do not occlude (only the solids do); a comet
//! nucleus is a sphere of its mean radius; TestBH-1 has no disk here (its shadow is picked as
//! a point, by its centre). A disk whose centre is off screen has no pick item, so it takes
//! the default priority of its kind ([`BODY_PRIORITY`], [`STAR_PRIORITY`],
//! [`COMET_PRIORITY`]), which only orders it against points in a near miss.

use glam::{DQuat, DVec2, DVec3, UVec2, Vec2};
use uni_core::CameraView;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{POINT_LOD_PX, PickItem, Snapshot};

/// The pick radius [px] at 1080 internal lines (plan C3); [`pick_radius_px`] scales it.
pub const PICK_RADIUS_PX_AT_1080: f32 = 12.0;

/// Internal height [lines] at which the pick radius is [`PICK_RADIUS_PX_AT_1080`].
const REFERENCE_LINES: f32 = 1080.0;

/// The priority of a body's or the Sun's disk with no pick item (`uni_sim::snapshot`'s pick
/// priority of bodies and the Sun).
pub const BODY_PRIORITY: u8 = 3;
/// The priority of a resolved comet nucleus with no pick item (`uni_sim::snapshot`'s comets).
pub const COMET_PRIORITY: u8 = 2;
/// The priority of a catalogue star's surface with no pick item (`uni_sim::snapshot`'s stars).
pub const STAR_PRIORITY: u8 = 1;

/// The pick radius [internal px] at an internal height of `internal_lines`: 12 px at 1080
/// lines, in proportion otherwise, so it covers the same share of the screen at any size.
pub fn pick_radius_px(internal_lines: u32) -> f32 {
    PICK_RADIUS_PX_AT_1080 * internal_lines as f32 / REFERENCE_LINES
}

/// The item of `items` within `radius_px` of `cursor_px` (on the radius counts) with the
/// highest `priority`; among equal priorities the nearest to the cursor, then the nearest to the
/// camera (`dist_m`). `None` when no item is that close or the cursor is not a number. Points
/// only: [`pick_object`] adds the disks and occlusion.
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

/// The camera a snapshot was taken from, in internal pixels: the cursor's ray and the screen
/// position of a snapshot's object. The same projection as `uni_sim::snapshot`'s (origin top
/// left, y down; pixel i spans [i, i + 1), as `pixel_dir` in `common/camera.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickView {
    /// Camera → ICRS axes (the camera looks along −Z, up +Y).
    to_world: DQuat,
    tan_half_y: f64,
    /// The internal size [px], at least 1 × 1.
    size: DVec2,
}

impl PickView {
    /// The view of `view` at internal size `viewport` (the snapshot's).
    pub fn new(view: &CameraView, viewport: UVec2) -> Self {
        Self {
            to_world: view.orient.normalize(),
            tan_half_y: (0.5 * view.fov_y_rad).tan(),
            size: viewport.max(UVec2::ONE).as_dvec2(),
        }
    }

    fn aspect(&self) -> f64 {
        self.size.x / self.size.y
    }

    /// The unit direction (ICRS axes) of the ray through internal pixel position `px` (a
    /// continuous position: the centre of pixel (i, j) is (i + ½, j + ½)).
    pub fn ray(&self, px: Vec2) -> DVec3 {
        let p = px.as_dvec2() / self.size;
        let ndc = DVec2::new(2.0 * p.x - 1.0, 1.0 - 2.0 * p.y);
        let camera = DVec3::new(
            ndc.x * self.aspect() * self.tan_half_y,
            ndc.y * self.tan_half_y,
            -1.0,
        );
        (self.to_world * camera).normalize()
    }

    /// Where a point at `rel` [m] from the camera (ICRS axes) projects [internal px], also off
    /// screen; `None` behind the camera.
    pub fn screen(&self, rel: DVec3) -> Option<DVec2> {
        let c = self.to_world.conjugate() * rel;
        if !(c.z < 0.0) {
            return None;
        }
        let ndc = DVec2::new(c.x / self.aspect(), c.y) / (-c.z * self.tan_half_y);
        Some(DVec2::new(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * self.size)
    }

    /// The projected radius [px] of a sphere of `radius_m` whose centre is `dist_m` away, as
    /// `uni_sim::snapshot` computes it (the half-angle's tangent over tan(fov/2)); +∞ with the
    /// camera inside.
    pub fn radius_px(&self, radius_m: f64, dist_m: f64) -> f64 {
        if !(dist_m > radius_m) {
            return f64::INFINITY;
        }
        let s = radius_m / dist_m;
        0.5 * self.size.y * (s / (1.0 - s * s).sqrt()) / self.tan_half_y
    }
}

/// A resolved object as a solid the cursor's ray can hit: an ellipsoid with semi-axes `radii_m`
/// along the body-fixed axes (a sphere when they are equal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickDisk {
    pub id: ObjectId,
    /// The centre relative to the camera [m], ICRS axes.
    pub rel_pos_m: DVec3,
    /// Body-fixed → ICRS rotation.
    pub orient: DQuat,
    pub radii_m: DVec3,
    pub priority: u8,
}

impl PickDisk {
    /// The distance [m] along the unit ray `dir` (ICRS axes) from the camera to where it enters
    /// the solid: 0 with the camera inside, `None` on a miss or with the solid behind.
    ///
    /// The ray is taken into the body's frame and scaled by 1/radii, where the solid is the
    /// unit sphere; the distance along the ray is unchanged by that linear map. The hit is
    /// found from the closest approach and the perpendicular distance there rather than from
    /// b² − 4ac, which loses precision for a far, small sphere (Haines, Günther & Akenine-Möller
    /// 2019, "Precision Improvements for Ray/Sphere Intersection", Ray Tracing Gems ch. 7).
    pub fn hit(&self, dir: DVec3) -> Option<f64> {
        let to_body = self.orient.normalize().conjugate();
        let o = (to_body * -self.rel_pos_m) / self.radii_m;
        let v = (to_body * dir) / self.radii_m;
        let a = v.length_squared();
        if !(a > 0.0 && o.is_finite() && a.is_finite()) {
            return None;
        }
        if o.length_squared() <= 1.0 {
            return Some(0.0);
        }
        let t_mid = -o.dot(v) / a;
        let h2 = (o + v * t_mid).length_squared();
        if !(t_mid > 0.0 && h2 <= 1.0) {
            return None;
        }
        Some(t_mid - ((1.0 - h2) / a).sqrt())
    }

    fn min_radius_m(&self) -> f64 {
        self.radii_m.min_element()
    }
}

/// The solids of `snap`'s resolved objects (module docs): its bodies (ellipsoids), its star
/// surfaces and its resolved comet nuclei (spheres), each with its pick item's priority, or its
/// kind's default without one (centre off screen).
pub fn pick_disks(snap: &Snapshot) -> Vec<PickDisk> {
    let priority = |id: ObjectId, default: u8| {
        snap.pick
            .iter()
            .find(|i| i.id == id)
            .map_or(default, |i| i.priority)
    };
    let bodies = snap.bodies.iter().map(|b| PickDisk {
        id: b.id,
        rel_pos_m: b.rel_pos_m,
        orient: b.orient,
        radii_m: b.radii_m,
        priority: priority(b.id, BODY_PRIORITY),
    });
    let stars = snap.star_surfaces.iter().map(|s| PickDisk {
        id: s.id,
        rel_pos_m: s.rel_pos_m,
        orient: s.orient,
        radii_m: DVec3::splat(s.radius_m),
        // The Sun has no catalogue index.
        priority: priority(
            s.id,
            if s.catalog_index.is_none() {
                BODY_PRIORITY
            } else {
                STAR_PRIORITY
            },
        ),
    });
    let nuclei = snap
        .comets
        .iter()
        .filter(|c| c.proj_radius_px >= POINT_LOD_PX)
        .map(|c| PickDisk {
            id: c.id,
            rel_pos_m: c.nucleus_rel_m,
            orient: DQuat::IDENTITY,
            radii_m: DVec3::splat(c.nucleus_radius_m),
            priority: priority(c.id, COMET_PRIORITY),
        });
    bodies.chain(stars).chain(nuclei).collect()
}

/// One object a click may pick: what it is, how far the cursor is from it [px], how far it is
/// from the camera [m], its priority and the pixel it is seen at.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    id: ObjectId,
    gap_px: f64,
    dist_m: f64,
    priority: u8,
    at: Vec2,
}

/// Higher priority first, then the smaller gap, then the nearer to the camera.
fn rank(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.priority
        .cmp(&a.priority)
        .then(a.gap_px.total_cmp(&b.gap_px))
        .then(a.dist_m.total_cmp(&b.dist_m))
}

/// What a click at `cursor_px` [internal px] picks among `items` (points) and `disks`
/// (solids) seen from `view`, within `radius_px` (module docs). `None` when nothing is under or
/// near the cursor, or the cursor is not a number.
pub fn pick_object(
    items: &[PickItem],
    disks: &[PickDisk],
    view: &PickView,
    cursor_px: Vec2,
    radius_px: f32,
) -> Option<ObjectId> {
    if !cursor_px.is_finite() {
        return None;
    }
    let radius = f64::from(radius_px);
    // Whether a disk other than `id` covers pixel `at` nearer than `dist_m`.
    let hidden = |id: ObjectId, at: Vec2, dist_m: f64| {
        let ray = view.ray(at);
        disks
            .iter()
            .any(|d| d.id != id && d.hit(ray).is_some_and(|t| t < dist_m))
    };
    let points: Vec<Candidate> = items
        .iter()
        .filter(|i| !disks.iter().any(|d| d.id == i.id))
        .filter_map(|i| {
            let gap_px = f64::from(i.screen_px.distance(cursor_px));
            (gap_px <= radius && !hidden(i.id, i.screen_px, i.dist_m)).then_some(Candidate {
                id: i.id,
                gap_px,
                dist_m: i.dist_m,
                priority: i.priority,
                at: i.screen_px,
            })
        })
        .collect();
    // The nearest disk under the cursor.
    let ray = view.ray(cursor_px);
    let under = disks
        .iter()
        .filter_map(|d| d.hit(ray).map(|t| (d, t)))
        .min_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((disk, _)) = under {
        // A point drawn over that disk (a moon in transit) wins; any other loses to it.
        let over = points
            .iter()
            .filter(|c| disk.hit(view.ray(c.at)).is_some())
            .min_by(|a, b| rank(a, b));
        return Some(over.map_or(disk.id, |c| c.id));
    }
    let cursor = cursor_px.as_dvec2();
    let near_misses = disks.iter().filter_map(|d| {
        let centre = view.screen(d.rel_pos_m)?;
        let dist_m = d.rel_pos_m.length();
        let edge_px = view.radius_px(d.min_radius_m(), dist_m);
        let gap_px = (centre.distance(cursor) - edge_px).max(0.0);
        let at = centre.as_vec2();
        (gap_px <= radius && at.is_finite() && !hidden(d.id, at, dist_m)).then_some(Candidate {
            id: d.id,
            gap_px,
            dist_m,
            priority: d.priority,
            at,
        })
    });
    points
        .iter()
        .copied()
        .chain(near_misses)
        .min_by(rank)
        .map(|c| c.id)
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
        let two_stars = [
            item(3, 7.0, 0.0, 1e16, STAR),
            item(4, -3.0, 0.0, 1e18, STAR),
        ];
        assert_eq!(nearest_pick(&two_stars, CURSOR, r), Some(4));
        // Same priority, same screen distance: the nearer to the camera.
        let twins = [
            item(5, 4.0, 0.0, 2e16, STAR),
            item(6, -4.0, 0.0, 1e16, STAR),
        ];
        assert_eq!(nearest_pick(&twins, CURSOR, r), Some(6));
        // A body outside the radius loses to a star inside it.
        let far_body = [
            item(7, 0.0, 13.0, 1e11, BODY),
            item(8, 0.0, 11.0, 1e17, STAR),
        ];
        assert_eq!(nearest_pick(&far_body, CURSOR, r), Some(8));
        // `pick_object` with no disks picks as `nearest_pick` does.
        let view = view();
        for set in [&planet_and_star[..], &two_stars, &twins, &far_body] {
            assert_eq!(
                pick_object(set, &[], &view, CURSOR, r),
                nearest_pick(set, CURSOR, r)
            );
        }
    }

    #[test]
    fn pick_none_beyond_radius() {
        assert_eq!(pick_radius_px(1080), PICK_RADIUS_PX_AT_1080);
        let items = [
            item(1, 13.0, 0.0, 1e11, BODY),
            item(2, 0.0, -20.0, 1e11, BODY),
        ];
        assert_eq!(nearest_pick(&items, CURSOR, pick_radius_px(1080)), None);
        // The radius scales with the internal height: 13 px is inside at 2160 lines.
        assert_eq!(pick_radius_px(2160), 24.0);
        assert_eq!(nearest_pick(&items, CURSOR, pick_radius_px(2160)), Some(1));
        // On the radius counts as within it.
        assert_eq!(
            nearest_pick(&[item(3, 12.0, 0.0, 1.0, STAR)], CURSOR, 12.0),
            Some(3)
        );
        // No items, or a cursor that is not a number: nothing.
        assert_eq!(nearest_pick(&[], CURSOR, 12.0), None);
        assert_eq!(nearest_pick(&items, Vec2::NAN, 1e9), None);
        assert_eq!(pick_object(&items, &[], &view(), Vec2::NAN, 1e9), None);
    }

    /// The internal size of the synthetic views.
    const SIZE: UVec2 = UVec2::new(1920, 1080);

    /// A 60° camera looking along −Z from the origin, up +Y.
    fn view() -> PickView {
        PickView::new(
            &CameraView {
                pos: uni_core::UniversePos::default(),
                orient: DQuat::IDENTITY,
                fov_y_rad: 60_f64.to_radians(),
                velocity_mps: DVec3::ZERO,
            },
            SIZE,
        )
    }

    /// The screen centre [px].
    fn centre() -> Vec2 {
        SIZE.as_vec2() / 2.0
    }

    /// A sphere of `radius_m` at `rel` [m] from the camera.
    fn sphere(id: ObjectId, rel: DVec3, radius_m: f64, priority: u8) -> PickDisk {
        PickDisk {
            id,
            rel_pos_m: rel,
            orient: DQuat::IDENTITY,
            radii_m: DVec3::splat(radius_m),
            priority,
        }
    }

    /// A point item for an object at `rel` [m] from the camera, seen from [`view`].
    fn point_at(id: ObjectId, rel: DVec3, priority: u8) -> PickItem {
        let p = view().screen(rel).expect("in front");
        PickItem {
            id,
            screen_px: p.as_vec2(),
            dist_m: rel.length(),
            priority,
        }
    }

    /// `PickView::ray` and `screen` are inverses, off axis too, and match the snapshot's
    /// projected radius.
    #[test]
    fn ray_and_screen_agree() {
        let v = PickView::new(
            &CameraView {
                pos: uni_core::UniversePos::default(),
                orient: DQuat::from_euler(glam::EulerRot::YXZ, 0.7, -0.3, 0.2),
                fov_y_rad: 0.9,
                velocity_mps: DVec3::ZERO,
            },
            SIZE,
        );
        for px in [
            Vec2::new(0.0, 0.0),
            Vec2::new(960.0, 540.0),
            Vec2::new(1900.5, 17.25),
        ] {
            let back = v.screen(v.ray(px) * 1e9).expect("in front");
            assert!(back.as_vec2().distance(px) < 1e-3, "{px} → {back}");
        }
        // The centre ray is the view axis.
        let axis = view().ray(centre());
        assert!((axis - DVec3::NEG_Z).length() < 1e-12, "{axis}");
        // A sphere seen on axis: its edge pixel is radius_px from the centre.
        let r = view().radius_px(1.0, 10.0);
        let edge = (1.0_f64 / 99.0_f64.sqrt()).atan(); // half-angle asin(1/10)
        let expect = 540.0 * edge.tan() / 30_f64.to_radians().tan();
        assert!((r - expect).abs() < 1e-9, "{r} vs {expect}");
        assert_eq!(view().radius_px(2.0, 1.0), f64::INFINITY);
    }

    /// The ray–ellipsoid test: the first entry along the ray, a rotated ellipsoid's axes, the
    /// camera inside, a miss beside and a solid behind.
    #[test]
    fn rays_hit_ellipsoids() {
        let ball = sphere(1, DVec3::new(0.0, 0.0, -10.0), 2.0, BODY);
        assert!((ball.hit(DVec3::NEG_Z).unwrap() - 8.0).abs() < 1e-12);
        assert_eq!(ball.hit(DVec3::Z), None, "behind");
        let beside = DVec3::new(0.21, 0.0, -1.0).normalize(); // passes 2.06 from the centre
        assert_eq!(ball.hit(beside), None);
        let grazing = DVec3::new(0.19, 0.0, -1.0).normalize();
        assert!(ball.hit(grazing).is_some());
        // An ellipsoid 4 × 1 × 1 along its own x, turned so x points along ICRS y.
        let long = PickDisk {
            radii_m: DVec3::new(4.0, 1.0, 1.0),
            orient: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            ..ball
        };
        let up = DVec3::new(0.0, 3.0, -10.0).normalize();
        let side = DVec3::new(3.0, 0.0, -10.0).normalize();
        assert!(long.hit(up).is_some() && long.hit(side).is_none());
        assert!((long.hit(DVec3::NEG_Z).unwrap() - 9.0).abs() < 1e-12);
        // A tiny, far sphere (Mars-sized 1 AU away) is still hit to the metre.
        let far = sphere(2, DVec3::new(0.0, 0.0, -1.496e11), 3.3895e6, BODY);
        let t = far.hit(DVec3::NEG_Z).unwrap();
        assert!((t - (1.496e11 - 3.3895e6)).abs() < 1.0, "{t}");
        // The camera inside: hit at once.
        let around = sphere(3, DVec3::new(0.0, 0.0, -0.5), 2.0, BODY);
        assert_eq!(around.hit(DVec3::X), Some(0.0));
    }

    /// Final review, app I2: a body is picked anywhere on its disk, not only within 12 px of its
    /// centre; just off its edge within the radius it is still picked (a near miss), beyond it
    /// not.
    #[test]
    fn a_body_is_picked_anywhere_on_its_disk() {
        let v = view();
        // A planet 425 px in radius at the centre.
        let rel = DVec3::new(0.0, 0.0, -1.0e7);
        let s = 425.0 * 30_f64.to_radians().tan() / 540.0;
        let radius = 1.0e7 * s / (1.0 + s * s).sqrt();
        let planet = sphere(1, rel, radius, BODY);
        assert!((v.radius_px(radius, 1.0e7) - 425.0).abs() < 1e-6);
        let items = [point_at(1, rel, BODY)];
        let disks = [planet];
        for dx in [0.0, 200.0, 420.0, -300.0] {
            let at = centre() + Vec2::new(dx, 0.0);
            assert_eq!(
                pick_object(&items, &disks, &v, at, 12.0),
                Some(1),
                "{dx} px"
            );
            // The old rule: nothing beyond 12 px of the centre.
            let old = nearest_pick(&items, at, 12.0);
            assert_eq!(old.is_some(), dx.abs() <= 12.0, "{dx} px");
        }
        // 8 px off the edge: a near miss; 20 px off: nothing.
        let off = |px: f32| {
            pick_object(
                &items,
                &disks,
                &v,
                centre() + Vec2::new(425.0 + px, 0.0),
                12.0,
            )
        };
        assert_eq!(off(8.0), Some(1));
        assert_eq!(off(20.0), None);
    }

    /// Final review, app I2: a body whose centre is off screen (it has no pick item) is picked
    /// by its disk.
    #[test]
    fn a_body_with_its_centre_off_screen_is_picked_by_its_disk() {
        let v = view();
        // A planet below the frame filling its lower half: centre 1.5 radii below the axis.
        let planet = sphere(4, DVec3::new(0.0, -3.0e6, -2.0e6), 2.5e6, BODY);
        let centre_px = v.screen(planet.rel_pos_m).unwrap();
        assert!(centre_px.y > f64::from(SIZE.y), "off screen: {centre_px}");
        let low = Vec2::new(960.0, 1060.0);
        assert_eq!(pick_object(&[], &[planet], &v, low, 12.0), Some(4));
        // Above the limb, far from it: nothing.
        assert_eq!(
            pick_object(&[], &[planet], &v, Vec2::new(960.0, 40.0), 12.0),
            None
        );
    }

    /// Final review, app I2: the nearest hit along the ray wins, and an object behind a disk is
    /// not picked there: a star or an occulted moon behind the planet, at the cursor or within
    /// 12 px of it, never; a moon in front of the disk (in transit) within the radius wins over
    /// the disk; one beside the disk loses to it while the cursor is on the disk.
    #[test]
    fn what_is_behind_a_disk_is_never_picked() {
        let v = view();
        let planet = sphere(1, DVec3::new(0.0, 0.0, -1.0e8), 7.0e6, BODY);
        let r_px = v.radius_px(7.0e6, 1.0e8) as f32;
        let disks = [planet];
        let limb = centre() + Vec2::new(0.95 * r_px, 0.0);
        let dir = v.ray(limb);
        // An occulted moon behind the limb, and a star there: hidden.
        let moon_behind = PickItem {
            id: 2,
            screen_px: limb,
            dist_m: 1.5e8,
            priority: BODY,
        };
        let star = PickItem {
            id: 3,
            screen_px: limb + Vec2::new(2.0, 0.0),
            dist_m: 1e17,
            priority: STAR,
        };
        let items = [point_at(1, planet.rel_pos_m, BODY), moon_behind, star];
        // The old rule flew to the occulted moon (nearest to the cursor, > 12 px from the
        // planet's centre).
        assert_eq!(nearest_pick(&items, limb, 12.0), Some(2));
        assert_eq!(pick_object(&items, &disks, &v, limb, 12.0), Some(1));
        assert_eq!(
            pick_object(&items, &disks, &v, limb + Vec2::new(2.0, 0.0), 12.0),
            Some(1)
        );
        // Just off the limb, within 12 px of the hidden moon and star: the planet's near miss,
        // never what it hides.
        let outside = centre() + Vec2::new(r_px + 4.0, 0.0);
        assert!(outside.distance(limb) < 12.0 && planet.hit(v.ray(outside)).is_none());
        assert_eq!(pick_object(&items, &disks, &v, outside, 12.0), Some(1));
        let only_hidden = [moon_behind, star];
        assert_eq!(
            pick_object(&only_hidden, &disks, &v, outside, 12.0),
            Some(1),
            "the planet's edge, not what it hides"
        );
        // A moon in transit, in front of the disk, 6 px from the cursor: the moon.
        let transit = PickItem {
            id: 5,
            screen_px: limb + Vec2::new(-6.0, 0.0),
            dist_m: 0.5e8,
            priority: BODY,
        };
        let with_transit = [items[0], moon_behind, star, transit];
        assert_eq!(pick_object(&with_transit, &disks, &v, limb, 12.0), Some(5));
        // A moon beside the disk (not over it), 6 px from a cursor on the disk: the disk.
        let edge = centre() + Vec2::new(r_px - 3.0, 0.0);
        let beside = PickItem {
            id: 6,
            screen_px: edge + Vec2::new(6.0, 0.0),
            dist_m: 0.9e8,
            priority: BODY,
        };
        assert!(planet.hit(v.ray(beside.screen_px)).is_none());
        assert_eq!(pick_object(&[beside], &disks, &v, edge, 12.0), Some(1));
        // Two disks along the ray: the nearer.
        let moon_disk = sphere(7, dir * 0.4e8, 1.0e6, BODY);
        assert_eq!(
            pick_object(&[], &[planet, moon_disk], &v, limb, 12.0),
            Some(7)
        );
        // A resolved moon behind the planet is hidden even at its own centre.
        let hidden_disk = sphere(8, dir * 1.5e8, 1.0e6, BODY);
        assert_eq!(
            pick_object(&[], &[planet, hidden_disk], &v, limb, 12.0),
            Some(1)
        );
    }
}
