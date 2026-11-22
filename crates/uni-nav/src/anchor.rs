//! Bookmark anchors: resolving a bookmark's `anchor`, `look_at` and `up` against the world's
//! objects (M1 Task A6).

use anyhow::Context;
use glam::{DMat3, DQuat, DVec3};
use uni_core::{CELL_SIZE_M, CameraView, FrameId, ROOT_FRAME, UniversePos};

use crate::bookmark::Bookmark;

/// Below this sine of the angle between the view direction and an up hint, the two count as
/// parallel and the next hint is tried (about 2e-4 arcsec).
const PARALLEL_SIN: f64 = 1e-9;

/// Finds a named object: its centre in the root frame and its frame.
pub trait ObjectLookup {
    fn find_object(&self, name: &str) -> Option<(UniversePos, FrameId)>;
}

/// The camera view and anchor frame a bookmark describes. `anchor` makes `view.pos` an offset
/// from that object's centre; `look_at` overrides `view.orient`. An unknown name is an error that
/// names it.
///
/// - Without `anchor`, `view.pos` is a root position and the frame is [`ROOT_FRAME`]; a bookmark
///   with neither `anchor` nor `look_at` (every M0 bookmark) resolves to its own view unchanged.
/// - With `anchor`, `view.pos` is the offset from the object's centre (normally cell 0, metres,
///   root axes; it is added cell by cell, so any cell works) and the frame is the object's.
/// - With `look_at`, the camera's −Z points at the object's centre and its +Y is as close as
///   possible to `up` (root axes; default ICRS +Z). When the view is parallel to `up`, +Y is
///   used instead, and ICRS +Z when the view is parallel to +Y as well. A camera at the
///   object's centre, or a zero or non-finite `up`, is an error. `up` is ignored without
///   `look_at`.
///
/// Objects are looked up where the world has them now, so the caller sets the bookmark's epoch
/// first. Field of view and velocity come from the bookmark unchanged.
pub fn resolve_bookmark(
    b: &Bookmark,
    world: &dyn ObjectLookup,
) -> anyhow::Result<(CameraView, FrameId)> {
    let find = |field: &str, name: &str| {
        world
            .find_object(name)
            .with_context(|| format!("bookmark \"{}\": unknown {field} object \"{name}\"", b.name))
    };

    let mut view = b.view.clone();
    let mut frame = ROOT_FRAME;
    if let Some(name) = &b.anchor {
        let (centre, anchor_frame) = find("anchor", name)?;
        let offset = b.view.pos;
        let centre = UniversePos {
            cell: centre.cell.saturating_add(offset.cell),
            local: centre.local,
        };
        view.pos = add_m(centre, offset.local);
        frame = anchor_frame;
    }
    if let Some(name) = &b.look_at {
        let (target, _) = find("look_at", name)?;
        view.orient = look_at_orient(delta_m(target, view.pos), b.up.unwrap_or(DVec3::Z))
            .with_context(|| format!("bookmark \"{}\": cannot look at \"{name}\"", b.name))?;
    }
    Ok((view, frame))
}

/// `origin` + `d` [m] with one rounding, of the result's cell-local metres: at most half their
/// spacing, 2^-11 m near a cell's edge. `UniversePos::offset` rounds `local + d` before
/// normalising, which costs up to 2^-10 m once the sum passes the edge (Review Focus 5).
pub(crate) fn add_m(origin: UniversePos, d: DVec3) -> UniversePos {
    add_dd(origin, d, DVec3::ZERO)
}

/// `origin` + (`hi` + `lo`) [m], where `hi` + `lo` is an unevaluated (double-double) sum with
/// |`lo`| at most half a spacing of `hi`. Like [`add_m`] it rounds once at the result's
/// cell-local metres; `lo` joins the exact rounding error `err` of `origin.local` + `hi` first,
/// which adds at most 2^-53 × (|`err`| + |`lo`|) ≤ 2^-53 × (2^-9 m + half a spacing of `hi`):
/// about 2e-16 m for |`hi`| up to 1 pc.
pub(crate) fn add_dd(origin: UniversePos, hi: DVec3, lo: DVec3) -> UniversePos {
    // Whole cells of `hi` first; `normalized` is exact and leaves |d.local| ≤ CELL_SIZE_M / 2.
    let d = UniversePos {
        cell: origin.cell,
        local: hi,
    }
    .normalized();
    let (sum, err) = two_sum(origin.local, d.local);
    // |sum| ≤ CELL_SIZE_M, so moving its whole cells out is exact too; `err` and `lo` then go in
    // with the one rounding.
    let p = UniversePos {
        cell: d.cell,
        local: sum,
    }
    .normalized();
    UniversePos {
        cell: p.cell,
        local: p.local + (err + lo),
    }
    .normalized()
}

/// `p` − `origin` [m], rounded once at the result's own magnitude (up to a 2^-53 relative
/// rounding of the part below its spacing). `UniversePos::delta_from` rounds the cell-local
/// difference first, which costs up to 2^-10 m when the two points lie on either side of a
/// cell's edge (Review Focus 5).
pub(crate) fn delta_m(p: UniversePos, origin: UniversePos) -> DVec3 {
    delta_dd(p, origin).0
}

/// `p` − `origin` [m] as a double-double (`hi`, `lo`): `hi` = fl(`hi` + `lo`), and `hi` + `lo`
/// is the exact difference up to a 2^-53 rounding of `lo`, while the points are fewer than
/// 2^53 cells (about 1.6e29 m, beyond the observable universe) apart along each axis.
pub(crate) fn delta_dd(p: UniversePos, origin: UniversePos) -> (DVec3, DVec3) {
    let (local, err) = two_sum(p.local, -origin.local);
    // Exact below 2^53 cells: the cell count converts exactly and CELL_SIZE_M is a power of two.
    let cells = (p.cell - origin.cell).as_dvec3() * CELL_SIZE_M;
    let (hi, err_hi) = two_sum(cells, local);
    two_sum(hi, err_hi + err)
}

/// Per component, `s` = fl(a + b) and `e` = (a + b) − `s` exactly (TwoSum: Møller 1965, BIT 5,
/// 37; Knuth, *The Art of Computer Programming* vol. 2, §4.2.2, Theorem B).
pub(crate) fn two_sum(a: DVec3, b: DVec3) -> (DVec3, DVec3) {
    let s = a + b;
    let b_virtual = s - a;
    let e = (a - (s - b_virtual)) + (b - b_virtual);
    (s, e)
}

/// The camera→root rotation whose −Z points along `dir` and whose +Y is as close as possible
/// to `up`, falling back to ICRS +Y, then +Z, when the view is parallel to the hint.
fn look_at_orient(dir: DVec3, up: DVec3) -> anyhow::Result<DQuat> {
    let up = up
        .try_normalize()
        .with_context(|| format!("up {up} is not a finite non-zero vector"))?;
    let forward = dir
        .try_normalize()
        .context("the camera is at the object's centre, so there is no direction to face")?;
    // For a unit `forward`, |forward × Y|² + |forward × Z|² = 1 + forward.x² ≥ 1, so when the
    // view is parallel to +Y, forward × Z has length 1.
    let right = [up, DVec3::Y]
        .into_iter()
        .map(|u| forward.cross(u))
        .find(|r| r.length() > PARALLEL_SIN)
        .unwrap_or_else(|| forward.cross(DVec3::Z))
        .normalize();
    let cam_up = right.cross(forward);
    Ok(DQuat::from_mat3(&DMat3::from_cols(right, cam_up, -forward)).normalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bookmark::{load_bookmarks, save_bookmarks};
    use crate::camera::FlyCamera;
    use crate::input::{Action, InputState};
    use glam::{DQuat, DVec3, I64Vec3};
    use std::path::Path;
    use uni_core::{AU, CELL_SIZE_M, Epoch, FrameMotion, FrameTree, ROOT_FRAME};

    /// A fixed table of named objects standing in for the world.
    struct FakeWorld(Vec<(&'static str, UniversePos, FrameId)>);

    impl ObjectLookup for FakeWorld {
        fn find_object(&self, name: &str) -> Option<(UniversePos, FrameId)> {
            self.0
                .iter()
                .find(|(n, ..)| *n == name)
                .map(|&(_, pos, frame)| (pos, frame))
        }
    }

    const MARS_FRAME: FrameId = 4;

    fn mars_pos() -> UniversePos {
        UniversePos::from_meters(DVec3::new(1.5 * AU, 0.0, 0.0))
    }

    fn world() -> FakeWorld {
        FakeWorld(vec![
            ("Sun", UniversePos::default(), 1),
            ("Mars", mars_pos(), MARS_FRAME),
        ])
    }

    /// A bookmark at `pos_m` (cell 0) with every optional field unset.
    fn bookmark(pos_m: DVec3) -> Bookmark {
        Bookmark {
            name: "test".to_owned(),
            view: CameraView {
                pos: UniversePos {
                    cell: I64Vec3::ZERO,
                    local: pos_m,
                },
                orient: DQuat::from_rotation_x(1.0),
                fov_y_rad: 0.5,
                velocity_mps: DVec3::new(1.0, 2.0, 3.0),
            },
            epoch_utc: None,
            scene_ev100: None,
            star_ev100: None,
            lens_scale: None,
            anchor: None,
            look_at: None,
            up: None,
            time_rate: None,
            exposure: None,
            eye_adapted: None,
            target: None,
            lens_env: None,
            cruise: None,
            highlight_belt: None,
        }
    }

    /// A bookmark anchored to Mars at `offset_m`, looking at Mars.
    fn facing_mars(offset_m: DVec3) -> Bookmark {
        let mut b = bookmark(offset_m);
        b.anchor = Some("Mars".to_owned());
        b.look_at = Some("Mars".to_owned());
        b
    }

    fn forward(v: &CameraView) -> DVec3 {
        v.orient * DVec3::NEG_Z
    }

    fn up(v: &CameraView) -> DVec3 {
        v.orient * DVec3::Y
    }

    /// `x` in exact units of 2^-60 m. Every value in `cell_edge_sums_round_once` is 0 or at
    /// least 2^-8 m in size, so its last bit is worth at least 2^-60 m.
    fn fixed(x: f64) -> i128 {
        let scaled = x * 2_f64.powi(60);
        assert_eq!(scaled.fract(), 0.0, "{x} is not a whole number of 2^-60 m");
        scaled as i128
    }

    /// The spacing of f64 values at `x`, in units of 2^-60 m.
    fn ulp(x: f64) -> i128 {
        fixed(x.abs().next_up() - x.abs())
    }

    #[test]
    fn cell_edge_sums_round_once() {
        // Review Focus 5. Near a cell's edge, cell-local metres are 2^-10 m apart and sums just
        // past it 2^-9 m apart. A step across the edge must round once, at the result (≤ half a
        // spacing, 2^-11 m), and so must the offset between points on either side (≤ half a
        // spacing of the result). Checked against exact integer arithmetic; `UniversePos::offset`
        // fails it.
        let half = CELL_SIZE_M / 2.0;
        let cell_units = fixed(CELL_SIZE_M);
        for i in 0..2000 {
            let f = (f64::from(i) * 0.618_033_988_749_895).fract();
            // x and y cross the cell's edge (x outwards, y inwards); z stays inside.
            let o = UniversePos {
                cell: I64Vec3::new(7, -3, 1_000_000_000),
                local: DVec3::new(half - 1.0e6 * f - 0.1, -half + 3.3e5 * f + 0.7, 1.0e12 * f),
            };
            let d = DVec3::new(2.0e6 * f + 1.3e5 + f, -7.0e5 * f - 0.3, 1.0e9 * f + 0.11);
            let p = add_m(o, d);
            assert!(p.local.abs().max_element() <= half, "{p:?}");
            let r = delta_m(p, o);
            let cells = p.cell - o.cell;
            for k in 0..3 {
                let exact =
                    i128::from(cells[k]) * cell_units + fixed(p.local[k]) - fixed(o.local[k]);
                let add_err = (exact - fixed(d[k])).abs();
                assert!(
                    2 * add_err <= ulp(p.local[k]),
                    "add, i = {i}, axis {k}: {p:?}"
                );
                let delta_err = (fixed(r[k]) - exact).abs();
                assert!(2 * delta_err <= ulp(r[k]), "delta, i = {i}, axis {k}: {r}");
            }
            // The double-double offset is exact (here every part is a whole number of 2^-60 m
            // under 2^-7 m, so even its tail adds without rounding), and adding it back to
            // `o` lands on `p` itself.
            let (hi, lo) = delta_dd(p, o);
            assert_eq!(hi, r, "i = {i}");
            for k in 0..3 {
                let exact =
                    i128::from(cells[k]) * cell_units + fixed(p.local[k]) - fixed(o.local[k]);
                assert_eq!(fixed(hi[k]) + fixed(lo[k]), exact, "dd, i = {i}, axis {k}");
            }
            assert_eq!(add_dd(o, hi, lo), p, "dd round trip, i = {i}");
        }
    }

    #[test]
    fn anchor_and_look_at_resolve() {
        let b = facing_mars(DVec3::new(0.0, 0.0, 1e7));
        let (view, frame) = resolve_bookmark(&b, &world()).unwrap();
        assert_eq!(frame, MARS_FRAME);
        assert_eq!(view.pos, mars_pos().offset(DVec3::new(0.0, 0.0, 1e7)));
        assert!((forward(&view) - DVec3::NEG_Z).length() < 1e-12, "{view:?}");
        // The default up (ICRS +Z) is parallel to the view here, so +Y is used instead.
        assert!((up(&view) - DVec3::Y).length() < 1e-12, "{view:?}");
        // Everything but the position and orientation comes from the bookmark.
        assert_eq!(
            (view.fov_y_rad, view.velocity_mps),
            (b.view.fov_y_rad, b.view.velocity_mps)
        );
    }

    #[test]
    fn look_at_keeps_up_closest_to_hint() {
        // From Mars + (1e7, 0, 3e6) m: −Z points at Mars, and +Y is the unit vector
        // perpendicular to the view that lies in the plane of the view and ICRS +Z, on +Z's side.
        let (view, _) =
            resolve_bookmark(&facing_mars(DVec3::new(1e7, 0.0, 3e6)), &world()).unwrap();
        let expected_fwd = DVec3::new(-1e7, 0.0, -3e6).normalize();
        assert!((forward(&view) - expected_fwd).length() < 1e-12, "{view:?}");
        let u = up(&view);
        assert!(u.dot(forward(&view)).abs() < 1e-12, "{u}");
        assert!(u.dot(expected_fwd.cross(DVec3::Z)).abs() < 1e-12, "{u}");
        assert!(u.z > 0.0, "{u}");

        // An explicit hint: from above Mars, looking down (−Z), with up = ICRS +X.
        let mut b = facing_mars(DVec3::new(0.0, 0.0, 1e7));
        b.up = Some(DVec3::new(2.0, 0.0, 0.0));
        let (view, _) = resolve_bookmark(&b, &world()).unwrap();
        assert!((forward(&view) - DVec3::NEG_Z).length() < 1e-12, "{view:?}");
        assert!((up(&view) - DVec3::X).length() < 1e-12, "{view:?}");
    }

    #[test]
    fn look_at_up_fallback() {
        // Looking along +Z (from below Mars) with up = +Z: +Y is used.
        let mut b = facing_mars(DVec3::new(0.0, 0.0, -1e7));
        b.up = Some(DVec3::Z);
        let (view, _) = resolve_bookmark(&b, &world()).unwrap();
        assert!(view.orient.is_finite(), "{view:?}");
        assert!((view.orient.length() - 1.0).abs() < 1e-12, "{view:?}");
        assert!((forward(&view) - DVec3::Z).length() < 1e-12, "{view:?}");
        assert!((up(&view) - DVec3::Y).length() < 1e-12, "{view:?}");

        // Looking along −Y with up = +Y: +Y is parallel too, so ICRS +Z is used.
        let mut b = facing_mars(DVec3::new(0.0, 1e7, 0.0));
        b.up = Some(DVec3::Y);
        let (view, _) = resolve_bookmark(&b, &world()).unwrap();
        assert!((view.orient.length() - 1.0).abs() < 1e-12, "{view:?}");
        assert!((forward(&view) - DVec3::NEG_Y).length() < 1e-12, "{view:?}");
        assert!((up(&view) - DVec3::Z).length() < 1e-12, "{view:?}");
    }

    #[test]
    fn unknown_anchor_is_error() {
        let mut b = bookmark(DVec3::ZERO);
        b.anchor = Some("Vulcan".to_owned());
        let err = resolve_bookmark(&b, &world()).unwrap_err();
        assert!(format!("{err:#}").contains("Vulcan"), "{err:#}");

        let mut b = bookmark(DVec3::ZERO);
        b.look_at = Some("Nemesis".to_owned());
        let err = resolve_bookmark(&b, &world()).unwrap_err();
        assert!(format!("{err:#}").contains("Nemesis"), "{err:#}");
    }

    #[test]
    fn degenerate_look_at_is_error() {
        // At Mars's centre there is no direction to face (Review Focus 2).
        let err = resolve_bookmark(&facing_mars(DVec3::ZERO), &world()).unwrap_err();
        assert!(format!("{err:#}").contains("Mars"), "{err:#}");

        // A zero or non-finite up hint is a broken bookmark, not a fallback case.
        for bad in [DVec3::ZERO, DVec3::new(f64::NAN, 0.0, 1.0)] {
            let mut b = facing_mars(DVec3::new(1e7, 0.0, 0.0));
            b.up = Some(bad);
            let err = resolve_bookmark(&b, &world()).unwrap_err();
            assert!(format!("{err:#}").contains("up"), "{err:#}");
        }
    }

    #[test]
    fn look_at_without_anchor_stays_in_root() {
        // From the SSB, looking at Mars: the position is the bookmark's own.
        let mut b = bookmark(DVec3::ZERO);
        b.look_at = Some("Mars".to_owned());
        let (view, frame) = resolve_bookmark(&b, &world()).unwrap();
        assert_eq!(frame, ROOT_FRAME);
        assert_eq!(view.pos, UniversePos::default());
        assert!((forward(&view) - DVec3::X).length() < 1e-12, "{view:?}");
        assert!((up(&view) - DVec3::Z).length() < 1e-12, "{view:?}");
    }

    #[test]
    fn m0_bookmarks_resolve_unchanged() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bookmarks/m0.ron");
        let marks = load_bookmarks(&path).unwrap();
        assert_eq!(marks.len(), 6);
        for b in &marks {
            let (view, frame) = resolve_bookmark(b, &FakeWorld(Vec::new())).unwrap();
            assert_eq!(view, b.view, "{}", b.name);
            assert_eq!(frame, ROOT_FRAME, "{}", b.name);
        }
    }

    #[test]
    fn anchored_view_resolves_back() {
        // F12: a camera anchored to a moving Mars saves `anchor: "Mars"` and its offset; loading
        // that bookmark at the same epoch puts the camera back where it was.
        let mut frames = FrameTree::new();
        let mars = frames.add(
            ROOT_FRAME,
            "Mars",
            FrameMotion::Custom(Box::new(|t: Epoch| {
                let y = 24_000.0 * t.seconds_since(Epoch::J2000);
                (
                    UniversePos::from_meters(DVec3::new(1.5 * AU, y, 0.0)),
                    DQuat::IDENTITY,
                )
            })),
        );
        let t = Epoch::J2000.add_seconds(86_400.0);
        let mars_now = frames.origin_in_root(mars, t);
        let mut cam = FlyCamera::new(CameraView {
            pos: mars_now.offset(DVec3::new(3.0e6, -4.0e6, 1.2e7)),
            orient: DQuat::from_rotation_z(0.3),
            fov_y_rad: 0.7,
            velocity_mps: DVec3::ZERO,
        });
        cam.reanchor(&frames, t, mars);
        cam.sync(&frames, t);

        let saved = cam.anchored_view();
        assert_eq!(saved.pos.cell, I64Vec3::ZERO);
        assert_eq!(saved.pos.local, cam.rel_m);
        let mut b = bookmark(DVec3::ZERO);
        b.view = saved;
        b.anchor = Some("Mars".to_owned());
        let world = FakeWorld(vec![("Mars", mars_now, mars)]);
        let (view, frame) = resolve_bookmark(&b, &world).unwrap();
        assert_eq!(frame, mars);
        assert_eq!((view.orient, view.fov_y_rad), (cam.view.orient, 0.7));
        // `sync` and `resolve_bookmark` add the same offset to the same centre the same way.
        assert_eq!(view.pos, cam.view.pos);

        // Anchored to the root frame, the saved position is the root position itself.
        cam.reanchor(&frames, t, ROOT_FRAME);
        assert_eq!(cam.anchored_view(), cam.view);
    }

    #[test]
    fn bookmark_view_names_the_anchor() {
        // F12 (review r0 I2): anchored to a named body, the anchored view and the body's name,
        // which resolve back to the same place; anchored to ROOT (which asks for no name) or
        // to a frame the world cannot name, the root view and no anchor.
        let mut frames = FrameTree::new();
        let mars = frames.add(
            ROOT_FRAME,
            "Mars",
            FrameMotion::Fixed {
                origin: mars_pos(),
                rotation: DQuat::IDENTITY,
            },
        );
        let t = Epoch::J2000;
        let mut cam = FlyCamera::new(CameraView {
            pos: mars_pos().offset(DVec3::new(1.0e7, -2.5e6, 0.125)),
            orient: DQuat::from_rotation_y(0.4),
            fov_y_rad: 0.7,
            velocity_mps: DVec3::ZERO,
        });

        let root = cam.bookmark_view(|_| -> Option<String> { panic!("ROOT needs no name") });
        assert_eq!(root, (cam.view.clone(), None));

        cam.reanchor(&frames, t, mars);
        cam.sync(&frames, t);
        let (view, anchor) = cam.bookmark_view(|f| (f == mars).then(|| "Mars".to_owned()));
        assert_eq!(view, cam.anchored_view());
        assert_eq!(anchor.as_deref(), Some("Mars"));
        let mut b = bookmark(DVec3::ZERO);
        (b.view, b.anchor) = (view, anchor);
        let world = FakeWorld(vec![("Mars", mars_pos(), mars)]);
        let (resolved, frame) = resolve_bookmark(&b, &world).unwrap();
        assert_eq!((resolved.pos, frame), (cam.view.pos, mars));

        assert_eq!(cam.bookmark_view(|_| None), (cam.view.clone(), None));
    }

    #[test]
    fn f12_bookmark_survives_the_file() {
        // Review r0/r1 I2, uni-nav half. B9's per-frame loop (update → sync → reanchor to the
        // dominant frame) flies a camera from ROOT into a moving Mars's frame. F12 names the
        // anchor through the world's object table (name, own frame), as uni-sim's
        // `World::objects()` gives it, and writes the file. Loading and resolving it at the
        // same epoch gives back the camera's frame and root position.
        let mut frames = FrameTree::new();
        let mars = frames.add(
            ROOT_FRAME,
            "Mars",
            FrameMotion::Custom(Box::new(|t: Epoch| {
                let y = 24_000.0 * t.seconds_since(Epoch::J2000);
                (
                    UniversePos::from_meters(DVec3::new(1.5 * AU, y, 0.0)),
                    DQuat::IDENTITY,
                )
            })),
        );
        // A frame-less object first, so the lookup has to match the frame.
        let objects = [("TestBH-1", None), ("Mars", Some(mars))];
        let path = std::env::temp_dir().join(format!("uni-nav-{}-f12.ron", std::process::id()));
        let f12 = |cam: &FlyCamera| {
            let (view, anchor) = cam.bookmark_view(|f| {
                objects
                    .iter()
                    .find(|(_, frame)| *frame == Some(f))
                    .map(|(name, _)| (*name).to_owned())
            });
            let mut b = bookmark(DVec3::ZERO);
            (b.view, b.anchor) = (view, anchor);
            save_bookmarks(&path, &[b]).unwrap();
            let text = std::fs::read_to_string(&path).unwrap();
            let mut loaded = load_bookmarks(&path).unwrap();
            let _ = std::fs::remove_file(&path);
            assert_eq!(loaded.len(), 1);
            (text, loaded.remove(0))
        };
        // Mars's frame dominates within 1e9 m of its centre (a stand-in for its Hill sphere).
        let dominant = |t: Epoch, p: UniversePos| {
            let near = delta_m(p, frames.origin_in_root(mars, t)).length() < 1.0e9;
            if near { mars } else { ROOT_FRAME }
        };

        let mut t = Epoch::J2000.add_seconds(86_400.0);
        let mut cam = FlyCamera::new(CameraView {
            pos: frames
                .origin_in_root(mars, t)
                .offset(DVec3::new(2.0e6, -3.0e6, 1.05e9)),
            orient: DQuat::IDENTITY,
            fov_y_rad: 0.7,
            velocity_mps: DVec3::ZERO,
        });

        // Outside Mars's sphere the camera is root-anchored: no anchor, the root view itself.
        let (text, b) = f12(&cam);
        assert!(!text.contains("anchor"), "{text}");
        let world = FakeWorld(vec![("Mars", frames.origin_in_root(mars, t), mars)]);
        assert_eq!(
            resolve_bookmark(&b, &world).unwrap(),
            (cam.view.clone(), ROOT_FRAME)
        );

        // 1 s toward Mars at 1e8 m/s crosses into its sphere, then 1 s at 0.1 m/s.
        let mut input = InputState::new();
        input.set_down(Action::Forward, true);
        for i in 0..120 {
            cam.speed_log10 = if i < 60 { 8.0 } else { -1.0 };
            cam.update(&mut input, 1.0 / 60.0);
            t = t.add_seconds(1.0 / 60.0);
            cam.sync(&frames, t);
            cam.reanchor(&frames, t, dominant(t, cam.view.pos));
        }
        assert_eq!(cam.anchor, mars);

        let (text, b) = f12(&cam);
        assert!(text.contains("anchor: Some(\"Mars\")"), "{text}");
        assert_eq!(b.view.pos.cell, I64Vec3::ZERO);
        let world = FakeWorld(vec![("Mars", frames.origin_in_root(mars, t), mars)]);
        let (view, frame) = resolve_bookmark(&b, &world).unwrap();
        assert_eq!(frame, mars);
        assert_eq!(view.orient, cam.view.orient);
        // The file keeps `rel_m` (f64, about 1e9 m: 1.2e-7 m apart) but not its tail, and each
        // side rounds once at Mars's cell-local metres (2^-15 m apart at 1.5 AU).
        let err = view.pos.delta_from(cam.view.pos).length();
        assert!(err < 1e-4, "resolved {err} m from the camera");
    }
}
