//! World state: epoch, time rate, frame tree, lenses and sky loading (M0 Task 11).
//!
//! M0 has two bodies: the Sun, fixed at the Solar System barycentre (SSB), and the test black
//! hole TestBH-1. Everything is in the ICRS root frame, SI units, f64.

use std::path::{Path, PathBuf};

use glam::{DQuat, DVec3};
use uni_core::{AU, Epoch, FrameMotion, FrameTree, Lens, M_SUN, ROOT_FRAME, UniversePos};
use uni_data::{CatalogStar, DataError};
use uni_gfx::LensView;

/// Name of the M0 test black hole.
pub const TEST_BH_NAME: &str = "TestBH-1";
/// TestBH-1's mass [M☉]: a typical stellar-mass black hole (design §9, M0 "test location").
pub const TEST_BH_MASS_SUN: f64 = 10.0;
/// TestBH-1's position: SSB + (1000 AU, 0, 0), far enough that the Sun's gravity is
/// irrelevant and the sky views from the SSB do not show it.
pub const TEST_BH_DISTANCE_AU: f64 = 1000.0;
/// TestBH-1's dimensionless spin a/M. 0 (Schwarzschild) in M0; Kerr arrives in M3.
pub const TEST_BH_SPIN: f64 = 0.0;
/// Name of the Sun, which M0 places at the SSB (its ~1 R☉ wobble about the SSB is ignored).
pub const SUN_NAME: &str = "Sun";

/// Simulation state the app renders: the epoch, how fast it runs, the frame tree and the lenses.
pub struct World {
    pub epoch: Epoch,
    /// Simulated seconds per real second; 0 is paused.
    pub time_rate: f64,
    pub frames: FrameTree,
    pub lenses: Vec<Lens>,
}

impl World {
    /// The M0 world at `epoch`, running in real time: the Sun at the SSB and TestBH-1, a
    /// 10 M☉ Schwarzschild hole held in a fixed frame at SSB + (1000 AU, 0, 0).
    pub fn new(epoch: Epoch) -> Self {
        let mut frames = FrameTree::new();
        frames.add(
            ROOT_FRAME,
            TEST_BH_NAME,
            FrameMotion::Fixed {
                origin: UniversePos::from_meters(DVec3::new(TEST_BH_DISTANCE_AU * AU, 0.0, 0.0)),
                rotation: DQuat::IDENTITY,
            },
        );
        let test_bh = Lens {
            name: TEST_BH_NAME.to_owned(),
            pos: UniversePos::default(), // placed from its frame below
            mass_kg: TEST_BH_MASS_SUN * M_SUN,
            spin: TEST_BH_SPIN,
        };
        let mut world = World {
            epoch,
            time_rate: 1.0,
            frames,
            lenses: vec![test_bh],
        };
        world.place_lenses();
        world
    }

    /// Advances the epoch by `dt_real_s` real seconds at the current time rate, and moves the
    /// lenses with their frames.
    pub fn advance(&mut self, dt_real_s: f64) {
        let dt = dt_real_s * self.time_rate;
        if dt.is_finite() && dt != 0.0 {
            self.epoch = self.epoch.add_seconds(dt);
            self.place_lenses();
        }
    }

    /// Puts each lens at the origin of the frame named after it, at the current epoch.
    fn place_lenses(&mut self) {
        for lens in &mut self.lenses {
            if let Some(id) = self.frames.find(&lens.name) {
                lens.pos = self.frames.origin_in_root(id, self.epoch);
            }
        }
    }

    /// The nearest body to `cam` and its centre distance [m]. M0 considers the Sun at the SSB
    /// and every lens; ties go to the Sun.
    pub fn nearest(&self, cam: &UniversePos) -> (String, f64) {
        let sun = (SUN_NAME, cam.delta_from(UniversePos::default()).length());
        let lenses = self
            .lenses
            .iter()
            .map(|l| (l.name.as_str(), cam.delta_from(l.pos).length()));
        let (name, dist) = lenses.fold(sun, |best, c| if c.1 < best.1 { c } else { best });
        (name.to_owned(), dist)
    }

    /// Every lens as seen from `cam`: lens − camera [m] in root axes, mass and spin.
    pub fn lens_views(&self, cam: &UniversePos) -> Vec<LensView> {
        self.lenses
            .iter()
            .map(|l| LensView {
                rel_pos_m: l.pos.delta_from(*cam),
                mass_kg: l.mass_kg,
                spin: l.spin,
            })
            .collect()
    }

    /// Loads the baked star catalogue at `path`. Never panics: a missing or corrupt file gives
    /// an empty sky and one warning for the HUD (Review Focus 3).
    pub fn load_sky(path: &Path) -> (Vec<CatalogStar>, Vec<String>) {
        match uni_data::load_bsc5(path) {
            Ok(stars) => (stars, Vec::new()),
            Err(e) => (Vec::new(), vec![sky_warning(&e)]),
        }
    }
}

/// The HUD line for a catalogue that could not be loaded.
fn sky_warning(e: &DataError) -> String {
    match e {
        DataError::NotFound(p) => format!("Star catalogue missing ({}): empty sky", p.display()),
        DataError::BadFormat(why) => format!("Star catalogue unreadable ({why}): empty sky"),
        DataError::Io(err) => format!("Star catalogue unreadable ({err}): empty sky"),
    }
}

/// `epoch` as `YYYY-MM-DDThh:mm:ssZ` (UTC, whole seconds), the form bookmarks use
/// (`Epoch::parse_iso_utc`). Built on the HUD's `format_epoch`, so both show the same second.
pub fn iso_utc(epoch: Epoch) -> String {
    let hud = uni_ui::format_epoch(epoch); // "YYYY-MM-DD hh:mm:ss UTC"
    match hud.strip_suffix(" UTC") {
        Some(t) => format!("{}Z", t.replacen(' ', "T", 1)),
        None => hud,
    }
}

/// The baked Yale Bright Star Catalogue, `<data dir>/baked/bsc5.bin` (uni-data, Task 7).
pub fn sky_path() -> PathBuf {
    uni_data::data_dir().join("baked").join("bsc5.bin")
}

/// `bookmarks/m0.ron` in the workspace: the M0 capture sheet and the windowed start view.
pub fn m0_bookmarks_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bookmarks/m0.ron")
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_core::{gravitational_radius, radec_to_icrs};
    use uni_nav::Bookmark;

    #[test]
    fn nearest_is_test_bh_at_start() {
        let world = World::new(Epoch::J2000);
        let bh = &world.lenses[0];
        let m = gravitational_radius(10.0 * M_SUN);
        let cam = bh.pos.offset(DVec3::new(-60.0 * m, 0.0, 0.0));
        let (name, dist) = world.nearest(&cam);
        assert_eq!(name, "TestBH-1");
        assert!((dist - 60.0 * m).abs() <= 1.0, "{dist} m vs {} m", 60.0 * m);
    }

    #[test]
    fn bookmarks_file_parses() {
        let marks = uni_nav::load_bookmarks(&m0_bookmarks_path()).expect("bookmarks/m0.ron loads");
        let names: Vec<&str> = marks.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "start",
                "sky-orion",
                "sky-galactic-center",
                "sky-zoom-sirius",
                "bh-shadow-30M",
                "bh-oblique-15M",
            ]
        );
    }

    #[test]
    fn missing_catalog_warns_not_panics() {
        // Review Focus 3: a missing data file gives an empty sky and one warning.
        let (stars, warnings) = World::load_sky(Path::new("/nonexistent"));
        assert!(stars.is_empty());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
    }

    #[test]
    fn corrupt_catalog_warns_not_panics() {
        let path = std::env::temp_dir().join(format!(
            "uni-app-{}-corrupt_catalog_warns_not_panics",
            std::process::id()
        ));
        std::fs::write(&path, [0_u8; 32]).unwrap();
        let (stars, warnings) = World::load_sky(&path);
        let _ = std::fs::remove_file(&path);
        assert!(stars.is_empty());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("unreadable"), "{warnings:?}");
    }

    #[test]
    fn committed_catalog_loads_without_warnings() {
        let (stars, warnings) = World::load_sky(&sky_path());
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(stars.len(), 9096);
    }

    #[test]
    fn nearest_is_the_sun_near_the_ssb() {
        let world = World::new(Epoch::J2000);
        let cam = UniversePos::from_meters(DVec3::new(0.0, 3.0 * AU, 4.0 * AU));
        let (name, dist) = world.nearest(&cam);
        assert_eq!(name, "Sun");
        assert!((dist - 5.0 * AU).abs() < 1e-3, "{dist}");
    }

    #[test]
    fn test_bh_is_10_solar_masses_1000_au_out_with_no_spin() {
        let world = World::new(Epoch::J2000);
        assert_eq!(world.lenses.len(), 1);
        let bh = &world.lenses[0];
        assert_eq!(bh.name, TEST_BH_NAME);
        assert_eq!(bh.mass_kg, 10.0 * M_SUN);
        assert_eq!(bh.spin, 0.0);
        assert_eq!(
            bh.pos.delta_from(UniversePos::default()),
            DVec3::new(1000.0 * AU, 0.0, 0.0)
        );
        let frame = world.frames.find(TEST_BH_NAME).expect("TestBH-1 frame");
        assert_eq!(world.frames.origin_in_root(frame, world.epoch), bh.pos);
    }

    #[test]
    fn lens_views_are_lens_minus_camera() {
        let world = World::new(Epoch::J2000);
        let cam = world.lenses[0].pos.offset(DVec3::new(-100.0, 20.0, 3.0));
        let views = world.lens_views(&cam);
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].rel_pos_m, DVec3::new(100.0, -20.0, -3.0));
        assert_eq!(views[0].mass_kg, 10.0 * M_SUN);
    }

    #[test]
    fn advance_scales_real_time_by_the_rate() {
        let mut world = World::new(Epoch::J2000);
        world.advance(2.0);
        assert_eq!(world.epoch.seconds_since(Epoch::J2000), 2.0);
        world.time_rate = 0.0;
        world.advance(5.0);
        world.time_rate = -10.0;
        world.advance(0.5);
        assert_eq!(world.epoch.seconds_since(Epoch::J2000), -3.0);
        world.time_rate = 1.0;
        world.advance(f64::NAN);
        assert_eq!(world.epoch.seconds_since(Epoch::J2000), -3.0);
    }

    #[test]
    fn iso_utc_round_trips_through_bookmark_parsing() {
        for s in [
            "2026-11-20T00:00:00Z",
            "2016-12-31T23:59:59Z",
            "1969-07-20T20:17:40Z",
        ] {
            let e = Epoch::parse_iso_utc(s).unwrap();
            assert_eq!(iso_utc(e), s);
        }
        assert_eq!(iso_utc(Epoch::J2000), "2000-01-01T11:58:55Z");
    }

    // ---- bookmarks/m0.ron says what its comments say -----------------------------------

    fn bookmark(name: &str) -> Bookmark {
        uni_nav::load_bookmarks(&m0_bookmarks_path())
            .expect("bookmarks/m0.ron loads")
            .into_iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("no bookmark {name}"))
    }

    /// Camera forward (−Z), right (+X) and up (+Y) in ICRS axes.
    fn axes(b: &Bookmark) -> (DVec3, DVec3, DVec3) {
        let q = b.view.orient;
        assert!(
            (q.length() - 1.0).abs() < 1e-12,
            "{}: |q| = {}",
            b.name,
            q.length()
        );
        (q * DVec3::NEG_Z, q * DVec3::X, q * DVec3::Y)
    }

    fn angle(a: DVec3, b: DVec3) -> f64 {
        a.cross(b).length().atan2(a.dot(b))
    }

    /// North up: the camera's right axis is level (no +z part) and its up axis leans north.
    fn assert_north_up(b: &Bookmark) {
        let (_, right, up) = axes(b);
        assert!(right.z.abs() < 1e-12, "{}: right {right}", b.name);
        assert!(up.z > 0.0, "{}: up {up}", b.name);
    }

    /// Hole direction from the bookmark's camera, and its distance in M.
    fn hole_from(b: &Bookmark) -> (DVec3, f64) {
        let world = World::new(Epoch::J2000);
        let bh = &world.lenses[0];
        let d = bh.pos.delta_from(b.view.pos);
        (d.normalize(), d.length() / gravitational_radius(bh.mass_kg))
    }

    #[test]
    fn every_bookmark_is_at_2026_11_20() {
        for b in uni_nav::load_bookmarks(&m0_bookmarks_path()).unwrap() {
            assert_eq!(
                b.epoch_utc.as_deref(),
                Some("2026-11-20T00:00:00Z"),
                "{}",
                b.name
            );
            assert!(Epoch::parse_iso_utc(b.epoch_utc.as_deref().unwrap()).is_some());
        }
    }

    #[test]
    fn sky_bookmarks_look_at_their_targets_from_the_ssb() {
        let deg = f64::to_radians;
        for (name, ra, dec, fov) in [
            ("sky-orion", 83.8, -1.2, 60.0),
            ("sky-galactic-center", 266.405, -28.936, 60.0),
            ("sky-zoom-sirius", 101.2872, -16.7161, 2.0),
        ] {
            let b = bookmark(name);
            assert_eq!(b.view.pos.delta_from(UniversePos::default()), DVec3::ZERO);
            assert!((b.view.fov_y_rad - deg(fov)).abs() < 1e-15, "{name}");
            let (fwd, _, _) = axes(&b);
            let a = angle(fwd, radec_to_icrs(deg(ra), deg(dec)));
            assert!(a < 1e-12, "{name}: {a} rad off target");
            assert_north_up(&b);
            assert_eq!(b.lens_scale, None, "{name}");
        }
    }

    #[test]
    fn start_is_60m_out_with_the_hole_15_deg_right_of_centre() {
        let b = bookmark("start");
        let (dir, r) = hole_from(&b);
        assert!((r - 60.0).abs() < 1e-6, "r = {r} M");
        let (fwd, right, up) = axes(&b);
        assert!((angle(fwd, dir) - 15_f64.to_radians()).abs() < 1e-6);
        assert!(dir.dot(right) > 0.0, "the hole is right of centre");
        assert!(dir.dot(up).abs() < 1e-6, "on the horizontal midline");
        assert_north_up(&b);
        assert!((b.view.fov_y_rad - 60_f64.to_radians()).abs() < 1e-15);
    }

    #[test]
    fn shadow_bookmark_is_a_static_observer_at_30m_looking_at_the_hole() {
        let b = bookmark("bh-shadow-30M");
        let (dir, r) = hole_from(&b);
        assert!((r - 30.0).abs() < 1e-6, "r = {r} M");
        let (fwd, _, _) = axes(&b);
        // 1e-7 rad is 0.03 px at 1080 rows and 60°: the shadow is centred.
        assert!(angle(fwd, dir) < 1e-7, "{} rad", angle(fwd, dir));
        assert_north_up(&b);
        assert!((b.view.fov_y_rad - 60_f64.to_radians()).abs() < 1e-15);
        assert_eq!(b.lens_scale, Some(1.0));
    }

    #[test]
    fn oblique_bookmark_is_15m_out_with_the_hole_20_deg_left() {
        let b = bookmark("bh-oblique-15M");
        let (dir, r) = hole_from(&b);
        assert!((r - 15.0).abs() < 1e-6, "r = {r} M");
        let (fwd, right, up) = axes(&b);
        assert!((angle(fwd, dir) - 20_f64.to_radians()).abs() < 1e-6);
        assert!(dir.dot(right) < 0.0, "the hole is left of centre");
        assert!(dir.dot(up).abs() < 1e-6, "on the horizontal midline");
        // Turned about its own up axis while looking down at the hole, so the horizon tilts;
        // up still leans north.
        assert!(up.z > 0.0, "up {up}");
        assert!((b.view.fov_y_rad - 60_f64.to_radians()).abs() < 1e-15);
    }
}
