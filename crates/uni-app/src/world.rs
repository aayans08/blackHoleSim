//! World state for the app: a thin adapter over [`uni_sim::world::World`] (M1 Task B9).
//!
//! uni-sim owns the Solar System (Standish planets, Meeus Pluto, the moons' theories), the
//! comets and asteroids, the AT-HYG stars, the frame tree and TestBH-1 (10 M☉ at SSB +
//! (1000 AU, 0, 0), as in M0). This module loads its baked data ([`load`]), keeps what the
//! renderer needs beside it (fact sheets, Saturn's ring profile, the map index: [`Assets`]),
//! and adds the few things the app asks of the world: the epoch and time rate, the lenses as
//! seen from the camera, the nearest body for the HUD, and object lookup for bookmark anchors
//! ([`uni_nav::ObjectLookup`]).

use std::path::{Path, PathBuf};

use uni_core::{Epoch, FrameId, ROOT_FRAME, UniversePos};
use uni_data::{BodyFacts, CatalogStar, DataError, MapInfo, RingProfile};
use uni_gfx::LensView;
use uni_sim::registry::ObjectId;
use uni_sim::world::{World as SimWorld, WorldData};

/// Name of the test black hole (uni-sim's `TEST_BH_NAME`).
#[cfg(test)]
pub const TEST_BH_NAME: &str = uni_sim::world::TEST_BH_NAME;

/// The world the app renders (see the module docs).
pub struct World {
    pub sim: SimWorld,
}

/// What the renderer needs beside the world: the files `World::new` consumes or does not read.
pub struct Assets {
    /// `body_facts.ron` (display names and p_V for the bodies pass).
    pub facts: Vec<BodyFacts>,
    /// `saturn_rings.ron`; `None` when missing (the ring pass is then not built, ruling A.3).
    pub rings: Option<RingProfile>,
    /// `maps/index.ron`; empty when missing (every body then draws uniform albedo).
    pub maps: Vec<MapInfo>,
    /// Where the map files are.
    pub map_dir: PathBuf,
    /// One line per missing or unreadable file (uni-sim's, then the map index's), for the HUD.
    pub warnings: Vec<String>,
}

impl Assets {
    /// Whether AT-HYG loaded: uni-sim warns about `athyg_m10.bin` otherwise.
    pub fn athyg_missing(&self) -> bool {
        self.warnings.iter().any(|w| w.contains("athyg_m10.bin"))
    }
}

/// Loads every baked file from `dir` (normally [`baked_dir`]) and builds the world at `epoch`.
/// Never panics: a missing file is one warning in [`Assets::warnings`].
pub fn load(dir: &Path, epoch: Epoch) -> (World, Assets) {
    let data = uni_sim::world::load_world_data(dir);
    let map_dir = dir.join("maps");
    let mut warnings = data.warnings.clone();
    let maps = match uni_data::load_map_index(&map_dir.join("index.ron")) {
        Ok(m) => m,
        Err(e) => {
            warnings.push(format!(
                "Surface maps unavailable ({}): every body draws uniform albedo",
                data_error(&e)
            ));
            Vec::new()
        }
    };
    if data.rings.is_none() {
        warnings.push("Saturn's ring profile missing: rings not drawn".to_owned());
    }
    let assets = Assets {
        facts: data.facts.clone(),
        rings: data.rings.clone(),
        maps,
        map_dir,
        warnings,
    };
    (World::new(data, epoch), assets)
}

/// A short reason for a data error.
fn data_error(e: &DataError) -> String {
    match e {
        DataError::NotFound(p) => format!("{} not found", p.display()),
        DataError::BadFormat(why) => format!("bad format: {why}"),
        DataError::Io(err) => err.to_string(),
    }
}

/// No baked files at all: the planets, the Sun, the Moon, the Galilean and Saturnian moons
/// from their theories, and TestBH-1; no stars, comets or asteroids.
#[cfg(test)]
pub fn empty_data() -> WorldData {
    WorldData {
        stars: Vec::new(),
        star_names: None,
        sats: Vec::new(),
        comets: Vec::new(),
        asteroids: Vec::new(),
        asteroid_names: Vec::new(),
        exo: Vec::new(),
        facts: Vec::new(),
        rings: None,
        warnings: Vec::new(),
    }
}

impl World {
    /// The world over `data` at `epoch`, its clock running at rate 1.
    pub fn new(data: WorldData, epoch: Epoch) -> Self {
        Self {
            sim: SimWorld::new(data, epoch),
        }
    }

    /// A world with no baked data ([`empty_data`]) at `epoch`: fast, for tests.
    #[cfg(test)]
    pub fn bare(epoch: Epoch) -> Self {
        Self::new(empty_data(), epoch)
    }

    /// The epoch of the cached states.
    pub fn epoch(&self) -> Epoch {
        self.sim.epoch()
    }

    /// Jumps to `epoch` and refreshes the cached states (one tick of zero wall time).
    pub fn set_epoch(&mut self, epoch: Epoch) {
        self.sim.time.set_epoch(epoch);
        self.sim.tick(0.0);
    }

    /// Simulated seconds per real second (0 while paused).
    pub fn time_rate(&self) -> f64 {
        if self.sim.time.is_paused() {
            0.0
        } else {
            self.sim.time.rate()
        }
    }

    /// Sets the time rate; 0 pauses (the rate itself is then left as it was).
    pub fn set_time_rate(&mut self, rate: f64) {
        if rate == 0.0 {
            if !self.sim.time.is_paused() {
                self.sim.time.toggle_pause();
            }
        } else {
            if self.sim.time.is_paused() {
                self.sim.time.toggle_pause();
            }
            self.sim.time.set_rate(rate);
        }
    }

    /// Advances by `dt_real_s` of wall time (A4's tick: states, near stars, comet catch-up).
    pub fn advance(&mut self, dt_real_s: f64) {
        self.sim.tick(dt_real_s);
    }

    /// Every lens as seen from `cam`: lens − camera [m] in root axes, mass and spin.
    pub fn lens_views(&self, cam: &UniversePos) -> Vec<LensView> {
        self.sim
            .lenses()
            .iter()
            .map(|l| LensView {
                rel_pos_m: l.pos.delta_from(*cam),
                mass_kg: l.mass_kg,
                spin: l.spin,
            })
            .collect()
    }

    /// The object with the nearest surface to `cam` (uni-sim's rule: bodies, TestBH-1's horizon,
    /// near stars' photospheres) and the distance from `cam` to its centre [m]; ("—", ∞) when
    /// there is none.
    pub fn nearest(&self, cam: &UniversePos) -> (String, f64) {
        match self.sim.nearest_surface(cam) {
            Some((id, _)) => (
                self.name(id).to_owned(),
                cam.delta_from(self.sim.position(id)).length(),
            ),
            None => ("\u{2014}".to_owned(), f64::INFINITY),
        }
    }

    /// The display name of object `id` ("" for an unknown id).
    pub fn name(&self, id: ObjectId) -> &str {
        self.sim
            .objects()
            .get(id as usize)
            .map_or("", |o| o.name.as_str())
    }

    /// The name of the body whose frame is `frame` (A4's frames are body-centred), for F12's
    /// anchored bookmark (A6 ruling I2).
    pub fn frame_body_name(&self, frame: FrameId) -> Option<String> {
        if frame == ROOT_FRAME {
            return None;
        }
        self.sim
            .objects()
            .iter()
            .find(|o| o.frame == Some(frame))
            .map(|o| o.name.clone())
    }

    /// Centre of object `name` (case-insensitive), root frame.
    pub fn position_of(&self, name: &str) -> Option<UniversePos> {
        self.sim.find(name).map(|id| self.sim.position(id))
    }
}

impl uni_nav::ObjectLookup for World {
    fn find_object(&self, name: &str) -> Option<(UniversePos, FrameId)> {
        let id = self.sim.find(name)?;
        let frame = self.sim.objects()[id as usize].frame.unwrap_or(ROOT_FRAME);
        Some((self.sim.position(id), frame))
    }
}

/// Loads the baked BSC5 catalogue at `path` (the window's fallback sky without AT-HYG).
/// Never panics: a missing or corrupt file gives an empty sky and one warning.
pub fn load_bsc5(path: &Path) -> (Vec<CatalogStar>, Vec<String>) {
    match uni_data::load_bsc5(path) {
        Ok(stars) => (stars, Vec::new()),
        Err(e) => (Vec::new(), vec![sky_warning(&e)]),
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

/// `epoch` as a bookmark date (UTC, whole seconds) that `crate::scene::parse_bookmark_epoch`
/// reads back to the same second, so F12 never saves a view `--capture` rejects (final review,
/// app minor 3):
/// - `YYYY-MM-DDThh:mm:ssZ` for years 0000–9999 (`Epoch::parse_iso_utc`), built on the HUD's
///   `format_epoch`, so both show the same second;
/// - ISO 8601's expanded form `±YYYYY…-MM-DDThh:mm:ssZ` (at least four digits; ISO 8601-1:2019
///   §4.3.2.2) outside them, built as the parser reads it: the date and time of the epoch
///   shifted by whole 400-year Gregorian cycles into the years 2000–2399, with the cycles'
///   years added back. (Formatting the far epoch directly would differ by the change of
///   TAI − UTC between the two eras, 27 s before 1972, and the parser would land that far off.)
pub fn iso_utc(epoch: Epoch) -> String {
    let plain = plain_iso_utc(epoch);
    let year = |s: &str| {
        let dash = 1 + s.get(1..)?.find('-')?;
        Some((s[..dash].parse::<i64>().ok()?, s[dash..].to_owned()))
    };
    match year(&plain) {
        Some((y, _)) if (0..=9999).contains(&y) => plain,
        Some((y, _)) => {
            let guess = (y - 2000).div_euclid(400);
            // The guess is off by one cycle only within seconds of a cycle's first instant.
            [guess, guess - 1, guess + 1]
                .into_iter()
                .find_map(|cycles| {
                    let shift = cycles as f64 * crate::scene::GREGORIAN_CYCLE_S;
                    let (base_year, rest) = year(&plain_iso_utc(epoch.add_seconds(-shift)))?;
                    (2000..2400).contains(&base_year).then(|| {
                        let y = base_year + 400 * cycles;
                        let sign = if y < 0 { '-' } else { '+' };
                        format!("{sign}{:04}{rest}", y.unsigned_abs())
                    })
                })
                .unwrap_or(plain)
        }
        None => plain,
    }
}

/// `epoch` as `YYYY-MM-DDThh:mm:ssZ` from the HUD's `format_epoch`, the year as it prints it.
fn plain_iso_utc(epoch: Epoch) -> String {
    let hud = uni_ui::format_epoch(epoch); // "YYYY-MM-DD hh:mm:ss UTC"
    match hud.strip_suffix(" UTC") {
        Some(t) => format!("{}Z", t.replacen(' ', "T", 1)),
        None => hud,
    }
}

/// The baked Yale Bright Star Catalogue, `<data dir>/baked/bsc5.bin` (uni-data, Task 7).
pub fn bsc5_path() -> PathBuf {
    baked_dir().join("bsc5.bin")
}

/// `<data dir>/baked`: every baked data file the app reads.
pub fn baked_dir() -> PathBuf {
    uni_data::data_dir().join("baked")
}

/// `<workspace>/bookmarks/<file>`.
pub fn bookmarks_path(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bookmarks")
        .join(file)
}

/// `bookmarks/m1.ron`: the M1 capture set and the window's start view.
pub fn m1_bookmarks_path() -> PathBuf {
    bookmarks_path("m1.ron")
}

/// `bookmarks/m0.ron`: the M0 capture set (still captured at each gate for GPU errors).
#[cfg(test)]
pub fn m0_bookmarks_path() -> PathBuf {
    bookmarks_path("m0.ron")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use glam::DVec3;
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
    use uni_core::{AU, M_SUN, gravitational_radius};

    /// The full baked world, loaded once per test binary and shared (AT-HYG and the satellite
    /// elements take about a second to load in a debug build). Tests lock it and set their own
    /// epoch.
    pub(crate) fn full_world() -> MutexGuard<'static, (World, Assets)> {
        static WORLD: OnceLock<Mutex<(World, Assets)>> = OnceLock::new();
        WORLD
            .get_or_init(|| {
                let epoch = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap();
                Mutex::new(load(&baked_dir(), epoch))
            })
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    #[test]
    fn committed_data_loads_without_warnings() {
        let guard = full_world();
        let (world, assets) = &*guard;
        assert!(assets.warnings.is_empty(), "{:?}", assets.warnings);
        assert!(!assets.athyg_missing());
        assert!(assets.rings.is_some());
        assert!(!assets.maps.is_empty());
        assert!(!assets.facts.is_empty());
        assert!(world.sim.star_data().len() >= 329_768);
        for name in [
            "Sun",
            "Earth",
            "Moon",
            "Saturn",
            "Titan",
            "Io",
            TEST_BH_NAME,
        ] {
            assert!(world.sim.find(name).is_some(), "{name}");
        }
    }

    #[test]
    fn missing_data_warns_not_panics() {
        // Review Focus 3: every missing file is a warning, never a panic.
        let (world, assets) = load(Path::new("/nonexistent"), Epoch::J2000);
        assert!(assets.athyg_missing());
        assert!(assets.rings.is_none());
        assert!(assets.maps.is_empty());
        assert!(assets.warnings.iter().any(|w| w.contains("Surface maps")));
        assert!(world.sim.star_data().is_empty());
        assert!(world.sim.find("Earth").is_some());
    }

    #[test]
    fn missing_or_corrupt_bsc5_warns_not_panics() {
        let (stars, warnings) = load_bsc5(Path::new("/nonexistent"));
        assert!(stars.is_empty());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let path = std::env::temp_dir().join(format!(
            "uni-app-{}-corrupt_catalog_warns_not_panics",
            std::process::id()
        ));
        std::fs::write(&path, [0_u8; 32]).unwrap();
        let (stars, warnings) = load_bsc5(&path);
        let _ = std::fs::remove_file(&path);
        assert!(stars.is_empty());
        assert!(warnings[0].contains("unreadable"), "{warnings:?}");
        let (stars, warnings) = load_bsc5(&bsc5_path());
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(stars.len(), 9096);
    }

    #[test]
    fn test_bh_is_10_solar_masses_1000_au_out_and_the_nearest_near_it() {
        let world = World::bare(Epoch::J2000);
        let bh = &world.sim.lenses()[0];
        assert_eq!(bh.name, TEST_BH_NAME);
        assert_eq!(bh.mass_kg, 10.0 * M_SUN);
        assert_eq!(bh.spin, 0.0);
        assert_eq!(
            bh.pos.delta_from(UniversePos::default()),
            DVec3::new(1000.0 * AU, 0.0, 0.0)
        );
        let m = gravitational_radius(bh.mass_kg);
        let cam = bh.pos.offset(DVec3::new(-60.0 * m, 0.0, 0.0));
        let (name, dist) = world.nearest(&cam);
        assert_eq!(name, TEST_BH_NAME);
        assert!((dist - 60.0 * m).abs() <= 1.0, "{dist}");
        let views = world.lens_views(&cam);
        assert_eq!(views.len(), 1);
        assert!((views[0].rel_pos_m - DVec3::new(60.0 * m, 0.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn epoch_and_rate_follow_the_clock() {
        let mut world = World::bare(Epoch::J2000);
        assert_eq!(world.time_rate(), 1.0);
        world.advance(2.0);
        assert_eq!(world.epoch().seconds_since(Epoch::J2000), 2.0);
        world.set_time_rate(0.0);
        assert_eq!(world.time_rate(), 0.0);
        world.advance(5.0);
        assert_eq!(world.epoch().seconds_since(Epoch::J2000), 2.0);
        world.set_time_rate(-10.0);
        world.advance(0.5);
        assert_eq!(world.epoch().seconds_since(Epoch::J2000), -3.0);
        let t = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap();
        let earth_then = {
            world.set_epoch(t);
            world.position_of("Earth").unwrap()
        };
        world.set_epoch(Epoch::J2000);
        assert_ne!(world.position_of("Earth").unwrap(), earth_then);
        world.set_epoch(t);
        assert_eq!(world.position_of("Earth").unwrap(), earth_then);
    }

    #[test]
    fn object_lookup_names_frames_and_centres() {
        use uni_nav::ObjectLookup as _;
        let world = World::bare(Epoch::J2000);
        let (pos, frame) = world.find_object("mars").unwrap();
        assert_eq!(pos, world.position_of("Mars").unwrap());
        assert_ne!(frame, ROOT_FRAME);
        assert_eq!(world.frame_body_name(frame).as_deref(), Some("Mars"));
        assert_eq!(world.frame_body_name(ROOT_FRAME), None);
        assert!(world.find_object("Vulcan").is_none());
    }

    /// Task C0 (ruling C.2): uni-sim's `ObjectInfo::atmosphere_top_m` (a copy, since uni-sim
    /// cannot depend on uni-atmo) equals uni-atmo's top radius for every body with a preset, and
    /// no other object has one.
    #[test]
    fn atmosphere_top_matches_atmo_presets() {
        let world = World::bare(Epoch::J2000);
        let mut found = Vec::new();
        for o in world.sim.objects() {
            let preset = o.naif.and_then(uni_atmo::preset_for);
            match (preset, o.atmosphere_top_m) {
                (Some(p), Some(top)) => {
                    assert_eq!(top, p.top_radius_m, "{}", o.name);
                    assert!(top > o.radii_m.z, "{}: top above the surface", o.name);
                    found.push(o.naif);
                }
                (None, None) => {}
                (p, top) => panic!("{}: preset {:?}, top {top:?}", o.name, p.is_some()),
            }
        }
        found.sort();
        assert_eq!(found, [Some(299), Some(399), Some(499), Some(606)]);
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

    /// Final review, app minor 3: F12 at years outside 0000–9999 (reachable with warp 10¹² and
    /// from `sky-orion-50kyr`) saves ISO 8601's expanded form, which bookmark parsing reads back
    /// to the same second: before, +52026 came out as "52026-…" (no sign) and −500 as "-500-…"
    /// (three digits), both rejected by `--capture`.
    #[test]
    fn far_epochs_round_trip_through_bookmark_parsing() {
        use crate::scene::parse_bookmark_epoch;
        let same = |a: Epoch, b: Epoch| a.seconds_since(b).abs() < 1e-3;
        for s in [
            "+52026-11-20T00:00:00Z",
            "-0500-03-21T06:30:15Z",
            "-0001-06-30T23:59:59Z",
            "-12345-07-04T12:00:00Z",
            "+2026000-02-29T01:02:03Z",
        ] {
            let e = parse_bookmark_epoch(s).unwrap_or_else(|| panic!("{s} parses"));
            assert_eq!(iso_utc(e), s);
        }
        // Within seconds of the years 0000 and 10000 the two forms meet (TAI − UTC differs by up
        // to 27 s between the eras); either form reads back to the same second.
        for s in [
            "+10000-01-01T00:00:00Z",
            "-0001-12-31T23:59:59Z",
            "+9999-12-31T23:59:58Z",
        ] {
            let e = parse_bookmark_epoch(s).unwrap_or_else(|| panic!("{s} parses"));
            let out = iso_utc(e);
            let back = parse_bookmark_epoch(&out).unwrap_or_else(|| panic!("{out} parses"));
            assert!(same(back, e), "{s} → {out}");
        }
        // An epoch reached by running time (not parsed): its string parses back to the second.
        let start = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap();
        for years in [-2_526.0, -2_027.5, 7_974.0, 50_000.25, -1.0e6] {
            let e = start.add_seconds(years * 365.25 * uni_core::DAY + 0.4);
            let s = iso_utc(e);
            let back = parse_bookmark_epoch(&s).unwrap_or_else(|| panic!("{s} parses"));
            let off = back.seconds_since(e);
            assert!((-1.0..=0.0).contains(&off), "{s}: {off} s");
            assert_eq!(iso_utc(back), s);
        }
        // Years 0000–9999 keep the plain form.
        let e = parse_bookmark_epoch("0001-01-01T00:00:00Z").unwrap();
        assert_eq!(iso_utc(e), "0001-01-01T00:00:00Z");
    }

    #[test]
    fn bookmark_files_parse() {
        let m0 = uni_nav::load_bookmarks(&m0_bookmarks_path()).expect("bookmarks/m0.ron loads");
        assert_eq!(m0.len(), 6);
        let m1 = uni_nav::load_bookmarks(&m1_bookmarks_path()).expect("bookmarks/m1.ron loads");
        assert_eq!(m1.len(), 27);
        let bench = uni_nav::load_bookmarks(&bookmarks_path("m1-bench.ron")).expect("bench loads");
        assert_eq!(bench.len(), 10);
    }
}
