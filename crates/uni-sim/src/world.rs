//! The world: data loading, the registry, the frame tree and per-tick cached states (M1 Task
//! A4).
//!
//! **Registry.** Object ids index [`World::objects`] in this order: the ephemeris bodies (Sun,
//! planets, Pluto, moons; parents first), comets, the asteroids that are objects (H ≤ 10, or
//! with a sheet in `body_facts.ron`), TestBH-1, then exoplanet hosts and named stars, all built
//! by [`World::new`]. Unnamed catalogue stars are not objects (plan A4's registry list) until a
//! camera comes within 0.1 pc of one: a snapshot or `nearest_surface` notes it, and the next
//! [`World::tick`] appends it as a `Star` named "AT-HYG <id>" (review I1), so it can be drawn
//! as a star surface (`StarSurfaceView::id` names an object) and its photosphere can be the
//! nearest surface. The registry only grows; an id, once given, never changes. Until that tick
//! (one frame) the star is a point without an id.
//!
//! **Frames.** ROOT (SSB, ICRS) → `Sun` → each planet → each moon, every one a
//! `FrameMotion::Custom` that translates only (ICRS axes); TestBH-1 sits in a fixed frame under
//! ROOT, as in M0. Body orientations come from `uni_physics::rotation`, not from frames.
//!
//! **States.** [`World::tick`] evaluates the body source once per body and caches every
//! Solar System position (and TestBH-1's) at the new epoch; stars are placed on demand from
//! their catalogue positions and velocities. Cached positions are summed cell by cell exactly
//! as [`FrameTree`] does, so `position(id)` equals the body frame's origin bit for bit.
//!
//! **Comets** (lead ruling after A3) move two-body from an osculating conic that follows the
//! shown date: once the date is more than 30 days from a comet's conic epoch, the comet is
//! re-osculated with `uni_physics::smallbody::osculate_at` (planet-perturbed) from whichever of
//! its catalogue conic, the conics it reached before and its current conic is nearest in time.
//! No comet is shown from its catalogue conic for the start date (review I3), and no frame
//! waits for a whole integration (review r1 N1):
//! - [`World::new`] settles every comet at the start date ([`World::settle_comets`]): one
//!   integration each, in 120-day segments that are all kept as checkpoints, spread over the
//!   CPU cores with `std::thread::scope` (ruling 3), longest first. That is a one-off start-up
//!   cost, listed there.
//! - A tick never settles. It runs a fixed number of short steps (`REOSC_STEPS_PER_TICK`, at
//!   most `REOSC_STEPS_PER_COMET` per comet) on scoped threads: first for the selected comet
//!   and the comets the last snapshot drew (brightest first), then for the other stale comets
//!   nearest the Sun. A step integrates `REOSC_CHUNK_STEPS` of the integrator's RK4 steps (at
//!   most 16 days), so its cost stays near 0.1 ms with optimised physics, near perihelion too.
//!   The work per tick does not depend on wall time, so the comets' state does not depend on
//!   the machine (review r1 N2).
//! - A comet keeps the conics it reached (`CHECKPOINT_SPACING_S`, 120 days, apart), from its
//!   catalogue epoch to the start date and wherever ticks took it. A jump to any such date
//!   adopts the nearest conic and needs at most ~38 days of integration, one or two ticks for
//!   a drawn comet; a return to the start date needs none (review r2 P1).
//! - Outside the spans a comet has covered (a jump past the start date, or before its catalogue
//!   epoch), and under a warp faster than the steps keep up with, a comet lags the date and is
//!   drawn two-body from the nearest conic it has until it catches up: a drawn comet at 24 RK4
//!   steps per tick (Halley, ten years past the start date, in 77 ticks, 1.3 s at 60 fps), the
//!   whole catalogue in minutes after a decades-long jump. No honesty label covers that lag yet
//!   (it needs a new `HonestyFlag`, a stub change; A4 report, fix round 3). Captures call
//!   [`World::settle_comets`] after setting a bookmark's epoch.
//!
//! Chunked steps cost metres at most against one integration. Perturbed motion is followed only
//! within `PERTURBED_SPAN_S` of the catalogue epoch; beyond it the comet is two-body from the
//! conic at that edge.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use glam::{DQuat, DVec3, UVec2};
use uni_core::{
    AU, CameraView, DAY, Epoch, FrameId, FrameMotion, FrameTree, GM_SUN, Lens, M_SUN, PC,
    R_SUN_M, ROOT_FRAME, UniversePos, bv_to_teff, radec_to_icrs, schwarzschild_radius,
    star_illuminance_lux,
};
use uni_data::{
    AsteroidRecord, BodyFacts, Comet, DataError, ExoSystem, FLAG_EXO_HOST, FLAG_FROM_ARCHIVE,
    FLAG_NO_VELOCITY, RingProfile, STAR3_EPOCH_JD_TDB, SatRecord, Star3, StarNames,
};
use uni_physics::bodies::{by_naif, system_gm_m3s2};
use uni_physics::calc::hill_radius;
use uni_physics::ephemeris::Ephemeris;
use uni_physics::frames::ecliptic_j2000_to_icrs;
use uni_physics::kepler::{Conic, perifocal_to_reference, solve_kepler};
use uni_physics::photometry::star_radius_m;
use uni_physics::sat_elements::SatElements;
use uni_physics::smallbody::{conic_from_perihelion, heliocentric_state_icrs, osculate_at};
use uni_physics::standish::{self, Planet, Validity};
use uni_physics::{pluto, rotation};

use crate::honesty::HonestyFlag;
use crate::registry::{
    NameTable, ObjectId, ObjectInfo, ObjectKind, Physical, Source, brdf_for, has_atmosphere,
};
use crate::snapshot::{self, Snapshot};
use crate::starindex::{StarIndex, star_pos_pc};
use crate::time::TimeController;

/// Julian year [s]: 365.25 days of 86 400 s (IAU 1976 System of Astronomical Constants).
const YEAR_S: f64 = 365.25 * DAY;
/// JD (TDB) of J2000.0 (IERS Conventions 2010).
const JD_J2000: f64 = 2_451_545.0;

/// The Sun's sphere of influence [m]: the Galactic tidal radius, 2.3 × 10⁵ AU (Chebotarev
/// 1965, Soviet Astronomy 8, 787). Outside it the dominant frame is ROOT.
const SUN_SPHERE_M: f64 = 2.3e5 * AU;
/// Catalogue stars nearer than this [pc] are drawn as surfaces or points by the snapshot, not
/// by the star pass (plan A4).
pub(crate) const NEAR_STAR_PC: f64 = 0.1;
/// At most this many near stars are taken out of the star pass (plan A4: `suppressed_stars`
/// ≤ 8).
pub(crate) const MAX_NEAR_STARS: usize = 8;
/// Star positions are linear motion; beyond this many years from the catalogue epoch the HUD
/// says so (plan "Honesty labels in M1").
const STAR_LINEAR_LIMIT_YR: f64 = 1.0e5;
/// The honesty frustum test does not know the viewport, so it takes the widest frame the app
/// renders (21:9 ≈ 2.33, rounded up): a Pluto just off a narrower screen may still raise the
/// label, never the reverse.
const HONESTY_MAX_ASPECT: f64 = 2.4;

/// Name of the M0 test black hole (as `uni_app::world::TEST_BH_NAME`).
pub const TEST_BH_NAME: &str = "TestBH-1";
/// TestBH-1's mass [M☉]: a typical stellar-mass black hole (design §9, M0 "test location").
const TEST_BH_MASS_SUN: f64 = 10.0;
/// TestBH-1 sits at SSB + (1000 AU, 0, 0) (M0, unchanged).
const TEST_BH_DISTANCE_AU: f64 = 1000.0;

/// Half-step [s] of the central differences that give ephemeris velocities. Phobos (period
/// 7.65 h) is the fastest-turning body: its truncation error is v(ωh)²/6 = 2 mm/s at 10 s, and
/// rounding at Neptune's 4.5 × 10¹² m is 10⁻³ m / 20 s = 5 × 10⁻⁵ m/s.
const VELOCITY_STEP_S: f64 = 10.0;

/// Radius [m] assumed for a moon whose size no data file gives (414 of JPL's 460 moons have
/// no radius in `sat_elements.ron`): 2 km, the same assumption plan C2 makes for comet nuclei
/// without a diameter. Not a measurement; such moons are a few km across at most.
const ASSUMED_RADIUS_M: f64 = 2.0e3;
/// Bulk density [kg/m³] assumed for a moon without a GM, used only for its Hill sphere (frame
/// anchoring), never shown: of the order of small moons' measured densities (0.4–0.6 g/cm³ for
/// Saturn's small inner moons, Thomas 2010, Icarus 208, 395; 1.9 g/cm³ for Phobos).
const ASSUMED_DENSITY_KG_M3: f64 = 1000.0;
/// V geometric albedo assumed for a moon without one in `body_facts.ron`. An assumption, not a
/// measurement: most such moons are small irregular moons, darker than this, so they may be
/// drawn somewhat too bright (they are V ≳ 15 points from anywhere but close by).
const DEFAULT_MOON_ALBEDO: f64 = 0.1;
/// V geometric albedo of comet nuclei, 0.04 (Lamy et al. 2004, in *Comets II*, 223).
const COMET_ALBEDO: f64 = 0.04;
/// Asteroid diameter from H and p_V: D = 1329 km / √p_V · 10^(−H/5) (Harris & Harris 1997,
/// Icarus 126, 450, eq. 2, as NASA's NEO Properties tool cites it; JPL CNEOS's size estimator,
/// https://cneos.jpl.nasa.gov/tools/ast_size_est.html, uses the same 10^3.1236 km = 1329 km).
const H_DIAMETER_KM: f64 = 1329.0;
/// p_V assumed for an asteroid with neither a measured albedo nor a diameter: 0.14 ± 0.02, the
/// debiased mean of near-Earth objects (Stuart & Binzel 2004, Icarus 170, 295). The point's
/// brightness at zero phase does not depend on it (D² p_V is fixed by H); only the size does.
const ASSUMED_ASTEROID_ALBEDO: f64 = 0.14;
/// Asteroids with H at or below this are registry objects (plan A4), and so is any asteroid
/// with a sheet in `body_facts.ron` (none today); the others are `uni-sky`'s points. An
/// asteroid without an H is not "H ≤ 10".
const REGISTRY_MAX_H: f32 = 10.0;
/// SBDB SPK-ID of (134340) Pluto: the ephemeris places Pluto, so its SBDB row is skipped.
const SPKID_PLUTO: u32 = 20_134_340;
/// SBDB SPK-IDs of the IAU's other dwarf planets (IAU 2006 Resolutions B5 and B6; the WGSBN
/// list): (1) Ceres, (136108) Haumea, (136199) Eris, (136472) Makemake.
const DWARF_PLANET_SPKIDS: [u32; 4] = [20_000_001, 20_136_108, 20_136_199, 20_136_472];

/// Saturn's ring radii [m] unless [`World::set_ring_profile`] gives others: the span of A8's
/// `saturn_rings.ron`, so `RingView` covers exactly the radii of the profile B2 draws (review
/// I2). A8's bake (`tools/bake/src/rings.rs`) fitted that span to Solar System Scope's ring
/// texture by placing the texture's Cassini-division drop at the Huygens gap, 117 680 km, and
/// its A ring's outer edge at 136 775 km, both the M1 plan's values (NSSDCA's Saturnian Rings
/// Fact Sheet gives 117 507 km and 136 780 km): 69 934–140 972 km. `WorldData` has no ring
/// field (A0b stub), so the test `saturn_carries_its_rings` pins these to the baked file, and a
/// re-bake that moves them fails there.
const SATURN_RINGS_DEFAULT_M: (f64, f64) = (69_933_828.337_874_65, 140_972_084.468_664_85);
/// NAIF id of Saturn, whose rings the snapshot reports.
pub(crate) const SATURN: i32 = 699;
const SUN: i32 = 10;
const PLUTO: i32 = 999;
const CHARON: i32 = 901;

/// A comet is re-osculated once the shown date is this far from its conic's epoch (lead ruling
/// after A3: "e.g. when the shown date moves > 30 days from the last osculation epoch").
const REOSC_INTERVAL_S: f64 = 30.0 * DAY;
/// `uni_physics::smallbody`'s RK4 step rule, repeated here (its private `STEP_ETA` and
/// `MAX_STEP_S`) to size the tick steps: η = 10⁻³ of the shortest local dynamical time
/// √(d³/GM) about the Sun and the eight planetary systems, at most 2 days
/// ([`smallbody_rk4_step_s`]).
pub(crate) const SMALLBODY_STEP_ETA: f64 = 1.0e-3;
/// See [`SMALLBODY_STEP_ETA`].
pub(crate) const SMALLBODY_MAX_STEP_S: f64 = 2.0 * DAY;
/// A tick step integrates this many of `uni_physics::smallbody`'s RK4 steps, sized at its
/// start: `REOSC_CHUNK_STEPS ·` [`smallbody_rk4_step_s`], so 16 days at most. One RK4 step costs
/// 16 Standish planet states; measured on the M3 over the baked catalogue's first steps (review
/// r1 N1), a step of about 45 RK4 steps took 1.25 ms in the debug build and 0.40 ms with
/// uni-physics and uni-core at opt-level 3. Eight keep a step near 0.09 ms there, near a planet
/// too, so a tick's comet work stays inside its share of the frame budget (ruling 3).
pub(crate) const REOSC_CHUNK_STEPS: f64 = 8.0;
/// Longest tick step [s] (far from the Sun, where the 2-day RK4 step applies).
pub(crate) const REOSC_CHUNK_S: f64 = REOSC_CHUNK_STEPS * SMALLBODY_MAX_STEP_S;
/// Shortest tick step [s], so a comet at r → 0 still moves on.
pub(crate) const REOSC_MIN_CHUNK_S: f64 = 60.0;
/// Most comet steps one tick runs, in all (review r1 N1: a tick never integrates a whole
/// re-osculation, so a date jump or the end of a warp never stalls a frame). A fixed count, not
/// a wall-time budget, so the comets' state after a run of ticks is the same on every machine and
/// in every run (review r1 N2). They run on up to [`std::thread::available_parallelism`] scoped
/// threads (ruling 3), [`REOSC_STEPS_PER_COMET`] at most for each comet. On the M3's 8 cores,
/// with every workspace crate at opt-level 3 in the test build (review r1 N1; the release
/// build is the gate's to measure), 24 one-step jobs took 0.41 ms of wall time and a whole
/// catch-up tick after a 30-year jump 0.47 ms; 16 jobs took 0.35 ms. A catalogue whose conics
/// all went stale at once catches up at about 24 × 8 = 192 RK4 steps per tick. The selected
/// comet and the comets the last snapshot drew go first, [`REOSC_STEPS_PER_COMET`] steps each
/// (review r2 P1), then the others nearest the Sun first. Measured the same way with those
/// priorities (fix round 3): the slowest of 30 ticks after the 30-year jump 0.51–0.62 ms
/// (two runs); catching up a jump to 2020, 0.38–0.39 ms per tick on average; a drawn Halley's
/// three steps plus 21 others, 0.50 ms on average.
const REOSC_STEPS_PER_TICK: usize = 24;
/// Most steps one comet takes per tick; they run one after another on one thread. The
/// selected and drawn comets always get this many while the budget lasts: 3 × 8 RK4 steps per
/// tick, the share of one of the M3's 8 threads.
const REOSC_STEPS_PER_COMET: usize = 3;
/// A comet keeps the conics it was re-osculated to, at least this far apart [s] (a whole number
/// of days, so a settle's segments, exactly this far apart, are all kept). A settle
/// ([`World::settle_comets`], so [`World::new`] too) integrates in segments of this length, so
/// a comet has a conic at most ~68 days (half of this plus one 16-day tick step) from every
/// date between its catalogue epoch and the start date, and from every date it reached later.
/// A jump to such a date needs at most ~38 days of integration after it adopts that conic,
/// one tick for a drawn comet far from the Sun, and none for a date within 30 days of one
/// (review r2 P1; a whole-year spacing left up to half a year). The baked catalogue's
/// construction spans (6694 comet-years) keep about 20 400 conics, 1.6 MB at 80 B each with the
/// epoch; the whole ±200-year perturbed span would be at most 1219 conics (98 kB) per comet.
const CHECKPOINT_SPACING_S: f64 = 120.0 * DAY;

/// The local dynamical time √(r³/GM☉) [s] at `r_m` from the Sun: the time scale that sets
/// `uni_physics::smallbody`'s RK4 step (DE440 GM☉; Kepler's third law: one radian of a
/// circular orbit).
fn dynamical_time_s(r_m: f64) -> f64 {
    (r_m.powi(3) / uni_physics::bodies::GM_SUN_DE440_M3S2).sqrt()
}

/// `uni_physics::smallbody`'s RK4 step [s] for a small body at heliocentric `r` [m] (ICRS) at
/// `t`: η times the least of √(|r|³/GM☉) and, for each planetary system's barycentre at
/// distance d, √(d³/GM) with its DE440 system GM (positions from Standish, as the integrator
/// takes them), at most 2 days. A NaN r (a broken conic) gives the 2-day step: `min` drops a
/// NaN.
pub(crate) fn smallbody_rk4_step_s(r: DVec3, t: Epoch) -> f64 {
    let mut t_dyn = dynamical_time_s(r.length());
    for planet in Planet::ALL {
        let gm = system_gm_m3s2(planet.barycentre_id()).unwrap_or(0.0);
        if gm > 0.0 {
            let d = (standish::heliocentric_state_icrs(planet, t).0 - r).length();
            t_dyn = t_dyn.min((d.powi(3) / gm).sqrt());
        }
    }
    (SMALLBODY_STEP_ETA * t_dyn).min(SMALLBODY_MAX_STEP_S)
}

/// The span [s] of a tick step from heliocentric `r` [m] at `t` ([`REOSC_CHUNK_STEPS`]).
pub(crate) fn short_span_s(r: DVec3, t: Epoch) -> f64 {
    (REOSC_CHUNK_STEPS * smallbody_rk4_step_s(r, t))
        .min(REOSC_CHUNK_S)
        .max(REOSC_MIN_CHUNK_S)
}

/// How far one comet re-osculation step integrates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chunk {
    /// All the way to the date, in segments of at most [`CHECKPOINT_SPACING_S`], each kept as
    /// a checkpoint ([`World::settle_comets`]).
    Whole,
    /// At most [`short_span_s`] (ticks).
    Short,
}
/// Perturbed motion is followed this far from a comet's catalogue epoch [s]. Beyond it the
/// comet's non-gravitational forces (not modelled; Yeomans & Kiang 1981 give Halley's as about
/// 4 days per orbit) dominate the error anyway, and `perturbed_state_icrs`'s cost grows with
/// the span.
const PERTURBED_SPAN_S: f64 = 200.0 * YEAR_S;

/// Every baked data file the world uses; a missing file leaves its field empty and adds one
/// warning.
pub struct WorldData {
    pub stars: Vec<Star3>,
    pub star_names: Option<StarNames>,
    pub sats: Vec<SatRecord>,
    pub comets: Vec<Comet>,
    pub asteroids: Vec<AsteroidRecord>,
    pub asteroid_names: Vec<String>,
    pub exo: Vec<ExoSystem>,
    pub facts: Vec<BodyFacts>,
    pub warnings: Vec<String>,
}

/// Loads every baked file from `dir` (normally `uni_data::data_dir()/baked`). Never panics;
/// each missing file gives one warning, and so does each unreadable one.
pub fn load_world_data(dir: &Path) -> WorldData {
    let mut warnings = Vec::new();
    let read = |file: &str| dir.join(file);

    let stars = warn_on(
        uni_data::load_athyg(&read("athyg_m10.bin")),
        "athyg_m10.bin",
        dir,
        "no AT-HYG stars (the app falls back to the BSC5 sky)",
        &mut warnings,
    );
    let star_names = warn_on(
        uni_data::load_star_names(&read("athyg_names.ron")),
        "athyg_names.ron",
        dir,
        "stars have no names",
        &mut warnings,
    );
    let sats = warn_on(
        uni_data::load_sat_elements(&read("sat_elements.ron")),
        "sat_elements.ron",
        dir,
        "only the Moon, the Galilean moons and Saturn's eight major moons (Meeus theories)",
        &mut warnings,
    );
    let comets = warn_on(
        uni_data::load_comets(&read("comets.ron")),
        "comets.ron",
        dir,
        "no comets",
        &mut warnings,
    );
    // Two files make one catalogue: each missing one is reported, and the pair loads only
    // when both are there.
    let pair = [read("asteroids.bin"), read("asteroid_names.txt")];
    let mut have_pair = true;
    for path in &pair {
        if !path.is_file() {
            let file = path
                .file_name()
                .map_or_else(String::new, |f| f.to_string_lossy().into_owned());
            warnings.push(format!(
                "{file} missing in {}: no catalogue asteroids",
                dir.display()
            ));
            have_pair = false;
        }
    }
    let (asteroids, asteroid_names) = if have_pair {
        warn_on(
            uni_data::load_asteroids(&pair[0], &pair[1]),
            "asteroids.bin",
            dir,
            "no catalogue asteroids",
            &mut warnings,
        )
        .unwrap_or_default()
    } else {
        Default::default()
    };
    let exo = warn_on(
        uni_data::load_exoplanets(&read("exoplanets.ron")),
        "exoplanets.ron",
        dir,
        "no exoplanet systems",
        &mut warnings,
    );
    let facts = warn_on(
        uni_data::load_body_facts(&read("body_facts.ron")),
        "body_facts.ron",
        dir,
        "no fact sheets (assumed albedos)",
        &mut warnings,
    );
    WorldData {
        stars: stars.unwrap_or_default(),
        star_names,
        sats: sats.unwrap_or_default(),
        comets: comets.unwrap_or_default(),
        asteroids,
        asteroid_names,
        exo: exo.unwrap_or_default(),
        facts: facts.unwrap_or_default(),
        warnings,
    }
}

/// `result`'s value, or `None` with one warning naming `file` and what is missing without it.
fn warn_on<T>(
    result: Result<T, DataError>,
    file: &str,
    dir: &Path,
    without: &str,
    warnings: &mut Vec<String>,
) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(DataError::NotFound(_)) => {
            warnings.push(format!("{file} missing in {}: {without}", dir.display()));
            None
        }
        Err(err) => {
            warnings.push(format!("{file} unreadable ({err}): {without}"));
            None
        }
    }
}

/// Where the Solar System bodies come from: the real [`Ephemeris`], or a fake in tests (the
/// brief's seam, so the world's logic is tested apart from A2's numbers).
pub(crate) trait BodySource: Send + Sync {
    /// (NAIF id, parent NAIF id) of every body; the Sun's parent is `None` (the SSB).
    fn bodies(&self) -> Vec<(i32, Option<i32>)>;
    /// Position relative to the parent [m], ICRS axes (the Sun: relative to the SSB).
    fn position_rel_parent(&self, naif: i32, t: Epoch) -> Option<DVec3>;
    /// Body-fixed → ICRS rotation, `None` without a rotation model.
    fn orient(&self, naif: i32, t: Epoch) -> Option<DQuat>;
}

impl BodySource for Ephemeris {
    fn bodies(&self) -> Vec<(i32, Option<i32>)> {
        Ephemeris::bodies(self)
            .iter()
            .map(|b| (b.naif, b.parent))
            .collect()
    }

    fn position_rel_parent(&self, naif: i32, t: Epoch) -> Option<DVec3> {
        Ephemeris::position_rel_parent(self, naif, t)
    }

    fn orient(&self, naif: i32, t: Epoch) -> Option<DQuat> {
        rotation::body_to_icrs(naif, t)
    }
}

/// The body's offset from its parent at `t`, or zero when the source has none (or a
/// non-finite one), so a bad value never spreads NaN down the frame tree.
fn rel_or_zero(source: &dyn BodySource, naif: i32, t: Epoch) -> DVec3 {
    source
        .position_rel_parent(naif, t)
        .filter(|v| v.is_finite())
        .unwrap_or(DVec3::ZERO)
}

/// `parent` + `rel` [m], summed cell by cell exactly as `FrameTree` resolves a root-aligned
/// frame, so a cached position equals its frame's origin bit for bit.
fn add_rel(parent: UniversePos, rel: DVec3) -> UniversePos {
    let o = UniversePos::from_meters(rel);
    UniversePos {
        cell: parent.cell.saturating_add(o.cell),
        local: parent.local + o.local,
    }
    .normalized()
}

/// The epoch of a TDB Julian Date.
fn epoch_from_jd(jd_tdb: f64) -> Epoch {
    Epoch::from_days_since_j2000(jd_tdb - JD_J2000)
}

/// `uni_data::SatRecord` (km, JD) → `uni_physics::sat_elements::SatElements` (m, Epoch).
fn sat_elements(r: &SatRecord) -> SatElements {
    SatElements {
        naif: r.naif,
        name: r.name.clone(),
        planet: r.planet_naif,
        a_m: r.a_km * 1e3,
        e: r.e,
        argp_deg: r.argp_deg,
        m_deg: r.m_deg,
        i_deg: r.i_deg,
        node_deg: r.node_deg,
        n_deg_per_day: r.n_deg_per_day,
        p_apsides_yr: r.p_apsides_yr,
        p_node_yr: r.p_node_yr,
        epoch: epoch_from_jd(r.epoch_jd_tdb),
        laplace_ra_deg: r.laplace_ra_deg,
        laplace_dec_deg: r.laplace_dec_deg,
    }
}

/// Luminous intensity [cd] of a star of absolute V magnitude `abs_mag_v`: its illuminance at
/// 10 pc times (10 pc)², on `star_illuminance_lux`'s zero point (Allen 1973).
pub(crate) fn luminous_intensity_cd(abs_mag_v: f64) -> f64 {
    let ten_pc = 10.0 * PC;
    star_illuminance_lux(abs_mag_v) * ten_pc * ten_pc
}

/// The Sun's luminous intensity [cd]: 1.26 × 10⁵ lx at 1 AU (M_V☉ = 4.83, Cox 2000).
pub(crate) fn sun_intensity_cd() -> f64 {
    luminous_intensity_cd(uni_core::M_V_SUN)
}

/// B−V of a star of effective temperature `teff_k`: the inverse of `bv_to_teff` (Ballesteros
/// 2012, which falls monotonically in B−V), by bisection over its clamp range [−0.4, 2.0].
/// Hotter or cooler stars get the range's ends.
fn bv_from_teff(teff_k: f64) -> f64 {
    let (mut lo, mut hi) = (-0.4_f64, 2.0_f64);
    if teff_k >= bv_to_teff(lo) {
        return lo;
    }
    if teff_k <= bv_to_teff(hi) {
        return hi;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if bv_to_teff(mid) > teff_k {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// SBDB "1 Ceres (A801 AA)" → "Ceres"; an unnamed "123456 (2001 AB12)" keeps its full name.
fn asteroid_display_name(full: &str) -> &str {
    let full = full.trim();
    let rest = full.trim_start_matches(|c: char| c.is_ascii_digit()).trim_start();
    let core = match rest.find(" (") {
        Some(at) => &rest[..at],
        None if rest.starts_with('(') => "",
        None => rest,
    };
    if core.is_empty() { full } else { core }
}

/// A comet's orbit: two-body from an osculating conic that follows the shown date (module
/// doc).
#[derive(Clone, Debug)]
struct CometOrbit {
    /// The epoch of SBDB's elements, the centre of the perturbed span.
    catalogue_epoch: Epoch,
    /// Conics this comet has held, sorted by epoch and at least [`CHECKPOINT_SPACING_S`]
    /// apart, starting with SBDB's (J2000 ecliptic). A re-osculation starts from the nearest
    /// of them and the current conic.
    checkpoints: Vec<(Epoch, Conic)>,
    /// The conic used now, osculating at `osc_epoch`.
    osc: Conic,
    osc_epoch: Epoch,
    /// SBDB's total-magnitude parameters (M1, K1), when it has both.
    magnitude: Option<(f64, f64)>,
    /// The perturbed integration failed (a non-finite result): stay two-body.
    failed: bool,
}

impl CometOrbit {
    fn new(c: &Comet) -> Self {
        let catalogue =
            conic_from_perihelion(c.q_au, c.e, c.i_deg, c.node_deg, c.argp_deg, c.tp_jd_tdb);
        let catalogue_epoch = epoch_from_jd(c.epoch_jd_tdb);
        CometOrbit {
            catalogue_epoch,
            checkpoints: vec![(catalogue_epoch, catalogue)],
            osc: catalogue,
            osc_epoch: catalogue_epoch,
            magnitude: c.m1.zip(c.k1),
            failed: false,
        }
    }

    /// `t` moved into the span where perturbed motion is followed (`t` itself, bit for bit,
    /// inside it).
    fn target(&self, t: Epoch) -> Epoch {
        let span = t.seconds_since(self.catalogue_epoch);
        if span.abs() <= PERTURBED_SPAN_S {
            return t;
        }
        self.catalogue_epoch
            .add_seconds(span.clamp(-PERTURBED_SPAN_S, PERTURBED_SPAN_S))
    }

    fn needs_work(&self, t: Epoch) -> bool {
        !self.failed && self.target(t).seconds_since(self.osc_epoch).abs() > REOSC_INTERVAL_S
    }

    /// The conic a re-osculation toward `target` starts from: the nearest in time of the
    /// checkpoints and the current conic (the current one on a tie).
    fn start_for(&self, target: Epoch) -> (Epoch, Conic) {
        let gap = |e: Epoch| target.seconds_since(e).abs();
        let at = self.checkpoints.partition_point(|(e, _)| e.seconds_since(target) < 0.0);
        let mut best = (self.osc_epoch, self.osc);
        for &(e, c) in self.checkpoints[at.saturating_sub(1)..].iter().take(2) {
            if gap(e) < gap(best.0) {
                best = (e, c);
            }
        }
        best
    }

    /// The integration a re-osculation toward `t` needs [s], from [`CometOrbit::start_for`]
    /// (what [`World::settle_comets`] orders its work by).
    fn work_span_s(&self, t: Epoch) -> f64 {
        let target = self.target(t);
        target.seconds_since(self.start_for(target).0).abs()
    }

    /// Takes the checkpoint nearest the shown date when it is within [`REOSC_INTERVAL_S`] of
    /// it, which needs no integration; returns whether the conic changed.
    fn adopt(&mut self, t: Epoch) -> bool {
        if !self.needs_work(t) {
            return false;
        }
        let target = self.target(t);
        let (epoch, conic) = self.start_for(target);
        let near = target.seconds_since(epoch).abs() <= REOSC_INTERVAL_S;
        if near && epoch != self.osc_epoch {
            (self.osc, self.osc_epoch) = (conic, epoch);
            return true;
        }
        false
    }

    /// Keeps the current conic as a checkpoint when no checkpoint is within
    /// [`CHECKPOINT_SPACING_S`] of it, or `always` (a settle's date, the likeliest return)
    /// unless one has its epoch.
    fn remember(&mut self, always: bool) {
        let e = self.osc_epoch;
        let at = self.checkpoints.partition_point(|(c, _)| c.seconds_since(e) < 0.0);
        if self.checkpoints.get(at).is_some_and(|(c, _)| *c == e) {
            return;
        }
        let near = |k: usize| {
            self.checkpoints
                .get(k)
                .is_some_and(|(c, _)| e.seconds_since(*c).abs() < CHECKPOINT_SPACING_S)
        };
        if always || !(near(at) || (at > 0 && near(at - 1))) {
            self.checkpoints.insert(at, (e, self.osc));
        }
    }

    /// One re-osculation toward the shown date, as far as `chunk` allows; returns whether the
    /// conic changed. Starts from [`CometOrbit::start_for`]. A short step stops within
    /// [`REOSC_INTERVAL_S`] of the date; a whole one reaches it, in segments of at most
    /// [`CHECKPOINT_SPACING_S`], each kept.
    fn step(&mut self, t: Epoch, chunk: Chunk) -> bool {
        if !self.needs_work(t) {
            return false;
        }
        let target = self.target(t);
        let (start_epoch, start) = self.start_for(target);
        if start_epoch != self.osc_epoch {
            self.osc = start;
            self.osc_epoch = start_epoch;
            if target.seconds_since(start_epoch).abs() <= REOSC_INTERVAL_S {
                return true;
            }
        }
        match chunk {
            Chunk::Whole => {
                while self.osc_epoch != target {
                    let left = target.seconds_since(self.osc_epoch);
                    let to = if left.abs() <= CHECKPOINT_SPACING_S {
                        target
                    } else {
                        self.osc_epoch.add_seconds(CHECKPOINT_SPACING_S.copysign(left))
                    };
                    if !self.advance(to, to == target) {
                        break;
                    }
                }
            }
            Chunk::Short => {
                let r = heliocentric_state_icrs(&self.osc, self.osc_epoch).0;
                let max_span_s = short_span_s(r, self.osc_epoch);
                let span = target
                    .seconds_since(self.osc_epoch)
                    .clamp(-max_span_s, max_span_s);
                self.advance(self.osc_epoch.add_seconds(span), false);
            }
        }
        true
    }

    /// Re-osculates the current conic at `to` (`uni_physics::smallbody::osculate_at`) and
    /// keeps it as a checkpoint ([`CometOrbit::remember`]); a non-finite result marks the comet
    /// failed (it stays two-body from its last good conic). Returns whether it moved.
    fn advance(&mut self, to: Epoch, always_keep: bool) -> bool {
        let next = osculate_at(&self.osc, self.osc_epoch, to);
        if next.q_m.is_finite() && next.q_m > 0.0 && next.e.is_finite() {
            self.osc = next;
            self.osc_epoch = to;
            self.remember(always_keep);
            true
        } else {
            self.failed = true;
            false
        }
    }

    /// Heliocentric position [m] and velocity [m/s], ICRS axes.
    fn state(&self, t: Epoch) -> (DVec3, DVec3) {
        heliocentric_state_icrs(&self.osc, t)
    }
}

/// An asteroid's two-body ellipse from its SBDB elements, with the perifocal axes rotated to
/// ICRS once, so a state costs one Kepler solve. The same orbit as
/// `smallbody::heliocentric_state_icrs(&smallbody::conic_from_mean_anomaly(..), t)`.
#[derive(Clone, Debug)]
struct AsteroidOrbit {
    a_m: f64,
    e: f64,
    /// Mean motion √(GM☉/a³) [rad/s] (Kepler's third law, DE440 GM☉).
    n_rad_s: f64,
    period_s: f64,
    m0_rad: f64,
    epoch: Epoch,
    /// Unit vectors toward periapsis and 90° ahead of it, ICRS.
    p: DVec3,
    q: DVec3,
}

impl AsteroidOrbit {
    fn new(r: &AsteroidRecord) -> Self {
        let a_m = r.a_au * AU;
        let n_rad_s = (uni_physics::bodies::GM_SUN_DE440_M3S2 / a_m).sqrt() / a_m;
        let rot = perifocal_to_reference(
            r.i_deg.to_radians(),
            r.node_deg.to_radians(),
            r.argp_deg.to_radians(),
        );
        AsteroidOrbit {
            a_m,
            e: r.e,
            n_rad_s,
            period_s: std::f64::consts::TAU / n_rad_s,
            m0_rad: r.m0_deg.to_radians(),
            epoch: epoch_from_jd(r.epoch_jd_tdb),
            p: ecliptic_j2000_to_icrs(rot.x_axis),
            q: ecliptic_j2000_to_icrs(rot.y_axis),
        }
    }

    /// Heliocentric position [m] and velocity [m/s], ICRS axes.
    fn state(&self, t: Epoch) -> (DVec3, DVec3) {
        // Whole periods are removed from Δt first, so the mean anomaly keeps its precision at
        // any epoch.
        let dt = t.seconds_since(self.epoch);
        let dt = dt - self.period_s * (dt / self.period_s).round();
        let ea = solve_kepler(self.m0_rad + self.n_rad_s * dt, self.e);
        let (sin_e, cos_e) = ea.sin_cos();
        let b = self.a_m * (1.0 - self.e * self.e).sqrt();
        let e_dot = self.n_rad_s / (1.0 - self.e * cos_e);
        (
            self.p * (self.a_m * (cos_e - self.e)) + self.q * (b * sin_e),
            (self.q * (b * cos_e) - self.p * (self.a_m * sin_e)) * e_dot,
        )
    }
}

/// The simulated Solar System and sky.
pub struct World {
    pub time: TimeController,
    pub(crate) source: Arc<dyn BodySource>,
    pub(crate) frames: FrameTree,
    pub(crate) objects: Vec<ObjectInfo>,
    /// Parallel to `objects`.
    pub(crate) physical: Vec<Physical>,
    names: NameTable,
    /// The epoch of the cached states.
    pub(crate) epoch: Epoch,
    /// Root positions at `epoch` of every Solar System object and TestBH-1: ids
    /// `0..cached.len()`. Stars (the ids after them) are placed on demand.
    pub(crate) cached: Vec<UniversePos>,
    /// The ephemeris bodies, parents first.
    pub(crate) ephem_order: Vec<ObjectId>,
    pub(crate) sun: Option<ObjectId>,
    comets: Vec<CometOrbit>,
    /// The registry object of each comet orbit.
    comet_objects: Vec<ObjectId>,
    asteroids: Vec<AsteroidOrbit>,
    /// AT-HYG stars, then archive-only exoplanet hosts.
    pub(crate) stars: Vec<Star3>,
    /// T_eff [K] of archive-only hosts whose archive row gives one, by star index.
    archive_teff: HashMap<u32, f64>,
    pub(crate) star_index: StarIndex,
    /// The registry object of a star, by star index.
    pub(crate) star_objects: HashMap<u32, ObjectId>,
    /// Unnamed catalogue stars a snapshot or [`World::nearest_surface`] found within 0.1 pc,
    /// registered as objects at the next tick (review I1).
    unnamed_near: Mutex<Vec<u32>>,
    /// The comets the last snapshot drew (comet index, illuminance at the camera [lx], +∞ for
    /// a resolved nucleus): ticks re-osculate the stale ones first, brightest first (review
    /// r2 P1). Each snapshot replaces it.
    shown_comets: Mutex<Vec<(usize, f64)>>,
    exo: Vec<ExoSystem>,
    lenses: Vec<Lens>,
    /// Saturn's ring radii (inner, outer) [m].
    pub(crate) ring_m: (f64, f64),
    band_enabled: bool,
    selected: Option<ObjectId>,
}

impl World {
    /// The world at `epoch`, every comet already re-osculated there ([`World::settle_comets`]:
    /// one integration per comet on every core, the cost listed there).
    pub fn new(data: WorldData, epoch: Epoch) -> Self {
        let sats = data.sats.iter().map(sat_elements).collect();
        World::with_source(Arc::new(Ephemeris::new(sats)), data, epoch)
    }

    /// The world over any body source (tests use a fake one), comets settled as in
    /// [`World::new`].
    pub(crate) fn with_source(source: Arc<dyn BodySource>, data: WorldData, epoch: Epoch) -> Self {
        let mut world = World {
            time: TimeController::new(epoch),
            source,
            frames: FrameTree::new(),
            objects: Vec::new(),
            physical: Vec::new(),
            names: NameTable::default(),
            epoch,
            cached: Vec::new(),
            ephem_order: Vec::new(),
            sun: None,
            comets: Vec::new(),
            comet_objects: Vec::new(),
            asteroids: Vec::new(),
            stars: Vec::new(),
            archive_teff: HashMap::new(),
            star_index: StarIndex::new(&[]),
            star_objects: HashMap::new(),
            unnamed_near: Mutex::new(Vec::new()),
            shown_comets: Mutex::new(Vec::new()),
            exo: Vec::new(),
            lenses: Vec::new(),
            ring_m: SATURN_RINGS_DEFAULT_M,
            band_enabled: true,
            selected: None,
        };
        let albedos: HashMap<i32, f64> = data
            .facts
            .iter()
            .filter_map(|f| Some((f.naif, f.geometric_albedo?)))
            .collect();
        let sheets: HashSet<i32> = data.facts.iter().map(|f| f.naif).collect();
        world.add_ephemeris_bodies(&data.sats, &albedos);
        world.add_comets(&data.comets);
        world.add_asteroids(&data.asteroids, &data.asteroid_names, &sheets);
        world.add_test_black_hole();
        world.cached = vec![UniversePos::default(); world.objects.len()];
        world.refresh_states();
        world.settle_comets();
        world.set_hill_radii();
        world.add_stars(data.stars, data.star_names, data.exo);
        world
    }

    /// Registers an object under the first free name among `names` (all of them, and
    /// `aliases`, also find it) and returns its id.
    fn push(
        &mut self,
        mut info: ObjectInfo,
        physical: Physical,
        names: &[&str],
        aliases: &[&str],
    ) -> ObjectId {
        let id = self.objects.len() as ObjectId;
        info.id = id;
        info.name = self.names.unique(names, id);
        self.names.insert(&info.name, id);
        for alias in names.iter().chain(aliases) {
            self.names.insert(alias, id);
        }
        self.objects.push(info);
        self.physical.push(physical);
        id
    }

    /// The source's bodies, parents first. A body whose parent the source does not list is
    /// left out, and so is a repeated NAIF id.
    fn add_ephemeris_bodies(&mut self, sats: &[SatRecord], albedos: &HashMap<i32, f64>) {
        let mut placed: HashMap<i32, ObjectId> = HashMap::new();
        let mut pending = self.source.bodies();
        loop {
            let mut rest = Vec::new();
            let before = pending.len();
            for (naif, parent) in pending {
                if placed.contains_key(&naif) {
                    continue;
                }
                let parent_id = match parent {
                    None => None,
                    Some(p) => match placed.get(&p) {
                        Some(&id) => Some(id),
                        None => {
                            rest.push((naif, parent));
                            continue;
                        }
                    },
                };
                let sat = sats.iter().find(|s| s.naif == naif);
                let id = self.add_ephem_body(naif, parent_id, sat, albedos.get(&naif).copied());
                placed.insert(naif, id);
            }
            if rest.is_empty() || rest.len() == before {
                break;
            }
            pending = rest;
        }
    }

    fn add_ephem_body(
        &mut self,
        naif: i32,
        parent: Option<ObjectId>,
        sat: Option<&SatRecord>,
        albedo: Option<f64>,
    ) -> ObjectId {
        let known = by_naif(naif);
        let name = known
            .map(|b| b.name.to_owned())
            .or_else(|| sat.map(|s| s.name.clone()))
            .unwrap_or_else(|| format!("NAIF {naif}"));
        let parent_naif = parent.and_then(|p| self.objects[p as usize].naif);
        let kind = match (naif, parent_naif) {
            (SUN, _) => ObjectKind::Sun,
            (PLUTO, _) => ObjectKind::DwarfPlanet,
            (_, None | Some(SUN)) => ObjectKind::Planet,
            _ => ObjectKind::Moon,
        };
        let radii_m = known.map(|b| b.radii_m).unwrap_or_else(|| {
            DVec3::splat(
                sat.and_then(|s| s.radius_km)
                    .filter(|r| r.is_finite() && *r > 0.0)
                    .map_or(ASSUMED_RADIUS_M, |r| r * 1e3),
            )
        });
        let gm_m3s2 = known.map(|b| b.gm_m3s2).unwrap_or_else(|| {
            sat.and_then(|s| s.gm_km3_s2)
                .filter(|g| g.is_finite() && *g > 0.0)
                .map_or(0.0, |g| g * 1e9)
        });
        let (parent_frame, depth) = match parent {
            Some(p) => (
                self.objects[p as usize].frame.unwrap_or(ROOT_FRAME),
                self.physical[p as usize].depth + 1,
            ),
            None => (ROOT_FRAME, 1),
        };
        // The frame carries the object's (unique) registry name, so `FrameTree::find` finds it.
        let name = self.names.unique(&[&name], self.objects.len() as ObjectId);
        let source = Arc::clone(&self.source);
        let frame = self.frames.add(
            parent_frame,
            &name,
            FrameMotion::Custom(Box::new(move |t| {
                (
                    UniversePos::from_meters(rel_or_zero(&*source, naif, t)),
                    DQuat::IDENTITY,
                )
            })),
        );
        let info = ObjectInfo {
            id: 0,
            name: String::new(),
            kind,
            naif: Some(naif),
            catalog_index: None,
            parent,
            radii_m,
            gm_m3s2,
            frame: Some(frame),
            hill_radius_m: 0.0,
        };
        let physical = Physical {
            source: Source::Ephem,
            depth,
            brdf: brdf_for(Some(naif)),
            geometric_albedo: albedo.unwrap_or(if naif == SUN { 0.0 } else { DEFAULT_MOON_ALBEDO }),
            mean_radius_m: mean_radius(radii_m),
            has_atmosphere: has_atmosphere(naif),
        };
        let id = self.push(info, physical, &[&name], &[]);
        if naif == SUN && self.sun.is_none() {
            self.sun = Some(id);
        }
        self.ephem_order.push(id);
        id
    }

    fn add_comets(&mut self, comets: &[Comet]) {
        for c in comets {
            let radius = c
                .diameter_km
                .filter(|d| d.is_finite() && *d > 0.0)
                .map_or(ASSUMED_RADIUS_M, |d| 0.5e3 * d);
            let info = ObjectInfo {
                id: 0,
                name: String::new(),
                kind: ObjectKind::Comet,
                naif: Some(c.spkid),
                catalog_index: None,
                parent: self.sun,
                radii_m: DVec3::splat(radius),
                gm_m3s2: 0.0,
                frame: None,
                hill_radius_m: 0.0,
            };
            let physical = Physical {
                source: Source::Comet(self.comets.len()),
                depth: 0,
                brdf: brdf_for(None),
                geometric_albedo: COMET_ALBEDO,
                mean_radius_m: radius,
                has_atmosphere: false,
            };
            self.comets.push(CometOrbit::new(c));
            let id = self.push(info, physical, &[&c.name], &[]);
            self.comet_objects.push(id);
        }
    }

    /// The asteroids that are objects (plan A4): H ≤ 10, or a sheet in `body_facts.ron`
    /// (`sheets`, NAIF ids, joined through `asteroid_naif_id`); never SBDB's Pluto row.
    fn add_asteroids(
        &mut self,
        records: &[AsteroidRecord],
        names: &[String],
        sheets: &HashSet<i32>,
    ) {
        for r in records {
            let naif = uni_data::smallbody::asteroid_naif_id(r.spkid);
            let has_sheet = naif.is_some_and(|n| sheets.contains(&n));
            if !(r.h <= REGISTRY_MAX_H || has_sheet) || r.spkid == SPKID_PLUTO {
                continue;
            }
            let full = names
                .get(r.name_index as usize)
                .cloned()
                .unwrap_or_else(|| format!("SPK-ID {}", r.spkid));
            let (diameter_km, albedo) = asteroid_size(r);
            let radius = 0.5e3 * diameter_km;
            let kind = if DWARF_PLANET_SPKIDS.contains(&r.spkid) {
                ObjectKind::DwarfPlanet
            } else {
                ObjectKind::Asteroid
            };
            let info = ObjectInfo {
                id: 0,
                name: String::new(),
                kind,
                naif,
                catalog_index: None,
                parent: self.sun,
                radii_m: DVec3::splat(radius),
                gm_m3s2: 0.0,
                frame: None,
                hill_radius_m: 0.0,
            };
            let physical = Physical {
                source: Source::Asteroid(self.asteroids.len()),
                depth: 0,
                brdf: brdf_for(None),
                geometric_albedo: albedo,
                mean_radius_m: radius,
                has_atmosphere: false,
            };
            self.asteroids.push(AsteroidOrbit::new(r));
            self.push(info, physical, &[asteroid_display_name(&full), &full], &[]);
        }
    }

    /// TestBH-1: a 10 M☉ Schwarzschild hole in a fixed frame at SSB + (1000 AU, 0, 0) (M0).
    fn add_test_black_hole(&mut self) {
        let pos = UniversePos::from_meters(DVec3::new(TEST_BH_DISTANCE_AU * AU, 0.0, 0.0));
        let frame = self.frames.add(
            ROOT_FRAME,
            TEST_BH_NAME,
            FrameMotion::Fixed {
                origin: pos,
                rotation: DQuat::IDENTITY,
            },
        );
        let mass_kg = TEST_BH_MASS_SUN * M_SUN;
        let horizon = schwarzschild_radius(mass_kg);
        let info = ObjectInfo {
            id: 0,
            name: String::new(),
            kind: ObjectKind::BlackHole,
            naif: None,
            catalog_index: None,
            parent: None,
            radii_m: DVec3::splat(horizon),
            gm_m3s2: TEST_BH_MASS_SUN * GM_SUN,
            frame: Some(frame),
            hill_radius_m: 0.0,
        };
        let physical = Physical {
            source: Source::BlackHole(self.lenses.len()),
            depth: 1,
            brdf: brdf_for(None),
            geometric_albedo: 0.0,
            mean_radius_m: horizon,
            has_atmosphere: false,
        };
        self.lenses.push(Lens {
            name: TEST_BH_NAME.to_owned(),
            pos,
            mass_kg,
            spin: 0.0,
        });
        self.push(info, physical, &[TEST_BH_NAME], &[]);
    }

    /// Hill radii at the construction epoch: each body's osculating orbit about its parent
    /// (a, e from `Conic::from_state` with μ = GM_parent + GM_body; a circle of the current
    /// distance when that is not an ellipse, e.g. a source that does not move), then
    /// `calc::hill_radius`. The Sun's sphere is [`SUN_SPHERE_M`]. TestBH-1's is where its pull
    /// equals the Sun's on the line between them, d·√q/(1 + √q) for the mass ratio q
    /// (Newton's law of gravitation), so the camera rides with the hole near it.
    fn set_hill_radii(&mut self) {
        let t = self.epoch;
        for k in 0..self.objects.len() {
            let id = k as ObjectId;
            let info = &self.objects[k];
            let hill = match (self.physical[k].source, info.parent) {
                (Source::Ephem, None) if info.naif == Some(SUN) => SUN_SPHERE_M,
                (Source::Ephem, Some(parent)) => {
                    let parent_gm = self.objects[parent as usize].gm_m3s2;
                    let gm = self.gm_for_hill(id);
                    let r = self.cached[k].delta_from(self.cached[parent as usize]);
                    let v = self.velocity(id) - self.velocity(parent);
                    let conic = Conic::from_state(r, v, parent_gm + gm, t);
                    let (a, e) = if conic.e.is_finite() && conic.e < 1.0 && conic.q_m > 0.0 {
                        (conic.q_m / (1.0 - conic.e), conic.e)
                    } else {
                        (r.length(), 0.0)
                    };
                    if parent_gm > 0.0 {
                        hill_radius(a, e, gm, parent_gm)
                    } else {
                        0.0
                    }
                }
                (Source::BlackHole(_), _) => match self.sun {
                    Some(sun) => {
                        let d = self.cached[k].delta_from(self.cached[sun as usize]).length();
                        let sqrt_q = (info.gm_m3s2 / self.objects[sun as usize].gm_m3s2).sqrt();
                        d * sqrt_q / (1.0 + sqrt_q)
                    }
                    None => f64::INFINITY,
                },
                _ => 0.0,
            };
            // A NaN (degenerate input) gives no sphere.
            self.objects[k].hill_radius_m = if hill >= 0.0 { hill } else { 0.0 };
        }
    }

    /// GM [m³/s²] for a body's Hill sphere: its own, or for a moon without one, its mean radius
    /// at [`ASSUMED_DENSITY_KG_M3`].
    fn gm_for_hill(&self, id: ObjectId) -> f64 {
        let gm = self.objects[id as usize].gm_m3s2;
        if gm > 0.0 {
            return gm;
        }
        let r = self.physical[id as usize].mean_radius_m;
        uni_core::G * ASSUMED_DENSITY_KG_M3 * 4.0 / 3.0 * std::f64::consts::PI * r * r * r
    }

    /// Archive-only exoplanet hosts (review I11), then one object per exoplanet system and one
    /// per named star; builds the star index.
    fn add_stars(&mut self, mut stars: Vec<Star3>, names: Option<StarNames>, exo: Vec<ExoSystem>) {
        let catalogue_len = stars.len();
        let mut hosts = Vec::with_capacity(exo.len());
        for (k, sys) in exo.iter().enumerate() {
            let index = match sys.star_index {
                Some(i) if (i as usize) < catalogue_len => Some(i),
                // No AT-HYG match, or no AT-HYG loaded: the archive's own values.
                _ => archive_star(sys, k as u32).map(|star| {
                    stars.push(star);
                    let index = (stars.len() - 1) as u32;
                    if let Some(teff) = sys.teff_k.filter(|t| t.is_finite() && *t > 0.0) {
                        self.archive_teff.insert(index, teff);
                    }
                    index
                }),
            };
            hosts.push(index);
        }
        self.stars = stars;
        let name_of = |index: u32| {
            names
                .as_ref()
                .filter(|_| (index as usize) < catalogue_len)
                .and_then(|n| n.get(index))
        };
        for (sys, index) in exo.iter().zip(hosts) {
            let Some(index) = index else { continue };
            let entry = name_of(index);
            let proper = entry.and_then(|e| e.proper.as_deref());
            let bayer = entry.and_then(|e| e.bayer_flamsteed.as_deref());
            let candidates: Vec<&str> = [proper, Some(sys.host.as_str()), bayer]
                .into_iter()
                .flatten()
                .collect();
            let id = self.push_star(index, ObjectKind::ExoHost, &candidates);
            self.star_objects.entry(index).or_insert(id);
        }
        if let Some(names) = &names {
            for entry in &names.entries {
                let index = entry.index;
                if index as usize >= catalogue_len || self.star_objects.contains_key(&index) {
                    continue;
                }
                let candidates: Vec<&str> =
                    [entry.proper.as_deref(), entry.bayer_flamsteed.as_deref()]
                        .into_iter()
                        .flatten()
                        .collect();
                if candidates.is_empty() {
                    continue;
                }
                let id = self.push_star(index, ObjectKind::Star, &candidates);
                self.star_objects.insert(index, id);
            }
        }
        self.exo = exo;
        self.star_index = StarIndex::new(&self.stars);
    }

    fn push_star(&mut self, index: u32, kind: ObjectKind, names: &[&str]) -> ObjectId {
        let radius = self.star_radius_m(index);
        let info = ObjectInfo {
            id: 0,
            name: String::new(),
            kind,
            naif: None,
            catalog_index: Some(index),
            parent: None,
            radii_m: DVec3::splat(radius),
            gm_m3s2: 0.0,
            frame: None,
            hill_radius_m: 0.0,
        };
        let physical = Physical {
            source: Source::Star,
            depth: 0,
            brdf: brdf_for(None),
            geometric_albedo: 0.0,
            mean_radius_m: radius,
            has_atmosphere: false,
        };
        self.push(info, physical, names, &[])
    }

    /// T_eff [K] of star `index`: the archive's for an archive-only host that has one, else
    /// from B−V (Ballesteros 2012).
    pub(crate) fn star_teff_k(&self, index: u32) -> f64 {
        match self.archive_teff.get(&index) {
            Some(&teff) => teff,
            None => self
                .stars
                .get(index as usize)
                .map_or(uni_core::TEFF_SUN_K, |s| bv_to_teff(f64::from(s.bv()))),
        }
    }

    /// Photosphere radius [m] of star `index` from M_V and T_eff (Torres 2010 bolometric
    /// corrections, Stefan–Boltzmann; `photometry::star_radius_m`); R☉ if that is not finite.
    pub(crate) fn star_radius_m(&self, index: u32) -> f64 {
        let Some(star) = self.stars.get(index as usize) else {
            return R_SUN_M;
        };
        let r = star_radius_m(f64::from(star.abs_mag_v), self.star_teff_k(index));
        if r.is_finite() && r > 0.0 { r } else { R_SUN_M }
    }

    /// Years since the star catalogue's epoch (J2000.0 for AT-HYG v4.0).
    pub(crate) fn star_years(&self) -> f64 {
        self.epoch
            .seconds_since(epoch_from_jd(STAR3_EPOCH_JD_TDB))
            / YEAR_S
    }

    /// Root position of star `index` at the current epoch.
    pub(crate) fn star_position(&self, index: u32) -> UniversePos {
        match self.stars.get(index as usize) {
            Some(star) => UniversePos::from_meters(star_pos_pc(star, self.star_years()) * PC),
            None => UniversePos::default(),
        }
    }

    /// Catalogue stars within [`NEAR_STAR_PC`] of `pos`, nearest first, at most
    /// [`MAX_NEAR_STARS`]: (star index, its registry object, `None` for an unnamed star not
    /// registered yet). Each such unnamed star is registered at the next tick (review I1).
    pub(crate) fn near_stars(&self, pos: &UniversePos) -> Vec<(u32, Option<ObjectId>)> {
        let p_pc = pos.delta_from(UniversePos::default()) / PC;
        let near: Vec<(u32, Option<ObjectId>)> = self
            .star_index
            .within(&self.stars, p_pc, self.star_years(), NEAR_STAR_PC, MAX_NEAR_STARS)
            .into_iter()
            .map(|(index, _)| (index, self.star_objects.get(&index).copied()))
            .collect();
        if near.iter().any(|(_, id)| id.is_none()) {
            let mut pending = self.unnamed_near.lock().unwrap_or_else(PoisonError::into_inner);
            for &(index, _) in near.iter().filter(|(_, id)| id.is_none()) {
                if !pending.contains(&index) {
                    pending.push(index);
                }
            }
        }
        near
    }

    /// Registers the unnamed catalogue stars found near a camera since the last tick (review
    /// I1): one `Star` object each, named "AT-HYG <id>" by its AT-HYG v4.0 id (the catalogue's
    /// `id` column), so it can become a star surface and its photosphere a nearest surface.
    /// Ids already handed out never change; the registry only grows.
    fn register_near_stars(&mut self) {
        let pending = std::mem::take(
            self.unnamed_near
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner),
        );
        for index in pending {
            let Some(star) = self.stars.get(index as usize) else {
                continue;
            };
            if self.star_objects.contains_key(&index) {
                continue;
            }
            let name = format!("AT-HYG {}", star.athyg_id());
            let id = self.push_star(index, ObjectKind::Star, &[&name]);
            self.star_objects.insert(index, id);
        }
    }

    /// Re-evaluates every cached position at `self.epoch`: one source evaluation per body.
    fn refresh_states(&mut self) {
        let t = self.epoch;
        for &id in &self.ephem_order {
            let info = &self.objects[id as usize];
            let parent = info.parent.map_or(UniversePos::default(), |p| self.cached[p as usize]);
            let naif = info.naif.unwrap_or_default();
            self.cached[id as usize] = add_rel(parent, rel_or_zero(&*self.source, naif, t));
        }
        self.refresh_small_bodies();
    }

    /// Comets, asteroids and TestBH-1 at `self.epoch`, relative to the cached Sun.
    fn refresh_small_bodies(&mut self) {
        let t = self.epoch;
        let sun = self.sun.map_or(UniversePos::default(), |s| self.cached[s as usize]);
        for k in 0..self.cached.len() {
            let helio = match self.physical[k].source {
                Source::Comet(c) => self.comets[c].state(t).0,
                Source::Asteroid(a) => self.asteroids[a].state(t).0,
                Source::BlackHole(l) => {
                    self.cached[k] = self.lenses[l].pos;
                    continue;
                }
                Source::Ephem | Source::Star => continue,
            };
            if helio.is_finite() {
                self.cached[k] = add_rel(sun, helio);
            }
        }
    }

    /// Root offset [m] of an ephemeris body from the SSB at `t`, summed up its parent chain.
    fn ssb_offset(&self, id: ObjectId, t: Epoch) -> DVec3 {
        let mut sum = DVec3::ZERO;
        let mut at = Some(id);
        while let Some(k) = at {
            let info = &self.objects[k as usize];
            sum += rel_or_zero(&*self.source, info.naif.unwrap_or_default(), t);
            at = info.parent;
        }
        sum
    }

    /// This tick's comet work at the cached epoch (review r1 N1): every stale comet with a
    /// checkpoint within 30 days of the date takes it (no integration); then at most
    /// [`REOSC_STEPS_PER_TICK`] short steps, in this order (review r2 P1):
    /// 1. the selected comet, then the comets the last snapshot drew, brightest first, each
    ///    [`REOSC_STEPS_PER_COMET`] steps while the budget lasts, so what is on screen catches
    ///    up first;
    /// 2. with the steps left (s), the s other stale comets nearest the Sun (by cached
    ///    position), each taking s / n of them (n such comets), at least one and at most
    ///    [`REOSC_STEPS_PER_COMET`].
    ///
    /// Ties go to the lower index. The changed comets' cached positions then follow their new
    /// conics. Which comets step, and how far, depends only on the world's state and the calls
    /// made on it, never on the machine or the wall time (review r1 N2).
    fn reosculate_comets(&mut self) {
        let t = self.epoch;
        let sun = self.sun.map_or(UniversePos::default(), |s| self.cached[s as usize]);
        for k in 0..self.comets.len() {
            if self.comets[k].adopt(t) {
                self.place_comet(k, sun);
            }
        }
        // (class, key) of each comet, ascending: the selected one (0), the drawn ones by falling
        // brightness (1), the others by distance from the Sun (2).
        let mut rank: Vec<(u8, f64)> = (0..self.comets.len())
            .map(|k| {
                let id = self.comet_objects[k] as usize;
                (2, self.cached[id].delta_from(sun).length_squared())
            })
            .collect();
        let shown = self.shown_comets.get_mut().unwrap_or_else(PoisonError::into_inner);
        for &(k, lux) in shown.iter() {
            if let Some(r) = rank.get_mut(k) {
                *r = (1, if r.0 == 1 { r.1.min(-lux) } else { -lux });
            }
        }
        let selected = self.selected.and_then(|id| match self.physical.get(id as usize)?.source {
            Source::Comet(k) => Some(k),
            _ => None,
        });
        if let Some(r) = selected.and_then(|k| rank.get_mut(k)) {
            *r = (0, 0.0);
        }
        let mut stale: Vec<(u8, f64, usize)> = (0..self.comets.len())
            .filter(|&k| self.comets[k].needs_work(t))
            .map(|k| (rank[k].0, rank[k].1, k))
            .collect();
        if stale.is_empty() {
            return;
        }
        stale.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut budget = REOSC_STEPS_PER_TICK;
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        let mut rest: Vec<usize> = Vec::new();
        for &(class, _, k) in &stale {
            if class < 2 {
                let steps = REOSC_STEPS_PER_COMET.min(budget);
                if steps == 0 {
                    break;
                }
                jobs.push((k, steps));
                budget -= steps;
            } else {
                rest.push(k);
            }
        }
        rest.truncate(budget);
        if !rest.is_empty() {
            let steps = (budget / rest.len()).clamp(1, REOSC_STEPS_PER_COMET);
            jobs.extend(rest.iter().map(|&k| (k, steps)));
        }
        run_comet_jobs(&mut self.comets, &jobs, t, Chunk::Short);
        for (k, _) in jobs {
            self.place_comet(k, sun);
        }
    }

    /// Records the comets a snapshot drew (comet index, illuminance [lx]), replacing the last
    /// snapshot's: the next ticks re-osculate the stale ones first (review r2 P1).
    pub(crate) fn note_shown_comets(&self, shown: Vec<(usize, f64)>) {
        *self.shown_comets.lock().unwrap_or_else(PoisonError::into_inner) = shown;
    }

    /// Comet `k`'s cached position from its conic at the cached epoch (kept if not finite).
    fn place_comet(&mut self, k: usize, sun: UniversePos) {
        let helio = self.comets[k].state(self.epoch).0;
        if helio.is_finite() {
            self.cached[self.comet_objects[k] as usize] = add_rel(sun, helio);
        }
    }
}

/// Runs comet re-osculation `jobs` (comet index, most steps) toward `t` on up to
/// [`std::thread::available_parallelism`] threads: the calling thread and scoped helpers
/// (ruling 3). A shared queue hands out one step at a time, always of the comet with the most
/// steps left (job order on ties), and a comet that has steps left goes back in after each one.
/// So one comet's steps run one after another (never two at once), the longest chains start
/// first and continue on whichever thread is free, and single steps fill the gaps: list
/// scheduling by the longest remaining chain, so neither a late helper nor the single steps
/// hold back a drawn comet's three steps (review r2 P1). Each comet depends only on its own
/// conic, so the result does not depend on the thread count or on which thread ran a step. A
/// repeated index runs once. A helper's panic is re-raised.
fn run_comet_jobs(comets: &mut [CometOrbit], jobs: &[(usize, usize)], t: Epoch, chunk: Chunk) {
    let mut slots: Vec<Option<&mut CometOrbit>> = comets.iter_mut().map(Some).collect();
    // (job order, comet, steps left).
    let queue: Vec<(usize, &mut CometOrbit, usize)> = jobs
        .iter()
        .filter(|&&(_, steps)| steps > 0)
        .filter_map(|&(k, steps)| Some((slots.get_mut(k)?.take()?, steps)))
        .enumerate()
        .map(|(order, (comet, steps))| (order, comet, steps))
        .collect();
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(queue.len());
    let queue = Mutex::new(queue);
    // A thread that finds the queue empty stops; a comet with steps left is put back by the
    // thread that ran its step, which then keeps going, so no step is ever left without one.
    let work = || {
        loop {
            let job = {
                let mut queue = queue.lock().unwrap_or_else(PoisonError::into_inner);
                let next = (0..queue.len()).max_by(|&a, &b| {
                    (queue[a].2.cmp(&queue[b].2)).then(queue[b].0.cmp(&queue[a].0))
                });
                next.map(|at| queue.swap_remove(at))
            };
            let Some((order, comet, steps)) = job else { break };
            if comet.step(t, chunk) && steps > 1 {
                queue
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((order, comet, steps - 1));
            }
        }
    };
    if workers <= 1 {
        work();
        return;
    }
    std::thread::scope(|scope| {
        let helpers: Vec<_> = (1..workers).map(|_| scope.spawn(&work)).collect();
        work();
        for helper in helpers {
            if let Err(panic) = helper.join() {
                std::panic::resume_unwind(panic);
            }
        }
    });
}

/// An exoplanet host built from the archive's RA, Dec, distance, V and T_eff (review I11):
/// M_V = V − 5 log₁₀(d / 10 pc) (the distance modulus, no extinction), at rest, flagged as an
/// archive host without velocity. `None` without a finite V and distance.
fn archive_star(sys: &ExoSystem, system_index: u32) -> Option<Star3> {
    let v = sys.v_mag.filter(|v| v.is_finite())?;
    let d = sys.dist_pc;
    if !(d.is_finite() && d > 0.0) {
        return None;
    }
    let pos = radec_to_icrs(sys.ra_deg.to_radians(), sys.dec_deg.to_radians()) * d;
    let abs_mag = v - 5.0 * (d / 10.0).log10();
    // 0.65: the colour A7's bake gives a star with no B−V either.
    let bv = sys
        .teff_k
        .filter(|t| t.is_finite() && *t > 0.0)
        .map_or(0.65, bv_from_teff);
    Some(Star3::new(
        pos.as_vec3().to_array(),
        abs_mag as f32,
        [0.0; 3],
        bv as f32,
        system_index,
        FLAG_EXO_HOST | FLAG_FROM_ARCHIVE | FLAG_NO_VELOCITY,
    ))
}

/// (diameter [km], p_V) of an asteroid: the catalogue's where it has them, the other from H by
/// D = 1329 km / √p_V · 10^(−H/5) where it has one, and p_V = 0.14 with D from H where it has
/// neither.
fn asteroid_size(r: &AsteroidRecord) -> (f64, f64) {
    let h = f64::from(r.h);
    let d = Some(f64::from(r.diameter_km)).filter(|d| d.is_finite() && *d > 0.0);
    let p = Some(f64::from(r.albedo)).filter(|p| p.is_finite() && *p > 0.0);
    let scale = H_DIAMETER_KM * 10_f64.powf(-h / 5.0);
    match (d, p) {
        (Some(d), Some(p)) => (d, p),
        (Some(d), None) => (d, (scale / d).powi(2)),
        (None, Some(p)) => (scale / p.sqrt(), p),
        (None, None) => (scale / ASSUMED_ASTEROID_ALBEDO.sqrt(), ASSUMED_ASTEROID_ALBEDO),
    }
}

/// Volumetric mean radius (a·b·c)^⅓ [m].
fn mean_radius(radii: DVec3) -> f64 {
    (radii.x * radii.y * radii.z).cbrt()
}

impl World {
    /// Advances time by `dt_real_s` of wall time and refreshes the cached states. Also picks
    /// up an epoch set directly on [`World::time`], registers the unnamed stars a camera came
    /// within 0.1 pc of, and moves the comets' conics toward the date by a fixed amount of work,
    /// paused or not (module doc). It never runs a whole settle: after a date jump the comets
    /// catch up over the following ticks, the selected comet and the ones the last snapshot
    /// drew first, then the others nearest the Sun.
    pub fn tick(&mut self, dt_real_s: f64) {
        self.time.advance(dt_real_s);
        self.register_near_stars();
        let t = self.time.epoch();
        if t != self.epoch {
            self.epoch = t;
            self.refresh_states();
        }
        self.reosculate_comets();
    }

    /// The epoch of the cached states: `time.epoch()` as of the last [`World::tick`].
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    pub fn frames(&self) -> &FrameTree {
        &self.frames
    }

    /// Every registry object, indexed by its id. The registry only grows: a tick can append
    /// the unnamed catalogue stars a camera came within 0.1 pc of (module doc), so the length
    /// can change after any [`World::tick`]; ids already handed out never change. Read the
    /// length again after a tick instead of sizing per-id arrays once (review r2 P2).
    pub fn objects(&self) -> &[ObjectInfo] {
        &self.objects
    }

    /// Case-insensitive lookup by name. Also accepts an object's other names: an asteroid's
    /// SBDB full name ("1 Ceres (A801 AA)"), a star's Bayer or Flamsteed designation, an
    /// exoplanet host's archive name.
    pub fn find(&self, name: &str) -> Option<ObjectId> {
        self.names.get(name)
    }

    /// Root position at [`World::epoch`]; the root origin for an unknown id.
    pub fn position(&self, id: ObjectId) -> UniversePos {
        if let Some(&pos) = self.cached.get(id as usize) {
            return pos;
        }
        match self.objects.get(id as usize) {
            Some(info) => info
                .catalog_index
                .map_or(UniversePos::default(), |i| self.star_position(i)),
            None => UniversePos::default(),
        }
    }

    /// Velocity [m/s], root axes: ephemeris bodies by central differences (±10 s), comets and
    /// asteroids from their conics plus the Sun's, stars from the catalogue (0 for "no
    /// velocity"), TestBH-1 at rest. Zero for an unknown id.
    pub fn velocity(&self, id: ObjectId) -> DVec3 {
        let Some(physical) = self.physical.get(id as usize) else {
            return DVec3::ZERO;
        };
        let t = self.epoch;
        let sun_velocity = || self.sun.map_or(DVec3::ZERO, |s| self.velocity(s));
        match physical.source {
            Source::Ephem => {
                let h = VELOCITY_STEP_S;
                (self.ssb_offset(id, t.add_seconds(h)) - self.ssb_offset(id, t.add_seconds(-h)))
                    / (2.0 * h)
            }
            Source::Comet(c) => sun_velocity() + self.comets[c].state(t).1,
            Source::Asteroid(a) => sun_velocity() + self.asteroids[a].state(t).1,
            Source::Star => self.objects[id as usize]
                .catalog_index
                .and_then(|i| self.stars.get(i as usize))
                .map_or(DVec3::ZERO, |s| {
                    crate::starindex::velocity_pc_per_yr(s) * (PC / YEAR_S)
                }),
            Source::BlackHole(_) => DVec3::ZERO,
        }
    }

    /// The deepest frame whose Hill sphere contains `pos`: a moon's inside its Hill sphere,
    /// else its planet's, else the Sun's out to 2.3 × 10⁵ AU, else ROOT. Where two spheres of
    /// the same depth overlap, the smaller one wins.
    pub fn dominant_frame(&self, pos: &UniversePos) -> FrameId {
        let mut best: Option<(u32, f64, FrameId)> = None;
        for (k, info) in self.objects.iter().enumerate().take(self.cached.len()) {
            let (Some(frame), hill) = (info.frame, info.hill_radius_m) else {
                continue;
            };
            if !(hill > 0.0) || !(pos.delta_from(self.cached[k]).length() < hill) {
                continue;
            }
            let depth = self.physical[k].depth;
            let better = match best {
                None => true,
                Some((d, h, _)) => depth > d || (depth == d && hill < h),
            };
            if better {
                best = Some((depth, hill, frame));
            }
        }
        best.map_or(ROOT_FRAME, |(_, _, frame)| frame)
    }

    /// The nearest body surface and the distance to it [m] (negative inside): every Solar
    /// System body (triaxial bodies along the direction to `pos`), TestBH-1's horizon, and the
    /// photospheres of catalogue stars within 0.1 pc. An unnamed star found here for the first
    /// time has no [`ObjectId`] yet; it is registered at the next [`World::tick`] and offered
    /// from then on (review I1).
    pub fn nearest_surface(&self, pos: &UniversePos) -> Option<(ObjectId, f64)> {
        self.nearest_surface_among(pos, &self.near_stars(pos))
    }

    /// [`World::nearest_surface`] with the near stars of `pos` already found (the snapshot
    /// searches once per frame).
    pub(crate) fn nearest_surface_among(
        &self,
        pos: &UniversePos,
        near: &[(u32, Option<ObjectId>)],
    ) -> Option<(ObjectId, f64)> {
        let mut best: Option<(ObjectId, f64)> = None;
        let mut offer = |id: ObjectId, altitude: f64| {
            if altitude.is_finite() && best.is_none_or(|(_, b)| altitude < b) {
                best = Some((id, altitude));
            }
        };
        for k in 0..self.cached.len() {
            let rel = pos.delta_from(self.cached[k]);
            offer(k as ObjectId, self.altitude(k as ObjectId, rel));
        }
        for &(index, id) in near {
            if let Some(id) = id {
                let d = pos.delta_from(self.star_position(index)).length();
                offer(id, d - self.objects[id as usize].radii_m.x);
            }
        }
        best
    }

    /// Height above the surface of a cached body at `rel` from its centre [m]: for a triaxial
    /// body, the distance minus the ellipsoid's radius along `rel` in body axes.
    fn altitude(&self, id: ObjectId, rel: DVec3) -> f64 {
        let info = &self.objects[id as usize];
        let r = info.radii_m;
        let d = rel.length();
        if r.x == r.y && r.y == r.z {
            return d - r.x;
        }
        if d == 0.0 {
            return -r.min_element();
        }
        let orient = match (self.physical[id as usize].source, info.naif) {
            (Source::Ephem, Some(naif)) => self.source.orient(naif, self.epoch),
            _ => None,
        }
        .unwrap_or(DQuat::IDENTITY);
        let u = orient.conjugate() * (rel / d);
        d - 1.0 / ((u / r).length())
    }

    /// What to draw from `view` at internal size `viewport` [px].
    pub fn snapshot(&self, view: &CameraView, viewport: UVec2) -> Snapshot {
        snapshot::build(self, view, viewport)
    }

    /// The world-state labels in effect: planets outside 3000 BC – AD 3000; Pluto outside
    /// 1885–2099 while Pluto or Charon is in the view or selected; stars beyond ±100 000 yr;
    /// the band while it is enabled. `ProceduralBelt` is the app's (C6).
    pub fn honesty(&self, view: &CameraView) -> Vec<HonestyFlag> {
        let t = self.epoch;
        let mut flags = Vec::new();
        if standish::validity(t) == Validity::Extrapolated {
            flags.push(HonestyFlag::PlanetsExtrapolated);
        }
        if !pluto::in_window(t) && self.pluto_shown(view) {
            flags.push(HonestyFlag::PlutoExtrapolated);
        }
        if self.star_years().abs() > STAR_LINEAR_LIMIT_YR {
            flags.push(HonestyFlag::StarsExtrapolated);
        }
        if self.band_enabled {
            flags.push(HonestyFlag::BandModel);
        }
        flags
    }

    /// Whether Pluto or Charon is selected or inside the view's frustum, taken as the cone
    /// through the corners of the widest frame ([`HONESTY_MAX_ASPECT`]) widened by the body's
    /// angular radius.
    fn pluto_shown(&self, view: &CameraView) -> bool {
        let tan_half = (0.5 * view.fov_y_rad).tan();
        let cone = (tan_half * (1.0 + HONESTY_MAX_ASPECT * HONESTY_MAX_ASPECT).sqrt()).atan();
        let forward = view.orient.normalize() * DVec3::NEG_Z;
        self.ephem_order.iter().any(|&id| {
            let info = &self.objects[id as usize];
            if !matches!(info.naif, Some(PLUTO | CHARON)) {
                return false;
            }
            if self.selected == Some(id) {
                return true;
            }
            let rel = self.cached[id as usize].delta_from(view.pos);
            let d = rel.length();
            let radius = info.radii_m.max_element();
            if d <= radius {
                return true;
            }
            let off_axis = forward.cross(rel).length().atan2(forward.dot(rel));
            off_axis <= cone + (radius / d).asin()
        })
    }

    pub fn lenses(&self) -> &[Lens] {
        &self.lenses
    }

    /// AT-HYG stars, then archive-only exoplanet hosts (review I11).
    pub fn star_data(&self) -> &[Star3] {
        &self.stars
    }

    pub fn exo_systems(&self) -> &[ExoSystem] {
        &self.exo
    }

    /// Whether the Milky Way band is drawn (default on): [`HonestyFlag::BandModel`] follows it.
    pub fn set_band_enabled(&mut self, on: bool) {
        self.band_enabled = on;
    }

    /// The object the user has selected: [`HonestyFlag::PlutoExtrapolated`] also applies while
    /// Pluto or Charon is selected off screen.
    pub fn set_selected(&mut self, id: Option<ObjectId>) {
        self.selected = id;
    }

    /// Saturn's ring radii from a ring profile. The default is already the span of A8's baked
    /// `saturn_rings.ron` (review I2), so the app need call this only for another profile. A
    /// profile with impossible radii is ignored.
    pub fn set_ring_profile(&mut self, profile: &RingProfile) {
        let (inner, outer) = (profile.inner_m, profile.outer_m);
        if inner.is_finite() && outer.is_finite() && 0.0 <= inner && inner < outer {
            self.ring_m = (inner, outer);
        }
    }

    /// Re-osculates every comet that needs it at [`World::epoch`] now, blocking: one perturbed
    /// integration per comet, from the nearest in time of its catalogue conic, the conics it
    /// reached before and its current conic, in segments of 120 days that are each kept as a
    /// checkpoint (so later jumps inside the span are cheap; review r2 P1), spread over the CPU
    /// cores with `std::thread::scope` (ruling 3). The longest integrations are handed out
    /// first, one comet at a time, so one slow comet does not hold back a whole share.
    /// [`World::new`] calls it; [`World::tick`] never does. A capture calls it after setting
    /// each bookmark's epoch, so its comets do not depend on the frames before it (review r1
    /// N2).
    ///
    /// The cost grows with the integration: the baked catalogue's 512 comets all have epochs at
    /// least a year from 2026-11-20, 6694 comet-years in all. On the M3's 8 cores that settle
    /// took 29 s in the debug build (230 s of CPU; 3D/Biela's 194 years alone 9.1 s) and 7.4 s
    /// with uni-physics and uni-core at opt-level 3; a 30-year jump took 42 s and 10.9 s. The
    /// 120-day segments (fix round 3) left it at 29–31 s in debug and 7.5 s with every
    /// workspace crate at opt-level 3. The release build is the gate's to measure.
    pub fn settle_comets(&mut self) {
        let t = self.epoch;
        let mut work: Vec<(f64, usize)> = self
            .comets
            .iter()
            .enumerate()
            .filter(|(_, c)| c.needs_work(t))
            .map(|(k, c)| (c.work_span_s(t), k))
            .collect();
        work.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let jobs: Vec<(usize, usize)> = work.iter().map(|&(_, k)| (k, usize::MAX)).collect();
        run_comet_jobs(&mut self.comets, &jobs, t, Chunk::Whole);
        let sun = self.sun.map_or(UniversePos::default(), |s| self.cached[s as usize]);
        for (k, _) in jobs {
            self.place_comet(k, sun);
        }
    }

    /// The source's orientation of an ephemeris body at the cached epoch (identity without
    /// a model, and for every other object).
    pub(crate) fn orientation(&self, id: ObjectId) -> DQuat {
        match (self.physical[id as usize].source, self.objects[id as usize].naif) {
            (Source::Ephem, Some(naif)) => self.source.orient(naif, self.epoch),
            _ => None,
        }
        .unwrap_or(DQuat::IDENTITY)
    }

    /// SBDB total-magnitude parameters (M1, K1) of a comet object.
    pub(crate) fn comet_magnitude(&self, id: ObjectId) -> Option<(f64, f64)> {
        match self.physical[id as usize].source {
            Source::Comet(c) => self.comets[c].magnitude,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use glam::{DQuat, DVec3, UVec2};
    use uni_core::{AU, M_V_SUN, PC, ROOT_FRAME, star_illuminance_lux};
    use uni_data::StarName;
    use uni_physics::bodies::by_naif;
    use uni_physics::kepler::Conic;
    use uni_physics::photometry::{
        Brdf, disk_illuminance_lux, planet_phase_correction, rho_from_geometric_albedo,
    };

    use super::*;
    use crate::honesty::label;
    use crate::registry::ObjectKind;
    use crate::snapshot::{BodyView, POINT_LOD_PX, proj_radius_px};

    const VIEWPORT: UVec2 = UVec2::new(1920, 1080);
    const YEAR_S: f64 = 365.25 * 86_400.0;

    /// Bodies held at fixed offsets from their parents; no rotation models.
    struct FakeSource(Vec<(i32, Option<i32>, DVec3)>);

    impl BodySource for FakeSource {
        fn bodies(&self) -> Vec<(i32, Option<i32>)> {
            self.0.iter().map(|&(n, p, _)| (n, p)).collect()
        }
        fn position_rel_parent(&self, naif: i32, _t: Epoch) -> Option<DVec3> {
            self.0.iter().find(|b| b.0 == naif).map(|b| b.2)
        }
        fn orient(&self, _naif: i32, _t: Epoch) -> Option<DQuat> {
            None
        }
    }

    fn empty_data() -> WorldData {
        WorldData {
            stars: Vec::new(),
            star_names: None,
            sats: Vec::new(),
            comets: Vec::new(),
            asteroids: Vec::new(),
            asteroid_names: Vec::new(),
            exo: Vec::new(),
            facts: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn fake_world(bodies: Vec<(i32, Option<i32>, DVec3)>, data: WorldData) -> World {
        World::with_source(Arc::new(FakeSource(bodies)), data, Epoch::J2000)
    }

    fn baked(file: &str) -> PathBuf {
        uni_data::data_dir().join("baked").join(file)
    }

    fn real_sats() -> Vec<SatRecord> {
        uni_data::load_sat_elements(&baked("sat_elements.ron")).expect("sat_elements.ron")
    }

    fn date(iso: &str) -> Epoch {
        Epoch::parse_iso_utc(iso).expect("ISO date")
    }

    /// A camera at `pos` whose −Z axis points at `target`.
    fn looking_at(pos: UniversePos, target: UniversePos, fov_deg: f64) -> CameraView {
        let dir = target.delta_from(pos).normalize();
        CameraView {
            pos,
            orient: DQuat::from_rotation_arc(DVec3::NEG_Z, dir),
            fov_y_rad: fov_deg.to_radians(),
            velocity_mps: DVec3::ZERO,
        }
    }

    /// A camera at `pos` looking along `dir`.
    fn looking_along(pos: UniversePos, dir: DVec3) -> CameraView {
        looking_at(pos, pos.offset(dir), 60.0)
    }

    fn id_of(world: &World, name: &str) -> ObjectId {
        world.find(name).unwrap_or_else(|| panic!("{name} not in the registry"))
    }

    fn sat_record(naif: i32, planet: i32, radius_km: f64) -> SatRecord {
        SatRecord {
            naif,
            name: format!("Test moon {naif}"),
            planet_naif: planet,
            a_km: 1.0e6,
            e: 0.0,
            argp_deg: 0.0,
            m_deg: 0.0,
            i_deg: 0.0,
            node_deg: 0.0,
            n_deg_per_day: 1.0,
            p_apsides_yr: f64::INFINITY,
            p_node_yr: f64::INFINITY,
            epoch_jd_tdb: 2_451_545.0,
            laplace_ra_deg: 0.0,
            laplace_dec_deg: 90.0,
            radius_km: Some(radius_km),
            gm_km3_s2: None,
        }
    }

    fn star_name(index: u32, proper: &str) -> StarName {
        StarName {
            index,
            proper: Some(proper.to_owned()),
            bayer_flamsteed: None,
            con: None,
            hip: None,
            spect: None,
        }
    }

    /// Review Focus 1, review C1: 10 000 frames at ×10¹² (1.67e14 s = 5.28 Myr) with the real
    /// Ephemeris and every JPL mean-element moon.
    #[test]
    fn max_warp_ten_thousand_frames_finite() {
        let mut data = empty_data();
        data.sats = real_sats();
        let start = date("2026-11-20T00:00:00Z");
        let mut world = World::new(data, start);
        world.time.set_rate(1.0e12);
        let sun = id_of(&world, "Sun");
        let planets: Vec<(ObjectId, i32)> = world
            .objects()
            .iter()
            .filter(|o| matches!(o.kind, ObjectKind::Planet | ObjectKind::DwarfPlanet))
            .filter_map(|o| Some((o.id, o.naif?)))
            .collect();
        assert_eq!(planets.len(), 9, "{planets:?}");
        let n = world.objects().len() as ObjectId;
        let clock = std::time::Instant::now();
        for frame in 0..10_000 {
            world.tick(1.0 / 60.0);
            for id in 0..n {
                let p = world.position(id);
                assert!(p.local.is_finite(), "frame {frame}: object {id} at {p:?}");
            }
            if frame % 100 == 99 {
                let t = world.epoch();
                for &(id, naif) in &planets {
                    let r = world.position(id).delta_from(world.position(sun));
                    let v = world.velocity(id) - world.velocity(sun);
                    let mu = by_naif(10).unwrap().gm_m3s2 + by_naif(naif).unwrap().gm_m3s2;
                    let e = Conic::from_state(r, v, mu, t).e;
                    assert!(e < 1.0, "frame {frame}: {naif} has e = {e}");
                }
            }
        }
        println!(
            "10 000 ticks of {} objects: {:.2} s (debug build)",
            n,
            clock.elapsed().as_secs_f64()
        );
        let elapsed = world.epoch().seconds_since(start);
        assert!((elapsed / (1.0e12 * 10_000.0 / 60.0) - 1.0).abs() < 1e-9, "{elapsed}");
        let view = looking_along(world.position(sun).offset(DVec3::X * AU), DVec3::X);
        assert!(world.honesty(&view).contains(&HonestyFlag::PlanetsExtrapolated));
    }

    #[test]
    fn world_data_missing_files_warns() {
        let dir = std::env::temp_dir().join(format!("uni-sim-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let data = load_world_data(&dir);
        assert!(data.stars.is_empty() && data.star_names.is_none());
        assert!(data.sats.is_empty() && data.comets.is_empty() && data.exo.is_empty());
        assert!(data.asteroids.is_empty() && data.asteroid_names.is_empty());
        assert!(data.facts.is_empty());
        let files = [
            "athyg_m10.bin",
            "athyg_names.ron",
            "sat_elements.ron",
            "comets.ron",
            "asteroids.bin",
            "asteroid_names.txt",
            "exoplanets.ron",
            "body_facts.ron",
        ];
        assert_eq!(data.warnings.len(), files.len(), "{:#?}", data.warnings);
        for file in files {
            assert_eq!(
                data.warnings.iter().filter(|w| w.contains(file)).count(),
                1,
                "{file}: {:#?}",
                data.warnings
            );
        }
        // The world still builds: the ephemeris bodies need no data file.
        let world = World::new(data, Epoch::J2000);
        assert!(world.find("Earth").is_some() && world.star_data().is_empty());
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn world_data_loads_the_baked_files() {
        let data = load_world_data(&uni_data::data_dir().join("baked"));
        assert!(data.warnings.is_empty(), "{:#?}", data.warnings);
        assert!(data.stars.len() > 300_000 && data.star_names.is_some());
        assert!(!data.sats.is_empty() && !data.comets.is_empty() && !data.exo.is_empty());
        assert!(!data.asteroids.is_empty() && !data.facts.is_empty());
    }

    #[test]
    fn find_is_case_insensitive() {
        let mut data = empty_data();
        data.stars = vec![Star3::new([2.0, 0.0, 0.0], 1.0, [0.0; 3], 0.0, 1, 0)];
        data.star_names = Some(StarNames {
            entries: vec![star_name(0, "Testar")],
        });
        let world = fake_world(
            vec![
                (10, None, DVec3::ZERO),
                (499, Some(10), DVec3::new(1.5 * AU, 0.0, 0.0)),
            ],
            data,
        );
        let mars = world.find("mars");
        assert!(mars.is_some());
        assert_eq!(world.find("MARS"), mars);
        assert_eq!(world.find("Mars"), mars);
        assert_eq!(world.find("tESTAR"), world.find("Testar"));
        assert!(world.find("Testar").is_some());
        assert_eq!(world.find("Vulcan"), None);
        let mut lower: Vec<String> = world
            .objects()
            .iter()
            .map(|o| o.name.to_lowercase())
            .collect();
        lower.sort();
        let count = lower.len();
        lower.dedup();
        assert_eq!(lower.len(), count, "names are unique case-insensitively");
        for (k, o) in world.objects().iter().enumerate() {
            assert_eq!(o.id as usize, k, "ids index objects()");
        }
    }

    /// Sun → Jupiter (5.2 AU) → Callisto, placed 10⁷ km out so that its Hill sphere (2.66 ×
    /// 10⁵ km about Jupiter) holds a point 10⁵ km away.
    #[test]
    fn dominant_frame_nesting() {
        let world = fake_world(
            vec![
                (10, None, DVec3::new(1.0e9, 0.0, 0.0)),
                (599, Some(10), DVec3::new(5.2 * AU, 0.0, 0.0)),
                (504, Some(599), DVec3::new(1.0e10, 0.0, 0.0)),
            ],
            empty_data(),
        );
        let frame = |name: &str| world.objects()[id_of(&world, name) as usize].frame;
        let at = |name: &str, d: DVec3| world.position(id_of(&world, name)).offset(d);
        let callisto = world.objects()[id_of(&world, "Callisto") as usize].hill_radius_m;
        assert!((callisto / 2.66e8 - 1.0).abs() < 0.01, "{callisto}");
        assert_eq!(
            Some(world.dominant_frame(&at("Callisto", DVec3::new(0.0, 1.0e8, 0.0)))),
            frame("Callisto")
        );
        assert_eq!(
            Some(world.dominant_frame(&at("Jupiter", DVec3::new(-1.0e10, 0.0, 0.0)))),
            frame("Jupiter")
        );
        assert_eq!(
            Some(world.dominant_frame(&at("Sun", DVec3::new(0.0, 0.0, 5.0 * AU)))),
            frame("Sun")
        );
        assert_eq!(
            world.dominant_frame(&at("Sun", DVec3::new(0.0, 0.0, 1.0e6 * AU))),
            ROOT_FRAME
        );
    }

    /// Review Focus 4: a 1000 km moon seen from 10⁵ m (inside it) to 10¹² m is in exactly one
    /// of `bodies` and `points`, and in `bodies` exactly when it projects to ≥ POINT_LOD_PX.
    #[test]
    fn body_is_point_or_resolved_never_both() {
        let mut data = empty_data();
        data.sats = vec![sat_record(790, 799, 1000.0)];
        let world = fake_world(
            vec![
                (10, None, DVec3::ZERO),
                (799, Some(10), DVec3::new(19.0 * AU, 0.0, 0.0)),
                (790, Some(799), DVec3::new(0.0, 1.0e9, 0.0)),
            ],
            data,
        );
        let moon = id_of(&world, "Test moon 790");
        assert_eq!(world.objects()[moon as usize].radii_m, DVec3::splat(1.0e6));
        let centre = world.position(moon);
        let (mut resolved, mut point) = (0, 0);
        for k in 0..=70 {
            let d = 10f64.powf(5.0 + 7.0 * k as f64 / 70.0);
            let eye = centre.offset(DVec3::new(0.3, -0.5, 0.8).normalize() * d);
            let view = looking_at(eye, centre, 60.0);
            let snap = world.snapshot(&view, VIEWPORT);
            let in_bodies = snap.bodies.iter().filter(|b| b.id == moon).count();
            let in_points = snap.points.iter().filter(|p| p.id == Some(moon)).count();
            assert_eq!(in_bodies + in_points, 1, "d = {d:e} m");
            let px = proj_radius_px(1.0e6, d, view.fov_y_rad, VIEWPORT.y as f64);
            assert_eq!(in_bodies == 1, px >= POINT_LOD_PX, "d = {d:e} m, {px} px");
            resolved += in_bodies;
            point += in_points;
        }
        assert!(resolved > 10 && point > 10, "{resolved} resolved, {point} points");
    }

    /// One named star 1 pc out, with Sirius's M_V; the camera 1 AU from it.
    #[test]
    fn near_star_goes_to_star_surfaces_and_is_suppressed() {
        let mut data = empty_data();
        data.stars = vec![
            Star3::new([1.0, 0.0, 0.0], 1.42, [0.0; 3], 0.0, 1, uni_data::FLAG_NAMED),
            Star3::new([-3.0, 1.0, 0.0], 4.0, [0.0; 3], 0.6, 2, 0),
        ];
        data.star_names = Some(StarNames {
            entries: vec![star_name(0, "Testar")],
        });
        let world = fake_world(vec![(10, None, DVec3::ZERO)], data);
        let star = id_of(&world, "Testar");
        assert_eq!(world.objects()[star as usize].kind, ObjectKind::Star);
        let centre = world.position(star);
        let view = looking_at(centre.offset(DVec3::new(0.0, AU, 0.0)), centre, 60.0);
        let snap = world.snapshot(&view, VIEWPORT);
        let surface = snap
            .star_surfaces
            .iter()
            .find(|s| s.id == star)
            .expect("the star is resolved");
        assert_eq!(surface.catalog_index, Some(0));
        assert!(surface.proj_radius_px >= POINT_LOD_PX, "{}", surface.proj_radius_px);
        assert!((surface.rel_pos_m - DVec3::new(0.0, -AU, 0.0)).length() < 1.0e3);
        assert_eq!(snap.suppressed_stars, vec![0]);
        assert!(snap.points.iter().all(|p| p.id != Some(star)), "never both");
        // Back at the Sun the star is 1 pc away: the star pass draws it again.
        let home = looking_at(UniversePos::default().offset(DVec3::Y * AU), centre, 60.0);
        assert!(world.snapshot(&home, VIEWPORT).suppressed_stars.is_empty());
    }

    /// Review Focus 4: the Sun seen from α Cen A's catalogue position is a point of
    /// star_illuminance_lux(M_V☉ + 5 log₁₀(d/10 pc)).
    #[test]
    fn sun_point_from_alpha_cen() {
        let mut data = empty_data();
        data.stars = uni_data::load_athyg(&baked("athyg_m10.bin")).expect("athyg_m10.bin");
        data.star_names =
            Some(uni_data::load_star_names(&baked("athyg_names.ron")).expect("athyg_names.ron"));
        let world = World::new(data, Epoch::J2000);
        let alpha_cen = id_of(&world, "Rigil Kentaurus");
        let sun = id_of(&world, "Sun");
        let view = looking_at(world.position(alpha_cen), world.position(sun), 20.0);
        let snap = world.snapshot(&view, VIEWPORT);
        let point = snap
            .points
            .iter()
            .find(|p| p.id == Some(sun))
            .expect("the Sun is a point from α Cen");
        assert!(snap.star_surfaces.iter().all(|s| s.id != sun));
        let d_pc = point.rel_pos_m.length() / PC;
        assert!((1.2..1.4).contains(&d_pc), "{d_pc} pc");
        let expected = star_illuminance_lux(M_V_SUN + 5.0 * (d_pc / 10.0).log10());
        assert!(
            (point.illuminance_lux / expected - 1.0).abs() < 0.005,
            "{} lx vs {expected} lx",
            point.illuminance_lux
        );
        // α Cen A itself (the camera sits at its centre) is drawn as a surface, not by the star
        // pass.
        let a_index = world.objects()[alpha_cen as usize].catalog_index.unwrap();
        assert!(snap.suppressed_stars.contains(&a_index));
        assert!(snap.suppressed_stars.len() <= 8);
    }

    /// Review I11: an archive host missing from AT-HYG (V 18.8 at 12.47 pc).
    #[test]
    fn archive_hosts_appended_and_flagged() {
        let mut data = empty_data();
        data.stars = vec![Star3::new([5.0, 0.0, 0.0], 3.0, [0.0; 3], 0.6, 1, 0)];
        data.exo = vec![ExoSystem {
            host: "Faint-1".to_owned(),
            star_index: None,
            ra_deg: 346.6,
            dec_deg: -5.04,
            dist_pc: 12.47,
            v_mag: Some(18.8),
            teff_k: Some(2566.0),
            radius_rsun: Some(0.12),
            planets: Vec::new(),
        }];
        let world = fake_world(vec![(10, None, DVec3::ZERO)], data);
        assert_eq!(world.star_data().len(), 2);
        let host = world.star_data()[1];
        let flags =
            uni_data::FLAG_EXO_HOST | uni_data::FLAG_FROM_ARCHIVE | uni_data::FLAG_NO_VELOCITY;
        assert_eq!(host.flags() & flags, flags);
        assert!((host.abs_mag_v - 18.32).abs() < 0.01, "{}", host.abs_mag_v);
        let p = DVec3::from_array(host.pos_pc.map(f64::from));
        assert!((p.length() - 12.47).abs() < 1e-4, "{p}");
        let id = id_of(&world, "Faint-1");
        let info = &world.objects()[id as usize];
        assert_eq!(info.kind, ObjectKind::ExoHost);
        assert_eq!(info.catalog_index, Some(1));
        let centre = world.position(id);
        let view = looking_at(centre.offset(DVec3::new(0.2 * AU, 0.0, 0.0)), centre, 60.0);
        let snap = world.snapshot(&view, VIEWPORT);
        let drawn = snap.star_surfaces.iter().filter(|s| s.id == id).count()
            + snap.points.iter().filter(|p| p.id == Some(id)).count();
        assert_eq!(drawn, 1);
        assert!(snap.suppressed_stars.contains(&1));
    }

    #[test]
    fn snapshot_rel_positions_are_f64_camera_relative() {
        let world = World::new(empty_data(), date("2026-11-20T00:00:00Z"));
        let neptune = id_of(&world, "Neptune");
        let cam = world.position(neptune).offset(DVec3::new(2.0e8, 3.0e7, -1.0e7));
        let view = looking_at(cam, world.position(neptune), 60.0);
        let snap = world.snapshot(&view, VIEWPORT);
        let body = snap.bodies.iter().find(|b| b.id == neptune).expect("Neptune resolved");
        let want = world.position(neptune).delta_from(cam);
        assert!((body.rel_pos_m - want).length() < 1e-3, "{} vs {want}", body.rel_pos_m);
        let sun = snap.sun.as_ref().expect("sun light");
        let sun_want = world.position(id_of(&world, "Sun")).delta_from(cam);
        assert!((sun.rel_pos_m - sun_want).length() < 1e-3);
        assert!((body.sun_rel_m - (sun_want - want)).length() < 1e-3);
    }

    #[test]
    fn honesty_flags() {
        let mut world = fake_world(
            vec![
                (10, None, DVec3::ZERO),
                (999, Some(10), DVec3::new(40.0 * AU, 0.0, 0.0)),
                (901, Some(999), DVec3::new(0.0, 1.96e7, 0.0)),
            ],
            empty_data(),
        );
        let pluto = id_of(&world, "Pluto");
        let set = |w: &mut World, e: Epoch| {
            w.time.set_epoch(e);
            w.tick(0.0);
        };
        let away = looking_along(UniversePos::default().offset(DVec3::Z * AU), DVec3::Z);
        assert_eq!(world.honesty(&away), vec![HonestyFlag::BandModel], "J2000: only the band");

        set(&mut world, Epoch::J2000.add_seconds(5000.0 * YEAR_S));
        let flags = world.honesty(&away);
        assert!(flags.contains(&HonestyFlag::PlanetsExtrapolated), "{flags:?}");
        assert!(!flags.contains(&HonestyFlag::StarsExtrapolated), "{flags:?}");

        set(&mut world, Epoch::J2000.add_seconds(2.0e5 * YEAR_S));
        let flags = world.honesty(&away);
        assert!(flags.contains(&HonestyFlag::StarsExtrapolated), "{flags:?}");
        assert!(flags.contains(&HonestyFlag::PlanetsExtrapolated), "{flags:?}");

        set(&mut world, date("2200-01-01T00:00:00Z"));
        let above = UniversePos::default().offset(DVec3::Z * AU);
        let at_pluto = looking_at(above, world.position(pluto), 60.0);
        let flags = world.honesty(&at_pluto);
        assert!(flags.contains(&HonestyFlag::PlutoExtrapolated), "{flags:?}");
        assert!(!flags.contains(&HonestyFlag::PlanetsExtrapolated), "{flags:?}");
        assert!(!world.honesty(&away).contains(&HonestyFlag::PlutoExtrapolated));
        world.set_selected(Some(pluto));
        assert!(world.honesty(&away).contains(&HonestyFlag::PlutoExtrapolated));
        world.set_selected(None);
        set(&mut world, date("2026-11-20T00:00:00Z"));
        assert!(!world.honesty(&at_pluto).contains(&HonestyFlag::PlutoExtrapolated));
        world.set_band_enabled(false);
        assert!(world.honesty(&at_pluto).is_empty());
        for flag in [
            HonestyFlag::PlanetsExtrapolated,
            HonestyFlag::PlutoExtrapolated,
            HonestyFlag::StarsExtrapolated,
            HonestyFlag::BandModel,
        ] {
            assert!(label(flag).contains("approximate") || label(flag).contains("model v0"));
        }
    }

    fn real_facts() -> Vec<BodyFacts> {
        uni_data::load_body_facts(&baked("body_facts.ron")).expect("body_facts.ron")
    }

    /// The app may hand the world to other threads (std::thread::scope, ruling 3).
    #[test]
    fn world_is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<World>();
    }

    /// Every body's cached position is its frame's origin, bit for bit, also after a tick.
    #[test]
    fn frames_match_cached_positions() {
        let mut data = empty_data();
        data.sats = real_sats();
        let mut world = World::new(data, date("2026-11-20T00:00:00Z"));
        for _ in 0..2 {
            let t = world.epoch();
            let mut framed = 0;
            for info in world.objects() {
                if let Some(frame) = info.frame {
                    let origin = world.frames().origin_in_root(frame, t);
                    assert_eq!(origin, world.position(info.id), "{}", info.name);
                    framed += 1;
                }
            }
            // The Sun, 8 planets, Pluto, every moon, TestBH-1.
            assert!(framed > 400, "{framed}");
            world.time.set_rate(86_400.0);
            world.tick(1.0);
        }
        let earth = world.objects()[id_of(&world, "Earth") as usize].frame.unwrap();
        assert_eq!(world.frames().find("Earth"), Some(earth));
        let bh = world.objects()[id_of(&world, TEST_BH_NAME) as usize].frame.unwrap();
        assert_eq!(world.frames().find(TEST_BH_NAME), Some(bh));
        let lens = &world.lenses()[0];
        assert_eq!(lens.name, TEST_BH_NAME);
        assert_eq!(lens.pos.delta_from(UniversePos::default()), DVec3::new(1000.0 * AU, 0.0, 0.0));
    }

    /// Velocities and Hill radii from the real ephemeris.
    #[test]
    fn velocities_and_hill_radii_are_physical() {
        let mut data = empty_data();
        data.sats = real_sats();
        let world = World::new(data, date("2026-11-20T00:00:00Z"));
        let id = |name: &str| id_of(&world, name);
        let v_earth = (world.velocity(id("Earth")) - world.velocity(id("Sun"))).length();
        assert!((29.0e3..30.8e3).contains(&v_earth), "{v_earth}");
        let v_moon = (world.velocity(id("Moon")) - world.velocity(id("Earth"))).length();
        assert!((0.96e3..1.09e3).contains(&v_moon), "{v_moon}");
        // Hill radii a(1 − e)∛(m/3M): Earth 1.47e9 m (about the Earth alone), the Moon
        // ≈ 5.8e7 m.
        let hill = |name: &str| world.objects()[id(name) as usize].hill_radius_m;
        assert!((1.40e9..1.52e9).contains(&hill("Earth")), "{}", hill("Earth"));
        assert!((5.0e7..6.5e7).contains(&hill("Moon")), "{}", hill("Moon"));
        assert!((4.6e10..5.4e10).contains(&hill("Jupiter")), "{}", hill("Jupiter"));
        assert_eq!(hill("Sun"), 2.3e5 * AU);
        // The camera rides with the Moon near it, with the Earth farther out, and with TestBH-1
        // within 760 AU of it.
        let near = |name: &str, d: f64| world.position(id(name)).offset(DVec3::new(0.0, d, 0.0));
        let frame = |name: &str| world.objects()[id(name) as usize].frame.unwrap();
        assert_eq!(world.dominant_frame(&near("Moon", 1.0e7)), frame("Moon"));
        assert_eq!(world.dominant_frame(&near("Earth", 5.0e8)), frame("Earth"));
        assert_eq!(world.dominant_frame(&near(TEST_BH_NAME, 100.0 * AU)), frame(TEST_BH_NAME));
        let bh_sphere = hill(TEST_BH_NAME) / AU;
        assert!((755.0..765.0).contains(&bh_sphere), "{bh_sphere}");
    }

    /// The asteroid fast path is the same orbit as uni-physics' conic, to well under a metre.
    #[test]
    fn asteroid_states_match_smallbody_conics() {
        let (records, names) =
            uni_data::load_asteroids(&baked("asteroids.bin"), &baked("asteroid_names.txt"))
                .expect("asteroids");
        let picked: Vec<AsteroidRecord> = records.iter().take(40).copied().collect();
        let mut data = empty_data();
        data.asteroids = picked.clone();
        data.asteroid_names = names;
        let mut world = World::new(data, date("2026-11-20T00:00:00Z"));
        let ceres = id_of(&world, "Ceres");
        assert_eq!(world.find("1 Ceres (A801 AA)"), Some(ceres));
        assert_eq!(world.objects()[ceres as usize].kind, ObjectKind::DwarfPlanet);
        assert_eq!(world.objects()[ceres as usize].naif, Some(2_000_001));
        let pallas = id_of(&world, "Pallas");
        assert_eq!(world.objects()[pallas as usize].kind, ObjectKind::Asteroid);
        for when in ["2026-11-20T00:00:00Z", "2100-06-01T00:00:00Z"] {
            world.time.set_epoch(date(when));
            world.tick(0.0);
            let sun = world.position(id_of(&world, "Sun"));
            let objects: Vec<&ObjectInfo> = world
                .objects()
                .iter()
                .filter(|o| matches!(o.kind, ObjectKind::Asteroid | ObjectKind::DwarfPlanet))
                .filter(|o| o.naif != Some(999))
                .collect();
            assert!(objects.len() >= 30, "{}", objects.len());
            for (info, r) in objects.iter().zip(picked.iter().filter(|r| r.h <= 10.0)) {
                let conic = uni_physics::smallbody::conic_from_mean_anomaly(
                    r.a_au, r.e, r.i_deg, r.node_deg, r.argp_deg, r.m0_deg, r.epoch_jd_tdb,
                );
                let (pos, vel) = heliocentric_state_icrs(&conic, world.epoch());
                let ours = world.position(info.id).delta_from(sun);
                // Two roundings of the same ellipse (M from the epoch vs from periapsis): ≈ 1 cm
                // at 2100.
                assert!((ours - pos).length() < 0.1, "{}: {} m", info.name, (ours - pos).length());
                let v = world.velocity(info.id) - world.velocity(id_of(&world, "Sun"));
                assert!((v - vel).length() < 1e-6, "{}: {} m/s", info.name, (v - vel).length());
            }
        }
    }

    /// 1P/Halley and C/2020 F3 (NEOWISE) from the baked catalogue.
    fn halley_and_neowise() -> (Comet, Comet) {
        let comets = uni_data::load_comets(&baked("comets.ron")).expect("comets.ron");
        let pick = |name: &str| comets.iter().find(|c| c.name == name).cloned().expect(name);
        (pick("1P/Halley"), pick("C/2020 F3 (NEOWISE)"))
    }

    fn catalogue_conic(c: &Comet) -> Conic {
        conic_from_perihelion(c.q_au, c.e, c.i_deg, c.node_deg, c.argp_deg, c.tp_jd_tdb)
    }

    /// The index of comet `name` among the world's comet orbits.
    fn comet_index(world: &World, name: &str) -> usize {
        match world.physical[id_of(world, name) as usize].source {
            Source::Comet(k) => k,
            other => panic!("{name} is not a comet: {other:?}"),
        }
    }

    /// Heliocentric position [m] of object `name`.
    fn helio(world: &World, name: &str) -> DVec3 {
        world.position(id_of(world, name)).delta_from(world.position(id_of(world, "Sun")))
    }

    /// Lead ruling after A3 (review I3): no comet is shown two-body from its catalogue conic,
    /// not even before the first tick: `World::new` settles every comet at the start date.
    #[test]
    fn comets_are_settled_at_construction() {
        let (halley, neowise) = halley_and_neowise();
        let t = date("1986-03-13T00:00:00Z");
        let data = WorldData { comets: vec![halley.clone(), neowise], ..empty_data() };
        let world = World::new(data, t);
        assert!(world.comets.iter().all(|c| !c.needs_work(t)), "settled");
        // The perturbed state at t (A3: 4990 km from Horizons there, 5.06 % of r for two-body).
        let catalogue = catalogue_conic(&halley);
        let perturbed = osculate_at(&catalogue, epoch_from_jd(halley.epoch_jd_tdb), t);
        let want = heliocentric_state_icrs(&perturbed, t).0;
        let err = (helio(&world, "1P/Halley") - want).length();
        assert!(err < 1.0, "{err} m from one integration");
        let two_body = heliocentric_state_icrs(&catalogue, t).0;
        let shift = (want - two_body).length() / want.length();
        assert!(shift > 0.01, "perturbations move Halley by {shift} of r");
    }

    /// Ticks until Halley (comet `h`) needs no work; returns how many it took.
    fn tick_until_settled(world: &mut World, h: usize) -> usize {
        let mut ticks = 0;
        while world.comets[h].needs_work(world.epoch()) {
            world.tick(1.0 / 60.0);
            ticks += 1;
            assert!(ticks < 20_000, "no convergence");
        }
        ticks
    }

    /// Review r1 N1: a tick never settles. After a date jump each tick moves a comet's conic
    /// by at most `REOSC_STEPS_PER_COMET` short steps, at real time, paused or under a warp, so
    /// the comets catch up over the following ticks and end within metres of one integration. A
    /// return to a date reached before needs no integration, and a date near one needs a few
    /// steps (checkpoints). Review r1 N2: the same ticks give the same conics, bit for bit.
    #[test]
    fn comets_catch_up_after_date_jumps_without_stalling() {
        let (halley, neowise) = halley_and_neowise();
        let t0 = date("1986-03-13T00:00:00Z");
        let data = || WorldData { comets: vec![halley.clone(), neowise.clone()], ..empty_data() };
        let mut world = World::new(data(), t0);
        let (h, n) = (comet_index(&world, "1P/Halley"), comet_index(&world, "C/2020 F3 (NEOWISE)"));
        let (osc_1986, epoch_1986) = (world.comets[h].osc, world.comets[h].osc_epoch);
        assert!(epoch_1986.seconds_since(t0).abs() < 1e-3, "settled at construction");
        let most = REOSC_STEPS_PER_COMET as f64 * REOSC_CHUNK_S * (1.0 + 1e-12);
        // Every history below, replayed on a second world, gives bit-identical conics.
        let history = |world: &mut World| {
            // A 34-year jump at real time: one tick moves Halley by at most three steps.
            world.time.set_epoch(date("2020-07-15T03:00:00Z"));
            let before = world.comets[h].osc_epoch;
            world.tick(1.0 / 60.0);
            let moved = world.comets[h].osc_epoch.seconds_since(before);
            assert!(moved > 0.0 && moved <= most, "{} days", moved / DAY);
            assert!(world.comets[h].needs_work(world.epoch()), "a tick never settles");
            // NEOWISE is nine days from its catalogue epoch: its catalogue conic, at once.
            assert_eq!(world.comets[n].osc_epoch, world.comets[n].catalogue_epoch);
            let ticks = 1 + tick_until_settled(world, h);
            // A fast warp (×10¹⁰: 5.3 years per tick) leaves it behind by design.
            world.time.set_rate(1.0e10);
            let before = world.comets[h].osc_epoch;
            world.tick(1.0 / 60.0);
            let moved = world.comets[h].osc_epoch.seconds_since(before);
            assert!(moved > 0.0 && moved <= most, "{} days under the warp", moved / DAY);
            assert!(world.comets[h].needs_work(world.epoch()));
            // Back to the start date, paused: the checkpoint, no integration.
            world.time.set_rate(1.0);
            world.time.toggle_pause();
            world.time.set_epoch(t0);
            world.tick(1.0 / 60.0);
            assert!(!world.comets[h].needs_work(world.epoch()));
            assert_eq!((world.comets[h].osc_epoch, world.comets[h].osc), (epoch_1986, osc_1986));
            // To 1996, inside the span caught up above: one tick from the nearest checkpoint
            // (120 days apart, review r2 P1).
            world.time.set_epoch(date("1996-01-01T00:00:00Z"));
            world.tick(1.0 / 60.0);
            let near = 1 + tick_until_settled(world, h);
            assert_eq!(near, 1, "{near} ticks from a checkpoint");
            ticks
        };
        let ticks = history(&mut world);
        // The chunked catch-up to 2020 matches one integration from the 1986 conic.
        world.time.set_epoch(date("2020-07-15T03:00:00Z"));
        world.tick(1.0 / 60.0);
        tick_until_settled(&mut world, h);
        let (osc, at) = (world.comets[h].osc, world.comets[h].osc_epoch);
        let one = osculate_at(&osc_1986, epoch_1986, at);
        let err = (heliocentric_state_icrs(&osc, at).0 - heliocentric_state_icrs(&one, at).0)
            .length();
        assert!(err < 10.0, "{err} m after {ticks} ticks");
        println!("Halley 1986 → 2020 caught up in {ticks} ticks, {err:.2} m from one run");
        let mut again = World::new(data(), t0);
        history(&mut again);
        again.time.set_epoch(date("2020-07-15T03:00:00Z"));
        again.tick(1.0 / 60.0);
        tick_until_settled(&mut again, h);
        assert_eq!((again.comets[h].osc_epoch, again.comets[h].osc), (at, osc), "deterministic");
    }

    /// Review I3, r1 N1: a tick step integrates `REOSC_CHUNK_STEPS` of the integrator's steps
    /// at its start, so a step near perihelion stays short and cheap; far out it is the full
    /// 16 days. Chunked steps reproduce one integration to the same epoch.
    #[test]
    fn comet_steps_stay_short_near_the_sun() {
        let (halley, _) = halley_and_neowise();
        let catalogue = catalogue_conic(&halley);
        let mut comet = CometOrbit::new(&halley);
        let t = date("1986-03-13T00:00:00Z");
        let (mut steps, mut shortest, mut longest) = (0, f64::INFINITY, 0.0_f64);
        let (mut slowest, mut total) = (0.0_f64, 0.0);
        loop {
            let start = comet.osc_epoch;
            let r = heliocentric_state_icrs(&comet.osc, start).0;
            let limit = short_span_s(r, start);
            // At most eight of the integrator's RK4 steps by the Sun's dynamical time alone
            // (A3's step rule); a planet can only shorten them.
            let sun_rk4 = (SMALLBODY_STEP_ETA * dynamical_time_s(r.length())).min(2.0 * DAY);
            assert!(limit <= (8.0 * sun_rk4).max(REOSC_MIN_CHUNK_S) * (1.0 + 1e-12));
            let clock = std::time::Instant::now();
            if !comet.step(t, Chunk::Short) {
                break;
            }
            let took = clock.elapsed().as_secs_f64();
            (slowest, total) = (slowest.max(took), total + took);
            let span = comet.osc_epoch.seconds_since(start).abs();
            assert!(span <= limit * (1.0 + 1e-9), "{span} s > {limit} s");
            (shortest, longest) = (shortest.min(span), longest.max(span));
            steps += 1;
            assert!(steps < 20_000, "no convergence");
        }
        assert!(t.seconds_since(comet.osc_epoch).abs() <= REOSC_INTERVAL_S);
        assert_eq!(longest, REOSC_CHUNK_S, "16-day steps far from the Sun");
        assert!(shortest < 0.5 * DAY, "{} days near perihelion", shortest / DAY);
        // 0.05 AU from Jupiter's barycentre its pull sets the step, as in the integrator:
        // √(d³/GM_J) = 63 days there, against 4.3 years for the Sun at 5.2 AU.
        let jupiter = standish::heliocentric_state_icrs(Planet::Jupiter, t).0;
        let close = jupiter + DVec3::new(0.0, 0.05 * AU, 0.0);
        let gm = system_gm_m3s2(Planet::Jupiter.barycentre_id()).expect("Jupiter's GM");
        let want = SMALLBODY_STEP_ETA * ((0.05 * AU).powi(3) / gm).sqrt();
        assert!((smallbody_rk4_step_s(close, t) / want - 1.0).abs() < 1e-9);
        assert!(short_span_s(close, t) < 0.1 * short_span_s(jupiter * 1.1, t));
        let one = osculate_at(&catalogue, epoch_from_jd(halley.epoch_jd_tdb), comet.osc_epoch);
        let err = (heliocentric_state_icrs(&comet.osc, t).0 - heliocentric_state_icrs(&one, t).0)
            .length();
        // Each restart's conic round trip costs ≈ 2 cm; the steps' own rounding of the RK4
        // grid adds a little (A3's integrator error is < 1 km over these 18 years).
        assert!(err < 10.0, "{steps} chunks vs one integration: {err} m");
        println!(
            "Halley 1968 → 1986 in {steps} short steps ({:.2}–{:.0} days), {err:.2} m from one \
             integration, mean {:.3} ms, slowest {:.3} ms (unoptimised uni-sim test build)",
            shortest / DAY,
            longest / DAY,
            total / f64::from(steps) * 1e3,
            slowest * 1e3
        );
    }

    /// Review r2 P1: a settle keeps a conic every 120 days from the catalogue epoch to the date,
    /// so any date in that span is at most 60 days from one, and a jump there takes one tick
    /// and ends within metres of one integration from the catalogue conic.
    #[test]
    fn settle_keeps_checkpoints_across_its_span() {
        let (halley, _) = halley_and_neowise();
        let t0 = date("1986-03-13T00:00:00Z");
        let data = WorldData { comets: vec![halley.clone()], ..empty_data() };
        let mut world = World::new(data, t0);
        let h = comet_index(&world, "1P/Halley");
        let comet = world.comets[h].clone();
        let marks: Vec<Epoch> = comet.checkpoints.iter().map(|(e, _)| *e).collect();
        let span = t0.seconds_since(comet.catalogue_epoch);
        assert_eq!(marks.first(), Some(&comet.catalogue_epoch));
        assert_eq!(marks.last(), Some(&t0));
        assert_eq!(marks.len(), (span / CHECKPOINT_SPACING_S).ceil() as usize + 1);
        for pair in marks.windows(2) {
            let gap = pair[1].seconds_since(pair[0]);
            assert!(gap > 0.0 && gap <= CHECKPOINT_SPACING_S, "{} days", gap / DAY);
        }
        let mut at = comet.catalogue_epoch;
        while at.seconds_since(t0) < 0.0 {
            let gap = at.seconds_since(comet.start_for(at).0).abs();
            assert!(gap <= 0.5 * CHECKPOINT_SPACING_S, "{} days", gap / DAY);
            at = at.add_seconds(7.0 * DAY);
        }
        // 1975, inside the span: one tick, paused or not.
        world.time.toggle_pause();
        world.time.set_epoch(date("1975-07-01T00:00:00Z"));
        world.tick(1.0 / 60.0);
        assert!(!world.comets[h].needs_work(world.epoch()), "one tick");
        let (osc, at) = (world.comets[h].osc, world.comets[h].osc_epoch);
        let one = osculate_at(&catalogue_conic(&halley), comet.catalogue_epoch, at);
        let err =
            (heliocentric_state_icrs(&osc, at).0 - heliocentric_state_icrs(&one, at).0).length();
        assert!(err < 10.0, "{err} m from one integration");
    }

    /// Review r2 P1: after a date jump the comets on screen catch up first. The selected comet
    /// and the comets the last snapshot drew (brightest first) take `REOSC_STEPS_PER_COMET`
    /// steps each while the budget lasts; the other stale comets nearest the Sun share the rest.
    #[test]
    fn drawn_and_selected_comets_catch_up_first() {
        let t0 = date("2026-11-20T00:00:00Z");
        let later = date("2028-11-20T00:00:00Z");
        // 30 catalogue comets with their epoch moved to t0 (to within the JD's rounding), so
        // `World::new` settles nothing.
        let jd0 = JD_J2000 + t0.seconds_since(Epoch::J2000) / DAY;
        let comets: Vec<Comet> = uni_data::load_comets(&baked("comets.ron"))
            .expect("comets.ron")
            .into_iter()
            .take(30)
            .map(|c| Comet { epoch_jd_tdb: jd0, ..c })
            .collect();
        let n = comets.len();
        // Paused, so each tick lands exactly on the date set.
        let make = || {
            let mut world = World::new(WorldData { comets: comets.clone(), ..empty_data() }, t0);
            world.time.toggle_pause();
            world
        };
        let moved = |world: &World| -> Vec<bool> {
            world.comets.iter().map(|c| c.osc_epoch != c.catalogue_epoch).collect()
        };

        // Nothing drawn: the 24 stale comets nearest the Sun take one step each.
        let mut plain = make();
        assert!(plain.comets.iter().all(|c| !c.needs_work(t0) && c.osc_epoch == c.catalogue_epoch));
        plain.time.set_epoch(later);
        plain.tick(1.0 / 60.0);
        let sun = plain.position(id_of(&plain, "Sun"));
        let dist = |k: usize| plain.position(plain.comet_objects[k]).delta_from(sun).length();
        let far = (0..n).max_by(|&a, &b| dist(a).total_cmp(&dist(b))).expect("comets");
        assert_eq!(moved(&plain).iter().filter(|&&m| m).count(), REOSC_STEPS_PER_TICK);
        assert!(!moved(&plain)[far], "the farthest comet waits");

        // `far`'s orbit after exactly `REOSC_STEPS_PER_COMET` short steps.
        let mut three = make().comets[far].clone();
        for _ in 0..REOSC_STEPS_PER_COMET {
            assert!(three.step(later, Chunk::Short));
        }

        // Drawn by a snapshot before the jump (as a point from 0.01 AU): three steps first.
        let mut drawn = make();
        let id = drawn.comet_objects[far];
        let pos = drawn.position(id);
        let view = looking_at(pos.offset(DVec3::new(0.0, 0.0, 0.01 * AU)), pos, 1.0);
        let snap = drawn.snapshot(&view, VIEWPORT);
        assert!(snap.points.iter().any(|p| p.id == Some(id)), "drawn as a point");
        let noted = drawn.shown_comets.lock().expect("lock").clone();
        assert!(noted.iter().any(|&(k, lux)| k == far && lux > 0.0), "{noted:?}");
        drawn.time.set_epoch(later);
        drawn.tick(1.0 / 60.0);
        let conic_of = |world: &World| (world.comets[far].osc_epoch, world.comets[far].osc);
        assert_eq!(conic_of(&drawn), (three.osc_epoch, three.osc));
        // The other 21 steps go to the 21 nearest the Sun, one each.
        let others = moved(&drawn).iter().filter(|&&m| m).count() - 1;
        assert_eq!(others, REOSC_STEPS_PER_TICK - REOSC_STEPS_PER_COMET);

        // A resolved nucleus is noted as infinitely bright.
        let close = looking_at(pos.offset(DVec3::new(0.0, 0.0, 1.0e5)), pos, 60.0);
        let near = make();
        assert!(near.snapshot(&close, VIEWPORT).bodies.iter().any(|b| b.id == id));
        let noted = near.shown_comets.lock().expect("lock").clone();
        assert!(noted.contains(&(far, f64::INFINITY)), "{noted:?}");

        // Selected, never drawn: the same three steps.
        let mut chosen = make();
        chosen.set_selected(Some(id));
        chosen.time.set_epoch(later);
        chosen.tick(1.0 / 60.0);
        assert_eq!(conic_of(&chosen), (three.osc_epoch, three.osc));

        // Nine drawn comets: the eight brightest take the whole budget, three steps each.
        let mut busy = make();
        busy.note_shown_comets((0..9).map(|k| (k, 1.0 + k as f64)).collect());
        busy.time.set_epoch(later);
        busy.tick(1.0 / 60.0);
        let want: Vec<bool> = (0..n).map(|k| (1..9).contains(&k)).collect();
        assert_eq!(moved(&busy), want, "comet 0 is the dimmest");

        // The selected comet goes before every drawn one: selecting the dimmest leaves out the
        // next dimmest instead.
        let mut both = make();
        both.note_shown_comets((0..9).map(|k| (k, 1.0 + k as f64)).collect());
        both.set_selected(Some(both.comet_objects[0]));
        both.time.set_epoch(later);
        both.tick(1.0 / 60.0);
        let want: Vec<bool> = (0..n).map(|k| k == 0 || (2..9).contains(&k)).collect();
        assert_eq!(moved(&both), want, "the selected comet first");
    }

    /// A planet's point flux is A3's disk integral times its clamped phase correction, so the
    /// point → disk handover keeps the brightness the bodies pass will draw.
    #[test]
    fn planet_point_flux_matches_disk_integral() {
        let facts = real_facts();
        let mut data = empty_data();
        data.facts = facts.clone();
        let world = World::new(data, date("2026-11-20T00:00:00Z"));
        let earth = world.position(id_of(&world, "Earth"));
        for (name, brdf) in [
            ("Mars", Brdf::Lambert),
            ("Venus", Brdf::Lambert),
            ("Mercury", Brdf::LunarLambert { l: 0.9 }),
            ("Moon", Brdf::LunarLambert { l: 0.9 }),
        ] {
            let id = id_of(&world, name);
            let cam = if name == "Moon" {
                earth.offset(DVec3::new(3.0e9, 0.0, 0.0))
            } else {
                earth
            };
            let view = looking_at(cam, world.position(id), 60.0);
            let snap = world.snapshot(&view, VIEWPORT);
            let point = snap.points.iter().find(|p| p.id == Some(id)).expect(name);
            let rel = world.position(id).delta_from(cam);
            let sun_rel = world.position(id_of(&world, "Sun")).delta_from(world.position(id));
            let phase = sun_rel.angle_between(-rel);
            let info = &world.objects()[id as usize];
            let fact = facts.iter().find(|f| Some(f.naif) == info.naif).unwrap();
            let p_v = fact.geometric_albedo.unwrap();
            let radius = (info.radii_m.x * info.radii_m.y * info.radii_m.z).cbrt();
            let e_sun = sun_intensity_cd() / sun_rel.length_squared();
            let rho = rho_from_geometric_albedo(brdf, p_v);
            let disk = disk_illuminance_lux(brdf, rho, radius, e_sun, phase, rel.length());
            let c = planet_phase_correction(info.naif.unwrap(), brdf, phase).min(10.0);
            let ratio = point.illuminance_lux / (disk * c);
            let phase_deg = phase.to_degrees();
            assert!((ratio - 1.0).abs() < 1e-3, "{name}: {ratio} at phase {phase_deg} deg");
        }
    }

    /// A Moon between the Earth and the Sun shadows the Earth; behind the Earth it is
    /// shadowed by it.
    #[test]
    fn occluders_follow_eclipse_geometry() {
        let cases = [(-3.844e8, true, false), (3.844e8, false, true)];
        for (moon_x, earth_has_moon, moon_has_earth) in cases {
            let world = fake_world(
                vec![
                    (10, None, DVec3::ZERO),
                    (399, Some(10), DVec3::new(AU, 0.0, 0.0)),
                    (301, Some(399), DVec3::new(moon_x, 0.0, 0.0)),
                ],
                empty_data(),
            );
            let (earth, moon) = (id_of(&world, "Earth"), id_of(&world, "Moon"));
            let cam = world.position(earth).offset(DVec3::new(0.0, 1.0e9, 0.0));
            let view = looking_at(cam, world.position(earth), 60.0);
            let snap = world.snapshot(&view, VIEWPORT);
            let body = |id| snap.bodies.iter().find(|b| b.id == id).expect("resolved");
            let near = |b: &BodyView, other: ObjectId| {
                let want = world.position(other).delta_from(cam);
                b.occluders.iter().any(|o| (o.rel_pos_m - want).length() < 1.0)
            };
            assert_eq!(near(body(earth), moon), earth_has_moon, "Moon at {moon_x}");
            assert_eq!(near(body(moon), earth), moon_has_earth, "Moon at {moon_x}");
            assert!(body(earth).has_atmosphere && !body(moon).has_atmosphere);
            let lux = body(earth).sun_illuminance_lux;
            assert!((lux / 1.26e5 - 1.0).abs() < 0.01, "{lux}");
        }
    }

    #[test]
    fn saturn_carries_its_rings() {
        let mut world = fake_world(
            vec![(10, None, DVec3::ZERO), (699, Some(10), DVec3::new(9.5 * AU, 0.0, 0.0))],
            empty_data(),
        );
        let saturn = id_of(&world, "Saturn");
        let centre = world.position(saturn);
        let view = looking_at(centre.offset(DVec3::new(0.0, 0.0, 1.0e9)), centre, 60.0);
        let ring = world.snapshot(&view, VIEWPORT).bodies[0].ring.expect("rings");
        // Review I2: by default the radii of A8's baked profile, the span B2's ring texture
        // covers. A re-bake that moves them fails here.
        let profile =
            uni_data::load_ring_profile(&baked("saturn_rings.ron")).expect("saturn_rings.ron");
        assert_eq!((ring.inner_m, ring.outer_m), (profile.inner_m, profile.outer_m));
        assert_eq!(ring.normal, DVec3::Z, "the fake has no rotation model: body z is ICRS z");
        let other = RingProfile {
            inner_m: 7.4658e7,
            outer_m: 1.36775e8,
            samples: profile.samples.clone(),
        };
        world.set_ring_profile(&other);
        let ring = world.snapshot(&view, VIEWPORT).bodies[0].ring.unwrap();
        assert_eq!((ring.inner_m, ring.outer_m), (other.inner_m, other.outer_m));
        // Impossible radii are ignored.
        world.set_ring_profile(&RingProfile { inner_m: 2.0e8, ..other.clone() });
        let ring = world.snapshot(&view, VIEWPORT).bodies[0].ring.unwrap();
        assert_eq!((ring.inner_m, ring.outer_m), (other.inner_m, other.outer_m));
    }

    /// Plan A4 (review I4): an asteroid is an object when H ≤ 10 or when `body_facts.ron` has
    /// a sheet for it (matched by NAIF id); the rest are `uni-sky`'s points.
    #[test]
    fn asteroids_with_a_fact_sheet_are_objects() {
        let record = |spkid: u32, h: f32, name_index: u32| AsteroidRecord {
            a_au: 1.13,
            e: 0.2,
            i_deg: 6.0,
            node_deg: 2.0,
            argp_deg: 66.0,
            m0_deg: 100.0,
            epoch_jd_tdb: 2_460_600.5,
            h,
            g: 0.15,
            diameter_km: f32::NAN,
            albedo: f32::NAN,
            spkid,
            name_index,
        };
        let mut data = empty_data();
        data.asteroids = vec![
            record(20_000_004, 3.2, 0),
            record(20_101_955, 20.2, 1),
            record(20_025_143, 19.2, 2),
            record(20_000_010, f32::NAN, 3),
        ];
        data.asteroid_names = [
            "4 Vesta (A807 FA)",
            "101955 Bennu (1999 RQ36)",
            "25143 Itokawa (1998 SF36)",
            "10 Hygiea (A849 GA)",
        ]
        .map(String::from)
        .to_vec();
        let bennu_naif = uni_data::smallbody::asteroid_naif_id(20_101_955).expect("numbered");
        assert_eq!(bennu_naif, 2_101_955);
        let mut sheet = real_facts().into_iter().next().expect("a fact sheet");
        (sheet.naif, sheet.name) = (bennu_naif, "Bennu".to_owned());
        data.facts = vec![sheet];
        let world = fake_world(vec![(10, None, DVec3::ZERO)], data);
        assert!(world.find("Vesta").is_some(), "H ≤ 10");
        let bennu = world.find("Bennu").expect("its fact sheet makes Bennu an object");
        assert_eq!(world.objects()[bennu as usize].naif, Some(bennu_naif));
        assert_eq!(world.objects()[bennu as usize].kind, ObjectKind::Asteroid);
        assert_eq!(world.find("Itokawa"), None, "H 19.2 without a sheet is a uni-sky point");
        assert_eq!(world.find("Hygiea"), None, "no H: not H ≤ 10");
    }

    /// Plan A4 (review I1): an unnamed catalogue star within 0.1 pc leaves the star pass and is
    /// drawn by the snapshot from f64 positions, once. The first snapshot that finds it near
    /// draws it as a point without an id; the next tick registers it ("AT-HYG <id>"), and from
    /// then on it resolves into `star_surfaces` at ≥ `POINT_LOD_PX` and its photosphere is the
    /// nearest surface. Beyond 0.1 pc the star pass draws it and it is never registered.
    #[test]
    fn unnamed_near_star_resolves_after_a_tick() {
        let mut data = empty_data();
        data.stars = vec![
            Star3::new([2.0, 0.0, 0.0], 4.83, [0.0; 3], 0.65, 4711, 0),
            Star3::new([-3.0, 0.0, 0.0], 4.83, [0.0; 3], 0.65, 4712, 0),
        ];
        let mut world = fake_world(vec![(10, None, DVec3::ZERO)], data);
        let objects = world.objects().len();
        assert!(world.objects().iter().all(|o| o.catalog_index.is_none()), "not an object");
        let centre = world.star_position(0);
        let radius = world.star_radius_m(0);
        assert!((radius / R_SUN_M - 1.0).abs() < 0.2, "{radius} m");
        let at = |d: f64| looking_at(centre.offset(DVec3::new(0.0, d, 0.0)), centre, 60.0);
        // Never near: the star pass draws it, and nothing is registered.
        let snap = world.snapshot(&at(0.2 * PC), VIEWPORT);
        assert!(snap.suppressed_stars.is_empty(), "the star pass draws it");
        assert!(snap.points.iter().all(|p| p.id.is_some()));
        world.tick(1.0 / 60.0);
        assert_eq!(world.objects().len(), objects);
        // 0.5 AU (8.7 px): resolved size, but no object yet, so a point without an id.
        let d = 0.5 * AU;
        let px = proj_radius_px(radius, d, 60f64.to_radians(), f64::from(VIEWPORT.y));
        assert!(px >= POINT_LOD_PX, "{px} px");
        let snap = world.snapshot(&at(d), VIEWPORT);
        assert_eq!(snap.suppressed_stars, vec![0]);
        assert!(snap.star_surfaces.is_empty(), "registered at the next tick");
        let points: Vec<_> = snap.points.iter().filter(|p| p.id.is_none()).collect();
        assert_eq!(points.len(), 1);
        assert!((points[0].rel_pos_m - DVec3::new(0.0, -d, 0.0)).length() < 1.0);
        // The catalogue holds M_V as f32.
        let intensity = luminous_intensity_cd(f64::from(4.83_f32));
        assert!((points[0].illuminance_lux / (intensity / (d * d)) - 1.0).abs() < 1e-9);
        // The tick registers it, under its AT-HYG id; ids given before do not move.
        world.tick(1.0 / 60.0);
        assert_eq!(world.objects().len(), objects + 1);
        let id = id_of(&world, "at-hyg 4711");
        assert_eq!(id as usize, objects);
        let info = &world.objects()[id as usize];
        assert_eq!((info.kind, info.catalog_index), (ObjectKind::Star, Some(0)));
        assert_eq!(info.radii_m, DVec3::splat(radius));
        assert_eq!(world.position(id), centre);
        // Now it resolves, drawn once, and its photosphere is the nearest surface.
        let snap = world.snapshot(&at(d), VIEWPORT);
        assert_eq!(snap.suppressed_stars, vec![0]);
        assert_eq!(snap.star_surfaces.len(), 1);
        let surface = &snap.star_surfaces[0];
        assert_eq!((surface.id, surface.catalog_index), (id, Some(0)));
        assert_eq!((surface.radius_m, surface.proj_radius_px), (radius, px));
        assert!((surface.luminous_intensity_cd / intensity - 1.0).abs() < 1e-12);
        assert!(snap.points.iter().all(|p| p.id.is_some()), "never also a point");
        let (near_id, altitude) = snap.nearest_surface.expect("a surface");
        assert_eq!(near_id, id);
        assert!((altitude - (d - radius)).abs() < 1.0, "{altitude}");
        // 0.05 pc: a point again, now with its id.
        let snap = world.snapshot(&at(0.05 * PC), VIEWPORT);
        assert!(snap.star_surfaces.is_empty());
        assert_eq!(snap.points.iter().filter(|p| p.id == Some(id)).count(), 1);
        // `nearest_surface` notes a star too; a second notice registers it only once.
        let other = world.star_position(1).offset(DVec3::new(AU, 0.0, 0.0));
        let sun = id_of(&world, "Sun");
        assert_eq!(world.nearest_surface(&other).map(|(s, _)| s), Some(sun), "not yet");
        world.snapshot(&looking_along(other, DVec3::X), VIEWPORT);
        world.tick(1.0 / 60.0);
        assert_eq!(world.objects().len(), objects + 2);
        let second = id_of(&world, "AT-HYG 4712");
        let (s, altitude) = world.nearest_surface(&other).expect("a surface");
        assert_eq!(s, second);
        assert!((altitude - (AU - world.star_radius_m(1))).abs() < 1.0);
    }

    /// Altitude over an oblate body follows the ellipsoid: 100 km over the Earth's pole and
    /// over its equator.
    #[test]
    fn nearest_surface_follows_the_ellipsoid() {
        let world = fake_world(
            vec![(10, None, DVec3::ZERO), (399, Some(10), DVec3::new(AU, 0.0, 0.0))],
            empty_data(),
        );
        let earth = id_of(&world, "Earth");
        let centre = world.position(earth);
        let r = world.objects()[earth as usize].radii_m;
        for offset in [DVec3::new(0.0, 0.0, r.z + 1.0e5), DVec3::new(r.x + 1.0e5, 0.0, 0.0)] {
            let (id, alt) = world.nearest_surface(&centre.offset(offset)).unwrap();
            assert_eq!(id, earth);
            assert!((alt - 1.0e5).abs() < 1.0, "{alt} at {offset}");
        }
        let (_, inside) = world.nearest_surface(&centre).unwrap();
        assert_eq!(inside, -r.z);
    }

    /// Picks: on-screen objects only, with C3's priorities.
    #[test]
    fn pick_lists_on_screen_objects_by_priority() {
        let mut data = empty_data();
        data.stars = vec![
            Star3::new([0.0, 0.0, 5.0], 1.0, [0.0; 3], 0.0, 1, uni_data::FLAG_NAMED),
            Star3::new([0.0, 0.0, -5.0], 1.0, [0.0; 3], 0.0, 2, uni_data::FLAG_NAMED),
        ];
        data.star_names = Some(StarNames {
            entries: vec![star_name(0, "Northstar"), star_name(1, "Southstar")],
        });
        let world = fake_world(
            vec![(10, None, DVec3::ZERO), (499, Some(10), DVec3::new(0.0, 0.0, 1.5 * AU))],
            data,
        );
        let eye = UniversePos::default().offset(DVec3::new(0.0, 0.0, -AU));
        let view = looking_along(eye, DVec3::Z);
        let snap = world.snapshot(&view, VIEWPORT);
        let ids: Vec<ObjectId> = snap.pick.iter().map(|p| p.id).collect();
        let (sun, mars) = (id_of(&world, "Sun"), id_of(&world, "Mars"));
        let (north, south) = (id_of(&world, "Northstar"), id_of(&world, "Southstar"));
        assert!(ids.contains(&sun) && ids.contains(&mars) && ids.contains(&north), "{ids:?}");
        assert!(!ids.contains(&south), "behind the camera");
        let item = |id| snap.pick.iter().find(|p| p.id == id).unwrap();
        assert_eq!(item(mars).priority, 3);
        assert_eq!(item(north).priority, 1);
        // Straight ahead: the middle of the 1920 × 1080 frame.
        assert!((item(mars).screen_px - glam::Vec2::new(960.0, 540.0)).length() < 1e-3);
        assert!((item(mars).dist_m - 2.5 * AU).abs() < 1.0);
    }

    /// The CPU budget is ≤ 2 ms per snapshot in the release build (plan A4, measured at the
    /// gate). This prints the debug build's cost over the full baked data.
    #[test]
    fn snapshot_cost_with_all_data() {
        let clock = std::time::Instant::now();
        let data = load_world_data(&uni_data::data_dir().join("baked"));
        let loaded = clock.elapsed().as_secs_f64();
        let mut world = World::new(data, date("2026-11-20T00:00:00Z"));
        let built = clock.elapsed().as_secs_f64() - loaded;
        let earth = world.position(id_of(&world, "Earth"));
        let views = [
            looking_at(earth.offset(DVec3::new(6.0e7, 0.0, 0.0)), earth, 60.0),
            looking_along(earth.offset(DVec3::new(0.0, 0.0, 6.0 * AU)), DVec3::NEG_Z),
            looking_along(earth, DVec3::new(-0.0548, -0.8734, -0.4838)),
        ];
        let mut worst: f64 = 0.0;
        for view in &views {
            let t0 = std::time::Instant::now();
            let snap = world.snapshot(view, VIEWPORT);
            worst = worst.max(t0.elapsed().as_secs_f64());
            assert!(!snap.pick.is_empty());
            assert!(snap.pick.len() <= 2000 + world.objects().len());
        }
        let t0 = std::time::Instant::now();
        world.tick(1.0 / 60.0);
        let tick = t0.elapsed().as_secs_f64();
        let start = world.epoch();
        assert!(world.comets.iter().all(|c| !c.needs_work(start)));
        // Review r1 N1: no tick settles. After a 30-year jump back, and after a warp to the end
        // of every comet's perturbed span, each tick steps at most REOSC_STEPS_PER_TICK comets.
        // The first tick after the jump refreshes every state; the timed ones after it (zero
        // wall time, so the date stays) do the comet work alone.
        let catch_up = |world: &mut World| {
            world.tick(1.0 / 60.0);
            let mut slowest: f64 = 0.0;
            for _ in 0..30 {
                let before: Vec<Epoch> = world.comets.iter().map(|c| c.osc_epoch).collect();
                let t0 = std::time::Instant::now();
                world.tick(0.0);
                slowest = slowest.max(t0.elapsed().as_secs_f64());
                let moved = world.comets.iter().zip(&before).filter(|(c, e)| c.osc_epoch != **e);
                assert!(moved.count() <= REOSC_STEPS_PER_TICK);
            }
            assert!(world.comets.iter().any(|c| c.needs_work(world.epoch())), "never settled");
            slowest
        };
        world.time.set_epoch(date("1996-11-20T00:00:00Z"));
        let jump = catch_up(&mut world);
        world.time.set_rate(1.0e12);
        world.tick(1.0 / 60.0);
        world.time.set_rate(1.0);
        let warp = catch_up(&mut world);
        // Back to the start date: every comet takes its start-date checkpoint in one tick.
        world.time.set_epoch(start);
        let t0 = std::time::Instant::now();
        world.tick(1.0 / 60.0);
        let back = t0.elapsed().as_secs_f64();
        assert!(world.comets.iter().all(|c| !c.needs_work(world.epoch())), "checkpoints");

        // Review r2 P1, measured. Ticks until all 512 comets catch up after a jump to 2020,
        // inside most comets' settled spans (120-day checkpoints): those with a later
        // catalogue epoch integrate back from it.
        let stale = |world: &World| {
            world.comets.iter().filter(|c| c.needs_work(world.epoch())).count()
        };
        world.time.set_epoch(date("2020-07-15T00:00:00Z"));
        world.tick(1.0 / 60.0);
        let stale_2020 = stale(&world);
        let mut ticks_2020 = 1;
        let (mut slowest_2020, mut total_2020): (f64, f64) = (0.0, 0.0);
        while stale(&world) > 0 {
            let t0 = std::time::Instant::now();
            world.tick(0.0);
            let took = t0.elapsed().as_secs_f64();
            (slowest_2020, total_2020) = (slowest_2020.max(took), total_2020 + took);
            ticks_2020 += 1;
            assert!(ticks_2020 < 20_000, "no convergence");
        }
        // Ten years past the start date, no checkpoint near: Halley, drawn (frame order
        // tick → snapshot), catches up while almost every other comet still lags.
        let h = comet_index(&world, "1P/Halley");
        let halley = world.comet_objects[h];
        world.time.set_epoch(start.add_seconds(10.0 * YEAR_S));
        world.tick(1.0 / 60.0);
        let mut ticks_drawn = 1;
        let (mut slowest_drawn, mut total_drawn): (f64, f64) = (0.0, 0.0);
        loop {
            let pos = world.position(halley);
            let view = looking_at(pos.offset(DVec3::new(0.0, 0.0, 0.01 * AU)), pos, 2.0);
            let snap = world.snapshot(&view, VIEWPORT);
            assert!(snap.points.iter().any(|p| p.id == Some(halley)), "Halley drawn");
            if !world.comets[h].needs_work(world.epoch()) {
                break;
            }
            let t0 = std::time::Instant::now();
            world.tick(0.0);
            let took = t0.elapsed().as_secs_f64();
            (slowest_drawn, total_drawn) = (slowest_drawn.max(took), total_drawn + took);
            ticks_drawn += 1;
            assert!(ticks_drawn < 20_000, "no convergence");
        }
        let still = stale(&world);
        assert!(still > 400, "{still} comets still lag");
        println!(
            "{} objects ({} cached, {} comets); load {:.0} ms, World::new {:.0} ms (comets \
             settled), tick {:.2} ms, slowest tick after a 30-year jump {:.2} ms and after a \
             warp past 200 years {:.2} ms, tick back to the start date {:.2} ms, worst snapshot \
             {:.2} ms (unoptimised uni-sim test build)",
            world.objects().len(),
            world.cached.len(),
            world.comets.len(),
            loaded * 1e3,
            built * 1e3,
            tick * 1e3,
            jump * 1e3,
            warp * 1e3,
            back * 1e3,
            worst * 1e3
        );
        println!(
            "jump to 2020-07-15: {stale_2020} comets stale, all caught up in {ticks_2020} ticks \
             (mean {:.2} ms, slowest {:.2} ms); jump to start + 10 yr: drawn Halley caught up \
             in {ticks_drawn} ticks (mean {:.2} ms, slowest {:.2} ms), {still} of {} comets \
             still lagging then; {} checkpoints in all (unoptimised uni-sim test build)",
            total_2020 / f64::from(ticks_2020 - 1) * 1e3,
            slowest_2020 * 1e3,
            total_drawn / f64::from(ticks_drawn - 1) * 1e3,
            slowest_drawn * 1e3,
            world.comets.len(),
            world.comets.iter().map(|c| c.checkpoints.len()).sum::<usize>()
        );
        assert!(back < 0.1, "{back} s");
        assert!(worst < 0.25, "{worst} s");
    }
}
