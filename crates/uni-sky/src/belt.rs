//! The procedural asteroid belt (Task C2): "drawn from the catalogue's statistical
//! distributions" (design §5.4). The app raises `HonestyFlag::ProceduralBelt` (grey) while it is
//! drawn and visible (C6).
//!
//! Everything comes from the real records (`World::asteroid_records`: SBDB's asteroids with
//! H ≤ 14, osculating elements in the J2000 ecliptic):
//! - **Main belt** (a = 1.8–5.5 AU outside the Trojan zone) and **Kuiper belt** (30–50 AU): a
//!   3-D histogram of (a, e, i) each, sampled cell by count and uniformly inside the cell, so
//!   the structure its bins resolve survives. 0.01 AU in a resolves the Kirkwood gaps: the
//!   catalogue holds 3 records in the 3:1 gap at [2.49, 2.51) AU against 528 and 1 297 in the
//!   0.05 AU on either side ([2.40, 2.45), [2.55, 2.60)). Node, argument of perihelion and mean
//!   anomaly are uniform.
//! - **Jupiter Trojans** ([`BeltPopulation::Trojan`]): the 4 319 records with a in
//!   [`TROJAN_A_AU`] whose mean longitude leads (L4, 2 561) or trails (L5, 1 758) Jupiter's by
//!   [`TROJAN_OFFSET_DEG`] give a 4-D histogram of (a, e, i, λ − λ_J). A procedural Trojan
//!   moves at Jupiter's mean motion, so it keeps its λ − λ_J at every date. That is an
//!   approximation: a real Trojan librates about L4 or L5 (periods of ~150 yr), which a
//!   two-body orbit cannot do, and a Kepler orbit with its own a would drift out of the cloud
//!   by ~1°/yr. The Trojans take the Trojan zone's share of the 1.8–5.5 AU draw (4 536 of
//!   24 918 records), so `n_main` counts them; the zone's 217 records outside the λ window
//!   (larger librations, a few strays) count in that share and are drawn as window Trojans.
//! - **Absolute magnitudes**: H > `h_cut` from N(<H) ∝ 10^(αH) (Jedicke & Metcalfe 1998,
//!   Icarus 131, 245), with α the least-squares slope of the population's own log₁₀ N(≤H) over
//!   [h_cut − 2, h_cut] ([`fit_h_slope`]). The n objects drawn are the next n beyond the cut
//!   under that law: H spans h_cut … h_cut + log₁₀(1 + n/N(≤h_cut))/α ([`HLaw`]). Main belt:
//!   α = 0.472 from 24 918 records, so the 200 000 objects span H 14–16.02 (the plan's
//!   "2 × 10⁵ procedural ones over H 14–16.0"). The catalogue's Kuiper belt is incomplete there
//!   (18 records in 12 < H ≤ 14), so it borrows the main belt's α. No procedural object
//!   duplicates a catalogue one: every H > h_cut, and the app's cut (14) is above the
//!   registry's (H ≤ 10).
//! - **Seeded**: object k of a population draws from its own `splitmix64` stream
//!   (`hash_u64s([seed, population, k])`), so the same seed gives the same belt, and a smaller
//!   draw (C6's ladder: 250 k → 100 k → 50 k) keeps the orbits of the first objects of each
//!   population; their H move to the brighter end, since n objects are the n brightest beyond
//!   the cut.
//!
//! The shader evaluates M = M₀ + n·(t − [`BELT_EPOCH`]) in f32 (statistical objects; no
//! targeting). [`BeltObject::heliocentric_icrs_m`] is the f64 position (C6's
//! `belt_radial_hist`). Catalogue facts for C2's arithmetic (counted from
//! `data/baked/asteroids.bin` for the C0 context): 33 678 records (H ≤ 14), 19 463 in the main
//! belt 2.1–3.3 AU, 6 095 in 30–50 AU.

use std::f64::consts::{PI, TAU};

use glam::DVec3;
use uni_core::{AU, DAY, Epoch, hash_u64s, splitmix64};
use uni_data::AsteroidRecord;
use uni_physics::bodies::GM_SUN_DE440_M3S2;
use uni_physics::frames::ecliptic_j2000_to_icrs;
use uni_physics::kepler::{perifocal_to_reference, solve_kepler};

/// Procedural main-belt objects (ruling 11: 200 k main + 50 k Kuiper is the starting point; C6
/// lowers it if the pass breaks its budget). The Jupiter Trojans are part of this count.
pub const BELT_MAIN_COUNT: u32 = 200_000;
/// Procedural Kuiper-belt objects (ruling 11).
pub const BELT_KUIPER_COUNT: u32 = 50_000;
/// The catalogue's own cut, and the `h_cut` the app passes: `asteroids.bin` holds SBDB's
/// asteroids with H ≤ 14 (ruling 11), so the procedural objects begin where it ends.
pub const BELT_H_CUT: f32 = 14.0;
/// The app's seed. Any value gives a valid belt; a fixed one makes captures repeat.
pub const BELT_SEED: u64 = 0x00C2_BE17;
/// The epoch of the procedural mean anomalies M₀: the shader's belt time is seconds since it.
pub const BELT_EPOCH: Epoch = Epoch::J2000;
/// The main belt's semi-major axes [AU] (plan): from the Hungarias at 1.8–2.0 AU past the Hildas
/// at 4.0 AU and the Trojans at 5.2 AU.
pub const MAIN_BELT_A_AU: [f64; 2] = [1.8, 5.5];
/// The Kuiper belt's semi-major axes [AU] (plan).
pub const KUIPER_BELT_A_AU: [f64; 2] = [30.0, 50.0];
/// The Trojan zone [AU]: Jupiter's 5.203 AU (Standish Table 1) ± 0.15 AU of libration in the
/// osculating a. It holds every catalogue record in the λ window but 28 (counted from
/// `asteroids.bin`).
pub const TROJAN_A_AU: [f64; 2] = [5.05, 5.35];
/// |λ − λ_J| of a Trojan [deg], 60 ± 25 (brief): leading (L4, λ − λ_J > 0) or trailing (L5).
/// It holds 4 319 of the Trojan zone's 4 536 records. Half-open, [35, 85): the λ bins below
/// start at −180° in steps of 5°, so ±35° and ±85° are bin edges and every draw stays inside.
pub const TROJAN_OFFSET_DEG: [f64; 2] = [35.0, 85.0];

/// Histogram cells of the main belt, (a [AU], e, i [deg]): the lower corner and the widths.
const MAIN_BINS: ([f64; 3], [f64; 3]) = ([1.8, 0.0, 0.0], [0.01, 0.02, 1.0]);
/// Kuiper belt cells: 0.1 AU in a resolves the Plutinos' 3:2 resonance at 39.4 AU.
const KUIPER_BINS: ([f64; 3], [f64; 3]) = ([30.0, 0.0, 0.0], [0.1, 0.02, 1.0]);
/// Trojan cells, (a, e, i, λ − λ_J [deg]).
const TROJAN_BINS: ([f64; 4], [f64; 4]) = ([5.05, 0.0, 0.0, -180.0], [0.01, 0.02, 1.0, 5.0]);
/// Largest eccentricity drawn. The catalogue's own largest in these ranges are 0.835 (main) and
/// 0.917 (Kuiper), so this only guards a foreign catalogue: it keeps every procedural orbit
/// bound and inside the f32 solver's converged range ([`crate::kepler_f32`]).
const MAX_DRAWN_E: f64 = 0.99;

/// Jupiter's mean longitude at J2000 [deg] and its rate [deg per Julian century], J2000
/// ecliptic: Standish, "Keplerian Elements for Approximate Positions of the Major Planets",
/// Table 1 (1800–2050), https://ssd.jpl.nasa.gov/planets/approx_pos.html (the row
/// `uni_physics::standish` copies; that module keeps its elements private).
const JUPITER_L_J2000_DEG: f64 = 34.396_440_51;
const JUPITER_L_RATE_DEG_PER_CY: f64 = 3_034.746_127_75;
/// One Julian century [s] (36 525 days, IAU 1976).
const JULIAN_CENTURY_S: f64 = 36_525.0 * DAY;
/// Julian Date (TDB) of J2000.0 (IERS Conventions 2010).
const JD_J2000: f64 = 2_451_545.0;

/// The cumulative-count window of the H fit [mag] below the cut, and its step.
const H_FIT_WINDOW_MAG: f64 = 2.0;
const H_FIT_STEP_MAG: f64 = 0.25;
/// Fewest records in the fit window for a population's own α. With fewer, each count is
/// uncertain by more than 10 % (Poisson, 1/√100) and the window is dominated by the catalogue's
/// incompleteness (the Kuiper belt: 18 records), so the population borrows the main belt's α.
const MIN_FIT_RECORDS: usize = 100;
/// α when no population can be fitted (only a tiny or foreign catalogue): the plan's
/// visibility arithmetic's value (review I12, "α = 0.35").
const ASSUMED_H_SLOPE: f64 = 0.35;

/// Which population a procedural object was drawn from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BeltPopulation {
    /// 1.8–5.5 AU outside the Trojan zone.
    Main,
    /// Jupiter's L4 and L5 clouds.
    Trojan,
    /// 30–50 AU.
    Kuiper,
}

impl BeltPopulation {
    /// The tag of the population's random streams.
    fn tag(self) -> u64 {
        match self {
            BeltPopulation::Main => 1,
            BeltPopulation::Trojan => 2,
            BeltPopulation::Kuiper => 3,
        }
    }
}

/// One procedural object: heliocentric two-body elements in the J2000 ecliptic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeltObject {
    pub population: BeltPopulation,
    /// Semi-major axis [AU].
    pub a_au: f64,
    pub e: f64,
    /// Inclination to the J2000 ecliptic [rad].
    pub i_rad: f64,
    /// Longitude of the ascending node [rad].
    pub node_rad: f64,
    /// Argument of perihelion [rad].
    pub argp_rad: f64,
    /// Mean anomaly at [`BELT_EPOCH`] [rad], in [−π, π).
    pub m0_rad: f64,
    /// Mean motion [rad/s]: √(GM☉/a³) (DE440's GM☉, as `uni_physics::smallbody`), or Jupiter's
    /// for a Trojan.
    pub n_rad_s: f64,
    /// Absolute magnitude H, above the cut; the H–G slope is `crate::DEFAULT_SLOPE_G`.
    pub h: f32,
}

impl BeltObject {
    /// Mean anomaly at `t` [rad], unwrapped: M₀ + n·(t − [`BELT_EPOCH`]).
    pub fn mean_anomaly_rad(&self, t: Epoch) -> f64 {
        self.m0_rad + self.n_rad_s * t.seconds_since(BELT_EPOCH)
    }

    /// Mean longitude Ω + ω + M at `t` [rad], unwrapped, J2000 ecliptic.
    pub fn mean_longitude_rad(&self, t: Epoch) -> f64 {
        self.node_rad + self.argp_rad + self.mean_anomaly_rad(t)
    }

    /// Heliocentric position at `t` [m], ICRS axes, in f64: uni_physics' Kepler solve
    /// (`solve_kepler`), the perifocal → ecliptic rotation and the J2000 obliquity
    /// (`ecliptic_j2000_to_icrs`).
    pub fn heliocentric_icrs_m(&self, t: Epoch) -> DVec3 {
        let ea = solve_kepler(self.mean_anomaly_rad(t), self.e);
        let a = self.a_au * AU;
        let b = a * ((1.0 - self.e) * (1.0 + self.e)).sqrt();
        let perifocal = DVec3::new(a * (ea.cos() - self.e), b * ea.sin(), 0.0);
        let rot = perifocal_to_reference(self.i_rad, self.node_rad, self.argp_rad);
        ecliptic_j2000_to_icrs(rot * perifocal)
    }
}

/// The H distribution of one population: N(<H) ∝ 10^(αH) beyond the cut (Jedicke & Metcalfe
/// 1998), normalised so the `drawn` objects are the next ones after the catalogue's `n_cut`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HLaw {
    /// α of N(<H) ∝ 10^(αH).
    pub alpha: f64,
    /// Whether α is this population's own fit (`false`: borrowed or assumed).
    pub fitted: bool,
    /// Catalogue records of the population with H ≤ h_cut, N(≤ h_cut).
    pub n_cut: u32,
    /// Objects drawn under the law.
    pub drawn: u32,
    /// The cut [mag]: every H drawn is above it.
    pub h_cut: f64,
    /// The faintest H drawn: N(< h_max) − N(< h_cut) = `drawn`.
    pub h_max: f64,
}

impl HLaw {
    fn new(alpha: f64, fitted: bool, n_cut: u32, drawn: u32, h_cut: f64) -> Self {
        let mut law = HLaw {
            alpha,
            fitted,
            n_cut,
            drawn,
            h_cut,
            h_max: h_cut,
        };
        law.h_max = law.quantile(1.0);
        law
    }

    /// The H at quantile `u` ∈ (0, 1] of the law truncated to (h_cut, h_max]: the inverse of
    /// its CDF, h_cut + log₁₀(1 + u·drawn/N(≤h_cut))/α. An empty population is taken as one
    /// record, so the law stays finite.
    pub fn quantile(&self, u: f64) -> f64 {
        let per_cut = f64::from(self.drawn) / f64::from(self.n_cut.max(1));
        self.h_cut + (u * per_cut).ln_1p() / (std::f64::consts::LN_10 * self.alpha)
    }
}

/// α of N(<H) ∝ 10^(αH): the least-squares slope of log₁₀ N(≤H) against H at H = h_cut − 2,
/// h_cut − 1.75, …, h_cut, with `h` the population's absolute magnitudes (Jedicke & Metcalfe
/// 1998 fit the same power law to the main belt's cumulative H counts). `None` when the window
/// holds fewer than 100 records (`MIN_FIT_RECORDS`), when N(≤ h_cut − 2) is 0, or when the slope
/// is not > 0.
pub fn fit_h_slope(h: &[f32], h_cut: f64) -> Option<f64> {
    let mut sorted: Vec<f64> = h
        .iter()
        .map(|&x| f64::from(x))
        .filter(|x| x.is_finite())
        .collect();
    sorted.sort_by(f64::total_cmp);
    let count_le = |x: f64| sorted.partition_point(|&v| v <= x);
    let lo = h_cut - H_FIT_WINDOW_MAG;
    if count_le(lo) == 0 || count_le(h_cut) - count_le(lo) < MIN_FIT_RECORDS {
        return None;
    }
    let steps = (H_FIT_WINDOW_MAG / H_FIT_STEP_MAG).round() as usize;
    let points: Vec<(f64, f64)> = (0..=steps)
        .map(|k| {
            let x = lo + H_FIT_STEP_MAG * k as f64;
            (x, (count_le(x) as f64).log10())
        })
        .collect();
    let n = points.len() as f64;
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x / n, sy + y / n));
    let (sxy, sxx) = points.iter().fold((0.0, 0.0), |(sxy, sxx), (x, y)| {
        (sxy + (x - mx) * (y - my), sxx + (x - mx) * (x - mx))
    });
    let alpha = sxy / sxx;
    (alpha.is_finite() && alpha > 0.0).then_some(alpha)
}

/// Jupiter's mean longitude at `t` [rad], unwrapped, J2000 ecliptic (Standish Table 1, linear in
/// T; the Trojans' λ − λ_J is measured from it).
pub fn jupiter_mean_longitude_rad(t: Epoch) -> f64 {
    (JUPITER_L_J2000_DEG + JUPITER_L_RATE_DEG_PER_CY * t.julian_centuries()).to_radians()
}

/// Jupiter's mean motion [rad/s], the rate of [`jupiter_mean_longitude_rad`].
pub fn jupiter_mean_motion_rad_s() -> f64 {
    JUPITER_L_RATE_DEG_PER_CY.to_radians() / JULIAN_CENTURY_S
}

/// λ − λ_J of record `r` at its own epoch [deg], wrapped to [−180, 180).
fn trojan_offset_deg(r: &AsteroidRecord) -> f64 {
    let t = Epoch::from_days_since_j2000(r.epoch_jd_tdb - JD_J2000);
    let lambda = r.node_deg + r.argp_deg + r.m0_deg;
    wrap_deg(lambda - jupiter_mean_longitude_rad(t).to_degrees())
}

/// `x` [deg] wrapped to [−180, 180).
fn wrap_deg(x: f64) -> f64 {
    x - 360.0 * ((x + 180.0) / 360.0).floor()
}

/// `x` [rad] wrapped to [−π, π).
fn wrap_rad(x: f64) -> f64 {
    x - TAU * ((x + PI) / TAU).floor()
}

/// Whether λ − λ_J = `dl` [deg] is in a Trojan cloud ([`TROJAN_OFFSET_DEG`], half-open).
fn in_trojan_window(dl: f64) -> bool {
    let [lo, hi] = TROJAN_OFFSET_DEG;
    (lo..hi).contains(&dl) || (-hi..-lo).contains(&dl)
}

fn in_range(range: [f64; 2], x: f64) -> bool {
    (range[0]..=range[1]).contains(&x)
}

/// A record whose elements describe a bound orbit (the loader checks this; a foreign slice
/// may not).
fn usable(r: &AsteroidRecord) -> bool {
    let finite = [r.a_au, r.e, r.i_deg, r.node_deg, r.argp_deg, r.m0_deg, r.epoch_jd_tdb]
        .iter()
        .all(|x| x.is_finite())
        && r.h.is_finite();
    finite && r.a_au > 0.0 && (0.0..1.0).contains(&r.e)
}

/// A `splitmix64` counter stream (Steele, Lea & Flood 2014): value k is
/// `splitmix64(start + k)`.
struct Stream(u64);

impl Stream {
    fn new(parts: &[u64]) -> Self {
        Stream(hash_u64s(parts))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(1);
        splitmix64(self.0)
    }

    /// Uniform in [0, 1), 53 bits.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in (0, 1], 53 bits.
    fn unit_open(&mut self) -> f64 {
        1.0 - self.unit()
    }

    /// Uniform integer in [0, n): the multiply-shift reduction (Lemire 2019, ACM TOMACS 29, 3),
    /// biased by under n/2⁶⁴.
    fn below(&mut self, n: u64) -> u64 {
        ((u128::from(self.next_u64()) * u128::from(n)) >> 64) as u64
    }
}

/// A histogram over `D` dimensions with cells of `width` from `lo`, stored sparsely: the occupied
/// cells in key order with cumulative counts.
#[derive(Clone, Debug)]
struct Histogram<const D: usize> {
    lo: [f64; D],
    width: [f64; D],
    keys: Vec<[i64; D]>,
    /// Records in cells 0..=k.
    cumulative: Vec<u64>,
}

impl<const D: usize> Histogram<D> {
    /// The histogram of `points`, or `None` when there are none.
    fn new((lo, width): ([f64; D], [f64; D]), points: &[[f64; D]]) -> Option<Self> {
        let mut cells: Vec<[i64; D]> = points
            .iter()
            .map(|p| std::array::from_fn(|d| ((p[d] - lo[d]) / width[d]).floor() as i64))
            .collect();
        cells.sort_unstable();
        let mut keys: Vec<[i64; D]> = Vec::new();
        let mut cumulative: Vec<u64> = Vec::new();
        for (n, cell) in cells.iter().enumerate() {
            if keys.last() == Some(cell) {
                if let Some(c) = cumulative.last_mut() {
                    *c += 1;
                }
            } else {
                keys.push(*cell);
                cumulative.push(n as u64 + 1);
            }
        }
        (!keys.is_empty()).then_some(Histogram {
            lo,
            width,
            keys,
            cumulative,
        })
    }

    /// A point drawn with the histogram's density: a cell with probability count/total, then
    /// uniform inside it.
    fn sample(&self, s: &mut Stream) -> [f64; D] {
        let total = self.cumulative.last().copied().unwrap_or(0);
        let pick = s.below(total);
        let k = self.cumulative.partition_point(|&c| c <= pick);
        let key = self.keys[k.min(self.keys.len() - 1)];
        std::array::from_fn(|d| self.lo[d] + (key[d] as f64 + s.unit()) * self.width[d])
    }
}

/// The sampled procedural elements: the main-belt objects, then the Trojans, then the Kuiper
/// belt (each in draw order).
#[derive(Clone, Debug)]
pub struct BeltModel {
    objects: Vec<BeltObject>,
    /// The main belt's and Trojans' H law (one law for 1.8–5.5 AU).
    main_law: Option<HLaw>,
    kuiper_law: Option<HLaw>,
}

impl BeltModel {
    /// Samples `n_main` main-belt objects (the Jupiter Trojans among them, at the catalogue's
    /// share) and `n_kuiper` Kuiper-belt objects fainter than `h_cut` from the distributions of
    /// `real`, deterministically from `seed` (module doc). `h_cut` should be the catalogue's own
    /// cut ([`BELT_H_CUT`]); a lower one makes the procedural set overlap the catalogue's H range.
    /// A population with no usable record in `real` gets no objects.
    pub fn from_catalogue(
        real: &[AsteroidRecord],
        n_main: u32,
        n_kuiper: u32,
        h_cut: f32,
        seed: u64,
    ) -> Self {
        let mut main_pts: Vec<[f64; 3]> = Vec::new();
        let mut trojan_pts: Vec<[f64; 4]> = Vec::new();
        let mut kuiper_pts: Vec<[f64; 3]> = Vec::new();
        let (mut main_h, mut kuiper_h) = (Vec::new(), Vec::new());
        let mut trojan_zone = 0u64;
        for r in real.iter().filter(|r| usable(r)) {
            let cell = [r.a_au, r.e, r.i_deg];
            if in_range(MAIN_BELT_A_AU, r.a_au) {
                main_h.push(r.h);
                if in_range(TROJAN_A_AU, r.a_au) {
                    trojan_zone += 1;
                    let dl = trojan_offset_deg(r);
                    if in_trojan_window(dl) {
                        trojan_pts.push([r.a_au, r.e, r.i_deg, dl]);
                    }
                } else {
                    main_pts.push(cell);
                }
            } else if in_range(KUIPER_BELT_A_AU, r.a_au) {
                kuiper_h.push(r.h);
                kuiper_pts.push(cell);
            }
        }
        let main_hist = Histogram::new(MAIN_BINS, &main_pts);
        let trojan_hist = Histogram::new(TROJAN_BINS, &trojan_pts);
        let kuiper_hist = Histogram::new(KUIPER_BINS, &kuiper_pts);

        let h_cut64 = f64::from(h_cut);
        let n_cut = |h: &[f32]| {
            u32::try_from(h.iter().filter(|&&x| x <= h_cut).count()).unwrap_or(u32::MAX)
        };
        let main_fit = fit_h_slope(&main_h, h_cut64);
        let main_law = (!main_h.is_empty()).then(|| {
            let (alpha, fitted) = main_fit.map_or((ASSUMED_H_SLOPE, false), |a| (a, true));
            HLaw::new(alpha, fitted, n_cut(&main_h), n_main, h_cut64)
        });
        let kuiper_law = (!kuiper_h.is_empty()).then(|| {
            let (alpha, fitted) = match fit_h_slope(&kuiper_h, h_cut64) {
                Some(a) => (a, true),
                None => (main_fit.unwrap_or(ASSUMED_H_SLOPE), false),
            };
            HLaw::new(alpha, fitted, n_cut(&kuiper_h), n_kuiper, h_cut64)
        });

        // The Trojans' share of the 1.8–5.5 AU draw: the Trojan zone's share of the records
        // (only populations with a histogram take part).
        let w_main = if main_hist.is_some() {
            main_pts.len() as u64
        } else {
            0
        };
        let w_trojan = if trojan_hist.is_some() {
            trojan_zone
        } else {
            0
        };
        let n_trojan = if w_main + w_trojan == 0 {
            0
        } else {
            (f64::from(n_main) * w_trojan as f64 / (w_main + w_trojan) as f64).round() as u32
        };
        let n_main_only = if w_main == 0 { 0 } else { n_main - n_trojan };

        let draw = Draw { seed, h_cut };
        let mut objects =
            Vec::with_capacity((n_main_only + n_trojan) as usize + n_kuiper as usize);
        if let (Some(hist), Some(law)) = (&main_hist, &main_law) {
            objects.extend(
                (0..n_main_only).map(|k| draw.object(BeltPopulation::Main, k, hist, law)),
            );
        }
        if let (Some(hist), Some(law)) = (&trojan_hist, &main_law) {
            objects.extend((0..n_trojan).map(|k| draw.trojan(k, hist, law)));
        }
        if let (Some(hist), Some(law)) = (&kuiper_hist, &kuiper_law) {
            objects.extend(
                (0..n_kuiper).map(|k| draw.object(BeltPopulation::Kuiper, k, hist, law)),
            );
        }
        BeltModel {
            objects,
            main_law,
            kuiper_law,
        }
    }

    /// Number of procedural objects.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// No procedural objects.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The objects: main belt, then Trojans, then Kuiper belt (the order of the GPU buffer).
    pub fn objects(&self) -> &[BeltObject] {
        &self.objects
    }

    /// The H law `population` was drawn under (the Trojans share the main belt's), or `None` when
    /// the catalogue has no records there.
    pub fn h_law(&self, population: BeltPopulation) -> Option<HLaw> {
        match population {
            BeltPopulation::Main | BeltPopulation::Trojan => self.main_law,
            BeltPopulation::Kuiper => self.kuiper_law,
        }
    }
}

/// One object's draws, in a fixed order per stream.
struct Draw {
    seed: u64,
    h_cut: f32,
}

impl Draw {
    /// H above the cut, strictly also after the f32 cast.
    fn h(&self, law: &HLaw, s: &mut Stream) -> f32 {
        (law.quantile(s.unit_open()) as f32).max(self.h_cut.next_up())
    }

    /// Object `k` of a main-belt or Kuiper-belt population: (a, e, i) from `hist`, H from
    /// `law`, node, argument of perihelion and mean anomaly uniform.
    fn object(
        &self,
        population: BeltPopulation,
        k: u32,
        hist: &Histogram<3>,
        law: &HLaw,
    ) -> BeltObject {
        let mut s = Stream::new(&[self.seed, population.tag(), u64::from(k)]);
        let [a_au, e, i_deg] = hist.sample(&mut s);
        let h = self.h(law, &mut s);
        BeltObject {
            population,
            a_au,
            e: e.min(MAX_DRAWN_E),
            i_rad: i_deg.clamp(0.0, 180.0).to_radians(),
            node_rad: TAU * s.unit(),
            argp_rad: TAU * s.unit(),
            m0_rad: TAU * s.unit() - PI,
            n_rad_s: (GM_SUN_DE440_M3S2 / (a_au * AU).powi(3)).sqrt(),
            h,
        }
    }

    /// Trojan `k`: (a, e, i, λ − λ_J) from `hist`, H from `law`, node and argument of
    /// perihelion uniform, and M₀ so that λ − λ_J at [`BELT_EPOCH`] is the drawn offset; it
    /// moves at Jupiter's mean motion, so the offset holds at every date.
    fn trojan(&self, k: u32, hist: &Histogram<4>, law: &HLaw) -> BeltObject {
        let mut s = Stream::new(&[self.seed, BeltPopulation::Trojan.tag(), u64::from(k)]);
        let [a_au, e, i_deg, offset_deg] = hist.sample(&mut s);
        let h = self.h(law, &mut s);
        let node_rad = TAU * s.unit();
        let argp_rad = TAU * s.unit();
        let lambda = jupiter_mean_longitude_rad(BELT_EPOCH) + offset_deg.to_radians();
        BeltObject {
            population: BeltPopulation::Trojan,
            a_au,
            e: e.min(MAX_DRAWN_E),
            i_rad: i_deg.clamp(0.0, 180.0).to_radians(),
            node_rad,
            argp_rad,
            m0_rad: wrap_rad(lambda - node_rad - argp_rad),
            n_rad_s: jupiter_mean_motion_rad_s(),
            h,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn catalogue() -> &'static [AsteroidRecord] {
        static CATALOGUE: OnceLock<Vec<AsteroidRecord>> = OnceLock::new();
        CATALOGUE.get_or_init(|| {
            let dir = uni_data::data_dir().join("baked");
            uni_data::load_asteroids(
                &dir.join("asteroids.bin"),
                &dir.join("asteroid_names.txt"),
            )
            .expect("committed asteroids.bin")
            .0
        })
    }

    /// The app's belt: the brief's counts, cut and seed.
    fn belt() -> &'static BeltModel {
        static BELT: OnceLock<BeltModel> = OnceLock::new();
        BELT.get_or_init(|| {
            BeltModel::from_catalogue(
                catalogue(),
                BELT_MAIN_COUNT,
                BELT_KUIPER_COUNT,
                BELT_H_CUT,
                BELT_SEED,
            )
        })
    }

    fn in_range(range: [f64; 2], x: f64) -> bool {
        (range[0]..=range[1]).contains(&x)
    }

    /// The two-sample Kolmogorov–Smirnov distance sup |F₁ − F₂| of `x` and `y`.
    fn ks_distance(mut x: Vec<f64>, mut y: Vec<f64>) -> f64 {
        x.sort_by(f64::total_cmp);
        y.sort_by(f64::total_cmp);
        let (nx, ny) = (x.len() as f64, y.len() as f64);
        let (mut i, mut j, mut d) = (0, 0, 0.0f64);
        while i < x.len() && j < y.len() {
            let v = x[i].min(y[j]);
            while i < x.len() && x[i] <= v {
                i += 1;
            }
            while j < y.len() && y[j] <= v {
                j += 1;
            }
            d = d.max((i as f64 / nx - j as f64 / ny).abs());
        }
        d
    }

    /// The procedural main-belt objects (the Trojans included: they are part of 1.8–5.5 AU)
    /// against the catalogue's records there, and the Kuiper belt likewise: KS distance on a
    /// below 0.02 (brief). The counts are the ones asked for.
    #[test]
    fn belt_matches_catalogue_a_distribution() {
        let belt = belt();
        assert_eq!(
            belt.len(),
            (BELT_MAIN_COUNT + BELT_KUIPER_COUNT) as usize
        );
        let of = |pops: &[BeltPopulation]| -> Vec<f64> {
            belt.objects()
                .iter()
                .filter(|o| pops.contains(&o.population))
                .map(|o| o.a_au)
                .collect()
        };
        let main = of(&[BeltPopulation::Main, BeltPopulation::Trojan]);
        let kuiper = of(&[BeltPopulation::Kuiper]);
        assert_eq!(main.len(), BELT_MAIN_COUNT as usize);
        assert_eq!(kuiper.len(), BELT_KUIPER_COUNT as usize);
        let real = |range: [f64; 2]| -> Vec<f64> {
            catalogue()
                .iter()
                .map(|r| r.a_au)
                .filter(|&a| in_range(range, a))
                .collect()
        };
        let d_main = ks_distance(main, real(MAIN_BELT_A_AU));
        let d_kuiper = ks_distance(kuiper, real(KUIPER_BELT_A_AU));
        assert!(d_main < 0.02, "main belt KS distance {d_main}");
        assert!(d_kuiper < 0.02, "Kuiper belt KS distance {d_kuiper}");
    }

    /// The 3:1 Kirkwood gap at 2.50 AU: the procedural density in [2.49, 2.51] AU is below 0.3 ×
    /// the mean of [2.40, 2.45] and [2.55, 2.60] (brief).
    #[test]
    fn belt_shows_kirkwood_gap() {
        let density = |lo: f64, hi: f64| {
            let n = belt()
                .objects()
                .iter()
                .filter(|o| o.population != BeltPopulation::Kuiper && (lo..hi).contains(&o.a_au))
                .count();
            n as f64 / (hi - lo)
        };
        let gap = density(2.49, 2.51);
        let sides = 0.5 * (density(2.40, 2.45) + density(2.55, 2.60));
        assert!(sides > 0.0);
        assert!(gap < 0.3 * sides, "gap {gap} per AU vs sides {sides} per AU");
    }

    /// Every procedural Trojan's λ − λ_J lies in ±(60 ± 25)° (brief), at any date (they move at
    /// Jupiter's mean motion), both clouds are populated, and the Trojans take the catalogue's
    /// share of the 1.8–5.5 AU population.
    #[test]
    fn trojans_cluster_at_l4_l5() {
        let trojans: Vec<&BeltObject> = belt()
            .objects()
            .iter()
            .filter(|o| o.population == BeltPopulation::Trojan)
            .collect();
        assert!(!trojans.is_empty());
        let dates = [
            BELT_EPOCH,
            Epoch::from_days_since_j2000(9_820.5),
            Epoch::from_days_since_j2000(-36_525.0 * 3.0),
            Epoch::from_days_since_j2000(36_525.0 * 5.0),
        ];
        let (mut l4, mut l5) = (0usize, 0usize);
        for t in dates {
            for o in &trojans {
                let dl = (o.mean_longitude_rad(t) - jupiter_mean_longitude_rad(t)).to_degrees();
                let dl = dl - 360.0 * ((dl + 180.0) / 360.0).floor();
                let off = dl.abs();
                assert!(
                    (TROJAN_OFFSET_DEG[0] - 1e-6..=TROJAN_OFFSET_DEG[1] + 1e-6).contains(&off),
                    "Trojan at λ − λ_J = {dl}° on day {}",
                    t.days_since_j2000()
                );
                assert!((35.0..=85.0).contains(&off));
                if t == BELT_EPOCH {
                    if dl > 0.0 { l4 += 1 } else { l5 += 1 }
                }
                assert_eq!(o.n_rad_s, jupiter_mean_motion_rad_s());
            }
        }
        // Both clouds, with the catalogue's leading excess (L4 2 673 vs L5 1 845 records).
        assert!(l4 > l5 && l5 > trojans.len() / 4, "L4 {l4} L5 {l5}");
        // The Trojans' share of the 1.8–5.5 AU draw is the catalogue's share of the Trojan zone.
        let zone = catalogue()
            .iter()
            .filter(|r| in_range(TROJAN_A_AU, r.a_au))
            .count() as f64;
        let main = catalogue()
            .iter()
            .filter(|r| in_range(MAIN_BELT_A_AU, r.a_au))
            .count() as f64;
        let share = trojans.len() as f64 / f64::from(BELT_MAIN_COUNT);
        assert!((share - zone / main).abs() < 1e-4, "{share} vs {}", zone / main);
    }

    /// Every procedural H is above the cut, so no procedural object duplicates a catalogue one
    /// (the registry, H ≤ 10, included), also for a cut inside the catalogue's range.
    #[test]
    fn procedural_h_above_cut() {
        for o in belt().objects() {
            assert!(o.h > BELT_H_CUT, "{o:?}");
        }
        let lower = BeltModel::from_catalogue(catalogue(), 2_000, 500, 12.0, 7);
        assert_eq!(lower.len(), 2_500);
        assert!(lower.objects().iter().all(|o| o.h > 12.0));
    }

    /// The same seed gives identical objects and identical GPU buffers; another seed does not.
    /// A smaller draw (C6's ladder) keeps the first objects' orbits of each population, and
    /// their H moves to the brighter end: the n objects are the n brightest beyond the cut.
    #[test]
    fn belt_deterministic() {
        let a = BeltModel::from_catalogue(catalogue(), 4_000, 1_000, BELT_H_CUT, 42);
        let b = BeltModel::from_catalogue(catalogue(), 4_000, 1_000, BELT_H_CUT, 42);
        assert_eq!(a.objects(), b.objects());
        let pack = |m: &BeltModel| -> Vec<u8> {
            let gpu: Vec<crate::asteroids::AsteroidGpu> =
                m.objects().iter().map(crate::asteroids::belt_gpu).collect();
            bytemuck::cast_slice(&gpu).to_vec()
        };
        assert_eq!(pack(&a), pack(&b));
        let c = BeltModel::from_catalogue(catalogue(), 4_000, 1_000, BELT_H_CUT, 43);
        assert_ne!(pack(&a), pack(&c));
        let small = BeltModel::from_catalogue(catalogue(), 2_000, 500, BELT_H_CUT, 42);
        for pop in [BeltPopulation::Main, BeltPopulation::Trojan, BeltPopulation::Kuiper] {
            let first: Vec<_> = small.objects().iter().filter(|o| o.population == pop).collect();
            let full: Vec<_> = a.objects().iter().filter(|o| o.population == pop).collect();
            assert!(!first.is_empty());
            for (s, f) in first.iter().zip(&full) {
                let orbit = |o: &BeltObject| {
                    [o.a_au, o.e, o.i_rad, o.node_rad, o.argp_rad, o.m0_rad, o.n_rad_s]
                };
                assert_eq!(orbit(s), orbit(f), "{pop:?}");
                assert!(s.h <= f.h && s.h > BELT_H_CUT, "{pop:?}: {} vs {}", s.h, f.h);
            }
        }
    }

    /// α is fitted to the main belt's own counts over [h_cut − 2, h_cut], so the 200 000 objects
    /// fill H 14–16 (the brief's "2 × 10⁵ procedural ones over H 14–16.0"); the Kuiper belt's
    /// window holds 18 records (the catalogue is incomplete there), so it borrows the main belt's
    /// α. An empty catalogue gives an empty belt.
    #[test]
    fn h_law_fitted_to_the_catalogue() {
        let main = belt().h_law(BeltPopulation::Main).expect("main law");
        assert!(main.fitted);
        assert!((0.3..0.7).contains(&main.alpha), "{main:?}");
        assert_eq!(main.n_cut, 24_918);
        assert_eq!(main.drawn, BELT_MAIN_COUNT);
        assert!((main.h_max - 16.0).abs() < 0.3, "{main:?}");
        assert_eq!(belt().h_law(BeltPopulation::Trojan), Some(main));
        let kuiper = belt().h_law(BeltPopulation::Kuiper).expect("kuiper law");
        assert!(!kuiper.fitted);
        assert_eq!(kuiper.alpha, main.alpha);
        assert_eq!(kuiper.n_cut, 6_095);
        let h_max = belt()
            .objects()
            .iter()
            .filter(|o| o.population != BeltPopulation::Kuiper)
            .map(|o| f64::from(o.h))
            .fold(0.0, f64::max);
        assert!(h_max <= main.h_max + 1e-5 && h_max > main.h_max - 0.01, "{h_max}");
        let empty = BeltModel::from_catalogue(&[], 1_000, 1_000, BELT_H_CUT, 1);
        assert!(empty.is_empty());
        assert_eq!(empty.h_law(BeltPopulation::Main), None);
    }

    /// The f64 position is the conic's: uni_physics' Kepler solve and ecliptic → ICRS rotation.
    #[test]
    fn belt_position_is_the_conic_state() {
        let o = belt().objects()[17];
        let t = Epoch::from_days_since_j2000(9_820.5);
        let conic = uni_physics::kepler::Conic::from_mean_anomaly(
            o.a_au * uni_core::AU,
            o.e,
            o.i_rad,
            o.node_rad,
            o.argp_rad,
            o.mean_anomaly_rad(t),
            t,
            uni_physics::bodies::GM_SUN_DE440_M3S2,
        );
        let want = uni_physics::smallbody::heliocentric_state_icrs(&conic, t).0;
        let got = o.heliocentric_icrs_m(t);
        assert!((got - want).length() < 1.0, "{got} vs {want}");
    }
}
