//! The M1 ephemeris: which theory places each body, and positions relative to its parent and to
//! the SSB (M1 Task A2).
//!
//! | Body | Theory | Relative to |
//! |---|---|---|
//! | Sun (10) | [`crate::ssb`]: the Standish planets' barycentre offset | SSB (the root) |
//! | Mercury … Neptune (199 … 899) | [`crate::standish`] (system barycentres) | Sun |
//! | Earth (399) | EMB − (GM☾ / (GM⊕ + GM☾))·r☾ | Sun |
//! | Pluto (999) | Meeus ch. 37 barycentre ([`crate::pluto`]) − (GM_Charon / (GM_Pluto + GM_Charon))·r_Charon | Sun |
//! | Moon (301) | Meeus ch. 47 ([`crate::moon`]) | Earth |
//! | Io … Callisto (501 … 504) | Meeus ch. 44, E5 ([`crate::galilean`]) | Jupiter |
//! | Mimas … Iapetus (601 … 608) | Meeus ch. 46 ([`crate::saturn_moons`]) | Saturn |
//! | every other moon given | JPL mean elements ([`crate::sat_elements`]) | its planet |
//!
//! Planets other than the Earth and Pluto are placed at their system barycentre: their moons
//! shift them by at most 290 km (Saturn by Titan; Jupiter ≈ 200 km), far below Standish's
//! 10⁵–10⁶ km errors, and every moon is placed relative to that same point, so the moon–planet
//! geometry is unaffected. Pluto's offset, ≈ 2 100 km, exceeds Pluto's radius, so it is applied
//! (from Charon's mean elements; without them Pluto is placed at the barycentre).
//!
//! Outside ±3000 yr of J2000 ([`MOON_THEORY_WINDOW_CY`]) the Meeus moon theories, whose
//! arguments are polynomials in T up to T⁴, are replaced by the same moon's JPL mean-element
//! orbit and [`EphemFlag::MoonTheoryFallback`] is raised. A Meeus moon whose elements were not
//! given (no `sat_elements.ron`) stays at its window-edge position there instead.
//!
//! The ephemeris caches nothing across epochs; uni-sim caches per tick.

use std::collections::HashMap;

use glam::DVec3;
use uni_core::Epoch;

use crate::NaifId;
use crate::bodies::by_naif;
use crate::galilean::{Galilean, jovicentric_icrs};
use crate::sat_elements::{SatElements, planetocentric_icrs};
use crate::saturn_moons::{SaturnMoon, saturnicentric_icrs};
use crate::standish::{Planet, Validity, heliocentric_state_icrs, validity};

/// Half-width of the Meeus moon theories' window [Julian centuries]: ±3000 yr from J2000.
/// Their arguments are polynomials in T up to T⁴; outside it the Ephemeris uses each moon's
/// mean-element orbit and raises [`EphemFlag::MoonTheoryFallback`].
pub const MOON_THEORY_WINDOW_CY: f64 = 30.0;

/// `t` moved to the nearer edge of the moon theories' window if it lies outside.
pub(crate) fn clamp_to_moon_theory_window(t: Epoch) -> Epoch {
    let tc = t.julian_centuries();
    if tc.abs() <= MOON_THEORY_WINDOW_CY {
        t
    } else {
        Epoch::from_days_since_j2000(MOON_THEORY_WINDOW_CY.copysign(tc) * 36_525.0)
    }
}

fn in_moon_theory_window(t: Epoch) -> bool {
    t.julian_centuries().abs() <= MOON_THEORY_WINDOW_CY
}

/// The theory that places a body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theory {
    SunSsb,
    Standish,
    MeeusPluto,
    EarthFromEmb,
    MeeusMoon,
    MeeusGalilean,
    MeeusSaturn,
    JplMeanElements,
}

/// One body the ephemeris places.
#[derive(Clone, Debug)]
pub struct EphemBody {
    pub naif: NaifId,
    pub name: String,
    /// Parent body (planets: the Sun, 10); `None` for the Sun.
    pub parent: Option<NaifId>,
    pub theory: Theory,
}

/// Conditions at an epoch that the HUD labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EphemFlag {
    PlanetsExtrapolated,
    PlutoExtrapolated,
    MoonTheoryFallback,
}

/// Every body of the M1 Solar System with its theory.
pub struct Ephemeris {
    bodies: Vec<EphemBody>,
    /// Index into `bodies` by NAIF id.
    index: HashMap<NaifId, usize>,
    /// Mean elements by NAIF id: the orbit of every `JplMeanElements` moon and the fall-back
    /// of the Meeus moons.
    sats: HashMap<NaifId, SatElements>,
}

/// The Standish bodies placed as planets, with their NAIF body ids.
const STANDISH_PLANETS: [(NaifId, Planet); 7] = [
    (199, Planet::Mercury),
    (299, Planet::Venus),
    (499, Planet::Mars),
    (599, Planet::Jupiter),
    (699, Planet::Saturn),
    (799, Planet::Uranus),
    (899, Planet::Neptune),
];

const GALILEANS: [(NaifId, Galilean); 4] = [
    (501, Galilean::Io),
    (502, Galilean::Europa),
    (503, Galilean::Ganymede),
    (504, Galilean::Callisto),
];

const SATURN_MOONS: [(NaifId, SaturnMoon); 8] = [
    (601, SaturnMoon::Mimas),
    (602, SaturnMoon::Enceladus),
    (603, SaturnMoon::Tethys),
    (604, SaturnMoon::Dione),
    (605, SaturnMoon::Rhea),
    (606, SaturnMoon::Titan),
    (607, SaturnMoon::Hyperion),
    (608, SaturnMoon::Iapetus),
];

const SUN: NaifId = 10;
const EARTH: NaifId = 399;
const MOON: NaifId = 301;
const PLUTO: NaifId = 999;
const CHARON: NaifId = 901;

/// GM [m³/s²] of a body in [`crate::bodies`]; 0 for one that is not there.
fn gm(naif: NaifId) -> f64 {
    by_naif(naif).map_or(0.0, |b| b.gm_m3s2)
}

impl Ephemeris {
    /// Builds the ephemeris from the moons' mean elements.
    ///
    /// The Sun, the planets, Pluto and the thirteen Meeus moons are always present; every other
    /// moon in `sats` is added under its planet. A moon whose planet is not one of the eight
    /// planets or Pluto is skipped; a repeated NAIF id keeps the last record.
    pub fn new(sats: Vec<SatElements>) -> Self {
        let name =
            |naif: NaifId| by_naif(naif).map_or_else(|| naif.to_string(), |b| b.name.to_string());
        let mut bodies = vec![EphemBody {
            naif: SUN,
            name: name(SUN),
            parent: None,
            theory: Theory::SunSsb,
        }];
        let planet = |naif: NaifId, theory: Theory| EphemBody {
            naif,
            name: name(naif),
            parent: Some(SUN),
            theory,
        };
        bodies.push(planet(199, Theory::Standish));
        bodies.push(planet(299, Theory::Standish));
        bodies.push(planet(EARTH, Theory::EarthFromEmb));
        for naif in [499, 599, 699, 799, 899] {
            bodies.push(planet(naif, Theory::Standish));
        }
        bodies.push(planet(PLUTO, Theory::MeeusPluto));
        let moon = |naif: NaifId, parent: NaifId, theory: Theory| EphemBody {
            naif,
            name: name(naif),
            parent: Some(parent),
            theory,
        };
        bodies.push(moon(MOON, EARTH, Theory::MeeusMoon));
        bodies.extend(
            GALILEANS
                .iter()
                .map(|&(id, _)| moon(id, 599, Theory::MeeusGalilean)),
        );
        bodies.extend(
            SATURN_MOONS
                .iter()
                .map(|&(id, _)| moon(id, 699, Theory::MeeusSaturn)),
        );

        let planets = [EARTH, 499, 599, 699, 799, 899, PLUTO];
        let mut by_id = HashMap::new();
        for el in sats {
            if planets.contains(&el.planet) {
                by_id.insert(el.naif, el);
            }
        }
        let mut extra: Vec<&SatElements> = by_id
            .values()
            .filter(|el| !bodies.iter().any(|b| b.naif == el.naif))
            .collect();
        extra.sort_by_key(|el| el.naif);
        for el in extra {
            bodies.push(EphemBody {
                naif: el.naif,
                name: el.name.clone(),
                parent: Some(el.planet),
                theory: Theory::JplMeanElements,
            });
        }
        let index = bodies
            .iter()
            .enumerate()
            .map(|(k, b)| (b.naif, k))
            .collect();
        Ephemeris {
            bodies,
            index,
            sats: by_id,
        }
    }

    pub fn bodies(&self) -> &[EphemBody] {
        &self.bodies
    }

    /// Position relative to the parent [m], ICRS axes; the parent of planets is the Sun (10).
    ///
    /// The Sun's position is relative to the SSB (the root). `None` for an unknown id.
    pub fn position_rel_parent(&self, naif: NaifId, t: Epoch) -> Option<DVec3> {
        let body = &self.bodies[*self.index.get(&naif)?];
        Some(match body.theory {
            Theory::SunSsb => crate::ssb::sun_from_ssb_icrs(t),
            Theory::Standish => {
                let &(_, p) = STANDISH_PLANETS.iter().find(|(id, _)| *id == naif)?;
                heliocentric_state_icrs(p, t).0
            }
            Theory::EarthFromEmb => {
                let emb = heliocentric_state_icrs(Planet::EmBary, t).0;
                let share = gm(MOON) / (gm(EARTH) + gm(MOON));
                emb - share * self.moon_geocentric(t)
            }
            Theory::MeeusPluto => {
                let barycentre = crate::pluto::heliocentric_state_icrs(t).0;
                let charon = self.position_rel_parent(CHARON, t).unwrap_or(DVec3::ZERO);
                let share = gm(CHARON) / (gm(PLUTO) + gm(CHARON));
                barycentre - share * charon
            }
            Theory::MeeusMoon => self.moon_geocentric(t),
            Theory::MeeusGalilean => {
                let &(_, m) = GALILEANS.iter().find(|(id, _)| *id == naif)?;
                self.meeus_or_mean(naif, t, |t| jovicentric_icrs(m, t))
            }
            Theory::MeeusSaturn => {
                let &(_, m) = SATURN_MOONS.iter().find(|(id, _)| *id == naif)?;
                self.meeus_or_mean(naif, t, |t| saturnicentric_icrs(m, t))
            }
            Theory::JplMeanElements => {
                let el = self.sats.get(&naif)?;
                planetocentric_icrs(el, gm(el.planet), t)
            }
        })
    }

    /// The Moon relative to the Earth: Meeus inside the window, else its mean elements.
    fn moon_geocentric(&self, t: Epoch) -> DVec3 {
        self.meeus_or_mean(MOON, t, crate::moon::geocentric_icrs)
    }

    /// `theory(t)` inside the window; outside it the moon's mean-element orbit, or the theory at
    /// the window edge when no elements were given.
    fn meeus_or_mean(&self, naif: NaifId, t: Epoch, theory: impl Fn(Epoch) -> DVec3) -> DVec3 {
        match self.sats.get(&naif) {
            Some(el) if !in_moon_theory_window(t) => planetocentric_icrs(el, gm(el.planet), t),
            _ => theory(t),
        }
    }

    /// Position relative to the SSB [m], ICRS axes: the positions relative to each parent
    /// summed up the chain (moon → planet → Sun → SSB).
    pub fn ssb_position(&self, naif: NaifId, t: Epoch) -> Option<DVec3> {
        let mut sum = DVec3::ZERO;
        let mut id = naif;
        // Chains are at most moon → planet → Sun; the bound guards against a malformed table.
        for _ in 0..4 {
            sum += self.position_rel_parent(id, t)?;
            match self.bodies[*self.index.get(&id)?].parent {
                Some(parent) => id = parent,
                None => return Some(sum),
            }
        }
        None
    }

    /// The labelled conditions at `t`: planets outside 3000 BC – AD 3000, Pluto outside
    /// 1885–2099, the Meeus moons outside ±3000 yr.
    pub fn flags(&self, t: Epoch) -> Vec<EphemFlag> {
        let mut flags = Vec::new();
        if validity(t) == Validity::Extrapolated {
            flags.push(EphemFlag::PlanetsExtrapolated);
        }
        if !crate::pluto::in_window(t) {
            flags.push(EphemFlag::PlutoExtrapolated);
        }
        if !in_moon_theory_window(t) {
            flags.push(EphemFlag::MoonTheoryFallback);
        }
        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodies::by_naif;
    use crate::sat_elements::test_support::jpl_sats;
    use crate::standish::{Planet, heliocentric_state_icrs};

    const YEAR_S: f64 = 365.25 * 86_400.0;

    fn eph() -> Ephemeris {
        Ephemeris::new(jpl_sats())
    }

    fn years(y: f64) -> Epoch {
        Epoch::J2000.add_seconds(y * YEAR_S)
    }

    /// ssb(body) = ssb(parent) + rel(body) down every chain; the Moon's parent is the Earth,
    /// Charon's is Pluto, and every chain ends at the Sun, whose parent is the root (SSB).
    #[test]
    fn ephemeris_parent_chain() {
        let e = eph();
        let t = years(26.9);
        let sun = e.ssb_position(10, t).expect("Sun");
        let earth = e.ssb_position(399, t).expect("Earth");
        let rel = e.position_rel_parent(399, t).expect("Earth rel");
        assert!(
            (earth - (sun + rel)).length() < 1e-3,
            "{earth} vs {}",
            sun + rel
        );
        assert!((sun - crate::ssb::sun_from_ssb_icrs(t)).length() < 1e-3);
        let parent = |id| {
            e.bodies()
                .iter()
                .find(|b| b.naif == id)
                .and_then(|b| b.parent)
        };
        assert_eq!(parent(301), Some(399));
        assert_eq!(parent(901), Some(999));
        assert_eq!(parent(10), None);
        assert!(e.bodies().len() > 400, "{} bodies", e.bodies().len());
        for body in e.bodies() {
            let mut id = body.naif;
            let mut steps = 0;
            while let Some(p) = parent(id) {
                id = p;
                steps += 1;
                assert!(steps < 5, "{} loops", body.name);
            }
            assert_eq!(id, 10, "{} does not reach the Sun", body.name);
            let ssb = e.ssb_position(body.naif, t).expect("position");
            let up = match body.parent {
                Some(p) => e.ssb_position(p, t).expect("parent"),
                None => DVec3::ZERO,
            };
            let rel = e.position_rel_parent(body.naif, t).expect("rel");
            assert!(
                ssb.is_finite() && (ssb - (up + rel)).length() < 1e-3,
                "{}",
                body.name
            );
        }
        assert_eq!(e.position_rel_parent(123_456, t), None);
        assert_eq!(e.ssb_position(123_456, t), None);
    }

    /// Theories: Earth from the EMB, the Moon, Galileans and Saturn's eight by Meeus, every other
    /// moon by JPL mean elements; Earth = EMB − (GM_Moon / (GM_Earth + GM_Moon))·r_Moon.
    #[test]
    fn earth_from_emb_and_theories() {
        let e = eph();
        let theory = |id| e.bodies().iter().find(|b| b.naif == id).map(|b| b.theory);
        assert_eq!(theory(10), Some(Theory::SunSsb));
        assert_eq!(theory(599), Some(Theory::Standish));
        assert_eq!(theory(399), Some(Theory::EarthFromEmb));
        assert_eq!(theory(999), Some(Theory::MeeusPluto));
        assert_eq!(theory(301), Some(Theory::MeeusMoon));
        assert_eq!(theory(503), Some(Theory::MeeusGalilean));
        assert_eq!(theory(606), Some(Theory::MeeusSaturn));
        assert_eq!(theory(401), Some(Theory::JplMeanElements));
        assert_eq!(e.bodies().iter().filter(|b| b.naif == 301).count(), 1);
        let t = years(26.9);
        let (emb, _) = heliocentric_state_icrs(Planet::EmBary, t);
        let gm_e = by_naif(399).expect("Earth").gm_m3s2;
        let gm_m = by_naif(301).expect("Moon").gm_m3s2;
        let earth = e.position_rel_parent(399, t).expect("Earth");
        let moon = e.position_rel_parent(301, t).expect("Moon");
        let bary = earth + gm_m / (gm_e + gm_m) * moon;
        assert!((bary - emb).length() < 1e-3, "{bary} vs {emb}");
        // The Earth sits 0.0123 of the Moon's distance from the EMB, opposite the Moon:
        // 4 300–5 000 km over the Moon's 356 000–407 000 km.
        let offset = (earth - emb).length() / 1e3;
        assert!((4_300.0..5_000.0).contains(&offset), "{offset} km");
        assert!((earth - emb).normalize().dot(moon.normalize()) < -0.999_999);
    }

    /// Pluto (999) = Pluto barycentre − (GM_Charon / (GM_Pluto + GM_Charon))·r_Charon:
    /// ≈ 2 100 km from the barycentre (more than Pluto's radius), opposite Charon.
    #[test]
    fn pluto_body_offset_from_barycentre() {
        let e = eph();
        for y in [-50.0, 0.0, 26.9, 50.0] {
            let t = years(y);
            let bary =
                crate::ssb::sun_from_ssb_icrs(t) + crate::pluto::heliocentric_state_icrs(t).0;
            let offset = e.ssb_position(999, t).expect("Pluto") - bary;
            let charon = e.position_rel_parent(901, t).expect("Charon");
            println!(
                "Pluto − barycentre at {y} yr: {:.1} km",
                offset.length() / 1e3
            );
            assert!(
                (offset.length() / 1e3 - 2_100.0).abs() < 210.0,
                "{y}: {offset}"
            );
            assert!(
                offset.normalize().dot(charon.normalize()) < -0.999_999,
                "{y}"
            );
        }
    }

    /// Review Focus 1: at J2000 ± 10⁶ yr the Meeus theories are replaced by the same moon's
    /// mean-element orbit: finite, the Moon 356 000 < Δ < 407 000 km, Io 4.1e5 < r < 4.3e5 km,
    /// and MoonTheoryFallback flagged (not flagged inside ±3000 yr).
    #[test]
    fn theories_fall_back_far_from_epoch() {
        let e = eph();
        for y in [-1e6, 1e6, -2e4, 3.5e3] {
            let t = years(y);
            let moon = e.position_rel_parent(301, t).expect("Moon").length() / 1e3;
            let io = e.position_rel_parent(501, t).expect("Io").length() / 1e3;
            assert!(moon > 356_000.0 && moon < 407_000.0, "{y}: Moon {moon} km");
            assert!(io > 4.1e5 && io < 4.3e5, "{y}: Io {io} km");
            assert!(e.flags(t).contains(&EphemFlag::MoonTheoryFallback), "{y}");
            for body in e.bodies() {
                let p = e.ssb_position(body.naif, t).expect("position");
                assert!(p.is_finite(), "{} at {y} yr", body.name);
            }
            for id in [
                301, 501, 502, 503, 504, 601, 602, 603, 604, 605, 606, 607, 608,
            ] {
                let r = e.position_rel_parent(id, t).expect("moon").length();
                let a = jpl_sats()
                    .iter()
                    .find(|s| s.naif == id)
                    .expect("elements")
                    .a_m;
                assert!((r / a - 1.0).abs() < 0.25, "{id} at {y} yr: r/a {}", r / a);
            }
        }
        for y in [-2_999.0, 0.0, 26.9, 2_999.0] {
            assert!(
                !e.flags(years(y)).contains(&EphemFlag::MoonTheoryFallback),
                "{y}"
            );
        }
    }

    /// Flags follow the theories' windows: planets outside 3000 BC – AD 3000, Pluto outside
    /// 1885–2099.
    #[test]
    fn flags_follow_windows() {
        let e = eph();
        assert!(e.flags(years(26.9)).is_empty());
        let f = e.flags(years(150.0));
        assert_eq!(f, vec![EphemFlag::PlutoExtrapolated]);
        let f = e.flags(years(5_000.0));
        assert!(f.contains(&EphemFlag::PlanetsExtrapolated));
        assert!(f.contains(&EphemFlag::PlutoExtrapolated));
        assert!(f.contains(&EphemFlag::MoonTheoryFallback));
    }

    /// 2026-11-20T12:00Z: the subsolar point from the Ephemeris and the WGCCRE Earth model
    /// against an independent formula: Meeus ch. 25's low-accuracy Sun (mean equinox of date)
    /// and Greenwich mean sidereal time (IAU 1982; Meeus 12.4). ≤ 0.3° (the review computed
    /// 0.087°; the WGCCRE Earth model alone is off by ≥ 150″, per NAIF's pck00011 notes).
    #[test]
    fn subsolar_point_matches_independent_formula() {
        let t = Epoch::parse_iso_utc("2026-11-20T12:00:00Z").expect("date");
        let e = eph();
        let sun = e.ssb_position(10, t).expect("Sun") - e.ssb_position(399, t).expect("Earth");
        let fixed = body_to_icrs_inverse(t) * sun;
        let lon = fixed.y.atan2(fixed.x).to_degrees();
        let lat = (fixed.z / fixed.length()).asin().to_degrees();

        // Meeus ch. 25 (low accuracy), T in Julian centuries TD.
        let jde = t.jd_tdb();
        let tc = (jde - 2_451_545.0) / 36_525.0;
        let l0 = 280.46646 + 36_000.769_83 * tc + 0.000_303_2 * tc * tc;
        let m = (357.529_11 + 35_999.050_29 * tc - 0.000_153_7 * tc * tc).to_radians();
        let c = (1.914_602 - 0.004_817 * tc - 0.000_014 * tc * tc) * m.sin()
            + (0.019_993 - 0.000_101 * tc) * (2.0 * m).sin()
            + 0.000_289 * (3.0 * m).sin();
        let lambda = (l0 + c).to_radians();
        let eps = (23.439_291_1 - 0.013_004_2 * tc).to_radians(); // Meeus 22.2, truncated
        let ra = (eps.cos() * lambda.sin()).atan2(lambda.cos()).to_degrees();
        let dec = (eps.sin() * lambda.sin()).asin().to_degrees();
        // GMST from UT (≈ UTC here; |UT1 − UTC| < 0.9 s is 0.004°), Meeus 12.4.
        let jd_ut = jde - 69.184 / 86_400.0;
        let tu = (jd_ut - 2_451_545.0) / 36_525.0;
        let gmst =
            280.460_618_37 + 360.985_647_366_29 * (jd_ut - 2_451_545.0) + 0.000_387_933 * tu * tu
                - tu * tu * tu / 38_710_000.0;
        let lon_ref = (ra - gmst).rem_euclid(360.0);
        let dlon = ((lon - lon_ref + 540.0).rem_euclid(360.0)) - 180.0;
        let dlat = lat - dec;
        println!(
            "subsolar: model ({lon:.3}°, {lat:.3}°), reference lon {lon_ref:.3}° lat {dec:.3}°: Δlon {dlon:.3}°, Δlat {dlat:.3}°"
        );
        assert!(dlon.abs() <= 0.3 && dlat.abs() <= 0.3, "{dlon} {dlat}");
    }

    fn body_to_icrs_inverse(t: Epoch) -> glam::DQuat {
        crate::rotation::body_to_icrs(399, t)
            .expect("Earth")
            .inverse()
    }

    /// One M1 bookmark that can show moons with Horizons spots, as the plan's table "M1 capture
    /// bookmarks" states it (docs/superpowers/plans/2026-11-22-m1-solar-system.md).
    struct PlanView {
        name: &'static str,
        /// The system's planet: Horizons' centre for the moons, and the pole that sets the
        /// camera's elevation and its "up".
        planet: NaifId,
        /// The body the camera is placed from and looks at: the planet, or a moon.
        anchor: NaifId,
        moons: &'static [NaifId],
        dist_km: f64,
        /// Camera elevations above the planet's equator [°]; empty: any (0.5° steps).
        elevations: &'static [f64],
        /// Sun–anchor–camera phase angles allowed [°].
        phase_deg: (f64, f64),
        fov_y_deg: f64,
        /// "Lit limb left": the Sun projects within 18° of the image's −x axis.
        lit_limb_left: bool,
    }

    /// The plan's geometry for each bookmark (table "M1 capture bookmarks"): `start` (#1),
    /// `mars-reference-angle` (#5), `jupiter-galileans` (#6), `jupiter-disk` (#7),
    /// `saturn-ring-plane` (#8: 4° above the lit face, the south one), `saturn-night-side` (#10)
    /// and `titan-backlit` (#14). The exact phases (60°, 150°) get the intent test's ±1 %, the
    /// "≈ 45°" ±5° and titan-backlit's 160° ±2°; upper bounds (< 10°, ≤ 1°) are scanned from 0°;
    /// "in its equatorial plane ±2°" in 0.5° steps; the other elevations are as stated.
    /// Distances are nominal. A moon in a planet's shadow still counts (conservative).
    const PLAN_VIEWS: [PlanView; 7] = [
        PlanView {
            name: "start",
            planet: 399,
            anchor: 399,
            moons: &[301],
            dist_km: 60_000.0,
            elevations: &[],
            phase_deg: (59.4, 60.6),
            fov_y_deg: 60.0,
            lit_limb_left: false,
        },
        PlanView {
            name: "mars-reference-angle",
            planet: 499,
            anchor: 499,
            moons: &[401, 402],
            dist_km: 8_210.0,
            elevations: &[],
            phase_deg: (40.0, 50.0),
            fov_y_deg: 60.0,
            lit_limb_left: true,
        },
        PlanView {
            name: "jupiter-galileans",
            planet: 599,
            anchor: 599,
            moons: &[501, 502, 503, 504],
            dist_km: 5.0e6,
            elevations: &[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            phase_deg: (0.0, 10.0),
            fov_y_deg: 30.0,
            lit_limb_left: false,
        },
        PlanView {
            name: "jupiter-disk",
            planet: 599,
            anchor: 599,
            moons: &[501, 502, 503, 504],
            dist_km: 3.0e6,
            elevations: &[],
            phase_deg: (0.0, 1.0),
            fov_y_deg: 3.5,
            lit_limb_left: false,
        },
        PlanView {
            name: "saturn-ring-plane",
            planet: 699,
            anchor: 699,
            moons: &[601, 602, 603, 604, 605, 606, 607, 608],
            dist_km: 1.5e6,
            elevations: &[-4.0],
            phase_deg: (0.0, 10.0),
            fov_y_deg: 20.0,
            lit_limb_left: false,
        },
        PlanView {
            name: "saturn-night-side",
            planet: 699,
            anchor: 699,
            moons: &[601, 602, 603, 604, 605, 606, 607, 608],
            dist_km: 1.5e6,
            elevations: &[25.0],
            phase_deg: (148.5, 151.5),
            fov_y_deg: 20.0,
            lit_limb_left: false,
        },
        PlanView {
            name: "titan-backlit",
            planet: 699,
            anchor: 606,
            moons: &[601, 602, 603, 604, 605, 607, 608],
            dist_km: 60_000.0,
            elevations: &[],
            phase_deg: (158.0, 162.0),
            fov_y_deg: 20.0,
            lit_limb_left: false,
        },
    ];

    /// Moons measured above 1 px at a plan view, with their bound (the measured worst + 10 %),
    /// for the lead's ruling (ruling 10; A2 report, fix round 2). Self-checking: each entry
    /// must still exceed 1 px. Measured worst (placements showing the moon, all of them ≥ 1 px
    /// unless stated):
    /// - `saturn-ring-plane`: Mimas 1.742 px (181 of 181), Dione 1.155 px (≥ 1 px at 142 of
    ///   181); no placement has every moon under 1 px.
    /// - `saturn-night-side`: Mimas 2.446 px (76 of 76); no placement has every moon under 1 px.
    /// - `titan-backlit`: Iapetus 5.569 px, in frame at 588 of 10 274 placements; the other 9 686
    ///   have every moon under 1 px.
    ///
    /// Mimas misses Horizons by 1 072 km along track at the spot (Meeus ch. 46, 0.34°), Dione
    /// by 469 km and Iapetus by 6 216 km (0.17 % of a): inside the brief's 2 % ceilings, but the
    /// cameras are close (1.5 × 10⁶ km at fov 20°: 1 px ≈ 400–600 km; 60 000 km from Titan).
    const OVER_ONE_PIXEL: [(&str, NaifId, f64); 4] = [
        ("saturn-ring-plane", 601, 1.92),
        ("saturn-ring-plane", 604, 1.27),
        ("saturn-night-side", 601, 2.69),
        ("titan-backlit", 608, 6.13),
    ];

    /// Whether the segment from `cam` to `p` passes within `r` of `c` (a disc in front of `p`).
    fn occulted(cam: DVec3, p: DVec3, c: DVec3, r: f64) -> bool {
        let d = p - cam;
        let s = (c - cam).dot(d) / d.length_squared();
        s > 0.0 && s < 1.0 && (cam + s * d - c).length() < r
    }

    /// Ruling 10 at the plan's bookmark geometry (A2 fix round 2, review r1 I2 and I3). At the
    /// 2026-11-20 00:00 UTC spot epoch (B9 Step 5) and every camera placement the plan's
    /// geometry allows (azimuth about the planet's pole in 0.1° steps from the Sun's; the stated
    /// elevations; the phase range), the angle between each moon's theory position and its
    /// Horizons spot as seen from the camera, in pixels of the 1920 × 1080 capture, for every
    /// moon inside the frame and not behind the planet or the anchor (polar-radius spheres, so
    /// fewer count as hidden). The camera looks at the anchor, "up" along the planet's pole
    /// (the error's size does not depend on the roll). Prints each moon's worst; fails at 1 px
    /// except the listed spots. B9's intent tests on `bookmarks/m1.ron` stay authoritative.
    #[test]
    fn bookmark_moon_errors_at_plan_geometry() {
        use crate::standish::golden::{epoch_of, horizons_spots, pos_m};
        const SPOT_JD: f64 = 2_461_364.500_8;
        let e = eph();
        let t = epoch_of(SPOT_JD);
        let polar_radius = |id: NaifId| by_naif(id).expect("body").radii_m.z;
        let mut failures = Vec::new();
        for view in &PLAN_VIEWS {
            let pole = crate::rotation::pole_icrs(view.planet, t).expect("pole");
            let rel = |id: NaifId| {
                if id == view.planet {
                    DVec3::ZERO
                } else {
                    e.position_rel_parent(id, t).expect("moon")
                }
            };
            let anchor = rel(view.anchor);
            let sun_from_planet = -e.position_rel_parent(view.planet, t).expect("planet");
            let sun = (sun_from_planet - anchor).normalize();
            let e1 = (sun - sun.dot(pole) * pole).normalize();
            let e2 = pole.cross(e1);
            let focal_px = 540.0 / (view.fov_y_deg.to_radians() / 2.0).tan();
            let moons: Vec<(NaifId, DVec3, DVec3)> = view
                .moons
                .iter()
                .map(|&id| {
                    let spot = horizons_spots(id, view.planet)
                        .into_iter()
                        .find(|s| (s.jd_tdb - SPOT_JD).abs() < 1e-6)
                        .expect("2026-11-20 spot");
                    (id, rel(id), pos_m(&spot))
                })
                .collect();
            let elevations: Vec<f64> = if view.elevations.is_empty() {
                (-180..=180).map(|k| f64::from(k) * 0.5).collect()
            } else {
                view.elevations.to_vec()
            };
            // Per moon: placements showing it, those at ≥ 1 px, and the worst (px, elevation,
            // azimuth, phase).
            let mut stats: Vec<(NaifId, u32, u32, [f64; 4])> =
                view.moons.iter().map(|&id| (id, 0, 0, [0.0; 4])).collect();
            let (mut placements, mut clean) = (0_u32, 0_u32);
            for &el in &elevations {
                for k in -1_800..1_800 {
                    let az = f64::from(k) / 10.0;
                    let (sin_a, cos_a) = az.to_radians().sin_cos();
                    let (sin_e, cos_e) = el.to_radians().sin_cos();
                    let dir = cos_e * (cos_a * e1 + sin_a * e2) + sin_e * pole;
                    let phase = dir.dot(sun).clamp(-1.0, 1.0).acos().to_degrees();
                    if phase < view.phase_deg.0 || phase > view.phase_deg.1 {
                        continue;
                    }
                    let fwd = -dir;
                    let up = (pole - pole.dot(fwd) * fwd).normalize();
                    let right = fwd.cross(up);
                    let sun_in_image = sun - sun.dot(fwd) * fwd;
                    if view.lit_limb_left && sun_in_image.dot(right) > -0.95 * sun_in_image.length()
                    {
                        continue;
                    }
                    placements += 1;
                    let cam = anchor + dir * view.dist_km * 1e3;
                    let project = |p: DVec3| {
                        let v = p - cam;
                        let z = v.dot(fwd);
                        (z, focal_px * v.dot(right) / z, focal_px * v.dot(up) / z)
                    };
                    let mut all_under = true;
                    for (&(_, model, truth), stat) in moons.iter().zip(stats.iter_mut()) {
                        let (z, x, y) = project(truth);
                        let hidden = occulted(cam, truth, DVec3::ZERO, polar_radius(view.planet))
                            || occulted(cam, truth, anchor, polar_radius(view.anchor));
                        if z <= 0.0 || x.abs() > 960.0 || y.abs() > 540.0 || hidden {
                            continue;
                        }
                        let (_, xm, ym) = project(model);
                        let px = (xm - x).hypot(ym - y);
                        stat.1 += 1;
                        if px >= 1.0 {
                            stat.2 += 1;
                            all_under = false;
                        }
                        if px > stat.3[0] {
                            stat.3 = [px, el, az, phase];
                        }
                    }
                    clean += u32::from(all_under);
                }
            }
            assert!(placements > 0, "{}: no placement fits the plan", view.name);
            println!(
                "{}: {placements} placements, {clean} with every moon in frame under 1 px",
                view.name
            );
            for &(id, shown, over, [px, el, az, phase]) in &stats {
                if shown == 0 {
                    println!("  {id}: never in frame");
                    continue;
                }
                println!(
                    "  {id}: in frame at {shown}, ≥ 1 px at {over}; worst {px:.3} px (elevation {el:+.1}°, azimuth {az:+.1}° from the Sun's, phase {phase:.2}°)"
                );
                let listed = OVER_ONE_PIXEL
                    .iter()
                    .find(|&&(name, naif, _)| name == view.name && naif == id);
                match listed {
                    Some(&(_, _, bound)) if px >= bound => {
                        failures.push(format!(
                            "{} {id}: {px:.3} px ≥ its bound {bound}",
                            view.name
                        ));
                    }
                    Some(_) if px <= 1.0 => failures.push(format!(
                        "{} {id} is now under 1 px: drop it from OVER_ONE_PIXEL",
                        view.name
                    )),
                    None if px >= 1.0 => failures.push(format!("{} {id}: {px:.3} px", view.name)),
                    _ => {}
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
