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
}
