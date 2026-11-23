//! Moons from JPL SSD mean elements on the local Laplace plane
//! (https://ssd.jpl.nasa.gov/sats/elem/) (M1 Task A2). JPL states these are not for precise
//! ephemerides; the info card labels them.
//!
//! JPL's model is "the elements of a precessing ellipse which has been fit in a least squares
//! sense to the numerically integrated orbit" (page read 2026-11-22). Each moon's elements are
//! referred to the plane with pole (`laplace_ra_deg`, `laplace_dec_deg`) (its Laplace plane,
//! the J2000 ecliptic or the planet's equator: A9's bake writes the right pole for each), and
//! the node is "measured from the node of the reference plane on the ICRF equator". So the
//! plane's frame is R_z(α + 90°)·R_x(90° − δ) (x̂ at that node, at RA α + 90°), and the orbit
//! within it is R_z(Ω)·R_x(i)·R_z(ω) ([`perifocal_to_reference`]).
//!
//! Rates. JPL gives the sidereal period P and the precession periods of the argument of
//! periapsis and of the node, without signs. Used here:
//! - ω̇ = +360°/P_apsis (apsides advance);
//! - Ω̇ = ∓360°/P_node: the node regresses on a prograde orbit (i < 90°) and advances on a
//!   retrograde one, as an oblate planet or the Sun drives it (the J2 rate ∝ −cos i);
//! - the argument of latitude u = ω + M advances at n_sid + 360°/P_node, because the sidereal
//!   rate is the rate of the in-plane angle Ω ± u along the motion; so Ṁ = n_sid + |Ω̇| − ω̇.
//!   Check: the Moon's JPL values (P = 27.322 d, 5.997 yr, 18.600 yr) give Ṁ =
//!   13.064 832°/d against Meeus's anomalistic 13.064 993°/d (47.4), within P's rounding
//!   (±0.000 24°/d for P's last digit).
//!
//! Rate refinements (A2 decisions; measured against Horizons in the A2 report):
//! - Deimos and Triton: JPL's P (1.2625 d, 5.8792 d) is 0.0134°/d and 0.0244°/d off their
//!   mean longitude rates, so they drift 131° and 240° by 2026-11-20. Both are
//!   synchronous, so their sidereal rate is taken as the WGCCRE 2015 prime-meridian rate |Ẇ|
//!   ([`crate::rotation`]): a synchronous moon's prime meridian turns with its mean longitude.
//! - Triton: the page lists P_apsis = 675.2 yr and P_node = 299.4 yr, but its orbit is circular
//!   (e = 0, so an apsidal period means nothing), and Horizons' node advances 0.535°/yr in this
//!   frame (a 673-yr period; WGCCRE's N7 gives 688 yr). The two columns are evidently swapped,
//!   so Triton's node period is 675.2 yr.
//! - Phobos: the page's periods (0.3187 d, 1.1 yr, 2.3 yr) leave its mean anomaly, periapsis
//!   and node undetermined after a few decades. Its rates come instead from Jacobson 2010
//!   (AJ 139, 668) as the WGCCRE 2015 report with its 2019 correction tabulates them
//!   (pck00011): the mean anomaly advances as the phase angle M5 = 189.632 715 60° +
//!   41 215 158.184 200 50°·T + 12.711 923 222°·T² (T in Julian centuries TDB), the node regresses
//!   with M1 at 15 917.108 186 95°/cy, and the mean longitude advances at |Ẇ| = 1 128.844 759 28°/d,
//!   so ω̇ = λ̇ − Ṁ − Ω̇ (periods 1.1316 yr and 2.2617 yr).
//!
//! Phobos's tidal secular acceleration (ruling round 2, review I10) is that T² term of M5:
//! 12.711 923 222°/cy² = 1.271 × 10⁻³ °/yr² as the coefficient of t², i.e. s = ½ṅ, not ṅ.
//! Brozović, Jacobson & Park 2025 (AJ 170, 1; MAR099, the ephemeris Horizons serves) name the
//! same quantity "s = ½ṅ" and fit 1.258 ± 0.058 × 10⁻³ °/yr². Without it Phobos misses Horizons
//! by 519 km at 1950 (3.2° along track); with it, by ≤ 21 km over 1950–2050. T is clamped to the
//! planets' window (3000 BC – AD 3000), as for W's quadratic term in [`crate::rotation`], so the
//! face Phobos turns to Mars stays consistent at any epoch.

use glam::{DMat3, DVec3};
use uni_core::Epoch;

use crate::NaifId;
use crate::kepler::{elliptic_perifocal, perifocal_to_reference};
use crate::standish::{TABLE2_T_MAX, TABLE2_T_MIN};

/// One moon's mean elements, in SI-friendly units (uni-sim maps `uni_data::SatRecord` to this).
#[derive(Clone, Debug, PartialEq)]
pub struct SatElements {
    pub naif: NaifId,
    pub name: String,
    /// NAIF id of the planet (body centre) it orbits.
    pub planet: NaifId,
    /// Semi-major axis [m].
    pub a_m: f64,
    pub e: f64,
    /// Argument of periapsis [deg].
    pub argp_deg: f64,
    /// Mean anomaly at `epoch` [deg].
    pub m_deg: f64,
    /// Inclination to the Laplace plane [deg].
    pub i_deg: f64,
    /// Ascending node on the Laplace plane [deg].
    pub node_deg: f64,
    /// Mean motion [deg/day].
    pub n_deg_per_day: f64,
    /// Apsidal precession period [yr].
    pub p_apsides_yr: f64,
    /// Nodal precession period [yr].
    pub p_node_yr: f64,
    pub epoch: Epoch,
    /// Laplace-plane pole, ICRS right ascension [deg].
    pub laplace_ra_deg: f64,
    /// Laplace-plane pole, ICRS declination [deg].
    pub laplace_dec_deg: f64,
}

/// One day [s], one Julian year and one Julian century [days].
const DAY_S: f64 = 86_400.0;
const JULIAN_YEAR_DAYS: f64 = 365.25;
const DAYS_PER_CENTURY: f64 = 36_525.0;

const PHOBOS: NaifId = 401;
const DEIMOS: NaifId = 402;
const TRITON: NaifId = 801;

/// WGCCRE 2015 |Ẇ| [°/day] of Deimos and Triton, used as their sidereal rates (module doc).
const DEIMOS_SIDEREAL_RATE: f64 = 285.161_888_99;
const TRITON_SIDEREAL_RATE: f64 = 61.257_263_7;
/// Triton's node period [yr]: the page's P_apsis column (module doc).
const TRITON_NODE_PERIOD_YR: f64 = 675.2;

/// Phobos (module doc): M5's rate [°/cy] and T² coefficient [°/cy²], M1's rate [°/cy], |Ẇ| [°/d].
const PHOBOS_M_RATE_PER_CY: f64 = 41_215_158.184_200_50;
const PHOBOS_M_ACCEL_PER_CY2: f64 = 12.711_923_222;
const PHOBOS_NODE_RATE_PER_CY: f64 = -15_917.108_186_95;
const PHOBOS_LONGITUDE_RATE: f64 = 1_128.844_759_28;

/// Mean anomaly, argument of periapsis and node at `t` [°], unreduced.
struct MeanAngles {
    m: f64,
    argp: f64,
    node: f64,
}

/// The mean-element rates [°/day] and the mean anomaly's T² coefficient [°/cy²].
struct Rates {
    m: f64,
    argp: f64,
    node: f64,
    m_per_cy2: f64,
}

/// 360°/P per day for a period `p_yr` in Julian years; 0 for "no precession" (∞) and for a
/// period that is not positive (A9's loader rejects those; kept finite here regardless).
fn precession_rate_deg_per_day(p_yr: f64) -> f64 {
    if p_yr.is_finite() && p_yr > 0.0 {
        360.0 / (p_yr * JULIAN_YEAR_DAYS)
    } else {
        0.0
    }
}

/// The rates of `el` (module doc): JPL's, with the Phobos, Deimos and Triton refinements.
/// `gm` [m³/s²] gives the sidereal rate by Kepler's third law only when the elements carry no
/// usable n (not finite and positive).
fn rates(el: &SatElements, gm: f64) -> Rates {
    if el.naif == PHOBOS {
        let m = PHOBOS_M_RATE_PER_CY / DAYS_PER_CENTURY;
        let node = PHOBOS_NODE_RATE_PER_CY / DAYS_PER_CENTURY;
        return Rates {
            m,
            argp: PHOBOS_LONGITUDE_RATE - m - node,
            node,
            m_per_cy2: PHOBOS_M_ACCEL_PER_CY2,
        };
    }
    let n_sid = match el.naif {
        DEIMOS => DEIMOS_SIDEREAL_RATE,
        TRITON => TRITON_SIDEREAL_RATE,
        _ if el.n_deg_per_day.is_finite() && el.n_deg_per_day > 0.0 => el.n_deg_per_day,
        _ => ((gm / el.a_m).sqrt() / el.a_m * DAY_S).to_degrees(),
    };
    let p_node = if el.naif == TRITON {
        TRITON_NODE_PERIOD_YR
    } else {
        el.p_node_yr
    };
    let argp = precession_rate_deg_per_day(el.p_apsides_yr);
    let node_speed = precession_rate_deg_per_day(p_node);
    // The node regresses on a prograde orbit and advances on a retrograde one.
    let node = if el.i_deg.to_radians().cos() < 0.0 {
        node_speed
    } else {
        -node_speed
    };
    Rates {
        m: n_sid + node_speed - argp,
        argp,
        node,
        m_per_cy2: 0.0,
    }
}

/// M, ω, Ω [°] at `t`: linear in time from the epoch, plus the mean anomaly's T² term with T
/// clamped to the planets' window (module doc).
fn mean_angles(el: &SatElements, r: &Rates, t: Epoch) -> MeanAngles {
    let days = t.seconds_since(el.epoch) / DAY_S;
    let mut m = el.m_deg + r.m * days;
    if r.m_per_cy2 != 0.0 {
        let cy =
            t.julian_centuries().clamp(TABLE2_T_MIN, TABLE2_T_MAX) - el.epoch.julian_centuries();
        m += r.m_per_cy2 * cy * cy;
    }
    MeanAngles {
        m,
        argp: el.argp_deg + r.argp * days,
        node: el.node_deg + r.node * days,
    }
}

/// The mean angles at `t` [°] (tests).
#[cfg(test)]
fn mean_angles_deg(el: &SatElements, t: Epoch) -> MeanAngles {
    mean_angles(el, &rates(el, 0.0), t)
}

/// The mean anomaly without the T² term [°] (tests).
#[cfg(test)]
fn linear_mean_anomaly_deg(el: &SatElements, t: Epoch) -> f64 {
    el.m_deg + rates(el, 0.0).m * t.seconds_since(el.epoch) / DAY_S
}

/// The reference plane's frame in ICRS: R_z(α + 90°)·R_x(90° − δ) (module doc).
fn laplace_to_icrs(el: &SatElements) -> DMat3 {
    DMat3::from_rotation_z((el.laplace_ra_deg + 90.0).to_radians())
        * DMat3::from_rotation_x((90.0 - el.laplace_dec_deg).to_radians())
}

/// Position relative to the planet's centre [m], ICRS axes; `gm_planet` [m³/s²].
///
/// `gm_planet` gives the mean motion by Kepler's third law only when the elements carry no
/// usable one (n not finite and positive); JPL's n otherwise, or a refined rate (module doc).
/// Finite for any finite epoch: the angles are reduced mod 360° before use.
pub fn planetocentric_icrs(el: &SatElements, gm_planet: f64, t: Epoch) -> DVec3 {
    let angles = mean_angles(el, &rates(el, gm_planet), t);
    let rad = |deg: f64| deg.rem_euclid(360.0).to_radians();
    // The velocity half of the perifocal state is unused; any positive n will do for it.
    let (pos, _) = elliptic_perifocal(el.a_m, el.e, rad(angles.m), 1.0);
    laplace_to_icrs(el)
        * perifocal_to_reference(el.i_deg.to_radians(), rad(angles.node), rad(angles.argp))
        * pos
}

/// Test support shared with the `saturn_moons` and `ephemeris` tests: JPL's mean elements from
/// `data/baked/sat_elements.ron` (A9), mapped the way uni-sim maps them (km → m, JD → Epoch).
#[cfg(test)]
pub(crate) mod test_support {
    use uni_core::Epoch;

    use super::SatElements;

    /// Every moon in `sat_elements.ron`. A missing or unreadable file fails the test.
    pub(crate) fn jpl_sats() -> Vec<SatElements> {
        let path = uni_data::data_dir().join("baked").join("sat_elements.ron");
        let records = uni_data::load_sat_elements(&path)
            .unwrap_or_else(|e| panic!("data file missing: {} ({e})", path.display()));
        records
            .into_iter()
            .map(|r| SatElements {
                naif: r.naif,
                name: r.name,
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
                epoch: Epoch::from_days_since_j2000(r.epoch_jd_tdb - 2_451_545.0),
                laplace_ra_deg: r.laplace_ra_deg,
                laplace_dec_deg: r.laplace_dec_deg,
            })
            .collect()
    }

    /// The moon with NAIF id `naif`.
    pub(crate) fn jpl_sat(naif: i32) -> SatElements {
        jpl_sats()
            .into_iter()
            .find(|s| s.naif == naif)
            .unwrap_or_else(|| panic!("moon {naif} not in sat_elements.ron"))
    }
}

#[cfg(test)]
mod tests {
    use glam::DMat3;

    use super::test_support::jpl_sat;
    use super::*;
    use crate::bodies::by_naif;
    use crate::frames::ecliptic_j2000_to_icrs;
    use crate::standish::golden::{epoch_of, horizons_spots, pos_m};

    fn gm(naif: NaifId) -> f64 {
        by_naif(naif).map_or(0.0, |b| b.gm_m3s2)
    }

    /// Spots over the brief's 5 % ceiling, asserted at their measured error + 10 % and listed
    /// in the A2 report for the lead (ruling 10). Triton: JPL's mean anomaly at the epoch is
    /// 4.4° from Horizons' (nep098) argument of latitude there (58.59° against 63.0°), a constant
    /// 28 000 km along track. Self-checking: each must still exceed 5 % of a.
    const OVER_CEILING: [(i32, f64, f64); 4] = [
        (801, 2_433_282.5, 31_100.0),
        (801, 2_451_545.0, 31_200.0),
        (801, 2_461_364.5008, 31_300.0),
        (801, 2_469_807.5, 31_200.0),
    ];

    /// Horizons 401, 402 (mar099), 801 (nep098), 703, 704 (ura184) and 901 (plu060), each about
    /// its planet's centre, at 1950, 2000, 2026-11-20 and 2050: prints the measured worst error
    /// per moon in km and as a fraction of a; ceiling 5 % of a except the listed spots. Phobos
    /// includes its tidal secular acceleration.
    #[test]
    fn mean_element_moons_vs_horizons() {
        for id in [401, 402, 801, 703, 704, 901] {
            let el = jpl_sat(id);
            let a_km = el.a_m / 1e3;
            let mut worst_km = 0.0_f64;
            for s in horizons_spots(id, el.planet) {
                let model = planetocentric_icrs(&el, gm(el.planet), epoch_of(s.jd_tdb));
                let miss_km = (model - pos_m(&s)).length() / 1e3;
                println!("{} JD {:.4}: |Δr| {miss_km:.1} km", el.name, s.jd_tdb);
                worst_km = worst_km.max(miss_km);
                let listed = OVER_CEILING
                    .iter()
                    .find(|&&(i, jd, _)| i == id && jd == s.jd_tdb);
                match listed {
                    Some(&(_, _, bound)) => {
                        assert!(miss_km < bound, "{} JD {}: {miss_km} km", el.name, s.jd_tdb);
                        assert!(
                            miss_km > 0.05 * a_km,
                            "{} JD {} is now under the ceiling: drop it from OVER_CEILING",
                            el.name,
                            s.jd_tdb
                        );
                    }
                    None => assert!(
                        miss_km < 0.05 * a_km,
                        "{} JD {}: {miss_km} km",
                        el.name,
                        s.jd_tdb
                    ),
                }
            }
            println!(
                "{} worst: {worst_km:.1} km = {:.4} of a (ceiling 5 % = {:.0} km)",
                el.name,
                worst_km / a_km,
                0.05 * a_km
            );
        }
    }

    /// A moon on the ecliptic "Laplace plane" (pole RA 270°, Dec 90° − ε₀) with node 0 at its
    /// epoch is placed exactly as the same elements rotated by ε₀ from the J2000 ecliptic: the
    /// plane's node on the ICRF equator is the equinox.
    #[test]
    fn ecliptic_pole_frame_is_ecliptic_j2000() {
        let mut el = jpl_sat(301);
        (el.p_apsides_yr, el.p_node_yr) = (f64::INFINITY, f64::INFINITY);
        let t = el.epoch;
        let model = planetocentric_icrs(&el, gm(399), t);
        let (i, node, argp) = (
            el.i_deg.to_radians(),
            el.node_deg.to_radians(),
            el.argp_deg.to_radians(),
        );
        let (pos, _) = crate::kepler::elliptic_perifocal(
            el.a_m,
            el.e,
            el.m_deg.to_radians(),
            el.n_deg_per_day.to_radians() / 86_400.0,
        );
        let expected = ecliptic_j2000_to_icrs(
            DMat3::from_rotation_z(node)
                * DMat3::from_rotation_x(i)
                * DMat3::from_rotation_z(argp)
                * pos,
        );
        assert!((model - expected).length() < 1e-3, "{model} vs {expected}");
    }

    /// Precession directions (module doc): the argument of periapsis advances, the node regresses
    /// on a prograde orbit and advances on a retrograde one, and the argument of latitude
    /// u = ω + M advances at the sidereal rate plus the nodal rate, |Ω̇|.
    #[test]
    fn precession_directions() {
        let mut el = jpl_sat(703);
        el.i_deg = 10.0;
        let year = el.epoch.add_seconds(365.25 * 86_400.0);
        let at0 = mean_angles_deg(&el, el.epoch);
        let at1 = mean_angles_deg(&el, year);
        let w_rate = 360.0 / el.p_apsides_yr;
        let node_rate = 360.0 / el.p_node_yr;
        assert!((at1.argp - at0.argp - w_rate).abs() < 1e-9);
        assert!((at1.node - at0.node + node_rate).abs() < 1e-9);
        el.i_deg = 170.0;
        let retro = mean_angles_deg(&el, year).node - mean_angles_deg(&el, el.epoch).node;
        assert!((retro - node_rate).abs() < 1e-9, "{retro}");
        // u̇ = n_sid + |Ω̇| on both, with JPL's n for a moon without a refined rate (Titania).
        let ti = jpl_sat(703);
        let day = ti.epoch.add_seconds(86_400.0);
        let (u0, u1) = (mean_angles_deg(&ti, ti.epoch), mean_angles_deg(&ti, day));
        let u_rate = (u1.argp + u1.m) - (u0.argp + u0.m);
        let expected = ti.n_deg_per_day + 360.0 / (ti.p_node_yr * 365.25);
        assert!((u_rate - expected).abs() < 1e-9, "{u_rate} vs {expected}");
    }

    /// Phobos's tidal secular acceleration: the mean anomaly gains s·t² with s = ½ṅ =
    /// 12.711 923 222°/cy² = 1.271 × 10⁻³ °/yr² (Jacobson 2010 via WGCCRE 2015 and its 2019
    /// correction), 3.18° at ±50 yr; no other moon has the term.
    #[test]
    fn phobos_secular_term_is_quadratic() {
        let phobos = jpl_sat(401);
        let deimos = jpl_sat(402);
        for years in [-50.0, 26.9, 50.0] {
            let dt_s = years * 365.25 * 86_400.0;
            for el in [&phobos, &deimos] {
                let t = el.epoch.add_seconds(dt_s);
                let quad = mean_angles_deg(el, t).m - linear_mean_anomaly_deg(el, t);
                let expected = if el.naif == 401 {
                    12.711_923_222 * (years / 100.0) * (years / 100.0)
                } else {
                    0.0
                };
                assert!(
                    (quad - expected).abs() < 1e-6,
                    "{} at {years} yr: {quad}",
                    el.name
                );
            }
        }
    }
}
