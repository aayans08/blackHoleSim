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
//! Rates. The page gives P, which it calls the "sidereal period" (A9's bake stores
//! n = 360°/P), and the precession periods of the argument of periapsis and of the node,
//! without signs. Used here:
//! - Ω̇ = ∓360°/P_node: the node regresses on a prograde orbit (i < 90°) and advances on a
//!   retrograde one, as an oblate planet or the Sun drives it (the J2 rate ∝ −cos i);
//! - ω̇ = +360°/P_apsis (apsides advance), except Io's and Europa's, which regress (below);
//! - λ̇, the sidereal (mean-longitude) rate, from the rules below; the argument of latitude
//!   u = ω + M advances at λ̇ + 360°/P_node, because the sidereal rate is the rate of the
//!   in-plane angle Ω ± u along the motion; so Ṁ = λ̇ + |Ω̇| − ω̇. Check: the Moon's JPL values
//!   (P = 27.322 d, 5.997 yr, 18.600 yr) give Ṁ = 13.064 832°/d against Meeus's anomalistic
//!   13.064 993°/d (47.4), within P's rounding (±0.000 24°/d for P's last digit).
//!
//! Which rate P is (A2 fix round 1, review I1). The rows mix two readings. Checked against
//! independent rates (E5's mean-longitude rates l̇, Meeus ch. 44; the prime-meridian rates |Ẇ|
//! of synchronous moons, WGCCRE 2015, Archinal et al. 2018 Table 2):
//! - P is the sidereal period, λ̇ = n, for the Moon, Mimas … Iapetus (SAT441), Ariel … Miranda
//!   and Charon: |n − |Ẇ|| ≤ 1.7 × 10⁻⁴ °/d, against 3 × 10⁻⁴ – 197 °/d under the other reading
//!   (Charon has no precession, so both readings agree). Also for the Trojans Helene, Telesto
//!   and Calypso: their n equals their host's (Dione's, Tethys's) to ≤ 1.0 × 10⁻⁴ °/d, as a
//!   co-orbital's long-term rate must (their WGCCRE |Ẇ| do not: 0.082 and 0.024 °/d off).
//! - P is the period of the fitted mean anomaly, n = Ṁ, for the Galileans, Amalthea and Thebe
//!   (JUP365) and Janus, Epimetheus, Atlas, Prometheus and Pandora (SAT415). Read that way the
//!   rows give λ̇ within 2 × 10⁻⁴ °/d of E5's l̇ (Io and Europa only with their periapses
//!   regressing) and within 0.064 °/d of |Ẇ|; read as λ̇ = n they are off by 0.0018–0.74 °/d
//!   (the Galileans) and 1.2–2.9 °/d (the inner moons). Their λ̇ is the independent rate itself:
//!   E5's l̇ for the Galileans ([`crate::galilean::E5_MEAN_LONGITUDE_RATES`]), which keeps the
//!   Laplace relation λ₁ − 3λ₂ + 2λ₃ = 180° in the fall-back beyond ±3000 yr; |Ẇ| for the inner
//!   moons ([`INNER_MOON_SIDEREAL_RATES`]), whose JPL precession periods have three significant
//!   digits, ±0.016°/d in n + ϖ̇.
//! - Io's and Europa's periapses regress at 0.7395°/d: their eccentricities are forced by the
//!   conjunctions in the Laplace resonance (E5's leading terms 0.47259° sin 2(l₁ − l₂) and
//!   1.06476° sin 2(l₂ − l₃), so ϖ₁ = 2l₂ − l₁, ϖ₂ = 2l₃ − l₂ and ϖ̇ = 2l̇₂ − l̇₁ = 2l̇₃ − l̇₂ =
//!   −0.7395°/d: a 1.333-yr period, Io's P_apsis).
//! - Every other row keeps the page's reading (P sidereal), unverified: the Uranian inner moons
//!   (URA184) and Neptune's (NEP098) disagree with their WGCCRE |Ẇ| by 0.006–2.9 °/d under
//!   either reading, and the remaining moons have no independent rate.
//!
//! Known data problem (A2 fix round 2; for the lead and A9): the SAT441 rows' epoch angles
//! do not match Horizons' sat441. With the page's angles as baked (checked against the page
//! 2026-11-23), the mean longitude in each row's Laplace frame is off by a near-constant amount
//! at all four spots 1950–2050: Enceladus +5°, Tethys −60°, Dione −151°, Rhea −157°, Titan
//! −150° to −163°, Hyperion −121° to −145°, Iapetus +139° to +147° (truth − model); Mimas
//! −22° to +50° (its 44° libration is not in the elements). The rates agree, and no single
//! epoch shift explains the offsets. Every other system's rows agree with Horizons to ≤ 0.25°
//! (Triton ≤ 3°), so the frame and angle conventions here are not the cause. Affected: Saturn's
//! eight beyond ±3000 yr, where their phase is anyway unknown after decades. Saturn's other
//! rows, which place their moons inside the window, are unchecked: none has a Horizons spot.
//!
//! Further refinements (A2 decisions; measured against Horizons in the A2 report):
//! - Deimos and Triton: JPL's P (1.2625 d, 5.8792 d) is 0.0134°/d and 0.0244°/d off their
//!   mean longitude rates, so they drift 131° and 240° by 2026-11-20. Both are synchronous, so
//!   their sidereal rate is their WGCCRE 2015 |Ẇ| ([`crate::rotation`]): a synchronous moon's
//!   prime meridian turns with its mean longitude.
//! - Triton: the page lists P_apsis = 675.2 yr and P_node = 299.4 yr, but its orbit is circular
//!   (e = 0, so an apsidal period means nothing), and Horizons' node advances 0.535°/yr in this
//!   frame (a 673-yr period; WGCCRE's N7 gives 688 yr). The two columns are evidently swapped,
//!   so Triton's node period is 675.2 yr.
//! - Triton's phase (fix round 1, review I2): the page's M = 63.0° is 4.4° ahead of Horizons
//!   (nep098) at every spot 1950–2050, 28 000 km along track. Triton's WGCCRE 2015 prime
//!   meridian faces Neptune (body +x points from Triton to Neptune: measured 2.7–2.8° from
//!   Horizons' direction at the same spots), so its epoch phase is taken from W: the argument
//!   of latitude of −x̂_body at the elements' epoch, 61.17°; with e = 0, M = u − ω. The worst
//!   error falls from 28 418 to 17 070 km (4.8 % of a).
//! - Phobos: the page's periods (0.3187 d, 1.1 yr, 2.3 yr) leave its mean anomaly, periapsis
//!   and node undetermined after a few decades. Its rates come instead from Jacobson 2010
//!   (AJ 139, 668) as the WGCCRE 2015 report with its 2019 correction tabulates them
//!   (pck00011): the mean anomaly advances as the phase angle M5 = 189.632 715 60° +
//!   41 215 158.184 200 50°·T + 12.711 923 222°·T² (T in Julian centuries TDB from J2000), the
//!   node regresses with M1 at 15 917.108 186 95°/cy, and the mean longitude advances at
//!   |Ẇ| = 1 128.844 759 28°/d, so ω̇ = λ̇ − Ṁ − Ω̇ (periods 1.1316 yr and 2.2617 yr).
//!
//! Phobos's tidal secular acceleration (ruling round 2, review I10) is that T² term of M5:
//! 12.711 923 222°/cy² = 1.271 × 10⁻³ °/yr² as the coefficient of t², i.e. s = ½ṅ, not ṅ.
//! Brozović, Jacobson & Park 2025 (AJ 170, 1; MAR099, the ephemeris Horizons serves) name the
//! same quantity "s = ½ṅ" and fit 1.258 ± 0.058 × 10⁻³ °/yr². Without it Phobos misses Horizons
//! by 519 km at 1950 (3.2° along track); with it, by ≤ 21 km over 1950–2050. T is measured from
//! J2000, as in M5, whatever the elements' epoch (M gains s·(T² − T₀²)), and is clamped to the
//! planets' window (3000 BC – AD 3000), as for W's quadratic term in [`crate::rotation`], so the
//! face Phobos turns to Mars stays consistent at any epoch.

use glam::{DMat3, DVec3};
use uni_core::Epoch;

use crate::NaifId;
use crate::galilean::E5_MEAN_LONGITUDE_RATES;
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

/// One day [s], one Julian year [days].
const DAY_S: f64 = uni_core::DAY;
const JULIAN_YEAR_DAYS: f64 = 365.25;
/// One Julian century [days].
const DAYS_PER_CENTURY: f64 = 36_525.0;

const PHOBOS: NaifId = 401;
const DEIMOS: NaifId = 402;
const IO: NaifId = 501;
const EUROPA: NaifId = 502;
const CALLISTO: NaifId = 504;
const TRITON: NaifId = 801;

/// Triton's node period [yr]: the page's P_apsis column (module doc).
const TRITON_NODE_PERIOD_YR: f64 = 675.2;

/// The inner moons whose tabulated P is the mean-anomaly period (module doc), with their
/// sidereal rates λ̇ = |Ẇ| [°/day] from WGCCRE 2015 (Archinal et al. 2018, Table 2; NAIF
/// pck00011 `BODYnnn_PM`, read 2026-11-23). They have no rotation model in [`crate::rotation`].
pub(crate) const INNER_MOON_SIDEREAL_RATES: [(NaifId, f64); 7] = [
    (505, 722.631_456_0), // Amalthea
    (514, 533.700_410_0), // Thebe
    (610, 518.235_987_6), // Janus
    (611, 518.490_723_9), // Epimetheus
    (615, 598.306_000_0), // Atlas
    (616, 587.289_000),   // Prometheus
    (617, 572.789_100_0), // Pandora
];

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

/// The sidereal (mean-longitude) rate λ̇ [°/day] of `el` (module doc): E5's l̇ for the
/// Galileans, WGCCRE |Ẇ| for Deimos, Triton and the inner moons whose P is anomalistic, else
/// JPL's n; Kepler's third law from `gm` [m³/s²] only when the row has no usable n.
fn sidereal_rate(el: &SatElements, gm: f64) -> f64 {
    let synchronous = |naif| crate::rotation::linear_w_rate_deg_per_day(naif).map(f64::abs);
    let independent = match el.naif {
        IO..=CALLISTO => Some(E5_MEAN_LONGITUDE_RATES[(el.naif - IO) as usize]),
        DEIMOS | TRITON => synchronous(el.naif),
        id => INNER_MOON_SIDEREAL_RATES
            .iter()
            .find(|&&(naif, _)| naif == id)
            .map(|&(_, rate)| rate),
    };
    match independent {
        Some(rate) => rate,
        None if el.n_deg_per_day.is_finite() && el.n_deg_per_day > 0.0 => el.n_deg_per_day,
        None => ((gm / el.a_m).sqrt() / el.a_m * DAY_S).to_degrees(),
    }
}

/// The rates of `el` (module doc). `gm` [m³/s²] gives the sidereal rate by Kepler's third law
/// only when the elements carry no usable n (not finite and positive).
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
    let lambda = sidereal_rate(el, gm);
    let p_node = if el.naif == TRITON {
        TRITON_NODE_PERIOD_YR
    } else {
        el.p_node_yr
    };
    // Io's and Europa's resonance-forced periapses regress (module doc).
    let apsides_sense = if matches!(el.naif, IO | EUROPA) {
        -1.0
    } else {
        1.0
    };
    let argp = apsides_sense * precession_rate_deg_per_day(el.p_apsides_yr);
    let node_speed = precession_rate_deg_per_day(p_node);
    // The node regresses on a prograde orbit and advances on a retrograde one.
    let node = if el.i_deg.to_radians().cos() < 0.0 {
        node_speed
    } else {
        -node_speed
    };
    Rates {
        m: lambda + node_speed - argp,
        argp,
        node,
        m_per_cy2: 0.0,
    }
}

/// The orbit plane's frame in ICRS at the node and inclination given [°]: x̂ at the ascending
/// node, ẑ the orbit normal.
fn orbit_plane_to_icrs(el: &SatElements, node_deg: f64) -> DMat3 {
    laplace_to_icrs(el)
        * DMat3::from_rotation_z(node_deg.to_radians())
        * DMat3::from_rotation_x(el.i_deg.to_radians())
}

/// The mean anomaly at the elements' epoch [°]: the page's, except Triton's, whose phase comes
/// from its WGCCRE prime meridian, which faces Neptune (module doc). Triton's e is 0, so its
/// mean anomaly is its argument of latitude less ω.
fn epoch_mean_anomaly_deg(el: &SatElements) -> f64 {
    if el.naif == TRITON
        && let Some(body) = crate::rotation::body_to_icrs(TRITON, el.epoch)
    {
        // Body +x points from Triton to Neptune, so Triton lies along −x̂ from Neptune.
        let toward_triton = -(body * DVec3::X);
        let in_plane = orbit_plane_to_icrs(el, el.node_deg).transpose() * toward_triton;
        return in_plane.y.atan2(in_plane.x).to_degrees() - el.argp_deg;
    }
    el.m_deg
}

/// M, ω, Ω [°] at `t`: linear in time from the epoch, plus the mean anomaly's T² term with T
/// from J2000 clamped to the planets' window (module doc).
fn mean_angles(el: &SatElements, r: &Rates, t: Epoch) -> MeanAngles {
    let days = t.seconds_since(el.epoch) / DAY_S;
    let mut m = epoch_mean_anomaly_deg(el) + r.m * days;
    if r.m_per_cy2 != 0.0 {
        let clamped = |t: Epoch| t.julian_centuries().clamp(TABLE2_T_MIN, TABLE2_T_MAX);
        let (tc, t0) = (clamped(t), clamped(el.epoch));
        m += r.m_per_cy2 * (tc * tc - t0 * t0);
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
    epoch_mean_anomaly_deg(el) + rates(el, 0.0).m * t.seconds_since(el.epoch) / DAY_S
}

/// The reference plane's frame in ICRS: R_z(α + 90°)·R_x(90° − δ) (module doc).
fn laplace_to_icrs(el: &SatElements) -> DMat3 {
    DMat3::from_rotation_z((el.laplace_ra_deg + 90.0).to_radians())
        * DMat3::from_rotation_x((90.0 - el.laplace_dec_deg).to_radians())
}

/// Position relative to the planet's centre [m], ICRS axes; `gm_planet` [m³/s²].
///
/// `gm_planet` gives the mean motion by Kepler's third law only when the elements carry no
/// usable one (n not finite and positive); JPL's n otherwise, or an independent rate (module
/// doc). Finite for any finite epoch: the angles are reduced mod 360° before use.
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
    use crate::ephemeris::gm;
    use crate::frames::ecliptic_j2000_to_icrs;
    use crate::standish::golden::{epoch_of, horizons_spots, pos_m};

    /// The mean longitude λ = Ω + ω + M [°] of a prograde orbit at `t`, unreduced.
    fn mean_longitude_deg(el: &SatElements, t: Epoch) -> f64 {
        let a = mean_angles_deg(el, t);
        a.node + a.argp + a.m
    }

    /// Horizons 401, 402 (mar099), 501–504 (jup365), 801 (nep098), 703, 704 (ura184) and 901
    /// (plu060), each about its planet's centre, at 1950, 2000, 2026-11-20 and 2050: prints the
    /// measured worst error per moon in km and as a fraction of a; ceiling 5 % of a. Phobos
    /// includes its tidal secular acceleration. The Galileans' mean elements are their fall-back
    /// beyond ±3000 yr (the Ephemeris uses E5 inside); they are tested here because their rows'
    /// P is anomalistic (review I1: read as sidereal, they missed by up to 1 964 000 km).
    #[test]
    fn mean_element_moons_vs_horizons() {
        for id in [401, 402, 501, 502, 503, 504, 801, 703, 704, 901] {
            let el = jpl_sat(id);
            let a_km = el.a_m / 1e3;
            let mut worst_km = 0.0_f64;
            for s in horizons_spots(id, el.planet) {
                let model = planetocentric_icrs(&el, gm(el.planet), epoch_of(s.jd_tdb));
                let miss_km = (model - pos_m(&s)).length() / 1e3;
                println!("{} JD {:.4}: |Δr| {miss_km:.1} km", el.name, s.jd_tdb);
                worst_km = worst_km.max(miss_km);
                assert!(
                    miss_km < 0.05 * a_km,
                    "{} JD {}: {miss_km} km",
                    el.name,
                    s.jd_tdb
                );
            }
            println!(
                "{} worst: {worst_km:.1} km = {:.4} of a (ceiling 5 % = {:.0} km)",
                el.name,
                worst_km / a_km,
                0.05 * a_km
            );
        }
    }

    /// Review I1: which rate each row's P is (module doc).
    /// - Anomalistic rows: JPL's n is the model's mean-anomaly rate Ṁ (to the rounding of the
    ///   page's precession periods), while the model's mean-longitude rate is the independent
    ///   one (E5's l̇, WGCCRE |Ẇ|), 0.0018–2.9°/d away from n.
    /// - Sidereal rows: JPL's n equals WGCCRE |Ẇ| to ≤ 1.7 × 10⁻⁴ °/d and is the model's λ̇.
    /// - Trojans: n equals the host's n.
    #[test]
    fn period_reading_per_row() {
        let day = |el: &SatElements| el.epoch.add_seconds(DAY_S);
        let galileans =
            [501, 502, 503, 504].map(|id| (id, E5_MEAN_LONGITUDE_RATES[id as usize - 501]));
        for (id, independent) in galileans.into_iter().chain(INNER_MOON_SIDEREAL_RATES) {
            let el = jpl_sat(id);
            let at0 = mean_angles_deg(&el, el.epoch);
            let at1 = mean_angles_deg(&el, day(&el));
            let m_rate = at1.m - at0.m;
            let lambda_rate = mean_longitude_deg(&el, day(&el)) - mean_longitude_deg(&el, el.epoch);
            println!(
                "{}: n {:.4}, model Ṁ {m_rate:.4}, λ̇ {lambda_rate:.4} (independent {independent:.4}) °/d",
                el.name, el.n_deg_per_day
            );
            assert!(
                (m_rate - el.n_deg_per_day).abs() < 0.065,
                "{}: Ṁ {m_rate}",
                el.name
            );
            assert!(
                (lambda_rate - independent).abs() < 1e-9,
                "{}: λ̇ {lambda_rate}",
                el.name
            );
            assert!(
                (el.n_deg_per_day - independent).abs() > 0.0017,
                "{}",
                el.name
            );
        }
        // Io's and Europa's periapses regress at 2l̇₂ − l̇₁ (E5's forced eccentricity).
        let [l1, l2, _, _] = E5_MEAN_LONGITUDE_RATES;
        for id in [501, 502] {
            let el = jpl_sat(id);
            let at0 = mean_angles_deg(&el, el.epoch);
            let at1 = mean_angles_deg(&el, day(&el));
            let varpi_rate = (at1.argp + at1.node) - (at0.argp + at0.node);
            assert!(
                (varpi_rate - (2.0 * l2 - l1)).abs() < 5e-4,
                "{}: ϖ̇ {varpi_rate}",
                el.name
            );
        }
        for id in [
            301, 601, 602, 603, 604, 605, 606, 608, 701, 702, 703, 704, 705, 901,
        ] {
            let el = jpl_sat(id);
            let w = crate::rotation::linear_w_rate_deg_per_day(id)
                .expect("model")
                .abs();
            assert!(
                el.i_deg < 90.0,
                "{}: λ = Ω + ω + M holds for prograde rows",
                el.name
            );
            let lambda_rate = mean_longitude_deg(&el, day(&el)) - mean_longitude_deg(&el, el.epoch);
            assert!(
                (el.n_deg_per_day - w).abs() < 1.7e-4,
                "{}: n − |Ẇ|",
                el.name
            );
            assert!(
                (lambda_rate - el.n_deg_per_day).abs() < 1e-9,
                "{}: λ̇",
                el.name
            );
        }
        for (trojan, host) in [(612, 604), (613, 603), (614, 603)] {
            let (t, h) = (jpl_sat(trojan), jpl_sat(host));
            assert!(
                (t.n_deg_per_day - h.n_deg_per_day).abs() < 1.05e-4,
                "{}",
                t.name
            );
        }
    }

    /// Review I1: beyond ±3000 yr the Galileans run on their mean elements, at E5's l̇, so the
    /// Laplace angle λ₁ − 3λ₂ + 2λ₃ stays at its epoch value (≈ 180°) for 10⁶ yr instead of
    /// circulating (with λ̇ = n it moved 2.2°/d).
    #[test]
    fn galilean_fallback_keeps_laplace_resonance() {
        let [io, europa, ganymede] = [501, 502, 503].map(jpl_sat);
        let laplace = |t: Epoch| {
            mean_longitude_deg(&io, t) - 3.0 * mean_longitude_deg(&europa, t)
                + 2.0 * mean_longitude_deg(&ganymede, t)
        };
        let at_epoch = laplace(io.epoch);
        println!(
            "Laplace angle at the epoch: {:.2}°",
            at_epoch.rem_euclid(360.0)
        );
        assert!(
            (at_epoch.rem_euclid(360.0) - 180.0).abs() < 1.0,
            "{at_epoch}"
        );
        for years in [-1e6, -3.5e3, 3.5e3, 1e6] {
            let t = io.epoch.add_seconds(years * JULIAN_YEAR_DAYS * DAY_S);
            let drift = (laplace(t) - at_epoch + 180.0).rem_euclid(360.0) - 180.0;
            println!("Laplace angle drift at {years} yr: {drift:.3}°");
            assert!(drift.abs() < 1.0, "{years} yr: {drift}°");
        }
    }

    /// Review I2: Triton's epoch phase comes from its WGCCRE prime meridian, which faces Neptune:
    /// at the elements' epoch the model places Triton along −x̂_body, in its orbit plane.
    #[test]
    fn triton_phase_from_prime_meridian() {
        let el = jpl_sat(801);
        let body = crate::rotation::body_to_icrs(801, el.epoch).expect("Triton model");
        let toward = -(body * DVec3::X);
        let model = planetocentric_icrs(&el, 0.0, el.epoch).normalize();
        let in_plane = orbit_plane_to_icrs(&el, el.node_deg).transpose() * toward;
        let expected =
            orbit_plane_to_icrs(&el, el.node_deg) * DVec3::new(in_plane.x, in_plane.y, 0.0);
        assert!(
            model.angle_between(expected.normalize()) < 1e-9,
            "{model} vs {expected}"
        );
        assert!(
            in_plane.z.abs() < 1e-3,
            "W meridian {} out of the orbit plane",
            in_plane.z
        );
        let m0 = epoch_mean_anomaly_deg(&el);
        println!("Triton M at the epoch: {m0:.3}° (page {:.1}°)", el.m_deg);
        assert!((m0 - 61.17).abs() < 0.01, "{m0}");
    }

    /// The independent rates in this module agree with [`crate::rotation`]'s models where both
    /// exist (Deimos, Triton), and the inner moons' WGCCRE |Ẇ| stay within 3°/d of JPL's n.
    #[test]
    fn independent_rates_are_consistent() {
        for id in [402, 801] {
            let el = jpl_sat(id);
            let w = crate::rotation::linear_w_rate_deg_per_day(id)
                .expect("model")
                .abs();
            assert_eq!(sidereal_rate(&el, 0.0), w, "{}", el.name);
        }
        for (id, rate) in INNER_MOON_SIDEREAL_RATES {
            let el = jpl_sat(id);
            assert_eq!(sidereal_rate(&el, 0.0), rate, "{}", el.name);
            assert!((el.n_deg_per_day - rate).abs() < 3.0, "{}", el.name);
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

    /// Phobos's tidal secular acceleration: the mean anomaly gains s·(T² − T₀²), T from J2000,
    /// with s = ½ṅ = 12.711 923 222°/cy² = 1.271 × 10⁻³ °/yr² (Jacobson 2010 via WGCCRE 2015 and
    /// its 2019 correction), 3.18° at ±50 yr; no other moon has the term. Review M3: an epoch
    /// other than J2000 shifts the term by −s·T₀², not into a spurious linear term.
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
        let mut shifted = phobos.clone();
        shifted.epoch = Epoch::J2000.add_seconds(25.0 * 365.25 * 86_400.0);
        let t = Epoch::J2000.add_seconds(75.0 * 365.25 * 86_400.0);
        let quad = mean_angles_deg(&shifted, t).m - linear_mean_anomaly_deg(&shifted, t);
        let expected = 12.711_923_222 * (0.75_f64.powi(2) - 0.25_f64.powi(2));
        assert!((quad - expected).abs() < 1e-6, "{quad} vs {expected}");
    }
}
