//! Pluto system barycentre (Horizons target 9) from Meeus, *Astronomical Algorithms* 2nd ed.,
//! ch. 37 (periodic terms, 1885–2099); a two-body conic outside that window (M1 Task A1,
//! review I1).
//!
//! The series (Meeus ch. 37, Table 37.A, 43 terms) gives heliocentric l, b, r referred to the
//! mean ecliptic and equinox of J2000. They are used as the Pluto system barycentre (ruling
//! round 2, review I1); Pluto itself is ≈ 2 100 km (0.06–0.1″ at 30–50 AU) from it, an offset
//! A2 adds from Charon's orbit. Meeus states 0.6″ in longitude and 0.2″ in
//! latitude over 1885–2099 (as quoted in SLALIB's sla_PLANET, note 5). Measured against DE441
//! the series holds that through 2026, drifts to 1.3″ by 2050 and 2.8″ by 2090, and its
//! distance is off by up to 43 000 km (A1 report, `pluto_vs_horizons`).
//!
//! Outside the window the state at the nearer window edge seeds a two-body conic about the
//! Sun with μ = GM☉ + GM(Pluto system) (DE440), propagated by [`Conic::state`]; A4 raises
//! `EphemFlag::PlutoExtrapolated` from [`in_window`].

use std::f64::consts::TAU;

use glam::DVec3;
use uni_core::{AU, DAY, Epoch};

use crate::bodies::{GM_SUN_DE440_M3S2, system_gm_m3s2};
use crate::frames::ecliptic_j2000_to_icrs;
use crate::kepler::Conic;

/// Validity window of Meeus ch. 37 [JD TDB]: 1885-01-01 – 2099-12-31.
pub const VALID_JD_TDB: (f64, f64) = (2409542.5, 2488068.5);

/// μ of the two-body fallback: GM☉ + GM of the Pluto system, both DE440.
const MU_SUN_PLUTO: f64 = GM_SUN_DE440_M3S2
    + match system_gm_m3s2(9) {
        Some(gm) => gm,
        None => panic!("bodies: no Pluto system GM"),
    };

/// One Julian century [s].
const JULIAN_CENTURY_S: f64 = 36_525.0 * DAY;

/// Meeus Table 37.A: argument multipliers i, j, k of J, S, P, then the sine and cosine
/// coefficients A, B of longitude and latitude [1e-6 °] and radius vector [1e-7 AU].
/// Transcribed from two independent published typings of the table (soniakeys/meeus
/// `pluto.go`, MIT; PyMeeus `Pluto.py`, LGPL), which agree on all 387 numbers, and checked
/// against Meeus's Example 37.a (`pluto_meeus_37a`).
#[rustfmt::skip]
const TERMS: [[i32; 9]; 43] = [
    [0, 0, 1, -19799805, 19850055, -5452852, -14974862, 66865439, 68951812],
    [0, 0, 2, 897144, -4954829, 3527812, 1672790, -11827535, -332538],
    [0, 0, 3, 611149, 1211027, -1050748, 327647, 1593179, -1438890],
    [0, 0, 4, -341243, -189585, 178690, -292153, -18444, 483220],
    [0, 0, 5, 129287, -34992, 18650, 100340, -65977, -85431],
    [0, 0, 6, -38164, 30893, -30697, -25823, 31174, -6032],
    [0, 1, -1, 20442, -9987, 4878, 11248, -5794, 22161],
    [0, 1, 0, -4063, -5071, 226, -64, 4601, 4032],
    [0, 1, 1, -6016, -3336, 2030, -836, -1729, 234],
    [0, 1, 2, -3956, 3039, 69, -604, -415, 702],
    [0, 1, 3, -667, 3572, -247, -567, 239, 723],
    [0, 2, -2, 1276, 501, -57, 1, 67, -67],
    [0, 2, -1, 1152, -917, -122, 175, 1034, -451],
    [0, 2, 0, 630, -1277, -49, -164, -129, 504],
    [1, -1, 0, 2571, -459, -197, 199, 480, -231],
    [1, -1, 1, 899, -1449, -25, 217, 2, -441],
    [1, 0, -3, -1016, 1043, 589, -248, -3359, 265],
    [1, 0, -2, -2343, -1012, -269, 711, 7856, -7832],
    [1, 0, -1, 7042, 788, 185, 193, 36, 45763],
    [1, 0, 0, 1199, -338, 315, 807, 8663, 8547],
    [1, 0, 1, 418, -67, -130, -43, -809, -769],
    [1, 0, 2, 120, -274, 5, 3, 263, -144],
    [1, 0, 3, -60, -159, 2, 17, -126, 32],
    [1, 0, 4, -82, -29, 2, 5, -35, -16],
    [1, 1, -3, -36, -29, 2, 3, -19, -4],
    [1, 1, -2, -40, 7, 3, 1, -15, 8],
    [1, 1, -1, -14, 22, 2, -1, -4, 12],
    [1, 1, 0, 4, 13, 1, -1, 5, 6],
    [1, 1, 1, 5, 2, 0, -1, 3, 1],
    [1, 1, 3, -1, 0, 0, 0, 6, -2],
    [2, 0, -6, 2, 0, 0, -2, 2, 2],
    [2, 0, -5, -4, 5, 2, 2, -2, -2],
    [2, 0, -4, 4, -7, -7, 0, 14, 13],
    [2, 0, -3, 14, 24, 10, -8, -63, 13],
    [2, 0, -2, -49, -34, -3, 20, 136, -236],
    [2, 0, -1, 163, -48, 6, 5, 273, 1065],
    [2, 0, 0, 9, -24, 14, 17, 251, 149],
    [2, 0, 1, -4, 1, -2, 0, -25, -9],
    [2, 0, 2, -3, 1, 0, 0, 9, -2],
    [2, 0, 3, 1, 3, 0, 0, -8, 7],
    [3, 0, -2, -3, -1, 0, 1, 2, -10],
    [3, 0, -1, 5, -3, 0, 0, 19, 35],
    [3, 0, 0, 0, 0, 1, 0, 10, 3],
];

/// Heliocentric ecliptic J2000 longitude l, latitude b [rad] and distance r [m]; `None`
/// outside [`VALID_JD_TDB`].
///
/// l is in [0, 2π).
pub fn heliocentric_ecliptic_j2000(t: Epoch) -> Option<(f64, f64, f64)> {
    if !in_window(t) {
        return None;
    }
    let ([l, b, r], _) = series(t.julian_centuries());
    Some((l.to_radians().rem_euclid(TAU), b.to_radians(), r * AU))
}

/// Heliocentric position [m] and velocity [m/s] in ICRS axes; two-body outside the window.
///
/// Inside the window the velocity is the analytic derivative of the series.
pub fn heliocentric_state_icrs(t: Epoch) -> (DVec3, DVec3) {
    if in_window(t) {
        return series_state_icrs(t);
    }
    let edge_jd = if t.jd_tdb() < VALID_JD_TDB.0 {
        VALID_JD_TDB.0
    } else {
        VALID_JD_TDB.1
    };
    let edge = Epoch::from_days_since_j2000(edge_jd - 2_451_545.0);
    let (r, v) = series_state_icrs(edge);
    Conic::from_state(r, v, MU_SUN_PLUTO, edge).state(t)
}

/// Whether `t` is inside [`VALID_JD_TDB`].
pub fn in_window(t: Epoch) -> bool {
    (VALID_JD_TDB.0..=VALID_JD_TDB.1).contains(&t.jd_tdb())
}

/// The series state in ICRS, at any epoch (the caller checks the window).
fn series_state_icrs(t: Epoch) -> (DVec3, DVec3) {
    let ([l, b, r], [l_rate, b_rate, r_rate]) = series(t.julian_centuries());
    let (sin_l, cos_l) = l.to_radians().sin_cos();
    let (sin_b, cos_b) = b.to_radians().sin_cos();
    let r = r * AU;
    let per_s = 1.0 / JULIAN_CENTURY_S;
    let (dl, db, dr) = (
        l_rate.to_radians() * per_s,
        b_rate.to_radians() * per_s,
        r_rate * AU * per_s,
    );
    let radial = DVec3::new(cos_b * cos_l, cos_b * sin_l, sin_b);
    let east = DVec3::new(-sin_l, cos_l, 0.0); // ∂r̂/∂l / cos b
    let north = DVec3::new(-sin_b * cos_l, -sin_b * sin_l, cos_b); // ∂r̂/∂b
    let pos = r * radial;
    let vel = dr * radial + r * cos_b * dl * east + r * db * north;
    (ecliptic_j2000_to_icrs(pos), ecliptic_j2000_to_icrs(vel))
}

/// Meeus (37.1) at `t_cy` Julian centuries from J2000: ([l °, b °, r AU], their rates per
/// century). J = 34.35° + 3034.9057°T, S = 50.08° + 1222.1138°T, P = 238.96° + 144.96°T;
/// l = 238.958116° + 144.96°T + Σl, b = −3.908239° + Σb, r = 40.7241346 AU + Σr, each Σ over
/// the terms A sin α + B cos α with α = iJ + jS + kP.
fn series(t_cy: f64) -> ([f64; 3], [f64; 3]) {
    const RATES: [f64; 3] = [3034.9057, 1222.1138, 144.96]; // J, S, P [°/cy]
    let angles = [
        34.35 + RATES[0] * t_cy,
        50.08 + RATES[1] * t_cy,
        238.96 + RATES[2] * t_cy,
    ];
    let mut sum = [0.0; 3];
    let mut rate = [0.0; 3];
    for term in &TERMS {
        let [i, j, k, coefficients @ ..] = term.map(f64::from);
        let alpha = (i * angles[0] + j * angles[1] + k * angles[2]).to_radians();
        let alpha_rate = (i * RATES[0] + j * RATES[1] + k * RATES[2]).to_radians(); // rad/cy
        let (sin_a, cos_a) = alpha.sin_cos();
        for q in 0..3 {
            let (a, b) = (coefficients[2 * q], coefficients[2 * q + 1]);
            sum[q] += a * sin_a + b * cos_a;
            rate[q] += (a * cos_a - b * sin_a) * alpha_rate;
        }
    }
    let scale = [1e-6, 1e-6, 1e-7];
    (
        [
            238.958116 + 144.96 * t_cy + sum[0] * scale[0],
            -3.908239 + sum[1] * scale[1],
            40.7241346 + sum[2] * scale[2],
        ],
        [
            144.96 + rate[0] * scale[0],
            rate[1] * scale[1],
            rate[2] * scale[2],
        ],
    )
}

#[cfg(test)]
mod tests {
    use uni_core::AU;

    use super::*;
    use crate::bodies::{GM_SUN_DE440_M3S2, system_gm_m3s2};
    use crate::standish::golden::{ecliptic_errors, epoch_of, horizons_spots, pos_m};

    /// Meeus, *Astronomical Algorithms* 2nd ed., Example 37.a: 1992 October 13.0 TD =
    /// JDE 2448908.5 gives l = 232.74071°, b = +14.58782°, R = 29.711111 AU (p. 266).
    #[test]
    fn pluto_meeus_37a() {
        let (l, b, r) = heliocentric_ecliptic_j2000(epoch_of(2_448_908.5)).expect("in window");
        assert!(
            (l.to_degrees() - 232.74071).abs() < 1e-5,
            "{}",
            l.to_degrees()
        );
        assert!(
            (b.to_degrees() - 14.58782).abs() < 1e-5,
            "{}",
            b.to_degrees()
        );
        assert!((r / AU - 29.711111).abs() < 1e-6, "{}", r / AU);
    }

    /// Against Horizons 9@10 (DE441). Meeus states 0.6″ in longitude and 0.2″ in latitude
    /// over 1885–2099 (as quoted by SLALIB, Wallace, sla_PLANET note 5) and no radius figure.
    /// Measured against DE441: within 0.6″/0.2″ through 2026; then 1.27″ at 2050 and 2.83″
    /// at 2090, so those two spots assert the measured value rounded up (raised with the
    /// lead). Radius: no published bound; measured worst 43 300 km (SLALIB measured 36 400 km
    /// RMS position error for this series against DE200 over 1960–2025).
    #[test]
    fn pluto_vs_horizons() {
        let jds = [
            2_415_020.5,
            2_433_282.5,
            2_451_545.0,
            2_461_364.5008,
            2_469_807.5,
            2_484_417.5,
        ];
        let spots = horizons_spots(9, 10);
        assert_eq!(spots.len(), jds.len());
        for s in spots {
            assert!(jds.contains(&s.jd_tdb), "unexpected spot JD {}", s.jd_tdb);
            let t = epoch_of(s.jd_tdb);
            assert!(in_window(t));
            let (r, _) = heliocentric_state_icrs(t);
            let [dl, db, dr] = ecliptic_errors(r, pos_m(&s));
            println!(
                "Pluto JD {:.4}: dλ {dl:+.3}″ dφ {db:+.3}″ dρ {dr:+.0} km",
                s.jd_tdb
            );
            // (λ, φ) bounds: Meeus's through 2026, the measured drift (rounded up) after it.
            let (bound_l, bound_b) = match s.jd_tdb {
                jd if jd <= 2_461_364.5008 => (0.6, 0.2),
                2_469_807.5 => (1.4, 0.35),
                _ => (3.1, 0.35),
            };
            assert!(dl.abs() <= bound_l, "JD {}: dλ {dl}″", s.jd_tdb);
            assert!(db.abs() <= bound_b, "JD {}: dφ {db}″", s.jd_tdb);
            assert!(dr.abs() <= 50_000.0, "JD {}: dρ {dr} km", s.jd_tdb);
        }
    }

    /// Review Focus 1: outside 1885–2099 Pluto follows the two-body conic from the nearer
    /// window edge (μ = GM☉ + GM of the Pluto system), finite and between 29 and 50 AU; the
    /// state is continuous across each edge.
    #[test]
    fn pluto_outside_window_is_two_body_and_flagged() {
        let mu = GM_SUN_DE440_M3S2 + system_gm_m3s2(9).expect("Pluto system GM");
        let energy = |r: DVec3, v: DVec3| v.length_squared() / 2.0 - mu / r.length();
        for (jd, edge_jd) in [(2_378_496.5, VALID_JD_TDB.0), (2_524_593.5, VALID_JD_TDB.1)] {
            let t = epoch_of(jd); // 1800-01-01, 2200-01-01
            assert!(!in_window(t));
            assert!(heliocentric_ecliptic_j2000(t).is_none());
            let (r, v) = heliocentric_state_icrs(t);
            assert!(r.is_finite() && v.is_finite());
            let d = r.length() / AU;
            assert!(d > 29.0 && d < 50.0, "JD {jd}: {d} AU");
            // Two-body: the energy and angular momentum of the edge state.
            let (re, ve) = heliocentric_state_icrs(epoch_of(edge_jd));
            assert!((energy(r, v) / energy(re, ve) - 1.0).abs() < 1e-10);
            assert!((r.cross(v) - re.cross(ve)).length() / re.cross(ve).length() < 1e-10);
        }
        // Continuity at each edge: 0.01 s either side (Pluto moves ≤ 61 m in 0.01 s).
        for (edge_jd, outward) in [(VALID_JD_TDB.0, -0.01), (VALID_JD_TDB.1, 0.01)] {
            let edge = epoch_of(edge_jd);
            let (inside, outside) = (edge.add_seconds(-outward), edge.add_seconds(outward));
            assert!(in_window(edge) && in_window(inside) && !in_window(outside));
            let (ri, vi) = heliocentric_state_icrs(inside);
            let (ro, vo) = heliocentric_state_icrs(outside);
            assert!(
                (ro - ri).length() < 1e3,
                "JD {edge_jd}: jump {} m",
                (ro - ri).length()
            );
            assert!(
                (vo - vi).length() < 1e-3,
                "JD {edge_jd}: Δv {} m/s",
                (vo - vi).length()
            );
        }
    }

    /// The analytic velocity of the series is the time derivative of its positions.
    #[test]
    fn velocity_matches_position_derivative() {
        for jd in [2_415_020.5, 2_448_908.5, 2_461_364.5, 2_484_417.5] {
            let t = epoch_of(jd);
            let (_, v) = heliocentric_state_icrs(t);
            let (r_minus, _) = heliocentric_state_icrs(t.add_seconds(-60.0));
            let (r_plus, _) = heliocentric_state_icrs(t.add_seconds(60.0));
            let rel = (v - (r_plus - r_minus) / 120.0).length() / v.length();
            assert!(rel < 1e-6, "JD {jd}: {rel:e}");
        }
    }
}
