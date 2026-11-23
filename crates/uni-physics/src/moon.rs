//! The Moon from Meeus, *Astronomical Algorithms* 2nd ed. (1998), ch. 47: the truncated
//! ELP-2000/82 of Chapront-Touzé & Chapront (1983) as Meeus gives it (M1 Task A2).
//!
//! Geometric geocentric λ, β, Δ referred to the mean ecliptic and equinox of date (no
//! nutation, no aberration, no light time), then the mean obliquity of date (Meeus 22.2) and
//! [`crate::frames::precession_to_j2000`] to ICRS. Measured against DE441 over 1950–2050
//! (`moon_vs_horizons`): ≤ 3.7″ in direction and ≤ 7.3 km in distance.
//!
//! Validity: the series' arguments are polynomials in T up to T⁴, so they are evaluated only
//! within ±3000 yr of J2000 ([`crate::ephemeris::MOON_THEORY_WINDOW_CY`]); called further out
//! these functions return the value at the nearer window edge (frozen, finite), and
//! [`crate::ephemeris::Ephemeris`] substitutes the Moon's JPL mean-element orbit instead.

use glam::DVec3;
use uni_core::Epoch;

use crate::ephemeris::clamp_to_moon_theory_window;
use crate::frames::ecliptic_of_date_to_icrs;

/// Meeus Table 47.A (pp. 339–340): multiples of D, M, M′, F, then the coefficient of
/// sin(argument) in Σl [10⁻⁶ °] and of cos(argument) in Σr [10⁻³ km]. Transcribed from two
/// independent published typings (soniakeys/meeus `moonposition.go`, MIT; PyMeeus `Moon.py`,
/// LGPL), which agree on all 360 numbers, and checked by Example 47.a (`moon_meeus_47a`).
#[rustfmt::skip]
const TABLE_47A: [[i32; 6]; 60] = [
    [ 0,  0,  1,  0,   6288774,  -20905355],
    [ 2,  0, -1,  0,   1274027,   -3699111],
    [ 2,  0,  0,  0,    658314,   -2955968],
    [ 0,  0,  2,  0,    213618,    -569925],
    [ 0,  1,  0,  0,   -185116,      48888],
    [ 0,  0,  0,  2,   -114332,      -3149],
    [ 2,  0, -2,  0,     58793,     246158],
    [ 2, -1, -1,  0,     57066,    -152138],
    [ 2,  0,  1,  0,     53322,    -170733],
    [ 2, -1,  0,  0,     45758,    -204586],
    [ 0,  1, -1,  0,    -40923,    -129620],
    [ 1,  0,  0,  0,    -34720,     108743],
    [ 0,  1,  1,  0,    -30383,     104755],
    [ 2,  0,  0, -2,     15327,      10321],
    [ 0,  0,  1,  2,    -12528,          0],
    [ 0,  0,  1, -2,     10980,      79661],
    [ 4,  0, -1,  0,     10675,     -34782],
    [ 0,  0,  3,  0,     10034,     -23210],
    [ 4,  0, -2,  0,      8548,     -21636],
    [ 2,  1, -1,  0,     -7888,      24208],
    [ 2,  1,  0,  0,     -6766,      30824],
    [ 1,  0, -1,  0,     -5163,      -8379],
    [ 1,  1,  0,  0,      4987,     -16675],
    [ 2, -1,  1,  0,      4036,     -12831],
    [ 2,  0,  2,  0,      3994,     -10445],
    [ 4,  0,  0,  0,      3861,     -11650],
    [ 2,  0, -3,  0,      3665,      14403],
    [ 0,  1, -2,  0,     -2689,      -7003],
    [ 2,  0, -1,  2,     -2602,          0],
    [ 2, -1, -2,  0,      2390,      10056],
    [ 1,  0,  1,  0,     -2348,       6322],
    [ 2, -2,  0,  0,      2236,      -9884],
    [ 0,  1,  2,  0,     -2120,       5751],
    [ 0,  2,  0,  0,     -2069,          0],
    [ 2, -2, -1,  0,      2048,      -4950],
    [ 2,  0,  1, -2,     -1773,       4130],
    [ 2,  0,  0,  2,     -1595,          0],
    [ 4, -1, -1,  0,      1215,      -3958],
    [ 0,  0,  2,  2,     -1110,          0],
    [ 3,  0, -1,  0,      -892,       3258],
    [ 2,  1,  1,  0,      -810,       2616],
    [ 4, -1, -2,  0,       759,      -1897],
    [ 0,  2, -1,  0,      -713,      -2117],
    [ 2,  2, -1,  0,      -700,       2354],
    [ 2,  1, -2,  0,       691,          0],
    [ 2, -1,  0, -2,       596,          0],
    [ 4,  0,  1,  0,       549,      -1423],
    [ 0,  0,  4,  0,       537,      -1117],
    [ 4, -1,  0,  0,       520,      -1571],
    [ 1,  0, -2,  0,      -487,      -1739],
    [ 2,  1,  0, -2,      -399,          0],
    [ 0,  0,  2, -2,      -381,      -4421],
    [ 1,  1,  1,  0,       351,          0],
    [ 3,  0, -2,  0,      -340,          0],
    [ 4,  0, -3,  0,       330,          0],
    [ 2, -1,  2,  0,       327,          0],
    [ 0,  2,  1,  0,      -323,       1165],
    [ 1,  1, -1,  0,       299,          0],
    [ 2,  0,  3,  0,       294,          0],
    [ 2,  0, -1, -2,         0,       8752],
];

/// Meeus Table 47.B (p. 341): multiples of D, M, M′, F and the coefficient of sin(argument) in
/// Σb [10⁻⁶ °]; same two typings, which agree on all 300 numbers.
#[rustfmt::skip]
const TABLE_47B: [[i32; 5]; 60] = [
    [ 0,  0,  0,  1,  5128122],
    [ 0,  0,  1,  1,   280602],
    [ 0,  0,  1, -1,   277693],
    [ 2,  0,  0, -1,   173237],
    [ 2,  0, -1,  1,    55413],
    [ 2,  0, -1, -1,    46271],
    [ 2,  0,  0,  1,    32573],
    [ 0,  0,  2,  1,    17198],
    [ 2,  0,  1, -1,     9266],
    [ 0,  0,  2, -1,     8822],
    [ 2, -1,  0, -1,     8216],
    [ 2,  0, -2, -1,     4324],
    [ 2,  0,  1,  1,     4200],
    [ 2,  1,  0, -1,    -3359],
    [ 2, -1, -1,  1,     2463],
    [ 2, -1,  0,  1,     2211],
    [ 2, -1, -1, -1,     2065],
    [ 0,  1, -1, -1,    -1870],
    [ 4,  0, -1, -1,     1828],
    [ 0,  1,  0,  1,    -1794],
    [ 0,  0,  0,  3,    -1749],
    [ 0,  1, -1,  1,    -1565],
    [ 1,  0,  0,  1,    -1491],
    [ 0,  1,  1,  1,    -1475],
    [ 0,  1,  1, -1,    -1410],
    [ 0,  1,  0, -1,    -1344],
    [ 1,  0,  0, -1,    -1335],
    [ 0,  0,  3,  1,     1107],
    [ 4,  0,  0, -1,     1021],
    [ 4,  0, -1,  1,      833],
    [ 0,  0,  1, -3,      777],
    [ 4,  0, -2,  1,      671],
    [ 2,  0,  0, -3,      607],
    [ 2,  0,  2, -1,      596],
    [ 2, -1,  1, -1,      491],
    [ 2,  0, -2,  1,     -451],
    [ 0,  0,  3, -1,      439],
    [ 2,  0,  2,  1,      422],
    [ 2,  0, -3, -1,      421],
    [ 2,  1, -1,  1,     -366],
    [ 2,  1,  0,  1,     -351],
    [ 4,  0,  0,  1,      331],
    [ 2, -1,  1,  1,      315],
    [ 2, -2,  0, -1,      302],
    [ 0,  0,  1,  3,     -283],
    [ 2,  1,  1, -1,     -229],
    [ 1,  1,  0, -1,      223],
    [ 1,  1,  0,  1,      223],
    [ 0,  1, -2, -1,     -220],
    [ 2,  1, -1, -1,     -220],
    [ 1,  0,  1,  1,     -185],
    [ 2, -1, -2, -1,      181],
    [ 0,  1,  2,  1,     -177],
    [ 4,  0, -2, -1,      176],
    [ 4, -1, -1, -1,      166],
    [ 1,  0,  1, -1,     -164],
    [ 4,  0,  1, -1,      132],
    [ 1,  0, -1, -1,     -119],
    [ 4, -1,  0, -1,      115],
    [ 2, -2,  0,  1,      107],
];

/// Horner evaluation of c₀ + c₁T + c₂T² + c₃T³ + c₄T⁴ [°], reduced to [0°, 360°).
fn poly_deg(t: f64, c: [f64; 5]) -> f64 {
    (c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])))).rem_euclid(360.0)
}

/// Geocentric ecliptic longitude λ, latitude β [rad] and distance Δ [m], mean equinox of date.
///
/// λ is in [0, 2π). Outside ±3000 yr the value at the nearer window edge (module doc).
pub fn ecliptic_of_date(t: Epoch) -> (f64, f64, f64) {
    let tc = clamp_to_moon_theory_window(t).julian_centuries();
    // Meeus (47.1)–(47.5): mean longitude L′, elongation D, the Sun's mean anomaly M, the
    // Moon's mean anomaly M′ and argument of latitude F [°]. M's T² coefficient is PyMeeus's
    // −0.0001536 (the Go typing has −0.0001535; 1e-7 °·T² either way).
    let lp = poly_deg(
        tc,
        [
            218.316_447_7,
            481_267.881_234_21,
            -0.001_578_6,
            1.0 / 538_841.0,
            -1.0 / 65_194_000.0,
        ],
    );
    let d = poly_deg(
        tc,
        [
            297.850_192_1,
            445_267.111_403_4,
            -0.001_881_9,
            1.0 / 545_868.0,
            -1.0 / 113_065_000.0,
        ],
    );
    let m = poly_deg(
        tc,
        [
            357.529_109_2,
            35_999.050_290_9,
            -0.000_153_6,
            1.0 / 24_490_000.0,
            0.0,
        ],
    );
    let mp = poly_deg(
        tc,
        [
            134.963_396_4,
            477_198.867_505_5,
            0.008_741_4,
            1.0 / 69_699.0,
            -1.0 / 14_712_000.0,
        ],
    );
    let f = poly_deg(
        tc,
        [
            93.272_095_0,
            483_202.017_523_3,
            -0.003_653_9,
            -1.0 / 3_526_000.0,
            1.0 / 863_310_000.0,
        ],
    );
    // Meeus ch. 47: A1 (Venus), A2 (Jupiter), A3 [°], and E (47.6), the decrease of the
    // eccentricity of the Earth's orbit, applied once per |multiple of M|.
    let a1 = (119.75 + 131.849 * tc).to_radians();
    let a2 = (53.09 + 479_264.290 * tc).to_radians();
    let a3 = (313.45 + 481_266.484 * tc).to_radians();
    let e = 1.0 - tc * (0.002_516 + 0.000_007_4 * tc);
    let (lp, d, m, mp, f) = (
        lp.to_radians(),
        d.to_radians(),
        m.to_radians(),
        mp.to_radians(),
        f.to_radians(),
    );
    let arg = |k: &[i32]| {
        f64::from(k[0]) * d + f64::from(k[1]) * m + f64::from(k[2]) * mp + f64::from(k[3]) * f
    };
    let e_factor = |k_m: i32| e.powi(k_m.abs());
    let (mut sum_l, mut sum_r) = (0.0, 0.0);
    for row in &TABLE_47A {
        let (s, c) = arg(row).sin_cos();
        let ef = e_factor(row[1]);
        sum_l += f64::from(row[4]) * ef * s;
        sum_r += f64::from(row[5]) * ef * c;
    }
    let mut sum_b = 0.0;
    for row in &TABLE_47B {
        sum_b += f64::from(row[4]) * e_factor(row[1]) * arg(row).sin();
    }
    // Meeus ch. 47: the additive terms (Venus, Jupiter, the Earth's flattening).
    sum_l += 3958.0 * a1.sin() + 1962.0 * (lp - f).sin() + 318.0 * a2.sin();
    sum_b += -2235.0 * lp.sin()
        + 382.0 * a3.sin()
        + 175.0 * (a1 - f).sin()
        + 175.0 * (a1 + f).sin()
        + 127.0 * (lp - mp).sin()
        - 115.0 * (lp + mp).sin();
    let lon = (lp + (sum_l * 1e-6).to_radians()).rem_euclid(std::f64::consts::TAU);
    let lat = (sum_b * 1e-6).to_radians();
    // Meeus ch. 47: Δ = 385 000.56 km + Σr.
    let dist_m = (385_000.56 + sum_r * 1e-3) * 1e3;
    (lon, lat, dist_m)
}

/// Geocentric position [m] in ICRS axes.
///
/// [`ecliptic_of_date`] rotated by the mean obliquity of date and precessed to J2000
/// ([`ecliptic_of_date_to_icrs`]), at the same (window-clamped) epoch.
pub fn geocentric_icrs(t: Epoch) -> DVec3 {
    let (lon, lat, dist) = ecliptic_of_date(t);
    ecliptic_of_date_to_icrs(lon, lat, dist, clamp_to_moon_theory_window(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standish::golden::{epoch_of, horizons_spots, pos_m};

    /// Meeus, *Astronomical Algorithms* 2nd ed., Example 47.a: 1992 April 12, 0h TD
    /// (JDE 2448724.5): λ = 133.162655°, β = −3.229126°, Δ = 368 409.7 km (geometric, mean
    /// equinox of date, before nutation).
    #[test]
    fn moon_meeus_47a() {
        let (lon, lat, dist) = ecliptic_of_date(epoch_of(2_448_724.5));
        assert!(
            (lon.to_degrees() - 133.162655).abs() < 1e-5,
            "λ {}",
            lon.to_degrees()
        );
        assert!(
            (lat.to_degrees() + 3.229126).abs() < 1e-5,
            "β {}",
            lat.to_degrees()
        );
        assert!((dist / 1e3 - 368_409.7).abs() < 0.5, "Δ {} km", dist / 1e3);
    }

    /// Horizons 301@399 (DE441) at 1950, 2000, 2026-11-20 and 2050: prints the measured
    /// direction and distance errors; ceiling 60″ and 60 km (ruling 10: the measured worst goes
    /// in the A2 report).
    #[test]
    fn moon_vs_horizons() {
        let (mut worst_arcsec, mut worst_km) = (0.0_f64, 0.0_f64);
        for s in horizons_spots(301, 399) {
            let model = geocentric_icrs(epoch_of(s.jd_tdb));
            let truth = pos_m(&s);
            let arcsec = model.angle_between(truth).to_degrees() * 3600.0;
            let d_km = (model.length() - truth.length()) / 1e3;
            let miss_km = (model - truth).length() / 1e3;
            println!(
                "Moon JD {:.4}: direction {arcsec:.2}″, distance {d_km:+.2} km, |Δr| {miss_km:.1} km",
                s.jd_tdb
            );
            worst_arcsec = worst_arcsec.max(arcsec);
            worst_km = worst_km.max(d_km.abs());
        }
        println!("Moon worst: {worst_arcsec:.2}″, {worst_km:.2} km (ceiling 60″, 60 km)");
        assert!(worst_arcsec < 60.0, "{worst_arcsec}″");
        assert!(worst_km < 60.0, "{worst_km} km");
    }
}
