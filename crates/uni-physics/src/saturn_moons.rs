//! Saturn's eight major moons from Meeus, *Astronomical Algorithms* 2nd ed. (1998), ch. 46:
//! G. Dourneau's theory (1987) with the corrections and constants of Harper & Taylor (1993,
//! A&A 268, 326), as Meeus gives it (M1 Task A2).
//!
//! Meeus gives each moon's longitude λ, radius r (in Saturn equatorial radii),
//! inclination γ to Saturn's equator and node Ω, longitudes counted from the B1950 equinox
//! along the B1950 ecliptic to Saturn's equatorial node (168.8112°), then along the equator.
//! His final steps rotate them to the B1950 ecliptic by Saturn's equatorial inclination
//! 28.0817° and node 168.8112°; [`ecliptic_of_date_to_icrs`] at B1950.0 (JDE 2433282.4235:
//! the B1950 mean obliquity, then IAU 1976 precession to J2000) takes them to ICRS. The 0.5″
//! FK4 → FK5 equinox offset is ignored (≤ 9 km at Iapetus). Geometric: no light time.
//!
//! Transcription: from the soniakeys/meeus typing (`saturnmoons.go`, MIT) checked number for
//! number against P. J. Naughter's independent AA+ typing (`AASaturnMoons.cpp`, v2.44): all
//! 225 constants agree except Hyperion's inclination term, 0.6434886 against AA+'s 0.643486
//! cos χ; the latter is used (2.6 × 10⁻⁶ °, 0.07 km).
//!
//! Page citations: the constants and series below are Meeus ch. 46's, printed before its
//! Example 46.a (p. 334, as the soniakeys/meeus and learnmeeus typings cite it); the page of
//! each moon's block could not be pinned, because the book was not available.
//!
//! Units (A2 fix round 1, review I3): r is multiplied by 60 438 km, the unit implied by JPL's
//! mean semi-major axes (https://ssd.jpl.nasa.gov/sats/elem/, SAT441) of the two outer regular
//! moons, Titan (1 221 900 km / 20.216 193 = 60 442 km) and Iapetus (3 561 700 km /
//! 58.935 028 = 60 434 km). Those two are used because JPL rounds a to 100 km (±2.5 and
//! ±0.9 km in the unit) and J2 shifts their mean distance by under 5 × 10⁻⁵; the inner moons'
//! ratios scatter more (Tethys 60 438, Dione 60 445, Rhea 60 417, Enceladus 60 489, Mimas
//! 60 610 km) and Hyperion's (60 454 km) mean distance is ill-defined in its resonance with
//! Titan. The 60 330 km Saturn reference radius used before (NAIF Cassini PCK
//! `cpck02Sep2010_Sci.tpc`) left the distances at 0.9983–0.9994 of DE441/sat441's (Titan
//! −2 200 km, Iapetus −6 100 km); 60 330 / 0.9982 = 60 439 km agrees with JPL's a, so the unit
//! is independent of the Horizons goldens that test it. Meeus's book, which states the unit
//! the theory uses, was not available.
//!
//! Validity: outside ±3000 yr of J2000 ([`crate::ephemeris::MOON_THEORY_WINDOW_CY`]) this
//! module returns the value at the nearer window edge and the Ephemeris uses the moons' JPL
//! mean elements instead.

use glam::DVec3;
use uni_core::Epoch;

use crate::ephemeris::clamp_to_moon_theory_window;
use crate::frames::ecliptic_of_date_to_icrs;

/// One of Saturn's eight major moons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaturnMoon {
    Mimas,
    Enceladus,
    Tethys,
    Dione,
    Rhea,
    Titan,
    Hyperion,
    Iapetus,
}

/// The theory's unit of length [m]: JPL's mean a of Titan and Iapetus over the theory's
/// (module doc).
const SATURN_UNIT_M: f64 = 60_438.0e3;

/// B1950.0 [JDE], the epoch of the theory's ecliptic and equinox.
const B1950_JDE: f64 = 2_433_282.4235;

/// Saturn's equator on the B1950 ecliptic: inclination and ascending node [°] (Meeus ch. 46,
/// before Example 46.a, p. 334).
const EQUATOR_INCLINATION_DEG: f64 = 28.0817;
const EQUATOR_NODE_DEG: f64 = 168.8112;

/// The time arguments t₁ … t₁₁ (t₃ and t₅ enter only W₀, W₃, W₄) and angles W₀ … W₈ [rad]
/// of Meeus ch. 46 (before Example 46.a, p. 334).
struct Args {
    t1: f64,
    t2: f64,
    t4: f64,
    t6: f64,
    t7: f64,
    t8: f64,
    t9: f64,
    t10: f64,
    t11: f64,
    w0: f64,
    w1: f64,
    w2: f64,
    w3: f64,
    w4: f64,
    w5: f64,
    w6: f64,
    w7: f64,
    w8: f64,
    /// e₁, the eccentricity of Saturn's orbit as the theory uses it.
    e1: f64,
}

/// Degrees → radians, after reducing to [0°, 360°).
fn rad(deg: f64) -> f64 {
    deg.rem_euclid(360.0).to_radians()
}

impl Args {
    fn new(jde: f64) -> Self {
        let t1 = jde - 2_411_093.0;
        let t2 = t1 / 365.25;
        let t3 = (jde - 2_433_282.423) / 365.25 + 1950.0;
        let t4 = jde - 2_411_368.0;
        let t5 = t4 / 365.25;
        let t6 = jde - 2_415_020.0;
        let t7 = t6 / 36_525.0;
        let t8 = t6 / 365.25;
        let t9 = (jde - 2_442_000.5) / 365.25;
        let t10 = jde - 2_409_786.0;
        let t11 = t10 / 36_525.0;
        Args {
            t1,
            t2,
            t4,
            t6,
            t7,
            t8,
            t9,
            t10,
            t11,
            w0: rad(5.095 * (t3 - 1866.39)),
            w1: rad(74.4 + 32.39 * t2),
            w2: rad(134.3 + 92.62 * t2),
            w3: rad(42.0 - 0.5118 * t5),
            w4: rad(276.59 + 0.5118 * t5),
            w5: rad(267.2635 + 1222.1136 * t7),
            w6: rad(175.4762 + 1221.5515 * t7),
            w7: rad(2.4891 + 0.002435 * t7),
            w8: rad(113.35 - 0.2597 * t7),
            e1: 0.05589 - 0.000346 * t7,
        }
    }
}

/// λ, r [Saturn radii], γ, Ω (angles in rad) of one moon: Meeus's output per satellite.
struct Orbit {
    lambda: f64,
    r: f64,
    gamma: f64,
    node: f64,
}

/// Mimas (Meeus ch. 46, before Example 46.a, p. 334).
fn mimas(q: &Args) -> Orbit {
    let l = rad(127.64 + 381.994_497 * q.t1
        - 43.57 * q.w0.sin()
        - 0.720 * (3.0 * q.w0).sin()
        - 0.02144 * (5.0 * q.w0).sin());
    let p = rad(106.1 + 365.549 * q.t2);
    let m = l - p;
    let c = (2.182_87 * m.sin() + 0.025_988 * (2.0 * m).sin() + 0.000_43 * (3.0 * m).sin())
        .to_radians();
    Orbit {
        lambda: l + c,
        r: 3.068_79 / (1.0 + 0.019_05 * (m + c).cos()),
        gamma: 1.563_f64.to_radians(),
        node: rad(54.5 - 365.072 * q.t2),
    }
}

/// Enceladus (Meeus ch. 46, before Example 46.a, p. 334).
fn enceladus(q: &Args) -> Orbit {
    let l = rad(200.317 + 262.731_900_2 * q.t1 + 0.256_67 * q.w1.sin() + 0.208_83 * q.w2.sin());
    let p = rad(309.107 + 123.441_21 * q.t2);
    let m = l - p;
    let c = (0.555_77 * m.sin() + 0.001_68 * (2.0 * m).sin()).to_radians();
    Orbit {
        lambda: l + c,
        r: 3.941_18 / (1.0 + 0.004_85 * (m + c).cos()),
        gamma: 0.0262_f64.to_radians(),
        node: rad(348.0 - 151.95 * q.t2),
    }
}

/// Tethys (Meeus ch. 46, before Example 46.a, p. 334).
fn tethys(q: &Args) -> Orbit {
    Orbit {
        lambda: rad(285.306
            + 190.697_912_26 * q.t1
            + 2.063 * q.w0.sin()
            + 0.034_09 * (3.0 * q.w0).sin()
            + 0.001_015 * (5.0 * q.w0).sin()),
        r: 4.880_998,
        gamma: 1.0976_f64.to_radians(),
        node: rad(111.33 - 72.2441 * q.t2),
    }
}

/// Dione (Meeus ch. 46, before Example 46.a, p. 334).
fn dione(q: &Args) -> Orbit {
    let l = rad(254.712 + 131.534_931_93 * q.t1 - 0.0215 * q.w1.sin() - 0.017_33 * q.w2.sin());
    let p = rad(174.8 + 30.82 * q.t2);
    let m = l - p;
    let c = (0.247_17 * m.sin() + 0.000_33 * (2.0 * m).sin()).to_radians();
    Orbit {
        lambda: l + c,
        r: 6.248_71 / (1.0 + 0.002_157 * (m + c).cos()),
        gamma: 0.0139_f64.to_radians(),
        node: rad(232.0 - 30.27 * q.t2),
    }
}

/// Rhea (Meeus ch. 46, before Example 46.a, p. 334), through the subroutine.
fn rhea(q: &Args) -> Orbit {
    let pp = rad(342.7 + 10.057 * q.t2);
    let a1 = 0.000_265 * pp.sin() + 0.001 * q.w4.sin();
    let a2 = 0.000_265 * pp.cos() + 0.001 * q.w4.cos();
    let n = rad(345.0 - 10.057 * q.t2);
    let lambda_dash = rad(359.244 + 79.690_047_2 * q.t1 + 0.086_754 * n.sin());
    let i = (28.0362 + 0.346_898 * n.cos() + 0.0193 * q.w3.cos()).to_radians();
    let node = rad(168.8034 + 0.736_936 * n.sin() + 0.041 * q.w3.sin());
    subroutine(lambda_dash, a1.atan2(a2), a1.hypot(a2), 8.725_924, node, i)
}

/// Titan (Meeus ch. 46, before Example 46.a, p. 334), through the subroutine.
fn titan(q: &Args) -> Orbit {
    let l = rad(261.1582 + 22.576_978_55 * q.t4 + 0.074_025 * q.w3.sin());
    let i_dash = (27.451_41 + 0.295_999 * q.w3.cos()).to_radians();
    let node_dash = rad(168.669_25 + 0.628_808 * q.w3.sin());
    let a1 = q.w7.sin() * (node_dash - q.w8).sin();
    let a2 = q.w7.cos() * i_dash.sin() - q.w7.sin() * i_dash.cos() * (node_dash - q.w8).cos();
    let g0 = 102.8623_f64.to_radians();
    let psi = a1.atan2(a2);
    let s = a1.hypot(a2);
    // ϖ and g by three fixed-point passes, as Meeus prescribes.
    let mut g = q.w4 - node_dash - psi;
    let mut varpi = 0.0;
    for _ in 0..3 {
        varpi = q.w4 + (0.375_15 * ((2.0 * g).sin() - (2.0 * g0).sin())).to_radians();
        g = varpi - node_dash - psi;
    }
    let e_dash = 0.029_092 + 0.000_190_48 * ((2.0 * g).cos() - (2.0 * g0).cos());
    let qq = 2.0 * (q.w5 - varpi);
    let b1 = i_dash.sin() * (node_dash - q.w8).sin();
    let b2 = q.w7.cos() * i_dash.sin() * (node_dash - q.w8).cos() - q.w7.sin() * i_dash.cos();
    let theta = b1.atan2(b2) + q.w8;
    let e = e_dash + 0.002_778_797 * e_dash * qq.cos();
    let p = varpi + (0.159_215 * qq.sin()).to_radians();
    let u = 2.0 * q.w5 - 2.0 * theta + psi;
    let h = 0.9375 * e_dash * e_dash * qq.sin() + 0.1875 * s * s * (2.0 * (q.w5 - theta)).sin();
    let lambda_dash = l
        - (0.254_744 * (q.e1 * q.w6.sin() + 0.75 * q.e1 * q.e1 * (2.0 * q.w6).sin() + h))
            .to_radians();
    let i = i_dash + (0.031_843 * s * u.cos()).to_radians();
    let node = node_dash + (0.031_843 * s * u.sin() / i_dash.sin()).to_radians();
    subroutine(lambda_dash, p, e, 20.216_193, node, i)
}

/// Hyperion (Meeus ch. 46, before Example 46.a, p. 334), through the subroutine.
fn hyperion(q: &Args) -> Orbit {
    let eta = rad(92.39 + 0.562_107_1 * q.t6);
    let zeta = rad(148.19 - 19.18 * q.t8);
    let theta = rad(184.8 - 35.41 * q.t9);
    let theta_dash = theta - 7.5_f64.to_radians();
    let a_s = rad(176.0 + 12.22 * q.t8);
    let b_s = rad(8.0 + 24.44 * q.t8);
    let c_s = b_s + 5.0_f64.to_radians();
    let varpi = rad(69.898 - 18.670_88 * q.t8);
    let phi = 2.0 * (varpi - q.w5);
    let chi = rad(94.9 - 2.292 * q.t8);
    let a = 24.506_01 - 0.086_86 * eta.cos() - 0.001_66 * (zeta + eta).cos()
        + 0.001_75 * (zeta - eta).cos();
    let e = 0.103_458 - 0.004_099 * eta.cos() - 0.000_167 * (zeta + eta).cos()
        + 0.000_235 * (zeta - eta).cos()
        + 0.023_03 * zeta.cos()
        - 0.002_12 * (2.0 * zeta).cos()
        + 0.000_151 * (3.0 * zeta).cos()
        + 0.000_13 * phi.cos();
    let p = varpi
        + (0.156_48 * chi.sin()
            - 0.4457 * eta.sin()
            - 0.2657 * (zeta + eta).sin()
            - 0.3573 * (zeta - eta).sin()
            - 12.872 * zeta.sin()
            + 1.668 * (2.0 * zeta).sin()
            - 0.2419 * (3.0 * zeta).sin()
            - 0.07 * phi.sin())
        .to_radians();
    let lambda_dash = rad(177.047 + 16.919_938_29 * q.t6)
        + (0.156_48 * chi.sin() + 9.142 * eta.sin() + 0.007 * (2.0 * eta).sin()
            - 0.014 * (3.0 * eta).sin()
            + 0.2275 * (zeta + eta).sin()
            + 0.2112 * (zeta - eta).sin()
            - 0.26 * zeta.sin()
            - 0.0098 * (2.0 * zeta).sin()
            - 0.013 * a_s.sin()
            + 0.017 * b_s.sin()
            - 0.0303 * phi.sin())
        .to_radians();
    let i = (27.3347 + 0.643_486 * chi.cos() + 0.315 * q.w3.cos() + 0.018 * theta.cos()
        - 0.018 * c_s.cos())
    .to_radians();
    let node = rad(
        168.6812 + 1.401_36 * chi.cos() + 0.685_99 * q.w3.sin() - 0.0392 * c_s.sin()
            + 0.0366 * theta_dash.sin(),
    );
    subroutine(lambda_dash, p, e, a, node, i)
}

/// Iapetus (Meeus ch. 46, before Example 46.a, p. 334), through the subroutine.
fn iapetus(q: &Args) -> Orbit {
    let l = rad(261.1582 + 22.576_978_55 * q.t4);
    let varpi_dash = rad(91.796 + 0.562 * q.t7);
    let psi = rad(4.367 - 0.195 * q.t7);
    let theta = rad(146.819 - 3.198 * q.t7);
    let phi = rad(60.470 + 1.521 * q.t7);
    let big_phi = rad(205.055 - 2.091 * q.t7);
    let e_dash = 0.028_298 + 0.001_156 * q.t11;
    let varpi0 = rad(352.91 + 11.71 * q.t11);
    let mu = rad(76.3852 + 4.537_951_25 * q.t10);
    let t11 = q.t11;
    let i_dash = (18.4602 + t11 * (-0.9518 + t11 * (-0.072 + t11 * 0.0054))).to_radians();
    let node_dash = rad(143.198 + t11 * (-3.919 + t11 * (0.116 + t11 * 0.008)));
    let l_ = mu - varpi0;
    let g = varpi0 - node_dash - psi;
    let g1 = varpi0 - node_dash - phi;
    let ls = q.w5 - varpi_dash;
    let gs = varpi_dash - theta;
    let lt = l - q.w4;
    let gt = q.w4 - big_phi;
    let u1 = 2.0 * (l_ + g - ls - gs);
    let u2 = l_ + g1 - lt - gt;
    let u3 = l_ + 2.0 * (g - ls - gs);
    let u4 = lt + gt - g1;
    let u5 = 2.0 * (ls + gs);
    let a = 58.935_028 + 0.004_638 * u1.cos() + 0.058_222 * u2.cos();
    let e = e_dash - 0.001_409_7 * (g1 - gt).cos()
        + 0.000_373_3 * (u5 - 2.0 * g).cos()
        + 0.000_118 * u3.cos()
        + 0.000_240_8 * l_.cos()
        + 0.000_284_9 * (l_ + u2).cos()
        + 0.000_619 * u4.cos();
    let w = (0.080_77 * (g1 - gt).sin() + 0.021_39 * (u5 - 2.0 * g).sin() - 0.006_76 * u3.sin()
        + 0.0138 * l_.sin()
        + 0.016_32 * (l_ + u2).sin()
        + 0.035_47 * u4.sin())
    .to_radians();
    let p = varpi0 + w / e_dash;
    let lambda_dash = mu
        + (-0.042_99 * u2.sin()
            - 0.007_89 * u1.sin()
            - 0.063_12 * ls.sin()
            - 0.002_95 * (2.0 * ls).sin()
            - 0.022_31 * u5.sin()
            + 0.0065 * (u5 + psi).sin())
        .to_radians();
    let i = i_dash
        + (0.042_04 * (u5 + psi).cos()
            + 0.002_35 * (l_ + g1 + lt + gt + phi).cos()
            + 0.0036 * (u2 + phi).cos())
        .to_radians();
    let w_dash = (0.042_04 * (u5 + psi).sin()
        + 0.002_35 * (l_ + g1 + lt + gt + phi).sin()
        + 0.003_58 * (u2 + phi).sin())
    .to_radians();
    let node = node_dash + w_dash / i_dash.sin();
    subroutine(lambda_dash, p, e, a, node, i)
}

/// Meeus's subroutine (ch. 46): from mean longitude λ′, perisaturnium p, e, a, and node Ω and
/// inclination i on the B1950 ecliptic to λ, r, γ and Ω on Saturn's equator. The equation of
/// the centre is its series in e to e⁵.
fn subroutine(lambda_dash: f64, p: f64, e: f64, a: f64, node: f64, i: f64) -> Orbit {
    let m = lambda_dash - p;
    let (e2, e3) = (e * e, e * e * e);
    let (e4, e5) = (e2 * e2, e3 * e2);
    let c = (2.0 * e - 0.25 * e3 + 0.052_083_333_3 * e5) * m.sin()
        + (1.25 * e2 - 0.458_333_333 * e4) * (2.0 * m).sin()
        + (1.083_333_333 * e3 - 0.671_875 * e5) * (3.0 * m).sin()
        + 1.072_917 * e4 * (4.0 * m).sin()
        + 1.142_708 * e5 * (5.0 * m).sin();
    let (s1, c1) = EQUATOR_INCLINATION_DEG.to_radians().sin_cos();
    let g = node - EQUATOR_NODE_DEG.to_radians();
    let (sin_i, cos_i) = i.sin_cos();
    let a1 = sin_i * g.sin();
    let a2 = c1 * sin_i * g.cos() - s1 * cos_i;
    let u = a1.atan2(a2);
    let h = c1 * sin_i - s1 * cos_i * g.cos();
    let psi = (s1 * g.sin()).atan2(h);
    Orbit {
        lambda: lambda_dash + c + u - g - psi,
        r: a * (1.0 - e2) / (1.0 + e * (m + c).cos()),
        gamma: a1.hypot(a2).asin(),
        node: EQUATOR_NODE_DEG.to_radians() + u,
    }
}

/// Position relative to Saturn's centre [m], ICRS axes.
///
/// Outside ±3000 yr of J2000 the value at the nearer window edge (module doc).
pub fn saturnicentric_icrs(m: SaturnMoon, t: Epoch) -> DVec3 {
    let t = clamp_to_moon_theory_window(t);
    let q = Args::new(t.jd_tdb());
    let o = match m {
        SaturnMoon::Mimas => mimas(&q),
        SaturnMoon::Enceladus => enceladus(&q),
        SaturnMoon::Tethys => tethys(&q),
        SaturnMoon::Dione => dione(&q),
        SaturnMoon::Rhea => rhea(&q),
        SaturnMoon::Titan => titan(&q),
        SaturnMoon::Hyperion => hyperion(&q),
        SaturnMoon::Iapetus => iapetus(&q),
    };
    // Meeus ch. 46: Saturn-equatorial coordinates (x toward the equator's node on the B1950
    // ecliptic), then the rotations "towards the plane of the ecliptic" and "towards the
    // vernal equinox".
    let u = o.lambda - o.node;
    let w = o.node - EQUATOR_NODE_DEG.to_radians();
    let (sin_u, cos_u) = u.sin_cos();
    let (sin_w, cos_w) = w.sin_cos();
    let (sin_g, cos_g) = o.gamma.sin_cos();
    let x = o.r * (cos_u * cos_w - sin_u * cos_g * sin_w);
    let y = o.r * (sin_u * cos_w * cos_g + cos_u * sin_w);
    let z = o.r * sin_u * sin_g;
    let (s1, c1) = EQUATOR_INCLINATION_DEG.to_radians().sin_cos();
    let (s2, c2) = EQUATOR_NODE_DEG.to_radians().sin_cos();
    let (a1, b1, c1z) = (x, c1 * y - s1 * z, s1 * y + c1 * z);
    let ecl = DVec3::new(c2 * a1 - s2 * b1, s2 * a1 + c2 * b1, c1z) * SATURN_UNIT_M;
    let b1950 = Epoch::from_days_since_j2000(B1950_JDE - 2_451_545.0);
    let dist = ecl.length();
    ecliptic_of_date_to_icrs(ecl.y.atan2(ecl.x), (ecl.z / dist).asin(), dist, b1950)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sat_elements::test_support::jpl_sat;
    use crate::standish::golden::{epoch_of, horizons_spots, pos_m};

    const ALL: [(SaturnMoon, i32); 8] = [
        (SaturnMoon::Mimas, 601),
        (SaturnMoon::Enceladus, 602),
        (SaturnMoon::Tethys, 603),
        (SaturnMoon::Dione, 604),
        (SaturnMoon::Rhea, 605),
        (SaturnMoon::Titan, 606),
        (SaturnMoon::Hyperion, 607),
        (SaturnMoon::Iapetus, 608),
    ];

    /// Horizons 601–608@699 (sat441) at 1950, 2000, 2026-11-20 and 2050: prints the measured
    /// worst error per moon in km and as a fraction of a (JPL's mean a); ceiling Titan
    /// 4 000 km, the others 2 % of a.
    #[test]
    fn saturn_moons_vs_horizons() {
        for (moon, id) in ALL {
            let a_km = jpl_sat(id).a_m / 1e3;
            let ceiling_km = if moon == SaturnMoon::Titan {
                4_000.0
            } else {
                0.02 * a_km
            };
            let mut worst_km = 0.0_f64;
            for s in horizons_spots(id, 699) {
                let model = saturnicentric_icrs(moon, epoch_of(s.jd_tdb));
                let truth = pos_m(&s);
                let miss_km = (model - truth).length() / 1e3;
                println!(
                    "{moon:?} JD {:.4}: |Δr| {miss_km:.1} km, r/r_true {:.5}",
                    s.jd_tdb,
                    model.length() / truth.length()
                );
                worst_km = worst_km.max(miss_km);
            }
            println!(
                "{moon:?} worst: {worst_km:.1} km = {:.4} of a (ceiling {ceiling_km:.0} km)",
                worst_km / a_km
            );
            assert!(worst_km < ceiling_km, "{moon:?}: {worst_km} km");
        }
    }
}
