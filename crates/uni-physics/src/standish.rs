//! Planet positions from JPL's "Approximate Positions of the Planets" (Standish & Williams;
//! https://ssd.jpl.nasa.gov/planets/approx_pos.html) (M1 Task A1). Pluto is in [`crate::pluto`]
//! (review I1); the page no longer lists it.
//!
//! The page's steps (physics research §6): T = (JD_TDB − 2451545.0)/36525; each element
//! x = x₀ + ẋT; ω = ϖ − Ω; M = L − ϖ + bT² + c cos(fT) + s sin(fT) (Table 2b terms for
//! Jupiter–Neptune with Table 2a only), reduced to [−180°, 180°]; Kepler's equation; the
//! orbital-plane position rotated by (ω, I, Ω) into the J2000 ecliptic and by ε₀ into ICRS
//! ([`crate::frames::ecliptic_j2000_to_icrs`]; the page's 23.43928° agrees with IAU 1976's
//! 84381.448″ = 23.4392911° to 0.04″). The positions are heliocentric, of the planetary-system barycentres
//! (EM Bary = Earth–Moon barycentre).
//!
//! Outside 3000 BC – AD 3000 (review C1) a, e, I, ϖ and Ω are frozen at the nearer window edge
//! (T_e = clamp(T, −50, 10)), the Table 2b terms are evaluated at T_e, and only L advances at
//! its Table 2a rate: the linear rates would push e past 1 and I to tens of degrees.
//!
//! Velocity: the analytic time derivative of these positions: the motion along the ellipse
//! at the tables' dM/dt plus the drift of a, e, I, ϖ and Ω (the apsidal and nodal rotation
//! alone is ϖ̇/L̇ = 1.5e-3 of Neptune's speed).
//!
//! Far from J2000 the mean longitude L₀ + L̇T resolves time only to the f64 spacing of T: at
//! 10⁶ yr that is ≈ 5 ms (Mercury steps ≈ 0.2 km), at 10⁷ yr ≈ 40 ms (≈ 2 km).

use glam::DVec3;
use uni_core::{AU, DAY, Epoch};

use crate::NaifId;
use crate::frames::ecliptic_j2000_to_icrs;
use crate::kepler::{elliptic_perifocal, perifocal_to_reference};

/// The bodies Standish's elements describe (planetary-system barycentres; EMB = Earth–Moon).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Planet {
    Mercury,
    Venus,
    EmBary,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
}

impl Planet {
    /// Every planet, in order from the Sun.
    pub const ALL: [Planet; 8] = [
        Planet::Mercury,
        Planet::Venus,
        Planet::EmBary,
        Planet::Mars,
        Planet::Jupiter,
        Planet::Saturn,
        Planet::Uranus,
        Planet::Neptune,
    ];

    /// NAIF id of the planetary-system barycentre, 1..=8 (the Horizons targets; review I2).
    pub fn barycentre_id(self) -> NaifId {
        self as NaifId + 1
    }
}

/// Which table an epoch uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Validity {
    /// Table 1: AD 1800–2050.
    Table1,
    /// Tables 2a + 2b: 3000 BC – AD 3000.
    Table2,
    /// Outside 3000 BC – AD 3000: elements frozen at the nearer window edge (review C1).
    Extrapolated,
}

/// Start of Table 2's window [Julian centuries from J2000] (3000 BC).
pub const TABLE2_T_MIN: f64 = -50.0;
/// End of Table 2's window [Julian centuries from J2000] (AD 3000).
pub const TABLE2_T_MAX: f64 = 10.0;

/// Table 1's window "1800 AD – 2050 AD" in Julian centuries from J2000, both ends inclusive:
/// T = −2 is 1799-12-30 12h TDB and T = 0.5 is 2050-01-01 0h TDB.
const TABLE1_T_MIN: f64 = -2.0;
const TABLE1_T_MAX: f64 = 0.5;

/// One Julian century [s] (36 525 days, IAU 1976).
const JULIAN_CENTURY_S: f64 = 36_525.0 * DAY;

/// One planet's row: the elements at J2000 and their rates per Julian century, in the page's
/// column order a [au], e, I [°], L [°], ϖ (long. peri.) [°], Ω (long. node) [°].
struct Row {
    at_j2000: [f64; 6],
    per_century: [f64; 6],
}

const fn row(at_j2000: [f64; 6], per_century: [f64; 6]) -> Row {
    Row {
        at_j2000,
        per_century,
    }
}

/// Table 1, "Keplerian elements and their rates, with respect to the mean ecliptic and
/// equinox of J2000, valid for the time-interval 1800 AD – 2050 AD"; copied digit for digit
/// from https://ssd.jpl.nasa.gov/planets/approx_pos.html (read 2026-11-22). Rows in
/// [`Planet::ALL`] order.
#[rustfmt::skip]
const TABLE_1: [Row; 8] = [
    row([ 0.38709927, 0.20563593,  7.00497902, 252.25032350,  77.45779628,  48.33076593],
        [ 0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]),
    row([ 0.72333566, 0.00677672,  3.39467605, 181.97909950, 131.60246718,  76.67984255],
        [ 0.00000390,-0.00004107, -0.00078890, 58517.81538729,  0.00268329, -0.27769418]),
    row([ 1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193,   0.0       ],
        [ 0.00000562,-0.00004392, -0.01294668, 35999.37244981,  0.32327364,   0.0       ]),
    row([ 1.52371034, 0.09339410,  1.84969142,  -4.55343205, -23.94362959,  49.55953891],
        [ 0.00001847, 0.00007882, -0.00813131, 19140.30268499,  0.44441088, -0.29257343]),
    row([ 5.20288700, 0.04838624,  1.30439695,  34.39644051,  14.72847983, 100.47390909],
        [-0.00011607,-0.00013253, -0.00183714,  3034.74612775,  0.21252668,  0.20469106]),
    row([ 9.53667594, 0.05386179,  2.48599187,  49.95424423,  92.59887831, 113.66242448],
        [-0.00125060,-0.00050991,  0.00193609,  1222.49362201, -0.41897216, -0.28867794]),
    row([19.18916464, 0.04725744,  0.77263783, 313.23810451, 170.95427630,  74.01692503],
        [-0.00196176,-0.00004397, -0.00242939,   428.48202785,  0.40805281,  0.04240589]),
    row([30.06992276, 0.00859048,  1.77004347, -55.12002969,  44.96476227, 131.78422574],
        [ 0.00026291, 0.00005105,  0.00035372,   218.45945325, -0.32241464, -0.00508664]),
];

/// Table 2a, the same elements "valid for the time-interval 3000 BC – 3000 AD"; same page and
/// date. Jupiter–Neptune need [`TABLE_2B`] with it.
#[rustfmt::skip]
const TABLE_2A: [Row; 8] = [
    row([ 0.38709843, 0.20563661,  7.00559432, 252.25166724,  77.45771895,  48.33961819],
        [ 0.00000000, 0.00002123, -0.00590158, 149472.67486623, 0.15940013, -0.12214182]),
    row([ 0.72332102, 0.00676399,  3.39777545, 181.97970850, 131.76755713,  76.67261496],
        [-0.00000026,-0.00005107,  0.00043494, 58517.81560260,  0.05679648, -0.27274174]),
    row([ 1.00000018, 0.01673163, -0.00054346, 100.46691572, 102.93005885,  -5.11260389],
        [-0.00000003,-0.00003661, -0.01337178, 35999.37306329,  0.31795260, -0.24123856]),
    row([ 1.52371243, 0.09336511,  1.85181869,  -4.56813164, -23.91744784,  49.71320984],
        [ 0.00000097, 0.00009149, -0.00724757, 19140.29934243,  0.45223625, -0.26852431]),
    row([ 5.20248019, 0.04853590,  1.29861416,  34.33479152,  14.27495244, 100.29282654],
        [-0.00002864, 0.00018026, -0.00322699,  3034.90371757,  0.18199196,  0.13024619]),
    row([ 9.54149883, 0.05550825,  2.49424102,  50.07571329,  92.86136063, 113.63998702],
        [-0.00003065,-0.00032044,  0.00451969,  1222.11494724,  0.54179478, -0.25015002]),
    row([19.18797948, 0.04685740,  0.77298127, 314.20276625, 172.43404441,  73.96250215],
        [-0.00020455,-0.00001550, -0.00180155,   428.49512595,  0.09266985,  0.05739699]),
    row([30.06952752, 0.00895439,  1.77005520, 304.22289287,  46.68158724, 131.78635853],
        [ 0.00006447, 0.00000818,  0.00022400,   218.46515314,  0.01009938, -0.00606302]),
];

/// Table 2b, the terms added to M for Jupiter, Saturn, Uranus and Neptune over 3000 BC –
/// AD 3000: b [°/cy²], c [°], s [°], f [°/cy]; same page and date.
#[rustfmt::skip]
const TABLE_2B: [[f64; 4]; 4] = [
    [-0.00012452,  0.06064060, -0.35635438, 38.35125000],
    [ 0.00025899, -0.13434469,  0.87320147, 38.35125000],
    [ 0.00058331, -0.97731848,  0.17689245,  7.67025000],
    [-0.00041348,  0.68346318, -0.10162547,  7.67025000],
];

/// The table used at `t`.
pub fn validity(t: Epoch) -> Validity {
    validity_at(t.julian_centuries())
}

fn validity_at(t_cy: f64) -> Validity {
    if (TABLE1_T_MIN..=TABLE1_T_MAX).contains(&t_cy) {
        Validity::Table1
    } else if (TABLE2_T_MIN..=TABLE2_T_MAX).contains(&t_cy) {
        Validity::Table2
    } else {
        Validity::Extrapolated
    }
}

/// Heliocentric position [m] and velocity [m/s] of `p` in ICRS axes.
///
/// Finite at every finite epoch (review C1); see [`validity`] for the table in use.
pub fn heliocentric_state_icrs(p: Planet, t: Epoch) -> (DVec3, DVec3) {
    let el = mean_elements(p, t.julian_centuries());
    let (a, e) = (el.a_au * AU, el.e);
    let per_s = 1.0 / JULIAN_CENTURY_S;
    let n = el.m_rate_deg_per_cy.to_radians() * per_s;
    let (pos, vel_mean) = elliptic_perifocal(a, e, el.m_deg.to_radians(), n);
    let (node, argp) = (
        el.node_deg.to_radians(),
        (el.peri_deg - el.node_deg).to_radians(),
    );
    let rot = perifocal_to_reference(el.i_deg.to_radians(), node, argp);

    // The drift of the elements, so the velocity is the time derivative of the positions.
    // In the plane, at fixed M: ∂r/∂a = r/a; ∂r/∂e = (−a, −e·y/(1 − e²)) + (∂r/∂M)·sin E with
    // sin E = y/(a√(1 − e²)) and ∂r/∂M = v/n.
    let [a_dot, e_dot, i_dot, peri_dot, node_dot] = el.drift_per_cy;
    let one_minus_e2 = (1.0 - e) * (1.0 + e);
    let sin_ea = pos.y / (a * one_minus_e2.sqrt());
    let dr_de = DVec3::new(-a, -e * pos.y / one_minus_e2, 0.0) + vel_mean * (sin_ea / n);
    let vel_plane = vel_mean + pos * (a_dot * AU * per_s / a) + dr_de * (e_dot * per_s);
    // The orbit's own rotation: angular velocity Ω̇ ẑ + İ n̂ + ω̇ ĥ for R_z(Ω)·R_x(I)·R_z(ω).
    let node_dir = DVec3::new(node.cos(), node.sin(), 0.0);
    let spin = (DVec3::Z * node_dot.to_radians()
        + node_dir * i_dot.to_radians()
        + rot.z_axis * (peri_dot - node_dot).to_radians())
        * per_s;
    let r_ecl = rot * pos;
    let v_ecl = rot * vel_plane + spin.cross(r_ecl);
    (ecliptic_j2000_to_icrs(r_ecl), ecliptic_j2000_to_icrs(v_ecl))
}

/// A planet's elements at one instant (J2000 ecliptic and equinox).
#[derive(Clone, Copy, Debug, PartialEq)]
struct MeanElements {
    validity: Validity,
    a_au: f64,
    e: f64,
    i_deg: f64,
    peri_deg: f64,
    node_deg: f64,
    /// Mean anomaly in [−180°, 180°).
    m_deg: f64,
    /// dM/dT [°/century].
    m_rate_deg_per_cy: f64,
    /// d/dT of a [au], e, I, ϖ, Ω [°] per century, in the table's column order without L;
    /// zero outside Table 2's window, where those elements are frozen.
    drift_per_cy: [f64; 5],
}

/// The elements of `p` at `t_cy` Julian centuries from J2000, by the page's steps 1–4.
fn mean_elements(p: Planet, t_cy: f64) -> MeanElements {
    let validity = validity_at(t_cy);
    let k = p as usize;
    let (table, t_el) = match validity {
        Validity::Table1 => (&TABLE_1[k], t_cy),
        _ => (&TABLE_2A[k], t_cy.clamp(TABLE2_T_MIN, TABLE2_T_MAX)),
    };
    let at = |j: usize| table.at_j2000[j] + table.per_century[j] * t_el;
    let (a, e, i, peri, node) = (at(0), at(1), at(2), at(4), at(5));
    // L keeps its rate at every T; the rest are frozen at T_el outside Table 2's window.
    let l = table.at_j2000[3] + table.per_century[3] * t_cy;
    let drifting = validity != Validity::Extrapolated;
    let rate = |j: usize| if drifting { table.per_century[j] } else { 0.0 };
    let mut m = l - peri;
    let mut m_rate = table.per_century[3] - rate(4);
    if validity != Validity::Table1 && k >= Planet::Jupiter as usize {
        let [b, c, s, f] = TABLE_2B[k - Planet::Jupiter as usize];
        let (sin_ft, cos_ft) = (f * t_el).to_radians().sin_cos();
        m += b * t_el * t_el + c * cos_ft + s * sin_ft;
        if drifting {
            m_rate += 2.0 * b * t_el + f.to_radians() * (s * cos_ft - c * sin_ft);
        }
    }
    MeanElements {
        validity,
        a_au: a,
        e,
        i_deg: i,
        peri_deg: peri,
        node_deg: node,
        m_deg: (m + 180.0).rem_euclid(360.0) - 180.0,
        m_rate_deg_per_cy: m_rate,
        drift_per_cy: [rate(0), rate(1), rate(2), rate(4), rate(5)],
    }
}

/// Test support shared with the `pluto` and `ssb` tests: the Horizons goldens (A9).
#[cfg(test)]
pub(crate) mod golden {
    use glam::{DMat3, DVec3};
    use uni_core::Epoch;
    use uni_data::EphemSpot;

    use crate::frames::OBLIQUITY_J2000_RAD;

    /// The spots of `target` relative to `center` in `data/golden/horizons_spots.ron`. A missing
    /// or unreadable file fails the test ("golden file missing: …"); it never skips.
    pub(crate) fn horizons_spots(target: i32, center: i32) -> Vec<EphemSpot> {
        let path = uni_data::data_dir()
            .join("golden")
            .join("horizons_spots.ron");
        let all = uni_data::load_ephem_spots(&path)
            .unwrap_or_else(|e| panic!("golden file missing: {} ({e})", path.display()));
        let spots: Vec<_> = all
            .into_iter()
            .filter(|s| s.target == target && s.center == center)
            .collect();
        assert!(!spots.is_empty(), "no Horizons spots for {target}@{center}");
        spots
    }

    /// The TDB epoch of a Julian date (TDB).
    pub(crate) fn epoch_of(jd_tdb: f64) -> Epoch {
        Epoch::from_days_since_j2000(jd_tdb - 2_451_545.0)
    }

    pub(crate) fn pos_m(s: &EphemSpot) -> DVec3 {
        DVec3::from_array(s.pos_km) * 1e3
    }

    pub(crate) fn vel_m_s(s: &EphemSpot) -> DVec3 {
        DVec3::from_array(s.vel_km_s) * 1e3
    }

    /// Heliocentric ecliptic-J2000 longitude, latitude [″] and distance [km] errors of `model`
    /// against `truth`, both ICRS [m], rotated by −ε₀. The 23 mas frame bias between ICRS and
    /// the J2000 mean equator, which the models ignore, stays in the errors; it is far below
    /// every bound tested.
    pub(crate) fn ecliptic_errors(model: DVec3, truth: DVec3) -> [f64; 3] {
        let to_ecliptic = DMat3::from_rotation_x(-OBLIQUITY_J2000_RAD);
        let spherical = |v: DVec3| {
            let e = to_ecliptic * v;
            (e.y.atan2(e.x), (e.z / e.length()).asin(), e.length())
        };
        let (lm, bm, rm) = spherical(model);
        let (lt, bt, rt) = spherical(truth);
        let dl = (lm - lt + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        [
            dl.to_degrees() * 3600.0,
            (bm - bt).to_degrees() * 3600.0,
            (rm - rt) / 1e3,
        ]
    }
}

#[cfg(test)]
mod tests {
    use uni_core::{AU, DAY};

    use super::golden::{ecliptic_errors, epoch_of, horizons_spots, pos_m, vel_m_s};
    use super::*;

    /// The JPL page's nominal errors (Accuracy table, read 2026-11-22), in its order: λ ["],
    /// φ ["], ρ [km] for Mercury, Venus, EM Bary, Mars, Jupiter, Saturn, Uranus, Neptune.
    const JPL_1800_2050: [[f64; 3]; 8] = [
        [15.0, 1.0, 1_000.0],
        [20.0, 1.0, 4_000.0],
        [20.0, 8.0, 6_000.0],
        [40.0, 2.0, 25_000.0],
        [400.0, 10.0, 600_000.0],
        [600.0, 25.0, 1_500_000.0],
        [50.0, 2.0, 1_000_000.0],
        [10.0, 1.0, 200_000.0],
    ];
    const JPL_3000BC_3000AD: [[f64; 3]; 8] = [
        [20.0, 15.0, 1_000.0],
        [40.0, 30.0, 8_000.0],
        [40.0, 15.0, 15_000.0],
        [100.0, 40.0, 30_000.0],
        [600.0, 100.0, 1_000_000.0],
        [1_000.0, 100.0, 4_000_000.0],
        [2_000.0, 30.0, 8_000_000.0],
        [400.0, 15.0, 4_000_000.0],
    ];

    /// The bounds the tests assert: the JPL value, except where Standish's elements measure
    /// worse than the page's nominal error against the DE441 goldens. There the bound is the
    /// measured worst error rounded up by about 10 % (A1 report, "Standish vs DE441", raised
    /// with the lead). Over the nominal, 1800–2050: Mercury λ 17.4″, φ 1.04″, ρ 1 212 km;
    /// Mars λ 58.8″; Saturn ρ 1.87e6 km; Uranus λ 79.0″, φ 2.23″, ρ 1.34e6 km; Neptune λ
    /// 45.1″, φ 1.33″, ρ 9.21e5 km.
    const ASSERTED_1800_2050: [[f64; 3]; 8] = [
        [19.0, 1.2, 1_350.0],
        [20.0, 1.0, 4_000.0],
        [20.0, 8.0, 6_000.0],
        [65.0, 2.0, 25_000.0],
        [400.0, 10.0, 600_000.0],
        [600.0, 25.0, 2_100_000.0],
        [87.0, 2.5, 1_500_000.0],
        [50.0, 1.5, 1_020_000.0],
    ];
    /// As above for 3000 BC – AD 3000 (spots AD 1000 and AD 2500). Over the nominal: Mercury
    /// ρ 1 127 km (AD 1000); Mars ρ 51 372 km (AD 2500).
    const ASSERTED_3000BC_3000AD: [[f64; 3]; 8] = [
        [20.0, 15.0, 1_250.0],
        [40.0, 30.0, 8_000.0],
        [40.0, 15.0, 15_000.0],
        [100.0, 40.0, 57_000.0],
        [600.0, 100.0, 1_000_000.0],
        [1_000.0, 100.0, 4_000_000.0],
        [2_000.0, 30.0, 8_000_000.0],
        [400.0, 15.0, 4_000_000.0],
    ];

    /// Compares every Horizons spot of each barycentre (1..=8 @ 10) whose JD passes `keep`
    /// with Standish, prints the per-spot and worst errors next to the JPL table, and asserts
    /// the `asserted` bounds and the expected `table`.
    fn check_against_horizons(
        keep: impl Fn(f64) -> bool,
        table: Validity,
        jpl: &[[f64; 3]; 8],
        asserted: &[[f64; 3]; 8],
    ) {
        let mut worst = [[0.0f64; 3]; 8];
        let mut worst_dv = 0.0f64;
        for (k, p) in Planet::ALL.into_iter().enumerate() {
            let spots: Vec<_> = horizons_spots(p.barycentre_id(), 10)
                .into_iter()
                .filter(|s| keep(s.jd_tdb))
                .collect();
            assert!(!spots.is_empty(), "{p:?}: no spots in the window");
            for s in spots {
                let t = epoch_of(s.jd_tdb);
                assert_eq!(validity(t), table, "{p:?} at JD {}", s.jd_tdb);
                let (r, v) = heliocentric_state_icrs(p, t);
                let err = ecliptic_errors(r, pos_m(&s));
                let dv = (v - vel_m_s(&s)).length() / vel_m_s(&s).length();
                worst_dv = worst_dv.max(dv);
                println!(
                    "{p:?} JD {:.4}: dλ {:+9.2}″ dφ {:+7.2}″ dρ {:+11.0} km |Δv|/v {dv:.1e}",
                    s.jd_tdb, err[0], err[1], err[2]
                );
                for q in 0..3 {
                    worst[k][q] = worst[k][q].max(err[q].abs());
                }
            }
        }
        println!("planet   | worst λ / JPL ″      | worst φ / JPL ″    | worst ρ / JPL km");
        for (k, p) in Planet::ALL.into_iter().enumerate() {
            let mark = |q: usize| if worst[k][q] > jpl[k][q] { " OVER" } else { "" };
            println!(
                "{:8} | {:7.2} / {:6.0}{:5} | {:6.2} / {:4.0}{:5} | {:10.0} / {:9.0}{}",
                format!("{p:?}"),
                worst[k][0],
                jpl[k][0],
                mark(0),
                worst[k][1],
                jpl[k][1],
                mark(1),
                worst[k][2],
                jpl[k][2],
                mark(2)
            );
        }
        println!("worst |Δv|/|v| {worst_dv:.2e}");
        for (k, p) in Planet::ALL.into_iter().enumerate() {
            for q in 0..3 {
                assert!(
                    worst[k][q] <= asserted[k][q],
                    "{p:?} quantity {q}: {} > {}",
                    worst[k][q],
                    asserted[k][q]
                );
            }
        }
        // Velocities: derivative of the mean ellipse, so within the position error's order.
        assert!(worst_dv < 1e-2, "{worst_dv}");
    }

    /// Every Horizons barycentre spot in AD 1800–2050 within the JPL 1800–2050 error table
    /// (where it holds; see ASSERTED_1800_2050). Target 399 is not the EMB and is not used.
    #[test]
    fn standish_within_error_table() {
        check_against_horizons(
            |jd| (2_378_495.0..=2_469_807.5).contains(&jd),
            Validity::Table1,
            &JPL_1800_2050,
            &ASSERTED_1800_2050,
        );
    }

    /// The spots at JD 2086307.5 (AD 1000-01-01, Julian calendar) and 2634166.5
    /// (AD 2500-01-01) within the JPL 3000 BC – AD 3000 table.
    #[test]
    fn standish_table2_window() {
        check_against_horizons(
            |jd| jd == 2_086_307.5 || jd == 2_634_166.5,
            Validity::Table2,
            &JPL_3000BC_3000AD,
            &ASSERTED_3000BC_3000AD,
        );
    }

    /// Review Focus 1, review C1: far from the tables every planet stays on a bound,
    /// low-inclination orbit; a, e, I, ϖ and Ω are frozen at the nearer window edge.
    #[test]
    fn standish_far_dates_finite_and_flagged() {
        for years in [1e6, -1e6, 1e7, -1e7] {
            let t = Epoch::from_days_since_j2000(years * 365.25);
            assert_eq!(validity(t), Validity::Extrapolated, "{years} yr");
            let edge = if years > 0.0 {
                TABLE2_T_MAX
            } else {
                TABLE2_T_MIN
            };
            for p in Planet::ALL {
                let far = mean_elements(p, t.julian_centuries());
                let at_edge = mean_elements(p, edge);
                assert_eq!(far.validity, Validity::Extrapolated);
                assert_eq!(
                    (far.a_au, far.e, far.i_deg, far.peri_deg, far.node_deg),
                    (
                        at_edge.a_au,
                        at_edge.e,
                        at_edge.i_deg,
                        at_edge.peri_deg,
                        at_edge.node_deg
                    ),
                    "{p:?} at {years} yr"
                );
                assert!((0.0..1.0).contains(&far.e), "{p:?}: e {}", far.e);
                assert!(far.i_deg.abs() < 20.0, "{p:?}: I {}", far.i_deg);
                assert!(
                    (-180.0..=180.0).contains(&far.m_deg),
                    "{p:?}: M {}",
                    far.m_deg
                );
                let (r, v) = heliocentric_state_icrs(p, t);
                assert!(r.is_finite() && v.is_finite(), "{p:?} at {years} yr");
                let d = r.length() / AU;
                assert!(d > 0.2 && d < 60.0, "{p:?} at {years} yr: {d} AU");
            }
        }
        // Edge values (review C1): e stays in [0.0062, 0.206] and |I| ≤ 7.31°; Saturn's e at
        // T = −50 is 0.05550825 + 0.00032044 × 50 = 0.0715.
        for edge in [TABLE2_T_MIN, TABLE2_T_MAX] {
            for p in Planet::ALL {
                let el = mean_elements(p, edge);
                assert!(
                    (0.0062..=0.206).contains(&el.e),
                    "{p:?} at T {edge}: e {}",
                    el.e
                );
                assert!(el.i_deg.abs() <= 7.31, "{p:?} at T {edge}: I {}", el.i_deg);
            }
        }
        let saturn = mean_elements(Planet::Saturn, TABLE2_T_MIN);
        assert!((saturn.e - 0.071_530_25).abs() < 1e-12, "{}", saturn.e);
    }

    /// Over 2026 the EMB's heliocentric distance spans its perihelion and aphelion, a(1 ∓ e),
    /// and stays inside [0.9832, 1.0168] AU.
    #[test]
    fn emb_distance_annual_range() {
        let jan1 = epoch_of(2_461_041.5); // 2026-01-01 0h TDB
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for day in 0..365 {
            let (r, _) =
                heliocentric_state_icrs(Planet::EmBary, jan1.add_seconds(day as f64 * DAY));
            let d = r.length() / AU;
            assert!((0.9832..=1.0168).contains(&d), "day {day}: {d} AU");
            (lo, hi) = (lo.min(d), hi.max(d));
        }
        assert!(lo < 0.9834 && hi > 1.0166, "range {lo}–{hi} AU");
    }

    /// The analytic velocity is the time derivative of the positions in each regime (Table 1,
    /// Table 2, extrapolated): central differences over ±300 s agree to 1e-7, which needs the
    /// element-drift terms (Neptune's apsidal and nodal turn alone is 1.5e-3). ±300 s keeps
    /// both the truncation error (Mercury ≈ 1e-8) and the rounding of L = L̇T (at T = 11,
    /// ≈ 8 µs of time) well below the bound. At 10⁶ yr the positions step with the ≈ 5 ms
    /// spacing of T, so the difference spans ±600 s and the bound is 1e-5.
    #[test]
    fn velocity_matches_position_derivative() {
        let cases = [
            (0.0, 300.0, 1e-7),
            (9_819.5, 300.0, 1e-7),
            (-100_000.0, 300.0, 1e-7),
            (400_000.0, 300.0, 1e-7),
            (3.65e8, 600.0, 1e-5),
        ];
        for (days, half_step, bound) in cases {
            let t = Epoch::from_days_since_j2000(days);
            for p in Planet::ALL {
                let (_, v) = heliocentric_state_icrs(p, t);
                let (r_minus, _) = heliocentric_state_icrs(p, t.add_seconds(-half_step));
                let (r_plus, _) = heliocentric_state_icrs(p, t.add_seconds(half_step));
                let numeric = (r_plus - r_minus) / (2.0 * half_step);
                let rel = (v - numeric).length() / v.length();
                assert!(rel < bound, "{p:?} at day {days}: {rel:e}");
            }
        }
    }

    /// Table 1 covers T ∈ [−2, 0.5] (1799-12-30 12h … 2050-01-01 0h TDB) inclusive, Table 2
    /// T ∈ [−50, 10]. Prints the position jump each planet makes where the tables switch.
    #[test]
    fn validity_windows_and_switch_jumps() {
        let at_t = |t_cy: f64| Epoch::from_days_since_j2000(t_cy * 36_525.0);
        assert_eq!(validity(Epoch::J2000), Validity::Table1);
        assert_eq!(validity(at_t(-2.0)), Validity::Table1);
        assert_eq!(validity(at_t(0.5)), Validity::Table1);
        assert_eq!(validity(at_t(0.5).add_seconds(1.0)), Validity::Table2);
        assert_eq!(validity(at_t(-2.0).add_seconds(-1.0)), Validity::Table2);
        assert_eq!(validity(at_t(TABLE2_T_MIN)), Validity::Table2);
        assert_eq!(validity(at_t(TABLE2_T_MAX)), Validity::Table2);
        assert_eq!(
            validity(at_t(TABLE2_T_MAX).add_seconds(1.0)),
            Validity::Extrapolated
        );
        assert_eq!(
            validity(at_t(TABLE2_T_MIN).add_seconds(-1.0)),
            Validity::Extrapolated
        );
        for (label, t) in [("1800 (T = −2)", at_t(-2.0)), ("2050 (T = 0.5)", at_t(0.5))] {
            let outside = if label.starts_with("1800") {
                -1e-3
            } else {
                1e-3
            };
            for p in Planet::ALL {
                let (r1, _) = heliocentric_state_icrs(p, t);
                let (r2, _) = heliocentric_state_icrs(p, t.add_seconds(outside));
                let jump_km = (r2 - r1).length() / 1e3;
                assert!(jump_km.is_finite());
                println!("Table 1 ↔ 2 switch at {label}: {p:?} jumps {jump_km:.0} km");
            }
        }
    }

    #[test]
    fn barycentre_ids() {
        let ids: Vec<_> = Planet::ALL.iter().map(|p| p.barycentre_id()).collect();
        assert_eq!(ids, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
