//! Planet positions from JPL's "Approximate Positions of the Planets" (Standish & Williams;
//! https://ssd.jpl.nasa.gov/planets/approx_pos.html) (M1 Task A1). Pluto is in [`crate::pluto`]
//! (review I1); the page no longer lists it.
//!
//! The page's steps (physics research §6): T = (JD_TDB − 2451545.0)/36525; each element
//! x = x₀ + ẋT; ω = ϖ − Ω; M = L − ϖ + bT² + c cos(fT) + s sin(fT) (Table 2b terms for
//! Jupiter–Neptune with Table 2a only), reduced to [−180°, 180°]; Kepler's equation; the
//! orbital-plane position rotated by (ω, I, Ω) into the J2000 ecliptic and by ε₀ into ICRS
//! ([`crate::frames::ecliptic_j2000_to_icrs`]; the page's 23.43928° agrees with IAU 1976's
//! 84381.448″ = 23.4392911° to 0.04″). The positions are of the planetary-system barycentres
//! (EM Bary = Earth–Moon barycentre).
//!
//! Origin of the fit (A1 fix round 1, review I1). The page calls every position heliocentric
//! and does not name the ephemeris it fitted. Neptune's elements nevertheless fit its orbit
//! about the Solar System barycentre (SSB), so [`heliocentric_state_icrs`] subtracts the Sun's
//! SSB state for Neptune only (`ssb_fitted`). Measured against DE441 at the six 1800–2050
//! Horizons spots, Neptune's worst longitude error is 45.1″ read about the Sun and 7.2″ read
//! about the SSB (the page's nominal error is 10″), and its RMS position error 8.7 × 10⁵ km
//! against 1.8 × 10⁵ km. The likely reason: a fixed ellipse about the Sun cannot follow the
//! Sun's own 12–30-year reflex motion about the SSB, 1.15 × 10⁶ km at J2000 (Horizons 10@0),
//! which is 52″ at Neptune. Every other planet stays heliocentric, as the page says: Jupiter's
//! RMS position error is 1.5× smaller about the Sun, and Saturn's and Uranus's differ by ≤ 10 %
//! between the two readings (Uranus: 1.05 × 10⁶ km about the Sun, 0.96 × 10⁶ km about the
//! SSB, but worst longitude 79″ against 114″). Read about the SSB, the inner planets would be
//! off by 800–2 700″. `fit_origin_against_horizons` measures and pins all of this.
//!
//! Table 1 ↔ 2 handover (A1b; a lead ruling, option (b), not from Standish). The blend bands
//! lie OUTSIDE Table 1's window, 1780–1800 (T ∈ [−2.2, −2]) and 2050–2070 (T ∈ [0.5, 0.7]):
//! there the state is (1 − w)·Table 2 + w·Table 1, with w a C¹ smoothstep of the distance
//! outside the window: Table 1 at the window edge, Table 2 from 20 years outside. Over
//! 1800–2050 the state is Table 1 alone, so A1's Horizons tests run against the production
//! state. The hard cut jumped up to 1.28e7 km (Uranus at 1800), a visible pop under time warp.
//! The cost: inside the bands Table 1 is evaluated up to 20 years PAST its stated window.
//! Measured (A1b fix round 2), over those 20 years the two tables diverge by at most about
//! 1.7× their largest in-window difference (worst: Saturn λ 1 233″ at 2070; Jupiter λ 345″ and
//! 359″), so nothing runs away, but the accuracy there is bounded (the blend lies between the
//! tables, so its error is at most the worse table's), not measured against truth: no Horizons
//! spot lies in the bands. [`validity`] reports Table 2 (the weaker table) inside the bands.
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
    /// Table 1 alone: AD 1800–2050, its stated window.
    Table1,
    /// Tables 2a + 2b: 3000 BC – AD 3000. This includes the 20-year blend bands just outside
    /// Table 1's window (1780–1800, 2050–2070), where the state is a Table 1 / Table 2 blend
    /// and Table 1 runs past its stated window (A1b, see the module doc): the weaker table
    /// is reported there so no accuracy label overstates it.
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

/// Width of each Table 1 ↔ 2 handover band outside the ends of Table 1's window [Julian
/// centuries]: 20 Julian years (A1b, lead ruling (b)).
const HANDOVER_BAND_CY: f64 = 0.2;

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

/// The table whose window holds `t`. Inside the blend bands just outside Table 1's window it
/// is Table 2 (the weaker table; the state there is a blend, see the module doc).
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
/// Finite at every finite epoch (review C1); see [`validity`] for the table in use. For a
/// planet whose elements fit its orbit about the SSB (`ssb_fitted`: Neptune) this is the
/// fitted state minus the Sun's SSB state ([`crate::ssb`]), so it costs eight more Kepler
/// solves.
pub fn heliocentric_state_icrs(p: Planet, t: Epoch) -> (DVec3, DVec3) {
    let (r, v) = fitted_state_icrs(p, t);
    if ssb_fitted(p) {
        let (sun_r, sun_v) = crate::ssb::sun_from_ssb_state_icrs(t);
        (r - sun_r, v - sun_v)
    } else {
        (r, v)
    }
}

/// Whether `p`'s elements fit its orbit about the SSB rather than the Sun: Neptune only (see
/// the module doc; A1 fix round 1). The same reading applies to Table 1, Tables 2a + 2b and
/// the frozen-edge extrapolation, so the Table 1 ↔ 2 switch does not move the origin.
pub(crate) const fn ssb_fitted(p: Planet) -> bool {
    matches!(p, Planet::Neptune)
}

/// The page's state of `p` [m, m/s], ICRS axes, about the origin its elements fit: the Sun,
/// or the SSB where [`ssb_fitted`]. In the [`HANDOVER_BAND_CY`] just outside each end of
/// Table 1's window it is the blend of Tables 1 and 2 (A1b, Table 1 past its window); [`crate::ssb`]'s Sun offset, and
/// so Neptune's heliocentric state, are built from these blended states.
pub(crate) fn fitted_state_icrs(p: Planet, t: Epoch) -> (DVec3, DVec3) {
    let t_cy = t.julian_centuries();
    let Some((w, w_rate_per_cy)) = handover_weight(t_cy) else {
        return elements_state_icrs(&mean_elements(p, t_cy));
    };
    let (r1, v1) = elements_state_icrs(&table_elements(p, t_cy, Validity::Table1));
    // Lead ruling (A1b, option (b)), not from Standish: avoids a discontinuity at the table
    // handover. Table 1 is evaluated up to 20 years past its stated window here (measured:
    // the tables diverge by ≤ ~1.7× their largest in-window difference, worst Saturn λ
    // ~1233″ at 2070; the accuracy there is bounded by the worse table, not measured).
    // State = (1 − w)·Table 2 + w·Table 1, and its exact time derivative
    // v = (1 − w)·v₂ + w·v₁ + (dw/dt)·(p₁ − p₂).
    let (r2, v2) = elements_state_icrs(&table_elements(p, t_cy, Validity::Table2));
    let w_dot = w_rate_per_cy / JULIAN_CENTURY_S;
    (
        r2 * (1.0 - w) + r1 * w,
        v2 * (1.0 - w) + v1 * w + (r1 - r2) * w_dot,
    )
}

/// Table 1's weight w in the handover blend at `t_cy` and its rate dw/dT [1/century], or
/// `None` where one table stands alone: inside Table 1's window (Table 1, ends included) and
/// more than [`HANDOVER_BAND_CY`] outside it (Table 2 or the extrapolation). With x the
/// distance outside the window from its nearer end in units of the band, w = 1 − (3x² − 2x³)
/// (the C¹ smoothstep): 1 at the window edge, 0 at the band's outer edge, with zero slope at
/// both, so position and velocity are continuous.
fn handover_weight(t_cy: f64) -> Option<(f64, f64)> {
    let (dist_cy, dist_rate) = if t_cy < TABLE1_T_MIN {
        (TABLE1_T_MIN - t_cy, -1.0)
    } else {
        (t_cy - TABLE1_T_MAX, 1.0)
    };
    if !(0.0 < dist_cy && dist_cy < HANDOVER_BAND_CY) {
        return None;
    }
    let x = dist_cy / HANDOVER_BAND_CY;
    let w = 1.0 - x * x * (3.0 - 2.0 * x);
    let w_rate = -6.0 * x * (1.0 - x) * dist_rate / HANDOVER_BAND_CY;
    Some((w, w_rate))
}

/// The state [m, m/s], ICRS axes, of the orbit `el` describes: the page's steps 4–6.
fn elements_state_icrs(el: &MeanElements) -> (DVec3, DVec3) {
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

/// The elements of `p` at `t_cy` Julian centuries from J2000 from the table in force there.
fn mean_elements(p: Planet, t_cy: f64) -> MeanElements {
    table_elements(p, t_cy, validity_at(t_cy))
}

/// The elements of `p` at `t_cy` Julian centuries from J2000 from the table of `validity`,
/// by the page's steps 1–4.
fn table_elements(p: Planet, t_cy: f64, validity: Validity) -> MeanElements {
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

    /// An error-table entry that the DE441 goldens exceed: (planet, quantity: 0 = λ, 1 = φ,
    /// 2 = ρ, ceiling in the table's unit).
    type Over = (Planet, usize, f64);
    const QUANTITY: [&str; 3] = ["λ [″]", "φ [″]", "ρ [km]"];

    /// How the error-table tests assert the page (A1 fix round 2, review r1 I1).
    ///
    /// The brief's bound: every spot within the page's nominal error. The tests assert exactly
    /// that for every planet and quantity, except the entries in an `*_OVER_*` list. The
    /// goldens (DE441) exceed the page there. Reading the fit about another origin (the SSB, or
    /// the planet's Jacobi centre) does not remove all of them: `fit_origin_against_horizons`
    /// shows that Mercury, Mars λ, Uranus λ and Neptune ρ stay over under every reading. About
    /// the SSB, Saturn ρ and Uranus φ and ρ would come within the page, while Uranus λ grows
    /// from 79″ to 114″. For each listed entry the test asserts:
    /// - that it is over its nominal, so each list holds exactly the entries the goldens
    ///   exceed: an entry back within the page fails the test until it is taken off the list;
    /// - that it is at or below its ceiling: the measured value × 1.1, rounded up to three
    ///   significant figures. That is a regression guard, not a published figure; whether
    ///   these exceedances are accepted is the lead's ruling (fix round 2, concern 1).
    ///
    /// The page says "nominal errors" and does not say whether they are maxima or RMS values
    /// (re-read 2026-11-22). The tests therefore also assert RMS over the spots ≤ nominal,
    /// with its own exact list. This is evidence for the RMS reading, which the page neither
    /// states nor rules out: every 1800–2050 RMS is within the page.
    ///
    /// 1800–2050, spots over the nominal (measured worst / nominal → ceiling): Mercury λ
    /// 17.40″ / 15″ → 19.2″ (2026-11-20), φ 1.043″ / 1″ → 1.15″ (1950), ρ 1 212 / 1 000 km →
    /// 1 340 km (1950); Mars λ 58.77″ / 40″ → 64.7″ (2050); Saturn ρ 1.874e6 / 1.5e6 km →
    /// 2.07e6 km (1900); Uranus λ 79.04″ / 50″ → 87.0″ (2050), φ 2.226″ / 2″ → 2.45″ (1900),
    /// ρ 1.341e6 / 1e6 km → 1.48e6 km (1800); Neptune ρ 2.419e5 / 2e5 km → 2.67e5 km (2000;
    /// read about the SSB, see the module doc).
    const PEAK_OVER_1800_2050: [Over; 9] = [
        (Planet::Mercury, 0, 19.2),
        (Planet::Mercury, 1, 1.15),
        (Planet::Mercury, 2, 1_340.0),
        (Planet::Mars, 0, 64.7),
        (Planet::Saturn, 2, 2_070_000.0),
        (Planet::Uranus, 0, 87.0),
        (Planet::Uranus, 1, 2.45),
        (Planet::Uranus, 2, 1_480_000.0),
        (Planet::Neptune, 2, 267_000.0),
    ];
    /// 3000 BC – AD 3000 (spots AD 1000 and AD 2500), spots over the nominal: Mercury ρ
    /// 1 127.4 / 1 000 km → 1 250 km (AD 1000); Mars ρ 51 372 / 30 000 km → 56 600 km (AD 2500).
    const PEAK_OVER_3000BC_3000AD: [Over; 2] =
        [(Planet::Mercury, 2, 1_250.0), (Planet::Mars, 2, 56_600.0)];
    /// The one RMS over its nominal: Mars ρ over the two Table 2 spots, 39 144 km against
    /// 30 000 km → 43 100 km (20 626 km at AD 1000, 51 372 km at AD 2500). Two spots do not
    /// sample a 6 000-year window.
    const RMS_OVER_3000BC_3000AD: [Over; 1] = [(Planet::Mars, 2, 43_100.0)];

    /// Asserts one measured `value` (a planet's worst or RMS error in quantity `q`) as
    /// described at [`PEAK_OVER_1800_2050`]: ≤ `nominal`, or, for an entry in `over`, above
    /// `nominal` and ≤ its ceiling.
    fn assert_entry(what: &str, p: Planet, q: usize, value: f64, nominal: f64, over: &[Over]) {
        let quantity = QUANTITY[q];
        match over.iter().find(|&&(pp, qq, _)| (pp, qq) == (p, q)) {
            None => assert!(
                value <= nominal,
                "{p:?} {quantity}: {what} {value} > the page's {nominal}"
            ),
            Some(&(_, _, ceiling)) => {
                assert!(
                    value > nominal,
                    "{p:?} {quantity}: {what} {value} is within the page's {nominal} again; take it off the over-nominal list"
                );
                assert!(
                    value <= ceiling,
                    "{p:?} {quantity}: {what} {value} > its ceiling {ceiling} (page: {nominal})"
                );
            }
        }
    }

    /// The heliocentric state [m, m/s] from the elements of `table` alone, with no handover
    /// blend: the page's table itself, evaluated even outside its window. It is the reference
    /// [`handover_bounds_hold`] blends between. For Neptune (SSB-fitted) the Sun's SSB offset
    /// is built from the same table's states by the closed form of [`crate::ssb`], repeated
    /// here. Wherever one table stands alone it equals [`heliocentric_state_icrs`] bit for bit,
    /// asserted by [`table_state_is_the_apps_outside_the_bands`] (A1b review r0 I2), so a
    /// change to [`crate::ssb`] or [`fitted_state_icrs`] that this copy misses fails the tests.
    fn table_heliocentric_state_icrs(p: Planet, t: Epoch, table: Validity) -> (DVec3, DVec3) {
        use crate::bodies::{GM_SUN_DE440_M3S2, system_gm_m3s2};
        let state =
            |q: Planet| elements_state_icrs(&table_elements(q, t.julian_centuries(), table));
        let (r, v) = state(p);
        if !ssb_fitted(p) {
            return (r, v);
        }
        let (mut total, mut moment, mut moment_rate) =
            (GM_SUN_DE440_M3S2, DVec3::ZERO, DVec3::ZERO);
        for q in Planet::ALL {
            let gm = system_gm_m3s2(q.barycentre_id()).expect("system GM");
            let (rq, vq) = state(q);
            moment += gm * rq;
            moment_rate += gm * vq;
            if !ssb_fitted(q) {
                total += gm;
            }
        }
        let (sun_r, sun_v) = (-moment / total, -moment_rate / total);
        (r - sun_r, v - sun_v)
    }

    /// Compares every Horizons spot of each barycentre (1..=8 @ 10) whose JD passes `keep`
    /// with the app's [`heliocentric_state_icrs`], prints the per-spot, worst and RMS errors
    /// next to the JPL table, and asserts the expected `table`, that no spot is in a blend
    /// band, and the bounds described at [`PEAK_OVER_1800_2050`].
    fn check_against_horizons(
        keep: impl Fn(f64) -> bool,
        table: Validity,
        jpl: &[[f64; 3]; 8],
        peak_over: &[Over],
        rms_over: &[Over],
    ) {
        let mut worst = [[0.0f64; 3]; 8];
        let mut sum_sq = [[0.0f64; 3]; 8];
        let mut count = [0usize; 8];
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
                assert_eq!(handover_weight(t.julian_centuries()), None, "{p:?} in a band");
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
                    sum_sq[k][q] += err[q] * err[q];
                }
                count[k] += 1;
            }
        }
        let rms = |k: usize, q: usize| (sum_sq[k][q] / count[k] as f64).sqrt();
        println!(
            "planet   | worst λ, RMS λ / JPL ″          | worst φ, RMS φ / JPL ″      | worst ρ, RMS ρ / JPL km"
        );
        for (k, p) in Planet::ALL.into_iter().enumerate() {
            let mark = |q: usize| match (rms(k, q) > jpl[k][q], worst[k][q] > jpl[k][q]) {
                (true, _) => " RMS OVER",
                (false, true) => " PEAK OVER",
                _ => "",
            };
            println!(
                "{:8} | {:7.2}, {:7.2} / {:5.0}{:10} | {:6.2}, {:6.2} / {:3.0}{:10} | {:9.0}, {:9.0} / {:8.0}{}",
                format!("{p:?}"),
                worst[k][0],
                rms(k, 0),
                jpl[k][0],
                mark(0),
                worst[k][1],
                rms(k, 1),
                jpl[k][1],
                mark(1),
                worst[k][2],
                rms(k, 2),
                jpl[k][2],
                mark(2)
            );
        }
        println!("worst |Δv|/|v| {worst_dv:.2e}");
        for (k, p) in Planet::ALL.into_iter().enumerate() {
            for q in 0..3 {
                assert_entry("worst", p, q, worst[k][q], jpl[k][q], peak_over);
                assert_entry("RMS", p, q, rms(k, q), jpl[k][q], rms_over);
            }
        }
        // Velocities: derivative of the mean ellipse, so within the position error's order.
        assert!(worst_dv < 1e-2, "{worst_dv}");
    }

    /// Every Horizons barycentre spot in AD 1800–2050 against the JPL 1800–2050 error table,
    /// asserted as described at [`PEAK_OVER_1800_2050`]. Target 399 is not the EMB and is not
    /// used. The state is the app's [`heliocentric_state_icrs`], which over 1800–2050 is
    /// Table 1 alone (A1b, option (b): the blend bands lie outside the window).
    #[test]
    fn standish_within_error_table() {
        check_against_horizons(
            |jd| (2_378_495.0..=2_469_807.5).contains(&jd),
            Validity::Table1,
            &JPL_1800_2050,
            &PEAK_OVER_1800_2050,
            &[],
        );
    }

    /// The spots at JD 2086307.5 (AD 1000-01-01, Julian calendar) and 2634166.5
    /// (AD 2500-01-01) against the JPL 3000 BC – AD 3000 table, read the same way. Both are
    /// centuries from the handover bands.
    #[test]
    fn standish_table2_window() {
        check_against_horizons(
            |jd| jd == 2_086_307.5 || jd == 2_634_166.5,
            Validity::Table2,
            &JPL_3000BC_3000AD,
            &PEAK_OVER_3000BC_3000AD,
            &RMS_OVER_3000BC_3000AD,
        );
    }

    /// A1b review r0 I2: the test-only single-table state ([`table_heliocentric_state_icrs`],
    /// whose Neptune path repeats [`crate::ssb`]'s closed form) is the app's
    /// [`heliocentric_state_icrs`] bit for bit wherever the table it is given stands alone:
    /// Table 1 across its window (both ends, just inside them, and deep inside), Table 2 just
    /// beyond each band and across its own window, and the frozen-edge extrapolation, for
    /// every planet.
    #[test]
    fn table_state_is_the_apps_outside_the_bands() {
        let one_day_cy = 1.0 / 36_525.0;
        let epochs_cy = [
            TABLE1_T_MIN,
            TABLE1_T_MIN + one_day_cy,
            -1.0,
            0.0,
            0.269, // ≈ 2026-11-20
            TABLE1_T_MAX - one_day_cy,
            TABLE1_T_MAX,
            TABLE1_T_MIN - BAND_CY - one_day_cy,
            TABLE1_T_MAX + BAND_CY + one_day_cy,
            -30.0,
            5.0,
            TABLE2_T_MIN,
            TABLE2_T_MAX,
            -1e4,
            1e4,
        ];
        for t_cy in epochs_cy {
            let t = at_cy(t_cy);
            let t_cy = t.julian_centuries();
            assert_eq!(handover_weight(t_cy), None, "T {t_cy} is in a band");
            for p in Planet::ALL {
                assert_eq!(
                    table_heliocentric_state_icrs(p, t, validity_at(t_cy)),
                    heliocentric_state_icrs(p, t),
                    "{p:?} at T {t_cy} ({:?})",
                    validity_at(t_cy)
                );
            }
        }
    }

    /// The evidence for [`ssb_fitted`] (A1 fix round 1, review I1). At the six 1800–2050 spots
    /// the page's (fitted) state of Jupiter–Neptune is compared with Horizons read about the
    /// Sun (x@10) and about the SSB (x@10 minus the SSB's heliocentric position, built from
    /// Horizons 1..=9 @ 10 with the DE440 system GMs; Pluto is missing at 1800 and 2050,
    /// 43 km). That construction matches Horizons 10@0 to ≤ 126 km at 1950, 2000 and 2026 (it
    /// leaves out the asteroids). Asserted, as measured (ratio of the RMS position errors,
    /// about the Sun / about the SSB): Neptune 4.83, so it reads about the SSB; Jupiter 0.65;
    /// Saturn 1.02 and Uranus 1.10, too close to leave the page's heliocentric reading. Also
    /// asserted: which page entries each planet exceeds about the Sun, the SSB and its Jacobi
    /// centre (fix round 2; see the comment at `expected_over`). The page's state is Table 1's
    /// alone at every spot (the app's fitted state: the blend bands lie outside 1800–2050).
    #[test]
    fn fit_origin_against_horizons() {
        use crate::bodies::{GM_SUN_DE440_M3S2, system_gm_m3s2};
        let gm = |id: i32| system_gm_m3s2(id).expect("system GM");
        // The page's fitted state (about the Sun, or the SSB for Neptune); over 1800–2050
        // it is Table 1 alone.
        let table1_fitted_state = fitted_state_icrs;
        let in_table1 = |jd: f64| (2_378_495.0..=2_469_807.5).contains(&jd);
        let spots: Vec<Vec<_>> = (1..=9)
            .map(|id| {
                horizons_spots(id, 10)
                    .into_iter()
                    .filter(|s| in_table1(s.jd_tdb))
                    .collect()
            })
            .collect();
        let jds: Vec<f64> = spots[0].iter().map(|s| s.jd_tdb).collect();
        assert_eq!(jds.len(), 6);
        // The SSB relative to the Sun at each spot, from the Horizons states.
        let ssb_from_sun: Vec<DVec3> = jds
            .iter()
            .map(|&jd| {
                let (mut moment, mut total) = (DVec3::ZERO, GM_SUN_DE440_M3S2);
                for (k, body) in spots.iter().enumerate() {
                    if let Some(s) = body.iter().find(|s| s.jd_tdb == jd) {
                        moment += gm(k as i32 + 1) * pos_m(s);
                        total += gm(k as i32 + 1);
                    }
                }
                moment / total
            })
            .collect();
        for s in horizons_spots(10, 0) {
            let k = jds
                .iter()
                .position(|&jd| jd == s.jd_tdb)
                .expect("10@0 spot");
            let miss_km = (-ssb_from_sun[k] - pos_m(&s)).length() / 1e3;
            assert!(
                miss_km < 200.0,
                "JD {}: SSB construction off {miss_km} km",
                s.jd_tdb
            );
        }
        // (planet, asserted range of RMS(about Sun) / RMS(about SSB)).
        let cases = [
            (Planet::Jupiter, 0.55..0.75),
            (Planet::Saturn, 0.9..1.15),
            (Planet::Uranus, 0.95..1.25),
            (Planet::Neptune, 4.3..5.5),
        ];
        for (p, expected) in cases {
            let (mut sq_sun, mut sq_ssb) = (0.0, 0.0);
            for (k, &jd) in jds.iter().enumerate() {
                let truth = pos_m(&spots[p as usize][k]);
                assert_eq!(spots[p as usize][k].jd_tdb, jd);
                let (fitted, _) = table1_fitted_state(p, epoch_of(jd));
                let (about_sun, about_ssb) = (
                    (fitted - truth).length() / 1e3,
                    (fitted - (truth - ssb_from_sun[k])).length() / 1e3,
                );
                println!(
                    "{p:?} JD {jd:.4}: |Δr| about the Sun {about_sun:9.0} km, about the SSB {about_ssb:9.0} km"
                );
                sq_sun += about_sun * about_sun;
                sq_ssb += about_ssb * about_ssb;
            }
            let (rms_sun, rms_ssb) = ((sq_sun / 6.0).sqrt(), (sq_ssb / 6.0).sqrt());
            let ratio = rms_sun / rms_ssb;
            println!(
                "{p:?}: RMS |Δr| about the Sun {rms_sun:.0} km, about the SSB {rms_ssb:.0} km, ratio {ratio:.2}"
            );
            assert!(expected.contains(&ratio), "{p:?}: ratio {ratio}");
            assert_eq!(ssb_fitted(p), ratio > 2.0, "{p:?}");
        }
        // A1 fix round 2 (review r1 I1): which of the page's 1800–2050 entries the spots exceed
        // when the page's state is read about each candidate origin: the Sun, the SSB, and the
        // planet's Jacobi centre (the barycentre of the Sun and the planets inside its orbit,
        // from the same Horizons states and DE440 GMs; for Mercury it is the Sun). Asserted as
        // measured, one string per reading (Sun, SSB, Jacobi) of the quantities over the page.
        // No single origin, and no per-planet choice, brings every entry within the page:
        // Mercury's three and Mars λ are over about the Sun and the Jacobi centre (≤ 750 km
        // from the Sun for Mars) and far over about the SSB; Uranus λ is over under all three
        // (79″, 114″, 127″ against 50″); Neptune ρ under all three. Saturn meets the page about
        // the SSB (λ 588″ of 600″, ρ 0.91 × 10⁶ of 1.5 × 10⁶ km) and not about the Sun (ρ
        // 1.87 × 10⁶ km). Both stay heliocentric (the page's word; RMS ratios 1.02 and 1.10
        // above, and Uranus λ is worse about the SSB); that is the lead's call (fix round 2,
        // concern 1).
        let expected_over: [[&str; 3]; 8] = [
            ["λφρ", "λφρ", "λφρ"], // Mercury
            ["", "λφρ", ""],       // Venus
            ["", "λφρ", ""],       // EM Bary
            ["λ", "λφρ", "λ"],     // Mars
            ["", "ρ", ""],         // Jupiter
            ["ρ", "", "λ"],        // Saturn
            ["λφρ", "λ", "λρ"],    // Uranus
            ["λφρ", "ρ", "ρ"],     // Neptune
        ];
        for (k, p) in Planet::ALL.into_iter().enumerate() {
            let mut worst = [[0.0f64; 3]; 3]; // [reading][quantity]
            for (j, &jd) in jds.iter().enumerate() {
                assert_eq!(spots[k][j].jd_tdb, jd);
                let (mut moment, mut total) = (DVec3::ZERO, GM_SUN_DE440_M3S2);
                for (inner, body) in spots.iter().enumerate().take(k) {
                    assert_eq!(body[j].jd_tdb, jd);
                    moment += gm(inner as i32 + 1) * pos_m(&body[j]);
                    total += gm(inner as i32 + 1);
                }
                let (fitted, _) = table1_fitted_state(p, epoch_of(jd));
                let origins = [DVec3::ZERO, ssb_from_sun[j], moment / total];
                for (reading, origin) in origins.into_iter().enumerate() {
                    let err = ecliptic_errors(fitted, pos_m(&spots[k][j]) - origin);
                    for q in 0..3 {
                        worst[reading][q] = worst[reading][q].max(err[q].abs());
                    }
                }
            }
            let over: Vec<String> = worst
                .iter()
                .map(|w| {
                    (0..3)
                        .filter(|&q| w[q] > JPL_1800_2050[k][q])
                        .map(|q| ["λ", "φ", "ρ"][q])
                        .collect()
                })
                .collect();
            for (reading, w) in ["Sun", "SSB", "Jacobi"].iter().zip(worst) {
                println!(
                    "{p:?} about the {reading}: worst λ {:.2}″ φ {:.2}″ ρ {:.0} km",
                    w[0], w[1], w[2]
                );
            }
            assert_eq!(
                over, expected_over[k],
                "{p:?}: over the page (Sun, SSB, Jacobi)"
            );
        }
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
    /// T ∈ [−50, 10]. Also prints how far each planet moves in the 1 ms across each end of
    /// Table 1's window; before A1b's blend the hard cut jumped up to 1.28e7 km there (A1
    /// report). The bound on that motion is asserted by [`handover_is_continuous`].
    #[test]
    fn validity_windows_and_handover_motion() {
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
                let jump_m = (r2 - r1).length();
                assert!(jump_m.is_finite());
                println!("Table 1 ↔ 2 handover at {label}: {p:?} moves {jump_m:.3} m in 1 ms");
            }
        }
    }

    /// The handover bands (lead ruling (b)): the 20 Julian years (0.2 cy) just outside each
    /// end of Table 1's window, pinned here as the ruling states it.
    const BAND_CY: f64 = 0.2;

    /// The epoch `t_cy` Julian centuries from J2000.
    fn at_cy(t_cy: f64) -> Epoch {
        Epoch::from_days_since_j2000(t_cy * 36_525.0)
    }

    /// A1b: no jump where the tables hand over. At both ends of Table 1's window and at each
    /// band's outer edge, for every planet (heliocentric, so Neptune's SSB path is in it), the
    /// position one second either side of the edge is within |v|·1 s + 1 m of the position at
    /// the edge. The hard cut jumped 1 260 km (Mercury) to 1.28e7 km (Uranus) (A1 report).
    #[test]
    fn handover_is_continuous() {
        let edges = [
            TABLE1_T_MIN - BAND_CY,
            TABLE1_T_MIN,
            TABLE1_T_MAX,
            TABLE1_T_MAX + BAND_CY,
        ];
        let mut worst_excess_m = f64::NEG_INFINITY;
        for t_cy in edges {
            let t = at_cy(t_cy);
            for p in Planet::ALL {
                let (r, v) = heliocentric_state_icrs(p, t);
                let bound_m = v.length() * 1.0 + 1.0;
                for side_s in [-1.0, 1.0] {
                    let (r_side, _) = heliocentric_state_icrs(p, t.add_seconds(side_s));
                    let step_m = (r_side - r).length();
                    worst_excess_m = worst_excess_m.max(step_m - v.length());
                    assert!(
                        step_m < bound_m,
                        "{p:?} at T {t_cy} ({side_s:+} s): moved {step_m:.3} m, bound {bound_m:.3} m"
                    );
                }
            }
        }
        println!("worst |Δr| − |v|·1 s over the handover edges: {worst_excess_m:.2e} m");
    }

    /// A1b: inside the bands (outside Table 1's window) the velocity is the time derivative of the blended position,
    /// including the blend's own term (dw/dt)·(p₁ − p₂). Central differences over ±60 s agree
    /// to 1e-6 relative, for every planet, at the bands' quarter points and 30 s inside each
    /// band end, so the outermost samples straddle the window edge and the band's outer edge.
    #[test]
    fn handover_velocity_is_derivative() {
        let half_step_s = 60.0;
        let near_cy = 30.0 / JULIAN_CENTURY_S;
        let mut worst_rel = 0.0f64;
        for (edge, inward) in [(TABLE1_T_MIN, -1.0), (TABLE1_T_MAX, 1.0)] {
            for depth_cy in [
                near_cy,
                0.25 * BAND_CY,
                0.5 * BAND_CY,
                0.75 * BAND_CY,
                BAND_CY - near_cy,
            ] {
                let t = at_cy(edge + inward * depth_cy);
                for p in Planet::ALL {
                    let (_, v) = heliocentric_state_icrs(p, t);
                    let (r_minus, _) = heliocentric_state_icrs(p, t.add_seconds(-half_step_s));
                    let (r_plus, _) = heliocentric_state_icrs(p, t.add_seconds(half_step_s));
                    let numeric = (r_plus - r_minus) / (2.0 * half_step_s);
                    let rel = (v - numeric).length() / v.length();
                    worst_rel = worst_rel.max(rel);
                    assert!(
                        rel < 1e-6,
                        "{p:?} at T {:.9}: |v − Δr/Δt|/|v| {rel:e}",
                        edge + inward * depth_cy
                    );
                }
            }
        }
        println!("worst |v − Δr/Δt|/|v| in the bands: {worst_rel:.2e}");
    }

    /// A1b: the blend keeps the page's bounds. Across both bands (41 epochs each, ends
    /// included) every planet's blended heliocentric position lies on the segment between
    /// Table 1's and Table 2's (|p − p₁| + |p − p₂| − |p₁ − p₂| < 1 m; Neptune too, as the
    /// Sun's SSB offset is linear in the states), so its error against any truth is at most
    /// the worse table's. No Horizons spot lies in the bands (1780–1800, 2050–2070: asserted),
    /// so the state there is bounded this way, not measured against truth; there is no
    /// over-the-page spot to report. The in-window spots, where Table 1 alone is the app's
    /// state, are [`standish_within_error_table`]'s. Also prints the largest table
    /// divergence over the bands for the report.
    #[test]
    fn handover_bounds_hold() {
        let mut worst_off_m = 0.0f64;
        let mut max_divergence_km = 0.0f64;
        for (edge, outward) in [(TABLE1_T_MIN, -1.0), (TABLE1_T_MAX, 1.0)] {
            for k in 0..=40 {
                let t_cy = edge + outward * BAND_CY * f64::from(k) / 40.0;
                let t = at_cy(t_cy);
                for p in Planet::ALL {
                    let (r, _) = heliocentric_state_icrs(p, t);
                    let (r1, _) = table_heliocentric_state_icrs(p, t, Validity::Table1);
                    let (r2, _) = table_heliocentric_state_icrs(p, t, Validity::Table2);
                    let off_m = (r - r1).length() + (r - r2).length() - (r1 - r2).length();
                    worst_off_m = worst_off_m.max(off_m);
                    max_divergence_km = max_divergence_km.max((r1 - r2).length() / 1e3);
                    assert!(
                        off_m < 1.0,
                        "{p:?} at T {t_cy}: {off_m} m off the Table 1–2 segment"
                    );
                }
            }
        }
        println!("worst |p − p₁| + |p − p₂| − |p₁ − p₂| in the bands: {worst_off_m:.2e} m");
        println!("largest |Table 1 − Table 2| in the bands: {max_divergence_km:.3e} km");
        // The brief's claim that the bands hold no Horizons spot, which is why no band spot
        // is checked against an error table.
        let in_band = |jd: f64| handover_weight(epoch_of(jd).julian_centuries()).is_some();
        let in_bands: Vec<f64> = (1..=9)
            .flat_map(|id| horizons_spots(id, 10))
            .map(|s| s.jd_tdb)
            .filter(|&jd| in_band(jd))
            .collect();
        assert!(in_bands.is_empty(), "Horizons spots in the bands: {in_bands:?}");
    }

    #[test]
    fn barycentre_ids() {
        let ids: Vec<_> = Planet::ALL.iter().map(|p| p.barycentre_id()).collect();
        assert_eq!(ids, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
