//! Comet and asteroid orbits from catalogue elements (J2000 ecliptic) (M1 Task A3).
//!
//! JPL SBDB gives heliocentric osculating elements referred to the ecliptic and mean equinox of
//! J2000, with T_p and the epoch as TDB Julian Dates. These build two-body [`Conic`]s about the
//! Sun (GM☉ of DE440, the value uni-physics uses throughout) in that ecliptic frame;
//! [`heliocentric_state_icrs`] rotates their states to ICRS axes. Two-body motion from the
//! osculating epoch ignores planetary perturbations and comets' non-gravitational forces, so it
//! drifts away from the epoch (the card says `Orbit: two-body from epoch YYYY-MM-DD
//! (approximate far from epoch)`).

use glam::DVec3;
use uni_core::{AU, Epoch};

use crate::bodies::GM_SUN_DE440_M3S2;
use crate::frames::ecliptic_j2000_to_icrs;
use crate::kepler::Conic;

/// Julian Date (TDB) of J2000.0 (IERS Conventions 2010).
const JD_J2000: f64 = 2_451_545.0;

/// The epoch of a TDB Julian Date.
fn epoch_from_jd_tdb(jd_tdb: f64) -> Epoch {
    Epoch::from_days_since_j2000(jd_tdb - JD_J2000)
}

/// Heliocentric conic from perihelion elements (q [AU], angles [deg], T_p [JD TDB]).
///
/// Any e ≥ 0 (the near-parabolic and hyperbolic long-period comets included); the frame is the
/// elements' (J2000 ecliptic for SBDB).
pub fn conic_from_perihelion(
    q_au: f64,
    e: f64,
    i_deg: f64,
    node_deg: f64,
    argp_deg: f64,
    tp_jd_tdb: f64,
) -> Conic {
    Conic {
        q_m: q_au * AU,
        e,
        i_rad: i_deg.to_radians(),
        node_rad: node_deg.to_radians(),
        argp_rad: argp_deg.to_radians(),
        tp: epoch_from_jd_tdb(tp_jd_tdb),
        mu_m3s2: GM_SUN_DE440_M3S2,
    }
}

/// Heliocentric conic from mean-anomaly elements (a [AU], angles [deg], epoch [JD TDB]).
///
/// Through [`Conic::from_mean_anomaly`] (e < 1 for asteroids; a hyperbola needs a < 0).
pub fn conic_from_mean_anomaly(
    a_au: f64,
    e: f64,
    i_deg: f64,
    node_deg: f64,
    argp_deg: f64,
    m0_deg: f64,
    epoch_jd_tdb: f64,
) -> Conic {
    Conic::from_mean_anomaly(
        a_au * AU,
        e,
        i_deg.to_radians(),
        node_deg.to_radians(),
        argp_deg.to_radians(),
        m0_deg.to_radians(),
        epoch_from_jd_tdb(epoch_jd_tdb),
        GM_SUN_DE440_M3S2,
    )
}

/// Heliocentric position [m] and velocity [m/s] in ICRS axes of a conic whose elements are in
/// the J2000 ecliptic.
///
/// One Kepler solve ([`Conic::state`]), then the rotation about +x by the J2000 obliquity
/// ([`ecliptic_j2000_to_icrs`]; the 23 mas frame bias is ignored there).
pub fn heliocentric_state_icrs(c: &Conic, t: Epoch) -> (DVec3, DVec3) {
    let (r, v) = c.state(t);
    (ecliptic_j2000_to_icrs(r), ecliptic_j2000_to_icrs(v))
}

#[cfg(test)]
mod tests {
    use uni_core::AU;

    use super::*;
    use crate::bodies::GM_SUN_DE440_M3S2 as GM_SUN;
    use crate::frames::OBLIQUITY_J2000_RAD;

    /// JD 2451545.0 TDB is J2000.0.
    fn at_jd(jd_tdb: f64) -> Epoch {
        Epoch::from_days_since_j2000(jd_tdb - 2_451_545.0)
    }

    /// Specific orbital energy v²/2 − μ/r [J/kg].
    fn energy(r: DVec3, v: DVec3) -> f64 {
        0.5 * v.length_squared() - GM_SUN / r.length()
    }

    /// A Horizons geometric state of `target` from the Sun (centre 10) in `horizons_spots.ron`.
    fn horizons_heliocentric(target: i32) -> (f64, DVec3, DVec3) {
        let path = uni_data::data_dir()
            .join("golden")
            .join("horizons_spots.ron");
        let spots = uni_data::load_ephem_spots(&path)
            .unwrap_or_else(|e| panic!("golden file missing: {} ({e})", path.display()));
        let spot = spots
            .iter()
            .find(|s| s.target == target && s.center == 10)
            .unwrap_or_else(|| panic!("no heliocentric Horizons spot for {target}"));
        let km = |v: [f64; 3]| DVec3::from_array(v) * 1e3;
        (spot.jd_tdb, km(spot.pos_km), km(spot.vel_km_s))
    }

    #[test]
    fn perihelion_elements_units_and_frame() {
        // i = Ω = 0, ω = 90°: perihelion lies on the ecliptic's +y axis, which is
        // (0, cos ε₀, sin ε₀) in ICRS; at T_p the speed is the vis-viva √(μ(1 + e)/q).
        let tp = 2_461_000.5;
        let c = conic_from_perihelion(0.5, 0.6, 0.0, 0.0, 90.0, tp);
        assert_eq!(c.q_m, 0.5 * AU);
        assert_eq!(c.mu_m3s2, GM_SUN);
        assert!((c.argp_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert!(c.tp.seconds_since(at_jd(tp)).abs() < 1e-4);
        let (r, v) = heliocentric_state_icrs(&c, at_jd(tp));
        let (s, co) = OBLIQUITY_J2000_RAD.sin_cos();
        let want = 0.5 * AU * DVec3::new(0.0, co, s);
        assert!((r - want).length() < 1e-3, "{r:?} vs {want:?}");
        let vq = (GM_SUN * 1.6 / (0.5 * AU)).sqrt();
        assert!((v.length() - vq).abs() < 1e-9 * vq);
        // Prograde in the ecliptic: angular momentum along the ecliptic pole (0, −sin ε₀, cos ε₀).
        let pole = r.cross(v).normalize();
        assert!(
            (pole - DVec3::new(0.0, -s, co)).length() < 1e-12,
            "{pole:?}"
        );
    }

    #[test]
    fn mean_anomaly_elements_place_the_body() {
        // M₀ = 0 puts the body at perihelion a(1 − e) at the epoch, M₀ = 180° at aphelion.
        let epoch = 2_461_200.5;
        let peri = conic_from_mean_anomaly(2.0, 0.25, 10.0, 80.0, 70.0, 0.0, epoch);
        let (r, _) = heliocentric_state_icrs(&peri, at_jd(epoch));
        assert!((r.length() - 1.5 * AU).abs() < 1e-3);
        let apo = conic_from_mean_anomaly(2.0, 0.25, 10.0, 80.0, 70.0, 180.0, epoch);
        let (r, _) = heliocentric_state_icrs(&apo, at_jd(epoch));
        assert!((r.length() - 2.5 * AU).abs() < 1e-3);
        // One period later (Kepler's third law) it is back.
        let period = std::f64::consts::TAU * ((2.0 * AU).powi(3) / GM_SUN).sqrt();
        let (r1, _) = heliocentric_state_icrs(&apo, at_jd(epoch).add_seconds(period));
        assert!((r1 - r).length() < 1.0, "{}", (r1 - r).length());
    }

    /// Review Focus 1 for comets: near-parabolic and strongly hyperbolic orbits stay finite and
    /// on their conic ±10 years from perihelion.
    #[test]
    fn hyperbolic_comet_conic_finite() {
        let tp = 2_460_000.5;
        for e in [1.0, 1.0002, 1.5] {
            let c = conic_from_perihelion(0.3, e, 128.9, 61.0, 37.3, tp);
            // Energy μ(e − 1)/(2q) of the conic.
            let want = GM_SUN * (e - 1.0) / (2.0 * 0.3 * AU);
            for days in [-3652.5, -365.25, -1.0, 0.0, 1.0, 365.25, 3652.5] {
                let t = at_jd(tp + days);
                let (r, v) = heliocentric_state_icrs(&c, t);
                assert!(r.is_finite() && v.is_finite(), "e {e} day {days}");
                assert!(r.length() >= 0.3 * AU * (1.0 - 1e-12), "e {e} day {days}");
                let scale = GM_SUN / r.length();
                let err = (energy(r, v) - want).abs() / scale;
                assert!(err < 1e-9, "e {e} day {days}: energy error {err:e} of μ/r");
            }
        }
    }

    /// SBDB's 1P/Halley elements (`comets.ron`) propagated two-body to 1986-03-13 against
    /// Horizons' geometric heliocentric state (JPL#75, DE441), about a month after perihelion.
    ///
    /// The brief's bound is 0.5 % of r. Measured: 5.06 % (6.7 million km, 2.79° in direction),
    /// escalated to the lead (A3 report). The elements cause it, not the propagation. SBDB's row
    /// osculates on JD 2439875.5 (1968-01-20), 18 years earlier. Its T_p is 1986-02-08.47 and its
    /// q is 0.5749 au. Horizons' own state on 1986-03-13 osculates at q 0.5871 au, e 0.96728,
    /// i 162.24°, Ω 58.86° and ω 111.87°, the familiar elements of the 1986 apparition (perihelion
    /// 1986-02-09.46). Planetary perturbations between the two epochs account for the difference.
    /// So the test checks two things. (1) The propagation matches an independent two-body solution
    /// of the same elements (Python: Newton on Kepler's equation, perifocal P/Q vectors, rotation
    /// by ε₀) to 100 m. (2) The Horizons miss stays at its recorded value. It therefore fails if
    /// either changes, including when re-baked elements bring the miss under the brief's bound.
    #[test]
    fn halley_1986_vs_horizons() {
        let path = uni_data::data_dir().join("baked").join("comets.ron");
        let comets = uni_data::load_comets(&path)
            .unwrap_or_else(|e| panic!("comet file missing: {} ({e})", path.display()));
        let h = comets
            .iter()
            .find(|c| c.spkid == 1_000_036)
            .unwrap_or_else(|| panic!("no 1P/Halley in {}", path.display()));
        let c = conic_from_perihelion(h.q_au, h.e, h.i_deg, h.node_deg, h.argp_deg, h.tp_jd_tdb);
        let (jd, r_ref, v_ref) = horizons_heliocentric(1_000_036);
        assert_eq!(jd, 2_446_502.5, "the golden is 1986-03-13 00:00 TDB");
        let (r, v) = heliocentric_state_icrs(&c, at_jd(jd));

        // (1) The independent solution of the same elements [km, ICRS].
        let independent = DVec3::new(-81_444_617.586, -96_122_657.881, -46_928_598.276) * 1e3;
        let dev = (r - independent).length();
        assert!(
            dev < 100.0,
            "{dev} m from the independent two-body solution"
        );

        // (2) The miss against Horizons.
        let rel = (r - r_ref).length() / r_ref.length();
        let angle = r.angle_between(r_ref).to_degrees() * 3600.0;
        let dv = (v - v_ref).length() / v_ref.length();
        println!(
            "Halley 1986-03-13: r = {:.6} au (Horizons {:.6}), |Δr| = {:.0} km = {rel:.4e} of r, \
             direction {angle:.1}″, |Δv|/v = {dv:.3e}; elements epoch JD {}, T_p JD {}; \
             {dev:.3} m from the independent solution",
            r.length() / AU,
            r_ref.length() / AU,
            (r - r_ref).length() / 1e3,
            h.epoch_jd_tdb,
            h.tp_jd_tdb
        );
        const RECORDED_REL: f64 = 5.058e-2;
        assert!(
            rel > 0.005 && (rel - RECORDED_REL).abs() < 1e-4,
            "|Δr|/r = {rel:.4e} is no longer the recorded {RECORDED_REL:e} over the 0.5 % bound"
        );
    }

    /// (1) Ceres from its `asteroids.bin` mean-anomaly elements, two-body from their epoch
    /// (2026-06-09), against Horizons (JPL#48, DE441) at 2026-11-20 00:00 UTC, 164 days later.
    /// Measured: 15 762 km, 4.0e-5 of r (planetary perturbations); bound 1e-4.
    #[test]
    fn ceres_mean_anomaly_vs_horizons() {
        let dir = uni_data::data_dir().join("baked");
        let (records, _) =
            uni_data::load_asteroids(&dir.join("asteroids.bin"), &dir.join("asteroid_names.txt"))
                .unwrap_or_else(|e| panic!("asteroid files missing in {} ({e})", dir.display()));
        let spkid = uni_data::smallbody::asteroid_spkid_from_naif(2_000_001);
        let a = records
            .iter()
            .find(|a| Some(a.spkid) == spkid)
            .unwrap_or_else(|| panic!("no (1) Ceres in asteroids.bin"));
        let c = conic_from_mean_anomaly(
            a.a_au,
            a.e,
            a.i_deg,
            a.node_deg,
            a.argp_deg,
            a.m0_deg,
            a.epoch_jd_tdb,
        );
        let (jd, r_ref, _) = horizons_heliocentric(2_000_001);
        let (r, _) = heliocentric_state_icrs(&c, at_jd(jd));
        let rel = (r - r_ref).length() / r_ref.length();
        println!(
            "Ceres JD {jd}: |Δr| = {:.0} km = {rel:.3e} of r after {:.1} d of two-body motion",
            (r - r_ref).length() / 1e3,
            jd - a.epoch_jd_tdb
        );
        assert!(rel < 1e-4, "|Δr|/r = {rel:e}");
    }
}
