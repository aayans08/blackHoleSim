//! Comet and asteroid orbits from catalogue elements (J2000 ecliptic) (M1 Task A3).
//!
//! JPL SBDB gives heliocentric osculating elements referred to the ecliptic and mean equinox of
//! J2000, with T_p and the epoch as TDB Julian Dates. These build two-body [`Conic`]s about the
//! Sun (GM☉ of DE440, the value uni-physics uses throughout) in that ecliptic frame;
//! [`heliocentric_state_icrs`] rotates their states to ICRS axes.
//!
//! Two-body motion from the osculating epoch ignores planetary perturbations and comets'
//! non-gravitational forces, so it drifts away from the epoch. For 1P/Halley, whose SBDB row
//! osculates in 1968, it is 5.06 % of r off on 1986-03-13 (`halley_1986_vs_horizons`).
//! [`perturbed_state_icrs`] integrates the eight planets' pulls from the epoch instead (3.8e-5
//! of r there), and [`osculate_at`] turns that into a fresh conic near the date shown, so the cheap
//! two-body path is accurate again around it. Neither models non-gravitational forces (a comet's
//! outgassing), which shift a periodic comet's perihelion by days per return after its
//! elements' epoch (Halley: about 4 days per orbit, Yeomans & Kiang 1981, MNRAS 197, 633).

use glam::{DMat3, DVec3};
use uni_core::{AU, DAY, Epoch};

use crate::bodies::{GM_SUN_DE440_M3S2, system_gm_m3s2};
use crate::frames::{OBLIQUITY_J2000_RAD, ecliptic_j2000_to_icrs};
use crate::kepler::Conic;
use crate::standish::{self, Planet};

/// Julian Date (TDB) of J2000.0 (IERS Conventions 2010).
const JD_J2000: f64 = 2_451_545.0;

/// The epoch of a TDB Julian Date.
pub(crate) fn epoch_from_jd_tdb(jd_tdb: f64) -> Epoch {
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

/// RK4 step as a fraction of the local dynamical time √(d³/GM), d the distance to the Sun or to
/// any perturbing planet.
pub const STEP_ETA: f64 = 1.0e-3;
/// Longest step [s]: 2 days, so the planets' motion (Mercury: 88 days) is resolved in the
/// indirect terms even where the small body is far out and slow. With 10 days, Mercury's
/// indirect term alone moved Halley's 1986 position by 4 km between two step sizes.
/// Together with [`STEP_ETA`] it keeps the integrator's own error under 1 km over Halley's 18
/// years from 1968 to 1986 (`perturbed_integrator_reproduces_two_body`).
pub const MAX_STEP_S: f64 = 2.0 * DAY;
/// Step cap: at Halley's rate (6848 steps for 18 years) it covers about 5 × 10⁴ years. Beyond
/// it (or for a body falling into the Sun, where the step shrinks to 0) the integration stops
/// and returns NaN.
const MAX_STEPS: usize = 20_000_000;

/// Position [m] of each perturber at `t`, heliocentric ICRS (Standish; the system barycentre).
fn perturber_positions(perturbers: &[Planet], t: Epoch) -> [DVec3; 8] {
    let mut out = [DVec3::ZERO; 8];
    for (slot, &p) in out.iter_mut().zip(perturbers) {
        *slot = standish::heliocentric_state_icrs(p, t).0;
    }
    out
}

/// Heliocentric acceleration [m/s²] at `x`: the Sun's pull plus each planet's direct pull and
/// the indirect term from the Sun's acceleration toward it (heliocentric frame is not inertial):
/// a = −μ☉x/|x|³ + Σ GMₚ[(xₚ − x)/|xₚ − x|³ − xₚ/|xₚ|³], the gradient of the direct and
/// indirect parts of the heliocentric disturbing function (Murray & Dermott 1999, *Solar System
/// Dynamics*, ch. 6).
fn acceleration(x: DVec3, mu_sun: f64, planets: &[DVec3], gms: &[f64]) -> DVec3 {
    let r2 = x.length_squared();
    let mut a = -mu_sun * x / (r2 * r2.sqrt());
    for (&xp, &gm) in planets.iter().zip(gms) {
        let d = xp - x;
        let d2 = d.length_squared();
        let p2 = xp.length_squared();
        a += gm * (d / (d2 * d2.sqrt()) - xp / (p2 * p2.sqrt()));
    }
    a
}

/// Cowell integration (classical RK4) of a heliocentric ICRS state from `t0` to `t1` under the
/// Sun (`mu_sun`) and `perturbers`, with the step rule above scaled by `step_scale` (1 in use;
/// tests halve it). Backward in time too. Returns the state and the number of RK4 steps.
fn integrate(
    (r0, v0): (DVec3, DVec3),
    mu_sun: f64,
    (t0, t1): (Epoch, Epoch),
    perturbers: &[Planet],
    step_scale: f64,
) -> ((DVec3, DVec3), usize) {
    const FAILED: (DVec3, DVec3) = (DVec3::NAN, DVec3::NAN);
    let total = t1.seconds_since(t0);
    if !total.is_finite() || !r0.is_finite() || !v0.is_finite() {
        return (FAILED, 0);
    }
    let (eta, max_step) = (STEP_ETA * step_scale, MAX_STEP_S * step_scale);
    let n = perturbers.len().min(8);
    let gms: Vec<f64> = perturbers[..n]
        .iter()
        .map(|p| system_gm_m3s2(p.barycentre_id()).unwrap_or(0.0))
        .collect();
    let at = |s: f64| perturber_positions(&perturbers[..n], t0.add_seconds(s));
    let accel = |x: DVec3, planets: &[DVec3; 8]| acceleration(x, mu_sun, &planets[..n], &gms);

    let dir = total.signum();
    let (mut r, mut v) = (r0, v0);
    let mut s = 0.0;
    let mut planets_now = at(0.0);
    for steps in 0..MAX_STEPS {
        let remaining = total - s;
        if remaining.abs() <= 1e-6 {
            return ((r, v), steps);
        }
        // Local dynamical times of the Sun and of every planet; the shortest sets the step.
        let mut t_dyn = (r.length_squared().powf(1.5) / mu_sun).sqrt();
        for (&xp, &gm) in planets_now[..n].iter().zip(&gms) {
            if gm > 0.0 {
                t_dyn = t_dyn.min(((xp - r).length_squared().powf(1.5) / gm).sqrt());
            }
        }
        let h = dir * (eta * t_dyn).min(max_step).min(remaining.abs());
        let planets_half = at(s + 0.5 * h);
        let planets_end = at(s + h);
        let k1v = accel(r, &planets_now);
        let k1r = v;
        let k2v = accel(r + 0.5 * h * k1r, &planets_half);
        let k2r = v + 0.5 * h * k1v;
        let k3v = accel(r + 0.5 * h * k2r, &planets_half);
        let k3r = v + 0.5 * h * k2v;
        let k4v = accel(r + h * k3r, &planets_end);
        let k4r = v + h * k3v;
        r += h / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        s += h;
        planets_now = planets_end;
        if !r.is_finite() || !v.is_finite() || h == 0.0 {
            break;
        }
    }
    (FAILED, MAX_STEPS)
}

/// Heliocentric position [m] and velocity [m/s] in ICRS axes at `t` of the small body whose
/// osculating heliocentric conic `c` (J2000 ecliptic, as this module builds it) holds at
/// `epoch`, with the planets' perturbations.
///
/// Cowell's method: the state at `epoch` ([`heliocentric_state_icrs`]) is integrated with
/// classical RK4 under the Sun (`c.mu_m3s2`) and the eight planetary-system barycentres of
/// [`standish`] with their DE440 GMs ([`system_gm_m3s2`]), direct and indirect terms. Left out:
/// Pluto, the asteroids, relativity and non-gravitational forces (see the module doc). Measured
/// against Horizons (`halley_1986_vs_horizons`, `ceres_mean_anomaly_vs_horizons`): Halley,
/// 1968 → 1986-03-13, 4990 km (3.8e-5 of r) against 5.06 % for two-body; Ceres over 164 days,
/// 27 km (6.7e-8) against 15 762 km (4.0e-5).
///
/// Cost: Halley's 18 years take 6848 RK4 steps, each with the eight Standish states at two new
/// times (about 1.1 × 10⁵ in all), 0.17 s in the unoptimised test build (release unmeasured).
/// So this is not a per-frame call: re-osculate with [`osculate_at`] when the shown date moves
/// far from the conic's epoch. NaN when the integration fails (a non-finite input, or a body
/// that falls into the Sun).
pub fn perturbed_state_icrs(c: &Conic, epoch: Epoch, t: Epoch) -> (DVec3, DVec3) {
    integrate(
        heliocentric_state_icrs(c, epoch),
        c.mu_m3s2,
        (epoch, t),
        &Planet::ALL,
        1.0,
    )
    .0
}

/// The osculating heliocentric conic (J2000 ecliptic, same μ) at `t` of the body whose conic
/// `c` holds at `epoch`: [`perturbed_state_icrs`] at `t`, rotated back from ICRS to the J2000
/// ecliptic and passed to [`Conic::from_state`].
///
/// Two-body motion from the result ([`heliocentric_state_icrs`]) reproduces the perturbed state
/// at `t` and stays close to it near `t`, so a caller re-osculates when the shown date moves
/// far from the conic's epoch, not every frame. Halley from its
/// 1968 SBDB elements (q 0.574864 au, e 0.967936) to 1986-03-13 gives q 0.587114 au,
/// e 0.967274, against q 0.587104, e 0.967276 for the osculating orbit of Horizons' state that
/// day; two-body from it stays within 1.3e-6 of r of the perturbed path 30 days either side
/// (`osculate_at_reproduces_horizons_orbit`).
pub fn osculate_at(c: &Conic, epoch: Epoch, t: Epoch) -> Conic {
    let (r, v) = perturbed_state_icrs(c, epoch, t);
    // ICRS → J2000 ecliptic: the inverse of `ecliptic_j2000_to_icrs`, about +x by −ε₀.
    let to_ecliptic = DMat3::from_rotation_x(-OBLIQUITY_J2000_RAD);
    Conic::from_state(to_ecliptic * r, to_ecliptic * v, c.mu_m3s2, t)
}

#[cfg(test)]
mod tests {
    use uni_core::AU;

    use super::*;
    use crate::bodies::GM_SUN_DE440_M3S2 as GM_SUN;
    use crate::frames::OBLIQUITY_J2000_RAD;

    /// JD 2451545.0 TDB is J2000.0.
    fn at_jd(jd_tdb: f64) -> Epoch {
        epoch_from_jd_tdb(jd_tdb)
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

    /// 1P/Halley's SBDB row in `comets.ron`: its conic and osculating epoch.
    fn halley() -> (Conic, Epoch) {
        let path = uni_data::data_dir().join("baked").join("comets.ron");
        let comets = uni_data::load_comets(&path)
            .unwrap_or_else(|e| panic!("comet file missing: {} ({e})", path.display()));
        let h = comets
            .iter()
            .find(|c| c.spkid == 1_000_036)
            .unwrap_or_else(|| panic!("no 1P/Halley in {}", path.display()));
        assert_eq!(
            h.epoch_jd_tdb, 2_439_875.5,
            "the row osculates on 1968-01-20"
        );
        let c = conic_from_perihelion(h.q_au, h.e, h.i_deg, h.node_deg, h.argp_deg, h.tp_jd_tdb);
        (c, at_jd(h.epoch_jd_tdb))
    }

    /// SBDB's 1P/Halley elements (`comets.ron`) propagated to 1986-03-13 against Horizons'
    /// geometric heliocentric state (JPL#75, DE441), about a month after perihelion. The brief's
    /// bound is 0.5 % of r.
    ///
    /// SBDB's row osculates on JD 2439875.5 (1968-01-20), 18 years earlier, with T_p
    /// 1986-02-08.47 and q 0.5749 au; Horizons' state on 1986-03-13 osculates at q 0.5871 au
    /// (perihelion 1986-02-09.46). Planetary perturbations between the two epochs account for the
    /// difference (A3 review r0 I2).
    ///
    /// (1) Two-body from the 1968 epoch misses by 5.06 % of r (6.7 million km, 2.79°). That is
    /// pinned as a record of the data's limit: it fails if the elements change. Its propagation
    /// matches a two-body solution of the same elements written separately in Python (Newton on
    /// Kepler's equation, perifocal P/Q vectors, rotation by ε₀; the script is not in the repo,
    /// so the vector below is a regression pin of this code's output, not an independent check)
    /// to 100 m.
    /// (2) [`perturbed_state_icrs`], the eight planets integrated from the same 1968 state, meets
    /// the brief's 0.5 % bound with room to spare: measured 4990 km, 3.8e-5 of r, 4.5″.
    #[test]
    fn halley_1986_vs_horizons() {
        let (c, epoch) = halley();
        let (jd, r_ref, v_ref) = horizons_heliocentric(1_000_036);
        assert_eq!(jd, 2_446_502.5, "the golden is 1986-03-13 00:00 TDB");
        let t = at_jd(jd);
        let report = |label: &str, r: DVec3, v: DVec3| -> f64 {
            let rel = (r - r_ref).length() / r_ref.length();
            println!(
                "Halley 1986-03-13, {label}: r = {:.6} au (Horizons {:.6}), |Δr| = {:.0} km = \
                 {rel:.4e} of r, direction {:.1}″, |Δv|/v = {:.3e}",
                r.length() / AU,
                r_ref.length() / AU,
                (r - r_ref).length() / 1e3,
                r.angle_between(r_ref).to_degrees() * 3600.0,
                (v - v_ref).length() / v_ref.length()
            );
            rel
        };

        // (1) Two-body from 1968, and the separate solution of the same elements [km, ICRS].
        let (r, v) = heliocentric_state_icrs(&c, t);
        let independent = DVec3::new(-81_444_617.586, -96_122_657.881, -46_928_598.276) * 1e3;
        let dev = (r - independent).length();
        assert!(dev < 100.0, "{dev} m from the separate two-body solution");
        let rel_two_body = report("two-body", r, v);
        const RECORDED_TWO_BODY: f64 = 5.058e-2;
        assert!(
            (rel_two_body - RECORDED_TWO_BODY).abs() < 1e-4,
            "two-body |Δr|/r = {rel_two_body:.4e} is no longer the recorded {RECORDED_TWO_BODY:e}"
        );

        // (2) With the planets: the brief's bound.
        let started = std::time::Instant::now();
        let (r, v) = perturbed_state_icrs(&c, epoch, t);
        let elapsed = started.elapsed().as_secs_f64();
        let rel = report("perturbed", r, v);
        println!("perturbed_state_icrs over 18.1 yr: {elapsed:.3} s (this build)");
        assert!(
            rel < 0.005,
            "perturbed |Δr|/r = {rel:.4e} over the brief's 0.5 %"
        );
    }

    /// The integrator alone: with no perturbers it must reproduce the conic. Halley 1968 → 1986
    /// (through perihelion at 0.575 au) and back.
    #[test]
    fn perturbed_integrator_reproduces_two_body() {
        let (c, epoch) = halley();
        let t = at_jd(2_446_502.5);
        let start = heliocentric_state_icrs(&c, epoch);
        let ((r, v), steps) = integrate(start, c.mu_m3s2, (epoch, t), &[], 1.0);
        let (r_conic, v_conic) = heliocentric_state_icrs(&c, t);
        let err = (r - r_conic).length();
        println!(
            "two-body limit 1968 → 1986: {err:.1} m, |Δv| {:.2e} m/s, {steps} RK4 steps",
            (v - v_conic).length()
        );
        assert!(err < 1000.0, "{err} m");
        let ((r_back, _), _) = integrate((r, v), c.mu_m3s2, (t, epoch), &[], 1.0);
        let err_back = (r_back - start.0).length();
        assert!(err_back < 2000.0, "back to 1968: {err_back} m");
        // No time to integrate: the start state, exactly.
        assert_eq!(perturbed_state_icrs(&c, epoch, epoch), start);
        // Halving every step (η and the 2-day cap) changes the perturbed answer by under 1 km.
        let ((coarse, _), steps) = integrate(start, c.mu_m3s2, (epoch, t), &Planet::ALL, 1.0);
        let ((fine, _), _) = integrate(start, c.mu_m3s2, (epoch, t), &Planet::ALL, 0.5);
        assert_eq!(coarse, perturbed_state_icrs(&c, epoch, t).0);
        let diff = (fine - coarse).length();
        println!("with planets: {steps} RK4 steps; half steps move the answer {diff:.1} m");
        assert!(diff < 1000.0, "{diff} m");
        // Non-finite input gives NaN, not a silent default.
        let bad = Conic { e: f64::NAN, ..c };
        assert!(perturbed_state_icrs(&bad, epoch, t).0.is_nan());
    }

    /// [`osculate_at`] on 1986-03-13 against the osculating orbit of Horizons' own state that
    /// day (q 0.5871 au, e 0.96728), and two-body from the new conic around that date.
    #[test]
    fn osculate_at_reproduces_horizons_orbit() {
        let (c, epoch) = halley();
        let (jd, r_ref, v_ref) = horizons_heliocentric(1_000_036);
        let t = at_jd(jd);
        let osc = osculate_at(&c, epoch, t);
        let to_ecliptic = DMat3::from_rotation_x(-OBLIQUITY_J2000_RAD);
        let truth = Conic::from_state(to_ecliptic * r_ref, to_ecliptic * v_ref, GM_SUN, t);
        println!(
            "Halley osculating 1986-03-13: q {:.6} au (Horizons {:.6}), e {:.6} ({:.6}), \
             i {:.4}° ({:.4}°), T_p JD {:.3} ({:.3}); SBDB 1968: q {:.6}, e {:.6}",
            osc.q_m / AU,
            truth.q_m / AU,
            osc.e,
            truth.e,
            osc.i_rad.to_degrees(),
            truth.i_rad.to_degrees(),
            osc.tp.jd_tdb(),
            truth.tp.jd_tdb(),
            c.q_m / AU,
            c.e
        );
        assert!((osc.q_m / truth.q_m - 1.0).abs() < 1e-4);
        assert!((osc.e - truth.e).abs() < 1e-5);
        assert!((osc.i_rad - truth.i_rad).abs() < 1e-5);
        assert!(osc.tp.seconds_since(truth.tp).abs() < 0.01 * DAY);
        // The new conic reproduces the perturbed state at t, and two-body from it stays within
        // 1e-3 of r of the perturbed path 30 days either side.
        let (r_osc, _) = heliocentric_state_icrs(&osc, t);
        let (r_pert, _) = perturbed_state_icrs(&c, epoch, t);
        assert!(
            (r_osc - r_pert).length() < 1.0,
            "{}",
            (r_osc - r_pert).length()
        );
        for days in [-30.0, 30.0] {
            let t2 = t.add_seconds(days * DAY);
            let a = heliocentric_state_icrs(&osc, t2).0;
            let b = perturbed_state_icrs(&c, epoch, t2).0;
            let rel = (a - b).length() / b.length();
            println!("two-body from the 1986-03-13 conic, {days:+} d: {rel:.2e} of r");
            assert!(rel < 1e-3, "{days} d: {rel:e}");
        }
    }

    /// How far two-body from the 1968 elements is from the perturbed path on the app's default
    /// date (2026-11-20) and at Halley's next perihelion (2061-07-28): a measured record for the
    /// card's label, no Horizons golden there. Non-gravitational forces from the 1986 passage
    /// (about 4 days per orbit in T_p) are in neither path.
    ///
    /// Measured (dev profile): 2026-11-20 r 35.001 au, 2.809e7 km = 5.365e-3 of r; 2061-07-28
    /// r 0.598 au, 4.854e8 km = 5.423e0 of r (the 1968 two-body orbit has the wrong period, so
    /// it is on the far side of the Sun). The bounds bracket those values by about 30 %.
    #[test]
    fn halley_two_body_drift_after_1986() {
        let (c, epoch) = halley();
        for (label, jd, lo, hi) in [
            ("2026-11-20", 2_461_364.5, 4e-3, 7e-3),
            ("2061-07-28", 2_474_033.5, 4.0, 7.0),
        ] {
            let t = at_jd(jd);
            let (r2, _) = heliocentric_state_icrs(&c, t);
            let (rp, _) = perturbed_state_icrs(&c, epoch, t);
            let rel = (r2 - rp).length() / rp.length();
            println!(
                "Halley {label}: r {:.3} au; two-body from 1968 is {:.3e} km = {rel:.3e} of r \
                 from the perturbed path",
                rp.length() / AU,
                (r2 - rp).length() / 1e3
            );
            assert!(rp.is_finite() && (lo..hi).contains(&rel), "{label}: drift {rel:.3e} of r");
        }
    }

    /// (1) Ceres from its `asteroids.bin` mean-anomaly elements, two-body from their epoch
    /// (2026-06-09), against Horizons (JPL#48, DE441) at 2026-11-20 00:00 UTC, 164 days later.
    /// Measured: 15 762 km, 4.0e-5 of r (planetary perturbations); bound 1e-4.
    /// (2) The same elements through [`perturbed_state_icrs`]: 27 km, 6.7e-8 of r; bound 1e-6.
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
        // The same elements with the planets' perturbations.
        let (rp, _) = perturbed_state_icrs(&c, at_jd(a.epoch_jd_tdb), at_jd(jd));
        let rel_p = (rp - r_ref).length() / r_ref.length();
        println!(
            "Ceres perturbed: |Δr| = {:.0} km = {rel_p:.3e} of r",
            (rp - r_ref).length() / 1e3
        );
        assert!(rel_p < 1e-6, "|Δr|/r = {rel_p:e}");
    }
}
