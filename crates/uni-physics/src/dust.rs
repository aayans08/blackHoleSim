//! Comet dust tails: Finson & Probstein 1968 (ApJ 154, 327) syndynes and synchrones, with β
//! from Burns, Lamy & Soter 1979 (Icarus 40, 1) (M1 Task A3).
//!
//! Finson–Probstein: a grain leaves the nucleus with zero velocity relative to it at the
//! emission time t_e = t_obs − age, and then moves on a two-body orbit about the Sun with the
//! reduced attraction μ(1 − β), β being the ratio of radiation pressure to solar gravity (both
//! fall as 1/r²). Grains of one β at all ages form a syndyne; grains of all β emitted at one
//! time form a synchrone. Neglected: ejection speeds, grain fragmentation, planetary
//! perturbations, and the Lorentz force on charged grains (the card says `Tails:
//! Finson–Probstein dust model; brightness approximate`).

use glam::DVec3;
use uni_core::{AU, DAY, Epoch};

use crate::frames::ecliptic_j2000_to_icrs;
use crate::kepler::Conic;

/// The grid of grains to emit.
///
/// `n_beta` values of β from `beta_min` to `beta_max` (both included) and `n_age` ages
/// (k + 1)·`max_age_days`/`n_age`, k = 0 … n_age − 1, so the oldest grain is exactly
/// `max_age_days` old and none sits on the nucleus at age 0. β is spaced logarithmically, so each
/// value stands for an equal step in ln β; a single β (`n_beta` = 1) is `beta_min` and may be 0
/// (the nucleus's own orbit). Domain: 0 ≤ β_min ≤ β_max < 1, β_min > 0 when n_beta ≥ 2,
/// n_beta ≥ 1, n_age ≥ 1, 0 < max_age_days < ∞; anything else yields no grains.
#[derive(Clone, Copy, Debug)]
pub struct DustParams {
    pub beta_min: f64,
    /// Below 1 (repulsive β ≥ 1 is excluded).
    pub beta_max: f64,
    pub n_beta: u32,
    pub max_age_days: f64,
    pub n_age: u32,
}

/// One grain at the observation time.
#[derive(Clone, Copy, Debug)]
pub struct DustGrain {
    /// Position relative to the nucleus [m], ICRS axes.
    pub pos_rel_nucleus_m: DVec3,
    /// Radiation-pressure to gravity ratio.
    pub beta: f64,
    /// Time since emission [s].
    pub age_s: f64,
    /// Relative brightness weight (production × size distribution).
    pub weight: f64,
}

impl DustParams {
    /// Whether the grid is inside the domain documented on [`DustParams`].
    ///
    /// β ≥ 1 is excluded: at β = 1 radiation pressure cancels gravity (μ(1 − β) = 0, a straight
    /// line), and above it the net force is repulsive, a hyperbola about the Sun's far focus that
    /// [`Conic`] (attractive μ > 0 only) does not describe. Such grains are sub-micron (β = 1 at
    /// s = 0.57 µm for ρ = 1 g cm⁻³, Burns et al. 1979) and, being smaller than the wavelength,
    /// scatter visible light weakly, so leaving them out costs little light.
    fn is_valid(&self) -> bool {
        let betas_ok = self.beta_min >= 0.0
            && self.beta_min <= self.beta_max
            && self.beta_max < 1.0
            && (self.n_beta == 1 || self.beta_min > 0.0);
        betas_ok
            && self.n_beta >= 1
            && self.n_age >= 1
            && self.max_age_days > 0.0
            && self.max_age_days.is_finite()
    }

    /// The j-th β: log-spaced from `beta_min` to `beta_max`, endpoints exact.
    fn beta(&self, j: u32) -> f64 {
        if j == 0 {
            self.beta_min
        } else if j + 1 == self.n_beta {
            self.beta_max
        } else {
            let f = f64::from(j) / f64::from(self.n_beta - 1);
            self.beta_min * (self.beta_max / self.beta_min).powf(f)
        }
    }

    /// The k-th age [s]: (k + 1)·max_age/n_age.
    fn age_s(&self, k: u32) -> f64 {
        f64::from(k + 1) * self.max_age_days * DAY / f64::from(self.n_age)
    }
}

/// Grains of `comet` at `t_obs`; `production(r_h_au)` is the relative dust production at
/// heliocentric distance r_h.
///
/// `comet` is a heliocentric conic in the J2000 ecliptic (as [`crate::smallbody`] builds them);
/// the grain positions are rotated to ICRS. Each grain: the nucleus state at its emission time
/// t_e = t_obs − age, `Conic::from_state(r, v, μ(1 − β), t_e)`, then `Conic::state(t_obs)`, one
/// elliptic or hyperbolic Kepler solve per grain (review I13); the nucleus is solved once per age.
///
/// Order: β-major, `grains[j·n_age + k]` has the j-th β and k-th age (rows are syndynes, columns
/// synchrones). Empty when `p` is outside its domain (see [`DustParams`]).
///
/// Weight = production(r_h at t_e) × √β. Dust size distribution dn/ds ∝ s^−3.5 (Dohnanyi 1969,
/// JGR 74, 2531), geometric cross-section ∝ s², and β ∝ 1/s (fixed Q_pr and density, Burns et
/// al. 1979) give a scattering cross-section per unit ln β ∝ s^−0.5 ∝ √β. Each grain stands for
/// an equal step in ln β and in age, so its weight is proportional to the light it carries, up to
/// its illumination and phase at t_obs, which the renderer applies. β = 0 grains weigh 0.
pub fn finson_probstein(
    comet: &Conic,
    t_obs: Epoch,
    p: &DustParams,
    production: &dyn Fn(f64) -> f64,
) -> Vec<DustGrain> {
    if !p.is_valid() {
        return Vec::new();
    }
    let mu = comet.mu_m3s2;
    let (nucleus_now, _) = comet.state(t_obs);
    // Emission states, one per age: (age, t_e, r, v, production).
    let emissions: Vec<(f64, Epoch, DVec3, DVec3, f64)> = (0..p.n_age)
        .map(|k| {
            let age_s = p.age_s(k);
            let t_e = t_obs.add_seconds(-age_s);
            let (r, v) = comet.state(t_e);
            (age_s, t_e, r, v, production(r.length() / AU))
        })
        .collect();
    let mut grains = Vec::with_capacity(p.n_beta as usize * p.n_age as usize);
    for j in 0..p.n_beta {
        let beta = p.beta(j);
        let size_weight = beta.sqrt();
        for &(age_s, t_e, r, v, prod) in &emissions {
            let orbit = Conic::from_state(r, v, mu * (1.0 - beta), t_e);
            let (grain, _) = orbit.state(t_obs);
            grains.push(DustGrain {
                pos_rel_nucleus_m: ecliptic_j2000_to_icrs(grain - nucleus_now),
                beta,
                age_s,
                weight: prod * size_weight,
            });
        }
    }
    grains
}

/// β = 5.7×10⁻⁵ Q_pr / (ρ[g cm⁻³] · s[cm]) for a grain of `radius_m` and `density_kg_m3`.
///
/// Burns, Lamy & Soter 1979 (Icarus 40, 1), for a spherical grain of radius s at any
/// heliocentric distance (radiation pressure and gravity both fall as 1/r²). Q_pr is the
/// radiation-pressure efficiency (≈ 1 for grains much larger than the wavelength). Radius and
/// density must be positive.
pub fn beta_for_grain(radius_m: f64, density_kg_m3: f64, q_pr: f64) -> f64 {
    let density_g_cm3 = density_kg_m3 * 1e-3;
    let radius_cm = radius_m * 1e2;
    5.7e-5 * q_pr / (density_g_cm3 * radius_cm)
}

#[cfg(test)]
mod tests {
    use uni_core::{AU, DAY};

    use super::*;
    use crate::bodies::GM_SUN_DE440_M3S2 as GM_SUN;
    use crate::smallbody::{conic_from_perihelion, heliocentric_state_icrs};

    fn at_jd(jd_tdb: f64) -> Epoch {
        Epoch::from_days_since_j2000(jd_tdb - 2_451_545.0)
    }

    /// One β and a regular age grid.
    fn one_beta(beta: f64, max_age_days: f64, n_age: u32) -> DustParams {
        DustParams {
            beta_min: beta,
            beta_max: beta,
            n_beta: 1,
            max_age_days,
            n_age,
        }
    }

    /// 1P/Halley's SBDB elements (`comets.ron`: q, e, i, Ω, ω, T_p).
    fn halley() -> Conic {
        conic_from_perihelion(
            0.574_863_831_374_341_3,
            0.967_935_995_695_321_1,
            162.190_530_043_912_9,
            59.098_947_206_124_37,
            112.241_431_463_776_4,
            2_446_469.973_616_146_5,
        )
    }

    #[test]
    fn beta_half_micron() {
        // s = 0.5 µm radius → 5.7e-5/(1·5e-5 cm).
        assert!((beta_for_grain(0.5e-6, 1000.0, 1.0) - 1.14).abs() < 0.01);
        // β ∝ Q_pr/(ρ s): a 10 µm grain of density 3 g cm⁻³ is 60× below that.
        let b = beta_for_grain(10e-6, 3000.0, 1.0);
        assert!((b - 1.14 / 60.0).abs() < 1e-12, "{b}");
    }

    /// Grains with no radiation pressure follow the nucleus's own orbit: from_state at emission
    /// and state at observation reproduce the nucleus to rounding, for an elliptic comet at
    /// perihelion and a hyperbolic one.
    #[test]
    fn beta_zero_grain_stays_on_nucleus() {
        let t_obs = at_jd(2_446_475.5);
        let hyperbolic = conic_from_perihelion(0.3, 1.5, 128.9, 61.0, 37.3, 2_446_470.5);
        for comet in [halley(), hyperbolic] {
            let grains = finson_probstein(&comet, t_obs, &one_beta(0.0, 30.0, 30), &|_| 1.0);
            assert_eq!(grains.len(), 30);
            for g in &grains {
                let d = g.pos_rel_nucleus_m.length();
                assert!(d < 1.0, "age {} d: {d} m from the nucleus", g.age_s / DAY);
            }
        }
    }

    /// A small β grain drifts anti-sunward by the radiation acceleration ½βg☉t² (the nucleus
    /// and the grain feel the same gravity, so only βGM☉/r² separates them over a short time).
    ///
    /// Observed a quarter orbit after perihelion, where the nucleus sits on the ecliptic's +y
    /// axis: a grain offset left in ecliptic axes would be off the ICRS radial by ε₀ = 23.4°.
    #[test]
    fn small_beta_drifts_antisunward() {
        let tp = 2_461_000.5;
        let circular = conic_from_perihelion(1.0, 0.0, 0.0, 0.0, 0.0, tp);
        let quarter_days = circular.period_s().map_or(f64::NAN, |p| p / 4.0 / DAY);
        let t_obs = at_jd(tp + quarter_days);
        let grains = finson_probstein(&circular, t_obs, &one_beta(0.1, 1.0, 1), &|_| 1.0);
        assert_eq!(grains.len(), 1);
        let g = grains[0];
        assert_eq!(g.age_s, DAY);
        let (nucleus, _) = heliocentric_state_icrs(&circular, t_obs);
        let radial = g.pos_rel_nucleus_m.dot(nucleus.normalize());
        let want = 0.5 * 0.1 * (GM_SUN / (AU * AU)) * DAY * DAY;
        assert!((want - 2.21e6).abs() < 0.01e6);
        assert!((radial / want - 1.0).abs() < 0.05, "{radial} m vs {want} m");
        // Along the radial to within the ≈ (2/3)·n·t = 0.7° that orbital motion turns it.
        let off = g.pos_rel_nucleus_m.angle_between(nucleus).to_degrees();
        assert!(off < 2.0, "{off}° off the anti-sunward direction");
    }

    /// The β × age grid, its order, and the weights (production at emission × √β).
    #[test]
    fn grid_order_and_weights() {
        let comet = halley();
        let t_obs = at_jd(2_446_480.5);
        let p = DustParams {
            beta_min: 0.001,
            beta_max: 0.9,
            n_beta: 4,
            max_age_days: 60.0,
            n_age: 3,
        };
        // A production law that reports the heliocentric distance it is given.
        let grains = finson_probstein(&comet, t_obs, &p, &|r_h| r_h);
        assert_eq!(grains.len(), 12);
        let ratio: f64 = 900.0_f64.powf(1.0 / 3.0);
        for (index, g) in grains.iter().enumerate() {
            let (j, k) = (index / 3, index % 3);
            // β-major: β_j = β_min·(β_max/β_min)^(j/(n−1)), endpoints exact.
            let beta = 0.001 * ratio.powi(j as i32);
            assert!((g.beta / beta - 1.0).abs() < 1e-12, "β[{j}] = {}", g.beta);
            // Ages (k + 1)·max/n: 20, 40, 60 days.
            assert!((g.age_s - (k as f64 + 1.0) * 20.0 * DAY).abs() < 1e-6);
            // Weight = production(r_h at emission) × √β.
            let (r_e, _) = comet.state(t_obs.add_seconds(-g.age_s));
            let want = r_e.length() / AU * g.beta.sqrt();
            assert!(
                (g.weight / want - 1.0).abs() < 1e-12,
                "{} vs {want}",
                g.weight
            );
        }
        assert_eq!(grains[0].beta, 0.001);
        assert_eq!(grains[11].beta, 0.9);
        // At a fixed age, larger β grains are pushed farther from the nucleus.
        for k in 0..3 {
            let d: Vec<f64> = (0..4)
                .map(|j| grains[j * 3 + k].pos_rel_nucleus_m.length())
                .collect();
            assert!(d.windows(2).all(|w| w[0] < w[1]), "age {k}: {d:?}");
        }
    }

    /// Parameters outside their domain give no grains (β ≥ 1 is repulsive and excluded).
    #[test]
    fn invalid_params_give_no_grains() {
        let comet = halley();
        let t = at_jd(2_446_480.5);
        let ok = DustParams {
            beta_min: 0.01,
            beta_max: 0.5,
            n_beta: 2,
            max_age_days: 10.0,
            n_age: 2,
        };
        assert_eq!(finson_probstein(&comet, t, &ok, &|_| 1.0).len(), 4);
        let bad = [
            DustParams {
                beta_max: 1.0,
                ..ok
            },
            DustParams {
                beta_max: 1.5,
                ..ok
            },
            DustParams {
                beta_min: -0.1,
                ..ok
            },
            DustParams {
                beta_min: 0.6,
                ..ok
            },
            DustParams {
                beta_min: 0.0,
                ..ok
            },
            DustParams {
                beta_min: f64::NAN,
                ..ok
            },
            DustParams { n_beta: 0, ..ok },
            DustParams { n_age: 0, ..ok },
            DustParams {
                max_age_days: 0.0,
                ..ok
            },
            DustParams {
                max_age_days: f64::INFINITY,
                ..ok
            },
        ];
        for p in bad {
            assert!(
                finson_probstein(&comet, t, &p, &|_| 1.0).is_empty(),
                "{p:?}"
            );
        }
    }
}
