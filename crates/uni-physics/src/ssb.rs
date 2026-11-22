//! The Sun's offset from the Solar System barycentre (M1 Task A1).
//!
//! sun_from_ssb = −Σᵢ (GMᵢ / (GM☉ + Σⱼ GMⱼ)) · rᵢ over the Standish bodies Mercury–Neptune
//! (system barycentres; the EMB carries Earth and Moon), with rᵢ heliocentric from
//! [`crate::standish`] and the GMs of DE440 (Park et al. 2021, AJ 161, 105). Pluto is omitted:
//! its pull moves the SSB by GM_Pluto-sys/GM☉ × 39.5 AU ≈ 43 km, far below Standish's error.
//!
//! Neptune's Standish elements fit its orbit about the SSB (`standish::ssb_fitted`, A1 fix
//! round 1): its heliocentric position is rᵢ = Rᵢ − S, with Rᵢ the fitted (SSB-relative) state
//! and S = sun_from_ssb. Substituting into the sum and solving for S gives the closed form
//! S = −(Σ_Sun-fitted GMᵢ rᵢ + Σ_SSB-fitted GMᵢ Rᵢ) / (GM☉ + Σⱼ GMⱼ − Σ_SSB-fitted GMᵢ),
//! and the same with velocities, so no iteration is needed.

use glam::DVec3;
use uni_core::Epoch;

use crate::bodies::{GM_SUN_DE440_M3S2, system_gm_m3s2};
use crate::standish::{Planet, fitted_state_icrs, ssb_fitted};

/// DE440 GM of each Standish body (barycentres 1..=8) [m³/s²], in [`Planet::ALL`] order.
const PLANET_GM_M3S2: [f64; 8] = {
    let mut gm = [0.0; 8];
    let mut k = 0;
    while k < 8 {
        gm[k] = match system_gm_m3s2(k as i32 + 1) {
            Some(value) => value,
            None => panic!("bodies: missing a system GM"),
        };
        k += 1;
    }
    gm
};

/// Position of the Sun relative to the SSB [m], ICRS axes.
pub fn sun_from_ssb_icrs(t: Epoch) -> DVec3 {
    sun_from_ssb_state_icrs(t).0
}

/// Position [m] and velocity [m/s] of the Sun relative to the SSB, ICRS axes (the closed form
/// in the module doc). The velocity is the exact time derivative of the position, because the
/// sum is linear in the Standish states, whose velocities are exact derivatives.
pub(crate) fn sun_from_ssb_state_icrs(t: Epoch) -> (DVec3, DVec3) {
    let mut total = GM_SUN_DE440_M3S2;
    let mut moment = DVec3::ZERO;
    let mut moment_rate = DVec3::ZERO;
    for (p, gm) in Planet::ALL.into_iter().zip(PLANET_GM_M3S2) {
        let (r, v) = fitted_state_icrs(p, t);
        moment += gm * r;
        moment_rate += gm * v;
        if !ssb_fitted(p) {
            total += gm;
        }
    }
    (-moment / total, -moment_rate / total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standish::golden::{epoch_of, horizons_spots, pos_m};
    use crate::standish::heliocentric_state_icrs;

    /// Horizons 10@0 at 1950, 2000 and 2026-11-20: |Δ| < 3 000 km. Budget: Standish's λ errors
    /// (Jupiter 400″ at 5.2 AU shifts the SSB by 1 440 km, Saturn 600″ at 9.5 AU by 1 180 km)
    /// give ≈ 1 900 km RSS, ≈ 2 600 km if aligned (review m1). Pluto's 43 km is omitted.
    #[test]
    fn sun_ssb_offset_vs_horizons() {
        let spots = horizons_spots(10, 0);
        assert_eq!(spots.len(), 3);
        for s in spots {
            let model = sun_from_ssb_icrs(epoch_of(s.jd_tdb));
            let truth = pos_m(&s);
            let miss_km = (model - truth).length() / 1e3;
            println!(
                "Sun − SSB at JD {:.4}: |Horizons| {:.0} km, |Δ| {miss_km:.0} km",
                s.jd_tdb,
                truth.length() / 1e3
            );
            assert!(miss_km < 3_000.0, "JD {}: {miss_km} km", s.jd_tdb);
        }
    }

    /// The closed form puts the SSB at the barycentre of the heliocentric states: with S the
    /// Sun and S + rᵢ each planet relative to the SSB, GM☉·S + Σ GMᵢ (S + rᵢ) = 0 to rounding,
    /// for position and velocity, in every regime (Table 1, Table 2, extrapolated). The rᵢ
    /// come from `heliocentric_state_icrs`, which itself uses S for Neptune.
    #[test]
    fn closed_form_is_the_barycentre() {
        for days in [0.0, 9_819.5, -100_000.0, 3.65e8] {
            let t = Epoch::from_days_since_j2000(days);
            let (s, s_dot) = sun_from_ssb_state_icrs(t);
            let (mut m, mut m_dot) = (GM_SUN_DE440_M3S2 * s, GM_SUN_DE440_M3S2 * s_dot);
            for (p, gm) in Planet::ALL.into_iter().zip(PLANET_GM_M3S2) {
                let (r, v) = heliocentric_state_icrs(p, t);
                m += gm * (s + r);
                m_dot += gm * (s_dot + v);
            }
            let total: f64 = GM_SUN_DE440_M3S2 + PLANET_GM_M3S2.iter().sum::<f64>();
            // Barycentre offset [m] and its velocity [m/s]; nonzero only by rounding (≈ 1e-7 m).
            let (miss, miss_dot) = ((m / total).length(), (m_dot / total).length());
            assert!(
                miss < 1e-3 && miss_dot < 1e-9,
                "day {days}: {miss} m, {miss_dot} m/s"
            );
        }
    }
}
