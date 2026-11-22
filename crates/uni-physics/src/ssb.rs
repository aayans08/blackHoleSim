//! The Sun's offset from the Solar System barycentre (M1 Task A1).
//!
//! sun_from_ssb = −Σᵢ (GMᵢ / (GM☉ + Σⱼ GMⱼ)) · rᵢ over the Standish bodies Mercury–Neptune
//! (system barycentres; the EMB carries Earth and Moon), with rᵢ heliocentric from
//! [`crate::standish`] and the GMs of DE440 (Park et al. 2021, AJ 161, 105). Pluto is omitted:
//! its pull moves the SSB by GM_Pluto-sys/GM☉ × 39.5 AU ≈ 43 km, far below Standish's error.

use glam::DVec3;
use uni_core::Epoch;

use crate::bodies::{GM_SUN_DE440_M3S2, system_gm_m3s2};
use crate::standish::{Planet, heliocentric_state_icrs};

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
    let mut total = GM_SUN_DE440_M3S2;
    let mut moment = DVec3::ZERO;
    for (p, gm) in Planet::ALL.into_iter().zip(PLANET_GM_M3S2) {
        let (r, _) = heliocentric_state_icrs(p, t);
        moment += gm * r;
        total += gm;
    }
    -moment / total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standish::golden::{epoch_of, horizons_spots, pos_m};

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
}
