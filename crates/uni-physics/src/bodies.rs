//! Physical constants of the major bodies (M1 Task A1): GM and triaxial radii.
//!
//! Radii: IAU WGCCRE 2015 (Archinal et al. 2018, CMDA 130:22). GM: DE440 (Park et al. 2021,
//! AJ 161, 105) and the JPL satellite physical parameters.
//!
//! **Implemented by the lead's A0b (fix round 2, review I3) ahead of A1**, because A9's
//! `facts_cover_major_bodies` (uni-data) reads this table while A1 runs in parallel. A1 still owns
//! this module: check every row against the sources below, add its `radii_order_and_flattening`
//! test, and put its tests inside the existing `mod tests` at the end of this file (a second
//! `mod tests` does not compile).
//!
//! Transcription sources, read 2026-11-22:
//! - Radii: NAIF `pck00011.tpc` (https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc),
//!   whose `BODYnnn_RADII` values are the WGCCRE 2015 report's. The Sun's 695 700 km is the IAU
//!   2015 Resolution B3 nominal solar radius. Bodies the report gives as spheres have a = b = c.
//! - GM: NAIF `gm_de440.tpc` (https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc):
//!   the DE440 "ASTRO-VALUES" for the Sun, Mercury, Venus, Earth and the Moon, and the JPL
//!   natural-satellite solutions' release values for the other moons and the planet bodies.
//!   Each value is the body's own GM (`BODYnnn_GM`), not its system barycentre's (`BODYn_GM`):
//!   for example Jupiter 1.266 865 319 × 10⁸ km³/s², Jupiter system 1.267 127 64 × 10⁸.
//!   Literals are the file's km³/s² with the exponent raised by 9, so they read digit for digit.

use glam::DVec3;

use crate::NaifId;

/// One body's constants.
pub struct BodyConst {
    pub name: &'static str,
    pub naif: NaifId,
    /// NAIF id of the body it orbits; `None` for the Sun.
    pub parent: Option<NaifId>,
    /// Gravitational parameter GM [m³/s²].
    pub gm_m3s2: f64,
    /// Triaxial radii [m], a ≥ b ≥ c: equatorial, equatorial, polar.
    pub radii_m: DVec3,
}

/// One table row; radii in m (the source's km with `e3`).
const fn row(
    name: &'static str,
    naif: NaifId,
    parent: Option<NaifId>,
    gm_m3s2: f64,
    a: f64,
    b: f64,
    c: f64,
) -> BodyConst {
    BodyConst {
        name,
        naif,
        parent,
        gm_m3s2,
        radii_m: DVec3::new(a, b, c),
    }
}

const SUN: Option<NaifId> = Some(10);

/// The contract's list, in its order. Columns: name, NAIF id, parent, GM [m³/s²] (`gm_de440.tpc`),
/// radii a, b, c [m] (`pck00011.tpc`, WGCCRE 2015).
#[rustfmt::skip]
static MAJOR_BODIES: [BodyConst; 29] = [
    row("Sun", 10, None, 1.3271244004127942e20, 695_700.0e3, 695_700.0e3, 695_700.0e3),
    row("Mercury", 199, SUN, 2.2031868551400003e13, 2_440.53e3, 2_440.53e3, 2_438.26e3),
    row("Venus", 299, SUN, 3.2485859200000000e14, 6_051.8e3, 6_051.8e3, 6_051.8e3),
    row("Earth", 399, SUN, 3.9860043550702266e14, 6_378.1366e3, 6_378.1366e3, 6_356.7519e3),
    row("Mars", 499, SUN, 4.282837362069909e13, 3_396.19e3, 3_396.19e3, 3_376.20e3),
    row("Jupiter", 599, SUN, 1.266865319003704e17, 71_492.0e3, 71_492.0e3, 66_854.0e3),
    row("Saturn", 699, SUN, 3.793120623436167e16, 60_268.0e3, 60_268.0e3, 54_364.0e3),
    row("Uranus", 799, SUN, 5.793951256527211e15, 25_559.0e3, 25_559.0e3, 24_973.0e3),
    row("Neptune", 899, SUN, 6.835103145462294e15, 24_764.0e3, 24_764.0e3, 24_341.0e3),
    row("Pluto", 999, SUN, 8.696138177608748e11, 1_188.3e3, 1_188.3e3, 1_188.3e3),
    row("Moon", 301, Some(399), 4.9028001184575496e12, 1_737.4e3, 1_737.4e3, 1_737.4e3),
    row("Io", 501, Some(599), 5.959915466180539e12, 1_829.4e3, 1_819.4e3, 1_815.7e3),
    row("Europa", 502, Some(599), 3.202712099607295e12, 1_562.6e3, 1_560.3e3, 1_559.5e3),
    row("Ganymede", 503, Some(599), 9.887832752719638e12, 2_631.2e3, 2_631.2e3, 2_631.2e3),
    row("Callisto", 504, Some(599), 7.179283402579837e12, 2_410.3e3, 2_410.3e3, 2_410.3e3),
    row("Mimas", 601, Some(699), 2.503488768152587e9, 207.8e3, 196.7e3, 190.6e3),
    row("Enceladus", 602, Some(699), 7.210366688598896e9, 256.6e3, 251.4e3, 248.3e3),
    row("Tethys", 603, Some(699), 4.121352885489587e10, 538.4e3, 528.3e3, 526.3e3),
    row("Dione", 604, Some(699), 7.311607172482067e10, 563.4e3, 561.3e3, 559.6e3),
    row("Rhea", 605, Some(699), 1.539417519146563e11, 765.0e3, 763.1e3, 762.4e3),
    row("Titan", 606, Some(699), 8.978137095521046e12, 2_575.15e3, 2_574.78e3, 2_574.47e3),
    row("Hyperion", 607, Some(699), 3.704913747932265e8, 180.1e3, 133.0e3, 102.7e3),
    row("Iapetus", 608, Some(699), 1.205151060137642e11, 745.7e3, 745.7e3, 712.1e3),
    row("Phobos", 401, Some(499), 7.087546066894452e5, 13.0e3, 11.4e3, 9.1e3),
    row("Deimos", 402, Some(499), 9.615569648120313e4, 7.8e3, 6.0e3, 5.1e3),
    row("Triton", 801, Some(899), 1.428495462910464e12, 1_352.6e3, 1_352.6e3, 1_352.6e3),
    row("Titania", 703, Some(799), 2.269437003741248e11, 788.9e3, 788.9e3, 788.9e3),
    row("Oberon", 704, Some(799), 2.053234302535623e11, 761.4e3, 761.4e3, 761.4e3),
    row("Charon", 901, Some(999), 1.058799888601881e11, 606.0e3, 606.0e3, 606.0e3),
];

/// Every body with constants: the Sun, the eight planets, Pluto, the Moon, the Galileans,
/// Saturn's eight major moons, Phobos, Deimos, Triton, Titania, Oberon and Charon.
///
/// Planets and Pluto orbit the Sun (10); each moon's parent is its planet's body id (399, 499,
/// 599, 699, 799, 899, 999), matching the planetocentric Horizons goldens.
pub fn major_bodies() -> &'static [BodyConst] {
    &MAJOR_BODIES
}

/// The body with NAIF id `id`, if it is in [`major_bodies`]. Barycentres (0–9) are not bodies.
pub fn by_naif(id: NaifId) -> Option<&'static BodyConst> {
    MAJOR_BODIES.iter().find(|b| b.naif == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The NAIF ids the contract lists, in its order, with the body each one orbits.
    const CONTRACT: [(NaifId, Option<NaifId>); 29] = [
        (10, None),
        (199, Some(10)),
        (299, Some(10)),
        (399, Some(10)),
        (499, Some(10)),
        (599, Some(10)),
        (699, Some(10)),
        (799, Some(10)),
        (899, Some(10)),
        (999, Some(10)),
        (301, Some(399)),
        (501, Some(599)),
        (502, Some(599)),
        (503, Some(599)),
        (504, Some(599)),
        (601, Some(699)),
        (602, Some(699)),
        (603, Some(699)),
        (604, Some(699)),
        (605, Some(699)),
        (606, Some(699)),
        (607, Some(699)),
        (608, Some(699)),
        (401, Some(499)),
        (402, Some(499)),
        (801, Some(899)),
        (703, Some(799)),
        (704, Some(799)),
        (901, Some(999)),
    ];

    /// Every body the contract names is present exactly once, with its parent, positive finite
    /// GM, a ≥ b ≥ c > 0, and a `by_naif` lookup that returns that row (A0b fix round 2, review
    /// I3: A9's `facts_cover_major_bodies` reads this table).
    #[test]
    fn major_bodies_cover_contract_list() {
        let all = major_bodies();
        assert_eq!(all.len(), CONTRACT.len());
        for (row, &(naif, parent)) in all.iter().zip(CONTRACT.iter()) {
            assert_eq!(row.naif, naif, "{}", row.name);
            assert_eq!(row.parent, parent, "{}", row.name);
            assert!(!row.name.is_empty());
            assert!(row.gm_m3s2.is_finite() && row.gm_m3s2 > 0.0, "{}", row.name);
            let r = row.radii_m;
            assert!(r.x >= r.y && r.y >= r.z && r.z > 0.0, "{}: {r:?}", row.name);
            let found = by_naif(naif).expect("listed id resolves");
            assert!(std::ptr::eq(found, row), "{}", row.name);
            assert_eq!(all.iter().filter(|b| b.naif == naif).count(), 1);
            if let Some(p) = parent {
                assert!(by_naif(p).is_some(), "{}'s parent {p}", row.name);
            }
        }
        // Barycentres (0–9) and unknown ids are not bodies.
        for id in [0, 3, 5, 9, 1_000_000] {
            assert!(by_naif(id).is_none(), "{id}");
        }
    }

    /// Spot rows against the cited sources, as transcribed: GM from NAIF `gm_de440.tpc`
    /// (km³/s² × 1e9) and radii from the WGCCRE 2015 values in NAIF `pck00011.tpc` (km × 1e3).
    #[test]
    fn major_bodies_spot_values() {
        let b = |id| by_naif(id).expect("present");
        assert_eq!(b(10).gm_m3s2, 1.3271244004127942e20);
        assert_eq!(b(399).gm_m3s2, 3.9860043550702266e14);
        assert_eq!(b(301).gm_m3s2, 4.9028001184575496e12);
        assert_eq!(b(599).gm_m3s2, 1.266865319003704e17);
        assert_eq!(b(606).gm_m3s2, 8.978137095521046e12);
        assert_eq!(b(10).radii_m, DVec3::splat(695_700.0e3));
        assert_eq!(
            b(399).radii_m,
            DVec3::new(6_378.1366e3, 6_378.1366e3, 6_356.7519e3)
        );
        assert_eq!(
            b(699).radii_m,
            DVec3::new(60_268.0e3, 60_268.0e3, 54_364.0e3)
        );
        assert_eq!(b(607).radii_m, DVec3::new(180.1e3, 133.0e3, 102.7e3));
        assert_eq!(b(999).radii_m, DVec3::splat(1_188.3e3));
        // Earth + Moon = DE440's Earth–Moon barycentre GM 403 503.235 625 km³/s² (to 1e-9).
        let emb = b(399).gm_m3s2 + b(301).gm_m3s2;
        assert!((emb / 4.0350323562548019e14 - 1.0).abs() < 1e-9, "{emb}");
    }
}
