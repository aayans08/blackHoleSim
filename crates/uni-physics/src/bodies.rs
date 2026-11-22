//! Physical constants of the major bodies (M1 Task A1): GM and triaxial radii.
//!
//! Radii: IAU WGCCRE 2015 (Archinal et al. 2018, CMDA 130:22). GM: DE440 (Park et al. 2021,
//! AJ 161, 105) and the JPL satellite physical parameters. A0b stub; A1 implements.

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

/// Every body with constants: the Sun, the eight planets, Pluto, the Moon, the Galileans,
/// Saturn's eight major moons, Phobos, Deimos, Triton, Titania, Oberon and Charon.
pub fn major_bodies() -> &'static [BodyConst] {
    todo!("A1")
}

/// The body with NAIF id `id`, if it is in [`major_bodies`].
pub fn by_naif(_id: NaifId) -> Option<&'static BodyConst> {
    todo!("A1")
}
