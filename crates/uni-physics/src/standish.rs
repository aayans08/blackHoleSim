//! Planet positions from JPL's "Approximate Positions of the Planets" (Standish & Williams;
//! https://ssd.jpl.nasa.gov/planets/approx_pos.html) (M1 Task A1). Pluto is in [`crate::pluto`]
//! (review I1). A0b stub; A1 implements.

use glam::DVec3;
use uni_core::Epoch;

use crate::NaifId;

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
    /// NAIF id of the planetary-system barycentre, 1..=8 (the Horizons targets; review I2).
    pub fn barycentre_id(self) -> NaifId {
        todo!("A1")
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

/// The table used at `t`.
pub fn validity(_t: Epoch) -> Validity {
    todo!("A1")
}

/// Heliocentric position [m] and velocity [m/s] of `p` in ICRS axes.
pub fn heliocentric_state_icrs(_p: Planet, _t: Epoch) -> (DVec3, DVec3) {
    todo!("A1")
}
