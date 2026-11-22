//! Body orientation from the IAU WGCCRE 2015 report (Archinal et al. 2018, CMDA 130:22, and
//! the 2019 erratum) (M1 Task A2). A0b stub; A2 implements.

use glam::{DQuat, DVec3};
use uni_core::Epoch;

use crate::NaifId;

/// Body-fixed → ICRS rotation, `None` for a body without a rotation model.
pub fn body_to_icrs(_naif: NaifId, _t: Epoch) -> Option<DQuat> {
    todo!("A2")
}

/// Prime-meridian angle W [deg].
pub fn prime_meridian_deg(_naif: NaifId, _t: Epoch) -> Option<f64> {
    todo!("A2")
}

/// North pole (unit vector, ICRS axes).
pub fn pole_icrs(_naif: NaifId, _t: Epoch) -> Option<DVec3> {
    todo!("A2")
}
