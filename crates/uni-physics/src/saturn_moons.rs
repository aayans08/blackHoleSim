//! Saturn's eight major moons from Meeus ch. 46 (Harper & Taylor 1993) (M1 Task A2).
//! A0b stub; A2 implements.

use glam::DVec3;
use uni_core::Epoch;

/// One of Saturn's eight major moons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaturnMoon {
    Mimas,
    Enceladus,
    Tethys,
    Dione,
    Rhea,
    Titan,
    Hyperion,
    Iapetus,
}

/// Position relative to Saturn's centre [m], ICRS axes.
pub fn saturnicentric_icrs(_m: SaturnMoon, _t: Epoch) -> DVec3 {
    todo!("A2")
}
