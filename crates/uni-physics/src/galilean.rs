//! Jupiter's Galilean moons from Meeus ch. 44 (Lieske's E5 theory) (M1 Task A2).
//! A0b stub; A2 implements.

use glam::DVec3;
use uni_core::Epoch;

/// One of the four Galilean moons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Galilean {
    Io,
    Europa,
    Ganymede,
    Callisto,
}

/// Position relative to Jupiter's centre [m], ICRS axes.
pub fn jovicentric_icrs(_m: Galilean, _t: Epoch) -> DVec3 {
    todo!("A2")
}
