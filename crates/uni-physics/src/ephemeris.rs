//! The M1 ephemeris: which theory places each body, and positions relative to its parent and to
//! the SSB (M1 Task A2). A0b stub; A2 implements.

use glam::DVec3;
use uni_core::Epoch;

use crate::NaifId;
use crate::sat_elements::SatElements;

/// The theory that places a body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theory {
    SunSsb,
    Standish,
    MeeusPluto,
    EarthFromEmb,
    MeeusMoon,
    MeeusGalilean,
    MeeusSaturn,
    JplMeanElements,
}

/// One body the ephemeris places.
#[derive(Clone, Debug)]
pub struct EphemBody {
    pub naif: NaifId,
    pub name: String,
    /// Parent body (planets: the Sun, 10); `None` for the Sun.
    pub parent: Option<NaifId>,
    pub theory: Theory,
}

/// Conditions at an epoch that the HUD labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EphemFlag {
    PlanetsExtrapolated,
    PlutoExtrapolated,
    MoonTheoryFallback,
}

/// Every body of the M1 Solar System with its theory.
pub struct Ephemeris {
    /// Placeholder for the private state; A2 replaces it.
    _private: (),
}

impl Ephemeris {
    /// Builds the ephemeris from the moons' mean elements.
    pub fn new(_sats: Vec<SatElements>) -> Self {
        todo!("A2")
    }

    pub fn bodies(&self) -> &[EphemBody] {
        todo!("A2")
    }

    /// Position relative to the parent [m], ICRS axes; the parent of planets is the Sun (10).
    pub fn position_rel_parent(&self, _naif: NaifId, _t: Epoch) -> Option<DVec3> {
        todo!("A2")
    }

    /// Position relative to the SSB [m], ICRS axes.
    pub fn ssb_position(&self, _naif: NaifId, _t: Epoch) -> Option<DVec3> {
        todo!("A2")
    }

    pub fn flags(&self, _t: Epoch) -> Vec<EphemFlag> {
        todo!("A2")
    }
}
