//! uni-physics: Kepler/Standish/Meeus orbits, IAU rotation and photometry from M1; Galaxy
//! potential, integrators and relativity calculators in M3-M4.
//!
//! M1 modules (A0b stubs; bodies `todo!("<task>")` until the named task lands):
//! - A1: [`bodies`], [`kepler`], [`frames`], [`standish`], [`pluto`], [`ssb`]
//! - A2: [`moon`], [`galilean`], [`saturn_moons`], [`sat_elements`], [`rotation`], [`ephemeris`]
//! - A3: [`photometry`], [`smallbody`], [`dust`], [`calc`]
//!
//! One exception: [`photometry::rho_from_geometric_albedo`] is already implemented (A0b fix
//! round 1, review I3), because A8 calls it before A3 lands.
//!
//! CPU maths is SI units in `f64`; positions are ICRS/J2000 axes (CLAUDE.md §3).

pub mod bodies;
pub mod calc;
pub mod dust;
pub mod ephemeris;
pub mod frames;
pub mod galilean;
pub mod kepler;
pub mod moon;
pub mod photometry;
pub mod pluto;
pub mod rotation;
pub mod sat_elements;
pub mod saturn_moons;
pub mod smallbody;
pub mod ssb;
pub mod standish;

/// NAIF integer ID of a body or barycentre (SPICE convention: 0 SSB, 1–9 planetary-system
/// barycentres, 10 Sun, 199 Mercury, 399 Earth, 301 Moon, 501 Io, 999 Pluto).
pub type NaifId = i32;
