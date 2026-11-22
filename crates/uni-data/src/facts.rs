//! Body fact sheets (`body_facts.ron`), hand-curated from the NASA NSSDCA planetary and
//! satellite fact sheets (M1 Task A9). A0b stub; A9 implements.
//!
//! Sequencing (A0b fix rounds 1–2, review I3): A9's `facts_cover_major_bodies` compares radii with
//! `uni_physics::bodies::major_bodies` (dev-dependency). The lead's A0b implemented that table
//! ahead of A1 (WGCCRE 2015 radii, DE440 GM), so the test runs for real while A1 works in
//! parallel. Its `radii_m` are triaxial (a, b, c); an NSSDCA mean radius compares with the
//! volumetric mean (a·b·c)^⅓ (Saturn: 58 232 km, not the 60 268 km equatorial radius).

use std::path::Path;

use crate::DataError;

/// The info-card facts of one body.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BodyFacts {
    pub naif: i32,
    pub name: String,
    pub kind: String,
    pub mean_radius_km: f64,
    pub mass_kg: f64,
    pub surface_gravity_ms2: Option<f64>,
    pub escape_velocity_kms: Option<f64>,
    pub sidereal_rotation_h: Option<f64>,
    pub orbital_period_d: Option<f64>,
    pub semi_major_au: Option<f64>,
    pub axial_tilt_deg: Option<f64>,
    pub surface_pressure_kpa: Option<f64>,
    pub atmosphere: Option<String>,
    pub mean_temp_k: Option<f64>,
    pub bond_albedo: Option<f64>,
    /// V-band geometric albedo.
    pub geometric_albedo: Option<f64>,
    pub moons: Vec<String>,
    /// The fact-sheet URL of every value.
    pub source: String,
}

/// Reads `body_facts.ron`.
pub fn load_body_facts(_path: &Path) -> Result<Vec<BodyFacts>, DataError> {
    todo!("A9")
}
