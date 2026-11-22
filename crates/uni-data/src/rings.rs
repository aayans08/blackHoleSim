//! Saturn's ring radial profile `saturn_rings.ron` (M1 Task A8). A0b stub; A8 implements.

use std::path::Path;

use crate::DataError;

/// Ring colour and opacity sampled uniformly in radius from `inner_m` to `outer_m`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RingProfile {
    pub inner_m: f64,
    pub outer_m: f64,
    /// Linear RGB reflectance and opacity per sample.
    pub samples: Vec<[f32; 4]>,
}

/// Reads `saturn_rings.ron`.
pub fn load_ring_profile(_path: &Path) -> Result<RingProfile, DataError> {
    todo!("A8")
}
