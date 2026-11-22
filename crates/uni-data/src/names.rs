//! Star names for the AT-HYG stars (`athyg_names.ron`, CC BY-SA 4.0) (M1 Task A7).
//! A0b stub; A7 implements.

use std::path::Path;

use crate::DataError;

/// Names of one star.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StarName {
    /// Index of the star in the baked `athyg_m10.bin`.
    pub index: u32,
    pub proper: Option<String>,
    pub bayer_flamsteed: Option<String>,
    /// Constellation abbreviation.
    pub con: Option<String>,
    /// Hipparcos number.
    pub hip: Option<u32>,
    /// Spectral type.
    pub spect: Option<String>,
}

/// Every named star.
#[derive(Clone, Debug, Default)]
pub struct StarNames {
    pub entries: Vec<StarName>,
}

/// Reads `athyg_names.ron`.
pub fn load_star_names(_path: &Path) -> Result<StarNames, DataError> {
    todo!("A7")
}
