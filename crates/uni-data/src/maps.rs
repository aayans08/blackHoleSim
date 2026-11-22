//! The surface-map index `data/baked/maps/index.ron` (M1 Task A8). A0b stub; A8 implements.

use std::path::Path;

use crate::DataError;

/// What a map holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MapKind {
    Albedo,
    Clouds,
    CloudTops,
}

/// One baked map.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MapInfo {
    pub body: String,
    pub naif: i32,
    pub kind: MapKind,
    /// The BC-compressed file, relative to the index.
    pub file: String,
    /// The 2k uncompressed copy, relative to the index.
    pub fallback_file: String,
    /// Decoded size of the source [texels].
    pub width: u32,
    pub height: u32,
    /// Planetocentric east longitude at u = 0 (left edge) [deg].
    pub lon0_deg: f64,
    /// Multiplies decoded linear albedo so the map's disk-averaged reflectance gives the body's
    /// V geometric albedo (review I4).
    pub albedo_scale: f32,
    pub source: String,
}

/// Reads `maps/index.ron`.
pub fn load_map_index(_path: &Path) -> Result<Vec<MapInfo>, DataError> {
    todo!("A8")
}
