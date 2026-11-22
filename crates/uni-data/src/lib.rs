//! uni-data: runtime loaders for baked catalogues. M0: the Yale Bright Star Catalogue (BSC5).
//! M1 (A0b stubs until the named task lands): AT-HYG stars, names and exoplanets (A7); surface
//! maps, textures and Saturn's rings (A8); small bodies, satellite elements, body facts and the
//! Horizons goldens (A9).

pub mod athyg;
pub mod bsc5;
pub mod exo;
pub mod f16;
pub mod facts;
pub mod golden;
pub mod maps;
pub mod names;
pub mod rings;
pub mod sats;
pub mod smallbody;
pub mod texture;

pub use athyg::{
    FLAG_EXO_HOST, FLAG_FROM_ARCHIVE, FLAG_NAMED, FLAG_NO_RV, FLAG_NO_VELOCITY, STAR3_EPOCH_JD_TDB,
    STAR3_SIZE, Star3, load_athyg, write_athyg,
};
pub use bsc5::{CatalogStar, load_bsc5, write_bsc5};
pub use exo::{ExoPlanet, ExoSystem, load_exoplanets};
pub use f16::{f16_to_f32, f32_to_f16};
pub use facts::{BodyFacts, load_body_facts};
pub use golden::{EphemSpot, SkySpot, load_ephem_spots, load_sky_spots};
pub use maps::{MapInfo, MapKind, load_map_index};
pub use names::{StarName, StarNames, load_star_names};
pub use rings::{RingProfile, load_ring_profile};
pub use sats::{SatRecord, load_sat_elements};
pub use smallbody::{AsteroidRecord, Comet, load_asteroids, load_comets};
pub use texture::{TexFile, TexFormat, load_tex, write_tex};

// A0b pre-adds every Wave A manifest entry (review I8), so a crate can be declared before the task
// that uses it lands. This line keeps `cargo::unused_dependencies` quiet until then; the lead
// deletes it once `ron` is used.
use ron as _;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Errors from loading or writing a baked data file.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("not found: {0}")]
    NotFound(PathBuf),
    #[error("bad format: {0}")]
    BadFormat(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Environment variable that overrides [`data_dir`].
pub const DATA_DIR_ENV: &str = "UNI_DATA_DIR";

/// The data directory: `$UNI_DATA_DIR` when it is set and non-empty, otherwise the workspace
/// `data/` directory, found at compile time from this crate's manifest directory.
pub fn data_dir() -> PathBuf {
    data_dir_from(std::env::var_os(DATA_DIR_ENV))
}

fn data_dir_from(override_dir: Option<OsString>) -> PathBuf {
    match override_dir {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
            std::fs::canonicalize(&dir).unwrap_or(dir)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_dir_honours_override() {
        let dir = data_dir_from(Some(OsString::from("/some/where/data")));
        assert_eq!(dir, PathBuf::from("/some/where/data"));
    }

    #[test]
    fn empty_override_falls_back_to_workspace_data() {
        assert_eq!(data_dir_from(Some(OsString::new())), data_dir_from(None));
    }

    #[test]
    fn default_data_dir_is_the_workspace_data_dir() {
        let dir = data_dir_from(None);
        assert!(dir.is_absolute(), "{}", dir.display());
        assert!(
            dir.join("manifest.toml").is_file(),
            "no manifest.toml in {}",
            dir.display()
        );
    }
}
