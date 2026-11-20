//! uni-data: runtime loaders for baked catalogues. M0: the Yale Bright Star Catalogue (BSC5).

pub mod bsc5;

pub use bsc5::{CatalogStar, load_bsc5, write_bsc5};

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
