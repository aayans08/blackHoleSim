//! Loader for the baked Yale Bright Star Catalogue, 5th revised ed. (Hoffleit & Warren 1991,
//! CDS V/50) (M0 Task 7). `uni-bake bsc5` writes the file from the raw catalogue.
//!
//! Binary format v1, all little-endian:
//! - header, 16 B: magic `b"UNIBSC5\0"`, `u32` version = 1, `u32` star count
//! - `count` records of 24 B: `f32 × 3` ICRS unit direction, `f32` V magnitude, `f32` Teff [K],
//!   `u32` HR number

use crate::DataError;
use glam::Vec3;
use std::path::Path;

/// One star of the baked catalogue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatalogStar {
    /// Unit direction from the Solar System barycentre in ICRS/J2000 axes.
    pub dir_icrs: Vec3,
    /// Apparent visual magnitude (Johnson V).
    pub v_mag: f32,
    /// Effective temperature [K].
    pub teff_k: f32,
    /// Harvard Revised number (= Bright Star number).
    pub hr: u32,
}

const MAGIC: [u8; 8] = *b"UNIBSC5\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 16;
const RECORD_LEN: usize = 24;
/// Largest accepted deviation of |direction| from 1. The bake stores f64 unit vectors rounded
/// to f32 (error about 1e-7), so anything near this bound means a corrupt file.
const UNIT_TOLERANCE: f32 = 1e-4;

/// Reads a baked `bsc5.bin`.
///
/// A missing file is [`DataError::NotFound`]. A file with the wrong magic, version or length, or
/// with a non-finite value or non-unit direction, is [`DataError::BadFormat`].
pub fn load_bsc5(path: &Path) -> Result<Vec<CatalogStar>, DataError> {
    let bytes = std::fs::read(path).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    })?;
    decode(&bytes).map_err(|why| DataError::BadFormat(format!("{}: {why}", path.display())))
}

/// Writes `stars` as a baked `bsc5.bin`, replacing any existing file.
pub fn write_bsc5(path: &Path, stars: &[CatalogStar]) -> Result<(), DataError> {
    let count = u32::try_from(stars.len()).map_err(|_| {
        DataError::BadFormat(format!("{} stars do not fit a u32 count", stars.len()))
    })?;
    let mut bytes = Vec::with_capacity(HEADER_LEN + RECORD_LEN * stars.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    for star in stars {
        let d = star.dir_icrs;
        for value in [d.x, d.y, d.z, star.v_mag, star.teff_k] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&star.hr.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn decode(bytes: &[u8]) -> Result<Vec<CatalogStar>, String> {
    let (header, body) = bytes.split_at_checked(HEADER_LEN).ok_or_else(|| {
        format!("{} bytes is shorter than the {HEADER_LEN}-byte header", bytes.len())
    })?;
    if header[..8] != MAGIC {
        return Err("bad magic: not a UNIBSC5 file".into());
    }
    let version = le_u32(header, 8);
    if version != VERSION {
        return Err(format!("version {version}; this build reads {VERSION}"));
    }
    let count = le_u32(header, 12) as usize;
    if Some(body.len()) != count.checked_mul(RECORD_LEN) {
        return Err(format!(
            "header says {count} stars ({} B) but {} B follow it",
            count.saturating_mul(RECORD_LEN),
            body.len()
        ));
    }
    body.chunks_exact(RECORD_LEN)
        .enumerate()
        .map(|(index, record)| decode_record(index, record))
        .collect()
}

fn decode_record(index: usize, record: &[u8]) -> Result<CatalogStar, String> {
    let f = |i: usize| f32::from_bits(le_u32(record, 4 * i));
    let star = CatalogStar {
        dir_icrs: Vec3::new(f(0), f(1), f(2)),
        v_mag: f(3),
        teff_k: f(4),
        hr: le_u32(record, 20),
    };
    let finite = star.dir_icrs.is_finite() && star.v_mag.is_finite() && star.teff_k.is_finite();
    if !finite || (star.dir_icrs.length() - 1.0).abs() > UNIT_TOLERANCE {
        return Err(format!("record {index} (HR {}) is invalid: {star:?}", star.hr));
    }
    Ok(star)
}

/// The little-endian `u32` at `offset`; callers guarantee `offset + 4 <= bytes.len()`.
fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(std::array::from_fn(|i| bytes[offset + i]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A per-process scratch path (no tempfile crate).
    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-{}-{}", std::process::id(), name))
    }

    fn sample_stars() -> [CatalogStar; 3] {
        [
            CatalogStar {
                dir_icrs: Vec3::X,
                v_mag: -1.46,
                teff_k: 10_125.0,
                hr: 2491,
            },
            CatalogStar {
                dir_icrs: Vec3::new(0.6, 0.0, -0.8),
                v_mag: 0.03,
                teff_k: 5800.0,
                hr: 7001,
            },
            CatalogStar {
                dir_icrs: Vec3::Z,
                v_mag: 6.5,
                teff_k: 3169.5,
                hr: 9110,
            },
        ]
    }

    #[test]
    fn record_is_24_bytes() {
        let path = temp_path("record_is_24_bytes.bin");
        write_bsc5(&path, &sample_stars()).expect("write");
        let len = std::fs::metadata(&path).expect("metadata").len();
        let _ = std::fs::remove_file(&path);
        assert_eq!(len, 16 + 3 * 24);
    }

    #[test]
    fn roundtrip() {
        let path = temp_path("roundtrip.bin");
        write_bsc5(&path, &sample_stars()).expect("write");
        let loaded = load_bsc5(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.expect("load"), sample_stars());
    }

    #[test]
    fn load_missing_file_is_not_found() {
        // Review Focus 3: a missing baked file is reported, not a panic.
        assert!(matches!(
            load_bsc5(Path::new("/nonexistent/x.bin")),
            Err(DataError::NotFound(_))
        ));
    }

    #[test]
    fn load_bad_magic_is_bad_format() {
        // Review Focus 3: a corrupt baked file is reported, not a panic.
        let path = temp_path("bad_magic.bin");
        std::fs::write(&path, [0u8; 32]).expect("write");
        let loaded = load_bsc5(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }

    #[test]
    fn load_truncated_file_is_bad_format() {
        let path = temp_path("truncated.bin");
        write_bsc5(&path, &sample_stars()).expect("write");
        let mut bytes = std::fs::read(&path).expect("read");
        bytes.truncate(bytes.len() - 1);
        std::fs::write(&path, &bytes).expect("rewrite");
        let loaded = load_bsc5(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }

    #[test]
    fn load_non_finite_value_is_bad_format() {
        let mut stars = sample_stars();
        stars[1].v_mag = f32::NAN;
        let path = temp_path("non_finite.bin");
        write_bsc5(&path, &stars).expect("write");
        let loaded = load_bsc5(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }

    #[test]
    fn committed_bake_loads() {
        // data/baked/bsc5.bin is committed (M0 Task 7): 9110 records minus 14 without a position.
        let path = crate::data_dir_from(None).join("baked/bsc5.bin");
        let stars = load_bsc5(&path).expect("committed bsc5.bin");
        assert_eq!(stars.len(), 9096);
        let brightest = stars
            .iter()
            .min_by(|a, b| a.v_mag.total_cmp(&b.v_mag))
            .expect("non-empty");
        assert_eq!((brightest.hr, brightest.v_mag), (2491, -1.46), "Sirius");
    }

    #[test]
    fn load_non_unit_direction_is_bad_format() {
        let mut stars = sample_stars();
        stars[2].dir_icrs = Vec3::new(0.0, 0.0, 2.0);
        let path = temp_path("non_unit.bin");
        write_bsc5(&path, &stars).expect("write");
        let loaded = load_bsc5(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }
}
