//! Baked textures in our own `UNITEX1` container (magic `b"UNITEX1\0"`): BC7 albedo and BC4
//! single-channel mip chains, plus 2k uncompressed fallbacks (M1 Task A8). There is no runtime BC
//! decoder (review I17): without `TEXTURE_COMPRESSION_BC` the renderer loads the fallback file.
//!
//! The container is a deliberate deviation from spec §5.4's KTX2 (no KTX2 crate is approved); the
//! texel formats are the spec's. Layout, all little-endian:
//! - header, 24 B: magic `b"UNITEX1\0"`, `u32` format code ([`TexFormat`]), `u32` width,
//!   `u32` height, `u32` mip count
//! - the mips, largest first, back to back. Mip `l` is `max(1, width >> l)` × `max(1, height >> l)`
//!   texels and holds exactly [`TexFormat::mip_len`] bytes: whole 4×4 blocks for the BC formats
//!   (rows of blocks top to bottom, blocks left to right), tightly packed texel rows otherwise.

use std::path::Path;

use crate::DataError;

/// Texel format of a baked texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TexFormat {
    Bc7RgbaUnormSrgb,
    /// Only if `intel_tex_2` fails to link (A8's own-encoder path).
    Bc1RgbaUnormSrgb,
    Bc4RUnorm,
    Rgba8UnormSrgb,
    R8Unorm,
}

impl TexFormat {
    /// Every format, in file-code order (code = index + 1).
    const ALL: [TexFormat; 5] = [
        TexFormat::Bc7RgbaUnormSrgb,
        TexFormat::Bc1RgbaUnormSrgb,
        TexFormat::Bc4RUnorm,
        TexFormat::Rgba8UnormSrgb,
        TexFormat::R8Unorm,
    ];

    fn code(self) -> u32 {
        match self {
            TexFormat::Bc7RgbaUnormSrgb => 1,
            TexFormat::Bc1RgbaUnormSrgb => 2,
            TexFormat::Bc4RUnorm => 3,
            TexFormat::Rgba8UnormSrgb => 4,
            TexFormat::R8Unorm => 5,
        }
    }

    fn from_code(code: u32) -> Option<TexFormat> {
        TexFormat::ALL.into_iter().find(|f| f.code() == code)
    }

    /// Texels per block edge: 4 for the BC formats, 1 for the uncompressed ones.
    pub fn block_dim(self) -> u32 {
        match self {
            TexFormat::Bc7RgbaUnormSrgb | TexFormat::Bc1RgbaUnormSrgb | TexFormat::Bc4RUnorm => 4,
            TexFormat::Rgba8UnormSrgb | TexFormat::R8Unorm => 1,
        }
    }

    /// Bytes per block (BC: one 4×4 block; uncompressed: one texel).
    pub fn block_bytes(self) -> u32 {
        match self {
            TexFormat::Bc7RgbaUnormSrgb => 16,
            TexFormat::Bc1RgbaUnormSrgb | TexFormat::Bc4RUnorm => 8,
            TexFormat::Rgba8UnormSrgb => 4,
            TexFormat::R8Unorm => 1,
        }
    }

    /// Bytes in one row of blocks of a mip `width` texels wide (wgpu's `bytes_per_row`).
    pub fn row_bytes(self, width: u32) -> u64 {
        u64::from(width.div_ceil(self.block_dim())) * u64::from(self.block_bytes())
    }

    /// Bytes in a mip of `width` × `height` texels.
    pub fn mip_len(self, width: u32, height: u32) -> u64 {
        self.row_bytes(width) * u64::from(height.div_ceil(self.block_dim()))
    }
}

/// A texture and its mip chain (mip 0 first).
#[derive(Clone, Debug, PartialEq)]
pub struct TexFile {
    pub format: TexFormat,
    pub width: u32,
    pub height: u32,
    pub mips: Vec<Vec<u8>>,
}

impl TexFile {
    /// Size of mip `level` [texels]: `max(1, width >> level)` × `max(1, height >> level)`.
    pub fn mip_size(&self, level: usize) -> (u32, u32) {
        mip_size(self.width, self.height, level)
    }
}

const MAGIC: [u8; 8] = *b"UNITEX1\0";
const HEADER_LEN: usize = 24;

fn mip_size(width: u32, height: u32, level: usize) -> (u32, u32) {
    let shift = |v: u32| v.checked_shr(level as u32).unwrap_or(0).max(1);
    (shift(width), shift(height))
}

/// Checks the header fields and returns each mip's byte length.
fn mip_lens(format: TexFormat, width: u32, height: u32, count: usize) -> Result<Vec<u64>, String> {
    if width == 0 || height == 0 {
        return Err(format!("size {width}×{height} is empty"));
    }
    // A full chain has 1 + floor(log2(max side)) levels.
    let max_levels = 32 - width.max(height).leading_zeros() as usize;
    if count == 0 || count > max_levels {
        return Err(format!(
            "{count} mips; a {width}×{height} texture has 1..={max_levels}"
        ));
    }
    Ok((0..count)
        .map(|level| {
            let (w, h) = mip_size(width, height, level);
            format.mip_len(w, h)
        })
        .collect())
}

/// Reads a `UNITEX1` file. Missing → [`DataError::NotFound`]; truncated or malformed →
/// [`DataError::BadFormat`].
pub fn load_tex(path: &Path) -> Result<TexFile, DataError> {
    let bytes = std::fs::read(path).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    })?;
    decode(&bytes).map_err(|why| DataError::BadFormat(format!("{}: {why}", path.display())))
}

/// Writes `t` as a `UNITEX1` file, replacing any existing file. A texture whose mip lengths do not
/// match its format and size is [`DataError::BadFormat`] and nothing is written.
pub fn write_tex(path: &Path, t: &TexFile) -> Result<(), DataError> {
    let lens = mip_lens(t.format, t.width, t.height, t.mips.len()).map_err(DataError::BadFormat)?;
    for (level, (mip, &len)) in t.mips.iter().zip(&lens).enumerate() {
        if mip.len() as u64 != len {
            return Err(DataError::BadFormat(format!(
                "mip {level} of a {:?} {}×{} texture has {} B, expected {len}",
                t.format,
                t.width,
                t.height,
                mip.len()
            )));
        }
    }
    let total: u64 = lens.iter().sum();
    let mut bytes = Vec::with_capacity(HEADER_LEN + total as usize);
    bytes.extend_from_slice(&MAGIC);
    for value in [t.format.code(), t.width, t.height, t.mips.len() as u32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for mip in &t.mips {
        bytes.extend_from_slice(mip);
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn decode(bytes: &[u8]) -> Result<TexFile, String> {
    let (header, mut body) = bytes.split_at_checked(HEADER_LEN).ok_or_else(|| {
        format!(
            "{} bytes is shorter than the {HEADER_LEN}-byte header",
            bytes.len()
        )
    })?;
    if header[..8] != MAGIC {
        return Err("bad magic: not a UNITEX1 file".into());
    }
    let field = |i: usize| u32::from_le_bytes(std::array::from_fn(|k| header[8 + 4 * i + k]));
    let code = field(0);
    let format = TexFormat::from_code(code).ok_or_else(|| format!("unknown format code {code}"))?;
    let (width, height, count) = (field(1), field(2), field(3) as usize);
    let lens = mip_lens(format, width, height, count)?;
    let total: u64 = lens.iter().sum();
    if body.len() as u64 != total {
        return Err(format!(
            "a {format:?} {width}×{height} texture with {count} mips needs {total} B of texels, \
             but {} B follow the header",
            body.len()
        ));
    }
    let mut mips = Vec::with_capacity(count);
    for len in lens {
        let (mip, rest) = body.split_at(len as usize);
        mips.push(mip.to_vec());
        body = rest;
    }
    Ok(TexFile {
        format,
        width,
        height,
        mips,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A per-process scratch path (no tempfile crate).
    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-tex-{}-{}", std::process::id(), name))
    }

    /// A 12×8 BC4 texture with its full block-aligned chain (12×8, 6×4, 3×2, 1×1).
    fn sample_bc4() -> TexFile {
        let mips = (0..4)
            .map(|level| {
                let (w, h) = mip_size(12, 8, level);
                let len = TexFormat::Bc4RUnorm.mip_len(w, h) as usize;
                (0..len).map(|i| (i * 7 + level) as u8).collect()
            })
            .collect();
        TexFile {
            format: TexFormat::Bc4RUnorm,
            width: 12,
            height: 8,
            mips,
        }
    }

    #[test]
    fn mip_lengths_follow_the_block_size() {
        // BC7: 16 B per 4×4 block; partial blocks round up (a 3×2 mip is one block).
        assert_eq!(TexFormat::Bc7RgbaUnormSrgb.mip_len(8192, 4096), 8192 * 4096);
        assert_eq!(TexFormat::Bc7RgbaUnormSrgb.mip_len(3, 2), 16);
        assert_eq!(TexFormat::Bc4RUnorm.mip_len(8, 4), 16);
        assert_eq!(
            TexFormat::Rgba8UnormSrgb.mip_len(2048, 1024),
            2048 * 1024 * 4
        );
        assert_eq!(TexFormat::R8Unorm.mip_len(3, 1), 3);
        assert_eq!(TexFormat::Bc7RgbaUnormSrgb.row_bytes(8192), 2048 * 16);
    }

    #[test]
    fn tex_roundtrip() {
        let path = temp_path("roundtrip.big");
        let tex = sample_bc4();
        write_tex(&path, &tex).expect("write");
        let loaded = load_tex(&path);
        let _ = std::fs::remove_file(&path);
        let loaded = loaded.expect("load");
        assert_eq!(loaded, tex);
        assert_eq!(loaded.mip_size(2), (3, 2));
        assert_eq!(loaded.mip_size(3), (1, 1));
    }

    #[test]
    fn every_format_roundtrips() {
        for format in TexFormat::ALL {
            let path = temp_path(&format!("{format:?}.big"));
            let tex = TexFile {
                format,
                width: 8,
                height: 4,
                mips: vec![vec![9; format.mip_len(8, 4) as usize]],
            };
            write_tex(&path, &tex).expect("write");
            let loaded = load_tex(&path);
            let _ = std::fs::remove_file(&path);
            assert_eq!(loaded.expect("load"), tex, "{format:?}");
        }
    }

    #[test]
    fn load_tex_missing_is_not_found() {
        // Review Focus 3: a missing baked map is reported, not a panic.
        assert!(matches!(
            load_tex(Path::new("/nonexistent/earth_albedo.big")),
            Err(DataError::NotFound(_))
        ));
    }

    #[test]
    fn load_tex_truncated_is_bad_format() {
        // Review Focus 3: a truncated `.big` (an interrupted bake or copy) is reported, not a panic.
        let path = temp_path("truncated.big");
        write_tex(&path, &sample_bc4()).expect("write");
        let full = std::fs::read(&path).expect("read");
        for cut in [1, 8, full.len() - HEADER_LEN, full.len() - 3] {
            std::fs::write(&path, &full[..full.len() - cut]).expect("rewrite");
            let loaded = load_tex(&path);
            assert!(
                matches!(loaded, Err(DataError::BadFormat(_))),
                "cut {cut}: {loaded:?}"
            );
        }
        // Trailing garbage is as wrong as a missing tail.
        let mut long = full.clone();
        long.push(0);
        std::fs::write(&path, &long).expect("rewrite");
        let loaded = load_tex(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }

    #[test]
    fn load_tex_bad_header_is_bad_format() {
        let path = temp_path("bad_header.big");
        let good = {
            write_tex(&path, &sample_bc4()).expect("write");
            std::fs::read(&path).expect("read")
        };
        let corrupt = |offset: usize, value: u32| {
            let mut bytes = good.clone();
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            bytes
        };
        let cases = [
            corrupt(0, 0x4E49_5500),  // magic
            corrupt(8, 0),            // format code 0
            corrupt(8, 99),           // unknown format code
            corrupt(12, 0),           // zero width
            corrupt(20, 0),           // no mips
            corrupt(20, 40),          // more mips than a 12×8 chain has
            corrupt(16, 0xFFFF_FFFF), // a height the file cannot hold
        ];
        for (i, bytes) in cases.iter().enumerate() {
            std::fs::write(&path, bytes).expect("rewrite");
            let loaded = load_tex(&path);
            assert!(
                matches!(loaded, Err(DataError::BadFormat(_))),
                "case {i}: {loaded:?}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_tex_rejects_wrong_mip_lengths() {
        let path = temp_path("wrong_len.big");
        let mut tex = sample_bc4();
        tex.mips[1].pop();
        assert!(matches!(
            write_tex(&path, &tex),
            Err(DataError::BadFormat(_))
        ));
        tex.mips.clear();
        assert!(matches!(
            write_tex(&path, &tex),
            Err(DataError::BadFormat(_))
        ));
        assert!(!path.exists());
    }
}
