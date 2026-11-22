//! Baked textures in our own `UNITEX1` container (magic `b"UNITEX1\0"`): BC7 albedo and BC4
//! single-channel mip chains, plus 2k uncompressed fallbacks (M1 Task A8). There is no runtime BC
//! decoder (review I17): without `TEXTURE_COMPRESSION_BC` the renderer loads the fallback file.
//! A0b stub; A8 implements.

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

/// A texture and its mip chain (mip 0 first).
#[derive(Clone, Debug, PartialEq)]
pub struct TexFile {
    pub format: TexFormat,
    pub width: u32,
    pub height: u32,
    pub mips: Vec<Vec<u8>>,
}

/// Reads a `UNITEX1` file. Missing → [`DataError::NotFound`]; truncated or malformed →
/// [`DataError::BadFormat`].
pub fn load_tex(_path: &Path) -> Result<TexFile, DataError> {
    todo!("A8")
}

/// Writes `t` as a `UNITEX1` file, replacing any existing file.
pub fn write_tex(_path: &Path, _t: &TexFile) -> Result<(), DataError> {
    todo!("A8")
}
