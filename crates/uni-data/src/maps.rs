//! The surface-map index `data/baked/maps/index.ron` (M1 Task A8), written by `uni-bake map`.
//!
//! Every map is an equirectangular (simple-cylindrical) global map: u runs east from `lon0_deg` at
//! the left edge through 360°, v runs from +90° latitude (top) to −90° (bottom), north up, with
//! square texels. `file` is the BC mip chain (`Bc7RgbaUnormSrgb` for `Albedo` and `CloudTops`,
//! `Bc4RUnorm` for `Clouds`); `fallback_file` is the box-filtered mip whose width is 2048 (or
//! mip 0 when the map is no wider) and the rest of its chain, uncompressed (`Rgba8UnormSrgb` or
//! `R8Unorm`), for GPUs without `TEXTURE_COMPRESSION_BC`.
//!
//! **Earth's clouds** (`Clouds`, one channel c ∈ [0, 1], stored linear) are composited over the
//! ground map as white clouds of coverage c, and both entries carry the same `albedo_scale` s:
//! ρ = s · [(1 − c) · ρ_ground + c]. The bake solves s with the clouds composited, so the
//! composite's disk average matches Earth's geometric albedo (review I4).

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
    /// Size of mip 0 of `file` [texels]: the decoded size of the source (every Solar System
    /// Scope map). A source that is not a 2:1 power-of-two size is box-filtered down to the
    /// largest one that fits, and `source` then gives its decoded size.
    pub width: u32,
    pub height: u32,
    /// Planetocentric east longitude at u = 0 (left edge) [deg].
    pub lon0_deg: f64,
    /// Multiplies decoded linear albedo so the map's disk-averaged reflectance gives the body's
    /// V geometric albedo (review I4).
    pub albedo_scale: f32,
    pub source: String,
}

/// Reads `maps/index.ron`. A missing file is [`DataError::NotFound`]; a file that does not parse,
/// or an entry with an empty file name, a zero size, a non-finite `lon0_deg` or a non-positive
/// `albedo_scale`, is [`DataError::BadFormat`].
pub fn load_map_index(path: &Path) -> Result<Vec<MapInfo>, DataError> {
    let text = std::fs::read_to_string(path).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    })?;
    let bad = |why: String| DataError::BadFormat(format!("{}: {why}", path.display()));
    let maps: Vec<MapInfo> = ron::from_str(&text).map_err(|err| bad(err.to_string()))?;
    for (i, map) in maps.iter().enumerate() {
        check(map).map_err(|why| bad(format!("entry {i} ({} {:?}): {why}", map.body, map.kind)))?;
    }
    Ok(maps)
}

fn check(map: &MapInfo) -> Result<(), String> {
    if map.file.is_empty() || map.fallback_file.is_empty() {
        return Err("empty file name".into());
    }
    if map.width == 0 || map.height == 0 {
        return Err(format!("size {}×{}", map.width, map.height));
    }
    if !map.lon0_deg.is_finite() {
        return Err(format!("lon0_deg {}", map.lon0_deg));
    }
    if !(map.albedo_scale.is_finite() && map.albedo_scale > 0.0) {
        return Err(format!("albedo_scale {}", map.albedo_scale));
    }
    Ok(())
}

/// Writes `maps` as `maps/index.ron` (one field per line), replacing any existing file.
pub fn write_map_index(path: &Path, maps: &[MapInfo]) -> Result<(), DataError> {
    let config = ron::ser::PrettyConfig::new().indentor("  ".to_string());
    let mut text = ron::ser::to_string_pretty(maps, config)
        .map_err(|err| DataError::BadFormat(format!("serialising the map index: {err}")))?;
    text.push('\n');
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::{TexFormat, load_tex};
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-maps-{}-{}", std::process::id(), name))
    }

    fn sample() -> Vec<MapInfo> {
        vec![
            MapInfo {
                body: "Earth".into(),
                naif: 399,
                kind: MapKind::Albedo,
                file: "earth_albedo.big".into(),
                fallback_file: "earth_albedo.rgba2k.big".into(),
                width: 8192,
                height: 4096,
                lon0_deg: -180.0,
                albedo_scale: 1.234_567_9,
                source: "Solar System Scope 8k_earth_daymap.jpg (CC BY 4.0)".into(),
            },
            MapInfo {
                body: "Earth".into(),
                naif: 399,
                kind: MapKind::Clouds,
                file: "earth_clouds.big".into(),
                fallback_file: "earth_clouds.r2k.big".into(),
                width: 8192,
                height: 4096,
                lon0_deg: -180.0,
                albedo_scale: 1.234_567_9,
                source: "Solar System Scope 8k_earth_clouds.jpg (CC BY 4.0)".into(),
            },
        ]
    }

    fn same(a: &MapInfo, b: &MapInfo) -> bool {
        a.body == b.body
            && a.naif == b.naif
            && a.kind == b.kind
            && a.file == b.file
            && a.fallback_file == b.fallback_file
            && (a.width, a.height) == (b.width, b.height)
            && a.lon0_deg.to_bits() == b.lon0_deg.to_bits()
            && a.albedo_scale.to_bits() == b.albedo_scale.to_bits()
            && a.source == b.source
    }

    #[test]
    fn map_index_ron_roundtrip() {
        let path = temp_path("index.ron");
        write_map_index(&path, &sample()).expect("write");
        let loaded = load_map_index(&path);
        let _ = std::fs::remove_file(&path);
        let loaded = loaded.expect("load");
        assert_eq!(loaded.len(), 2);
        for (a, b) in loaded.iter().zip(&sample()) {
            assert!(same(a, b), "{a:?} != {b:?}");
        }
    }

    #[test]
    fn load_map_index_missing_is_not_found() {
        assert!(matches!(
            load_map_index(Path::new("/nonexistent/maps/index.ron")),
            Err(DataError::NotFound(_))
        ));
    }

    #[test]
    fn load_map_index_rejects_bad_entries() {
        let path = temp_path("bad_index.ron");
        let mut cases = vec!["[(body: \"Earth\")]".to_string(), "not ron".to_string()];
        for edit in 0..4 {
            let mut maps = sample();
            match edit {
                0 => maps[0].file.clear(),
                1 => maps[0].width = 0,
                2 => maps[0].lon0_deg = f64::NAN,
                _ => maps[0].albedo_scale = 0.0,
            }
            cases.push(ron::to_string(&maps).expect("serialise"));
        }
        for (i, text) in cases.iter().enumerate() {
            std::fs::write(&path, text).expect("write");
            let loaded = load_map_index(&path);
            assert!(
                matches!(loaded, Err(DataError::BadFormat(_))),
                "case {i}: {loaded:?}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The committed index: every entry names its files, and every map baked on this machine
    /// (the `.big` files are git-ignored) loads with the index's size, the 2k fallback and the
    /// formats the renderer expects.
    #[test]
    fn baked_index_maps_load() {
        let dir = crate::data_dir().join("baked/maps");
        let maps = load_map_index(&dir.join("index.ron")).expect("committed maps/index.ron");
        assert!(!maps.is_empty());
        let mut baked = 0;
        for map in &maps {
            assert!((-360.0..=360.0).contains(&map.lon0_deg), "{map:?}");
            let bc = match load_tex(&dir.join(&map.file)) {
                Ok(tex) => tex,
                Err(DataError::NotFound(_)) => continue, // not baked on this machine
                Err(err) => panic!("{}: {err}", map.file),
            };
            let fallback = load_tex(&dir.join(&map.fallback_file)).expect("fallback");
            let (bc_format, fallback_format) = match map.kind {
                MapKind::Albedo | MapKind::CloudTops => {
                    (TexFormat::Bc7RgbaUnormSrgb, TexFormat::Rgba8UnormSrgb)
                }
                MapKind::Clouds => (TexFormat::Bc4RUnorm, TexFormat::R8Unorm),
            };
            assert_eq!((bc.format, fallback.format), (bc_format, fallback_format));
            assert_eq!(
                (bc.width, bc.height),
                (map.width, map.height),
                "{}",
                map.file
            );
            assert_eq!(map.width, 2 * map.height, "{}", map.file);
            assert_eq!(fallback.width, map.width.min(2048), "{}", map.fallback_file);
            assert_eq!(2 * fallback.height, fallback.width, "{}", map.fallback_file);
            assert_eq!(
                bc.mip_size(bc.mips.len() - 1).1,
                4,
                "{}: chain ends at 4 high",
                map.file
            );
            baked += 1;
        }
        eprintln!("{baked} of {} maps baked here and checked", maps.len());
    }
}
