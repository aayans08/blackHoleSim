//! The surface-map index `data/baked/maps/index.ron` (M1 Task A8), written by `uni-bake map`.
//!
//! Every map is an equirectangular (simple-cylindrical) global map: u runs east from `lon0_deg` at
//! the left edge through 360°, v runs from +90° latitude (top) to −90° (bottom), north up, with
//! square texels. `file` is the BC mip chain (`Bc7RgbaUnormSrgb` for `Albedo` and `CloudTops`,
//! `Bc4RUnorm` for `Clouds`); `fallback_file` is the box-filtered mip whose width is 2048 (or
//! mip 0 when the map is no wider) and the rest of its chain, uncompressed (`Rgba8UnormSrgb` or
//! `R8Unorm`), for GPUs without `TEXTURE_COMPRESSION_BC`.
//!
//! **Sizes.** `width`/`height` record the source's decoded size. The files hold mip 0 at
//! [`baked_size`]`(width)`: the decoded size itself for a 2:1 power of two (every Solar System
//! Scope map), otherwise the nearest 2:1 power of two, so every mip stays block-aligned (Ganymede
//! 8271×4135 and Ceres 7383×3691 are stored at 8192×4096). A loaded [`crate::TexFile`] carries
//! its stored size.
//!
//! **Reflectance ≤ 1.** The reflectance of a texel is `albedo_scale` × its decoded linear value,
//! per channel, and it never exceeds 1 (energy conservation; the M1 plan's "ρ is capped at 1",
//! review I4). For most maps the bake solves `albedo_scale` so the map's disk average gives the
//! body's geometric albedo, with that cap inside the disk average. When the geometric albedo needs
//! texels brighter than 1, the file stores the capped reflectance itself, min(1, s · decoded), and
//! `albedo_scale` is 1; the solved s is in `source`. The renderer multiplies by `albedo_scale` and
//! needs no clamp of its own. Two maps are not normalised to p_V (A8 fix round 2; `source` says
//! how each was calibrated and what disk average it gives):
//! - **Earth's ground** is calibrated on its ocean (the map's modal colour) at the sea surface's
//!   measured reflectance. p_V also holds Rayleigh scattering, which the atmosphere draws, and
//!   Earth's real cloud cover; pushed into the ground they made it 4–8× too bright.
//! - **Venus's cloud tops** (`CloudTops`) are neutral grey: the source's luminance pattern with a
//!   disk-averaged ρ below 1 and the largest contrast that stays ≤ 1. p_V 0.689 exceeds a Lambert
//!   sphere's 2/3, so matching it would leave only a uniform white map.
//!
//! **Earth's clouds** (`Clouds`, one channel c ∈ [0, 1], stored linear: the source's DN/255 as
//! is) are read as the **areal coverage of opaque white clouds**, c = cover fraction × cloud
//! albedo, and composited over the ground as ρ = (1 − c) · ρ_ground + c · s_c, with
//! ρ_ground = the ground entry's `albedo_scale` × its decoded texel and s_c the cloud entry's
//! `albedo_scale` (1 for the baked map, so ρ = (1 − c) · ρ_ground + c; constraints §4a, ruling
//! W-B2). This is a coverage-weighted mix, not a transmitting layer: a conservative layer of
//! albedo c over a Lambert ground would give c + (1 − c)²ρ/(1 − cρ), because light reflected by
//! the ground crosses it twice (A8 review R2-2); on the baked maps the two differ by about 4 % in
//! Earth's disk composite (0.302 here against 0.290).

use std::path::Path;

use crate::DataError;

/// The size `uni-bake map` stores a 2:1 map of decoded width `width` at: the 2:1 power of two
/// whose width is nearest `width` on a log scale, so the resample factor stays within 2^±½ and
/// every mip is block-aligned down to 4 texels high. A decoded 2:1 power of two keeps its size.
pub fn baked_size(width: u32) -> (u32, u32) {
    if width < 2 {
        return (2, 1);
    }
    let below = 1_u32 << (31 - width.leading_zeros());
    // Round up past below·√2, i.e. when width² > 2·below².
    let up = u64::from(width).pow(2) > 2 * u64::from(below).pow(2);
    let w = match below.checked_mul(2) {
        Some(above) if up => above,
        _ => below,
    };
    (w, w / 2)
}

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
    /// Multiplies decoded linear albedo to give the reflectance (review I4). For most maps the
    /// bake solves it so the map's disk-averaged reflectance gives the body's V geometric albedo;
    /// it is 1 where the solved scale would push texels above 1 (the file then stores the capped
    /// reflectance), and for these maps, which are not normalised to p_V (module docs; `source`
    /// says how each was calibrated):
    /// - Earth's ground (`Albedo`): calibrated on the ocean at the sea surface's measured
    ///   reflectance; the atmosphere (B3) supplies the rest of Earth's p_V (ruling W-B1);
    /// - Earth's clouds (`Clouds`): 1, the cloud coverage c used as is (ruling W-B2);
    /// - Venus's cloud tops (`CloudTops`): a neutral grey pattern with a disk-averaged ρ below 1
    ///   (p_V 0.689 exceeds a Lambert sphere's 2/3; ruling W-B3 keeps the baked 0.95).
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

    #[test]
    fn baked_size_is_the_nearest_power_of_two() {
        // 2:1 powers of two keep their size (every Solar System Scope map).
        for w in [8, 1024, 2048, 4096, 8192] {
            assert_eq!(baked_size(w), (w, w / 2));
        }
        // Ganymede rounds down by 1 %, Ceres up by 11 % (review r0 Important 2).
        assert_eq!(baked_size(8271), (8192, 4096));
        assert_eq!(baked_size(7383), (8192, 4096));
        // The switch is at 4096·√2 = 5792.6.
        assert_eq!(baked_size(5792), (4096, 2048));
        assert_eq!(baked_size(5793), (8192, 4096));
        assert_eq!(baked_size(200), (256, 128));
        assert_eq!(baked_size(u32::MAX), (1 << 31, 1 << 30));
    }

    /// IEC 61966-2-1 (sRGB) decoding of an 8-bit code.
    fn srgb_to_linear(code: u8) -> f64 {
        let c = f64::from(code) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Linear BT.709 luminance of an RGBA8 sRGB texel.
    fn luminance(t: &[u8]) -> f64 {
        0.2126 * srgb_to_linear(t[0])
            + 0.7152 * srgb_to_linear(t[1])
            + 0.0722 * srgb_to_linear(t[2])
    }

    /// The committed index: every entry names its files, no reflectance exceeds 1, and every map
    /// baked on this machine (the `.big` files are git-ignored) loads at the size the bake stores
    /// for the recorded decoded size, with the 2k fallback and the formats the renderer expects.
    /// Earth's and Venus's calibrations (A8 fix round 2) are checked on the files as baked.
    #[test]
    fn baked_index_maps_load() {
        let dir = crate::data_dir().join("baked/maps");
        let maps = load_map_index(&dir.join("index.ron")).expect("committed maps/index.ron");
        assert!(!maps.is_empty());
        let mut baked = 0;
        for map in &maps {
            assert!((-360.0..=360.0).contains(&map.lon0_deg), "{map:?}");
            // A cloud layer's reflectance at full cover is its scale: at most 1 (review r0
            // Important 1: it was 2.33), and 1 for the baked map, whose c is the cloud albedo
            // (fix round 2, review r1 N2).
            if map.kind == MapKind::Clouds {
                assert_eq!(map.albedo_scale, 1.0, "{map:?}");
            }
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
                baked_size(map.width),
                "{}: decoded {}×{}",
                map.file,
                map.width,
                map.height
            );
            assert!(map.width.abs_diff(2 * map.height) <= 2, "{}", map.file);
            assert_eq!(fallback.width, bc.width.min(2048), "{}", map.fallback_file);
            // The fallback's texels are averages of mip 0's, so none may exceed 1 once scaled.
            if fallback.format == TexFormat::Rgba8UnormSrgb {
                let brightest = fallback.mips[0]
                    .chunks(4)
                    .flat_map(|t| t[..3].iter().copied())
                    .max()
                    .unwrap_or(0);
                let rho = f64::from(map.albedo_scale) * srgb_to_linear(brightest);
                assert!(
                    rho <= 1.0 + 1e-6,
                    "{}: brightest ρ {rho}",
                    map.fallback_file
                );
                let scale = f64::from(map.albedo_scale);
                let texels: Vec<f64> = fallback.mips[0].chunks(4).map(luminance).collect();
                // Earth's ground (review r1 N2): its modal texel, the open ocean, is at the sea
                // surface's 0.06 (Payne 1972), not the 0.4 p_V calibration gave it.
                if (map.naif, map.kind) == (399, MapKind::Albedo) {
                    let mut counts = std::collections::HashMap::new();
                    for t in fallback.mips[0].chunks(4) {
                        *counts.entry([t[0], t[1], t[2]]).or_insert(0_usize) += 1;
                    }
                    let (mode, n) = counts
                        .into_iter()
                        .max_by_key(|&(k, n)| (n, std::cmp::Reverse(k)))
                        .unwrap_or_default();
                    let ocean = luminance(&mode) * scale;
                    assert!(
                        n * 2 > texels.len() && (ocean - 0.06).abs() < 2e-3,
                        "{}: modal texel {mode:?} ({n} of {}) ρ {ocean}",
                        map.fallback_file,
                        texels.len()
                    );
                }
                // Venus's cloud tops (review r1 N1): neutral and patterned, not uniform white.
                if (map.naif, map.kind) == (299, MapKind::CloudTops) {
                    assert!(
                        fallback.mips[0]
                            .chunks(4)
                            .all(|t| t[0] == t[1] && t[1] == t[2]),
                        "{}: not grey",
                        map.fallback_file
                    );
                    let (lo, hi) = texels
                        .iter()
                        .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &v| {
                            (lo.min(v), hi.max(v))
                        });
                    let mean = texels.iter().sum::<f64>() / texels.len() as f64;
                    assert!(
                        hi - lo > 0.05 && (0.85..1.0).contains(&(mean * scale)),
                        "{}: ρ {lo}..{hi}, texel mean {mean}",
                        map.fallback_file
                    );
                }
            }
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
