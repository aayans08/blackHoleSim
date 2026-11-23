//! Surface materials of the bodies pass (Task B1).
//!
//! Per body (keyed by NAIF id, never sized per `ObjectId`: `World::objects()` grows at run time):
//! - the **albedo map**, or Venus's **cloud-top map** (`MapKind::CloudTops`): the BC mip chain
//!   (`file`) when `GpuContext::has_bc_compression()`, else the 2k uncompressed copy
//!   (`fallback_file`). There is no CPU BC decoder (round 2): the BC texels go to the GPU as they
//!   are. Reflectance = `albedo_scale` × the decoded linear texel (sRGB decoded by the texture
//!   format), ≤ 1 in the files themselves (A8), so no clamp here;
//! - Earth's **cloud map** (`MapKind::Clouds`, BC4 or R8), composited over the ground as
//!   ρ = (1 − c)·ρ_ground + c·s_c: the areal coverage c of opaque white clouds, not a transmitting
//!   layer (constraints §4a, ruling W-B2; `uni_data::maps` module docs), lit with the ground's
//!   BRDF. A static map: the card says `Clouds: static map (no weather)` (C4);
//! - for a body without a usable map, a **uniform albedo** ρ = `rho_from_geometric_albedo(brdf,
//!   p_V)` (review I4), the point path's own ρ, with the HUD warning
//!   "No surface map: uniform albedo (<name>)".
//!
//! The CPU also reads each map's 2k copy once at start-up (uncompressed, so no decoding beyond
//! sRGB) for two statistics: the mean reflectance colour that tints the body's point sprite
//! (`BodyRenderer::mean_albedo_rgb`, carry-over 6) and the disk-averaged reflectance ρ̄ that the
//! point → disk cross-fade of ruling W-B1 needs.

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI};
use std::path::Path;

use anyhow::{Context as _, anyhow};
use glam::{DVec3, Vec3};
use uni_data::{BodyFacts, MapInfo, MapKind, TexFile, TexFormat, load_tex};
use uni_gfx::GpuContext;
use uni_physics::photometry::{Brdf, rho_from_geometric_albedo};

/// The CPU statistics read the first mip at most this wide (256 × 128 texels: 0.7° per texel,
/// far finer than the latitude weighting varies).
const STATS_MAX_WIDTH: u32 = 256;

/// ITU-R BT.709-6 luminance weights of linear R, G, B (item 3.2), as A8's bake uses for ρ.
const BT709: DVec3 = DVec3::new(0.2126, 0.7152, 0.0722);

/// The HUD warning for a body drawn without a surface map (exact text, B1 context and
/// constraints "Honesty labels in M1": `No surface map: uniform albedo`).
pub(crate) fn no_map_warning(name: &str) -> String {
    format!("No surface map: uniform albedo ({name})")
}

/// Normal reflectance ρ of a body drawn without a map: `rho_from_geometric_albedo(brdf, p_V)`
/// (review I4), the same ρ as the point path (uni-sim `reflected_lux`), so the point ↔ disk
/// handover does not pop. A NaN (negative p_V) gives 0.
pub(crate) fn uniform_rho(brdf: Brdf, p_v: f64) -> f32 {
    rho_from_geometric_albedo(brdf, p_v).max(0.0) as f32
}

/// The albedo (or cloud-top) map and the cloud map of one body, from `maps/index.ron`.
#[derive(Clone, Debug, Default)]
pub(crate) struct MapPair {
    /// `MapKind::Albedo`, or `MapKind::CloudTops` for Venus (the visible deck is the surface).
    pub albedo: Option<MapInfo>,
    /// `MapKind::Clouds` (Earth).
    pub clouds: Option<MapInfo>,
}

/// The index's maps grouped by body. The first entry of each kind wins.
pub(crate) fn maps_by_body(maps: &[MapInfo]) -> HashMap<i32, MapPair> {
    let mut pairs: HashMap<i32, MapPair> = HashMap::new();
    for map in maps {
        let pair = pairs.entry(map.naif).or_default();
        let slot = match map.kind {
            MapKind::Albedo | MapKind::CloudTops => &mut pair.albedo,
            MapKind::Clouds => &mut pair.clouds,
        };
        if slot.is_none() {
            *slot = Some(map.clone());
        }
    }
    pairs
}

/// Linear value of an 8-bit sRGB code: c/12.92 up to 0.04045, else ((c + 0.055)/1.055)^2.4
/// (IEC 61966-2-1:1999, §5.2), what an `…UnormSrgb` texture decodes to on the GPU.
pub(crate) fn srgb_decode(code: u8) -> f32 {
    let c = f64::from(code) / 255.0;
    let linear = if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    };
    linear as f32
}

/// One latitude row of a map's CPU statistics.
#[derive(Clone, Debug)]
struct StatsRow {
    /// Latitudes of the row's top and bottom edges [rad].
    top: f64,
    bottom: f64,
    /// Mean reflectance over the row (linear RGB, `albedo_scale` applied, clouds composited).
    rgb: DVec3,
}

/// Per-row mean reflectance of a body's map (from its 2k copy), for the disk average and the
/// mean colour.
#[derive(Clone, Debug)]
pub(crate) struct MapStats {
    rows: Vec<StatsRow>,
}

/// The first mip of `tex` at most [`STATS_MAX_WIDTH`] wide (the last one if none is).
fn stats_level(tex: &TexFile) -> usize {
    (0..tex.mips.len())
        .find(|&l| tex.mip_size(l).0 <= STATS_MAX_WIDTH)
        .unwrap_or(tex.mips.len().saturating_sub(1))
}

impl MapStats {
    /// Row means of `albedo` (an `Rgba8UnormSrgb` 2k copy) × `albedo_scale`, with Earth's
    /// `clouds` (an `R8Unorm` coverage map and its scale s_c) composited per texel as
    /// (1 − c)·ρ_ground + c·s_c (ruling W-B2). `None` for a BC texture (no CPU decoder) or an
    /// empty one; a cloud map in another format is ignored.
    pub(crate) fn from_maps(
        albedo: &TexFile,
        albedo_scale: f32,
        clouds: Option<(&TexFile, f32)>,
    ) -> Option<Self> {
        if albedo.format != TexFormat::Rgba8UnormSrgb || albedo.mips.is_empty() {
            return None;
        }
        let level = stats_level(albedo);
        let (w, h) = albedo.mip_size(level);
        let texels = &albedo.mips[level];
        let clouds = clouds.filter(|(c, _)| c.format == TexFormat::R8Unorm && !c.mips.is_empty());
        let cloud_mip = clouds.map(|(c, scale)| {
            let level = (0..c.mips.len())
                .find(|&l| c.mip_size(l).0 == w)
                .unwrap_or_else(|| stats_level(c));
            (c.mip_size(level), &c.mips[level], f64::from(scale))
        });
        let decode: [f64; 256] = std::array::from_fn(|code| f64::from(srgb_decode(code as u8)));
        let scale = f64::from(albedo_scale);
        let dlat = PI / f64::from(h);
        let rows = (0..h as usize)
            .map(|y| {
                let mut sum = DVec3::ZERO;
                for x in 0..w as usize {
                    let t = &texels[4 * (y * w as usize + x)..][..3];
                    let ground = DVec3::new(
                        decode[t[0] as usize],
                        decode[t[1] as usize],
                        decode[t[2] as usize],
                    ) * scale;
                    sum += match cloud_mip {
                        Some(((cw, ch), cloud, s_c)) => {
                            // Nearest cloud texel (the two 2k copies share their size in practice).
                            let cx = x * cw as usize / w as usize;
                            let cy = y * ch as usize / h as usize;
                            let c = f64::from(cloud[cy * cw as usize + cx]) / 255.0;
                            ground * (1.0 - c) + DVec3::splat(c * s_c)
                        }
                        None => ground,
                    };
                }
                let top = FRAC_PI_2 - y as f64 * dlat;
                StatsRow {
                    top,
                    bottom: top - dlat,
                    rgb: sum / f64::from(w),
                }
            })
            .collect();
        Some(Self { rows })
    }

    /// Disk-averaged reflectance colour at phase 0, averaged over a full rotation seen from the
    /// equatorial plane: rows weighted by 2L cos²φ + (1 − L)(π/2) cos³φ integrated over the row,
    /// the weighting of A8's bake (`tools/bake/src/map.rs`, `DiskWeights`): at phase 0 the
    /// lunar-Lambert radiance factor is ρ[L + (1 − L)μ] (McEwen 1991, Icarus 92, 298), each
    /// element weighs that × μ × its area, and the rotation average of μ^k over longitude gives
    /// cos^k φ (k = 1, 2) times a constant.
    pub(crate) fn disk_mean_rgb(&self, brdf: Brdf) -> DVec3 {
        let l = match brdf {
            Brdf::Lambert => 0.0,
            Brdf::LunarLambert { l } => l,
        };
        // ∫cos²φ dφ = φ/2 + sin 2φ/4; ∫cos³φ dφ = sin φ − sin³φ/3.
        let int_cos2 = |p: f64| p / 2.0 + (2.0 * p).sin() / 4.0;
        let int_cos3 = |p: f64| p.sin() - p.sin().powi(3) / 3.0;
        let (mut sum, mut total) = (DVec3::ZERO, 0.0);
        for row in &self.rows {
            let w = 2.0 * l * (int_cos2(row.top) - int_cos2(row.bottom))
                + (1.0 - l) * FRAC_PI_2 * (int_cos3(row.top) - int_cos3(row.bottom));
            sum += row.rgb * w;
            total += w;
        }
        if total > 0.0 {
            sum / total
        } else {
            DVec3::ZERO
        }
    }

    /// The disk-averaged reflectance ρ̄ (BT.709 luminance of [`Self::disk_mean_rgb`]).
    pub(crate) fn disk_rho(&self, brdf: Brdf) -> f64 {
        self.disk_mean_rgb(brdf).dot(BT709)
    }

    /// The map's mean reflectance colour (the Lambert disk average), normalised to luminance 1;
    /// `None` for a black map.
    pub(crate) fn mean_rgb(&self) -> Option<Vec3> {
        let rgb = self.disk_mean_rgb(Brdf::Lambert);
        let y = rgb.dot(BT709);
        (y > 0.0 && y.is_finite()).then(|| (rgb / y).as_vec3())
    }
}

/// What a loader reports for one body whose maps reached the GPU.
pub(crate) struct Loaded {
    /// Index of the body's material bind group (≥ 1; 0 is the uniform default).
    pub slot: usize,
    /// Statistics from the 2k copy, if it could be read.
    pub stats: Option<MapStats>,
    /// Whether the cloud map is on the GPU.
    pub clouds: bool,
    /// Anything that did not load (a missing BC file replaced by the 2k copy, a missing cloud
    /// map), for the HUD.
    pub note: Option<String>,
}

/// What the shader needs of one body's surface this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SurfaceParams {
    /// Material bind group (0: the uniform default, white 1 × 1 albedo, no clouds).
    pub slot: usize,
    /// Reflectance multiplier: the map's `albedo_scale`, or the uniform ρ without a map.
    pub albedo: f32,
    pub has_map: bool,
    /// Planetocentric east longitude of the albedo map's left edge [deg].
    pub lon0_deg: f32,
    /// Earth's cloud layer: (s_c, lon0 [deg]).
    pub clouds: Option<(f32, f32)>,
    /// Disk-averaged reflectance ρ̄ of what is drawn (the map's, or the uniform ρ); `None` when
    /// a map's 2k copy could not be read.
    pub disk_rho: Option<f64>,
    /// The map is Venus's cloud tops (super-rotating, `venus_cloud_longitude_offset_deg`).
    pub cloud_tops: bool,
}

/// One body whose maps are on the GPU.
struct MappedEntry {
    slot: usize,
    albedo_scale: f32,
    lon0_deg: f32,
    clouds: Option<(f32, f32)>,
    cloud_tops: bool,
    stats: Option<MapStats>,
    note: Option<String>,
}

/// Every body's material, keyed by NAIF id.
pub(crate) struct MaterialTable {
    mapped: HashMap<i32, MappedEntry>,
    /// Listed in the index, but the map could not be loaded: why.
    failed: HashMap<i32, String>,
    /// Display names: the fact sheets', then the map index's.
    names: HashMap<i32, String>,
}

impl MaterialTable {
    /// Groups `maps` by body and calls `load` once per body with an albedo (or cloud-top) map, in
    /// NAIF order. A body whose `load` fails draws uniform albedo, and its warning says why. A
    /// cloud map without a ground map is not used.
    pub(crate) fn build(
        maps: &[MapInfo],
        facts: &[BodyFacts],
        mut load: impl FnMut(i32, &MapPair) -> anyhow::Result<Loaded>,
    ) -> Self {
        let pairs = maps_by_body(maps);
        let mut names: HashMap<i32, String> =
            facts.iter().map(|f| (f.naif, f.name.clone())).collect();
        for map in maps {
            names.entry(map.naif).or_insert_with(|| map.body.clone());
        }
        let mut naifs: Vec<i32> = pairs.keys().copied().collect();
        naifs.sort_unstable();
        let (mut mapped, mut failed) = (HashMap::new(), HashMap::new());
        for naif in naifs {
            let pair = &pairs[&naif];
            let Some(albedo) = &pair.albedo else {
                continue;
            };
            match load(naif, pair) {
                Ok(loaded) => {
                    let clouds = pair
                        .clouds
                        .as_ref()
                        .filter(|_| loaded.clouds)
                        .map(|c| (c.albedo_scale, c.lon0_deg as f32));
                    mapped.insert(
                        naif,
                        MappedEntry {
                            slot: loaded.slot,
                            albedo_scale: albedo.albedo_scale,
                            lon0_deg: albedo.lon0_deg as f32,
                            clouds,
                            cloud_tops: albedo.kind == MapKind::CloudTops,
                            stats: loaded.stats,
                            note: loaded.note,
                        },
                    );
                }
                Err(err) => {
                    failed.insert(naif, format!("{err:#}"));
                }
            }
        }
        Self {
            mapped,
            failed,
            names,
        }
    }

    /// The surface of body `naif` drawn with `brdf`, and the HUD warnings it raises. Without a
    /// usable map: uniform ρ from `p_v`, the view's p_V (uni-sim fills it from the same
    /// `body_facts.ron`, or from its exported assumed defaults, so the point and the disk use one
    /// value; B0 ruling A.1), and "No surface map: uniform albedo (<name>)", with the reason when
    /// a listed map failed to load.
    pub(crate) fn surface_for(
        &self,
        naif: Option<i32>,
        brdf: Brdf,
        p_v: f64,
        name: &str,
    ) -> (SurfaceParams, Vec<String>) {
        if let Some(entry) = naif.and_then(|n| self.mapped.get(&n)) {
            let warnings = entry
                .note
                .iter()
                .map(|note| format!("Surface map incomplete ({name}; {note})"))
                .collect();
            let surface = SurfaceParams {
                slot: entry.slot,
                albedo: entry.albedo_scale,
                has_map: true,
                lon0_deg: entry.lon0_deg,
                clouds: entry.clouds,
                disk_rho: entry.stats.as_ref().map(|s| s.disk_rho(brdf)),
                cloud_tops: entry.cloud_tops,
            };
            return (surface, warnings);
        }
        let rho = uniform_rho(brdf, p_v);
        let warning = match naif.and_then(|n| self.failed.get(&n)) {
            Some(why) => format!("No surface map: uniform albedo ({name}; {why})"),
            None => no_map_warning(name),
        };
        let surface = SurfaceParams {
            slot: 0,
            albedo: rho,
            has_map: false,
            lon0_deg: 0.0,
            clouds: None,
            disk_rho: Some(f64::from(rho)),
            cloud_tops: false,
        };
        (surface, vec![warning])
    }

    /// The display name of body `naif` from the fact sheets or the map index.
    pub(crate) fn name(&self, naif: i32) -> Option<&str> {
        self.names.get(&naif).map(String::as_str)
    }

    /// Mean reflectance colour of body `naif`'s map, luminance 1 (`BodyRenderer::mean_albedo_rgb`).
    pub(crate) fn mean_albedo_rgb(&self, naif: i32) -> Option<Vec3> {
        self.mapped.get(&naif)?.stats.as_ref()?.mean_rgb()
    }
}

/// The wgpu format of a baked texture's texels.
pub(crate) fn wgpu_format(format: TexFormat) -> wgpu::TextureFormat {
    match format {
        TexFormat::Bc7RgbaUnormSrgb => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        TexFormat::Bc1RgbaUnormSrgb => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
        TexFormat::Bc4RUnorm => wgpu::TextureFormat::Bc4RUnorm,
        TexFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TexFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
    }
}

/// Uploads `tex` and its mip chain as it is stored (BC blocks go to the GPU undecoded). Mips
/// larger than the device's `max_texture_dimension_2d` are skipped, so an 8k map still loads on
/// a 4k device. A BC texture needs `TEXTURE_COMPRESSION_BC`. Validation errors come back as an
/// `Err`.
pub(crate) fn upload_tex(
    gpu: &GpuContext,
    label: &str,
    tex: &TexFile,
) -> anyhow::Result<wgpu::Texture> {
    let block = tex.format.block_dim();
    anyhow::ensure!(
        block == 1 || gpu.has_bc_compression(),
        "{label}: {:?} needs TEXTURE_COMPRESSION_BC",
        tex.format
    );
    let max = gpu.device.limits().max_texture_dimension_2d;
    let first = (0..tex.mips.len())
        .find(|&l| {
            let (w, h) = tex.mip_size(l);
            w <= max && h <= max
        })
        .with_context(|| format!("{label}: no mip fits the device's {max}-texel limit"))?;
    let (width, height) = tex.mip_size(first);
    anyhow::ensure!(
        width % block == 0 && height % block == 0,
        "{label}: mip {first} is {width}×{height}, not whole {block}×{block} blocks"
    );
    let device = &gpu.device;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: (tex.mips.len() - first) as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu_format(tex.format),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (mip, level) in (first..tex.mips.len()).enumerate() {
        let (w, h) = tex.mip_size(level);
        let row_bytes = u32::try_from(tex.format.row_bytes(w)).context("row too long")?;
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &tex.mips[level],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(h.div_ceil(block)),
            },
            // The physical size: whole blocks, also for a mip smaller than one block.
            wgpu::Extent3d {
                width: w.div_ceil(block) * block,
                height: h.div_ceil(block) * block,
                depth_or_array_layers: 1,
            },
        );
    }
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("{label}: uploading the texture: {err}");
    }
    Ok(texture)
}

/// A 1 × 1 texture holding `texel` (the uniform material's stand-in maps).
pub(crate) fn solid_texture(
    gpu: &GpuContext,
    label: &str,
    format: TexFormat,
    texel: &[u8],
) -> anyhow::Result<wgpu::Texture> {
    upload_tex(
        gpu,
        label,
        &TexFile {
            format,
            width: 1,
            height: 1,
            mips: vec![texel.to_vec()],
        },
    )
}

/// One body's maps on the GPU.
pub(crate) struct GpuMaps {
    pub albedo: wgpu::Texture,
    pub clouds: Option<wgpu::Texture>,
}

/// What [`load_pair`] read.
pub(crate) struct LoadedPair {
    pub maps: GpuMaps,
    pub stats: Option<MapStats>,
    pub note: Option<String>,
}

/// Loads one body's maps from `map_dir` onto the GPU: the BC files when `bc`, else (or when a BC
/// file cannot be read) the 2k uncompressed copies; and the CPU statistics from the 2k copies. An
/// albedo map that cannot be loaded at all is an `Err` (the body draws uniform albedo); a cloud
/// map that cannot is a note.
pub(crate) fn load_pair(
    gpu: &GpuContext,
    map_dir: &Path,
    pair: &MapPair,
    bc: bool,
) -> anyhow::Result<LoadedPair> {
    let albedo = pair.albedo.as_ref().context("no albedo map")?;
    let read = |file: &str| -> anyhow::Result<TexFile> {
        load_tex(&map_dir.join(file)).map_err(|e| anyhow!("{file}: {e}"))
    };
    let mut notes = Vec::new();
    // The GPU copy of one map, and the 2k copy read on the CPU (kept for the statistics).
    let mut to_gpu = |info: &MapInfo| -> anyhow::Result<(wgpu::Texture, Option<TexFile>)> {
        let small = read(&info.fallback_file);
        if bc {
            match read(&info.file).and_then(|t| upload_tex(gpu, &info.file, &t)) {
                Ok(texture) => return Ok((texture, small.ok())),
                Err(err) => notes.push(format!("{err:#}; using the 2k copy")),
            }
        }
        let small = small?;
        Ok((upload_tex(gpu, &info.fallback_file, &small)?, Some(small)))
    };
    let (albedo_gpu, albedo_2k) = to_gpu(albedo)?;
    let (clouds_gpu, clouds_2k) = match pair.clouds.as_ref().map(|c| (c, to_gpu(c))) {
        Some((info, Ok((texture, small)))) => {
            (Some(texture), small.map(|s| (s, info.albedo_scale)))
        }
        Some((_, Err(err))) => {
            notes.push(format!("{err:#}; no cloud layer"));
            (None, None)
        }
        None => (None, None),
    };
    let stats = albedo_2k.as_ref().and_then(|a| {
        MapStats::from_maps(
            a,
            albedo.albedo_scale,
            clouds_2k.as_ref().map(|(c, s)| (c, *s)),
        )
    });
    Ok(LoadedPair {
        maps: GpuMaps {
            albedo: albedo_gpu,
            clouds: clouds_gpu,
        },
        stats,
        note: (!notes.is_empty()).then(|| notes.join("; ")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_data::{MapKind, TexFormat};
    use uni_physics::photometry::rho_from_geometric_albedo;

    const LUNAR_LAMBERT: Brdf = Brdf::LunarLambert { l: 0.9 };

    fn map(body: &str, naif: i32, kind: MapKind, file: &str) -> MapInfo {
        MapInfo {
            body: body.into(),
            naif,
            kind,
            file: format!("{file}.big"),
            fallback_file: format!("{file}.rgba2k.big"),
            width: 8192,
            height: 4096,
            lon0_deg: -180.0,
            albedo_scale: 0.5,
            source: "test entry".into(),
        }
    }

    /// A facts entry with only what the bodies pass reads (name, p_V); test values.
    fn facts(naif: i32, name: &str, p_v: Option<f64>) -> BodyFacts {
        BodyFacts {
            naif,
            name: name.into(),
            kind: "moon".into(),
            mean_radius_km: 788.9,
            mass_kg: 3.4e21,
            surface_gravity_ms2: None,
            escape_velocity_kms: None,
            sidereal_rotation_h: None,
            orbital_period_d: None,
            semi_major_au: None,
            axial_tilt_deg: None,
            surface_pressure_kpa: None,
            atmosphere: None,
            mean_temp_k: None,
            bond_albedo: None,
            geometric_albedo: p_v,
            moons: Vec::new(),
            source: "test entry".into(),
        }
    }

    fn index() -> Vec<MapInfo> {
        vec![
            map("Earth", 399, MapKind::Albedo, "earth_albedo"),
            map("Earth", 399, MapKind::Clouds, "earth_clouds"),
            map("Venus", 299, MapKind::CloudTops, "venus_cloud_tops"),
            map("Moon", 301, MapKind::Albedo, "moon_albedo"),
        ]
    }

    /// A loader that "loads" every pair, numbering the slots from 1 (slot 0 is the default).
    fn ok_loader() -> impl FnMut(i32, &MapPair) -> anyhow::Result<Loaded> {
        let mut next = 0;
        move |_, pair| {
            next += 1;
            Ok(Loaded {
                slot: next,
                stats: None,
                clouds: pair.clouds.is_some(),
                note: None,
            })
        }
    }

    /// An RGBA8 sRGB 2:1 texture of one colour, with a full chain down to 1 texel high.
    fn solid_rgba(w: u32, code: [u8; 4]) -> TexFile {
        let mut mips = Vec::new();
        let (mut mw, mut mh) = (w, w / 2);
        loop {
            mips.push(code.repeat((mw * mh) as usize));
            if mh == 1 {
                break;
            }
            (mw, mh) = (mw / 2, mh / 2);
        }
        TexFile {
            format: TexFormat::Rgba8UnormSrgb,
            width: w,
            height: w / 2,
            mips,
        }
    }

    /// One-channel linear clouds of one value.
    fn solid_r8(w: u32, code: u8) -> TexFile {
        let t = solid_rgba(w, [code; 4]);
        TexFile {
            format: TexFormat::R8Unorm,
            mips: t
                .mips
                .iter()
                .map(|m| m.iter().step_by(4).copied().collect())
                .collect(),
            ..t
        }
    }

    /// Review Focus 3: a body the map index does not list (Titania) draws uniform albedo with ρ
    /// from its p_V, and the HUD gets the exact warning; a listed map whose file failed to load
    /// falls back the same way and says why.
    #[test]
    fn missing_map_falls_back_to_uniform_albedo() {
        let facts = [facts(703, "Titania", Some(0.35))];
        let table = MaterialTable::build(&index(), &facts, ok_loader());
        let (surface, warnings) = table.surface_for(Some(703), LUNAR_LAMBERT, 0.35, "Titania");
        assert_eq!(warnings, ["No surface map: uniform albedo (Titania)"]);
        assert!(!surface.has_map);
        assert_eq!(surface.slot, 0);
        let rho = rho_from_geometric_albedo(LUNAR_LAMBERT, 0.35);
        assert!(
            (f64::from(surface.albedo) - rho).abs() < 1e-7,
            "{surface:?}"
        );
        assert_eq!(surface.disk_rho, Some(f64::from(surface.albedo)));
        assert_eq!(surface.clouds, None);
        // A body without a NAIF id (an asteroid) falls back the same way, with its view's p_V.
        let (s, w) = table.surface_for(None, LUNAR_LAMBERT, 0.14, "(433) Eros");
        assert_eq!(w, ["No surface map: uniform albedo ((433) Eros)"]);
        assert!(
            (f64::from(s.albedo) - rho_from_geometric_albedo(LUNAR_LAMBERT, 0.14)).abs() < 1e-7
        );
        // Listed and loaded: mapped, no warning; Earth carries its cloud layer, Venus its
        // cloud tops.
        let (earth, w) = table.surface_for(Some(399), Brdf::Lambert, 0.434, "Earth");
        assert!(
            earth.has_map && earth.slot > 0 && w.is_empty(),
            "{earth:?} {w:?}"
        );
        assert_eq!(earth.albedo, 0.5);
        assert_eq!(earth.clouds, Some((0.5, -180.0)));
        let (venus, _) = table.surface_for(Some(299), Brdf::Lambert, 0.689, "Venus");
        assert!(venus.has_map && venus.cloud_tops && venus.clouds.is_none());
        // Listed but unreadable: uniform, and the warning says why.
        let failing = MaterialTable::build(&index(), &facts, |naif, _| {
            anyhow::ensure!(naif != 301, "moon_albedo.big: not found");
            Ok(Loaded {
                slot: 1,
                stats: None,
                clouds: false,
                note: None,
            })
        });
        let (moon, w) = failing.surface_for(Some(301), LUNAR_LAMBERT, 0.12, "Moon");
        assert!(!moon.has_map);
        assert_eq!(
            w,
            ["No surface map: uniform albedo (Moon; moon_albedo.big: not found)"]
        );
        // Names: the facts' first, then the index's.
        assert_eq!(table.name(703), Some("Titania"));
        assert_eq!(table.name(301), Some("Moon"));
        assert_eq!(table.name(42), None);
    }

    #[test]
    fn maps_pair_by_body() {
        let pairs = maps_by_body(&index());
        assert_eq!(pairs.len(), 3);
        let earth = &pairs[&399];
        assert_eq!(earth.albedo.as_ref().unwrap().kind, MapKind::Albedo);
        assert_eq!(earth.clouds.as_ref().unwrap().kind, MapKind::Clouds);
        assert_eq!(
            pairs[&299].albedo.as_ref().unwrap().kind,
            MapKind::CloudTops
        );
        assert!(pairs[&301].clouds.is_none());
    }

    /// The CPU disk averages of the real 2k copies reproduce the ρ̄ that A8's bake reports in
    /// `maps/index.ron` (`source`, computed there on the full-resolution maps): Earth's
    /// ground + cloud composite 0.3019, Mars 0.2550, Jupiter 0.8072, Venus's cloud tops 0.95,
    /// Mercury 0.1469 (lunar-Lambert), all rotation-averaged. The W-B1 cross-fade divides by
    /// Earth's. Skipped (with a message) where the git-ignored `.big` files are absent.
    #[test]
    fn disk_averages_match_the_bake() {
        let dir = uni_data::data_dir().join("baked/maps");
        let Ok(index) = uni_data::load_map_index(&dir.join("index.ron")) else {
            eprintln!("skipped: no maps/index.ron");
            return;
        };
        let pairs = maps_by_body(&index);
        let cases = [
            (399, Brdf::Lambert, 0.3019),
            (499, Brdf::Lambert, 0.2550),
            (599, Brdf::Lambert, 0.8072),
            (299, Brdf::Lambert, 0.95),
            (199, LUNAR_LAMBERT, 0.1469),
        ];
        for (naif, brdf, want) in cases {
            let pair = &pairs[&naif];
            let albedo = pair.albedo.as_ref().unwrap();
            let Ok(ground) = uni_data::load_tex(&dir.join(&albedo.fallback_file)) else {
                eprintln!("skipped {naif}: {} absent", albedo.fallback_file);
                continue;
            };
            let clouds = pair.clouds.as_ref().map(|c| {
                (
                    uni_data::load_tex(&dir.join(&c.fallback_file)).unwrap(),
                    c.albedo_scale,
                )
            });
            let stats = MapStats::from_maps(
                &ground,
                albedo.albedo_scale,
                clouds.as_ref().map(|(t, s)| (t, *s)),
            )
            .unwrap();
            let got = stats.disk_rho(brdf);
            eprintln!("{} ({naif}): ρ̄ {got:.4} (bake {want})", albedo.body);
            assert!((got / want - 1.0).abs() < 0.02, "{naif}: {got} vs {want}");
            if naif == 499 {
                let c = stats.mean_rgb().unwrap();
                assert!(c.x > c.y && c.y > c.z, "Mars is red: {c}");
            }
        }
    }

    /// sRGB decoding (IEC 61966-2-1): the end points, the linear toe and mid grey.
    #[test]
    fn srgb_decode_matches_the_standard() {
        assert_eq!(srgb_decode(0), 0.0);
        assert!((srgb_decode(255) - 1.0).abs() < 1e-7);
        assert!((srgb_decode(10) - 10.0 / 255.0 / 12.92).abs() < 1e-7);
        assert!((srgb_decode(128) - 0.215_861).abs() < 1e-5);
    }

    /// The CPU statistics: a uniform map gives its own colour (normalised to luminance 1) and
    /// its scaled luminance as the disk average for either BRDF; Earth's composite
    /// ρ = (1 − c)·ρ_ground + c·s_c (ruling W-B2) is averaged per texel.
    #[test]
    fn map_stats_uniform_colour_and_cloud_composite() {
        let ground = solid_rgba(512, [200, 100, 50, 255]);
        let lin = Vec3::new(srgb_decode(200), srgb_decode(100), srgb_decode(50));
        let lum = 0.2126 * lin.x + 0.7152 * lin.y + 0.0722 * lin.z;
        let stats = MapStats::from_maps(&ground, 0.5, None).unwrap();
        for brdf in [Brdf::Lambert, LUNAR_LAMBERT] {
            assert!((stats.disk_rho(brdf) - 0.5 * f64::from(lum)).abs() < 1e-6);
        }
        let colour = stats.mean_rgb().unwrap();
        assert!((colour - lin / lum).abs().max_element() < 1e-5, "{colour}");
        // 40 % cloud cover of albedo-1 clouds over the same ground.
        let clouds = solid_r8(512, 102);
        let c = 102.0 / 255.0;
        let stats = MapStats::from_maps(&ground, 0.5, Some((&clouds, 1.0))).unwrap();
        let want = (1.0 - c) * 0.5 * f64::from(lum) + c;
        assert!((stats.disk_rho(Brdf::Lambert) - want).abs() < 1e-6);
        // A BC texture has no CPU decoder: no statistics.
        let bc = TexFile {
            format: TexFormat::Bc7RgbaUnormSrgb,
            ..ground.clone()
        };
        assert!(MapStats::from_maps(&bc, 1.0, None).is_none());
    }
}
