//! `uni-bake map` (M1 Task A8): surface maps (JPEG/PNG/TIFF) → `UNITEX1` BC mip chains plus 2k
//! uncompressed fallbacks, and the index `maps/index.ron`.
//!
//! Command line (one group per map; each `--input` starts a new group, and the groups are baked
//! in parallel with rayon):
//! ```text
//! uni-bake map [--index <index.ron>]
//!   --input <jpg|png|tif> --body Earth --naif 399 --kind albedo|cloud-tops --lon0 -180
//!   --geometric-albedo 0.434 [--brdf lambert|lunar-lambert] [--view-lon <deg>]
//!   [--encoding srgb|linear] [--fill-nodata <max DN>] [--clouds <img> --clouds-out <big>]
//!   --source "<credit>" --out data/baked/maps/earth_albedo.big [--preview <png>]
//!   [--input … next map]
//! ```
//! - `--out x.big` writes the BC chain there and the fallback beside it as `x.rgba2k.big`
//!   (`--clouds-out y.big`: `y.r2k.big`). `--index` defaults to `index.ron` in the first map's
//!   directory; entries with the same body and kind are replaced, others kept.
//! - `--encoding` is the source's transfer function: `srgb` for display images (Solar System
//!   Scope), `linear` for USGS mosaics, whose 8-bit DN are a linear stretch of I/F (their ISIS
//!   labels' `Base`/`Multiplier`). Albedo is stored sRGB-encoded (BC7 / RGBA8 sRGB); Earth's
//!   cloud coverage is stored linear (BC4 / R8).
//! - `--fill-nodata T` replaces the no-data texels (dark regions, DN ≤ T, that touch a pole or
//!   cover ≥ 0.1 % of the map: unimaged polar caps and gaps) with the mean albedo of the imaged
//!   area, i.e. uniform albedo where nothing was imaged.
//! - `--view-lon` fixes the sub-observer longitude of the disk average (the Moon: 0°, the hemisphere
//!   its geometric albedo is measured on); without it the disk average is taken over a full
//!   rotation seen from the equatorial plane.
//! - A source whose decoded size is not a 2:1 power of two (tier-B mosaics) is area-resampled to
//!   the nearest 2:1 power of two (`uni_data::maps::baked_size`), so every mip is block-aligned;
//!   the index records the decoded size, as `MapInfo` documents. Solar System Scope maps bake at
//!   their decoded size.
//! - `albedo_scale` is solved with each channel's reflectance capped at 1 inside the disk average
//!   (`DiskHistogram::solve`). When the geometric albedo needs texels brighter than 1, the files
//!   store the capped reflectance min(1, s·a) and the index's scale is 1, so the stored texel ×
//!   `albedo_scale` never exceeds 1 (review r0 Important 1).

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use uni_data::{MapInfo, MapKind, TexFile, TexFormat};
use uni_physics::photometry::{Brdf, rho_from_geometric_albedo};

use crate::bc;

/// Lunar-Lambert limb-darkening weight for airless bodies, the value B1 renders with
/// (McEwen 1991, Icarus 92, 298).
const LUNAR_LAMBERT_L: f64 = 0.9;
/// Width of the uncompressed fallback [texels] (round 2, review I17).
const FALLBACK_WIDTH: u32 = 2048;

/// Source and storage transfer function of 8-bit texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transfer {
    Srgb,
    Linear,
}

/// sRGB decoding of every 8-bit code (IEC 61966-2-1:1999).
static SRGB_TO_LINEAR: LazyLock<[f32; 256]> = LazyLock::new(|| {
    std::array::from_fn(|code| {
        let c = code as f64 / 255.0;
        let l = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
        l as f32
    })
});

impl Transfer {
    /// Linear value [0, 1] of an 8-bit code.
    fn decode(self, code: u8) -> f32 {
        match self {
            Transfer::Srgb => SRGB_TO_LINEAR[usize::from(code)],
            Transfer::Linear => f32::from(code) / 255.0,
        }
    }

    /// Nearest 8-bit code of a linear value (clamped to [0, 1]); sRGB encoding per
    /// IEC 61966-2-1:1999.
    fn encode(self, linear: f32) -> u8 {
        let l = f64::from(linear).clamp(0.0, 1.0);
        let c = match self {
            Transfer::Linear => l,
            Transfer::Srgb if l <= 0.003_130_8 => 12.92 * l,
            Transfer::Srgb => 1.055 * l.powf(1.0 / 2.4) - 0.055,
        };
        (c * 255.0).round() as u8
    }
}

/// An 8-bit image: `channels` (1 or 3) bytes per texel, row-major, top row (north) first.
#[derive(Clone, Debug, PartialEq)]
struct Image8 {
    width: u32,
    height: u32,
    channels: usize,
    transfer: Transfer,
    data: Vec<u8>,
}

impl Image8 {
    /// Linear value of channel `c` of texel `t`.
    fn linear(&self, t: usize, c: usize) -> f32 {
        self.transfer.decode(self.data[t * self.channels + c])
    }

    /// The same texels re-encoded with `transfer`.
    fn with_transfer(self, transfer: Transfer) -> Image8 {
        if self.transfer == transfer {
            return self;
        }
        let data = self
            .data
            .par_iter()
            .map(|&code| transfer.encode(self.transfer.decode(code)))
            .collect();
        Image8 {
            transfer,
            data,
            ..self
        }
    }
}

/// The surface kinds the command line bakes directly (clouds come with `--clouds`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurfaceKind {
    Albedo,
    CloudTops,
}

/// Which hemisphere the disk average is taken over.
#[derive(Clone, Copy, Debug, PartialEq)]
enum View {
    /// Observer in the equatorial plane, averaged over a full rotation.
    RotationAveraged,
    /// Observer above planetocentric east longitude `sub_observer_lon_deg` on the equator.
    Fixed { sub_observer_lon_deg: f64 },
}

/// One map group of the command line.
#[derive(Clone, Debug, PartialEq)]
struct MapSpec {
    input: PathBuf,
    body: String,
    naif: i32,
    kind: SurfaceKind,
    lon0_deg: f64,
    geometric_albedo: f64,
    brdf: Brdf,
    view: View,
    encoding: Transfer,
    fill_nodata: Option<u8>,
    clouds: Option<(PathBuf, PathBuf)>,
    source: String,
    out: PathBuf,
    preview: Option<PathBuf>,
}

/// Runs `uni-bake map` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let (specs, index) = parse(options)?;
    let index = match index {
        Some(index) => index,
        None => dir_of(&specs[0].out).join("index.ron"),
    };
    for spec in &specs {
        let outs = std::iter::once(&spec.out).chain(spec.clouds.as_ref().map(|(_, out)| out));
        for out in outs {
            ensure!(
                dir_of(out) == dir_of(&index),
                "{} is not beside the index {} (the index names files relative to itself)",
                out.display(),
                index.display()
            );
        }
    }
    let started = Instant::now();
    let results: Vec<Result<Vec<Baked>>> = specs.par_iter().map(bake_group).collect();
    let mut baked = Vec::new();
    for (spec, result) in specs.iter().zip(results) {
        baked.extend(result.with_context(|| format!("baking {}", spec.input.display()))?);
    }
    let mut maps = match uni_data::load_map_index(&index) {
        Ok(maps) => maps,
        Err(uni_data::DataError::NotFound(_)) => Vec::new(),
        Err(err) => return Err(err).context("reading the existing index"),
    };
    for map in baked.iter().map(|b| &b.info) {
        maps.retain(|m| (m.body.as_str(), m.kind) != (map.body.as_str(), map.kind));
        maps.push(map.clone());
    }
    maps.sort_by_key(|m| (m.naif, kind_order(m.kind)));
    uni_data::maps::write_map_index(&index, &maps)?;
    for b in &baked {
        println!("{}", b.report);
    }
    println!(
        "map: {} maps in {:.1} s -> {} ({} entries)",
        baked.len(),
        started.elapsed().as_secs_f64(),
        index.display(),
        maps.len()
    );
    Ok(())
}

fn kind_order(kind: MapKind) -> u8 {
    match kind {
        MapKind::Albedo => 0,
        MapKind::Clouds => 1,
        MapKind::CloudTops => 2,
    }
}

/// The directory a path names a file in (`.` for a bare file name).
fn dir_of(path: &Path) -> PathBuf {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Parses the groups and the optional `--index`.
fn parse(options: &[OsString]) -> anyhow::Result<(Vec<MapSpec>, Option<PathBuf>)> {
    let mut index = None;
    let mut groups: Vec<BTreeMap<String, OsString>> = Vec::new();
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let name = flag
            .to_str()
            .with_context(|| format!("option {flag:?} is not UTF-8"))?;
        let value = options
            .next()
            .with_context(|| format!("{name} needs a value"))?
            .clone();
        if name == "--index" {
            index = Some(PathBuf::from(value));
            continue;
        }
        if name == "--input" {
            groups.push(BTreeMap::new());
        }
        let group = groups
            .last_mut()
            .with_context(|| format!("{name} comes before the first --input"))?;
        ensure!(
            group.insert(name.to_string(), value).is_none(),
            "{name} given twice for one map"
        );
    }
    ensure!(
        !groups.is_empty(),
        "no map named: start each map with --input"
    );
    let specs = groups.iter().map(spec_from).collect::<Result<Vec<_>>>()?;
    Ok((specs, index))
}

/// Builds one map's spec from its flags.
fn spec_from(group: &BTreeMap<String, OsString>) -> Result<MapSpec> {
    const KNOWN: [&str; 15] = [
        "--input",
        "--body",
        "--naif",
        "--kind",
        "--lon0",
        "--geometric-albedo",
        "--brdf",
        "--view-lon",
        "--encoding",
        "--fill-nodata",
        "--clouds",
        "--clouds-out",
        "--source",
        "--out",
        "--preview",
    ];
    if let Some(unknown) = group.keys().find(|k| !KNOWN.contains(&k.as_str())) {
        bail!("unknown option {unknown}");
    }
    let text = |name: &str| -> Result<Option<&str>> {
        group
            .get(name)
            .map(|v| {
                v.to_str()
                    .with_context(|| format!("{name} {v:?} is not UTF-8"))
            })
            .transpose()
    };
    let required =
        |name: &str| -> Result<&str> { text(name)?.with_context(|| format!("missing {name}")) };
    let number = |name: &str, value: &str| -> Result<f64> {
        let x: f64 = value
            .parse()
            .with_context(|| format!("{name} {value:?} is not a number"))?;
        ensure!(x.is_finite(), "{name} {value:?} is not finite");
        Ok(x)
    };
    let path = |name: &str| text(name).map(|v| v.map(PathBuf::from));

    let kind = match required("--kind")? {
        "albedo" => SurfaceKind::Albedo,
        "cloud-tops" => SurfaceKind::CloudTops,
        other => {
            bail!("--kind {other:?}: use albedo or cloud-tops (Earth's clouds come with --clouds)")
        }
    };
    let geometric_albedo = number("--geometric-albedo", required("--geometric-albedo")?)?;
    ensure!(
        geometric_albedo > 0.0,
        "--geometric-albedo must be positive"
    );
    let brdf = match text("--brdf")?.unwrap_or("lambert") {
        "lambert" => Brdf::Lambert,
        "lunar-lambert" => Brdf::LunarLambert { l: LUNAR_LAMBERT_L },
        other => bail!("--brdf {other:?}: use lambert or lunar-lambert"),
    };
    let view = match text("--view-lon")? {
        Some(v) => View::Fixed {
            sub_observer_lon_deg: number("--view-lon", v)?,
        },
        None => View::RotationAveraged,
    };
    let encoding = match text("--encoding")?.unwrap_or("srgb") {
        "srgb" => Transfer::Srgb,
        "linear" => Transfer::Linear,
        other => bail!("--encoding {other:?}: use srgb or linear"),
    };
    let fill_nodata = text("--fill-nodata")?
        .map(|v| {
            v.parse::<u8>()
                .with_context(|| format!("--fill-nodata {v:?} is not 0–255"))
        })
        .transpose()?;
    let clouds = match (path("--clouds")?, path("--clouds-out")?) {
        (Some(input), Some(out)) => {
            ensure!(
                kind == SurfaceKind::Albedo,
                "--clouds goes with --kind albedo"
            );
            Some((input, out))
        }
        (None, None) => None,
        _ => bail!("--clouds and --clouds-out go together"),
    };
    Ok(MapSpec {
        input: PathBuf::from(required("--input")?),
        body: required("--body")?.to_string(),
        naif: required("--naif")?
            .parse()
            .with_context(|| "--naif is not an integer".to_string())?,
        kind,
        lon0_deg: number("--lon0", required("--lon0")?)?,
        geometric_albedo,
        brdf,
        view,
        encoding,
        fill_nodata,
        clouds,
        source: required("--source")?.to_string(),
        out: PathBuf::from(required("--out")?),
        preview: path("--preview")?,
    })
}

/// One baked map: its index entry and its report line.
struct Baked {
    info: MapInfo,
    report: String,
}

/// The fallback's path: `x.big` → `x.rgba2k.big` (RGBA) or `x.r2k.big` (single channel).
fn fallback_path(out: &Path, single_channel: bool) -> PathBuf {
    let stem = out
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tag = if single_channel { "r2k" } else { "rgba2k" };
    out.with_file_name(format!("{stem}.{tag}.big"))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Decodes an image file to 8 bits: one channel for greyscale sources (or when `grey`), three
/// otherwise.
fn decode_source(path: &Path, transfer: Transfer, grey: bool) -> Result<Image8> {
    // An 8k RGB map decodes to ~100 MB, inside `image`'s default 512 MiB allocation limit.
    let decoded = image::ImageReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("decoding {}", path.display()))?;
    let (width, height) = (decoded.width(), decoded.height());
    let (channels, data) = if grey || !decoded.color().has_color() {
        (1, decoded.into_luma8().into_raw())
    } else {
        (3, decoded.into_rgb8().into_raw())
    };
    Ok(Image8 {
        width,
        height,
        channels,
        transfer,
        data,
    })
}

/// Bakes one group: the ground (or cloud-top) map and, with `--clouds`, Earth's cloud map.
fn bake_group(spec: &MapSpec) -> Result<Vec<Baked>> {
    let started = Instant::now();
    let (ground, ground_note, decoded) = prepare(
        decode_source(&spec.input, spec.encoding, false)?,
        spec.fill_nodata,
    )?;
    let ground = ground.with_transfer(Transfer::Srgb);
    let clouds = match &spec.clouds {
        Some((input, _)) => {
            let (clouds, note, cloud_decoded) =
                prepare(decode_source(input, Transfer::Linear, true)?, None)?;
            ensure!(
                cloud_decoded == decoded,
                "clouds {}×{} do not match the ground's {}×{}",
                cloud_decoded.0,
                cloud_decoded.1,
                decoded.0,
                decoded.1
            );
            Some((clouds, note))
        }
        None => None,
    };
    let cloud_map = clouds.as_ref().map(|(c, _)| c);
    let histogram = DiskHistogram::new(&ground, cloud_map, spec.brdf, spec.view, spec.lon0_deg);
    let average = histogram.average(1.0);
    let target = rho_from_geometric_albedo(spec.brdf, spec.geometric_albedo);
    let solved = histogram
        .solve(target)
        .with_context(|| format!("solving albedo_scale for {}", spec.body))?;
    let capped = solved > histogram.first_cap();
    // For the report only (the brief asks for both Earth scales): the ground map without its
    // clouds, its disk average and its own capped scale, and the mean cloud cover.
    let ground_alone = cloud_map.map(|clouds| {
        let alone = DiskHistogram::new(&ground, None, spec.brdf, spec.view, spec.lon0_deg);
        let cover = DiskHistogram::new(clouds, None, spec.brdf, spec.view, spec.lon0_deg);
        let alone_scale = match alone.solve(target) {
            Ok(s) => format!("{s:.4}"),
            Err(err) => format!("none ({err})"),
        };
        format!(
            "disk-weighted mean cloud cover {:.4}; ground alone: disk-averaged ρ {:.4} → capped \
             scale {alone_scale}",
            cover.average(1.0),
            alone.average(1.0)
        )
    });
    let (ground, scale) = apply_scale(ground, solved, histogram.first_cap());
    // Check on the texels as stored (re-quantised when capped): their disk average × scale.
    let stored =
        DiskHistogram::new(&ground, cloud_map, spec.brdf, spec.view, spec.lon0_deg).average(scale);
    let brdf = match spec.brdf {
        Brdf::Lambert => "Lambert".to_string(),
        Brdf::LunarLambert { l } => format!("lunar-Lambert L = {l}"),
    };
    let view = match spec.view {
        View::RotationAveraged => "rotation-averaged".to_string(),
        View::Fixed {
            sub_observer_lon_deg,
        } => format!("hemisphere centred on {sub_observer_lon_deg}°E"),
    };
    let cap_note = if capped {
        format!(
            "solved with each channel capped at ρ = 1: s = {solved:.4}, {:.1} % of the disk's \
             luminance weight at the cap, texels stored as min(1, s·a) (disk average as stored \
             {stored:.4})",
            100.0 * histogram.share_at_cap(solved)
        )
    } else {
        format!(
            "no texel reaches ρ = 1 (brightest {:.3})",
            solved / histogram.first_cap()
        )
    };
    let scale_note = format!(
        "albedo_scale {scale:.4}: disk-averaged ρ {average:.4} ({view}{}) → ρ {target:.4} for \
         {brdf}, p_V {}; {cap_note}",
        if clouds.is_some() {
            ", clouds composited"
        } else {
            ""
        },
        spec.geometric_albedo
    );
    let max_ground = (0..(ground.width * ground.height) as usize)
        .into_par_iter()
        .map(|t| {
            (0..ground.channels)
                .map(|c| ground.linear(t, c))
                .fold(0.0, f32::max)
        })
        .reduce(|| 0.0, f32::max);
    let kind = match spec.kind {
        SurfaceKind::Albedo => MapKind::Albedo,
        SurfaceKind::CloudTops => MapKind::CloudTops,
    };
    let mut baked = vec![write_map(
        spec,
        kind,
        ground,
        decoded,
        &spec.out,
        format!(
            "{}: {}; {ground_note}; {scale_note}",
            file_name(&spec.input),
            spec.source
        ),
        scale,
        format!(
            "max scaled texel {:.3}; stored disk average × scale {stored:.4} vs ρ {target:.4}",
            f64::from(max_ground) * scale
        ),
        spec.preview.as_deref(),
    )?];
    if let (Some((clouds, note)), Some((input, out)), Some(ground_alone)) =
        (clouds, &spec.clouds, ground_alone)
    {
        baked.push(write_map(
            spec,
            MapKind::Clouds,
            clouds,
            decoded,
            out,
            format!(
                "{}: {}; {note}; coverage c, composited (1 − c)·ρ + c; {scale_note}",
                file_name(input),
                spec.source
            ),
            scale,
            format!(
                "cloud reflectance at full cover {:.3}; {ground_alone}",
                scale.min(1.0)
            ),
            None,
        )?);
    }
    let seconds = started.elapsed().as_secs_f64();
    for b in &mut baked {
        b.report.push_str(&format!(", group {seconds:.1} s"));
    }
    Ok(baked)
}

/// Fills no-data and resamples to the stored size ([`uni_data::maps::baked_size`]); returns the
/// image, the note on what was done for the index's `source`, and the decoded size.
fn prepare(mut image: Image8, fill: Option<u8>) -> Result<(Image8, String, (u32, u32))> {
    let decoded = (image.width, image.height);
    let mut note = format!("decoded {}×{}", image.width, image.height);
    if let Some(max_dn) = fill {
        let filled = fill_nodata(&mut image, max_dn);
        let share = 100.0 * filled as f64 / f64::from(image.width * image.height);
        note.push_str(&format!(
            "; {filled} unimaged texels ({share:.1} %, DN ≤ {max_dn}) set to the imaged mean albedo"
        ));
    }
    let image = to_power_of_two(image)?;
    if (image.width, image.height) != decoded {
        note.push_str(&format!(
            "; area-resampled to {}×{}, the nearest 2:1 power of two",
            image.width, image.height
        ));
    }
    Ok((image, note, decoded))
}

/// Builds the chain, writes the BC file and the fallback, and returns the index entry.
#[allow(clippy::too_many_arguments)]
fn write_map(
    spec: &MapSpec,
    kind: MapKind,
    mip0: Image8,
    decoded: (u32, u32),
    out: &Path,
    source: String,
    scale: f64,
    extra: String,
    preview: Option<&Path>,
) -> Result<Baked> {
    let started = Instant::now();
    let single = mip0.channels == 1 && mip0.transfer == Transfer::Linear;
    let (width, height) = (mip0.width, mip0.height);
    let chain = mip_chain(mip0);
    let start = fallback_start(&chain);
    let compressed = pack(&chain, true)?;
    let fallback = pack(&chain[start..], false)?;
    let fallback_out = fallback_path(out, single);
    uni_data::write_tex(out, &compressed)?;
    uni_data::write_tex(&fallback_out, &fallback)?;
    if let Some(png) = preview {
        let level = &chain[start];
        let color = if level.channels == 3 {
            image::ExtendedColorType::Rgb8
        } else {
            image::ExtendedColorType::L8
        };
        image::save_buffer(png, &level.data, level.width, level.height, color)?;
    }
    let size = |tex: &TexFile| tex.mips.iter().map(Vec::len).sum::<usize>();
    let report = format!(
        "{} {kind:?}: {width}×{height}, {} mips {:?} {:.1} MB + fallback {}×{} {:?} {:.1} MB; {}; \
         {extra}; encode {:.1} s",
        spec.body,
        chain.len(),
        compressed.format,
        size(&compressed) as f64 / 1e6,
        fallback.width,
        fallback.height,
        fallback.format,
        size(&fallback) as f64 / 1e6,
        source,
        started.elapsed().as_secs_f64()
    );
    Ok(Baked {
        info: MapInfo {
            body: spec.body.clone(),
            naif: spec.naif,
            kind,
            file: file_name(out),
            fallback_file: file_name(&fallback_out),
            // The stub's meaning: the source's decoded size (mip 0 is `baked_size` of it).
            width: decoded.0,
            height: decoded.1,
            lon0_deg: spec.lon0_deg,
            albedo_scale: scale as f32,
            source,
        },
        report,
    })
}

/// The mip chain from mip 0: a 2×2 box filter in linear light, each level re-encoded with mip 0's
/// transfer. It continues while both sides of the last level are multiples of 8, so every level
/// is block-aligned and the chain ends at a level 4 texels high (or wide).
fn mip_chain(mip0: Image8) -> Vec<Image8> {
    let (transfer, channels) = (mip0.transfer, mip0.channels);
    let mut chain = vec![mip0];
    // Linear values of the last level once it is no longer mip 0, so each level is filtered
    // from unquantised values.
    let mut linear: Option<Vec<f32>> = None;
    loop {
        let last = &chain[chain.len() - 1];
        let (w, h) = (last.width as usize, last.height as usize);
        if w % 8 != 0 || h % 8 != 0 {
            break;
        }
        let (w2, h2) = (w / 2, h / 2);
        let mut next = vec![0.0_f32; w2 * h2 * channels];
        let source = linear.as_deref();
        next.par_chunks_mut(w2 * channels)
            .enumerate()
            .for_each(|(y, row)| {
                let at = |sx: usize, sy: usize, c: usize| {
                    let i = (sy * w + sx) * channels + c;
                    match source {
                        Some(values) => values[i],
                        None => transfer.decode(last.data[i]),
                    }
                };
                for x in 0..w2 {
                    for c in 0..channels {
                        // 2×2 box in linear light; each mip texel covers exactly its four children,
                        // so no neighbour across the u seam or the poles is read.
                        row[x * channels + c] = 0.25
                            * (at(2 * x, 2 * y, c)
                                + at(2 * x + 1, 2 * y, c)
                                + at(2 * x, 2 * y + 1, c)
                                + at(2 * x + 1, 2 * y + 1, c));
                    }
                }
            });
        let data = next.par_iter().map(|&l| transfer.encode(l)).collect();
        chain.push(Image8 {
            width: w2 as u32,
            height: h2 as u32,
            channels,
            transfer,
            data,
        });
        linear = Some(next);
    }
    chain
}

/// Index of the fallback's first level: the level 2048 wide, or the widest one narrower than
/// that, or mip 0 when the map is no wider than 2048.
fn fallback_start(chain: &[Image8]) -> usize {
    chain
        .iter()
        .position(|level| level.width <= FALLBACK_WIDTH)
        .unwrap_or(chain.len() - 1)
}

/// Packs `levels` as a `UNITEX1` texture: BC7 (sRGB albedo) or BC4 (linear single channel) when
/// `compressed`, else RGBA8 sRGB (alpha 255) or R8.
fn pack(levels: &[Image8], compressed: bool) -> anyhow::Result<uni_data::TexFile> {
    let first = levels.first().context("no levels to pack")?;
    let single = match (first.channels, first.transfer) {
        (1, Transfer::Linear) => true,
        (1 | 3, Transfer::Srgb) => false,
        (channels, transfer) => bail!("cannot pack {channels}-channel {transfer:?} texels"),
    };
    let format = match (single, compressed) {
        (true, true) => TexFormat::Bc4RUnorm,
        (true, false) => TexFormat::R8Unorm,
        (false, true) => TexFormat::Bc7RgbaUnormSrgb,
        (false, false) => TexFormat::Rgba8UnormSrgb,
    };
    let mips = levels
        .iter()
        .map(|level| {
            let (w, h) = (level.width, level.height);
            Ok(match format {
                TexFormat::Bc4RUnorm => bc::encode_bc4(&level.data, w, h)?,
                TexFormat::R8Unorm => level.data.clone(),
                TexFormat::Bc7RgbaUnormSrgb => bc::encode_bc7_srgb(&to_rgba8(level), w, h)?,
                TexFormat::Rgba8UnormSrgb => to_rgba8(level),
                TexFormat::Bc1RgbaUnormSrgb => bail!("BC1 is the own-encoder path, not built"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TexFile {
        format,
        width: first.width,
        height: first.height,
        mips,
    })
}

/// RGBA8 texels (grey replicated, alpha 255) of an sRGB level.
fn to_rgba8(level: &Image8) -> Vec<u8> {
    let channels = level.channels;
    level
        .data
        .par_chunks(channels)
        .flat_map_iter(|t| {
            let rgb = if channels == 3 {
                [t[0], t[1], t[2]]
            } else {
                [t[0]; 3]
            };
            [rgb[0], rgb[1], rgb[2], 255]
        })
        .collect()
}

/// A map's phase-0 disk-average weights binned by 8-bit code, so the disk average of the scaled
/// reflectance capped at 1, min(1, s·a) per channel, is a sum over 256 codes per channel for any
/// scale s.
///
/// At phase 0 (μ₀ = μ) a sphere's radiance factor is I/F = ρ·f(μ) with f = μ (Lambert) or
/// L + (1 − L)μ (lunar-Lambert, McEwen 1991, Icarus 92, 298), and the geometric albedo is its
/// mean over the projected disk, p = (1/π)∫ ρ f(μ) μ dA over the visible hemisphere. So texel
/// weights are f(μ)·μ·dA, dA = cos φ dφ dλ. A full rotation seen from the equatorial plane
/// (μ = cos φ cos(λ − θ), averaged over θ) gives the latitude weight
/// 2L cos²φ + (1 − L)(π/2) cos³φ, integrated exactly over each texel row.
///
/// An RGB texel's reflectance is its BT.709 luminance (ITU-R BT.709, the closest standard match
/// to the V band) after each channel is capped. Earth's clouds (coverage c) are white clouds over
/// the ground, each layer capped at 1: ρ = (1 − c)·min(1, s·ρ_ground) + c·min(1, s).
struct DiskHistogram {
    /// Per ground channel: Σ w·(1 − c) over the texels holding each code.
    ground: Vec<[f64; 256]>,
    /// Σ w·c: the weight of the white clouds (0 without clouds).
    clouds: f64,
    /// Σ w.
    total: f64,
    /// Linear value of each code in the ground's transfer.
    decode: [f64; 256],
}

/// ITU-R BT.709 luminance weights of linear R, G, B.
const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

impl DiskHistogram {
    fn new(
        ground: &Image8,
        clouds: Option<&Image8>,
        brdf: Brdf,
        view: View,
        lon0_deg: f64,
    ) -> DiskHistogram {
        let l = match brdf {
            Brdf::Lambert => 0.0,
            Brdf::LunarLambert { l } => l,
        };
        let (w, h, ch) = (
            ground.width as usize,
            ground.height as usize,
            ground.channels,
        );
        let dlat = std::f64::consts::PI / h as f64;
        let empty = || (vec![[0.0_f64; 256]; ch], 0.0_f64, 0.0_f64);
        let (hist, cloud_weight, total) = (0..h)
            .into_par_iter()
            .fold(empty, |(mut hist, mut cloud_weight, mut total), y| {
                let top = std::f64::consts::FRAC_PI_2 - y as f64 * dlat;
                let bottom = top - dlat;
                let mid = top - 0.5 * dlat;
                // ∫cos²φ dφ = φ/2 + sin 2φ/4; ∫cos³φ dφ = sin φ − sin³φ/3.
                let int_cos2 = |p: f64| p / 2.0 + (2.0 * p).sin() / 4.0;
                let int_cos3 = |p: f64| p.sin() - p.sin().powi(3) / 3.0;
                let row_weight = 2.0 * l * (int_cos2(top) - int_cos2(bottom))
                    + (1.0 - l) * std::f64::consts::FRAC_PI_2 * (int_cos3(top) - int_cos3(bottom));
                for x in 0..w {
                    let weight = match view {
                        View::RotationAveraged => row_weight,
                        View::Fixed {
                            sub_observer_lon_deg,
                        } => {
                            let lon = lon0_deg + (x as f64 + 0.5) * 360.0 / w as f64;
                            let mu = mid.cos() * (lon - sub_observer_lon_deg).to_radians().cos();
                            if mu <= 0.0 {
                                continue;
                            }
                            (l + (1.0 - l) * mu) * mu * mid.cos() * dlat
                        }
                    };
                    let t = y * w + x;
                    let c = clouds.map_or(0.0, |clouds| f64::from(clouds.linear(t, 0)));
                    for (k, bins) in hist.iter_mut().enumerate() {
                        bins[usize::from(ground.data[t * ch + k])] += weight * (1.0 - c);
                    }
                    cloud_weight += weight * c;
                    total += weight;
                }
                (hist, cloud_weight, total)
            })
            .reduce(empty, |mut a, b| {
                for (bins, other) in a.0.iter_mut().zip(&b.0) {
                    for (n, m) in bins.iter_mut().zip(other) {
                        *n += m;
                    }
                }
                (a.0, a.1 + b.1, a.2 + b.2)
            });
        DiskHistogram {
            ground: hist,
            clouds: cloud_weight,
            total,
            decode: std::array::from_fn(|code| f64::from(ground.transfer.decode(code as u8))),
        }
    }

    /// Luminance weight of each ground channel.
    fn luma(&self) -> &'static [f64] {
        if self.ground.len() == 3 {
            &LUMA
        } else {
            &[1.0]
        }
    }

    /// Disk-averaged reflectance with every channel scaled by `scale` and capped at 1. At
    /// `scale` 1 nothing is capped (texels are ≤ 1), so this is the map's own disk average.
    fn average(&self, scale: f64) -> f64 {
        let ground: f64 = self
            .ground
            .iter()
            .zip(self.luma())
            .map(|(bins, k)| {
                k * bins
                    .iter()
                    .zip(&self.decode)
                    .map(|(n, a)| n * (scale * a).min(1.0))
                    .sum::<f64>()
            })
            .sum();
        (ground + self.clouds * scale.min(1.0)) / self.total
    }

    /// Share of the disk-averaged luminance weight that sits at the cap at `scale`.
    fn share_at_cap(&self, scale: f64) -> f64 {
        let ground: f64 = self
            .ground
            .iter()
            .zip(self.luma())
            .map(|(bins, k)| {
                k * bins
                    .iter()
                    .zip(&self.decode)
                    .filter(|&(_, a)| scale * a >= 1.0)
                    .map(|(n, _)| n)
                    .sum::<f64>()
            })
            .sum();
        let clouds = if scale >= 1.0 { self.clouds } else { 0.0 };
        (ground + clouds) / self.total
    }

    /// Linear values of the lit (non-zero) codes that carry weight, and 1 for the clouds.
    fn lit(&self) -> impl Iterator<Item = f64> + '_ {
        let ground = self.ground.iter().flat_map(|bins| {
            bins.iter()
                .zip(&self.decode)
                .filter(|&(n, a)| *n > 0.0 && *a > 0.0)
                .map(|(_, a)| *a)
        });
        ground.chain((self.clouds > 0.0).then_some(1.0))
    }

    /// The largest scale at which no texel reaches the cap: 1 / the brightest lit value.
    fn first_cap(&self) -> f64 {
        1.0 / self.lit().fold(0.0, f64::max)
    }

    /// The scale s whose capped disk average is `target` (review I4 with the plan's "ρ is capped
    /// at 1", review r0 Important 1). Below [`Self::first_cap`] the average is linear in s, so s
    /// is the plain ratio target / average; above it the average is piecewise linear and
    /// non-decreasing, and s is found by bisection. Fails when the target is out of reach even
    /// with every lit texel at the cap.
    fn solve(&self, target: f64) -> Result<f64> {
        ensure!(
            target > 0.0 && target <= 1.0,
            "target reflectance {target} is outside (0, 1]"
        );
        let dimmest = self.lit().fold(f64::INFINITY, f64::min);
        ensure!(dimmest.is_finite(), "the map is black");
        let linear = target / self.average(1.0);
        if linear <= self.first_cap() {
            return Ok(linear);
        }
        // From 1 / the dimmest lit value on, every lit texel is at the cap.
        let (mut lo, mut hi) = (self.first_cap(), 1.0 / dimmest);
        let reach = self.average(hi);
        ensure!(
            target <= reach * (1.0 + 1e-12),
            "reflectance {target:.4} is out of reach: with every channel capped at 1 this map's \
             disk average is at most {reach:.4}"
        );
        while hi - lo > 1e-12 * hi {
            let mid = 0.5 * (lo + hi);
            if self.average(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(hi)
    }
}

/// The map's phase-0 disk-averaged reflectance (Earth's `clouds` composited), unscaled.
#[cfg(test)]
fn disk_average(
    ground: &Image8,
    clouds: Option<&Image8>,
    brdf: Brdf,
    view: View,
    lon0_deg: f64,
) -> f64 {
    DiskHistogram::new(ground, clouds, brdf, view, lon0_deg).average(1.0)
}

/// `image` with every channel scaled by `scale` and capped at 1, re-encoded in its transfer: the
/// stored reflectance of a map whose scale would push texels past 1.
fn scaled_capped(image: Image8, scale: f64) -> Image8 {
    let transfer = image.transfer;
    let table: [u8; 256] = std::array::from_fn(|code| {
        let linear = scale * f64::from(transfer.decode(code as u8));
        transfer.encode(linear.min(1.0) as f32)
    });
    let data = image
        .data
        .par_iter()
        .map(|&code| table[usize::from(code)])
        .collect();
    Image8 { data, ..image }
}

/// The scale for the index and the texels to store: `scale` itself when no texel reaches the cap,
/// else the capped reflectance with scale 1 (so `albedo_scale` × texel ≤ 1 holds in the file,
/// and every mip averages capped values).
fn apply_scale(ground: Image8, scale: f64, first_cap: f64) -> (Image8, f64) {
    if scale <= first_cap {
        (ground, scale)
    } else {
        (scaled_capped(ground, scale), 1.0)
    }
}

/// Replaces the no-data texels with the cos-latitude-weighted mean of the others, in linear
/// light, and returns how many. No data is a 4-connected region (wrapping in u) of dark texels,
/// DN ≤ `max_dn` in every channel, that touches the top or bottom row (an unimaged polar cap) or
/// covers at least 0.1 % of the map and 64 texels (an unimaged gap). Smaller enclosed dark spots
/// (black paterae, shadows) are data.
fn fill_nodata(image: &mut Image8, max_dn: u8) -> usize {
    let (w, h, ch) = (image.width as usize, image.height as usize, image.channels);
    let min_gap = (w * h / 1000).max(64);
    let dark = |t: usize| {
        image.data[t * ch..(t + 1) * ch]
            .iter()
            .all(|&v| v <= max_dn)
    };
    let mut seen = vec![false; w * h];
    let mut nodata = vec![false; w * h];
    for seed in 0..w * h {
        if seen[seed] || !dark(seed) {
            continue;
        }
        seen[seed] = true;
        let (mut region, mut stack, mut polar) = (Vec::new(), vec![seed], false);
        while let Some(t) = stack.pop() {
            region.push(t);
            let (x, y) = (t % w, t / w);
            polar |= y == 0 || y + 1 == h;
            let mut neighbours = vec![y * w + (x + 1) % w, y * w + (x + w - 1) % w];
            if y > 0 {
                neighbours.push(t - w);
            }
            if y + 1 < h {
                neighbours.push(t + w);
            }
            for n in neighbours {
                if !seen[n] && dark(n) {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        if polar || region.len() >= min_gap {
            for t in region {
                nodata[t] = true;
            }
        }
    }
    let filled = nodata.iter().filter(|&&n| n).count();
    if filled == 0 || filled == w * h {
        return filled;
    }
    let mut sum = vec![0.0_f64; ch];
    let mut weight = 0.0_f64;
    for y in 0..h {
        let cos_lat = ((y as f64 + 0.5) / h as f64 * std::f64::consts::PI
            - std::f64::consts::FRAC_PI_2)
            .cos();
        for x in 0..w {
            let t = y * w + x;
            if !nodata[t] {
                for (c, s) in sum.iter_mut().enumerate() {
                    *s += cos_lat * f64::from(image.linear(t, c));
                }
                weight += cos_lat;
            }
        }
    }
    let fill: Vec<u8> = sum
        .iter()
        .map(|s| image.transfer.encode((s / weight) as f32))
        .collect();
    for (t, _) in nodata.iter().enumerate().filter(|&(_, &n)| n) {
        image.data[t * ch..(t + 1) * ch].copy_from_slice(&fill);
    }
    filled
}

/// Resamples a roughly 2:1 image to its stored size, the 2:1 power of two nearest its width
/// ([`uni_data::maps::baked_size`]; review r0 Important 2: rounding always down halved Ceres's
/// resolution). Each output texel is the area-weighted mean, in linear light, of the source
/// texels it overlaps (wrapping in u, clamping in v): a box filter when shrinking, and when
/// enlarging a blend of the one or two source texels under it, so the map's mean is kept either
/// way. A 2:1 power of two is returned unchanged.
fn to_power_of_two(image: Image8) -> anyhow::Result<Image8> {
    let (w, h) = (image.width, image.height);
    ensure!(
        (i64::from(w) - 2 * i64::from(h)).abs() <= 2,
        "{w}×{h} is not a 2:1 equirectangular map"
    );
    if w.is_power_of_two() && w == 2 * h {
        return Ok(image);
    }
    let (w2, h2) = uni_data::maps::baked_size(w);
    ensure!(h2 >= 4, "{w}×{h} is too small to bake");
    let ch = image.channels;
    let (wu, hu, w2u, h2u) = (w as usize, h as usize, w2 as usize, h2 as usize);
    let linear: Vec<f32> = image
        .data
        .par_iter()
        .map(|&v| image.transfer.decode(v))
        .collect();
    // Horizontal pass (wrap in u), then vertical (clamp in v).
    let x_taps = area_taps(wu, w2u);
    let mut across = vec![0.0_f32; w2u * hu * ch];
    across
        .par_chunks_mut(w2u * ch)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, taps) in x_taps.iter().enumerate() {
                for c in 0..ch {
                    row[x * ch + c] = taps
                        .iter()
                        .map(|&(sx, wt)| wt * linear[(y * wu + sx % wu) * ch + c])
                        .sum();
                }
            }
        });
    let y_taps = area_taps(hu, h2u);
    let mut out = vec![0_u8; w2u * h2u * ch];
    out.par_chunks_mut(w2u * ch)
        .enumerate()
        .for_each(|(y, row)| {
            for (i, value) in row.iter_mut().enumerate() {
                let l: f32 = y_taps[y]
                    .iter()
                    .map(|&(sy, wt)| wt * across[sy.min(hu - 1) * w2u * ch + i])
                    .sum();
                *value = image.transfer.encode(l);
            }
        });
    Ok(Image8 {
        width: w2,
        height: h2,
        channels: ch,
        transfer: image.transfer,
        data: out,
    })
}

/// For each of `m` output texels spanning `n` input texels, the inputs it overlaps and the
/// overlap weights (summing to 1).
fn area_taps(n: usize, m: usize) -> Vec<Vec<(usize, f32)>> {
    let step = n as f64 / m as f64;
    (0..m)
        .map(|i| {
            let (start, end) = (i as f64 * step, (i + 1) as f64 * step);
            (start.floor() as usize..end.ceil() as usize)
                .filter_map(|j| {
                    let overlap = end.min(j as f64 + 1.0) - start.max(j as f64);
                    (overlap > 0.0).then_some((j, (overlap / step) as f32))
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_physics::photometry::rho_from_geometric_albedo;

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    fn image(
        width: u32,
        height: u32,
        channels: usize,
        transfer: Transfer,
        f: impl Fn(u32, u32, usize) -> u8,
    ) -> Image8 {
        let data = (0..height)
            .flat_map(|y| (0..width).flat_map(move |x| (0..channels).map(move |c| (x, y, c))))
            .map(|(x, y, c)| f(x, y, c))
            .collect();
        Image8 {
            width,
            height,
            channels,
            transfer,
            data,
        }
    }

    fn srgb_to_linear(v: u8) -> f64 {
        // IEC 61966-2-1 (sRGB) decoding.
        let c = f64::from(v) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn linear_to_srgb(l: f64) -> u8 {
        let c = if l <= 0.003_130_8 {
            12.92 * l
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        (c * 255.0).round().clamp(0.0, 255.0) as u8
    }

    // ---- command line ----

    #[test]
    fn parses_map_groups() {
        let (specs, index) = parse(&args(&[
            "--index",
            "maps/index.ron",
            "--input",
            "moon.jpg",
            "--body",
            "Moon",
            "--naif",
            "301",
            "--kind",
            "albedo",
            "--lon0",
            "-180",
            "--geometric-albedo",
            "0.12",
            "--brdf",
            "lunar-lambert",
            "--view-lon",
            "0",
            "--source",
            "SSS",
            "--out",
            "moon_albedo.big",
            "--input",
            "earth.jpg",
            "--body",
            "Earth",
            "--naif",
            "399",
            "--kind",
            "albedo",
            "--lon0",
            "-180",
            "--geometric-albedo",
            "0.434",
            "--clouds",
            "clouds.jpg",
            "--clouds-out",
            "earth_clouds.big",
            "--source",
            "SSS",
            "--out",
            "earth_albedo.big",
            "--input",
            "europa.jpg",
            "--body",
            "Europa",
            "--naif",
            "502",
            "--kind",
            "albedo",
            "--lon0",
            "0",
            "--geometric-albedo",
            "0.68",
            "--brdf",
            "lunar-lambert",
            "--encoding",
            "linear",
            "--fill-nodata",
            "4",
            "--source",
            "USGS",
            "--out",
            "europa_albedo.big",
        ]))
        .expect("valid");
        assert_eq!(index, Some(PathBuf::from("maps/index.ron")));
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].brdf, Brdf::LunarLambert { l: 0.9 });
        assert_eq!(
            specs[0].view,
            View::Fixed {
                sub_observer_lon_deg: 0.0
            }
        );
        assert_eq!(specs[0].naif, 301);
        assert_eq!(specs[1].brdf, Brdf::Lambert);
        assert_eq!(specs[1].view, View::RotationAveraged);
        assert_eq!(specs[1].encoding, Transfer::Srgb);
        assert_eq!(
            specs[1].clouds,
            Some((
                PathBuf::from("clouds.jpg"),
                PathBuf::from("earth_clouds.big")
            ))
        );
        assert_eq!(specs[2].encoding, Transfer::Linear);
        assert_eq!(specs[2].fill_nodata, Some(4));
        assert_eq!(specs[2].lon0_deg, 0.0);
    }

    #[test]
    fn rejects_bad_map_command_lines() {
        let good = [
            "--input",
            "a.jpg",
            "--body",
            "Mars",
            "--naif",
            "499",
            "--kind",
            "albedo",
            "--lon0",
            "-180",
            "--geometric-albedo",
            "0.17",
            "--source",
            "SSS",
            "--out",
            "m.big",
        ];
        assert!(parse(&args(&good)).is_ok());
        assert!(parse(&args(&[])).is_err(), "no map");
        // Every required flag is required.
        for required in [
            "--body",
            "--naif",
            "--kind",
            "--lon0",
            "--geometric-albedo",
            "--source",
            "--out",
        ] {
            let i = good.iter().position(|a| *a == required).expect("present");
            let mut line = good.to_vec();
            line.drain(i..i + 2);
            assert!(parse(&args(&line)).is_err(), "missing {required}");
        }
        // Sets a flag (replacing its value when the good line has it) and parses.
        let with = |extra: &[&str]| {
            let mut line = good.to_vec();
            for pair in extra.chunks(2) {
                match line.iter().position(|a| *a == pair[0]) {
                    Some(i) if pair.len() == 2 => line[i + 1] = pair[1],
                    _ => line.extend_from_slice(pair),
                }
            }
            parse(&args(&line))
        };
        assert!(with(&["--brdf", "lunar-lambert"]).is_ok());
        let mut twice = good.to_vec();
        twice.extend_from_slice(&["--body", "Venus"]);
        assert!(parse(&args(&twice)).is_err(), "a flag twice for one map");
        assert!(
            with(&["--kind", "clouds"]).is_err(),
            "clouds come with --clouds"
        );
        assert!(with(&["--brdf", "phong"]).is_err());
        assert!(
            with(&["--clouds", "c.jpg"]).is_err(),
            "--clouds needs --clouds-out"
        );
        assert!(with(&["--naif", "x"]).is_err());
        assert!(with(&["--geometric-albedo", "-0.1"]).is_err());
        assert!(with(&["--bogus", "1"]).is_err());
        assert!(with(&["--out"]).is_err(), "flag without a value");
        assert!(
            parse(&args(&["--body", "Mars"])).is_err(),
            "option before --input"
        );
    }

    // ---- mips and packing ----

    #[test]
    fn bc7_block_layout() {
        // A square map halves down to 4×4; a 2:1 map ends at 8×4.
        for (w, h, last) in [(64_u32, 64_u32, (4, 4)), (128, 64, (8, 4))] {
            let mip0 = image(w, h, 3, Transfer::Srgb, |x, y, c| {
                ((x * 5 + y * 3 + c as u32 * 70) % 256) as u8
            });
            let chain = mip_chain(mip0);
            let sizes: Vec<(u32, u32)> = chain.iter().map(|l| (l.width, l.height)).collect();
            assert_eq!(sizes.first(), Some(&(w, h)));
            assert_eq!(sizes.last(), Some(&last));
            for pair in sizes.windows(2) {
                assert_eq!(pair[1], (pair[0].0 / 2, pair[0].1 / 2), "{sizes:?}");
            }
            let tex = pack(&chain, true).expect("BC7");
            assert_eq!(tex.format, uni_data::TexFormat::Bc7RgbaUnormSrgb);
            assert_eq!((tex.width, tex.height), (w, h));
            assert_eq!(tex.mips.len(), chain.len());
            for (level, mip) in tex.mips.iter().enumerate() {
                let (mw, mh) = sizes[level];
                assert_eq!(
                    mip.len(),
                    (mw.div_ceil(4) * mh.div_ceil(4) * 16) as usize,
                    "mip {level}"
                );
            }
        }
        // Single-channel coverage packs as BC4 (8 B per block) or R8.
        let clouds = mip_chain(image(32, 16, 1, Transfer::Linear, |x, _, _| (x * 8) as u8));
        let bc4 = pack(&clouds, true).expect("BC4");
        assert_eq!(bc4.format, uni_data::TexFormat::Bc4RUnorm);
        assert_eq!(bc4.mips[0].len(), 8 * 4 * 8);
        let r8 = pack(&clouds, false).expect("R8");
        assert_eq!(r8.format, uni_data::TexFormat::R8Unorm);
        assert_eq!(r8.mips[0], clouds[0].data);
    }

    #[test]
    fn mips_box_filter_in_linear_light() {
        // A 2×2 checker of sRGB 0 and 255 averages to linear 0.5, i.e. sRGB 188, not 128.
        let mip0 = image(8, 8, 3, Transfer::Srgb, |x, y, _| {
            if (x + y) % 2 == 0 { 255 } else { 0 }
        });
        let chain = mip_chain(mip0);
        assert!(
            chain[1].data.iter().all(|&v| v == linear_to_srgb(0.5)),
            "{:?}",
            &chain[1].data[..6]
        );
        // Linear storage averages the codes themselves.
        let mip0 = image(8, 8, 1, Transfer::Linear, |x, y, _| {
            if (x + y) % 2 == 0 { 200 } else { 100 }
        });
        assert!(mip_chain(mip0)[1].data.iter().all(|&v| v == 150));
    }

    /// The fallback is exactly the box-filtered mip of width 2048, computed here independently
    /// as one 4×4 box in linear light (f64) from a synthetic 8192×4096 source.
    #[test]
    fn fallback_is_the_2k_mip() {
        let (w, h) = (8192_u32, 4096_u32);
        let texel = |x: u32, y: u32, c: usize| -> u8 {
            let v =
                (x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503) ^ (c as u32 * 977)) >> 7;
            (v % 256) as u8
        };
        let chain = mip_chain(image(w, h, 3, Transfer::Srgb, texel));
        let start = fallback_start(&chain);
        assert_eq!((chain[start].width, chain[start].height), (2048, 1024));
        let tex = pack(&chain[start..], false).expect("RGBA8");
        assert_eq!(tex.format, uni_data::TexFormat::Rgba8UnormSrgb);
        assert_eq!((tex.width, tex.height), (2048, 1024));
        assert_eq!(tex.mips.len(), chain.len() - start);
        let fallback = &tex.mips[0];
        let (mut exact, mut worst) = (0_usize, 0_i32);
        for y in 0..1024 {
            for x in 0..2048 {
                for c in 0..3 {
                    let mut sum = 0.0;
                    for dy in 0..4 {
                        for dx in 0..4 {
                            sum += srgb_to_linear(texel(4 * x + dx, 4 * y + dy, c));
                        }
                    }
                    let want = i32::from(linear_to_srgb(sum / 16.0));
                    let got = i32::from(fallback[((y * 2048 + x) * 4) as usize + c]);
                    worst = worst.max((want - got).abs());
                    exact += usize::from(want == got);
                }
                assert_eq!(fallback[((y * 2048 + x) * 4 + 3) as usize], 255);
            }
        }
        let total = 2048 * 1024 * 3;
        assert!(worst <= 1, "largest difference {worst} codes");
        assert!(
            exact as f64 >= 0.999 * total as f64,
            "{exact} of {total} texels exact"
        );
    }

    #[test]
    fn fallback_is_mip0_for_narrow_maps() {
        let chain = mip_chain(image(1024, 512, 1, Transfer::Linear, |x, _, _| x as u8));
        assert_eq!(fallback_start(&chain), 0);
    }

    /// Seam safety: a 1-texel-wide stripe centred on u = 0 (half in the last column, half in the
    /// first, as area sampling rasterises it) shows up in mip 2 at both edges equally, with its
    /// energy kept; shifting the source by one mip-2 texel shifts mip 2 by one texel across the
    /// seam; and v does not wrap (a stripe along the top row stays at the top).
    #[test]
    fn mip_wraps_in_u() {
        let (w, h) = (64_u32, 32_u32);
        let seam = image(w, h, 1, Transfer::Linear, |x, _, _| {
            if x == 0 || x == w - 1 { 100 } else { 0 }
        });
        let mip2 = &mip_chain(seam)[2];
        assert_eq!((mip2.width, mip2.height), (16, 8));
        for y in 0..8 {
            let row = &mip2.data[(y * 16) as usize..((y + 1) * 16) as usize];
            assert_eq!((row[0], row[15]), (25, 25), "row {y}: {row:?}");
            assert!(row[1..15].iter().all(|&v| v == 0), "row {y}: {row:?}");
            let edge_average = (f64::from(row[0]) + f64::from(row[15])) / 2.0;
            assert_eq!(
                edge_average,
                200.0 / 8.0,
                "a 200-high stripe over 8 texels at mip 2"
            );
        }
        // Rolling the source by 4 texels (one mip-2 texel) rolls mip 2 by one texel.
        let rolled = image(w, h, 1, Transfer::Linear, |x, _, _| {
            if x == 3 || x == 4 { 100 } else { 0 }
        });
        let mip2 = &mip_chain(rolled)[2];
        assert_eq!(&mip2.data[..3], &[25, 25, 0]);
        assert_eq!(mip2.data[15], 0);
        // Clamp in v: a top-row stripe never reaches the bottom.
        let top = image(
            w,
            h,
            1,
            Transfer::Linear,
            |_, y, _| if y == 0 { 200 } else { 0 },
        );
        let mip2 = &mip_chain(top)[2];
        assert!(mip2.data[..16].iter().all(|&v| v == 50));
        assert!(mip2.data[7 * 16..].iter().all(|&v| v == 0));
    }

    // ---- albedo scale ----

    /// Linear reflectance of an sRGB texel as the bake defines it: grey, or BT.709 luminance.
    fn texel_rho(map: &Image8, t: usize) -> f64 {
        let at = |c: usize| srgb_to_linear(map.data[t * map.channels + c]);
        if map.channels == 3 {
            0.2126 * at(0) + 0.7152 * at(1) + 0.0722 * at(2)
        } else {
            at(0)
        }
    }

    /// Disk-averaged ρ̄ by brute force: render the visible disk at phase 0 for `views`
    /// sub-observer longitudes, look each pixel's surface point up in the map (nearest texel),
    /// weight it by the BRDF's phase-0 radiance factor f(μ) (Lambert μ; lunar-Lambert
    /// L + (1 − L)μ, McEwen 1991) and divide by the summed weights.
    fn rendered_disk_average(map: &Image8, lon0_deg: f64, brdf: Brdf, sub_lons_deg: &[f64]) -> f64 {
        let l = match brdf {
            Brdf::Lambert => 0.0,
            Brdf::LunarLambert { l } => l,
        };
        let n = 600;
        let (mut num, mut den) = (0.0, 0.0);
        for &sub_lon in sub_lons_deg {
            for j in 0..n {
                for i in 0..n {
                    let x = -1.0 + (2.0 * i as f64 + 1.0) / n as f64;
                    let y = -1.0 + (2.0 * j as f64 + 1.0) / n as f64;
                    let r2 = x * x + y * y;
                    if r2 >= 1.0 {
                        continue;
                    }
                    let mu = (1.0 - r2).sqrt();
                    let lat = y.asin().to_degrees();
                    let lon = sub_lon + x.atan2(mu).to_degrees();
                    let u = ((lon - lon0_deg) / 360.0).rem_euclid(1.0);
                    let col = ((u * f64::from(map.width)) as u32).min(map.width - 1);
                    let row =
                        (((90.0 - lat) / 180.0 * f64::from(map.height)) as u32).min(map.height - 1);
                    let rho = texel_rho(map, (row * map.width + col) as usize);
                    let f = l + (1.0 - l) * mu;
                    num += rho * f;
                    den += f;
                }
            }
        }
        num / den
    }

    /// A grey synthetic map whose albedo varies with latitude and longitude.
    fn synthetic_albedo_map() -> Image8 {
        image(128, 64, 3, Transfer::Srgb, |x, y, _| {
            let lat = 90.0 - (f64::from(y) + 0.5) * 180.0 / 64.0;
            let lon = (f64::from(x) + 0.5) * 360.0 / 128.0;
            let l = 0.15 + 0.35 * (lat.to_radians().sin().abs()) + 0.1 * lon.to_radians().cos();
            linear_to_srgb(l)
        })
    }

    /// What the bake does with a map: solve the capped scale for p_V and store the texels.
    fn bake_scale(
        map: &Image8,
        clouds: Option<&Image8>,
        brdf: Brdf,
        view: View,
        p_v: f64,
    ) -> (Image8, f64) {
        let histogram = DiskHistogram::new(map, clouds, brdf, view, -180.0);
        let solved = histogram
            .solve(rho_from_geometric_albedo(brdf, p_v))
            .expect("reachable");
        apply_scale(map.clone(), solved, histogram.first_cap())
    }

    /// The largest stored channel value × scale.
    fn brightest_rho(map: &Image8, scale: f64) -> f64 {
        map.data
            .iter()
            .map(|&v| srgb_to_linear(v))
            .fold(0.0, f64::max)
            * scale
    }

    /// 36 sub-observer longitudes sample a full rotation.
    fn rotation() -> Vec<f64> {
        (0..36).map(|k| f64::from(k) * 10.0).collect()
    }

    /// The brief's test, through the capped solve and the texels as stored: this map (albedo
    /// 0.05–0.6) needs a scale of about 3, so its bright texels reach the cap and are stored
    /// capped, yet the rendered disk average × scale is still 0.807 and no texel exceeds 1.
    #[test]
    fn albedo_scale_hits_geometric_albedo() {
        let map = synthetic_albedo_map();
        let average = disk_average(&map, None, Brdf::Lambert, View::RotationAveraged, -180.0);
        let (stored, scale) = bake_scale(&map, None, Brdf::Lambert, View::RotationAveraged, 0.538);
        assert!(
            0.807 / average > 2.0,
            "the plain ratio would put texels at {}",
            0.6 * 0.807 / average
        );
        assert_eq!(scale, 1.0, "capped texels are stored as reflectance");
        let product = rendered_disk_average(&stored, -180.0, Brdf::Lambert, &rotation()) * scale;
        assert!(
            (product - 0.807).abs() <= 1e-3,
            "disk-averaged ρ × scale = {product}"
        );
        assert!(brightest_rho(&stored, scale) <= 1.0);
        assert!((rho_from_geometric_albedo(Brdf::Lambert, 0.538) - 0.807).abs() < 1e-12);
    }

    /// A saturated blue map like Solar System Scope's Neptune (luminance 0.14, review r0
    /// Important 1): the plain ratio 4.6 would put the blue channel at 2.1. Capped, the blue
    /// channel saturates, red and green carry the rest, and the disk average still hits ρ.
    #[test]
    fn capped_scale_conserves_energy_on_a_saturated_map() {
        let map = image(128, 64, 3, Transfer::Srgb, |x, y, c| {
            let band = 1.0 + 0.15 * (f64::from(y) * 0.4).sin() + 0.05 * (f64::from(x) * 0.3).cos();
            linear_to_srgb([0.05, 0.13, 0.45][c] * band)
        });
        let average = disk_average(&map, None, Brdf::Lambert, View::RotationAveraged, -180.0);
        let want = rho_from_geometric_albedo(Brdf::Lambert, 0.442);
        assert!(
            want / average * 0.45 > 1.5,
            "plain ratio {}",
            want / average
        );
        let (stored, scale) = bake_scale(&map, None, Brdf::Lambert, View::RotationAveraged, 0.442);
        let product = rendered_disk_average(&stored, -180.0, Brdf::Lambert, &rotation()) * scale;
        assert!((product - want).abs() <= 1e-3, "{product} vs {want}");
        assert!(brightest_rho(&stored, scale) <= 1.0);
        // Blue is at the cap everywhere; the hue order is kept (blue ≥ green ≥ red).
        let mean = |c: usize| {
            stored
                .data
                .chunks(3)
                .map(|t| srgb_to_linear(t[c]))
                .sum::<f64>()
                / (128.0 * 64.0)
        };
        assert_eq!(mean(2), 1.0);
        assert!(
            mean(2) >= mean(1) && mean(1) >= mean(0),
            "{}, {}",
            mean(0),
            mean(1)
        );
    }

    /// A target no capped map can reach is an error, not a silent shortfall.
    #[test]
    fn out_of_reach_target_is_an_error() {
        // Pure red: with red at the cap the luminance is 0.2126 at most.
        let red = image(
            64,
            32,
            3,
            Transfer::Srgb,
            |_, _, c| if c == 0 { 120 } else { 0 },
        );
        let histogram = DiskHistogram::new(&red, None, Brdf::Lambert, View::RotationAveraged, 0.0);
        assert!(histogram.solve(0.5).is_err());
        let reachable = histogram.solve(0.2).expect("reachable");
        assert!((histogram.average(reachable) - 0.2).abs() < 1e-9);
        assert!(histogram.solve(0.0).is_err() && histogram.solve(1.5).is_err());
        let black = image(64, 32, 1, Transfer::Srgb, |_, _, _| 0);
        let histogram =
            DiskHistogram::new(&black, None, Brdf::Lambert, View::RotationAveraged, 0.0);
        assert!(histogram.solve(0.1).is_err());
    }

    /// Earth: the white clouds are capped at 1 like the ground (they were at 2.33, review r0
    /// Important 1), so the ground carries the rest. Ground 0.05 and cover c everywhere:
    /// (1 − c)·min(1, s·0.05) + c = 0.651.
    #[test]
    fn clouds_are_capped_at_one() {
        let ground = image(64, 32, 3, Transfer::Srgb, |_, _, _| linear_to_srgb(0.05));
        let clouds = image(64, 32, 1, Transfer::Linear, |_, _, _| 128);
        let (c, g) = (128.0 / 255.0, srgb_to_linear(linear_to_srgb(0.05)));
        let want = rho_from_geometric_albedo(Brdf::Lambert, 0.434);
        let histogram = DiskHistogram::new(
            &ground,
            Some(&clouds),
            Brdf::Lambert,
            View::RotationAveraged,
            -180.0,
        );
        assert!(
            (histogram.first_cap() - 1.0).abs() < 1e-12,
            "clouds cap at s = 1"
        );
        let solved = histogram.solve(want).expect("reachable");
        let expected = (want - c) / (1.0 - c) / g;
        // The bake decodes through an f32 table, the test in f64.
        assert!(
            (solved / expected - 1.0).abs() < 1e-6,
            "{solved} vs {expected}"
        );
        let (stored, scale) = apply_scale(ground, solved, histogram.first_cap());
        assert_eq!(scale, 1.0, "cloud reflectance at full cover is 1");
        let stored_ground = srgb_to_linear(stored.data[0]);
        // Within half an sRGB step (0.0055 near 0.3).
        assert!(
            (stored_ground - expected * g).abs() < 3e-3,
            "{stored_ground}"
        );
    }

    #[test]
    fn albedo_scale_for_a_fixed_hemisphere() {
        // The Moon's p_V is the Earth-facing hemisphere's: lunar-Lambert, sub-observer lon 0.
        // Its scale is below 1, so no texel reaches the cap: the map is stored as it is and the
        // scale is the plain ratio.
        let map = synthetic_albedo_map();
        let brdf = Brdf::LunarLambert { l: 0.9 };
        let view = View::Fixed {
            sub_observer_lon_deg: 0.0,
        };
        let want = rho_from_geometric_albedo(brdf, 0.12);
        let (stored, scale) = bake_scale(&map, None, brdf, view, 0.12);
        assert_eq!(stored, map);
        let plain = want / disk_average(&map, None, brdf, view, -180.0);
        assert!((scale / plain - 1.0).abs() < 1e-12, "{scale} vs {plain}");
        let rendered = rendered_disk_average(&map, -180.0, brdf, &[0.0]);
        assert!(
            (rendered * scale / want - 1.0).abs() < 2e-3,
            "{} vs {want}",
            rendered * scale
        );
        // This map's far side is brighter, so the hemisphere matters.
        let far = disk_average(
            &map,
            None,
            brdf,
            View::Fixed {
                sub_observer_lon_deg: 180.0,
            },
            -180.0,
        );
        assert!((far - disk_average(&map, None, brdf, view, -180.0)).abs() > 0.01);
    }

    #[test]
    fn clouds_composite_over_the_ground() {
        // Ground ρ 0.2 everywhere, cloud coverage 0.5 everywhere: (1 − 0.5)·0.2 + 0.5 = 0.6.
        let ground = image(64, 32, 3, Transfer::Srgb, |_, _, _| linear_to_srgb(0.2));
        let clouds = image(64, 32, 1, Transfer::Linear, |_, _, _| 128);
        let c = 128.0 / 255.0;
        let rho_ground = srgb_to_linear(linear_to_srgb(0.2));
        let want = (1.0 - c) * rho_ground + c;
        let got = disk_average(
            &ground,
            Some(&clouds),
            Brdf::Lambert,
            View::RotationAveraged,
            -180.0,
        );
        // The bake decodes through an f32 table, the test in f64.
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        let bare = disk_average(&ground, None, Brdf::Lambert, View::RotationAveraged, -180.0);
        assert!((bare - rho_ground).abs() < 1e-6);
    }

    // ---- tier-B preparation ----

    #[test]
    fn fill_nodata_fills_unimaged_regions_only() {
        // Rows 0 and 15 are no data (0), as is an enclosed 8×8 gap (64 texels, the smallest
        // filled gap); a dark crater (0) of 2 texels in the middle is real data.
        let mut map = image(32, 16, 1, Transfer::Linear, |x, y, _| match (x, y) {
            (_, 0) | (_, 15) => 0,
            (20..28, 4..12) => 1,
            (8..10, 7) => 0,
            _ => 120,
        });
        let filled = fill_nodata(&mut map, 2);
        assert_eq!(filled, 2 * 32 + 64);
        // Filled with the imaged mean: 120 everywhere but the two dark crater texels.
        let fill = map.data[0];
        assert!((118..120).contains(&fill), "{fill}");
        let at = |x: usize, y: usize| map.data[y * 32 + x];
        assert!((0..32).all(|x| at(x, 0) == fill && at(x, 15) == fill));
        assert!((20..28).all(|x| (4..12).all(|y| at(x, y) == fill)));
        assert_eq!(
            (at(8, 7), at(9, 7)),
            (0, 0),
            "a small enclosed dark spot is data"
        );
        assert_eq!(at(10, 7), 120);
    }

    #[test]
    fn power_of_two_resample_keeps_the_mean() {
        let mean =
            |im: &Image8| im.data.iter().map(|&v| f64::from(v)).sum::<f64>() / im.data.len() as f64;
        // 12 is nearer 16 than 8 on a log scale, so this one is enlarged (like Ceres, 7383 →
        // 8192, review r0 Important 2); 20 → 16 and 11 → 8 shrink (like Ganymede, 8271 → 8192).
        for (w, want) in [(12_u32, (16, 8)), (20, (16, 8)), (11, (8, 4))] {
            let source = image(w, w / 2, 1, Transfer::Linear, |x, y, _| {
                ((x * 20 + y * 7) % 256) as u8
            });
            let out = to_power_of_two(source.clone()).expect("resample");
            assert_eq!((out.width, out.height), want);
            assert!(
                (mean(&out) - mean(&source)).abs() < 1.0,
                "{w}: {} vs {}",
                mean(&out),
                mean(&source)
            );
        }
        // Enlarging blends at most two neighbours: a constant stays constant, a step stays a
        // step one texel wide.
        let step = image(
            12,
            6,
            1,
            Transfer::Linear,
            |x, _, _| if x < 6 { 0 } else { 240 },
        );
        let out = to_power_of_two(step).expect("resample");
        let row: Vec<u8> = out.data[..16].to_vec();
        assert_eq!(&row[..7], &[0; 7], "{row:?}");
        assert_eq!(&row[9..15], &[240; 6], "{row:?}");
        let pow2 = image(16, 8, 1, Transfer::Linear, |x, _, _| x as u8);
        assert_eq!(to_power_of_two(pow2.clone()).expect("unchanged"), pow2);
        assert!(
            to_power_of_two(image(12, 12, 1, Transfer::Linear, |_, _, _| 0)).is_err(),
            "not 2:1"
        );
    }

    // ---- end to end ----

    /// A whole `uni-bake map` run on two small synthetic maps (one with clouds): files, index
    /// entries, sizes, formats and the shared Earth scale.
    #[test]
    fn bakes_small_maps_end_to_end() {
        let dir = std::env::temp_dir().join(format!("uni-bake-map-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let ground = dir.join("ground.png");
        let clouds = dir.join("clouds.png");
        let grey = dir.join("grey.png");
        image::RgbImage::from_fn(256, 128, |x, y| {
            image::Rgb([(x % 256) as u8, (y * 2) as u8, 90])
        })
        .save(&ground)
        .expect("png");
        image::GrayImage::from_fn(256, 128, |x, _| image::Luma([(x / 2) as u8]))
            .save(&clouds)
            .expect("png");
        image::GrayImage::from_fn(200, 100, |x, y| {
            image::Luma([if y < 3 { 0 } else { 60 + (x % 50) as u8 }])
        })
        .save(&grey)
        .expect("png");
        let p = |name: &str| dir.join(name).to_string_lossy().into_owned();
        run(&args(&[
            "--input",
            &p("ground.png"),
            "--body",
            "Earth",
            "--naif",
            "399",
            "--kind",
            "albedo",
            "--lon0",
            "-180",
            "--geometric-albedo",
            "0.434",
            "--clouds",
            &p("clouds.png"),
            "--clouds-out",
            &p("earth_clouds.big"),
            "--source",
            "test",
            "--out",
            &p("earth_albedo.big"),
            "--input",
            &p("grey.png"),
            "--body",
            "Ceres",
            "--naif",
            "2000001",
            "--kind",
            "albedo",
            "--lon0",
            "0",
            "--geometric-albedo",
            "0.09",
            "--brdf",
            "lunar-lambert",
            "--encoding",
            "linear",
            "--fill-nodata",
            "2",
            "--source",
            "test",
            "--out",
            &p("ceres_albedo.big"),
        ]))
        .expect("bake");
        let maps = uni_data::load_map_index(&dir.join("index.ron")).expect("index");
        assert_eq!(maps.len(), 3);
        let find = |body: &str, kind: uni_data::MapKind| {
            maps.iter()
                .find(|m| m.body == body && m.kind == kind)
                .expect("entry")
                .clone()
        };
        let earth = find("Earth", uni_data::MapKind::Albedo);
        let earth_clouds = find("Earth", uni_data::MapKind::Clouds);
        let ceres = find("Ceres", uni_data::MapKind::Albedo);
        assert_eq!(
            earth.albedo_scale, earth_clouds.albedo_scale,
            "scaled together"
        );
        assert_eq!((earth.width, earth.height), (256, 128));
        // The index records the decoded size (the stub's meaning); the files hold the nearest
        // 2:1 power of two, 256×128 (review r0 Important 2).
        assert_eq!((ceres.width, ceres.height), (200, 100));
        assert!(ceres.source.contains("200×100"), "{}", ceres.source);
        assert!(ceres.source.contains("256×128"), "{}", ceres.source);
        assert_eq!(earth_clouds.fallback_file, "earth_clouds.r2k.big");
        // Earth's p_V needs clouds brighter than 1 at the plain ratio; capped, the cloud
        // reflectance is the shared scale, at most 1 (review r0 Important 1).
        assert!(earth.albedo_scale <= 1.0, "{}", earth.albedo_scale);
        for map in [&earth, &earth_clouds, &ceres] {
            let bc = uni_data::load_tex(&dir.join(&map.file)).expect("bc");
            let fallback = uni_data::load_tex(&dir.join(&map.fallback_file)).expect("fallback");
            let stored = uni_data::maps::baked_size(map.width);
            assert_eq!((bc.width, bc.height), stored, "{}", map.file);
            assert_eq!((fallback.width, fallback.height), stored, "{}", map.file);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
