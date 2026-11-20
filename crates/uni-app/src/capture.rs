//! Headless --capture mode: renders bookmarks to PNG plus JSON metrics (M0 Task 11).
//!
//! For each bookmark, `--frames` frames are rendered one at a time (each waits for the GPU),
//! the last one is saved as `<out>/<name>.png`, and `<out>/<name>.json` records its size, the
//! GPU time of each pass, the CPU time to update and encode it, the star count, the lens scale
//! and the epoch (design §8, "Headless capture").

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context as _;
use glam::UVec2;
use uni_core::Epoch;
use uni_gfx::{CAPTURE_FORMAT, CaptureTarget, GpuContext};
use uni_ui::Hud;

use crate::scene::{
    CAPTURE_LENS_SCALE, Perf, Scene, apply_bookmark, default_controls, exposure, frame_ctx,
    hud_data,
};
use crate::world::{World, iso_utc, sky_path};

/// Default capture size (brief).
pub const DEFAULT_SIZE: UVec2 = UVec2::new(1920, 1080);
/// Default frames rendered per bookmark before the capture (brief).
pub const DEFAULT_FRAMES: u32 = 4;
/// Epoch of a bookmark that sets none: J2000.0, so captures never depend on the clock.
const DEFAULT_EPOCH: Epoch = Epoch::J2000;
/// Frame time handed to the features in captures [s]; nothing moves, so it only fills
/// `CameraUniform::time`.
const CAPTURE_DT_S: f64 = 1.0 / 60.0;
/// Physical pixels per egui point for `--hud` (lead ruling: 1:1, the capture is 1920×1080
/// logical = physical).
const HEADLESS_HUD_PIXELS_PER_POINT: f32 = 1.0;

/// `universe --capture <bookmarks.ron> --out <dir> [--size WxH] [--frames N] [--hud]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptureArgs {
    pub bookmarks: PathBuf,
    pub out: PathBuf,
    pub size: UVec2,
    pub frames: u32,
    pub hud: bool,
}

/// Why a capture run failed. The binary exits with 2 for [`CaptureError::MissingData`] and
/// 1 otherwise.
#[derive(Debug)]
pub enum CaptureError {
    /// A baked data file is missing or unreadable (Review Focus 3).
    MissingData(String),
    Failed(anyhow::Error),
}

impl From<anyhow::Error> for CaptureError {
    fn from(e: anyhow::Error) -> Self {
        CaptureError::Failed(e)
    }
}

/// `<name>.json`: the capture's metrics.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CaptureMetrics {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Pass name → GPU milliseconds; empty without timestamp queries (Review Focus 4).
    pub gpu_ms: BTreeMap<String, f32>,
    /// CPU milliseconds to update and encode the captured frame.
    pub cpu_ms: f32,
    pub star_count: u32,
    /// Lens tracer resolution scale used.
    pub lens_scale: f32,
    /// The bookmark's epoch, `YYYY-MM-DDThh:mm:ssZ` (UTC, whole seconds).
    pub epoch_utc: String,
}

/// Renders every bookmark in `args.bookmarks` into `args.out`.
pub fn run(args: &CaptureArgs) -> Result<(), CaptureError> {
    let sky = sky_path();
    let (stars, sky_warnings) = World::load_sky(&sky);
    if !sky_warnings.is_empty() {
        return Err(CaptureError::MissingData(format!(
            "{} (bake it with `cargo run -p uni-bake -- bsc5 ...`, see data/manifest.toml)",
            sky_warnings.join("; ")
        )));
    }
    let bookmarks = uni_nav::load_bookmarks(&args.bookmarks)?;
    check_names(&bookmarks)?;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;

    let gpu = pollster::block_on(GpuContext::new_headless())?;
    let max = gpu.device.limits().max_texture_dimension_2d;
    if args.size.max_element() > max {
        return Err(
            anyhow::anyhow!("--size {} exceeds the GPU's {max} px limit", args.size).into(),
        );
    }
    let mut scene = Scene::new(&gpu, args.size, CAPTURE_FORMAT, &stars, CAPTURE_LENS_SCALE)?;
    drop(stars);
    let target = CaptureTarget::new(&gpu.device, args.size);
    let mut hud = args
        .hud
        .then(|| Hud::new_headless(&gpu, CAPTURE_FORMAT, HEADLESS_HUD_PIXELS_PER_POINT));
    let mut world = World::new(DEFAULT_EPOCH);
    world.time_rate = 0.0; // a capture is a still frame

    let mut frame_index = 0_u64;
    for b in &bookmarks {
        world.epoch = DEFAULT_EPOCH;
        let mut controls = default_controls(CAPTURE_LENS_SCALE);
        let view = apply_bookmark(b, &mut world, &mut controls)?;
        scene.set_lens_scale(controls.lens_scale);
        let size = scene.renderer.targets().size;

        let mut cpu_ms = 0.0;
        for f in 0..args.frames {
            let start = Instant::now();
            let ctx = frame_ctx(
                &world,
                &view,
                size,
                exposure(&controls),
                frame_index,
                CAPTURE_DT_S,
            );
            let mut cmds = vec![scene.render(&gpu, &ctx, &target.view)];
            if let Some(hud) = hud.as_mut().filter(|_| f + 1 == args.frames) {
                let perf = Perf {
                    fps: 0.0,
                    cpu_ms,
                    gpu_ms: scene.renderer.gpu_ms(),
                    gpu_timing_available: gpu.has_timestamps(),
                };
                let data = hud_data(&world, &ctx, perf, 0.0, scene.star_count, &[]);
                let mut enc = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("capture hud"),
                    });
                cmds.extend(hud.draw_headless(
                    &gpu,
                    &mut enc,
                    &target.view,
                    target.size.to_array(),
                    &data,
                    &mut controls,
                ));
                cmds.push(enc.finish());
            }
            cpu_ms = start.elapsed().as_secs_f32() * 1e3;
            gpu.queue.submit(cmds);
            // One frame at a time, so every frame's GPU timings are read back.
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("waiting for a capture frame")?;
            frame_index += 1;
        }
        let png = args.out.join(format!("{}.png", b.name));
        target.save_png(&gpu, &png)?;

        // GPU timings lag one render (uni-gfx PassTimer): render once more to collect the
        // captured frame's.
        let ctx = frame_ctx(
            &world,
            &view,
            size,
            exposure(&controls),
            frame_index,
            CAPTURE_DT_S,
        );
        gpu.queue
            .submit(std::iter::once(scene.render(&gpu, &ctx, &target.view)));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("waiting for the timing frame")?;
        frame_index += 1;

        let metrics = CaptureMetrics {
            name: b.name.clone(),
            width: target.size.x,
            height: target.size.y,
            gpu_ms: scene
                .renderer
                .gpu_ms()
                .into_iter()
                .map(|(pass, ms)| (pass.to_owned(), ms))
                .collect(),
            cpu_ms,
            star_count: scene.star_count,
            lens_scale: controls.lens_scale,
            epoch_utc: iso_utc(world.epoch),
        };
        let json_path = args.out.join(format!("{}.json", b.name));
        write_metrics(&json_path, &metrics)?;
        println!("{}", summary(&png, &metrics));
    }
    Ok(())
}

/// Bookmark names become file names: each must be a plain, unique file stem.
fn check_names(bookmarks: &[uni_nav::Bookmark]) -> anyhow::Result<()> {
    anyhow::ensure!(
        !bookmarks.is_empty(),
        "the bookmark file lists no bookmarks"
    );
    let mut seen = std::collections::HashSet::new();
    for b in bookmarks {
        anyhow::ensure!(
            is_file_stem(&b.name),
            "bookmark name {:?} is not a plain file name (use letters, digits, '-', '_', '.')",
            b.name
        );
        anyhow::ensure!(
            seen.insert(&b.name),
            "bookmark name {:?} appears twice",
            b.name
        );
    }
    Ok(())
}

/// Letters, digits, `-`, `_` and `.`, not starting with `.`: safe as `<out>/<name>.png`.
fn is_file_stem(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn write_metrics(path: &Path, metrics: &CaptureMetrics) -> anyhow::Result<()> {
    let text = serde_json::to_string_pretty(metrics).context("serialising capture metrics")?;
    std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))
}

/// One line per capture for the terminal.
fn summary(png: &Path, m: &CaptureMetrics) -> String {
    let gpu = if m.gpu_ms.is_empty() {
        "GPU ms n/a".to_owned()
    } else {
        m.gpu_ms
            .iter()
            .map(|(pass, ms)| format!("{pass} {ms:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
            + " GPU ms"
    };
    format!(
        "captured {} ({}×{}): {gpu}; CPU {:.2} ms; {} stars; lens scale {}",
        png.display(),
        m.width,
        m.height,
        m.cpu_ms,
        m.star_count,
        m.lens_scale
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_json_has_the_brief_fields() {
        let m = CaptureMetrics {
            name: "bh-shadow-30M".to_owned(),
            width: 1920,
            height: 1080,
            gpu_ms: BTreeMap::from([("lens".to_owned(), 3.5), ("stars".to_owned(), 0.25)]),
            cpu_ms: 1.5,
            star_count: 9096,
            lens_scale: 1.0,
            epoch_utc: "2026-11-20T00:00:00Z".to_owned(),
        };
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "cpu_ms",
                "epoch_utc",
                "gpu_ms",
                "height",
                "lens_scale",
                "name",
                "star_count",
                "width"
            ]
        );
        assert_eq!(v["gpu_ms"]["lens"], 3.5);
        assert_eq!(v["gpu_ms"]["stars"], 0.25);
        assert_eq!(v["width"], 1920);
        assert_eq!(v["epoch_utc"], "2026-11-20T00:00:00Z");
    }

    #[test]
    fn metrics_are_written_as_json() {
        let path = std::env::temp_dir().join(format!(
            "uni-app-{}-metrics_are_written_as_json.json",
            std::process::id()
        ));
        let m = CaptureMetrics {
            name: "x".to_owned(),
            width: 2,
            height: 1,
            gpu_ms: BTreeMap::new(),
            cpu_ms: 0.0,
            star_count: 0,
            lens_scale: 0.5,
            epoch_utc: "2000-01-01T11:58:55Z".to_owned(),
        };
        write_metrics(&path, &m).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["gpu_ms"], serde_json::json!({}));
        assert_eq!(v["lens_scale"], 0.5);
    }

    #[test]
    fn bookmark_names_must_be_unique_file_stems() {
        for ok in ["start", "bh-shadow-30M", "sky_orion.v2"] {
            assert!(is_file_stem(ok), "{ok}");
        }
        for bad in ["", ".hidden", "a/b", "..", "a b", "c:\\x", "naïve"] {
            assert!(!is_file_stem(bad), "{bad}");
        }
        let marks = uni_nav::load_bookmarks(&crate::world::m0_bookmarks_path()).unwrap();
        check_names(&marks).unwrap();
        let mut twice = marks.clone();
        twice.push(marks[0].clone());
        assert!(check_names(&twice).is_err());
        assert!(check_names(&[]).is_err());
    }

    #[test]
    fn summary_reports_missing_timestamps() {
        let m = CaptureMetrics {
            name: "x".to_owned(),
            width: 4,
            height: 3,
            gpu_ms: BTreeMap::new(),
            cpu_ms: 0.5,
            star_count: 7,
            lens_scale: 1.0,
            epoch_utc: String::new(),
        };
        let line = summary(Path::new("out/x.png"), &m);
        assert!(line.contains("GPU ms n/a"), "{line}");
        assert!(line.contains("4×3"), "{line}");
    }
}
