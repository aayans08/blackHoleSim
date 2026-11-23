//! Headless --bench mode: frame-time and GPU-time statistics per bookmark (review I6, ledger
//! T13 "window FPS not measured").
//!
//! `universe --bench <bookmarks.ron> [--frames N] [--size WxH]` renders each bookmark headless
//! at the internal resolution `--size` (default 1920×1080, the "1080p internal" of design §6
//! and the M1 gate) into an offscreen sRGB target of the same size, so the tonemap does no
//! upscale. Each bookmark is set up as the window would open at it: the window's lens scale
//! (0.5) unless the bookmark names one, and the bookmark's epoch and exposures.
//!
//! Per bookmark, [`WARMUP_FRAMES`] uncounted frames come first, then `N` measured frames. Each
//! measured frame is built, encoded, submitted and waited for on its own, and its GPU timings
//! are read back before the next one starts. Reported, per bookmark:
//! - **frame**: wall-clock time from building the frame to the GPU finishing it, mean and p95
//!   (nearest rank). Frames do not overlap here, so this is the serial cost CPU + GPU + the
//!   submit/wait round trip: an upper bound on the window's frame time, where CPU and GPU work
//!   overlap. The fps shown is 1000 / mean frame ms.
//! - **GPU frame**: the timer's whole-frame GPU time (uni-gfx `PassTimer` frame brackets),
//!   mean and p95, or n/a if the device returned no valid measurement.
//! - **CPU**: the time to build and encode the frame, mean.
//! - **GPU passes**: the mean of each pass's valid measurements, "n/a" when it has none, and
//!   the count of valid ones when some were invalid (review I5: never shown as 0).
//!
//! With `--sky v1` the scene is the sky v1 stack (`Scene::new_v1`), with auto-exposure settling
//! each frame as in captures; a header line says so. Without the flag the run is as before.
//!
//! Not measured: present and vsync (there is no window), the HUD (the window draws egui on
//! top) and thermal throttling over minutes (the M3 Air is fanless; a short bench runs cool).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context as _;
use glam::UVec2;
use uni_core::Epoch;
use uni_gfx::{CAPTURE_FORMAT, CaptureTarget};

use crate::capture::{CaptureError, headless_gpu, load_catalog, ms_or_na, submit_and_wait};
use crate::scene::{
    INTERACTIVE_LENS_SCALE, Scene, SkyExposure, SkyMode, apply_bookmark, default_controls,
    frame_ctx,
};
use crate::world::World;

/// Default internal resolution of a bench run (design §6, "1080p internal").
pub const DEFAULT_SIZE: UVec2 = UVec2::new(1920, 1080);
/// Default measured frames per bookmark.
pub const DEFAULT_FRAMES: u32 = 120;
/// Frames rendered per bookmark before measuring (pipeline warm-up, resource creation).
pub const WARMUP_FRAMES: u32 = 8;
/// Epoch of a bookmark that sets none, as in captures.
const DEFAULT_EPOCH: Epoch = Epoch::J2000;
/// Frame time handed to the features [s] (a 60 Hz frame; nothing moves).
const BENCH_DT_S: f64 = 1.0 / 60.0;

/// `universe --bench <bookmarks.ron> [--frames N] [--size WxH]`.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchArgs {
    pub bookmarks: PathBuf,
    pub size: UVec2,
    pub frames: u32,
    /// `--sky`: the M0 stack (default) or sky v1.
    pub sky: SkyMode,
}

/// Mean and 95th percentile (nearest rank) of a set of samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub mean: f32,
    pub p95: f32,
    pub count: usize,
}

/// [`Stats`] of `samples`, or `None` when there are none. Non-finite samples are ignored.
pub fn stats(samples: &[f32]) -> Option<Stats> {
    let mut v: Vec<f32> = samples.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f32::total_cmp);
    let n = v.len();
    let mean = (v.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64) as f32;
    // Nearest rank: the smallest sample with at least 95 % of the samples at or below it.
    let rank = (0.95 * n as f64).ceil() as usize;
    Some(Stats {
        mean,
        p95: v[rank.clamp(1, n) - 1],
        count: n,
    })
}

/// One pass's GPU times across the measured frames.
#[derive(Clone, Debug, Default, PartialEq)]
struct PassSamples {
    /// Valid measurements [ms].
    valid: Vec<f32>,
    /// Frames in which the pass ran (valid or not).
    runs: usize,
}

/// Per-pass GPU samples in the order passes first appeared.
#[derive(Clone, Debug, Default, PartialEq)]
struct PassTable(Vec<(&'static str, PassSamples)>);

impl PassTable {
    fn add(&mut self, frame: &[(&'static str, Option<f32>)]) {
        for &(label, ms) in frame {
            let i = match self.0.iter().position(|(l, _)| *l == label) {
                Some(i) => i,
                None => {
                    self.0.push((label, PassSamples::default()));
                    self.0.len() - 1
                }
            };
            let s = &mut self.0[i].1;
            s.runs += 1;
            s.valid.extend(ms);
        }
    }

    /// `"<pass> <mean ms>"` per pass, "n/a" with no valid sample, and "(k/n valid)" when some
    /// were invalid.
    fn line(&self) -> String {
        if self.0.is_empty() {
            return "n/a".to_owned();
        }
        self.0
            .iter()
            .map(|(label, s)| {
                let mean = stats(&s.valid).map(|st| st.mean);
                let partial = if !s.valid.is_empty() && s.valid.len() < s.runs {
                    format!(" ({}/{} valid)", s.valid.len(), s.runs)
                } else {
                    String::new()
                };
                format!("{label} {}{partial}", ms_or_na(mean))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// One bookmark's results.
struct BookmarkBench {
    name: String,
    lens_scale: f32,
    frame_ms: Vec<f32>,
    cpu_ms: Vec<f32>,
    gpu_frame_ms: Vec<f32>,
    passes: PassTable,
}

impl BookmarkBench {
    /// The two report lines.
    fn report(&self, size: UVec2) -> String {
        let frame = stats(&self.frame_ms);
        let fps = frame.map_or_else(|| "n/a".to_owned(), |s| format!("{:.1}", 1e3 / s.mean));
        let gpu = stats(&self.gpu_frame_ms);
        let cpu = stats(&self.cpu_ms);
        format!(
            "bench {} ({}×{}, {} frames, lens scale {}): frame {} ms mean, {} ms p95 ({fps} fps); \
             GPU frame {} ms mean, {} ms p95{}; CPU {} ms mean\n  GPU passes (mean ms): {}",
            self.name,
            size.x,
            size.y,
            self.frame_ms.len(),
            self.lens_scale,
            ms_or_na(frame.map(|s| s.mean)),
            ms_or_na(frame.map(|s| s.p95)),
            ms_or_na(gpu.map(|s| s.mean)),
            ms_or_na(gpu.map(|s| s.p95)),
            match gpu {
                Some(s) if s.count < self.frame_ms.len() => {
                    format!(" ({}/{} valid)", s.count, self.frame_ms.len())
                }
                _ => String::new(),
            },
            ms_or_na(cpu.map(|s| s.mean)),
            self.passes.line(),
        )
    }
}

/// Benchmarks every bookmark in `args.bookmarks`, printing two lines per bookmark.
pub fn run(args: &BenchArgs) -> Result<(), CaptureError> {
    let catalog = load_catalog(args.sky)?;
    let bookmarks = uni_nav::load_bookmarks(&args.bookmarks)?;
    if bookmarks.is_empty() {
        return Err(anyhow::anyhow!("the bookmark file lists no bookmarks").into());
    }
    let gpu = headless_gpu(args.size, "--size")?;
    let mut scene = Scene::for_sky(
        &gpu,
        args.size,
        CAPTURE_FORMAT,
        args.sky,
        &catalog,
        INTERACTIVE_LENS_SCALE,
    )?;
    drop(catalog);
    if args.sky == SkyMode::V1 {
        println!(
            "bench: sky v1 (band, {} AT-HYG stars in 3-D, auto-exposure, bloom)",
            scene.star_count
        );
    }
    let target = CaptureTarget::new(&gpu.device, args.size);
    let mut world = World::new(DEFAULT_EPOCH);
    world.time_rate = 0.0;
    if !gpu.has_timestamps() {
        println!("bench: the device has no timestamp queries; GPU times are n/a");
    }

    let mut frame_index = 0_u64;
    for b in &bookmarks {
        world.epoch = DEFAULT_EPOCH;
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        let view = apply_bookmark(b, &mut world, &mut controls)?;
        let sky_exposure = SkyExposure::from_bookmark(b);
        scene.set_lens_scale(controls.lens_scale);
        let mut result = BookmarkBench {
            name: b.name.clone(),
            lens_scale: scene.lens_scale(),
            frame_ms: Vec::new(),
            cpu_ms: Vec::new(),
            gpu_frame_ms: Vec::new(),
            passes: PassTable::default(),
        };
        for f in 0..WARMUP_FRAMES + args.frames {
            let start = Instant::now();
            let fe = scene.set_exposure(&controls, sky_exposure, true);
            let ctx = frame_ctx(
                &world,
                &view,
                args.size,
                args.size,
                fe.exposure,
                frame_index,
                BENCH_DT_S,
            );
            let cmds = vec![scene.render(&gpu, &ctx, &target.view)];
            let encoded = start.elapsed();
            // `render` collected every earlier frame's timings, so newer ones are this frame's.
            let before = scene.renderer.gpu_timings().map(|t| t.seq);
            submit_and_wait(&gpu, cmds, &format!("bench frame of {:?}", b.name))
                .context("bench")?;
            let done = start.elapsed();
            frame_index += 1;
            // The wait finished this frame, so its timings are the latest after this.
            scene.renderer.poll_timings(&gpu);
            if f < WARMUP_FRAMES {
                continue;
            }
            result.frame_ms.push(done.as_secs_f32() * 1e3);
            result.cpu_ms.push(encoded.as_secs_f32() * 1e3);
            // Only this frame's timings: a frame whose read-back was dropped adds nothing.
            if let Some(t) = scene
                .renderer
                .gpu_timings()
                .filter(|t| Some(t.seq) != before)
            {
                result.gpu_frame_ms.extend(t.frame_ms);
                result.passes.add(&t.passes);
            }
        }
        println!("{}", result.report(args.size));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_mean_and_nearest_rank_p95() {
        assert_eq!(stats(&[]), None);
        assert_eq!(stats(&[f32::NAN]), None);
        let s = stats(&[3.0]).unwrap();
        assert_eq!((s.mean, s.p95, s.count), (3.0, 3.0, 1));
        // 1..=100: mean 50.5, p95 = the 95th smallest = 95.
        let v: Vec<f32> = (1..=100).rev().map(|i| i as f32).collect();
        let s = stats(&v).unwrap();
        assert!((s.mean - 50.5).abs() < 1e-5);
        assert_eq!(s.p95, 95.0);
        // 20 samples: rank ceil(19) = 19.
        let v: Vec<f32> = (1..=20).map(|i| i as f32).collect();
        assert_eq!(stats(&v).unwrap().p95, 19.0);
        // Non-finite samples are ignored.
        let s = stats(&[1.0, f32::INFINITY, 3.0]).unwrap();
        assert_eq!((s.mean, s.count), (2.0, 2));
    }

    /// Review I5: a pass with no valid measurement reads "n/a", and a partly valid one says how
    /// many were valid; nothing invalid becomes 0.
    #[test]
    fn pass_table_reports_na_and_partial_validity() {
        let mut t = PassTable::default();
        assert_eq!(t.line(), "n/a");
        t.add(&[
            ("lens", Some(10.0)),
            ("tonemap", None),
            ("stars", Some(1.0)),
        ]);
        t.add(&[("lens", Some(20.0)), ("tonemap", None), ("stars", None)]);
        assert_eq!(t.line(), "lens 15.00, tonemap n/a, stars 1.00 (1/2 valid)");
        assert!(!t.line().contains("0.00 "));
    }

    #[test]
    fn report_shows_frame_gpu_and_pass_lines() {
        let mut passes = PassTable::default();
        passes.add(&[("lens", Some(12.0)), ("tonemap", None)]);
        let r = BookmarkBench {
            name: "start".to_owned(),
            lens_scale: 0.5,
            frame_ms: vec![10.0, 20.0],
            cpu_ms: vec![0.5, 0.5],
            gpu_frame_ms: vec![9.0],
            passes,
        };
        let text = r.report(UVec2::new(1920, 1080));
        assert!(
            text.starts_with(
                "bench start (1920×1080, 2 frames, lens scale 0.5): frame 15.00 ms mean, \
                 20.00 ms p95 (66.7 fps); GPU frame 9.00 ms mean, 9.00 ms p95 (1/2 valid); \
                 CPU 0.50 ms mean"
            ),
            "{text}"
        );
        assert!(
            text.ends_with("GPU passes (mean ms): lens 12.00, tonemap n/a"),
            "{text}"
        );
    }
}
