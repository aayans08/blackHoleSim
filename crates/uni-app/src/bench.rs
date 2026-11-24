//! Headless `--bench` v2: frame-time and GPU-time statistics per bookmark (M1 Task B9 Step 3).
//!
//! `universe --bench <bookmarks.ron> [--size WxH] [--frames N] [--sustained <s>] [--json <out>]`
//! renders each bookmark headless at the internal resolution `--size` (default 1920×1080, the
//! "1080p internal" of design §6 and the M1 gate) into an offscreen sRGB target of the same size,
//! so the tonemap does no upscale. Each bookmark is set up as the window opens it: the window's
//! lens scale (0.5) unless the bookmark names one, its epoch, view and exposure choices; time is
//! paused and auto-exposure adapts as in the window (no settle). [`WARMUP_FRAMES`] uncounted
//! frames come first (pipelines, the band's first cube build).
//!
//! - **Serial** (default): `N` frames, each built, encoded, submitted and waited for on its own;
//!   its GPU timings are read back before the next one starts. `frame_ms` is the serial cost
//!   (CPU + GPU + the submit/wait round trip), an upper bound on the window's frame time.
//! - **Sustained** (`--sustained s`): each bookmark renders continuously for `s` seconds with up
//!   to two frames in flight, as the window does (warm-up frames too); GPU timings are collected
//!   without blocking. Each frame carries a frame tag (`crate::tap::FrameTag`), so a timing
//!   record, which arrives frames later, is attached to the frame it came from (B9t; review B9
//!   r0 minor 9: records were attached to the oldest frame without one, so when the timer's ring
//!   dropped records the backlog grew and the last 10 s got none). A frame whose record the
//!   timer dropped has no GPU numbers; its interval and CPU time still count.
//!   `interval_ms` is the time between consecutive frame starts (the frame rate the window would
//!   reach), `frame_ms` a frame's latency from its start to the GPU finishing it. The M3 Air is
//!   fanless, so the first and the last 10 s are reported apart (plan "Performance budget").
//!
//! Every frame records: `cpu_ms` (snapshot, inputs and encode), `snapshot_cpu_ms`
//! (`World::snapshot`), `gpu_frame_ms` (the timer's frame brackets), and per pass the GPU
//! **duration** (`scene::pass_durations` of `scene::pass_times`; invalid ones stay `null`, never
//! 0, review I5). Each window also gives `pass_sum_ms`, the per-frame sum of the valid
//! durations, to set against `gpu_frame_ms` (B9t: the sum should be within 5 % of the frame).
//! `comets_cpu_ms` is the comet pass's whole `prepare` (C1's `cpu_ms`: culling, the tail work,
//! the uploads; budget ≤ 1.0 ms) and `asteroids_cpu_ms` the asteroid pass's (C2's `cpu_ms`: the
//! mean anomalies and their upload; budget ≤ 0.5 ms), per frame (Task C6). Each bookmark's
//! `target`, `highlight_belt` and `cruise` apply as in a capture.
//! Windows `all`, `first10s` and `last10s` are cut by the time since the bookmark's first measured
//! frame; a window with no frame is `null`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context as _;
use glam::UVec2;
use uni_gfx::{CAPTURE_FORMAT, CaptureTarget};

use crate::capture::{CaptureError, headless_gpu, load_world, ms_or_na, submit_and_wait};
use crate::scene::{
    INTERACTIVE_LENS_SCALE, Scene, StarCatalog, apply_bookmark, default_controls, pass_durations,
    pass_times,
};
use crate::tap::{FRAME_TAG_COUNT, frame_tag_of};

/// Default internal resolution of a bench run (design §6, "1080p internal").
pub const DEFAULT_SIZE: UVec2 = UVec2::new(1920, 1080);
/// Default measured frames per bookmark (serial mode).
pub const DEFAULT_FRAMES: u32 = 120;
/// Frames rendered per bookmark before measuring (pipeline warm-up, the band's cube build).
pub const WARMUP_FRAMES: u32 = 8;
/// Frames in flight in sustained mode (as the window: one being recorded, one on the GPU).
pub const FRAMES_IN_FLIGHT: usize = 2;
/// Length of the first and last windows of a sustained run [s] (plan "Performance budget").
pub const WINDOW_S: f64 = 10.0;
/// Frame time handed to the features [s] (a 60 Hz frame; time is paused).
const BENCH_DT_S: f64 = 1.0 / 60.0;

/// `universe --bench <bookmarks.ron> [--frames N] [--size WxH] [--sustained s] [--json f]`.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchArgs {
    pub bookmarks: PathBuf,
    pub size: UVec2,
    pub frames: u32,
    pub sustained_s: Option<f64>,
    pub json: Option<PathBuf>,
}

/// Mean, 95th percentile (nearest rank) and count of a set of samples.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
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

/// One measured frame.
#[derive(Clone, Debug, Default, PartialEq)]
struct Sample {
    /// Seconds since the bookmark's first measured frame started.
    t_s: f64,
    frame_ms: Option<f32>,
    interval_ms: Option<f32>,
    cpu_ms: f32,
    snapshot_cpu_ms: f32,
    comets_cpu_ms: f32,
    asteroids_cpu_ms: f32,
    /// Filled when the frame's GPU timings arrive.
    gpu_frame_ms: Option<f32>,
    passes: Option<Vec<(&'static str, Option<f32>)>>,
}

/// One pass's GPU durations in a window: the mean of the valid ones, how many were valid and in
/// how many frames it ran (B8's `{mean, valid, runs}`).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct PassStats {
    pub mean: Option<f32>,
    pub valid: usize,
    pub runs: usize,
}

/// One window's statistics (B9 Step 3's schema).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct WindowStats {
    pub frames: usize,
    pub frame_ms: Option<Stats>,
    pub interval_ms: Option<Stats>,
    pub gpu_frame_ms: Option<Stats>,
    pub cpu_ms: Option<Stats>,
    pub passes: BTreeMap<String, PassStats>,
    /// Per frame, the sum of its valid pass durations [ms] (B9t: against `gpu_frame_ms`).
    pub pass_sum_ms: Option<Stats>,
    pub snapshot_cpu_ms: Option<Stats>,
    pub comets_cpu_ms: Option<Stats>,
    pub asteroids_cpu_ms: Option<Stats>,
}

/// The statistics of `samples`.
fn window(samples: &[&Sample]) -> Option<WindowStats> {
    if samples.is_empty() {
        return None;
    }
    let col = |f: &dyn Fn(&Sample) -> Option<f32>| -> Vec<f32> {
        samples.iter().filter_map(|s| f(s)).collect()
    };
    let mut passes: BTreeMap<String, (Vec<f32>, usize)> = BTreeMap::new();
    for s in samples {
        for &(name, ms) in s.passes.iter().flatten() {
            let e = passes.entry(name.to_owned()).or_default();
            e.1 += 1;
            e.0.extend(ms);
        }
    }
    Some(WindowStats {
        frames: samples.len(),
        frame_ms: stats(&col(&|s| s.frame_ms)),
        interval_ms: stats(&col(&|s| s.interval_ms)),
        gpu_frame_ms: stats(&col(&|s| s.gpu_frame_ms)),
        cpu_ms: stats(&col(&|s| Some(s.cpu_ms))),
        passes: passes
            .into_iter()
            .map(|(k, (v, runs))| {
                let mean = stats(&v).map(|s| s.mean);
                (
                    k,
                    PassStats {
                        mean,
                        valid: v.len(),
                        runs,
                    },
                )
            })
            .collect(),
        pass_sum_ms: stats(&col(&|s| {
            s.passes
                .as_ref()
                .map(|p| p.iter().filter_map(|&(_, ms)| ms).sum())
        })),
        snapshot_cpu_ms: stats(&col(&|s| Some(s.snapshot_cpu_ms))),
        comets_cpu_ms: stats(&col(&|s| Some(s.comets_cpu_ms))),
        asteroids_cpu_ms: stats(&col(&|s| Some(s.asteroids_cpu_ms))),
    })
}

/// A timing record's frame tag (`crate::tap::FrameTag`) and its passes without the tag.
fn split_tag(
    passes: &[(&'static str, Option<f32>)],
) -> (Option<usize>, Vec<(&'static str, Option<f32>)>) {
    let tag = passes.iter().find_map(|(l, _)| frame_tag_of(l));
    let rest = passes
        .iter()
        .filter(|(l, _)| frame_tag_of(l).is_none())
        .copied()
        .collect();
    (tag, rest)
}

/// Sustained mode: attaches a timing record (its `pass_times` and frame time) to the frame its
/// tag names (`tag_sample[tag]`, the sample index of the last measured frame rendered with that
/// tag), unless that frame has one already. Returns the sample it went to.
fn attach_by_tag(
    samples: &mut [Sample],
    tag_sample: &[Option<usize>; FRAME_TAG_COUNT],
    times: &[(&'static str, Option<f32>)],
    frame_ms: Option<f32>,
) -> Option<usize> {
    let (tag, passes) = split_tag(times);
    let i = tag_sample[tag?]?;
    let s = samples.get_mut(i).filter(|s| s.passes.is_none())?;
    s.gpu_frame_ms = frame_ms;
    s.passes = Some(pass_durations(&passes));
    Some(i)
}

/// One bookmark's result.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BookmarkResult {
    pub name: String,
    pub lens_scale: f32,
    /// `all`, `first10s`, `last10s`.
    pub windows: BTreeMap<&'static str, Option<WindowStats>>,
}

/// The bench JSON (B9 Step 3).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BenchFile {
    pub size: [u32; 2],
    pub frames: u32,
    pub sustained_s: Option<f64>,
    pub bookmarks: Vec<BookmarkResult>,
}

/// The windows of one bookmark's samples.
fn windows(samples: &[Sample]) -> BTreeMap<&'static str, Option<WindowStats>> {
    let end = samples.last().map_or(0.0, |s| s.t_s);
    let all: Vec<&Sample> = samples.iter().collect();
    let first: Vec<&Sample> = samples.iter().filter(|s| s.t_s < WINDOW_S).collect();
    let last: Vec<&Sample> = samples.iter().filter(|s| s.t_s >= end - WINDOW_S).collect();
    BTreeMap::from([
        ("all", window(&all)),
        ("first10s", window(&first)),
        ("last10s", window(&last)),
    ])
}

/// Two report lines for a bookmark's `all` window.
fn report(r: &BookmarkResult, size: UVec2) -> String {
    let Some(Some(w)) = r.windows.get("all") else {
        return format!("bench {}: no frames", r.name);
    };
    let pick = |s: Option<Stats>| (s.map(|s| s.mean), s.map(|s| s.p95));
    let (fm, fp) = pick(w.frame_ms);
    let (im, ip) = pick(w.interval_ms);
    let (gm, gp) = pick(w.gpu_frame_ms);
    let (cm, cp) = pick(w.cpu_ms);
    let (sm, _) = pick(w.snapshot_cpu_ms);
    let (cm_c, _) = pick(w.comets_cpu_ms);
    let (am, _) = pick(w.asteroids_cpu_ms);
    let (psm, _) = pick(w.pass_sum_ms);
    let share = match (psm, gm) {
        (Some(p), Some(g)) if g > 0.0 => format!(" ({:.0} % of the GPU frame)", 100.0 * p / g),
        _ => String::new(),
    };
    let fps = w
        .interval_ms
        .or(w.frame_ms)
        .map_or_else(|| "n/a".to_owned(), |s| format!("{:.1}", 1e3 / s.mean));
    let passes = if w.passes.is_empty() {
        "n/a".to_owned()
    } else {
        w.passes
            .iter()
            .map(|(k, p)| {
                let partial = if p.valid > 0 && p.valid < p.runs {
                    format!(" ({}/{} valid)", p.valid, p.runs)
                } else {
                    String::new()
                };
                format!("{k} {}{partial}", ms_or_na(p.mean))
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "bench {} ({}×{}, {} frames, lens scale {}): frame {} ms mean, {} ms p95; interval {} \
         ms mean, {} ms p95 ({fps} fps); GPU frame {} ms mean, {} ms p95; CPU {} ms mean, {} ms \
         p95 (snapshot {} ms, comets {} ms, asteroids {} ms)\n  GPU passes (mean duration ms): \
         {passes}; sum {} ms{share}",
        r.name,
        size.x,
        size.y,
        w.frames,
        r.lens_scale,
        ms_or_na(fm),
        ms_or_na(fp),
        ms_or_na(im),
        ms_or_na(ip),
        ms_or_na(gm),
        ms_or_na(gp),
        ms_or_na(cm),
        ms_or_na(cp),
        ms_or_na(sm),
        ms_or_na(cm_c),
        ms_or_na(am),
        ms_or_na(psm),
    )
}

/// Benchmarks every bookmark in `args.bookmarks`: two lines per bookmark on stdout, and the
/// JSON with `--json`.
pub fn run(args: &BenchArgs) -> Result<(), CaptureError> {
    let bookmarks = uni_nav::load_bookmarks(&args.bookmarks)?;
    if bookmarks.is_empty() {
        return Err(anyhow::anyhow!("the bookmark file lists no bookmarks").into());
    }
    let (mut world, assets) = load_world()?;
    let gpu = headless_gpu(args.size, "--size")?;
    let mut scene = Scene::new(
        &gpu,
        args.size,
        CAPTURE_FORMAT,
        StarCatalog::AtHyg(world.sim.star_data()),
        &assets,
        INTERACTIVE_LENS_SCALE,
        world.sim.asteroid_records(),
    )?;
    let target = CaptureTarget::new(&gpu.device, args.size);
    if !gpu.has_timestamps() {
        println!("bench: the device has no timestamp queries; GPU times are n/a");
    }
    let mut results = Vec::new();
    let mut frame_index = 0_u64;
    for b in &bookmarks {
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        crate::capture::settle_at_bookmark_epoch(b, &mut world)?;
        let applied = apply_bookmark(b, &mut world, &mut controls)?;
        world.set_time_rate(0.0);
        let ui = crate::capture::bookmark_ui(b, &mut world)?;
        scene.set_lens_scale(controls.lens_scale);
        let mut samples: Vec<Sample> = Vec::new();
        // Submitted frames not yet finished: (sample index for a measured frame, submission,
        // start).
        let mut in_flight: std::collections::VecDeque<(
            Option<usize>,
            wgpu::SubmissionIndex,
            Instant,
        )> = std::collections::VecDeque::new();
        // The measured frame each frame tag was last rendered with (sustained mode).
        let mut tag_sample: [Option<usize>; FRAME_TAG_COUNT] = [None; FRAME_TAG_COUNT];
        let sustained = args.sustained_s.is_some();
        let mut last_seq = scene.renderer.gpu_timings().map(|t| t.seq);
        let mut bench_start: Option<Instant> = None;
        let mut prev_start: Option<Instant> = None;
        let mut f = 0_u32;
        loop {
            let measuring = f >= WARMUP_FRAMES;
            if measuring {
                match (args.sustained_s, bench_start) {
                    (Some(s), Some(t0)) if t0.elapsed().as_secs_f64() >= s => break,
                    (None, _) if f >= WARMUP_FRAMES + args.frames => break,
                    _ => {}
                }
            }
            let start = Instant::now();
            let tag = (frame_index % FRAME_TAG_COUNT as u64) as usize;
            scene.frame_tag.set(sustained.then_some(tag));
            world.advance(BENCH_DT_S);
            let frame = scene.prepare(
                &world,
                &applied.view,
                args.size,
                args.size,
                &controls,
                applied.choice,
                ui,
                false,
                frame_index,
                BENCH_DT_S,
            );
            let cmd = scene.render(&gpu, &frame, &target.view);
            let cpu_ms = start.elapsed().as_secs_f32() * 1e3;
            let (comets_cpu_ms, asteroids_cpu_ms) =
                (scene.comets_cpu_ms(), scene.asteroids_cpu_ms());
            frame_index += 1;
            f += 1;
            let sample_index = measuring.then(|| {
                let t0 = *bench_start.get_or_insert(start);
                samples.push(Sample {
                    t_s: start.duration_since(t0).as_secs_f64(),
                    interval_ms: prev_start.map(|p| start.duration_since(p).as_secs_f32() * 1e3),
                    cpu_ms,
                    snapshot_cpu_ms: frame.snapshot_cpu_ms,
                    comets_cpu_ms,
                    asteroids_cpu_ms,
                    ..Sample::default()
                });
                samples.len() - 1
            });
            tag_sample[tag] = sample_index;
            prev_start = measuring.then_some(start);
            if args.sustained_s.is_none() {
                // Serial: wait for this frame and read its timings back before the next.
                submit_and_wait(&gpu, vec![cmd], &format!("bench frame of {:?}", b.name))
                    .context("bench")?;
                let done = start.elapsed().as_secs_f32() * 1e3;
                scene.renderer.poll_timings(&gpu);
                let t = scene.renderer.gpu_timings();
                if let (Some(i), Some(t)) = (sample_index, t.filter(|t| Some(t.seq) != last_seq)) {
                    samples[i].frame_ms = Some(done);
                    samples[i].gpu_frame_ms = t.frame_ms;
                    samples[i].passes = Some(pass_durations(&split_tag(pass_times(t)).1));
                } else if let Some(i) = sample_index {
                    samples[i].frame_ms = Some(done);
                }
                last_seq = scene.renderer.gpu_timings().map(|t| t.seq);
                continue;
            }
            // Sustained: keep at most FRAMES_IN_FLIGHT frames on the GPU, warm-up frames too
            // (review B9 r0 minor 9).
            let idx = gpu.queue.submit(std::iter::once(cmd));
            in_flight.push_back((sample_index, idx, start));
            while in_flight.len() >= FRAMES_IN_FLIGHT {
                let Some((i, idx, s)) = in_flight.pop_front() else {
                    break;
                };
                gpu.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(idx),
                        timeout: None,
                    })
                    .context("waiting for a bench frame")?;
                if let Some(i) = i {
                    samples[i].frame_ms = Some(s.elapsed().as_secs_f32() * 1e3);
                }
            }
            let errors = gpu.take_errors();
            if !errors.is_empty() {
                return Err(anyhow::anyhow!("bench {:?}: {}", b.name, errors.join("; ")).into());
            }
            // Timings arrive a few frames late (read without blocking in `render`): attach each
            // new record to the frame its tag names.
            if let Some(t) = scene.renderer.gpu_timings()
                && Some(t.seq) != last_seq
            {
                last_seq = Some(t.seq);
                attach_by_tag(&mut samples, &tag_sample, pass_times(t), t.frame_ms);
            }
        }
        // Drain what is still in flight, and collect the newest record.
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("draining the bench frames")?;
        for (i, _, s) in in_flight.drain(..) {
            if let Some(i) = i {
                samples[i].frame_ms = Some(s.elapsed().as_secs_f32() * 1e3);
            }
        }
        if sustained {
            scene.renderer.poll_timings(&gpu);
            if let Some(t) = scene.renderer.gpu_timings()
                && Some(t.seq) != last_seq
            {
                attach_by_tag(&mut samples, &tag_sample, pass_times(t), t.frame_ms);
            }
        }
        scene.frame_tag.set(None);
        let result = BookmarkResult {
            name: b.name.clone(),
            lens_scale: scene.lens_scale(),
            windows: windows(&samples),
        };
        println!("{}", report(&result, args.size));
        results.push(result);
    }
    if let Some(path) = &args.json {
        let file = BenchFile {
            size: args.size.to_array(),
            frames: args.frames,
            sustained_s: args.sustained_s,
            bookmarks: results,
        };
        let text = serde_json::to_string_pretty(&file).context("serialising the bench JSON")?;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
        println!("bench: wrote {}", path.display());
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
        let v: Vec<f32> = (1..=100).rev().map(|i| i as f32).collect();
        let s = stats(&v).unwrap();
        assert!((s.mean - 50.5).abs() < 1e-5);
        assert_eq!(s.p95, 95.0);
        let v: Vec<f32> = (1..=20).map(|i| i as f32).collect();
        assert_eq!(stats(&v).unwrap().p95, 19.0);
        let s = stats(&[1.0, f32::INFINITY, 3.0]).unwrap();
        assert_eq!((s.mean, s.count), (2.0, 2));
    }

    fn sample(t_s: f64, ms: f32, passes: Option<Vec<(&'static str, Option<f32>)>>) -> Sample {
        Sample {
            t_s,
            frame_ms: Some(ms),
            interval_ms: (t_s > 0.0).then_some(ms),
            cpu_ms: 1.0,
            snapshot_cpu_ms: 0.5,
            comets_cpu_ms: 0.25,
            asteroids_cpu_ms: 0.125,
            gpu_frame_ms: passes.as_ref().map(|_| ms - 1.0),
            passes,
        }
    }

    /// The windows split by time; a pass with no valid sample has `mean: null` and its counts
    /// (review I5: never 0).
    #[test]
    fn windows_split_by_time_and_keep_na() {
        let samples: Vec<Sample> = (0..300)
            .map(|i| {
                let t = f64::from(i) * 0.1;
                let ms = if t < 10.0 { 10.0 } else { 20.0 };
                let passes = (i % 2 == 0)
                    .then(|| vec![("stars", Some(1.0)), ("lens", None), ("bloom", Some(0.5))]);
                sample(t, ms, passes)
            })
            .collect();
        let w = windows(&samples);
        let all = w["all"].as_ref().unwrap();
        let first = w["first10s"].as_ref().unwrap();
        let last = w["last10s"].as_ref().unwrap();
        assert_eq!(all.frames, 300);
        assert_eq!(first.frames, 100);
        assert_eq!(last.frames, 101); // t ≥ 29.9 − 10
        assert_eq!(first.frame_ms.unwrap().p95, 10.0);
        assert_eq!(last.frame_ms.unwrap().mean, 20.0);
        assert_eq!(
            all.passes["lens"],
            PassStats {
                mean: None,
                valid: 0,
                runs: 150
            }
        );
        assert_eq!(all.passes["stars"].mean, Some(1.0));
        assert_eq!(all.passes["stars"].valid, 150);
        assert_eq!(all.comets_cpu_ms.unwrap().mean, 0.25);
        assert_eq!(all.asteroids_cpu_ms.unwrap().mean, 0.125);
        let v = serde_json::to_value(all).unwrap();
        assert_eq!(v["passes"]["lens"]["mean"], serde_json::Value::Null);
        assert_eq!(v["comets_cpu_ms"]["mean"], 0.25);
        assert_eq!(v["interval_ms"]["count"], 299);
        assert!(windows(&[])["all"].is_none());
    }

    /// B9t (d), review B9 r0 minor 9: sustained records attach to the frame their tag names.
    /// 600 measured frames (10 s at 60 fps, then 10 s more), the timer's ring keeping only every
    /// other record and each arriving 3 frames late: every record lands on its own frame, the
    /// last 10 s get half their frames timed, and no frame gets another frame's numbers. The
    /// old rule (the oldest frame without a record) piles every record onto the first frames
    /// and leaves the last 10 s with none, the "no valid GPU timing" B9 and B9p saw.
    #[test]
    fn sustained_records_attach_to_their_own_frames() {
        let n = 1_200;
        let mut samples: Vec<Sample> = (0..n)
            .map(|i| sample(f64::from(i) / 60.0, 16.0, None))
            .collect();
        let mut old_rule = samples.clone();
        let mut tag_sample = [None; FRAME_TAG_COUNT];
        // A record: (pass times, frame time), both naming the frame by value.
        let record = |frame: u32| {
            let ms = f32::from(u16::try_from(frame).unwrap());
            let times = vec![
                (
                    crate::tap::FRAME_TAGS[frame as usize % FRAME_TAG_COUNT],
                    None,
                ),
                ("bloom", Some(ms)),
            ];
            (times, Some(ms))
        };
        for frame in 0..n {
            tag_sample[frame as usize % FRAME_TAG_COUNT] = Some(frame as usize);
            // Frame `frame − 3`'s record arrives now, if the ring kept it (even frames).
            if frame >= 3 && (frame - 3) % 2 == 0 {
                let (times, ms) = record(frame - 3);
                assert_eq!(
                    attach_by_tag(&mut samples, &tag_sample, &times, ms),
                    Some((frame - 3) as usize)
                );
                if let Some(s) = old_rule.iter_mut().find(|s| s.passes.is_none()) {
                    s.gpu_frame_ms = ms;
                    s.passes = Some(pass_durations(&times));
                }
            }
        }
        for (i, s) in samples.iter().enumerate() {
            match s.gpu_frame_ms {
                Some(ms) => assert_eq!(ms, i as f32, "frame {i} got another frame's record"),
                None => assert!(i % 2 == 1 || i + 3 >= n as usize, "frame {i}"),
            }
            if let Some(p) = &s.passes {
                assert!(
                    p.iter().all(|(l, _)| frame_tag_of(l).is_none()),
                    "tag left in"
                );
            }
        }
        let w = windows(&samples);
        let last = w["last10s"].as_ref().unwrap();
        assert!(
            last.gpu_frame_ms.unwrap().count >= 290,
            "{:?}",
            last.gpu_frame_ms
        );
        let old = windows(&old_rule);
        assert!(old["last10s"].as_ref().unwrap().gpu_frame_ms.is_none());
        // A record whose frame has one already, or a tag no frame carries, attaches nowhere.
        let (times, ms) = record(n - 4);
        assert_eq!(attach_by_tag(&mut samples, &tag_sample, &times, ms), None);
        let untagged = [("bloom", Some(1.0))];
        assert_eq!(
            attach_by_tag(&mut samples, &[None; FRAME_TAG_COUNT], &untagged, Some(1.0)),
            None
        );
    }

    /// The JSON is B8's reader schema: `{size, frames, sustained_s, bookmarks: [{name,
    /// lens_scale, windows: {all|first10s|last10s: {...}}}]}`.
    #[test]
    fn bench_json_has_the_schema_b8_reads() {
        let samples = vec![sample(0.0, 12.0, Some(vec![("stars", Some(1.0))]))];
        let file = BenchFile {
            size: [1920, 1080],
            frames: 120,
            sustained_s: Some(60.0),
            bookmarks: vec![BookmarkResult {
                name: "start".to_owned(),
                lens_scale: 0.5,
                windows: windows(&samples),
            }],
        };
        let v = serde_json::to_value(&file).unwrap();
        assert_eq!(v["size"], serde_json::json!([1920, 1080]));
        assert_eq!(v["sustained_s"], 60.0);
        let w = &v["bookmarks"][0]["windows"];
        for key in ["all", "first10s", "last10s"] {
            for field in [
                "frame_ms",
                "interval_ms",
                "gpu_frame_ms",
                "cpu_ms",
                "passes",
                "snapshot_cpu_ms",
                "comets_cpu_ms",
                "asteroids_cpu_ms",
            ] {
                assert!(w[key].get(field).is_some(), "{key}.{field}");
            }
        }
        assert_eq!(w["all"]["gpu_frame_ms"]["p95"], 11.0);
        assert_eq!(w["all"]["passes"]["stars"]["runs"], 1);
        let line = report(&file.bookmarks[0], UVec2::new(1920, 1080));
        assert!(
            line.contains("GPU passes (mean duration ms): stars 1.00"),
            "{line}"
        );
    }
}
