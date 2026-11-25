//! Headless `--capture`: renders bookmarks to PNG plus the capture JSON v2 (M1 Task B9 Step 2),
//! and with `--hdr` the pre-bloom HDR sidecar.
//!
//! Per bookmark: the world goes to its epoch, every comet is settled there (A4 ruling: captures
//! settle per bookmark, so they do not depend on the frames before), the camera is placed (its
//! anchor and `look_at` resolved, A6), the bookmark's `target` is selected, its
//! `highlight_belt` and `cruise` set the belt highlight and the flight label (Task C6), a
//! snapshot notes any unnamed near star and a zero tick registers it (A4 parked item; the frames
//! then draw it with its id). The comet pass finishes every due tail within the frame (C1's
//! settle mode), and is rebuilt whenever a bookmark changes the epoch, so no tail survives from
//! another date (C1 review m2).
//! - **`--hud`**: the HUD with the overlay (markers, the target's bracket, orbits) is drawn on
//!   the captured frame, with the card open when the bookmark has a target and the notes list
//!   always open, so every label is readable (lead ruling on the C0 context, owner point 4);
//!   `hud.card_rows` lists the card's row labels in order.
//! - **Auto exposure** (review I16, round 2): frames render with `settle` until the resolved
//!   scene and star EV each move less than [`SETTLE_EV`] between frames (at least
//!   `--frames` − 2 frames, at most [`MAX_SETTLE_FRAMES`]), then [`MANUAL_FRAMES`] frames in
//!   Manual at the settled EVs. The last is captured, so the written exposure is exactly the
//!   captured frame's.
//! - **Manual** (the bookmark's, or the debug grid's): `--frames` frames; the last is captured.
//!
//! The internal render size equals `--size`, so the tonemap reads `hdr` 1:1. GPU times are per
//! pass **durations** (`scene::pass_durations`; `null` = no valid measurement, never 0; review
//! I5). Any uncaptured GPU error fails the run (review m6). `<name>.json` holds the M0 keys plus
//! `ev100`, `honesty`, `projected`, `metrics`, `regions`, `samples`, `features`, `flux`, `limbs`,
//! `notes`, `settled`, and with `--hdr` the `hdr` header of `<name>.hdr.bin` (f32 LE RGBA, rows
//! top-down, the pre-exposed HDR target copied just before bloom); `hud` with `--hud`. The HDR
//! taps' copies are timed passes of their own, `tap-world` and `tap-prebloom` in `gpu_ms`
//! (`crate::tap`); `gpu_frame_ms` includes them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context as _;
use glam::UVec2;
use uni_core::Epoch;
use uni_gfx::{CAPTURE_FORMAT, CaptureTarget, GpuContext};
use uni_post::ExposureMode;
use uni_ui::Hud;

use crate::measure::{self, Measurements};
use crate::scene::{
    CAPTURE_LENS_SCALE, Frame, HudExtras, OVERLAY_SETTINGS, Perf, Scene, StarCatalog, UiState,
    ViewChoices, apply_bookmark, default_controls, frame_durations, hud_data, label_texts, nav_hud,
    still_status,
};
use crate::world::{self, Assets, World, baked_dir, iso_utc};

/// Default capture size (brief).
pub const DEFAULT_SIZE: UVec2 = UVec2::new(1920, 1080);
/// Default frames rendered per Manual bookmark, and the least an Auto bookmark renders.
pub const DEFAULT_FRAMES: u32 = 4;
/// Auto exposure has settled when the scene and star EV each move less than this between two
/// frames [EV] (brief).
pub const SETTLE_EV: f32 = 0.05;
/// Most settle frames per Auto bookmark (brief).
pub const MAX_SETTLE_FRAMES: u32 = 16;
/// Manual frames at the settled EVs after settling (brief: the last two).
pub const MANUAL_FRAMES: u32 = 2;
/// The world's epoch before the first bookmark: the baked data's start date (no comet settles).
const START_EPOCH_UTC: &str = "2026-11-20T00:00:00Z";
/// Frame time handed to the features in captures [s]; time is paused, so it only fills
/// `CameraUniform::time`.
const CAPTURE_DT_S: f64 = 1.0 / 60.0;
/// Physical pixels per egui point for `--hud` (lead ruling: 1:1).
const HEADLESS_HUD_PIXELS_PER_POINT: f32 = 1.0;
/// `hdr.layout` (B8's reader requires the prefix "f32 LE RGBA").
pub const HDR_LAYOUT: &str = "f32 LE RGBA, rows top-down";

/// `universe --capture <bookmarks.ron> --out <dir> [--size WxH] [--frames N] [--hud] [--hdr]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptureArgs {
    pub bookmarks: PathBuf,
    pub out: PathBuf,
    pub size: UVec2,
    pub frames: u32,
    pub hud: bool,
    /// Also write `<name>.hdr.bin` (33.2 MB at 1920×1080).
    pub hdr: bool,
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

/// The `hdr` header of a capture JSON (B8's reader).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct HdrHeader {
    /// The sidecar's file name, relative to the JSON's directory.
    pub file: String,
    /// The HDR target's (internal) size.
    pub width: u32,
    pub height: u32,
    /// The capture camera's vertical field of view [rad] (B8's `flux` needs Ω_c).
    pub fov_y_rad: f64,
    /// Exposure factors [scene, star]: pre-exposed value = luminance [cd/m²] × pre.
    pub pre: [f32; 2],
    pub ev100: [f32; 2],
    pub layout: &'static str,
}

/// `hud` (with `--hud`): what the HUD showed (C4's HUD v2).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct HudInfo {
    /// The target card's row labels in order (its live rows, then its facts); empty without a
    /// card.
    pub card_rows: Vec<String>,
    /// The card's kicker and title ("TARGET", "Mars"), when there is one.
    pub card: Option<[String; 2]>,
    /// The HUD blocks drawn (`uni_ui::Hud::drawn_blocks`): the collapsed lines or the expanded
    /// panels, the card, the time line or panel, the notes line and list.
    pub panels: Vec<String>,
    /// The notes the HUD showed, in order.
    pub notes: Vec<String>,
}

/// `<name>.json`: the capture's metrics (JSON v2).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CaptureMetrics {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Pass name → GPU duration [ms] (differences of the timer's stamps), `null` where the
    /// device returned no valid measurement; empty without timestamp queries.
    pub gpu_ms: BTreeMap<String, Option<f32>>,
    /// Whole-frame GPU milliseconds (between the timer's frame brackets), `null` if not
    /// measured or invalid.
    pub gpu_frame_ms: Option<f32>,
    /// CPU milliseconds to build (snapshot included) and encode the captured frame.
    pub cpu_ms: f32,
    pub star_count: u32,
    pub lens_scale: f32,
    /// The bookmark's epoch, `YYYY-MM-DDThh:mm:ssZ` (UTC, whole seconds).
    pub epoch_utc: String,
    /// [scene, star] EV100 of the captured frame.
    pub ev100: Option<[f32; 2]>,
    /// "auto" or "manual"; for auto, the frames it took to settle.
    pub exposure: &'static str,
    pub settle_frames: u32,
    /// Whether the exposure the frame was captured at is settled: `false` when Auto exposure
    /// still moved by [`SETTLE_EV`] or more after [`MAX_SETTLE_FRAMES`] frames (the capture then
    /// uses the last EVs and says so in `notes`; review B9 r0 minor 8); `true` for Manual.
    pub settled: bool,
    /// The honesty labels in effect (exact strings).
    pub honesty: Vec<String>,
    #[serde(flatten)]
    pub measured: Measurements,
    /// Data and pass warnings in effect (a missing map, a camera inside a body, ...).
    pub notes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hdr: Option<HdrHeader>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hud: Option<HudInfo>,
}

/// A headless GPU that can render `size` (`flag` names the option in the error).
pub fn headless_gpu(size: UVec2, flag: &str) -> anyhow::Result<GpuContext> {
    let gpu = pollster::block_on(GpuContext::new_headless())?;
    let max = gpu.device.limits().max_texture_dimension_2d;
    anyhow::ensure!(
        size.max_element() <= max,
        "{flag} {size} exceeds the GPU's {max} px limit"
    );
    Ok(gpu)
}

/// Submits `cmds`, waits until the GPU has finished them, and fails on any uncaptured GPU
/// error raised so far (review m6). `what` names the work in the error.
pub fn submit_and_wait(
    gpu: &GpuContext,
    cmds: Vec<wgpu::CommandBuffer>,
    what: &str,
) -> anyhow::Result<()> {
    gpu.queue.submit(cmds);
    check_gpu_errors(gpu, what)
}

/// Waits until the GPU has finished everything submitted, then fails on any uncaptured GPU
/// error raised so far, naming `what`. Captures call it after a bookmark's read-backs (the PNG
/// and the taps submit their own copies after the frame's check) and the bench after its final
/// drain, so an error raised there fails the run and is blamed on its own bookmark, not the
/// next one, or on none after the last (final review, app minor 6).
pub fn check_gpu_errors(gpu: &GpuContext, what: &str) -> anyhow::Result<()> {
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .with_context(|| format!("waiting for {what}"))?;
    let errors = gpu.take_errors();
    anyhow::ensure!(errors.is_empty(), "{what}: {}", errors.join("; "));
    Ok(())
}

/// The world and its assets for a headless run, or [`CaptureError::MissingData`] without
/// AT-HYG (every M1 bookmark draws the AT-HYG sky; a headless run never shows the BSC5
/// fallback silently).
pub fn load_world() -> Result<(World, Assets), CaptureError> {
    let epoch = Epoch::parse_iso_utc(START_EPOCH_UTC)
        .context("internal: bad start epoch")
        .map_err(CaptureError::Failed)?;
    let (world, assets) = world::load(&baked_dir(), epoch);
    for w in &assets.warnings {
        log::warn!("baked data: {w}");
    }
    if assets.athyg_missing() {
        return Err(CaptureError::MissingData(format!(
            "{} (bake it with `cargo run -p uni-bake -- athyg ...`, see data/manifest.toml)",
            assets.warnings.join("; ")
        )));
    }
    Ok((world, assets))
}

/// Renders every bookmark in `args.bookmarks` into `args.out`.
pub fn run(args: &CaptureArgs) -> Result<(), CaptureError> {
    let bookmarks = uni_nav::load_bookmarks(&args.bookmarks)?;
    check_names(&bookmarks)?;
    let (mut world, assets) = load_world()?;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;

    let gpu = headless_gpu(args.size, "--size")?;
    let mut scene = Scene::new(
        &gpu,
        args.size,
        CAPTURE_FORMAT,
        StarCatalog::AtHyg(world.sim.star_data()),
        &assets,
        CAPTURE_LENS_SCALE,
        world.sim.asteroid_records(),
    )?;
    scene.set_settle_tails(true);
    let target = CaptureTarget::new(&gpu.device, args.size);
    let mut hud = args
        .hud
        .then(|| Hud::new_headless(&gpu, CAPTURE_FORMAT, HEADLESS_HUD_PIXELS_PER_POINT));
    let size = scene.renderer.targets().size;

    let mut frame_index = 0_u64;
    let mut last_epoch: Option<Epoch> = None;
    for b in &bookmarks {
        let mut controls = default_controls(CAPTURE_LENS_SCALE);
        // The comets are settled at the bookmark's epoch before its anchor and `look_at` are
        // resolved, so a view of a comet is placed against where the comet is drawn.
        let comets_settle_ms = settle_at_bookmark_epoch(b, &mut world)?;
        let applied = apply_bookmark(b, &mut world, &mut controls)?;
        world.set_time_rate(0.0); // a capture is a still frame
        if last_epoch.is_some_and(|e| e != world.epoch()) {
            scene.reset_comets(&gpu)?;
        }
        last_epoch = Some(world.epoch());
        let ui = bookmark_ui(b, &mut world)?;
        if args.hud {
            controls.card_expanded = ui.selected.is_some();
            controls.notes_expanded = true;
        }
        // Note any unnamed near star, then register it (one zero tick) before the frames.
        let _ = world.sim.snapshot(&applied.view, size);
        world.advance(0.0);
        scene.set_lens_scale(controls.lens_scale);
        let mut choice = applied.choice;
        let auto = choice.mode() == ExposureMode::Auto;

        let job = FrameJob {
            gpu: &gpu,
            target: &target,
            assets: &assets,
            view: &applied.view,
            size,
            name: &b.name,
            hdr: args.hdr,
            ui,
            cruise: b.cruise == Some(true),
            anchor: applied.anchor,
        };

        let mut settle_frames = 0;
        let mut settled = true;
        let last = if auto {
            // At most MAX_SETTLE_FRAMES, or a long --frames would never let it settle (review
            // B9g r0 minor 4).
            let min = args
                .frames
                .saturating_sub(MANUAL_FRAMES)
                .clamp(2, MAX_SETTLE_FRAMES);
            let mut prev: Option<[f32; 2]> = None;
            settled = false;
            for i in 0..MAX_SETTLE_FRAMES {
                render_one(
                    &job,
                    &mut scene,
                    &world,
                    hud.as_mut(),
                    &controls,
                    choice,
                    true,
                    false,
                    frame_index,
                )?;
                frame_index += 1;
                settle_frames = i + 1;
                let now = scene.resolved_ev100();
                let still = match (prev, now) {
                    (Some(p), Some(n)) => {
                        (n[0] - p[0]).abs() < SETTLE_EV && (n[1] - p[1]).abs() < SETTLE_EV
                    }
                    _ => false,
                };
                if still && settle_frames >= min {
                    settled = true;
                    break;
                }
                prev = now;
            }
            let ev = scene
                .resolved_ev100()
                .context("auto exposure resolved nothing to settle on")?;
            controls.scene_ev100 = ev[0];
            controls.star_ev100 = ev[1];
            choice.exposure = ExposureMode::Manual;
            let mut last = None;
            for f in 0..MANUAL_FRAMES {
                last = Some(render_one(
                    &job,
                    &mut scene,
                    &world,
                    hud.as_mut(),
                    &controls,
                    choice,
                    false,
                    f + 1 == MANUAL_FRAMES,
                    frame_index,
                )?);
                frame_index += 1;
            }
            last
        } else {
            let mut last = None;
            for f in 0..args.frames {
                last = Some(render_one(
                    &job,
                    &mut scene,
                    &world,
                    hud.as_mut(),
                    &controls,
                    choice,
                    false,
                    f + 1 == args.frames,
                    frame_index,
                )?);
                frame_index += 1;
            }
            last
        };
        let (frame, cpu_ms, hud_info) = last.context("no frame rendered")?;

        let png = args.out.join(format!("{}.png", b.name));
        target.save_png(&gpu, &png)?;
        let world_img = scene.world_tap.take(&gpu)?;
        let hdr_img = scene.hdr_tap.take(&gpu)?;
        check_gpu_errors(&gpu, &format!("read-backs of {:?}", b.name))?;
        let ev100 = scene.resolved_ev100();
        let ev = ev100.unwrap_or([
            frame.settings.manual.scene_ev100,
            frame.settings.manual.star_ev100,
        ]);

        let hdr = match &hdr_img {
            Some(img) => {
                let file = format!("{}.hdr.bin", b.name);
                let path = args.out.join(&file);
                std::fs::write(&path, img.to_le_bytes())
                    .with_context(|| format!("writing {}", path.display()))?;
                Some(HdrHeader {
                    file,
                    width: img.size.x,
                    height: img.size.y,
                    fov_y_rad: applied.view.fov_y_rad,
                    pre: ev.map(uni_gfx::exposure_scale),
                    ev100: ev,
                    layout: HDR_LAYOUT,
                })
            }
            None => None,
        };
        let overlay = capture_overlay(&world, &frame, &frame.ctx.camera, args.size);
        let measured = measure::measure(&measure::Input {
            name: &b.name,
            world: &world,
            frame: &frame,
            ev100: ev,
            world_hdr: world_img.as_ref(),
            v_lim: scene.v_lim(),
            comets_settle_ms,
            rings: assets.rings.as_ref(),
            maps: &assets.maps,
            map_dir: &assets.map_dir,
            belt: Some(&scene.belt),
            overlay: Some(&overlay),
            comets_cpu_ms: Some(scene.comets_cpu_ms()),
            asteroids_cpu_ms: Some(scene.asteroids_cpu_ms()),
            asteroid_counts: Some(scene.asteroid_counts()),
        });
        let mut notes = assets.warnings.clone();
        notes.extend(scene.notices.iter().cloned());
        notes.extend(scene.pass_warnings());
        notes.extend(frame.coverage.warning().map(str::to_owned));
        if !settled {
            notes.push(unsettled_note());
        }
        notes.extend(crate::scene::duplicate_pass_names(&scene.renderer.gpu_ms()));
        let metrics = CaptureMetrics {
            name: b.name.clone(),
            width: target.size.x,
            height: target.size.y,
            gpu_ms: frame_durations(&scene.renderer)
                .into_iter()
                .map(|(pass, ms)| (pass.to_owned(), ms))
                .collect(),
            gpu_frame_ms: scene.renderer.gpu_frame_ms(),
            cpu_ms,
            star_count: scene.star_count,
            lens_scale: controls.lens_scale,
            epoch_utc: iso_utc(world.epoch()),
            ev100,
            exposure: if auto { "auto" } else { "manual" },
            settle_frames,
            settled,
            honesty: label_texts(&frame.labels),
            measured,
            notes,
            hdr,
            hud: hud_info,
        };
        let json_path = args.out.join(format!("{}.json", b.name));
        write_metrics(&json_path, &metrics)?;
        println!("{}", summary(&png, &metrics));
    }
    Ok(())
}

/// What every frame of one bookmark shares.
struct FrameJob<'a> {
    gpu: &'a GpuContext,
    target: &'a CaptureTarget,
    assets: &'a Assets,
    view: &'a uni_core::CameraView,
    size: UVec2,
    name: &'a str,
    hdr: bool,
    ui: UiState,
    /// The bookmark asks for cruise (the HUD's Speed panel and flight label).
    cruise: bool,
    /// The frame the camera is anchored to (its absolute velocity for closing speeds).
    anchor: uni_core::FrameId,
}

/// Moves `world` to bookmark `b`'s epoch (if it names one) and settles every comet there
/// (`World::settle_comets`); returns the settle's wall time [ms]. Captures and the bench call it
/// before `apply_bookmark`, so a bookmark anchored on or looking at a comet resolves against
/// the comet's settled position (Task C6: resolved first, the 1986 Halley view missed Halley by
/// the 40 years of perturbations between its 2026 conic and its settled one).
pub fn settle_at_bookmark_epoch(b: &uni_nav::Bookmark, world: &mut World) -> anyhow::Result<f32> {
    if let Some(s) = &b.epoch_utc {
        let e = crate::scene::parse_bookmark_epoch(s)
            .with_context(|| format!("bookmark {:?}: bad epoch_utc {s:?}", b.name))?;
        world.set_epoch(e);
    }
    let start = Instant::now();
    world.sim.settle_comets();
    Ok(start.elapsed().as_secs_f32() * 1e3)
}

/// The bookmark's UI state (Task C6): its `target` selected in `world` (an unknown name is an
/// error naming the bookmark), its `highlight_belt`, and its flight label (`cruise`: the cruise
/// label; otherwise free flight, at rest, so below c). Captures, the bench and the window's
/// `--start` apply it.
pub fn bookmark_ui(b: &uni_nav::Bookmark, world: &mut World) -> anyhow::Result<UiState> {
    let selected = match &b.target {
        Some(name) => Some(
            world
                .sim
                .find(name)
                .with_context(|| format!("bookmark {:?}: no target object {name:?}", b.name))?,
        ),
        None => None,
    };
    world.sim.set_selected(selected);
    world.sim.set_goto_target(None);
    // Cruise carries its label at any speed (plan table): no view is needed for the label.
    let flight = if b.cruise == Some(true) {
        uni_ui::FlightLabel::Cruise
    } else {
        uni_ui::FlightLabel::Free
    };
    Ok(UiState {
        flight,
        ftl: false,
        highlight_belt: b.highlight_belt == Some(true),
        selected,
    })
}

/// The overlay of a captured frame (built for every capture: the JSON's `overlay_orbits`; drawn
/// only with `--hud`), at 1 pt per pixel (`HEADLESS_HUD_PIXELS_PER_POINT`).
fn capture_overlay(
    world: &World,
    frame: &Frame,
    view: &uni_core::CameraView,
    size: UVec2,
) -> uni_ui::Overlay {
    uni_ui::build_overlay(
        &world.sim,
        &frame.snapshot,
        view,
        size,
        size.as_vec2() / HEADLESS_HUD_PIXELS_PER_POINT,
        frame.ui.selected,
        None,
        None,
        &OVERLAY_SETTINGS,
    )
}

/// One capture frame: build, render (the HUD and the taps on the `captured` one), wait for the
/// GPU and read the frame's timings. Returns the frame, its CPU time [ms] and, with the HUD on
/// the captured frame, what the HUD showed.
#[allow(clippy::too_many_arguments)]
fn render_one(
    job: &FrameJob<'_>,
    scene: &mut Scene,
    world: &World,
    hud: Option<&mut Hud>,
    controls: &uni_ui::HudControls,
    choice: ViewChoices,
    settle: bool,
    captured: bool,
    frame_index: u64,
) -> anyhow::Result<(Frame, f32, Option<HudInfo>)> {
    let gpu = job.gpu;
    let start = Instant::now();
    let frame = scene.prepare(
        world,
        job.view,
        job.size,
        job.size,
        controls,
        choice,
        job.ui,
        settle,
        frame_index,
        CAPTURE_DT_S,
    );
    scene.world_tap.arm(captured);
    scene.hdr_tap.arm(captured && job.hdr);
    let mut cmds = vec![scene.render(gpu, &frame, &job.target.view)];
    scene.world_tap.arm(false);
    scene.hdr_tap.arm(false);
    let mut info = None;
    if let Some(hud) = hud.filter(|_| captured) {
        // The HUD shows this frame's measured CPU time so far (snapshot, inputs and the
        // renderer's encode; review B9 r0 minor 2: it showed 0.00). fps 0 leaves the fps part
        // out: a capture has no frame rate.
        let cpu_so_far_ms = start.elapsed().as_secs_f32() * 1e3;
        let perf = Perf::from_renderer(&scene.renderer, gpu, 0.0, cpu_so_far_ms);
        let mut warnings = job.assets.warnings.clone();
        warnings.extend(scene.notices.iter().cloned());
        let statuses = scene.pass_warnings();
        let status = still_status(world, job.view, job.cruise, job.ui.selected);
        let seen = crate::scene::absolute_view(world, job.view, job.anchor);
        let nav = nav_hud(world, &seen, &status, |n| scene.has_map(n));
        let data = hud_data(
            world,
            &frame,
            nav,
            HudExtras {
                perf,
                star_count: scene.star_count,
                warnings: &warnings,
                statuses: &statuses,
            },
        );
        let overlay = capture_overlay(world, &frame, &seen, job.size);
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture hud"),
            });
        let mut c = *controls;
        let (hud_cmds, _hud_out) = hud.draw_headless(
            gpu,
            &mut enc,
            &job.target.view,
            job.target.size.to_array(),
            &data,
            &overlay,
            &mut c,
        );
        cmds.extend(hud_cmds);
        cmds.push(enc.finish());
        info = Some(hud_info(&data, hud.drawn_blocks()));
    }
    let cpu_ms = start.elapsed().as_secs_f32() * 1e3;
    submit_and_wait(gpu, cmds, &format!("capture frame of {:?}", job.name))?;
    scene.renderer.poll_timings(gpu);
    Ok((frame, cpu_ms, info))
}

/// The capture JSON's `hud` block for what the HUD drew from `data`; `blocks` are the blocks
/// its last draw showed (`uni_ui::Hud::drawn_blocks`).
fn hud_info(data: &uni_ui::HudData, blocks: Vec<&'static str>) -> HudInfo {
    let card_rows = data
        .target
        .iter()
        .flat_map(|c| c.live.iter().chain(c.facts.iter()))
        .map(|r| r.label.clone())
        .collect();
    // Only what the HUD drew (lead rulings C4w r3 and r4: the TRAPPIST-1 capture listed a
    // Gravity panel it did not draw; Gravity and Location give way to the honesty notes).
    let panels = blocks.into_iter().map(str::to_owned).collect();
    HudInfo {
        card_rows,
        card: data
            .target
            .as_ref()
            .map(|c| [c.kicker.clone(), c.title.clone()]),
        panels,
        notes: data.notes.iter().map(|(t, _)| t.clone()).collect(),
    }
}

/// The `notes` line of a capture whose Auto exposure did not settle in [`MAX_SETTLE_FRAMES`].
fn unsettled_note() -> String {
    format!(
        "Auto exposure did not settle in {MAX_SETTLE_FRAMES} frames (the EVs still moved by \
         {SETTLE_EV} or more): captured at the last EVs"
    )
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

/// `ms` to two decimals, or "n/a" for no valid measurement (review I5: never 0.00).
pub fn ms_or_na(ms: Option<f32>) -> String {
    ms.map_or_else(|| "n/a".to_owned(), |ms| format!("{ms:.2}"))
}

/// One line per capture for the terminal.
fn summary(png: &Path, m: &CaptureMetrics) -> String {
    let gpu = if m.gpu_ms.is_empty() {
        "GPU ms n/a".to_owned()
    } else {
        format!(
            "frame {}, {} GPU ms",
            ms_or_na(m.gpu_frame_ms),
            m.gpu_ms
                .iter()
                .map(|(pass, ms)| format!("{pass} {}", ms_or_na(*ms)))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let ev = m.ev100.map_or_else(String::new, |[scene, star]| {
        format!("; EV100 scene {scene:.2}, stars {star:.2}")
    });
    format!(
        "captured {} ({}×{}, {} {}): {gpu}; CPU {:.2} ms; {} stars; lens scale {}{ev}",
        png.display(),
        m.width,
        m.height,
        m.exposure,
        m.settle_frames,
        m.cpu_ms,
        m.star_count,
        m.lens_scale
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics() -> CaptureMetrics {
        CaptureMetrics {
            name: "bh-shadow-30M-grid".to_owned(),
            width: 1920,
            height: 1080,
            gpu_ms: BTreeMap::from([
                ("lens".to_owned(), Some(3.5)),
                ("stars".to_owned(), Some(0.25)),
                ("tonemap".to_owned(), None),
            ]),
            gpu_frame_ms: Some(4.0),
            cpu_ms: 1.5,
            star_count: 333_294,
            lens_scale: 1.0,
            epoch_utc: "2026-11-20T00:00:00Z".to_owned(),
            ev100: Some([0.0, -4.0]),
            exposure: "manual",
            settle_frames: 0,
            settled: true,
            honesty: vec!["Lens background: M0 debug grid".to_owned()],
            measured: Measurements::default(),
            notes: Vec::new(),
            hdr: None,
            hud: None,
        }
    }

    /// The JSON v2 keys B8's readers and the sheet expect (B9 Step 2).
    #[test]
    fn metrics_json_has_the_v2_fields() {
        let v: serde_json::Value = serde_json::to_value(metrics()).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "cpu_ms",
                "epoch_utc",
                "ev100",
                "exposure",
                "features",
                "flux",
                "gpu_frame_ms",
                "gpu_ms",
                "height",
                "honesty",
                "lens_scale",
                "limbs",
                "metrics",
                "name",
                "notes",
                "projected",
                "regions",
                "samples",
                "settle_frames",
                "settled",
                "star_count",
                "width"
            ]
        );
        // Review I5: an invalid measurement is null, not 0.
        assert_eq!(v["gpu_ms"]["tonemap"], serde_json::Value::Null);
        assert_eq!(v["ev100"], serde_json::json!([0.0, -4.0]));
        let mut m = metrics();
        m.hdr = Some(HdrHeader {
            file: "x.hdr.bin".to_owned(),
            width: 1920,
            height: 1080,
            fov_y_rad: 1.0,
            pre: [1.0, 2.0],
            ev100: [0.0, -1.0],
            layout: HDR_LAYOUT,
        });
        m.hud = Some(HudInfo {
            card_rows: Vec::new(),
            card: None,
            panels: vec!["place line".to_owned()],
            notes: Vec::new(),
        });
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["hdr"]["file"], "x.hdr.bin");
        assert!(
            v["hdr"]["layout"]
                .as_str()
                .unwrap()
                .starts_with("f32 LE RGBA")
        );
        assert_eq!(v["hud"]["card_rows"], serde_json::json!([]));
    }

    /// Review B9 r0 minor 8: an Auto capture that ran out of settle frames says so in the JSON
    /// (`settled: false`) and in `notes`.
    #[test]
    fn an_unsettled_capture_is_flagged() {
        let mut m = metrics();
        m.exposure = "auto";
        m.settle_frames = MAX_SETTLE_FRAMES;
        m.settled = false;
        m.notes.push(unsettled_note());
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["settled"], serde_json::Value::Bool(false));
        assert_eq!(v["settle_frames"], 16);
        let note = v["notes"][0].as_str().unwrap();
        assert!(note.contains("did not settle in 16 frames"), "{note}");
        assert_eq!(
            serde_json::to_value(metrics()).unwrap()["settled"],
            serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn metrics_are_written_as_json() {
        let path = std::env::temp_dir().join(format!(
            "uni-app-{}-metrics_are_written_as_json.json",
            std::process::id()
        ));
        write_metrics(&path, &metrics()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["lens_scale"], 1.0);
        assert_eq!(v["gpu_frame_ms"], 4.0);
    }

    #[test]
    fn bookmark_names_must_be_unique_file_stems() {
        for ok in ["start", "bh-shadow-30M", "sky_orion.v2"] {
            assert!(is_file_stem(ok), "{ok}");
        }
        for bad in ["", ".hidden", "a/b", "..", "a b", "c:\\x", "naïve"] {
            assert!(!is_file_stem(bad), "{bad}");
        }
        for file in [
            crate::world::m1_bookmarks_path(),
            crate::world::m0_bookmarks_path(),
            crate::world::bookmarks_path("m1-bench.ron"),
        ] {
            let marks = uni_nav::load_bookmarks(&file).unwrap();
            check_names(&marks).unwrap();
            let mut twice = marks.clone();
            twice.push(marks[0].clone());
            assert!(check_names(&twice).is_err());
        }
        assert!(check_names(&[]).is_err());
    }

    #[test]
    fn summary_shows_invalid_timings_as_na() {
        let line = summary(Path::new("out/x.png"), &metrics());
        assert!(line.contains("tonemap n/a"), "{line}");
        assert!(!line.contains("0.00 "), "{line}");
        let mut m = metrics();
        m.gpu_ms.clear();
        assert!(summary(Path::new("x.png"), &m).contains("GPU ms n/a"));
    }
}
