//! winit ApplicationHandler: window, main loop and per-frame wiring (M0 Task 11, M1 Task B9).
//!
//! Starts at the `start` bookmark of `bookmarks/m1.ron` (Earth from 60 000 km), renders
//! continuously (`ControlFlow::Poll`, Fifo present) while the window is visible, and feeds winit
//! input to the HUD and the fly camera. While the window is minimised, hidden or covered it
//! sleeps and retries four times a second, and nothing moves (Task C6w, [`crate::present`]). The scene renders at a fixed internal resolution of about 1080 lines at the
//! window's aspect ([`internal_size`]); the tonemap upscales it to the swapchain and the HUD
//! draws at full resolution on top (review I6).
//!
//! Each frame (B9 Step 1, Task C6 with the navigator; uni-nav `nav` module docs, "the app's
//! frame"): a click picks with the last snapshot and the view it was taken with
//! (`Navigator::pick`: a body anywhere on its disk, nothing hidden behind one; the cursor scaled
//! from the window to the internal resolution); the navigator moves the camera in free flight, cruise or
//! a goto (`Navigator::update`, with the last snapshot's destinations for the aim ring); the time
//! keys (P, `.`, `,`); any Esc the navigator left closes the card, else the notes list, else
//! deselects (the cancel chain: aim → goto → card → notes → selection); the selection and the
//! goto target go to the world; the world notes any unnamed star the camera is now within 0.1 pc
//! of and ticks with the same dt the navigator used (a goto predicts that tick); the camera
//! follows its anchor (`FlyCamera::sync`) and joins the dominant frame, or during a goto the
//! navigator's anchor (none for a frameless target: it keeps its own); then the scene takes the
//! snapshot, sets every shared input and renders, and the HUD draws the overlay (markers, the
//! target bracket, orbits) and the panels, whose buttons act after the draw.
//!
//! Tab hides the HUD, F1 shows the controls list, I opens the card, T the time-warp buttons,
//! F12 saves the current view as an anchored Bookmark v2 (A6 ruling I2: the offset from the
//! anchor body, so a reload lands where the camera was relative to the body). The window closes
//! on Cmd+Q or its close button (Esc is the cancel chain now). GPU errors are logged and shown on
//! the HUD instead of aborting the app. Without AT-HYG the window falls back to the BSC5 stars
//! with a HUD notice.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use glam::{DQuat, DVec3, UVec2};
use uni_core::{CameraView, Epoch, UniversePos};
use uni_gfx::GpuContext;
use uni_nav::pick::PickView;
use uni_nav::{
    Action, Bookmark, ExposureChoice, FlightMode, FlyCamera, InputState, LensEnvChoice, NavStatus,
    Navigator,
};
use uni_post::ExposureMode;
use uni_rel::LensEnv;
use uni_sim::snapshot::Snapshot;
use uni_ui::{CardAction, Hud, HudControls, HudOutput};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::present::{self, Attempt, Change, PresentGate};
use crate::scene::{
    HudExtras, INTERACTIVE_LENS_SCALE, OVERLAY_SETTINGS, Perf, Scene, StarCatalog, UiState,
    ViewChoices, absolute_view, apply_bookmark, default_controls, flight_label, hud_data,
    internal_size, nav_hud,
};
use crate::world::{self, Assets, World, baked_dir, bsc5_path, iso_utc, m1_bookmarks_path};

/// Bookmark the window opens at (brief).
const START_BOOKMARK: &str = "start";
/// The world's epoch before the start bookmark sets its own.
const START_EPOCH_UTC: &str = "2026-11-20T00:00:00Z";
const WINDOW_TITLE: &str = "Universe";
/// Initial window size in logical pixels.
const WINDOW_SIZE: LogicalSize<f64> = LogicalSize::new(1280.0, 800.0);
/// Field of view of the fallback view used when the start bookmark cannot be loaded [rad].
const FALLBACK_FOV_Y_RAD: f64 = std::f64::consts::FRAC_PI_3;
/// How long a one-off HUD notice (such as F12's) stays up.
const FLASH_DURATION: Duration = Duration::from_secs(5);
/// Weight of the newest frame in the smoothed FPS readout.
const FPS_SMOOTHING: f32 = 0.1;
/// How often the window logs its frame rate at `info` level (the HUD's perf line, for runs read
/// from the terminal: `RUST_LOG=info`).
const FPS_LOG_INTERVAL: Duration = Duration::from_secs(10);
/// Longest frame time the camera moves by [s]. After a stall (a debugger, a slow frame) a held
/// key moves the camera one short step instead of speed × the stall. (After a pause in
/// presenting, such as a minimised window, the first frame's dt is 0: [`crate::present`].)
const MAX_CAMERA_DT_S: f64 = 0.1;
/// GPU error lines kept on the HUD (the most recent ones; review m6).
const MAX_GPU_ERROR_NOTICES: usize = 3;

/// Opens the window at bookmark `start` of `bookmarks/m1.ron` ([`START_BOOKMARK`] when `None`)
/// and runs until it is closed.
pub fn run(start: Option<&str>) -> anyhow::Result<()> {
    let start_epoch = Epoch::parse_iso_utc(START_EPOCH_UTC).context("internal: start epoch")?;
    let (mut world, assets) = world::load(&baked_dir(), start_epoch);
    let mut notices = assets.warnings.clone();
    let bsc5 = if assets.athyg_missing() {
        let (stars, w) = world::load_bsc5(&bsc5_path());
        notices.extend(w);
        Some(stars)
    } else {
        None
    };
    for n in &notices {
        log::warn!("{n}");
    }
    let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
    let (camera, choice) = start_view(
        &m1_bookmarks_path(),
        start.unwrap_or(START_BOOKMARK),
        &mut world,
        &mut controls,
        &mut notices,
    );

    let event_loop = EventLoop::new().context("creating the event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut desc = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
        event_loop.owned_display_handle(),
    ));
    desc.backends = wgpu::Backends::METAL;
    // Replaced with the first frame's; the empty first snapshot picks nothing anyway.
    let last_view = PickView::new(&camera.view, UVec2::ONE);
    let mut app = App {
        instance: Some(wgpu::Instance::new(desc)),
        assets,
        bsc5,
        choice,
        notices,
        flash: None,
        world,
        controls,
        camera,
        nav: Navigator::new(),
        last_snap: Snapshot::default(),
        last_view,
        input: InputState::new(),
        modifiers: ModifiersState::empty(),
        last_nav: None,
        gfx: None,
        error: None,
        gpu_errors: Vec::new(),
        frame_index: 0,
        last_frame: None,
        fps: 0.0,
        last_fps_log: None,
        cpu_ms: 0.0,
        cursor_grabbed: false,
        present: PresentGate::new(),
    };
    event_loop
        .run_app(&mut app)
        .context("running the event loop")?;
    app.error.map_or(Ok(()), Err)
}

/// The camera at bookmark `name` in `path` (its epoch, exposure and lens settings applied, its
/// anchor joined) and its view choices, or a camera at the SSB with the default choices plus a
/// HUD notice if it cannot be loaded.
fn start_view(
    path: &Path,
    name: &str,
    world: &mut World,
    controls: &mut HudControls,
    notices: &mut Vec<String>,
) -> (FlyCamera, ViewChoices) {
    let start = uni_nav::load_bookmarks(path).and_then(|marks| {
        marks
            .into_iter()
            .find(|b| b.name == name)
            .with_context(|| format!("no bookmark {name:?} in {}", path.display()))
    });
    // As a capture: the comets settled at the bookmark's epoch before it is resolved (a no-op
    // at the baked data's 2026-11-20).
    let applied = start.and_then(|b| {
        crate::capture::settle_at_bookmark_epoch(&b, world)?;
        apply_bookmark(&b, world, controls)
    });
    match applied {
        Ok(applied) => {
            let mut camera = FlyCamera::new(applied.view);
            camera.reanchor(world.sim.frames(), world.epoch(), applied.anchor);
            (camera, applied.choice)
        }
        Err(e) => {
            let msg = format!("Start view unavailable, starting at the SSB: {e:#}");
            log::warn!("{msg}");
            notices.push(msg);
            let view = CameraView {
                pos: UniversePos::default(),
                orient: DQuat::IDENTITY,
                fov_y_rad: FALLBACK_FOV_Y_RAD,
                velocity_mps: DVec3::ZERO,
            };
            (FlyCamera::new(view), ViewChoices::default())
        }
    }
}

/// Everything that exists once the window does.
struct Gfx {
    window: Arc<Window>,
    gpu: GpuContext,
    scene: Scene,
    hud: Hud,
    /// The surface's sRGB view format, which the tonemap renders through.
    srgb_format: wgpu::TextureFormat,
}

impl Gfx {
    /// Follows a window resize: only the swapchain changes. The renderer's targets follow the
    /// internal resolution the next frame asks for ([`internal_size`]). A zero axis (minimised)
    /// changes nothing: frames are skipped until the window comes back (Review Focus 1).
    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.gpu.resize_surface(w, h);
    }
}

struct App {
    /// Moved into the GPU context when the window is created.
    instance: Option<wgpu::Instance>,
    assets: Assets,
    /// The BSC5 stars, when AT-HYG is missing (B9 Step 1's fallback).
    bsc5: Option<Vec<uni_data::CatalogStar>>,
    /// The start view's exposure and lens choices.
    choice: ViewChoices,
    /// HUD warnings that stay up (data, start view).
    notices: Vec<String>,
    /// A HUD notice shown until the given time.
    flash: Option<(String, Instant)>,
    world: World,
    controls: HudControls,
    camera: FlyCamera,
    /// Selection, the aim ring, cruise and the goto (C3).
    nav: Navigator,
    /// The last frame's snapshot: the navigator's pick items, disks and destinations.
    last_snap: Snapshot,
    /// The camera and internal size `last_snap` was taken with (a click's ray).
    last_view: PickView,
    input: InputState,
    /// The keyboard modifiers held (for Cmd+Q).
    modifiers: ModifiersState,
    /// The navigator's last logged state (mode, aim candidate, selection).
    last_nav: Option<NavLog>,
    gfx: Option<Gfx>,
    /// Set if the window or GPU could not be created; `run` returns it.
    error: Option<anyhow::Error>,
    /// The latest uncaptured GPU errors (at most [`MAX_GPU_ERROR_NOTICES`]), shown on the HUD.
    gpu_errors: Vec<String>,
    frame_index: u64,
    last_frame: Option<Instant>,
    fps: f32,
    /// When the frame rate was last logged.
    last_fps_log: Option<Instant>,
    /// CPU time of the previous frame's update and encode [ms].
    cpu_ms: f32,
    /// The cursor is hidden and locked for mouse look.
    cursor_grabbed: bool,
    /// When to try to present again while the surface gives no texture (Task C6w).
    present: PresentGate,
}

impl App {
    fn create_gfx(
        &mut self,
        el: &ActiveEventLoop,
        instance: wgpu::Instance,
    ) -> anyhow::Result<Gfx> {
        // The least size keeps room for the HUD's honesty notes (lead ruling C4w-I1).
        let attrs = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_inner_size(WINDOW_SIZE)
            .with_min_inner_size(LogicalSize::new(
                uni_ui::hud::MIN_WINDOW_PT[0],
                uni_ui::hud::MIN_WINDOW_PT[1],
            ));
        let window = Arc::new(el.create_window(attrs).context("creating the window")?);
        let gpu = pollster::block_on(GpuContext::new_windowed(instance, Arc::clone(&window)))?;
        let config = &gpu
            .surface
            .as_ref()
            .context("the windowed GPU context has no surface")?
            .config;
        let plain_format = config.format;
        let srgb_format = plain_format.add_srgb_suffix();
        anyhow::ensure!(
            config.view_formats.contains(&srgb_format),
            "the surface has no {srgb_format:?} view"
        );
        let internal = internal_size(UVec2::new(config.width, config.height));
        let catalog = match &self.bsc5 {
            Some(stars) => StarCatalog::Bsc5(stars),
            None => StarCatalog::AtHyg(self.world.sim.star_data()),
        };
        let scene = Scene::new(
            &gpu,
            internal,
            srgb_format,
            catalog,
            &self.assets,
            self.controls.lens_scale,
            self.world.sim.asteroid_records(),
        )?;
        self.bsc5 = None;
        let hud = Hud::new(&window, &gpu, plain_format);
        Ok(Gfx {
            window,
            gpu,
            scene,
            hud,
            srgb_format,
        })
    }

    /// Updates and renders one frame if the surface gives a texture; otherwise tells the
    /// [`PresentGate`] why not, and returns (the gate decides when to try again).
    fn redraw(&mut self) {
        let Some(g) = self.gfx.as_mut() else {
            return;
        };
        let size = g.window.inner_size();
        let attempt_at = Instant::now();
        if size.width == 0 || size.height == 0 {
            // Minimised on a platform that zeroes the size (Review Focus 1; macOS keeps it and
            // reports `Occluded` instead).
            let change = self.present.on_attempt(Attempt::Occluded, attempt_at);
            log_present(change, self.frame_index);
            return;
        }
        let Some(surface) = g.gpu.surface.as_ref() else {
            return;
        };
        let acquired = surface.surface.get_current_texture();
        let frame = match acquired {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            other => {
                let attempt = present::attempt_of(&other);
                drop(other);
                if attempt == Attempt::Reconfigured {
                    g.resize(size.width, size.height);
                }
                let change = self.present.on_attempt(attempt, attempt_at);
                log_present(change, self.frame_index);
                return;
            }
        };

        // dt runs from the last presented frame; the first frame after a pause moves nothing
        // (the hidden time is not simulated: `present` module docs).
        let now = Instant::now();
        let resumed = !self.present.presenting();
        let change = self.present.on_attempt(Attempt::Frame, now);
        log_present(change, self.frame_index);
        let dt = match self.last_frame {
            Some(t) if !resumed => (now - t).as_secs_f64(),
            _ => 0.0,
        };
        self.last_frame = Some(now);
        self.fps = smoothed_fps(self.fps, dt);
        if self
            .last_fps_log
            .is_none_or(|t| now - t >= FPS_LOG_INTERVAL)
        {
            self.last_fps_log = Some(now);
            log::info!(
                "window: {:.1} fps, CPU {:.2} ms, frame {} (the HUD's perf line)",
                self.fps,
                self.cpu_ms,
                self.frame_index
            );
        }

        let start = Instant::now();
        if self.input.take_pressed_once(Action::ToggleHud) {
            self.controls.visible = !self.controls.visible;
        }
        if self.input.take_pressed_once(Action::Help) {
            self.controls.show_help = !self.controls.show_help;
        }
        if self.input.take_pressed_once(Action::TimePanel) {
            self.controls.show_warp = !self.controls.show_warp;
        }
        if self.input.take_pressed_once(Action::ToggleCard) {
            self.controls.card_expanded = !self.controls.card_expanded;
        }
        if self.input.take_pressed_once(Action::Screenshot) {
            let saved = save_view(
                &views_dir(),
                &self.camera,
                &self.world,
                &self.controls,
                self.choice,
            );
            let msg = match saved {
                Ok(path) => format!("Saved view {} (render with --capture)", path.display()),
                Err(e) => format!("Could not save the view: {e:#}"),
            };
            log::info!("{msg}");
            self.flash = Some((msg, now + FLASH_DURATION));
        }
        let output = UVec2::new(frame.texture.width(), frame.texture.height());
        let internal = internal_size(output);
        // The navigator's frame (module docs). One dt for the navigator and the world's tick:
        // a goto predicts that tick (uni-nav `nav` docs, C3 report §6.1).
        let dt_frame = dt.min(MAX_CAMERA_DT_S);
        if self.input.take_pressed_once(Action::Select)
            && let Some(cursor) = self.input.cursor_px()
        {
            let to_internal = internal.as_dvec2() / output.as_dvec2();
            let at = (cursor * to_internal).as_vec2();
            let radius = uni_nav::pick_radius_px(internal.y);
            self.nav.pick(&self.last_snap, &self.last_view, at, radius);
        }
        let status = self.nav.update(
            &mut self.camera,
            &self.world.sim,
            &self.last_snap,
            &mut self.input,
            dt_frame,
        );
        uni_nav::handle_time_keys(&mut self.input, &mut self.world.sim.time);
        if self.input.take_pressed_once(Action::Deselect) {
            esc_after_navigator(&mut self.controls, &mut self.nav);
        }
        log_navigation(
            &self.world,
            &self.nav,
            &self.controls,
            &mut self.last_nav,
            &status,
        );
        let goto_target = goto_target(&status);
        self.world.sim.set_goto_target(goto_target);
        self.world.sim.set_selected(self.nav.selected);
        // Note near stars at the new place, tick, follow the anchor, join the dominant frame
        // (or the goto's anchor).
        let _ = self.world.sim.nearest_surface(&self.camera.view.pos);
        self.world.advance(dt_frame);
        let (frames, epoch) = (self.world.sim.frames(), self.world.epoch());
        self.camera.sync(frames, epoch);
        let anchor = match status.mode {
            FlightMode::Goto { .. } => status.anchor,
            FlightMode::Free | FlightMode::Cruise => {
                Some(self.world.sim.dominant_frame(&self.camera.view.pos))
            }
        };
        if let Some(anchor) = anchor {
            self.camera.reanchor(frames, epoch, anchor);
        }
        if self.input.is_looking() != self.cursor_grabbed {
            self.cursor_grabbed = self.input.is_looking();
            grab_cursor(&g.window, self.cursor_grabbed);
        }

        g.scene.set_lens_scale(self.controls.lens_scale);
        let ui = UiState {
            flight: flight_label(&status),
            ftl: status.ftl,
            highlight_belt: self.controls.highlight_belt,
            selected: self.nav.selected,
        };
        // The window adapts its exposure over time (no settle).
        let f = g.scene.prepare(
            &self.world,
            &self.camera.view,
            internal,
            output,
            &self.controls,
            self.choice,
            ui,
            false,
            self.frame_index,
            dt,
        );
        let srgb_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(g.srgb_format),
            ..Default::default()
        });
        let plain_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut cmds = vec![g.scene.render(&g.gpu, &f, &srgb_view)];

        if self.flash.as_ref().is_some_and(|(_, until)| now >= *until) {
            self.flash = None;
        }
        keep_latest(
            &mut self.gpu_errors,
            g.gpu.take_errors(),
            MAX_GPU_ERROR_NOTICES,
        );
        let mut warnings = self.notices.clone();
        warnings.extend(g.scene.notices.iter().cloned());
        warnings.extend(self.gpu_errors.iter().cloned());
        warnings.extend(self.flash.iter().map(|(msg, _)| msg.clone()));
        let statuses = g.scene.pass_warnings();
        let perf = Perf::from_renderer(&g.scene.renderer, &g.gpu, self.fps, self.cpu_ms);
        let seen = absolute_view(&self.world, &self.camera.view, self.camera.anchor);
        let nav = nav_hud(&self.world, &seen, &status, |n| g.scene.has_map(n));
        let data = hud_data(
            &self.world,
            &f,
            nav,
            HudExtras {
                perf,
                star_count: g.scene.star_count,
                warnings: &warnings,
                statuses: &statuses,
            },
        );
        // The overlay in egui points: the window's physical size over its scale factor (C5:
        // `output_pt` equals egui's content rectangle).
        let scale = g.window.scale_factor() as f32;
        let output_pt = output.as_vec2() / scale.max(f32::MIN_POSITIVE);
        let overlay = uni_ui::build_overlay(
            &self.world.sim,
            &f.snapshot,
            &seen,
            internal,
            output_pt,
            self.nav.selected,
            status.aim.map(|a| (a.target, a.fill)),
            goto_target,
            &OVERLAY_SETTINGS,
        );
        let mut enc = g
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("hud") });
        let size_px = [frame.texture.width(), frame.texture.height()];
        let (hud_cmds, hud_out) = g.hud.draw(
            &g.window,
            &g.gpu,
            &mut enc,
            &plain_view,
            size_px,
            &data,
            &overlay,
            &mut self.controls,
        );
        cmds.extend(hud_cmds);
        cmds.push(enc.finish());
        apply_hud_output(&hud_out, &mut self.nav, &mut self.world, &mut self.controls);
        self.last_snap = f.snapshot;
        self.last_view = PickView::new(&self.camera.view, internal);
        self.cpu_ms = start.elapsed().as_secs_f32() * 1e3;

        g.gpu.queue.submit(cmds);
        g.window.pre_present_notify();
        g.gpu.queue.present(frame);
        self.frame_index += 1;
    }
}

/// Logs a change of the window's present state (`present` module docs), so a run read from the
/// terminal shows why frames stopped (`RUST_LOG=universe=info`; the C6 run's stall was an
/// occluded window).
fn log_present(change: Option<Change>, frame: u64) {
    match change {
        None => {}
        // The window is occluded for a moment before it is first shown.
        Some(Change::Paused(Attempt::Occluded)) if frame == 0 => {
            log::debug!("window: occluded before the first frame")
        }
        Some(Change::Paused(Attempt::Occluded)) => log::info!(
            "window: occluded (minimised, hidden or covered), rendering paused after frame {frame}"
        ),
        Some(Change::Paused(a)) => log::debug!("window: no surface texture ({a:?}), retrying"),
        Some(Change::Stuck { for_s, last }) => {
            log::warn!("window: no surface texture for {for_s:.1} s (last: {last:?}), retrying")
        }
        Some(Change::Resumed { for_s, skipped }) if for_s >= 0.25 => {
            log::info!("window: presenting again at frame {frame} after {for_s:.1} s ({skipped})")
        }
        Some(Change::Resumed { for_s, skipped }) => {
            log::debug!("window: presenting again at frame {frame} after {for_s:.3} s ({skipped})")
        }
    }
}

/// The navigator state [`log_navigation`] compares: mode, aim candidate, selection.
type NavLog = (
    FlightMode,
    Option<uni_sim::registry::ObjectId>,
    Option<uni_sim::registry::ObjectId>,
);

/// Logs the navigator's changes at `info` (mode, aim candidate, selection, drop-outs), so a
/// window run can be followed from the terminal (`RUST_LOG=universe=info`).
fn log_navigation(
    world: &World,
    nav: &Navigator,
    controls: &HudControls,
    last: &mut Option<NavLog>,
    status: &NavStatus,
) {
    let name = |id| world.name(id).to_owned();
    let now = (status.mode, status.aim.map(|a| a.target), nav.selected);
    if *last != Some(now) {
        log::info!(
            "navigation: mode {}, aim {:?}, selected {:?}, card {}, speed {:.3e} m/s",
            match status.mode {
                FlightMode::Goto { target } => format!("goto {}", name(target)),
                m => format!("{m:?}"),
            },
            status.aim.map(|a| name(a.target)),
            nav.selected.map(name),
            controls.card_expanded,
            status.speed_mps
        );
        *last = Some(now);
    }
    if let Some(d) = status.dropout {
        log::info!("navigation: drop-out {d:?}");
    }
}

/// The goto target of `status`, if a goto is running.
fn goto_target(status: &NavStatus) -> Option<uni_sim::registry::ObjectId> {
    match status.mode {
        FlightMode::Goto { target } => Some(target),
        FlightMode::Free | FlightMode::Cruise => None,
    }
}

/// An Esc press the navigator left (it cancels an aim, else a goto, and takes the press only
/// then): closes the card, else the notes list, else deselects (Task C6, the cancel chain).
fn esc_after_navigator(controls: &mut HudControls, nav: &mut Navigator) {
    if controls.card_expanded {
        controls.card_expanded = false;
    } else if controls.notes_expanded {
        controls.notes_expanded = false;
    } else {
        nav.deselect();
    }
}

/// Acts on the HUD's buttons (C4's `HudOutput`): Go to flies to the selection, Close closes the
/// card, a moon from the Moons list becomes the selection, a warp preset sets the rate (and
/// unpauses: a rate pressed while paused means "run at this rate"; C4 review m4), Pause toggles
/// the clock. The Moons button's list is the HUD's own state.
fn apply_hud_output(
    out: &HudOutput,
    nav: &mut Navigator,
    world: &mut World,
    controls: &mut HudControls,
) {
    for a in &out.actions {
        match a {
            CardAction::Goto => {
                nav.goto_selected();
            }
            CardAction::Close => controls.card_expanded = false,
            CardAction::Moons => {}
        }
    }
    if let Some(id) = out.selected_moon.as_deref().and_then(|n| world.sim.find(n)) {
        nav.selected = Some(id);
    }
    if let Some(&rate) = out
        .warp_preset
        .and_then(|i| uni_sim::time::WARP_PRESETS.get(i))
    {
        world.sim.time.set_rate(rate);
        if world.sim.time.is_paused() {
            world.sim.time.toggle_pause();
        }
    }
    if out.toggle_pause {
        world.sim.time.toggle_pause();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let Some(instance) = self.instance.take() else {
            return;
        };
        match self.create_gfx(el, instance) {
            Ok(gfx) => {
                gfx.window.request_redraw();
                self.gfx = Some(gfx);
            }
            Err(e) => {
                self.error = Some(e);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(g) = self.gfx.as_mut() else {
            return;
        };
        let consumed = g.hud.on_window_event(&g.window, &event);
        // Releases always reach the camera input, so a key or mouse look cannot stick; the
        // cursor's moves too, so the click picks where the cursor is (C3 report §6.2).
        if !consumed || is_release(&event) || is_cursor(&event) {
            self.input.on_window_event(&event);
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            // Cmd+Q quits (Esc is the cancel chain: module docs).
            WindowEvent::KeyboardInput { event: key, .. }
                if key.state == ElementState::Pressed
                    && self.modifiers.super_key()
                    && key.physical_key == PhysicalKey::Code(KeyCode::KeyQ) =>
            {
                el.exit();
            }
            WindowEvent::Resized(size) => g.resize(size.width, size.height),
            WindowEvent::Occluded(occluded) => self.present.on_occluded(occluded, Instant::now()),
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        self.input.on_device_event(&event);
    }

    /// Renders continuously (`ControlFlow::Poll`; Fifo paces the frames) while the window
    /// presents; otherwise sleeps until the [`PresentGate`]'s next attempt or the next event
    /// (Task C6w: an occluded window used to spin one core).
    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let Some(g) = &self.gfx else {
            return;
        };
        match self.present.next_attempt(Instant::now()) {
            None => {
                el.set_control_flow(ControlFlow::Poll);
                g.window.request_redraw();
            }
            Some(at) => el.set_control_flow(ControlFlow::WaitUntil(at)),
        }
    }
}

/// Whether `e` moves the cursor or takes it out of the window.
fn is_cursor(e: &WindowEvent) -> bool {
    matches!(
        e,
        WindowEvent::CursorMoved { .. } | WindowEvent::CursorLeft { .. }
    )
}

/// Whether `e` releases a key or a mouse button.
fn is_release(e: &WindowEvent) -> bool {
    match e {
        WindowEvent::KeyboardInput { event, .. } => event.state == ElementState::Released,
        WindowEvent::MouseInput { state, .. } => *state == ElementState::Released,
        _ => false,
    }
}

/// Hides and locks the cursor during mouse look, and restores it after.
fn grab_cursor(window: &Window, look: bool) {
    window.set_cursor_visible(!look);
    let mode = if look {
        CursorGrabMode::Locked
    } else {
        CursorGrabMode::None
    };
    if let Err(e) = window.set_cursor_grab(mode) {
        log::debug!("cursor grab {mode:?}: {e}");
    }
}

/// Appends `new` to `kept`, keeping only the `max` most recent entries.
fn keep_latest(kept: &mut Vec<String>, new: Vec<String>, max: usize) {
    kept.extend(new);
    let excess = kept.len().saturating_sub(max);
    kept.drain(..excess);
}

/// FPS readout: exponentially smoothed 1/dt; the first frame (dt = 0) leaves it unchanged.
fn smoothed_fps(fps: f32, dt_s: f64) -> f32 {
    if !(dt_s > 0.0) {
        return fps;
    }
    let now = (1.0 / dt_s) as f32;
    if fps > 0.0 {
        fps + FPS_SMOOTHING * (now - fps)
    } else {
        now
    }
}

/// Where F12 saves views: `captures/views/` in the workspace (git-ignored).
fn views_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../captures/views")
}

/// The F12 bookmark of `camera`: anchored to its anchor body when it has one (A6 ruling I2:
/// `FlyCamera::bookmark_view`, the offset from the body's centre), with the epoch, the exposure
/// sliders and the view choices. The window's lens scale is left out (review m2): a capture of
/// the view traces the lens at the full capture scale.
fn view_bookmark(
    name: &str,
    camera: &FlyCamera,
    world: &World,
    controls: &HudControls,
    choice: ViewChoices,
) -> Bookmark {
    let (view, anchor) = camera.bookmark_view(|f| world.frame_body_name(f));
    Bookmark {
        name: name.to_owned(),
        view,
        epoch_utc: Some(iso_utc(world.epoch())),
        scene_ev100: Some(controls.scene_ev100),
        star_ev100: Some(controls.star_ev100),
        lens_scale: None,
        anchor,
        look_at: None,
        up: None,
        time_rate: None,
        exposure: Some(match choice.exposure {
            ExposureMode::Auto => ExposureChoice::Auto,
            ExposureMode::Manual => ExposureChoice::Manual,
        }),
        eye_adapted: Some(choice.eye_adapted),
        target: None,
        lens_env: Some(match choice.lens_env {
            LensEnv::Band => LensEnvChoice::Band,
            LensEnv::DebugGrid => LensEnvChoice::DebugGrid,
        }),
        cruise: None,
        highlight_belt: None,
    }
}

/// F12: saves the current view as a one-bookmark RON file in `dir`, named after the wall-clock
/// time, so `universe --capture <file>` renders it. Returns the file's path.
fn save_view(
    dir: &Path,
    camera: &FlyCamera,
    world: &World,
    controls: &HudControls,
    choice: ViewChoices,
) -> anyhow::Result<PathBuf> {
    let unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let name = format!("view-{unix_ms}");
    let bookmark = view_bookmark(&name, camera, world, controls, choice);
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(format!("{name}.ron"));
    uni_nav::save_bookmarks(&path, &[bookmark])?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::CAPTURE_LENS_SCALE;

    #[test]
    fn start_view_is_earth_from_60_000_km() {
        let mut world = World::bare(Epoch::J2000);
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        let mut notices = Vec::new();
        let (camera, choice) = start_view(
            &m1_bookmarks_path(),
            START_BOOKMARK,
            &mut world,
            &mut controls,
            &mut notices,
        );
        assert!(notices.is_empty(), "{notices:?}");
        assert_eq!(choice, ViewChoices::default());
        assert_eq!(
            world.epoch(),
            Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap()
        );
        let earth = world.position_of("Earth").unwrap();
        let d = camera.view.pos.delta_from(earth).length();
        assert!((d - 6.0e7).abs() < 1.0, "{d}");
        assert_eq!(
            world.frame_body_name(camera.anchor).as_deref(),
            Some("Earth")
        );
        // The start bookmark sets no lens scale: the interactive default stays.
        assert_eq!(controls.lens_scale, INTERACTIVE_LENS_SCALE);
    }

    #[test]
    fn missing_bookmarks_start_at_the_ssb_with_a_notice() {
        let mut world = World::bare(Epoch::J2000);
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        let mut notices = Vec::new();
        let (camera, _) = start_view(
            Path::new("/nonexistent/m1.ron"),
            START_BOOKMARK,
            &mut world,
            &mut controls,
            &mut notices,
        );
        assert_eq!(camera.view.pos, UniversePos::default());
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("/nonexistent/m1.ron"), "{notices:?}");
    }

    /// A6 ruling I2 (the exact app.rs change): F12 saves an anchored camera as the offset from
    /// its body, so a reload at the same (whole-second) epoch lands where the camera was relative
    /// to the body, within 1e-4 m; the root form would miss by about the body's motion in the
    /// dropped fraction of a second.
    #[test]
    fn f12_saves_an_anchored_view_that_loads_back_relative_to_the_body() {
        let mut world = World::bare(Epoch::J2000);
        // A fractional-second epoch: F12 floors it to the whole UTC second.
        world.set_epoch(
            Epoch::parse_iso_utc("2026-11-20T00:00:00Z")
                .unwrap()
                .add_seconds(0.75),
        );
        let mars = world.sim.find("Mars").unwrap();
        let mars_frame = world.sim.objects()[mars as usize].frame.unwrap();
        let offset = DVec3::new(5.4e6, -1.0e6, 2.0e5);
        let mut camera = FlyCamera::new(CameraView {
            pos: world.sim.position(mars).offset(offset),
            orient: DQuat::from_rotation_y(0.3),
            fov_y_rad: 0.8,
            velocity_mps: DVec3::ZERO,
        });
        camera.reanchor(world.sim.frames(), world.epoch(), mars_frame);
        let mut controls = default_controls(0.75);
        controls.scene_ev100 = 2.5;
        let choice = ViewChoices::default();
        let dir = std::env::temp_dir().join(format!(
            "uni-app-{}-f12_saves_an_anchored_view",
            std::process::id()
        ));
        let path = save_view(&dir, &camera, &world, &controls, choice).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let loaded = uni_nav::load_bookmarks(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(text.contains("anchor: Some(\"Mars\")"), "{text}");
        let b = &loaded[0];
        assert!(b.name.starts_with("view-"));
        assert_eq!(b.epoch_utc.as_deref(), Some("2026-11-20T00:00:00Z"));
        assert_eq!(b.scene_ev100, Some(2.5));
        // Review m2: the window's interactive lens scale is not saved.
        assert_eq!(b.lens_scale, None);
        assert_eq!(b.exposure, Some(ExposureChoice::Auto));
        assert_eq!(b.lens_env, Some(LensEnvChoice::Band));
        let mut again = World::bare(Epoch::J2000);
        let mut c = default_controls(CAPTURE_LENS_SCALE);
        let applied = apply_bookmark(b, &mut again, &mut c).unwrap();
        let rel = applied
            .view
            .pos
            .delta_from(again.sim.position(again.sim.find("Mars").unwrap()));
        assert!((rel - offset).length() < 1e-4, "{rel} vs {offset}");
        assert_eq!(
            again.frame_body_name(applied.anchor).as_deref(),
            Some("Mars")
        );
        assert_eq!(c.lens_scale, CAPTURE_LENS_SCALE);
        // A root-anchored camera saves its root position with no anchor.
        let root = FlyCamera::new(CameraView {
            pos: UniversePos::from_meters(DVec3::splat(1e9)),
            ..camera.view.clone()
        });
        let b = view_bookmark("r", &root, &world, &controls, choice);
        assert_eq!(b.anchor, None);
        assert_eq!(b.view.pos, root.view.pos);
    }

    /// Task C6: the Esc the navigator leaves (it took the aim and the goto) closes the card,
    /// then the notes list, then deselects.
    #[test]
    fn esc_closes_the_card_then_the_notes_then_deselects() {
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        controls.card_expanded = true;
        controls.notes_expanded = true;
        let mut nav = Navigator::new();
        nav.selected = Some(3);
        esc_after_navigator(&mut controls, &mut nav);
        assert!(!controls.card_expanded && controls.notes_expanded);
        assert_eq!(nav.selected, Some(3));
        esc_after_navigator(&mut controls, &mut nav);
        assert!(!controls.notes_expanded);
        assert_eq!(nav.selected, Some(3));
        esc_after_navigator(&mut controls, &mut nav);
        assert_eq!(nav.selected, None);
    }

    /// The HUD's buttons (C4's `HudOutput`): a warp preset sets the rate and, pressed while
    /// paused, unpauses (C4 review m4); Pause toggles; Close closes the card; a moon from the
    /// list becomes the selection; Go to flies to the selection.
    #[test]
    fn hud_buttons_act_and_a_rate_unpauses() {
        let mut world = World::bare(Epoch::J2000);
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        controls.card_expanded = true;
        let mut nav = Navigator::new();
        world.sim.time.toggle_pause();
        assert!(world.sim.time.is_paused());
        let out = HudOutput {
            actions: vec![CardAction::Close],
            selected_moon: Some("Moon".to_owned()),
            warp_preset: Some(2),
            toggle_pause: false,
        };
        apply_hud_output(&out, &mut nav, &mut world, &mut controls);
        assert!(
            !world.sim.time.is_paused(),
            "a rate pressed while paused runs"
        );
        assert_eq!(world.sim.time.rate(), uni_sim::time::WARP_PRESETS[2]);
        assert!(!controls.card_expanded);
        assert_eq!(nav.selected, world.sim.find("Moon"));
        let out = HudOutput {
            actions: vec![CardAction::Goto],
            toggle_pause: true,
            ..HudOutput::default()
        };
        apply_hud_output(&out, &mut nav, &mut world, &mut controls);
        assert!(world.sim.time.is_paused());
        assert_eq!(
            nav.mode,
            FlightMode::Goto {
                target: world.sim.find("Moon").unwrap()
            }
        );
    }

    #[test]
    fn gpu_error_notices_keep_the_latest() {
        let mut kept = vec!["a".to_owned()];
        keep_latest(&mut kept, vec!["b".into(), "c".into(), "d".into()], 3);
        assert_eq!(kept, ["b", "c", "d"]);
        keep_latest(&mut kept, Vec::new(), 3);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn fps_is_smoothed_and_ignores_the_first_frame() {
        assert_eq!(smoothed_fps(0.0, 0.0), 0.0);
        assert_eq!(smoothed_fps(0.0, 0.02), 50.0);
        let next = smoothed_fps(50.0, 0.01);
        assert!((next - 55.0).abs() < 1e-4, "{next}");
        assert_eq!(smoothed_fps(60.0, f64::NAN), 60.0);
    }
}
