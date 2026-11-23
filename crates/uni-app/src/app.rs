//! winit ApplicationHandler: window, main loop and per-frame wiring (M0 Task 11, M1 Task B9).
//!
//! Starts at the `start` bookmark of `bookmarks/m1.ron` (Earth from 60 000 km), renders
//! continuously (`ControlFlow::Poll`, Fifo present), and feeds winit input to the HUD and the
//! fly camera. The scene renders at a fixed internal resolution of about 1080 lines at the
//! window's aspect ([`internal_size`]); the tonemap upscales it to the swapchain and the HUD
//! draws at full resolution on top (review I6).
//!
//! Each frame (B9 Step 1): the camera moves (relative to its anchor body); the world notes any
//! unnamed star the camera is now within 0.1 pc of and ticks (time, comet catch-up, and the
//! star's registration, so the snapshot draws it with its id: A4 parked item); the camera
//! follows its anchor to the new epoch (`FlyCamera::sync`) and joins the dominant frame
//! (`reanchor` to `World::dominant_frame`, A6); then the scene takes the snapshot, sets every
//! shared input and renders.
//!
//! Tab hides the HUD, F1 shows the controls list, F12 saves the current view as an anchored
//! Bookmark v2 (A6 ruling I2: the offset from the anchor body, so a reload lands where the
//! camera was relative to the body), and Esc quits. GPU errors are logged and shown on the HUD
//! instead of aborting the app. Without AT-HYG the window falls back to the BSC5 stars with a
//! HUD notice.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use glam::{DQuat, DVec3, UVec2};
use uni_core::{CameraView, Epoch, UniversePos};
use uni_gfx::GpuContext;
use uni_nav::{Action, Bookmark, ExposureChoice, FlyCamera, InputState, LensEnvChoice};
use uni_post::ExposureMode;
use uni_rel::LensEnv;
use uni_ui::{Hud, HudControls};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::scene::{
    INTERACTIVE_LENS_SCALE, Perf, Scene, StarCatalog, ViewChoices, apply_bookmark,
    default_controls, hud_data, internal_size,
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
/// Longest frame time the camera moves by [s]. After a stall (a minimised window, a debugger)
/// a held key moves the camera one short step instead of speed × the stall.
const MAX_CAMERA_DT_S: f64 = 0.1;
/// GPU error lines kept on the HUD (the most recent ones; review m6).
const MAX_GPU_ERROR_NOTICES: usize = 3;

/// Opens the window and runs until it is closed.
pub fn run() -> anyhow::Result<()> {
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
        START_BOOKMARK,
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
        input: InputState::new(),
        gfx: None,
        error: None,
        gpu_errors: Vec::new(),
        frame_index: 0,
        last_frame: None,
        fps: 0.0,
        cpu_ms: 0.0,
        cursor_grabbed: false,
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
    match start.and_then(|b| apply_bookmark(&b, world, controls)) {
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
    input: InputState,
    gfx: Option<Gfx>,
    /// Set if the window or GPU could not be created; `run` returns it.
    error: Option<anyhow::Error>,
    /// The latest uncaptured GPU errors (at most [`MAX_GPU_ERROR_NOTICES`]), shown on the HUD.
    gpu_errors: Vec<String>,
    frame_index: u64,
    last_frame: Option<Instant>,
    fps: f32,
    /// CPU time of the previous frame's update and encode [ms].
    cpu_ms: f32,
    /// The cursor is hidden and locked for mouse look.
    cursor_grabbed: bool,
}

impl App {
    fn create_gfx(
        &mut self,
        el: &ActiveEventLoop,
        instance: wgpu::Instance,
    ) -> anyhow::Result<Gfx> {
        let attrs = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_inner_size(WINDOW_SIZE);
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

    /// Updates and renders one frame, unless the window is minimised.
    fn redraw(&mut self) {
        let Some(g) = self.gfx.as_mut() else {
            return;
        };
        let size = g.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return; // minimised (Review Focus 1)
        }
        let now = Instant::now();
        let dt = self.last_frame.map_or(0.0, |t| (now - t).as_secs_f64());
        self.last_frame = Some(now);
        self.fps = smoothed_fps(self.fps, dt);

        let Some(surface) = g.gpu.surface.as_ref() else {
            return;
        };
        let frame = match surface.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(_)
            | wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost => {
                g.resize(size.width, size.height);
                return;
            }
            _ => return, // timeout, occluded or a validation error: try the next frame
        };

        let start = Instant::now();
        if self.input.take_pressed_once(Action::ToggleHud) {
            self.controls.visible = !self.controls.visible;
        }
        if self.input.take_pressed_once(Action::Help) {
            self.controls.show_help = !self.controls.show_help;
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
        // Move, note near stars at the new place, tick, follow the anchor, join the dominant
        // frame (module docs).
        self.camera.update(&mut self.input, dt.min(MAX_CAMERA_DT_S));
        let _ = self.world.sim.nearest_surface(&self.camera.view.pos);
        self.world.advance(dt);
        let (frames, epoch) = (self.world.sim.frames(), self.world.epoch());
        self.camera.sync(frames, epoch);
        let dominant = self.world.sim.dominant_frame(&self.camera.view.pos);
        self.camera.reanchor(frames, epoch, dominant);
        if self.input.is_looking() != self.cursor_grabbed {
            self.cursor_grabbed = self.input.is_looking();
            grab_cursor(&g.window, self.cursor_grabbed);
        }

        g.scene.set_lens_scale(self.controls.lens_scale);
        let output = UVec2::new(frame.texture.width(), frame.texture.height());
        // The window adapts its exposure over time (no settle).
        let f = g.scene.prepare(
            &self.world,
            &self.camera.view,
            internal_size(output),
            output,
            &self.controls,
            self.choice,
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
        let mut notices = self.notices.clone();
        notices.extend(g.scene.notices.iter().cloned());
        notices.extend(g.scene.pass_warnings());
        notices.extend(self.gpu_errors.iter().cloned());
        notices.extend(self.flash.iter().map(|(msg, _)| msg.clone()));
        let perf = Perf::from_renderer(&g.scene.renderer, &g.gpu, self.fps, self.cpu_ms);
        let data = hud_data(
            &self.world,
            &f,
            perf,
            self.camera.speed_mps(),
            g.scene.star_count,
            &notices,
        );
        let mut enc = g
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("hud") });
        let size_px = [frame.texture.width(), frame.texture.height()];
        // No overlay and no HUD actions until C6 wires them (Task C0: call site only).
        let (hud_cmds, _hud_out) = g.hud.draw(
            &g.window,
            &g.gpu,
            &mut enc,
            &plain_view,
            size_px,
            &data,
            &uni_ui::Overlay::default(),
            &mut self.controls,
        );
        cmds.extend(hud_cmds);
        cmds.push(enc.finish());
        self.cpu_ms = start.elapsed().as_secs_f32() * 1e3;

        g.gpu.queue.submit(cmds);
        g.window.pre_present_notify();
        g.gpu.queue.present(frame);
        self.frame_index += 1;
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
        // Releases always reach the camera input, so a key or mouse look cannot stick.
        if !consumed || is_release(&event) {
            self.input.on_window_event(&event);
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput { event: key, .. }
                if !consumed
                    && key.state == ElementState::Pressed
                    && key.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
            {
                el.exit();
            }
            WindowEvent::Resized(size) => g.resize(size.width, size.height),
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        self.input.on_device_event(&event);
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }
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
