//! winit ApplicationHandler: window, main loop and per-frame wiring (M0 Task 11).
//!
//! Starts at the `start` bookmark of `bookmarks/m0.ron`, renders continuously
//! (`ControlFlow::Poll`, Fifo present), and feeds winit input to the HUD and the fly camera.
//! The scene renders at a fixed internal resolution of about 1080 lines at the window's aspect
//! ([`internal_size`]), decoupled from the window's physical size; the tonemap upscales it to
//! the swapchain and the HUD draws at full resolution on top (review I6).
//! Tab hides the HUD, F1 shows the controls list, F12 saves the current view (position,
//! orientation, field of view, epoch and exposures) as a bookmark that `--capture` renders,
//! with the lens traced at the capture scale of 1.0 rather than the window's slider value,
//! and Esc quits. GPU errors are logged and shown on the HUD instead of aborting the app.
//!
//! With `--sky v1` the window runs the sky v1 stack (`Scene::new_v1`) and opens at
//! `sky-wide-galactic-center` of `bookmarks/sky-v1.ron`. Its auto-exposure adapts over time (no
//! settle); the HUD's scene slider is the manual EV that Auto starts from, the star slider the
//! eye-adapted star EV. Without AT-HYG it falls back to the BSC5 stars with a HUD warning.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use glam::{DQuat, DVec3, UVec2};
use uni_core::{CameraView, Epoch, UniversePos};
use uni_gfx::GpuContext;
use uni_nav::{Action, Bookmark, FlyCamera, InputState};
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
    INTERACTIVE_LENS_SCALE, Perf, Scene, SkyCatalog, SkyExposure, SkyMode, apply_bookmark,
    default_controls, frame_ctx, hud_data, internal_size,
};
use crate::world::{
    World, baked_dir, iso_utc, load_sky_v1_stars, m0_bookmarks_path, sky_path,
    sky_v1_bookmarks_path,
};

/// Bookmark the window opens at (brief).
const START_BOOKMARK: &str = "start";
/// Bookmark the window opens at with `--sky v1`, in `bookmarks/sky-v1.ron`.
const SKY_V1_START_BOOKMARK: &str = "sky-wide-galactic-center";
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

/// Opens the window with the `sky` stack and runs until it is closed.
pub fn run(sky: SkyMode) -> anyhow::Result<()> {
    let (catalog, mut notices) = window_catalog(sky);
    for n in &notices {
        log::warn!("{n}");
    }
    let mut world = World::new(Epoch::J2000);
    let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
    let (path, name) = match sky {
        SkyMode::M0 => (m0_bookmarks_path(), START_BOOKMARK),
        SkyMode::V1 => (sky_v1_bookmarks_path(), SKY_V1_START_BOOKMARK),
    };
    let (view, sky_exposure) = start_view(&path, name, &mut world, &mut controls, &mut notices);

    let event_loop = EventLoop::new().context("creating the event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut desc = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
        event_loop.owned_display_handle(),
    ));
    desc.backends = wgpu::Backends::METAL;
    let mut app = App {
        instance: Some(wgpu::Instance::new(desc)),
        sky,
        catalog: Some(catalog),
        sky_exposure,
        notices,
        flash: None,
        world,
        controls,
        camera: FlyCamera::new(view),
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

/// The window's star catalogue for `sky`, with its HUD notices: BSC5 for M0 (a missing file
/// gives an empty sky and a notice, as before); for sky v1 AT-HYG as `uni_sim::World::star_data()`,
/// or, when AT-HYG is missing, the BSC5 stars plus a notice saying so (B9 Step 1's fallback).
fn window_catalog(sky: SkyMode) -> (SkyCatalog, Vec<String>) {
    if sky == SkyMode::V1 {
        let (stars, warnings) = load_sky_v1_stars(&baked_dir());
        if !stars.is_empty() {
            return (SkyCatalog::AtHyg(stars), warnings);
        }
        let (bsc5, mut notices) = World::load_sky(&sky_path());
        notices.extend(warnings);
        notices.push("No AT-HYG stars: sky v1 shows the BSC5 stars instead".to_owned());
        return (SkyCatalog::Bsc5(bsc5), notices);
    }
    let (stars, notices) = World::load_sky(&sky_path());
    (SkyCatalog::Bsc5(stars), notices)
}

/// Bookmark `name`'s view in `path` (applying its epoch and exposure) and its sky v1 exposure
/// choices, or a view from the SSB with the default choices plus a HUD notice if it cannot be
/// loaded.
fn start_view(
    path: &Path,
    name: &str,
    world: &mut World,
    controls: &mut HudControls,
    notices: &mut Vec<String>,
) -> (CameraView, SkyExposure) {
    let start = uni_nav::load_bookmarks(path).and_then(|marks| {
        marks
            .into_iter()
            .find(|b| b.name == name)
            .with_context(|| format!("no bookmark {name:?} in {}", path.display()))
    });
    let applied = start.and_then(|b| {
        let view = apply_bookmark(&b, world, controls)?;
        Ok((view, SkyExposure::from_bookmark(&b)))
    });
    match applied {
        Ok(start) => start,
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
            (view, SkyExposure::default())
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
    /// internal resolution the next frame asks for ([`internal_size`]; `Renderer::render`
    /// resizes them when the frame's viewport changes). A zero axis (minimised) changes nothing:
    /// frames are skipped until the window comes back (Review Focus 1).
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
    /// The sky stack (`--sky`).
    sky: SkyMode,
    /// Handed to the star pass when the window is created.
    catalog: Option<SkyCatalog>,
    /// Sky v1's exposure choices from the start bookmark (unused by the M0 stack).
    sky_exposure: SkyExposure,
    /// HUD warnings that stay up (catalogue, start view).
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
        let catalog = self
            .catalog
            .take()
            .context("the star catalogue was already handed to a scene")?;
        let scene = Scene::for_sky(
            &gpu,
            internal,
            srgb_format,
            self.sky,
            &catalog,
            self.controls.lens_scale,
        )?;
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
            let saved = save_view(&views_dir(), &self.camera.view, &self.world, &self.controls);
            let msg = match saved {
                Ok(path) => format!("Saved view {} (render with --capture)", path.display()),
                Err(e) => format!("Could not save the view: {e:#}"),
            };
            log::info!("{msg}");
            self.flash = Some((msg, now + FLASH_DURATION));
        }
        self.camera.update(&mut self.input, dt.min(MAX_CAMERA_DT_S));
        self.world.advance(dt);
        if self.input.is_looking() != self.cursor_grabbed {
            self.cursor_grabbed = self.input.is_looking();
            grab_cursor(&g.window, self.cursor_grabbed);
        }

        g.scene.set_lens_scale(self.controls.lens_scale);
        // Sky v1 adapts over time in the window (no settle).
        let fe = g
            .scene
            .set_exposure(&self.controls, self.sky_exposure, false);
        let output = UVec2::new(frame.texture.width(), frame.texture.height());
        let ctx = frame_ctx(
            &self.world,
            &self.camera.view,
            internal_size(output),
            output,
            fe.exposure,
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
        let mut cmds = vec![g.scene.render(&g.gpu, &ctx, &srgb_view)];

        if self.flash.as_ref().is_some_and(|(_, until)| now >= *until) {
            self.flash = None;
        }
        keep_latest(
            &mut self.gpu_errors,
            g.gpu.take_errors(),
            MAX_GPU_ERROR_NOTICES,
        );
        let mut notices = self.notices.clone();
        notices.extend(self.gpu_errors.iter().cloned());
        notices.extend(self.flash.iter().map(|(msg, _)| msg.clone()));
        let perf = Perf::from_renderer(&g.scene.renderer, &g.gpu, self.fps, self.cpu_ms);
        let data = hud_data(
            &self.world,
            &ctx,
            perf,
            self.camera.speed_mps(),
            g.scene.star_count,
            &notices,
            fe.settings.as_ref(),
        );
        let mut enc = g
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("hud") });
        let size_px = [frame.texture.width(), frame.texture.height()];
        cmds.extend(g.hud.draw(
            &g.window,
            &g.gpu,
            &mut enc,
            &plain_view,
            size_px,
            &data,
            &mut self.controls,
        ));
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

/// F12: saves the current view, epoch and exposures as a one-bookmark RON file in `dir`, named
/// after the wall-clock time, so `universe --capture <file>` renders it. Returns the file's
/// path. The window's lens scale is left out (review m2): it is an interactive speed setting,
/// and a capture of the view traces the lens at the full capture scale.
fn save_view(
    dir: &Path,
    view: &CameraView,
    world: &World,
    controls: &HudControls,
) -> anyhow::Result<PathBuf> {
    let unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let name = format!("view-{unix_ms}");
    let bookmark = Bookmark {
        name: name.clone(),
        view: view.clone(),
        epoch_utc: Some(iso_utc(world.epoch)),
        scene_ev100: Some(controls.scene_ev100),
        star_ev100: Some(controls.star_ev100),
        lens_scale: None,
        anchor: None,
        look_at: None,
        up: None,
        time_rate: None,
        exposure: None,
        eye_adapted: None,
        target: None,
        lens_env: None,
        cruise: None,
        highlight_belt: None,
    };
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(format!("{name}.ron"));
    uni_nav::save_bookmarks(&path, &[bookmark])?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::INTERACTIVE_LENS_SCALE;

    #[test]
    fn start_view_is_the_start_bookmark() {
        let mut world = World::new(Epoch::J2000);
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        let mut notices = Vec::new();
        let (view, sky_exposure) = start_view(
            &m0_bookmarks_path(),
            START_BOOKMARK,
            &mut world,
            &mut controls,
            &mut notices,
        );
        assert!(notices.is_empty(), "{notices:?}");
        assert_eq!(sky_exposure, SkyExposure::default());
        let (name, _) = world.nearest(&view.pos);
        assert_eq!(name, "TestBH-1");
        assert_eq!(
            world.epoch,
            Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap()
        );
        // The start bookmark sets no lens scale: the interactive default stays.
        assert_eq!(controls.lens_scale, INTERACTIVE_LENS_SCALE);
    }

    #[test]
    fn sky_v1_start_view_is_the_wide_galactic_centre() {
        let mut world = World::new(Epoch::J2000);
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        let mut notices = Vec::new();
        let (view, sky_exposure) = start_view(
            &sky_v1_bookmarks_path(),
            SKY_V1_START_BOOKMARK,
            &mut world,
            &mut controls,
            &mut notices,
        );
        assert!(notices.is_empty(), "{notices:?}");
        assert_eq!(view.pos, UniversePos::default());
        assert!((view.fov_y_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert_eq!(sky_exposure, SkyExposure::default());
        assert_eq!(
            world.epoch,
            Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap()
        );
    }

    #[test]
    fn missing_bookmarks_start_at_the_ssb_with_a_notice() {
        let mut world = World::new(Epoch::J2000);
        let mut controls = default_controls(INTERACTIVE_LENS_SCALE);
        let mut notices = Vec::new();
        let (view, _) = start_view(
            Path::new("/nonexistent/m0.ron"),
            START_BOOKMARK,
            &mut world,
            &mut controls,
            &mut notices,
        );
        assert_eq!(view.pos, UniversePos::default());
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("/nonexistent/m0.ron"), "{notices:?}");
    }

    #[test]
    fn saved_view_loads_back_as_a_capturable_bookmark() {
        let dir = std::env::temp_dir().join(format!(
            "uni-app-{}-saved_view_loads_back",
            std::process::id()
        ));
        let mut world = World::new(Epoch::J2000);
        let mut controls = default_controls(0.75);
        let (view, _) = start_view(
            &m0_bookmarks_path(),
            START_BOOKMARK,
            &mut world,
            &mut controls,
            &mut Vec::new(),
        );
        controls.scene_ev100 = 2.5;
        let path = save_view(&dir, &view, &world, &controls).unwrap();
        let loaded = uni_nav::load_bookmarks(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(loaded.len(), 1);
        let b = &loaded[0];
        assert!(b.name.starts_with("view-"));
        assert_eq!(b.view, view);
        assert_eq!(b.epoch_utc.as_deref(), Some("2026-11-20T00:00:00Z"));
        assert_eq!(b.scene_ev100, Some(2.5));
        // Review m2: the window's interactive lens scale is not saved, so `--capture` traces
        // the view at its full capture scale.
        assert_eq!(b.lens_scale, None);
        let mut again = World::new(Epoch::J2000);
        let mut c = default_controls(crate::scene::CAPTURE_LENS_SCALE);
        assert_eq!(apply_bookmark(b, &mut again, &mut c).unwrap(), view);
        assert_eq!(c.lens_scale, crate::scene::CAPTURE_LENS_SCALE);
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
