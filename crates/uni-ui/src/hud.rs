//! The egui HUD drawn over the tonemapped frame (M0 Task 10; M1 Task C4: the approved design).
//!
//! **Collapsed by default** (owner feedback 2026-11-22, lead ruling on the C0 context, owner
//! point 4): four quiet lines over the view, each on a soft dark halo instead of a panel:
//! - top left, the place line ([`NavLine`]) beside the "i" button; a click on either opens the
//!   card (I does too: the app sets `HudControls::card_expanded`);
//! - top centre, one mono time line (ruling 15);
//! - bottom left, the honesty line: short labels joined by " · ", 11 pt, muted, an amber dot
//!   before each non-physical one; it holds the flight label, so ruling 6 is always on screen.
//!   A click opens "What is approximate here", the full sentences in their tones;
//! - bottom right, the perf line and the key hints.
//!
//! **Expanded** (`card_expanded`): the mockup's panels. Location (top left, 300 pt) with the
//! notes stack under it (12 pt, in tone), Gravity (bottom left, 260 pt; amber kicker, ruling 6),
//! Speed (bottom centre, 520 pt; amber header, the log bar), and the target card (top right,
//! 420 pt) when there is a target; the time stays one line at top centre while the card shows
//! (ruling 15), else the Time panel sits at the top right. T (`show_warp`) shows the time-warp
//! panel (380 pt) in the Time panel's place, or at top centre while the card shows.
//!
//! Every panel is an `egui::Area` laid out at fixed margins (24 pt) and painted with the
//! mockup's tokens (`crate::theme`). The overlay (markers, brackets, orbits; C5) is drawn first,
//! in a full-screen, non-interactive area at `Order::Background` (ruling C.7), so it sits under
//! the panels and never takes a click. Tab (`visible = false`) hides all of it.
//!
//! A windowed HUD ([`Hud::new`]) reads its input and screen size from winit. A headless HUD
//! ([`Hud::new_headless`], for `--capture ... --hud`) has no window: it is laid out for the
//! capture size instead.

use std::ops::RangeInclusive;

use egui::{Align2, Id, Order};
use winit::{
    event::{ElementState, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::cards::{CardAction, TargetCard, show_card};
use crate::overlay::{Overlay, draw_overlay};
use crate::panels::{self, GravityPanel, LocationPanel, NavLine, SpeedPanel, TimePanel, WarpPick};
use crate::theme::{
    self, BELT_HIGHLIGHT_LABEL, BELT_TOGGLE_LABEL, CARD_PADDING_Y, DEBUG_GRID_LABEL,
    EYE_ADAPTED_STARS_LABEL, GRAVITY_WIDTH, HINT_CARD, HINT_COLLAPSED, INK_2, INK_3, INK_SOFT,
    INSIDE_HORIZON_LABEL, KICKER_TRACKING_EM, LOCATION_WIDTH, LabelTone, MUTED, NON_PHYSICAL_SOFT,
    NOTE_DOT, NOTES_LINE_MAX_WIDTH, NOTES_POPUP_TITLE, NOTES_POPUP_WIDTH, NOTES_SIZE,
    SCREEN_MARGIN, SMALL_SIZE, SPEED_WIDTH, TIME_LINE_SIZE, UNLENSED_STARS_LABEL, Weight,
    mono_font, short_label, text_font, tone_color,
};

/// Slider range for both exposures, EV at ISO 100. A UI choice, not physics: it spans the
/// photographic EV of a moonless night sky (about −6; the star default is −6.5, a Milky Way
/// photograph, ruling B4L exposure) to direct sunlight (about 15, the "sunny 16" rule) with
/// margin either side.
const EV100_RANGE: RangeInclusive<f32> = -10.0..=20.0;
/// The lens tracer's resolution-scale range (`LensTracer::set_resolution_scale`, Task 9).
const LENS_SCALE_RANGE: RangeInclusive<f32> = 0.25..=1.0;

/// The key map (uni-nav: M0 Task 6, M1 C3's actions), shown when F1 toggles
/// `HudControls::show_help`: (action, keys).
const CONTROLS: [(&str, &str); 15] = [
    ("Forward / back / left / right", "W / S / A / D"),
    ("Up / down", "Space / Left Ctrl"),
    ("Roll", "Q / E"),
    ("Look", "Right mouse (hold)"),
    ("Speed ×10^±0.1 per notch", "Wheel or = / -"),
    ("Select; a marker flies there", "Left click"),
    ("Go to the selection", "F"),
    ("Cruise (FTL, non-physical)", "C"),
    ("Open / close the card", "I"),
    ("Cancel aim or goto, close the card", "Esc"),
    ("Time-warp panel", "T"),
    ("Pause / faster / slower time", "P / . / ,"),
    ("Show / hide HUD", "Tab"),
    ("Save view bookmark (captures/views/)", "F12"),
    ("Show / hide this list", "F1"),
];

/// What is approximate wherever the lens pass draws (Global Constraints, Honesty): it bends a
/// debug grid or the band, and the catalogue stars are drawn unlensed over it. Shown only while
/// `HudData::lens_on_screen` (the M1 wording of [`crate::theme`]).
const LENS_LABELS: [&str; 2] = [DEBUG_GRID_LABEL, UNLENSED_STARS_LABEL];

/// The app's own hotkeys in uni-nav's key map (Task 6): Tab hides the HUD, F1 shows the
/// controls list, F12 saves the view as a bookmark. Matched by physical key, as uni-nav does.
const APP_HOTKEYS: [KeyCode; 3] = [KeyCode::Tab, KeyCode::F1, KeyCode::F12];

/// Most egui runs per headless draw. egui lays out a new area invisibly on its first run (a
/// sizing pass) and draws it from the second (measured with egui 0.37;
/// `headless_draw_settles_then_costs_one_run`). The rest is margin.
const HEADLESS_MAX_RUNS: usize = 4;

/// Room kept free at the bottom right for the perf line and hints [pt].
const PERF_RESERVE: f32 = 40.0;
/// Gap between neighbouring panels [pt].
const PANEL_GAP: f32 = 16.0;

/// What the HUD shows this frame. Filled by the app; the HUD only reads it.
#[derive(Clone, Debug)]
pub struct HudData {
    /// Smoothed frames per second of the window (0 in captures, where it is not shown).
    pub fps: f32,
    pub cpu_ms: f32,
    /// (pass name, GPU ms) per pass, from the pass timer; `None` where the device returned no
    /// valid measurement (shown as "n/a", never as 0). Listed in the F1 panel.
    pub gpu_ms: Vec<(&'static str, Option<f32>)>,
    /// Whole-frame GPU time [ms], if measured and valid.
    pub gpu_frame_ms: Option<f32>,
    /// False when the device has no timestamp queries; the HUD then shows "GPU n/a".
    pub gpu_timing_available: bool,
    /// Internal render resolution [px] (the frame targets).
    pub render_size: [u32; 2],
    /// Output resolution [px] (the swapchain or capture) the frame is upscaled to.
    pub output_size: [u32; 2],
    pub dist_from_sun_m: f64,
    pub nearest_name: String,
    pub nearest_dist_m: f64,
    pub speed_mps: f64,
    pub epoch: uni_core::Epoch,
    /// Simulated seconds per real second; 0 is paused.
    pub time_rate: f64,
    pub star_count: u32,
    /// Renderer warnings ("Inside Sun", "No surface map: …"): in the notes, amber.
    pub warnings: Vec<String>,
    /// Shows "Inside event horizon" in red, first in the notes.
    pub inside_horizon: bool,
    /// True when the lens pass draws this frame: the notes then hold, in amber, "Lens
    /// background: M0 debug grid" and "Stars near the hole are not yet lensed (M3)". Sky views
    /// skip the pass, so neither line would be true there.
    pub lens_on_screen: bool,
    /// True while "Eye-adapted stars" is in effect (`ExposureSettings::eye_adapted_in_effect`,
    /// review m13): the notes then hold [`EYE_ADAPTED_STARS_LABEL`] in amber.
    pub eye_adapted_stars: bool,
    // ---- v2 (Task C0 interface, drawn by C4) ----
    /// The Location panel.
    pub location: LocationPanel,
    /// The Time panel (one mono line at top centre while the card is open).
    pub time: TimePanel,
    /// The Gravity panel of the dominant body; `None` far from every body.
    pub gravity: Option<GravityPanel>,
    /// The Speed panel.
    pub speed: SpeedPanel,
    /// The target card of the selected object (or the goto target); `None` without one.
    pub target: Option<TargetCard>,
    /// The notes line: honesty labels and warnings with their tones (world labels from
    /// `uni_sim::honesty`, UI labels from `theme::ui_labels`, renderer warnings). The HUD adds
    /// the labels of its own flags above (each label is shown once).
    pub notes: Vec<(String, LabelTone)>,
    /// The collapsed HUD's place line.
    pub nav_line: NavLine,
}

/// Settings the HUD edits. The app reads them back after [`Hud::draw`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HudControls {
    pub scene_ev100: f32,
    pub star_ev100: f32,
    /// Lens tracer resolution scale, 0.25..=1.0.
    pub lens_scale: f32,
    /// False hides the whole HUD, the overlay and the controls list included (Tab).
    pub visible: bool,
    /// Shows the controls and settings list (F1). Its close button clears this.
    pub show_help: bool,
    // ---- v2 UI state (Task C0, ruling C.4; all false at start) ----
    /// The target card is open (the "i" button or I).
    pub card_expanded: bool,
    /// The notes line is expanded to the full sentences ("What is approximate here").
    pub notes_expanded: bool,
    /// The time-warp buttons show (T).
    pub show_warp: bool,
    /// "Highlight asteroid belt" is on (the amber `theme::BELT_HIGHLIGHT_LABEL` follows it).
    pub highlight_belt: bool,
}

/// One-shot actions the HUD's buttons asked for this frame (Task C0; persistent UI state is in
/// [`HudControls`]). The app acts on them after [`Hud::draw`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HudOutput {
    /// Card buttons pressed.
    pub actions: Vec<CardAction>,
    /// A moon picked from the card's Moons list, by name.
    pub selected_moon: Option<String>,
    /// A warp button pressed: an index into `uni_sim::time::WARP_PRESETS`.
    pub warp_preset: Option<usize>,
    /// The Pause button pressed.
    pub toggle_pause: bool,
}

/// The egui context, its winit input state and its wgpu renderer.
pub struct Hud {
    ctx: egui::Context,
    /// winit input. `None` for a headless HUD, which has no window to read.
    state: Option<egui_winit::State>,
    renderer: egui_wgpu::Renderer,
    /// Layout scale for [`Self::draw_headless`]. A windowed [`Self::draw`] reads the window's.
    pixels_per_point: f32,
    max_texture_side: usize,
}

impl Hud {
    /// `plain_format` must be the non-sRGB swapchain format: egui writes gamma-encoded colours.
    pub fn new(
        window: &Window,
        gpu: &uni_gfx::GpuContext,
        plain_format: wgpu::TextureFormat,
    ) -> Self {
        let ctx = hud_context();
        let max_texture_side = gpu.device.limits().max_texture_dimension_2d as usize;
        let scale = window.scale_factor() as f32;
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(scale),
            window.theme(),
            Some(max_texture_side),
        );
        Self {
            ctx,
            state: Some(state),
            renderer: hud_renderer(gpu, plain_format),
            pixels_per_point: scale,
            max_texture_side,
        }
    }

    /// A HUD with no window, for headless captures (`--capture ... --hud`).
    ///
    /// [`Self::draw_headless`] lays it out for its `size_px` at `pixels_per_point` physical
    /// pixels per egui point: 1.0 draws it at 1:1, 2.0 matches a Retina window. A value that is
    /// not finite and positive falls back to 1.0. Nothing animates, so every draw is a finished
    /// frame.
    ///
    /// `target_format` is the format of the view it will draw over. A non-sRGB view matches
    /// the window exactly. An sRGB one, such as `uni_gfx::CAPTURE_FORMAT`, also works:
    /// egui-wgpu then decodes its colours to linear in the shader so the view re-encodes them
    /// (and logs a warning), and translucent edges blend in linear rather than gamma space.
    pub fn new_headless(
        gpu: &uni_gfx::GpuContext,
        target_format: wgpu::TextureFormat,
        pixels_per_point: f32,
    ) -> Self {
        Self {
            ctx: headless_context(),
            state: None,
            renderer: hud_renderer(gpu, target_format),
            pixels_per_point,
            max_texture_side: gpu.device.limits().max_texture_dimension_2d as usize,
        }
    }

    /// Feeds a window event to egui. Returns true if egui consumed it, in which case the app
    /// should not also act on it.
    ///
    /// - The app's hotkeys (Tab, F1, F12) never reach egui and are never consumed, so Tab
    ///   always toggles the HUD and never moves egui's keyboard focus.
    /// - Other keys reach egui, but a press is consumed only while a text field has focus
    ///   (typing a number into a slider's value box), so typing "-2" does not also slow the
    ///   camera. A focused slider or button does not hold the keyboard: flying keeps working.
    ///   This overrides egui-winit's own verdict, which consumes Tab always and every key while
    ///   any widget has focus.
    /// - Releases of keys and mouse buttons are never consumed, so a held key or mouse look
    ///   cannot stick (uni-nav `InputState::on_window_event`).
    /// - Other pointer events are consumed while the pointer is over the HUD or dragging one of
    ///   its sliders.
    /// - A headless HUD has no window input: it ignores the event and returns false.
    pub fn on_window_event(&mut self, window: &Window, e: &WindowEvent) -> bool {
        let Some(state) = self.state.as_mut() else {
            return false;
        };
        if let WindowEvent::KeyboardInput { event, .. } = e {
            let route = route_key(
                event.physical_key,
                event.state.is_pressed(),
                self.ctx.text_edit_focused(),
            );
            return match route {
                KeyRoute::AppOnly => false,
                KeyRoute::Shared { consumed } => {
                    // egui-winit's verdict is replaced; see the doc comment.
                    let _ = state.on_window_event(window, e);
                    consumed
                }
            };
        }
        let egui_consumed = state.on_window_event(window, e).consumed;
        pointer_consumed(e, egui_consumed)
    }

    /// Runs the HUD and records its draw into `enc` over `target` (`LoadOp::Load`, so it
    /// composites over the frame). `target` is the plain (non-sRGB) swapchain view. `overlay`
    /// (markers, brackets, orbits) is drawn under the panels.
    ///
    /// A HUD made by [`Self::new_headless`] draws as [`Self::draw_headless`] does.
    ///
    /// Returns command buffers that must be submitted before `enc.finish()`, and the one-shot
    /// actions of the HUD's buttons.
    #[allow(clippy::too_many_arguments)] // The frozen M1 interface (Task C0).
    pub fn draw(
        &mut self,
        window: &Window,
        gpu: &uni_gfx::GpuContext,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size_px: [u32; 2],
        data: &HudData,
        overlay: &Overlay,
        controls: &mut HudControls,
    ) -> (Vec<wgpu::CommandBuffer>, HudOutput) {
        let Some(state) = self.state.as_mut() else {
            return self.draw_headless(gpu, enc, target, size_px, data, overlay, controls);
        };
        let raw = state.take_egui_input(window);
        let (mut out, hud) = run_hud(&self.ctx, raw, data, overlay, controls);
        state.handle_platform_output(window, std::mem::take(&mut out.platform_output));
        (self.render(gpu, enc, target, size_px, out), hud)
    }

    /// Runs the HUD and records its draw as [`Self::draw`] does, but with no window: it is laid
    /// out for `size_px` at the HUD's `pixels_per_point`. For headless captures.
    ///
    /// egui draws a new panel only from its second run, so this runs egui until it stops asking
    /// for an immediate repaint (three runs on the first call, then one). One call draws the
    /// whole HUD.
    ///
    /// Returns command buffers that must be submitted before `enc.finish()`, and the HUD's
    /// output (as [`Self::draw`]).
    #[allow(clippy::too_many_arguments)] // The frozen M1 interface (Task C0).
    pub fn draw_headless(
        &mut self,
        gpu: &uni_gfx::GpuContext,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size_px: [u32; 2],
        data: &HudData,
        overlay: &Overlay,
        controls: &mut HudControls,
    ) -> (Vec<wgpu::CommandBuffer>, HudOutput) {
        let (out, hud, _runs) = run_hud_settled(
            &self.ctx,
            size_px,
            self.pixels_per_point,
            self.max_texture_side,
            data,
            overlay,
            controls,
        );
        (self.render(gpu, enc, target, size_px, out), hud)
    }

    /// Tessellates one egui output and records it into `enc` over `target`.
    fn render(
        &mut self,
        gpu: &uni_gfx::GpuContext,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size_px: [u32; 2],
        out: egui::FullOutput,
    ) -> Vec<wgpu::CommandBuffer> {
        let egui::FullOutput {
            mut textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = out;
        let jobs = self.ctx.tessellate(shapes, pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size_px,
            pixels_per_point,
        };

        // egui 0.37: several deltas per texture. Drained, not borrowed: epaint's
        // `TexturesDelta` debug-asserts on drop that every delta was taken out.
        for (id, deltas) in textures_delta.set.drain() {
            for delta in &deltas {
                self.renderer
                    .update_texture(&gpu.device, &gpu.queue, id, delta);
            }
        }
        let cmds = self
            .renderer
            .update_buffers(&gpu.device, &gpu.queue, enc, &jobs, &screen);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui hud"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime(); // egui-wgpu wants RenderPass<'static>.
            self.renderer.render(&mut pass, &jobs, &screen);
        }
        for id in textures_delta.free.drain() {
            self.renderer.free_texture(&id);
        }
        cmds
    }
}

/// The egui context every HUD uses.
fn hud_context() -> egui::Context {
    let ctx = egui::Context::default();
    ctx.set_fonts(theme::font_definitions());
    // The HUD always sits over space, so it stays dark whatever the system theme is.
    ctx.set_theme(egui::Theme::Dark);
    ctx
}

/// The context of a headless HUD: as [`hud_context`], but nothing animates, since a capture is
/// one still frame and a fade-in would leave the HUD half drawn.
fn headless_context() -> egui::Context {
    let ctx = hud_context();
    ctx.all_styles_mut(|style| style.animation_time = 0.0);
    ctx
}

/// The wgpu renderer every HUD uses, writing to `target_format`.
fn hud_renderer(
    gpu: &uni_gfx::GpuContext,
    target_format: wgpu::TextureFormat,
) -> egui_wgpu::Renderer {
    egui_wgpu::Renderer::new(
        &gpu.device,
        target_format,
        egui_wgpu::RendererOptions::default(),
    )
}

/// Who gets a keyboard event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyRoute {
    /// An app hotkey: egui never sees it and the app always acts on it.
    AppOnly,
    /// egui sees it. If `consumed`, the app should not also act on it.
    Shared { consumed: bool },
}

/// Routes a key: see [`Hud::on_window_event`].
fn route_key(key: PhysicalKey, pressed: bool, text_edit_focused: bool) -> KeyRoute {
    if matches!(key, PhysicalKey::Code(code) if APP_HOTKEYS.contains(&code)) {
        return KeyRoute::AppOnly;
    }
    KeyRoute::Shared {
        consumed: pressed && text_edit_focused,
    }
}

/// egui's verdict on a non-keyboard event, except that a mouse button release is never
/// consumed.
fn pointer_consumed(e: &WindowEvent, egui_consumed: bool) -> bool {
    match e {
        WindowEvent::MouseInput {
            state: ElementState::Released,
            ..
        } => false,
        _ => egui_consumed,
    }
}

/// The input egui-winit's `take_egui_input` builds from a window, built from a target size
/// instead: the screen is `size_px / pixels_per_point` points.
fn headless_input(
    size_px: [u32; 2],
    pixels_per_point: f32,
    max_texture_side: usize,
) -> egui::RawInput {
    let ppp = if pixels_per_point.is_finite() && pixels_per_point > 0.0 {
        pixels_per_point
    } else {
        1.0
    };
    let size = egui::vec2(size_px[0] as f32, size_px[1] as f32) / ppp;
    let mut raw = egui::RawInput {
        // As egui-winit: a zero-sized target has no screen rect.
        screen_rect: (size.x > 0.0 && size.y > 0.0)
            .then(|| egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        max_texture_side: Some(max_texture_side),
        ..Default::default()
    };
    raw.viewports
        .entry(egui::ViewportId::ROOT)
        .or_default()
        .native_pixels_per_point = Some(ppp);
    raw
}

/// One egui run of the HUD. egui may call the UI closure more than once in a run (a discarded
/// pass); each call starts again from `controls` as it was, and the last call's state and
/// actions are the run's, so a click is never applied twice.
fn run_hud(
    ctx: &egui::Context,
    raw: egui::RawInput,
    data: &HudData,
    overlay: &Overlay,
    controls: &mut HudControls,
) -> (egui::FullOutput, HudOutput) {
    let start = *controls;
    let mut result = (start, HudOutput::default());
    let full = ctx.run_ui(raw, |ui| {
        let mut c = start;
        let mut out = HudOutput::default();
        if c.visible {
            draw_hud(ui.ctx(), data, overlay, &mut c, &mut out);
        }
        result = (c, out);
    });
    *controls = result.0;
    (full, result.1)
}

/// Runs the HUD headless until egui stops asking for an immediate repaint, at most
/// [`HEADLESS_MAX_RUNS`] times. Returns the merged output, the last run's actions and the
/// number of runs.
fn run_hud_settled(
    ctx: &egui::Context,
    size_px: [u32; 2],
    pixels_per_point: f32,
    max_texture_side: usize,
    data: &HudData,
    overlay: &Overlay,
    controls: &mut HudControls,
) -> (egui::FullOutput, HudOutput, usize) {
    let input = || headless_input(size_px, pixels_per_point, max_texture_side);
    let (mut out, mut hud) = run_hud(ctx, input(), data, overlay, controls);
    let mut runs = 1;
    // A new egui area spends its first run sizing itself, invisibly, and asks for a repaint.
    let mut again = wants_repaint_now(&out);
    while again && runs < HEADLESS_MAX_RUNS {
        let (next, next_hud) = run_hud(ctx, input(), data, overlay, controls);
        hud = next_hud;
        runs += 1;
        // Read before `append`, which keeps the shortest repaint delay of all runs.
        again = wants_repaint_now(&next);
        // Keeps every run's texture uploads (the first has the font atlas) and the last
        // run's shapes.
        out.append(next);
    }
    (out, hud, runs)
}

/// Whether egui asked to run again straight away.
fn wants_repaint_now(out: &egui::FullOutput) -> bool {
    out.viewport_output
        .get(&egui::ViewportId::ROOT)
        .is_some_and(|v| v.repaint_delay.is_zero())
}

/// A fixed, non-movable HUD area.
fn area(id: &str, order: Order) -> egui::Area {
    egui::Area::new(Id::unique(("hud", id)))
        .order(order)
        .movable(false)
}

/// Everything the HUD draws in one run (module docs).
fn draw_hud(
    ctx: &egui::Context,
    data: &HudData,
    overlay: &Overlay,
    c: &mut HudControls,
    out: &mut HudOutput,
) {
    let screen = ctx.content_rect();
    let m = SCREEN_MARGIN;
    area("overlay", Order::Background)
        .fixed_pos(screen.min)
        .interactable(false)
        .show(ctx, |ui| {
            ui.set_min_size(screen.size());
            draw_overlay(ui, overlay);
        });

    let notes = hud_notes(data, c);
    let card = data.target.as_ref().filter(|_| c.card_expanded);
    if c.card_expanded {
        area("location", Order::Middle)
            .anchor(Align2::LEFT_TOP, [m, m])
            .show(ctx, |ui| {
                if panels::show_location(ui, &data.location) {
                    c.card_expanded = false;
                }
                // The notes stack (12 pt, in tone); the flight label is the Speed header.
                let stack: Vec<_> = notes
                    .iter()
                    .filter(|(t, _)| *t != data.speed.header)
                    .collect();
                if !stack.is_empty() {
                    ui.add_space(8.0);
                    ui.set_max_width(LOCATION_WIDTH);
                    notes_stack(ui, &stack);
                }
            });
        if let Some(g) = &data.gravity {
            area("gravity", Order::Middle)
                .anchor(Align2::LEFT_BOTTOM, [m, -m])
                .show(ctx, |ui| panels::show_gravity(ui, g));
        }
        // Centred, but clear of the Gravity panel on a narrow screen.
        let mut left = screen.center().x - SPEED_WIDTH / 2.0;
        if data.gravity.is_some() {
            left = left.max(screen.left() + m + GRAVITY_WIDTH + PANEL_GAP);
        }
        area("speed", Order::Middle)
            .anchor(Align2::LEFT_BOTTOM, [left - screen.left(), -m])
            .show(ctx, |ui| panels::show_speed(ui, &data.speed));
        if let Some(card) = card {
            area("card", Order::Middle)
                .anchor(Align2::RIGHT_TOP, [-m, m])
                .show(ctx, |ui| {
                    // The header (≈ 90 pt) and the padding stay; the body scrolls.
                    let max_body = screen.height()
                        - 2.0 * m
                        - 2.0 * CARD_PADDING_Y
                        - 2.0
                        - 90.0
                        - PERF_RESERVE;
                    let clicks = show_card(ui, card, max_body.max(80.0));
                    if clicks.actions.contains(&CardAction::Close) {
                        c.card_expanded = false;
                    }
                    out.actions.extend(clicks.actions);
                    if clicks.selected_moon.is_some() {
                        out.selected_moon = clicks.selected_moon;
                    }
                });
        }
    } else {
        area("place", Order::Middle)
            .anchor(Align2::LEFT_TOP, [m, m])
            .show(ctx, |ui| {
                let halo = ui.painter().add(egui::Shape::Noop);
                let line = panels::show_place(ui, &data.nav_line, false);
                ui.painter()
                    .set(halo, theme::halo_shape(line.rect.expand(6.0)));
                if line.clicked() {
                    c.card_expanded = true;
                }
            });
        if !notes.is_empty() {
            let line_w = NOTES_LINE_MAX_WIDTH.max(screen.width() - 2.0 * m - 320.0);
            let line_h = area("notes", Order::Middle)
                .anchor(Align2::LEFT_BOTTOM, [m, -m + 4.0])
                .show(ctx, |ui| {
                    let line = notes_line(ui, &notes, line_w);
                    if line.clicked() {
                        c.notes_expanded = !c.notes_expanded;
                    }
                    line.rect.height()
                })
                .inner;
            if c.notes_expanded {
                area("notes popup", Order::Foreground)
                    .anchor(Align2::LEFT_BOTTOM, [m, -m - line_h - 6.0])
                    .show(ctx, |ui| notes_popup(ui, &notes));
            }
        }
    }

    // Time: the warp panel (T), the Time panel (expanded, no card) or the one-line time.
    let time_anchor = if card.is_some() {
        (Align2::CENTER_TOP, [0.0, m])
    } else {
        (Align2::RIGHT_TOP, [-m, m])
    };
    if c.show_warp {
        area("warp", Order::Middle)
            .anchor(time_anchor.0, time_anchor.1)
            .show(ctx, |ui| match panels::show_warp(ui, &data.time) {
                Some(WarpPick::Pause) => out.toggle_pause = true,
                Some(WarpPick::Preset(i)) => out.warp_preset = Some(i),
                None => {}
            });
    } else if c.card_expanded && card.is_none() {
        area("time", Order::Middle)
            .anchor(Align2::RIGHT_TOP, [-m, m])
            .show(ctx, |ui| panels::show_time(ui, &data.time));
    } else {
        area("time line", Order::Middle)
            .anchor(Align2::CENTER_TOP, [0.0, m])
            .interactable(false)
            .show(ctx, |ui| {
                let font = mono_font(TIME_LINE_SIZE, Weight::Regular);
                haloed(ui, |ui| {
                    theme::text(ui, &panels::time_line(&data.time), font, INK_SOFT)
                });
            });
    }

    let hint = if card.is_some() {
        HINT_CARD
    } else {
        HINT_COLLAPSED
    };
    area("perf", Order::Middle)
        .anchor(Align2::RIGHT_BOTTOM, [-m, -m + 2.0])
        .interactable(false)
        .show(ctx, |ui| {
            ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
                haloed(ui, |ui| {
                    let font = mono_font(SMALL_SIZE, Weight::Regular);
                    theme::text(ui, &perf_line(data), font.clone(), INK_3);
                    theme::text(ui, hint, font, INK_3);
                });
            });
        });

    if c.show_help {
        show_controls_window(ctx, data, c);
    }
}

/// Paints `add` and a soft halo behind it.
fn haloed(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    let halo = ui.painter().add(egui::Shape::Noop);
    let rect = ui.scope(add).response.rect;
    ui.painter().set(halo, theme::halo_shape(rect.expand(6.0)));
}

/// The notes in their order: "Inside event horizon" (red), the app's notes, then the labels of
/// the HUD's own flags (the belt toggle, eye-adapted stars, the lens) and the renderer warnings
/// (amber); each text once.
fn hud_notes(data: &HudData, c: &HudControls) -> Vec<(String, LabelTone)> {
    fn push(notes: &mut Vec<(String, LabelTone)>, text: &str, tone: LabelTone) {
        if !notes.iter().any(|(t, _)| t == text) {
            notes.push((text.to_owned(), tone));
        }
    }
    let mut notes = Vec::new();
    if data.inside_horizon {
        push(&mut notes, INSIDE_HORIZON_LABEL, LabelTone::Danger);
    }
    for (text, tone) in &data.notes {
        push(&mut notes, text, *tone);
    }
    if c.highlight_belt {
        push(&mut notes, BELT_HIGHLIGHT_LABEL, LabelTone::NonPhysical);
    }
    if data.eye_adapted_stars {
        push(&mut notes, EYE_ADAPTED_STARS_LABEL, LabelTone::NonPhysical);
    }
    if data.lens_on_screen {
        for label in LENS_LABELS {
            push(&mut notes, label, LabelTone::NonPhysical);
        }
    }
    for warning in &data.warnings {
        push(&mut notes, warning, LabelTone::NonPhysical);
    }
    notes
}

/// The collapsed honesty line: each note's short form, muted, joined by " · ", an amber dot
/// before each non-physical one; "…" where the line would pass `max_w`. Clickable.
fn notes_line(ui: &mut egui::Ui, notes: &[(String, LabelTone)], max_w: f32) -> egui::Response {
    let font = text_font(SMALL_SIZE, Weight::Regular);
    let layout = |ui: &egui::Ui, s: &str| {
        ui.painter()
            .layout_job(theme::job(s, font.clone(), MUTED, 0.0, f32::INFINITY))
    };
    let sep = layout(ui, " · ");
    let more = layout(ui, " …");
    let dot_w = NOTE_DOT + 6.0;
    // (galley, dotted, start x, tone) per shown item.
    let mut items = Vec::new();
    let mut x = 0.0;
    let mut cut = false;
    for (i, (text, tone)) in notes.iter().enumerate() {
        let galley = layout(ui, short_label(text));
        let dotted = matches!(tone, LabelTone::NonPhysical | LabelTone::Danger);
        let lead = if i > 0 { sep.size().x } else { 0.0 };
        let need = lead + if dotted { dot_w } else { 0.0 } + galley.size().x;
        let reserve = if i + 1 < notes.len() {
            more.size().x
        } else {
            0.0
        };
        if x + need + reserve > max_w && !items.is_empty() {
            cut = true;
            break;
        }
        items.push((galley, dotted, x + lead, *tone));
        x += need;
    }
    let height = sep.size().y;
    let width = x + if cut { more.size().x } else { 0.0 };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
    ui.painter().add(theme::halo_shape(rect.expand(6.0)));
    let painter = ui.painter();
    let mid = rect.center().y;
    for (i, (galley, dotted, start, tone)) in items.into_iter().enumerate() {
        if i > 0 {
            painter.galley(
                egui::pos2(rect.left() + start - sep.size().x, rect.top()),
                sep.clone(),
                MUTED,
            );
        }
        let mut gx = rect.left() + start;
        if dotted {
            let color = if tone == LabelTone::Danger {
                tone_color(tone)
            } else {
                NON_PHYSICAL_SOFT
            };
            painter.circle_filled(egui::pos2(gx + NOTE_DOT / 2.0, mid), NOTE_DOT / 2.0, color);
            gx += dot_w;
        }
        // Hover lightens the text (preview `.np-line:hover`).
        if response.hovered() {
            painter.galley_with_override_text_color(egui::pos2(gx, rect.top()), galley, INK_2);
        } else {
            painter.galley(egui::pos2(gx, rect.top()), galley, MUTED);
        }
    }
    if cut {
        painter.galley(egui::pos2(rect.left() + x, rect.top()), more, MUTED);
    }
    response
}

/// "What is approximate here": every note in full, in its tone, with its dot.
fn notes_popup(ui: &mut egui::Ui, notes: &[(String, LabelTone)]) {
    theme::panel_frame(theme::HAIRLINE)
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_width(panels::inner_width(NOTES_POPUP_WIDTH, 14.0));
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
            theme::kicker(
                ui,
                NOTES_POPUP_TITLE,
                INK_3,
                Weight::Medium,
                KICKER_TRACKING_EM,
            );
            ui.add_space(2.0);
            for (text, tone) in notes {
                ui.horizontal_top(|ui| {
                    let (dot, _) = ui.allocate_exact_size(
                        egui::vec2(NOTE_DOT, NOTES_SIZE + 2.0),
                        egui::Sense::hover(),
                    );
                    ui.painter()
                        .circle_filled(dot.center(), NOTE_DOT / 2.0, tone_color(*tone));
                    theme::text(
                        ui,
                        text,
                        text_font(NOTES_SIZE, Weight::Regular),
                        tone_color(*tone),
                    );
                });
            }
        });
}

/// The notes stack under the Location panel: each note in full, 12 pt, in its tone.
fn notes_stack(ui: &mut egui::Ui, notes: &[&(String, LabelTone)]) {
    haloed(ui, |ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
        for (text, tone) in notes {
            theme::text(
                ui,
                text,
                text_font(NOTES_SIZE, Weight::Regular),
                tone_color(*tone),
            );
        }
    });
}

/// The perf line (mockup): `"{fps} fps · GPU {ms} ms · CPU {ms} ms"`, "n/a" for an invalid
/// GPU measurement or a device without timestamps (review I5: never 0); the fps part is left
/// out when there is none (captures).
fn perf_line(data: &HudData) -> String {
    let gpu = match data.gpu_frame_ms.filter(|_| data.gpu_timing_available) {
        Some(ms) => format!("GPU {ms:.1} ms"),
        None => "GPU n/a".to_owned(),
    };
    let rest = format!("{gpu} · CPU {:.1} ms", data.cpu_ms);
    if data.fps > 0.0 && data.fps.is_finite() {
        format!("{:.0} fps · {rest}", data.fps)
    } else {
        rest
    }
}

/// The F1 list (top centre): the key map, the settings (exposures, lens scale, "Highlight
/// asteroid belt" with its amber tag) and the performance detail (render size, GPU ms per
/// pass). Its "Close · F1" button clears `show_help`.
fn show_controls_window(ctx: &egui::Context, data: &HudData, c: &mut HudControls) {
    area("controls", Order::Foreground)
        .anchor(Align2::CENTER_TOP, [0.0, 2.0 * SCREEN_MARGIN + 16.0])
        .show(ctx, |ui| {
            theme::panel_frame(theme::HAIRLINE).show(ui, |ui| controls_list(ui, data, c));
        });
}

/// The F1 list's content.
fn controls_list(ui: &mut egui::Ui, data: &HudData, c: &mut HudControls) {
    ui.set_width(panels::inner_width(
        theme::TIME_WARP_WIDTH,
        theme::PANEL_PADDING_X,
    ));
    ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
    theme::kicker(ui, "Controls", INK_3, Weight::Medium, KICKER_TRACKING_EM);
    ui.add_space(4.0);
    for (action, keys) in CONTROLS {
        theme::row(ui, action, keys, true, theme::INK);
    }
    ui.add_space(8.0);
    theme::kicker(ui, "Settings", INK_3, Weight::Medium, KICKER_TRACKING_EM);
    ui.add(
        egui::Slider::new(&mut c.scene_ev100, EV100_RANGE)
            .text("Scene EV100")
            .fixed_decimals(1)
            // A bookmark may set an exposure outside the slider's range; keep it.
            .clamping(egui::SliderClamping::Edits),
    );
    ui.add(
        egui::Slider::new(&mut c.star_ev100, EV100_RANGE)
            .text("Star EV100")
            .fixed_decimals(1)
            .clamping(egui::SliderClamping::Edits),
    );
    ui.add(
        egui::Slider::new(&mut c.lens_scale, LENS_SCALE_RANGE)
            .text("Lens scale")
            .fixed_decimals(2),
    );
    ui.horizontal(|ui| {
        ui.checkbox(&mut c.highlight_belt, BELT_TOGGLE_LABEL);
        theme::text(
            ui,
            "non-physical",
            text_font(SMALL_SIZE, Weight::Regular),
            NON_PHYSICAL_SOFT,
        );
    });
    ui.add_space(8.0);
    theme::kicker(ui, "Performance", INK_3, Weight::Medium, KICKER_TRACKING_EM);
    let font = mono_font(SMALL_SIZE, Weight::Regular);
    for line in perf_detail(data) {
        theme::text(ui, &line, font.clone(), INK_2);
    }
    ui.add_space(8.0);
    let font = text_font(theme::BODY_SIZE, Weight::Medium);
    if theme::button(ui, "Close · F1", font, None, theme::ButtonKind::Neutral).clicked() {
        c.show_help = false;
    }
}

/// `"Render <w>×<h> → <w>×<h>"` (internal → output), `"Stars <n>"`, then `"GPU frame <ms>
/// ms"` and one `"GPU <pass> <ms> ms"` per timed pass, "n/a" for an invalid measurement
/// (review I5: never 0.00); `"GPU ms n/a"` without timestamp queries; `"GPU ms pending"`
/// before the first read-back.
fn perf_detail(data: &HudData) -> Vec<String> {
    let [rw, rh] = data.render_size;
    let [ow, oh] = data.output_size;
    let mut lines = vec![
        format!("Render {rw}×{rh} → {ow}×{oh}"),
        format!(
            "Stars {}",
            crate::format::format_grouped(f64::from(data.star_count), 0)
        ),
    ];
    if !data.gpu_timing_available {
        lines.push("GPU ms n/a".into());
    } else if data.gpu_ms.is_empty() {
        lines.push("GPU ms pending".into());
    } else {
        let ms = |v: Option<f32>| v.map_or_else(|| "n/a".to_owned(), |ms| format!("{ms:.2} ms"));
        lines.push(format!("GPU frame {}", ms(data.gpu_frame_ms)));
        lines.extend(
            data.gpu_ms
                .iter()
                .map(|(pass, v)| format!("GPU {pass} {}", ms(*v))),
        );
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::CardRow;
    use crate::theme::{
        BELT_HIGHLIGHT_LABEL, CARD_WIDTH, CRUISE_BORDER, CRUISE_LABEL, DEBUG_GRID_LABEL, FONT_MONO,
        FONT_MONO_LIGHT, FONT_TEXT, FONT_TEXT_LIGHT, FONT_TEXT_MEDIUM, FREE_FLIGHT_LABEL,
        FlightLabel, HINT_CARD, HINT_COLLAPSED, NAV_BUTTON_BORDER, NON_PHYSICAL, NON_PHYSICAL_SOFT,
        SCREEN_MARGIN, TARGET, TARGET_LINE, UNLENSED_STARS_LABEL, WARP_BUTTONS, WARP_NOTES,
        WARP_SELECTED_BORDER, gravity_kicker, ui_labels,
    };
    use egui::{Color32, FontFamily, Pos2, Vec2};
    use winit::event::{DeviceId, MouseButton};

    const SIZE: [u32; 2] = [1920, 1080];

    /// What the app fills in flight near Mars (the mockup's in-flight artboard).
    fn data() -> HudData {
        let epoch = uni_core::Epoch::parse_iso_utc("2026-11-20T04:12:09Z").expect("date");
        HudData {
            fps: 60.0,
            cpu_ms: 1.04,
            gpu_ms: vec![("stars", Some(0.5)), ("tonemap", None)],
            gpu_frame_ms: Some(16.5),
            gpu_timing_available: true,
            render_size: [1728, 1080],
            output_size: [2560, 1600],
            dist_from_sun_m: 1.524 * uni_core::AU,
            nearest_name: "Mars".into(),
            nearest_dist_m: 1.42e7,
            speed_mps: 12_410.0,
            epoch,
            time_rate: 1000.0,
            star_count: 9096,
            warnings: Vec::new(),
            inside_horizon: false,
            lens_on_screen: false,
            eye_adapted_stars: false,
            location: LocationPanel {
                region: "Near Mars".into(),
                from_sun_m: 1.524 * uni_core::AU,
                light_time_s: 1.524 * uni_core::AU / uni_core::C,
                frame: "Mars".into(),
            },
            time: TimePanel {
                epoch,
                rate: 1000.0,
                paused: false,
            },
            gravity: Some(GravityPanel {
                body: "Mars".into(),
                local_g: 0.138,
                escape_mps: 2210.0,
                circular_mps: 1560.0,
            }),
            speed: SpeedPanel {
                header: FREE_FLIGHT_LABEL.into(),
                tone: LabelTone::NonPhysical,
                speed_mps: 12_410.0,
                note: None,
            },
            target: None,
            notes: ui_labels(FlightLabel::Free, false, false)
                .into_iter()
                .map(|(s, t)| (s.to_owned(), t))
                .collect(),
            nav_line: NavLine::Here {
                name: "Mars".into(),
                dist_m: 1.42e7,
            },
        }
    }

    fn row(label: &str, value: &str, mono: bool) -> CardRow {
        CardRow {
            label: label.into(),
            value: value.into(),
            mono,
            tone: LabelTone::Neutral,
        }
    }

    /// A short Mars card (the mockup's target artboard).
    fn mars_card() -> TargetCard {
        TargetCard {
            kicker: "TARGET".into(),
            title: "Mars".into(),
            subtitle: "Terrestrial planet · Sol IV · real object".into(),
            live: vec![
                row("Distance", "14 200 km", true),
                row("Closing speed", "2.10 km/s", true),
                row("Light time", "0.047 s", true),
            ],
            facts: vec![
                row("Mean radius", "3 389.5 km", true),
                row("Atmosphere", "95.1 % CO₂", false),
            ],
            notes: vec![crate::theme::NOTE_ORBIT_MEAN_ELEMENTS.into()],
            sources: "Data: NASA Mars Fact Sheet".into(),
            actions: vec![CardAction::Goto, CardAction::Moons, CardAction::Close],
            moons: vec!["Phobos".into(), "Deimos".into()],
        }
    }

    fn controls() -> HudControls {
        HudControls {
            scene_ev100: 0.0,
            star_ev100: -6.5,
            lens_scale: 1.0,
            visible: true,
            show_help: false,
            card_expanded: false,
            notes_expanded: false,
            show_warp: false,
            highlight_belt: false,
        }
    }

    fn expanded() -> HudControls {
        HudControls {
            card_expanded: true,
            ..controls()
        }
    }

    /// A text shape egui painted.
    #[derive(Debug)]
    struct Painted {
        text: String,
        /// Top left and size, in points.
        pos: Pos2,
        size: Vec2,
        /// The colour, font and letter spacing of its first section.
        color: Color32,
        font: egui::FontId,
        tracking: f32,
    }

    impl Painted {
        fn center(&self) -> Pos2 {
            self.pos + self.size / 2.0
        }
    }

    /// Everything one draw painted: text, rectangles (rect, fill, stroke colour) and circles
    /// (centre, fill).
    #[derive(Debug, Default)]
    struct Frame {
        texts: Vec<Painted>,
        rects: Vec<(egui::Rect, Color32, Color32)>,
        circles: Vec<(Pos2, Color32)>,
    }

    impl Frame {
        fn find(&self, needle: &str) -> Option<&Painted> {
            self.texts.iter().find(|p| p.text.contains(needle))
        }
        fn get(&self, needle: &str) -> &Painted {
            self.find(needle)
                .unwrap_or_else(|| panic!("no {needle:?} in {:#?}", self.all_text()))
        }
        fn all_text(&self) -> Vec<&str> {
            self.texts.iter().map(|p| p.text.as_str()).collect()
        }
        fn has_stroke(&self, color: Color32) -> Option<egui::Rect> {
            self.rects.iter().find(|r| r.2 == color).map(|r| r.0)
        }
    }

    fn collect(out: &egui::FullOutput) -> Frame {
        fn walk(shape: &egui::Shape, f: &mut Frame) {
            match shape {
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, f)),
                egui::Shape::Text(t) => {
                    let first = t.galley.job.sections.first();
                    let rect = t.galley.rect.translate(t.pos.to_vec2());
                    f.texts.push(Painted {
                        text: t.galley.text().to_owned(),
                        pos: rect.min,
                        size: rect.size(),
                        color: first.map_or(Color32::PLACEHOLDER, |s| s.format.color),
                        font: first.map_or_else(Default::default, |s| s.format.font_id.clone()),
                        tracking: first.map_or(0.0, |s| s.format.extra_letter_spacing),
                    });
                }
                egui::Shape::Rect(r) => f.rects.push((r.rect, r.fill, r.stroke.color)),
                egui::Shape::Circle(c) => f.circles.push((c.center, c.fill)),
                _ => {}
            }
        }
        let mut f = Frame::default();
        for clipped in &out.shapes {
            walk(&clipped.shape, &mut f);
        }
        f
    }

    /// A fresh headless HUD's settled draw of `d` at 1920×1080 and `ppp` (no GPU).
    fn draw_ppp(ppp: f32, d: &HudData, c: &mut HudControls) -> (f32, Frame) {
        let ctx = headless_context();
        (ppp, look(&ctx, ppp, d, c))
    }

    /// A settled draw of `d` at 1920×1080 and `ppp` with `ctx` (its UI memory kept).
    fn look(ctx: &egui::Context, ppp: f32, d: &HudData, c: &mut HudControls) -> Frame {
        let (mut out, _hud, _runs) =
            run_hud_settled(ctx, SIZE, ppp, 8192, d, &Overlay::default(), c);
        // No GPU here to upload to; epaint asserts that deltas are not silently dropped.
        out.textures_delta.clear();
        assert_eq!(out.pixels_per_point, ppp);
        collect(&out)
    }

    fn draw(d: &HudData, mut c: HudControls) -> Frame {
        draw_ppp(1.0, d, &mut c).1
    }

    fn family(name: &str) -> FontFamily {
        FontFamily::Name(name.into())
    }

    /// Owner feedback (2026-11-22) and the lead's ruling: collapsed by default, the HUD is four
    /// quiet lines: the place line with its "i" (top left), one mono time line (top centre),
    /// the honesty line holding the flight label (bottom left), the perf line (bottom right).
    #[test]
    fn collapsed_by_default_four_quiet_lines() {
        let f = draw(&data(), controls());
        let place = f.get("Mars");
        assert!(place.pos.x < 100.0 && place.pos.y < 100.0, "{place:?}");
        assert!(f.find("14 200 km away").is_some(), "{:?}", f.all_text());
        let time = f.get("2026-11-20 04:12:09 UTC · × 1 000");
        assert!(
            (time.center().x - 960.0).abs() < 2.0 && time.pos.y < 60.0,
            "{time:?}"
        );
        assert_eq!(time.font.family, family(FONT_MONO));
        let flight = f.get(FREE_FLIGHT_LABEL);
        assert!(flight.pos.x < 100.0 && flight.pos.y > 1000.0, "{flight:?}");
        let perf = f.get("60 fps · GPU 16.5 ms · CPU 1.0 ms");
        assert!(
            perf.pos.x + perf.size.x > 1880.0 && perf.pos.y > 1000.0,
            "{perf:?}"
        );
        assert!(f.find(HINT_COLLAPSED).is_some());
        for hidden in [
            "LOCATION",
            "TIME",
            "Local g",
            &gravity_kicker("Mars"),
            "km/s",
        ] {
            assert!(f.find(hidden).is_none(), "{hidden:?} shown while collapsed");
        }
    }

    /// I (or a click) expands: Location, Gravity and Speed panels, the target card at the right
    /// with the time collapsed to one line at top centre (ruling 15).
    #[test]
    fn expanded_shows_panels_and_card() {
        let mut d = data();
        d.target = Some(mars_card());
        let f = draw(&d, expanded());
        let kicker = f.get("LOCATION");
        assert!(kicker.pos.x < 60.0 && kicker.pos.y < 60.0, "{kicker:?}");
        assert!(f.find("Near Mars").is_some());
        assert_eq!(f.get("12 min 40 s").font.family, family(FONT_MONO));
        assert_eq!(f.get("Mars · non-rotating").font.family, family(FONT_TEXT));
        // Ruling 6: gravity calculators, never applied, in amber.
        assert_eq!(f.get(&gravity_kicker("Mars")).color, NON_PHYSICAL);
        assert!(f.find("0.138 m/s²").is_some(), "{:?}", f.all_text());
        let speed = f.get("12.41 km/s");
        assert_eq!(speed.font, egui::FontId::new(40.0, family(FONT_MONO_LIGHT)));
        assert!(
            (speed.center().x - 960.0).abs() < 2.0 && speed.pos.y > 900.0,
            "{speed:?}"
        );
        assert!(f.find("4.14 × 10⁻⁵ c").is_some(), "{:?}", f.all_text());
        // The card: right, 420 wide, cyan border, title 30 pt light.
        let border = f.has_stroke(TARGET_LINE).expect("card border");
        assert!((border.width() - CARD_WIDTH).abs() < 1.5, "{border:?}");
        assert!(
            (border.right() - (1920.0 - SCREEN_MARGIN)).abs() < 1.5,
            "{border:?}"
        );
        let title = f.get("Mars");
        let titles: Vec<&Painted> = f.texts.iter().filter(|p| p.text == "Mars").collect();
        assert!(
            titles
                .iter()
                .any(|p| p.font == egui::FontId::new(30.0, family(FONT_TEXT_LIGHT))),
            "{titles:?} ({title:?})"
        );
        assert_eq!(f.get("TARGET").color, TARGET);
        assert!(f.find("Terrestrial planet · Sol IV").is_some());
        assert!(f.find("Data: NASA Mars Fact Sheet").is_some());
        assert!(f.find(crate::theme::NOTE_ORBIT_MEAN_ELEMENTS).is_some());
        // Buttons ≥ 44 pt; "Hold orbit · O" waits for M3.
        let go = f.has_stroke(NAV_BUTTON_BORDER).expect("Go to button");
        assert!(go.height() >= 44.0, "{go:?}");
        for label in ["Go to · F", "Moons", "Close · Esc"] {
            assert!(f.find(label).is_some(), "no {label:?}");
        }
        assert!(f.find("Hold orbit").is_none());
        // Ruling 15: one mono time line at top centre while the card is open.
        assert!(f.find("TIME").is_none(), "{:?}", f.all_text());
        let time = f.get("2026-11-20 04:12:09 UTC");
        assert!((time.center().x - 960.0).abs() < 2.0, "{time:?}");
        assert!(f.find(HINT_CARD).is_some());
    }

    /// Expanded with nothing selected: the Time panel at the top right (mockup).
    #[test]
    fn expanded_without_target_shows_time_panel() {
        let f = draw(&data(), expanded());
        let kicker = f.get("TIME");
        assert!(kicker.pos.x > 1920.0 - 24.0 - 280.0 - 1.0, "{kicker:?}");
        assert_eq!(f.get("2026-11-20 04:12:09 UTC").font.size, 15.0);
        assert!(f.find("× 1 000").is_some());
        assert!(f.find(HINT_COLLAPSED).is_some());
    }

    /// The mockup's type: kickers 11 pt Inter 500 with 0.12 em tracking, values mono.
    #[test]
    fn type_follows_mockup() {
        let f = draw(&data(), expanded());
        let kicker = f.get("LOCATION");
        assert_eq!(
            kicker.font,
            egui::FontId::new(11.0, family(FONT_TEXT_MEDIUM))
        );
        assert!((kicker.tracking - 0.12 * 11.0).abs() < 1e-5, "{kicker:?}");
        assert_eq!(kicker.color, crate::theme::INK_3);
        let region = f.get("Near Mars");
        assert_eq!(region.font, egui::FontId::new(18.0, family(FONT_TEXT)));
        let label = f.get("From Sun");
        assert_eq!((label.font.size, label.color), (13.0, crate::theme::INK_2));
        let value = f.get("1.524 AU");
        assert_eq!(value.font, egui::FontId::new(13.0, family(FONT_MONO)));
        assert_eq!(value.color, crate::theme::INK);
    }

    /// The Speed panel in cruise: amber header and border (plan table, mockup).
    #[test]
    fn cruise_header_is_amber() {
        let mut d = data();
        d.speed = SpeedPanel {
            header: CRUISE_LABEL.into(),
            tone: LabelTone::NonPhysical,
            speed_mps: 4210.0 * uni_core::C,
            note: Some("auto drop-out near stars, atmospheres and black holes".into()),
        };
        let f = draw(&d, expanded());
        let header = f
            .texts
            .iter()
            .find(|p| p.text == CRUISE_LABEL)
            .unwrap_or_else(|| panic!("no cruise header in {:?}", f.all_text()));
        assert_eq!(header.color, NON_PHYSICAL);
        assert!(f.has_stroke(CRUISE_BORDER).is_some(), "no amber border");
        assert!(f.find("4.21 × 10³ c").is_some(), "{:?}", f.all_text());
        assert!(
            f.find("per hour · auto drop-out near stars, atmospheres and black holes")
                .is_some(),
            "{:?}",
            f.all_text()
        );
        // Free flight: the amber header, a grey border.
        let f = draw(&data(), expanded());
        let header = f
            .texts
            .iter()
            .find(|p| p.text == FREE_FLIGHT_LABEL)
            .expect("free-flight header");
        assert_eq!(header.color, NON_PHYSICAL);
        assert!(f.has_stroke(CRUISE_BORDER).is_none());
    }

    /// Free flight faster than light: the speed itself says FTL, in amber (M0 honesty rule).
    #[test]
    fn free_flight_above_c_is_labelled() {
        let mut d = data();
        d.speed.speed_mps = 10.0 * uni_core::C;
        let f = draw(&d, expanded());
        assert_eq!(f.get("FTL · non-physical").color, NON_PHYSICAL);
    }

    /// Round 2 (review I12): the amber label follows the "Highlight asteroid belt" toggle.
    #[test]
    fn belt_highlight_label_follows_toggle() {
        assert_eq!(
            ui_labels(FlightLabel::Free, true, false),
            [
                (FREE_FLIGHT_LABEL, LabelTone::NonPhysical),
                (BELT_HIGHLIGHT_LABEL, LabelTone::NonPhysical),
            ]
        );
        assert_eq!(
            ui_labels(FlightLabel::Cruise, false, true),
            [
                (CRUISE_LABEL, LabelTone::NonPhysical),
                (EYE_ADAPTED_STARS_LABEL, LabelTone::NonPhysical),
            ]
        );
        for on in [true, false] {
            let c = HudControls {
                highlight_belt: on,
                notes_expanded: true,
                ..controls()
            };
            // The app's notes come from `ui_labels`; the HUD also follows its own toggle, so a
            // click shows the label in the same frame.
            {
                let mut d = data();
                d.notes = ui_labels(FlightLabel::Free, on, false)
                    .into_iter()
                    .map(|(s, t)| (s.to_owned(), t))
                    .collect();
                let f = draw(&d, c);
                let full = f.texts.iter().find(|p| p.text == BELT_HIGHLIGHT_LABEL);
                match (on, full) {
                    (true, Some(p)) => assert_eq!(p.color, NON_PHYSICAL),
                    (true, None) => panic!("toggle on, no label in {:?}", f.all_text()),
                    (false, Some(p)) => panic!("toggle off, yet {p:?}"),
                    (false, None) => {}
                }
                assert_eq!(
                    on,
                    f.find("Asteroid belt brightness exaggerated").is_some(),
                    "{:?}",
                    f.all_text()
                );
            }
            let mut d = data();
            d.notes.clear();
            let f = draw(&d, c);
            assert_eq!(
                on,
                f.find(BELT_HIGHLIGHT_LABEL).is_some(),
                "HUD's own toggle"
            );
        }
    }

    /// The collapsed honesty line: short items, muted, an amber dot per non-physical item.
    #[test]
    fn notes_line_marks_non_physical_items() {
        let mut d = data();
        d.notes.push((
            "Milky Way band: galaxy model v0 (real dust map in M4)".into(),
            LabelTone::Neutral,
        ));
        d.eye_adapted_stars = true;
        let f = draw(&d, controls());
        let eye = f.get("Eye-adapted stars");
        assert_eq!(eye.text, "Eye-adapted stars");
        assert_eq!(eye.color, crate::theme::MUTED);
        assert!(f.find("Milky Way band: galaxy model v0").is_some());
        assert!(f.find("(real dust map in M4)").is_none());
        let dots = f
            .circles
            .iter()
            .filter(|(p, fill)| *fill == NON_PHYSICAL_SOFT && p.y > 1000.0)
            .count();
        assert_eq!(dots, 2, "free flight + eye-adapted: {:?}", f.circles);
        // The pop-up has the full sentences in their tones.
        let f = draw(
            &d,
            HudControls {
                notes_expanded: true,
                ..controls()
            },
        );
        assert!(
            f.find("WHAT IS APPROXIMATE HERE").is_some(),
            "{:?}",
            f.all_text()
        );
        assert_eq!(f.get(EYE_ADAPTED_STARS_LABEL).color, NON_PHYSICAL);
        assert_eq!(
            f.get("Milky Way band: galaxy model v0 (real dust map in M4)")
                .color,
            crate::theme::MUTED
        );
    }

    /// Review m13: the eye-adapted label exactly while it is in effect; the M1 lens labels
    /// exactly while the lens pass draws (M0 gate, the "(M3)" wording).
    #[test]
    fn eye_adapted_and_lens_labels_follow_state() {
        let c = HudControls {
            notes_expanded: true,
            ..controls()
        };
        for on in [true, false] {
            let mut d = data();
            d.eye_adapted_stars = on;
            d.lens_on_screen = on;
            let f = draw(&d, c);
            for label in [
                EYE_ADAPTED_STARS_LABEL,
                DEBUG_GRID_LABEL,
                UNLENSED_STARS_LABEL,
            ] {
                let in_tone = f
                    .texts
                    .iter()
                    .any(|p| p.text == label && p.color == NON_PHYSICAL);
                assert_eq!(on, in_tone, "{label:?} in {:?}", f.all_text());
                assert_eq!(on, f.find(short_label(label)).is_some(), "{label:?}");
            }
        }
        assert_eq!(
            UNLENSED_STARS_LABEL,
            "Stars near the hole are not yet lensed (M3)"
        );
        // The same label from the app's notes and from the flag is drawn once.
        let mut d = data();
        d.eye_adapted_stars = true;
        d.notes
            .push((EYE_ADAPTED_STARS_LABEL.into(), LabelTone::NonPhysical));
        let f = draw(&d, c);
        let n = f
            .texts
            .iter()
            .filter(|p| p.text == EYE_ADAPTED_STARS_LABEL)
            .count();
        assert_eq!(n, 1, "once in the pop-up");
        let n = f
            .texts
            .iter()
            .filter(|p| p.text == "Eye-adapted stars")
            .count();
        assert_eq!(n, 1, "once on the line");
    }

    /// Expanded, the notes stack sits under the Location panel, in tone, 12 pt.
    #[test]
    fn notes_stack_under_location_when_expanded() {
        let mut d = data();
        d.inside_horizon = true;
        d.warnings.push("Inside Sun".into());
        let f = draw(&d, expanded());
        let kicker = f.get("LOCATION");
        let red = f.get("Inside event horizon");
        assert_eq!(red.color, crate::theme::DANGER);
        assert!(
            red.pos.x < 60.0 && red.pos.y > kicker.pos.y + 80.0,
            "{red:?}"
        );
        assert_eq!(red.font.size, 12.0);
        assert_eq!(f.get("Inside Sun").color, NON_PHYSICAL);
    }

    /// Review I5: invalid timings read "n/a", never 0.
    #[test]
    fn perf_line_na() {
        let mut d = data();
        assert_eq!(perf_line(&d), "60 fps · GPU 16.5 ms · CPU 1.0 ms");
        d.gpu_frame_ms = None;
        assert_eq!(perf_line(&d), "60 fps · GPU n/a · CPU 1.0 ms");
        d.gpu_frame_ms = Some(16.5);
        d.gpu_timing_available = false;
        assert_eq!(perf_line(&d), "60 fps · GPU n/a · CPU 1.0 ms");
        // A capture has no frame rate.
        d.fps = 0.0;
        assert_eq!(perf_line(&d), "GPU n/a · CPU 1.0 ms");
        let f = draw(&d, controls());
        assert!(f.find("GPU n/a").is_some());
    }

    /// Tab hides everything: no shape at all, whatever is open.
    #[test]
    fn hidden_hud_draws_nothing() {
        let mut d = data();
        d.target = Some(mars_card());
        d.lens_on_screen = true;
        let c = HudControls {
            visible: false,
            show_help: true,
            card_expanded: true,
            notes_expanded: true,
            show_warp: true,
            ..controls()
        };
        let ctx = headless_context();
        let mut c2 = c;
        let (mut out, hud, _) =
            run_hud_settled(&ctx, SIZE, 1.0, 8192, &d, &Overlay::default(), &mut c2);
        out.textures_delta.clear();
        assert!(out.shapes.is_empty(), "{} shapes", out.shapes.len());
        assert_eq!(hud, HudOutput::default());
        assert_eq!(c2, c);
    }

    /// T: six touch-sized buttons, the current rate's in cyan, and the mockup's line for it.
    #[test]
    fn warp_panel_offers_six_rates() {
        let c = HudControls {
            show_warp: true,
            ..controls()
        };
        let f = draw(&data(), c);
        for label in WARP_BUTTONS {
            assert!(f.texts.iter().any(|p| p.text == label), "no {label:?}");
        }
        assert!(f.find(WARP_NOTES[2]).is_some(), "{:?}", f.all_text());
        let selected = f.has_stroke(WARP_SELECTED_BORDER).expect("selected rate");
        assert!(selected.height() >= 44.0);
        let ten3 = f.texts.iter().find(|p| p.text == "10³").expect("10³");
        assert!(selected.contains(ten3.center()), "{selected:?} {ten3:?}");
        assert!(f.get("TIME WARP").pos.x > 1920.0 - 24.0 - 380.0 - 1.0);
        let mut d = data();
        d.time.paused = true;
        assert!(draw(&d, c).find(WARP_NOTES[0]).is_some());
    }

    /// Clicks: the place line expands, the card's buttons report their actions, the warp
    /// buttons pick a preset or pause, the honesty line opens its pop-up.
    #[test]
    fn clicks_drive_state_and_actions() {
        let ctx = headless_context();
        let ov = Overlay::default();
        let click = |d: &HudData, c: &mut HudControls, at: Pos2| -> HudOutput {
            let (mut o, _, _) = run_hud_settled(&ctx, SIZE, 1.0, 8192, d, &ov, c);
            o.textures_delta.clear();
            let press = |pressed| egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            let mut raw = headless_input(SIZE, 1.0, 8192);
            raw.events = vec![egui::Event::PointerMoved(at), press(true)];
            let (mut o, _) = run_hud(&ctx, raw, d, &ov, c);
            o.textures_delta.clear();
            let mut raw = headless_input(SIZE, 1.0, 8192);
            raw.events = vec![press(false)];
            let (mut o, hud) = run_hud(&ctx, raw, d, &ov, c);
            o.textures_delta.clear();
            hud
        };
        let mut d = data();
        d.target = Some(mars_card());
        let mut c = controls();
        let place = look(&ctx, 1.0, &d, &mut c).get("14 200 km away").center();
        click(&d, &mut c, place);
        assert!(c.card_expanded, "the place line expands the card");
        let f = look(&ctx, 1.0, &d, &mut c);
        let out = click(&d, &mut c, f.get("Go to · F").center());
        assert_eq!(out.actions, [CardAction::Goto]);
        let out = click(&d, &mut c, f.get("Moons").center());
        assert_eq!(out.actions, [CardAction::Moons]);
        let phobos = look(&ctx, 1.0, &d, &mut c).get("Phobos").center();
        let out = click(&d, &mut c, phobos);
        assert_eq!(out.selected_moon.as_deref(), Some("Phobos"));
        let out = click(&d, &mut c, f.get("Close · Esc").center());
        assert_eq!(out.actions, [CardAction::Close]);
        assert!(!c.card_expanded);
        c.show_warp = true;
        let f = look(&ctx, 1.0, &d, &mut c);
        let out = click(
            &d,
            &mut c,
            f.texts
                .iter()
                .find(|p| p.text == "10⁶")
                .expect("10⁶")
                .center(),
        );
        assert_eq!(out.warp_preset, Some(2));
        let out = click(&d, &mut c, f.get("Pause").center());
        assert!(out.toggle_pause);
        c.show_warp = false;
        let line = look(&ctx, 1.0, &d, &mut c).get(FREE_FLIGHT_LABEL).center();
        click(&d, &mut c, line);
        assert!(c.notes_expanded);
    }

    /// One headless draw is a finished HUD inside the target at 1 and 2 points per pixel.
    #[test]
    fn first_headless_draw_fits_the_target() {
        let mut d = data();
        d.target = Some(mars_card());
        d.lens_on_screen = true;
        d.eye_adapted_stars = true;
        for ppp in [1.0, 2.0] {
            let mut c = HudControls {
                notes_expanded: true,
                ..expanded()
            };
            let (out_ppp, f) = draw_ppp(ppp, &d, &mut c);
            assert_eq!(out_ppp, ppp);
            let screen =
                egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(1920.0 / ppp, 1080.0 / ppp));
            for needle in ["LOCATION", "Mars", "12.41 km/s", "GPU 16.5 ms"] {
                assert!(f.find(needle).is_some(), "ppp {ppp}: no {needle:?}");
            }
            for p in &f.texts {
                let r = egui::Rect::from_min_size(p.pos, p.size);
                assert!(
                    screen.expand(0.5).contains_rect(r),
                    "ppp {ppp}: {:?} at {r:?} is off {screen:?}",
                    p.text
                );
            }
        }
    }

    /// The F1 list names the M1 keys and holds the settings, the belt toggle included.
    #[test]
    fn controls_list_has_m1_keys_and_settings() {
        let c = HudControls {
            show_help: true,
            ..controls()
        };
        let f = draw(&data(), c);
        for needle in [
            "Save view bookmark (captures/views/)",
            "Go to the selection",
            "Cruise",
            "Time-warp panel",
            "Open / close the card",
            crate::theme::BELT_TOGGLE_LABEL,
            "Scene EV100",
            "Star EV100",
            "Render 1728×1080 → 2560×1600",
            "GPU tonemap n/a",
        ] {
            assert!(
                f.find(needle).is_some(),
                "no {needle:?} in {:?}",
                f.all_text()
            );
        }
        assert!(
            CONTROLS
                .iter()
                .all(|(_, action)| !action.contains("Screenshot"))
        );
    }

    /// One headless draw must not be a bare sizing pass, must end on egui's own signal rather
    /// than the cap, and once settled must cost a single egui run.
    #[test]
    fn headless_draw_settles_then_costs_one_run() {
        let ctx = headless_context();
        let d = data();
        let mut c = HudControls {
            show_help: true,
            ..expanded()
        };
        let ov = Overlay::default();
        let (mut first, _, runs) = run_hud_settled(&ctx, SIZE, 2.0, 8192, &d, &ov, &mut c);
        first.textures_delta.clear();
        assert!(
            runs > 1 && runs < HEADLESS_MAX_RUNS,
            "first draw took {runs} runs"
        );
        let (mut next, _, runs) = run_hud_settled(&ctx, SIZE, 2.0, 8192, &d, &ov, &mut c);
        next.textures_delta.clear();
        assert_eq!(runs, 1);
    }

    /// Review r0 I1: egui-winit consumes Tab always, so the app never saw ToggleHud.
    #[test]
    fn app_hotkeys_never_reach_egui_and_are_never_consumed() {
        for code in [KeyCode::Tab, KeyCode::F1, KeyCode::F12] {
            for pressed in [true, false] {
                for text_edit_focused in [true, false] {
                    assert_eq!(
                        route_key(PhysicalKey::Code(code), pressed, text_edit_focused),
                        KeyRoute::AppOnly,
                        "{code:?} pressed={pressed} text_edit_focused={text_edit_focused}"
                    );
                }
            }
        }
    }

    /// Review r0 I1: a focused slider (Tab or a click) made egui-winit consume every key.
    #[test]
    fn flying_keys_pass_through_unless_typing() {
        let flying = [
            KeyCode::KeyW,
            KeyCode::KeyA,
            KeyCode::KeyS,
            KeyCode::KeyD,
            KeyCode::KeyQ,
            KeyCode::KeyE,
            KeyCode::Space,
            KeyCode::ControlLeft,
            KeyCode::Equal,
            KeyCode::Minus,
            KeyCode::KeyF,
            KeyCode::KeyC,
            KeyCode::KeyT,
            KeyCode::KeyI,
            KeyCode::Escape,
        ];
        for code in flying {
            assert_eq!(
                route_key(PhysicalKey::Code(code), true, false),
                KeyRoute::Shared { consumed: false },
                "{code:?}"
            );
        }
        // Typing "-2" into an exposure field must not also slow the camera.
        for code in [KeyCode::Minus, KeyCode::Digit2, KeyCode::KeyW] {
            assert_eq!(
                route_key(PhysicalKey::Code(code), true, true),
                KeyRoute::Shared { consumed: true },
                "{code:?}"
            );
        }
    }

    #[test]
    fn key_releases_are_never_consumed() {
        for text_edit_focused in [true, false] {
            assert_eq!(
                route_key(PhysicalKey::Code(KeyCode::KeyW), false, text_edit_focused),
                KeyRoute::Shared { consumed: false }
            );
        }
    }

    #[test]
    fn mouse_button_releases_are_never_consumed() {
        let button = |state| WindowEvent::MouseInput {
            device_id: DeviceId::dummy(),
            state,
            button: MouseButton::Right,
        };
        // Releasing mouse look over the HUD must still end mouse look.
        assert!(!pointer_consumed(&button(ElementState::Released), true));
        // A press on the HUD is the HUD's.
        assert!(pointer_consumed(&button(ElementState::Pressed), true));
        assert!(!pointer_consumed(&button(ElementState::Pressed), false));
        let left = WindowEvent::CursorLeft {
            device_id: DeviceId::dummy(),
        };
        assert!(pointer_consumed(&left, true));
    }

    /// Review r0 I2: a headless HUD is laid out for the capture size, not a window's.
    #[test]
    fn headless_input_uses_the_target_size() {
        let raw = headless_input([1920, 1080], 2.0, 8192);
        let points = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(960.0, 540.0));
        assert_eq!(raw.screen_rect, Some(points));
        assert_eq!(raw.viewport().native_pixels_per_point, Some(2.0));
        assert_eq!(raw.max_texture_side, Some(8192));
    }

    #[test]
    fn headless_input_guards_size_and_scale() {
        // As egui-winit: a zero-sized target has no screen rect.
        assert_eq!(headless_input([0, 1080], 1.0, 8192).screen_rect, None);
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let raw = headless_input([100, 50], bad, 8192);
            assert_eq!(raw.viewport().native_pixels_per_point, Some(1.0), "{bad}");
            assert_eq!(
                raw.screen_rect.map(|r| r.size()),
                Some(egui::vec2(100.0, 50.0)),
                "{bad}"
            );
        }
    }

    /// The HUD's contexts draw in the mockup's faces.
    #[test]
    fn hud_contexts_use_the_mockup_fonts() {
        let f = draw(&data(), controls());
        let ours = |fam: &FontFamily| match fam {
            FontFamily::Name(n) => n.starts_with(FONT_TEXT) || n.starts_with(FONT_MONO),
            _ => false,
        };
        let others: Vec<&Painted> = f.texts.iter().filter(|p| !ours(&p.font.family)).collect();
        assert!(others.is_empty(), "{others:?}");
    }
}
