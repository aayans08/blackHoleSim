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
    self, BELT_HIGHLIGHT_LABEL, BELT_TOGGLE_LABEL, CARD_PADDING_Y, EYE_ADAPTED_STARS_LABEL,
    GRAVITY_WIDTH, HINT_CARD, HINT_COLLAPSED, INK_2, INK_3, INK_SOFT, INSIDE_HORIZON_LABEL,
    KICKER_TRACKING_EM, LOCATION_WIDTH, LabelTone, MUTED, NON_PHYSICAL_SOFT, NOTE_DOT,
    NOTES_LINE_MAX_WIDTH, NOTES_POPUP_TITLE, NOTES_POPUP_WIDTH, NOTES_SIZE, SCREEN_MARGIN,
    SMALL_SIZE, SPEED_WIDTH, TIME_LINE_SIZE, UNLENSED_STARS_LABEL, Weight, mono_font, short_label,
    text_font, tone_color,
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

/// The app's own hotkeys in uni-nav's key map (Task 6): Tab hides the HUD, F1 shows the
/// controls list, F12 saves the view as a bookmark. Matched by physical key, as uni-nav does.
const APP_HOTKEYS: [KeyCode; 3] = [KeyCode::Tab, KeyCode::F1, KeyCode::F12];

/// Most egui runs per headless draw. egui lays out a new area invisibly on its first run (a
/// sizing pass) and draws it from the second (measured with egui 0.37;
/// `headless_draw_settles_then_costs_one_run`). The rest is margin.
const HEADLESS_MAX_RUNS: usize = 4;

/// Room kept free at the bottom right for the perf line and hints [pt].
const PERF_RESERVE: f32 = 40.0;
/// Gap between neighbouring panels [pt]; also the least gap a notes block keeps from the
/// other HUD blocks ([`rise`], [`fall`]).
const PANEL_GAP: f32 = 16.0;
/// The window's least inner size [pt] (lead ruling C4w-I1; uni-app sets it on the window).
/// Chosen from the C4w size sweep (`honesty_notes_stay_whole_on_screen`):
/// - 960 pt keeps the honesty line beside the perf block (its 560 pt least width and the perf
///   block need about 880 pt), the 520 pt Speed panel on screen, and Location clear of the
///   centred time line while the card is open;
/// - 600 pt leaves the expanded notes stack, between Location and the Gravity and Speed panels,
///   206 pt, room for the most honesty notes M1 raises at once (nine, 192 pt with their wrapped
///   lines; at 540 pt the room is 146 pt).
pub const MIN_WINDOW_PT: [f64; 2] = [960.0, 600.0];
/// Gap from the honesty line's top to its pop-up's bottom [pt] (C4's geometry: 6 pt over the
/// line's area, which hangs 4 pt lower than the margin).
const POPUP_GAP: f32 = 10.0;
/// Gap between the perf line and the key hints under it [pt].
const PERF_LINE_GAP: f32 = 4.0;

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
    /// True when a lens is traced on screen this frame, in either lens environment (the second
    /// flag of `uni_rel::lens_labels`): the notes then hold, in amber, "Stars near the hole are
    /// not yet lensed (M3)". Sky views skip the pass, so the line would not be true there. The
    /// flag never adds "Lens background: M0 debug grid": that label depends on the lens
    /// environment (M1's default lenses the band), which only the app knows, so the app puts it
    /// in `notes` (the first flag of `uni_rel::lens_labels`; review C4 r0 I1).
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
    /// `uni_sim::honesty`, UI labels from `theme::ui_labels`, the debug-grid label, renderer
    /// warnings), in any order. The HUD shows "Inside event horizon" first, then the flight
    /// label (the note equal to `speed.header`, added from the Speed panel if missing here), so
    /// the collapsed line always holds it however narrow the window (ruling 6; review C4 r0
    /// I2); then these notes in their order, then the labels of its own flags above (each
    /// label is shown once).
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
///
/// Layout (C4w): the top blocks first, then the bottom ones, then the notes. egui places an
/// anchored area from the size it had on its last pass, so a bottom block that grew since (a
/// longer note list, a wrapped note, the FTL line) was drawn too low, over the block under it;
/// a capture, which draws each bookmark once on one HUD, kept it there. Every bottom block (and
/// the centred time line) is therefore placed from its size measured this pass ([`measure`],
/// [`show_measured`]), and the notes blocks (the honesty line, its pop-up, the stack under
/// Location) keep [`PANEL_GAP`] clear of every block drawn before them ([`rise`], [`fall`]).
/// A pop-up or stack with too little room keeps its honesty notes whole and first, and scrolls
/// or leaves out the others ([`plan_notes`]; lead ruling C4w-I1).
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
    // Read once: a click below may close the card, which takes effect on the next pass.
    let expanded = c.card_expanded;
    let card = data.target.as_ref().filter(|_| expanded);
    // The blocks drawn so far; the notes keep clear of them.
    let mut taken: Vec<egui::Rect> = Vec::new();

    // The top row.
    if let Some(card) = card {
        let rect = area("card", Order::Middle)
            .anchor(Align2::RIGHT_TOP, [-m, m])
            .show(ctx, |ui| {
                // The header (≈ 90 pt) and the padding stay; the body scrolls.
                let max_body =
                    screen.height() - 2.0 * m - 2.0 * CARD_PADDING_Y - 2.0 - 90.0 - PERF_RESERVE;
                let clicks = show_card(ui, card, max_body.max(80.0));
                if clicks.actions.contains(&CardAction::Close) {
                    c.card_expanded = false;
                }
                out.actions.extend(clicks.actions);
                if clicks.selected_moon.is_some() {
                    out.selected_moon = clicks.selected_moon;
                }
                ui.is_sizing_pass()
            });
        taken.extend(shown_rect(&rect));
    }
    if !expanded {
        let rect = area("place", Order::Middle)
            .anchor(Align2::LEFT_TOP, [m, m])
            .show(ctx, |ui| {
                let halo = ui.painter().add(egui::Shape::Noop);
                let line = panels::show_place(ui, &data.nav_line, false);
                ui.painter()
                    .set(halo, theme::halo_shape(line.rect.expand(6.0)));
                if line.clicked() {
                    c.card_expanded = true;
                }
                ui.is_sizing_pass()
            });
        taken.extend(shown_rect(&rect));
    }

    // Time: the warp panel (T), the Time panel (expanded, no card) or the one-line time.
    let time_anchor = if card.is_some() {
        (Align2::CENTER_TOP, [0.0, m])
    } else {
        (Align2::RIGHT_TOP, [-m, m])
    };
    let time = if c.show_warp {
        let warp = area("warp", Order::Middle)
            .anchor(time_anchor.0, time_anchor.1)
            .show(ctx, |ui| {
                match panels::show_warp(ui, &data.time) {
                    Some(WarpPick::Pause) => out.toggle_pause = true,
                    Some(WarpPick::Preset(i)) => out.warp_preset = Some(i),
                    None => {}
                }
                ui.is_sizing_pass()
            });
        shown_rect(&warp)
    } else if expanded && card.is_none() {
        let panel = area("time", Order::Middle)
            .anchor(Align2::RIGHT_TOP, [-m, m])
            .show(ctx, |ui| {
                panels::show_time(ui, &data.time);
                ui.is_sizing_pass()
            });
        shown_rect(&panel)
    } else {
        // Centred from its width this pass: a new rate changes it (review C4w M4).
        let time_line = panels::time_line(&data.time);
        let line = Measured {
            id: "time line",
            order: Order::Middle,
            align: Align2::CENTER_TOP,
            at: egui::pos2(screen.center().x, screen.top() + m),
            max_width: screen.width(),
            interactable: false,
        };
        Some(show_measured(ctx, line, |ui| {
            let font = mono_font(TIME_LINE_SIZE, Weight::Regular);
            haloed(ui, |ui| theme::text(ui, &time_line, font, INK_SOFT));
        }))
    };
    taken.extend(time);

    // The bottom row, each block placed from its size this pass.
    let hint = if card.is_some() {
        HINT_CARD
    } else {
        HINT_COLLAPSED
    };
    let perf_lines = [perf_line(data), hint.to_owned()];
    let perf = show_measured(
        ctx,
        Measured {
            id: "perf",
            order: Order::Middle,
            align: Align2::RIGHT_BOTTOM,
            at: screen.right_bottom() + egui::vec2(-m, -m + 2.0),
            max_width: screen.width(),
            interactable: false,
        },
        |ui| perf_block(ui, &perf_lines),
    );
    taken.push(perf);

    let left = screen.left() + m;
    if expanded {
        if let Some(g) = &data.gravity {
            let rect = show_measured(
                ctx,
                Measured::bottom_left("gravity", egui::pos2(left, screen.bottom() - m)),
                |ui| panels::show_gravity(ui, g),
            );
            taken.push(rect);
        }
        // Centred, but clear of the Gravity panel on a narrow screen.
        let mut speed_left = screen.center().x - SPEED_WIDTH / 2.0;
        if data.gravity.is_some() {
            speed_left = speed_left.max(left + GRAVITY_WIDTH + PANEL_GAP);
        }
        let rect = show_measured(
            ctx,
            Measured::bottom_left("speed", egui::pos2(speed_left, screen.bottom() - m)),
            |ui| panels::show_speed(ui, &data.speed),
        );
        taken.push(rect);

        // Anchored at its top left, so its rect holds even on egui's sizing pass.
        let location = area("location", Order::Middle)
            .anchor(Align2::LEFT_TOP, [m, m])
            .show(ctx, |ui| {
                if panels::show_location(ui, &data.location) {
                    c.card_expanded = false;
                }
            })
            .response
            .rect;
        // The notes stack under Location (12 pt, in tone; the flight label is the Speed
        // header), 11 pt under it as in C4 (egui's item spacing and 8 pt). It ends
        // `PANEL_GAP` over the Gravity panel (or any block below it); with less room, its
        // honesty notes stay whole and the others scroll or are left out ([`plan_notes`]).
        let stack = block_notes(&notes, &data.warnings, Some(&data.speed.header));
        if !stack.is_empty() {
            let height = |fit| {
                measure(ctx, "notes stack", LOCATION_WIDTH, |ui| {
                    notes_stack(ui, &stack, fit)
                })
                .y
            };
            let all = height(NotesFit::All);
            let x = egui::Rangef::new(left, left + LOCATION_WIDTH);
            let top = location.bottom() + ctx.global_style().spacing.item_spacing.y + 8.0;
            let slot = fall(x, top, all, screen.bottom() - m, &taken, PANEL_GAP);
            let (fit, _) = plan_notes(slot.height, all, || height(NotesFit::HONESTY_ONLY));
            placed("notes stack", Order::Middle, egui::pos2(left, slot.top)).show(ctx, |ui| {
                ui.set_max_width(LOCATION_WIDTH);
                notes_stack(ui, &stack, fit);
            });
        }
    } else if !notes.is_empty() {
        // The honesty line, bottom left: beside the perf block when the line's least width fits
        // there, else over it (a narrow window).
        let beside = perf.left() - PANEL_GAP - left;
        let mut line_w = NOTES_LINE_MAX_WIDTH.max(screen.width() - 2.0 * m - 320.0);
        if beside >= NOTES_LINE_MAX_WIDTH {
            line_w = line_w.min(beside);
        }
        let size = measure(ctx, "notes line", line_w, |ui| {
            notes_line(ui, &notes, line_w);
        });
        let x = egui::Rangef::new(left, left + size.x);
        let bottom = screen.bottom() - m + 4.0;
        let slot = rise(x, bottom, size.y, screen.top() + m, &taken, PANEL_GAP);
        // One line, always drawn whole from the slot's top (it holds the flight label, ruling
        // 6): a screen with less than a line of room, far under the window's least size
        // (`MIN_WINDOW_PT`), would see it over the block that lifted it (review C4w M2).
        let line = placed("notes", Order::Middle, egui::pos2(left, slot.top))
            .show(ctx, |ui| {
                let line = notes_line(ui, &notes, line_w);
                if line.clicked() {
                    c.notes_expanded = !c.notes_expanded;
                }
                line.rect
            })
            .inner;
        // Its pop-up stands on it, from its height this pass, and stops under the top row;
        // with less room, its honesty notes stay whole and the others scroll or are left out.
        if c.notes_expanded {
            let popup = block_notes(&notes, &data.warnings, None);
            let size = |fit| {
                measure(ctx, "notes popup", NOTES_POPUP_WIDTH, |ui| {
                    notes_popup(ui, &popup, fit)
                })
            };
            let all = size(NotesFit::All);
            let x = egui::Rangef::new(left, left + all.x);
            let bottom = line.top() - POPUP_GAP;
            let slot = rise(x, bottom, all.y, screen.top() + m, &taken, PANEL_GAP);
            let (fit, h) = plan_notes(slot.height, all.y, || size(NotesFit::HONESTY_ONLY).y);
            let top = slot.top + slot.height - h;
            placed("notes popup", Order::Foreground, egui::pos2(left, top))
                .show(ctx, |ui| notes_popup(ui, &popup, fit));
        }
    }

    if c.show_help {
        show_controls_window(ctx, data, c);
    }
}

/// The rect of a top-row area just shown, whose closure returned `ui.is_sizing_pass()`. None
/// on egui's first, invisible pass of a new area: egui then places it from a guessed size (its
/// default area size), so its rect is wrong for a right or centre anchor (the card at 800 pt
/// came out 180 pt left of its place), and the notes would make room for a block not there.
fn shown_rect(r: &egui::InnerResponse<bool>) -> Option<egui::Rect> {
    (!r.inner).then_some(r.response.rect)
}

/// A fixed HUD area with its top left at `left_top`, for a block the HUD placed from its size
/// this pass: not pushed back on screen by its size on the last pass, as egui's constraint
/// would (the placement keeps it on screen).
fn placed(id: &str, order: Order, left_top: egui::Pos2) -> egui::Area {
    area(id, order).fixed_pos(left_top).constrain(false)
}

/// Where [`show_measured`] puts a block: its `align` corner at `at`.
struct Measured {
    id: &'static str,
    order: Order,
    align: Align2,
    at: egui::Pos2,
    /// The width the block may lay out to [pt].
    max_width: f32,
    interactable: bool,
}

impl Measured {
    /// A panel at the bottom left, its bottom-left corner at `at`.
    fn bottom_left(id: &'static str, at: egui::Pos2) -> Self {
        Self {
            id,
            order: Order::Middle,
            align: Align2::LEFT_BOTTOM,
            at,
            max_width: SPEED_WIDTH,
            interactable: true,
        }
    }
}

/// Shows `add` in a fixed area placed by `at` from the size `add` lays out to this pass
/// ([`measure`]), so a block that grew or shrank since the last pass is never drawn at its old
/// size (C4w). Returns the area's rect.
fn show_measured(
    ctx: &egui::Context,
    at: Measured,
    mut add: impl FnMut(&mut egui::Ui),
) -> egui::Rect {
    let size = measure(ctx, at.id, at.max_width, &mut add);
    let corner = egui::vec2(
        at.align.x().to_factor() * size.x,
        at.align.y().to_factor() * size.y,
    );
    placed(at.id, at.order, at.at - corner)
        .interactable(at.interactable)
        .show(ctx, |ui| {
            // The width it was measured at, not the width egui stored for the area last pass,
            // so text wraps as it did when measured.
            ui.set_max_width(at.max_width);
            add(ui)
        })
        .response
        .rect
}

/// The size `add` lays out to this pass, at most `max_width` wide: it runs in an invisible,
/// disabled `Ui` far off screen, so it paints nothing and no pointer reaches it. The text it
/// lays out is cached, so drawing it again costs little.
fn measure(
    ctx: &egui::Context,
    id: &str,
    max_width: f32,
    add: impl FnOnce(&mut egui::Ui),
) -> egui::Vec2 {
    let rect = egui::Rect::from_min_size(
        egui::pos2(-MEASURE_OFFSET, -MEASURE_OFFSET),
        egui::vec2(max_width.max(1.0), MEASURE_OFFSET / 2.0),
    );
    let mut ui = egui::Ui::new(
        ctx.clone(),
        Id::unique(("hud measure", id)),
        egui::UiBuilder::new().max_rect(rect).invisible(),
    );
    add(&mut ui);
    ui.min_size()
}

/// How far up and left of the screen [`measure`] lays its blocks out [pt]: more than any HUD
/// block is wide or tall, small enough for f32 to keep sub-point positions (2⁻⁹ pt).
const MEASURE_OFFSET: f32 = 20_000.0;

/// One note of a notes block (the pop-up or the stack): its sentence and tone, and whether it
/// is an honesty note, which no block scrolls out of view or leaves out (lead ruling C4w-I1).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Note<'a> {
    text: &'a str,
    tone: LabelTone,
    honesty: bool,
}

/// The notes a block lists: every note but `skip` (expanded, the flight label, which the Speed
/// header shows), in order. Honesty notes are the non-physical and danger ones and the
/// renderer warnings (`HudData::warnings`, whatever tone the app gave the same text).
fn block_notes<'a>(
    notes: &'a [(String, LabelTone)],
    warnings: &[String],
    skip: Option<&str>,
) -> Vec<Note<'a>> {
    notes
        .iter()
        .filter(|(t, _)| Some(t.as_str()) != skip)
        .map(|(text, tone)| Note {
            text,
            tone: *tone,
            honesty: matches!(tone, LabelTone::NonPhysical | LabelTone::Danger)
                || warnings.contains(text),
        })
        .collect()
}

/// How a notes block lays out its notes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum NotesFit {
    /// Every note, in order.
    All,
    /// Its honesty notes first, whole, in order; then the others in a scroll area at most
    /// `room` tall, or none when `room` is 0.
    HonestyFirst { room: f32 },
}

impl NotesFit {
    /// The honesty notes alone (the block's least height).
    const HONESTY_ONLY: Self = Self::HonestyFirst { room: 0.0 };
}

/// The vertical gap between a block's note rows [pt] (also before its scroll area).
const NOTE_ROW_GAP: f32 = 4.0;
/// The least scroll area for the informational notes [pt]: a 12 pt row is about 15 pt, so
/// less would show a sliver; they are then left out.
const MIN_SCROLL_ROOM: f32 = 24.0;

/// How a notes block whose notes take `all` points fits `room`, and the height it is drawn at
/// (lead ruling C4w-I1: honesty notes are never scrolled out of view or dropped):
/// - it fits: every note in order, `all` tall;
/// - else its honesty notes (`honesty_only()` tall with the block's chrome) are drawn whole and
///   first, and the informational ones scroll in what is left, or are left out when that is
///   under [`MIN_SCROLL_ROOM`]. Honesty notes taller than `room` are drawn whole all the same
///   (taller than the slot; the window's least size, [`MIN_WINDOW_PT`], keeps M1's worst set
///   inside it).
fn plan_notes(room: f32, all: f32, honesty_only: impl FnOnce() -> f32) -> (NotesFit, f32) {
    if all <= room + 0.5 {
        return (NotesFit::All, all);
    }
    let honesty = honesty_only();
    if honesty + 0.5 >= all {
        // Nothing informational to give up.
        return (NotesFit::All, all);
    }
    let rest = room - honesty - NOTE_ROW_GAP;
    if rest >= MIN_SCROLL_ROOM {
        (NotesFit::HonestyFirst { room: rest }, room)
    } else {
        (NotesFit::HONESTY_ONLY, honesty)
    }
}

/// Draws a block's note rows with `row`, as `fit` says.
fn note_rows(
    ui: &mut egui::Ui,
    id: &str,
    notes: &[Note<'_>],
    fit: NotesFit,
    row: impl Fn(&mut egui::Ui, &Note<'_>),
) {
    match fit {
        NotesFit::All => notes.iter().for_each(|n| row(ui, n)),
        NotesFit::HonestyFirst { room } => {
            notes.iter().filter(|n| n.honesty).for_each(|n| row(ui, n));
            if room > 0.0 {
                egui::ScrollArea::vertical()
                    .id_salt(("hud notes", id))
                    .max_height(room)
                    .show(ui, |ui| {
                        notes.iter().filter(|n| !n.honesty).for_each(|n| row(ui, n));
                    });
            }
        }
    }
}

/// A block's place on the screen's vertical axis [pt]: its top edge and the height it gets,
/// which is less than its own only when it has no room ([`plan_notes`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Slot {
    top: f32,
    height: f32,
}

/// The blocks of `taken` that share columns with `x`.
fn in_columns(taken: &[egui::Rect], x: egui::Rangef) -> impl Iterator<Item = &egui::Rect> {
    taken
        .iter()
        .filter(move |r| r.max.x > x.min && r.min.x < x.max)
}

/// Places a block `h` tall in the columns `x`, standing on `bottom` (its bottom edge), clear of
/// the blocks in `taken` by `gap`:
/// - a block in those columns that reaches across the bottom edge lifts it to `gap` over that
///   block (again for any block met there);
/// - the lowest block wholly above, or `ceiling`, caps its top `gap` under it. A block taller
///   than the room gets the room (never less than 0).
fn rise(
    x: egui::Rangef,
    bottom: f32,
    h: f32,
    ceiling: f32,
    taken: &[egui::Rect],
    gap: f32,
) -> Slot {
    let mut bottom = bottom;
    // Each block lifts it at most once: after that the edge is over that block.
    for _ in 0..=taken.len() {
        match in_columns(taken, x).find(|r| r.min.y - gap < bottom && r.max.y + gap > bottom) {
            Some(r) => bottom = r.min.y - gap,
            None => break,
        }
    }
    let top = in_columns(taken, x)
        .filter(|r| r.max.y + gap <= bottom)
        .map(|r| r.max.y + gap)
        .fold(ceiling, f32::max);
    let height = h.min(bottom - top).max(0.0);
    Slot {
        top: bottom - height,
        height,
    }
}

/// [`rise`] downward: a block `h` tall in the columns `x` hanging from `top`, pushed down past
/// any block of `taken` across that edge, its bottom capped `gap` over the highest block wholly
/// below, or at `floor`.
fn fall(x: egui::Rangef, top: f32, h: f32, floor: f32, taken: &[egui::Rect], gap: f32) -> Slot {
    let mut top = top;
    for _ in 0..=taken.len() {
        match in_columns(taken, x).find(|r| r.min.y - gap < top && r.max.y + gap > top) {
            Some(r) => top = r.max.y + gap,
            None => break,
        }
    }
    let bottom = in_columns(taken, x)
        .filter(|r| r.min.y - gap >= top)
        .map(|r| r.min.y - gap)
        .fold(floor, f32::min);
    Slot {
        top,
        height: h.min(bottom - top).max(0.0),
    }
}

/// The perf line over the key hints, both right-aligned on one halo (bottom right). It is laid
/// out at its own width, so its area's rect is the block itself (a right-aligned egui layout
/// would stretch the area to its last width).
fn perf_block(ui: &mut egui::Ui, lines: &[String; 2]) {
    let font = mono_font(SMALL_SIZE, Weight::Regular);
    let galleys = lines.each_ref().map(|s| {
        ui.painter()
            .layout_job(theme::job(s, font.clone(), INK_3, 0.0, f32::INFINITY))
    });
    let width = galleys.iter().map(|g| g.size().x).fold(0.0, f32::max);
    let height = galleys.iter().map(|g| g.size().y).sum::<f32>() + PERF_LINE_GAP;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter();
    painter.add(theme::halo_shape(rect.expand(6.0)));
    let mut y = rect.top();
    for galley in galleys {
        let size = galley.size();
        painter.galley(egui::pos2(rect.right() - size.x, y), galley, INK_3);
        y += size.y + PERF_LINE_GAP;
    }
}

/// Paints `add` and a soft halo behind it.
fn haloed(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    let halo = ui.painter().add(egui::Shape::Noop);
    let rect = ui.scope(add).response.rect;
    ui.painter().set(halo, theme::halo_shape(rect.expand(6.0)));
}

/// The notes in their order: "Inside event horizon" (red), the flight label, the app's notes,
/// then the labels of the HUD's own flags (the belt toggle, eye-adapted stars, the traced lens)
/// and the renderer warnings (amber); each text once.
///
/// The flight label is the Speed panel's header (`FREE FLIGHT · GRAVITY NOT SIMULATED` or
/// `CRUISE · …`), in the tone the app's matching note gives it, else the panel's. Collapsed,
/// the Speed panel is hidden and the honesty line is the only place it shows; [`notes_line`]
/// always keeps the line's first item and the first two always fit its least width, so the
/// label stays on screen however many notes the app lists (ruling 6; review C4 r0 I2).
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
    let flight = data.speed.header.as_str();
    if !flight.is_empty() {
        let tone = data
            .notes
            .iter()
            .find(|(t, _)| t == flight)
            .map_or(data.speed.tone, |(_, tone)| *tone);
        push(&mut notes, flight, tone);
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
    // A traced lens: the stars over it are unlensed until M3. The debug-grid label is the
    // app's, in `data.notes` (it knows the lens environment; review C4 r0 I1).
    if data.lens_on_screen {
        push(&mut notes, UNLENSED_STARS_LABEL, LabelTone::NonPhysical);
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

/// "What is approximate here": every note in full, in its tone, with its dot, as `fit` says.
fn notes_popup(ui: &mut egui::Ui, notes: &[Note<'_>], fit: NotesFit) {
    theme::panel_frame(theme::HAIRLINE)
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_width(panels::inner_width(NOTES_POPUP_WIDTH, 14.0));
            ui.spacing_mut().item_spacing = egui::vec2(8.0, NOTE_ROW_GAP);
            theme::kicker(
                ui,
                NOTES_POPUP_TITLE,
                INK_3,
                Weight::Medium,
                KICKER_TRACKING_EM,
            );
            ui.add_space(2.0);
            note_rows(ui, "notes popup", notes, fit, |ui, n| {
                ui.horizontal_top(|ui| {
                    let (dot, _) = ui.allocate_exact_size(
                        egui::vec2(NOTE_DOT, NOTES_SIZE + 2.0),
                        egui::Sense::hover(),
                    );
                    ui.painter()
                        .circle_filled(dot.center(), NOTE_DOT / 2.0, tone_color(n.tone));
                    theme::text(
                        ui,
                        n.text,
                        text_font(NOTES_SIZE, Weight::Regular),
                        tone_color(n.tone),
                    );
                });
            });
        });
}

/// The notes stack under the Location panel: each note in full, 12 pt, in its tone, as `fit`
/// says.
fn notes_stack(ui: &mut egui::Ui, notes: &[Note<'_>], fit: NotesFit) {
    haloed(ui, |ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, NOTE_ROW_GAP);
        note_rows(ui, "notes stack", notes, fit, |ui, n| {
            theme::text(
                ui,
                n.text,
                text_font(NOTES_SIZE, Weight::Regular),
                tone_color(n.tone),
            );
        });
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
        /// The clip rect it was painted with (a scroll area's viewport clips its rows).
        clip: egui::Rect,
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
        fn walk(shape: &egui::Shape, clip: egui::Rect, f: &mut Frame) {
            match shape {
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, clip, f)),
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
                        clip,
                    });
                }
                egui::Shape::Rect(r) => f.rects.push((r.rect, r.fill, r.stroke.color)),
                egui::Shape::Circle(c) => f.circles.push((c.center, c.fill)),
                _ => {}
            }
        }
        let mut f = Frame::default();
        for clipped in &out.shapes {
            walk(&clipped.shape, clipped.clip_rect, &mut f);
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

    /// Review m13: the eye-adapted label exactly while it is in effect; the unlensed-stars label
    /// exactly while a lens is traced (the "(M3)" wording). The debug-grid label is not the
    /// flag's (`lens_flag_never_claims_the_debug_grid`).
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
            for label in [EYE_ADAPTED_STARS_LABEL, UNLENSED_STARS_LABEL] {
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

    /// Review C4 r0 I1 (plan table "Honesty labels in M1"): `lens_on_screen` means a lens is
    /// traced, in either lens environment. M1's default environment is the band, so the flag
    /// alone must never claim "Lens background: M0 debug grid"; that label depends on the
    /// environment, which only the app knows (`uni_rel::lens_labels`), and arrives in `notes`.
    #[test]
    fn lens_flag_never_claims_the_debug_grid() {
        let c = HudControls {
            notes_expanded: true,
            ..controls()
        };
        // A band-lensed hole on screen: only the unlensed-stars label.
        let mut d = data();
        d.lens_on_screen = true;
        let texts: Vec<String> = hud_notes(&d, &c).into_iter().map(|(t, _)| t).collect();
        assert!(texts.iter().any(|t| t == UNLENSED_STARS_LABEL), "{texts:?}");
        assert!(!texts.iter().any(|t| t == DEBUG_GRID_LABEL), "{texts:?}");
        let f = draw(&d, c);
        assert!(f.find(DEBUG_GRID_LABEL).is_none(), "{:?}", f.all_text());
        assert!(
            f.texts
                .iter()
                .any(|p| p.text == UNLENSED_STARS_LABEL && p.color == NON_PHYSICAL),
            "{:?}",
            f.all_text()
        );
        // The debug environment: the app lists the grid label, which the HUD shows in amber.
        d.notes
            .push((DEBUG_GRID_LABEL.into(), LabelTone::NonPhysical));
        let f = draw(&d, c);
        assert!(
            f.texts
                .iter()
                .any(|p| p.text == DEBUG_GRID_LABEL && p.color == NON_PHYSICAL),
            "{:?}",
            f.all_text()
        );
        // Without a traced lens and without the app's note: neither label.
        d.notes.retain(|(t, _)| t != DEBUG_GRID_LABEL);
        d.lens_on_screen = false;
        let f = draw(&d, c);
        assert!(f.find(DEBUG_GRID_LABEL).is_none());
        assert!(f.find(short_label(UNLENSED_STARS_LABEL)).is_none());
    }

    /// A settled draw of `d` on a fresh headless HUD at `size_px` and `ppp`.
    fn draw_at(size_px: [u32; 2], ppp: f32, d: &HudData, mut c: HudControls) -> Frame {
        let ctx = headless_context();
        let (mut out, _hud, _runs) =
            run_hud_settled(&ctx, size_px, ppp, 8192, d, &Overlay::default(), &mut c);
        out.textures_delta.clear();
        collect(&out)
    }

    /// Review C4 r0 I2 (ruling 6, the lead's collapsed-HUD ruling): collapsed, the honesty line
    /// is the only place the flight label shows, so it leads the line (after the red "Inside
    /// event horizon") however many world labels the app lists before it, and survives the "…"
    /// cut on the owner's 1280–1470 pt wide window (the M3 at 2560 × 1664 px, 2 px per point,
    /// and macOS's default 1470 × 956 pt) and at the line's least width (560 pt, an 800 pt
    /// window), with the longer cruise label behind the red one. It leads even when the app's
    /// notes lack it.
    #[test]
    fn flight_label_leads_a_narrow_notes_line() {
        use crate::theme::{SATURN_MOONS_LABEL, VIVID_STAR_COLOURS_LABEL};
        use uni_sim::honesty::{HonestyFlag, label};
        // A typical M1 set of world labels, listed by the app before its UI labels.
        let world: Vec<(String, LabelTone)> = [
            (label(HonestyFlag::BandModel), LabelTone::Neutral),
            (
                label(HonestyFlag::MilkyWayDetailProcedural),
                LabelTone::Neutral,
            ),
            (VIVID_STAR_COLOURS_LABEL, LabelTone::NonPhysical),
            (label(HonestyFlag::ProceduralBelt), LabelTone::Neutral),
            (SATURN_MOONS_LABEL, LabelTone::Neutral),
            (
                label(HonestyFlag::PlanetsExtrapolated),
                LabelTone::NonPhysical,
            ),
        ]
        .into_iter()
        .map(|(s, t)| (s.to_owned(), t))
        .collect();
        let band = short_label(label(HonestyFlag::BandModel));
        for (flight, header) in [
            (FlightLabel::Free, FREE_FLIGHT_LABEL),
            (FlightLabel::Cruise, CRUISE_LABEL),
        ] {
            for (horizon, with_ui_labels) in [(false, true), (true, true), (false, false)] {
                let mut d = data();
                d.speed.header = header.into();
                d.inside_horizon = horizon;
                d.notes = world.clone();
                if with_ui_labels {
                    d.notes.extend(
                        ui_labels(flight, true, true)
                            .into_iter()
                            .map(|(s, t)| (s.to_owned(), t)),
                    );
                }
                let case = format!("{header:?}, horizon {horizon}, ui_labels {with_ui_labels}");
                // The order: red horizon, the flight label, then the app's notes in order.
                let notes = hud_notes(&d, &controls());
                let lead = usize::from(horizon);
                if horizon {
                    assert_eq!(
                        notes[0],
                        (INSIDE_HORIZON_LABEL.to_owned(), LabelTone::Danger),
                        "{case}"
                    );
                }
                assert_eq!(
                    notes[lead],
                    (header.to_owned(), LabelTone::NonPhysical),
                    "{case}: {notes:?}"
                );
                assert_eq!(notes[lead + 1..lead + 1 + world.len()], world[..], "{case}");
                assert_eq!(
                    notes.iter().filter(|(t, _)| t == header).count(),
                    1,
                    "{case}"
                );
                for (w, h) in [(800_u32, 600_u32), (1280, 832), (1470, 956)] {
                    // 800 pt: the line is at its least width, `NOTES_LINE_MAX_WIDTH` (560 pt;
                    // 800 − 2 × 24 − 320 = 432 is less).
                    let f = draw_at([2 * w, 2 * h], 2.0, &d, controls());
                    let case = format!("{case}, {w} × {h} pt");
                    // The line really is cut here, so the test exercises the "…".
                    let more = f
                        .texts
                        .iter()
                        .find(|p| p.text == " …")
                        .unwrap_or_else(|| panic!("{case}: no cut in {:?}", f.all_text()));
                    let shown = f
                        .texts
                        .iter()
                        .find(|p| p.text == header)
                        .unwrap_or_else(|| panic!("{case}: no flight label in {:?}", f.all_text()));
                    // The bottom row, beside the perf block. At 800 pt a cut line too long for
                    // the room beside it sits over the perf block instead, never on it (C4w).
                    let perf = f.get("60 fps · GPU 16.5 ms");
                    let perf_left = perf.pos.x.min(f.get(HINT_COLLAPSED).pos.x);
                    let line_right = line_texts(&f)
                        .iter()
                        .map(|p| p.pos.x + p.size.x)
                        .fold(0.0, f32::max);
                    let beside =
                        shown.pos.y > h as f32 - 60.0 && line_right <= perf_left - PANEL_GAP + 1.0;
                    let over = shown.pos.y + shown.size.y <= perf.pos.y - PANEL_GAP + 1.0
                        && shown.pos.y > h as f32 - 100.0;
                    assert!(
                        beside || (w == 800 && over),
                        "{case}: {shown:?}, line right {line_right}, {perf:?}"
                    );
                    let overlaps = overlapping_texts(&f);
                    assert!(overlaps.is_empty(), "{case}: {overlaps:?}");
                    assert!(
                        shown.pos.x + shown.size.x <= more.pos.x,
                        "{case}: {shown:?}"
                    );
                    if horizon {
                        let red = f.get(INSIDE_HORIZON_LABEL);
                        assert!(red.pos.x < shown.pos.x, "{case}: {red:?} {shown:?}");
                    } else {
                        // First on the line: just after its amber dot.
                        assert!(
                            shown.pos.x <= SCREEN_MARGIN + NOTE_DOT + 6.0 + 1.0,
                            "{case}: {shown:?}"
                        );
                    }
                    // The app's notes follow, in its order (the first one may be cut at 560 pt).
                    match f.find(band) {
                        Some(p) => assert!(p.pos.x > shown.pos.x, "{case}: {p:?}"),
                        None => assert_eq!(w, 800, "{case}: {:?}", f.all_text()),
                    }
                }
            }
        }
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

    /// The honesty notes of the M1 comet captures (C1L r1 phase 2, the capture JSON `hud.notes`):
    /// NEOWISE's six, and with `halley` Halley's seventh, which wraps in the pop-up.
    fn comet_notes(halley: bool) -> Vec<(String, LabelTone)> {
        use crate::theme::VIVID_STAR_COLOURS_LABEL;
        use uni_sim::honesty::{HonestyFlag, label};
        let mut notes: Vec<(String, LabelTone)> = [
            (FREE_FLIGHT_LABEL, LabelTone::NonPhysical),
            (label(HonestyFlag::BandModel), LabelTone::Neutral),
            (
                label(HonestyFlag::MilkyWayDetailProcedural),
                LabelTone::Neutral,
            ),
            (label(HonestyFlag::ProceduralBelt), LabelTone::Neutral),
            (EYE_ADAPTED_STARS_LABEL, LabelTone::NonPhysical),
            (VIVID_STAR_COLOURS_LABEL, LabelTone::NonPhysical),
        ]
        .into_iter()
        .map(|(s, t)| (s.to_owned(), t))
        .collect();
        if halley {
            notes.push((
                "Comet dust tail partly drawn at this range (6 % of its light; renderer limit)"
                    .into(),
                LabelTone::Neutral,
            ));
        }
        notes
    }

    /// The collapsed honesty line's texts (11 pt regular text face; the perf line is mono).
    fn line_texts(f: &Frame) -> Vec<&Painted> {
        f.texts
            .iter()
            .filter(|p| p.font == egui::FontId::new(SMALL_SIZE, family(FONT_TEXT)))
            .collect()
    }

    /// Pairs of painted texts whose rects overlap by more than half a point.
    fn overlapping_texts(f: &Frame) -> Vec<(String, String)> {
        let rects: Vec<(&str, egui::Rect)> = f
            .texts
            .iter()
            .filter(|p| !p.text.trim().is_empty())
            .map(|p| {
                let r = egui::Rect::from_min_size(p.pos, p.size).shrink(0.5);
                (p.text.as_str(), r)
            })
            .collect();
        let mut pairs = Vec::new();
        for (i, (a, ra)) in rects.iter().enumerate() {
            for (b, rb) in &rects[i + 1..] {
                if ra.intersects(*rb) {
                    pairs.push(((*a).to_owned(), (*b).to_owned()));
                }
            }
        }
        pairs
    }

    /// C4w (C1L r1 phase 2, Halley's dust note): egui places an anchored area from the size it
    /// had on its last pass. A capture draws each bookmark once on one HUD, so on Halley (after
    /// NEOWISE) the pop-up was placed for six notes and drew seven, its last two lines over the
    /// honesty line. It is now placed from its height measured this pass, so it sits 10 pt over
    /// the line whatever the list was before, longer or shorter.
    #[test]
    fn notes_popup_sits_on_the_line_whatever_came_before() {
        let ctx = headless_context();
        for halley in [false, true, false, true, true] {
            let mut d = data();
            d.notes = comet_notes(halley);
            d.eye_adapted_stars = true;
            let mut c = HudControls {
                notes_expanded: true,
                ..controls()
            };
            let f = look(&ctx, 1.0, &d, &mut c);
            let case = format!("halley {halley}");
            let popup = f.has_stroke(theme::HAIRLINE).expect("pop-up frame");
            let line = line_texts(&f);
            assert!(!line.is_empty(), "{case}: no honesty line");
            let line_top = line.iter().map(|p| p.pos.y).fold(f32::INFINITY, f32::min);
            assert!(
                (line_top - 10.0 - popup.bottom()).abs() <= 1.0,
                "{case}: pop-up bottom {} vs line top {line_top}",
                popup.bottom()
            );
            let full =
                "Comet dust tail partly drawn at this range (6 % of its light; renderer limit)";
            let dust = f.texts.iter().find(|p| p.text == full);
            assert_eq!(dust.is_some(), halley, "{case}");
            if let Some(dust) = dust {
                let r = egui::Rect::from_min_size(dust.pos, dust.size);
                assert!(r.height() > 20.0, "{case}: the note wraps: {dust:?}");
                assert!(popup.contains_rect(r), "{case}: {dust:?} outside {popup:?}");
            }
            assert_eq!(
                overlapping_texts(&f),
                Vec::<(String, String)>::new(),
                "{case}"
            );
        }
    }

    /// C4w: [`rise`] stands a block on its bottom edge, lifted over a block across that edge,
    /// capped under the lowest block above (or the ceiling), and given only the room it has.
    #[test]
    fn rise_stacks_a_block_over_its_edge_and_under_the_blocks_above() {
        let x = egui::Rangef::new(24.0, 404.0);
        let rect = |x0: f32, y0: f32, x1: f32, y1: f32| {
            egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1))
        };
        let place = rect(24.0, 24.0, 300.0, 50.0);
        let perf = rect(1600.0, 1020.0, 1896.0, 1058.0);
        // Room: it stands on its edge, whatever is in other columns.
        let s = rise(x, 1037.0, 200.0, 24.0, &[place, perf], 16.0);
        assert_eq!(
            s,
            Slot {
                top: 837.0,
                height: 200.0
            }
        );
        // No room: it fills the room, 16 pt under the lowest block above.
        let s = rise(x, 1037.0, 2000.0, 24.0, &[place, perf], 16.0);
        assert_eq!(
            s,
            Slot {
                top: 66.0,
                height: 971.0
            }
        );
        // Nothing above: the ceiling caps it.
        let s = rise(x, 1037.0, 2000.0, 24.0, &[perf], 16.0);
        assert_eq!(
            s,
            Slot {
                top: 24.0,
                height: 1013.0
            }
        );
        // A block across its edge in its columns (a narrow window's perf block) lifts it, and
        // a second one met there lifts it again.
        let low = rect(300.0, 1000.0, 600.0, 1060.0);
        let s = rise(x, 1050.0, 14.0, 24.0, &[low], 16.0);
        assert_eq!(
            s,
            Slot {
                top: 970.0,
                height: 14.0
            }
        );
        let next = rect(0.0, 950.0, 100.0, 990.0);
        let s = rise(x, 1050.0, 14.0, 24.0, &[low, next], 16.0);
        assert_eq!(
            s,
            Slot {
                top: 920.0,
                height: 14.0
            }
        );
        // The gap counts: a block ending 10 pt under the edge still lifts it.
        let s = rise(
            x,
            1037.0,
            20.0,
            24.0,
            &[rect(24.0, 1047.0, 300.0, 1060.0)],
            16.0,
        );
        assert_eq!(s.top + s.height, 1031.0);
        // No room at all: zero height, never negative.
        let s = rise(x, 60.0, 50.0, 24.0, &[place], 16.0);
        assert_eq!(s.height, 0.0);
    }

    /// C4w: [`fall`] hangs a block from its top edge and stops it `gap` over the highest block
    /// below it in its columns (the Gravity panel under the Location stack), or at the floor.
    #[test]
    fn fall_hangs_a_block_and_stops_over_the_blocks_below() {
        let x = egui::Rangef::new(24.0, 324.0);
        let gravity = egui::Rect::from_min_max(egui::pos2(24.0, 900.0), egui::pos2(284.0, 1056.0));
        let card = egui::Rect::from_min_max(egui::pos2(1476.0, 24.0), egui::pos2(1896.0, 700.0));
        let s = fall(x, 250.0, 300.0, 1056.0, &[gravity, card], 16.0);
        assert_eq!(
            s,
            Slot {
                top: 250.0,
                height: 300.0
            }
        );
        let s = fall(x, 250.0, 900.0, 1056.0, &[gravity, card], 16.0);
        assert_eq!(
            s,
            Slot {
                top: 250.0,
                height: 634.0
            }
        );
        let s = fall(x, 250.0, 900.0, 1056.0, &[card], 16.0);
        assert_eq!(
            s,
            Slot {
                top: 250.0,
                height: 806.0
            }
        );
        // A block across its top pushes it under that block.
        let s = fall(
            x,
            250.0,
            10.0,
            1056.0,
            &[
                gravity,
                egui::Rect::from_min_max(egui::pos2(0.0, 200.0), egui::pos2(50.0, 260.0)),
            ],
            16.0,
        );
        assert_eq!(
            s,
            Slot {
                top: 276.0,
                height: 10.0
            }
        );
    }

    /// Lead ruling C4w-I1: a block that fits keeps every note in order; one that does not keeps
    /// its honesty notes whole and first, the others in what is left (left out under
    /// `MIN_SCROLL_ROOM`); honesty notes taller than the room are drawn whole all the same.
    #[test]
    fn plan_notes_keeps_honesty_notes_whole() {
        let never = || -> f32 { panic!("measured although it fits") };
        assert_eq!(plan_notes(200.0, 150.0, never), (NotesFit::All, 150.0));
        // Halley's stack at 960 × 540 pt: 154.5 pt of notes in 148.
        assert_eq!(
            plan_notes(148.0, 154.5, || 60.0),
            (NotesFit::HonestyFirst { room: 84.0 }, 148.0)
        );
        // Under a scroll area's least room the informational notes are left out.
        assert_eq!(
            plan_notes(80.0, 200.0, || 60.0),
            (NotesFit::HONESTY_ONLY, 60.0)
        );
        // Honesty notes never shrink, even past the room.
        assert_eq!(
            plan_notes(100.0, 300.0, || 191.5),
            (NotesFit::HONESTY_ONLY, 191.5)
        );
        // Only honesty notes: nothing to give up.
        assert_eq!(plan_notes(100.0, 150.0, || 150.0), (NotesFit::All, 150.0));
    }

    /// Honesty notes are the non-physical and danger ones and the renderer warnings, whatever
    /// tone the app gave a warning's text; expanded, the stack leaves the flight label out.
    #[test]
    fn block_notes_mark_honesty_notes() {
        let notes: Vec<(String, LabelTone)> = [
            (FREE_FLIGHT_LABEL, LabelTone::NonPhysical),
            ("Milky Way band: galaxy model v0", LabelTone::Neutral),
            (INSIDE_HORIZON_LABEL, LabelTone::Danger),
            ("No surface map: uniform albedo", LabelTone::Neutral),
        ]
        .into_iter()
        .map(|(s, t)| (s.to_owned(), t))
        .collect();
        let warnings = vec!["No surface map: uniform albedo".to_owned()];
        let all = block_notes(&notes, &warnings, None);
        let honesty: Vec<bool> = all.iter().map(|n| n.honesty).collect();
        assert_eq!(honesty, [true, false, true, true]);
        let stack = block_notes(&notes, &[], Some(FREE_FLIGHT_LABEL));
        let texts: Vec<&str> = stack.iter().map(|n| n.text).collect();
        assert_eq!(
            texts,
            [
                "Milky Way band: galaxy model v0",
                INSIDE_HORIZON_LABEL,
                "No surface map: uniform albedo"
            ]
        );
        assert_eq!(stack[2].tone, LabelTone::Neutral);
        assert!(!stack[2].honesty, "not a warning here");
    }

    /// Review C4w M3: what the HUD's measuring costs. Prints the mean of one settled HUD run
    /// (collapsed with the pop-up; expanded with the card) and of one frame's measure calls
    /// alone (all six measured blocks, more than either mode makes)
    /// (`cargo test -p uni-ui --lib -- --ignored --nocapture hud_measure_cost`).
    #[test]
    #[ignore = "print-only timing"]
    fn hud_measure_cost() {
        let n = 200;
        let mut d = data();
        d.notes = comet_notes(true);
        d.eye_adapted_stars = true;
        d.lens_on_screen = true;
        d.target = Some(mars_card());
        let ov = Overlay::default();
        for (mode, start) in [
            (
                "collapsed + pop-up",
                HudControls {
                    notes_expanded: true,
                    ..controls()
                },
            ),
            ("expanded + card", expanded()),
        ] {
            let ctx = headless_context();
            let mut c = start;
            let (mut o, _, _) = run_hud_settled(&ctx, SIZE, 1.0, 8192, &d, &ov, &mut c);
            o.textures_delta.clear();
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let (mut o, _) = run_hud(&ctx, headless_input(SIZE, 1.0, 8192), &d, &ov, &mut c);
                o.textures_delta.clear();
            }
            let run_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n);
            println!("{mode}: one HUD run {run_us:.0} µs");
        }
        let ctx = headless_context();
        let c = HudControls {
            notes_expanded: true,
            ..controls()
        };
        let notes = hud_notes(&d, &c);
        let popup = block_notes(&notes, &d.warnings, None);
        let lines = [perf_line(&d), HINT_COLLAPSED.to_owned()];
        let g = d.gravity.clone().expect("gravity");
        // One pass per frame, as in the HUD; only the measure calls are timed.
        let mut total = std::time::Duration::ZERO;
        for _ in 0..n {
            let mut o = ctx.run_ui(headless_input(SIZE, 1.0, 8192), |ui| {
                let ctx = ui.ctx().clone();
                let t0 = std::time::Instant::now();
                measure(&ctx, "perf", 1920.0, |ui| perf_block(ui, &lines));
                measure(&ctx, "gravity", SPEED_WIDTH, |ui| {
                    panels::show_gravity(ui, &g)
                });
                measure(&ctx, "speed", SPEED_WIDTH, |ui| {
                    panels::show_speed(ui, &d.speed)
                });
                measure(&ctx, "notes line", 1000.0, |ui| {
                    notes_line(ui, &notes, 1000.0);
                });
                measure(&ctx, "notes popup", NOTES_POPUP_WIDTH, |ui| {
                    notes_popup(ui, &popup, NotesFit::All)
                });
                measure(&ctx, "notes stack", LOCATION_WIDTH, |ui| {
                    notes_stack(ui, &popup, NotesFit::All)
                });
                total += t0.elapsed();
            });
            o.textures_delta.clear();
        }
        let us = total.as_secs_f64() * 1e6 / f64::from(n);
        println!("one frame's measure calls (6 blocks): {us:.0} µs");
    }

    /// The HUD's visible blocks after the last run on `ctx`: (area name, rect).
    fn blocks(ctx: &egui::Context) -> Vec<(&'static str, egui::Rect)> {
        const NAMES: [&str; 12] = [
            "card",
            "place",
            "warp",
            "time",
            "time line",
            "perf",
            "gravity",
            "speed",
            "location",
            "notes stack",
            "notes",
            "notes popup",
        ];
        ctx.memory(|mem| {
            let visible = mem.areas().visible_layer_ids();
            NAMES
                .iter()
                .filter_map(|&name| {
                    let id = Id::unique(("hud", name));
                    let rect = mem.area_rect(id)?;
                    visible.iter().any(|l| l.id == id).then_some((name, rect))
                })
                .collect()
        })
    }

    /// The C4w sweep sizes (px, points per pixel): the capture size, the owner's windows
    /// (1280 × 832 and 1470 × 956 pt), the window's least size ([`MIN_WINDOW_PT`]), and 960 ×
    /// 540 and 800 × 600 pt.
    fn sweep_sizes() -> Vec<([u32; 2], f32)> {
        let min = [
            (2.0 * MIN_WINDOW_PT[0]) as u32,
            (2.0 * MIN_WINDOW_PT[1]) as u32,
        ];
        vec![
            ([1920, 1080], 1.0),
            ([2560, 1664], 2.0),
            ([2940, 1912], 2.0),
            (min, 2.0),
            ([1920, 1080], 2.0),
            ([1600, 1200], 2.0),
        ]
    }

    /// A screen at least the window's least size.
    fn holds_min(size_px: [u32; 2], ppp: f32) -> bool {
        f64::from(size_px[0]) / f64::from(ppp) >= MIN_WINDOW_PT[0]
            && f64::from(size_px[1]) / f64::from(ppp) >= MIN_WINDOW_PT[1]
    }

    /// The comet notes and nine long informational notes, each two lines in the pop-up.
    fn long_notes() -> Vec<(String, LabelTone)> {
        let mut long = comet_notes(true);
        for i in 0..9 {
            long.push((
                format!(
                    "Test note {i}: an approximation stated in a sentence long enough to wrap \
                     onto two lines in the pop-up (renderer limit)"
                ),
                LabelTone::Neutral,
            ));
        }
        long
    }

    /// The most honesty notes M1 raises at once: inside a horizon at a traced hole on the debug
    /// grid, planets extrapolated, the belt highlighted, two renderer warnings (one wraps), on
    /// top of the comet captures' labels.
    fn worst_honesty(d: &mut HudData, c: &mut HudControls) {
        use uni_sim::honesty::{HonestyFlag, label};
        d.notes = comet_notes(true);
        d.notes
            .push((DEBUG_GRID_LABEL.into(), LabelTone::NonPhysical));
        d.notes.push((
            label(HonestyFlag::PlanetsExtrapolated).into(),
            LabelTone::NonPhysical,
        ));
        d.inside_horizon = true;
        d.warnings = vec![
            "Inside Sun".into(),
            "GPU error: validation error in the bodies pass (the frame may be incomplete; see \
             the log)"
                .into(),
        ];
        c.highlight_belt = true;
    }

    /// One case of the sweep: a fresh HUD draws `d` with only the plain notes, then with its
    /// own (C1L r1: a list that grew since the last draw). Returns the HUD and the second draw.
    fn draw_grown(
        size_px: [u32; 2],
        ppp: f32,
        d: &HudData,
        c: HudControls,
    ) -> (egui::Context, Frame) {
        let ctx = headless_context();
        let plain = HudData {
            notes: data().notes,
            warnings: Vec::new(),
            ..d.clone()
        };
        let mut c1 = c;
        let (mut o, _, _) = run_hud_settled(
            &ctx,
            size_px,
            ppp,
            8192,
            &plain,
            &Overlay::default(),
            &mut c1,
        );
        o.textures_delta.clear();
        let mut c2 = c;
        let (mut o, _, _) =
            run_hud_settled(&ctx, size_px, ppp, 8192, d, &Overlay::default(), &mut c2);
        o.textures_delta.clear();
        let f = collect(&o);
        (ctx, f)
    }

    /// Every case of the sweep: (case name, size, ppp, data, controls, expanded). The lists:
    /// the comet captures', the long one, and the worst honesty set on screens at least the
    /// window's least size; collapsed with the pop-up, expanded, expanded with the card; the
    /// warp panel off and on.
    fn sweep_cases() -> Vec<(String, [u32; 2], f32, HudData, HudControls, bool)> {
        let mut cases = Vec::new();
        for (size_px, ppp) in sweep_sizes() {
            for list in ["comets", "long", "worst"] {
                if list == "worst" && !holds_min(size_px, ppp) {
                    continue;
                }
                for (mode, card_expanded, card) in [
                    ("collapsed", false, false),
                    ("expanded", true, false),
                    ("card", true, true),
                ] {
                    for show_warp in [false, true] {
                        let mut d = data();
                        let mut c = HudControls {
                            card_expanded,
                            notes_expanded: true,
                            show_warp,
                            ..controls()
                        };
                        d.target = card.then(mars_card);
                        d.eye_adapted_stars = true;
                        d.lens_on_screen = true;
                        match list {
                            "comets" => d.notes = comet_notes(true),
                            "long" => d.notes = long_notes(),
                            _ => worst_honesty(&mut d, &mut c),
                        }
                        let case = format!("{size_px:?} @ {ppp}, {list}, {mode}, warp {show_warp}");
                        cases.push((case, size_px, ppp, d, c, card_expanded));
                    }
                }
            }
        }
        cases
    }

    /// C4w (the owner: soft and uncluttered): over the sweep (every size, list, mode, the warp
    /// panel off and on, the list grown since the last draw), no notes block touches another
    /// HUD block or leaves the screen. Informational notes that would not fit scroll or are
    /// left out.
    #[test]
    fn notes_never_overlap_other_blocks() {
        for (case, size_px, ppp, d, c, expanded) in sweep_cases() {
            let screen = egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(size_px[0] as f32, size_px[1] as f32) / ppp,
            );
            let (ctx, _) = draw_grown(size_px, ppp, &d, c);
            let blocks = blocks(&ctx);
            let names: Vec<&str> = blocks.iter().map(|b| b.0).collect();
            let note_blocks: &[&str] = if expanded {
                &["notes stack"]
            } else {
                &["notes", "notes popup"]
            };
            for name in note_blocks {
                assert!(names.contains(name), "{case}: no {name} in {names:?}");
            }
            for (name, rect) in &blocks {
                if !note_blocks.contains(name) {
                    continue;
                }
                assert!(
                    screen.expand(0.5).contains_rect(*rect),
                    "{case}: {name} {rect:?} off {screen:?}"
                );
                for (other, r) in &blocks {
                    if other != name {
                        assert!(
                            !rect.shrink(0.5).intersects(*r),
                            "{case}: {name} {rect:?} touches {other} {r:?}"
                        );
                    }
                }
            }
        }
    }

    /// The honesty notes of `d` under `c`: non-physical or danger in tone, or a renderer
    /// warning; expanded, without the flight label (the Speed header shows it).
    fn honesty_texts(d: &HudData, c: &HudControls) -> Vec<String> {
        hud_notes(d, c)
            .into_iter()
            .filter(|(t, tone)| {
                matches!(tone, LabelTone::NonPhysical | LabelTone::Danger) || d.warnings.contains(t)
            })
            .filter(|(t, _)| !c.card_expanded || *t != d.speed.header)
            .map(|(t, _)| t)
            .collect()
    }

    /// Lead ruling C4w-I1: the honesty notes (non-physical, danger, renderer warnings) are never
    /// scrolled out of view or dropped. Over the whole sweep, 960 × 540 pt and the window's
    /// least size included, every one is painted in full in the pop-up or the stack: inside its
    /// clip rect and on screen. Before the ruling they came last and a scroll area cut them
    /// first (960 × 540 pt, expanded: "Stars near the hole are not yet lensed (M3)").
    #[test]
    fn honesty_notes_stay_whole_on_screen() {
        for (case, size_px, ppp, d, c, _) in sweep_cases() {
            let screen = egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(size_px[0] as f32, size_px[1] as f32) / ppp,
            );
            let (_, f) = draw_grown(size_px, ppp, &d, c);
            let expected = honesty_texts(&d, &c);
            assert!(expected.len() >= 3, "{case}: {expected:?}");
            for text in &expected {
                let p = f
                    .texts
                    .iter()
                    .find(|p| p.text == *text && p.font.size == NOTES_SIZE)
                    .unwrap_or_else(|| panic!("{case}: {text:?} not drawn: {:?}", f.all_text()));
                let r = egui::Rect::from_min_size(p.pos, p.size);
                assert!(
                    p.clip.expand(0.5).contains_rect(r) && screen.expand(0.5).contains_rect(r),
                    "{case}: {text:?} at {r:?} cut by {:?} or off {screen:?}",
                    p.clip
                );
            }
        }
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
