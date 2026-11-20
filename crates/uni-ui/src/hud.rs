//! egui HUD overlay drawn over the tonemapped frame (M0 Task 10).
//!
//! v0 is functional only: one translucent panel at the top left and an F1 controls list. The
//! final visual design comes from the M1 mockup (Task 14).
//!
//! A windowed HUD ([`Hud::new`]) reads its input and screen size from winit. A headless HUD
//! ([`Hud::new_headless`], for `--capture ... --hud`) has no window: it is laid out for the
//! capture size instead.

use std::ops::RangeInclusive;

use egui::{Color32, RichText};
use winit::{
    event::{ElementState, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::format::{format_distance, format_epoch, format_speed, format_time_rate};

/// Warnings and non-physical labels (owner-approved HUD palette).
const AMBER: Color32 = Color32::from_rgb(0xF5, 0xB0, 0x4C);
/// Danger (owner-approved HUD palette).
const RED: Color32 = Color32::from_rgb(0xF8, 0x71, 0x71);

/// Slider range for both exposures, EV at ISO 100. A UI choice, not physics: it spans the
/// photographic EV of a moonless night sky (about −6) to direct sunlight (about 15, the
/// "sunny 16" rule) with margin either side.
const EV100_RANGE: RangeInclusive<f32> = -10.0..=20.0;
/// The lens tracer's resolution-scale range (`LensTracer::set_resolution_scale`, Task 9).
const LENS_SCALE_RANGE: RangeInclusive<f32> = 0.25..=1.0;

/// The M0 key map (uni-nav, Task 6), shown when F1 toggles `HudControls::show_help`.
const CONTROLS: [(&str, &str); 8] = [
    ("W / S / A / D", "Forward / back / left / right"),
    ("Space / Left Ctrl", "Up / down"),
    ("Q / E", "Roll"),
    ("Right mouse (hold)", "Look"),
    ("Wheel or = / -", "Speed ×10^±0.1 per notch"),
    ("Tab", "Show / hide HUD"),
    ("F12", "Save view bookmark (captures/views/)"),
    ("F1", "Show / hide this list"),
];

/// What is approximate wherever the lens pass draws in M0 (Global Constraints, Honesty): it
/// bends a debug grid, not the real sky, and the catalogue stars are drawn unlensed over it.
/// Shown only while it draws ([`honesty_labels`]).
const HONESTY_LABELS: [&str; 2] = [
    "Lens background: M0 debug grid",
    "Stars near the hole are not yet lensed (M0)",
];

/// The app's own hotkeys in uni-nav's key map (Task 6): Tab hides the HUD, F1 shows the
/// controls list, F12 saves the view as a bookmark. Matched by physical key, as uni-nav does.
const APP_HOTKEYS: [KeyCode; 3] = [KeyCode::Tab, KeyCode::F1, KeyCode::F12];

/// Most egui runs per headless draw. egui lays out a new window invisibly on its first run
/// (a sizing pass), draws it from the second, and stops asking for a repaint after the third
/// (measured with egui 0.37; `headless_draw_settles_then_costs_one_run`). The fourth is margin.
const HEADLESS_MAX_RUNS: usize = 4;

/// What the HUD shows this frame. Filled by the app; the HUD only reads it.
#[derive(Clone, Debug)]
pub struct HudData {
    pub fps: f32,
    pub cpu_ms: f32,
    /// (pass name, GPU ms) per pass, from the pass timer.
    pub gpu_ms: Vec<(&'static str, f32)>,
    /// False when the device has no timestamp queries; the HUD then shows "GPU ms n/a".
    pub gpu_timing_available: bool,
    pub dist_from_sun_m: f64,
    pub nearest_name: String,
    pub nearest_dist_m: f64,
    pub speed_mps: f64,
    pub epoch: uni_core::Epoch,
    /// Simulated seconds per real second; 0 is paused.
    pub time_rate: f64,
    pub star_count: u32,
    /// Shown in amber, one per line.
    pub warnings: Vec<String>,
    /// Shows "Inside event horizon" in red.
    pub inside_horizon: bool,
    /// True when the lens pass draws this frame. Only then does the HUD show, in amber, "Lens
    /// background: M0 debug grid" and "Stars near the hole are not yet lensed (M0)". Sky views
    /// skip the pass, so neither line would be true there.
    pub lens_on_screen: bool,
}

/// Settings the HUD edits. The app reads them back after [`Hud::draw`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HudControls {
    pub scene_ev100: f32,
    pub star_ev100: f32,
    /// Lens tracer resolution scale, 0.25..=1.0.
    pub lens_scale: f32,
    /// False hides the whole HUD, the controls list included (Tab).
    pub visible: bool,
    /// Shows the controls list (F1). Its close button clears this.
    pub show_help: bool,
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
    /// composites over the frame). `target` is the plain (non-sRGB) swapchain view.
    ///
    /// A HUD made by [`Self::new_headless`] draws as [`Self::draw_headless`] does.
    ///
    /// Returns command buffers that must be submitted before `enc.finish()`.
    #[allow(clippy::too_many_arguments)] // The signature is the frozen M0 interface.
    pub fn draw(
        &mut self,
        window: &Window,
        gpu: &uni_gfx::GpuContext,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size_px: [u32; 2],
        data: &HudData,
        controls: &mut HudControls,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(state) = self.state.as_mut() else {
            return self.draw_headless(gpu, enc, target, size_px, data, controls);
        };
        let raw = state.take_egui_input(window);
        let mut out = run_hud(&self.ctx, raw, data, controls);
        state.handle_platform_output(window, std::mem::take(&mut out.platform_output));
        self.render(gpu, enc, target, size_px, out)
    }

    /// Runs the HUD and records its draw as [`Self::draw`] does, but with no window: it is laid
    /// out for `size_px` at the HUD's `pixels_per_point`. For headless captures.
    ///
    /// egui draws a new panel only from its second run, so this runs egui until it stops asking
    /// for an immediate repaint (three runs on the first call, then one). One call draws the
    /// whole HUD.
    ///
    /// Returns command buffers that must be submitted before `enc.finish()`.
    pub fn draw_headless(
        &mut self,
        gpu: &uni_gfx::GpuContext,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size_px: [u32; 2],
        data: &HudData,
        controls: &mut HudControls,
    ) -> Vec<wgpu::CommandBuffer> {
        let (out, _runs) = run_hud_settled(
            &self.ctx,
            size_px,
            self.pixels_per_point,
            self.max_texture_side,
            data,
            controls,
        );
        self.render(gpu, enc, target, size_px, out)
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

/// One egui run of the HUD.
fn run_hud(
    ctx: &egui::Context,
    raw: egui::RawInput,
    data: &HudData,
    controls: &mut HudControls,
) -> egui::FullOutput {
    ctx.run_ui(raw, |ui| {
        if !controls.visible {
            return;
        }
        show_panel(ui.ctx(), data, controls);
        if controls.show_help {
            show_controls_list(ui.ctx(), &mut controls.show_help);
        }
    })
}

/// Runs the HUD headless until egui stops asking for an immediate repaint, at most
/// [`HEADLESS_MAX_RUNS`] times. Returns the merged output and the number of runs.
fn run_hud_settled(
    ctx: &egui::Context,
    size_px: [u32; 2],
    pixels_per_point: f32,
    max_texture_side: usize,
    data: &HudData,
    controls: &mut HudControls,
) -> (egui::FullOutput, usize) {
    let input = || headless_input(size_px, pixels_per_point, max_texture_side);
    let mut out = run_hud(ctx, input(), data, controls);
    let mut runs = 1;
    // A new egui window spends its first run sizing itself, invisibly, and asks for a repaint.
    let mut again = wants_repaint_now(&out);
    while again && runs < HEADLESS_MAX_RUNS {
        let next = run_hud(ctx, input(), data, controls);
        runs += 1;
        // Read before `append`, which keeps the shortest repaint delay of all runs.
        again = wants_repaint_now(&next);
        // Keeps every run's texture uploads (the first has the font atlas) and the last
        // run's shapes.
        out.append(next);
    }
    (out, runs)
}

/// Whether egui asked to run again straight away.
fn wants_repaint_now(out: &egui::FullOutput) -> bool {
    out.viewport_output
        .get(&egui::ViewportId::ROOT)
        .is_some_and(|v| v.repaint_delay.is_zero())
}

/// Dark translucent glass with a hairline border.
fn panel_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(Color32::from_rgba_unmultiplied(8, 12, 18, 200))
        .stroke(egui::Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(255, 255, 255, 40),
        ))
        .corner_radius(6)
        .inner_margin(10)
}

/// The readout panel, top left.
fn show_panel(ctx: &egui::Context, data: &HudData, controls: &mut HudControls) {
    egui::Window::new("HUD")
        .title_bar(false)
        .resizable(false)
        .anchor(egui::Align2::LEFT_TOP, [12.0, 12.0])
        .frame(panel_frame())
        .show(ctx, |ui| {
            if data.inside_horizon {
                ui.label(RichText::new("Inside event horizon").color(RED).strong());
            }
            ui.monospace(format!("{:.0} fps · CPU {:.2} ms", data.fps, data.cpu_ms));
            for line in gpu_timing_lines(data) {
                ui.monospace(line);
            }
            ui.separator();

            egui::Grid::new("hud readouts")
                .num_columns(2)
                .show(ui, |ui| {
                    let row = |ui: &mut egui::Ui, label: &str, value: RichText| {
                        ui.label(label);
                        ui.label(value.monospace());
                        ui.end_row();
                    };
                    row(ui, "Sun", format_distance(data.dist_from_sun_m).into());
                    let nearest = format!(
                        "{} · {}",
                        data.nearest_name,
                        format_distance(data.nearest_dist_m)
                    );
                    row(ui, "Nearest", nearest.into());
                    let speed = RichText::new(format_speed(data.speed_mps));
                    let speed = if data.speed_mps > uni_core::C {
                        speed.color(AMBER)
                    } else {
                        speed
                    };
                    row(ui, "Speed", speed);
                    row(ui, "Epoch", format_epoch(data.epoch).into());
                    row(ui, "Rate", format_time_rate(data.time_rate).into());
                    row(ui, "Stars", data.star_count.to_string().into());
                });

            for warning in &data.warnings {
                ui.label(RichText::new(warning).color(AMBER));
            }
            ui.separator();

            ui.add(
                egui::Slider::new(&mut controls.scene_ev100, EV100_RANGE)
                    .text("Scene EV100")
                    .fixed_decimals(1)
                    // A bookmark may set an exposure outside the slider's range; keep it.
                    .clamping(egui::SliderClamping::Edits),
            );
            ui.add(
                egui::Slider::new(&mut controls.star_ev100, EV100_RANGE)
                    .text("Star EV100")
                    .fixed_decimals(1)
                    .clamping(egui::SliderClamping::Edits),
            );
            ui.add(
                egui::Slider::new(&mut controls.lens_scale, LENS_SCALE_RANGE)
                    .text("Lens scale")
                    .fixed_decimals(2),
            );
            for label in honesty_labels(data.lens_on_screen) {
                ui.label(RichText::new(*label).color(AMBER));
            }
            ui.small("F1 controls · Tab hide HUD");
        });
}

/// The F1 controls list, top right. Its close button clears `open`.
fn show_controls_list(ctx: &egui::Context, open: &mut bool) {
    egui::Window::new("Controls")
        .open(open)
        .resizable(false)
        .collapsible(false)
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .frame(panel_frame())
        .show(ctx, |ui| {
            egui::Grid::new("hud controls")
                .num_columns(2)
                .show(ui, |ui| {
                    for (keys, action) in CONTROLS {
                        ui.monospace(keys);
                        ui.label(action);
                        ui.end_row();
                    }
                });
        });
}

/// The amber honesty lines for a frame: [`HONESTY_LABELS`] when the lens pass draws, else none,
/// since a label must be true where it is shown (design anti-goal 2).
fn honesty_labels(lens_on_screen: bool) -> &'static [&'static str] {
    if lens_on_screen { &HONESTY_LABELS } else { &[] }
}

/// One `"GPU <pass> <ms> ms"` line per timed pass, or `"GPU ms n/a"` when the device has no
/// timestamp queries (Review Focus 4). Before the first read-back there is nothing to show yet.
fn gpu_timing_lines(data: &HudData) -> Vec<String> {
    if !data.gpu_timing_available {
        return vec!["GPU ms n/a".into()];
    }
    if data.gpu_ms.is_empty() {
        return vec!["GPU ms pending".into()];
    }
    data.gpu_ms
        .iter()
        .map(|(pass, ms)| format!("GPU {pass} {ms:.2} ms"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::event::{DeviceId, MouseButton};

    fn data(gpu_ms: Vec<(&'static str, f32)>, gpu_timing_available: bool) -> HudData {
        HudData {
            fps: 60.0,
            cpu_ms: 1.0,
            gpu_ms,
            gpu_timing_available,
            dist_from_sun_m: 0.0,
            nearest_name: "Sun".into(),
            nearest_dist_m: 0.0,
            speed_mps: 0.0,
            epoch: uni_core::Epoch::J2000,
            time_rate: 1.0,
            star_count: 0,
            warnings: Vec::new(),
            inside_horizon: false,
            lens_on_screen: false,
        }
    }

    /// The M0 honesty lines, verbatim (Task 10b brief).
    const HONESTY_LINES: [&str; 2] = [
        "Lens background: M0 debug grid",
        "Stars near the hole are not yet lensed (M0)",
    ];

    fn controls(show_help: bool) -> HudControls {
        HudControls {
            scene_ev100: 0.0,
            star_ev100: 0.0,
            lens_scale: 1.0,
            visible: true,
            show_help,
        }
    }

    /// A text shape egui painted.
    #[derive(Debug)]
    struct Painted {
        text: String,
        /// Top left, in points.
        pos: egui::Pos2,
        /// The colour of its first section: `Color32::PLACEHOLDER` for the theme's text colour.
        color: Color32,
    }

    /// Every text shape egui painted.
    fn texts(out: &egui::FullOutput) -> Vec<Painted> {
        fn walk(shape: &egui::Shape, acc: &mut Vec<Painted>) {
            match shape {
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, acc)),
                egui::Shape::Text(t) => acc.push(Painted {
                    text: t.galley.text().to_owned(),
                    pos: t.pos,
                    color: t
                        .galley
                        .job
                        .sections
                        .first()
                        .map_or(Color32::PLACEHOLDER, |s| s.format.color),
                }),
                _ => {}
            }
        }
        let mut acc = Vec::new();
        for clipped in &out.shapes {
            walk(&clipped.shape, &mut acc);
        }
        acc
    }

    /// A fresh headless HUD's first draw of `d` at 1920×1080: its pixels per point and its text.
    fn first_headless_draw(ppp: f32, d: &HudData, mut c: HudControls) -> (f32, Vec<Painted>) {
        let ctx = headless_context();
        let (mut out, _runs) = run_hud_settled(&ctx, [1920, 1080], ppp, 8192, d, &mut c);
        // No GPU here to upload to; epaint asserts that deltas are not silently dropped.
        out.textures_delta.clear();
        (out.pixels_per_point, texts(&out))
    }

    fn find<'a>(texts: &'a [Painted], needle: &str) -> Option<&'a Painted> {
        texts.iter().find(|p| p.text.contains(needle))
    }

    /// M0 gate: the lens labels are true only while the lens pass draws (design anti-goal 2).
    /// Sky views skip the pass and show black space with real stars.
    #[test]
    fn honesty_labels_iff_lens_on_screen() {
        assert_eq!(honesty_labels(true), HONESTY_LINES);
        assert!(honesty_labels(false).is_empty());
    }

    #[test]
    fn honesty_lines_are_drawn_in_amber_iff_lens_on_screen() {
        for lens_on_screen in [true, false] {
            let mut d = data(Vec::new(), false);
            d.lens_on_screen = lens_on_screen;
            let (_, texts) = first_headless_draw(1.0, &d, controls(false));
            for line in HONESTY_LINES {
                match (lens_on_screen, find(&texts, line)) {
                    (true, Some(p)) => assert_eq!(p.color, AMBER, "{line:?}"),
                    (true, None) => panic!("lens on screen: no {line:?} in {texts:?}"),
                    (false, Some(p)) => panic!("no lens on screen, yet {p:?} is drawn"),
                    (false, None) => {}
                }
            }
        }
    }

    /// M0 gate: F12 saves the view as a bookmark in `captures/views/` (uni-app); it takes no
    /// screenshot.
    #[test]
    fn help_names_f12_save_view_bookmark() {
        let f12 = CONTROLS
            .iter()
            .find(|(keys, _)| *keys == "F12")
            .map(|(_, action)| *action);
        assert_eq!(f12, Some("Save view bookmark (captures/views/)"));
        assert!(
            CONTROLS
                .iter()
                .all(|(_, action)| !action.contains("Screenshot")),
            "{CONTROLS:?}"
        );
    }

    #[test]
    fn gpu_timing_without_timestamps_is_na() {
        assert_eq!(gpu_timing_lines(&data(Vec::new(), false)), ["GPU ms n/a"]);
    }

    #[test]
    fn gpu_timing_lists_each_pass() {
        let d = data(vec![("lens", 1.234), ("stars", 0.5)], true);
        assert_eq!(
            gpu_timing_lines(&d),
            ["GPU lens 1.23 ms", "GPU stars 0.50 ms"]
        );
    }

    #[test]
    fn gpu_timing_before_first_readback_is_pending() {
        assert_eq!(
            gpu_timing_lines(&data(Vec::new(), true)),
            ["GPU ms pending"]
        );
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

    /// One headless draw is a finished HUD, laid out for `size_px / pixels_per_point`: the
    /// panel at the top left, the controls list against the right edge.
    #[test]
    fn first_headless_draw_shows_the_hud_laid_out_for_the_target() {
        let mut d = data(Vec::new(), false);
        // Every line the panel can show, the longest honesty line included.
        d.lens_on_screen = true;
        for ppp in [1.0, 2.0] {
            let (out_ppp, texts) = first_headless_draw(ppp, &d, controls(true));
            assert_eq!(out_ppp, ppp);
            let screen =
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1920.0 / ppp, 1080.0 / ppp));
            for needle in ["fps", "GPU ms n/a", "Show / hide HUD", "Save view bookmark"]
                .into_iter()
                .chain(HONESTY_LINES)
            {
                assert!(
                    find(&texts, needle).is_some(),
                    "ppp {ppp}: no {needle:?} in {texts:?}"
                );
            }
            for Painted { text, pos, .. } in &texts {
                assert!(
                    screen.contains(*pos),
                    "ppp {ppp}: {text:?} at {pos:?} is off {screen:?}"
                );
            }
            let fps = find(&texts, "fps").map(|p| p.pos).unwrap_or_default();
            assert!(
                fps.x < 50.0 && fps.y < 50.0,
                "ppp {ppp}: panel not top left: {fps:?}"
            );
            let list = find(&texts, "Show / hide HUD")
                .map(|p| p.pos)
                .unwrap_or_default();
            assert!(
                list.x > screen.width() / 2.0,
                "ppp {ppp}: controls list not on the right half of {screen:?}: {list:?}"
            );
        }
    }

    /// One headless draw must not be a bare sizing pass, must end on egui's own signal rather
    /// than the cap, and once settled must cost a single egui run.
    #[test]
    fn headless_draw_settles_then_costs_one_run() {
        let ctx = headless_context();
        let d = data(Vec::new(), false);
        let mut c = controls(true);
        let (mut first, runs) = run_hud_settled(&ctx, [1920, 1080], 2.0, 8192, &d, &mut c);
        first.textures_delta.clear();
        assert!(
            runs > 1 && runs < HEADLESS_MAX_RUNS,
            "first draw took {runs} runs"
        );
        let (mut next, runs) = run_hud_settled(&ctx, [1920, 1080], 2.0, 8192, &d, &mut c);
        next.textures_delta.clear();
        assert_eq!(runs, 1);
    }

    #[test]
    fn hidden_headless_hud_draws_nothing() {
        let mut c = controls(true);
        c.visible = false;
        let mut d = data(Vec::new(), false);
        d.lens_on_screen = true;
        assert!(first_headless_draw(1.0, &d, c).1.is_empty());
    }
}
