//! egui HUD overlay drawn over the tonemapped frame (M0 Task 10).
//!
//! v0 is functional only: one translucent panel at the top left and an F1 controls list. The
//! final visual design comes from the M1 mockup (Task 14).

use std::ops::RangeInclusive;

use egui::{Color32, RichText};
use winit::{event::WindowEvent, window::Window};

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
    ("F12", "Screenshot"),
    ("F1", "Show / hide this list"),
];

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
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl Hud {
    /// `plain_format` must be the non-sRGB swapchain format: egui writes gamma-encoded colours.
    pub fn new(
        window: &Window,
        gpu: &uni_gfx::GpuContext,
        plain_format: wgpu::TextureFormat,
    ) -> Self {
        let ctx = egui::Context::default();
        // The HUD always sits over space, so it stays dark whatever the system theme is.
        ctx.set_theme(egui::Theme::Dark);
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            plain_format,
            egui_wgpu::RendererOptions::default(),
        );
        Self {
            ctx,
            state,
            renderer,
        }
    }

    /// Feeds a window event to egui. Returns true if egui consumed it, in which case the app
    /// should not also act on it.
    pub fn on_window_event(&mut self, window: &Window, e: &WindowEvent) -> bool {
        self.state.on_window_event(window, e).consumed
    }

    /// Runs the HUD and records its draw into `enc` over `target` (`LoadOp::Load`, so it
    /// composites over the frame). `target` is the plain (non-sRGB) swapchain view.
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
        let raw = self.state.take_egui_input(window);
        let out = self.ctx.run_ui(raw, |ui| {
            if !controls.visible {
                return;
            }
            show_panel(ui.ctx(), data, controls);
            if controls.show_help {
                show_controls_list(ui.ctx(), &mut controls.show_help);
            }
        });
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = out;
        self.state.handle_platform_output(window, platform_output);
        let jobs = self.ctx.tessellate(shapes, pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size_px,
            pixels_per_point,
        };

        // egui 0.37: several deltas per texture.
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                self.renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, delta);
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
        for id in &textures_delta.free {
            self.renderer.free_texture(id);
        }
        cmds
    }
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
            ui.label(RichText::new("Lens background: M0 debug grid").color(AMBER));
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
        }
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
        assert_eq!(gpu_timing_lines(&data(Vec::new(), true)), ["GPU ms pending"]);
    }
}
