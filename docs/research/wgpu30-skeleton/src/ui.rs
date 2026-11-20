//! egui 0.37 overlay (egui + egui-winit + egui-wgpu 0.37.0, which depend on wgpu ^30 / winit ^0.30.13).
use winit::{event::WindowEvent, window::Window};

pub struct Overlay {
    pub ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl Overlay {
    /// `surface_format` must be the non-sRGB swapchain format (egui prefers Bgra8Unorm/Rgba8Unorm).
    pub fn new(window: &Window, device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window, // &dyn HasDisplayHandle
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer =
            egui_wgpu::Renderer::new(device, surface_format, egui_wgpu::RendererOptions::default());
        Self { ctx, state, renderer }
    }

    /// Returns true if egui consumed the event.
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Runs the UI and records its draw into `encoder` (LoadOp::Load over the 3D image).
    /// Returns extra command buffers that must be submitted *before* `encoder.finish()`.
    pub fn frame(
        &mut self,
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size_in_pixels: [u32; 2],
        mut build: impl FnMut(&mut egui::Ui),
    ) -> Vec<wgpu::CommandBuffer> {
        let raw = self.state.take_egui_input(window);
        // 0.34+: Context::run -> run_ui (closure gets &mut Ui); windows still take &Context.
        let out = self.ctx.run_ui(raw, |ui| {
            egui::Window::new("HUD").show(ui.ctx(), |ui| build(ui));
        });
        let egui::FullOutput { platform_output, textures_delta, shapes, pixels_per_point, .. } = out;
        self.state.handle_platform_output(window, platform_output);
        let jobs = self.ctx.tessellate(shapes, pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels, pixels_per_point };

        // 0.37: `set` is HashMap<TextureId, SmallVec<[ImageDelta; 1]>> (several deltas per texture).
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                self.renderer.update_texture(device, queue, *id, delta);
            }
        }
        let cmds = self.renderer.update_buffers(device, queue, encoder, &jobs, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime(); // egui-wgpu wants RenderPass<'static>
            self.renderer.render(&mut pass, &jobs, &screen);
        }
        for id in &textures_delta.free {
            self.renderer.free_texture(id);
        }
        cmds
    }
}
