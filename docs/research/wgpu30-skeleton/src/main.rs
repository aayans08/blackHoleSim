// Skeleton targeting: wgpu 30.0.2, winit 0.30.13, egui/egui-wgpu/egui-winit 0.37.0, pollster 1.0.3
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const COMPUTE_WGSL: &str = r#"
struct Params { time: f32, exposure: f32, _pad: vec2<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var hdr_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(hdr_out);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let c = vec3<f32>(uv, 0.5 + 0.5 * sin(params.time)) * params.exposure;
    textureStore(hdr_out, gid.xy, vec4<f32>(c, 1.0));
}
"#;

const BLIT_WGSL: &str = r#"
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    var o: VsOut;
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(hdr_tex, hdr_samp, in.uv).rgb;
    return vec4<f32>(c / (c + vec3<f32>(1.0)), 1.0); // Reinhard; sRGB surface does the gamma
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params { time: f32, exposure: f32, _pad: [f32; 2] }

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    params_buf: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    blit_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    compute_bg: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    start: std::time::Instant,
}

impl Gfx {
    async fn new(window: Arc<Window>, instance: &wgpu::Instance) -> Self {
        let size = window.inner_size();
        // Arc<Window> -> Surface<'static>; no unsafe, no lifetime juggling.
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .expect("adapter"); // Result since v25 (was Option)
        let af = adapter.features();
        let wanted = wgpu::Features::SHADER_F16
            | wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("main device"),
                required_features: af & wanted,
                // Ask for the adapter's real limits instead of WebGPU defaults (bigger buffers etc.)
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device"); // no 2nd `trace_path` arg since v25

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto, // new in v30
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- compute pipeline (writes an Rgba16Float storage texture) ----
        let cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_WGSL.into()),
        });
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute layout"),
            bind_group_layouts: &[Some(&compute_bgl)], // Option<&BGL> since v29
            immediate_size: 0,                         // was push_constant_ranges (v28)
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute"),
            layout: Some(&compute_layout),
            module: &cs,
            entry_point: Some("cs_main"), // Option since v23
            compilation_options: Default::default(),
            cache: None,
        });

        // ---- blit pipeline (samples the HDR texture, tone-maps to the surface) ----
        let bs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit layout"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &bs,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[], // &[Option<VertexBufferLayout>] since v30
            },
            fragment: Some(wgpu::FragmentState {
                module: &bs,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None, // was `multiview` before v28
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest, // own enum since v28
            ..Default::default()
        });

        // ---- egui ----
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        let (compute_bg, blit_bg) =
            Self::make_targets(&device, &config, &compute_bgl, &blit_bgl, &params_buf, &sampler);
        let _ = (compute_bgl, blit_bgl); // layouts are kept alive by pipelines; re-fetch via get_bind_group_layout on resize
        Self {
            window, surface, device, queue, config, params_buf, compute_pipeline, blit_pipeline,
            sampler, compute_bg, blit_bg, egui_ctx, egui_state, egui_renderer,
            start: std::time::Instant::now(),
        }
    }

    fn make_targets(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        compute_bgl: &wgpu::BindGroupLayout,
        blit_bgl: &wgpu::BindGroupLayout,
        params_buf: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let hdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr"),
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = hdr.create_view(&wgpu::TextureViewDescriptor::default());
        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute bg"),
            layout: compute_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });
        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit bg"),
            layout: blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });
        (compute_bg, blit_bg)
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        let cbgl = self.compute_pipeline.get_bind_group_layout(0);
        let bbgl = self.blit_pipeline.get_bind_group_layout(0);
        (self.compute_bg, self.blit_bg) =
            Self::make_targets(&self.device, &self.config, &cbgl, &bbgl, &self.params_buf, &self.sampler);
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => f,
            wgpu::CurrentSurfaceTexture::Suboptimal(_) | wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return, // Timeout / Occluded / Validation: skip this frame
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let t = self.start.elapsed().as_secs_f32();
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&Params { time: t, exposure: 2.0, _pad: [0.0; 2] }));

        // ---- egui frame ----
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut out = self.egui_ctx.run_ui(raw_input, |ui| {
            egui::Window::new("HUD").show(ui.ctx(), |ui| {
                ui.label(format!("t = {t:.2} s"));
            });
        });
        self.egui_state.handle_platform_output(&self.window, std::mem::take(&mut out.platform_output));
        let jobs = self.egui_ctx.tessellate(std::mem::take(&mut out.shapes), out.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: out.pixels_per_point,
        };
        // egui 0.37: `set` is a HashMap<TextureId, SmallVec<[ImageDelta;1]>>, and dropping an
        // un-drained TexturesDelta debug-panics -> always drain both `set` and `free`.
        for (id, deltas) in out.textures_delta.set.drain() {
            for d in &deltas {
                self.egui_renderer.update_texture(&self.device, &self.queue, id, d);
            }
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let egui_cmds = self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &jobs, &screen);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("trace"), timestamp_writes: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bg, &[]);
            cpass.dispatch_workgroups(self.config.width.div_ceil(8), self.config.height.div_ceil(8), 1);
        }
        {
            // egui_wgpu::Renderer::render wants RenderPass<'static> -> forget_lifetime()
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blit+egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &self.blit_bg, &[]);
            rpass.draw(0..3, 0..1);
            self.egui_renderer.render(&mut rpass, &jobs, &screen);
        }
        for id in out.textures_delta.free.drain() {
            self.egui_renderer.free_texture(&id);
        }
        self.queue.submit(egui_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        self.window.pre_present_notify();
        self.queue.present(frame); // was frame.present() before v30
    }
}

struct App {
    instance: wgpu::Instance,
    gfx: Option<Gfx>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title("Universe")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(el.create_window(attrs).expect("window"));
        self.gfx = Some(pollster::block_on(Gfx::new(window, &self.instance)));
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(g) = self.gfx.as_mut() else { return };
        let resp = g.egui_state.on_window_event(&g.window, &event);
        if resp.consumed { return; } // egui took it (e.g. click on a HUD window)
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => g.resize(s.width, s.height),
            WindowEvent::RedrawRequested => g.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(g) = &self.gfx { g.window.request_redraw(); }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    // v29+: InstanceDescriptor has no Default; say whether a display handle is used.
    let mut desc = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(event_loop.owned_display_handle()));
    desc.backends = wgpu::Backends::METAL;
    let instance = wgpu::Instance::new(desc);
    let mut app = App { instance, gfx: None };
    event_loop.run_app(&mut app).expect("run");
}
