use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

// Uniform buffer structures (must be 16-byte aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    position: [f32; 4],      // vec3 + padding
    forward: [f32; 4],       // vec3 + padding
    right: [f32; 4],         // vec3 + padding
    up: [f32; 4],            // vec3 + padding
    fov: f32,
    aspect: f32,
    _padding: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BlackHoleUniform {
    position: [f32; 4],      // vec3 + padding
    schwarzschild_radius: f32,
    mass: f32,
    _padding: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DiskParamsUniform {
    inner_radius: f32,
    outer_radius: f32,
    thickness: f32,
    density: f32,
    temperature_inner: f32,
    temperature_outer: f32,
    _padding: [f32; 2],
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    fov: f32,
}

impl Camera {
    fn new() -> Self {
        // Camera at 10 Schwarzschild radii, looking at disk edge-on
        let schwarzschild_radius = 29.54; // km for 10 solar masses
        let distance = schwarzschild_radius * 10.0;

        Self {
            position: Vec3::new(distance, distance * 0.2, 0.0),
            yaw: -std::f32::consts::PI * 0.5,
            pitch: -0.2,
            fov: 75.0_f32.to_radians(),  // Wider FOV to see more
        }
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }

    fn to_uniform(&self, aspect: f32) -> CameraUniform {
        let forward = self.forward();
        let right = self.right();
        let up = self.up();

        CameraUniform {
            position: [self.position.x, self.position.y, self.position.z, 0.0],
            forward: [forward.x, forward.y, forward.z, 0.0],
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            fov: self.fov,
            aspect,
            _padding: [0.0; 2],
        }
    }
}

struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            mouse_pressed: false,
            last_mouse: None,
        }
    }
}

struct Renderer<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    // Compute pipeline
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,

    // Display pipeline
    display_pipeline: wgpu::RenderPipeline,
    display_bind_group: wgpu::BindGroup,

    // Buffers
    camera_buffer: wgpu::Buffer,
    black_hole_buffer: wgpu::Buffer,
    disk_params_buffer: wgpu::Buffer,

    // Textures
    compute_texture: wgpu::Texture,
    environment_texture: wgpu::Texture,

    // Samplers
    sampler: wgpu::Sampler,

    // Camera
    camera: Camera,
    input: InputState,

    // Timing
    last_frame: Instant,
}

impl<'window> Renderer<'window> {
    async fn new(window: &'window Window) -> Self {
        let size = window.inner_size();

        // Create wgpu instance with Metal backend preference
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL, // Force Metal for M3
            ..Default::default()
        });

        // Create surface
        let surface = instance.create_surface(window).unwrap();

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        println!("Using adapter: {:?}", adapter.get_info());

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo, // VSync for consistent 60 FPS
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create camera
        let camera = Camera::new();

        // Create uniform buffers
        let camera_uniform = camera.to_uniform(size.width as f32 / size.height as f32);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let schwarzschild_radius = 29.54; // km for 10 solar masses
        let black_hole_uniform = BlackHoleUniform {
            position: [0.0, 0.0, 0.0, 0.0],
            schwarzschild_radius,
            mass: 10.0, // solar masses
            _padding: [0.0; 2],
        };
        let black_hole_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Black Hole Buffer"),
            contents: bytemuck::cast_slice(&[black_hole_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let disk_params_uniform = DiskParamsUniform {
            inner_radius: schwarzschild_radius * 3.0,  // ISCO at 3 Rs
            outer_radius: schwarzschild_radius * 12.0, // Outer edge
            thickness: schwarzschild_radius * 0.5,     // Vertical thickness
            density: 1.0,
            temperature_inner: 10000000.0, // 10^7 K
            temperature_outer: 10000.0,    // 10^4 K
            _padding: [0.0; 2],
        };
        let disk_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Disk Params Buffer"),
            contents: bytemuck::cast_slice(&[disk_params_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create compute output texture
        let compute_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compute Output"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create environment map (procedural starfield)
        let environment_texture = Self::create_starfield(&device, &queue);

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Load shaders
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compute.wgsl").into()),
        });

        let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/display.wgsl").into()),
        });

        // Create compute bind group layout
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
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
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create compute bind group
        let compute_texture_view = compute_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Compute Texture View"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let environment_view = environment_texture.create_view(&Default::default());

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: black_hole_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: disk_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&compute_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&environment_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create compute pipeline
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        // Create display bind group layout
        let display_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Display Bind Group Layout"),
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

        // Create display bind group
        let display_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Display Bind Group"),
            layout: &display_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&compute_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create display pipeline
        let display_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Display Pipeline Layout"),
                bind_group_layouts: &[&display_bind_group_layout],
                push_constant_ranges: &[],
            });

        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Display Pipeline"),
            layout: Some(&display_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &display_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &display_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            compute_pipeline,
            compute_bind_group,
            display_pipeline,
            display_bind_group,
            camera_buffer,
            black_hole_buffer,
            disk_params_buffer,
            compute_texture,
            environment_texture,
            sampler,
            camera,
            input: InputState::default(),
            last_frame: Instant::now(),
        }
    }

    fn create_starfield(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        // Create a procedural starfield texture
        let size = 2048;
        let mut pixels = vec![0u8; (size * size * 4) as usize];

        // Fill with dark blue background
        for i in 0..(size * size) {
            let idx = (i * 4) as usize;
            pixels[idx] = 5;      // R
            pixels[idx + 1] = 10;  // G
            pixels[idx + 2] = 20;  // B
            pixels[idx + 3] = 255; // A
        }

        // Simple random number generator
        let mut seed = 12345u32;
        let random = |s: &mut u32| -> f32 {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (*s / 65536) as f32 / 32768.0
        };

        // Generate many bright stars
        for _ in 0..50000 {
            let x = (random(&mut seed) * size as f32) as usize;
            let y = (random(&mut seed) * size as f32) as usize;
            let brightness = (128.0 + random(&mut seed) * 127.0) as u8;

            if x < size as usize && y < size as usize {
                let idx = (y * size as usize + x) * 4;
                pixels[idx] = brightness;
                pixels[idx + 1] = brightness;
                pixels[idx + 2] = brightness;
                pixels[idx + 3] = 255;
            }
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Environment Map"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        texture
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Recreate compute texture
            self.compute_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Compute Output"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            // Recreate bind groups with new texture
            let compute_texture_view =
                self.compute_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: Some("Compute Texture View"),
                        format: Some(wgpu::TextureFormat::Rgba16Float),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: None,
                        base_array_layer: 0,
                        array_layer_count: None,
                    });

            let environment_view = self.environment_texture.create_view(&Default::default());

            // Recreate compute bind group
            let compute_bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
            self.compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.black_hole_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.disk_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&compute_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&environment_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            // Recreate display bind group
            let display_bind_group_layout = self.display_pipeline.get_bind_group_layout(0);
            self.display_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Display Bind Group"),
                layout: &display_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&compute_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Camera movement speed (km/s)
        let move_speed = 50.0 * dt;

        let forward = self.camera.forward();
        let right = self.camera.right();

        // WASD movement
        if self.input.forward {
            self.camera.position += forward * move_speed;
        }
        if self.input.backward {
            self.camera.position -= forward * move_speed;
        }
        if self.input.right {
            self.camera.position += right * move_speed;
        }
        if self.input.left {
            self.camera.position -= right * move_speed;
        }
        if self.input.up {
            self.camera.position.y += move_speed;
        }
        if self.input.down {
            self.camera.position.y -= move_speed;
        }

        // Update camera buffer
        let camera_uniform = self
            .camera
            .to_uniform(self.size.width as f32 / self.size.height as f32);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

            let workgroup_size = 8;
            let workgroups_x = (self.size.width + workgroup_size - 1) / workgroup_size;
            let workgroups_y = (self.size.height + workgroup_size - 1) / workgroup_size;

            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.display_pipeline);
            render_pass.set_bind_group(0, &self.display_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match key {
                    KeyCode::KeyW => {
                        self.input.forward = is_pressed;
                        true
                    }
                    KeyCode::KeyS => {
                        self.input.backward = is_pressed;
                        true
                    }
                    KeyCode::KeyA => {
                        self.input.left = is_pressed;
                        true
                    }
                    KeyCode::KeyD => {
                        self.input.right = is_pressed;
                        true
                    }
                    KeyCode::Space => {
                        self.input.up = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft => {
                        self.input.down = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.input.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.input.mouse_pressed {
                    if let Some(last) = self.input.last_mouse {
                        let dx = (position.x - last.0) as f32;
                        let dy = (position.y - last.1) as f32;

                        self.camera.yaw -= dx * 0.002;
                        self.camera.pitch -= dy * 0.002;

                        // Clamp pitch
                        self.camera.pitch = self
                            .camera
                            .pitch
                            .clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
                    }
                    self.input.last_mouse = Some((position.x, position.y));
                } else {
                    self.input.last_mouse = None;
                }
                true
            }
            _ => false,
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Black Hole Renderer - Rust + wgpu")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800))
            .build(&event_loop)
            .unwrap(),
    );

    let window_clone = window.clone();
    let mut renderer = pollster::block_on(Renderer::new(&window_clone));

    println!("=== Black Hole Renderer ===");
    println!("Controls:");
    println!("  WASD - Move camera");
    println!("  Space/Shift - Up/Down");
    println!("  Left Mouse + Drag - Rotate camera");
    println!("  ESC - Exit");
    println!("\nRendering black hole with:");
    println!("  Mass: 10 solar masses");
    println!("  Schwarzschild radius: 29.54 km");
    println!("  Accretion disk: 3-12 Rs");
    println!("  Physics: Full Schwarzschild geodesics");

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window.id() => {
                    if !renderer.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        state: ElementState::Pressed,
                                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                                        ..
                                    },
                                ..
                            } => elwt.exit(),
                            WindowEvent::Resized(physical_size) => {
                                renderer.resize(*physical_size);
                            }
                            WindowEvent::RedrawRequested => {
                                renderer.update();
                                match renderer.render() {
                                    Ok(_) => {}
                                    Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                                    Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                    Err(e) => eprintln!("Render error: {:?}", e),
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .unwrap();
}
