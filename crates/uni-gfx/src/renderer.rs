//! The fixed-order pass scheduler: features run in push order into shared frame targets.
//!
//! The renderer owns group 0 (the camera and exposure uniforms of `shaders/common/camera.wgsl`)
//! and is its only writer; every feature binds [`FrameBindings::camera`] there. The frame
//! targets live at the internal render resolution (`FrameCtx::viewport`), decoupled from the
//! output view; the tonemap upscales to the output. The pass order and what each feature may
//! write are documented in [`crate::targets`].

use glam::UVec2;

use crate::context::GpuContext;
use crate::frame::{CameraUniform, ExposureUniform, FrameCtx};
use crate::targets::FrameTargets;
use crate::timer::{GpuTimings, PassTimer};

/// The shared per-frame bindings a feature receives in [`RenderFeature::encode`].
pub struct FrameBindings<'a> {
    /// Group 0 for every feature (layout: [`Renderer::camera_bind_group_layout`]):
    /// binding 0 `CameraUniform`, binding 1 `ExposureUniform`.
    pub camera: &'a wgpu::BindGroup,
}

/// One render feature (sky, lens, tonemap, ...). Every render crate implements this.
pub trait RenderFeature {
    /// Stable name, used for GPU timing labels.
    fn name(&self) -> &'static str;
    /// Uploads per-frame data. The camera and exposure are the renderer's (group 0), so a
    /// feature uploads only its own parameters here.
    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets);
    /// Records this feature's passes into `enc`. `frame.camera` is bound at group 0;
    /// `output` is the output view (only the tonemap writes it).
    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        output: &wgpu::TextureView,
        timer: &mut PassTimer,
    );
    /// Rebuilds size-dependent resources after the targets change.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {}
}

/// Most timed passes per frame, the two frame brackets included; plenty for M0.
const MAX_TIMED_PASSES: u32 = 32;

/// Owns the frame targets, group 0 (camera and exposure uniforms) and the features, in push
/// order.
pub struct Renderer {
    features: Vec<Box<dyn RenderFeature>>,
    targets: FrameTargets,
    camera_buffer: wgpu::Buffer,
    exposure_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,
    timer: PassTimer,
}

impl Renderer {
    /// Creates targets at the internal resolution `internal_size`, group 0 and the pass timer.
    pub fn new(gpu: &GpuContext, internal_size: UVec2) -> Self {
        let device = &gpu.device;
        let targets = FrameTargets::new(device, internal_size);
        let camera_size = std::mem::size_of::<CameraUniform>() as u64;
        let exposure_size = std::mem::size_of::<ExposureUniform>() as u64;
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: camera_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // STORAGE and COPY_SRC too, so M1's auto-exposure can write it on the GPU (as a storage
        // binding in its own pass, or by copy) with no CPU round trip.
        let exposure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure uniform"),
            size: exposure_size,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform_entry = |binding: u32, size: u64| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX
                | wgpu::ShaderStages::FRAGMENT
                | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        };
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bgl"),
                entries: &[
                    uniform_entry(0, camera_size),
                    uniform_entry(1, exposure_size),
                ],
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bg"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exposure_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            features: Vec::new(),
            targets,
            camera_buffer,
            exposure_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            timer: PassTimer::new(gpu, MAX_TIMED_PASSES),
        }
    }

    /// Layout of group 0: binding 0 `CameraUniform`, binding 1 `ExposureUniform`. Features
    /// put it first in their pipeline layouts.
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_bind_group_layout
    }

    /// Group 0 itself (also handed to every feature as [`FrameBindings::camera`]).
    pub fn camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_bind_group
    }

    /// The exposure buffer at group 0, binding 1 (for a future GPU auto-exposure pass).
    pub fn exposure_buffer(&self) -> &wgpu::Buffer {
        &self.exposure_buffer
    }

    /// Appends a feature; features encode in push order.
    pub fn push(&mut self, f: Box<dyn RenderFeature>) {
        self.features.push(f);
    }

    /// The current frame targets.
    pub fn targets(&self) -> &FrameTargets {
        &self.targets
    }

    /// Recreates the targets at the internal resolution `size` (each axis clamped to
    /// 1..=`max_texture_dimension_2d`) and notifies every feature.
    pub fn resize(&mut self, gpu: &GpuContext, size: UVec2) {
        self.targets = FrameTargets::new(&gpu.device, size);
        for f in &mut self.features {
            f.resize(gpu, &self.targets);
        }
    }

    /// Records one frame into `output` and returns the finished command buffer.
    ///
    /// Order: collect finished GPU timings from earlier frames (non-blocking); recreate the
    /// targets if `ctx.viewport` changed (so the targets always match the viewport every
    /// feature plans with); write the camera and exposure uniforms; every feature's `prepare`,
    /// in push order; the opening frame bracket; clear `hdr` and `lens_mask` to 0 and `depth`
    /// to 0.0 (infinitely far, reversed-Z); every feature's `encode`, in push order; the
    /// closing frame bracket; close the pass timer's record (its queries are resolved once the
    /// GPU has finished the frame). The caller submits the buffer.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        ctx: &FrameCtx,
        output: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        self.timer.read_back(gpu);
        if FrameTargets::fit(&gpu.device, ctx.viewport) != self.targets.size {
            self.resize(gpu, ctx.viewport);
        }
        gpu.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::from_ctx(ctx)),
        );
        gpu.queue.write_buffer(
            &self.exposure_buffer,
            0,
            bytemuck::bytes_of(&ExposureUniform::from_exposure(ctx.exposure)),
        );
        for f in &mut self.features {
            f.prepare(gpu, ctx, &self.targets);
        }

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        // The brackets anchor on hdr, so Metal orders them around the frame's work (timer docs).
        self.timer
            .begin_frame(&gpu.device, &mut enc, &self.targets.hdr_view);
        clear_targets(&mut enc, &self.targets);
        let frame = FrameBindings {
            camera: &self.camera_bind_group,
        };
        for f in &mut self.features {
            f.encode(
                gpu,
                &mut enc,
                &frame,
                &self.targets,
                output,
                &mut self.timer,
            );
        }
        self.timer
            .end_frame(&gpu.device, &mut enc, &self.targets.hdr_view);
        self.timer.resolve(&mut enc);
        enc.finish()
    }

    /// Headless runs: after the caller has submitted a frame and waited for it, resolves and
    /// reads back its GPU timings, blocking until they are in, so they become the latest. The
    /// window never calls this: `render` collects timings without blocking, a few frames late.
    pub fn poll_timings(&mut self, gpu: &GpuContext) {
        self.timer.read_back_blocking(gpu);
    }

    /// Latest per-pass GPU times in milliseconds, `None` where the measurement was invalid;
    /// empty without timestamps or before the first read-back.
    pub fn gpu_ms(&self) -> Vec<(&'static str, Option<f32>)> {
        self.timer.last_ms()
    }

    /// Latest whole-frame GPU time in milliseconds, if measured and valid.
    pub fn gpu_frame_ms(&self) -> Option<f32> {
        self.timer.last_frame_ms()
    }

    /// The latest frame's GPU timings with its sequence number, if any.
    pub fn gpu_timings(&self) -> Option<&GpuTimings> {
        self.timer.last()
    }
}

/// One empty render pass whose load ops clear every frame target.
fn clear_targets(enc: &mut wgpu::CommandEncoder, targets: &FrameTargets) {
    let clear = |view| {
        Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })
    };
    let _pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear targets"),
        color_attachments: &[clear(&targets.hdr_view), clear(&targets.lens_mask_view)],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &targets.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}
