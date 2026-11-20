//! The fixed-order pass scheduler: features run in push order into shared frame targets.

use glam::UVec2;

use crate::context::GpuContext;
use crate::frame::{CameraUniform, FrameCtx};
use crate::targets::FrameTargets;
use crate::timer::PassTimer;

/// One render feature (sky, lens, tonemap, ...). Every render crate implements this.
pub trait RenderFeature {
    /// Stable name, used for GPU timing labels.
    fn name(&self) -> &'static str;
    /// Uploads per-frame data.
    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets);
    /// Records this feature's passes into `enc`.
    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        targets: &FrameTargets,
        output: &wgpu::TextureView,
        timer: &mut PassTimer,
    );
    /// Rebuilds size-dependent resources after the targets change.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {}
}

/// Most timed passes per frame; plenty for M0's lens, stars and tonemap features.
const MAX_TIMED_PASSES: u32 = 32;

/// Owns the frame targets, the camera uniform and the features, in push order.
pub struct Renderer {
    features: Vec<Box<dyn RenderFeature>>,
    targets: FrameTargets,
    camera_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,
    timer: PassTimer,
}

impl Renderer {
    /// Creates targets at `size`, the camera buffer and bind group, and the pass timer.
    pub fn new(gpu: &GpuContext, size: UVec2) -> Self {
        let device = &gpu.device;
        let targets = FrameTargets::new(device, size);
        let uniform_size = std::mem::size_of::<CameraUniform>() as u64;
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(uniform_size),
                    },
                    count: None,
                }],
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bg"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        Self {
            features: Vec::new(),
            targets,
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            timer: PassTimer::new(gpu, MAX_TIMED_PASSES),
        }
    }

    /// Layout of the camera bind group (group 0, binding 0 = `CameraUniform`).
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_bind_group_layout
    }

    /// The camera bind group for group 0.
    pub fn camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_bind_group
    }

    /// Appends a feature; features encode in push order.
    pub fn push(&mut self, f: Box<dyn RenderFeature>) {
        self.features.push(f);
    }

    /// The current frame targets.
    pub fn targets(&self) -> &FrameTargets {
        &self.targets
    }

    /// Recreates targets at `size` (0 is clamped to 1) and notifies every feature.
    pub fn resize(&mut self, gpu: &GpuContext, size: UVec2) {
        self.targets = FrameTargets::new(&gpu.device, size);
        for f in &mut self.features {
            f.resize(gpu, &self.targets);
        }
    }

    /// Records one frame into `output` and returns the finished command buffer.
    ///
    /// Order: collect finished GPU timings from earlier frames (non-blocking); write the
    /// camera uniform; every feature's `prepare`, in push order; clear `hdr` and
    /// `lens_mask` to 0 and `depth` to 0.0 (infinitely far, reversed-Z); every feature's
    /// `encode`, in push order; resolve the pass timer. The caller submits the buffer.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        ctx: &FrameCtx,
        output: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        self.timer.read_back(gpu);
        gpu.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::from_ctx(ctx)),
        );
        for f in &mut self.features {
            f.prepare(gpu, ctx, &self.targets);
        }

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        clear_targets(&mut enc, &self.targets);
        for f in &mut self.features {
            f.encode(gpu, &mut enc, &self.targets, output, &mut self.timer);
        }
        self.timer.resolve(&mut enc);
        enc.finish()
    }

    /// Latest per-pass GPU times in milliseconds; empty without timestamps.
    pub fn gpu_ms(&self) -> Vec<(&'static str, f32)> {
        self.timer.last_ms()
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
