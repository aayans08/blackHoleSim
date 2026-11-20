//! The fixed-order pass scheduler: features run in push order into shared frame targets.

use glam::UVec2;

use crate::context::GpuContext;
use crate::frame::FrameCtx;
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
        todo!()
    }

    /// Layout of the camera bind group (group 0, binding 0 = `CameraUniform`).
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        todo!()
    }

    /// The camera bind group for group 0.
    pub fn camera_bind_group(&self) -> &wgpu::BindGroup {
        todo!()
    }

    /// Appends a feature; features encode in push order.
    pub fn push(&mut self, f: Box<dyn RenderFeature>) {
        todo!()
    }

    /// The current frame targets.
    pub fn targets(&self) -> &FrameTargets {
        todo!()
    }

    /// Recreates targets at `size` (0 is clamped to 1) and notifies every feature.
    pub fn resize(&mut self, gpu: &GpuContext, size: UVec2) {
        todo!()
    }

    /// Records one frame into `output` and returns the finished command buffer.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        ctx: &FrameCtx,
        output: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        todo!()
    }

    /// Latest per-pass GPU times in milliseconds; empty without timestamps.
    pub fn gpu_ms(&self) -> Vec<(&'static str, f32)> {
        todo!()
    }
}
