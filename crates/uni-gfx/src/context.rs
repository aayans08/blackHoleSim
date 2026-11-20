//! GPU device, queue and (optional) window surface setup.

use std::sync::Arc;

/// The window surface and its configuration.
/// `config.format` is `Bgra8Unorm` with `view_formats = [Bgra8UnormSrgb]`.
pub struct SurfaceState {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}

/// Everything needed to talk to the GPU. `surface` is `None` when headless.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: Option<SurfaceState>,
}

impl GpuContext {
    /// Creates a device that can present to `window`.
    pub async fn new_windowed(
        instance: wgpu::Instance,
        window: Arc<winit::window::Window>,
    ) -> anyhow::Result<Self> {
        todo!()
    }

    /// Creates a device with no surface, for captures and tests.
    pub async fn new_headless() -> anyhow::Result<Self> {
        todo!()
    }

    /// Reconfigures the surface; a no-op if `w` or `h` is 0.
    pub fn resize_surface(&mut self, w: u32, h: u32) {
        todo!()
    }

    /// True when the device was created with `TIMESTAMP_QUERY`.
    pub fn has_timestamps(&self) -> bool {
        todo!()
    }
}
