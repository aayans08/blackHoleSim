//! GPU device, queue and (optional) window surface setup.

use std::sync::Arc;

use anyhow::Context as _;

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

/// The surface's storage format. Tonemapping renders through the sRGB view
/// ([`SURFACE_SRGB_VIEW_FORMAT`]) so the hardware encodes; egui draws to the plain view.
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const SURFACE_SRGB_VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

/// Optional device features we use when the adapter has them (global constraints, "GPU").
/// Everything must still work without `TIMESTAMP_QUERY` (the timer then records nothing).
const WANTED_FEATURES: wgpu::Features = wgpu::Features::SHADER_F16
    .union(wgpu::Features::FLOAT32_FILTERABLE)
    .union(wgpu::Features::TIMESTAMP_QUERY);

impl GpuContext {
    /// Creates a device that can present to `window`.
    pub async fn new_windowed(
        instance: wgpu::Instance,
        window: Arc<winit::window::Window>,
    ) -> anyhow::Result<Self> {
        let size = window.inner_size();
        // Arc<Window> -> Surface<'static>; the surface keeps the window alive.
        let surface = instance
            .create_surface(window)
            .context("creating the window surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .context("no Metal adapter can present to this window")?;
        let (device, queue) = request_device(&adapter).await?;

        let caps = surface.get_capabilities(&adapter);
        if !caps.formats.contains(&SURFACE_FORMAT) {
            anyhow::bail!(
                "surface does not support {SURFACE_FORMAT:?} (supported: {:?})",
                caps.formats
            );
        }
        let alpha_mode = *caps
            .alpha_modes
            .first()
            .context("surface reports no alpha modes")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: SURFACE_FORMAT,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            // Fifo is always supported and is what uni-app's windowed mode asks for.
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![SURFACE_SRGB_VIEW_FORMAT],
        };
        surface.configure(&device, &config);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: Some(SurfaceState { surface, config }),
        })
    }

    /// Creates a device with no surface, for captures and tests.
    pub async fn new_headless() -> anyhow::Result<Self> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .context("no Metal adapter available")?;
        let (device, queue) = request_device(&adapter).await?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: None,
        })
    }

    /// Reconfigures the surface; a no-op if `w` or `h` is 0.
    pub fn resize_surface(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        if let Some(s) = &mut self.surface {
            s.config.width = w;
            s.config.height = h;
            s.surface.configure(&self.device, &s.config);
        }
    }

    /// True when the device was created with `TIMESTAMP_QUERY`.
    pub fn has_timestamps(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }
}

/// Requests the device with `adapter.features() ∩ WANTED_FEATURES` and the adapter's own
/// limits (not the WebGPU defaults), as the skeleton does.
async fn request_device(adapter: &wgpu::Adapter) -> anyhow::Result<(wgpu::Device, wgpu::Queue)> {
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("main device"),
            required_features: adapter.features() & WANTED_FEATURES,
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .context("requesting the GPU device")
}
