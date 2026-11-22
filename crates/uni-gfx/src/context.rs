//! GPU device, queue and (optional) window surface setup.
//!
//! Error policy (review m6). wgpu's default uncaptured-error handler panics, which would kill
//! the window app on any per-frame validation error. [`GpuContext`] installs a handler that logs
//! the error and keeps it for [`GpuContext::take_errors`], and a device-lost callback that does
//! the same. The window shows them on the HUD; `--capture`, `--bench` and `--selftest` fail on
//! them, so a broken frame is never published as a measurement.

use std::sync::{Arc, Mutex};

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
    /// Uncaptured GPU errors and device loss since the last [`GpuContext::take_errors`].
    errors: GpuErrors,
}

/// Most errors kept between two [`GpuContext::take_errors`] calls; a per-frame error repeats
/// every frame, and the first ones say what went wrong.
const MAX_KEPT_ERRORS: usize = 16;

/// Shared between the device's error callbacks and the context.
#[derive(Clone, Default)]
struct GpuErrors(Arc<Mutex<Vec<String>>>);

impl GpuErrors {
    fn push(&self, msg: String) {
        log::error!("{msg}");
        // A poisoned lock only means another thread panicked mid-push; keep going.
        let mut v = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if v.len() < MAX_KEPT_ERRORS {
            v.push(msg);
        }
    }

    fn take(&self) -> Vec<String> {
        let mut v = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *v)
    }

    /// Routes the device's uncaptured errors and its loss into `self`.
    fn install(&self, device: &wgpu::Device) {
        let errors = self.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            errors.push(format!("GPU error: {e}"));
        }));
        let errors = self.clone();
        device.set_device_lost_callback(move |reason, msg| {
            errors.push(format!("GPU device lost ({reason:?}): {msg}"));
        });
    }
}

/// The surface's storage format. Tonemapping renders through the sRGB view
/// ([`SURFACE_SRGB_VIEW_FORMAT`]) so the hardware encodes; egui draws to the plain view.
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const SURFACE_SRGB_VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

/// Optional device features we use when the adapter has them (global constraints, "GPU"; M1
/// ruling 1). The device is created with `adapter.features() ∩ WANTED_FEATURES`, and the app
/// must run without any of the last three:
/// - no `TIMESTAMP_QUERY`: the timer records nothing and GPU times show "n/a";
/// - no `TEXTURE_COMPRESSION_BC`: maps load their 2k uncompressed copies;
/// - no `DUAL_SOURCE_BLENDING`: the atmosphere uses its grey-transmittance variant.
///
/// The M3 has all five (docs/research/wgpu30-skeleton/m3-probe-output.txt).
const WANTED_FEATURES: wgpu::Features = wgpu::Features::SHADER_F16
    .union(wgpu::Features::FLOAT32_FILTERABLE)
    .union(wgpu::Features::TIMESTAMP_QUERY)
    .union(wgpu::Features::TEXTURE_COMPRESSION_BC)
    .union(wgpu::Features::DUAL_SOURCE_BLENDING);

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

        let errors = GpuErrors::default();
        errors.install(&device);
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: Some(SurfaceState { surface, config }),
            errors,
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
        let errors = GpuErrors::default();
        errors.install(&device);
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: None,
            errors,
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

    /// Uncaptured GPU errors (and device loss) since the last call, oldest first, at most 16.
    /// Each was also logged at error level when it happened.
    pub fn take_errors(&self) -> Vec<String> {
        self.errors.take()
    }

    /// True when the device was created with `TIMESTAMP_QUERY`.
    pub fn has_timestamps(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// True when the device was created with `TEXTURE_COMPRESSION_BC` (BC7 and BC4 maps);
    /// without it, maps load their uncompressed 2k copies.
    pub fn has_bc_compression(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
    }

    /// True when the device was created with `DUAL_SOURCE_BLENDING` (`@blend_src` outputs; see
    /// `shader::validate_wgsl_with`); without it, the atmosphere uses its grey variant.
    pub fn has_dual_source_blending(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wanted_features_include_bc_and_dual_source() {
        assert!(WANTED_FEATURES.contains(
            wgpu::Features::SHADER_F16
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::DUAL_SOURCE_BLENDING
        ));
        // Nothing else: every feature we request has a fallback, and no other is needed.
        assert_eq!(WANTED_FEATURES.iter().count(), 5);
    }

    #[test]
    fn errors_are_kept_bounded_and_taken_once() {
        let e = GpuErrors::default();
        for i in 0..40 {
            e.push(format!("e{i}"));
        }
        let got = e.take();
        assert_eq!(got.len(), MAX_KEPT_ERRORS);
        assert_eq!(got[0], "e0");
        assert!(e.take().is_empty());
    }
}
