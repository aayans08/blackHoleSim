//! uni-gfx: the wgpu 30 platform layer (Metal backend).
//!
//! Device and surface setup ([`GpuContext`]), per-frame render targets ([`FrameTargets`]),
//! the fixed-order pass scheduler ([`Renderer`] + [`RenderFeature`]), GPU pass timing
//! ([`PassTimer`]), the WGSL `#import` preprocessor ([`ShaderLib`]) and headless
//! capture ([`CaptureTarget`]). GPU data is f32 and camera-relative; depth is
//! reversed-Z with an infinite far plane.

pub mod capture;
pub mod context;
pub mod frame;
pub mod proj;
pub mod renderer;
pub mod shader;
pub mod targets;
pub mod timer;

pub use capture::{CAPTURE_FORMAT, CaptureTarget};
pub use context::{GpuContext, SurfaceState};
pub use frame::{CameraUniform, Exposure, FrameCtx, LensView};
pub use proj::reversed_z_infinite_proj;
pub use renderer::{RenderFeature, Renderer};
pub use shader::{ShaderError, ShaderLib, validate_wgsl};
pub use targets::{DEPTH_FORMAT, FrameTargets, HDR_FORMAT, LENS_MASK_FORMAT};
pub use timer::PassTimer;
