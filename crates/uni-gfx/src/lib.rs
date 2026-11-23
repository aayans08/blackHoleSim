//! uni-gfx: the wgpu 30 platform layer (Metal backend).
//!
//! Device and surface setup ([`GpuContext`]), per-frame render targets ([`FrameTargets`]),
//! the fixed-order pass scheduler ([`Renderer`] + [`RenderFeature`], with the shared group 0
//! in [`FrameBindings`]), per-frame app → feature inputs ([`SharedInput`]), GPU pass timing
//! ([`PassTimer`]), the WGSL `#import` preprocessor ([`ShaderLib`]), the environment cubemap
//! convention ([`env`]), GPU probes for `--selftest` ([`probe`]) and headless capture
//! ([`CaptureTarget`], [`read_hdr`]). GPU data is f32 and camera-relative; depth is reversed-Z
//! with an infinite far plane; the HDR target is pre-exposed and at the internal render
//! resolution (see [`targets`] for the M1 frame contract).

pub mod capture;
pub mod context;
pub mod env;
pub mod frame;
pub mod probe;
pub mod proj;
pub mod renderer;
pub mod shader;
pub mod shared;
pub mod targets;
pub mod timer;

pub use capture::{CAPTURE_FORMAT, CaptureTarget, read_hdr};
pub use context::{GpuContext, SurfaceState};
pub use env::{ENV_UNIT_CD_M2, EnvMap};
pub use frame::{
    CameraUniform, Exposure, ExposureUniform, FrameCtx, LensView, LightGpu, LightUniform,
    LightView, MAX_LIGHTS, SIM_TIME_WRAP_S, exposure_scale, pre_mix,
};
pub use proj::reversed_z_infinite_proj;
pub use renderer::{FrameBindings, RenderFeature, Renderer};
pub use shader::{ShaderError, ShaderLib, validate_wgsl, validate_wgsl_with};
pub use shared::SharedInput;
pub use targets::{DEPTH_FORMAT, FrameTargets, HDR_FORMAT, HDR_MAX, LENS_MASK_FORMAT};
pub use timer::{GpuTimings, PassTimer};
