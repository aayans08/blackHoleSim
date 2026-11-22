// common/camera.wgsl: the per-frame group 0 every render feature binds, pixel_dir and the
// pre-exposure helpers. Every render feature imports this file. (A pass that writes the
// exposure buffer imports only common/exposure.wgsl, which has no bindings.)
//
// Group 0 is owned by uni_gfx::Renderer, the only writer of its buffers (crates/uni-gfx/src/
// renderer.rs); a feature binds `FrameBindings::camera` there and never keeps a copy:
// - binding 0: `camera`, the twin of `uni_gfx::CameraUniform` (crates/uni-gfx/src/frame.rs).
//   Member order and types must match the Rust struct exactly: four mat4x4<f32>, then five
//   vec4<f32>, 336 bytes, no vec3 members. `camera_uniform_layout_matches_wgsl` checks the size
//   and every offset with naga.
// - binding 1: `exposure`, the twin of `uni_gfx::ExposureUniform` (common/exposure.wgsl).
// - binding 2: `lights`, the twin of `uni_gfx::LightUniform` (144 bytes;
//   `light_uniform_layout_matches_wgsl`).
//
// The exposure contract (pre-exposed HDR, scene_pre vs star_pre) is in common/exposure.wgsl
// and crates/uni-gfx/src/targets.rs.

#import "common/exposure.wgsl"

struct CameraUniform {
    // Root → camera rotation only; the camera sits at the origin (camera-relative rendering).
    view_rot: mat4x4<f32>,
    // Reversed-Z, infinite far plane: depth 1 at the near plane, 0 at infinity.
    proj: mat4x4<f32>,
    // inverse(proj * view_rot): clip space → root axes.
    inv_view_proj: mat4x4<f32>,
    // The previous frame's proj * view_rot (the current one on the first frame); M2 TAA.
    prev_view_proj: mat4x4<f32>,
    // Internal render resolution (the frame targets): w, h, 1/w, 1/h in pixels.
    viewport: vec4<f32>,
    // Output view (swapchain or capture) the tonemap writes: w, h, 1/w, 1/h in pixels. Equal
    // to `viewport` in captures; larger in the window, where the tonemap upscales.
    output: vec4<f32>,
    // frame_index (wrapped below 2^24), dt_real_s, fov_y [rad], aspect (w/h).
    time: vec4<f32>,
    // x: TDB seconds since J2000 modulo 2^22 s (uni_gfx::SIM_TIME_WRAP_S, ≈ 48.5 d; steps of at
    // most 0.25 s in f32), y: simulated seconds this frame, zw: projection jitter [px] (M2 TAA;
    // 0 in M1).
    sim: vec4<f32>,
    // xyz: camera translation since the previous frame [m], root axes (0 on the first frame);
    // w: 0.
    cam_delta: vec4<f32>,
}

// One light (a star), camera-relative.
struct LightGpu {
    // xyz: light centre − camera [m], root axes; w: photosphere radius [m].
    pos_radius: vec4<f32>,
    // rgb: linear Rec. 709 colour, luminance 1; w: luminous intensity [cd] (illuminance at
    // distance d is w / d² lux).
    color_intensity: vec4<f32>,
}

struct LightUniform {
    // x: number of valid entries in `lights` (≤ 4); yzw: 0.
    info: vec4<u32>,
    // 4 = uni_gfx::MAX_LIGHTS; valid entries first.
    lights: array<LightGpu, 4>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
// Written by the CPU each frame from FrameCtx::exposure; M1's auto-exposure then overwrites it
// on the GPU, metering world light before any sky light or point source is written.
@group(0) @binding(1) var<uniform> exposure: ExposureUniform;
// The first uni_gfx::MAX_LIGHTS finite lights of FrameCtx::lights.
@group(0) @binding(2) var<uniform> lights: LightUniform;

// Unit view direction, in root (ICRS) axes, through the centre of internal-resolution pixel
// `pix`.
//
// Convention:
// - `pix` is the integer pixel index (column, row) as f32; row 0 is the TOP row. In a
//   compute shader pass `vec2<f32>(gid.xy)`; in a fragment shader pass
//   `floor(position.xy)`, because @builtin(position) is already at the pixel centre.
// - The pixel centre is pix + 0.5. It maps to NDC x = 2u − 1 and y = 1 − 2v (y flipped),
//   with u, v = (pix + 0.5) / (w, h).
// - The NDC point is taken on the near plane, z = 1 (reversed-Z), w = 1, and unprojected
//   with inv_view_proj. The camera is at the origin, so the unprojected point (after the
//   divide by w) is itself the ray direction; it is normalised before returning.
fn pixel_dir(pix: vec2<f32>) -> vec3<f32> {
    let uv = (pix + vec2<f32>(0.5)) * camera.viewport.zw;
    let ndc = vec4<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, 1.0, 1.0);
    let p = camera.inv_view_proj * ndc;
    return normalize(p.xyz / p.w);
}

// The factor world light [cd/m²] is multiplied by before it is written to hdr.
fn scene_pre() -> f32 {
    return exposure.pre.x;
}

// The factor sky light and point sources [cd/m²] are multiplied by before they are written to
// hdr.
fn star_pre() -> f32 {
    return exposure.pre.y;
}

// Simulation time for shader animation: TDB seconds since J2000 modulo 2^22 s (see
// CameraUniform.sim). Animations must be periodic in it or tolerate the wrap.
fn sim_time_s() -> f32 {
    return camera.sim.x;
}
