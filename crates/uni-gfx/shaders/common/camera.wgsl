// common/camera.wgsl: the per-frame group 0 every render feature binds, pixel_dir and the
// pre-exposure helpers. Every render feature imports this file.
//
// Group 0 is owned by uni_gfx::Renderer, the only writer of both buffers (crates/uni-gfx/src/
// renderer.rs); a feature binds `FrameBindings::camera` there and never keeps a copy:
// - binding 0: `camera`, the twin of `uni_gfx::CameraUniform` (crates/uni-gfx/src/frame.rs).
//   Member order and types must match the Rust struct exactly: three mat4x4<f32>, then three
//   vec4<f32>, 240 bytes, no vec3 members. `camera_uniform_layout_matches_wgsl` checks the size
//   and every offset with naga.
// - binding 1: `exposure`, the twin of `uni_gfx::ExposureUniform` (32 bytes, two vec4s;
//   `exposure_uniform_layout_matches_wgsl`).
//
// Pre-exposed HDR (the frame's exposure contract). The hdr target holds scene-linear Rec. 709
// light ALREADY MULTIPLIED by its exposure, so the tonemap applies AgX directly and values stay
// inside f16 range at any exposure (the Sun's disk, ~1.6e9 cd/m², is ~4.1e4 at scene EV100 15):
// - world light (debug grid, later bodies, atmosphere, band): luminance [cd/m²] × scene_pre();
// - catalogue stars (uni-sky): E/Ω × star_pre(). With star EV ≠ scene EV this is the
//   "Eye-adapted stars" dual exposure of design §7 item 4, and the HUD labels it.
// Every feature clamps what it writes to HDR_MAX, below f16's largest finite value.

struct CameraUniform {
    // Root → camera rotation only; the camera sits at the origin (camera-relative rendering).
    view_rot: mat4x4<f32>,
    // Reversed-Z, infinite far plane: depth 1 at the near plane, 0 at infinity.
    proj: mat4x4<f32>,
    // inverse(proj * view_rot): clip space → root axes.
    inv_view_proj: mat4x4<f32>,
    // Internal render resolution (the frame targets): w, h, 1/w, 1/h in pixels.
    viewport: vec4<f32>,
    // Output view (swapchain or capture) the tonemap writes: w, h, 1/w, 1/h in pixels. Equal
    // to `viewport` in captures; larger in the window, where the tonemap upscales.
    output: vec4<f32>,
    // frame_index (wrapped below 2^24), dt_real_s, fov_y [rad], aspect (w/h).
    time: vec4<f32>,
}

struct ExposureUniform {
    // x: scene pre-exposure = exposure_scale(scene EV100); y: star pre-exposure =
    // exposure_scale(star EV100); zw: 0.
    pre: vec4<f32>,
    // x: scene EV100, y: star EV100 (for reference; shaders use `pre`); zw: 0.
    ev100: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
// M0: written by the CPU each frame from FrameCtx::exposure. Target (M1): histogram
// auto-exposure writes it on the GPU, metering the hdr target before the star pass (stars are
// pre-exposed on their own scale and must not feed back into the scene exposure).
@group(0) @binding(1) var<uniform> exposure: ExposureUniform;

// Largest value any feature writes into the Rgba16Float hdr target, below the largest finite
// f16 (IEEE 754 binary16: 65504), so additive blending of a few such values stays finite in
// practice and no pass writes infinity. Twin of `uni_gfx::HDR_MAX`.
const HDR_MAX: f32 = 60000.0;

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

// Scale from scene luminance [cd/m²] to normalised exposure for a camera at `ev100`.
// Saturation-based sensitivity: L_max = (78 / (S·q)) · 2^EV100 with S = 100, q = 0.65,
// so L_max = 1.2 · 2^EV100. Source: Lagarde & de Rousiers 2014, "Moving Frostbite to
// Physically Based Rendering 3.0", §4.1.1 (after ISO 12232:2006). Twin of
// `uni_gfx::exposure_scale`.
fn exposure_scale(ev100: f32) -> f32 {
    return 1.0 / (1.2 * exp2(ev100));
}

// The factor world light [cd/m²] is multiplied by before it is written to hdr.
fn scene_pre() -> f32 {
    return exposure.pre.x;
}

// The factor catalogue-star radiance [cd/m²] is multiplied by before it is written to hdr.
fn star_pre() -> f32 {
    return exposure.pre.y;
}
