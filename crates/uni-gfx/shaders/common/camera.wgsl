// common/camera.wgsl: the per-frame camera uniform at group(0) binding(0), pixel_dir and
// exposure_scale. Every render feature imports this file.
//
// `CameraUniform` is the twin of `uni_gfx::CameraUniform` (crates/uni-gfx/src/frame.rs).
// Member order and types must match the Rust struct exactly: three mat4x4<f32>, then three
// vec4<f32>, 240 bytes, no vec3 members. `camera_uniform_layout_matches_wgsl` checks the
// size and every offset with naga.
struct CameraUniform {
    // Root → camera rotation only; the camera sits at the origin (camera-relative rendering).
    view_rot: mat4x4<f32>,
    // Reversed-Z, infinite far plane: depth 1 at the near plane, 0 at infinity.
    proj: mat4x4<f32>,
    // inverse(proj * view_rot): clip space → root axes.
    inv_view_proj: mat4x4<f32>,
    // w, h, 1/w, 1/h in pixels.
    viewport: vec4<f32>,
    // scene_ev100, star_ev100, 0, 0.
    exposure: vec4<f32>,
    // frame_index (wrapped below 2^24), dt_real_s, fov_y [rad], aspect (w/h).
    time: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Unit view direction, in root (ICRS) axes, through the centre of pixel `pix`.
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
// Physically Based Rendering 3.0", §4.1.1 (after ISO 12232:2006).
fn exposure_scale(ev100: f32) -> f32 {
    return 1.0 / (1.2 * exp2(ev100));
}
