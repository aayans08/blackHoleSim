// rel/lens_trace.wgsl: the lens pass (brute-force Schwarzschild ground truth; M0 Task 9, M1 Task
// B7).
//
// One compute pass, two dispatches (crates/uni-rel/src/tracer.rs):
// 1. cs_trace, over the traced grid: for each sample of the lens rectangle (at the resolution
//    scale), build the camera ray with pixel_dir, apply static_observer_ray, integrate it with
//    binet_trace (r_far = max(2·r_cam, 1000) M, at most 2000 steps) and store the exit
//    direction and capture flag.
// 2. cs_resolve, over the lens rectangle only: upsample the samples bilinearly into hdr
//    (captured → black) and lens_mask (r = captured fraction, g = 0, b = 1, a = 0; M1 frame
//    contract, crates/uni-gfx/src/targets.rs). If the camera is inside the horizon, the
//    rectangle is the whole frame and every pixel is black with lens_mask (1, 0, 1, 0).
//    Pixels a body drew (depth ≠ 0) are skipped in both cases: foreground objects are not
//    lensed (design §5.3, review m7), and they keep their colour and mask.
//
// The environment escaped rays show (LensEnvParams.mode.x, set from uni_rel::LensEnv):
// - LENS_ENV_BAND: the environment cubemap (uni_gfx::EnvMap, the Milky Way band written by
//   uni-sky's band-update) at the exit direction, as sky light: texel units ×
//   LENS_ENV_UNIT_CD_M2 × star_pre(), clamped to HDR_MAX. The band pass draws the same cubemap,
//   at level 0, around the rectangle and skips the pixels this pass wrote (lens_mask.b).
// - LENS_ENV_DEBUG_GRID: the M0 debug grid (rel/debug_sky.wgsl), display-referred, exactly as
//   at m0.1; rel/debug_background.wgsl draws the same grid around the rectangle.
// Either way the rectangle's edge is invisible: there the deflection is under half a pixel.
//
// The cubemap's mip level comes from each sample's footprint on the sky: the angle between the
// exit directions of neighbouring samples, i.e. the Jacobian of exit direction against screen
// position by finite differences, times the sample spacing (physics research §1.4, the cheap
// substitute for DNGR's ray bundles; James et al. 2015). Strongly demagnified sky near the
// photon ring is therefore read from a coarse level instead of aliasing.
//
// The pass writes nothing outside its rectangle (uni-gfx targets.rs, pass contract): there the
// deflection is under half a pixel, and what was drawn before the lens pass (or the band drawn
// after it) stays.
//
// Step budget. The step cap is 5 M, so 2000 steps carry a ray at most about 1e4 M. Measured
// with the CPU twin, as the camera moves out:
// - from about 7500 M, rays grazing the photon sphere run out before they leave it;
// - from about 8600 M, rays inside the shadow are no longer captured;
// - beyond about 1e4 M, every ray runs out before it gets near the hole, so all of the
//   deflection and the capture are lost, not just the end of the path.
// The host therefore traces only cameras within TRACE_REACH = 5000 M of the hole
// (crates/uni-rel/src/tracer.rs). Farther out the grid is empty, and the frame shows the
// unlensed sky with no shadow, which the app labels on the HUD (LensCoverage::BeyondReach).
//
// Within the reach, an escaping ray usually runs out on its way back out. cs_trace then shades
// it along its last tangent. It has passed periapsis by then, and the deflection still to come
// is under 2e-5 rad; the CPU twin checks this in gpu_budget_is_exact_within_trace_reach. Rays
// that orbit the photon sphere many times, in the sub-pixel band around the critical impact
// parameter, still lose the rest of their path.

#import "common/camera.wgsl"
#import "rel/binet.wgsl"
#import "rel/debug_sky.wgsl"

// Twin of `LensParams` in crates/uni-rel/src/tracer.rs: vec4 members only (Global
// Constraints, uniform layout rule). `lens_params_layout_unchanged` checks it with naga.
struct LensParams {
    // xyz: camera position relative to the hole [M], root axes; w: r_cam [M].
    cam: vec4<f32>,
    // Traced rectangle in full-resolution pixels: min x, min y, width, height (the whole frame
    // when the camera is inside the horizon).
    rect: vec4<u32>,
    // Samples across the rectangle (x, y; 0 = nothing traced), max steps,
    // camera inside the horizon (1 = the whole frame is black).
    grid: vec4<u32>,
    // Full-resolution pixels per sample (x, y), r_far [M], 1 / resolution scale.
    scale: vec4<f32>,
}

// Twin of `LensEnvParams` in crates/uni-rel/src/tracer.rs (one vec4;
// `lens_env_params_layout_matches_wgsl`).
struct LensEnvParams {
    // x: the environment escaped rays show, LENS_ENV_DEBUG_GRID or LENS_ENV_BAND; yzw: 0.
    mode: vec4<u32>,
}

// LensEnvParams.mode.x codes (twins of the constants in tracer.rs, `lens_env_constants_match_wgsl`).
const LENS_ENV_DEBUG_GRID: u32 = 0u;
const LENS_ENV_BAND: u32 = 1u;
// Luminance [cd/m²] of one environment texel unit. Twin of uni_gfx::ENV_UNIT_CD_M2.
const LENS_ENV_UNIT_CD_M2: f32 = 1.0e-4;

@group(1) @binding(0) var<uniform> lens: LensParams;
// Ray samples (cs_trace writes, cs_resolve reads): xyz = exit direction in root axes,
// w = 1 if captured (xyz = 0), 0 if escaped.
@group(1) @binding(1) var samples_out: texture_storage_2d<rgba32float, write>;
@group(1) @binding(2) var samples_in: texture_2d<f32>;
@group(1) @binding(3) var hdr_out: texture_storage_2d<rgba16float, write>;
// uni_gfx::LENS_MASK_FORMAT (M1 frame contract, crates/uni-gfx/src/targets.rs), written only
// on the pixels this pass repaints: r = captured fraction of the pixel (sky passes multiply by
// 1 − r), g = 0 (no sky occlusion on a repainted pixel), b = 1 (the lensed environment is here;
// the band multiplies by 1 − b), a = 0 (reserved for M3).
@group(1) @binding(4) var lens_mask_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var<uniform> lens_env: LensEnvParams;
// uni_gfx::FrameTargets::depth (reversed-Z, cleared to 0 = infinitely far). Nothing is attached
// in this compute pass; every opaque pass has already run.
@group(1) @binding(6) var lens_depth: texture_depth_2d;
// The environment cubemap (uni_gfx::EnvMap): texel units of LENS_ENV_UNIT_CD_M2, root axes,
// faces in the WebGPU convention (common/cube.wgsl), and its trilinear sampler.
@group(1) @binding(7) var lens_env_map: texture_cube<f32>;
@group(1) @binding(8) var lens_env_sampler: sampler;

// The full-resolution pixel index whose centre (pixel_dir adds 0.5) is the centre of sample
// `s`. At resolution scale 1 this is exactly the integer pixel index.
fn sample_pixel(s: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(lens.rect.xy) + (s + 0.5) * lens.scale.xy - 0.5;
}

@compute @workgroup_size(8, 8, 1)
fn cs_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= lens.grid.x || gid.y >= lens.grid.y) {
        return;
    }
    let cam = lens.cam.xyz;
    let v0 = static_observer_ray(cam, pixel_dir(sample_pixel(vec2<f32>(gid.xy))));
    let end = binet_trace(cam, v0, lens.scale.z, lens.grid.z);
    var result = vec4<f32>(normalize(end.v), 0.0);
    if (end.status == BINET_CAPTURED) {
        result = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    textureStore(samples_out, gid.xy, result);
}

// Mip level of a cube with `face_texels` texels per face edge and `levels` mips for a sample
// whose footprint on the sky is `footprint` [rad], in direction `dir` (any length). A face is
// the gnomonic projection of the sky onto the plane at distance 1, which stretches angles
// radially by 1/m² (m = the unit direction's major-axis component), so a texel spans 2m²/N rad
// there (2/N at a face centre). The level is log2(footprint / that angle), clamped to
// [0, levels − 1]: the level a GPU picks from the larger screen-space derivative of the face
// coordinates (Vulkan specification, "Scale Factor Operation, LOD Operation and Image Level(s)
// Selection"). CPU twin: uni_rel::tracer::env_lod.
fn lens_env_lod(dir: vec3<f32>, footprint: f32, face_texels: f32, levels: f32) -> f32 {
    let d = abs(normalize(dir));
    let m = max(d.x, max(d.y, d.z));
    let texel = 2.0 * m * m / face_texels;
    let lod = log2(max(footprint, 1e-12) / texel);
    return max(min(lod, levels - 1.0), 0.0);
}

// The environment in direction `dir` (root axes) for a sample of footprint `footprint` [rad],
// in texel units (absolute luminance / LENS_ENV_UNIT_CD_M2), at lens_env_lod's level.
fn lens_env_units(env: texture_cube<f32>, env_sampler: sampler, dir: vec3<f32>, footprint: f32) -> vec3<f32> {
    let size = textureDimensions(env);
    let lod = lens_env_lod(dir, footprint, f32(size.x), f32(textureNumLevels(env)));
    return textureSampleLevel(env, env_sampler, dir, lod).rgb;
}

// Angle (as a chord) between the exit directions of two samples; 0 if either was captured.
fn sample_gap(a: vec4<f32>, b: vec4<f32>) -> f32 {
    if (a.w > 0.5 || b.w > 0.5) {
        return 0.0;
    }
    return length(a.xyz - b.xyz);
}

// Pre-exposed colour (rgb) and captured flag (a) of one sample, `footprint` [rad] across.
fn shade_sample(s: vec4<f32>, footprint: f32) -> vec4<f32> {
    if (s.w > 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if (lens_env.mode.x == LENS_ENV_BAND) {
        // Sky light at star exposure (M1 frame contract: hdr holds luminance × star_pre()).
        let units = lens_env_units(lens_env_map, lens_env_sampler, s.xyz, footprint);
        let light = units * (LENS_ENV_UNIT_CD_M2 * star_pre());
        return vec4<f32>(clamp(light, vec3<f32>(0.0), vec3<f32>(HDR_MAX)), 0.0);
    }
    return vec4<f32>(debug_sky(s.xyz, footprint), 0.0);
}

// Dispatched over the rectangle only: `gid` is the pixel's offset inside it.
// CPU model: `resolve_model` in tracer.rs's tests, statement for statement.
@compute @workgroup_size(8, 8, 1)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= lens.rect.z || gid.y >= lens.rect.w) {
        return;
    }
    let p = lens.rect.xy + gid.xy;
    let dims = textureDimensions(hdr_out);
    if (p.x >= dims.x || p.y >= dims.y) {
        return;
    }
    if (textureLoad(lens_depth, p, 0) != 0.0) {
        // A body is in front of the environment here: it is not lensed (review m7).
        return;
    }
    if (lens.grid.w != 0u) {
        // Camera inside the horizon: no light from outside reaches it.
        textureStore(hdr_out, p, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(lens_mask_out, p, vec4<f32>(1.0, 0.0, 1.0, 0.0));
        return;
    }
    let pf = vec2<f32>(p);
    let rel = vec2<f32>(gid.xy);
    // Bilinear upsampling: this pixel's centre in sample coordinates (samples sit at their
    // centres, see sample_pixel). At scale 1, q is the integer sample index and t = 0.
    let q = (rel + 0.5) / lens.scale.xy - 0.5;
    let q0 = floor(q);
    let t = clamp(q - q0, vec2<f32>(0.0), vec2<f32>(1.0));
    let last = vec2<i32>(lens.grid.xy) - 1;
    let i0 = clamp(vec2<i32>(q0), vec2<i32>(0), last);
    let i1 = clamp(vec2<i32>(q0) + 1, vec2<i32>(0), last);
    let s00 = textureLoad(samples_in, i0, 0);
    let s10 = textureLoad(samples_in, vec2<i32>(i1.x, i0.y), 0);
    let s01 = textureLoad(samples_in, vec2<i32>(i0.x, i1.y), 0);
    let s11 = textureLoad(samples_in, i1, 0);
    // Footprint of one sample on the sky (antialiasing; the grid's line width, the band's mip
    // level): the widest gap between neighbouring escaped samples, i.e. the larger column of
    // the finite-difference Jacobian times the sample spacing. As a chord it is the angle to
    // within 0.07% wherever it selects a level below the last (≤ 0.125 rad on the band's
    // 256-texel cube). Large across the photon ring, where the sky is strongly demagnified.
    var fp = max(max(sample_gap(s00, s10), sample_gap(s01, s11)), max(sample_gap(s00, s01), sample_gap(s10, s11)));
    if (fp <= 0.0) {
        // No escaped neighbour pair (rectangle corner or shadow edge): use the unlensed size.
        fp = pixel_footprint(pf, pixel_dir(pf)) * max(lens.scale.x, lens.scale.y);
    }
    let top = mix(shade_sample(s00, fp), shade_sample(s10, fp), t.x);
    let bottom = mix(shade_sample(s01, fp), shade_sample(s11, fp), t.x);
    let c = mix(top, bottom, t.y);
    textureStore(hdr_out, p, vec4<f32>(c.rgb, 1.0));
    textureStore(lens_mask_out, p, vec4<f32>(c.a, 0.0, 1.0, 0.0));
}
