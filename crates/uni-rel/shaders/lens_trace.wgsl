// rel/lens_trace.wgsl: the M0 lens pass (brute-force Schwarzschild ground truth).
//
// One compute pass, two dispatches (crates/uni-rel/src/tracer.rs):
// 1. cs_trace, over the traced grid: for each sample of the lens rectangle (at the resolution
//    scale), build the camera ray with pixel_dir, apply static_observer_ray, integrate it with
//    binet_trace (r_far = max(2·r_cam, 1000) M, at most 2000 steps) and store the exit
//    direction and capture flag.
// 2. cs_resolve, over the full frame: inside the rectangle, upsample the samples bilinearly
//    (captured → black and lens_mask 1); outside it, draw the unlensed debug sky, since the
//    deflection there is under half a pixel. If the camera is inside the horizon, every pixel
//    is black with lens_mask 1.
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
// Constraints, uniform layout rule). `lens_params_layout_matches_wgsl` checks it with naga.
struct LensParams {
    // xyz: camera position relative to the hole [M], root axes; w: r_cam [M].
    cam: vec4<f32>,
    // Traced rectangle in full-resolution pixels: min x, min y, width, height.
    rect: vec4<u32>,
    // Samples across the rectangle (x, y; 0 = nothing traced), max steps,
    // camera inside the horizon (1 = the whole frame is black).
    grid: vec4<u32>,
    // Full-resolution pixels per sample (x, y), r_far [M], 1 / resolution scale.
    scale: vec4<f32>,
}

@group(1) @binding(0) var<uniform> lens: LensParams;
// Ray samples (cs_trace writes, cs_resolve reads): xyz = exit direction in root axes,
// w = 1 if captured (xyz = 0), 0 if escaped.
@group(1) @binding(1) var samples_out: texture_storage_2d<rgba32float, write>;
@group(1) @binding(2) var samples_in: texture_2d<f32>;
@group(1) @binding(3) var hdr_out: texture_storage_2d<rgba16float, write>;
// 1.0 where the ray was captured (read by the star pass as 1 − lens_mask).
@group(1) @binding(4) var lens_mask_out: texture_storage_2d<r32float, write>;

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

// Unlensed angular size [rad] of full-resolution pixel `p`, whose direction is `d`.
fn pixel_footprint(p: vec2<f32>, d: vec3<f32>) -> f32 {
    let dx = length(pixel_dir(p + vec2<f32>(1.0, 0.0)) - d);
    let dy = length(pixel_dir(p + vec2<f32>(0.0, 1.0)) - d);
    return max(dx, dy);
}

// Angle (as a chord) between the exit directions of two samples; 0 if either was captured.
fn sample_gap(a: vec4<f32>, b: vec4<f32>) -> f32 {
    if (a.w > 0.5 || b.w > 0.5) {
        return 0.0;
    }
    return length(a.xyz - b.xyz);
}

// Colour (rgb) and lens mask (a) of one sample.
fn shade_sample(s: vec4<f32>, footprint: f32, ev100: f32) -> vec4<f32> {
    if (s.w > 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(debug_sky(s.xyz, footprint, ev100), 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(hdr_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let p = gid.xy;
    if (lens.grid.w != 0u) {
        // Camera inside the horizon: no light from outside reaches it.
        textureStore(hdr_out, p, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(lens_mask_out, p, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }
    let ev100 = camera.exposure.x;
    let pf = vec2<f32>(p);
    let rel = vec2<i32>(p) - vec2<i32>(lens.rect.xy);
    let traced = lens.grid.x > 0u && lens.grid.y > 0u;
    let in_rect = traced && all(rel >= vec2<i32>(0)) && all(rel < vec2<i32>(lens.rect.zw));
    if (!in_rect) {
        // Unlensed. Lines match the traced region's width (one sample = 1/scale pixels).
        let d = pixel_dir(pf);
        let c = debug_sky(d, pixel_footprint(pf, d) * lens.scale.w, ev100);
        textureStore(hdr_out, p, vec4<f32>(c, 1.0));
        return;
    }
    // Bilinear upsampling: this pixel's centre in sample coordinates (samples sit at their
    // centres, see sample_pixel). At scale 1, q is the integer sample index and t = 0.
    let q = (vec2<f32>(rel) + 0.5) / lens.scale.xy - 0.5;
    let q0 = floor(q);
    let t = clamp(q - q0, vec2<f32>(0.0), vec2<f32>(1.0));
    let last = vec2<i32>(lens.grid.xy) - 1;
    let i0 = clamp(vec2<i32>(q0), vec2<i32>(0), last);
    let i1 = clamp(vec2<i32>(q0) + 1, vec2<i32>(0), last);
    let s00 = textureLoad(samples_in, i0, 0);
    let s10 = textureLoad(samples_in, vec2<i32>(i1.x, i0.y), 0);
    let s01 = textureLoad(samples_in, vec2<i32>(i0.x, i1.y), 0);
    let s11 = textureLoad(samples_in, i1, 0);
    // Footprint of one sample on the sky (antialiasing): the widest gap between neighbouring
    // escaped samples. Large across the photon ring, where the sky is strongly demagnified.
    var fp = max(max(sample_gap(s00, s10), sample_gap(s01, s11)), max(sample_gap(s00, s01), sample_gap(s10, s11)));
    if (fp <= 0.0) {
        // No escaped neighbour pair (rectangle corner or shadow edge): use the unlensed size.
        fp = pixel_footprint(pf, pixel_dir(pf)) * max(lens.scale.x, lens.scale.y);
    }
    let top = mix(shade_sample(s00, fp, ev100), shade_sample(s10, fp, ev100), t.x);
    let bottom = mix(shade_sample(s01, fp, ev100), shade_sample(s11, fp, ev100), t.x);
    let c = mix(top, bottom, t.y);
    textureStore(hdr_out, p, vec4<f32>(c.rgb, 1.0));
    textureStore(lens_mask_out, p, vec4<f32>(c.a, 0.0, 0.0, 0.0));
}
