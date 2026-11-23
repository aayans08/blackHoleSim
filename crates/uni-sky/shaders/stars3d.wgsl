// sky/stars3d.wgsl: the 3-D star pass "stars" (Task B4): per star, position at the frame's epoch
// from its sector origin + f32 offset + proper motion, apparent magnitude, illuminance, M0's
// energy-normalised PSF, E / Omega x star_pre() x (1 - r)(1 - g) at the far plane.
//
// B0 stub: valid and neutral (draws nothing). B4 replaces everything below the imports and keeps
// `vivid_color`, the WGSL twin of `uni_sky::vivid_color`.

#import "common/camera.wgsl"
#import "common/color.wgsl"

// The "Vivid star colours - non-physical" boost: saturation beyond Planck chromaticity that keeps
// the colour's Rec. 709 luminance. Stub: the identity (B4 implements it).
fn vivid_color(c: vec3<f32>) -> vec3<f32> {
    return c;
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
