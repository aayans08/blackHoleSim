// sky/asteroids.wgsl: the asteroid pass "asteroids" (Task C2).
//
// C0 stub: valid and neutral (draws nothing). C2 replaces everything below the import with the
// f32 Kepler solve, the double-single camera offset and the PSF splats.

#import "common/camera.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
