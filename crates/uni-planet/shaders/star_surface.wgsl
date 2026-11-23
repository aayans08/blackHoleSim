// planet/star_surface.wgsl: the star-surface pass "sun" (Task B2): limb darkening, granulation.
//
// B0 stub: valid and neutral (draws nothing). B2 replaces everything below the imports.

#import "common/camera.wgsl"
#import "common/hash.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
