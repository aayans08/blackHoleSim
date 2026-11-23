// planet/body.wgsl: the bodies pass (Task B1).
//
// B0 stub: valid and neutral (draws nothing), and it imports what B1 builds on so the
// cross-crate chain composes from day one: group 0 (common/camera.wgsl), sun_visible_fraction
// (planet/eclipse.wgsl, B0's twin) and the ground-lighting group 2 (atmo/ground_lighting.wgsl,
// uni-atmo). B1 replaces everything below the imports.

#import "common/camera.wgsl"
#import "planet/eclipse.wgsl"
#import "atmo/ground_lighting.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
