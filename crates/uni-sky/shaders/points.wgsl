// sky/points.wgsl: the point-sprite pass "points" (Task B4): bodies, the Sun and near stars as
// PSF splats, x star_pre() x (1 - r), vertex depth from the true distance, compare Greater.
//
// B0 stub: valid and neutral (draws nothing). B4 replaces everything below the import.

#import "common/camera.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
