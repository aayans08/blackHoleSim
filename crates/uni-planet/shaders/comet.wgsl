// planet/comet.wgsl: the comet pass "comets" (Task C1).
//
// C0 stub: valid and neutral (draws nothing). C1 replaces everything below the import with the
// nucleus mesh, the coma quad and the tail sprites.

#import "common/camera.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
