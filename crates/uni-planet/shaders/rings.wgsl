// planet/rings.wgsl: Saturn's ring pass "rings" (Task B2).
//
// B0 stub: valid and neutral (draws nothing), importing sun_visible_fraction for the planet's
// penumbra on the rings. B2 replaces everything below the imports.

#import "common/camera.wgsl"
#import "planet/eclipse.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
