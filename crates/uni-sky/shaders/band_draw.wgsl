// sky/band_draw.wgsl: the band draw pass "band" (Task B5): fullscreen triangle, pixel_dir ->
// cubemap mip 0 x ENV_UNIT_CD_M2 x star_pre() x (1 - r)(1 - g)(1 - b), additive, depth test
// GreaterEqual at depth 0 (ruling 4), clamped to HDR_MAX.
//
// B0 stub: valid and neutral (adds nothing). B5 replaces everything below the imports.

#import "common/camera.wgsl"
#import "common/cube.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
