// sky/band_draw.wgsl: the band draw pass "band" (Task B5). A full-screen triangle on the far
// plane samples the environment cubemap (mip 0) in each pixel's view direction and adds it to hdr
// as sky light:
//     hdr += units × ENV_UNIT_CD_M2 × star_pre() × (1 − r)(1 − g)(1 − b), clamped to HDR_MAX,
// with r, g, b the lens_mask channels (M1 frame contract, uni_gfx::targets): r the captured
// fraction, g the sky occlusion by bodies, rings and atmospheres, b where the lens pass already
// drew the lensed environment. The pipeline blends additively and tests depth GreaterEqual
// against the cleared 0.0 without writing it, so the band shows only where no body was drawn
// (ruling 4: the band comes after the opaque passes and the meter).

#import "common/camera.wgsl"

// Luminance [cd/m²] of one texel unit. Twin of uni_gfx::ENV_UNIT_CD_M2.
const BAND_ENV_UNIT_CD_M2: f32 = 1.0e-4;

// The environment cubemap (uni_gfx::EnvMap, written by band_update.wgsl): texel units, root axes.
@group(1) @binding(0) var band_env: texture_cube<f32>;
@group(1) @binding(1) var band_sampler: sampler;
// uni_gfx::FrameTargets::lens_mask, read with textureLoad at integer pixels.
@group(1) @binding(2) var band_lens_mask: texture_2d<f32>;

// A triangle covering the screen, at depth 0 (the far plane of the reversed-Z projection).
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // @builtin(position) is at the pixel centre; pixel_dir takes the integer pixel index.
    let pix = floor(pos.xy);
    let units = textureSampleLevel(band_env, band_sampler, pixel_dir(pix), 0.0).rgb;
    let mask = textureLoad(band_lens_mask, vec2<i32>(pix), 0);
    let visible = saturate(1.0 - mask.r) * saturate(1.0 - mask.g) * saturate(1.0 - mask.b);
    let light = units * (BAND_ENV_UNIT_CD_M2 * star_pre() * visible);
    return vec4<f32>(clamp(light, vec3<f32>(0.0), vec3<f32>(HDR_MAX)), 0.0);
}
