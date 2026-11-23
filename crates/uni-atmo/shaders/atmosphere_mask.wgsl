// atmo/atmosphere_mask.wgsl: the dual-source variant's second pipeline, writing
// lens_mask.g = 1 - luminance(T) and lens_mask.a = the star share (atmo_star_share), both with
// BlendOperation::Max (M1 frame contract, lens_mask channels; lead ruling B6 I1). It repeats the
// march's steps for the transmittance only (no LUT samples, no scattering), and discards the same
// pixels as the colour pipeline.

#import "atmo/atmosphere_common.wgsl"
#import "common/color.wgsl"

@fragment
fn fs_mask(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let t = atmo_view_transmittance(pos.xy);
    return vec4<f32>(0.0, clamp(1.0 - luminance(t), 0.0, 1.0), 0.0, atmo_star_share());
}
