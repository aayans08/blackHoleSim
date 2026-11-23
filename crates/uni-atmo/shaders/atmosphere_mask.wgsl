// atmo/atmosphere_mask.wgsl: the dual-source variant's second pipeline, writing
// lens_mask.g = 1 - luminance(T) with BlendOperation::Max (M1 frame contract, lens_mask channels).
//
// B0 stub: valid and neutral (g = 0). B3 writes the real pass.

#import "atmo/atmosphere_common.wgsl"
#import "common/color.wgsl"

@fragment
fn fs_mask(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let s = atmo_segment(pos.xy);
    return vec4<f32>(0.0, 1.0 - luminance(s.transmittance), 0.0, 0.0);
}
