// atmo/atmosphere_grey.wgsl: the grey-transmittance variant for GPUs without dual-source blending
// (review I7): one pipeline, two targets. hdr gets dst * (1 - alpha) + L with
// alpha = 1 - luminance(T); lens_mask.g gets alpha with Max.
//
// B0 stub: valid and neutral (adds nothing, alpha 0). B3 writes the real pass.

#import "atmo/atmosphere_common.wgsl"
#import "common/color.wgsl"

struct AtmoGreyOut {
    @location(0) color: vec4<f32>,
    @location(1) mask: vec4<f32>,
}

@fragment
fn fs_grey(@builtin(position) pos: vec4<f32>) -> AtmoGreyOut {
    let s = atmo_segment(pos.xy);
    let alpha = 1.0 - luminance(s.transmittance);
    return AtmoGreyOut(
        vec4<f32>(min(s.inscatter * scene_pre(), vec3<f32>(HDR_MAX)), alpha),
        vec4<f32>(0.0, alpha, 0.0, 0.0),
    );
}
