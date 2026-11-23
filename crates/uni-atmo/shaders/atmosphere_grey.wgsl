// atmo/atmosphere_grey.wgsl: the grey-transmittance variant for GPUs without dual-source blending
// (review I7): the half-resolution march (fs_march, task B9p), then one composite pipeline with
// two targets. hdr gets dst * (1 - alpha) + L with
// alpha = 1 - luminance(T) (a grey approximation of the per-channel transmittance); lens_mask.g
// gets alpha and lens_mask.a the star share (atmo_star_share), both with Max (lead ruling B6 I1).

#import "atmo/atmosphere_common.wgsl"
#import "common/color.wgsl"

struct AtmoGreyOut {
    @location(0) color: vec4<f32>,
    @location(1) mask: vec4<f32>,
}

// The composite (full resolution, task B9p): the half-resolution samples upsampled, or the
// pixel's own march at an edge (atmo_composite).
@fragment
fn fs_grey(@builtin(position) pos: vec4<f32>) -> AtmoGreyOut {
    let s = atmo_composite(pos.xy);
    let alpha = clamp(1.0 - luminance(s.transmittance), 0.0, 1.0);
    return AtmoGreyOut(
        vec4<f32>(s.light, alpha),
        vec4<f32>(0.0, alpha, 0.0, atmo_star_share()),
    );
}

// The half-resolution march into the scratch targets (atmo_march_half).
@fragment
fn fs_march(@builtin(position) pos: vec4<f32>) -> AtmoHalfOut {
    return atmo_march_half(pos.xy);
}
