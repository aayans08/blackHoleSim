enable dual_source_blending;
// atmo/atmosphere_dual.wgsl: the dual-source variant's colour pipeline (review I7). Source 0 is
// the inscattered light x E_sun x scene_pre(), source 1 the transmittance; the blend is
// dst * T + src. Needs wgpu::Features::DUAL_SOURCE_BLENDING; ShaderLib::compose hoists the
// `enable` above to the top of the composed source. The lens-mask write is
// atmosphere_mask.wgsl (dual-source blending allows one colour target).
//
// B0 stub: valid and neutral (adds nothing, transmittance 1). B3 writes the real pass.

#import "atmo/atmosphere_common.wgsl"

struct AtmoBlendOut {
    @location(0) @blend_src(0) color: vec4<f32>,
    @location(0) @blend_src(1) transmittance: vec4<f32>,
}

@fragment
fn fs_dual(@builtin(position) pos: vec4<f32>) -> AtmoBlendOut {
    let s = atmo_segment(pos.xy);
    return AtmoBlendOut(
        vec4<f32>(min(s.inscatter * scene_pre(), vec3<f32>(HDR_MAX)), 0.0),
        vec4<f32>(s.transmittance, 1.0),
    );
}
