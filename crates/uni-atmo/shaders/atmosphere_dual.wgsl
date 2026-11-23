enable dual_source_blending;
// atmo/atmosphere_dual.wgsl: the dual-source variant's march and colour pipelines (review I7;
// task B9p: the march at half resolution, fs_march, then the full-resolution composite,
// fs_dual). Source 0 is the inscattered light x E_sun x the Sun's colour x the exposure, source
// 1 the transmittance;
// the blend is dst * T + src (src factor One, dst factor Src1), so the scene behind is
// attenuated per channel. Needs wgpu::Features::DUAL_SOURCE_BLENDING; ShaderLib::compose hoists
// the `enable` above to the top of the composed source. The lens-mask write is
// atmosphere_mask.wgsl (dual-source blending allows one colour target).

#import "atmo/atmosphere_common.wgsl"

struct AtmoBlendOut {
    @location(0) @blend_src(0) color: vec4<f32>,
    @location(0) @blend_src(1) transmittance: vec4<f32>,
}

// The composite (full resolution, task B9p): the half-resolution samples upsampled, or the
// pixel's own march at an edge (atmo_composite).
@fragment
fn fs_dual(@builtin(position) pos: vec4<f32>) -> AtmoBlendOut {
    let s = atmo_composite(pos.xy);
    return AtmoBlendOut(
        vec4<f32>(s.light, 0.0),
        vec4<f32>(s.transmittance, 1.0),
    );
}

// The half-resolution march into the scratch targets (atmo_march_half).
@fragment
fn fs_march(@builtin(position) pos: vec4<f32>) -> AtmoHalfOut {
    return atmo_march_half(pos.xy);
}
