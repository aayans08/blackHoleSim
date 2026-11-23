// atmo/atmosphere_common.wgsl: the per-pixel view march shared by the three draw variants
// (atmosphere_dual, atmosphere_mask, atmosphere_grey). B3: intersect the shell, clip by the scene
// depth (reversed-Z infinite) and the ground, march 32 steps (48 on long limb paths) with single
// scattering through the transmittance LUT plus Hillaire 2020's multiple-scattering term.
//
// B0 stub: valid and neutral (no atmosphere: nothing scattered, transmittance 1), with a
// fullscreen-triangle vertex stage. B3 replaces the body and the bindings.

#import "common/camera.wgsl"

// What a pixel's ray picks up through the atmosphere.
struct AtmoSegment {
    // Inscattered radiance per lux of sun illuminance [cd/m2 per lx], per channel.
    inscatter: vec3<f32>,
    // Transmittance along the ray to the scene behind, per channel.
    transmittance: vec3<f32>,
}

fn atmo_segment(pix: vec2<f32>) -> AtmoSegment {
    return AtmoSegment(vec3<f32>(0.0), vec3<f32>(1.0));
}

// Fullscreen triangle (vertices 0, 1, 2 cover the viewport).
@vertex
fn vs_atmosphere(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
