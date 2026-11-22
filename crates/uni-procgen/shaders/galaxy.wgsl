// Milky Way galaxy model v0 (M1 Task A5): the WGSL twin of `uni_procgen::galaxy`.
// ShaderLib path "procgen/galaxy.wgsl". B5 imports this file and binds `GalaxyGpu` at its own
// group. Galactocentric frame, distances in pc (see galaxy.rs).
//
// A0b stub: valid empty functions that return 0 until A5 implements them.

// The model's parameters; vec4-sized members only (CLAUDE.md §3 uniform layout rule).
struct GalaxyGpu {
    // A0b placeholder member; A5 replaces it with the model's vec4 parameters.
    placeholder: vec4<f32>,
}

// Star density [stars/pc³] at galactocentric position `p_gc_pc` [pc].
fn gal_density(p_gc_pc: vec3<f32>) -> f32 {
    return 0.0;
}

// V-band luminosity density [L☉,V/pc³].
fn gal_luminosity(p_gc_pc: vec3<f32>) -> f32 {
    return 0.0;
}

// Dust extinction A_V per unit length [mag/pc].
fn gal_extinction(p_gc_pc: vec3<f32>) -> f32 {
    return 0.0;
}

// Band radiance [cd/m², linear Rec.709] seen from `cam_gc_pc` along `dir`, from stars fainter
// than `v_lim`.
fn gal_band_radiance(cam_gc_pc: vec3<f32>, dir: vec3<f32>, v_lim: f32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
