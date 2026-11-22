// common/exposure.wgsl: the exposure contract with NO bindings. common/camera.wgsl imports it
// and binds `exposure` at group 0, binding 1.
//
// A pass that writes the exposure buffer as storage (M1's auto-exposure resolve) imports only
// this file: binding the same buffer as a uniform in the same dispatch would be a usage-scope
// conflict (review m14), so it must not touch group 0.
//
// Pre-exposed HDR (the frame's exposure contract). The hdr target holds scene-linear Rec. 709
// light ALREADY MULTIPLIED by its exposure, so the tonemap applies AgX directly and values stay
// inside f16 range at any exposure (the Sun's disk, ~1.6e9 cd/m², is ~4.1e4 at scene EV100 15):
// - world light (bodies, the Sun's disk, rings, atmospheres): luminance [cd/m²] × scene_pre();
// - sky light and point sources (stars, the band, unresolved bodies): luminance × star_pre().
//   With star EV ≠ scene EV this is the "Eye-adapted stars" dual exposure of design §7 item 4,
//   and the HUD labels it.
// Every feature clamps what it writes to HDR_MAX, below f16's largest finite value.

// Twin of `uni_gfx::ExposureUniform` (crates/uni-gfx/src/frame.rs): 32 bytes, two vec4s
// (`exposure_uniform_layout_matches_wgsl`).
struct ExposureUniform {
    // x: scene pre-exposure = exposure_scale(scene EV100); y: star pre-exposure =
    // exposure_scale(star EV100); zw: 0.
    pre: vec4<f32>,
    // x: scene EV100, y: star EV100 (for reference; shaders use `pre`); zw: 0.
    ev100: vec4<f32>,
}

// Largest value any feature writes into the Rgba16Float hdr target, below the largest finite
// f16 (IEEE 754 binary16: 65504), so additive blending of a few such values stays finite in
// practice and no pass writes infinity. Twin of `uni_gfx::HDR_MAX`.
const HDR_MAX: f32 = 60000.0;

// Scale from scene luminance [cd/m²] to normalised exposure for a camera at `ev100`.
// Saturation-based sensitivity: L_max = (78 / (S·q)) · 2^EV100 with S = 100, q = 0.65,
// so L_max = 1.2 · 2^EV100. Source: Lagarde & de Rousiers 2014, "Moving Frostbite to
// Physically Based Rendering 3.0", §4.1.1 (after ISO 12232:2006). Twin of
// `uni_gfx::exposure_scale`.
fn exposure_scale(ev100: f32) -> f32 {
    return 1.0 / (1.2 * exp2(ev100));
}
