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

// The pre-exposure a fraction t of the way from pre-exposure a to b, blended in log space (in
// stops): exp2(mix(log2 a, log2 b, t)) = a^(1 - t) b^t (task B11). The LOD-band writers (bodies,
// rings, star surfaces, atmospheres in the 1.5-4 px band) draw at
// pre_mix(star_pre(), scene_pre(), w), and the meter reads a pixel back at
// pre_mix(scene_pre(), star_pre(), s) with s = 1 - w, its star share in lens_mask.a. An exposure
// factor is a multiplier, so equal steps of t move it by equal stops; a linear mix would stay
// near the larger factor until t is close to 1 and drop a 19-stop star/scene gap at the end.
// t <= 0 returns a and t >= 1 returns b unchanged (exp2(log2(a)) need not round back to a, and
// s = 0 must read scene_pre() exactly). A non-positive a or b (log2 undefined; the renderer's
// factors are always positive) falls back to the linear mix. Twin of `uni_gfx::frame::pre_mix`.
fn pre_mix(a: f32, b: f32, t: f32) -> f32 {
    if (t <= 0.0) {
        return a;
    }
    if (t >= 1.0) {
        return b;
    }
    if (!(a > 0.0 && b > 0.0)) {
        return mix(a, b, t);
    }
    return exp2(mix(log2(a), log2(b), t));
}
