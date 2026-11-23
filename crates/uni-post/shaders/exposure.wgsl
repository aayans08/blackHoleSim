// post/exposure.wgsl: the exposure resolve pass "exposure" (Task B6), pass 1 of the M1 frame
// contract. One thread:
// 1. adapts the scene EV toward the meter's target from the previous frame (Auto), or takes the
//    manual EV (Manual), and keeps it in the auto-exposure state;
// 2. writes the frame's ExposureUniform into the renderer's exposure buffer: scene EV adapted or
//    manual; star EV = the eye-adapted EV while eye-adapted stars are in effect, else the scene
//    EV.
//
// It imports only common/exposure.wgsl (no bindings) and never group 0: binding the exposure
// buffer as a uniform and as storage in one dispatch is a usage-scope conflict (review m14).
// Its own bindings are at group 1: 0 exposure_out (the renderer's exposure buffer),
// 1 resolve_params, 2 resolve_state.
//
// CPU twins (crates/uni-post/src/exposure.rs and histogram.rs): clamp_ev100, target_ev, adapt,
// next_adapted_ev (the state step) and resolve_exposure (the uniform), same order of
// operations. The CPU sanitises the params (ResolveParamsGpu::new), so nothing here sees a NaN.

#import "common/exposure.wgsl"

// Dark-adapted target with no metered world light [EV100] (uni_post::NIGHT_EV100).
const NIGHT_EV100: f32 = -4.0;
// Every EV the resolve writes lies in [EV100_MIN, EV100_MAX] (uni_post::EV100_MIN/MAX).
const EV100_MIN: f32 = -30.0;
const EV100_MAX: f32 = 40.0;
// Lit-pixel thresholds of the target rule (plan B6; uni_post::histogram).
const METER_MIN_LIT_PX: f32 = 50.0;
const METER_FULL_LIT_PX: f32 = 400.0;

// Twin of uni_post::exposure::ResolveParamsGpu (48 bytes).
struct ResolveParams {
    // x: 1 = Manual, 0 = Auto; y: 1 = eye-adapted stars in effect; z: 0; w: dt [s] (≥ 0).
    mode: vec4<f32>,
    // x: manual scene EV100, y: eye-adapted star EV100 (both clamped), z: compensation [EV].
    ev: vec4<f32>,
    // x: tau while the scene brightens, y: tau while it darkens [s] (0 = settle).
    tau: vec4<f32>,
}

// Twin of uni_post::exposure::ExposureStateGpu (32 bytes); also in post/histogram.wgsl.
struct ExposureState {
    // Written by the meter: lit_px (1920 × 1080), ev_avg, ev_hi, 1 once it has read a frame.
    meter: vec4<f32>,
    // The adapted scene EV100, 1 once it holds a value (0 before).
    adapted: vec4<f32>,
}

@group(1) @binding(0) var<storage, read_write> exposure_out: ExposureUniform;
@group(1) @binding(1) var<uniform> resolve_params: ResolveParams;
@group(1) @binding(2) var<storage, read_write> resolve_state: ExposureState;

// NaN test on the bits: Metal compiles with fast math, where `x != x` may fold to false.
fn is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// ev clamped to [EV100_MIN, EV100_MAX]; NaN becomes NIGHT_EV100.
fn clamp_ev(ev: f32) -> f32 {
    if (is_nan(ev)) {
        return NIGHT_EV100;
    }
    return clamp(ev, EV100_MIN, EV100_MAX);
}

// The target rule (plan B6): NIGHT below METER_MIN_LIT_PX lit pixels; max(ev_avg, ev_hi) +
// compensation from METER_FULL_LIT_PX; smoothstep in ln(lit_px) between. m = the meter's
// (lit_px, ev_avg, ev_hi, -).
fn target_ev(m: vec4<f32>, compensation: f32) -> f32 {
    if (!(m.x >= METER_MIN_LIT_PX)) {
        return NIGHT_EV100;
    }
    let metered = max(m.y, m.z) + compensation;
    let t = clamp(
        (log(m.x) - log(METER_MIN_LIT_PX)) / (log(METER_FULL_LIT_PX) - log(METER_MIN_LIT_PX)),
        0.0,
        1.0,
    );
    let w = t * t * (3.0 - 2.0 * t);
    return clamp_ev(metered * w + NIGHT_EV100 * (1.0 - w));
}

// One adaptation step, ev += (goal - ev)(1 - e^(-dt/tau)), tau.x while the scene brightens (goal
// above), tau.y while it darkens; tau 0 jumps to the goal (Pattanaik et al. 2000; Krawczyk et
// al. 2005; see DEFAULT_ADAPT_TAU_S in exposure.rs).
fn adapt(prev: f32, goal: f32, dt: f32, tau: vec2<f32>) -> f32 {
    let t = select(tau.y, tau.x, goal > prev);
    if (!(t > 0.0)) {
        return goal;
    }
    return prev + (goal - prev) * (1.0 - exp(-dt / t));
}

@compute @workgroup_size(1, 1, 1)
fn cs_resolve() {
    let p = resolve_params;
    let manual = p.mode.x > 0.5;
    var adapted = resolve_state.adapted;
    if (manual) {
        // Keep the manual EV, so switching to Auto adapts from it.
        adapted = vec4<f32>(p.ev.x, 1.0, 0.0, 0.0);
    } else if (resolve_state.meter.w > 0.5) {
        // The meter read the previous frame; before the first adapted EV, start from the manual
        // EV that frame was exposed at.
        let ev_from = select(p.ev.x, adapted.x, adapted.y > 0.5);
        let goal = target_ev(resolve_state.meter, p.ev.z);
        adapted = vec4<f32>(clamp_ev(adapt(ev_from, goal, p.mode.w, p.tau.xy)), 1.0, 0.0, 0.0);
    }
    // (Auto without a reading holds the adapted EV.)
    resolve_state.adapted = adapted;

    let scene = select(p.ev.x, clamp_ev(adapted.x), !manual && adapted.y > 0.5);
    let star = select(scene, p.ev.y, p.mode.y > 0.5);
    exposure_out.pre = vec4<f32>(exposure_scale(scene), exposure_scale(star), 0.0, 0.0);
    exposure_out.ev100 = vec4<f32>(scene, star, 0.0, 0.0);
}
