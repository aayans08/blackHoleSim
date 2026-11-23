// post/histogram.wgsl: the exposure meter "histogram" (Task B6), pass 9 of the M1 frame
// contract: after the atmospheres, before any sky light or point source is written, so it meters
// world light only.
//
// - cs_histogram: each pixel's world luminance, Rec. 709 Y of hdr / the pre-exposure it was
//   written with [cd/m²] (meter_pre: scene_pre() for plain world light, see "Star share"), into
//   256 log2 bins over [-10, 32] with workgroup-shared atomics, flushed to the global bins once
//   per workgroup (bruop 2019, "Automatic Exposure Using a Luminance Histogram",
//   https://bruop.github.io/exposure/).
// - cs_reduce: one workgroup reads and clears the bins and reduces them to the Meter
//   (lit_px, ev_avg, ev_hi) in the auto-exposure state; the next frame's resolve pass
//   (post/exposure.wgsl) turns it into its target.
//
// Each bin also sums its pixels' sub-bin step (0..15), so it stands for their mean log2 L
// rather than its centre (B6 decision; crates/uni-post/src/histogram.rs). In workgroup memory
// one atomic per bin holds both: count << 16 | step sum (a workgroup has at most 256 pixels per
// bin: count ≤ 256, step sum ≤ 3840 < 2^16).
//
// Star share (B6 fix rounds 1-2, review r0 Important 1; lead ruling B6 I1, writers task B10,
// log-space blend task B11): lens_mask.a at this pass is the pixel's star share s = 1 - w, where
// w = lod_exposure_mix is the weight its writer used in pre = pre_mix(star_pre(), scene_pre(), w)
// (common/exposure.wgsl), composited with the Max blend of the writers' lens_mask target. The
// meter divides hdr's Y by pre_mix(scene_pre(), star_pre(), s), the writer's pre read back.
// s = 0, the clear value and what plain world light stores, is the plan's rule (÷ scene_pre)
// exactly: pre_mix returns scene_pre() unchanged there. hdr's own alpha is not read. A body in
// the 1.5-4 px band read ÷ scene_pre would be 2^(s (EV_scene - EV_star)) too bright (with B10's
// linear blend the error grew as 2^(EV_scene - EV_star) and ran the loop away); with a
// Max-composited share no reading grows with the scene EV (crates/uni-post/src/histogram.rs
// module docs).
//
// CPU twins (crates/uni-post/src/histogram.rs): meter_pre, meter_luminance, meter_pixel (the
// per-pixel step of cs_histogram), meter_bin, Histogram::meter (meter_reduce here, same order
// of operations). The constants are checked against Rust by
// `post_shaders_validate`; MeterBins and ExposureState by `post_layouts_match_wgsl`; the
// bindings by `meter_reads_the_share_from_lens_mask`.
//
// Bindings: group 0 = the renderer's (scene_pre, star_pre); group 1: 0 meter_hdr, 1 meter_bins,
// 2 meter_state, 3 meter_lens_mask.

#import "common/camera.wgsl"
#import "common/color.wgsl"

// A pixel below this world luminance [cd/m²] is not lit (plan B6).
const METER_MIN_CD_M2: f32 = 1.0e-3;
// The bins span [METER_LOG2_MIN, METER_LOG2_MAX] in log2 cd/m² (plan B6).
const METER_LOG2_MIN: f32 = -10.0;
const METER_LOG2_MAX: f32 = 32.0;
const METER_BINS: u32 = 256u;
// (METER_LOG2_MAX - METER_LOG2_MIN) / METER_BINS = 42/256, exact in binary.
const METER_BIN_EV: f32 = 0.1640625;
const METER_BIN_STEPS: u32 = 16u;
// meter_bin clamps the bin coordinate this far inside the top edge (bins).
const BIN_COORD_MARGIN: f32 = 1.0e-3;
// lit_px is normalised to a 1920 × 1080 frame (plan B6).
const METER_REFERENCE_PX: f32 = 2073600.0;
// ev_avg averages log2 L over these lit-pixel percentiles (plan B6: 50th-95th).
const AVERAGE_BAND_LO: f32 = 0.50;
const AVERAGE_BAND_HI: f32 = 0.95;
// log2(S/K), S = 100, K = 12.5: Lagarde & de Rousiers 2014, "Moving Frostbite to PBR 3.0",
// §4.1.1, after ISO 2720:1974.
const LOG2_S_OVER_K: f32 = 3.0;
// The highlight rule: the HIGHLIGHT_PERCENTILE lit pixel lands at HIGHLIGHT_PRE (plan B6).
const HIGHLIGHT_PERCENTILE: f32 = 0.995;
const HIGHLIGHT_PRE: f32 = 4.0;
// L_max = 1.2·2^EV100 (exposure_scale in common/exposure.wgsl; Lagarde & de Rousiers 2014
// §4.1.1, after ISO 12232:2006).
const SATURATION_FACTOR: f32 = 1.2;

// Twin of uni_post::histogram::MeterBinsGpu (2048 bytes).
struct MeterBins {
    counts: array<atomic<u32>, 256>,
    steps: array<atomic<u32>, 256>,
}

// Twin of uni_post::exposure::ExposureStateGpu (32 bytes); also in post/exposure.wgsl.
struct ExposureState {
    // lit_px, ev_avg, ev_hi, 1 once the meter has read a frame (0 before).
    meter: vec4<f32>,
    // The adapted scene EV100, 1 once it holds a value; written by the resolve pass.
    adapted: vec4<f32>,
}

@group(1) @binding(0) var meter_hdr: texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> meter_bins: MeterBins;
@group(1) @binding(2) var<storage, read_write> meter_state: ExposureState;
// The frame's lens_mask (same size as hdr); only its alpha, the star share, is read.
@group(1) @binding(3) var meter_lens_mask: texture_2d<f32>;

// cs_histogram: count << 16 | step sum per bin.
var<workgroup> wg_packed: array<atomic<u32>, 256>;
// cs_reduce: the bins, copied out of the global buffer.
var<workgroup> wg_counts: array<u32, 256>;
var<workgroup> wg_steps: array<u32, 256>;

// NaN test on the bits: Metal compiles with fast math, where `x != x` may fold to false.
fn is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// The world luminance [cd/m²] of pre-exposed pixel `rgb` written at pre-exposure `pre`: Rec. 709
// Y / pre, channels clamped to [0, HDR_MAX] first; a pixel with a channel at HDR_MAX (clamped by
// its writer) counts as at least HDR_MAX / pre (plan B6): a lower bound when all three channels
// clipped, up to 3.8 EV high for one frame when only one did (it no longer clips the next).
fn meter_luminance(rgb: vec3<f32>, pre: f32) -> f32 {
    let c = clamp(rgb, vec3<f32>(0.0), vec3<f32>(HDR_MAX));
    let l = luminance(c) / pre;
    if (max(c.r, max(c.g, c.b)) >= HDR_MAX) {
        return max(l, HDR_MAX / pre);
    }
    return l;
}

// The pre-exposure a pixel was written with, from its star share s (lens_mask alpha, not NaN):
// the log-space pre_mix(scene_pre(), star_pre(), s) = scene_pre^(1 - s) star_pre^s, s clamped to
// [0, 1], i.e. the writers' pre_mix(star_pre(), scene_pre(), w) with w = 1 - s; s = 0 gives
// scene_pre() exactly.
fn meter_pre(s: f32) -> f32 {
    return pre_mix(scene_pre(), star_pre(), clamp(s, 0.0, 1.0));
}

// Bin (x) and sub-bin step (y) of lit luminance l ≥ METER_MIN_CD_M2; values beyond the range go
// to the end bins.
fn meter_bin(l: f32) -> vec2<u32> {
    let x = clamp(
        (log2(l) - METER_LOG2_MIN) / METER_BIN_EV,
        0.0,
        f32(METER_BINS) - BIN_COORD_MARGIN,
    );
    let bin = min(u32(x), METER_BINS - 1u);
    let step = min(u32((x - f32(bin)) * f32(METER_BIN_STEPS)), METER_BIN_STEPS - 1u);
    return vec2<u32>(bin, step);
}

@compute @workgroup_size(16, 16, 1)
fn cs_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    atomicStore(&wg_packed[li], 0u);
    workgroupBarrier();
    let dims = textureDimensions(meter_hdr);
    if (gid.x < dims.x && gid.y < dims.y) {
        let p = vec2<i32>(gid.xy);
        let s = textureLoad(meter_lens_mask, p, 0).a;
        // A NaN star share: the pixel's pre is unknown, so it is not metered.
        if (!is_nan(s)) {
            let l = meter_luminance(textureLoad(meter_hdr, p, 0).rgb, meter_pre(s));
            if (l >= METER_MIN_CD_M2 && !is_nan(l)) {
                let b = meter_bin(l);
                atomicAdd(&wg_packed[b.x], (1u << 16u) | b.y);
            }
        }
    }
    workgroupBarrier();
    let packed = atomicLoad(&wg_packed[li]);
    if (packed != 0u) {
        atomicAdd(&meter_bins.counts[li], packed >> 16u);
        atomicAdd(&meter_bins.steps[li], packed & 0xffffu);
    }
}

// The log2 luminance bin `bin` stands for: its pixels' mean position (step sum / count, + half a
// step for the truncation in meter_bin).
fn bin_log2_l(bin: u32) -> f32 {
    let steps = f32(wg_steps[bin]) / f32(wg_counts[bin]);
    return METER_LOG2_MIN + (f32(bin) + (steps + 0.5) / f32(METER_BIN_STEPS)) * METER_BIN_EV;
}

// The Meter of the bins in workgroup memory for a frame of total_px pixels: (lit_px, ev_avg,
// ev_hi, 1), or (0, 0, 0, 1) when nothing is lit. Twin of `reduce` in histogram.rs.
fn meter_reduce(total_px: u32) -> vec4<f32> {
    var lit = 0u;
    for (var bin = 0u; bin < METER_BINS; bin++) {
        lit += wg_counts[bin];
    }
    if (lit == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let n = f32(lit);
    let lo = AVERAGE_BAND_LO * n;
    let hi = AVERAGE_BAND_HI * n;
    let rank = HIGHLIGHT_PERCENTILE * n;
    var below = 0.0;
    var sum = 0.0;
    var p_hi = METER_LOG2_MIN;
    var found = false;
    for (var bin = 0u; bin < METER_BINS; bin++) {
        let count = wg_counts[bin];
        if (count == 0u) {
            continue;
        }
        let l = bin_log2_l(bin);
        let above = below + f32(count);
        sum += max(min(above, hi) - max(below, lo), 0.0) * l;
        if (!found && above >= rank) {
            p_hi = l;
            found = true;
        }
        below = above;
    }
    return vec4<f32>(
        n * (METER_REFERENCE_PX / f32(max(total_px, 1u))),
        sum / (hi - lo) + LOG2_S_OVER_K,
        p_hi - log2(SATURATION_FACTOR * HIGHLIGHT_PRE),
        1.0,
    );
}

@compute @workgroup_size(256, 1, 1)
fn cs_reduce(@builtin(local_invocation_index) li: u32) {
    // Read and clear in one step, so the next frame starts from empty bins.
    wg_counts[li] = atomicExchange(&meter_bins.counts[li], 0u);
    wg_steps[li] = atomicExchange(&meter_bins.steps[li], 0u);
    workgroupBarrier();
    if (li == 0u) {
        let dims = textureDimensions(meter_hdr);
        meter_state.meter = meter_reduce(dims.x * dims.y);
    }
}
