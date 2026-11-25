// post/tonemap.wgsl: the M0 display transform, as one fullscreen-triangle pass.
//
//   hdr (pre-exposed scene-linear Rec. 709, internal resolution) → upscale to the output
//   → agx → display-linear → sRGB code space + triangular dither of ±1/255
//   → display-linear → output
//
// The hdr target is pre-exposed (common/camera.wgsl: features write light × exposure), so this
// pass applies no exposure of its own. When the output is the internal resolution (captures,
// --bench) every output pixel reads its own hdr texel with textureLoad, exactly; when it is
// larger (the window), the hdr is sampled bilinearly at the output pixel's centre. Upscaling
// happens before AgX, in the linear HDR domain (TAAU takes this slot in M7).
//
// The output view is sRGB (Bgra8UnormSrgb for the window, Rgba8UnormSrgb for captures), so
// the fragment writes display-LINEAR values and the hardware applies the sRGB encoding.
//
// Bindings: group 0 = `camera`, `exposure` (common/camera.wgsl); group 1 binding 0 = `hdr_tex`,
// binding 1 = `hdr_sampler` (linear, clamp to edge). A shader that imports this file only for
// `agx` must keep its own bindings clear of groups 0 and 1.
//
// `agx` has a CPU twin, `uni_post::agx` (crates/uni-post/src/agx.rs). Keep the constants and
// the order of operations identical; `universe --selftest` compares the two within 1e-3 and
// `wgsl_agx_constants_match_cpu_twin` compares the constants.

#import "common/camera.wgsl"
#import "common/color.wgsl"

// ---- AgX-style transform --------------------------------------------------------------
// The idea is Troy Sobotka's AgX (github.com/sobotka/AgX): inset, log2 encoding around middle
// grey, sigmoid, outset. Every constant is derived in crates/uni-post/src/agx.rs (final fix
// wave F3; the M0 constants were copied from a post with no licence): the inset moves each
// Rec. 709 primary 20 % of the way to D65 in CIE 1931 xy with white kept (BT.709-6 primaries,
// SMPTE RP 177 construction), the outset is its exact inverse, the log range is 10 stops below
// and 6.5 above middle grey, and the sigmoid is agx.rs's own toe/shoulder curve.

// Inset matrix (columns): (1 − t)·I + t·𝟙wᵀ, t = 0.2, w_j = S_j/ΣS (agx.rs AGX_INSET).
const AGX_INSET: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(0.8423989290495315, 0.04239892904953144, 0.04239892904953144),
    vec3<f32>(0.07843016510486392, 0.8784301651048639, 0.07843016510486392),
    vec3<f32>(0.07917090584560466, 0.07917090584560466, 0.8791709058456048),
);

// Outset matrix (columns): the exact inverse of AGX_INSET, applied after the sigmoid.
const AGX_OUTSET: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(1.197001338688086, -0.0529986613119143, -0.0529986613119143),
    vec3<f32>(-0.09803770638107989, 1.15196229361892, -0.09803770638107989),
    vec3<f32>(-0.09896363230700582, -0.09896363230700582, 1.151036367692994),
);

// Log2 encoding range [EV relative to 1.0]: log2 0.18 − 10 and log2 0.18 + 6.5.
const AGX_MIN_EV: f32 = -12.473931188332411;
const AGX_MAX_EV: f32 = 4.026068811667588;

// Floor before log2 so zero or negative input stays finite; far below 2^AGX_MIN_EV, so it
// lands at the bottom of the range like any other near-black value.
const AGX_LOG_FLOOR: f32 = 1e-10;

// The sigmoid's pivot (middle grey's position 10/16.5 → code value 0.5), its slope there and the
// toe's and shoulder's powers (agx.rs SIGMOID_*).
const SIGMOID_PIVOT_IN: f32 = 0.6060606060606061;
const SIGMOID_PIVOT_OUT: f32 = 0.5;
const SIGMOID_SLOPE: f32 = 2.02;
const SIGMOID_TOE_POWER: f32 = 2.9;
const SIGMOID_SHOULDER_POWER: f32 = 2.8;

// k·v / (1 + (k^p − 1)·v^p)^(1/p): 0 at 0, 1 at 1, slope k at 0. Twin of agx.rs rational_curve.
fn rational_curve(v: f32, k: f32, p: f32) -> f32 {
    return k * v / pow(1.0 + (pow(k, p) - 1.0) * pow(v, p), 1.0 / p);
}

// The sigmoid on [0, 1], sRGB-encoded output: toe below the pivot, shoulder above. Twin of
// agx.rs agx_sigmoid.
fn agx_sigmoid(x: f32) -> f32 {
    let xp = SIGMOID_PIVOT_IN;
    let yp = SIGMOID_PIVOT_OUT;
    if (x >= xp) {
        let k = SIGMOID_SLOPE * (1.0 - xp) / (1.0 - yp);
        return yp + (1.0 - yp) * rational_curve((x - xp) / (1.0 - xp), k, SIGMOID_SHOULDER_POWER);
    }
    let k = SIGMOID_SLOPE * xp / yp;
    return yp - yp * rational_curve((xp - x) / xp, k, SIGMOID_TOE_POWER);
}

// sRGB code value → display-linear, per channel (inverse of `linear_to_srgb` in
// common/color.wgsl). Source: IEC 61966-2-1:1999, §5.2: c / 12.92 for c ≤ 0.04045, else
// ((c + 0.055) / 1.055)^2.4. Negative inputs are clamped to 0 so `pow` never sees them.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    let lo = x / 12.92;
    let hi = pow((x + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, x <= vec3<f32>(0.04045));
}

// AgX-style: exposed scene-linear Rec. 709 RGB (middle grey 0.18) → display-linear RGB in [0, 1].
// The sigmoid's output is an sRGB-encoded signal; it is decoded with the exact inverse of the
// sRGB view's encoder, so the code values on screen are exactly AgX's.
fn agx(c: vec3<f32>) -> vec3<f32> {
    // 1. Inset.
    let inset = AGX_INSET * c;
    // 2. Log2 encoding, normalised so [AGX_MIN_EV, AGX_MAX_EV] maps to [0, 1].
    let ev = clamp(
        log2(max(inset, vec3<f32>(AGX_LOG_FLOOR))),
        vec3<f32>(AGX_MIN_EV),
        vec3<f32>(AGX_MAX_EV),
    );
    let x = (ev - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);
    // 3. Sigmoid → sRGB-encoded signal.
    let encoded = vec3<f32>(agx_sigmoid(x.x), agx_sigmoid(x.y), agx_sigmoid(x.z));
    // 4. Outset. Bright saturated colours can overshoot [0, 1] here, so clamp to the display.
    let outset = clamp(AGX_OUTSET * encoded, vec3<f32>(0.0), vec3<f32>(1.0));
    // 5. Decode to display-linear.
    return srgb_to_linear(outset);
}

// ---- Dither ----------------------------------------------------------------------------

// Peak dither amplitude in sRGB code space: ±1/255, one 8-bit step, so the triangular PDF is
// the sum of two ±½-step uniforms and the quantisation error is independent of the signal
// (M1 review deferral T5: faint band and atmosphere gradients; M0 had ±0.5/255).
const DITHER_AMPLITUDE: f32 = 1.0 / 255.0;

// Interleaved gradient noise in [0, 1) at integer pixel `pix`.
// Source: Jimenez 2014, "Next Generation Post Processing in Call of Duty: Advanced Warfare",
// SIGGRAPH Advances in Real-Time Rendering (constants 52.9829189, 0.06711056, 0.00583715).
fn interleaved_gradient_noise(pix: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pix, vec2<f32>(0.06711056, 0.00583715))));
}

// Uniform [0, 1) → triangular PDF on [−1, 1] by the inverse CDF, keeping the noise's spatial
// structure. Source: Mikkel Gjøl (Playdead) 2016, "Banding in Games: A Noisy Rant"
// (`remap_pdf_tri_unity`).
fn uniform_to_triangular(u: f32) -> f32 {
    let v = 2.0 * u - 1.0;
    return sign(v) * (1.0 - sqrt(1.0 - abs(v)));
}

// ---- Pass ------------------------------------------------------------------------------

@group(1) @binding(0) var hdr_tex: texture_2d<f32>;
@group(1) @binding(1) var hdr_sampler: sampler;

// Fullscreen triangle: vertices (−1, 1), (3, 1), (−1, −3) in clip space cover the viewport.
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(2.0 * x - 1.0, 1.0 - 2.0 * y, 0.0, 1.0);
}

// The pre-exposed hdr colour for the output pixel whose centre is `position` (output pixels).
// Same size: the pixel's own texel, read exactly. Otherwise: bilinear at the same relative
// position, (position / output size) in uv.
fn hdr_at(position: vec2<f32>) -> vec3<f32> {
    let dims = textureDimensions(hdr_tex);
    if (all(vec2<f32>(dims) == camera.output.xy)) {
        // Clamped so a pixel can never read out of bounds.
        let last = vec2<i32>(dims) - vec2<i32>(1);
        return textureLoad(hdr_tex, clamp(vec2<i32>(floor(position)), vec2<i32>(0), last), 0).rgb;
    }
    return textureSampleLevel(hdr_tex, hdr_sampler, position * camera.output.zw, 0.0).rgb;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Integer output-pixel index; @builtin(position) is at the pixel centre (x + 0.5, y + 0.5).
    let pix = floor(position.xy);
    let display = agx(hdr_at(position.xy));

    // Dither in sRGB code space (the space the 8-bit target quantises in), then back to
    // display-linear for the sRGB view. The same noise on all channels keeps it achromatic.
    let noise = uniform_to_triangular(interleaved_gradient_noise(pix)) * DITHER_AMPLITUDE;
    let code = clamp(linear_to_srgb(display) + vec3<f32>(noise), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(srgb_to_linear(code), 1.0);
}
