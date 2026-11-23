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

// ---- AgX -------------------------------------------------------------------------------
// Source: Troy Sobotka, AgX (github.com/sobotka/AgX), "base" transform with the default
// contrast look, in the real-time form of Benjamin Wrensch 2023, "Minimal AgX Implementation"
// (iolite-engine.com/blog_posts/minimal_agx_implementation). Matrices, EV range and the
// polynomial are verbatim from there.

// Inset matrix (columns): mixes a few percent of each primary into the others, so very bright
// saturated colours move toward white instead of clipping per channel.
const AGX_INSET: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
    vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
    vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
);

// Outset matrix (columns): the inverse of AGX_INSET, applied after the sigmoid.
const AGX_OUTSET: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
    vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
    vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
);

// Log2 encoding range [EV relative to 1.0]: 10 stops below and 6.5 stops above middle grey
// (log2 0.18 = −2.47393).
const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

// Floor before log2 so zero or negative input stays finite; far below 2^AGX_MIN_EV, so it
// lands at the bottom of the range like any other near-black value.
const AGX_LOG_FLOOR: f32 = 1e-10;

// 6th-order polynomial fit of AgX's default-contrast sigmoid on [0, 1] (Wrensch 2023,
// `agxDefaultContrastApprox`; mean squared error 3.67e-6). Output is sRGB-encoded.
fn agx_default_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2
        + 0.1191 * x - 0.00232;
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

// AgX: exposed scene-linear Rec. 709 RGB (middle grey 0.18) → display-linear RGB in [0, 1].
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
    // 3. Default-contrast sigmoid → sRGB-encoded signal.
    let encoded = agx_default_contrast(x);
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
