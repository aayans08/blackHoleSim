// post/bloom.wgsl: bloom "bloom" (Task B6), pass 15 of the M1 frame contract.
//
// Jimenez 2014 (SIGGRAPH Advances in Real-Time Rendering, "Next Generation Post Processing in
// Call of Duty: Advanced Warfare"):
// - cs_down_first: level 1 from hdr, the 13-tap downsample with every source texel clamped to
//   BLOOM_INPUT_MAX first and the partial Karis average (per block of four taps); bilinear taps
//   wherever that clamp changes nothing, texel by texel where it does (down_first);
// - cs_down: level k from level k − 1, the 13-tap downsample of bilinear taps;
// - cs_up: u_k = d_k + tent(u_{k+1}), a 3 × 3 tent of bilinear taps one lower-level texel apart
//   (u_L = d_L), evaluated as 2 × 2 bilinear taps with the same weights (tent);
// - vs_composite / fs_composite: bloom = tent(u_1) / L at full resolution; the pipeline's blend
//   (constant, 1 - constant) with the constant = strength makes hdr = mix(hdr, bloom, strength).
// No threshold. Every step is normalised, so the chain conserves energy away from the edges.
//
// Task B9p (performance, no change to the result beyond rounding): the first downsample read each
// window's 36 texels one by one (4.4 ms at 1920 x 1080; now 0.26 ms), and the tents took 9 taps
// (u_1 0.20 -> 0.08 ms, composite 0.76 -> 0.38 ms).
//
// CPU twin: uni_post::bloom::bloom_reference (crates/uni-post/src/bloom.rs), the same texels and
// weights (the GPU groups some sums differently: bilinear taps, the 2 x 2-tap tent);
// `universe --selftest` compares the two on an impulse (1 %).
//
// Bindings: group 0 = the renderer's (the composite reads camera.viewport, the internal size =
// hdr's); group 1: 0 bloom_src (the level read), 1 bloom_dst (the level written, storage),
// 2 bloom_sampler (linear, clamp to edge), 3 bloom_same (cs_up: d_k), 4 bloom_params
// (composite).

#import "common/camera.wgsl"
#import "common/color.wgsl"

// The first downsample clamps its pre-exposed input to this: 2 stops over AgX white (plan B6;
// uni_post::BLOOM_INPUT_MAX).
const BLOOM_INPUT_MAX: f32 = 64.0;
// Floor of the largest channel in bloom_input's division (a normal f32, kept by fast math).
const MIN_CHANNEL: f32 = 1e-30;
// The 13-tap pattern's blocks of four taps: the centre block 1/2, each corner block 1/8
// (Jimenez 2014).
const CENTRE_WEIGHT: f32 = 0.5;
const CORNER_WEIGHT: f32 = 0.125;

// Twin of uni_post::bloom::BloomParamsGpu (16 bytes).
struct BloomParams {
    // x: 1 / the number of levels; yzw: 0.
    scale: vec4<f32>,
}

@group(1) @binding(0) var bloom_src: texture_2d<f32>;
@group(1) @binding(1) var bloom_dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var bloom_sampler: sampler;
@group(1) @binding(3) var bloom_same: texture_2d<f32>;
@group(1) @binding(4) var<uniform> bloom_params: BloomParams;

// The input clamp, per texel: c scaled so its largest channel is at most BLOOM_INPUT_MAX
// (negatives to 0); scaling keeps a bright star's colour in its glow.
fn bloom_input(c: vec3<f32>) -> vec3<f32> {
    let v = max(c, vec3<f32>(0.0));
    return v * min(BLOOM_INPUT_MAX / max(max(v.r, max(v.g, v.b)), MIN_CHANNEL), 1.0);
}

// The 13-tap result from its blocks (each the mean of four taps): weights 1/2 and 1/8, divided
// by 1 + the block's luminance when `karis` (Jimenez 2014, partial Karis average), renormalised.
fn bloom_combine(
    centre: vec3<f32>,
    tl: vec3<f32>,
    tr: vec3<f32>,
    bl: vec3<f32>,
    br: vec3<f32>,
    karis: bool,
) -> vec3<f32> {
    var wc = CENTRE_WEIGHT;
    var w = vec4<f32>(CORNER_WEIGHT);
    if (karis) {
        wc = wc / (1.0 + luminance(centre));
        w = w / (vec4<f32>(1.0)
            + vec4<f32>(luminance(tl), luminance(tr), luminance(bl), luminance(br)));
    }
    let sum = wc * centre + w.x * tl + w.y * tr + w.z * bl + w.w * br;
    return sum / (wc + w.x + w.y + w.z + w.w);
}

// One 2 x 2 tap of level 1's 13-tap pattern, exactly: the mean of the four source texels around
// the corner `p` (texel units, texel t spans [t, t + 1]), each clamped to the edges and to
// BLOOM_INPUT_MAX first.
fn first_tap_exact(p: vec2<f32>, last: vec2<i32>) -> vec3<f32> {
    let i = vec2<i32>(p);
    let a = bloom_input(textureLoad(bloom_src, clamp(i - vec2<i32>(1, 1), vec2<i32>(0), last), 0).rgb);
    let b = bloom_input(textureLoad(bloom_src, clamp(i - vec2<i32>(0, 1), vec2<i32>(0), last), 0).rgb);
    let c = bloom_input(textureLoad(bloom_src, clamp(i - vec2<i32>(1, 0), vec2<i32>(0), last), 0).rgb);
    let d = bloom_input(textureLoad(bloom_src, clamp(i, vec2<i32>(0), last), 0).rgb);
    return ((a + b) + (c + d)) * 0.25;
}

// Level-1 texel `dst` from hdr: the 6 x 6 source texels 2·dst - 2 ..= 2·dst + 3 as the 13 taps of
// cs_down: the five blocks (the centre, taps j k l m at ±1; the corners, taps a..i at {-2, 0} or
// {0, 2}) are each the mean of four taps = the mean of their 4 x 4 texels, combined with the
// Karis average. Task B9p: the 13 taps are bilinear (4.4 ms when every thread read its 36 texels
// one by one into a private array; 0.26 ms at 1920 x 1080 now), which equals clamping each texel
// first whenever no tap's mean has a channel above BLOOM_INPUT_MAX / 4 (then no texel is above
// BLOOM_INPUT_MAX) or below 0 (pre-exposed light is not negative). Otherwise, decided once after
// all 13 taps are issued (a test per tap serialised the fetches: +0.3 to 0.6 ms), every tap is
// redone texel by texel with the clamp (first_tap_exact).
fn down_first(dst: vec2<u32>) -> vec3<f32> {
    let last = vec2<i32>(textureDimensions(bloom_src)) - vec2<i32>(1);
    let texel = 1.0 / vec2<f32>(textureDimensions(bloom_src));
    let p = 2.0 * vec2<f32>(dst) + 1.0;
    var a = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(-2.0, -2.0)) * texel, 0.0).rgb;
    var b = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(0.0, -2.0)) * texel, 0.0).rgb;
    var c = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(2.0, -2.0)) * texel, 0.0).rgb;
    var d = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(-2.0, 0.0)) * texel, 0.0).rgb;
    var e = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(0.0, 0.0)) * texel, 0.0).rgb;
    var f = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(2.0, 0.0)) * texel, 0.0).rgb;
    var g = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(-2.0, 2.0)) * texel, 0.0).rgb;
    var h = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(0.0, 2.0)) * texel, 0.0).rgb;
    var i = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(2.0, 2.0)) * texel, 0.0).rgb;
    var j = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(-1.0, -1.0)) * texel, 0.0).rgb;
    var k = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(1.0, -1.0)) * texel, 0.0).rgb;
    var l = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(-1.0, 1.0)) * texel, 0.0).rgb;
    var m = textureSampleLevel(bloom_src, bloom_sampler, (p + vec2<f32>(1.0, 1.0)) * texel, 0.0).rgb;
    var hi = 0.0;
    var lo = 0.0;
    hi = max(hi, max(a.r, max(a.g, a.b)));
    hi = max(hi, max(b.r, max(b.g, b.b)));
    hi = max(hi, max(c.r, max(c.g, c.b)));
    hi = max(hi, max(d.r, max(d.g, d.b)));
    hi = max(hi, max(e.r, max(e.g, e.b)));
    hi = max(hi, max(f.r, max(f.g, f.b)));
    hi = max(hi, max(g.r, max(g.g, g.b)));
    hi = max(hi, max(h.r, max(h.g, h.b)));
    hi = max(hi, max(i.r, max(i.g, i.b)));
    hi = max(hi, max(j.r, max(j.g, j.b)));
    hi = max(hi, max(k.r, max(k.g, k.b)));
    hi = max(hi, max(l.r, max(l.g, l.b)));
    hi = max(hi, max(m.r, max(m.g, m.b)));
    lo = min(lo, min(a.r, min(a.g, a.b)));
    lo = min(lo, min(b.r, min(b.g, b.b)));
    lo = min(lo, min(c.r, min(c.g, c.b)));
    lo = min(lo, min(d.r, min(d.g, d.b)));
    lo = min(lo, min(e.r, min(e.g, e.b)));
    lo = min(lo, min(f.r, min(f.g, f.b)));
    lo = min(lo, min(g.r, min(g.g, g.b)));
    lo = min(lo, min(h.r, min(h.g, h.b)));
    lo = min(lo, min(i.r, min(i.g, i.b)));
    lo = min(lo, min(j.r, min(j.g, j.b)));
    lo = min(lo, min(k.r, min(k.g, k.b)));
    lo = min(lo, min(l.r, min(l.g, l.b)));
    lo = min(lo, min(m.r, min(m.g, m.b)));
    if (hi > 0.25 * BLOOM_INPUT_MAX || lo < 0.0) {
        a = first_tap_exact(p + vec2<f32>(-2.0, -2.0), last);
        b = first_tap_exact(p + vec2<f32>(0.0, -2.0), last);
        c = first_tap_exact(p + vec2<f32>(2.0, -2.0), last);
        d = first_tap_exact(p + vec2<f32>(-2.0, 0.0), last);
        e = first_tap_exact(p + vec2<f32>(0.0, 0.0), last);
        f = first_tap_exact(p + vec2<f32>(2.0, 0.0), last);
        g = first_tap_exact(p + vec2<f32>(-2.0, 2.0), last);
        h = first_tap_exact(p + vec2<f32>(0.0, 2.0), last);
        i = first_tap_exact(p + vec2<f32>(2.0, 2.0), last);
        j = first_tap_exact(p + vec2<f32>(-1.0, -1.0), last);
        k = first_tap_exact(p + vec2<f32>(1.0, -1.0), last);
        l = first_tap_exact(p + vec2<f32>(-1.0, 1.0), last);
        m = first_tap_exact(p + vec2<f32>(1.0, 1.0), last);
    }
    let centre = (j + k + l + m) * 0.25;
    let tl = (a + b + d + e) * 0.25;
    let tr = (b + c + e + f) * 0.25;
    let bl = (d + e + g + h) * 0.25;
    let br = (e + f + h + i) * 0.25;
    return bloom_combine(centre, tl, tr, bl, br, true);
}

@compute @workgroup_size(8, 8, 1)
fn cs_down_first(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= textureDimensions(bloom_dst))) {
        return;
    }
    textureStore(bloom_dst, gid.xy, vec4<f32>(down_first(gid.xy), 1.0));
}

// Bilinear tap of bloom_src at source position `p` (texel units, texel t spans [t, t + 1]).
fn tap_at(p: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(bloom_src, bloom_sampler, p * texel, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn cs_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= textureDimensions(bloom_dst))) {
        return;
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(bloom_src));
    // The corner shared by source texels 2·dst and 2·dst + 1: the destination texel's centre.
    let p = 2.0 * vec2<f32>(gid.xy) + 1.0;
    let a = tap_at(p + vec2<f32>(-2.0, -2.0), texel);
    let b = tap_at(p + vec2<f32>(0.0, -2.0), texel);
    let c = tap_at(p + vec2<f32>(2.0, -2.0), texel);
    let d = tap_at(p + vec2<f32>(-2.0, 0.0), texel);
    let e = tap_at(p, texel);
    let f = tap_at(p + vec2<f32>(2.0, 0.0), texel);
    let g = tap_at(p + vec2<f32>(-2.0, 2.0), texel);
    let h = tap_at(p + vec2<f32>(0.0, 2.0), texel);
    let i = tap_at(p + vec2<f32>(2.0, 2.0), texel);
    let j = tap_at(p + vec2<f32>(-1.0, -1.0), texel);
    let k = tap_at(p + vec2<f32>(1.0, -1.0), texel);
    let l = tap_at(p + vec2<f32>(-1.0, 1.0), texel);
    let m = tap_at(p + vec2<f32>(1.0, 1.0), texel);
    let centre = (j + k + l + m) * 0.25;
    let tl = (a + b + d + e) * 0.25;
    let tr = (b + c + e + f) * 0.25;
    let bl = (d + e + g + h) * 0.25;
    let br = (e + f + h + i) * 0.25;
    textureStore(bloom_dst, gid.xy, vec4<f32>(bloom_combine(centre, tl, tr, bl, br, false), 1.0));
}

// The 3 × 3 tent ([1 2 1] ⊗ [1 2 1] / 16) of bilinear taps of bloom_src one texel apart, around
// `uv`, as 2 × 2 bilinear taps (task B9p; 9 taps before). Per axis, with p = uv·size − ½ = i + f,
// the three taps weigh texels i − 1 ..= i + 2 by (1 − f)/4, f/4 + (1 − f)/2, f/2 + (1 − f)/4, f/4;
// one bilinear tap between texels i − 1 and i carries the first two ((3 − 2f)/4 at i − 1 + (2 − f)/
// (3 − 2f)), one between i + 1 and i + 2 the last two ((1 + 2f)/4 at i + 1 + f/(1 + 2f)). The same
// texels and weights as the nine taps (the CPU twin's form), up to the sampler's sub-texel
// precision; clamp to edge picks the same edge texels.
fn tent(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(bloom_src));
    let p = uv * size - 0.5;
    let i = floor(p);
    let f = p - i;
    let wa = (3.0 - 2.0 * f) * 0.25;
    let wb = (1.0 + 2.0 * f) * 0.25;
    // Texel-centre coordinates of the two taps per axis, then + ½ and / size for uv.
    let pa = (i - 1.0 + (2.0 - f) / (3.0 - 2.0 * f) + 0.5) / size;
    let pb = (i + 1.0 + f / (1.0 + 2.0 * f) + 0.5) / size;
    let s00 = textureSampleLevel(bloom_src, bloom_sampler, vec2<f32>(pa.x, pa.y), 0.0).rgb;
    let s10 = textureSampleLevel(bloom_src, bloom_sampler, vec2<f32>(pb.x, pa.y), 0.0).rgb;
    let s01 = textureSampleLevel(bloom_src, bloom_sampler, vec2<f32>(pa.x, pb.y), 0.0).rgb;
    let s11 = textureSampleLevel(bloom_src, bloom_sampler, vec2<f32>(pb.x, pb.y), 0.0).rgb;
    return wa.y * (wa.x * s00 + wb.x * s10) + wb.y * (wa.x * s01 + wb.x * s11);
}

@compute @workgroup_size(8, 8, 1)
fn cs_up(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(bloom_dst);
    if (any(gid.xy >= dims)) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) * (1.0 / vec2<f32>(dims));
    let same = textureLoad(bloom_same, vec2<i32>(gid.xy), 0).rgb;
    textureStore(bloom_dst, gid.xy, vec4<f32>(same + tent(uv), 1.0));
}

// Fullscreen triangle: vertices (-1, 1), (3, 1), (-1, -3) in clip space cover the viewport.
@vertex
fn vs_composite(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(2.0 * x - 1.0, 1.0 - 2.0 * y, 0.0, 1.0);
}

// The bloom for the hdr pixel whose centre is `position`; the blend state mixes it in.
@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = position.xy * camera.viewport.zw;
    return vec4<f32>(tent(uv) * bloom_params.scale.x, 1.0);
}
