// post/bloom.wgsl: bloom "bloom" (Task B6), pass 15 of the M1 frame contract.
//
// Jimenez 2014 (SIGGRAPH Advances in Real-Time Rendering, "Next Generation Post Processing in
// Call of Duty: Advanced Warfare"):
// - cs_down_first: level 1 from hdr, the 13-tap downsample with every source texel clamped to
//   BLOOM_INPUT_MAX first and the partial Karis average (per block of four taps);
// - cs_down: level k from level k − 1, the 13-tap downsample of bilinear taps;
// - cs_up: u_k = d_k + tent(u_{k+1}), a 3 × 3 tent of bilinear taps one lower-level texel apart
//   (u_L = d_L);
// - vs_composite / fs_composite: bloom = tent(u_1) / L at full resolution; the pipeline's blend
//   (constant, 1 - constant) with the constant = strength makes hdr = mix(hdr, bloom, strength).
// No threshold. Every step is normalised, so the chain conserves energy away from the edges.
//
// CPU twin: uni_post::bloom::bloom_reference (crates/uni-post/src/bloom.rs), same taps, blocks and
// order of operations; `universe --selftest` compares the two on an impulse.
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

@compute @workgroup_size(8, 8, 1)
fn cs_down_first(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= textureDimensions(bloom_dst))) {
        return;
    }
    // The 6 × 6 source texels 2·dst - 2 ..= 2·dst + 3, clamped to the edges and to
    // BLOOM_INPUT_MAX one by one (a bilinear tap would average a hot texel before clamping it).
    let last = vec2<i32>(textureDimensions(bloom_src)) - vec2<i32>(1);
    let base = 2 * vec2<i32>(gid.xy) - vec2<i32>(2);
    var t: array<array<vec3<f32>, 6>, 6>;
    for (var y = 0; y < 6; y++) {
        for (var x = 0; x < 6; x++) {
            let p = clamp(base + vec2<i32>(x, y), vec2<i32>(0), last);
            t[y][x] = bloom_input(textureLoad(bloom_src, p, 0).rgb);
        }
    }
    // The five blocks, each the mean of its 4 × 4 texels (= its four 2 × 2 taps): the centre
    // (taps at ±1) at origin (1, 1), the corners (taps at {-2, 0} or {0, 2}) at (0, 0), (2, 0),
    // (0, 2), (2, 2).
    var origins = array<vec2<i32>, 5>(
        vec2<i32>(1, 1),
        vec2<i32>(0, 0),
        vec2<i32>(2, 0),
        vec2<i32>(0, 2),
        vec2<i32>(2, 2),
    );
    var blocks: array<vec3<f32>, 5>;
    for (var b = 0; b < 5; b++) {
        var s = vec3<f32>(0.0);
        for (var y = 0; y < 4; y++) {
            for (var x = 0; x < 4; x++) {
                s += t[origins[b].y + y][origins[b].x + x];
            }
        }
        blocks[b] = s / 16.0;
    }
    let c = bloom_combine(blocks[0], blocks[1], blocks[2], blocks[3], blocks[4], true);
    textureStore(bloom_dst, gid.xy, vec4<f32>(c, 1.0));
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
// `uv`.
fn tent(uv: vec2<f32>) -> vec3<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(bloom_src));
    var sum = vec3<f32>(0.0);
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let w = f32(2 - abs(ox)) * f32(2 - abs(oy));
            let o = vec2<f32>(f32(ox), f32(oy));
            sum += w * textureSampleLevel(bloom_src, bloom_sampler, uv + o * texel, 0.0).rgb;
        }
    }
    return sum / 16.0;
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
