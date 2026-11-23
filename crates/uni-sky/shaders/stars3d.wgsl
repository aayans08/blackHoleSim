// sky/stars3d.wgsl: the 3-D star pass "stars" (Task B4), and the splat code the point pass
// (sky/points.wgsl) shares by importing this file.
//
// Each star is one instance of a screen-aligned quad (four vertices, two triangles from an index
// buffer) centred on its exact sub-pixel projected position, drawn at the far plane (depth 0,
// GreaterEqual). Per star the vertex stage:
//   1. emits a degenerate quad for a suppressed instance slot (near stars drawn elsewhere);
//   2. places the star relative to the camera [pc]:
//        p = (sector centre − camera) + offset + v·t,
//      where the sector centre comes relative to an anchor near the camera and the uniform
//      carries camera − anchor (both f64 differences on the CPU, then f32; design §5.2);
//   3. m = M_V + 5 log10 d − 5, E = 10^(−0.4 (m + 13.99)) lx (Allen 1973);
//   4. splat_vertex: M0's PSF (imported) sizing at E × twinkle / Ω(θ) × star_pre(), culled below
//      VISIBILITY_CUTOFF (the twinkle is 1 in space; under an atmosphere it follows Young's 1967
//      scintillation index on a TDB clock, see twinkle_factor);
//   5. for a drawn splat only, the colour from B−V through the Planck table, boosted by
//      vivid_color when the option is on.
// The fragment stage integrates the PSF over its pixel (erf differences, so the footprint sums
// to the star's energy at any sub-pixel position) and writes it × (1 − r)(1 − g) of lens_mask.
//
//   hdr value = E / Ω(θ) × star_pre() × (1 − r)(1 − g) × PSF weight,  Ω(θ) = Ω_c·cos³θ.
//
// M0's PSF is used as M0 wrote it, not copied (review r0 I3): this file imports M0's
// sky/stars.wgsl for its PSF constants, psf_erf, pixel_coverage, moffat, halo_weight,
// pixel_solid_angle_off_axis, MIN_CLIP_W, CULLED and its `lens_mask` declaration (group 1,
// binding 1), which both 3-D passes read. The import also brings M0's own pass (StarInstance,
// `stars` at binding 0, VsOut, vs_main, fs_main), which no 3-D entry point uses, so wgpu leaves
// it out of the 3-D pipelines; this file's bindings take 2–5 and the point pass's 6–7, so no two
// globals of a composed module share a slot. The one PSF function of B4's own is
// stars3d_footprint_radius_px: M0's footprint_radius_px with the 2-px minimum (review m16).
//
// CPU twins: crates/uni-sky/src/stars3d.rs (and psf.rs, stars.rs for M0's PSF and solid angle).
// `shader_constants_match_cpu` keeps the constants equal; `stars3d_points_layouts_match_wgsl`
// checks the structs; `splat_layouts_match_entry_point_use` checks the bindings against the
// pipeline layouts; `crate::probes` runs the maths on the GPU under `universe --selftest`.

#import "common/camera.wgsl"
#import "common/color.wgsl"
#import "common/constants.wgsl"
#import "common/hash.wgsl"
#import "sky/stars.wgsl"

// Twin of `uni_sky::stars3d::StarGpu`: 32 bytes, two vec4s (review m15).
struct StarGpu {
    // xyz: offset from the star's sector centre [pc] at the catalogue epoch; w: M_V.
    offset_absmag: vec4<f32>,
    // x: f16 (vx, vy) [km/s]; y: f16 (vz, B−V), each pair low half first; z: uni_data flag
    // bits; w: sector index.
    vel_bv_flags: vec4<u32>,
}

// Twin of `uni_sky::stars3d::StarsParams`: 80 bytes, written once per frame.
struct StarsParams {
    // xyz: camera − anchor [pc]; w: years since the catalogue epoch (J2000.0).
    camera_years: vec4<f32>,
    // x: twinkle log-amplitude sigma (0 = steady); y: 1 with vivid colours, else 0; z: twinkle
    // phase in [0, 1] from draw twinkle_step.x to the next; w: 0.
    look: vec4<f32>,
    // x: the twinkle draw: TDB seconds since J2000 / 0.1 s, floored, modulo 2^32 (computed in
    // f64 on the CPU: sim_time_s() is f32 and steps by up to 0.25 s); yzw: 0.
    twinkle_step: vec4<u32>,
    // Instance slots of the suppressed stars; NO_STAR where unused.
    suppressed: array<vec4<u32>, 2>,
}

// Twin of `uni_sky::stars3d::BvLut`: Planck colours (luminance 1) at B−V evenly spaced over
// [BV_LUT_MIN, BV_LUT_MAX].
struct BvColourLut {
    entries: array<vec4<f32>, 128>,
}

// Group 1 binding 1 is M0's `lens_mask` (sky/stars.wgsl, imported above): uni_gfx's lens mask,
// r = captured fraction, g = sky occlusion. Read with textureLoad at the fragment's pixel; it has
// the size of hdr, so the vertex stage reads the attachment size from it. The point pass binds
// it at the same slot. Binding 0 is M0's `stars`, unused here.
@group(1) @binding(2) var<uniform> stars_params: StarsParams;
@group(1) @binding(3) var<uniform> bv_lut: BvColourLut;
// The star records in instance order.
@group(1) @binding(4) var<storage, read> catalogue: array<StarGpu>;
// xyz: sector centre − anchor [pc]; w: 0.
@group(1) @binding(5) var<storage, read> sectors: array<vec4<f32>>;

// The PSF constants (CORE_SIGMA_PX, HALO_*, MAX_RADIUS_PX, VISIBILITY_CUTOFF), MIN_CLIP_W and
// CULLED are M0's, from sky/stars.wgsl; crates/uni-sky/src/psf.rs gives their sources.
// The 3-D pass's smallest footprint radius (review m16; M0's 3 px is unchanged).
const STARS3D_MIN_RADIUS_PX: f32 = 2.0;
// 1 km/s in pc per Julian year (IAU 1976 year, IAU 2012 B2 parsec).
const PC_PER_YR_PER_KMS: f32 = 1.02271217e-6;
// log10(2) and log2(10): WGSL has log2 and exp2 only.
const LOG10_2: f32 = 0.30102999566398120;
const LOG2_10: f32 = 3.32192809488736235;
// V zero point of uni_core::star_illuminance_lux: E = 10^(−0.4 (m + 13.99)) lx (Allen 1973).
const LUX_ZERO_POINT_MAG: f32 = 13.99;
// Largest display energy of a splat, so a source at the camera stays finite.
const MAX_SPLAT_ENERGY: f32 = 1.0e30;
// The B−V table's length and range (uni_core::bv_to_teff's clamp, Ballesteros 2012).
const BV_LUT_LEN: u32 = 128u;
const BV_LUT_MIN: f32 = -0.4;
const BV_LUT_MAX: f32 = 2.0;
// Saturation factor of "Vivid star colours · non-physical" (the owner-approved preview's 1.6).
const VIVID_SATURATION: f32 = 1.6;
// The twinkle hash's third lane.
const TWINKLE_SEED: u32 = 0x7f4a7c15u;
// An unused entry of StarsParams.suppressed.
const NO_STAR: u32 = 0xffffffffu;

// Footprint radius [px]: where the core and the halo have both fallen below VISIBILITY_CUTOFF,
// clamped to [STARS3D_MIN_RADIUS_PX, MAX_RADIUS_PX]. M0's footprint_radius_px (sky/stars.wgsl)
// but for the minimum, which M0 fixes at its own 3 px, so it cannot be called with 2 px.
// footprint_matches_m0_above_three_px holds the two together above 3 px. Twin of
// `uni_sky::stars3d::footprint_radius_px`.
fn stars3d_footprint_radius_px(energy: f32, halo_fraction: f32) -> f32 {
    var r = STARS3D_MIN_RADIUS_PX;
    let core_peak = (1.0 - halo_fraction) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
    if (core_peak > VISIBILITY_CUTOFF) {
        r = max(r, CORE_SIGMA_PX * sqrt(2.0 * log(core_peak / VISIBILITY_CUTOFF)));
    }
    let halo_peak = halo_fraction * energy * moffat(0.0);
    if (halo_peak > VISIBILITY_CUTOFF) {
        let q = pow(halo_peak / VISIBILITY_CUTOFF, 1.0 / HALO_BETA);
        r = max(r, HALO_ALPHA_PX * sqrt(q - 1.0));
    }
    return min(r, MAX_RADIUS_PX);
}

// Whether a splat's core peak or halo peak rises above VISIBILITY_CUTOFF. Twin of
// `uni_sky::stars3d::star_is_visible`.
fn star_is_visible(energy: f32, halo_fraction: f32) -> bool {
    let core_peak = (1.0 - halo_fraction) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
    let halo_peak = halo_fraction * energy * moffat(0.0);
    return core_peak > VISIBILITY_CUTOFF || halo_peak > VISIBILITY_CUTOFF;
}

// IEEE 754 binary16 bits (the low 16 bits of h) → f32, exactly: normal numbers by moving the
// sign, exponent (bias 15 → 127) and mantissa into place; zero and subnormals as mantissa × 2^−24
// (exact, a power-of-two scale); infinities and NaN as f32 ones (the CPU never stores them).
// Built-in unpack2x16float needs naga's SHADER_FLOAT16_IN_FLOAT32 capability, which
// uni_gfx::validate_wgsl does not enable. Twin of `uni_data::f16_to_f32`.
fn f16_bits_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 0x1fu;
    let m = h & 0x3ffu;
    if (e == 0u) {
        return bitcast<f32>(sign | bitcast<u32>(f32(m) * 5.9604645e-8));
    }
    if (e == 31u) {
        return bitcast<f32>(sign | 0x7f800000u | (m << 13u));
    }
    return bitcast<f32>(sign | ((e + 112u) << 23u) | (m << 13u));
}

// The two binary16 halves of w, low half first (the order of uni_data::Star3's packing). Twin of
// `uni_sky::stars3d::unpack_f16_pair`.
fn unpack_f16_pair(w: u32) -> vec2<f32> {
    return vec2<f32>(f16_bits_to_f32(w & 0xffffu), f16_bits_to_f32(w >> 16u));
}

// Star position relative to the camera [pc] at `years` after the catalogue epoch:
// (sector centre − camera + offset) + v·years. Twin of `uni_sky::stars3d::star_rel_pos_pc`.
fn star_rel_pos_pc(star: StarGpu, sector_rel: vec3<f32>, years: f32) -> vec3<f32> {
    let vxy = unpack_f16_pair(star.vel_bv_flags.x);
    let vz = unpack_f16_pair(star.vel_bv_flags.y).x;
    let v = vec3<f32>(vxy, vz) * PC_PER_YR_PER_KMS;
    return (sector_rel + star.offset_absmag.xyz) + v * years;
}

// Apparent V magnitude: M + 5 log10 d − 5. Twin of `uni_sky::star_apparent_mag`.
fn star_apparent_mag(abs_mag: f32, d_pc: f32) -> f32 {
    return abs_mag + 5.0 * LOG10_2 * log2(d_pc) - 5.0;
}

// Illuminance [lux] of apparent V magnitude v: 10^(−0.4 (v + 13.99)). Twin of
// `uni_sky::stars3d::star_lux`.
fn star_lux(v_mag: f32) -> f32 {
    return exp2(-0.4 * LOG2_10 * (v_mag + LUX_ZERO_POINT_MAG));
}

// Planck colour (luminance 1) of colour index bv, linear in B−V, clamped to the table's ends.
// Twin of `uni_sky::stars3d::bv_colour`.
fn bv_colour(bv: f32) -> vec3<f32> {
    let x = clamp((bv - BV_LUT_MIN) / (BV_LUT_MAX - BV_LUT_MIN), 0.0, 1.0)
        * f32(BV_LUT_LEN - 1u);
    let i = min(u32(x), BV_LUT_LEN - 2u);
    let f = x - f32(i);
    let a = bv_lut.entries[i].rgb;
    let b = bv_lut.entries[i + 1u].rgb;
    return a + f * (b - a);
}

// The "Vivid star colours · non-physical" boost: the colour's distance from the grey of its own
// Rec. 709 luminance scaled by VIVID_SATURATION, negative channels clamped to 0 and the result
// rescaled to the original luminance. Twin of `uni_sky::vivid_color`.
fn vivid_color(c: vec3<f32>) -> vec3<f32> {
    let y = luminance(c);
    if (!(y > 0.0)) {
        return c;
    }
    let grey = vec3<f32>(y);
    let boosted = max(grey + VIVID_SATURATION * (c - grey), vec3<f32>(0.0));
    return boosted * (y / luminance(boosted));
}

// One twinkle draw of unit variance: pcg3d (Jarzynski & Olano 2020) of (slot, step, seed), the
// sum of three 24-bit uniforms (Irwin–Hall) shifted and scaled to mean 0, variance 1. Twin of
// `uni_sky::stars3d::twinkle_noise`.
fn twinkle_noise(slot: u32, step: u32) -> f32 {
    let h = pcg3d(vec3<u32>(slot, step, TWINKLE_SEED));
    let u = vec3<f32>(h >> vec3<u32>(8u)) * (1.0 / 16777216.0);
    return (u.x + u.y + u.z - 1.5) * 2.0;
}

// I/<I> = exp(σn − σ²/2), log-normal (Dravins et al. 1997, PASP 109, 173), with σ from Young's
// (1967) scintillation index (computed on the CPU, `uni_sky::stars3d::scintillation_index`).
// n is draw `step`'s value blended smoothly (smoothstep of `phase`) into draw step + 1's and
// renormalised to unit variance. The mean is 1 to 1e-4: at a draw n is Irwin–Hall, whose exact
// mean factor is (sinh σ/σ)³·e^(−σ²/2) = 0.99993 at σ = 0.25. Exactly 1 for σ ≤ 0. Twin of
// `uni_sky::stars3d::twinkle_factor`.
fn twinkle_factor(slot: u32, step: u32, phase: f32, sigma: f32) -> f32 {
    if (!(sigma > 0.0)) {
        return 1.0;
    }
    let f = clamp(phase, 0.0, 1.0);
    let t = f * f * (3.0 - 2.0 * f);
    let n0 = twinkle_noise(slot, step);
    let n1 = twinkle_noise(slot, step + 1u);
    let n = (n0 + t * (n1 - n0)) / sqrt((1.0 - t) * (1.0 - t) + t * t);
    return exp(sigma * n - 0.5 * sigma * sigma);
}

// Corner (±1, ±1) of index-buffer vertex vi: 0 (−,−), 1 (+,−), 2 (−,+), 3 (+,+); the index
// buffer draws (0, 1, 2) and (2, 1, 3). (M0's quad_corner maps six non-indexed vertices.)
fn splat_corner(vi: u32) -> vec2<f32> {
    return vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
}

struct SplatOut {
    @builtin(position) pos: vec4<f32>,
    // Source centre in framebuffer pixels: x right, y down, pixel k spans [k, k + 1].
    @location(0) @interpolate(flat) centre_px: vec2<f32>,
    // Pre-exposed hdr value per unit PSF weight, before the lens mask: colour·E/Ω(θ)·star_pre().
    @location(1) @interpolate(flat) radiance: vec3<f32>,
    // x: halo share h; y: footprint radius R [px].
    @location(2) @interpolate(flat) halo: vec2<f32>,
}

fn culled_splat() -> SplatOut {
    var out: SplatOut;
    out.pos = CULLED;
    out.centre_px = vec2<f32>(0.0);
    out.radiance = vec3<f32>(0.0);
    out.halo = vec2<f32>(0.0);
    return out;
}

// Vertex vi of the splat of a source in unit direction `dir` (root axes) with steady illuminance
// `lux` [lx] scaled by `twinkle` (1 for a steady source) and colour `colour` (luminance 1), at
// NDC depth `depth` (0 for stars at the far plane). Culled behind the camera, without light,
// below VISIBILITY_CUTOFF or off screen by more than its footprint. Twin of
// `uni_sky::stars3d::twin::splat` and `vertex`.
fn splat_vertex(
    vi: u32, dir: vec3<f32>, lux: f32, twinkle: f32, colour: vec3<f32>, depth: f32,
) -> SplatOut {
    let out_culled = culled_splat();
    // Directions go in with w = 0: the reversed-Z infinite projection maps view (x, y, z, 0) to
    // clip (f·x/aspect, f·y, 0, −z), and clip.w = −z is positive only in front of the camera.
    let view_dir = (camera.view_rot * vec4<f32>(dir, 0.0)).xyz;
    let clip = camera.proj * vec4<f32>(view_dir, 0.0);
    if (!(clip.w > MIN_CLIP_W) || !(lux > 0.0)) {
        return out_culled;
    }
    // The rasteriser maps NDC onto the attachment, so use the attachment's size (M0).
    let size = vec2<f32>(textureDimensions(lens_mask));
    let ndc = clip.xy / clip.w;
    let centre = vec2<f32>(0.5 * (ndc.x + 1.0) * size.x, 0.5 * (1.0 - ndc.y) * size.y);
    let cos_theta = -view_dir.z / length(view_dir);
    let omega = pixel_solid_angle_off_axis(camera.time.z, size.y, cos_theta);
    // Display-normalised energy: the footprint's sum of pre-exposed hdr values.
    let energy = min(lux * twinkle * star_pre() / omega, MAX_SPLAT_ENERGY);
    // The halo share follows the steady illuminance, so it does not switch on and off at the
    // twinkle rate (review r0 M7).
    let h = select(0.0, HALO_ENERGY_FRACTION, lux > HALO_MIN_LUX);
    if (!star_is_visible(energy, h)) {
        return out_culled;
    }
    let radius = stars3d_footprint_radius_px(energy, h);
    let reach = vec2<f32>(radius + 1.0);
    if (!(all(centre > -reach) && all(centre < size + reach))) {
        return out_culled;
    }
    var out: SplatOut;
    // Half-size R + 0.5 px, so every pixel that overlaps the footprint is shaded.
    out.pos = vec4<f32>(ndc + splat_corner(vi) * (radius + 0.5) * 2.0 / size, depth, 1.0);
    out.centre_px = centre;
    out.radiance = colour * energy;
    out.halo = vec2<f32>(h, radius);
    return out;
}

// PSF weight of pixel `pix` (integer index) for a splat centred at `centre` with halo (h, R):
// (1 − h)·Gaussian core integrated over the pixel + h·Moffat halo at the pixel's centre.
fn psf_weight(pix: vec2<f32>, centre: vec2<f32>, halo: vec2<f32>) -> f32 {
    let edge = pix - centre;
    var weight = (1.0 - halo.x) * pixel_coverage(edge.x) * pixel_coverage(edge.y);
    if (halo.x > 0.0) {
        weight += halo.x * halo_weight(length(pix + vec2<f32>(0.5) - centre), halo.y);
    }
    return weight;
}

// The share of sky light kept under lens-mask value `mask`: (1 − r)(1 − g) (frame contract).
// Twin of `uni_sky::stars3d::twin::star_sky_factor`.
fn star_sky_factor(mask: vec4<f32>) -> f32 {
    return (1.0 - clamp(mask.r, 0.0, 1.0)) * (1.0 - clamp(mask.g, 0.0, 1.0));
}

fn is_suppressed(slot: u32) -> bool {
    let s = stars_params.suppressed;
    return any(s[0] == vec4<u32>(slot)) || any(s[1] == vec4<u32>(slot));
}

@vertex
fn vs_stars(@builtin(vertex_index) vi: u32, @builtin(instance_index) slot: u32) -> SplatOut {
    if (is_suppressed(slot)) {
        return culled_splat();
    }
    let star = catalogue[slot];
    let sector_rel = sectors[star.vel_bv_flags.w].xyz - stars_params.camera_years.xyz;
    let p = star_rel_pos_pc(star, sector_rel, stars_params.camera_years.w);
    let d = length(p);
    if (!(d > 0.0)) {
        return culled_splat();
    }
    let lux = star_lux(star_apparent_mag(star.offset_absmag.w, d));
    let twinkle = twinkle_factor(
        slot, stars_params.twinkle_step.x, stars_params.look.z, stars_params.look.x);
    var out = splat_vertex(vi, p / d, lux, twinkle, vec3<f32>(1.0), 0.0);
    // The colour only for a drawn splat (review r0 M8); a culled one has radius 0.
    if (out.halo.y > 0.0) {
        var colour = bv_colour(unpack_f16_pair(star.vel_bv_flags.y).y);
        if (stars_params.look.y > 0.5) {
            colour = vivid_color(colour);
        }
        out.radiance = colour * out.radiance;
    }
    return out;
}

@fragment
fn fs_stars(in: SplatOut) -> @location(0) vec4<f32> {
    // @builtin(position).xy is this pixel's centre (k + 0.5), so floor gives the pixel index k.
    let pix = floor(in.pos.xy);
    let mask = textureLoad(lens_mask, vec2<i32>(pix), 0);
    let rgb = in.radiance * (psf_weight(pix, in.centre_px, in.halo) * star_sky_factor(mask));
    return vec4<f32>(min(rgb, vec3<f32>(HDR_MAX)), 0.0);
}
