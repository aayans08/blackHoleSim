// planet/comet.wgsl: the comet pass "comets" (Task C1). Four pipelines in one render pass:
//
// - nucleus (vs_nucleus / fs_nucleus): the procedural mesh (crates/uni-planet/src/comet/
//   nucleus.rs), opaque, depth write, back faces culled; lunar-Lambert L = 0.9 (McEwen 1991) with
//   ρ from p_V 0.04 (Lamy et al. 2004); exposure pre_mix(star_pre, scene_pre, w) for the LOD band;
//   lens_mask.g = 1 (coverage) and lens_mask.a = the star share 1 - w, both with Max.
// - coma (vs_coma / fs_coma): a quad facing the camera through the nucleus, the profile
//   coma_drawn_brightness (Haser-like K/ρ e^(-ρ/ℓ), softened over one pixel), additive.
// - ion tail (vs_ion / fs_ion): a ribbon along the tail's axis that turns to face the camera,
//   ion_tail_drawn_brightness (σ at least ION_MIN_SIGMA_PX on screen, light kept), additive.
// - dust tail (vs_dust / fs_dust): one elliptical Gaussian splat per Finson-Probstein grain or
//   merged block (C1L; EWA splatting, Zwicker et al. 2001): its 3D covariance projected onto the
//   sprite's plane (splat_axes), lit by E☉ ∝ 1/r² at the grain and the Henyey-Greenstein phase
//   function (dust_grain_light) × the tail's emission weight × the dust's light production
//   (TailGpu.sun.w = I_coma / DUST_HEAD_TIME_S), additive. Splats at least COARSE_MIN_SIGMA_PX
//   wide are drawn into a coarse target (1/8 of the lines; vs_dust_coarse / fs_dust_coarse, its
//   own pass before this one) and added back by fs_dust_composite with a cubic B-spline.
//
// Glow (coma, ion, dust) whose brightest point would land below GLOW_CULL_PRE pre-exposed is not
// drawn: a thousandth of AgX's black point, invisible at the frame's exposure.
//
// World light: luminance [cd/m²] × scene_pre() (the frame contract); the coma and the tails are
// optically thin and write nothing to lens_mask (their location 1 has an empty write mask).
// Everything is camera-relative f32 in root (ICRS) axes. CPU twins (crates/uni-planet/src/comet/):
// coma_surface_brightness, coma_drawn_brightness, ion_tail_brightness,
// ion_tail_drawn_brightness, ion_drawn_sigma, dust_phase, dust_grain_light, sprite_profile,
// nucleus_reflectance; `universe --selftest` compares them on the GPU.

#import "common/camera.wgsl"
#import "common/constants.wgsl"

// Splat slots per tail: 16 β × 64 ages (twin of tails::GRAINS_PER_TAIL).
const GRAINS_PER_TAIL: u32 = 1024u;
// Segments of the ion-tail ribbon (twin of tails::ION_SEGMENTS).
const ION_SEGMENTS: u32 = 32u;
// e-foldings of the ion tail's fall-off over its length (tails::ION_TAIL_EFOLDS).
const ION_EFOLDS: f32 = 3.0;
// The ion tail's FWHM [m] (tails::ION_TAIL_WIDTH_M) and the ribbon's half width in σ.
const ION_WIDTH_M: f32 = 1.0e8;
const ION_HALF_WIDTH_SIGMAS: f32 = 3.5;
// The ribbon's smallest on-screen σ [px] (tails::ION_MIN_SIGMA_PX, review C1 I2): a Gaussian of
// σ ≥ 0.75 px sampled at pixel centres sums to its integral within 2·e^(−2π²σ²) ≈ 3e-5, so a
// sub-pixel tail neither beads nor breaks into dashes; its light across is kept.
const ION_MIN_SIGMA_PX: f32 = 0.75;
// FWHM / σ of a Gaussian: 2√(2 ln 2).
const FWHM_PER_SIGMA: f32 = 2.354820045;
// Henyey-Greenstein asymmetry of the dust (tails::DUST_PHASE_G).
const DUST_G: f32 = 0.6;
// Grain splats are cut at this many σ (tails::SPRITE_SIGMAS).
const SPRITE_SIGMAS: f32 = 2.0;
// A splat's σ gets this many pixels added in quadrature on both axes (tails::SPRITE_MIN_PX), so
// a sub-pixel splat neither aliases nor vanishes; its light is kept.
const SPRITE_MIN_PX: f32 = 0.75;
// ...and each axis is at most this many on the full-resolution path (thin needles only; a wider
// splat goes to the coarse path) and COARSE_MAX_PX on the coarse one: a wider splat keeps its
// surface brightness and loses the light beyond (a camera inside a tail; approximate).
const SPRITE_MAX_PX: f32 = 256.0;
const COARSE_MAX_PX: f32 = 1024.0;
// A splat whose minor σ is at least this many pixels is drawn into the coarse target, 1/8 of the
// lines (comet::COARSE_DOWNSAMPLE), there ≥ 2 texels wide, and upsampled with a cubic B-spline.
const COARSE_MIN_SIGMA_PX: f32 = 16.0;
// Glow whose brightest point is below this pre-exposed value is not drawn (comet::GLOW_CULL_PRE:
// AgX's black point 2^-12.47393 / 1000).
const GLOW_CULL_PRE: f32 = 1.75e-7;
// The coma quad reaches this many scale lengths (coma::COMA_EXTENT_SCALES).
const COMA_EXTENT_SCALES: f32 = 10.0;
// Lunar-Lambert weight of the nucleus (nucleus::NUCLEUS_BRDF, McEwen 1991).
const NUCLEUS_L: f32 = 0.9;

// Twin of uni_planet::comet::NucleusGpu: 96 bytes, six vec4s.
struct NucleusGpu {
    // xyz: nucleus − camera [m], root axes; w: the exposure weight w (1 = scene exposure).
    centre: vec4<f32>,
    // The body → root rotation's columns (w: 0).
    axis_x: vec4<f32>,
    axis_y: vec4<f32>,
    axis_z: vec4<f32>,
    // xyz: unit direction from the nucleus to the Sun; w: sunlight E☉ [lx] at the nucleus.
    sun: vec4<f32>,
    // rgb: the Sun's colour (luminance 1); w: normal reflectance ρ.
    light: vec4<f32>,
}

// Twin of uni_planet::comet::ComaGpu: 48 bytes.
struct ComaGpu {
    // xyz: nucleus − camera [m]; w: the quad's half size [m].
    centre: vec4<f32>,
    // x: luminous intensity [cd]; y: scale length ℓ [m]; z: pixel radius [m]; w: 0.
    profile: vec4<f32>,
    // rgb: colour (luminance 1); w: 0.
    colour: vec4<f32>,
}

// Twin of uni_planet::comet::TailGpu: 80 bytes, one per tail slot.
struct TailGpu {
    // xyz: nucleus − camera [m]; w: the ion tail's luminous intensity [cd].
    nucleus: vec4<f32>,
    // xyz: Sun − nucleus [m]; w: the dust's light production [cd/s per unit of HG]
    // (tails::dust_emission_cd_per_s: I_coma / DUST_HEAD_TIME_S).
    sun: vec4<f32>,
    // xyz: the ion tail's unit direction; w: its length [m].
    ion_axis: vec4<f32>,
    // rgb: the ion tail's colour (luminance 1); w: 0.
    ion_colour: vec4<f32>,
    // rgb: the dust's colour (sunlight, luminance 1); w: 0.
    dust_colour: vec4<f32>,
}

// Twin of uni_planet::comet::GrainGpu: 48 bytes, one dust splat.
struct GrainGpu {
    // xyz: offset from the nucleus [m]; w: the emission weight [s] (tails::TailGrain::weight).
    offset: vec4<f32>,
    // The 3D covariance's Cholesky factor L [m] (tails::cov_factor): L00, L10, L20, L11...
    cov_l0: vec4<f32>,
    // ...and L21, L22 (zw: 0).
    cov_l1: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> nuclei: array<NucleusGpu>;
@group(1) @binding(1) var<storage, read> comae: array<ComaGpu>;
@group(1) @binding(2) var<storage, read> tails: array<TailGpu>;
@group(1) @binding(3) var<storage, read> grains: array<GrainGpu>;

// ---------------------------------------------------------------------------------------------
// Twins of the CPU functions (same order of operations, f32).

// Coma surface brightness [cd/m²] K/ρ · e^(−ρ/ℓ), K = I/(2πℓ) (Haser 1957; ∫S dA = I). Twin of
// coma::coma_surface_brightness.
fn coma_surface_brightness(rho: f32, intensity: f32, scale: f32) -> f32 {
    return intensity / (TAU * scale) / rho * exp(-rho / scale);
}

// The drawn coma: softened over the pixel radius `pixel`, ℓ' = max(ℓ, pixel), ∫S dA = I kept.
// Twin of coma::coma_drawn_brightness.
fn coma_drawn_brightness(rho: f32, intensity: f32, scale: f32, pixel: f32) -> f32 {
    let l = max(scale, pixel);
    let u = sqrt(rho * rho + pixel * pixel);
    return coma_surface_brightness(u, intensity * exp(pixel / l), l);
}

// The ion tail as the ribbon draws it [cd/m²], at s along and x across the axis, for a tail of
// `length` carrying `intensity`, seen at sin ψ to its axis, with its Gaussian cross-section
// widened to `sigma` (≥ the physical σ) and renormalised (the light across is kept). Twin of
// tails::ion_tail_drawn_brightness.
fn ion_tail_drawn_brightness(
    s: f32,
    x: f32,
    length: f32,
    intensity: f32,
    sin_psi: f32,
    sigma: f32,
) -> f32 {
    if (!(s >= 0.0 && s < length && sigma > 0.0)) {
        return 0.0;
    }
    let lambda = length / ION_EFOLDS;
    let tip = exp(-ION_EFOLDS);
    let norm = lambda * (1.0 - tip) - length * tip;
    let along = (exp(-s / lambda) - tip) / norm;
    let t = x / sigma;
    let across = exp(-0.5 * t * t) / (sqrt(TAU) * sigma);
    let sin_eff = max(sin_psi, ION_WIDTH_M / length);
    return intensity * along * across / sin_eff;
}

// The ion tail's surface brightness [cd/m²] at its physical width (FWHM ION_WIDTH_M). Twin of
// tails::ion_tail_brightness.
fn ion_tail_brightness(s: f32, x: f32, length: f32, intensity: f32, sin_psi: f32) -> f32 {
    let sigma = ION_WIDTH_M / FWHM_PER_SIGMA;
    return ion_tail_drawn_brightness(s, x, length, intensity, sin_psi, sigma);
}

// The σ [m] the ribbon is drawn with where a pixel spans `pixel` metres: the physical σ, at
// least ION_MIN_SIGMA_PX pixels. Twin of tails::ion_drawn_sigma_m.
fn ion_drawn_sigma(pixel: f32) -> f32 {
    return max(ION_WIDTH_M / FWHM_PER_SIGMA, ION_MIN_SIGMA_PX * pixel);
}

// Henyey & Greenstein 1941 (ApJ 93, 70), g = DUST_G, mean 1 over the sphere. Twin of
// tails::dust_phase.
fn dust_phase(cos_theta: f32) -> f32 {
    let g = DUST_G;
    let denom = 1.0 + g * g - 2.0 * g * clamp(cos_theta, -1.0, 1.0);
    return (1.0 - g * g) / (denom * sqrt(denom));
}

// A grain's light per unit of the dust's light production: emission weight [s] × (r_h/r)² (E☉
// at the grain relative to the nucleus's) × HG(θ), θ between the Sun → grain and grain → camera
// directions. `offset`: grain − nucleus, `nucleus_rel`: nucleus − camera, `sun_rel`: Sun −
// nucleus [m]. 0 with the camera within 1 m of the grain. Twin of tails::dust_grain_light.
fn dust_grain_light(
    offset: vec3<f32>,
    weight: f32,
    nucleus_rel: vec3<f32>,
    sun_rel: vec3<f32>,
) -> f32 {
    let rel = nucleus_rel + offset;
    let dist = length(rel);
    let helio = offset - sun_rel;
    let helio2 = dot(helio, helio);
    if (!(dist > 1.0 && helio2 > 0.0)) {
        return 0.0;
    }
    let lit = dot(sun_rel, sun_rel) / helio2;
    let cos_theta = dot(normalize(helio), -rel / dist);
    return weight * lit * dust_phase(cos_theta);
}

// A grain sprite's profile [1/m²]: a Gaussian cut at SPRITE_SIGMAS σ, shifted to 0 there and
// renormalised to a unit integral. Twin of tails::sprite_profile.
fn sprite_profile(r: f32, sigma: f32) -> f32 {
    if (!(sigma > 0.0 && abs(r) < SPRITE_SIGMAS * sigma)) {
        return 0.0;
    }
    let cut2 = SPRITE_SIGMAS * SPRITE_SIGMAS;
    let edge = exp(-0.5 * cut2);
    let norm = TAU * sigma * sigma * (1.0 - edge * (1.0 + 0.5 * cut2));
    let t = r / sigma;
    return (exp(-0.5 * t * t) - edge) / norm;
}

// A splat's ellipse on the sprite plane (C1L): the 3D covariance L Lᵀ (l0 = (L00, L10, L20,
// L11), l1 = (L21, L22)) projected on the unit vectors right and up, plus floor² on both axes.
// With uᵢ the projected columns of L: trace Σ|uᵢ|² + 2f², determinant Σ_{i<j}(uᵢ × uⱼ)² +
// f²Σ|uᵢ|² + f⁴ (Cauchy–Binet: no cancellation for elongated splats), σ₁² = m + √(m² − det),
// σ₂² = det/σ₁². Returns (σ₁, σ₂, φ): σ₁ ≥ σ₂ along cos φ·right + sin φ·up and −sin φ·right +
// cos φ·up. Twin of tails::splat_axes.
fn splat_axes(
    l0: vec4<f32>,
    l1: vec2<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    floor: f32,
) -> vec3<f32> {
    let c0 = vec3<f32>(l0.x, l0.y, l0.z);
    let c1 = vec3<f32>(0.0, l0.w, l1.x);
    let c2 = vec3<f32>(0.0, 0.0, l1.y);
    let u0 = vec2<f32>(dot(right, c0), dot(up, c0));
    let u1 = vec2<f32>(dot(right, c1), dot(up, c1));
    let u2 = vec2<f32>(dot(right, c2), dot(up, c2));
    let f2 = floor * floor;
    let p = u0.x * u0.x + u1.x * u1.x + u2.x * u2.x;
    let q = u0.x * u0.y + u1.x * u1.y + u2.x * u2.y;
    let r = u0.y * u0.y + u1.y * u1.y + u2.y * u2.y;
    let c01 = u0.x * u1.y - u0.y * u1.x;
    let c02 = u0.x * u2.y - u0.y * u2.x;
    let c12 = u1.x * u2.y - u1.y * u2.x;
    let det = c01 * c01 + c02 * c02 + c12 * c12 + f2 * (p + r) + f2 * f2;
    let m = 0.5 * (p + r) + f2;
    let l_1 = m + sqrt(max(m * m - det, 0.0));
    let l_2 = det / l_1;
    return vec3<f32>(sqrt(l_1), sqrt(l_2), 0.5 * atan2(2.0 * q, p - r));
}

// Lunar-Lambert reflectance factor (McEwen 1991, Icarus 92, 298), L = NUCLEUS_L:
// 2Lμ₀/(μ₀ + μ) + (1 − L)μ₀, 0 when unlit or facing away. Twin of
// uni_physics::photometry::reflectance(NUCLEUS_BRDF, μ₀, μ).
fn nucleus_reflectance(mu0: f32, mu: f32) -> f32 {
    if (mu0 <= 0.0 || mu <= 0.0) {
        return 0.0;
    }
    return 2.0 * NUCLEUS_L * mu0 / (mu0 + mu) + (1.0 - NUCLEUS_L) * mu0;
}

// The star share a world writer stores in lens_mask.a (lead ruling B6 I1): 1 − w where it covers
// anything. Twin of uni_planet::body::star_share.
fn star_share(w: f32, coverage: f32) -> f32 {
    return select(0.0, 1.0 - w, coverage > 0.0);
}

// ---------------------------------------------------------------------------------------------
// Shared geometry.

// The camera's +x and +y axes in root axes (rows 0 and 1 of view_rot).
fn camera_right() -> vec3<f32> {
    return vec3<f32>(camera.view_rot[0][0], camera.view_rot[1][0], camera.view_rot[2][0]);
}
fn camera_up() -> vec3<f32> {
    return vec3<f32>(camera.view_rot[0][1], camera.view_rot[1][1], camera.view_rot[2][1]);
}

// Right and up of a quad facing the camera across the line of sight `dir` (unit, camera → quad).
fn facing_right(dir: vec3<f32>) -> vec3<f32> {
    let r = cross(dir, camera_up());
    if (dot(r, r) < 1e-8) {
        return camera_right();
    }
    return normalize(r);
}

// Corner k (0..6) of a quad as two triangles, in [−1, 1]².
fn quad_corner(k: u32) -> vec2<f32> {
    var c = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    return c[k];
}

fn to_clip(rel: vec3<f32>) -> vec4<f32> {
    return camera.proj * camera.view_rot * vec4<f32>(rel, 1.0);
}

// Pixel size [m] at distance d along a ray: d·2·tan(fov/2)/height (proj[1][1] = 1/tan(fov/2)).
fn pixel_size_at(d: f32) -> f32 {
    return d * 2.0 / (camera.proj[1][1] * camera.viewport.y);
}

// Light written by the additive parts (location 1 is masked off: they leave lens_mask alone).
struct GlowOut {
    @location(0) color: vec4<f32>,
    @location(1) mask: vec4<f32>,
}

fn glow(radiance: vec3<f32>) -> GlowOut {
    var out: GlowOut;
    out.color = vec4<f32>(min(radiance * scene_pre(), vec3<f32>(HDR_MAX)), 0.0);
    out.mask = vec4<f32>(0.0);
    return out;
}

// ---------------------------------------------------------------------------------------------
// Nucleus.

struct NucleusOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) k: u32,
}

@vertex
fn vs_nucleus(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) k: u32,
) -> NucleusOut {
    let n = nuclei[k];
    let rot = mat3x3<f32>(n.axis_x.xyz, n.axis_y.xyz, n.axis_z.xyz);
    let rel = n.centre.xyz + rot * pos;
    var out: NucleusOut;
    out.clip = to_clip(rel);
    out.rel = rel;
    out.normal = rot * normal;
    out.k = k;
    return out;
}

struct NucleusFsOut {
    @location(0) color: vec4<f32>,
    // lens_mask: g = coverage (1), a = the star share; blended with Max.
    @location(1) mask: vec4<f32>,
}

@fragment
fn fs_nucleus(in: NucleusOut) -> NucleusFsOut {
    let n = nuclei[in.k];
    let normal = normalize(in.normal);
    let mu0 = dot(normal, n.sun.xyz);
    let mu = dot(normal, -normalize(in.rel));
    // L = ρ · R(μ₀, μ) · E☉/π [cd/m²] (the radiance convention of uni_physics::photometry).
    let radiance = n.light.w * nucleus_reflectance(mu0, mu) * n.sun.w / PI * n.light.rgb;
    let w = n.centre.w;
    let pre = pre_mix(star_pre(), scene_pre(), w);
    var out: NucleusFsOut;
    out.color = vec4<f32>(min(radiance * pre, vec3<f32>(HDR_MAX)), 1.0);
    out.mask = vec4<f32>(0.0, 1.0, 0.0, star_share(w, 1.0));
    return out;
}

// ---------------------------------------------------------------------------------------------
// Coma.

struct ComaOut {
    @builtin(position) clip: vec4<f32>,
    // Position in the quad's plane relative to the nucleus [m].
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) k: u32,
}

@vertex
fn vs_coma(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> ComaOut {
    let c = comae[k];
    var out: ComaOut;
    out.k = k;
    // Visibility (C1L): the drawn profile is at most C/ρ with C = I·e^(a/ℓ')/(2πℓ'), ℓ' =
    // max(ℓ, a), and at most I/(2πℓ'a) at the centre; nothing beyond ρ = C·pre/GLOW_CULL_PRE (or
    // at all) can reach the cut.
    let l = max(c.profile.y, c.profile.z);
    let big_c = c.profile.x * exp(c.profile.z / l) / (TAU * l);
    let pre = scene_pre();
    let peak = c.profile.x / (TAU * l * c.profile.z) * pre;
    if (!(peak >= GLOW_CULL_PRE)) {
        out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.local = vec2<f32>(0.0);
        return out;
    }
    let half = min(c.centre.w, big_c * pre / GLOW_CULL_PRE);
    let dir = normalize(c.centre.xyz);
    let right = facing_right(dir);
    let up = cross(right, dir);
    let local = quad_corner(vi % 6u) * half;
    out.clip = to_clip(c.centre.xyz + right * local.x + up * local.y);
    out.local = local;
    return out;
}

@fragment
fn fs_coma(in: ComaOut) -> GlowOut {
    let c = comae[in.k];
    let rho = length(in.local);
    if (rho > c.centre.w) {
        discard;
    }
    // (The quad may be smaller than c.centre.w: vs_coma's visibility bound.)
    let s = coma_drawn_brightness(rho, c.profile.x, c.profile.y, c.profile.z);
    return glow(c.colour.rgb * s);
}

// ---------------------------------------------------------------------------------------------
// Ion tail.

struct IonOut {
    @builtin(position) clip: vec4<f32>,
    // x: distance along the axis [m]; y: across it [m]; z: sin ψ (the axis vs the line of sight);
    // w: the drawn σ [m] (ion_drawn_sigma at the vertex's distance; the ribbon's edges sit at
    // ±ION_HALF_WIDTH_SIGMAS of it, so x/σ is ±3.5 along both edges).
    @location(0) coords: vec4<f32>,
    @location(1) @interpolate(flat) slot: u32,
}

@vertex
fn vs_ion(@builtin(vertex_index) vi: u32, @builtin(instance_index) slot: u32) -> IonOut {
    let t = tails[slot];
    let seg = vi / 6u;
    let corner = quad_corner(vi % 6u);
    // Along: corner −1 → this segment's start, +1 → its end; across: the side.
    let a = f32(seg) + 0.5 * (corner.y + 1.0);
    let s = t.ion_axis.w * a / f32(ION_SEGMENTS);
    let axis = t.ion_axis.xyz;
    let p = t.nucleus.xyz + axis * s;
    let dist = length(p);
    let dir = normalize(p);
    var side = cross(axis, dir);
    let sin_psi = length(side);
    if (sin_psi < 1e-6) {
        side = camera_right();
    } else {
        side = side / sin_psi;
    }
    // The pixel floor (review C1 I2): `side` is across the line of sight, so one pixel there
    // spans pixel_size_at(dist) metres.
    let sigma = ion_drawn_sigma(pixel_size_at(dist));
    let x = corner.x * ION_HALF_WIDTH_SIGMAS * sigma;
    var out: IonOut;
    out.slot = slot;
    // Visibility (C1L): a segment whose start (its brightest point, the axis at the segment's
    // first s) stays below GLOW_CULL_PRE is not drawn; all six vertices decide alike.
    let s0 = t.ion_axis.w * f32(seg) / f32(ION_SEGMENTS);
    let p0 = t.nucleus.xyz + axis * s0;
    let sin0 = length(cross(axis, normalize(p0)));
    let sigma0 = ion_drawn_sigma(pixel_size_at(length(p0)));
    let peak = ion_tail_drawn_brightness(s0, 0.0, t.ion_axis.w, t.nucleus.w, sin0, sigma0);
    if (!(peak * scene_pre() >= GLOW_CULL_PRE)) {
        out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.coords = vec4<f32>(0.0);
        return out;
    }
    out.clip = to_clip(p + side * x);
    out.coords = vec4<f32>(s, x, sin_psi, sigma);
    return out;
}

@fragment
fn fs_ion(in: IonOut) -> GlowOut {
    let t = tails[in.slot];
    let c = in.coords;
    let s = ion_tail_drawn_brightness(c.x, c.y, t.ion_axis.w, t.nucleus.w, c.z, c.w);
    return glow(t.ion_colour.rgb * s);
}

// ---------------------------------------------------------------------------------------------
// Dust tail.

struct DustOut {
    @builtin(position) clip: vec4<f32>,
    // xy: position on the splat in units of its σ₁, σ₂; z: σ₁σ₂ [m²]; w: intensity [cd].
    @location(0) sprite: vec4<f32>,
    @location(1) @interpolate(flat) slot: u32,
}

// One corner of splat g, on the full-resolution path (coarse = false) or the coarse one. Both
// paths compute the same ellipse; a splat whose minor σ is at least COARSE_MIN_SIGMA_PX pixels
// belongs to the coarse path, and the other path collapses it (nothing drawn).
fn dust_vertex(vi: u32, g: u32, coarse: bool) -> DustOut {
    let slot = g / GRAINS_PER_TAIL;
    let t = tails[slot];
    let grain = grains[g];
    let rel = t.nucleus.xyz + grain.offset.xyz;
    let dist = length(rel);
    var out: DustOut;
    out.slot = slot;
    out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.sprite = vec4<f32>(0.0);
    if (!(dist > 1.0)) {
        // Degenerate (the camera on the grain): every corner at one point, nothing drawn.
        return out;
    }
    let dir = rel / dist;
    // The splat's light: the dust's light production × its emission weight × (r_h/r)² × HG.
    var intensity =
        t.sun.w * dust_grain_light(grain.offset.xyz, grain.offset.w, t.nucleus.xyz, t.sun.xyz);
    let right = facing_right(dir);
    let up = cross(right, dir);
    let px = pixel_size_at(dist);
    let axes = splat_axes(grain.cov_l0, grain.cov_l1.xy, right, up, SPRITE_MIN_PX * px);
    if ((axes.y >= COARSE_MIN_SIGMA_PX * px) != coarse) {
        return out;
    }
    var s1 = axes.x;
    var s2 = axes.y;
    let s_max = select(SPRITE_MAX_PX, COARSE_MAX_PX, coarse) * px;
    if (s1 > s_max) {
        intensity = intensity * (s_max / s1);
        s1 = s_max;
    }
    if (s2 > s_max) {
        intensity = intensity * (s_max / s2);
        s2 = s_max;
    }
    // Visibility (C1L): the splat's centre is its brightest point.
    let area = s1 * s2;
    if (!(intensity * sprite_profile(0.0, 1.0) / area * scene_pre() >= GLOW_CULL_PRE)) {
        return out;
    }
    let c = cos(axes.z);
    let sn = sin(axes.z);
    let e1 = c * right + sn * up;
    let e2 = c * up - sn * right;
    let q = quad_corner(vi % 6u) * SPRITE_SIGMAS;
    out.clip = to_clip(rel + e1 * (q.x * s1) + e2 * (q.y * s2));
    out.sprite = vec4<f32>(q, area, intensity);
    return out;
}

// The splat's pre-exposed radiance at a fragment: the unit profile in σ units over the
// ellipse's area σ₁σ₂ (∫ = 1 over the plane) × its intensity, in the dust's colour.
fn dust_radiance(in: DustOut) -> vec3<f32> {
    let s = in.sprite.w * sprite_profile(length(in.sprite.xy), 1.0) / in.sprite.z;
    return min(tails[in.slot].dust_colour.rgb * s * scene_pre(), vec3<f32>(HDR_MAX));
}

@vertex
fn vs_dust(@builtin(vertex_index) vi: u32, @builtin(instance_index) g: u32) -> DustOut {
    return dust_vertex(vi, g, false);
}

@fragment
fn fs_dust(in: DustOut) -> GlowOut {
    var out: GlowOut;
    out.color = vec4<f32>(dust_radiance(in), 0.0);
    out.mask = vec4<f32>(0.0);
    return out;
}

// The coarse path (C1L): wide splats, drawn into the coarse target (1/8 of the lines, cleared
// every frame, no depth), at 1/64 of the fill.
@vertex
fn vs_dust_coarse(@builtin(vertex_index) vi: u32, @builtin(instance_index) g: u32) -> DustOut {
    return dust_vertex(vi, g, true);
}

@fragment
fn fs_dust_coarse(in: DustOut) -> @location(0) vec4<f32> {
    return vec4<f32>(dust_radiance(in), 0.0);
}

// The coarse target, read back into hdr by the composite (group 2 of its own layout).
@group(2) @binding(0) var dust_coarse: texture_2d<f32>;
@group(2) @binding(1) var dust_sampler: sampler;

// The coarse target at uv with a cubic B-spline: four bilinear taps (Sigg & Hadwiger 2005,
// "Fast Third-Order Texture Filtering", GPU Gems 2, ch. 20). The weights sum to 1, so the light
// is kept; the B-spline is smooth (C²) where bilinear upsampling by 8 would show its facets.
fn sample_coarse(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(dust_coarse));
    let p = uv * size - 0.5;
    let i = floor(p);
    let f = p - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let t0 = (i - 1.0 + w1 / g0 + 0.5) / size;
    let t1 = (i + 1.0 + w3 / g1 + 0.5) / size;
    let a = textureSampleLevel(dust_coarse, dust_sampler, vec2<f32>(t0.x, t0.y), 0.0).rgb;
    let b = textureSampleLevel(dust_coarse, dust_sampler, vec2<f32>(t1.x, t0.y), 0.0).rgb;
    let c = textureSampleLevel(dust_coarse, dust_sampler, vec2<f32>(t0.x, t1.y), 0.0).rgb;
    let d = textureSampleLevel(dust_coarse, dust_sampler, vec2<f32>(t1.x, t1.y), 0.0).rgb;
    return g0.y * (g0.x * a + g1.x * b) + g1.y * (g0.x * c + g1.x * d);
}

// The composite: a triangle covering the screen (scissored to the coarse splats' bounds) at depth
// 0, the far plane, drawn with GreaterEqual: the wide dust shows where no body wrote depth (a
// tail in front of a body is hidden by it; approximate).
@vertex
fn vs_dust_composite(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_dust_composite(@builtin(position) pos: vec4<f32>) -> GlowOut {
    var out: GlowOut;
    let light = sample_coarse(pos.xy / camera.viewport.xy);
    out.color = vec4<f32>(min(light, vec3<f32>(HDR_MAX)), 0.0);
    out.mask = vec4<f32>(0.0);
    return out;
}
