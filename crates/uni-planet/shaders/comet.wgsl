// planet/comet.wgsl: the comet pass "comets" (Task C1). Four pipelines in one render pass:
//
// - nucleus (vs_nucleus / fs_nucleus): the procedural mesh (crates/uni-planet/src/comet/
//   nucleus.rs), opaque, depth write, back faces culled; lunar-Lambert L = 0.9 (McEwen 1991) with
//   ρ from p_V 0.04 (Lamy et al. 2004); exposure pre_mix(star_pre, scene_pre, w) for the LOD band;
//   lens_mask.g = 1 (coverage) and lens_mask.a = the star share 1 - w, both with Max.
// - coma (vs_coma / fs_coma): a quad facing the camera through the nucleus, the profile
//   coma_drawn_brightness (Haser-like K/ρ e^(-ρ/ℓ), softened over one pixel), additive.
// - ion tail (vs_ion / fs_ion): a ribbon along the tail's axis that turns to face the camera,
//   ion_tail_brightness, additive.
// - dust tail (vs_dust / fs_dust): one Gaussian sprite per Finson-Probstein grain, lit by
//   E☉ ∝ 1/r² at the grain and the Henyey-Greenstein phase function, additive.
//
// World light: luminance [cd/m²] × scene_pre() (the frame contract); the coma and the tails are
// optically thin and write nothing to lens_mask (their location 1 has an empty write mask).
// Everything is camera-relative f32 in root (ICRS) axes. CPU twins (crates/uni-planet/src/comet/):
// coma_surface_brightness, coma_drawn_brightness, ion_tail_brightness, dust_phase,
// sprite_profile, nucleus_reflectance; `universe --selftest` compares them on the GPU.

#import "common/camera.wgsl"
#import "common/constants.wgsl"

// Grains per tail slot: 32 β × 64 ages (twin of tails::GRAINS_PER_TAIL).
const GRAINS_PER_TAIL: u32 = 2048u;
// Segments of the ion-tail ribbon (twin of tails::ION_SEGMENTS).
const ION_SEGMENTS: u32 = 32u;
// e-foldings of the ion tail's fall-off over its length (tails::ION_TAIL_EFOLDS).
const ION_EFOLDS: f32 = 3.0;
// The ion tail's FWHM [m] (tails::ION_TAIL_WIDTH_M) and the ribbon's half width in σ.
const ION_WIDTH_M: f32 = 1.0e8;
const ION_HALF_WIDTH_SIGMAS: f32 = 3.5;
// FWHM / σ of a Gaussian: 2√(2 ln 2).
const FWHM_PER_SIGMA: f32 = 2.354820045;
// Henyey-Greenstein asymmetry of the dust (tails::DUST_PHASE_G).
const DUST_G: f32 = 0.6;
// Grain sprites are cut at this many σ (tails::SPRITE_SIGMAS).
const SPRITE_SIGMAS: f32 = 3.0;
// A grain sprite is at least this many pixels wide (σ), so a sub-pixel sprite neither aliases nor
// vanishes; its light is kept.
const SPRITE_MIN_PX: f32 = 0.75;
// ...and at most this many: a nearer sprite keeps its central surface brightness but is cut to
// this size (a camera inside a tail sees a veil, not 2 048 screen-filling quads; approximate).
const SPRITE_MAX_PX: f32 = 32.0;
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
    // xyz: Sun − nucleus [m]; w: the dust tail's phase-averaged luminous intensity [cd].
    sun: vec4<f32>,
    // xyz: the ion tail's unit direction; w: its length [m].
    ion_axis: vec4<f32>,
    // rgb: the ion tail's colour (luminance 1); w: 0.
    ion_colour: vec4<f32>,
    // rgb: the dust's colour (sunlight, luminance 1); w: 0.
    dust_colour: vec4<f32>,
}

// Twin of uni_planet::comet::GrainGpu: 32 bytes.
struct GrainGpu {
    // xyz: offset from the nucleus [m]; w: the sprite's σ [m].
    offset: vec4<f32>,
    // x: light weight (Σ w·(r_h/r)² = 1 over the tail); yzw: 0.
    weight: vec4<f32>,
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

// The ion tail's surface brightness [cd/m²] on its ribbon at s along and x across the axis, for
// a tail of `length` carrying `intensity`, seen at sin ψ to its axis. Twin of
// tails::ion_tail_brightness.
fn ion_tail_brightness(s: f32, x: f32, length: f32, intensity: f32, sin_psi: f32) -> f32 {
    if (!(s >= 0.0 && s < length)) {
        return 0.0;
    }
    let lambda = length / ION_EFOLDS;
    let tip = exp(-ION_EFOLDS);
    let norm = lambda * (1.0 - tip) - length * tip;
    let along = (exp(-s / lambda) - tip) / norm;
    let sigma = ION_WIDTH_M / FWHM_PER_SIGMA;
    let t = x / sigma;
    let across = exp(-0.5 * t * t) / (sqrt(TAU) * sigma);
    let sin_eff = max(sin_psi, ION_WIDTH_M / length);
    return intensity * along * across / sin_eff;
}

// Henyey & Greenstein 1941 (ApJ 93, 70), g = DUST_G, mean 1 over the sphere. Twin of
// tails::dust_phase.
fn dust_phase(cos_theta: f32) -> f32 {
    let g = DUST_G;
    let denom = 1.0 + g * g - 2.0 * g * clamp(cos_theta, -1.0, 1.0);
    return (1.0 - g * g) / (denom * sqrt(denom));
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
    let dir = normalize(c.centre.xyz);
    let right = facing_right(dir);
    let up = cross(right, dir);
    let local = quad_corner(vi % 6u) * c.centre.w;
    var out: ComaOut;
    out.clip = to_clip(c.centre.xyz + right * local.x + up * local.y);
    out.local = local;
    out.k = k;
    return out;
}

@fragment
fn fs_coma(in: ComaOut) -> GlowOut {
    let c = comae[in.k];
    let rho = length(in.local);
    if (rho > c.centre.w) {
        discard;
    }
    let s = coma_drawn_brightness(rho, c.profile.x, c.profile.y, c.profile.z);
    return glow(c.colour.rgb * s);
}

// ---------------------------------------------------------------------------------------------
// Ion tail.

struct IonOut {
    @builtin(position) clip: vec4<f32>,
    // x: distance along the axis [m]; y: across it [m]; z: sin ψ (the axis vs the line of sight).
    @location(0) coords: vec3<f32>,
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
    let dir = normalize(p);
    var side = cross(axis, dir);
    let sin_psi = length(side);
    if (sin_psi < 1e-6) {
        side = camera_right();
    } else {
        side = side / sin_psi;
    }
    let x = corner.x * ION_HALF_WIDTH_SIGMAS * ION_WIDTH_M / FWHM_PER_SIGMA;
    var out: IonOut;
    out.clip = to_clip(p + side * x);
    out.coords = vec3<f32>(s, x, sin_psi);
    out.slot = slot;
    return out;
}

@fragment
fn fs_ion(in: IonOut) -> GlowOut {
    let t = tails[in.slot];
    let s = ion_tail_brightness(in.coords.x, in.coords.y, t.ion_axis.w, t.nucleus.w, in.coords.z);
    return glow(t.ion_colour.rgb * s);
}

// ---------------------------------------------------------------------------------------------
// Dust tail.

struct DustOut {
    @builtin(position) clip: vec4<f32>,
    // xy: position in the sprite's plane relative to the grain [m]; z: σ [m]; w: intensity [cd].
    @location(0) sprite: vec4<f32>,
    @location(1) @interpolate(flat) slot: u32,
}

@vertex
fn vs_dust(@builtin(vertex_index) vi: u32, @builtin(instance_index) g: u32) -> DustOut {
    let slot = g / GRAINS_PER_TAIL;
    let t = tails[slot];
    let grain = grains[g];
    let rel = t.nucleus.xyz + grain.offset.xyz;
    let dist = length(rel);
    var out: DustOut;
    out.slot = slot;
    if (!(dist > 1.0)) {
        // Degenerate (the camera on the grain): every corner at one point, nothing drawn.
        out.clip = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.sprite = vec4<f32>(0.0);
        return out;
    }
    let dir = rel / dist;
    // Sunlight at the grain relative to the nucleus's, (r_h/r)², and the scattering angle
    // between the Sun → grain and grain → camera directions.
    let helio = grain.offset.xyz - t.sun.xyz;
    let lit = dot(t.sun.xyz, t.sun.xyz) / dot(helio, helio);
    let cos_theta = dot(normalize(helio), -dir);
    var intensity = t.sun.w * grain.weight.x * lit * dust_phase(cos_theta);
    let px = pixel_size_at(dist);
    var sigma = max(grain.offset.w, SPRITE_MIN_PX * px);
    let sigma_max = SPRITE_MAX_PX * px;
    if (sigma > sigma_max) {
        intensity = intensity * (sigma_max / sigma) * (sigma_max / sigma);
        sigma = sigma_max;
    }
    let right = facing_right(dir);
    let up = cross(right, dir);
    let local = quad_corner(vi % 6u) * (SPRITE_SIGMAS * sigma);
    out.clip = to_clip(rel + right * local.x + up * local.y);
    out.sprite = vec4<f32>(local, sigma, intensity);
    return out;
}

@fragment
fn fs_dust(in: DustOut) -> GlowOut {
    let t = tails[in.slot];
    let s = in.sprite.w * sprite_profile(length(in.sprite.xy), in.sprite.z);
    return glow(t.dust_colour.rgb * s);
}
