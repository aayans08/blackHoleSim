// Milky Way galaxy model v0 (M1 Task A5): the WGSL twin of `uni_procgen::galaxy::GalaxyModel`.
// Every function here mirrors the Rust function of the same role line by line; the sources and
// approximations of each parameter are documented in galaxy.rs and lf.rs.
//
// ShaderLib path "procgen/galaxy.wgsl". Galactocentric frame: origin at Sgr A*, +x from the Sun's
// projection toward the centre, +y toward Galactic longitude 90°, +z toward the north Galactic
// pole; distances in pc.
//
// How to use it: the importer declares the module-scope variable `galaxy: GalaxyGpu` that every
// gal_* function reads, at its own group and in its own address space, e.g. the band update's
//     @group(1) @binding(0) var<uniform> galaxy: GalaxyGpu;
// or, for a probe whose one storage input carries the model ahead of its data, a
// `var<private> galaxy: GalaxyGpu;` it fills first. (The gal_* signatures take no parameter
// block, and a binding here would fix the group for every importer. Until B4L this file held
// `var<private> galaxy` and every importer copied its uniform into it; on the M3 that 1.6 KB
// copy per invocation, read with dynamic indices, spilled to thread memory and made the cube's
// first build take ~0.8 s.)

// The model's parameters, filled by `GalaxyModel::gpu()`; vec4-sized members only (CLAUDE.md §3
// uniform layout rule). Lengths in pc.
struct GalaxyGpu {
    // L1, H1, L2, H2: thin and thick disk scale lengths and heights.
    disk: vec4<f32>,
    // R0, f (thick/thin at the Sun), thin-disk density in the plane at R0 [stars/pc³],
    // mean V luminosity per star [L☉,V].
    norm: vec4<f32>,
    // f_h, n_h, core radius, q.
    halo: vec4<f32>,
    // cos α, sin α of the bar angle, central density [stars/pc³], half-length.
    bar: vec4<f32>,
    // x0, y0, z0 scale lengths, end-taper width.
    bar_scale: vec4<f32>,
    // Gaussian σ, light contrast, dust contrast, x shift into Reid et al. 2019's frame.
    arm_shape: vec4<f32>,
    // Inner fade-in start and end radii, 0, 0.
    arm_taper: vec4<f32>,
    // Per arm: β_ref [rad], ln(R_ref / 1 pc), 1 / tan ψ, sin ψ.
    arms: array<vec4<f32>, 4>,
    // κ0 [A_V mag/pc in the dust midplane at R0 between the arms], h_r, h_z at R0, z_d (height of
    // the dust midplane: 19.5 pc, the gas layer's, not the stellar disk's z = 0).
    dust: vec4<f32>,
    // Flare h(R) = h0 + h1 max(R − r_f, 0): h0, h1 [pc/pc], r_f, h(R0); h_z(R) = h_z(R0) h(R)/h(R0).
    dust_flare: vec4<f32>,
    // Radiance per V luminosity column [cd/m² per (L☉,V/pc²)], 0, 0, 0.
    light: vec4<f32>,
    // Linear Rec.709 colours (Y = 1): thin disk, thick disk + halo, bar; w = 0.
    color_thin: vec4<f32>,
    color_old: vec4<f32>,
    color_bar: vec4<f32>,
    // Per luminosity-function bin: x_bright, x_faint, light weight, 0 (`lf::faint_bins`).
    lf: array<vec4<f32>, 21>,
    // Detail (B4L, galaxy.rs "Detail"): σ_ln of the dust, star and cloud fields; fBm gain.
    detail_sigma: vec4<f32>,
    // 1 / largest wavelength [1/pc] of the dust, star and cloud fields; no noise nearer than w [pc].
    detail_scale: vec4<f32>,
    // Octaves of the dust, star and cloud fields; number of named clouds.
    detail_octaves: vec4<u32>,
    // Seeds of the dust, star and cloud fields; 0.
    detail_seed: vec4<u32>,
    // Rows of the rotation between octaves; w of row 0: the lacunarity; w of row 1: z☉ [pc] (the
    // Sun is at (−R₀, 0, z☉)); w of row 2: 0.
    noise_rot: array<vec4<f32>, 3>,
    // Per named cloud, four vec4s: centre − Sun [pc] and peak A_V rate [mag/pc]; then the rows of
    // its whitening matrix W [1/pc] (the extinction rate is κ₀ exp(−|W (x − c)|²/2)).
    clouds: array<vec4<f32>, 48>,
}

// The parameters every gal_* function read are the importer's module-scope `galaxy` (above).

const GAL_PI: f32 = 3.14159265358979;
const GAL_TWO_PI: f32 = 6.28318530717959;
// Band integration: 1 pc to 30 kpc in 128 log-spaced steps (galaxy.rs BAND_*).
const GAL_BAND_S_MIN_PC: f32 = 1.0;
const GAL_BAND_S_MAX_PC: f32 = 30000.0;
const GAL_BAND_STEPS: u32 = 128u;
const GAL_BAND_SERIES_MAX_DTAU: f32 = 1e-3;
// Optical depth per magnitude of extinction, 0.4 ln 10 (Pogson's scale).
const GAL_TAU_PER_MAG: f32 = 0.921034037;
const GAL_LF_BINS: u32 = 21u;
const GAL_ARM_COUNT: u32 = 4u;
// Detail (galaxy.rs, B4L): the LOD fade of an octave (NOISE_LOD_FADE), the variance of one value-
// noise octave (VALUE_NOISE_VAR = (26/35)³/3), the octave seed step, the cloud skip and keep
// thresholds (CLOUD_Q_MAX, CLOUD_MIN_TAU, CLOUD_BEHIND_SIGMAS), the per-step noise threshold (NOISE_MIN_DTAU) and the
// named-cloud capacity (NAMED_CLOUD_COUNT).
const GAL_NOISE_FADE_START: f32 = 0.3;
const GAL_NOISE_FADE_END: f32 = 0.6;
const GAL_VALUE_NOISE_VAR: f32 = 0.136645287;
const GAL_OCTAVE_SEED_STEP: u32 = 0x68E31DA4u;
const GAL_CLOUD_Q_MAX: f32 = 50.0;
const GAL_CLOUD_MIN_TAU: f32 = 1e-4;
const GAL_CLOUD_BEHIND_SIGMAS: f32 = 6.0;
const GAL_NOISE_MIN_DTAU: f32 = 1e-3;
// The march stops once the optical depth passes this (BAND_TAU_STOP: e^−15 = 3e-7).
const GAL_BAND_TAU_STOP: f32 = 15.0;
const GAL_MAX_CLOUDS: u32 = 12u;

// `a` wrapped into [−π, π).
fn gal_wrap_pi(a: f32) -> f32 {
    return a - GAL_TWO_PI * floor((a + GAL_PI) / GAL_TWO_PI);
}

// 1 on a spiral arm's centre line, a Gaussian in the distance to the nearest arm, faded in
// between the arm_taper radii (galaxy.rs `arm_strength`).
fn gal_arm_strength(p: vec3<f32>) -> f32 {
    let sigma = galaxy.arm_shape.x;
    let q = vec2<f32>(p.x + galaxy.arm_shape.w, p.y);
    let r = max(length(q), 1.0);
    let beta = GAL_PI - atan2(q.y, q.x);
    var strength = 0.0;
    for (var k = 0u; k < GAL_ARM_COUNT; k = k + 1u) {
        let arm = galaxy.arms[k];
        let d_beta = gal_wrap_pi(beta - arm.x + (log(r) - arm.y) * arm.z);
        let d = r * abs(d_beta) * arm.w / sigma;
        strength = max(strength, exp(-0.5 * d * d));
    }
    return strength * smoothstep(galaxy.arm_taper.x, galaxy.arm_taper.y, r);
}

// Bar star density [stars/pc³] (galaxy.rs `bar_density`).
fn gal_bar_density(p: vec3<f32>) -> f32 {
    let xb = p.x * galaxy.bar.x - p.y * galaxy.bar.y;
    let yb = p.x * galaxy.bar.y + p.y * galaxy.bar.x;
    let rs = length(vec3<f32>(xb, yb, p.z) / galaxy.bar_scale.xyz);
    let beyond = max(length(vec2<f32>(xb, yb)) - galaxy.bar.w, 0.0) / galaxy.bar_scale.w;
    return galaxy.bar.z * exp(-rs - 0.5 * beyond * beyond);
}

// Star density of each component [stars/pc³] given the arm strength at `p`: thin disk (with arms),
// thick disk, halo, bar (galaxy.rs `components`).
fn gal_components(p: vec3<f32>, arms: f32) -> vec4<f32> {
    let r0 = galaxy.norm.x;
    let r = length(p.xy);
    let z = abs(p.z);
    let thin = exp(-(r - r0) / galaxy.disk.x - z / galaxy.disk.y)
        * (1.0 + galaxy.arm_shape.y * arms);
    let thick = galaxy.norm.y * exp(-(r - r0) / galaxy.disk.z - z / galaxy.disk.w);
    let zq = p.z / galaxy.halo.w;
    let r2 = r * r + zq * zq;
    let r_core = galaxy.halo.z;
    let halo = galaxy.halo.x * pow(r0 * r0 / max(r2, r_core * r_core), 0.5 * galaxy.halo.y);
    let n_thin0 = galaxy.norm.z;
    return vec4<f32>(n_thin0 * thin, n_thin0 * thick, n_thin0 * halo, gal_bar_density(p));
}

// Dust extinction [A_V mag/pc] given the arm strength at `p` (galaxy.rs `extinction`).
fn gal_extinction_at(p: vec3<f32>, arms: f32) -> f32 {
    let flare = galaxy.dust_flare;
    let r = length(p.xy);
    let h_z = galaxy.dust.z * (flare.x + flare.y * max(r - flare.z, 0.0)) / flare.w;
    let z = abs(p.z - galaxy.dust.w);
    return galaxy.dust.x * exp(-(r - galaxy.norm.x) / galaxy.dust.y - z / h_z)
        * (1.0 + galaxy.arm_shape.z * arms);
}

// Fraction of the light from stars fainter than the limit, at flux limit
// y = 10^(−0.4 v_lim) (d / 10 pc)² (lf.rs `faint_fraction_at`).
fn gal_faint_fraction(y: f32) -> f32 {
    var f = 0.0;
    for (var i = 0u; i < GAL_LF_BINS; i = i + 1u) {
        let b = galaxy.lf[i];
        f = f + b.z * clamp((y - b.y) / (b.x - b.y), 0.0, 1.0);
    }
    return f;
}

// Star density [stars/pc³] at galactocentric position `p_gc_pc` [pc].
fn gal_density(p_gc_pc: vec3<f32>) -> f32 {
    let c = gal_components(p_gc_pc, gal_arm_strength(p_gc_pc));
    return c.x + c.y + c.z + c.w;
}

// V-band luminosity density [L☉,V/pc³].
fn gal_luminosity(p_gc_pc: vec3<f32>) -> f32 {
    return gal_density(p_gc_pc) * galaxy.norm.w;
}

// Dust extinction A_V per unit length [mag/pc].
fn gal_extinction(p_gc_pc: vec3<f32>) -> f32 {
    return gal_extinction_at(p_gc_pc, gal_arm_strength(p_gc_pc));
}

// pcg3d (Jarzynski & Olano 2020, JCGT 9(3)): a verbatim copy of common/hash.wgsl's `pcg3d`, the
// twin of uni_core::pcg3d (this file is composed on its own by uni-procgen's naga tests, so it
// cannot #import; `gal_pcg3d_is_the_engine_hash` holds the two bodies equal).
fn gal_pcg3d(v_in: vec3<u32>) -> vec3<u32> {
    // LCG step (Numerical Recipes constants 1664525, 1013904223).
    var v = v_in * 1664525u + 1013904223u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    v ^= v >> vec3<u32>(16u);

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    return v;
}

// The lattice value at corner b + (dx, dy, dz): uniform on [−1, 1) from 24 bits of gal_pcg3d's
// first lane (galaxy.rs `value_noise`'s `corner`).
fn gal_lattice(b: vec3<u32>, dx: u32, dy: u32, dz: u32, seed: u32) -> f32 {
    let h = gal_pcg3d(vec3<u32>(b.x + dx, b.y + dy, (b.z + dz) ^ seed));
    return f32(h.x >> 8u) * (2.0 / 16777216.0) - 1.0;
}

// One octave of value noise at `p` [lattice units]: the 8 lattice values around it, uniform on
// [−1, 1) from 24 bits of gal_pcg3d's first lane, blended with smoothstep weights (galaxy.rs
// `value_noise`).
fn gal_value_noise(p: vec3<f32>, seed: u32) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (vec3<f32>(3.0) - 2.0 * f);
    let b = bitcast<vec3<u32>>(vec3<i32>(cell));
    let x00 = mix(gal_lattice(b, 0u, 0u, 0u, seed), gal_lattice(b, 1u, 0u, 0u, seed), u.x);
    let x10 = mix(gal_lattice(b, 0u, 1u, 0u, seed), gal_lattice(b, 1u, 1u, 0u, seed), u.x);
    let x01 = mix(gal_lattice(b, 0u, 0u, 1u, seed), gal_lattice(b, 1u, 0u, 1u, seed), u.x);
    let x11 = mix(gal_lattice(b, 0u, 1u, 1u, seed), gal_lattice(b, 1u, 1u, 1u, seed), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// fBm of value noise at `q` [pc, relative to the Sun] (galaxy.rs `fbm`): octave k has wavelength
// λ₀/2.03^k (inv_scale = 1/λ₀), amplitude gain^k, its lattice rotated k times, and fades out as
// (1 − smoothstep(0.3, 0.6, fp/λ_k)) for a footprint of `fp_pc` [pc]. Returns (n, kept): n
// scaled to unit variance with every octave kept, kept the share of that variance kept.
fn gal_fbm(q: vec3<f32>, inv_scale: f32, octaves: u32, seed: u32, fp_pc: f32) -> vec2<f32> {
    let gain = galaxy.detail_sigma.w;
    let lacunarity = galaxy.noise_rot[0].w;
    let g2 = gain * gain;
    let total = (1.0 - pow(g2, f32(octaves))) / (1.0 - g2);
    var p = q * inv_scale;
    var inv_lambda = inv_scale;
    var amp = 1.0;
    var sum = 0.0;
    var kept = 0.0;
    for (var k = 0u; k < octaves; k = k + 1u) {
        let w = 1.0 - smoothstep(GAL_NOISE_FADE_START, GAL_NOISE_FADE_END, fp_pc * inv_lambda);
        if (!(w > 0.0)) {
            break;
        }
        sum = sum + w * amp * gal_value_noise(p, seed + k * GAL_OCTAVE_SEED_STEP);
        kept = kept + w * w * amp * amp;
        p = vec3<f32>(dot(galaxy.noise_rot[0].xyz, p), dot(galaxy.noise_rot[1].xyz, p),
            dot(galaxy.noise_rot[2].xyz, p)) * lacunarity;
        inv_lambda = inv_lambda * lacunarity;
        amp = amp * gain;
    }
    return vec2<f32>(sum / sqrt(total * GAL_VALUE_NOISE_VAR), kept / total);
}

// erf(x) by Abramowitz & Stegun 1964, eq. 7.1.26 (|error| ≤ 1.5e-7; galaxy.rs `erf`).
fn gal_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * exp(-a * a);
    return select(-y, y, x >= 0.0);
}

// The named clouds one ray meets (galaxy.rs `RayClouds`), set up by gal_band_radiance and packed
// at the front (gal_cl_count of them; most rays meet none or one): per cloud its whole column
// τ_c, the distance s_m of its peak along the ray, and √a, the inverse of its σ along the ray.
var<private> gal_cl_tau: array<f32, 12>;
var<private> gal_cl_mid: array<f32, 12>;
var<private> gal_cl_sqrt_a: array<f32, 12>;
var<private> gal_cl_count: u32;

// The named clouds' optical depth from −∞ to `s` along the ray: Σ τ_c Φ((s − s_m)√a), Φ the
// normal CDF (galaxy.rs `RayClouds::column_to`; the same sum in cloud order).
fn gal_cloud_column(s: f32) -> f32 {
    var sum = 0.0;
    for (var k = 0u; k < gal_cl_count; k = k + 1u) {
        let x = (s - gal_cl_mid[k]) * gal_cl_sqrt_a[k];
        sum = sum + gal_cl_tau[k] * 0.5 * (1.0 + gal_erf(x * 0.70710678118654752));
    }
    return sum;
}

// Band radiance [cd/m², linear Rec.709] seen from `cam_gc_pc` along `dir`, from stars fainter
// than `v_lim`, for a sample of footprint `footprint` [rad] (galaxy.rs `band_radiance`):
// ∫ j f_faint e^{−τ} ds from 1 pc to 30 kpc in 128 log-spaced steps, j at each step's geometric
// midpoint, the dust within a step a uniform slab; with the detail: the clumpy dust and star
// clouds on steps past the Local Bubble that hold dust, and the named clouds' exact columns.
fn gal_band_radiance(cam_gc_pc: vec3<f32>, dir: vec3<f32>, v_lim: f32, footprint: f32)
    -> vec3<f32> {
    let d = normalize(dir);
    let x_lim = pow(10.0, -0.4 * v_lim);
    let ratio = GAL_BAND_S_MAX_PC / GAL_BAND_S_MIN_PC;
    let fp_rad = max(footprint, 0.0);
    // The Sun's galactocentric position: (−R₀, 0, z☉), z☉ in noise_rot[1].w.
    let cam_rel = cam_gc_pc - vec3<f32>(-galaxy.norm.x, 0.0, galaxy.noise_rot[1].w);
    // The named clouds this ray meets (galaxy.rs `ray_clouds`).
    let n_clouds = min(galaxy.detail_octaves.w, GAL_MAX_CLOUDS);
    gal_cl_count = 0u;
    for (var k = 0u; k < n_clouds; k = k + 1u) {
        let c = galaxy.clouds[4u * k];
        if (!(c.w > 0.0)) {
            continue;
        }
        let w0 = galaxy.clouds[4u * k + 1u].xyz;
        let w1 = galaxy.clouds[4u * k + 2u].xyz;
        let w2 = galaxy.clouds[4u * k + 3u].xyz;
        let v = c.xyz - cam_rel;
        let dw = vec3<f32>(dot(w0, d), dot(w1, d), dot(w2, d));
        let vw = vec3<f32>(dot(w0, v), dot(w1, v), dot(w2, v));
        let a = dot(dw, dw);
        let s_mid = dot(dw, vw) / a;
        let perp = vw - dw * s_mid;
        let q = dot(perp, perp);
        if (!(q < GAL_CLOUD_Q_MAX) || s_mid * sqrt(a) < -GAL_CLOUD_BEHIND_SIGMAS) {
            continue;
        }
        var tau_c = c.w * exp(-0.5 * q) * sqrt(GAL_TWO_PI / a) * GAL_TAU_PER_MAG;
        let s_near = max(s_mid, 0.0);
        let sig = galaxy.detail_sigma.z;
        let n = gal_fbm(cam_rel + d * s_near, galaxy.detail_scale.z, galaxy.detail_octaves.z,
            galaxy.detail_seed.z + k, fp_rad * max(s_near, 1.0));
        tau_c = tau_c * exp(sig * n.x - 0.5 * sig * sig * n.y);
        if (tau_c >= GAL_CLOUD_MIN_TAU) {
            gal_cl_tau[gal_cl_count] = tau_c;
            gal_cl_mid[gal_cl_count] = s_mid;
            gal_cl_sqrt_a[gal_cl_count] = sqrt(a);
            gal_cl_count = gal_cl_count + 1u;
        }
    }
    var cloud_tau0 = gal_cloud_column(GAL_BAND_S_MIN_PC);
    let sig_dust = galaxy.detail_sigma.x;
    let sig_star = galaxy.detail_sigma.y;
    var s0 = GAL_BAND_S_MIN_PC;
    var tau = 0.0;
    var sum = vec3<f32>(0.0);
    for (var i = 1u; i <= GAL_BAND_STEPS; i = i + 1u) {
        let s1 = GAL_BAND_S_MIN_PC * pow(ratio, f32(i) / f32(GAL_BAND_STEPS));
        let s = sqrt(s0 * s1);
        let ds = s1 - s0;
        let p = cam_gc_pc + d * s;
        let arms = gal_arm_strength(p);
        let c = gal_components(p, arms);
        var dtau = gal_extinction_at(p, arms) * GAL_TAU_PER_MAG * ds;
        var star_clouds = 1.0;
        if (s > galaxy.detail_scale.w && dtau >= GAL_NOISE_MIN_DTAU) {
            let q = cam_rel + d * s;
            let fp = fp_rad * s;
            let nd = gal_fbm(q, galaxy.detail_scale.x, galaxy.detail_octaves.x,
                galaxy.detail_seed.x, fp);
            dtau = dtau * exp(sig_dust * nd.x - 0.5 * sig_dust * sig_dust * nd.y);
            let ns = gal_fbm(q, galaxy.detail_scale.y, galaxy.detail_octaves.y,
                galaxy.detail_seed.y, fp);
            star_clouds = exp(sig_star * ns.x - 0.5 * sig_star * sig_star * ns.y);
        }
        let cloud_tau1 = gal_cloud_column(s1);
        dtau = dtau + (cloud_tau1 - cloud_tau0);
        cloud_tau0 = cloud_tau1;
        let rgb = c.x * star_clouds * galaxy.color_thin.xyz + (c.y + c.z) * galaxy.color_old.xyz
            + c.w * galaxy.color_bar.xyz;
        // ∫₀^ds e^{−κt} dt = ds (1 − e^{−dτ}) / dτ; below dτ = 1e-3 its series, where f32
        // 1 − e^{−dτ} would lose digits.
        var path = ds * (1.0 - dtau * (0.5 - dtau / 6.0));
        if (dtau >= GAL_BAND_SERIES_MAX_DTAU) {
            path = ds * (1.0 - exp(-dtau)) / dtau;
        }
        let s10 = s / 10.0;
        let faint = gal_faint_fraction(x_lim * s10 * s10);
        sum = sum + rgb * (faint * exp(-tau) * path);
        tau = tau + dtau;
        s0 = s1;
        if (tau > GAL_BAND_TAU_STOP) {
            break;
        }
    }
    return sum * (galaxy.norm.w * galaxy.light.x);
}
