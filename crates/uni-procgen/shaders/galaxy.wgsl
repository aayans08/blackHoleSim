// Milky Way galaxy model v0 (M1 Task A5): the WGSL twin of `uni_procgen::galaxy::GalaxyModel`.
// Every function here mirrors the Rust function of the same role line by line; the sources and
// approximations of each parameter are documented in galaxy.rs and lf.rs.
//
// ShaderLib path "procgen/galaxy.wgsl". Galactocentric frame: origin at Sgr A*, +x from the Sun's
// projection toward the centre, +y toward Galactic longitude 90°, +z toward the north Galactic
// pole; distances in pc.
//
// How to use it: the importer declares `GalaxyGpu` at its own group, e.g.
//     @group(1) @binding(0) var<uniform> band_galaxy: GalaxyGpu;
// and copies it into this file's `galaxy` at the start of its entry point, before calling any
// gal_* function:
//     galaxy = band_galaxy;
// (The gal_* signatures take no parameter block, and a binding here would fix the group for
// every importer.)

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
}

// The parameters every gal_* function reads; the importer copies its uniform here.
var<private> galaxy: GalaxyGpu;

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

// Band radiance [cd/m², linear Rec.709] seen from `cam_gc_pc` along `dir`, from stars fainter
// than `v_lim` (galaxy.rs `band_radiance`): ∫ j f_faint e^{−τ} ds from 1 pc to 30 kpc in 128
// log-spaced steps, j at each step's geometric midpoint, the dust within a step a uniform slab.
fn gal_band_radiance(cam_gc_pc: vec3<f32>, dir: vec3<f32>, v_lim: f32) -> vec3<f32> {
    let d = normalize(dir);
    let x_lim = pow(10.0, -0.4 * v_lim);
    let ratio = GAL_BAND_S_MAX_PC / GAL_BAND_S_MIN_PC;
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
        let rgb = c.x * galaxy.color_thin.xyz + (c.y + c.z) * galaxy.color_old.xyz
            + c.w * galaxy.color_bar.xyz;
        let dtau = gal_extinction_at(p, arms) * GAL_TAU_PER_MAG * ds;
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
    }
    return sum * (galaxy.norm.w * galaxy.light.x);
}
