// atmo/atmosphere_common.wgsl: the medium, the LUT lookups and the per-pixel view march shared by
// the three draw variants (atmosphere_dual, atmosphere_mask, atmosphere_grey), the LUT passes and
// the probes. Hillaire 2020 (CGF 39(4), "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique"; MIT sample code used as the algorithm reference, see ATTRIBUTION.md)
// with Bruneton 2017's LUT parameterisation. CPU twins live in crates/uni-atmo/src/lut.rs
// (`medium`, `march`, `step_count`, ...) and feature.rs (`segment_radiance`).
//
// Positions and directions are in the body's *scaled* frame: body-fixed axes stretched so the
// body's reference ellipsoid becomes the sphere of its mean radius (AtmoDrawGpu.to_body_*), so
// the shell follows the oblate ground. A ray parameter t is a true distance in metres.
//
// Bindings: group 0 camera (common/camera.wgsl); group 1 the draw (uniform with a dynamic offset)
// and the depth target, and for the composite the two half-resolution scratch targets; group 2
// the atmosphere: binding 1 transmittance LUT and 3 sampler (from atmo/ground_lighting.wgsl), 4
// multiple-scattering LUT, 5 parameters.
//
// The pass (task B9p, plan fallback ladder steps 1-3): the march runs at half resolution into
// two scratch targets (atmo_march_half), a full-resolution composite upsamples them where the
// neighbouring samples belong to the same ray geometry and marches the rest exactly
// (atmo_composite), and the dual variant's mask takes T from the transmittance-LUT ratio
// (atmo_view_transmittance).

#import "common/camera.wgsl"
#import "common/constants.wgsl"
#import "atmo/ground_lighting.wgsl"

// Twin of `uni_atmo::feature::AtmoParamsGpu` (nine vec4; `atmo_uniforms_layout_match_wgsl`).
// Coefficients [1/m] at (680, 550, 440) nm.
struct AtmoParamsGpu {
    // x: bottom radius, y: top radius, z: Rayleigh scale height, w: Mie scale height [m].
    radii: vec4<f32>,
    // xyz: Rayleigh scattering; w: layer-2 tent centre altitude [m].
    rayleigh_scattering: vec4<f32>,
    // xyz: Mie scattering; w: layer-2 tent full width [m].
    mie_scattering: vec4<f32>,
    // xyz: Mie extinction; w: 0.
    mie_extinction: vec4<f32>,
    // xyz: Mie Cornette-Shanks g per channel; w: 0.
    mie_g: vec4<f32>,
    // xyz: layer-2 scattering at the tent's peak; w: 0.
    layer2_scattering: vec4<f32>,
    // xyz: layer-2 extinction at the tent's peak; w: 0.
    layer2_extinction: vec4<f32>,
    // xyz: layer-2 Cornette-Shanks g per channel; w: 0.
    layer2_g: vec4<f32>,
    // xyz: ground Lambert albedo (multiple-scattering LUT only); w: 0.
    ground_albedo: vec4<f32>,
}

// Twin of `uni_atmo::feature::AtmoDrawGpu` (seven vec4).
struct AtmoDrawGpu {
    // Columns of the root (ICRS) -> scaled body frame matrix: d_body = x * d.x + y * d.y + z * d.z.
    to_body_x: vec4<f32>,
    to_body_y: vec4<f32>,
    to_body_z: vec4<f32>,
    // xyz: camera - body centre in the scaled body frame [m]; w: camera height above the bottom.
    origin: vec4<f32>,
    // xyz: unit direction to the Sun in the scaled body frame; w: sun illuminance at the body [lx].
    sun: vec4<f32>,
    // NDC x0, y0, x1, y1 of the screen quad bounding the shell.
    rect: vec4<f32>,
    // x: exposure mix w, the body's (0 = star_pre, 1 = scene_pre; snapped to 1 when
    // 1 - w < 2^-14, uni_atmo::feature::atmo_exposure_mix); y: 1 = draw a fullscreen triangle;
    // z: depth surfaces lower than this above the ground [m] are the body's own ground; w: the
    // draw's index + 1 (it tags the draw's half-resolution samples).
    misc: vec4<f32>,
}

@group(1) @binding(0) var<uniform> atmo_draw: AtmoDrawGpu;
@group(1) @binding(1) var atmo_depth: texture_depth_2d;
// The half-resolution scratch targets (the composite's group 1 only): rgb the pre-exposed
// in-scattered light (atmo_light), a the segment length [km]; rgb the transmittance, a the
// draw's index + 1 (0 where nothing was marched: the clear value).
@group(1) @binding(2) var atmo_half_light: texture_2d<f32>;
@group(1) @binding(3) var atmo_half_trans: texture_2d<f32>;
// Multiple-scattering LUT (32 x 32, Rgba16Float): Psi_ms [1/sr per lux].
@group(2) @binding(4) var atmo_multiscatter_lut: texture_2d<f32>;
@group(2) @binding(5) var<uniform> atmo_params: AtmoParamsGpu;

// Twins of uni_atmo::lut::{MARCH_STEPS, MARCH_STEPS_LIMB, LUT_GROUND_OFFSET_M}: the LUT passes'
// and the probes' march.
const ATMO_MARCH_STEPS: f32 = 32.0;
const ATMO_MARCH_STEPS_LIMB: f32 = 48.0;
// Twins of uni_atmo::lut::{VIEW_MARCH_STEPS, VIEW_MARCH_STEPS_LIMB}: the per-pixel view march
// (plan fallback ladder, atmosphere step 2, task B9p: 32 -> 16 and 48 -> 24).
const ATMO_VIEW_MARCH_STEPS: f32 = 16.0;
const ATMO_VIEW_MARCH_STEPS_LIMB: f32 = 24.0;
const ATMO_LUT_GROUND_OFFSET_M: f32 = 1.0;

// What a ray picks up through the atmosphere.
struct AtmoSegment {
    // Inscattered radiance per lux of sun illuminance [cd/m2 per lx], per channel.
    inscatter: vec3<f32>,
    // Transmittance along the ray to the scene behind, per channel.
    transmittance: vec3<f32>,
}

// Scattering and extinction [1/m] at one point.
struct AtmoMedium {
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
    layer2: vec3<f32>,
    extinction: vec3<f32>,
}

// The medium at height h_m above the bottom (Hillaire 2020 section 4): exponential Rayleigh and
// Mie, a tent for layer 2. Twin of uni_atmo::lut::medium.
fn atmo_medium(h_m: f32) -> AtmoMedium {
    let h = max(h_m, 0.0);
    let d_r = exp(-h / atmo_params.radii.z);
    let d_m = exp(-h / atmo_params.radii.w);
    let half_width = 0.5 * atmo_params.mie_scattering.w;
    var d_2 = 0.0;
    if (half_width > 0.0) {
        d_2 = max(1.0 - abs(h - atmo_params.rayleigh_scattering.w) / half_width, 0.0);
    }
    let rayleigh = atmo_params.rayleigh_scattering.xyz * d_r;
    let mie = atmo_params.mie_scattering.xyz * d_m;
    let layer2 = atmo_params.layer2_scattering.xyz * d_2;
    let extinction = rayleigh + atmo_params.mie_extinction.xyz * d_m
        + atmo_params.layer2_extinction.xyz * d_2;
    return AtmoMedium(rayleigh, mie, layer2, extinction);
}

// Integral of exp(-ext s) ds over [0, dt] per channel (Hillaire 2015, "Physically Based and
// Unified Volumetric Rendering in Frostbite", slide 28), with a series below x = 1e-3. Twin of
// uni_atmo::lut::step_integral.
fn atmo_step_integral(ext: vec3<f32>, dt: f32) -> vec3<f32> {
    return atmo_step_integral_with(ext, dt, exp(-ext * dt));
}

// atmo_step_integral given the step's transmittance tr = exp(-ext dt), which the march needs
// anyway (one exp per step instead of two; task B9p).
fn atmo_step_integral_with(ext: vec3<f32>, dt: f32, tr: vec3<f32>) -> vec3<f32> {
    let x = ext * dt;
    let series = dt * (1.0 - 0.5 * x + x * x / 6.0);
    let exact = (1.0 - tr) / max(ext, vec3<f32>(1e-30));
    return select(exact, series, x < vec3<f32>(1e-3));
}

// Rayleigh phase function [1/sr]. Twin of uni_atmo::lut::rayleigh_phase.
fn atmo_rayleigh_phase(nu: f32) -> f32 {
    return 3.0 * (1.0 + nu * nu) / (16.0 * PI);
}

// Cornette-Shanks phase function [1/sr] per channel (Cornette & Shanks 1992), nu = cos of the
// angle between the view ray and the direction to the Sun. Twin of
// uni_atmo::lut::cornette_shanks.
fn atmo_cornette_shanks(g: vec3<f32>, nu: f32) -> vec3<f32> {
    let g2 = g * g;
    let k = 3.0 / (8.0 * PI) * (1.0 - g2) / (2.0 + g2);
    let denom = 1.0 + g2 - 2.0 * g * nu;
    return k * (1.0 + nu * nu) / (denom * sqrt(denom));
}

// The transmittance LUT at (r, mu), r clamped to the shell (no ground test).
fn atmo_lut_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let bottom = atmo_params.radii.x;
    let top = atmo_params.radii.y;
    let uv = atmo_transmittance_uv(bottom, top, clamp(r, bottom, top), mu);
    return textureSampleLevel(atmo_transmittance_lut, atmo_sampler, uv, 0.0).rgb;
}

// Transmittance toward the Sun from radius r: 0 in the planet's shadow (Hillaire 2020's planet
// shadow test), else the LUT. Twin of uni_atmo::lut::sun_transmittance_with.
fn atmo_sun_lut_transmittance(r: f32, mu_s: f32) -> vec3<f32> {
    if (atmo_ray_hits_ground(atmo_params.radii.x, r, mu_s)) {
        return vec3<f32>(0.0);
    }
    return atmo_lut_transmittance(r, mu_s);
}

// Psi_ms at (r, mu_s) from the multiple-scattering LUT.
fn atmo_lut_multiscatter(r: f32, mu_s: f32) -> vec3<f32> {
    let uv = atmo_rmus_uv(
        atmo_params.radii.x, atmo_params.radii.y, r, mu_s, ATMO_MULTISCATTER_SIZE
    );
    return textureSampleLevel(atmo_multiscatter_lut, atmo_sampler, uv, 0.0).rgb;
}

// (r, mu) of a transmittance-LUT texel coordinate (Bruneton 2017,
// GetRMuFromTransmittanceTextureUv). Twin of uni_atmo::transmittance_r_mu.
fn atmo_transmittance_r_mu(uv: vec2<f32>) -> vec2<f32> {
    let bottom = atmo_params.radii.x;
    let top = atmo_params.radii.y;
    let x = atmo_uv_to_unit(uv, ATMO_TRANSMITTANCE_SIZE);
    let h = sqrt((top - bottom) * (top + bottom));
    let rho = h * x.y;
    let r = sqrt(rho * rho + bottom * bottom);
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + x.x * (d_max - d_min);
    var mu = 1.0;
    if (d > 0.0) {
        mu = (h * h - rho * rho - d * d) / (2.0 * r * d);
    }
    return vec2<f32>(r, clamp(mu, -1.0, 1.0));
}

// (r, mu_s) a multiple-scattering or irradiance texel is evaluated at, r at least 1 m above the
// bottom. Twin of uni_atmo::lut::texel_rmus.
fn atmo_texel_rmus(uv: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let x = clamp(atmo_uv_to_unit(uv, size), vec2<f32>(0.0), vec2<f32>(1.0));
    let bottom = atmo_params.radii.x;
    let top = atmo_params.radii.y;
    let r = max(bottom + x.y * (top - bottom), bottom + ATMO_LUT_GROUND_OFFSET_M);
    return vec2<f32>(r, x.x * 2.0 - 1.0);
}

// Ray-sphere distances (near, far) for origin o and direction d (any length; t is in units of
// |d|), or (1, -1) (far < near) when the line misses. Perpendicular-distance discriminant for
// precision far from the sphere (Haines et al. 2019, Ray Tracing Gems ch. 7). Twin of
// uni_atmo::lut::ray_sphere.
fn atmo_ray_sphere(o: vec3<f32>, d: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(d, d);
    let b = dot(o, d);
    let perp = o - d * (b / a);
    let pl = length(perp);
    let disc = a * (radius - pl) * (radius + pl);
    if (disc < 0.0) {
        return vec2<f32>(1.0, -1.0);
    }
    let s = sqrt(disc);
    return vec2<f32>((-b - s) / a, (-b + s) / a);
}

// Steps for a segment of length len: `base`, rising linearly to `limb` as len goes from H to 2H,
// H = sqrt(top^2 - bottom^2). Twin of uni_atmo::lut::step_count_between.
fn atmo_step_count_between(len: f32, base: f32, limb: f32) -> f32 {
    let bottom = atmo_params.radii.x;
    let top = atmo_params.radii.y;
    let h = sqrt((top - bottom) * (top + bottom));
    return base + (limb - base) * clamp((len - h) / h, 0.0, 1.0);
}

// The LUT passes' and probes' step count: 32, rising to 48. Twin of uni_atmo::lut::step_count.
fn atmo_step_count(len: f32) -> f32 {
    return atmo_step_count_between(len, ATMO_MARCH_STEPS, ATMO_MARCH_STEPS_LIMB);
}

// The view march's step count: 16, rising to 24 (ladder step 2). Twin of
// uni_atmo::lut::view_step_count.
fn atmo_view_step_count(len: f32) -> f32 {
    return atmo_step_count_between(len, ATMO_VIEW_MARCH_STEPS, ATMO_VIEW_MARCH_STEPS_LIMB);
}

// A half-segment's step boundary k, measured from its dense end: len * min((k/c)^2, 1).
fn atmo_half_step(len: f32, c: f32, k: f32) -> f32 {
    let x = k / c;
    return len * min(x * x, 1.0);
}

// The phase values of one ray (the Sun is at infinity, so they are constant along it).
struct AtmoPhases {
    rayleigh: f32,
    mie: vec3<f32>,
    layer2: vec3<f32>,
}

// One march step with its sample at `pos`, length dt. With `scatter` false only the
// transmittance is accumulated (the mask pass).
fn atmo_march_step(
    acc: ptr<function, AtmoSegment>,
    pos: vec3<f32>,
    sun: vec3<f32>,
    dt: f32,
    phases: AtmoPhases,
    scatter: bool,
) {
    let r = length(pos);
    let m = atmo_medium(r - atmo_params.radii.x);
    let tr = exp(-m.extinction * dt);
    if (scatter) {
        let mu_s = dot(pos, sun) / r;
        let t_sun = atmo_sun_lut_transmittance(r, mu_s);
        let psi = atmo_lut_multiscatter(r, mu_s);
        let single = m.rayleigh * phases.rayleigh + m.mie * phases.mie + m.layer2 * phases.layer2;
        let s = t_sun * single + psi * (m.rayleigh + m.mie + m.layer2);
        (*acc).inscatter += (*acc).transmittance * s * atmo_step_integral_with(m.extinction, dt, tr);
    }
    (*acc).transmittance *= tr;
}

// Marches from o (body-centred position) along d over [t0, t1] with the Sun in the unit
// direction sun, nu = cos(view, Sun). Single scattering through the transmittance LUT (zero in
// the planet's shadow) with Rayleigh and per-channel Cornette-Shanks phases, plus Psi_ms x
// sigma_s (Hillaire 2020 section 5.5), integrated per step. The segment is split at the closest
// approach to the centre and each half's steps are spaced quadratically from that densest
// point; each step is sampled at its midpoint. atmo_step_count steps (the LUT passes and the
// probes). Twin of uni_atmo::lut::march.
fn atmo_march(o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32, sun: vec3<f32>, nu: f32, scatter: bool) -> AtmoSegment {
    return atmo_march_n(o, d, t0, t1, sun, nu, scatter, atmo_step_count(t1 - t0));
}

// atmo_march with n steps. Twin of uni_atmo::lut::march_steps.
fn atmo_march_n(
    o: vec3<f32>,
    d: vec3<f32>,
    t0: f32,
    t1: f32,
    sun: vec3<f32>,
    nu: f32,
    scatter: bool,
    n: f32,
) -> AtmoSegment {
    var acc = AtmoSegment(vec3<f32>(0.0), vec3<f32>(1.0));
    let len = t1 - t0;
    if (!(len > 0.0)) {
        return acc;
    }
    let phases = AtmoPhases(
        atmo_rayleigh_phase(nu),
        atmo_cornette_shanks(atmo_params.mie_g.xyz, nu),
        atmo_cornette_shanks(atmo_params.layer2_g.xyz, nu),
    );
    let tc = clamp(-dot(o, d) / dot(d, d), t0, t1);
    let len_a = tc - t0;
    let len_b = t1 - tc;
    let c_a = n * len_a / len;
    let c_b = n * len_b / len;
    // Half A, camera side: from t0 toward tc, the dense end at tc.
    if (len_a > 0.0) {
        var k = ceil(c_a);
        loop {
            if (k < 1.0) {
                break;
            }
            let q_far = atmo_half_step(len_a, c_a, k);
            let q_near = atmo_half_step(len_a, c_a, k - 1.0);
            let dt = q_far - q_near;
            if (dt > 0.0) {
                atmo_march_step(&acc, o + d * (tc - q_near - 0.5 * dt), sun, dt, phases, scatter);
            }
            k -= 1.0;
        }
    }
    // Half B: from tc outward, the dense end at tc.
    if (len_b > 0.0) {
        let steps = ceil(c_b);
        var k = 0.0;
        loop {
            if (k >= steps) {
                break;
            }
            let q0 = atmo_half_step(len_b, c_b, k);
            let q1 = atmo_half_step(len_b, c_b, k + 1.0);
            let dt = q1 - q0;
            if (dt > 0.0) {
                atmo_march_step(&acc, o + d * (tc + q0 + 0.5 * dt), sun, dt, phases, scatter);
            }
            k += 1.0;
        }
    }
    return acc;
}

// A pixel's ray through this draw's shell, re-centred at its entry point so the march adds small
// offsets to a nearby origin (f32 precision from far away).
struct AtmoViewRay {
    // Entry point (or the camera, inside the shell) in the scaled body frame [m].
    o: vec3<f32>,
    // Direction in the scaled body frame (|d| ~ 1; t is a true distance).
    d: vec3<f32>,
    // Segment length [m].
    len: f32,
    // cos(view, Sun).
    nu: f32,
    valid: bool,
}

// The segment of pixel `pix`'s ray inside the shell: from the camera or the shell entry to the
// shell exit, the ground, or the scene depth, whichever comes first. The scene distance comes
// from the reversed-Z infinite depth (view z = near / depth, divided by the ray's cosine to the
// view axis). A depth surface lower than atmo_draw.misc.z above the ground is the body's own
// surface drawn on a proxy mesh slightly outside it; there the ground sphere ends the ray
// instead, so the air down to the true ground is marched.
fn atmo_view_ray(pix: vec2<f32>) -> AtmoViewRay {
    var ray: AtmoViewRay;
    ray.valid = false;
    let dir = pixel_dir(pix);
    let d = atmo_draw.to_body_x.xyz * dir.x + atmo_draw.to_body_y.xyz * dir.y
        + atmo_draw.to_body_z.xyz * dir.z;
    let o = atmo_draw.origin.xyz;
    let bottom = atmo_params.radii.x;
    let top = atmo_params.radii.y;
    let shell = atmo_ray_sphere(o, d, top);
    if (shell.x > shell.y || shell.y <= 0.0) {
        return ray;
    }
    let t0 = max(shell.x, 0.0);
    var t1 = shell.y;
    let ground = atmo_ray_sphere(o, d, bottom);
    if (ground.x <= ground.y && ground.y > 0.0) {
        if (ground.x > 0.0) {
            t1 = min(t1, ground.x);
        } else if (dot(o, d) < 0.0) {
            // Camera on the ground (f32 rounding put it inside): a downward ray sees ground.
            t1 = t0;
        }
    }
    let depth = textureLoad(atmo_depth, vec2<i32>(pix), 0);
    if (depth > 0.0) {
        let z_per_t = -(camera.view_rot * vec4<f32>(dir, 0.0)).z;
        let t_scene = camera.proj[3].z / (depth * max(z_per_t, 1e-6));
        let h_scene = length(o + d * t_scene) - bottom;
        if (h_scene > atmo_draw.misc.z) {
            t1 = min(t1, t_scene);
        }
    }
    if (!(t1 > t0)) {
        return ray;
    }
    ray.o = o + d * t0;
    ray.d = d;
    ray.len = t1 - t0;
    ray.nu = dot(normalize(d), atmo_draw.sun.xyz);
    ray.valid = true;
    return ray;
}

// Transmittance along the segment from o over len (o + d·t, the ray's own parameterisation)
// from the transmittance LUT, without a march (Bruneton 2017, GetTransmittance): T(r0, mu0) /
// T(r1, mu1) for a ray that misses the ground, T(r1, -mu1) / T(r0, -mu0) for one that hits it
// (both then read toward the top), clamped to 1. Plan fallback ladder, atmosphere step 1 (task
// B9p): the mask pass's `g`. With the camera outside the shell (every Solar System bookmark) one
// of the two factors is 1 and the result is one LUT read; from inside, near-horizon rays divide
// two small numbers and lose precision. CPU twin: uni_atmo::feature::segment_transmittance_lut.
fn atmo_segment_lut_transmittance(o: vec3<f32>, d: vec3<f32>, len: f32) -> vec3<f32> {
    let bottom = atmo_params.radii.x;
    let dn = normalize(d);
    let r0 = length(o);
    let mu0 = dot(o, dn) / r0;
    let e = o + d * len;
    let r1 = length(e);
    let mu1 = dot(e, dn) / r1;
    var num: vec3<f32>;
    var den: vec3<f32>;
    if (atmo_ray_hits_ground(bottom, r0, mu0)) {
        num = atmo_lut_transmittance(r1, -mu1);
        den = atmo_lut_transmittance(r0, -mu0);
    } else {
        num = atmo_lut_transmittance(r0, mu0);
        den = atmo_lut_transmittance(r1, mu1);
    }
    return min(num / max(den, vec3<f32>(1e-30)), vec3<f32>(1.0));
}

// The transmittance of pixel `pix`'s ray (the fragment position) from the LUT ratio (the mask
// pass; ladder step 1). Discards like atmo_composite.
fn atmo_view_transmittance(pix: vec2<f32>) -> vec3<f32> {
    let ray = atmo_view_ray(floor(pix));
    if (!ray.valid) {
        discard;
    }
    return atmo_segment_lut_transmittance(ray.o, ray.d, ray.len);
}

// Inscattered light per lux -> pre-exposed hdr: x sun illuminance x the Sun's colour (light 0,
// the brightest, is the Sun; white if there is none) x the draw's exposure, the body's own
// pre_mix(star_pre(), scene_pre(), w) (the log-space blend of common/exposure.wgsl, task B11;
// scene_pre from 4 px on; lead ruling B6 I1), clamped to HDR_MAX.
fn atmo_light(inscatter_per_lux: vec3<f32>) -> vec3<f32> {
    var sun_rgb = vec3<f32>(1.0);
    if (lights.info.x >= 1u) {
        sun_rgb = lights.lights[0].color_intensity.rgb;
    }
    let pre = pre_mix(star_pre(), scene_pre(), atmo_draw.misc.x);
    return min(inscatter_per_lux * atmo_draw.sun.w * sun_rgb * pre, vec3<f32>(HDR_MAX));
}

// The star share every fragment of this draw stores in lens_mask.a, blended with Max (M1 frame
// contract, lead ruling B6 I1): 1 - w. Each fragment the march does not discard adds in-scatter
// at pre_mix(star_pre(), scene_pre(), w), so it covers its pixel for this purpose. The meter
// (post/histogram.wgsl, meter_pre) reads the pixel's pre back as
// pre_mix(scene_pre(), star_pre(), share). 0 from 4 px on (w = 1). CPU twin:
// uni_atmo::feature::atmo_star_share.
fn atmo_star_share() -> f32 {
    return 1.0 - atmo_draw.misc.x;
}

// ---- Half resolution (plan fallback ladder, atmosphere step 3; task B9p) ------------------------

// Largest relative difference between a half-resolution sample's segment length and the pixel's
// own for the sample to be used. A ray that grazes the ground and one that just misses it differ
// by a factor of about 2, a ray stopped by a body in front by more; inside one geometry the length
// changes by under 10 % per sample except in the outermost few kilometres of the shell, where the
// light is faint and the pixel is marched instead. Twin of uni_atmo::feature::UPSAMPLE_LEN_TOL.
const ATMO_UPSAMPLE_LEN_TOL: f32 = 0.1;
// Segment lengths are stored in km (f16 holds up to 65504 km at 1e-3 relative).
const ATMO_KM_PER_M: f32 = 1e-3;

// The march pass's outputs (two scratch targets, no blending).
struct AtmoHalfOut {
    @location(0) light: vec4<f32>,
    @location(1) trans: vec4<f32>,
}

// The march of half-resolution texel `half_pos` (the fragment position): the exact ray of
// full-resolution pixel 2 x the texel index (pixel_dir and the depth there), so even pixels get
// their own march back. Discards where that ray misses the shell (the clear value stays).
fn atmo_march_half(half_pos: vec2<f32>) -> AtmoHalfOut {
    let ray = atmo_view_ray(2.0 * floor(half_pos));
    if (!ray.valid) {
        discard;
    }
    let s = atmo_march_n(
        ray.o, ray.d, 0.0, ray.len, atmo_draw.sun.xyz, ray.nu, true, atmo_view_step_count(ray.len)
    );
    return AtmoHalfOut(
        vec4<f32>(atmo_light(s.inscatter), ray.len * ATMO_KM_PER_M),
        vec4<f32>(s.transmittance, atmo_draw.misc.w),
    );
}

// What the composite applies to a pixel: hdr = hdr x transmittance + light.
struct AtmoComposite {
    // Pre-exposed in-scattered light (atmo_light).
    light: vec3<f32>,
    transmittance: vec3<f32>,
}

// Pixel `pos` (the fragment position) from the half-resolution samples: the bilinear neighbours
// (pixel p lies at p/2 in samples: even pixels on a sample, odd ones halfway), keeping only
// samples of this draw whose segment length is within ATMO_UPSAMPLE_LEN_TOL of the pixel's own
// (the depth-aware upsample, generalised to the ray's end: a body, the ground or the shell);
// with none kept, the pixel's own exact march (CPU twin: segment_radiance). Discards pixels whose
// ray misses the shell. CPU twin of the weights: uni_atmo::feature::upsample_weights.
fn atmo_composite(pos: vec2<f32>) -> AtmoComposite {
    let pix = floor(pos);
    let ray = atmo_view_ray(pix);
    if (!ray.valid) {
        discard;
    }
    let len_km = ray.len * ATMO_KM_PER_M;
    let ip = vec2<i32>(pix);
    let base = ip / 2;
    let odd = vec2<f32>(ip % 2);
    let last = vec2<i32>(textureDimensions(atmo_half_light)) - vec2<i32>(1);
    var light = vec3<f32>(0.0);
    var trans = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1u);
        let wx = select(1.0 - 0.5 * odd.x, 0.5 * odd.x, o.x == 1);
        let wy = select(1.0 - 0.5 * odd.y, 0.5 * odd.y, o.y == 1);
        let w = wx * wy;
        if (w > 0.0) {
            let q = min(base + o, last);
            let l = textureLoad(atmo_half_light, q, 0);
            let t = textureLoad(atmo_half_trans, q, 0);
            if (t.a == atmo_draw.misc.w && abs(l.a - len_km) <= ATMO_UPSAMPLE_LEN_TOL * len_km) {
                light += w * l.rgb;
                trans += w * t.rgb;
                wsum += w;
            }
        }
    }
    if (wsum > 0.0) {
        return AtmoComposite(light / wsum, trans / wsum);
    }
    let s = atmo_march_n(
        ray.o, ray.d, 0.0, ray.len, atmo_draw.sun.xyz, ray.nu, true, atmo_view_step_count(ray.len)
    );
    return AtmoComposite(atmo_light(s.inscatter), s.transmittance);
}

// The screen quad bounding the shell (rect, 6 vertices), or a fullscreen triangle (vertices
// 0..2) when misc.y is set (the camera inside or near the shell).
@vertex
fn vs_atmosphere(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    if (atmo_draw.misc.y > 0.5) {
        let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
        return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    }
    // Two triangles: (0,0) (1,0) (0,1) and (0,1) (1,0) (1,1).
    let x = select(0.0, 1.0, i == 1u || i == 4u || i == 5u);
    let y = select(0.0, 1.0, i == 2u || i == 3u || i == 5u);
    let r = atmo_draw.rect;
    return vec4<f32>(mix(r.xy, r.zw, vec2<f32>(x, y)), 0.0, 1.0);
}
