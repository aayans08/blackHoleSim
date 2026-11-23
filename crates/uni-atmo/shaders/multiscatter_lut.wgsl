// atmo/multiscatter_lut.wgsl: Hillaire 2020's multiple-scattering LUT pass (32 x 32, Rgba16Float),
// computed once per atmosphere at start-up, after the transmittance LUT (section 5.5; eq. 5-10 as the sample code numbers them).
// For each texel's (r, mu_s), 8 x 8 directions on the sphere are marched in 20 midpoint steps
// with an isotropic phase: L2 (second-order light) and f_ms (the transfer fraction), plus the
// sunlit ground's Lambert bounce; Psi_ms = L2 / (1 - f_ms). CPU twin:
// uni_atmo::lut::multiscatter_with (uni_atmo::multiscatter).

#import "atmo/atmosphere_common.wgsl"

@group(3) @binding(0) var atmo_lut_out: texture_storage_2d<rgba16float, write>;

// Twins of uni_atmo::lut::{MULTISCATTER_SQRT_DIRS, MULTISCATTER_STEPS, MAX_F_MS}.
const ATMO_MS_SQRT_DIRS: u32 = 8u;
const ATMO_MS_STEPS: u32 = 20u;
const ATMO_MAX_F_MS: f32 = 0.99;

struct AtmoMsRay {
    l: vec3<f32>,
    f: vec3<f32>,
}

// One direction from o (inside the shell). Twin of uni_atmo::lut::multiscatter_ray.
fn atmo_ms_ray(o: vec3<f32>, d: vec3<f32>, sun: vec3<f32>) -> AtmoMsRay {
    let bottom = atmo_params.radii.x;
    let r = length(o);
    let mu = dot(o, d) / r;
    let hits = atmo_ray_hits_ground(bottom, r, mu);
    var t_max = atmo_distance_to_top(atmo_params.radii.y, r, mu);
    if (hits) {
        let disc = (bottom - r) * (bottom + r) + r * r * mu * mu;
        t_max = max(-r * mu - sqrt(max(disc, 0.0)), 0.0);
    }
    let dt = t_max / f32(ATMO_MS_STEPS);
    let iso = 1.0 / (4.0 * PI);
    var throughput = vec3<f32>(1.0);
    var out = AtmoMsRay(vec3<f32>(0.0), vec3<f32>(0.0));
    for (var s = 0u; s < ATMO_MS_STEPS; s++) {
        let pos = o + d * ((f32(s) + 0.5) * dt);
        let rr = length(pos);
        let m = atmo_medium(rr - bottom);
        let t_sun = atmo_sun_lut_transmittance(rr, dot(pos, sun) / rr);
        let scat = m.rayleigh + m.mie + m.layer2;
        let integ = atmo_step_integral(m.extinction, dt);
        out.l += throughput * t_sun * scat * iso * integ;
        out.f += throughput * scat * integ;
        throughput *= exp(-m.extinction * dt);
    }
    if (hits) {
        let pos = o + d * t_max;
        let mu_s = dot(pos, sun) / length(pos);
        let t_sun = atmo_lut_transmittance(bottom, mu_s);
        out.l += t_sun * throughput * max(mu_s, 0.0) * atmo_params.ground_albedo.xyz / PI;
    }
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn cs_multiscatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(ATMO_MULTISCATTER_SIZE);
    if (any(gid.xy >= size)) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / ATMO_MULTISCATTER_SIZE;
    let rmus = atmo_texel_rmus(uv, ATMO_MULTISCATTER_SIZE);
    let mu_s = rmus.y;
    let o = vec3<f32>(0.0, 0.0, rmus.x);
    let sun = vec3<f32>(sqrt(max(1.0 - mu_s * mu_s, 0.0)), 0.0, mu_s);
    var l2 = vec3<f32>(0.0);
    var f_ms = vec3<f32>(0.0);
    let n = f32(ATMO_MS_SQRT_DIRS);
    for (var a = 0u; a < ATMO_MS_SQRT_DIRS; a++) {
        for (var b = 0u; b < ATMO_MS_SQRT_DIRS; b++) {
            let theta = TAU * (f32(a) + 0.5) / n;
            let phi = acos(1.0 - 2.0 * (f32(b) + 0.5) / n);
            let d = vec3<f32>(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
            let ray = atmo_ms_ray(o, d, sun);
            l2 += ray.l;
            f_ms += ray.f;
        }
    }
    let count = n * n;
    let psi = (l2 / count) / (vec3<f32>(1.0) - min(f_ms / count, vec3<f32>(ATMO_MAX_F_MS)));
    textureStore(atmo_lut_out, vec2<i32>(gid.xy), vec4<f32>(psi, 1.0));
}
