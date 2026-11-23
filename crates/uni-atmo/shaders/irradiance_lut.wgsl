// atmo/irradiance_lut.wgsl: the ground-irradiance LUT pass (64 x 16, Rgba16Float), computed once
// per atmosphere at start-up after the transmittance and multiple-scattering LUTs;
// atmo_sky_irradiance samples it. E(r, mu_s) = integral of L(w) cos(theta) over the upper
// hemisphere, L from the view march (single + multiple scattering), by a cosine-weighted
// 16 x 16 quadrature: E = pi * mean(L). CPU twin: uni_atmo::sky_irradiance.

#import "atmo/atmosphere_common.wgsl"

@group(3) @binding(0) var atmo_lut_out: texture_storage_2d<rgba16float, write>;

// Twin of uni_atmo::lut::IRRADIANCE_DIRS.
const ATMO_IRRADIANCE_DIRS: vec2<u32> = vec2<u32>(16u, 16u);

@compute @workgroup_size(8, 8, 1)
fn cs_irradiance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(ATMO_IRRADIANCE_SIZE);
    if (any(gid.xy >= size)) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / ATMO_IRRADIANCE_SIZE;
    let rmus = atmo_texel_rmus(uv, ATMO_IRRADIANCE_SIZE);
    let r = rmus.x;
    let mu_s = rmus.y;
    let o = vec3<f32>(0.0, 0.0, r);
    let sun = vec3<f32>(sqrt(max(1.0 - mu_s * mu_s, 0.0)), 0.0, mu_s);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < ATMO_IRRADIANCE_DIRS.x; i++) {
        let u = (f32(i) + 0.5) / f32(ATMO_IRRADIANCE_DIRS.x);
        let sin_t = sqrt(u);
        let cos_t = sqrt(1.0 - u);
        let t_top = atmo_distance_to_top(atmo_params.radii.y, r, cos_t);
        for (var j = 0u; j < ATMO_IRRADIANCE_DIRS.y; j++) {
            let phi = TAU * (f32(j) + 0.5) / f32(ATMO_IRRADIANCE_DIRS.y);
            let d = vec3<f32>(sin_t * cos(phi), sin_t * sin(phi), cos_t);
            sum += atmo_march(o, d, 0.0, t_top, sun, dot(d, sun), true).inscatter;
        }
    }
    let e = sum * (PI / f32(ATMO_IRRADIANCE_DIRS.x * ATMO_IRRADIANCE_DIRS.y));
    textureStore(atmo_lut_out, vec2<i32>(gid.xy), vec4<f32>(e, 1.0));
}
