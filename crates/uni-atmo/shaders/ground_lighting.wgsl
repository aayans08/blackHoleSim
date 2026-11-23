// atmo/ground_lighting.wgsl: the ground-lighting contract between uni-atmo (B3) and the bodies
// pass (uni-planet B1), which imports this file and binds `uni_atmo::GroundLighting` at group 2.
//
// Bodies without an atmosphere get the neutral group (has_atmosphere = 0): the Sun's light
// arrives unattenuated and there is no skylight. Bodies with one get its transmittance and
// ground-irradiance LUTs (Hillaire 2020, CGF 39(4); Bruneton 2017's (r, mu) parameterisation).
// CPU twin of both functions: `uni_atmo::ground::ground_lighting`.
//
// r_m for a point on a body is the atmosphere's bottom radius (atmo_ground.radii.x) plus the
// point's height above the body's reference surface: the atmosphere follows the body's
// ellipsoid, so every point on the ground passes the bottom radius. mu_s is the cosine of the
// Sun's zenith angle about the surface normal.
//
// The pure helpers below (no bindings) are shared with atmo/atmosphere_common.wgsl, which also
// reuses this file's transmittance-LUT and sampler bindings for the view march.

// Twin of `uni_atmo::AtmoGround` (one vec4; `atmo_ground_layout_matches_wgsl`).
struct AtmoGround {
    // x: bottom radius [m], y: top radius [m], z: has_atmosphere (0 or 1), w: 0.
    radii: vec4<f32>,
}

@group(2) @binding(0) var<uniform> atmo_ground: AtmoGround;
// Transmittance LUT (256 x 64, Rgba16Float).
@group(2) @binding(1) var atmo_transmittance_lut: texture_2d<f32>;
// Ground-irradiance LUT (64 x 16, Rgba16Float).
@group(2) @binding(2) var atmo_irradiance_lut: texture_2d<f32>;
// Linear, clamp to edge.
@group(2) @binding(3) var atmo_sampler: sampler;

// LUT sizes (twins of uni_atmo::lut::{TRANSMITTANCE_SIZE, MULTISCATTER_SIZE, IRRADIANCE_SIZE}).
const ATMO_TRANSMITTANCE_SIZE: vec2<f32> = vec2<f32>(256.0, 64.0);
const ATMO_MULTISCATTER_SIZE: vec2<f32> = vec2<f32>(32.0, 32.0);
const ATMO_IRRADIANCE_SIZE: vec2<f32> = vec2<f32>(64.0, 16.0);

// [0, 1] -> texture coordinate, mapping 0 and 1 to the first and last texel centres
// (Bruneton 2017, GetTextureCoordFromUnitRange), and back.
fn atmo_unit_to_uv(x: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return 0.5 / size + x * (1.0 - 1.0 / size);
}

fn atmo_uv_to_unit(u: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return (u - 0.5 / size) / (1.0 - 1.0 / size);
}

// True when the ray from radius r with zenith cosine mu hits the sphere of radius `bottom`
// (Bruneton 2017, RayIntersectsGround). Twin of uni_atmo::lut::ray_hits_ground.
fn atmo_ray_hits_ground(bottom: f32, r: f32, mu: f32) -> bool {
    return mu < 0.0 && (bottom - r) * (bottom + r) + r * r * mu * mu >= 0.0;
}

// Distance from radius r along zenith cosine mu to the sphere of radius `top` (Bruneton 2017,
// DistanceToTopAtmosphereBoundary). Twin of uni_atmo::lut::distance_to_top.
fn atmo_distance_to_top(top: f32, r: f32, mu: f32) -> f32 {
    let disc = (top - r) * (top + r) + r * r * mu * mu;
    return max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
}

// Transmittance-LUT coordinates of (r, mu) (Bruneton 2017, GetTransmittanceTextureUvFromRMu).
// Twin of uni_atmo::transmittance_uv.
fn atmo_transmittance_uv(bottom: f32, top: f32, r: f32, mu: f32) -> vec2<f32> {
    let h = sqrt((top - bottom) * (top + bottom));
    let rho = sqrt(max((r - bottom) * (r + bottom), 0.0));
    let d = atmo_distance_to_top(top, r, mu);
    let d_min = top - r;
    let d_max = rho + h;
    var x_mu = 0.0;
    if (d_max > d_min) {
        x_mu = (d - d_min) / (d_max - d_min);
    }
    return atmo_unit_to_uv(vec2<f32>(x_mu, rho / h), ATMO_TRANSMITTANCE_SIZE);
}

// Coordinates of (r, mu_s) in a multiple-scattering or irradiance LUT of `size`: x = mu_s/2 + 1/2,
// y = (r - bottom)/(top - bottom). Twin of uni_atmo::lut::rmus_uv.
fn atmo_rmus_uv(bottom: f32, top: f32, r: f32, mu_s: f32, size: vec2<f32>) -> vec2<f32> {
    let x = clamp(mu_s * 0.5 + 0.5, 0.0, 1.0);
    let y = clamp((r - bottom) / (top - bottom), 0.0, 1.0);
    return atmo_unit_to_uv(vec2<f32>(x, y), size);
}

// Fraction of the Sun's light reaching radius r_m [m] with the Sun at zenith cosine mu_s, per
// channel: 1 without an atmosphere; 0 below the geometric horizon. r is clamped to the shell.
fn atmo_sun_transmittance(r_m: f32, mu_s: f32) -> vec3<f32> {
    if (atmo_ground.radii.z < 0.5) {
        return vec3<f32>(1.0);
    }
    let bottom = atmo_ground.radii.x;
    let top = atmo_ground.radii.y;
    let r = clamp(r_m, bottom, top);
    let mu = clamp(mu_s, -1.0, 1.0);
    if (atmo_ray_hits_ground(bottom, r, mu)) {
        return vec3<f32>(0.0);
    }
    let uv = atmo_transmittance_uv(bottom, top, r, mu);
    return textureSampleLevel(atmo_transmittance_lut, atmo_sampler, uv, 0.0).rgb;
}

// Skylight irradiance on the ground at radius r_m per unit sun illuminance, with the Sun at
// zenith cosine mu_s, per channel: 0 without an atmosphere. r is clamped to the shell.
fn atmo_sky_irradiance(r_m: f32, mu_s: f32) -> vec3<f32> {
    if (atmo_ground.radii.z < 0.5) {
        return vec3<f32>(0.0);
    }
    let bottom = atmo_ground.radii.x;
    let top = atmo_ground.radii.y;
    let r = clamp(r_m, bottom, top);
    let uv = atmo_rmus_uv(bottom, top, r, clamp(mu_s, -1.0, 1.0), ATMO_IRRADIANCE_SIZE);
    return textureSampleLevel(atmo_irradiance_lut, atmo_sampler, uv, 0.0).rgb;
}
