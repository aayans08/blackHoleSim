// atmo/ground_lighting.wgsl: the ground-lighting contract between uni-atmo (B3) and the bodies
// pass (uni-planet B1), which imports this file and binds `uni_atmo::GroundLighting` at group 2.
//
// B0 stub: valid and neutral. Every group the stub builds has has_atmosphere = 0, so the Sun's
// light arrives unattenuated and there is no skylight. B3 samples the LUTs below for bodies with
// an atmosphere (Hillaire 2020, CGF 39(4); Bruneton 2017's (r, mu) parameterisation) and keeps the
// struct, the bindings and the two signatures.

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

// Fraction of the Sun's light reaching radius r_m [m] with the Sun at zenith cosine mu_s, per
// channel: 1 without an atmosphere; 0 below the geometric horizon.
fn atmo_sun_transmittance(r_m: f32, mu_s: f32) -> vec3<f32> {
    if (atmo_ground.radii.z < 0.5) {
        return vec3<f32>(1.0);
    }
    // B3: the transmittance LUT at (r_m, mu_s), 0 where the ray to the Sun hits the ground.
    return vec3<f32>(1.0);
}

// Skylight irradiance on the ground at radius r_m per unit sun illuminance, with the Sun at
// zenith cosine mu_s, per channel: 0 without an atmosphere.
fn atmo_sky_irradiance(r_m: f32, mu_s: f32) -> vec3<f32> {
    if (atmo_ground.radii.z < 0.5) {
        return vec3<f32>(0.0);
    }
    // B3: the irradiance LUT at (r_m, mu_s).
    return vec3<f32>(0.0);
}
