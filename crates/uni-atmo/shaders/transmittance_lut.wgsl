// atmo/transmittance_lut.wgsl: the transmittance LUT pass (256 x 64, Rgba16Float), computed once
// per atmosphere at start-up (Hillaire 2020; Bruneton 2017's (r, mu) parameterisation and
// ComputeOpticalLengthToTopAtmosphereBoundary: trapezoid rule, 500 intervals). Texel (i, j) holds
// exp(-optical depth) from the (r, mu) of its centre to the top, ignoring the ground. CPU twin:
// uni_atmo::lut::transmittance_to_top (via uni_atmo::transmittance).

#import "atmo/atmosphere_common.wgsl"

@group(3) @binding(0) var atmo_lut_out: texture_storage_2d<rgba16float, write>;

// Twin of uni_atmo::lut::TRANSMITTANCE_STEPS.
const ATMO_TRANSMITTANCE_STEPS: u32 = 500u;

@compute @workgroup_size(8, 8, 1)
fn cs_transmittance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(ATMO_TRANSMITTANCE_SIZE);
    if (any(gid.xy >= size)) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / ATMO_TRANSMITTANCE_SIZE;
    let rmu = atmo_transmittance_r_mu(uv);
    let r = rmu.x;
    let mu = rmu.y;
    let bottom = atmo_params.radii.x;
    let d = atmo_distance_to_top(atmo_params.radii.y, r, mu);
    let dx = d / f32(ATMO_TRANSMITTANCE_STEPS);
    var tau = vec3<f32>(0.0);
    for (var i = 0u; i <= ATMO_TRANSMITTANCE_STEPS; i++) {
        let di = f32(i) * dx;
        let ri = sqrt(di * di + 2.0 * r * mu * di + r * r);
        // Height as (ri^2 - bottom^2) / (ri + bottom): no cancellation near the ground in f32.
        let h = (di * (di + 2.0 * r * mu) + (r - bottom) * (r + bottom)) / (ri + bottom);
        var w = 1.0;
        if (i == 0u || i == ATMO_TRANSMITTANCE_STEPS) {
            w = 0.5;
        }
        tau += atmo_medium(h).extinction * (w * dx);
    }
    textureStore(atmo_lut_out, vec2<i32>(gid.xy), vec4<f32>(exp(-tau), 1.0));
}
