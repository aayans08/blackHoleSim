// atmo/irradiance_lut.wgsl: the ground-irradiance LUT pass (64 x 16, Rgba16Float), computed once
// per atmosphere at start-up; atmo_sky_irradiance samples it.
//
// B0 stub: valid, writes nothing. B3 writes the pass and its bindings.

@compute @workgroup_size(8, 8, 1)
fn cs_irradiance(@builtin(global_invocation_id) gid: vec3<u32>) {
}
