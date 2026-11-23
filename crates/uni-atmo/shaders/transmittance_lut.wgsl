// atmo/transmittance_lut.wgsl: the transmittance LUT pass (256 x 64, Rgba16Float), computed once
// per atmosphere at start-up (Hillaire 2020; Bruneton 2017's (r, mu) parameterisation).
//
// B0 stub: valid, writes nothing. B3 writes the pass and its bindings.

@compute @workgroup_size(8, 8, 1)
fn cs_transmittance(@builtin(global_invocation_id) gid: vec3<u32>) {
}
