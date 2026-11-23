// atmo/multiscatter_lut.wgsl: Hillaire 2020's multiple-scattering LUT pass (32 x 32,
// Rgba16Float), computed once per atmosphere at start-up.
//
// B0 stub: valid, writes nothing. B3 writes the pass and its bindings.

@compute @workgroup_size(8, 8, 1)
fn cs_multiscatter(@builtin(global_invocation_id) gid: vec3<u32>) {
}
