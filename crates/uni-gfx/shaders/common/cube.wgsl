// common/cube.wgsl: the cube-face convention of the environment cubemap. Twin of
// `uni_gfx::env::cube_dir` (crates/uni-gfx/src/env.rs); `universe --selftest` checks the two
// agree on the GPU (probe `cube_dir`).

// Unit direction through the point `uv` of cube face `face`, in the WebGPU convention: layers
// [+X, −X, +Y, −Y, +Z, −Z] (WebGPU, GPUTextureViewDimension "cube"), uv in [0, 1]² with (0, 0)
// the top-left corner of the face image, +U right and +V down. It inverts the face selection a
// cube sample uses (Vulkan specification, "Cube Map Face Selection"), so a texel written at
// cube_dir(face, uv) is what textureSample(cube, dir) returns for that dir. Faces above 5 are
// treated as −Z.
fn cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let a = 2.0 * uv.x - 1.0;
    let b = 2.0 * uv.y - 1.0;
    var d: vec3<f32>;
    switch face {
        case 0u: { d = vec3<f32>(1.0, -b, -a); }
        case 1u: { d = vec3<f32>(-1.0, -b, a); }
        case 2u: { d = vec3<f32>(a, 1.0, b); }
        case 3u: { d = vec3<f32>(a, -1.0, -b); }
        case 4u: { d = vec3<f32>(a, -b, 1.0); }
        default: { d = vec3<f32>(-a, -b, -1.0); }
    }
    return normalize(d);
}
