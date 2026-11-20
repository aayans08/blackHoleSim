// common/hash.wgsl: integer hashes, the GPU twins of `uni_core::hash`.
//
// pcg3d: Jarzynski & Olano 2020, "Hash Functions for GPU Rendering", JCGT 9(3), §2.
// It must stay bit-identical to `uni_core::pcg3d`; `universe --selftest` checks 4096
// inputs on the GPU against the CPU. Golden values: docs/research/m0-golden-values.py.
// All arithmetic is u32 and wraps modulo 2^32 (WGSL integer overflow is defined to wrap).

fn pcg3d(v_in: vec3<u32>) -> vec3<u32> {
    // LCG step (Numerical Recipes constants 1664525, 1013904223).
    var v = v_in * 1664525u + 1013904223u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    v ^= v >> vec3<u32>(16u);

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    return v;
}
