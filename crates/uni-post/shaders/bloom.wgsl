// post/bloom.wgsl: bloom "bloom" (Task B6): Jimenez 2014's 13-tap downsample (Karis average on
// the first level, input clamped to BLOOM_INPUT_MAX) and 3x3 tent upsample over 6 levels, then
// hdr = mix(hdr, bloom, strength).
//
// B0 stub: valid and neutral (changes nothing). B6 replaces everything below the import.

#import "common/camera.wgsl"

@compute @workgroup_size(8, 8, 1)
fn cs_bloom(@builtin(global_invocation_id) gid: vec3<u32>) {
}
