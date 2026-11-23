// post/histogram.wgsl: the exposure meter "histogram" (Task B6): world luminance of hdr divided
// by scene_pre into 256 log2 bins over [-10, 32] with workgroup-shared atomics, then one
// reduction (bruop 2019, "Automatic Exposure Using a Luminance Histogram").
//
// B0 stub: valid and neutral (counts nothing). B6 replaces everything below the import.

#import "common/camera.wgsl"

@compute @workgroup_size(16, 16, 1)
fn cs_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
}
