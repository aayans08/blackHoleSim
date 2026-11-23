// post/exposure.wgsl: the exposure resolve pass "exposure" (Task B6): writes the frame's
// ExposureUniform (scene EV adapted or manual; star EV eye-adapted or equal to the scene's) into
// the renderer's exposure buffer through a storage binding in this pass's own bind group.
//
// It imports only common/exposure.wgsl (no bindings) and never group 0: binding the exposure
// buffer as a uniform and as storage in one dispatch is a usage-scope conflict (review m14).
//
// B0 stub: valid and neutral (writes nothing, so the Renderer's CPU-written exposure stands).
// B6 replaces everything below the import.

#import "common/exposure.wgsl"

@group(1) @binding(0) var<storage, read_write> exposure_out: ExposureUniform;

@compute @workgroup_size(1, 1, 1)
fn cs_resolve() {
}
