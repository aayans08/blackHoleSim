// sky/band_update.wgsl: the band-update compute pass (Task B5): per cubemap texel,
// gal_band_radiance(cam_gc, dir_gc, v_lim) from uni-procgen's WGSL twin, divided by
// ENV_UNIT_CD_M2 (1e-4 cd/m2) and clamped to 60000.
//
// B0 stub: valid and neutral (writes nothing). It already imports what B5 builds on, so the
// cross-crate chain composes: procgen/galaxy.wgsl (registered by uni_sky::register_shaders,
// ruling A.6) and common/cube.wgsl. Per galaxy.wgsl's contract the importer declares GalaxyGpu at
// its own group and copies it into `galaxy` before calling any gal_* function. B5 replaces
// everything below the imports.

#import "common/cube.wgsl"
#import "procgen/galaxy.wgsl"

@group(1) @binding(0) var<uniform> band_galaxy: GalaxyGpu;

@compute @workgroup_size(8, 8, 1)
fn cs_band_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    galaxy = band_galaxy;
}
