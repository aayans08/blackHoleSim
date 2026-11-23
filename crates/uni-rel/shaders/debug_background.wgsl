// rel/debug_background.wgsl: the M0 debug-grid background, kept in M1 as the opt-in lens
// environment uni_rel::LensEnv::DebugGrid (ruling 14; DebugSkyBackground in
// crates/uni-rel/src/background.rs).
//
// One compute pass over the whole frame, pass 3 of the M1 order (uni-gfx targets.rs), long
// before the lens pass: it writes the unlensed debug grid into every hdr pixel. Bodies then draw
// over it, and the lens pass overwrites only the sky pixels it traces with the same grid seen
// through the lens, so outside the lens rectangle (where the deflection is under half a pixel)
// the grid continues unlensed and the edge is invisible. The host runs it only with
// LensEnv::DebugGrid while the lens is traced; with LensEnv::Band the band pass draws the sky
// around the lens instead. Display-referred: the app runs Manual exposure while it is selected.

#import "common/camera.wgsl"
#import "rel/debug_sky.wgsl"

// Twin of `BackgroundParams` in crates/uni-rel/src/background.rs (vec4 members only).
struct BackgroundParams {
    // x: line-width factor, 1 / the lens tracer's resolution scale, so the unlensed lines are as
    // wide as the traced region's (one ray sample = 1/scale pixels); yzw: 0.
    scale: vec4<f32>,
}

@group(1) @binding(0) var<uniform> background: BackgroundParams;
@group(1) @binding(1) var hdr_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_background(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(hdr_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let pf = vec2<f32>(gid.xy);
    let d = pixel_dir(pf);
    let c = debug_sky(d, pixel_footprint(pf, d) * background.scale.x);
    textureStore(hdr_out, gid.xy, vec4<f32>(c, 1.0));
}
