// sky/world_points.wgsl: point sources drawn as WORLD light (final review render I2): a light
// bright enough to light-adapt the eye (uni_sky::world_points::ADAPTING_LUX, the Sun from inside
// ~356 AU) that projects below 1.5 px. The star-surface pass ("sun", uni-planet) draws them in
// its own render pass, before the bodies and the exposure meter, so the meter sees their glare
// and the eye adapts to it; the point pass then skips their sprites.
//
// The splat is the star pass's (splat_vertex_at and psf_weight from sky/stars3d.wgsl: the
// erf-integrated core plus the eye's glare halo, tapered to black inside its 64-px clamp with
// its light kept up to the clamp), written × scene_pre() (world light, M1 frame contract,
// ruling 5) on the internal target (camera.viewport). Vertex depth at the instance's depth
// distance, tested Greater and never written, so a star disk in front hides it and bodies drawn
// later cover it; additive into hdr only (no lens_mask channel: glare is light, not an occluder,
// and its star share is 0).
//
// Group 1 binding 6: the instances. stars3d.wgsl (and M0's stars.wgsl through it) declare
// bindings 0–5 of group 1; these entry points use none of them.
//
// CPU twin: crates/uni-sky/src/world_points.rs (`world_splat`), with stars3d.rs's `halo_shape`
// and `psf_weight`.

#import "sky/stars3d.wgsl"

// Twin of `uni_sky::world_points::WorldPointGpu`: 32 bytes, two vec4s.
struct WorldPointGpu {
    // xyz: unit direction source − camera, root (ICRS) axes; w: the distance [m] its depth is
    // taken at.
    dir_depth: vec4<f32>,
    // rgb: colour (linear Rec. 709, luminance 1); w: illuminance at the camera [lx].
    color_lux: vec4<f32>,
}

@group(1) @binding(6) var<storage, read> world_points: array<WorldPointGpu>;

@vertex
fn vs_world_points(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SplatOut {
    let p = world_points[ii];
    // NDC depth at the depth distance, as vs_points: near / (d·cos θ) under the reversed-Z
    // infinite projection. Behind the camera or inside the near plane: no quad.
    let view_dir = (camera.view_rot * vec4<f32>(p.dir_depth.xyz, 0.0)).xyz;
    let clip = camera.proj * vec4<f32>(view_dir * p.dir_depth.w, 1.0);
    let depth = clip.z / clip.w;
    if (!(depth > 0.0 && depth <= 1.0)) {
        return culled_splat();
    }
    return splat_vertex_at(vi, p.dir_depth.xyz, p.color_lux.w, 1.0, p.color_lux.rgb, depth,
        camera.viewport.xy, scene_pre());
}

@fragment
fn fs_world_points(in: SplatOut) -> @location(0) vec4<f32> {
    let pix = floor(in.pos.xy);
    let rgb = in.radiance * psf_weight(pix, in.centre_px, in.halo);
    return vec4<f32>(min(rgb, vec3<f32>(HDR_MAX)), 0.0);
}
