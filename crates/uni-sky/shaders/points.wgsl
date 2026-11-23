// sky/points.wgsl: the point-sprite pass "points" (Task B4): unresolved bodies, the Sun and near
// stars as PSF splats, × star_pre() × (1 − r) of lens_mask (review m6), with vertex depth from
// the true distance and compare Greater, so bodies in front hide them.
//
// The splat itself (projection, E/Ω(θ) × star_pre(), M0's PSF, the 2-px minimum footprint, the
// cull) is `splat_vertex` and `psf_weight` from sky/stars3d.wgsl, imported here, so a point and
// a star of equal illuminance and direction leave the same energy. This pass binds only group 1's
// `lens_mask` (binding 4, declared in stars3d.wgsl) and its own instances (binding 5).
//
// CPU twins: crates/uni-sky/src/points.rs (`twin::point_splat`, `twin::point_sky_factor`).

#import "sky/stars3d.wgsl"

// Twin of `uni_sky::points::PointGpu`: 32 bytes, two vec4s.
struct PointGpu {
    // xyz: unit direction point − camera, root (ICRS) axes; w: distance [m].
    dir_dist: vec4<f32>,
    // rgb: colour (linear Rec. 709, luminance 1); w: illuminance at the camera [lx].
    color_lux: vec4<f32>,
}

@group(1) @binding(5) var<storage, read> points: array<PointGpu>;

// The share of a point's light kept under lens-mask value `mask`: 1 − r (review m6). Points are
// depth-tested against bodies instead of using g. Twin of
// `uni_sky::points::twin::point_sky_factor`.
fn point_sky_factor(mask: vec4<f32>) -> f32 {
    return 1.0 - clamp(mask.r, 0.0, 1.0);
}

@vertex
fn vs_points(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SplatOut {
    let point = points[ii];
    // NDC depth of the point at its true distance: proj × (view_dir·d, 1) gives
    // near / (d·cos θ) under the reversed-Z infinite projection, the depth a body's vertex there
    // gets, so a nearer body (greater depth) hides it. Behind the camera or inside the near
    // plane: no quad.
    let view_dir = (camera.view_rot * vec4<f32>(point.dir_dist.xyz, 0.0)).xyz;
    let clip = camera.proj * vec4<f32>(view_dir * point.dir_dist.w, 1.0);
    let depth = clip.z / clip.w;
    if (!(depth > 0.0 && depth <= 1.0)) {
        return culled_splat();
    }
    return splat_vertex(vi, point.dir_dist.xyz, point.color_lux.w, point.color_lux.rgb, depth);
}

@fragment
fn fs_points(in: SplatOut) -> @location(0) vec4<f32> {
    let pix = floor(in.pos.xy);
    let mask = textureLoad(lens_mask, vec2<i32>(pix), 0);
    let rgb = in.radiance * (psf_weight(pix, in.centre_px, in.halo) * point_sky_factor(mask));
    return vec4<f32>(min(rgb, vec3<f32>(HDR_MAX)), 0.0);
}
