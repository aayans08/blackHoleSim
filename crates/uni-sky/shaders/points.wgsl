// sky/points.wgsl: the point-sprite pass "points" (Task B4): unresolved bodies, the Sun and near
// stars as PSF splats, × star_pre() × (1 − r) of lens_mask (review m6), and × (1 − g) as well
// while the camera is inside an atmosphere shell (lead's r0 ruling), with vertex depth from the
// true distance and compare Greater, so bodies in front hide them.
//
// The splat itself (projection, E/Ω(θ) × star_pre(), M0's PSF, the 2-px minimum footprint, the
// cull) is `splat_vertex` and `psf_weight` from sky/stars3d.wgsl, imported here, so a point and
// a star of equal illuminance and direction leave the same energy. This pass binds only group 1's
// `lens_mask` (binding 1, M0's declaration in sky/stars.wgsl, which stars3d.wgsl imports), its
// own instances (binding 6) and its uniform (binding 7); bindings 0 and 2–5 are the star passes'
// and unused here.
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

// Twin of `uni_sky::points::PointsParams`: 16 bytes, written once per frame.
struct PointsParams {
    // x: 1 while the camera is inside an atmosphere shell (uni_sky::StarInput::scintillation
    // > 0, B3), else 0; yzw: 0.
    sky: vec4<f32>,
}

@group(1) @binding(6) var<storage, read> points: array<PointGpu>;
@group(1) @binding(7) var<uniform> points_params: PointsParams;

// The share of a point's light kept under lens-mask value `mask`. From space 1 − r (review m6):
// points are depth-tested against bodies instead of using g, so a moon's point in front of a
// planet's limb is not dimmed by the limb's g. With the camera inside an atmosphere every point
// is seen through the air: (1 − r)(1 − g), the star pass's factor (lead's r0 ruling). Twin of
// `uni_sky::points::twin::point_sky_factor`.
fn point_sky_factor(mask: vec4<f32>, inside_atmosphere: bool) -> f32 {
    let kept = 1.0 - clamp(mask.r, 0.0, 1.0);
    return select(kept, kept * (1.0 - clamp(mask.g, 0.0, 1.0)), inside_atmosphere);
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
    return splat_vertex(
        vi, point.dir_dist.xyz, point.color_lux.w, 1.0, point.color_lux.rgb, depth);
}

@fragment
fn fs_points(in: SplatOut) -> @location(0) vec4<f32> {
    let pix = floor(in.pos.xy);
    let mask = textureLoad(lens_mask, vec2<i32>(pix), 0);
    let inside = points_params.sky.x > 0.5;
    let rgb = in.radiance
        * (psf_weight(pix, in.centre_px, in.halo) * point_sky_factor(mask, inside));
    return vec4<f32>(min(rgb, vec3<f32>(HDR_MAX)), 0.0);
}
