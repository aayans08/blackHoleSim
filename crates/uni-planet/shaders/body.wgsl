// planet/body.wgsl: the bodies pass (Task B1): analytic ellipsoids with surface maps.
//
// Per resolved body the CPU draws a camera-facing cap (B0's cap_mesh, made camera-relative in
// f64, then f32) that only supplies coverage and vertex depth; this shader re-intersects the
// exact ellipsoid per pixel in the body's unit-sphere space (body-fixed coordinates divided by
// the radii). Depth is the mesh's vertex depth, never frag_depth (Apple GPUs are tile-based).
//
// Radiance convention (uni_physics::photometry): a surface of normal reflectance rho lit by an
// illuminance E at normal incidence has radiance L = rho * R(mu0, mu) * E / pi [cd/m^2], with R
// the lunar-Lambert reflectance factor (Lambert when L = 0). World light: written as
// L * pre, pre = pre_mix(star_pre(), scene_pre(), w) (the log-space blend of common/exposure.wgsl,
// task B11), clamped to HDR_MAX (M1 frame contract), and the star share 1 - w into lens_mask.a
// (star_share).
//
// Bindings: group 0 the renderer's (common/camera.wgsl); group 1 the body (`BodyUniform` at a
// dynamic offset, its maps, Saturn's ring profile); group 2 uni-atmo's ground lighting
// (atmo/ground_lighting.wgsl), neutral for airless bodies.

#import "common/camera.wgsl"
#import "common/constants.wgsl"
#import "planet/eclipse.wgsl"
#import "atmo/ground_lighting.wgsl"

// Twin of `uni_planet::body::BodyUniform` (15 vec4, 240 bytes, vec4 members only;
// `body_uniforms_layout_match_wgsl`).
struct BodyUniform {
    // Rows of the ICRS -> body-fixed rotation (xyz); w: 0. Body +z is the pole, +x the prime
    // meridian (uni_physics::rotation).
    to_body_x: vec4<f32>,
    to_body_y: vec4<f32>,
    to_body_z: vec4<f32>,
    // xyz: radii of the drawn ellipsoid [m] (Venus: its cloud tops); w: 0.
    radii: vec4<f32>,
    // xyz: the camera in unit-sphere space; w: c = |xyz|^2 - 1, computed in f64 on the CPU
    // (review I5: in f32 it loses its sign within metres of the surface).
    cam_unit: vec4<f32>,
    // xyz: Sun centre - body centre, body-fixed [m]; w: the Sun's radius [m].
    sun: vec4<f32>,
    // rgb: E_sun [lx] x the Sun's colour (luminance 1) x the phase correction C; w: exposure
    // weight (0 star_pre ... 1 scene_pre).
    light: vec4<f32>,
    // rgb: reflectance multiplier (the map's albedo_scale, or the uniform rho without a map);
    // w: lunar-Lambert L (0 = Lambert).
    albedo: vec4<f32>,
    // x: 1 with an albedo map; y: cloud albedo s_c (0 = no cloud layer); z: albedo map lon0
    // [deg]; w: cloud map lon0 [deg].
    maps: vec4<f32>,
    // x: map longitude offset [rad] (Venus's super-rotating cloud tops); y, z: ring shadow
    // inner and outer radius [m] (z <= y: none); w: number of occluders.
    extra: vec4<f32>,
    // Ruling W-B1's point -> disk handover (see flux_match): x: w (0 at 1.5 px, 1 from 4 px);
    // y: A, the point's flux over this pass's disk flux at k = 1; z: B, the atmosphere pass's
    // in-scatter flux over the same; w: the bound of k.
    handover: vec4<f32>,
    // xyz: occluder centre - body centre, body-fixed [m]; w: its radius [m].
    occluders: array<vec4<f32>, 4>,
}

@group(1) @binding(0) var<uniform> body_u: BodyUniform;
// Albedo (or cloud-top) map: BC7 or Rgba8 sRGB, decoded to linear by the format.
@group(1) @binding(1) var body_albedo: texture_2d<f32>;
// Earth's cloud coverage (BC4 or R8, linear), red channel.
@group(1) @binding(2) var body_clouds: texture_2d<f32>;
// Linear, mipmapped, anisotropic; repeat in u (longitude), clamp in v.
@group(1) @binding(3) var body_map_sampler: sampler;
// Saturn's ring opacity, one row of R16Float samples from inner to outer radius.
@group(1) @binding(4) var body_ring: texture_2d<f32>;
// Linear, clamp to edge.
@group(1) @binding(5) var body_ring_sampler: sampler;

// A texel spanning more than BICUBIC_START_PX pixels starts blending to the cubic B-spline,
// which takes over fully at BICUBIC_FULL_PX (B1 brief: bicubic above 1.5 px per texel; the
// half-pixel blend keeps the switch from showing as a line).
const BICUBIC_START_PX: f32 = 1.5;
const BICUBIC_FULL_PX: f32 = 2.0;
// Emission cosine used at the limb, where the analytic value is 0 (the AA rim pixels).
const LIMB_MU: f32 = 1.0e-4;

// Ray - unit sphere in unit-sphere space: o + t d, c = |o|^2 - 1 from the CPU. Returns (t0, t1,
// hit) with t0 <= t1, hit 1 on an intersection, else (0, 0, 0). Twin of
// `uni_planet::ray_ellipsoid`: keep the order of operations identical.
//
// Numerically stable quadratic (Haines et al. 2019, Ray Tracing Gems ch. 7): the discriminant
// b^2 - a c is formed as a (1 - |o - (b/a) d|^2), from the ray's distance to the centre rather
// than a difference of two large squares, and q = -(b + sign(b) sqrt(disc)) gives the roots q/a
// and c/q without cancellation.
fn ray_ellipsoid(o: vec3<f32>, d: vec3<f32>, c: f32) -> vec3<f32> {
    let a = dot(d, d);
    let b = dot(o, d);
    let l = o - d * (b / a);
    let disc = a * (1.0 - dot(l, l));
    if (!(disc >= 0.0)) {
        return vec3<f32>(0.0);
    }
    let q = -(b + select(-1.0, 1.0, b >= 0.0) * sqrt(disc));
    if (q == 0.0) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let t0 = q / a;
    let t1 = c / q;
    return vec3<f32>(min(t0, t1), max(t0, t1), 1.0);
}

// Equirectangular texture coordinates of planetocentric east longitude lon_rad and latitude
// lat_rad, for a map whose left edge is at east longitude lon0_deg: u east through 360 deg,
// v from +90 deg (0) to -90 deg (1). Twin of `uni_planet::equirect_uv`.
fn equirect_uv(lon_rad: f32, lat_rad: f32, lon0_deg: f32) -> vec2<f32> {
    let x = (lon_rad - radians(lon0_deg)) / TAU;
    return vec2<f32>(x - floor(x), 0.5 - lat_rad / PI);
}

// Radius [m] at which the ray from p toward the Sun (unit s) crosses the ring plane z = 0 of
// the body's axes inside [inner, outer]; -1 when it does not (Sun in the plane, crossing behind
// p, or outside the annulus). Twin of `uni_planet::body::ring_crossing_radius`.
fn ring_crossing_radius(p: vec3<f32>, s: vec3<f32>, inner: f32, outer: f32) -> f32 {
    if (s.z == 0.0) {
        return -1.0;
    }
    let t = -p.z / s.z;
    if (!(t > 0.0)) {
        return -1.0;
    }
    let r = length(p.xy + t * s.xy);
    if (r >= inner && r <= outer) {
        return r;
    }
    return -1.0;
}

// Fraction of sunlight reaching p through Saturn's rings: 1 - opacity at the crossing, the
// profile sampled linearly between sample centres (twin of `uni_planet::ring_transmittance`
// with `ring_profile_opacity`). The opacity is the profile's as is, with no slant-path
// correction, matching the ring pass's use of it (B2).
fn ring_transmittance(p: vec3<f32>, s: vec3<f32>, inner: f32, outer: f32) -> f32 {
    let r = ring_crossing_radius(p, s, inner, outer);
    if (r < 0.0) {
        return 1.0;
    }
    let u = (r - inner) / (outer - inner);
    return 1.0 - textureSampleLevel(body_ring, body_ring_sampler, vec2<f32>(u, 0.5), 0.0).r;
}

// Ruling W-B1's factor k on this pass's light (CPU twin `uni_planet::body::flux_match`):
// k = (1 - w) A + w + B (w - sigma), clamped to [0, k_max], sigma = pre_atmo / pre. This pass
// at k = 1 sends a disk flux D, the point sends A D, and the atmosphere pass adds B D at
// pre_atmo, so the total k D pre + B D pre_atmo is pre [(1 - w) A D + w (D + B D)]: the
// point's flux at 1.5 px, the real rendering from 4 px (k = 1), a linear cross-fade of the flux
// between, written at the log-space exposure pre = pre_mix(star_pre(), scene_pre(), w) (task
// B11). fs_main passes sigma = 1: the atmosphere draws at its body's exposure weight and the same
// pre_mix (lead ruling B6 I1, task B10; uni_atmo::AtmoDraw::exposure_mix =
// uni_planet::body::exposure_weight), so k does not depend on how pre blends.
fn flux_match(h: vec4<f32>, sigma: f32) -> f32 {
    return clamp((1.0 - h.x) * h.y + h.x + h.z * (h.x - sigma), 0.0, h.w);
}

// The star share this fragment stores in lens_mask.a, blended with Max (M1 frame contract, lead
// ruling B6 I1): 1 - w where it covers anything, else 0 (0 = scene exposure, the clear value).
// The meter (post/histogram.wgsl, meter_pre) reads the pixel's pre back as
// pre_mix(scene_pre(), star_pre(), share) = this pass's pre_mix(star_pre(), scene_pre(), w). The
// CPU snaps w to 1 when 1 - w < 2^-14 (uni_planet::body::exposure_weight), so the share is 0 or
// an f16 normal, never a subnormal Metal could flush. CPU twin: uni_planet::body::star_share.
fn star_share(w: f32, coverage: f32) -> f32 {
    return select(0.0, 1.0 - w, coverage > 0.0);
}

// Reflectance factor R(mu0, mu), normalised by rho: lunar-Lambert 2 L mu0 / (mu0 + mu) +
// (1 - L) mu0 (McEwen 1991, Icarus 92, 298), Lambert at L = 0; 0 when unlit or facing away.
// Twin of `uni_physics::photometry::reflectance`.
fn lunar_lambert(l: f32, mu0: f32, mu: f32) -> f32 {
    if (mu0 <= 0.0 || mu <= 0.0) {
        return 0.0;
    }
    return 2.0 * l * mu0 / (mu0 + mu) + (1.0 - l) * mu0;
}

// Cubic B-spline filtering at mip 0 from four bilinear taps (Sigg & Hadwiger 2005, "Fast
// Third-Order Texture Filtering", GPU Gems 2 ch. 20): the B-spline is C2 and never overshoots,
// so a magnified map shows no texel diamonds. Texel centres sit at (i + 0.5) / size.
fn bspline_sample(t: texture_2d<f32>, uv: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
    let tc = uv * size - 0.5;
    let i = floor(tc);
    let f = tc - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    // Each pair of weights becomes one bilinear tap between its two texels.
    let h0 = (i - 0.5 + w1 / g0) / size;
    let h1 = (i + 1.5 + w3 / g1) / size;
    let t00 = textureSampleLevel(t, body_map_sampler, vec2<f32>(h0.x, h0.y), 0.0);
    let t10 = textureSampleLevel(t, body_map_sampler, vec2<f32>(h1.x, h0.y), 0.0);
    let t01 = textureSampleLevel(t, body_map_sampler, vec2<f32>(h0.x, h1.y), 0.0);
    let t11 = textureSampleLevel(t, body_map_sampler, vec2<f32>(h1.x, h1.y), 0.0);
    return g0.y * (g0.x * t00 + g1.x * t10) + g1.y * (g0.x * t01 + g1.x * t11);
}

// A map sample with explicit gradients (trilinear + anisotropic while minified or mildly
// magnified), blended into the cubic B-spline once a texel spans more than BICUBIC_START_PX.
fn sample_map(t: texture_2d<f32>, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(t, 0));
    let texels_per_px = max(length(ddx * size), length(ddy * size));
    let w = smoothstep(BICUBIC_START_PX, BICUBIC_FULL_PX, 1.0 / max(texels_per_px, 1.0e-20));
    var c = vec4<f32>(0.0);
    if (w < 1.0) {
        c = textureSampleGrad(t, body_map_sampler, uv, ddx, ddy);
    }
    if (w > 0.0) {
        c = mix(c, bspline_sample(t, uv, size), w);
    }
    return c;
}

fn to_body(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(body_u.to_body_x.xyz, v),
        dot(body_u.to_body_y.xyz, v),
        dot(body_u.to_body_z.xyz, v),
    );
}

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    // Camera-relative root-axes position [m] (cast from f64 on the CPU).
    return camera.proj * (camera.view_rot * vec4<f32>(pos, 1.0));
}

struct BodyOut {
    // Premultiplied: light x coverage, coverage.
    @location(0) hdr: vec4<f32>,
    // lens_mask.g = coverage, lens_mask.a = the star share (write mask GREEN | ALPHA,
    // BlendOperation::Max).
    @location(1) mask: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> BodyOut {
    let radii = body_u.radii.xyz;
    let o = body_u.cam_unit.xyz;
    let c = body_u.cam_unit.w;
    let d_body = to_body(pixel_dir(floor(frag.xy)));
    let d_unit = d_body / radii;
    let hit = ray_ellipsoid(o, d_unit, c);

    // Silhouette AA. Signed angle between the pixel ray and the limb cone in unit-sphere space
    // (positive outside): the ray's angle from the centre, atan2(|l|, -b), minus the cone's
    // half-angle asin(1/|o|) = atan2(1, sqrt(c)). Its screen-space gradient turns it into a
    // signed distance in pixels, which absorbs the ellipsoid's anisotropy and the projection's.
    let dn = normalize(d_unit);
    let b = dot(o, dn);
    let l = o - b * dn;
    let limb = atan2(length(l), -b) - atan2(1.0, sqrt(max(c, 0.0)));
    let limb_px = limb / max(length(vec2<f32>(dpdx(limb), dpdy(limb))), 1.0e-30);
    let coverage = clamp(0.5 - limb_px, 0.0, 1.0);

    // Surface point: the entry on a hit, else (the rim just outside the limb) the limb point
    // nearest the ray, so the AA rim is shaded like the limb.
    let p_unit = select(l / max(length(l), 1.0e-30), o + hit.x * d_unit, hit.z > 0.5);
    let p_body = p_unit * radii;
    let lon = atan2(p_body.y, p_body.x) - body_u.extra.x;
    let lat = atan2(p_body.z, length(p_body.xy));
    let uv = equirect_uv(lon, lat, body_u.maps.z);
    // Seam-safe gradients (Tarini 2012, "Cylindrical and toroidal parameterizations without
    // vertex seams", JGT 16(3)): of u and of u shifted by half a turn, the one with the smaller
    // derivative is continuous here.
    let u_alt = fract(uv.x + 0.5) - 0.5;
    let gu = vec2<f32>(dpdx(uv.x), dpdy(uv.x));
    let ga = vec2<f32>(dpdx(u_alt), dpdy(u_alt));
    let g = select(gu, ga, dot(ga, ga) < dot(gu, gu));
    let ddx = vec2<f32>(g.x, dpdx(uv.y));
    let ddy = vec2<f32>(g.y, dpdy(uv.y));
    if (coverage < 1.0 / 255.0) {
        discard;
    }

    // Lighting, body-fixed: the ellipsoid normal (gradient of sum x^2/r^2), toward the camera
    // and toward the Sun.
    let n = normalize(p_unit / radii);
    let to_sun = body_u.sun.xyz - p_body;
    let s = normalize(to_sun);
    let mu0 = dot(n, s);
    let mu = max(dot(n, -d_body), LIMB_MU);
    let refl = lunar_lambert(body_u.albedo.w, mu0, mu);

    // Eclipses: the Sun's disk past each occluder, from this point (planar uniform disks).
    var vis = 1.0;
    let sun_dist = length(to_sun);
    let sun_ang = asin(min(body_u.sun.w / sun_dist, 1.0));
    let n_occ = min(u32(body_u.extra.w), 4u);
    for (var k = 0u; k < n_occ; k++) {
        let occ = body_u.occluders[k];
        let to_occ = occ.xyz - p_body;
        let d_occ = length(to_occ);
        if (occ.w > 0.0 && d_occ < sun_dist) {
            let o_dir = to_occ / d_occ;
            let occ_ang = asin(min(occ.w / d_occ, 1.0));
            let sep = atan2(length(cross(s, o_dir)), dot(s, o_dir));
            vis *= sun_visible_fraction(sun_ang, occ_ang, sep);
        }
    }
    if (body_u.extra.z > body_u.extra.y) {
        vis *= ring_transmittance(p_body, s, body_u.extra.y, body_u.extra.z);
    }

    // Atmosphere (uni-atmo group 2; neutral for airless bodies): the Sun's light left after the
    // path down to the ground, and skylight per unit sun illuminance on horizontal ground. B3's
    // contract (atmo/ground_lighting.wgsl): r is the atmosphere's bottom radius plus the height
    // above the reference ellipsoid, 0 without terrain in M1 (|p| would put Earth's equator 7 km
    // up and its poles 14 km down), and mu_s is taken about the surface normal. The W-B1
    // handover tables evaluate the ground the same way (uni_planet::body::AtmoHandover).
    let r_m = atmo_ground.radii.x;
    let mu_s = mu0;
    let sun_t = atmo_sun_transmittance(r_m, mu_s);
    let sky = atmo_sky_irradiance(r_m, mu_s);

    // Reflectance: uniform, or the map x albedo_scale with Earth's clouds over it as areal
    // coverage of opaque clouds, rho = (1 - c) rho_ground + c s_c (ruling W-B2).
    var rho = body_u.albedo.rgb;
    if (body_u.maps.x > 0.5) {
        rho *= sample_map(body_albedo, uv, ddx, ddy).rgb;
        if (body_u.maps.y > 0.0) {
            let uv_clouds = equirect_uv(lon, lat, body_u.maps.w);
            let cover = sample_map(body_clouds, uv_clouds, ddx, ddy).r;
            rho = (1.0 - cover) * rho + cover * body_u.maps.y;
        }
    }

    // Skylight is dimmed by an eclipse too (the sky above is lit by the same Sun).
    let pre = pre_mix(star_pre(), scene_pre(), body_u.light.w);
    // sigma = 1: the atmosphere over this body draws at the same exposure weight (flux_match).
    let k = flux_match(body_u.handover, 1.0);
    let radiance = rho * body_u.light.rgb * k * (refl * sun_t + sky) * vis / PI;
    let lit = min(radiance * pre, vec3<f32>(HDR_MAX));
    var out: BodyOut;
    out.hdr = vec4<f32>(lit * coverage, coverage);
    out.mask = vec4<f32>(0.0, coverage, 0.0, star_share(body_u.light.w, coverage));
    return out;
}
