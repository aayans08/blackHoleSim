// planet/star_surface.wgsl: the star-surface pass "sun" (Task B2): limb darkening, granulation.
//
// Per resolved star the CPU uploads a camera-facing cap mesh (uni_planet::cap_mesh, drawn a
// little larger than the star so the limb can be antialiased) as camera-relative vertices, and
// one `StarGpu` record (uni_planet::star_surface::StarGpu). The fragment shader re-intersects the
// exact sphere per pixel in the star's unit-sphere space, so the mesh supplies only coverage and
// vertex depth.
//
// CPU twins (crates/uni-planet/src/star_surface.rs): `limb_darkening`, `granulation`,
// `granulation_fade`. Keep the order of operations identical; `universe --selftest` compares
// limb_darkening (64 μ) and granulation (256 points: the lattice hash and each layer's nearest
// granule bit for bit, the value within a tolerance scaled by the pattern's slope) on the GPU.

#import "common/camera.wgsl"
#import "common/hash.wgsl"
#import "common/constants.wgsl"

// ---- Limb darkening ----------------------------------------------------------------------
//
// Neckel & Labs 1994, "Solar limb darkening 1986–1990 (λλ303 to 1099 nm)", Solar Physics 153,
// 91–114, Table I: fifth-order polynomials P5(μ) = Σ A_n μ^n (their eq. 5, Σ A_n = 1), mean of
// the 1986 and 1987 east–west averages, corrected for scattered light. Channels (r, g, b) use
// the rows nearest 650, 550 and 450 nm: 640.970, 541.760 and 445.125 nm (the last is marked in
// the table as observed only since 1988).
const LD_A0: vec3<f32> = vec3<f32>(0.33644, 0.26073, 0.15248);
const LD_A1: vec3<f32> = vec3<f32>(1.30590, 1.27428, 1.38517);
const LD_A2: vec3<f32> = vec3<f32>(-1.79238, -1.30352, -1.49615);
const LD_A3: vec3<f32> = vec3<f32>(2.45040, 1.47085, 1.99886);
const LD_A4: vec3<f32> = vec3<f32>(-1.89979, -0.96618, -1.48155);
const LD_A5: vec3<f32> = vec3<f32>(0.59943, 0.26384, 0.44119);

// I(μ)/I(1) per channel, μ clamped to [0, 1]; Horner's rule.
fn limb_darkening(mu_in: f32) -> vec3<f32> {
    let mu = clamp(mu_in, 0.0, 1.0);
    return LD_A0 + mu * (LD_A1 + mu * (LD_A2 + mu * (LD_A3 + mu * (LD_A4 + mu * LD_A5))));
}

// ---- Granulation ---------------------------------------------------------------------------
//
// Cellular (Worley F2 − F1) noise on an equal-angle cube-sphere lattice of about one granule per
// cell, two layers whose patterns are regenerated every GRAN_PERIOD_S and cross-faded half a
// period apart (Neyret 2003, "Advected textures", SCA), renormalised so the variance does not dip
// mid-fade. See `granulation` in star_surface.rs for the sources of every number.
const GRAN_PERIOD_S: f32 = 1024.0;
const GRAN_EPOCHS: i32 = 4096;
const GRAN_MAX_CELLS: f32 = 8192.0;
const GRAN_JITTER_MIN: f32 = 0.1;
const GRAN_JITTER_SPAN: f32 = 0.8;
const GRAN_LANE: f32 = 0.3;
const GRAN_BRIGHT_MIN: f32 = 0.55;
const GRAN_BRIGHT_SPAN: f32 = 0.45;
const GRAN_MEAN: f32 = 0.5109;
const GRAN_SIGMA: f32 = 0.3112;
const GRAN_RMS_550: f32 = 0.18;
const GRAN_FADE_LO_PX: f32 = 1.25;
const GRAN_FADE_HI_PX: f32 = 2.0;
const FRAC_4_PI: f32 = 1.2732395447351628;
const FRAC_PI_4: f32 = 0.7853981633974483;
const GRAN_NONE: u32 = 0xffffffffu;

struct CubeCoord {
    face: u32,
    st: vec2<f32>,
}

// Cube face of unit vector p and its gnomonic coordinates in [-1, 1]: faces 0/1 = ±x with
// (y, z), 2/3 = ±y with (z, x), 4/5 = ±z with (x, y).
fn gran_cube(p: vec3<f32>) -> CubeCoord {
    let a = abs(p);
    if (a.x >= a.y && a.x >= a.z) {
        return CubeCoord(select(1u, 0u, p.x >= 0.0), vec2<f32>(p.y, p.z) / a.x);
    }
    if (a.y >= a.z) {
        return CubeCoord(select(3u, 2u, p.y >= 0.0), vec2<f32>(p.z, p.x) / a.y);
    }
    return CubeCoord(select(5u, 4u, p.z >= 0.0), vec2<f32>(p.x, p.y) / a.z);
}

// Unit vector of face `face` at gnomonic coordinates `st` (the inverse of gran_cube).
fn gran_dir(face: u32, st: vec2<f32>) -> vec3<f32> {
    let sg = select(-1.0, 1.0, face % 2u == 0u);
    var v = vec3<f32>(st.x, st.y, sg);
    if (face / 2u == 0u) {
        v = vec3<f32>(sg, st.x, st.y);
    } else if (face / 2u == 1u) {
        v = vec3<f32>(st.y, sg, st.x);
    }
    return normalize(v);
}

// 24 high bits of x as a float in [0, 1), exact in f32.
fn gran_unit(x: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(x >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

// The lattice cell (face, i, j) that cell (ci, cj) of face `face` names: itself when inside the
// face, else the cell of the neighbouring face that holds the would-be cell's centre.
fn gran_wrap(face: u32, ci: i32, cj: i32, n: i32) -> vec3<u32> {
    if (ci >= 0 && ci < n && cj >= 0 && cj < n) {
        return vec3<u32>(face, u32(ci), u32(cj));
    }
    let nf = f32(n);
    let a = (vec2<f32>(f32(ci), f32(cj)) + 0.5) / nf * 2.0 - 1.0;
    let cc = gran_cube(gran_dir(face, tan(a * FRAC_PI_4)));
    let q = (atan(cc.st) * FRAC_4_PI + 1.0) * 0.5 * nf;
    let ij = clamp(vec2<i32>(floor(q)), vec2<i32>(0), vec2<i32>(n - 1));
    return vec3<u32>(cc.face, u32(ij.x), u32(ij.y));
}

// pcg3d of lattice cell (face, i, j) for pattern `salt` (epoch × 2 + layer).
fn gran_hash(c: vec3<u32>, salt: u32) -> vec3<u32> {
    return pcg3d(vec3<u32>(c.y, c.z, c.x + 6u * salt));
}

struct Nearest {
    f1: f32,
    f2: f32,
    id1: u32,
    b1: f32,
}

// Nearest-two update with feature `id` at distance `d` and brightness `b`; a feature already
// nearest (a cell reached twice across a cube corner) is skipped.
fn gran_consider(nr: Nearest, id: u32, d: f32, b: f32) -> Nearest {
    if (id == nr.id1) {
        return nr;
    }
    if (d < nr.f1) {
        return Nearest(d, nr.f1, id, b);
    }
    if (d < nr.f2) {
        return Nearest(nr.f1, d, nr.id1, nr.b1);
    }
    return nr;
}

// The granule (feature point) of cell c for pattern salt: its unit vector, and brightness in w.
fn gran_feature(c: vec3<u32>, salt: u32, nf: f32) -> vec4<f32> {
    let h = gran_unit(gran_hash(c, salt));
    let pos = vec2<f32>(c.yz) + GRAN_JITTER_MIN + GRAN_JITTER_SPAN * h.xy;
    return vec4<f32>(gran_dir(c.x, tan((pos / nf * 2.0 - 1.0) * FRAC_PI_4)), h.z);
}

// Intensity of one layer: dark lanes where the two nearest granules are equidistant, granule
// brightness varying per granule.
fn gran_intensity(nr: Nearest) -> f32 {
    return smoothstep(0.0, GRAN_LANE, nr.f2 - nr.f1) * (GRAN_BRIGHT_MIN + GRAN_BRIGHT_SPAN * nr.b1);
}

// Lattice cells per cube-face edge for granules of cell_m on a star of radius_m.
fn gran_cells(cell_m: f32, radius_m: f32) -> i32 {
    let n = floor(PI * radius_m / (2.0 * cell_m) + 0.5);
    return i32(clamp(select(1.0, n, n >= 1.0), 1.0, GRAN_MAX_CELLS));
}

// Pattern salt of `layer` at time t_s: the epoch modulo GRAN_EPOCHS (so the pattern is periodic
// in the 2^22 s wrap of sim_time_s), times two, plus the layer.
fn gran_salt(phase: f32, layer: u32) -> u32 {
    let e = i32(floor(phase));
    return u32(((e % GRAN_EPOCHS) + GRAN_EPOCHS) % GRAN_EPOCHS) * 2u + layer;
}

// Cross-fade weight of a layer at `phase`: 0 when its pattern is regenerated, 1 half a period
// later.
fn gran_weight(phase: f32) -> f32 {
    return 1.0 - abs(2.0 * (phase - floor(phase)) - 1.0);
}

struct GranLayers {
    l0: Nearest,
    l1: Nearest,
}

// Both layers' two nearest granules at the unit vector p at t_s (layer 1 half a period later):
// one loop over the 3 × 3 cells around p's (twin of uni_planet's gran_layers).
fn gran_layers(p: vec3<f32>, cell_m: f32, radius_m: f32, t_s: f32) -> GranLayers {
    let n = gran_cells(cell_m, radius_m);
    let nf = f32(n);
    let phase0 = t_s / GRAN_PERIOD_S;
    let salt0 = gran_salt(phase0, 0u);
    let salt1 = gran_salt(phase0 + 0.5, 1u);
    let cc = gran_cube(p);
    let q = (atan(cc.st) * FRAC_4_PI + 1.0) * 0.5 * nf;
    let base = clamp(vec2<i32>(floor(q)), vec2<i32>(0), vec2<i32>(n - 1));
    let to_cells = 2.0 * nf / PI;
    var near0 = Nearest(3.0e38, 3.0e38, GRAN_NONE, 0.0);
    var near1 = Nearest(3.0e38, 3.0e38, GRAN_NONE, 0.0);
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            let c = gran_wrap(cc.face, base.x + di, base.y + dj, n);
            let id = (c.x * u32(n) + c.y) * u32(n) + c.z;
            let f0 = gran_feature(c, salt0, nf);
            near0 = gran_consider(near0, id, length(p - f0.xyz) * to_cells, f0.w);
            let f1 = gran_feature(c, salt1, nf);
            near1 = gran_consider(near1, id, length(p - f1.xyz) * to_cells, f1.w);
        }
    }
    return GranLayers(near0, near1);
}

// Multiplicative granulation intensity at 550 nm, mean 1 (twin of uni_planet::granulation).
fn granulation(p_unit: vec3<f32>, cell_m: f32, radius_m: f32, t_s: f32) -> f32 {
    if (!(dot(p_unit, p_unit) > 0.0)) {
        return 1.0;
    }
    let p = normalize(p_unit);
    let near = gran_layers(p, cell_m, radius_m, t_s);
    let phase0 = t_s / GRAN_PERIOD_S;
    let phase1 = phase0 + 0.5;
    let v0 = (gran_intensity(near.l0) - GRAN_MEAN) / GRAN_SIGMA;
    let v1 = (gran_intensity(near.l1) - GRAN_MEAN) / GRAN_SIGMA;
    let w0 = gran_weight(phase0);
    let w1 = gran_weight(phase1);
    return 1.0 + GRAN_RMS_550 * (w0 * v0 + w1 * v1) / sqrt(w0 * w0 + w1 * w1);
}

// The packed lattice id of each layer's nearest granule at p at t_s (the probe's bit-identical
// check; twin of uni_planet's granulation_nearest_ids).
fn granulation_nearest_ids(p_unit: vec3<f32>, cell_m: f32, radius_m: f32, t_s: f32) -> vec2<u32> {
    let near = gran_layers(normalize(p_unit), cell_m, radius_m, t_s);
    return vec2<u32>(near.l0.id1, near.l1.id1);
}

// The lattice hash of p's own cell for layer 0 at t_s (the probe's bit-identical check).
fn granulation_cell_hash(p_unit: vec3<f32>, cell_m: f32, radius_m: f32, t_s: f32) -> vec3<u32> {
    let p = normalize(p_unit);
    let n = gran_cells(cell_m, radius_m);
    let cc = gran_cube(p);
    let q = (atan(cc.st) * FRAC_4_PI + 1.0) * 0.5 * f32(n);
    let base = clamp(vec2<i32>(floor(q)), vec2<i32>(0), vec2<i32>(n - 1));
    return gran_hash(vec3<u32>(cc.face, u32(base.x), u32(base.y)), gran_salt(t_s / GRAN_PERIOD_S, 0u));
}

// Fraction of the granulation contrast kept when a cell spans cell_px pixels: 0 at and below
// 1.25 px, 1 at and above 2 px (the footprint filter: no aliasing below the Nyquist size).
fn granulation_fade(cell_px: f32) -> f32 {
    return smoothstep(GRAN_FADE_LO_PX, GRAN_FADE_HI_PX, cell_px);
}

// ---- The pass ------------------------------------------------------------------------------

// Twin of uni_planet::star_surface::StarGpu: one star, 96 bytes, six vec4s
// (star_ring_uniforms_layout_match_wgsl).
struct StarGpu {
    // xyz: the camera in the star's unit-sphere space, world axes (camera − centre)/R; w: |xyz|² − 1
    // from the CPU in f64 (the ray–sphere constant, exact even 1 m above the photosphere).
    cam_unit: vec4<f32>,
    // Rows of the world → star-fixed rotation (granules ride with the rotating surface); w: 0.
    to_body_x: vec4<f32>,
    to_body_y: vec4<f32>,
    to_body_z: vec4<f32>,
    // rgb: disk-centre luminance per channel [cd/m²], I/(πR²) × colour / <LD>; w: the exposure
    // blend weight (lod_exposure_mix; 1 = scene exposure).
    l0: vec4<f32>,
    // x: radius [m], y: granule cell size [m], z, w: the granulation contrast of the r and b
    // channels relative to g (Planck's sensitivity to temperature at their wavelengths).
    params: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> stars: array<StarGpu>;

struct StarVsOut {
    @builtin(position) clip: vec4<f32>,
    // Camera-relative position [m], root axes; its direction is the pixel's ray.
    @location(0) rel: vec3<f32>,
    @location(1) @interpolate(flat) star: u32,
}

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @builtin(instance_index) k: u32) -> StarVsOut {
    var out: StarVsOut;
    out.clip = camera.proj * camera.view_rot * vec4<f32>(pos, 1.0);
    out.rel = pos;
    out.star = k;
    return out;
}

struct StarFsOut {
    // Premultiplied pre-exposed light; alpha = coverage.
    @location(0) color: vec4<f32>,
    // lens_mask: g = coverage (blended with Max, green only).
    @location(1) mask: vec4<f32>,
}

// Fraction of the pixel the disk covers, from the ray's angular distance to the silhouette:
// f is the ray's distance from the centre in radii, beta = o·d, c = |o|² − 1, px_angle the
// pixel's angular size. Rays pointing away from the star cover nothing.
fn star_coverage(f: f32, beta: f32, c: f32, px_angle: f32) -> f32 {
    if (beta >= 0.0) {
        return 0.0;
    }
    // θ_s − θ ≈ (sin θ_s − sin θ)/cos θ_s = (1 − f)/√(|o|² − 1).
    let d_angle = (1.0 - f) / sqrt(max(c, 1e-12));
    return clamp(0.5 + d_angle / px_angle, 0.0, 1.0);
}

@fragment
fn fs_main(in: StarVsOut) -> StarFsOut {
    let s = stars[in.star];
    let d = normalize(in.rel);
    let o = s.cam_unit.xyz;
    let c = s.cam_unit.w;
    let beta = dot(o, d);
    // Distance of the ray from the centre in radii, |o × d|: precise for small disks.
    let f = length(cross(o, d));
    // Angular size of a pixel at the centre of the frame.
    let px_angle = 2.0 * tan(0.5 * camera.time.z) * camera.viewport.w;
    let cov = star_coverage(f, beta, c, px_angle);
    if (cov <= 0.0) {
        discard;
    }
    // Stable ray–sphere (Haines et al. 2019, Ray Tracing Gems ch. 7): discriminant 1 − f², near
    // root c/q with q = −β + √Δ. Past the silhouette (the antialiased rim) the ray's closest
    // point stands in for the limb.
    let disc = max(1.0 - f * f, 0.0);
    let mu = sqrt(disc);
    let t = select(-beta, c / (-beta + mu), disc > 0.0);
    let hit = normalize(o + t * d);
    let radius = s.params.x;
    let cell = s.params.y;

    // Granulation, faded out as a cell shrinks below 2 px (its footprint across the
    // foreshortened direction is t·R·px_angle/μ).
    let footprint = t * radius * px_angle / max(mu, 1e-4);
    let fade = granulation_fade(cell / footprint);
    var g = vec3<f32>(1.0);
    if (fade > 0.0) {
        let p_body = vec3<f32>(dot(s.to_body_x.xyz, hit), dot(s.to_body_y.xyz, hit), dot(s.to_body_z.xyz, hit));
        let g550 = granulation(p_body, cell, radius, sim_time_s());
        let k = vec3<f32>(s.params.z, 1.0, s.params.w);
        g = max(vec3<f32>(0.0), vec3<f32>(1.0) + (g550 - 1.0) * fade * k);
    }

    let lum = s.l0.rgb * limb_darkening(mu) * g;
    let pre = mix(star_pre(), scene_pre(), s.l0.w);
    var out: StarFsOut;
    out.color = vec4<f32>(min(lum * pre, vec3<f32>(HDR_MAX)) * cov, cov);
    out.mask = vec4<f32>(0.0, cov, 0.0, 0.0);
    return out;
}
