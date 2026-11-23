// sky/asteroids.wgsl: the asteroid pass "asteroids" (Task C2): the catalogue's asteroids (SBDB,
// H ≤ 14) that uni-sim does not draw itself, and the procedural belt, as PSF splats.
//
// One instance per orbit: instances [0, counts.x) are the real records, in catalogue order; the
// procedural objects follow. Per instance the vertex stage
//   1. takes the mean anomaly M: a real record's from the CPU (f64 M₀ + n·(t − epoch), wrapped to
//      [−π, π], uploaded as a double-single pair (hi, lo); hi = SUPPRESSED_ANOMALY for a record
//      not drawn here: the registry asteroids, which uni-sim draws from f64 states, and SBDB's
//      Pluto row), or a procedural object's M₀ + n·t in f32 (belt_anomaly; statistical objects,
//      no targeting; lo = 0);
//   2. solves Kepler's equation for hi in f32 (kepler_f32: Danby's starter capped by two upper
//      bounds of the root, then three steps of Danby's quartic correction), corrects sin E and
//      cos E to first order for the residual of the full M = hi + lo (asteroid_helio_icrs), and
//      builds the heliocentric position from the orbit's packed axes a·P and b·Q. The CPU rotated
//      those axes from the J2000 ecliptic to ICRS in f64 (the obliquity folded into the orbit,
//      one f32 rotation less per vertex);
//   3. subtracts the camera's heliocentric position, passed as a double-single pair (hi, lo)
//      ("GPU RTE": Cozzi & Ring 2011, *3D Engine Design for Virtual Globes*, ch. 5), so the
//      camera is exact to centimetres where one f32 position is off by up to half an ulp
//      (262 km at 30 au);
//   4. V from H, G at the true phase angle (Bowell et al. 1989), as illuminance
//      E = 10^(−0.4 (H + 13.99)) · Φ(α) / (r Δ)² lx with r, Δ in au (Allen 1973's zero point,
//      sky/stars3d.wgsl's star_lux), times the uniform's factor (the app's reflected-point
//      exposure factor × BELT_HIGHLIGHT_GAIN while "Highlight asteroid belt" is on);
//   5. splat_vertex: the star pass's PSF at E / Ω(θ) × star_pre(), culled below its visibility
//      cutoff (so the faint belt costs no fragments), at the true depth (compare Greater: a body
//      in front hides it).
// The fragment stage is the point pass's: the PSF integrated over the pixel × (1 − r) of
// lens_mask, and × (1 − g) as well while the camera is inside an atmosphere (point_sky_factor,
// imported from sky/points.wgsl).
//
// Precision (review I12): an f32 orbit is good to a few ulps of a (2⁻²⁴ × 3 au = 27 km per ulp),
// about 100 km in the main belt, 6 × 10⁻⁴ px at 1 au and 1080 lines over 60°. The CPU twin
// measures 94 km at most over the catalogue's 16 675 records within 3.3 au (median 23 km);
// with M in one f32 and the rotation in the shader it was 135 km.
//
// Bindings, group 1: 1 lens_mask (M0's declaration in sky/stars.wgsl, through sky/points.wgsl
// and sky/stars3d.wgsl), 8 the orbits, 9 the real records' mean anomaly pairs, 10 the uniform.
// The imports' own bindings (0, 2–7) are unused here, so no two globals share a slot.
//
// CPU twins: crates/uni-sky/src/asteroids.rs (`kepler_f32`, `split_f64`, `twin::*`); its probe
// runs under `universe --selftest`.

#import "sky/points.wgsl"

// Twin of `uni_sky::asteroids::AsteroidGpu`: 48 bytes, three vec4s.
struct AsteroidGpu {
    // xyz: a·P [m], P the unit vector to perihelion, ICRS axes; w: e.
    p_e: vec4<f32>,
    // xyz: b·Q [m], b = a√(1 − e²), Q = P turned 90° in the direction of motion, ICRS; w: H.
    q_h: vec4<f32>,
    // x: the H–G slope G; y: mean motion n [rad/s]; z: M₀ [rad] at the belt epoch; w: 0. Real
    // records: y = z = 0 (their M comes from the CPU).
    slope_motion: vec4<f32>,
}

// Twin of `uni_sky::asteroids::AsteroidParams`: 80 bytes, written once per frame.
struct AsteroidParams {
    // xyz: camera − Sun [m], ICRS, the f32 high parts of the double-single pair; w: 0.
    camera_hi: vec4<f32>,
    // xyz: the low parts (camera − Sun − hi); w: 0.
    camera_lo: vec4<f32>,
    // x: the illuminance factor; y: 1 while the camera is inside an atmosphere, else 0;
    // z: the belt time t [s since the belt epoch]; w: 0.
    look: vec4<f32>,
    // rgb: the colour of sunlight (linear Rec. 709, luminance 1); w: 0.
    colour: vec4<f32>,
    // x: real records (instances below it are real); yzw: 0.
    counts: vec4<u32>,
}

@group(1) @binding(8) var<storage, read> asteroids: array<AsteroidGpu>;
// Two mean anomalies [rad] per vec4 as double-single pairs: real record k's (hi, lo) is
// [k / 2].xy for even k and .zw for odd k.
@group(1) @binding(9) var<storage, read> real_anomalies: array<vec4<f32>>;
@group(1) @binding(10) var<uniform> asteroid_params: AsteroidParams;

// The mean anomaly of a record the pass must not draw (outside [−π, π]).
const SUPPRESSED_ANOMALY: f32 = 16.0;
// Danby's correction steps and his starter's k (see kepler_f32).
const KEPLER_STEPS: u32 = 3u;
const DANBY_K: f32 = 0.85;
// Floor of the starter's cube-root argument and of 1 − e (fast math keeps no infinities).
const KEPLER_TINY: f32 = 1e-30;
// The astronomical unit [m] (IAU 2012 Resolution B2).
const AU_M: f32 = 1.495978707e11;
// The H, G phase function's Φᵢ = exp(−Aᵢ tan^Bᵢ(α/2)) (Bowell et al. 1989, Asteroids II, 524).
const HG_A1: f32 = 3.33;
const HG_B1: f32 = 0.63;
const HG_A2: f32 = 1.87;
const HG_B2: f32 = 1.22;
// Floor of tan(α/2) before its log2 (fast math keeps no infinities): Φ(α = 0) = 1 to 1e-12.
const HG_TAN_MIN: f32 = 1e-20;

// The eccentric anomaly E [rad] for mean anomaly m ∈ [−π, π] and eccentricity e < 1, in f32.
// Danby's starter |E₀| = |M| + k e, k = 0.85, with the sign of M (Danby's sign(sin M) on
// [−π, π]; Danby 1988, *Fundamentals of Celestial Mechanics*, 2nd ed., §6.6), capped by two
// upper bounds of the root for |M| ≤ π,
// ∛(12|M|) and |M|/(1 − e) (the bounds of uni_physics::kepler's solver: E − sin E ≥
// (E³/6)(1 − π²/20) on [0, π], and sin E ≤ E), so that near perihelion on a near-parabolic orbit
// the start is already close. Then KEPLER_STEPS of Danby's quartic correction: with
// f = E − e sin E − M, δ₁ = −f/f′, δ₂ = −f/(f′ + δ₁f″/2), δ₃ = −f/(f′ + δ₂f″/2 + δ₂²f‴/6),
// E += δ₃ (Danby & Burkardt 1983, Celest. Mech. 31, 95; Murray & Dermott 1999, *Solar System
// Dynamics*, §2.4). The same sin and cos per step as Newton's method; three steps leave
// ≤ 5.5 × 10⁻¹³ of r for every e ≤ 0.9999 (f64 analysis) where three Newton steps from Danby's
// start leave 0.8 % of r at e = 0.89 (asteroids.rs, KEPLER_STEPS). Twin of
// `uni_sky::asteroids::kepler_f32`.
fn kepler_f32(m: f32, e: f32) -> f32 {
    let m_abs = abs(m);
    let start = min(
        min(m_abs + DANBY_K * e, pow(max(12.0 * m_abs, KEPLER_TINY), 1.0 / 3.0)),
        m_abs / max(1.0 - e, KEPLER_TINY));
    var ea = select(-start, start, m >= 0.0);
    for (var k = 0u; k < KEPLER_STEPS; k++) {
        let s = sin(ea);
        let c = cos(ea);
        let f = ea - e * s - m;
        let f1 = 1.0 - e * c;
        let f2 = e * s;
        let f3 = e * c;
        let d1 = -f / f1;
        let d2 = -f / (f1 + 0.5 * d1 * f2);
        let d3 = -f / (f1 + 0.5 * d2 * f2 + d2 * d2 * f3 / 6.0);
        ea += d3;
    }
    return ea;
}

// Heliocentric position [m], ICRS axes, of orbit o at the mean anomaly m = m.x + m.y [rad] (a
// double-single pair): E from kepler_f32(m.x), then one Newton step for the full M applied to
// sin E and cos E to first order, δ = −((E − m.x) − e sin E − m.y)/(1 − e cos E) (E − m.x is
// exact in f32 by Sterbenz's lemma wherever m.x/2 ≤ E ≤ 2 m.x, so the low part is not lost), and
// a(cos E − e)·P + b sin E·Q. The neglected δ²/2 is below 10⁻¹³. Twin of
// `twin::asteroid_helio_icrs`.
fn asteroid_helio_icrs(o: AsteroidGpu, m: vec2<f32>) -> vec3<f32> {
    let e = o.p_e.w;
    let ea = kepler_f32(m.x, e);
    let s = sin(ea);
    let c = cos(ea);
    let delta = -(((ea - m.x) - e * s) - m.y) / (1.0 - e * c);
    return o.p_e.xyz * ((c - s * delta) - e) + o.q_h.xyz * (s + c * delta);
}

// p − camera for a camera given as the double-single pair (hi, lo): (p − hi) − lo, so the low
// part survives the subtraction (GPU RTE). Twin of `twin::rte_offset`.
fn rte_offset(p: vec3<f32>, hi: vec3<f32>, lo: vec3<f32>) -> vec3<f32> {
    return (p - hi) - lo;
}

// A procedural object's mean anomaly at belt time t [s]: M₀ + n·t wrapped to [−π, π) in f32.
// Twin of `twin::belt_anomaly`.
fn belt_anomaly(m0: f32, n: f32, t: f32) -> f32 {
    let m = m0 + n * t;
    return m - TAU * floor(m / TAU + 0.5);
}

// tan(α/2) of the phase angle α at an object at heliocentric p seen from offset rel = object −
// camera (r = |p|, d = |rel|): α is the angle between p and rel, so with the unit vectors
// u = p/r, v = rel/d, tan(α/2) = sin α/(1 + cos α) = |u × v|/(1 + u·v). The cross product keeps
// small angles that cos α cannot resolve in f32 (an object at opposition: 1 − cos 0.08° = 10⁻⁶,
// 16 f32 spacings); the unit vectors keep |p × rel|², 10⁴⁶ m⁴ in the belt, inside f32. Negative
// when α = π (unlit). Twin of `twin::tan_half_phase`.
fn tan_half_phase(p: vec3<f32>, rel: vec3<f32>, r: f32, d: f32) -> f32 {
    let u = p / r;
    let v = rel / d;
    let denom = 1.0 + dot(u, v);
    if (!(denom > 0.0)) {
        return -1.0;
    }
    return length(cross(u, v)) / denom;
}

// The H, G phase function Φ(α) from tan(α/2) (Bowell et al. 1989; uni_physics::photometry::
// hg_phase). 0 for a negative tan_half (α = π, unlit). Twin of `twin::hg_phase`.
fn hg_phase(tan_half: f32, g: f32) -> f32 {
    if (!(tan_half >= 0.0)) {
        return 0.0;
    }
    let lt = log2(max(tan_half, HG_TAN_MIN));
    let p1 = exp(-HG_A1 * exp2(HG_B1 * lt));
    let p2 = exp(-HG_A2 * exp2(HG_B2 * lt));
    return p1 + g * (p2 - p1);
}

// Illuminance at the camera [lx] of an asteroid of absolute magnitude h and slope g at r_m from
// the Sun and d_m from the camera [m], phase tan(α/2) = tan_half: 10^(−0.4 (V + 13.99)) with
// V = H + 5 log₁₀(rΔ) − 2.5 log₁₀ Φ(α), r and Δ in au (Bowell et al. 1989), written as
// star_lux(H)·Φ/(rΔ)². Twin of `twin::asteroid_lux`.
fn asteroid_lux(h: f32, g: f32, r_m: f32, d_m: f32, tan_half: f32) -> f32 {
    let q = (r_m / AU_M) * (d_m / AU_M);
    return star_lux(h) * hg_phase(tan_half, g) / (q * q);
}

// Instance ii's mean anomaly [rad] as a pair (hi, lo); hi = SUPPRESSED_ANOMALY: not drawn.
fn asteroid_anomaly(ii: u32, o: AsteroidGpu) -> vec2<f32> {
    if (ii < asteroid_params.counts.x) {
        let pairs = real_anomalies[ii / 2u];
        return select(pairs.xy, pairs.zw, (ii & 1u) == 1u);
    }
    let m = belt_anomaly(o.slope_motion.z, o.slope_motion.y, asteroid_params.look.z);
    return vec2<f32>(m, 0.0);
}

@vertex
fn vs_asteroids(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SplatOut {
    let o = asteroids[ii];
    let m = asteroid_anomaly(ii, o);
    if (m.x >= SUPPRESSED_ANOMALY) {
        return culled_splat();
    }
    let helio = asteroid_helio_icrs(o, m);
    let rel = rte_offset(helio, asteroid_params.camera_hi.xyz, asteroid_params.camera_lo.xyz);
    let d = length(rel);
    let r = length(helio);
    if (!(d > 0.0) || !(r > 0.0)) {
        return culled_splat();
    }
    let lux = asteroid_lux(o.q_h.w, o.slope_motion.x, r, d, tan_half_phase(helio, rel, r, d))
        * asteroid_params.look.x;
    // NDC depth at the true distance, as the point pass: proj × (view rel, 1) gives near / (−z)
    // under the reversed-Z infinite projection. Behind the camera: no quad.
    let clip = camera.proj * vec4<f32>((camera.view_rot * vec4<f32>(rel, 0.0)).xyz, 1.0);
    let depth = clip.z / clip.w;
    if (!(depth > 0.0 && depth <= 1.0)) {
        return culled_splat();
    }
    return splat_vertex(vi, rel / d, lux, 1.0, asteroid_params.colour.rgb, depth);
}

@fragment
fn fs_asteroids(in: SplatOut) -> @location(0) vec4<f32> {
    let pix = floor(in.pos.xy);
    let mask = textureLoad(lens_mask, vec2<i32>(pix), 0);
    let inside = asteroid_params.look.y > 0.5;
    let rgb = in.radiance
        * (psf_weight(pix, in.centre_px, in.halo) * point_sky_factor(mask, inside));
    return vec4<f32>(min(rgb, vec3<f32>(HDR_MAX)), 0.0);
}
