// planet/rings.wgsl: Saturn's ring pass "rings" (Task B2).
//
// An annulus of RING_SEGMENTS quads in the ring plane (Saturn's equatorial plane), built in the
// vertex shader from the vertex index: inner vertices on the inner circle, outer ones on the
// circumscribed polygon of the outer circle, so the mesh covers the whole annulus and the
// fragment shader trims it to the exact radii. The mesh is planar, so its interpolated depth is
// the plane's exact depth. Depth-tested (reversed-Z Greater) without depth write; premultiplied
// over hdr; lens_mask.g = the opacity along the view ray, blended with Max.
//
// CPU twins (crates/uni-planet/src/rings.rs): `ring_phase`, `ring_layer_factor` and
// `planet_shadow_on_ring` (its f32 core `shadow_core`). `universe --selftest` compares all
// three on the GPU (64 cases each).

#import "common/camera.wgsl"
#import "common/constants.wgsl"
#import "planet/eclipse.wgsl"

// 256 segments (B2 brief; the performance ladder's step is 128).
const RING_SEGMENTS: u32 = 256u;

// Lit face: a two-lobe Henyey–Greenstein phase function, weight 0.8 on a backscattering lobe
// (g = −0.3, icy regolith-covered particles) and 0.2 on a forward lobe (g = +0.6, the
// diffraction of the smaller particles); unlit face: the forward lobe alone. Values as the M1
// plan gives them (B2 brief); see rings.rs for what they were checked against.
const RING_LIT_W: f32 = 0.8;
const RING_LIT_G1: f32 = -0.3;
const RING_LIT_G2: f32 = 0.6;
const RING_UNLIT_G: f32 = 0.6;

// Henyey & Greenstein 1941 (ApJ 93, 70): (1 − g²)/(1 + g² − 2g cos θ)^{3/2}, normalised so its
// mean over the sphere is 1; θ is the scattering angle (0 = forward).
fn henyey_greenstein(g: f32, cos_t: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_t;
    return (1.0 - g2) / (denom * sqrt(denom));
}

// The ring phase function at scattering-angle cosine cos_alpha (twin of uni_planet::ring_phase).
fn ring_phase(cos_alpha: f32, lit_face: bool) -> f32 {
    let c = clamp(cos_alpha, -1.0, 1.0);
    if (lit_face) {
        return RING_LIT_W * henyey_greenstein(RING_LIT_G1, c)
            + (1.0 - RING_LIT_W) * henyey_greenstein(RING_LIT_G2, c);
    }
    return henyey_greenstein(RING_UNLIT_G, c);
}

// Geometry of single scattering in a plane-parallel particle layer of normal optical depth tau,
// lit at μ₀ and seen at μ (cosines from the ring normal), times 2 so that an opaque layer seen
// from the Sun's direction gives 1 (twin of uni_planet::rings::ring_layer_factor): first-order
// scattering in a slab (Chandrasekhar 1960, Radiative Transfer), the "classical single-scattering
// model" of Saturn's rings (Cuzzi, French & Dones 2002, Icarus 158, 199):
//   lit face:   2 μ₀/(μ₀ + μ) · (1 − e^{−τ(1/μ₀ + 1/μ)})
//   unlit face: 2 μ₀/(μ₀ − μ) · (e^{−τ/μ₀} − e^{−τ/μ}),  2 (τ/μ) e^{−τ/μ} at μ = μ₀.
const RING_MU_MIN: f32 = 1e-6;
fn ring_layer_factor(tau: f32, mu0_in: f32, mu_in: f32, lit_face: bool) -> f32 {
    if (!(tau > 0.0) || !(mu0_in > 0.0)) {
        return 0.0;
    }
    let mu0 = max(mu0_in, RING_MU_MIN);
    let mu = max(mu_in, RING_MU_MIN);
    if (lit_face) {
        return 2.0 * mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu0 + 1.0 / mu)));
    }
    let e0 = exp(-tau / mu0);
    let e1 = exp(-tau / mu);
    if (abs(mu0 - mu) <= 1e-4 * mu) {
        return 2.0 * (tau / mu) * e1;
    }
    return 2.0 * mu0 / (mu0 - mu) * (e0 - e1);
}

// Fraction of the Sun visible from p (a ring point in the planet's unit-ellipsoid space: body
// coordinates divided by the radii) for the Sun direction d in that space (unit), the Sun's
// angular radius sun_ang [rad]. The planet is the unit sphere there; the penumbra comes from
// sun_visible_fraction (planet/eclipse.wgsl). Twin of uni_planet::rings::shadow_core.
fn planet_shadow_on_ring(p: vec3<f32>, d: vec3<f32>, sun_ang: f32) -> f32 {
    let r = length(p);
    if (!(r > 1.0)) {
        return 0.0;
    }
    let to_c = -p / r;
    let sep = atan2(length(cross(to_c, d)), dot(to_c, d));
    return sun_visible_fraction(sun_ang, asin(1.0 / r), sep);
}

// Twin of uni_planet::rings::RingGpu: one ringed body, 128 bytes, eight vec4s
// (star_ring_uniforms_layout_match_wgsl).
struct RingGpu {
    // xyz: planet centre − camera [m], root axes; w: the exposure blend weight (1 = scene).
    centre: vec4<f32>,
    // xyz: the body x axis (root axes); w: inner radius [m].
    axis_x: vec4<f32>,
    // xyz: the body y axis (root axes); w: outer radius [m].
    axis_y: vec4<f32>,
    // xyz: the ring normal, the body z axis (root axes); w: the equatorial radius a [m].
    normal: vec4<f32>,
    // xyz: unit direction to the Sun from the planet (root axes); w: sunlight [lx] at the planet.
    sun: vec4<f32>,
    // xyz: the Sun's direction in the unit-ellipsoid space (unit); w: the Sun's angular radius [rad].
    sun_scaled: vec4<f32>,
    // xy: a / radii.x, a / radii.y (plane coordinates in a → unit-ellipsoid space); zw: 0.
    radii: vec4<f32>,
    // rgb: the Sun's colour, linear Rec. 709 with luminance 1; w: 0.
    sun_color: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> rings: array<RingGpu>;
// The radial profile, premultiplied (ρ·α rgb, α), u = (r − inner)/(outer − inner), with mips.
@group(1) @binding(1) var ring_profile: texture_2d<f32>;
@group(1) @binding(2) var ring_sampler: sampler;

struct RingVsOut {
    @builtin(position) clip: vec4<f32>,
    // Camera-relative position [m], root axes.
    @location(0) rel: vec3<f32>,
    // Position in the ring plane along the body x and y axes, in units of a.
    @location(1) plane: vec2<f32>,
    @location(2) @interpolate(flat) ring: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> RingVsOut {
    let ring = rings[k];
    let seg = vi / 6u;
    let corner = vi % 6u;
    // Quad corners: (seg, in), (seg, out), (seg+1, out), (seg, in), (seg+1, out), (seg+1, in).
    let next = corner == 2u || corner == 4u || corner == 5u;
    let outer = corner == 1u || corner == 2u || corner == 4u;
    let phi = TAU * f32(seg + select(0u, 1u, next)) / f32(RING_SEGMENTS);
    let r = select(ring.axis_x.w, ring.axis_y.w / cos(PI / f32(RING_SEGMENTS)), outer);
    let cs = vec2<f32>(cos(phi), sin(phi));
    let rel = ring.centre.xyz + r * (cs.x * ring.axis_x.xyz + cs.y * ring.axis_y.xyz);
    var out: RingVsOut;
    out.clip = camera.proj * camera.view_rot * vec4<f32>(rel, 1.0);
    out.rel = rel;
    out.plane = cs * (r / ring.normal.w);
    out.ring = k;
    return out;
}

struct RingFsOut {
    // Premultiplied pre-exposed light; alpha = the opacity along the view ray.
    @location(0) color: vec4<f32>,
    // lens_mask: g = the same opacity (blended with Max, green only).
    @location(1) mask: vec4<f32>,
}

// Normal opacities at or above this are treated as this (τ ≤ 9.2), so τ stays finite.
const RING_ALPHA_MAX: f32 = 0.9999;
// Profile opacities at or below this are not drawn.
const RING_ALPHA_DRAWN_MIN: f32 = 1e-5;

@fragment
fn fs_main(in: RingVsOut) -> RingFsOut {
    let ring = rings[in.ring];
    let r = length(in.plane) * ring.normal.w;
    let u = (r - ring.axis_x.w) / (ring.axis_y.w - ring.axis_x.w);
    // Sampled before any branch (implicit derivatives pick the mip, so the fine structure is
    // averaged, not aliased, when a pixel spans many samples).
    let tex = textureSample(ring_profile, ring_sampler, vec2<f32>(u, 0.5));
    if (u < 0.0 || u > 1.0 || tex.a <= RING_ALPHA_DRAWN_MIN) {
        discard;
    }
    let alpha_n = min(tex.a, RING_ALPHA_MAX);
    let rho = tex.rgb / tex.a;
    // Normal optical depth from the normal opacity: α = 1 − e^{−τ}.
    let tau = -log(1.0 - alpha_n);

    let n = ring.normal.xyz;
    let v = -normalize(in.rel);
    let s = ring.sun.xyz;
    let sn = dot(s, n);
    let vn = dot(v, n);
    let lit = sn * vn > 0.0;
    let mu0 = abs(sn);
    let mu = max(abs(vn), RING_MU_MIN);
    // The sunlight travels along −s and leaves toward v.
    let phase = ring_phase(-dot(s, v), lit);
    // The single-scattering layer factor stands where the brief has the opacity factors α (lit
    // face) and α(1 − α) (unlit face): it depends on the Sun's and the camera's elevations, so
    // the rings darken at equinox and the opaque B ring is dark on its unlit face (rings.rs).
    let layer = ring_layer_factor(tau, mu0, mu, lit);
    let p_unit = vec3<f32>(in.plane * ring.radii.xy, 0.0);
    let shadow = planet_shadow_on_ring(p_unit, ring.sun_scaled.xyz, ring.sun_scaled.w);
    // L = (E/π) · ρ · P · layer [cd/m²]: I/F = ρ·P·layer (CPU twin: uni_planet::rings::ring_i_over_f).
    let radiance = ring.sun.w * shadow / PI * ring.sun_color.rgb * rho * phase * layer;
    // Opacity along the view ray (slant path τ/μ).
    let alpha_view = 1.0 - exp(-tau / mu);
    let pre = mix(star_pre(), scene_pre(), ring.centre.w);
    var out: RingFsOut;
    out.color = vec4<f32>(min(radiance * pre, vec3<f32>(HDR_MAX)), alpha_view);
    out.mask = vec4<f32>(0.0, alpha_view, 0.0, 0.0);
    return out;
}
