//! GPU probes of uni-sky's WGSL against its CPU twins, run by `universe --selftest` (Task B4).
//!
//! - `stars3d_mag_dir`: `star_rel_pos_pc` and `star_apparent_mag` of `stars3d.wgsl` for 64
//!   packed stars (sector offsets, f16 velocities, proper motion over ±10⁵ yr) against the CPU
//!   twins: direction within 1e-6 rad, apparent magnitude within 1e-3 mag (brief).
//! - `stars3d_psf`: the PSF (review m8: M0's `psf_erf`, `pixel_coverage` and `halo_weight`,
//!   which `stars3d.wgsl` imports from M0's `stars.wgsl`, and B4's
//!   `stars3d_footprint_radius_px` with its 2-px minimum), `star_is_visible`, `star_lux`,
//!   `vivid_color`, `twinkle_factor` and the f16 decoding `f16_bits_to_f32` (exact) against
//!   their CPU twins.
//!
//! - `stars3d_halo` (Task B9, for B4L): the eye's glare halo, `stars3d_moffat`,
//!   `stars3d_halo_energy` and the windowed, renormalised `stars3d_halo_weight`, against their
//!   CPU twins over footprints of 2–64 px.
//! - `galaxy_detail` (Task B9, for B4L): `gal_band_radiance` toward the named dark clouds, the
//!   Cygnus and Aquila rifts and Baade's window with footprints from 0 (every octave of the
//!   clumpy dust and star clouds) to 0.2 rad (the octaves faded out), against
//!   `GalaxyModel::band_radiance`.
//!
//! B5's band probes are `band::band_probes`, called below.

use anyhow::Context as _;
use glam::{DVec3, Vec3, Vec4};
use uni_core::splitmix64;
use uni_data::Star3;
use uni_gfx::probe::{ProbeResult, run_compute};
use uni_gfx::{GpuContext, ShaderLib};

use crate::psf::{self, HALO_ENERGY_FRACTION};
use crate::stars3d::{
    StarGpu, footprint_radius_px, pack_stars, scintillation_index, star_apparent_mag,
    star_is_visible, star_lux, star_rel_pos_pc, twinkle_factor, twinkle_log_sigma, vivid_color,
};

/// Shader path of the magnitude-and-direction probe.
const MAG_DIR_PROBE: &str = "sky/probe_stars3d_mag_dir.wgsl";
const MAG_DIR_PROBE_WGSL: &str = r#"// sky/probe_stars3d_mag_dir.wgsl: `universe --selftest` probe of star_rel_pos_pc and
// star_apparent_mag (crates/uni-sky/src/probes.rs). Group 2: stars3d.wgsl declares groups 0 and 1.
#import "sky/stars3d.wgsl"

// Twin of `probes::StarProbeIn`.
struct StarProbeIn {
    star: StarGpu,
    // xyz: sector centre − camera [pc]; w: years since the catalogue epoch.
    rel_years: vec4<f32>,
}

@group(2) @binding(0) var<storage, read> probe_in: array<StarProbeIn>;
// xyz: unit direction camera → star; w: apparent V magnitude.
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    let p = star_rel_pos_pc(c.star, c.rel_years.xyz, c.rel_years.w);
    let d = length(p);
    probe_out[gid.x] = vec4<f32>(p / d, star_apparent_mag(c.star.offset_absmag.w, d));
}
"#;

/// Shader path of the PSF probe.
const PSF_PROBE: &str = "sky/probe_stars3d_psf.wgsl";
const PSF_PROBE_WGSL: &str = r#"// sky/probe_stars3d_psf.wgsl: `universe --selftest` probe of the star pass's PSF, visibility,
// illuminance, vivid colour and twinkle (crates/uni-sky/src/probes.rs). Group 2: stars3d.wgsl
// declares groups 0 and 1.
#import "sky/stars3d.wgsl"

// Twin of `probes::PsfProbeIn`.
struct PsfProbeIn {
    // x: edge [px] for psf_erf and pixel_coverage; y: r [px] and z: radius [px] for halo_weight;
    // w: display energy for stars3d_footprint_radius_px and star_is_visible.
    a: vec4<f32>,
    // x: halo fraction; y: slot, z: twinkle step (exact integers) and w: sigma for
    // twinkle_factor.
    b: vec4<f32>,
    // rgb: colour for vivid_color; w: V magnitude for star_lux.
    c: vec4<f32>,
    // x: binary16 bits (an exact integer) for f16_bits_to_f32; y: twinkle phase; zw: 0.
    d: vec4<f32>,
}

// Twin of `probes::PsfProbeOut`.
struct PsfProbeOut {
    // psf_erf(a.x), pixel_coverage(a.x), halo_weight(a.y, a.z),
    // stars3d_footprint_radius_px(a.w, b.x).
    a: vec4<f32>,
    // rgb: vivid_color(c.rgb); w: twinkle_factor(b.y, b.z, d.y, b.w).
    b: vec4<f32>,
    // x: star_lux(c.w); y: 1 when star_is_visible(a.w, b.x), else 0; z: f16_bits_to_f32(d.x);
    // w: 0.
    c: vec4<f32>,
}

@group(2) @binding(0) var<storage, read> probe_in: array<PsfProbeIn>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<PsfProbeOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    var out: PsfProbeOut;
    out.a = vec4<f32>(psf_erf(c.a.x), pixel_coverage(c.a.x), halo_weight(c.a.y, c.a.z),
        stars3d_footprint_radius_px(c.a.w, c.b.x));
    out.b = vec4<f32>(vivid_color(c.c.rgb),
        twinkle_factor(u32(c.b.y), u32(c.b.z), c.d.y, c.b.w));
    out.c = vec4<f32>(star_lux(c.c.w), select(0.0, 1.0, star_is_visible(c.a.w, c.b.x)),
        f16_bits_to_f32(u32(c.d.x)), 0.0);
    probe_out[gid.x] = out;
}
"#;

/// One input of the magnitude-and-direction probe: twin of `StarProbeIn` (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct StarProbeIn {
    star: StarGpu,
    /// xyz: sector centre − camera [pc]; w: years since the catalogue epoch.
    rel_years: [f32; 4],
}

/// One input of the PSF probe: twin of `PsfProbeIn` (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct PsfProbeIn {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
}

/// One output of the PSF probe: twin of `PsfProbeOut` (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct PsfProbeOut {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
}

/// Largest GPU − CPU direction difference of the magnitude-and-direction probe [rad] (brief).
const DIR_TOLERANCE_RAD: f64 = 1e-6;
/// Largest GPU − CPU apparent-magnitude difference [mag] (brief).
const MAG_TOLERANCE: f32 = 1e-3;
/// erf and pixel coverage: absolute. Both are the same A&S polynomial and one exp.
const ERF_TOLERANCE: f32 = 1e-5;
/// halo weight, illuminance and twinkle: relative (pow, exp2 and exp on the GPU).
const REL_TOLERANCE: f32 = 1e-4;
/// Footprint radius [px] (log, sqrt and pow on the GPU).
const RADIUS_TOLERANCE_PX: f32 = 1e-3;
/// vivid colour channels: absolute (colours have luminance 1).
const COLOUR_TOLERANCE: f32 = 1e-4;

/// Every uni-sky probe, registering its shaders in `lib`.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    crate::register_shaders(lib);
    lib.add(MAG_DIR_PROBE, MAG_DIR_PROBE_WGSL);
    lib.add(PSF_PROBE, PSF_PROBE_WGSL);
    lib.add(HALO_PROBE, HALO_PROBE_WGSL);
    lib.add(DETAIL_PROBE, DETAIL_PROBE_WGSL);
    lib.add(TAPER_PROBE, TAPER_PROBE_WGSL);
    lib.add(WORLD_POINTS_PROBE, WORLD_POINTS_PROBE_WGSL);
    let mut results = vec![
        result("stars3d_mag_dir", check_mag_dir(gpu, lib)),
        result("stars3d_psf", check_psf(gpu, lib)),
        result("stars3d_halo", check_halo(gpu, lib)),
        result("stars3d_halo_taper", check_taper(gpu, lib)),
        result("world_points_shape", check_world_points(gpu, lib)),
        result("galaxy_detail", check_galaxy_detail(gpu, lib)),
    ];
    results.extend(crate::band::band_probes(gpu, lib));
    // C2's asteroid probe (Task C0 hook; none until C2 lands).
    results.extend(crate::asteroids::asteroid_probes(gpu, lib));
    results
}

fn result(name: &str, outcome: anyhow::Result<(bool, String)>) -> ProbeResult {
    let (pass, detail) = outcome.unwrap_or_else(|e| (false, format!("{e:#}")));
    ProbeResult {
        name: name.to_owned(),
        pass,
        detail,
    }
}

/// Uniform in [lo, hi) from a splitmix64 stream.
fn uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
    *state = state.wrapping_add(1);
    let bits = splitmix64(*state) >> 11;
    lo + (hi - lo) * (bits as f64 / (1u64 << 53) as f64)
}

/// 64 deterministic stars, 0.3 pc to 3 kpc from a camera 120 pc from the SSB, packed as the star
/// pass packs them, each with its sector's centre − camera and a time within ±10⁵ yr.
fn mag_dir_inputs() -> Vec<StarProbeIn> {
    let mut state = 0xB4_5EED_u64;
    let camera = DVec3::new(3.7, -120.2, 45.1);
    let stars: Vec<Star3> = (0..64)
        .map(|i| {
            let dir = DVec3::new(
                uniform(&mut state, -1.0, 1.0),
                uniform(&mut state, -1.0, 1.0),
                uniform(&mut state, -1.0, 1.0),
            )
            .normalize();
            let d = 10f64.powf(uniform(&mut state, -0.5, 3.5));
            let v = [0; 3].map(|_| uniform(&mut state, -300.0, 300.0) as f32);
            let pos = (camera + dir * d).as_vec3().to_array();
            let abs_mag = uniform(&mut state, -8.0, 16.0) as f32;
            let bv = uniform(&mut state, -0.4, 2.0) as f32;
            Star3::new(pos, abs_mag, v, bv, i, 0)
        })
        .collect();
    let packed = pack_stars(&stars);
    let rel = packed.table.centres_rel(camera);
    (0..packed.gpu.len())
        .map(|slot| {
            let sector = Vec4::from(rel[packed.table.sector_index[slot] as usize]);
            let years = uniform(&mut state, -1e5, 1e5) as f32;
            StarProbeIn {
                star: packed.gpu[slot],
                rel_years: sector.truncate().extend(years).to_array(),
            }
        })
        .collect()
}

/// The CPU twin's answer for one magnitude-and-direction input.
fn mag_dir_expected(c: &StarProbeIn) -> Vec4 {
    let rel = Vec3::from_slice(&c.rel_years[..3]);
    let p = star_rel_pos_pc(&c.star, rel, c.rel_years[3]);
    let d = p.length();
    (p / d).extend(star_apparent_mag(c.star.offset_absmag[3], d))
}

fn check_mag_dir(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let inputs = mag_dir_inputs();
    let src = lib.compose(MAG_DIR_PROBE)?;
    let n = u32::try_from(inputs.len()).context("too many probe inputs")?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        bytemuck::cast_slice(&inputs),
        (inputs.len() * std::mem::size_of::<[f32; 4]>()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&bytes);
    // A short read-back must not pass on fewer stars (review r0 M4).
    if out.len() != inputs.len() {
        return Ok((
            false,
            format!("read back {} results for {n} stars", out.len()),
        ));
    }
    let (mut worst_dir, mut worst_mag) = (0.0f64, 0.0f32);
    for (c, g) in inputs.iter().zip(&out) {
        let want = mag_dir_expected(c);
        let got = Vec4::from(*g);
        // The angle between the unit vectors (|Δ| equals it to 1e-12 at these sizes); NaN fails.
        worst_dir = nan_max(
            worst_dir,
            (got.truncate() - want.truncate()).as_dvec3().length(),
        );
        worst_mag = nan_max(worst_mag, (got.w - want.w).abs());
    }
    Ok((
        worst_dir <= DIR_TOLERANCE_RAD && worst_mag <= MAG_TOLERANCE,
        format!(
            "{n} stars: direction max {worst_dir:.1e} rad (tolerance {DIR_TOLERANCE_RAD:.0e}), \
             magnitude max {worst_mag:.1e} (tolerance {MAG_TOLERANCE:.0e})"
        ),
    ))
}

/// PSF probe inputs: edges across a footprint, halo radii up to each footprint radius, display
/// energies from below the visibility cutoff to the radius cap with and without the halo,
/// Planck colours from 2000 to 40 000 K, twinkle slots, steps and phases (0 to 1) at Young's
/// sea-level zenith amplitude, V from −27 to 15, and binary16 bit patterns: ±0, subnormals,
/// normals and the largest finite value.
fn psf_inputs() -> Vec<PsfProbeIn> {
    let twinkle_sigma = twinkle_log_sigma(scintillation_index(1.0, 1.0));
    let mut inputs = Vec::new();
    let radii = [2.0, 3.0, 8.0, 20.0, 64.0];
    for i in 0..64u32 {
        let t = i as f32 / 63.0;
        let edge = -4.0 + 7.0 * t;
        let radius = radii[i as usize % radii.len()];
        let r = radius * (i % 7) as f32 / 6.0;
        let energy = 10f32.powf(-6.0 + 16.0 * t);
        let halo = if i % 2 == 0 {
            0.0
        } else {
            HALO_ENERGY_FRACTION
        };
        let kelvin = 2000.0 * 20f64.powf(f64::from(t));
        let c = uni_core::blackbody_linear_srgb(kelvin);
        let c = c / c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let sigma = if i % 5 == 0 { 0.0 } else { twinkle_sigma };
        let phase = (i % 9) as f32 / 8.0;
        let v_mag = -27.0 + 42.0 * t;
        let f16_bits = F16_PATTERNS
            .get(i as usize)
            .copied()
            .unwrap_or_else(|| (i * 1031) & 0x7bff | (i & 1) << 15);
        inputs.push(PsfProbeIn {
            a: [edge, r, radius, energy],
            b: [halo, (i * 977) as f32, (i * 31 + 5) as f32, sigma],
            c: c.extend(v_mag).to_array(),
            d: [f16_bits as f32, phase, 0.0, 0.0],
        });
    }
    inputs
}

/// The CPU twins' answer for one PSF input.
fn psf_expected(c: &PsfProbeIn) -> PsfProbeOut {
    let [edge, r, radius, energy] = c.a;
    let [halo, slot, step, sigma] = c.b;
    PsfProbeOut {
        a: [
            psf::erf(edge),
            psf::pixel_coverage(edge, psf::CORE_SIGMA_PX),
            psf::halo_weight(r, radius),
            footprint_radius_px(energy, halo),
        ],
        b: vivid_color(Vec3::from_slice(&c.c[..3]))
            .extend(twinkle_factor(slot as u32, step as u32, c.d[1], sigma))
            .to_array(),
        c: [
            star_lux(c.c[3]),
            if star_is_visible(energy, halo) {
                1.0
            } else {
                0.0
            },
            uni_data::f16_to_f32(c.d[0] as u16),
            0.0,
        ],
    }
}

/// binary16 patterns the PSF probe decodes first: +0, −0, the smallest and largest subnormals,
/// the smallest normal, 1, −2, 0.65 (the unknown B−V), 1000 and the largest finite value.
const F16_PATTERNS: [u32; 10] = [
    0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xc000, 0x3933, 0x63d0, 0x7bff,
];

fn check_psf(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let inputs = psf_inputs();
    let src = lib.compose(PSF_PROBE)?;
    let n = u32::try_from(inputs.len()).context("too many probe inputs")?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        bytemuck::cast_slice(&inputs),
        // One 48-byte `PsfProbeOut` per input (B9: the 64-byte input size here read back 85
        // outputs for 64 inputs, so `psf_worst` saw a length mismatch and reported NaN).
        (inputs.len() * std::mem::size_of::<PsfProbeOut>()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<PsfProbeOut> = bytemuck::pod_collect_to_vec(&bytes);
    let worst = psf_worst(&inputs, &out);
    let pass = worst[0] <= ERF_TOLERANCE
        && worst[1] <= REL_TOLERANCE
        && worst[2] <= RADIUS_TOLERANCE_PX
        && worst[3] <= COLOUR_TOLERANCE
        && worst[4] <= REL_TOLERANCE
        && worst[5] == 0.0
        && worst[6] == 0.0;
    Ok((
        pass,
        format!(
            "{n} inputs: erf/coverage {:.1e} (tol {ERF_TOLERANCE:.0e}), halo weight/lux/twinkle \
             rel {:.1e}/{:.1e} (tol {REL_TOLERANCE:.0e}), footprint {:.1e} px (tol \
             {RADIUS_TOLERANCE_PX:.0e}), vivid {:.1e} (tol {COLOUR_TOLERANCE:.0e}), visibility \
             mismatches {}, f16 mismatches {}",
            worst[0], worst[1], worst[4], worst[2], worst[3], worst[5], worst[6]
        ),
    ))
}

/// Worst GPU − CPU differences of the PSF probe: [erf and coverage (abs), halo weight (rel),
/// footprint (px), vivid colour (abs), illuminance and twinkle (rel), visibility mismatches,
/// f16 decoding mismatches (bit-exact)].
fn psf_worst(inputs: &[PsfProbeIn], out: &[PsfProbeOut]) -> [f32; 7] {
    let rel = |got: f32, want: f32| (got - want).abs() / want.abs().max(1e-30);
    let mut w = [0.0f32; 7];
    if out.len() != inputs.len() {
        return [f32::NAN; 7];
    }
    for (c, g) in inputs.iter().zip(out) {
        let e = psf_expected(c);
        w[0] = nan_max(
            w[0],
            nan_max((g.a[0] - e.a[0]).abs(), (g.a[1] - e.a[1]).abs()),
        );
        // halo_weight is ~1e-6 near its edge; compare it relative to its scale at the centre.
        let halo_scale = psf::halo_weight(0.0, c.a[2]);
        w[1] = nan_max(w[1], (g.a[2] - e.a[2]).abs() / halo_scale);
        w[2] = nan_max(w[2], (g.a[3] - e.a[3]).abs());
        let vivid = Vec3::from_slice(&g.b[..3]) - Vec3::from_slice(&e.b[..3]);
        w[3] = vivid.abs().to_array().into_iter().fold(w[3], nan_max);
        let lux = if e.c[0] > 0.0 {
            rel(g.c[0], e.c[0])
        } else {
            g.c[0].abs()
        };
        w[4] = nan_max(w[4], nan_max(lux, rel(g.b[3], e.b[3])));
        w[5] += (g.c[1] != e.c[1]) as u32 as f32;
        w[6] += (g.c[2].to_bits() != e.c[2].to_bits()) as u32 as f32;
    }
    w
}

/// max that propagates NaN, so a NaN anywhere fails its check.
fn nan_max<T: PartialOrd + Copy>(a: T, b: T) -> T {
    #[allow(clippy::eq_op)] // x != x is the NaN test for a generic float
    let nan = |x: T| x != x;
    if nan(a) {
        a
    } else if nan(b) || b > a {
        b
    } else {
        a
    }
}

/// Shader path of the halo probe (B9, for B4L).
const HALO_PROBE: &str = "sky/probe_stars3d_halo.wgsl";
const HALO_PROBE_WGSL: &str = r#"// sky/probe_stars3d_halo.wgsl: `universe --selftest` probe of the 3-D stars' glare halo
// (crates/uni-sky/src/probes.rs). Group 2: stars3d.wgsl declares groups 0 and 1.
#import "sky/stars3d.wgsl"

// x: r [px]; y: footprint radius R [px]; zw: 0.
@group(2) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
// x: stars3d_moffat(r); y: stars3d_halo_energy(R); z: stars3d_halo_weight(r, R, 1/E(R), 2); w: 0.
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    let e = stars3d_halo_energy(c.y);
    probe_out[gid.x] = vec4<f32>(stars3d_moffat(c.x), e,
        stars3d_halo_weight(c.x, c.y, 1.0 / e, 2.0), 0.0);
}
"#;

/// The halo probe's tolerance: relative to each value's scale (pow and division on the GPU;
/// the weight relative to the footprint's central weight).
const HALO_TOLERANCE: f32 = 1e-4;

/// 64 (r, R) pairs: footprints from the 2-px minimum to the 64-px cap (Sirius's 42.5 px at EV
/// −4 among them), r from the centre to just past the edge.
fn halo_inputs() -> Vec<[f32; 4]> {
    let radii = [2.0, 3.0, 5.0, 8.0, 13.5, 21.0, 42.5, 64.0];
    let mut out = Vec::with_capacity(64);
    for (i, &radius) in radii.iter().enumerate() {
        for k in 0..8 {
            let r = radius * (k as f32 / 7.0) * 1.05 + 0.01 * i as f32;
            out.push([r, radius, 0.0, 0.0]);
        }
    }
    out
}

/// The CPU twins' answer for one halo input: [`stars3d_moffat`](crate::stars3d::stars3d_moffat)
/// (r), `stars3d_halo_energy`(R), `stars3d_halo_weight`(r, R, 1/E(R)), 0 — the probe's output
/// layout.
fn halo_expected(c: &[f32; 4]) -> [f32; 4] {
    use crate::stars3d::{stars3d_halo_energy, stars3d_halo_weight, stars3d_moffat};
    let (r, radius) = (c[0], c[1]);
    let e = stars3d_halo_energy(radius);
    [
        stars3d_moffat(r),
        e,
        stars3d_halo_weight(r, radius, 1.0 / e, 2.0),
        0.0,
    ]
}

/// Worst GPU − CPU differences of the halo probe, each relative to its scale: [moffat (over its
/// peak), halo energy (it is ≤ 1), windowed weight (over the footprint's central weight)]. NaN
/// anywhere, or a read-back of the wrong length, gives NaN (so the check fails).
fn halo_worst(inputs: &[[f32; 4]], out: &[[f32; 4]]) -> [f32; 3] {
    use crate::stars3d::{stars3d_halo_weight, stars3d_moffat};
    if out.len() != inputs.len() {
        return [f32::NAN; 3];
    }
    let mut worst = [0.0_f32; 3];
    for (c, g) in inputs.iter().zip(out) {
        let cpu = halo_expected(c);
        let scale = [
            stars3d_moffat(0.0),
            1.0,
            stars3d_halo_weight(0.0, c[1], 1.0 / cpu[1], 2.0),
        ];
        for k in 0..3 {
            worst[k] = nan_max(worst[k], (g[k] - cpu[k]).abs() / scale[k]);
        }
    }
    worst
}

/// Whether the halo probe's worst differences are within [`HALO_TOLERANCE`] (NaN fails).
fn halo_pass(worst: &[f32; 3]) -> bool {
    worst.iter().all(|w| *w <= HALO_TOLERANCE)
}

fn check_halo(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let inputs = halo_inputs();
    let src = lib.compose(HALO_PROBE)?;
    let n = u32::try_from(inputs.len()).context("too many probe inputs")?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        bytemuck::cast_slice(&inputs),
        (inputs.len() * std::mem::size_of::<[f32; 4]>()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&bytes);
    let worst = halo_worst(&inputs, &out);
    Ok((
        halo_pass(&worst),
        format!(
            "{n} (r, R) pairs, R 2–64 px: moffat {:.1e}, halo energy {:.1e}, windowed weight \
             {:.1e} (relative to scale; tolerance {HALO_TOLERANCE:.0e})",
            worst[0], worst[1], worst[2]
        ),
    ))
}

/// Shader path of the clamped-halo probe (final review render I2).
const TAPER_PROBE: &str = "sky/probe_stars3d_taper.wgsl";
const TAPER_PROBE_WGSL: &str = r#"// sky/probe_stars3d_taper.wgsl: `universe --selftest` probe of the clamped
// halo's shape (final review render I2; crates/uni-sky/src/probes.rs). Group 2: stars3d.wgsl
// declares groups 0 and 1.
#import "sky/stars3d.wgsl"

// x: display energy; y: r [px]; zw: 0.
@group(2) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
// x: R; y: weight scale; z: window exponent n (stars3d_halo_shape(energy, h)); w: the halo
// weight at r with that shape.
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    let s = stars3d_halo_shape(c.x, STARS3D_HALO_FRACTION);
    probe_out[gid.x] = vec4<f32>(s, stars3d_halo_weight(c.y, s.x, s.y, s.z));
}
"#;

/// The clamped-halo probe's tolerance, relative (logs and pows of up to 1e30 on the GPU).
const TAPER_TOLERANCE: f32 = 1e-3;

/// 64 (energy, r) pairs: display energies from 1e2 (unclamped) to the 1e30 cap, 8 radii each
/// from the centre to the edge.
fn taper_inputs() -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(64);
    for i in 0..8 {
        let energy = 10f32.powi(2 + 4 * i);
        for k in 0..8 {
            out.push([energy.min(1.0e30), 64.5 * k as f32 / 7.0, 0.0, 0.0]);
        }
    }
    out
}

/// The CPU twins' answer for one taper input (the probe's output layout).
fn taper_expected(c: &[f32; 4]) -> [f32; 4] {
    use crate::stars3d::{STARS3D_HALO_FRACTION, halo_shape, stars3d_halo_weight};
    let s = halo_shape(c[0], STARS3D_HALO_FRACTION);
    [
        s.radius_px,
        s.scale,
        s.taper,
        stars3d_halo_weight(c[1], s.radius_px, s.scale, s.taper),
    ]
}

/// Worst relative GPU − CPU difference of the taper probe: R, scale and n each relative to
/// themselves, the weight relative to the footprint's central weight. NaN or a short read-back
/// gives NaN.
fn taper_worst(inputs: &[[f32; 4]], out: &[[f32; 4]]) -> f32 {
    use crate::stars3d::{stars3d_halo_weight, stars3d_moffat};
    if out.len() != inputs.len() {
        return f32::NAN;
    }
    let mut worst = 0.0_f32;
    for (c, g) in inputs.iter().zip(out) {
        let cpu = taper_expected(c);
        let centre =
            stars3d_halo_weight(0.0, cpu[0], cpu[1], cpu[2]).max(stars3d_moffat(0.0) * 1e-6);
        let scale = [cpu[0], cpu[1], cpu[2], centre];
        for k in 0..4 {
            worst = nan_max(worst, (g[k] - cpu[k]).abs() / scale[k].abs());
        }
    }
    worst
}

fn check_taper(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let inputs = taper_inputs();
    let src = lib.compose(TAPER_PROBE)?;
    let n = u32::try_from(inputs.len()).context("too many probe inputs")?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        bytemuck::cast_slice(&inputs),
        (inputs.len() * std::mem::size_of::<[f32; 4]>()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&bytes);
    let worst = taper_worst(&inputs, &out);
    Ok((
        worst <= TAPER_TOLERANCE,
        format!(
            "{n} (energy 1e2–1e30, r) pairs: halo radius, scale, taper and weight within \
             {worst:.1e} (relative; tolerance {TAPER_TOLERANCE:.0e})"
        ),
    ))
}

/// Shader path of the world-point probe (final review render I2): the world-light splat's
/// footprint and weight at scene exposure, through `splat_vertex_at`'s maths.
const WORLD_POINTS_PROBE: &str = "sky/probe_world_points.wgsl";
const WORLD_POINTS_PROBE_WGSL: &str = r#"// sky/probe_world_points.wgsl: `universe --selftest` probe of
// the world-light point's PSF weight (crates/uni-sky/src/probes.rs). Group 2: stars3d.wgsl
// declares groups 0 and 1.
#import "sky/stars3d.wgsl"

// x: display energy; y, z: pixel offset from the splat's centre [px] (pixel index − centre);
// w: 0.
@group(2) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
// x: psf_weight at that pixel with the energy's halo shape; yzw: 0.
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    let h = STARS3D_HALO_FRACTION;
    let s = stars3d_halo_shape(c.x, h);
    let w = psf_weight(c.yz, vec2<f32>(0.0), vec4<f32>(h, s));
    probe_out[gid.x] = vec4<f32>(w, 0.0, 0.0, 0.0);
}
"#;

/// 64 (energy, pixel) inputs: the belt view's Sun at scene exposure (≈ 7e3) and at star
/// exposure (≈ 2e11), a Sirius-like 650 and 1e6, 16 pixels each from the centre out to the edge.
fn world_point_inputs() -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(64);
    for energy in [650.0_f32, 7.0e3, 1.0e6, 2.0e11] {
        for k in 0..16 {
            let d = k as f32 * 4.3 - 0.37;
            out.push([energy, d, 0.6 * d - 0.21, 0.0]);
        }
    }
    out
}

fn check_world_points(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    use crate::stars3d::{STARS3D_HALO_FRACTION, halo_shape, psf_weight};
    let inputs = world_point_inputs();
    let src = lib.compose(WORLD_POINTS_PROBE)?;
    let n = u32::try_from(inputs.len()).context("too many probe inputs")?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        bytemuck::cast_slice(&inputs),
        (inputs.len() * std::mem::size_of::<[f32; 4]>()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&bytes);
    let h = STARS3D_HALO_FRACTION;
    let mut worst = if out.len() == inputs.len() {
        0.0_f32
    } else {
        f32::NAN
    };
    for (c, g) in inputs.iter().zip(&out) {
        let s = halo_shape(c[0], h);
        let cpu = psf_weight(glam::Vec2::new(c[1], c[2]), glam::Vec2::ZERO, h, &s);
        // Relative to the footprint's peak pixel weight.
        let peak = psf_weight(glam::Vec2::ZERO, glam::Vec2::ZERO, h, &s);
        worst = nan_max(worst, (g[0] - cpu).abs() / peak);
    }
    Ok((
        worst <= HALO_TOLERANCE,
        format!(
            "{n} pixels of 4 splats (energy 650–2e11): PSF weight within {worst:.1e} of the \
             CPU twin, relative to the peak (tolerance {HALO_TOLERANCE:.0e})"
        ),
    ))
}

/// Shader path of the galaxy-detail probe (B9, for B4L).
const DETAIL_PROBE: &str = "sky/probe_galaxy_detail.wgsl";
const DETAIL_PROBE_WGSL: &str = r#"// sky/probe_galaxy_detail.wgsl: `universe --selftest` probe of
// gal_band_radiance's detail (B4L) against `GalaxyModel::band_radiance`.
#import "procgen/galaxy.wgsl"

struct DetailRay {
    // xyz: camera, galactocentric [pc]; w: v_lim [V mag].
    cam_v_lim: vec4<f32>,
    // xyz: unit direction, galactic axes; w: footprint [rad].
    dir: vec4<f32>,
}

struct DetailIn {
    galaxy: GalaxyGpu,
    rays: array<DetailRay>,
}

@group(0) @binding(0) var<storage, read> probe_in: DetailIn;
// procgen/galaxy.wgsl reads the importer's `galaxy`; the probe fills it from its input.
var<private> galaxy: GalaxyGpu;
// rgb: band radiance [cd/m²]; w: 0.
@group(0) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in.rays)) {
        return;
    }
    galaxy = probe_in.galaxy;
    let ray = probe_in.rays[gid.x];
    let rgb = gal_band_radiance(ray.cam_v_lim.xyz, ray.dir.xyz, ray.cam_v_lim.w, ray.dir.w);
    probe_out[gid.x] = vec4<f32>(rgb, 0.0);
}
"#;

/// Relative |GPU − CPU| of the band radiance (B5 brief: ±1 %).
const DETAIL_TOLERANCE: f64 = 1e-2;

/// One ray of the galaxy-detail probe: twin of `DetailRay` in its WGSL (32 bytes, two vec4s;
/// `detail_probe_layout_matches_wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct DetailRay {
    /// xyz: camera, galactocentric [pc]; w: v_lim [V mag].
    cam_v_lim: [f32; 4],
    /// xyz: unit direction, galactic axes; w: footprint [rad].
    dir: [f32; 4],
}

/// 60 rays from the Sun at V = 10: toward the named dark clouds and the dusty band (Galactic l,
/// b [deg]: Taurus 172/−15, Ophiuchus ρ Oph 353/17, the Pipe 0/5, the Coalsack 301/−1, the
/// Cygnus Rift 80/0, the Aquila Rift 30/3, Orion 209/−19, Baade's window 1/−3.9, Chamaeleon
/// 300/−16, Lupus 339/16, Perseus 160/−20, Corona Australis 0/−18), each with footprints 0,
/// one 512² texel, 0.01, 0.05 and 0.2 rad.
fn detail_rays(model: &uni_procgen::galaxy::GalaxyModel) -> Vec<DetailRay> {
    let sun = model.sun_galactocentric_pc();
    let targets = [
        (172.0, -15.0),
        (353.0, 17.0),
        (0.0, 5.0),
        (301.0, -1.0),
        (80.0, 0.0),
        (30.0, 3.0),
        (209.0, -19.0),
        (1.0, -3.9),
        (300.0, -16.0),
        (339.0, 16.0),
        (160.0, -20.0),
        (0.0, -18.0),
    ];
    let footprints = [0.0, 2.0 / crate::band::ENV_SIZE as f32, 0.01, 0.05, 0.2];
    let mut out = Vec::with_capacity(60);
    for (l, b) in targets {
        let (l, b): (f64, f64) = (f64::to_radians(l), f64::to_radians(b));
        let d = DVec3::new(b.cos() * l.cos(), b.cos() * l.sin(), b.sin());
        for fp in footprints {
            out.push(DetailRay {
                cam_v_lim: [sun.x as f32, sun.y as f32, sun.z as f32, 10.0],
                dir: [d.x as f32, d.y as f32, d.z as f32, fp],
            });
        }
    }
    out
}

/// The galaxy-detail probe's input buffer: the model's `GalaxyGpu`, then the rays, packed as
/// `DetailIn` in the WGSL (the rays start at `size_of::<GalaxyGpu>()`;
/// `detail_probe_layout_matches_wgsl`).
fn detail_input(model: &uni_procgen::galaxy::GalaxyModel, rays: &[DetailRay]) -> Vec<u8> {
    let mut input = bytemuck::bytes_of(&model.gpu()).to_vec();
    input.extend_from_slice(bytemuck::cast_slice(rays));
    input
}

/// The CPU twin's band radiance for each ray [cd/m²] (`GalaxyModel::band_radiance`).
fn detail_expected(model: &uni_procgen::galaxy::GalaxyModel, rays: &[DetailRay]) -> Vec<DVec3> {
    rays.iter()
        .map(|r| {
            model.band_radiance(
                Vec3::from_slice(&r.cam_v_lim[..3]).as_dvec3(),
                Vec3::from_slice(&r.dir[..3]).as_dvec3(),
                f64::from(r.cam_v_lim[3]),
                f64::from(r.dir[3]),
            )
        })
        .collect()
}

/// Worst relative GPU − CPU difference of the galaxy-detail probe (each ray's largest channel
/// difference over its brightest CPU channel). NaN anywhere, or a read-back of the wrong length,
/// gives NaN (so the check fails).
fn detail_worst(expected: &[DVec3], out: &[[f32; 4]]) -> f64 {
    if out.len() != expected.len() {
        return f64::NAN;
    }
    expected.iter().zip(out).fold(0.0, |w, (cpu, g)| {
        let gpu_v = Vec3::from_slice(&g[..3]).as_dvec3();
        // glam's max_element skips NaN (f64::max), so a NaN channel is caught here.
        if !gpu_v.is_finite() {
            return f64::NAN;
        }
        nan_max(w, (gpu_v - *cpu).abs().max_element() / cpu.max_element())
    })
}

fn check_galaxy_detail(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let model = uni_procgen::galaxy::GalaxyModel::milky_way();
    let rays = detail_rays(&model);
    let n = u32::try_from(rays.len()).context("too many probe rays")?;
    let src = lib.compose(DETAIL_PROBE)?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        &detail_input(&model, &rays),
        (rays.len() * std::mem::size_of::<[f32; 4]>()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&bytes);
    let expected = detail_expected(&model, &rays);
    let worst = detail_worst(&expected, &out);
    let (lo, hi) = expected
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(lo, hi), c| {
            (lo.min(c.y), hi.max(c.y))
        });
    Ok((
        worst <= DETAIL_TOLERANCE,
        format!(
            "{n} rays (12 dusty directions × 5 footprints 0–0.2 rad), band G {lo:.2e}–{hi:.2e} \
             cd/m², max relative |GPU − CPU| {worst:.1e} (tolerance {DETAIL_TOLERANCE:.0e})"
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stars3d::tests_support::{calls, entry, func, wgsl_struct};
    use core::mem::size_of;
    use uni_gfx::validate_wgsl;
    use wgpu::naga;

    fn probe_module(path: &'static str, src: &'static str) -> naga::Module {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        lib.add(path, src);
        let composed = lib.compose(path).unwrap();
        validate_wgsl(&composed).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    #[test]
    fn probe_shaders_validate_with_matching_layouts() {
        let m = probe_module(MAG_DIR_PROBE, MAG_DIR_PROBE_WGSL);
        let (size, _, _) = wgsl_struct(&m, "StarProbeIn");
        assert_eq!(size, size_of::<StarProbeIn>());
        let m = probe_module(PSF_PROBE, PSF_PROBE_WGSL);
        for (name, rust) in [
            ("PsfProbeIn", size_of::<PsfProbeIn>()),
            ("PsfProbeOut", size_of::<PsfProbeOut>()),
        ] {
            let (size, _, vec4_only) = wgsl_struct(&m, name);
            assert_eq!(size, rust, "{name}");
            assert!(vec4_only, "{name}");
        }
        // The probe measures the 3-D footprint (2-px minimum), which its CPU answer uses, not
        // M0's 3-px one, which the module also holds (stars3d.wgsl imports M0's PSF, review r0
        // I3).
        let cs = entry(&m, "cs_main");
        assert!(calls(&cs.body, func(&m, "stars3d_footprint_radius_px")));
        assert!(!calls(&cs.body, func(&m, "footprint_radius_px")));
    }

    /// The probes' CPU answers are finite and the inputs span what the brief asks for; a perfect
    /// GPU (the CPU's own answers) passes every check.
    #[test]
    fn probe_inputs_and_expectations() {
        let inputs = mag_dir_inputs();
        assert_eq!(inputs.len(), 64);
        let mags: Vec<f32> = inputs.iter().map(|c| mag_dir_expected(c).w).collect();
        assert!(mags.iter().all(|m| m.is_finite()));
        assert!(mags.iter().cloned().fold(f32::MAX, f32::min) < 0.0);
        assert!(mags.iter().cloned().fold(f32::MIN, f32::max) > 15.0);
        for c in &inputs {
            let dir = mag_dir_expected(c).truncate();
            assert!((dir.length() - 1.0).abs() < 1e-6);
            // The twin agrees with an f64 evaluation of the same packed star to 1e-6 rad.
            let v = [c.star.vel_bv_flags[0], c.star.vel_bv_flags[1]];
            let vel = DVec3::new(
                f64::from(uni_data::f16_to_f32(v[0] as u16)),
                f64::from(uni_data::f16_to_f32((v[0] >> 16) as u16)),
                f64::from(uni_data::f16_to_f32(v[1] as u16)),
            );
            let p = Vec3::from_slice(&c.rel_years[..3]).as_dvec3()
                + Vec3::from_slice(&c.star.offset_absmag[..3]).as_dvec3()
                + vel * f64::from(crate::stars3d::PC_PER_YR_PER_KMS) * f64::from(c.rel_years[3]);
            assert!(dir.as_dvec3().angle_between(p) < 1e-6);
        }

        let inputs = psf_inputs();
        let perfect: Vec<PsfProbeOut> = inputs.iter().map(psf_expected).collect();
        assert!(
            perfect
                .iter()
                .all(|o| { [o.a, o.b, o.c].iter().flatten().all(|x| x.is_finite()) })
        );
        assert_eq!(psf_worst(&inputs, &perfect), [0.0; 7]);
        // Signed zeros, subnormals and the largest finite binary16 are among the inputs.
        let decoded: Vec<f32> = perfect.iter().map(|o| o.c[2]).collect();
        assert!(decoded.iter().any(|x| *x == 0.0 && x.is_sign_negative()));
        assert!(decoded.contains(&5.960_464_5e-8) && decoded.contains(&65504.0));
        // Both sides of the visibility cutoff and both footprint regimes are exercised.
        let visible = perfect.iter().filter(|o| o.c[1] == 1.0).count();
        assert!(visible > 0 && visible < inputs.len());
        assert!(perfect.iter().any(|o| o.a[3] == 2.0));
        assert!(perfect.iter().any(|o| o.a[3] == psf::MAX_RADIUS_PX));
        // The twinkle at both ends of a draw's phase, twinkling and steady.
        let twinkling = |phase: f32| {
            inputs
                .iter()
                .zip(&perfect)
                .any(|(c, o)| c.d[1] == phase && c.b[3] > 0.0 && o.b[3] != 1.0)
        };
        assert!(twinkling(0.0) && twinkling(1.0));
        assert!(inputs.iter().any(|c| c.b[3] == 0.0));
        // A wrong GPU fails.
        let mut wrong = perfect.clone();
        wrong[3].a[3] += 0.01;
        assert!(psf_worst(&inputs, &wrong)[2] > RADIUS_TOLERANCE_PX);
        wrong[3].a[0] = f32::NAN;
        assert!(psf_worst(&inputs, &wrong)[0].is_nan());
        wrong[5].b[1] = f32::NAN;
        assert!(psf_worst(&inputs, &wrong)[3].is_nan());
    }

    /// The two B9 probes compose and validate with naga (CLAUDE.md §3), bind only binding 0 and
    /// 1 of one group (what `run_compute` binds), and call the functions they claim to test.
    #[test]
    fn b9_probe_shaders_validate() {
        for (path, src, group, called) in [
            (
                HALO_PROBE,
                HALO_PROBE_WGSL,
                2,
                &[
                    "stars3d_moffat",
                    "stars3d_halo_energy",
                    "stars3d_halo_weight",
                ][..],
            ),
            (
                TAPER_PROBE,
                TAPER_PROBE_WGSL,
                2,
                &["stars3d_halo_shape", "stars3d_halo_weight"][..],
            ),
            (
                WORLD_POINTS_PROBE,
                WORLD_POINTS_PROBE_WGSL,
                2,
                &["stars3d_halo_shape", "psf_weight"][..],
            ),
            (
                DETAIL_PROBE,
                DETAIL_PROBE_WGSL,
                0,
                &["gal_band_radiance"][..],
            ),
        ] {
            let m = probe_module(path, src);
            let ep = m.entry_points.iter().find(|e| e.name == "cs_main").unwrap();
            assert_eq!(ep.workgroup_size, [64, 1, 1], "{path}");
            let cs = entry(&m, "cs_main");
            for f in called {
                assert!(calls(&cs.body, func(&m, f)), "{path}: no call to {f}");
            }
            let mut bindings: Vec<(u32, u32)> = m
                .global_variables
                .iter()
                .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
                .filter(|(g, _)| *g == group)
                .collect();
            bindings.sort_unstable();
            assert_eq!(bindings, [(group, 0), (group, 1)], "{path}");
        }
    }

    /// Review (B9 r0 I2): the galaxy-detail probe's packed input is `GalaxyGpu` followed by the
    /// ray array, so in the WGSL `DetailIn.rays` starts at `size_of::<GalaxyGpu>()` and each
    /// `DetailRay` is the Rust struct's 32 bytes with the same member offsets.
    #[test]
    fn detail_probe_layout_matches_wgsl() {
        use core::mem::offset_of;
        use uni_procgen::galaxy::GalaxyGpu;
        let m = probe_module(DETAIL_PROBE, DETAIL_PROBE_WGSL);
        let (size, members, vec4_only) = wgsl_struct(&m, "DetailRay");
        assert_eq!(size, size_of::<DetailRay>());
        assert!(vec4_only);
        assert_eq!(
            members,
            [
                ("cam_v_lim".to_owned(), offset_of!(DetailRay, cam_v_lim)),
                ("dir".to_owned(), offset_of!(DetailRay, dir)),
            ]
        );
        let (galaxy_size, _, _) = wgsl_struct(&m, "GalaxyGpu");
        assert_eq!(galaxy_size, size_of::<GalaxyGpu>());
        let (_, members, _) = wgsl_struct(&m, "DetailIn");
        assert_eq!(
            members,
            [
                ("galaxy".to_owned(), 0),
                ("rays".to_owned(), size_of::<GalaxyGpu>()),
            ]
        );
        // The buffer the probe uploads is exactly that: the model, then the rays.
        let model = uni_procgen::galaxy::GalaxyModel::milky_way();
        let rays = detail_rays(&model);
        let input = detail_input(&model, &rays);
        assert_eq!(
            input.len(),
            size_of::<GalaxyGpu>() + rays.len() * size_of::<DetailRay>()
        );
        assert_eq!(
            &input[size_of::<GalaxyGpu>()..size_of::<GalaxyGpu>() + size_of::<DetailRay>()],
            bytemuck::bytes_of(&rays[0])
        );
        // The halo probe's buffers are plain vec4 arrays: 16 bytes a slot.
        assert_eq!(size_of::<[f32; 4]>(), 16);
    }

    /// The halo probe: inputs span the footprints and radii it claims; a perfect GPU (the CPU's
    /// own answers) passes, and a wrong, NaN or short read-back fails.
    #[test]
    fn halo_probe_expectations() {
        let inputs = halo_inputs();
        assert_eq!(inputs.len(), 64);
        let radii: Vec<f32> = inputs.iter().map(|c| c[1]).collect();
        assert_eq!(radii.iter().cloned().fold(f32::MAX, f32::min), 2.0);
        assert_eq!(radii.iter().cloned().fold(f32::MIN, f32::max), 64.0);
        assert!(radii.contains(&42.5)); // Sirius at EV −4 (B4L)
        // r from the centre to past the footprint's edge, where the weight is 0.
        assert!(inputs.iter().any(|c| c[0] < 0.1));
        assert!(inputs.iter().any(|c| c[0] > c[1]));
        let perfect: Vec<[f32; 4]> = inputs.iter().map(halo_expected).collect();
        assert!(perfect.iter().flatten().all(|x| x.is_finite()));
        assert!(perfect.iter().any(|o| o[2] == 0.0) && perfect.iter().any(|o| o[2] > 0.0));
        // The windowed energy tends to 1 for a wide footprint.
        assert!(perfect.iter().all(|o| o[1] > 0.0 && o[1] < 1.0));
        let w = halo_worst(&inputs, &perfect);
        assert_eq!(w, [0.0; 3]);
        assert!(halo_pass(&w));
        // A wrong GPU fails, channel by channel.
        for k in 0..3 {
            let mut wrong = perfect.clone();
            wrong[10][k] *= 1.01;
            wrong[10][k] += 1e-3;
            let w = halo_worst(&inputs, &wrong);
            assert!(
                w[k] > HALO_TOLERANCE && !halo_pass(&w),
                "channel {k}: {w:?}"
            );
        }
        let mut wrong = perfect.clone();
        wrong[3][2] = f32::NAN;
        assert!(halo_worst(&inputs, &wrong)[2].is_nan());
        assert!(!halo_pass(&halo_worst(&inputs, &wrong)));
        assert!(
            halo_worst(&inputs, &perfect[..63])
                .iter()
                .all(|x| x.is_nan())
        );
        assert!(!halo_pass(&halo_worst(&inputs, &perfect[..63])));
    }

    /// The galaxy-detail probe: 12 directions × 5 footprints, all finite and lit; a perfect GPU
    /// passes, and a wrong, NaN or short read-back fails.
    #[test]
    fn galaxy_detail_probe_expectations() {
        let model = uni_procgen::galaxy::GalaxyModel::milky_way();
        let rays = detail_rays(&model);
        assert_eq!(rays.len(), 60);
        let footprints: std::collections::BTreeSet<u32> =
            rays.iter().map(|r| r.dir[3].to_bits()).collect();
        assert_eq!(footprints.len(), 5);
        assert!(rays.iter().any(|r| r.dir[3] == 0.0) && rays.iter().any(|r| r.dir[3] == 0.2));
        assert!(
            rays.iter()
                .all(|r| (Vec3::from_slice(&r.dir[..3]).length() - 1.0).abs() < 1e-6)
        );
        let expected = detail_expected(&model, &rays);
        assert!(
            expected
                .iter()
                .all(|c| c.is_finite() && c.max_element() > 0.0)
        );
        // The footprint matters: the detail's octaves fade, so a direction's radiance changes
        // between footprint 0 and 0.2 rad somewhere among the 12.
        assert!(
            expected
                .chunks(5)
                .any(|c| (c[0].y - c[4].y).abs() > 1e-3 * c[0].y)
        );
        let perfect: Vec<[f32; 4]> = expected
            .iter()
            .map(|c| [c.x as f32, c.y as f32, c.z as f32, 0.0])
            .collect();
        let w = detail_worst(&expected, &perfect);
        assert!(w < 1e-6, "{w}");
        let mut wrong = perfect.clone();
        wrong[7][1] *= 1.05;
        assert!(detail_worst(&expected, &wrong) > DETAIL_TOLERANCE);
        wrong[7][1] = f32::NAN;
        assert!(detail_worst(&expected, &wrong).is_nan());
        assert!(detail_worst(&expected, &perfect[..59]).is_nan());
    }
}
