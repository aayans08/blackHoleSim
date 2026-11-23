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
    let mut results = vec![
        result("stars3d_mag_dir", check_mag_dir(gpu, lib)),
        result("stars3d_psf", check_psf(gpu, lib)),
    ];
    results.extend(crate::band::band_probes(gpu, lib));
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
        std::mem::size_of_val(inputs.as_slice()) as u64,
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
}
