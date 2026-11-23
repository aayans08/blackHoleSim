//! GPU probes of uni-planet's WGSL against its CPU twins, run by `universe --selftest` (Task B1).
//!
//! B1's probes: `ray_ellipsoid` (256 rays, including a camera 1 m above Earth), `equirect_uv`
//! (256), `sun_visible_fraction` (256, B0's twin in `planet/eclipse.wgsl`), `reflectance` (64,
//! the lunar-Lambert twin), `ring_transmittance` (64), `flux_match` (64, ruling W-B1's handover
//! factor, review I3), `pre_mix` (64, uni-gfx's log-space exposure blend that body.wgsl, the
//! rings, the star surfaces, the atmosphere and the meter use; task B11), `bodies_disk_flux`
//! (the whole bodies pass renders an unmapped Lambert sphere into a 256² target; Σ L·Ω_pixel
//! against `disk_illuminance_lux` within 2 %) and
//! `bc_matches_fallback` (Earth's BC7 mip of width 2048 decoded on the GPU against the 2k
//! uncompressed copy over a 256 × 256 window: PSNR ≥ 38 dB; "skipped: no BC" without
//! `TEXTURE_COMPRESSION_BC`). B2's probes are its own modules'
//! `star_probes` and `ring_probes`, called below.

use std::f64::consts::PI;
use std::path::Path;

use anyhow::{Context as _, anyhow};
use glam::{DQuat, DVec3, Mat4, UVec2, Vec2, Vec3};
use uni_core::{CameraView, Epoch, UniversePos};
use uni_data::{MapKind, TexFormat, load_map_index, load_tex};
use uni_gfx::frame::pre_mix;
use uni_gfx::probe::{ProbeResult, run_compute};
use uni_gfx::{
    Exposure, FrameCtx, GpuContext, Renderer, ShaderLib, SharedInput, exposure_scale, read_hdr,
    reversed_z_infinite_proj,
};
use uni_physics::photometry::{Brdf, disk_illuminance_lux, reflectance, rho_from_geometric_albedo};
use uni_sim::snapshot::{BodyView, proj_radius_px};

use crate::body::{BodyRenderer, BodyScene, FLUX_MATCH_MAX, flux_match, ring_crossing_radius};
use crate::{equirect_uv, ray_ellipsoid, ring_transmittance, sun_visible_fraction};

/// Every uni-planet probe, registering its shaders in `lib`.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    crate::register_shaders(lib);
    lib.add(PROBE_PATH, PROBE_WGSL);
    let mut results = vec![
        probe("ray_ellipsoid", check_ray_ellipsoid(gpu, lib)),
        probe("equirect_uv", check_equirect_uv(gpu, lib)),
        probe("sun_visible_fraction", check_sun_visible_fraction(gpu, lib)),
        probe("reflectance", check_reflectance(gpu, lib)),
        probe("ring_transmittance", check_ring_transmittance(gpu, lib)),
        probe("flux_match", check_flux_match(gpu, lib)),
        probe("pre_mix", check_pre_mix(gpu, lib)),
        probe("bodies_disk_flux", check_disk_flux(gpu, lib)),
        probe("bc_matches_fallback", check_bc_matches_fallback(gpu)),
    ];
    results.extend(crate::star_surface::star_probes(gpu, lib));
    results.extend(crate::rings::ring_probes(gpu, lib));
    // C1's comet probes (Task C0 hook; none until C1 lands).
    results.extend(crate::comet::comet_probes(gpu, lib));
    results
}

/// A `ProbeResult` from a check's (pass, detail), or its error as a failure.
fn probe(name: &str, outcome: anyhow::Result<(bool, String)>) -> ProbeResult {
    let (pass, detail) = outcome.unwrap_or_else(|e| (false, format!("{e:#}")));
    ProbeResult {
        name: name.to_owned(),
        pass,
        detail,
    }
}

/// Shader path of the twin probes.
const PROBE_PATH: &str = "probe/planet_body.wgsl";

/// One compute entry point per twin; each reads vec4s from `probe_in` and writes one vec4 per
/// case. Group 3, so the imported body.wgsl's groups 0–2 (unused by these entry points) do not
/// collide.
const PROBE_WGSL: &str = r#"// probe/planet_body.wgsl: `universe --selftest` probes of planet/body.wgsl (uni-planet B1).
#import "planet/body.wgsl"

@group(3) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(3) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

// In: (o.xyz, c), (d.xyz, 0). Out: (t0, t1, hit, 0).
@compute @workgroup_size(64)
fn cs_ray_ellipsoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_out)) {
        return;
    }
    let a = probe_in[2u * gid.x];
    let b = probe_in[2u * gid.x + 1u];
    probe_out[gid.x] = vec4<f32>(ray_ellipsoid(a.xyz, b.xyz, a.w), 0.0);
}

// In: (lon [rad], lat [rad], lon0 [deg], 0). Out: (u, v, 0, 0).
@compute @workgroup_size(64)
fn cs_equirect_uv(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_out)) {
        return;
    }
    let a = probe_in[gid.x];
    probe_out[gid.x] = vec4<f32>(equirect_uv(a.x, a.y, a.z), 0.0, 0.0);
}

// In: (sun radius, occluder radius, separation [rad], 0). Out: (fraction, 0, 0, 0).
@compute @workgroup_size(64)
fn cs_sun_visible_fraction(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_out)) {
        return;
    }
    let a = probe_in[gid.x];
    probe_out[gid.x] = vec4<f32>(sun_visible_fraction(a.x, a.y, a.z), 0.0, 0.0, 0.0);
}

// In: (L, mu0, mu, 0). Out: (R(mu0, mu), 0, 0, 0).
@compute @workgroup_size(64)
fn cs_reflectance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_out)) {
        return;
    }
    let a = probe_in[gid.x];
    probe_out[gid.x] = vec4<f32>(lunar_lambert(a.x, a.y, a.z), 0.0, 0.0, 0.0);
}

// The probe's ring opacity (the renderer's comes from a texture): 3.6 x (1 - x), x the
// fraction across the annulus. Twin of `probe_opacity` in probes.rs.
fn probe_opacity(r: f32, inner: f32, outer: f32) -> f32 {
    let x = (r - inner) / (outer - inner);
    return 3.6 * x * (1.0 - x);
}

// In: (p.xyz, inner), (s.xyz, outer). Out: (transmittance, 0, 0, 0).
@compute @workgroup_size(64)
fn cs_ring_transmittance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_out)) {
        return;
    }
    let a = probe_in[2u * gid.x];
    let b = probe_in[2u * gid.x + 1u];
    let r = ring_crossing_radius(a.xyz, b.xyz, a.w, b.w);
    let t = select(1.0, 1.0 - probe_opacity(r, a.w, b.w), r >= 0.0);
    probe_out[gid.x] = vec4<f32>(t, 0.0, 0.0, 0.0);
}

// In: (w, A, B, k_max), (sigma, 0, 0, 0). Out: (k, 0, 0, 0).
@compute @workgroup_size(64)
fn cs_flux_match(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_out)) {
        return;
    }
    let h = probe_in[2u * gid.x];
    let s = probe_in[2u * gid.x + 1u];
    probe_out[gid.x] = vec4<f32>(flux_match(h, s.x), 0.0, 0.0, 0.0);
}

// In: (a, b, t, 0). Out: (pre_mix(a, b, t), 0, 0, 0) (common/exposure.wgsl, task B11).
@compute @workgroup_size(64)
fn cs_pre_mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_out)) {
        return;
    }
    let a = probe_in[gid.x];
    probe_out[gid.x] = vec4<f32>(pre_mix(a.x, a.y, a.z), 0.0, 0.0, 0.0);
}
"#;

/// Twin of `probe_opacity` in [`PROBE_WGSL`].
fn probe_opacity(r: f32, inner: f32, outer: f32) -> f32 {
    let x = (r - inner) / (outer - inner);
    3.6 * x * (1.0 - x)
}

/// Runs entry point `entry` of the twin probe over `input` (`per_case` vec4s per case) and
/// returns one vec4 per case.
fn run_twin(
    gpu: &GpuContext,
    lib: &ShaderLib,
    entry: &str,
    input: &[[f32; 4]],
    per_case: usize,
) -> anyhow::Result<Vec<[f32; 4]>> {
    let src = lib.compose(PROBE_PATH)?;
    let cases = input.len() / per_case;
    let n = u32::try_from(cases).context("too many probe cases")?;
    let bytes = run_compute(
        gpu,
        &src,
        entry,
        bytemuck::cast_slice(input),
        (cases * size_of::<[f32; 4]>()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    Ok(bytemuck::pod_collect_to_vec(&bytes))
}

/// The worst of `errors`, NaN if any is NaN (a NaN fails every check).
fn worst(errors: impl IntoIterator<Item = f64>) -> f64 {
    errors.into_iter().fold(0.0_f64, |a, e| {
        if a.is_nan() || e.is_nan() {
            f64::NAN
        } else {
            a.max(e)
        }
    })
}

/// WGS 84 equatorial and polar radii [m] (NIMA TR8350.2, 3rd ed., 2000, Table 3.1).
const EARTH_RADII: DVec3 = DVec3::new(6_378_137.0, 6_378_137.0, 6_356_752.314_2);

/// One ray of the `ray_ellipsoid` probe: (o.xyz, c) and (d.xyz, 0), unit-sphere space, f32
/// from f64 (c computed in f64, as the renderer does).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RayCase {
    pub o: [f32; 4],
    pub d: [f32; 4],
}

/// The fractional part of `i`·`step`, a low-discrepancy sequence for the probe inputs.
fn frac_seq(i: usize, step: f64) -> f64 {
    (i as f64 * step).fract()
}

/// 256 rays at Earth's ellipsoid: 32 from each of eight cameras (1 m, 100 m, 10 km and 1000 km
/// up, 3 and 60 radii away, inside, the centre). The surface cameras include the nadir and 45°
/// rays; the rest alternate between rays aimed at spread-out surface points and directions
/// spread over the sphere. Rays within 1e-3 of grazing (|1 − |l|²| ≤ 1e-3, l the ray's offset
/// from the centre in unit-sphere space) are left out: there a 1-ulp difference may turn a hit
/// into a miss.
pub(crate) fn ray_inputs() -> Vec<RayCase> {
    let radii = EARTH_RADII;
    let (lat, lon) = (37f64.to_radians(), 20f64.to_radians());
    let on_sphere = DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
    let normal = (on_sphere / radii).normalize();
    let surface = on_sphere * radii;
    let east = DVec3::Z.cross(normal).normalize();
    let up = |h: f64| (surface + normal * h) / radii;
    let cameras = [
        up(1.0),
        up(100.0),
        up(1.0e4),
        up(1.0e6),
        DVec3::new(-2.0, 2.0, 1.0),
        DVec3::new(30.0, -40.0, 30.0),
        DVec3::new(0.3, -0.2, 0.5),
        DVec3::ZERO,
    ];
    let mut cases = Vec::with_capacity(256);
    for (k, &o) in cameras.iter().enumerate() {
        let c = o.length_squared() - 1.0;
        let mut dirs = Vec::new();
        if k < 4 {
            dirs.push(-normal);
            dirs.push((east - normal).normalize());
        }
        let mut i = 0;
        while dirs.len() < 32 {
            i += 1;
            let z = 2.0 * frac_seq(i + 97 * k, 0.618_033_988_7) - 1.0;
            let phi = 2.0 * PI * frac_seq(i + 31 * k, 0.754_877_666_2);
            let unit = DVec3::new(
                (1.0 - z * z).sqrt() * phi.cos(),
                (1.0 - z * z).sqrt() * phi.sin(),
                z,
            );
            // Aimed at a surface point, or a direction over the sphere.
            let dir = if i % 2 == 0 {
                (unit * radii - o * radii).normalize()
            } else {
                unit
            };
            let du = (dir / radii).normalize();
            let l = o - du * o.dot(du);
            if (1.0 - l.length_squared()).abs() > 1e-3 && dir.is_finite() {
                dirs.push(dir);
            }
        }
        for dir in dirs {
            let d = dir / radii;
            cases.push(RayCase {
                o: [o.x as f32, o.y as f32, o.z as f32, c as f32],
                d: [d.x as f32, d.y as f32, d.z as f32, 0.0],
            });
        }
    }
    cases
}

/// GPU `ray_ellipsoid` against the CPU twin: the same hit or miss, and each root within
/// 1e-4 relative.
fn check_ray_ellipsoid(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let cases = ray_inputs();
    let input: Vec<[f32; 4]> = cases.iter().flat_map(|r| [r.o, r.d]).collect();
    let out = run_twin(gpu, lib, "cs_ray_ellipsoid", &input, 2)?;
    let mut mismatched = 0;
    let errors = cases.iter().zip(&out).map(|(r, g)| {
        let o = Vec3::from_slice(&r.o[..3]);
        let d = Vec3::from_slice(&r.d[..3]);
        match (ray_ellipsoid(o, d, r.o[3]), g[2] > 0.5) {
            (Some((t0, t1)), true) => {
                let rel = |cpu: f32, gpu: f32| {
                    f64::from((cpu - gpu).abs()) / f64::from(cpu.abs().max(1e-6))
                };
                rel(t0, g[0]).max(rel(t1, g[1]))
            }
            (None, false) => 0.0,
            _ => {
                mismatched += 1;
                f64::INFINITY
            }
        }
    });
    let worst = worst(errors);
    const TOLERANCE: f64 = 1e-4;
    Ok((
        worst <= TOLERANCE,
        format!(
            "{} rays (1 m altitude to 60 radii, inside, centre): {mismatched} hit/miss \
             mismatches, max relative |GPU − CPU| of t {worst:.1e} (tolerance {TOLERANCE:.0e})",
            cases.len()
        ),
    ))
}

/// 256 (lon, lat, lon0) triples away from the seam.
fn equirect_inputs() -> Vec<[f32; 4]> {
    (0..256)
        .map(|i| {
            let lon = -3.1 + 6.2 * frac_seq(i, 0.618_033_988_7);
            let lat = -1.55 + 3.1 * frac_seq(i + 7, 0.381_966_011_3);
            let lon0 = [-180.0, 0.0, 90.0, -90.0][i % 4];
            [lon as f32, lat as f32, lon0, 0.0]
        })
        .collect()
}

/// GPU `equirect_uv` against the CPU twin, u compared modulo 1.
fn check_equirect_uv(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let input = equirect_inputs();
    let out = run_twin(gpu, lib, "cs_equirect_uv", &input, 1)?;
    let worst = worst(input.iter().zip(&out).map(|(a, g)| {
        let cpu = equirect_uv(a[0], a[1], a[2]);
        let du = (cpu.x - g[0]).abs();
        f64::from(du.min(1.0 - du).max((cpu.y - g[1]).abs()))
    }));
    const TOLERANCE: f64 = 2e-6;
    Ok((
        worst <= TOLERANCE,
        format!(
            "{} points, max |GPU − CPU| {worst:.1e} (tolerance {TOLERANCE:.0e})",
            input.len()
        ),
    ))
}

/// The Sun's angular radius from 1 AU [rad]: R☉/AU (IAU 2015 B3 nominal radius, IAU 2012 B2 au).
const SUN_FROM_1AU: f64 = 6.957e8 / 1.495_978_707e11;

/// 256 eclipse geometries: occluder/Sun ratios 0.1–3, Sun radii from 0.5 to 2 × the Sun's from
/// 1 AU, separations from 0 to 1.2 × the touching distance (total, annular, partial and apart).
fn eclipse_inputs() -> Vec<[f32; 4]> {
    let ratios = [0.1, 0.5, 0.97, 1.0, 1.07, 3.0, 0.3, 1.5];
    (0..256)
        .map(|i| {
            let rs = SUN_FROM_1AU * (0.5 + 1.5 * frac_seq(i, 0.381_966_011_3));
            let ro = rs * ratios[i % ratios.len()];
            let sep = 1.2 * (rs + ro) * (i / ratios.len()) as f64 / 31.0;
            [rs as f32, ro as f32, sep as f32, 0.0]
        })
        .collect()
}

/// GPU `sun_visible_fraction` (B0's `planet/eclipse.wgsl`) against its CPU twin.
fn check_sun_visible_fraction(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let input = eclipse_inputs();
    let out = run_twin(gpu, lib, "cs_sun_visible_fraction", &input, 1)?;
    let worst = worst(
        input
            .iter()
            .zip(&out)
            .map(|(a, g)| f64::from((sun_visible_fraction(a[0], a[1], a[2]) - g[0]).abs())),
    );
    // B0's accuracy bound against the numerical integral (eclipse_fraction_cases).
    const TOLERANCE: f64 = 1e-4;
    Ok((
        worst <= TOLERANCE,
        format!(
            "{} geometries (total, annular, partial, apart), max |GPU − CPU| {worst:.1e} \
             (tolerance {TOLERANCE:.0e})",
            input.len()
        ),
    ))
}

/// 64 (L, μ₀, μ): L ∈ {0, 0.5, 0.9, 1}, μ₀ and μ each over unlit/back-facing, grazing and lit.
fn reflectance_inputs() -> Vec<[f32; 4]> {
    let mut input = Vec::with_capacity(64);
    for l in [0.0, 0.5, 0.9, 1.0] {
        for mu0 in [-0.1, 0.0, 0.2, 0.7] {
            for mu in [-0.1, 0.05, 0.5, 1.0] {
                input.push([l, mu0, mu, 0.0]);
            }
        }
    }
    input
}

/// GPU `lunar_lambert` against `uni_physics::photometry::reflectance` (f64).
fn check_reflectance(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let input = reflectance_inputs();
    let out = run_twin(gpu, lib, "cs_reflectance", &input, 1)?;
    let worst = worst(input.iter().zip(&out).map(|(a, g)| {
        let brdf = if a[0] == 0.0 {
            Brdf::Lambert
        } else {
            Brdf::LunarLambert { l: f64::from(a[0]) }
        };
        (reflectance(brdf, f64::from(a[1]), f64::from(a[2])) - f64::from(g[0])).abs()
    }));
    const TOLERANCE: f64 = 1e-6;
    Ok((
        worst <= TOLERANCE,
        format!(
            "{} cases, max |GPU − CPU| {worst:.1e} (tolerance {TOLERANCE:.0e})",
            input.len()
        ),
    ))
}

/// Inner and outer radius [m] of the ring probe's annulus (about Saturn's C to A ring).
const PROBE_RING: (f32, f32) = (7.0e7, 1.4e8);

/// 64 globe points of Saturn (latitudes ±40°) with Suns 5°–40° above or below the ring plane;
/// none whose crossing lies within 1 km of the annulus edges.
fn ring_inputs() -> Vec<[f32; 4]> {
    let (inner, outer) = PROBE_RING;
    let mut input = Vec::with_capacity(128);
    let mut i = 0;
    while input.len() < 128 {
        i += 1;
        let lat = (-40.0 + 80.0 * frac_seq(i, 0.618_033_988_7)).to_radians();
        let lon = 2.0 * PI * frac_seq(i, 0.754_877_666_2);
        let r = 60_268.0e3 * (1.0 - 0.098 * lat.sin().powi(2));
        let p = DVec3::new(
            r * lat.cos() * lon.cos(),
            r * lat.cos() * lon.sin(),
            r * lat.sin(),
        );
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let elev = sign * (5.0 + 35.0 * frac_seq(i, 0.381_966_011_3)).to_radians();
        let az = lon + (frac_seq(i, 0.569_840_290_9) - 0.5) * 1.5;
        let s = DVec3::new(elev.cos() * az.cos(), elev.cos() * az.sin(), elev.sin());
        let t = -p.z / s.z;
        let cross = (p.truncate() + t * s.truncate()).length();
        let near_edge = t > 0.0
            && ((cross - f64::from(inner)).abs() < 1e3 || (cross - f64::from(outer)).abs() < 1e3);
        if !near_edge {
            input.push([p.x as f32, p.y as f32, p.z as f32, inner]);
            input.push([s.x as f32, s.y as f32, s.z as f32, outer]);
        }
    }
    input
}

/// GPU `ring_crossing_radius` with the probe's opacity against the CPU `ring_transmittance`.
fn check_ring_transmittance(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let input = ring_inputs();
    let out = run_twin(gpu, lib, "cs_ring_transmittance", &input, 2)?;
    let (inner, outer) = PROBE_RING;
    let mut shaded = 0;
    let worst = worst(input.chunks_exact(2).zip(&out).map(|(pair, g)| {
        let p = Vec3::from_slice(&pair[0][..3]);
        let s = Vec3::from_slice(&pair[1][..3]);
        let cpu = ring_transmittance(p, s, inner, outer, |r| probe_opacity(r, inner, outer));
        if ring_crossing_radius(p, s, inner, outer).is_some() {
            shaded += 1;
        }
        f64::from((cpu - g[0]).abs())
    }));
    const TOLERANCE: f64 = 1e-5;
    Ok((
        worst <= TOLERANCE,
        format!(
            "{} globe points ({shaded} under the annulus), max |GPU − CPU| {worst:.1e} \
             (tolerance {TOLERANCE:.0e})",
            out.len()
        ),
    ))
}

/// 64 handover cases ((w, A, B, k_max), (σ, 0, 0, 0)): w across the band, A and B from the
/// bodies' measured shares (Earth 2.864/0.405, Mars 2.463/1.622) and extremes that clamp k to 0
/// and to k_max, and σ = 1, 1/2, 1/16 and 1/256. body.wgsl passes σ = 1 since task B10 (the
/// atmosphere draws at its body's own `pre_mix` exposure, in either exposure mode); the other
/// values keep the general formula's σ term checked (σ = pre_atmo/pre for an atmosphere drawn 1,
/// 4 or 8 stops below the body's exposure, B1's pre-B10 case).
fn flux_match_inputs() -> Vec<[f32; 4]> {
    let mut input = Vec::with_capacity(128);
    for w in [0.0, 0.3, 0.97, 1.0] {
        for (a, b) in [(1.0, 0.0), (2.864, 0.405), (2.463, 1.622), (40.0, 50.0)] {
            for sigma in [1.0, 0.5, 1.0 / 16.0, 1.0 / 256.0] {
                input.push([w, a, b, FLUX_MATCH_MAX]);
                input.push([sigma, 0.0, 0.0, 0.0]);
            }
        }
    }
    input
}

/// GPU `flux_match` against its CPU twin.
fn check_flux_match(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let input = flux_match_inputs();
    let out = run_twin(gpu, lib, "cs_flux_match", &input, 2)?;
    let worst = worst(
        input
            .chunks_exact(2)
            .zip(&out)
            .map(|(c, g)| f64::from((flux_match(c[0], c[1][0]) - g[0]).abs())),
    );
    const TOLERANCE: f64 = 2e-5;
    Ok((
        worst <= TOLERANCE,
        format!(
            "{} cases, max |GPU − CPU| {worst:.1e} (tolerance {TOLERANCE:.0e})",
            out.len()
        ),
    ))
}

/// 64 `pre_mix` cases (a, b, t, 0) (task B11): 8 pairs of pre-exposures (`exposure_scale` of
/// star EV −4 against scene EV −4, 0, 10 and 15, both directions, and EV −10 → 40, the meter's
/// widest span) × t at both ends (returned unchanged, so compared bit for bit), next to them
/// (2⁻¹⁴ and 1 − 2⁻¹⁴, the star-share snap) and inside the band (w 0.05, 0.2076 at 2 px, ½,
/// 0.875).
fn pre_mix_inputs() -> Vec<[f32; 4]> {
    let mut input = Vec::with_capacity(64);
    for (ea, eb) in [
        (-4.0, 10.0),
        (-4.0, 15.0),
        (-4.0, 0.0),
        (-4.0, -4.0),
        (10.0, -4.0),
        (15.0, -4.0),
        (0.0, -4.0),
        (-10.0, 40.0),
    ] {
        let (a, b) = (exposure_scale(ea), exposure_scale(eb));
        for t in [
            0.0,
            2f32.powi(-14),
            0.05,
            0.2076,
            0.5,
            0.875,
            1.0 - 2f32.powi(-14),
            1.0,
        ] {
            input.push([a, b, t, 0.0]);
        }
    }
    input
}

/// GPU `pre_mix` (uni-gfx `common/exposure.wgsl`, imported through `body.wgsl`) against its CPU
/// twin `uni_gfx::frame::pre_mix`: the ends bit for bit, the rest within
/// [`PRE_MIX_TOLERANCE_STOPS`] (the GPU's exp2 and log2 may be less exact than the CPU's).
fn check_pre_mix(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let input = pre_mix_inputs();
    let out = run_twin(gpu, lib, "cs_pre_mix", &input, 1)?;
    let mut ends_exact = true;
    let worst = worst(input.iter().zip(&out).map(|(c, g)| {
        let cpu = pre_mix(c[0], c[1], c[2]);
        if c[2] == 0.0 || c[2] == 1.0 {
            ends_exact &= cpu.to_bits() == g[0].to_bits();
        }
        (f64::from(g[0]) / f64::from(cpu)).log2().abs()
    }));
    Ok((
        ends_exact && worst <= PRE_MIX_TOLERANCE_STOPS,
        format!(
            "{} cases, ends exact: {ends_exact}, max |log₂(GPU/CPU)| {worst:.1e} stops \
             (tolerance {PRE_MIX_TOLERANCE_STOPS:.0e})",
            out.len()
        ),
    ))
}

/// Tolerance of the `pre_mix` probe [stops]: a 50-stop log₂ to a few f32 ulps (4·2⁻¹⁸ ≈
/// 1.5·10⁻⁵) with room for a fast-math exp2; a linear mix instead misses by stops.
const PRE_MIX_TOLERANCE_STOPS: f64 = 2.0e-4;

/// The `bodies_disk_flux` scene: an unmapped Lambert sphere (p_V 0.3) of 1000 km at 500 000 km,
/// 98 px in radius in a 256² frame of fov 0.3°, phase 60°, lit by 10⁵ lx, at scene EV 12. At
/// 500 radii the far-field integral `disk_illuminance_lux` is good to ~R/d = 0.2 % of the
/// perspective view (at 50 radii the near side's 1/r² made the CPU raster 0.5 % brighter).
struct DiskFluxScene {
    size: u32,
    fov_y_rad: f64,
    radius_m: f64,
    dist_m: f64,
    phase_rad: f64,
    sun_lux: f64,
    p_v: f64,
    ev100: f32,
}

impl DiskFluxScene {
    fn new(size: u32) -> Self {
        Self {
            size,
            fov_y_rad: 0.3f64.to_radians(),
            radius_m: 1.0e6,
            dist_m: 5.0e8,
            phase_rad: 60f64.to_radians(),
            sun_lux: 1.0e5,
            p_v: 0.3,
            ev100: 12.0,
        }
    }

    /// The camera at the origin looks along −Z (identity orientation); the body is straight
    /// ahead and the Sun 60° from the camera as seen from the body.
    fn view(&self) -> BodyView {
        let to_camera = DVec3::Z;
        let to_sun = DQuat::from_rotation_y(self.phase_rad) * to_camera;
        BodyView {
            id: 0,
            naif: None,
            rel_pos_m: -to_camera * self.dist_m,
            orient: DQuat::from_rotation_x(0.4),
            radii_m: DVec3::splat(self.radius_m),
            proj_radius_px: proj_radius_px(
                self.radius_m,
                self.dist_m,
                self.fov_y_rad,
                f64::from(self.size),
            ),
            sun_rel_m: to_sun * 1.5e11,
            sun_illuminance_lux: self.sun_lux,
            occluders: Vec::new(),
            ring: None,
            has_atmosphere: false,
            brdf: Brdf::Lambert,
            geometric_albedo: self.p_v,
            mean_radius_m: self.radius_m,
        }
    }

    /// `disk_illuminance_lux` for the scene (C = 1: no NAIF id).
    fn expected_lux(&self) -> f64 {
        let rho = rho_from_geometric_albedo(Brdf::Lambert, self.p_v);
        disk_illuminance_lux(
            Brdf::Lambert,
            rho,
            self.radius_m,
            self.sun_lux,
            self.phase_rad,
            self.dist_m,
        )
    }

    /// Focal length [px] of the square frame.
    fn focal_px(&self) -> f64 {
        0.5 * f64::from(self.size) / (0.5 * self.fov_y_rad).tan()
    }

    /// Unit view direction (camera = root axes) through pixel (i, j)'s centre, row 0 at the top
    /// (`pixel_dir` in `common/camera.wgsl`), and the pixel's solid angle cos³θ/f² [sr].
    fn pixel(&self, i: u32, j: u32) -> (DVec3, f64) {
        let half = 0.5 * f64::from(self.size);
        let f = self.focal_px();
        let x = (f64::from(i) + 0.5 - half) / f;
        let y = (half - f64::from(j) - 0.5) / f;
        let v = DVec3::new(x, y, -1.0);
        let cos = 1.0 / v.length();
        (v * cos, cos.powi(3) / (f * f))
    }

    fn frame_ctx(&self) -> FrameCtx {
        let size = UVec2::splat(self.size);
        FrameCtx {
            frame_index: 0,
            epoch: Epoch::J2000,
            dt_real_s: 0.0,
            camera: CameraView {
                pos: UniversePos::default(),
                orient: DQuat::IDENTITY,
                fov_y_rad: self.fov_y_rad,
                velocity_mps: DVec3::ZERO,
            },
            view_rot: Mat4::IDENTITY,
            proj: reversed_z_infinite_proj(self.fov_y_rad as f32, 1.0, 0.1),
            viewport: size,
            output_size: size,
            exposure: Exposure {
                scene_ev100: self.ev100,
                star_ev100: self.ev100,
            },
            lenses: Vec::new(),
            lights: Vec::new(),
            sim_dt_s: 0.0,
            jitter_px: Vec2::ZERO,
        }
    }
}

/// Side of the `bodies_disk_flux` frame [px].
const DISK_PROBE_SIZE: u32 = 256;

/// BT.709 luminance (ITU-R BT.709-6, item 3.2).
fn luminance(r: f64, g: f64, b: f64) -> f64 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// The `bodies_disk_flux` sum done on the CPU with the CPU twins (point-sampled pixels, the
/// shader's lighting): (Σ L·Ω_pixel, `disk_illuminance_lux`). The probe's arithmetic without the
/// GPU (`disk_flux_probe_arithmetic_on_cpu`).
#[cfg(test)]
pub(crate) fn raster_disk_flux_cpu(size: u32) -> (f64, f64) {
    let scene = DiskFluxScene::new(size);
    let view = scene.view();
    let rho = rho_from_geometric_albedo(Brdf::Lambert, scene.p_v);
    let to_body = view.orient.inverse();
    let o = (to_body * -view.rel_pos_m) / view.radii_m;
    let c = (o.length_squared() - 1.0) as f32;
    let sun = to_body * view.sun_rel_m;
    let mut sum = 0.0;
    for j in 0..size {
        for i in 0..size {
            let (dir, omega) = scene.pixel(i, j);
            let d_body = to_body * dir;
            let d_unit = (d_body / view.radii_m).as_vec3();
            let Some((t0, _)) = ray_ellipsoid(o.as_vec3(), d_unit, c) else {
                continue;
            };
            let p_unit = o + f64::from(t0) * d_unit.as_dvec3();
            let p = p_unit * view.radii_m;
            let n = (p_unit / view.radii_m).normalize();
            let s = (sun - p).normalize();
            let r = reflectance(Brdf::Lambert, n.dot(s), n.dot(-d_body));
            sum += rho * r * scene.sun_lux / PI * omega;
        }
    }
    (sum, scene.expected_lux())
}

/// Renders the scene with the real bodies pass (a `Renderer` with only `BodyRenderer`) and
/// compares Σ L·Ω_pixel from the HDR target (luminance / scene_pre) with `disk_illuminance_lux`.
fn check_disk_flux(gpu: &GpuContext, lib: &mut ShaderLib) -> anyhow::Result<(bool, String)> {
    let scene = DiskFluxScene::new(DISK_PROBE_SIZE);
    let mut renderer = Renderer::new(gpu, UVec2::splat(scene.size));
    let atmospheres = uni_atmo::AtmosphereSet::new(gpu, lib, &[])?;
    let input = SharedInput::new(BodyScene {
        bodies: vec![scene.view()],
        ..BodyScene::default()
    });
    let bodies = BodyRenderer::new(
        gpu,
        lib,
        renderer.camera_bind_group_layout(),
        &[],
        Path::new("."),
        &[],
        None,
        atmospheres.ground_lighting(),
        input,
    )?;
    renderer.push(Box::new(bodies));
    // `Renderer::render` needs an output view; only a tonemap would write it.
    let output = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("disk probe output"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let commands = renderer.render(gpu, &scene.frame_ctx(), &output_view);
    gpu.queue.submit(std::iter::once(commands));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("rendering the test sphere: {err}");
    }
    let hdr = read_hdr(gpu, renderer.targets())?;
    let pre = f64::from(exposure_scale(scene.ev100));
    let mut sum = 0.0;
    let mut lit_px = 0;
    for j in 0..scene.size {
        for i in 0..scene.size {
            let k = 4 * (j * scene.size + i) as usize;
            let y = luminance(hdr[k].into(), hdr[k + 1].into(), hdr[k + 2].into());
            if !y.is_finite() {
                return Ok((false, format!("pixel ({i}, {j}) is not finite")));
            }
            if y > 0.0 {
                lit_px += 1;
            }
            sum += y / pre * scene.pixel(i, j).1;
        }
    }
    let want = scene.expected_lux();
    let ratio = sum / want;
    const TOLERANCE: f64 = 0.02;
    Ok((
        (ratio - 1.0).abs() <= TOLERANCE,
        format!(
            "Lambert sphere, phase 60°, {lit_px} lit px: Σ L·Ω = {sum:.5e} lx vs \
             disk_illuminance_lux {want:.5e} lx (ratio {ratio:.4}, tolerance ±{TOLERANCE})"
        ),
    ))
}

/// Origin (column, row) of the 256 × 256 window of Earth's 2048 × 1024 map that the BC probe
/// compares: 7.7° W – 37.3° E, 60.5° N – 15.5° N (Europe, the Mediterranean, North Africa),
/// block-aligned.
const BC_WINDOW: (u32, u32) = (980, 168);
/// Side of the BC probe's window [texels].
const BC_WINDOW_SIZE: u32 = 256;

/// Reads 256 × 256 texels of the BC mip with `textureLoad` (the texture is viewed as a non-sRGB
/// format, so each texel comes back as its stored code / 255).
const BC_PROBE_WGSL: &str = r#"// probe/planet_bc.wgsl: `universe --selftest` probe of the BC7 maps (uni-planet B1).
@group(0) @binding(0) var bc_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bc_out: array<vec4<f32>>;

const WINDOW_ORIGIN: vec2<u32> = vec2<u32>(980u, 168u);
const WINDOW_SIZE: u32 = 256u;

@compute @workgroup_size(16, 16, 1)
fn cs_bc_window(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WINDOW_SIZE || gid.y >= WINDOW_SIZE) {
        return;
    }
    let texel = textureLoad(bc_texture, WINDOW_ORIGIN + gid.xy, 0);
    bc_out[gid.y * WINDOW_SIZE + gid.x] = texel;
}
"#;

/// Earth's BC7 mip of width 2048 decoded on the GPU against its 2k uncompressed copy, over
/// [`BC_WINDOW`]: PSNR of the 8-bit sRGB codes (RGB), ≥ 38 dB. Without
/// `TEXTURE_COMPRESSION_BC`: "skipped: no BC" (a pass, there is nothing to compare). A missing
/// map is a failure (the selftest needs the baked maps).
fn check_bc_matches_fallback(gpu: &GpuContext) -> anyhow::Result<(bool, String)> {
    if !gpu.has_bc_compression() {
        return Ok((true, "skipped: no BC".to_owned()));
    }
    let dir = uni_data::data_dir().join("baked/maps");
    let index = load_map_index(&dir.join("index.ron")).map_err(|e| anyhow!("{e}"))?;
    let earth = index
        .iter()
        .find(|m| m.naif == 399 && m.kind == MapKind::Albedo)
        .context("no Earth albedo map in maps/index.ron")?;
    let bc = load_tex(&dir.join(&earth.file)).map_err(|e| anyhow!("{e}"))?;
    let small = load_tex(&dir.join(&earth.fallback_file)).map_err(|e| anyhow!("{e}"))?;
    anyhow::ensure!(
        small.format == TexFormat::Rgba8UnormSrgb,
        "{}: {:?}",
        earth.fallback_file,
        small.format
    );
    let level = (0..bc.mips.len())
        .find(|&l| bc.mip_size(l) == (small.width, small.height))
        .with_context(|| format!("{} has no {}-wide mip", earth.file, small.width))?;
    let format = match bc.format {
        TexFormat::Bc7RgbaUnormSrgb => wgpu::TextureFormat::Bc7RgbaUnorm,
        TexFormat::Bc1RgbaUnormSrgb => wgpu::TextureFormat::Bc1RgbaUnorm,
        other => anyhow::bail!("{}: {other:?} is not an sRGB BC format", earth.file),
    };
    let (w, h) = bc.mip_size(level);
    let (x0, y0) = BC_WINDOW;
    anyhow::ensure!(
        x0 + BC_WINDOW_SIZE <= w && y0 + BC_WINDOW_SIZE <= h,
        "window outside {w}×{h}"
    );
    let codes = decode_bc_window(gpu, &bc.mips[level], bc.format.row_bytes(w), format, (w, h))?;
    let mut sq = 0.0;
    for (k, texel) in codes.iter().enumerate() {
        let (x, y) = (
            x0 + k as u32 % BC_WINDOW_SIZE,
            y0 + k as u32 / BC_WINDOW_SIZE,
        );
        let stored = &small.mips[0][4 * (y * w + x) as usize..][..3];
        for (c, &code) in stored.iter().enumerate() {
            let diff = f64::from(texel[c]) * 255.0 - f64::from(code);
            sq += diff * diff;
        }
    }
    let mse = sq / (3.0 * codes.len() as f64);
    let psnr = 10.0 * (255.0 * 255.0 / mse.max(1e-12)).log10();
    const MIN_PSNR_DB: f64 = 38.0;
    Ok((
        psnr >= MIN_PSNR_DB,
        format!(
            "{} mip {level} ({w}×{h}) vs {} over {BC_WINDOW_SIZE}² texels at {BC_WINDOW:?}: \
             PSNR {psnr:.2} dB (minimum {MIN_PSNR_DB} dB)",
            earth.file, earth.fallback_file
        ),
    ))
}

/// Uploads one BC mip (`blocks`, `row_bytes` per row of blocks, `size` texels) as `format` and
/// reads [`BC_WINDOW`] back through `textureLoad`: RGBA in [0, 1] per texel, rows top-down.
fn decode_bc_window(
    gpu: &GpuContext,
    blocks: &[u8],
    row_bytes: u64,
    format: wgpu::TextureFormat,
    size: (u32, u32),
) -> anyhow::Result<Vec<[f32; 4]>> {
    let device = &gpu.device;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bc probe"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        blocks,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(u32::try_from(row_bytes).context("row too long")?),
            rows_per_image: Some(size.1.div_ceil(4)),
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bc probe"),
        source: wgpu::ShaderSource::Wgsl(BC_PROBE_WGSL.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bc probe bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bc probe layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bc probe"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("cs_bc_window"),
        compilation_options: Default::default(),
        cache: None,
    });
    let out_bytes = u64::from(BC_WINDOW_SIZE * BC_WINDOW_SIZE) * size_of::<[f32; 4]>() as u64;
    let buffer = |label: &str, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: out_bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let output = buffer(
        "bc probe output",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        "bc probe read-back",
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bc probe bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bc probe"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bc probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = BC_WINDOW_SIZE.div_ceil(16);
        pass.dispatch_workgroups(groups, groups, 1);
    }
    enc.copy_buffer_to_buffer(&output, 0, &readback, 0, out_bytes);
    gpu.queue.submit(std::iter::once(enc.finish()));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("bc probe: {err}");
    }
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        // The receiver outlives the wait below; a send error cannot be acted on here.
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("bc probe: waiting for the GPU")?;
    rx.try_recv()
        .context("bc probe: the read-back callback did not run")?
        .context("bc probe: mapping the output")?;
    let texels = {
        let mapped = readback
            .get_mapped_range(..)
            .context("bc probe: reading the output")?;
        bytemuck::pod_collect_to_vec(&mapped)
    };
    readback.unmap();
    Ok(texels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_gfx::validate_wgsl;

    /// The probe shader composes with `body.wgsl` and validates, with one compute entry point
    /// per probe.
    #[test]
    fn probe_shader_validates() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        lib.add(PROBE_PATH, PROBE_WGSL);
        let src = lib.compose(PROBE_PATH).unwrap();
        let m = validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        for entry in [
            "cs_ray_ellipsoid",
            "cs_equirect_uv",
            "cs_sun_visible_fraction",
            "cs_reflectance",
            "cs_ring_transmittance",
            "cs_flux_match",
            "cs_pre_mix",
        ] {
            assert!(
                m.entry_points.iter().any(|e| e.name == entry),
                "no entry point {entry}"
            );
        }
        // The BC probe's own shader.
        validate_wgsl(BC_PROBE_WGSL).unwrap();
    }

    /// 256 rays, including a camera 1 m above the surface, one inside and one at the centre,
    /// and no ray within 1e-3 of grazing (where the GPU and the CPU could disagree on hit/miss).
    #[test]
    fn ray_probe_inputs_cover_the_cases() {
        let rays = ray_inputs();
        assert_eq!(rays.len(), 256);
        let cs: Vec<f32> = rays.iter().map(|r| r.o[3]).collect();
        assert!(cs.iter().any(|&c| c > 0.0 && c < 1e-6), "a 1 m camera");
        assert!(cs.iter().any(|&c| c == -1.0), "the centre");
        assert!(cs.iter().any(|&c| c < 0.0 && c > -1.0), "inside");
        let hits = rays
            .iter()
            .filter(|r| {
                let o = glam::Vec3::from_slice(&r.o[..3]);
                let d = glam::Vec3::from_slice(&r.d[..3]);
                crate::ray_ellipsoid(o, d, r.o[3]).is_some()
            })
            .count();
        assert!(hits > 64 && hits < 256 - 32, "{hits} hits");
    }

    /// The `flux_match` probe's 64 cases reach both clamps and the interior.
    #[test]
    fn flux_match_probe_inputs_cover_the_cases() {
        let input = flux_match_inputs();
        assert_eq!(input.len(), 2 * 64);
        let ks: Vec<f32> = input
            .chunks_exact(2)
            .map(|c| flux_match(c[0], c[1][0]))
            .collect();
        assert!(ks.contains(&0.0), "clamped to 0");
        assert!(ks.contains(&FLUX_MATCH_MAX), "clamped to k_max");
        assert!(
            ks.iter().any(|&k| k > 1.0 && k < FLUX_MATCH_MAX),
            "interior"
        );
    }

    /// The `pre_mix` probe's 64 cases: both ends (where the twin returns its input bit for bit),
    /// the interior in both directions and with equal pre-exposures, all finite and positive.
    /// Inside the band the log blend differs from a linear one by far more than the probe's
    /// tolerance, so the probe tells them apart.
    #[test]
    fn pre_mix_probe_inputs_cover_the_cases() {
        let input = pre_mix_inputs();
        assert_eq!(input.len(), 64);
        assert!(input.iter().flatten().all(|v| v.is_finite() && *v >= 0.0));
        let ends = input.iter().filter(|c| c[2] == 0.0 || c[2] == 1.0).count();
        assert_eq!(ends, 16);
        for c in &input {
            let p = pre_mix(c[0], c[1], c[2]);
            assert!(p.is_finite() && p > 0.0, "{c:?}");
            if c[2] == 0.0 {
                assert_eq!(p.to_bits(), c[0].to_bits());
            }
            if c[2] == 1.0 {
                assert_eq!(p.to_bits(), c[1].to_bits());
            }
        }
        assert!(input.iter().any(|c| c[0] < c[1]) && input.iter().any(|c| c[0] > c[1]));
        assert!(input.iter().any(|c| c[0] == c[1]));
        // Star EV −4 against scene EV 15, half way: the linear mix is 8.5 stops off.
        let (a, b) = (exposure_scale(-4.0), exposure_scale(15.0));
        let linear = a + (b - a) * 0.5;
        let off = f64::from(linear / pre_mix(a, b, 0.5)).log2();
        assert!(off > 8.4 && off > 1e3 * PRE_MIX_TOLERANCE_STOPS, "{off}");
    }

    /// The bodies_disk_flux probe's arithmetic on the CPU: the CPU twins rasterised over the
    /// probe's 256² grid, summed as Σ L·Ω_pixel, match `disk_illuminance_lux` within 2 %.
    #[test]
    fn disk_flux_probe_arithmetic_on_cpu() {
        let (sum, want) = raster_disk_flux_cpu(256);
        eprintln!("CPU raster {sum:.6e} lx vs integral {want:.6e} lx");
        assert!((sum / want - 1.0).abs() < 0.02, "{sum} vs {want}");
    }
}
