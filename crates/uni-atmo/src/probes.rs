//! GPU probes of uni-atmo's WGSL against its CPU twins, run by `universe --selftest` (Task B3).
//!
//! - the transmittance LUT against [`crate::transmittance`]'s integral at 64 texel centres
//!   (±1e-3);
//! - the multiple-scattering LUT against [`crate::multiscatter`] at 16 texel centres (±2 %);
//! - the irradiance LUT against [`crate::sky_irradiance`] at 16 texel centres (±2 %);
//! - `atmo_sun_transmittance` and `atmo_sky_irradiance` through the ground-lighting group, 32
//!   each, against [`crate::ground::ground_lighting`] (review §4);
//! - `atmo_march` (32–48 steps: the irradiance LUT's march) against the CPU `lut::march`, 32 rays
//!   (±2 %);
//! - the per-pixel view march (`atmo_march_n` with `atmo_view_step_count`, 16–24 steps: what the
//!   atmosphere pass draws, task B9p) against [`crate::segment_radiance`]'s march, the same rays
//!   (±2 %);
//! - `atmo_segment_lut_transmittance` (the dual-source mask pass's `g`, task B9p) against
//!   `feature::lut_ratio_transmittance` on the same rays, also cut short at half their segment
//!   (a body inside the shell): |ΔT| ≤ 1e-3·(1 + T)/den + 1e-4 per channel, at most 0.02, den
//!   being the ratio's denominator (a LUT error of 1e-3, the transmittance-LUT probe's bound,
//!   grows by (1 + T)/den through the division).
//!
//! Earth and Mars are probed. `uni_gfx::probe::run_compute` binds only its own group, and these
//! kernels also need the atmosphere's group 2, so [`run`] here binds both (input and output at
//! group 3).

use anyhow::Context as _;
use glam::{DVec2, DVec3, Vec3};
use uni_gfx::probe::ProbeResult;
use uni_gfx::{GpuContext, ShaderLib};
use wgpu::util::DeviceExt as _;

use crate::feature::{camera_origin, lut_ratio_parts, segment_bounds};
use crate::lut::{
    AtmosphereSet, IRRADIANCE_SIZE, MULTISCATTER_SIZE, TRANSMITTANCE_SIZE, cpu_luts, march,
    march_steps, texel_rmus, transmittance_r_mu, transmittance_to_top, view_step_count,
};
use crate::params::{AtmosphereParams, earth, mars};

/// `ShaderLib` path of the probe kernels.
pub const PROBE_WGSL: &str = "atmo/probes.wgsl";
/// Entry points of [`PROBE_WGSL`].
pub const PROBE_ENTRIES: [&str; 7] = [
    "cs_probe_lut_t",
    "cs_probe_lut_ms",
    "cs_probe_lut_irr",
    "cs_probe_ground",
    "cs_probe_march",
    "cs_probe_view_march",
    "cs_probe_lut_ratio",
];

const PROBE_SRC: &str = r#"// atmo/probes.wgsl: `universe --selftest` kernels for uni-atmo (crates/uni-atmo/src/probes.rs).
#import "atmo/atmosphere_common.wgsl"

@group(3) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(3) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

// in: (u, v, 0, 0); out: the transmittance LUT there (group 2 = an atmosphere's view group).
@compute @workgroup_size(64, 1, 1)
fn cs_probe_lut_t(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&probe_in)) {
        return;
    }
    let uv = probe_in[i].xy;
    probe_out[i] = vec4<f32>(textureSampleLevel(atmo_transmittance_lut, atmo_sampler, uv, 0.0).rgb, 0.0);
}

// in: (u, v, 0, 0); out: the multiple-scattering LUT there (view group).
@compute @workgroup_size(64, 1, 1)
fn cs_probe_lut_ms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&probe_in)) {
        return;
    }
    let uv = probe_in[i].xy;
    probe_out[i] = vec4<f32>(textureSampleLevel(atmo_multiscatter_lut, atmo_sampler, uv, 0.0).rgb, 0.0);
}

// in: (u, v, 0, 0); out: the irradiance LUT there (group 2 = the ground-lighting group).
@compute @workgroup_size(64, 1, 1)
fn cs_probe_lut_irr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&probe_in)) {
        return;
    }
    let uv = probe_in[i].xy;
    probe_out[i] = vec4<f32>(textureSampleLevel(atmo_irradiance_lut, atmo_sampler, uv, 0.0).rgb, 0.0);
}

// in: (r_m, mu_s, 0, 0); out[2i]: atmo_sun_transmittance, out[2i + 1]: atmo_sky_irradiance
// (ground-lighting group).
@compute @workgroup_size(64, 1, 1)
fn cs_probe_ground(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[i];
    probe_out[2u * i] = vec4<f32>(atmo_sun_transmittance(c.x, c.y), 0.0);
    probe_out[2u * i + 1u] = vec4<f32>(atmo_sky_irradiance(c.x, c.y), 0.0);
}

// in[3i]: (origin, 0), in[3i + 1]: (direction, length), in[3i + 2]: (sun, nu);
// out[2i]: inscatter per lux, out[2i + 1]: transmittance (view group).
@compute @workgroup_size(64, 1, 1)
fn cs_probe_march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (3u * i + 2u >= arrayLength(&probe_in)) {
        return;
    }
    let a = probe_in[3u * i];
    let b = probe_in[3u * i + 1u];
    let c = probe_in[3u * i + 2u];
    let s = atmo_march(a.xyz, b.xyz, 0.0, b.w, c.xyz, c.w, true);
    probe_out[2u * i] = vec4<f32>(s.inscatter, 0.0);
    probe_out[2u * i + 1u] = vec4<f32>(s.transmittance, 0.0);
}

// As cs_probe_march, with the view march's step count (16 to 24): the march the atmosphere pass
// runs per pixel (atmo_march_half, atmo_composite).
@compute @workgroup_size(64, 1, 1)
fn cs_probe_view_march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (3u * i + 2u >= arrayLength(&probe_in)) {
        return;
    }
    let a = probe_in[3u * i];
    let b = probe_in[3u * i + 1u];
    let c = probe_in[3u * i + 2u];
    let s = atmo_march_n(a.xyz, b.xyz, 0.0, b.w, c.xyz, c.w, true, atmo_view_step_count(b.w));
    probe_out[2u * i] = vec4<f32>(s.inscatter, 0.0);
    probe_out[2u * i + 1u] = vec4<f32>(s.transmittance, 0.0);
}

// in[2i]: (segment start, length), in[2i + 1]: (direction, 0); out[i]: the transmittance-LUT
// ratio along the segment (view group), the mask pass's T.
@compute @workgroup_size(64, 1, 1)
fn cs_probe_lut_ratio(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (2u * i + 1u >= arrayLength(&probe_in)) {
        return;
    }
    let a = probe_in[2u * i];
    let b = probe_in[2u * i + 1u];
    probe_out[i] = vec4<f32>(atmo_segment_lut_transmittance(a.xyz, b.xyz, a.w), 0.0);
}
"#;

/// Registers the probe kernels in `lib` (they import the uni-atmo shaders).
pub fn register_probe_shaders(lib: &mut ShaderLib) {
    crate::register_shaders(lib);
    lib.add(PROBE_WGSL, PROBE_SRC);
}

/// Every uni-atmo probe, registering its shaders in `lib`.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    register_probe_shaders(lib);
    let fail = |name: &str, e: anyhow::Error| ProbeResult {
        name: name.to_owned(),
        pass: false,
        detail: format!("{e:#}"),
    };
    let presets = [(399, earth()), (499, mars())];
    let set = match AtmosphereSet::new(gpu, lib, &presets) {
        Ok(set) => set,
        Err(e) => return vec![fail("atmo LUTs", e)],
    };
    let probe = match Probe::new(gpu, lib) {
        Ok(p) => p,
        Err(e) => return vec![fail("atmo probe kernels", e)],
    };
    let mut results = Vec::new();
    for (naif, params) in &presets {
        let name = |what: &str| format!("atmo {what} ({naif})");
        let Some(entry) = set.entry(*naif) else {
            results.push(fail(
                &name("set"),
                anyhow::anyhow!("no atmosphere for NAIF {naif}"),
            ));
            continue;
        };
        let view = (set.view_bgl(), &entry.view_group);
        let ground_lighting = set.ground_lighting();
        let ground = (&ground_lighting.bgl, ground_lighting.group(Some(*naif)));
        let checks: [(&str, anyhow::Result<(bool, String)>); 8] = [
            (
                "transmittance LUT",
                check_transmittance_lut(&probe, gpu, view, params),
            ),
            (
                "multiscatter LUT",
                check_multiscatter_lut(&probe, gpu, view, params),
            ),
            (
                "irradiance LUT",
                check_irradiance_lut(&probe, gpu, ground, params),
            ),
            (
                "atmo_sun_transmittance",
                check_ground_transmittance(&probe, gpu, ground, params),
            ),
            (
                "atmo_sky_irradiance",
                check_ground_irradiance(&probe, gpu, ground, params),
            ),
            ("march", check_march(&probe, gpu, view, params, March::Lut)),
            (
                "view march",
                check_march(&probe, gpu, view, params, March::View),
            ),
            ("lut ratio", check_lut_ratio(&probe, gpu, view, params)),
        ];
        for (what, result) in checks {
            results.push(match result {
                Ok((pass, detail)) => ProbeResult {
                    name: name(what),
                    pass,
                    detail,
                },
                Err(e) => fail(&name(what), e),
            });
        }
    }
    results
}

/// The probe module and its input/output layout (group 3).
struct Probe {
    module: wgpu::ShaderModule,
    io_bgl: wgpu::BindGroupLayout,
}

type Group<'a> = (&'a wgpu::BindGroupLayout, &'a wgpu::BindGroup);

impl Probe {
    fn new(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<Self> {
        let device = &gpu.device;
        let module = lib.module(device, PROBE_WGSL)?;
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let io_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo probe io"),
            entries: &[storage(0, true), storage(1, false)],
        });
        Ok(Self { module, io_bgl })
    }

    /// Runs `entry` over `threads` items with `group2` at group 2 and `input` at group 3, and
    /// reads back `outputs` vec4s.
    fn run(
        &self,
        gpu: &GpuContext,
        entry: &str,
        group2: Group<'_>,
        input: &[[f32; 4]],
        threads: u32,
        outputs: usize,
    ) -> anyhow::Result<Vec<[f32; 4]>> {
        anyhow::ensure!(
            !input.is_empty() && outputs > 0,
            "probe {entry}: empty input or output"
        );
        let device = &gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[None, None, Some(group2.0), Some(&self.io_bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &self.module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("atmo probe input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bytes = (outputs * size_of::<[f32; 4]>()) as u64;
        let buffer = |label: &str, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let output = buffer(
            "atmo probe output",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = buffer(
            "atmo probe read-back",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmo probe io"),
            layout: &self.io_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(2, group2.1, &[]);
            pass.set_bind_group(3, &io, &[]);
            pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
        gpu.queue.submit(std::iter::once(enc.finish()));
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("probe {entry}: {err}");
        }
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            // The receiver outlives the wait below; a send error cannot be acted on here.
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .with_context(|| format!("probe {entry}: waiting for the GPU"))?;
        rx.try_recv()
            .with_context(|| format!("probe {entry}: the read-back callback did not run"))?
            .with_context(|| format!("probe {entry}: mapping the output"))?;
        let out = {
            let view = readback
                .get_mapped_range(..)
                .with_context(|| format!("probe {entry}: reading the output"))?;
            bytemuck::pod_collect_to_vec(&view)
        };
        readback.unmap();
        Ok(out)
    }
}

fn rgb(v: [f32; 4]) -> DVec3 {
    Vec3::from_slice(&v[..3]).as_dvec3()
}

/// Texel-centre uv of texel (i, j) of a LUT of `size`.
fn texel_uv(i: u32, j: u32, size: [u32; 2]) -> DVec2 {
    DVec2::new(
        (f64::from(i) + 0.5) / f64::from(size[0]),
        (f64::from(j) + 0.5) / f64::from(size[1]),
    )
}

/// `n` × `m` texels spread evenly over a LUT of `size`, the last column skipped when
/// `skip_last_col` (the transmittance LUT's horizon column).
fn texel_grid(n: u32, m: u32, size: [u32; 2], skip_last_col: bool) -> Vec<DVec2> {
    let cols = size[0] - 1 - u32::from(skip_last_col);
    let mut uvs = Vec::new();
    for j in 0..m {
        for i in 0..n {
            let x = i * cols / (n - 1).max(1);
            let y = j * (size[1] - 1) / (m - 1).max(1);
            uvs.push(texel_uv(x, y, size));
        }
    }
    uvs
}

fn uv_inputs(uvs: &[DVec2]) -> Vec<[f32; 4]> {
    uvs.iter()
        .map(|uv| [uv.x as f32, uv.y as f32, 0.0, 0.0])
        .collect()
}

/// Worst |GPU − CPU| of vectors, and the worst relative error against `scale` (per sample:
/// the CPU vector's largest channel plus `floor`); NaN anywhere makes both NaN.
fn compare(pairs: &[(DVec3, DVec3)], floor: f64) -> (f64, f64) {
    let mut worst_abs: f64 = 0.0;
    let mut worst_rel: f64 = 0.0;
    for (gpu, cpu) in pairs {
        let err = (*gpu - *cpu).abs().max_element();
        if err.is_nan() || !gpu.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        worst_abs = worst_abs.max(err);
        worst_rel = worst_rel.max(err / (cpu.abs().max_element() + floor));
    }
    (worst_abs, worst_rel)
}

fn check_transmittance_lut(
    probe: &Probe,
    gpu: &GpuContext,
    view: Group<'_>,
    p: &AtmosphereParams,
) -> anyhow::Result<(bool, String)> {
    let uvs = texel_grid(8, 8, TRANSMITTANCE_SIZE, false);
    let out = probe.run(gpu, "cs_probe_lut_t", view, &uv_inputs(&uvs), 64, 64)?;
    let pairs: Vec<_> = uvs
        .iter()
        .zip(&out)
        .map(|(uv, g)| {
            let (r, mu) = transmittance_r_mu(p, *uv);
            (rgb(*g), transmittance_to_top(p, r, mu))
        })
        .collect();
    let (abs, _) = compare(&pairs, 0.0);
    Ok((
        abs <= 1e-3,
        format!("64 texels, max |GPU − CPU| {abs:.2e} (tolerance 1e-3)"),
    ))
}

fn check_multiscatter_lut(
    probe: &Probe,
    gpu: &GpuContext,
    view: Group<'_>,
    p: &AtmosphereParams,
) -> anyhow::Result<(bool, String)> {
    let uvs = texel_grid(4, 4, MULTISCATTER_SIZE, false);
    let out = probe.run(gpu, "cs_probe_lut_ms", view, &uv_inputs(&uvs), 16, 16)?;
    let pairs: Vec<_> = uvs
        .iter()
        .zip(&out)
        .map(|(uv, g)| {
            let (r, mu_s) = texel_rmus(p, *uv, MULTISCATTER_SIZE);
            (rgb(*g), crate::multiscatter(p, r, mu_s).as_dvec3())
        })
        .collect();
    let floor = 1e-4
        * pairs
            .iter()
            .map(|(_, c)| c.max_element())
            .fold(0.0, f64::max);
    let (_, rel) = compare(&pairs, floor);
    Ok((
        rel <= 0.02,
        format!("16 texels, max relative error {rel:.2e} (tolerance 2e-2)"),
    ))
}

fn check_irradiance_lut(
    probe: &Probe,
    gpu: &GpuContext,
    ground: Group<'_>,
    p: &AtmosphereParams,
) -> anyhow::Result<(bool, String)> {
    let uvs = texel_grid(4, 4, IRRADIANCE_SIZE, false);
    let out = probe.run(gpu, "cs_probe_lut_irr", ground, &uv_inputs(&uvs), 16, 16)?;
    let pairs: Vec<_> = uvs
        .iter()
        .zip(&out)
        .map(|(uv, g)| {
            let (r, mu_s) = texel_rmus(p, *uv, IRRADIANCE_SIZE);
            (rgb(*g), crate::sky_irradiance(p, r, mu_s).as_dvec3())
        })
        .collect();
    let floor = 1e-4
        * pairs
            .iter()
            .map(|(_, c)| c.max_element())
            .fold(0.0, f64::max);
    let (_, rel) = compare(&pairs, floor);
    Ok((
        rel <= 0.02,
        format!("16 texels, max relative error {rel:.2e} (tolerance 2e-2)"),
    ))
}

fn check_ground_transmittance(
    probe: &Probe,
    gpu: &GpuContext,
    ground: Group<'_>,
    p: &AtmosphereParams,
) -> anyhow::Result<(bool, String)> {
    // 28 points at transmittance-LUT texel centres (the horizon column skipped), 4 below the
    // horizon (exactly 0).
    let mut points: Vec<(f64, f64)> = texel_grid(7, 4, TRANSMITTANCE_SIZE, true)
        .iter()
        .map(|uv| transmittance_r_mu(p, *uv))
        .collect();
    for k in 0..4 {
        points.push((
            p.bottom_radius_m + 1e3 * f64::from(k),
            -0.2 - 0.2 * f64::from(k),
        ));
    }
    let input: Vec<[f32; 4]> = points
        .iter()
        .map(|(r, mu)| [*r as f32, *mu as f32, 0.0, 0.0])
        .collect();
    let out = probe.run(gpu, "cs_probe_ground", ground, &input, 32, 64)?;
    let pairs: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, (r, mu))| {
            let cpu = crate::ground::ground_lighting(Some(p), *r, *mu)
                .0
                .as_dvec3();
            (rgb(out[2 * i]), cpu)
        })
        .collect();
    let (abs, _) = compare(&pairs, 0.0);
    Ok((
        abs <= 2e-3,
        format!("32 points, max |GPU − CPU| {abs:.2e} (tolerance 2e-3)"),
    ))
}

fn check_ground_irradiance(
    probe: &Probe,
    gpu: &GpuContext,
    ground: Group<'_>,
    p: &AtmosphereParams,
) -> anyhow::Result<(bool, String)> {
    let points: Vec<(f64, f64)> = texel_grid(8, 4, IRRADIANCE_SIZE, false)
        .iter()
        .map(|uv| texel_rmus(p, *uv, IRRADIANCE_SIZE))
        .collect();
    let input: Vec<[f32; 4]> = points
        .iter()
        .map(|(r, mu)| [*r as f32, *mu as f32, 0.0, 0.0])
        .collect();
    let out = probe.run(gpu, "cs_probe_ground", ground, &input, 32, 64)?;
    let pairs: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, (r, mu))| {
            let cpu = crate::ground::ground_lighting(Some(p), *r, *mu)
                .1
                .as_dvec3();
            (rgb(out[2 * i + 1]), cpu)
        })
        .collect();
    let floor = 1e-4
        * pairs
            .iter()
            .map(|(_, c)| c.max_element())
            .fold(0.0, f64::max);
    let (_, rel) = compare(&pairs, floor);
    Ok((
        rel <= 0.02,
        format!("32 points, max relative error {rel:.2e} (tolerance 2e-2)"),
    ))
}

/// 32 rays: from the ground at several elevations and Sun angles, from 10 km and 400 km up
/// (down, at the limb, sideways), and from 3 radii out across the limb and onto the disk.
fn probe_rays(p: &AtmosphereParams) -> Vec<(DVec3, DVec3, DVec3)> {
    let b = p.bottom_radius_m;
    let dir = |elev: f64, az: f64| {
        let (e, a) = (elev.to_radians(), az.to_radians());
        DVec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin())
    };
    let mut rays = Vec::new();
    for (k, (elev, sun_elev)) in [
        (90.0, 60.0),
        (45.0, 30.0),
        (10.0, 5.0),
        (2.0, 1.0),
        (30.0, -3.0),
        (5.0, 45.0),
    ]
    .into_iter()
    .enumerate()
    {
        let ro = DVec3::new(0.0, 0.0, b + 1.0);
        rays.push((ro, dir(elev, 30.0 * k as f64), dir(sun_elev, 0.0)));
        rays.push((ro, dir(elev, 180.0), dir(sun_elev, 0.0)));
    }
    for (k, elev) in [-30.0, -2.0, 0.0, 3.0, 20.0, 89.0].into_iter().enumerate() {
        let ro = DVec3::new(0.0, 0.0, b + 10e3);
        rays.push((ro, dir(elev, 45.0 * k as f64), dir(20.0, 0.0)));
    }
    let up = DVec3::new(0.0, 0.0, b + 400e3);
    let dip = -(b / (b + 400e3)).acos().to_degrees();
    for (k, elev) in [-90.0, -60.0, dip + 0.5, dip - 0.2, dip + 2.0, -5.0, 10.0]
        .into_iter()
        .enumerate()
    {
        rays.push((up, dir(elev, 20.0 * k as f64), dir(-1.0 + k as f64, 10.0)));
    }
    let far = DVec3::new(3.0 * b, 0.0, 0.0);
    for k in 0..7 {
        let target = DVec3::new(0.0, 0.0, b * (0.2 + 0.13 * f64::from(k)));
        let d = (target - far).normalize();
        rays.push((far, d, dir(10.0 * f64::from(k), 60.0)));
    }
    rays.truncate(32);
    rays
}

/// Which march a probe runs: `atmo_march` at 32–48 steps (the irradiance LUT's), or the view
/// march at 16–24 (`atmo_march_n` with `atmo_view_step_count`, the atmosphere pass's).
#[derive(Clone, Copy, Debug, PartialEq)]
enum March {
    Lut,
    View,
}

fn check_march(
    probe: &Probe,
    gpu: &GpuContext,
    view: Group<'_>,
    p: &AtmosphereParams,
    which: March,
) -> anyhow::Result<(bool, String)> {
    let luts = cpu_luts(p);
    let mut input = Vec::new();
    let mut cpu = Vec::new();
    for (ro, rd, sun) in probe_rays(p) {
        let Some(ro) = camera_origin(p, ro) else {
            continue;
        };
        let Some((t0, t1)) = segment_bounds(p, ro, rd, f64::INFINITY) else {
            continue;
        };
        let o = ro + rd * t0;
        let m = match which {
            March::Lut => march(&luts, o, rd, 0.0, t1 - t0, sun),
            March::View => {
                march_steps(&luts, o, rd, 0.0, t1 - t0, sun, view_step_count(p, t1 - t0))
            }
        };
        input.push(o.as_vec3().extend(0.0).to_array());
        input.push(rd.as_vec3().extend((t1 - t0) as f32).to_array());
        input.push(sun.as_vec3().extend(rd.dot(sun) as f32).to_array());
        cpu.push(m);
    }
    let n = cpu.len();
    anyhow::ensure!(n > 0, "no probe ray crosses the shell");
    let entry = match which {
        March::Lut => "cs_probe_march",
        March::View => "cs_probe_view_march",
    };
    let out = probe.run(gpu, entry, view, &input, n as u32, 2 * n)?;
    let light: Vec<_> = cpu
        .iter()
        .enumerate()
        .map(|(i, m)| (rgb(out[2 * i]), m.inscatter))
        .collect();
    let trans: Vec<_> = cpu
        .iter()
        .enumerate()
        .map(|(i, m)| (rgb(out[2 * i + 1]), m.transmittance))
        .collect();
    Ok(march_verdict(&light, &trans))
}

/// The march probes' verdict: inscatter within 2 % of the largest CPU value's channel (relative,
/// with a floor of 1e-3 of the brightest ray) and transmittance within 2e-3, per ray and
/// channel; NaN anywhere fails. (gpu, cpu) pairs.
fn march_verdict(light: &[(DVec3, DVec3)], trans: &[(DVec3, DVec3)]) -> (bool, String) {
    let floor = 1e-3
        * light
            .iter()
            .map(|(_, c)| c.max_element())
            .fold(0.0, f64::max);
    let (_, rel) = compare(light, floor);
    let (t_abs, _) = compare(trans, 0.0);
    (
        rel <= 0.02 && t_abs <= 2e-3,
        format!(
            "{} rays, inscatter max relative error {rel:.2e} (tolerance 2e-2), transmittance max \
             |GPU − CPU| {t_abs:.2e} (tolerance 2e-3)",
            light.len()
        ),
    )
}

/// The transmittance-LUT probe's error bound for a LUT error of this size (the
/// transmittance-LUT probe's tolerance), and its cap.
const LUT_RATIO_LUT_ERR: f64 = 1e-3;
const LUT_RATIO_CAP: f64 = 0.02;
/// f32 geometry (r, μ of the two ends) on top.
const LUT_RATIO_FLOOR: f64 = 1e-4;

/// `atmo_segment_lut_transmittance` on the GPU against `lut_ratio_transmittance` on the CPU, on
/// the probe rays from their entry to their exit and to half way (a body inside the shell).
fn check_lut_ratio(
    probe: &Probe,
    gpu: &GpuContext,
    view: Group<'_>,
    p: &AtmosphereParams,
) -> anyhow::Result<(bool, String)> {
    let luts = cpu_luts(p);
    let mut input = Vec::new();
    let mut cpu = Vec::new();
    for (ro, rd, _) in probe_rays(p) {
        let Some(ro) = camera_origin(p, ro) else {
            continue;
        };
        let Some((t0, t1)) = segment_bounds(p, ro, rd, f64::INFINITY) else {
            continue;
        };
        for frac in [1.0, 0.5] {
            // The GPU gets f32 inputs; the CPU twin runs on the same rounded values.
            let o = (ro + rd * t0).as_vec3();
            let d = rd.as_vec3();
            let len = ((t1 - t0) * frac) as f32;
            let (num, den) = lut_ratio_parts(&luts, o.as_dvec3(), d.as_dvec3(), f64::from(len));
            let t = (num / den.max(DVec3::splat(1e-30))).min(DVec3::ONE);
            input.push(o.extend(len).to_array());
            input.push(d.extend(0.0).to_array());
            cpu.push((t, den));
        }
    }
    let n = cpu.len();
    anyhow::ensure!(n > 0, "no probe ray crosses the shell");
    let out = probe.run(gpu, "cs_probe_lut_ratio", view, &input, n as u32, n)?;
    let cases: Vec<(DVec3, DVec3, DVec3)> = cpu
        .iter()
        .zip(&out)
        .map(|((t, den), g)| (rgb(*g), *t, *den))
        .collect();
    Ok(lut_ratio_verdict(&cases))
}

/// The LUT-ratio probe's verdict on (gpu, cpu, cpu denominator) triples: per channel
/// |GPU − CPU| ≤ min(1e-3·(1 + T)/den + 1e-4, 0.02); NaN or ∞ fails.
fn lut_ratio_verdict(cases: &[(DVec3, DVec3, DVec3)]) -> (bool, String) {
    let mut worst_err: f64 = 0.0;
    let mut worst_ratio: f64 = 0.0;
    let mut finite = true;
    for (gpu, cpu, den) in cases {
        finite &= gpu.is_finite() && cpu.is_finite();
        for c in 0..3 {
            let bound = (LUT_RATIO_LUT_ERR * (1.0 + cpu[c]) / den[c].max(1e-30) + LUT_RATIO_FLOOR)
                .min(LUT_RATIO_CAP);
            let err = (gpu[c] - cpu[c]).abs();
            worst_err = worst_err.max(err);
            worst_ratio = worst_ratio.max(err / bound);
        }
    }
    (
        finite && worst_ratio <= 1.0,
        format!(
            "{} segments, max |GPU − CPU| {worst_err:.2e}, max error/bound {worst_ratio:.2} \
             (bound 1e-3·(1 + T)/den + 1e-4, at most 0.02); finite {finite}",
            cases.len()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_gfx::validate_wgsl;

    /// The probe kernels compose and validate, and the B9p kernels call the routines they
    /// check: the view march's step count and the mask pass's LUT ratio.
    #[test]
    fn probe_kernels_validate_and_call_the_pass_routines() {
        let mut lib = ShaderLib::with_common();
        register_probe_shaders(&mut lib);
        let src = lib.compose(PROBE_WGSL).unwrap();
        let m = validate_wgsl(&src).unwrap_or_else(|e| panic!("{PROBE_WGSL}: {e}"));
        for entry in PROBE_ENTRIES {
            assert!(
                m.entry_points
                    .iter()
                    .any(|e| e.name == entry && e.stage == wgpu::naga::ShaderStage::Compute),
                "{entry}"
            );
        }
        assert!(src.contains(
            "atmo_march_n(a.xyz, b.xyz, 0.0, b.w, c.xyz, c.w, true, atmo_view_step_count(b.w))"
        ));
        assert!(src.contains("atmo_segment_lut_transmittance(a.xyz, b.xyz, a.w)"));
    }

    /// The march verdict passes f32-sized differences and fails a 5 % inscatter error, a
    /// transmittance off by 5e-3, a NaN channel, and the 32-step march compared with the
    /// 16-step one where they differ.
    #[test]
    fn march_verdict_passes_agreement_and_fails_errors() {
        let light = vec![(DVec3::new(1.0, 2.0, 3.0), DVec3::new(1.0, 2.0, 3.0)); 4];
        let trans = vec![(DVec3::splat(0.5), DVec3::splat(0.5)); 4];
        assert!(march_verdict(&light, &trans).0);
        let mut near = light.clone();
        near[1].0 *= 1.0 + 1e-4;
        assert!(march_verdict(&near, &trans).0);
        let mut off = light.clone();
        off[2].0.z *= 1.05;
        assert!(!march_verdict(&off, &trans).0);
        let mut t_off = trans.clone();
        t_off[0].0.y += 5e-3;
        assert!(!march_verdict(&light, &t_off).0);
        let mut nan = light.clone();
        nan[3].0.x = f64::NAN;
        assert!(!march_verdict(&nan, &trans).0);
    }

    /// The LUT-ratio verdict passes a LUT-sized error and fails what a slip in the WGSL would
    /// give, on segments whose denominator is not 1 (cut short inside the shell, or reaching the
    /// ground from inside): numerator and denominator swapped, the sky branch used for a ground
    /// ray, μ not negated in the ground branch, NaN.
    #[test]
    fn lut_ratio_verdict_passes_agreement_and_fails_slips() {
        use crate::lut::lut_transmittance;
        let p = earth();
        let luts = cpu_luts(&p);
        let b = p.bottom_radius_m;
        let lut = |r: f64, mu: f64| lut_transmittance(&luts, r, mu);
        let space = DVec3::new(3.0 * b, 0.0, 0.0);
        let inside = DVec3::new(0.0, 0.0, b + 10e3);
        let mut good = Vec::new();
        let mut swapped = Vec::new();
        let mut sky_branch = Vec::new();
        let mut no_negation = Vec::new();
        for (ro, rd, frac) in [
            // From space onto the disk, cut short at half the air column.
            (space, -DVec3::X, 0.5),
            // From space through the limb, cut short at half the path.
            (
                space,
                (DVec3::new(0.0, b + 20e3, 0.0) - space).normalize(),
                0.5,
            ),
            // From inside down to the ground at 20° and 60°.
            (inside, DVec3::new(0.94, 0.0, -0.342).normalize(), 1.0),
            (inside, DVec3::new(0.5, 0.0, -0.866).normalize(), 1.0),
        ] {
            let ro = camera_origin(&p, ro).unwrap();
            let (t0, t1) = segment_bounds(&p, ro, rd, f64::INFINITY).unwrap();
            let (o, len) = (ro + rd * t0, (t1 - t0) * frac);
            let (num, den) = lut_ratio_parts(&luts, o, rd, len);
            let t = (num / den).min(DVec3::ONE);
            good.push((t + 5e-4 * den.min_element(), t, den));
            swapped.push(((den / num).min(DVec3::ONE), t, den));
            let e = o + rd * len;
            let (r0, r1) = (o.length(), e.length());
            let (mu0, mu1) = (o.dot(rd) / r0, e.dot(rd) / r1);
            sky_branch.push(((lut(r0, mu0) / lut(r1, mu1)).min(DVec3::ONE), t, den));
            no_negation.push(((lut(r1, mu1) / lut(r0, mu0)).min(DVec3::ONE), t, den));
        }
        let (pass, detail) = lut_ratio_verdict(&good);
        assert!(pass, "{detail}");
        for (what, cases) in [
            ("swapped", &swapped),
            ("sky branch", &sky_branch),
            ("no negation", &no_negation),
        ] {
            let (pass, detail) = lut_ratio_verdict(cases);
            assert!(!pass, "{what}: {detail}");
        }
        let mut nan = good.clone();
        nan[0].0.y = f64::NAN;
        assert!(!lut_ratio_verdict(&nan).0);
    }
}
