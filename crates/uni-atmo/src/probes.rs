//! GPU probes of uni-atmo's WGSL against its CPU twins, run by `universe --selftest` (Task B3).
//!
//! - the transmittance LUT against [`crate::transmittance`]'s integral at 64 texel centres
//!   (±1e-3);
//! - the multiple-scattering LUT against [`crate::multiscatter`] at 16 texel centres (±2 %);
//! - the irradiance LUT against [`crate::sky_irradiance`] at 16 texel centres (±2 %);
//! - `atmo_sun_transmittance` and `atmo_sky_irradiance` through the ground-lighting group, 32
//!   each, against [`crate::ground::ground_lighting`] (review §4);
//! - the view march (`atmo_march`) against [`crate::segment_radiance`]'s march, 32 rays (±2 %).
//!
//! Earth and Mars are probed. `uni_gfx::probe::run_compute` binds only its own group, and these
//! kernels also need the atmosphere's group 2, so [`run`] here binds both (input and output at
//! group 3).

use anyhow::Context as _;
use glam::{DVec2, DVec3, Vec3};
use uni_gfx::probe::ProbeResult;
use uni_gfx::{GpuContext, ShaderLib};
use wgpu::util::DeviceExt as _;

use crate::feature::{camera_origin, segment_bounds};
use crate::lut::{
    AtmosphereSet, IRRADIANCE_SIZE, MULTISCATTER_SIZE, TRANSMITTANCE_SIZE, cpu_luts, march,
    texel_rmus, transmittance_r_mu, transmittance_to_top,
};
use crate::params::{AtmosphereParams, earth, mars};

/// `ShaderLib` path of the probe kernels.
pub const PROBE_WGSL: &str = "atmo/probes.wgsl";
/// Entry points of [`PROBE_WGSL`].
pub const PROBE_ENTRIES: [&str; 5] = [
    "cs_probe_lut_t",
    "cs_probe_lut_ms",
    "cs_probe_lut_irr",
    "cs_probe_ground",
    "cs_probe_march",
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
        let checks: [(&str, anyhow::Result<(bool, String)>); 6] = [
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
            ("march", check_march(&probe, gpu, view, params)),
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

fn check_march(
    probe: &Probe,
    gpu: &GpuContext,
    view: Group<'_>,
    p: &AtmosphereParams,
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
        let m = march(&luts, o, rd, 0.0, t1 - t0, sun);
        input.push(o.as_vec3().extend(0.0).to_array());
        input.push(rd.as_vec3().extend((t1 - t0) as f32).to_array());
        input.push(sun.as_vec3().extend(rd.dot(sun) as f32).to_array());
        cpu.push(m);
    }
    let n = cpu.len();
    anyhow::ensure!(n > 0, "no probe ray crosses the shell");
    let out = probe.run(gpu, "cs_probe_march", view, &input, n as u32, 2 * n)?;
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
    let floor = 1e-3
        * light
            .iter()
            .map(|(_, c)| c.max_element())
            .fold(0.0, f64::max);
    let (_, rel) = compare(&light, floor);
    let (t_abs, _) = compare(&trans, 0.0);
    Ok((
        rel <= 0.02 && t_abs <= 2e-3,
        format!(
            "{n} rays, inscatter max relative error {rel:.2e} (tolerance 2e-2), transmittance max |GPU − CPU| {t_abs:.2e} (tolerance 2e-3)"
        ),
    ))
}
