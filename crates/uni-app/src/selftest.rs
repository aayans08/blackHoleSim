//! Headless --selftest mode: compares GPU shader twins with their CPU twins (M0 Task 11).
//!
//! Each check runs a small compute probe that `#import`s the shader under test, reads its
//! output back and compares it with the Rust twin:
//! - `pcg3d`: `common/hash.wgsl` vs `uni_core::pcg3d`, 4096 inputs, bit-exact;
//! - `agx`: `post/tonemap.wgsl` vs `uni_post::agx`, 256 inputs, within 1e-3;
//! - `binet`: `rel/binet.wgsl` vs `uni_rel::binet`, 64 rays traced as the lens pass traces
//!   them (same step rule and budget), exit direction within 2e-3 rad, same capture flags;
//! - then every `gpu_probes` of the M1 render crates (`uni_gfx::probe::ProbeResult`), so far
//!   uni-gfx's `cube_dir` (`common/cube.wgsl` vs `uni_gfx::env::cube_dir`).
//!
//! Probe buffers sit in bind group 2: `post/tonemap.wgsl` claims groups 0 and 1.

use std::f64::consts::PI;

use anyhow::Context as _;
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, UVec3, Vec3, Vec4};
use uni_core::{pcg3d, splitmix64};
use uni_gfx::{GpuContext, ShaderLib};
use uni_rel::binet::{self, RayEnd};
use wgpu::util::DeviceExt as _;

/// Bind group of every probe's input (binding 0) and output (binding 1) storage buffers.
const PROBE_GROUP: u32 = 2;
/// `@workgroup_size` of every probe's `cs_main`.
const PROBE_WORKGROUP: u32 = 64;

const PCG3D_PROBE: &str = "selftest/pcg3d.wgsl";
const AGX_PROBE: &str = "selftest/agx.wgsl";
const BINET_PROBE: &str = "selftest/binet.wgsl";

/// Runs `pcg3d` from common/hash.wgsl on each input; w is unused.
const PCG3D_PROBE_WGSL: &str = r#"// selftest/pcg3d.wgsl: `universe --selftest` probe for pcg3d.
#import "common/hash.wgsl"

@group(2) @binding(0) var<storage, read> probe_in: array<vec4<u32>>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<u32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    probe_out[gid.x] = vec4<u32>(pcg3d(probe_in[gid.x].xyz), 0u);
}
"#;

/// Runs `agx` from post/tonemap.wgsl on each input colour (xyz); w is unused.
const AGX_PROBE_WGSL: &str = r#"// selftest/agx.wgsl: `universe --selftest` probe for agx.
#import "post/tonemap.wgsl"

@group(2) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    probe_out[gid.x] = vec4<f32>(agx(probe_in[gid.x].xyz), 0.0);
}
"#;

/// Traces one ray per input exactly as `cs_trace` in rel/lens_trace.wgsl does:
/// `static_observer_ray`, then `binet_trace`.
const BINET_PROBE_WGSL: &str = r#"// selftest/binet.wgsl: `universe --selftest` probe for binet_trace.
#import "rel/binet.wgsl"

// Twin of `BinetCase` in crates/uni-app/src/selftest.rs (vec4 members only).
struct BinetCase {
    // xyz: static camera position relative to the hole [M]; w: escape radius r_far [M].
    cam: vec4<f32>,
    // xyz: the ray's direction in the camera's local rest frame; w unused.
    dir: vec4<f32>,
    // x: step budget; yzw unused.
    budget: vec4<u32>,
}

// Twin of `BinetOutcome` in crates/uni-app/src/selftest.rs (vec4 members only).
struct BinetOutcome {
    // xyz: final tangent dx/dλ; w unused.
    v: vec4<f32>,
    // x: BINET_CAPTURED, BINET_ESCAPED or BINET_MAX_STEPS; yzw unused.
    status: vec4<u32>,
}

@group(2) @binding(0) var<storage, read> probe_in: array<BinetCase>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<BinetOutcome>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    let v0 = static_observer_ray(c.cam.xyz, c.dir.xyz);
    let end = binet_trace(c.cam.xyz, v0, c.cam.w, c.budget.x);
    probe_out[gid.x] = BinetOutcome(vec4<f32>(end.v, 0.0), vec4<u32>(end.status, 0u, 0u, 0u));
}
"#;

/// Inputs of the `pcg3d` check.
const PCG3D_INPUTS: usize = 4096;
/// Inputs of the `agx` check.
const AGX_INPUTS: usize = 256;
/// Largest |GPU − CPU| of any AgX output channel (brief).
const AGX_TOLERANCE: f32 = 1e-3;
/// Rays of the `binet` check.
const BINET_RAYS: usize = 64;
/// Largest angle between GPU and CPU exit directions [rad] (brief).
const BINET_TOLERANCE_RAD: f64 = 2e-3;
/// The lens pass's step budget (`GPU_MAX_STEPS` in uni-rel's tracer.rs, private there).
const LENS_MAX_STEPS: u32 = 2000;
/// The lens pass's smallest escape radius [M] (`GPU_MIN_R_FAR` in uni-rel's tracer.rs).
const LENS_MIN_R_FAR: f64 = 1000.0;

/// GPU status codes from rel/binet.wgsl.
const BINET_CAPTURED: u32 = 0;
const BINET_ESCAPED: u32 = 1;
const BINET_MAX_STEPS: u32 = 2;

/// Twin of `struct BinetCase` in the binet probe.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BinetCase {
    cam: [f32; 4],
    dir: [f32; 4],
    budget: [u32; 4],
}

/// Twin of `struct BinetOutcome` in the binet probe.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BinetOutcome {
    v: [f32; 4],
    status: [u32; 4],
}

/// Runs every check on a headless GPU, printing `PASS name: …` or `FAIL name: …` for each.
/// Returns true only if all pass.
pub fn run() -> bool {
    let gpu = match pollster::block_on(GpuContext::new_headless()) {
        Ok(gpu) => gpu,
        Err(e) => {
            println!("FAIL gpu: {e:#}");
            return false;
        }
    };
    let mut lib = probe_lib();
    let checks: [(&str, fn(&GpuContext, &ShaderLib) -> Result<String, String>); 3] = [
        ("pcg3d", check_pcg3d),
        ("agx", check_agx),
        ("binet", check_binet),
    ];
    let mut all_pass = true;
    for (name, check) in checks {
        match check(&gpu, &lib) {
            Ok(detail) => println!("PASS {name}: {detail}"),
            Err(detail) => {
                println!("FAIL {name}: {detail}");
                all_pass = false;
            }
        }
    }
    for probe in uni_gfx::probe::gpu_probes(&gpu, &mut lib) {
        let verdict = if probe.pass { "PASS" } else { "FAIL" };
        println!("{verdict} {}: {}", probe.name, probe.detail);
        all_pass &= probe.pass;
    }
    // An uncaptured GPU error during the probes fails the run (review m6).
    let errors = gpu.take_errors();
    if !errors.is_empty() {
        println!("FAIL gpu errors: {}", errors.join("; "));
        all_pass = false;
    }
    all_pass
}

/// The common shaders, the shaders under test and the three probes.
fn probe_lib() -> ShaderLib {
    let mut lib = ShaderLib::with_common();
    uni_post::register_shaders(&mut lib);
    uni_rel::register_shaders(&mut lib);
    lib.add(PCG3D_PROBE, PCG3D_PROBE_WGSL);
    lib.add(AGX_PROBE, AGX_PROBE_WGSL);
    lib.add(BINET_PROBE, BINET_PROBE_WGSL);
    lib
}

fn check_pcg3d(gpu: &GpuContext, lib: &ShaderLib) -> Result<String, String> {
    let inputs = pcg3d_inputs();
    let gpu_in: Vec<[u32; 4]> = inputs.iter().map(|v| v.extend(0).to_array()).collect();
    let out: Vec<[u32; 4]> =
        run_probe(gpu, lib, PCG3D_PROBE, &gpu_in).map_err(|e| format!("{e:#}"))?;
    let mismatches: Vec<usize> = (0..inputs.len())
        .filter(|&i| UVec3::from_slice(&out[i][..3]) != pcg3d(inputs[i]))
        .collect();
    match mismatches.first() {
        None => Ok(format!("{} inputs bit-exact", inputs.len())),
        Some(&i) => Err(format!(
            "{} of {} differ; first #{i}: input {} GPU {} CPU {}",
            mismatches.len(),
            inputs.len(),
            inputs[i],
            UVec3::from_slice(&out[i][..3]),
            pcg3d(inputs[i])
        )),
    }
}

fn check_agx(gpu: &GpuContext, lib: &ShaderLib) -> Result<String, String> {
    let inputs = agx_inputs();
    let gpu_in: Vec<[f32; 4]> = inputs.iter().map(|c| c.extend(0.0).to_array()).collect();
    let out: Vec<[f32; 4]> =
        run_probe(gpu, lib, AGX_PROBE, &gpu_in).map_err(|e| format!("{e:#}"))?;
    let errors: Vec<f32> = (0..inputs.len())
        .map(|i| {
            let g = Vec4::from_array(out[i]).truncate();
            let err = (g - uni_post::agx(inputs[i])).abs().max_element();
            // NaN must fail, so it maps to infinity.
            if err.is_nan() { f32::INFINITY } else { err }
        })
        .collect();
    let (worst, max_err) = errors
        .iter()
        .copied()
        .enumerate()
        .fold((0, 0.0_f32), |m, (i, e)| if e > m.1 { (i, e) } else { m });
    let summary = format!("{} inputs, max |GPU − CPU| {max_err:.2e}", inputs.len());
    if max_err <= AGX_TOLERANCE {
        Ok(summary)
    } else {
        Err(format!(
            "{summary} > {AGX_TOLERANCE:e} at input {}: GPU {} CPU {}",
            inputs[worst],
            Vec4::from_array(out[worst]).truncate(),
            uni_post::agx(inputs[worst])
        ))
    }
}

fn check_binet(gpu: &GpuContext, lib: &ShaderLib) -> Result<String, String> {
    let cases = binet_cases();
    let out: Vec<BinetOutcome> =
        run_probe(gpu, lib, BINET_PROBE, &cases).map_err(|e| format!("{e:#}"))?;
    let mut captured = 0;
    let mut max_err = 0.0_f64;
    let mut worst = 0;
    let mut failures = Vec::new();
    for (i, (c, o)) in cases.iter().zip(&out).enumerate() {
        let cpu = cpu_trace(c);
        let gpu_v = Vec4::from_array(o.v).truncate().as_dvec3();
        match (cpu, o.status[0]) {
            (RayEnd::Captured, BINET_CAPTURED) => captured += 1,
            (RayEnd::Escaped { dir }, BINET_ESCAPED) => {
                let err = angle_between(dir, gpu_v);
                if err > max_err {
                    (max_err, worst) = (err, i);
                }
                if !(err <= BINET_TOLERANCE_RAD) {
                    failures.push(format!("ray #{i}: exit directions {err:.2e} rad apart"));
                }
            }
            (cpu, status) => failures.push(format!(
                "ray #{i}: CPU {cpu:?}, GPU status {status} ({})",
                status_name(status)
            )),
        }
    }
    let summary = format!(
        "{} rays ({captured} captured), max exit-direction error {max_err:.2e} rad \
         (ray #{worst}: r = {} M, b = {:.3} M)",
        cases.len(),
        case_r(&cases[worst]),
        case_b(&cases[worst]),
    );
    if failures.is_empty() {
        Ok(summary)
    } else {
        Err(format!("{summary}; {}", failures.join("; ")))
    }
}

/// Camera distance of a case [M], rounded to the nearest M.
fn case_r(c: &BinetCase) -> f64 {
    Vec4::from_array(c.cam).truncate().length().round().into()
}

/// Impact parameter of a case [M]: b = r·sin δ/√(1 − 2/r), δ from the inward direction.
fn case_b(c: &BinetCase) -> f64 {
    let cam = Vec4::from_array(c.cam).truncate().as_dvec3();
    let dir = Vec4::from_array(c.dir).truncate().as_dvec3().normalize();
    let r = cam.length();
    r * dir.cross(-cam / r).length() / (1.0 - 2.0 / r).sqrt()
}

fn status_name(status: u32) -> &'static str {
    match status {
        BINET_CAPTURED => "captured",
        BINET_ESCAPED => "escaped",
        BINET_MAX_STEPS => "out of steps",
        _ => "unknown",
    }
}

/// The CPU twin of the probe for one case, from the same f32 inputs the GPU gets.
fn cpu_trace(c: &BinetCase) -> RayEnd {
    let cam = Vec4::from_array(c.cam).truncate().as_dvec3();
    let dir = Vec4::from_array(c.dir).truncate().as_dvec3();
    match binet::static_observer_ray(cam, dir) {
        Some(v0) => binet::trace(cam, v0, f64::from(c.cam[3]), c.budget[0]),
        // Unreachable for `binet_cases` (every camera is outside the horizon).
        None => RayEnd::MaxSteps,
    }
}

/// Angle between two directions [rad], accurate at small angles; π if either is not finite.
fn angle_between(a: DVec3, b: DVec3) -> f64 {
    let a = a.normalize();
    let b = b.normalize();
    let angle = a.cross(b).length().atan2(a.dot(b));
    if angle.is_finite() { angle } else { PI }
}

/// 4096 pcg3d inputs: the corners of the u32 cube, then SplitMix64 noise (Steele et al. 2014).
fn pcg3d_inputs() -> Vec<UVec3> {
    let corners = [
        UVec3::ZERO,
        UVec3::MAX,
        UVec3::new(1, 2, 3),
        UVec3::new(u32::MAX, 0, 1),
        UVec3::new(0x8000_0000, 0x7FFF_FFFF, 0xDEAD_BEEF),
    ];
    let noise = (0_u64..).map(|i| {
        let a = splitmix64(2 * i);
        let b = splitmix64(2 * i + 1);
        UVec3::new(a as u32, (a >> 32) as u32, b as u32)
    });
    corners
        .into_iter()
        .chain(noise)
        .take(PCG3D_INPUTS)
        .collect()
}

/// 256 scene-linear colours for AgX: black, a grey ramp over 2^−14 … 2^14, saturated
/// primaries and secondaries from dim to 10⁶, and SplitMix64 colours with each channel
/// log-uniform over 2^−12 … 2^12.
fn agx_inputs() -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO];
    out.extend((0..64).map(|i| Vec3::splat((-14.0 + 28.0 * i as f32 / 63.0).exp2())));
    let hues = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
    ];
    for level in [1e-3, 0.18, 1.0, 20.0, 1e3, 1e6] {
        out.extend(hues.iter().map(|&h| h * level));
    }
    let unit = |x: u64| (x >> 40) as f32 / (1u64 << 24) as f32; // 24 random bits in [0, 1)
    let noise = (0_u64..).map(|i| {
        let a = splitmix64(1000 + 2 * i);
        let b = splitmix64(1001 + 2 * i);
        let ev = |u: f32| (u * 24.0 - 12.0).exp2();
        Vec3::new(ev(unit(a)), ev(unit(a << 24)), ev(unit(b)))
    });
    out.extend(noise.take(AGX_INPUTS - out.len()));
    out
}

/// 64 rays leaving static cameras at 15, 30, 60 and 1000 M (16 each), with the lens pass's
/// budget: 2000 steps and r_far = max(2·r, 1000) M (uni-rel tracer.rs).
///
/// Per camera: three rays inside the shadow (0, ½ and 0.9 of its angular radius α, Synge
/// 1966), three just outside it (1.25α, 1.5α, 2α; strong deflection), and ten from 30° to 180°
/// off the inward direction (weak deflection, then rays heading away). Each ray's azimuth
/// about the inward axis steps by the golden angle so the rays fill 3D. No ray comes within 5%
/// of b_c (`binet_cases_avoid_b_c_and_cover_both_outcomes`): there the exit direction depends
/// chaotically on the last bits of the input (the ray winds around the photon sphere
/// ~ln(1/|b/b_c − 1|) times), which no f32/f64 twin pair can match.
///
/// Why 1.25α and not closer: near b_c the exit direction grows like 1/(b − b_c), so the f32
/// rounding of a far camera's position (~6e-5 M at 1000 M) shows up in it. Measured on the
/// M3: at 1.1α from 1000 M (b = 1.1 b_c) the f32 GPU twin is 1.6e-3 rad from the f64 CPU twin,
/// within 2e-3 but with no margin; at 1.25α it is 1.3e-4 rad. That is f32 conditioning, not a
/// twin mismatch: from 15–60 M the same rays agree to ~2e-6 rad.
fn binet_cases() -> Vec<BinetCase> {
    let cameras = [
        (15.0, DVec3::new(2.0, -1.0, 2.0)),
        (30.0, DVec3::NEG_X),
        (60.0, DVec3::new(-1.0, 0.0, 0.2)),
        (1000.0, DVec3::new(0.3, 0.9, -0.4)),
    ];
    let golden_angle = PI * (3.0 - 5_f64.sqrt());
    let mut cases = Vec::with_capacity(BINET_RAYS);
    for (r, axis) in cameras {
        let cam = axis.normalize() * r;
        let inward = -cam / r;
        let e1 = inward.any_orthonormal_vector();
        let e2 = inward.cross(e1);
        let alpha = binet::shadow_angular_radius(r);
        let shadow = [0.0, 0.5, 0.9, 1.25, 1.5, 2.0].map(|k| k * alpha);
        let wide = [
            30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 140.0, 160.0, 180.0_f64,
        ]
        .map(f64::to_radians);
        for (k, delta) in shadow.into_iter().chain(wide).enumerate() {
            let psi = golden_angle * (cases.len() + k) as f64;
            let side = e1 * psi.cos() + e2 * psi.sin();
            let dir = inward * delta.cos() + side * delta.sin();
            let r_far = (2.0 * r).max(LENS_MIN_R_FAR);
            cases.push(BinetCase {
                cam: cam.as_vec3().extend(r_far as f32).to_array(),
                dir: dir.as_vec3().extend(0.0).to_array(),
                budget: [LENS_MAX_STEPS, 0, 0, 0],
            });
        }
    }
    cases
}

/// Uploads `input` to binding 0 of group [`PROBE_GROUP`], runs the probe's `cs_main` once per
/// element and reads back binding 1, which holds one `O` per input. Validation errors come back
/// as `Err`.
fn run_probe<I: Pod, O: Pod>(
    gpu: &GpuContext,
    lib: &ShaderLib,
    probe: &str,
    input: &[I],
) -> anyhow::Result<Vec<O>> {
    let device = &gpu.device;
    let module = lib.module(device, probe)?;
    let n = u32::try_from(input.len()).context("too many probe inputs")?;
    let out_size = (input.len() * size_of::<O>()) as u64;

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
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
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("selftest probe bgl"),
        entries: &[storage(0, true), storage(1, false)],
    });
    // Groups below PROBE_GROUP are left empty: the probe binds nothing there.
    let mut groups: Vec<Option<&wgpu::BindGroupLayout>> = vec![None; PROBE_GROUP as usize];
    groups.push(Some(&bgl));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("selftest probe layout"),
        bind_group_layouts: &groups,
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(probe),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("selftest input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer = |label: &str, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: out_size,
            usage,
            mapped_at_creation: false,
        })
    };
    let output_buffer = buffer(
        "selftest output",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        "selftest read-back",
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("selftest probe bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("selftest"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(probe),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(PROBE_GROUP, &bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(PROBE_WORKGROUP), 1, 1);
    }
    enc.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, out_size);
    gpu.queue.submit(std::iter::once(enc.finish()));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("{probe}: {err}");
    }

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        // The receiver outlives the wait below; a send error cannot be acted on here.
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("waiting for the probe")?;
    rx.try_recv()
        .context("probe read-back callback did not run")?
        .context("mapping the probe output")?;
    let out = {
        let view = readback
            .get_mapped_range(..)
            .context("reading the probe output")?;
        bytemuck::pod_collect_to_vec(&view)
    };
    readback.unmap();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;
    use uni_gfx::validate_wgsl;
    use wgpu::naga;

    fn probe_module(probe: &str) -> naga::Module {
        let src = probe_lib().compose(probe).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{probe}: {e}\n{src}"))
    }

    #[test]
    fn probes_validate_and_bind_only_group_2() {
        for probe in [PCG3D_PROBE, AGX_PROBE, BINET_PROBE] {
            let module = probe_module(probe);
            assert!(
                module.entry_points.iter().any(|e| e.name == "cs_main"
                    && e.stage == naga::ShaderStage::Compute
                    && e.workgroup_size == [PROBE_WORKGROUP, 1, 1]),
                "{probe}: no cs_main @workgroup_size(64)"
            );
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16,
            )
            .validate(&module)
            .unwrap();
            let (ep_index, _) = module
                .entry_points
                .iter()
                .enumerate()
                .find(|(_, e)| e.name == "cs_main")
                .unwrap();
            let used: Vec<naga::ResourceBinding> = module
                .global_variables
                .iter()
                .filter(|(h, _)| !info.get_entry_point(ep_index)[*h].is_empty())
                .filter_map(|(_, v)| v.binding)
                .collect();
            let expected = [0, 1].map(|binding| naga::ResourceBinding {
                group: PROBE_GROUP,
                binding,
            });
            assert_eq!(used, expected, "{probe}");
        }
    }

    /// (member name, offset) pairs of WGSL struct `name` in `module`, and its size.
    fn wgsl_layout(module: &naga::Module, name: &str) -> (Vec<(String, usize)>, usize) {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("struct {name}"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let fields = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        (fields, layouter[handle].size as usize)
    }

    fn named(fields: &[(&str, usize)]) -> Vec<(String, usize)> {
        fields.iter().map(|&(n, o)| (n.to_owned(), o)).collect()
    }

    #[test]
    fn binet_probe_structs_match_wgsl() {
        // Global Constraints: storage structs are vec4-only and match naga's layout.
        let module = probe_module(BINET_PROBE);
        let (fields, size) = wgsl_layout(&module, "BinetCase");
        assert_eq!(size, size_of::<BinetCase>());
        assert_eq!(
            fields,
            named(&[
                ("cam", offset_of!(BinetCase, cam)),
                ("dir", offset_of!(BinetCase, dir)),
                ("budget", offset_of!(BinetCase, budget)),
            ])
        );
        let (fields, size) = wgsl_layout(&module, "BinetOutcome");
        assert_eq!(size, size_of::<BinetOutcome>());
        assert_eq!(
            fields,
            named(&[
                ("v", offset_of!(BinetOutcome, v)),
                ("status", offset_of!(BinetOutcome, status)),
            ])
        );
    }

    #[test]
    fn binet_status_codes_match_wgsl() {
        let module = probe_module(BINET_PROBE);
        for (name, value) in [
            ("BINET_CAPTURED", BINET_CAPTURED),
            ("BINET_ESCAPED", BINET_ESCAPED),
            ("BINET_MAX_STEPS", BINET_MAX_STEPS),
        ] {
            let (_, c) = module
                .constants
                .iter()
                .find(|(_, c)| c.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("const {name}"));
            let naga::Expression::Literal(naga::Literal::U32(v)) =
                module.global_expressions[c.init]
            else {
                panic!("{name} is not a u32 literal");
            };
            assert_eq!(v, value, "{name}");
        }
    }

    #[test]
    fn input_counts_match_the_brief() {
        assert_eq!(pcg3d_inputs().len(), 4096);
        assert_eq!(agx_inputs().len(), 256);
        assert_eq!(binet_cases().len(), 64);
        let inputs = pcg3d_inputs();
        let mut unique = inputs.clone();
        unique.sort_by_key(|v| v.to_array());
        unique.dedup();
        assert_eq!(unique.len(), inputs.len(), "pcg3d inputs repeat");
        assert!(
            agx_inputs()
                .iter()
                .all(|c| c.is_finite() && c.min_element() >= 0.0)
        );
    }

    /// No selftest ray's impact parameter is within this fraction of b_c (see `binet_cases`).
    const NEAR_CRITICAL_MARGIN: f64 = 0.05;

    #[test]
    fn binet_cases_avoid_b_c_and_cover_both_outcomes() {
        let mut captured = 0;
        for c in binet_cases() {
            let r = Vec4::from_array(c.cam).truncate().as_dvec3().length();
            let b = case_b(&c);
            let off = (b / binet::CRITICAL_IMPACT - 1.0).abs();
            assert!(off > NEAR_CRITICAL_MARGIN, "b = {b} M is {off} from b_c");
            assert_eq!(c.cam[3] as f64, (2.0 * r).max(1000.0).round());
            assert_eq!(c.budget[0], 2000);
            match cpu_trace(&c) {
                RayEnd::Captured => captured += 1,
                RayEnd::Escaped { .. } => {}
                RayEnd::MaxSteps => panic!("ray from r = {r} ran out of steps"),
            }
        }
        // Three rays inside the shadow per camera.
        assert_eq!(captured, 12);
    }

    #[test]
    fn angle_between_is_exact_at_small_angles_and_pi_for_nan() {
        let a = DVec3::X;
        let b = DVec3::new(1.0, 1e-6, 0.0);
        assert!((angle_between(a, b) - 1e-6).abs() < 1e-15);
        assert_eq!(angle_between(a, DVec3::splat(f64::NAN)), PI);
    }
}
