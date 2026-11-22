//! GPU probes: small compute dispatches that check a WGSL routine against its CPU twin.
//!
//! Every WGSL maths routine that matters has a CPU twin and a probe (M1 global constraints).
//! Each render crate exports `gpu_probes(gpu, lib) -> Vec<ProbeResult>`, and `universe
//! --selftest` runs them all on the GPU; `cargo test` cannot reach the GPU. A probe is a compute
//! shader with a read-only storage input at binding 0 and a storage output at binding 1 of one
//! bind group, run by [`run_compute`]. `--selftest`'s M0 checks (`pcg3d`, `agx`, `binet`) run
//! through it too, so there is one GPU read-back path to fix.

use anyhow::{Context as _, anyhow};
use glam::{Vec2, Vec3};
use wgpu::util::DeviceExt as _;

use crate::context::GpuContext;
use crate::env::cube_dir;
use crate::shader::ShaderLib;

/// The outcome of one GPU probe, printed by `universe --selftest` as `PASS name: detail` or
/// `FAIL name: detail`.
#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub name: String,
    pub pass: bool,
    pub detail: String,
}

/// Runs compute entry point `entry` of the (composed) WGSL `wgsl` once and returns its output.
///
/// The shader reads `input` from a `var<storage, read>` at binding 0 and writes
/// `output_bytes` bytes to a `var<storage, read_write>` at binding 1. Both sit in one bind
/// group, found from the shader, so a probe can import files that claim lower groups; the entry
/// point may use no other binding. `workgroups` is the dispatch size. Blocks until the GPU has
/// finished and the output is read back. Errors: an empty input, an output size that is not a
/// positive multiple of 4, a shader that does not parse or uses other bindings, and any GPU
/// validation error.
pub fn run_compute(
    gpu: &GpuContext,
    wgsl: &str,
    entry: &str,
    input: &[u8],
    output_bytes: u64,
    workgroups: [u32; 3],
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(!input.is_empty(), "probe {entry}: the input is empty");
    anyhow::ensure!(
        output_bytes > 0 && output_bytes % wgpu::COPY_BUFFER_ALIGNMENT == 0,
        "probe {entry}: output_bytes {output_bytes} is not a positive multiple of 4"
    );
    let parsed = naga::front::wgsl::parse_str(wgsl)
        .map_err(|e| anyhow!("probe {entry}: {}", e.emit_to_string(wgsl)))?;
    let group = probe_group(&parsed, entry)?;

    let device = &gpu.device;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
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
        label: Some("probe bgl"),
        entries: &[storage(0, true), storage(1, false)],
    });
    // Groups below the probe's are left empty: the entry point binds nothing there.
    let mut groups: Vec<Option<&wgpu::BindGroupLayout>> = vec![None; group as usize];
    groups.push(Some(&bgl));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("probe layout"),
        bind_group_layouts: &groups,
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probe input"),
        contents: input,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer = |label: &str, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: output_bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let output_buffer = buffer(
        "probe output",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        "probe read-back",
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe bg"),
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
        label: Some("probe"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(group, &bind_group, &[]);
        let [x, y, z] = workgroups;
        pass.dispatch_workgroups(x, y, z);
    }
    enc.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes);
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
    // The wait above runs the callback; `try_recv` turns a missing callback into an error.
    rx.try_recv()
        .with_context(|| format!("probe {entry}: the read-back callback did not run"))?
        .with_context(|| format!("probe {entry}: mapping the output"))?;
    let out = {
        let view = readback
            .get_mapped_range(..)
            .with_context(|| format!("probe {entry}: reading the output"))?;
        view.to_vec()
    };
    readback.unmap();
    Ok(out)
}

/// The bind group holding the probe's input (binding 0, `var<storage, read>`) and output
/// (binding 1, `var<storage, read_write>`), from the bindings compute entry point `entry`
/// actually uses. An error names any other binding it uses, since `run_compute` binds nothing
/// else.
fn probe_group(module: &naga::Module, entry: &str) -> anyhow::Result<u32> {
    use naga::{AddressSpace, StorageAccess};
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| anyhow!("probe {entry}: {e}"))?;
    let index = module
        .entry_points
        .iter()
        .position(|ep| ep.name == entry && ep.stage == naga::ShaderStage::Compute)
        .with_context(|| format!("probe: no compute entry point {entry:?}"))?;
    let used = info.get_entry_point(index);
    let (mut input, mut output, mut others) = (None, None, Vec::new());
    for (handle, var) in module.global_variables.iter() {
        let Some(rb) = &var.binding else {
            continue; // private and workgroup variables
        };
        if used[handle].is_empty() {
            continue;
        }
        match (rb.binding, var.space) {
            (0, AddressSpace::Storage { access })
                if access == StorageAccess::LOAD && input.is_none() =>
            {
                input = Some(rb.group);
            }
            (1, AddressSpace::Storage { access })
                if access.contains(StorageAccess::STORE) && output.is_none() =>
            {
                output = Some(rb.group);
            }
            _ => others.push(format!(
                "{} (group {}, binding {})",
                var.name.as_deref().unwrap_or("?"),
                rb.group,
                rb.binding
            )),
        }
    }
    anyhow::ensure!(
        others.is_empty(),
        "probe {entry} uses bindings run_compute does not provide: {}",
        others.join(", ")
    );
    match (input, output) {
        (Some(i), Some(o)) if i == o => Ok(i),
        _ => anyhow::bail!(
            "probe {entry} needs var<storage, read> at binding 0 and var<storage, read_write> at \
             binding 1 of one bind group (found input group {input:?}, output group {output:?})"
        ),
    }
}

/// Shader path of the `cube_dir` probe.
const CUBE_DIR_PROBE: &str = "probe/cube_dir.wgsl";

/// Runs `cube_dir` from common/cube.wgsl on each input.
const CUBE_DIR_PROBE_WGSL: &str = r#"// probe/cube_dir.wgsl: `universe --selftest` probe for cube_dir.
#import "common/cube.wgsl"

// x: face, yz: uv, w: unused.
@group(0) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
// xyz: cube_dir(face, uv), w: 0.
@group(0) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    probe_out[gid.x] = vec4<f32>(cube_dir(u32(c.x), c.yz), 0.0);
}
"#;

/// Largest |GPU − CPU| of a `cube_dir` direction. Both are unit vectors from the same few
/// operations; a wrong face or sign errs by ≥ 0.1.
const CUBE_DIR_TOLERANCE: f32 = 1e-5;

/// Every face on a 9 × 9 grid of uv from 0 to 1, edges and corners included: (face, u, v, 0).
fn cube_dir_inputs() -> Vec<[f32; 4]> {
    let mut inputs = Vec::with_capacity(6 * 81);
    for face in 0..6 {
        for i in 0..=8 {
            for j in 0..=8 {
                inputs.push([face as f32, i as f32 / 8.0, j as f32 / 8.0, 0.0]);
            }
        }
    }
    inputs
}

/// The probes for uni-gfx's own WGSL (`cube_dir`), registering their shaders in `lib`.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    lib.add(CUBE_DIR_PROBE, CUBE_DIR_PROBE_WGSL);
    let (pass, detail) = match check_cube_dir(gpu, lib) {
        Ok(result) => result,
        Err(e) => (false, format!("{e:#}")),
    };
    vec![ProbeResult {
        name: "cube_dir".to_owned(),
        pass,
        detail,
    }]
}

/// `cube_dir` on the GPU against `uni_gfx::env::cube_dir`: (pass, detail).
fn check_cube_dir(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let inputs = cube_dir_inputs();
    let src = lib.compose(CUBE_DIR_PROBE)?;
    let n = u32::try_from(inputs.len()).context("too many cube_dir inputs")?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        bytemuck::cast_slice(&inputs),
        std::mem::size_of_val(inputs.as_slice()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&bytes);
    // Worst error; a NaN anywhere makes it NaN, which fails the check.
    let worst = inputs
        .iter()
        .zip(&out)
        .map(|(c, g)| {
            let cpu = cube_dir(c[0] as u32, Vec2::new(c[1], c[2]));
            (cpu - Vec3::from_slice(&g[..3])).length()
        })
        .fold(0.0_f32, |a, e| {
            if a.is_nan() || e.is_nan() {
                f32::NAN
            } else {
                a.max(e)
            }
        });
    Ok((
        worst <= CUBE_DIR_TOLERANCE,
        format!(
            "{} directions, max |GPU − CPU| {worst:.1e} (tolerance {CUBE_DIR_TOLERANCE:.0e})",
            inputs.len()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::ShaderLib;

    fn module(src: &str) -> naga::Module {
        naga::front::wgsl::parse_str(src).unwrap()
    }

    const IO: &str = "@group(GROUP) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(GROUP) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;
";

    fn probe_src(group: u32, body: &str) -> String {
        format!(
            "{}@compute @workgroup_size(64)\nfn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n{body}\n}}\n",
            IO.replace("GROUP", &group.to_string())
        )
    }

    #[test]
    fn probe_group_is_found_from_the_shader() {
        let src = probe_src(2, "probe_out[gid.x] = probe_in[gid.x];");
        assert_eq!(probe_group(&module(&src), "cs_main").unwrap(), 2);
        let src = probe_src(0, "probe_out[gid.x] = probe_in[gid.x] * 2.0;");
        assert_eq!(probe_group(&module(&src), "cs_main").unwrap(), 0);

        // The cube_dir probe imports a common file and binds group 0.
        let mut lib = ShaderLib::with_common();
        lib.add(CUBE_DIR_PROBE, CUBE_DIR_PROBE_WGSL);
        let src = lib.compose(CUBE_DIR_PROBE).unwrap();
        crate::shader::validate_wgsl(&src).unwrap();
        assert_eq!(probe_group(&module(&src), "cs_main").unwrap(), 0);
    }

    #[test]
    fn probe_group_rejects_other_bindings_and_missing_io() {
        // Uses the camera uniform as well: run_compute cannot bind it.
        let mut lib = ShaderLib::with_common();
        lib.add(
            "test/probe.wgsl",
            "#import \"common/camera.wgsl\"\n@group(1) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;\n@group(1) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;\n@compute @workgroup_size(64)\nfn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    probe_out[gid.x] = probe_in[gid.x] * scene_pre();\n}\n",
        );
        let src = lib.compose("test/probe.wgsl").unwrap();
        let err = probe_group(&module(&src), "cs_main").unwrap_err();
        assert!(format!("{err:#}").contains("exposure"), "{err:#}");

        // Input and output in different groups.
        let src = "@group(0) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) { probe_out[gid.x] = probe_in[gid.x]; }
";
        assert!(probe_group(&module(src), "cs_main").is_err());

        // A writable input, or a missing entry point.
        let src = probe_src(0, "probe_out[gid.x] = probe_in[gid.x];").replace(
            "var<storage, read> probe_in",
            "var<storage, read_write> probe_in",
        );
        assert!(probe_group(&module(&src), "cs_main").is_err());
        let src = probe_src(0, "probe_out[gid.x] = probe_in[gid.x];");
        assert!(probe_group(&module(&src), "cs_other").is_err());
    }

    #[test]
    fn cube_dir_probe_inputs_cover_every_face_and_edge() {
        let inputs = cube_dir_inputs();
        assert_eq!(inputs.len(), 6 * 9 * 9);
        for face in 0..6 {
            assert!(
                inputs
                    .iter()
                    .any(|c| c[0] == face as f32 && c[1] == 0.0 && c[2] == 1.0)
            );
        }
    }
}
