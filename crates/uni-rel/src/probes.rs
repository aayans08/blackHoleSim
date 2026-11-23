//! GPU probes of uni-rel's WGSL against its CPU twins, run by `universe --selftest` (Task B7).
//!
//! `lens_env`: the lens pass's environment lookup, `lens_env_units` and `lens_env_lod` in
//! `shaders/lens_trace.wgsl` (the code `cs_resolve` runs for every escaped ray with
//! `LensEnv::Band`), on the GPU against a CPU lookup through `uni_gfx::env::cube_dir`'s
//! convention. A small test cube (6 × 8², 4 mips, Rgba32Float) is uploaded with every texel
//! holding its own address (face, level, row · size + column); 32 exit directions of rays traced
//! by the CPU Binet twin from 30 M are each sampled (nearest) through `lens_env_units`, at a
//! footprint chosen so that `lens_env_lod` lands on a whole level, covering levels 0–3 and both
//! clamps. The texel read must be the one the CPU lookup names, and the GPU level must match
//! [`crate::tracer::env_lod`]. This is the write-then-sample check of the cube convention that
//! A0a deferred, run through the lens's own sampling code (B5 checks the band's).
//!
//! M0's Binet check stays in the app's selftest (`selftest.rs`, ruling B.10).

use anyhow::Context as _;
use glam::{DVec3, UVec2, Vec2, Vec3};
use uni_gfx::probe::ProbeResult;
use uni_gfx::{GpuContext, ShaderLib};
use wgpu::util::DeviceExt as _;

use crate::binet;
use crate::tracer::env_lod;

/// Texels per face edge of the probe cube's level 0.
const PROBE_FACE: u32 = 8;
/// Mip levels of the probe cube (8, 4, 2, 1 texels).
const PROBE_LEVELS: u32 = 4;
/// Rays in the probe.
const PROBE_RAYS: usize = 32;
/// Direction of the probe camera from the hole (off every axis) and its distance [M], the
/// distance of the `bh-shadow-30M` bookmark.
const PROBE_CAMERA_DIR: DVec3 = DVec3::new(0.4, -0.5, 0.77);
const PROBE_CAMERA_R: f64 = 30.0;
/// Escape radius [M] and step budget of the CPU traces (ample: every probe ray escapes).
const PROBE_R_FAR: f64 = 1000.0;
const PROBE_MAX_STEPS: u32 = 1_000_000;
/// Levels (before clamping) the probe footprints aim at, cycled over the rays; −2 and 6 test
/// the clamps to 0 and to the last level.
const PROBE_TARGET_LEVELS: [i32; 6] = [-2, 0, 1, 2, 3, 6];
/// Largest |GPU − CPU| of the level. Both evaluate the same few f32 operations; a wrong
/// formula errs by ≥ 0.5.
const LOD_TOLERANCE: f32 = 1e-4;
/// Workgroup size of `cs_lens_env`.
const PROBE_WORKGROUP: u32 = 64;
/// Shader path of the `lens_env` probe.
const LENS_ENV_PROBE: &str = "probe/lens_env.wgsl";

/// The probe kernel: `lens_env_units` and `lens_env_lod` from the lens shader itself, on the
/// probe's own cube. Its four resources sit at group 0, bindings 8–11, clear of the camera's
/// (group 0, bindings 0–2) and the lens pass's (group 1) that the import declares.
const LENS_ENV_PROBE_WGSL: &str = r#"// probe/lens_env.wgsl: `universe --selftest` probe of the
// lens pass's environment lookup (Task B7; crates/uni-rel/src/probes.rs).
#import "rel/lens_trace.wgsl"

// xyz: exit direction (root axes); w: footprint [rad].
@group(0) @binding(8) var<storage, read> probe_in: array<vec4<f32>>;
// xyz: the texel lens_env_units read (face, level, row · size + column); w: lens_env_lod.
@group(0) @binding(9) var<storage, read_write> probe_out: array<vec4<f32>>;
@group(0) @binding(10) var probe_cube: texture_cube<f32>;
@group(0) @binding(11) var probe_nearest: sampler;

@compute @workgroup_size(64, 1, 1)
fn cs_lens_env(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let ray = probe_in[gid.x];
    let size = textureDimensions(probe_cube);
    let lod = lens_env_lod(ray.xyz, ray.w, f32(size.x), f32(textureNumLevels(probe_cube)));
    probe_out[gid.x] = vec4<f32>(lens_env_units(probe_cube, probe_nearest, ray.xyz, ray.w), lod);
}
"#;

/// Every uni-rel probe (`lens_env`), registering its shaders in `lib`.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    crate::register_shaders(lib);
    lib.add(LENS_ENV_PROBE, LENS_ENV_PROBE_WGSL);
    let (pass, detail) = check_lens_env(gpu, lib).unwrap_or_else(|e| (false, format!("{e:#}")));
    vec![ProbeResult {
        name: "lens_env".to_owned(),
        pass,
        detail,
    }]
}

/// Face and uv of the point where direction `dir` (any length) meets the cube, as a cube
/// sample selects it: the major axis picks the face and (s_c, t_c, r_c) come from the Vulkan
/// specification's "Cube Map Face Selection" table, which WebGPU's face orientations follow
/// (+X (−z, −y, x), −X (z, −y, x), +Y (x, z, y), −Y (x, −z, y), +Z (x, −y, z), −Z (−x, −y, z)),
/// with uv = (s_c/|r_c| + 1)/2, (t_c/|r_c| + 1)/2. The inverse of `uni_gfx::env::cube_dir`
/// (`cube_face_uv_inverts_cube_dir`).
fn cube_face_uv(dir: Vec3) -> (u32, Vec2) {
    let a = dir.abs();
    let (face, sc, tc, rc) = if a.x >= a.y && a.x >= a.z {
        if dir.x > 0.0 {
            (0, -dir.z, -dir.y, a.x)
        } else {
            (1, dir.z, -dir.y, a.x)
        }
    } else if a.y >= a.z {
        if dir.y > 0.0 {
            (2, dir.x, dir.z, a.y)
        } else {
            (3, dir.x, -dir.z, a.y)
        }
    } else if dir.z > 0.0 {
        (4, dir.x, -dir.y, a.z)
    } else {
        (5, -dir.x, -dir.y, a.z)
    };
    (face, (Vec2::new(sc, tc) / rc + 1.0) * 0.5)
}

/// The texel holding `uv` at a level `n` texels across (column, row).
fn texel_of(uv: Vec2, n: u32) -> UVec2 {
    (uv * n as f32)
        .floor()
        .clamp(Vec2::ZERO, Vec2::splat((n - 1) as f32))
        .as_uvec2()
}

/// Texels per face edge at `level` of the probe cube.
fn level_size(level: u32) -> u32 {
    PROBE_FACE >> level
}

/// What the probe cube holds at (face, level, texel): its own address, all small integers and
/// so exact in f32.
fn texel_value(face: u32, level: u32, texel: UVec2) -> [f32; 4] {
    let id = texel.y * level_size(level) + texel.x;
    [face as f32, level as f32, id as f32, 1.0]
}

/// One ray of the probe: the exit direction (unit, root axes), the footprint [rad] fed to
/// `lens_env_lod`, and the level that footprint selects after clamping.
struct ProbeRay {
    dir: Vec3,
    footprint: f32,
    level: u32,
}

/// The probe's 32 rays: from a static camera 30 M from the hole, at angles from the inward
/// radial running from just outside the shadow (9.6° at 30 M; Synge 1966) almost straight back
/// out, turned by the golden angle each time, traced by the CPU Binet twin to their exit
/// directions. Each footprint is 2^t texel angles (2m²/N, [`env_lod`]) for a target level t
/// from [`PROBE_TARGET_LEVELS`]. A ray that does not escape is left out (none does; the test
/// pins 32).
fn probe_rays() -> Vec<ProbeRay> {
    let n = PROBE_CAMERA_DIR.normalize();
    let cam = n * PROBE_CAMERA_R;
    let u = n.any_orthonormal_vector();
    let w = n.cross(u);
    (0..PROBE_RAYS)
        .filter_map(|k| {
            let kf = k as f64;
            let delta = (10.5 + 5.3 * kf).to_radians();
            let psi = (137.508 * kf).to_radians();
            let local = -n * delta.cos() + (u * psi.cos() + w * psi.sin()) * delta.sin();
            let v0 = binet::static_observer_ray(cam, local)?;
            let binet::RayEnd::Escaped { dir } =
                binet::trace(cam, v0, PROBE_R_FAR, PROBE_MAX_STEPS)
            else {
                return None;
            };
            let dir = dir.as_vec3().normalize();
            let target = PROBE_TARGET_LEVELS[k % PROBE_TARGET_LEVELS.len()];
            let m = dir.abs().max_element();
            let footprint = 2f32.powi(target) * 2.0 * m * m / PROBE_FACE as f32;
            let level = target.clamp(0, PROBE_LEVELS as i32 - 1) as u32;
            Some(ProbeRay {
                dir,
                footprint,
                level,
            })
        })
        .collect()
}

/// The texel value a nearest sample of `ray` must return: at its level, the texel the CPU
/// lookup names.
fn expected_texel(ray: &ProbeRay) -> [f32; 4] {
    let (face, uv) = cube_face_uv(ray.dir);
    texel_value(face, ray.level, texel_of(uv, level_size(ray.level)))
}

/// Uploads the probe cube, runs `cs_lens_env` over the rays and compares: (pass, detail).
fn check_lens_env(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let device = &gpu.device;
    let module = lib.module(device, LENS_ENV_PROBE)?;
    let rays = probe_rays();
    anyhow::ensure!(
        rays.len() == PROBE_RAYS,
        "lens_env: {} of {PROBE_RAYS} probe rays escaped",
        rays.len()
    );
    let input: Vec<[f32; 4]> = rays
        .iter()
        .map(|r| r.dir.extend(r.footprint).to_array())
        .collect();
    let n = u32::try_from(input.len()).context("too many lens_env rays")?;
    let out_bytes = size_of_val(input.as_slice()) as u64;

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let cube = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lens_env probe cube"),
        size: wgpu::Extent3d {
            width: PROBE_FACE,
            height: PROBE_FACE,
            depth_or_array_layers: 6,
        },
        mip_level_count: PROBE_LEVELS,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for level in 0..PROBE_LEVELS {
        let size = level_size(level);
        let texels: Vec<[f32; 4]> = (0..6)
            .flat_map(|face| {
                (0..size).flat_map(move |y| {
                    (0..size).map(move |x| texel_value(face, level, UVec2::new(x, y)))
                })
            })
            .collect();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &cube,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 16),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
        );
    }
    let cube_view = cube.create_view(&wgpu::TextureViewDescriptor {
        label: Some("lens_env probe cube view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    // Nearest in every filter (the default), so a sample returns one texel of one level.
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("lens_env probe nearest"),
        ..Default::default()
    });
    let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    let storage = |binding: u32, read_only: bool| {
        entry(
            binding,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        )
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lens_env probe bgl"),
        entries: &[
            storage(8, true),
            storage(9, false),
            entry(
                10,
                wgpu::BindingType::Texture {
                    // Rgba32Float needs FLOAT32_FILTERABLE to filter; nearest needs none.
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
            ),
            entry(
                11,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            ),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("lens_env probe"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("lens_env probe"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("cs_lens_env"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lens_env probe input"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer = |label: &str, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: out_bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let output = buffer(
        "lens_env probe output",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        "lens_env probe read-back",
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lens_env probe bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 8,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(&cube_view),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("lens_env probe"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lens_env probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(PROBE_WORKGROUP), 1, 1);
    }
    enc.copy_buffer_to_buffer(&output, 0, &readback, 0, out_bytes);
    gpu.queue.submit(std::iter::once(enc.finish()));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("lens_env: {err}");
    }

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        // The receiver outlives the wait below; a send error cannot be acted on here.
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("lens_env: waiting for the GPU")?;
    // The wait above runs the callback; `try_recv` turns a missing callback into an error.
    rx.try_recv()
        .context("lens_env: the read-back callback did not run")?
        .context("lens_env: mapping the output")?;
    let out: Vec<[f32; 4]> = {
        let view = readback
            .get_mapped_range(..)
            .context("lens_env: reading the output")?;
        bytemuck::pod_collect_to_vec(&view)
    };
    readback.unmap();
    anyhow::ensure!(out.len() == rays.len(), "lens_env: short read-back");

    let mut right_texel = 0;
    // Worst level error; a NaN anywhere makes it NaN, which fails the check.
    let mut worst_lod = 0.0_f32;
    for (ray, got) in rays.iter().zip(&out) {
        if got[..3] == expected_texel(ray)[..3] {
            right_texel += 1;
        }
        let cpu = env_lod(
            ray.dir,
            ray.footprint,
            PROBE_FACE as f32,
            PROBE_LEVELS as f32,
        );
        let err = (got[3] - cpu).abs();
        worst_lod = if worst_lod.is_nan() || err.is_nan() {
            f32::NAN
        } else {
            worst_lod.max(err)
        };
    }
    Ok((
        right_texel == rays.len() && worst_lod <= LOD_TOLERANCE,
        format!(
            "{n} exit directions (CPU Binet from {PROBE_CAMERA_R} M) through lens_env_units on a \
             6 × {PROBE_FACE}² cube with {PROBE_LEVELS} levels: right texel and level \
             {right_texel}/{n}, max |lod GPU − CPU| {worst_lod:.1e} (tolerance \
             {LOD_TOLERANCE:.0e})"
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_gfx::env::cube_dir;
    use uni_gfx::validate_wgsl;
    use wgpu::naga;

    /// The CPU lookup inverts `cube_dir`: every texel centre of every face comes back to its own
    /// face and uv, so it uses the same face order and orientation as the band's writes.
    #[test]
    fn cube_face_uv_inverts_cube_dir() {
        let n = PROBE_FACE;
        for face in 0..6 {
            for y in 0..n {
                for x in 0..n {
                    let uv = (Vec2::new(x as f32, y as f32) + 0.5) / n as f32;
                    let (f, back) = cube_face_uv(cube_dir(face, uv));
                    assert_eq!(f, face, "{face} {uv}");
                    assert!((back - uv).length() < 1e-6, "{face} {uv} → {back}");
                    assert_eq!(texel_of(back, n), UVec2::new(x, y));
                    // The direction's length does not matter.
                    assert_eq!(cube_face_uv(cube_dir(face, uv) * 3.5).0, face);
                }
            }
        }
    }

    /// 32 escaped rays whose exit directions reach all six faces, each well inside one texel
    /// (so GPU rounding cannot pick a neighbour) and at a footprint whose level (before
    /// clamping) is an integer, covering levels 0–3 and both clamps.
    #[test]
    fn probe_rays_are_exit_directions_clear_of_texel_edges() {
        let rays = probe_rays();
        assert_eq!(rays.len(), PROBE_RAYS);
        let mut faces = [0; 6];
        let mut levels = [0; PROBE_LEVELS as usize];
        for (k, ray) in rays.iter().enumerate() {
            assert!((ray.dir.length() - 1.0).abs() < 1e-6, "ray {k}");
            let a = ray.dir.abs();
            let mut sorted = a.to_array();
            sorted.sort_by(f32::total_cmp);
            assert!(sorted[2] - sorted[1] > 1e-3, "ray {k}: near a cube edge");
            let (face, uv) = cube_face_uv(ray.dir);
            faces[face as usize] += 1;
            // At least 0.01 texel (1.25e-3 in uv) from a level-0 texel edge, and so from every
            // coarser level's edges: ~1000× the f32 error of a GPU's cube coordinates.
            let texels = uv * PROBE_FACE as f32;
            let margin = (texels - texels.round()).abs().min_element();
            assert!(margin > 0.01, "ray {k}: {margin} texel from an edge");
            let lod = env_lod(
                ray.dir,
                ray.footprint,
                PROBE_FACE as f32,
                PROBE_LEVELS as f32,
            );
            assert!((lod - ray.level as f32).abs() < 1e-5, "ray {k}: {lod}");
            levels[ray.level as usize] += 1;
        }
        assert!(faces.iter().all(|&n| n > 0), "{faces:?}");
        assert!(levels.iter().all(|&n| n > 0), "{levels:?}");
        // Both clamps are exercised.
        assert!(PROBE_TARGET_LEVELS.iter().any(|&t| t < 0));
        assert!(PROBE_TARGET_LEVELS.iter().any(|&t| t >= PROBE_LEVELS as i32));
    }

    /// The cube's contents: every (face, level, texel) has its own value, so a wrong face,
    /// level, row or column reads back as a different texel.
    #[test]
    fn probe_texels_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for level in 0..PROBE_LEVELS {
            let size = level_size(level);
            assert!(size >= 1);
            for face in 0..6 {
                for y in 0..size {
                    for x in 0..size {
                        let v = texel_value(face, level, UVec2::new(x, y));
                        assert!(seen.insert(v.map(f32::to_bits)), "{v:?}");
                    }
                }
            }
        }
        assert_eq!(level_size(PROBE_LEVELS - 1), 1);
    }

    /// The probe shader composes, validates, and its entry point binds only its own four
    /// resources (group 0, bindings 8–11), none of the lens pass's or the camera's.
    #[test]
    fn lens_env_probe_shader_binds_only_its_own_resources() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        lib.add(LENS_ENV_PROBE, LENS_ENV_PROBE_WGSL);
        let src = lib.compose(LENS_ENV_PROBE).unwrap();
        let module = validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}"));
        let index = module
            .entry_points
            .iter()
            .position(|e| e.name == "cs_lens_env")
            .expect("entry point cs_lens_env");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16,
        )
        .validate(&module)
        .unwrap();
        let used = info.get_entry_point(index);
        let mut bindings: Vec<(u32, u32)> = module
            .global_variables
            .iter()
            .filter(|(h, _)| !used[*h].is_empty())
            .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
            .collect();
        bindings.sort();
        assert_eq!(bindings, [(0, 8), (0, 9), (0, 10), (0, 11)]);
        assert_eq!(PROBE_WORKGROUP, 64, "@workgroup_size(64, 1, 1)");
    }
}
