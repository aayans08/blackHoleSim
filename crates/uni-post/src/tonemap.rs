//! Tonemap render feature: exposure, AgX and dither from the HDR target to the output view (M0 Task 5).
//!
//! One fullscreen-triangle render pass (`shaders/tonemap.wgsl`): it reads `targets.hdr` 1:1
//! with `textureLoad`, multiplies by `exposure_scale(scene_ev100)`, applies [`crate::agx`],
//! adds a triangular-PDF interleaved-gradient-noise dither of ±0.5/255 in sRGB code space and
//! writes display-linear colour to the output view, which must be an sRGB view.

use anyhow::Context as _;
use uni_gfx::{
    CameraUniform, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
};

/// Path of `shaders/tonemap.wgsl` in the [`ShaderLib`]. Another shader can
/// `#import "post/tonemap.wgsl"` to call `agx` (the `--selftest` does).
pub const TONEMAP_SHADER: &str = "post/tonemap.wgsl";

/// Feature name, also the GPU timing label.
const NAME: &str = "tonemap";

/// Bind group indices: the camera (`camera` in common/camera.wgsl) and the HDR input
/// (`hdr_tex`, binding 0).
const CAMERA_GROUP: u32 = 0;
const HDR_GROUP: u32 = 1;

/// Registers uni-post's WGSL in `lib` ([`TONEMAP_SHADER`]). [`TonemapFeature::new`] calls it;
/// it is public so a shader can import `post/tonemap.wgsl` without building the feature.
pub fn register_shaders(lib: &mut ShaderLib) {
    lib.add(TONEMAP_SHADER, include_str!("../shaders/tonemap.wgsl"));
}

/// The pass writes display-linear values and relies on the view's sRGB encoding, so the
/// output must be an sRGB format (`Bgra8UnormSrgb` window view, `Rgba8UnormSrgb` capture).
fn check_output_format(format: wgpu::TextureFormat) -> anyhow::Result<()> {
    anyhow::ensure!(
        format.is_srgb(),
        "tonemap output must be an sRGB view format (the pass writes display-linear colour), \
         got {format:?}"
    );
    Ok(())
}

/// Exposure, AgX tonemap and dither from `targets.hdr` to the output view.
pub struct TonemapFeature {
    pipeline: wgpu::RenderPipeline,
    /// Layout of the HDR input group ([`HDR_GROUP`]).
    hdr_layout: wgpu::BindGroupLayout,
    /// This feature's copy of the camera uniform, bound at [`CAMERA_GROUP`].
    /// `RenderFeature::encode` does not receive the `Renderer`'s camera bind group, so the
    /// feature writes `CameraUniform::from_ctx` (the same value) into its own buffer.
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    /// Bind group for the current `targets.hdr_view`; `None` until the first `encode` and
    /// after a resize.
    hdr_binding: Option<HdrBinding>,
}

/// A bind group over one HDR view, with that view, so a stale group can be detected.
struct HdrBinding {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl TonemapFeature {
    /// Builds the pipeline for `output_format`, which must be an sRGB format.
    ///
    /// `camera_bgl` is `Renderer::camera_bind_group_layout()`. Registers
    /// `post/tonemap.wgsl` in `lib`. Shader and pipeline validation errors are returned
    /// rather than reaching the device's uncaptured-error handler.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        output_format: wgpu::TextureFormat,
    ) -> anyhow::Result<Self> {
        check_output_format(output_format)?;
        register_shaders(lib);
        let device = &gpu.device;
        let module = lib
            .module(device, TONEMAP_SHADER)
            .context("compiling the tonemap shader")?;

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let hdr_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap hdr bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                // Read with textureLoad only, so no filtering is required.
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap layout"),
            // Index = group: CAMERA_GROUP (0), HDR_GROUP (1).
            bind_group_layouts: &[Some(camera_bgl), Some(&hdr_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tonemap camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap camera bg"),
            layout: camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the tonemap pipeline: {err}");
        }

        Ok(Self {
            pipeline,
            hdr_layout,
            camera_buffer,
            camera_bind_group,
            hdr_binding: None,
        })
    }
}

/// The bind group for `targets.hdr_view`, rebuilt if `binding` was made for another view.
/// A free function over the two fields so the caller can keep borrowing the others.
fn hdr_bind_group<'a>(
    binding: &'a mut Option<HdrBinding>,
    layout: &wgpu::BindGroupLayout,
    device: &wgpu::Device,
    targets: &FrameTargets,
) -> &'a wgpu::BindGroup {
    if binding.as_ref().is_some_and(|b| b.view != targets.hdr_view) {
        *binding = None;
    }
    let b = binding.get_or_insert_with(|| HdrBinding {
        view: targets.hdr_view.clone(),
        bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap hdr bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
            }],
        }),
    });
    &b.bind_group
}

impl RenderFeature for TonemapFeature {
    fn name(&self) -> &'static str {
        NAME
    }

    /// Uploads the camera uniform (the pass reads `camera.exposure.x` = scene EV100).
    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, _targets: &FrameTargets) {
        gpu.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::from_ctx(ctx)),
        );
    }

    /// One fullscreen triangle into `output`, which covers every pixel (cleared first).
    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        targets: &FrameTargets,
        output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        let hdr_bg = hdr_bind_group(
            &mut self.hdr_binding,
            &self.hdr_layout,
            &gpu.device,
            targets,
        );
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(NAME),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timer.render_writes(NAME),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(CAMERA_GROUP, &self.camera_bind_group, &[]);
        pass.set_bind_group(HDR_GROUP, hdr_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Drops the bind group over the old HDR view (releasing that texture now); the next
    /// `encode` binds the new one.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {
        self.hdr_binding = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_gfx::validate_wgsl;
    use wgpu::naga;

    fn tonemap_module() -> naga::Module {
        let mut lib = ShaderLib::with_common();
        register_shaders(&mut lib);
        let src = lib.compose(TONEMAP_SHADER).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"))
    }

    #[test]
    fn tonemap_shader_validates() {
        let m = tonemap_module();

        // `fn agx(c: vec3<f32>) -> vec3<f32>`: the signature `--selftest` calls.
        let vec3f = naga::TypeInner::Vector {
            size: naga::VectorSize::Tri,
            scalar: naga::Scalar::F32,
        };
        let (_, agx) = m
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("agx"))
            .expect("fn agx in post/tonemap.wgsl");
        assert_eq!(agx.arguments.len(), 1);
        assert_eq!(m.types[agx.arguments[0].ty].inner, vec3f);
        let result = agx.result.as_ref().expect("agx returns a value");
        assert_eq!(m.types[result.ty].inner, vec3f);

        // The two entry points the pipeline names.
        let stage_of = |name: &str| {
            m.entry_points
                .iter()
                .find(|e| e.name == name)
                .map(|e| e.stage)
        };
        assert_eq!(stage_of("vs_main"), Some(naga::ShaderStage::Vertex));
        assert_eq!(stage_of("fs_main"), Some(naga::ShaderStage::Fragment));
    }

    #[test]
    fn tonemap_bindings_match_pipeline_layout() {
        // The Rust pipeline layout is [camera_bgl, hdr layout]; the shader must agree.
        let m = tonemap_module();
        let binding_of = |name: &str| {
            let (_, v) = m
                .global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("var {name} in post/tonemap.wgsl"));
            (v.binding.clone(), m.types[v.ty].inner.clone())
        };

        let (camera, _) = binding_of("camera");
        assert_eq!(
            camera,
            Some(naga::ResourceBinding {
                group: CAMERA_GROUP,
                binding: 0
            })
        );

        let (hdr, hdr_ty) = binding_of("hdr_tex");
        assert_eq!(
            hdr,
            Some(naga::ResourceBinding {
                group: HDR_GROUP,
                binding: 0
            })
        );
        // texture_2d<f32>, matching BindingType::Texture { Float, D2, multisampled: false }.
        assert_eq!(
            hdr_ty,
            naga::TypeInner::Image {
                dim: naga::ImageDimension::D2,
                arrayed: false,
                class: naga::ImageClass::Sampled {
                    kind: naga::ScalarKind::Float,
                    multi: false
                },
            }
        );
    }

    #[test]
    fn agx_importable_from_compute_shader() {
        // The `--selftest` pattern: a compute shader that imports post/tonemap.wgsl and runs
        // `agx` over a buffer. Its own bindings sit in group 2, clear of camera and hdr_tex.
        let mut lib = ShaderLib::with_common();
        register_shaders(&mut lib);
        lib.add(
            "test/agx_probe.wgsl",
            r#"#import "post/tonemap.wgsl"
@group(2) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;
@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) { return; }
    probe_out[gid.x] = vec4<f32>(agx(probe_in[gid.x].xyz), 1.0);
}
"#,
        );
        let src = lib.compose("test/agx_probe.wgsl").unwrap();
        if let Err(e) = validate_wgsl(&src) {
            panic!("{e}\n{src}");
        }
    }

    #[test]
    fn output_format_must_be_srgb() {
        // The fragment writes display-linear values; only an sRGB view encodes them.
        assert!(check_output_format(wgpu::TextureFormat::Bgra8UnormSrgb).is_ok());
        assert!(check_output_format(wgpu::TextureFormat::Rgba8UnormSrgb).is_ok());
        assert!(check_output_format(uni_gfx::CAPTURE_FORMAT).is_ok());
        assert!(check_output_format(wgpu::TextureFormat::Bgra8Unorm).is_err());
        assert!(check_output_format(uni_gfx::HDR_FORMAT).is_err());
    }
}
