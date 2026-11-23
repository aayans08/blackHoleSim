//! The M0 debug-grid background, kept in M1 as an opt-in lens environment (ruling 14): the
//! unlensed debug grid over the whole frame.
//!
//! [`DebugSkyBackground`] is pass 3 (uni-gfx `targets` pass contract), before the opaque passes.
//! While the lens environment is [`LensEnv::DebugGrid`] and the lens is traced
//! ([`LensCoverage::Traced`]) it writes the debug grid (`shaders/debug_sky.wgsl`) into every
//! `hdr` pixel; the lens pass then overwrites only its traced rectangle with the same grid seen
//! through the lens. Everywhere else (the [`LensEnv::Band`] environment, no lens on screen, a
//! lens beyond the trace reach, or inside a horizon) it writes nothing. With `Band` the band pass
//! draws the sky around the lens rectangle instead, from the same cubemap the lens samples. The
//! HUD's "Lens background: M0 debug grid" label is shown under exactly the condition the grid
//! draws ([`lens_labels`]). The grid is display-referred, so the app runs Manual exposure
//! whenever `DebugGrid` is selected (M1 frame contract).

use anyhow::Context as _;
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, PassTimer, RenderFeature,
    ShaderLib, SharedInput,
};

use crate::LensEnv;
use crate::tracer::{LensCoverage, LensScale, lens_coverage};

/// Feature name, also the GPU timing label.
const NAME: &str = "background";
/// Workgroup edge of `cs_background` (`@workgroup_size(8, 8, 1)`).
const WORKGROUP: u32 = 8;

/// GPU twin of `struct BackgroundParams` in `shaders/debug_background.wgsl` (one vec4;
/// `background_params_layout_matches_wgsl` checks it with naga).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct BackgroundParams {
    /// x: 1 / the lens resolution scale (line-width factor); yzw: 0.
    scale: [f32; 4],
}

/// Whether the debug grid background draws for a frame with this lens coverage and lens
/// environment: only with [`LensEnv::DebugGrid`] and while the lens is traced, so the grid is
/// never shown where no lens bends it (m0.1 took the coverage only; with `DebugGrid` the result
/// is m0.1's).
pub fn debug_background_draws(coverage: LensCoverage, env: LensEnv) -> bool {
    env == LensEnv::DebugGrid && coverage == LensCoverage::Traced
}

/// Which lens honesty labels the HUD shows for this coverage and environment: (the debug-grid
/// label "Lens background: M0 debug grid", the unlensed-stars label "Stars near the hole are not
/// yet lensed (M3)"; the strings live in uni-ui, C4).
/// - The debug label exactly while the grid draws ([`debug_background_draws`]): `DebugGrid`
///   and `Traced`.
/// - The unlensed-stars label whenever the lens is traced, in either environment: the star
///   pass draws catalogue stars at their unlensed positions over the traced pixels (× (1 − r)),
///   so stars are not lensed until M3. Outside `Traced` no pixel is lensed (from inside a
///   horizon the frame is black; beyond the reach `BEYOND_REACH_WARNING` says so).
pub fn lens_labels(coverage: LensCoverage, env: LensEnv) -> (bool, bool) {
    (
        debug_background_draws(coverage, env),
        coverage == LensCoverage::Traced,
    )
}

/// Group 1 of `cs_background`: the params uniform and the hdr storage texture.
fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<BackgroundParams>() as u64),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: HDR_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
    ]
}

/// The debug-grid background (`RenderFeature` named "background").
pub struct DebugSkyBackground {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    scale: LensScale,
    /// The lens environment the app selects; the grid draws only with `DebugGrid`.
    mode: SharedInput<LensEnv>,
    /// Decided in `prepare` for this frame's `encode`.
    draws: bool,
    /// Group 1 over the current `hdr` view; `None` until first used and after a resize.
    binding: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

impl DebugSkyBackground {
    /// Registers the uni-rel shaders in `lib` and builds the pass. `camera_bgl` is
    /// `Renderer::camera_bind_group_layout()`; `scale` is the lens tracer's resolution-scale
    /// handle, so the unlensed lines are as wide as the traced ones; `mode` is the lens
    /// environment the app sets each frame (the pass draws only with `DebugGrid` and `Traced`).
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        scale: LensScale,
        mode: SharedInput<LensEnv>,
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let device = &gpu.device;
        let module = lib
            .module(device, crate::DEBUG_BACKGROUND_WGSL)
            .context("compiling the debug background shader")?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("debug background bgl"),
            entries: &layout_entries(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("debug background"),
            bind_group_layouts: &[Some(camera_bgl), Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("debug background"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("cs_background"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug background params"),
            size: size_of::<BackgroundParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the debug background pipeline: {err}");
        }
        Ok(Self {
            pipeline,
            bgl,
            params_buffer,
            scale,
            mode,
            draws: false,
            binding: None,
        })
    }
}

impl RenderFeature for DebugSkyBackground {
    fn name(&self) -> &'static str {
        NAME
    }

    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets) {
        self.draws = debug_background_draws(lens_coverage(ctx), self.mode.get());
        if !self.draws {
            return;
        }
        let params = BackgroundParams {
            scale: [1.0 / self.scale.get(), 0.0, 0.0, 0.0],
        };
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        if self
            .binding
            .as_ref()
            .is_none_or(|(view, _)| *view != targets.hdr_view)
        {
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("debug background bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
                    },
                ],
            });
            self.binding = Some((targets.hdr_view.clone(), bind_group));
        }
    }

    fn encode(
        &mut self,
        _gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if !self.draws {
            return;
        }
        let Some((_, bind_group)) = &self.binding else {
            return;
        };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(NAME),
            timestamp_writes: timer.compute_writes(NAME),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.dispatch_workgroups(
            targets.size.x.div_ceil(WORKGROUP),
            targets.size.y.div_ceil(WORKGROUP),
            1,
        );
    }

    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {
        self.binding = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;
    use uni_gfx::validate_wgsl;
    use wgpu::naga;

    fn module() -> naga::Module {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let src = lib.compose(crate::DEBUG_BACKGROUND_WGSL).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"))
    }

    #[test]
    fn background_params_layout_matches_wgsl() {
        let m = module();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("BackgroundParams"))
            .expect("struct BackgroundParams");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("not a struct");
        };
        assert_eq!(
            layouter[handle].size as usize,
            size_of::<BackgroundParams>()
        );
        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        assert_eq!(
            wgsl,
            [("scale".to_owned(), offset_of!(BackgroundParams, scale))]
        );
    }

    /// `cs_background` uses group 0's camera and exactly the group-1 bindings of its layout.
    #[test]
    fn bindings_match_the_layout() {
        let m = module();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16,
        )
        .validate(&m)
        .unwrap();
        assert_eq!(m.entry_points.len(), 1);
        assert_eq!(m.entry_points[0].name, "cs_background");
        let used = info.get_entry_point(0);
        let mut bindings: Vec<(u32, u32, String)> = m
            .global_variables
            .iter()
            .filter(|(h, _)| !used[*h].is_empty())
            .map(|(_, v)| {
                let rb = v.binding.as_ref().expect("resource binding");
                (rb.group, rb.binding, v.name.clone().unwrap_or_default())
            })
            .collect();
        bindings.sort();
        assert_eq!(
            bindings,
            [
                (0, 0, "camera".to_owned()),
                (1, 0, "background".to_owned()),
                (1, 1, "hdr_out".to_owned()),
            ]
        );
        let entries = layout_entries();
        assert_eq!(entries.map(|e| e.binding), [0, 1]);
        assert!(matches!(
            entries[1].ty,
            wgpu::BindingType::StorageTexture {
                format: HDR_FORMAT,
                ..
            }
        ));
    }

    /// Review I3 and B7: the grid is the lens's environment only while it is the selected one
    /// ([`LensEnv::DebugGrid`]), and it is drawn exactly when the lens is traced, never in sky
    /// views, beyond the trace reach or inside a horizon. With [`LensEnv::Band`] the band pass
    /// draws the sky around the lens instead.
    #[test]
    fn debug_background_only_in_debug_mode() {
        assert!(debug_background_draws(
            LensCoverage::Traced,
            LensEnv::DebugGrid
        ));
        assert!(!debug_background_draws(LensCoverage::Traced, LensEnv::Band));
        assert!(!debug_background_draws(
            LensCoverage::Unlensed,
            LensEnv::DebugGrid
        ));
        for c in [
            LensCoverage::Unlensed,
            LensCoverage::BeyondReach,
            LensCoverage::InsideHorizon,
        ] {
            for env in [LensEnv::DebugGrid, LensEnv::Band] {
                assert!(!debug_background_draws(c, env), "{c:?} {env:?}");
            }
        }
    }

    /// B7: "Lens background: M0 debug grid" exactly while the grid draws (DebugGrid and Traced);
    /// "Stars near the hole are not yet lensed (M3)" whenever the lens is traced, in either
    /// environment, because the star pass draws catalogue stars unlensed over the traced pixels.
    #[test]
    fn lens_labels_by_mode() {
        for env in [LensEnv::DebugGrid, LensEnv::Band] {
            let (debug, unlensed_stars) = lens_labels(LensCoverage::Traced, env);
            assert_eq!(debug, env == LensEnv::DebugGrid, "{env:?}");
            assert!(unlensed_stars, "{env:?}");
            for c in [
                LensCoverage::Unlensed,
                LensCoverage::BeyondReach,
                LensCoverage::InsideHorizon,
            ] {
                assert_eq!(lens_labels(c, env), (false, false), "{c:?} {env:?}");
            }
        }
        // The debug label follows the background pass exactly.
        for c in [
            LensCoverage::Unlensed,
            LensCoverage::Traced,
            LensCoverage::BeyondReach,
            LensCoverage::InsideHorizon,
        ] {
            for env in [LensEnv::DebugGrid, LensEnv::Band] {
                assert_eq!(lens_labels(c, env).0, debug_background_draws(c, env));
            }
        }
    }
}
