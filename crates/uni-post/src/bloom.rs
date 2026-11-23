//! Bloom, pass "bloom" (Task B6), pass 15 of the M1 frame contract.
//!
//! Jimenez 2014 (SIGGRAPH Advances in Real-Time Rendering, "Next Generation Post Processing in
//! Call of Duty: Advanced Warfare"): a 6-level 13-tap downsample with a Karis average on the
//! first level, a 3 × 3 tent upsample, hdr = mix(hdr, bloom, strength), no threshold. The first
//! downsample clamps its input to [`BLOOM_INPUT_MAX`]. (The LearnOpenGL "Physically Based Bloom"
//! article was read as a reference only; its CC BY-NC code is not used.)
//!
//! The chain, for levels 1..=L (L = [`BLOOM_LEVELS`], each level half the one above):
//! - d₁ = down(clamp(hdr)) with the Karis average, d_k = down(d_{k−1}): the 13 bilinear taps
//!   at ±2, ±1, 0 source texels around the destination texel, grouped into the centre block of
//!   four taps (weight ½) and four corner blocks (⅛ each). The Karis average is Jimenez's partial
//!   form (slide "Fireflies – Partial Karis Average"): each block's weight is divided by
//!   1 + its luminance, then renormalised.
//! - u_L = d_L, u_k = d_k + tent(u_{k+1}): a 3 × 3 tent ([1 2 1] ⊗ [1 2 1] / 16) of bilinear
//!   taps one lower-level texel apart.
//! - bloom = tent(u₁) / L at full resolution, mixed by the composite render pass with blend
//!   factors (constant, 1 − constant), the constant = strength.
//!
//! Each step is normalised, so away from the frame's edges every level carries the input's
//! energy, and averaging the L levels conserves it (`bloom_impulse_energy_conserved`): bloom
//! moves light, it does not add any. Above the clamp, mix() takes strength × (hdr −
//! [`BLOOM_INPUT_MAX`]) from a pixel without spreading it; such a pixel is over 2 stops above
//! AgX white either way.

use anyhow::Context as _;
use glam::{UVec2, Vec2, Vec3};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, PassTimer, RenderFeature,
    ShaderLib, SharedInput,
};
use wgpu::util::DeviceExt as _;

/// Default bloom strength, the mix weight of the bloom chain (plan B6; an aesthetic choice of a
/// camera's or eye's faint scatter, not a measured value).
pub const DEFAULT_BLOOM_STRENGTH: f32 = 0.04;
/// The first downsample clamps its pre-exposed input to this: 2 stops over AgX white
/// (4 × 16.29 ≈ 65; uni-post `agx`), so the Sun's saturated glare does not wash out the faint
/// highlighted asteroid belt (plan B6's arithmetic).
pub const BLOOM_INPUT_MAX: f32 = 64.0;
/// Levels of the pyramid (plan B6; the fallback ladder is 6 → 5 → 4). Fewer when the frame is
/// too small to halve that often.
pub const BLOOM_LEVELS: usize = 6;

/// The first downsample's input clamp, per texel: `c` scaled so its largest channel is at most
/// [`BLOOM_INPUT_MAX`] (negative channels to 0). Scaling, not a per-channel min, keeps a bright
/// star's colour in its glow. Twin of `bloom_input` in `bloom.wgsl`.
pub fn bloom_input(c: Vec3) -> Vec3 {
    let v = c.max(Vec3::ZERO);
    v * (BLOOM_INPUT_MAX / v.max_element().max(MIN_CHANNEL)).min(1.0)
}

/// Floor of the largest channel in [`bloom_input`]'s division, so black stays black (a normal
/// f32, kept by Metal's fast math).
const MIN_CHANNEL: f32 = 1e-30;
/// Weights of the 13-tap pattern's blocks of four taps: the centre block ½, each corner block ⅛
/// (Jimenez 2014). They sum to 1.
const CENTRE_WEIGHT: f32 = 0.5;
const CORNER_WEIGHT: f32 = 0.125;
/// Relative luminance weights of linear Rec. 709 for the Karis average: ITU-R BT.709-6 (2015),
/// item 3.2 (`luminance` in common/color.wgsl).
const REC709_Y: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// The sizes of the pyramid's levels 1..=L for a frame of `size`: each half the one above it
/// (rounded down, at least 1), with L = [`BLOOM_LEVELS`] or fewer for a frame too small to halve
/// that often (the mips of a texture of the level-1 size).
pub fn bloom_level_sizes(size: UVec2) -> Vec<UVec2> {
    let first = (size / 2).max(UVec2::ONE);
    // floor(log2(largest side)) + 1 mips.
    let mips = (u32::BITS - first.max_element().leading_zeros()) as usize;
    (0..mips.min(BLOOM_LEVELS) as u32)
        .map(|k| UVec2::new((first.x >> k).max(1), (first.y >> k).max(1)))
        .collect()
}

/// CPU twin of the whole chain (`bloom.wgsl`): the full-resolution bloom of a `size` frame
/// `input` (row-major, top row first) that the composite mixes into `hdr` with weight
/// `strength`. `karis` turns the first level's Karis average on (the GPU always has it on; off,
/// the chain conserves energy away from the edges).
pub fn bloom_reference(input: &[Vec3], size: UVec2, karis: bool) -> Vec<Vec3> {
    let source = Image::new(size, input.to_vec());
    let sizes = bloom_level_sizes(size);
    let mut down: Vec<Image> = Vec::with_capacity(sizes.len());
    for &level in &sizes {
        let next = match down.last() {
            None => downsample_13(&source, level, Some(karis)),
            Some(above) => downsample_13(above, level, None),
        };
        down.push(next);
    }
    // u_L = d_L, u_k = d_k + tent(u_{k+1}), from the smallest level up.
    let Some(top) = down
        .into_iter()
        .rev()
        .reduce(|lower, same| tent_upsample(&lower, same.size, Some(&same)))
    else {
        return vec![Vec3::ZERO; input.len()]; // unreachable: there is always a level 1
    };
    let scale = 1.0 / sizes.len() as f32;
    tent_upsample(&top, size, None)
        .px
        .into_iter()
        .map(|v| v * scale)
        .collect()
}

/// A row-major image (top row first) read with clamp-to-edge addressing, like the chain's
/// textures and sampler.
#[derive(Clone)]
struct Image {
    size: UVec2,
    px: Vec<Vec3>,
}

impl Image {
    /// `px` must hold `size.x × size.y` texels.
    fn new(size: UVec2, px: Vec<Vec3>) -> Self {
        assert_eq!(
            px.len(),
            (size.x * size.y) as usize,
            "bloom image: {} texels for {size}",
            px.len()
        );
        Self { size, px }
    }

    /// Texel (x, y), clamped to the edges.
    fn texel(&self, x: i32, y: i32) -> Vec3 {
        let x = x.clamp(0, self.size.x as i32 - 1) as u32;
        let y = y.clamp(0, self.size.y as i32 - 1) as u32;
        self.px[(y * self.size.x + x) as usize]
    }

    /// Bilinear sample at `uv` (0..1 across the image; texel centres at (i + ½)/size), clamp to
    /// edge: the linear sampler of bloom.wgsl.
    fn sample(&self, uv: Vec2) -> Vec3 {
        let p = uv * self.size.as_vec2() - 0.5;
        let p0 = p.floor();
        let f = p - p0;
        let (x, y) = (p0.x as i32, p0.y as i32);
        let row = |y: i32| self.texel(x, y).lerp(self.texel(x + 1, y), f.x);
        row(y).lerp(row(y + 1), f.y)
    }

    /// Builds a `size` image texel by texel.
    fn from_fn(size: UVec2, f: impl Fn(u32, u32) -> Vec3) -> Self {
        let px = (0..size.y)
            .flat_map(|y| (0..size.x).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect();
        Self::new(size, px)
    }
}

/// The 13-tap result from its five blocks [centre, four corners], each the mean of four taps:
/// weights ½ and ⅛, each divided by 1 + the block's luminance with the Karis average (Jimenez
/// 2014, partial Karis average), renormalised. Twin of `bloom_combine` in bloom.wgsl.
fn combine(blocks: [Vec3; 5], karis: bool) -> Vec3 {
    let mut w = [
        CENTRE_WEIGHT,
        CORNER_WEIGHT,
        CORNER_WEIGHT,
        CORNER_WEIGHT,
        CORNER_WEIGHT,
    ];
    if karis {
        for (w, b) in w.iter_mut().zip(blocks) {
            *w /= 1.0 + REC709_Y.dot(b);
        }
    }
    let sum = w[0] * blocks[0]
        + w[1] * blocks[1]
        + w[2] * blocks[2]
        + w[3] * blocks[3]
        + w[4] * blocks[4];
    sum / (w[0] + w[1] + w[2] + w[3] + w[4])
}

/// Origins (x, y) in the 6 × 6 source block of the five 4 × 4 blocks of the first downsample:
/// centre (taps at ±1), then the corners (taps at {−2, 0} or {0, 2} on each axis).
const FIRST_BLOCKS: [(usize, usize); 5] = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2)];

/// One downsample of `src` to `dst_size` with the 13-tap pattern. `first = Some(karis)` is the
/// first level (`cs_down_first`): it reads the 6 × 6 source texels 2·dst − 2 ..= 2·dst + 3
/// one by one, clamped with [`bloom_input`], and each block is the mean of its 4 × 4 texels
/// (= its four 2 × 2 taps). `None` is a later level (`cs_down`): bilinear taps at the texel
/// corners 2·dst + 1 + offset.
fn downsample_13(src: &Image, dst_size: UVec2, first: Option<bool>) -> Image {
    Image::from_fn(dst_size, |i, j| match first {
        Some(karis) => first_level_texel(src, i, j, true, karis),
        None => {
            let corner = 2.0 * Vec2::new(i as f32, j as f32) + 1.0;
            let texel = 1.0 / src.size.as_vec2();
            let tap = |ox: f32, oy: f32| src.sample((corner + Vec2::new(ox, oy)) * texel);
            let centre = (tap(-1.0, -1.0) + tap(1.0, -1.0) + tap(-1.0, 1.0) + tap(1.0, 1.0)) * 0.25;
            let tl = (tap(-2.0, -2.0) + tap(0.0, -2.0) + tap(-2.0, 0.0) + tap(0.0, 0.0)) * 0.25;
            let tr = (tap(0.0, -2.0) + tap(2.0, -2.0) + tap(0.0, 0.0) + tap(2.0, 0.0)) * 0.25;
            let bl = (tap(-2.0, 0.0) + tap(0.0, 0.0) + tap(-2.0, 2.0) + tap(0.0, 2.0)) * 0.25;
            let br = (tap(0.0, 0.0) + tap(2.0, 0.0) + tap(0.0, 2.0) + tap(2.0, 2.0)) * 0.25;
            combine([centre, tl, tr, bl, br], false)
        }
    })
}

/// Level-1 texel (i, j) of `src`: the 6 × 6 source block, optionally clamped, its five 4 × 4
/// blocks, combined with or without the Karis average.
fn first_level_texel(src: &Image, i: u32, j: u32, clamp: bool, karis: bool) -> Vec3 {
    let (bx, by) = (2 * i as i32 - 2, 2 * j as i32 - 2);
    let mut t = [[Vec3::ZERO; 6]; 6];
    for (y, row) in t.iter_mut().enumerate() {
        for (x, v) in row.iter_mut().enumerate() {
            let c = src.texel(bx + x as i32, by + y as i32);
            *v = if clamp { bloom_input(c) } else { c };
        }
    }
    let blocks = FIRST_BLOCKS.map(|(ox, oy)| {
        let mut s = Vec3::ZERO;
        for row in &t[oy..oy + 4] {
            for &v in &row[ox..ox + 4] {
                s += v;
            }
        }
        s / 16.0
    });
    combine(blocks, karis)
}

/// Tent weights of the 3 × 3 upsample per axis ([1 2 1] / 4; Jimenez 2014).
const TENT: [f32; 3] = [1.0, 2.0, 1.0];

/// `low` upsampled to `dst_size` with the 3 × 3 tent of bilinear taps one `low` texel apart,
/// plus `same` (a level of `dst_size`) when given: `cs_up`, or the composite's bloom without
/// `same`.
fn tent_upsample(low: &Image, dst_size: UVec2, same: Option<&Image>) -> Image {
    let texel = 1.0 / low.size.as_vec2();
    let inv = 1.0 / dst_size.as_vec2();
    Image::from_fn(dst_size, |i, j| {
        let uv = (Vec2::new(i as f32, j as f32) + 0.5) * inv;
        let mut sum = Vec3::ZERO;
        for (oy, wy) in TENT.iter().enumerate() {
            for (ox, wx) in TENT.iter().enumerate() {
                let o = Vec2::new(ox as f32 - 1.0, oy as f32 - 1.0);
                sum += wx * wy * low.sample(uv + o * texel);
            }
        }
        let up = sum / 16.0;
        same.map_or(up, |s| s.px[(j * dst_size.x + i) as usize] + up)
    })
}

/// Twin of `struct BloomParams` in `bloom.wgsl` (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BloomParamsGpu {
    /// x: 1 / the number of levels (the composite's normalisation); yzw 0.
    pub(crate) scale: [f32; 4],
}

/// Feature name and the GPU timing label of the compute chain.
const NAME: &str = "bloom";
/// GPU timing label of the composite render pass.
const COMPOSITE_NAME: &str = "bloom composite";
/// Bind group indices: the renderer's group 0 and the bloom's own.
const CAMERA_GROUP: u32 = 0;
const BLOOM_GROUP: u32 = 1;
/// Workgroup edge of the compute passes (8 × 8 threads, one per destination texel).
const WORKGROUP_EDGE: u32 = 8;
/// Format of the pyramid's textures (filterable, and a storage format on every adapter).
const PYRAMID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The bloom pass (`RenderFeature` named "bloom"): one compute pass for the pyramid ("bloom") and
/// one render pass mixing it into `hdr` ("bloom composite"). Push it after the last sky pass and
/// before the tonemap (M1 frame contract, pass 15).
pub struct Bloom {
    down_first: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    up: wgpu::ComputePipeline,
    composite: wgpu::RenderPipeline,
    /// Group 1 of the downsamples: 0 source, 1 destination (storage), 2 sampler.
    down_layout: wgpu::BindGroupLayout,
    /// Group 1 of the upsamples: the down layout + 3 the same-level downsample.
    up_layout: wgpu::BindGroupLayout,
    /// Group 1 of the composite: 0 u₁, 2 sampler, 4 [`BloomParamsGpu`].
    composite_layout: wgpu::BindGroupLayout,
    /// Linear, clamp to edge.
    sampler: wgpu::Sampler,
    strength: SharedInput<f32>,
    /// This frame's mix weight in [0, 1], read from `strength` in `prepare`.
    weight: f32,
    /// The pyramid for the current `hdr` view; `None` before the first frame and after a resize.
    chain: Option<Chain>,
}

/// The pyramid's textures, as bind groups, for one `hdr` view.
struct Chain {
    /// The `hdr` view this chain reads and writes.
    hdr_view: wgpu::TextureView,
    /// Level 1 first: the bind group producing that level and its size.
    down: Vec<(wgpu::BindGroup, UVec2)>,
    /// Level L − 1 first, down to level 1: the bind group of u_k = d_k + tent(u_{k+1}), size.
    up: Vec<(wgpu::BindGroup, UVec2)>,
    composite: wgpu::BindGroup,
    /// The textures and the params buffer the bind groups use, held for the chain's lifetime.
    _resources: (wgpu::Texture, Option<wgpu::Texture>, wgpu::Buffer),
}

impl Bloom {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`; `strength` is the
    /// mix weight, set by the app (HUD), [`DEFAULT_BLOOM_STRENGTH`] by default.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        strength: SharedInput<f32>,
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let device = &gpu.device;
        let module = lib
            .module(device, crate::BLOOM_WGSL)
            .context("compiling the bloom shader")?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let down_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom down bgl"),
            entries: &compute_layout_entries(false),
        });
        let up_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom up bgl"),
            entries: &compute_layout_entries(true),
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom composite bgl"),
            entries: &composite_layout_entries(),
        });
        let pipeline_layout = |label: &str, layout: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                // Index = group: CAMERA_GROUP (0), BLOOM_GROUP (1).
                bind_group_layouts: &[Some(camera_bgl), Some(layout)],
                immediate_size: 0,
            })
        };
        let down_pl = pipeline_layout("bloom down layout", &down_layout);
        let up_pl = pipeline_layout("bloom up layout", &up_layout);
        let compute = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let down_first = compute("cs_down_first", &down_pl);
        let down = compute("cs_down", &down_pl);
        let up = compute("cs_up", &up_pl);
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(COMPOSITE_NAME),
            layout: Some(&pipeline_layout(
                "bloom composite layout",
                &composite_layout,
            )),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_composite"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_composite"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    // hdr = bloom · strength + hdr · (1 − strength), strength = the blend
                    // constant: mix(hdr, bloom, strength). Alpha is left alone.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Constant,
                            dst_factor: wgpu::BlendFactor::OneMinusConstant,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom bilinear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // ClampToEdge (the default) in u, v, w, as the CPU twin reads.
            ..Default::default()
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the bloom pipelines: {err}");
        }
        Ok(Self {
            down_first,
            down,
            up,
            composite,
            down_layout,
            up_layout,
            composite_layout,
            sampler,
            strength,
            weight: 0.0,
            chain: None,
        })
    }

    /// The pyramid for `targets`: two textures of the level-1 size (the downsamples, L mips; the
    /// upsamples, L − 1 mips) and a bind group per step.
    fn build_chain(&self, device: &wgpu::Device, targets: &FrameTargets) -> Chain {
        let sizes = bloom_level_sizes(targets.size);
        let levels = sizes.len();
        let texture = |label: &str, mips: usize| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: sizes[0].x,
                    height: sizes[0].y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mips as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: PYRAMID_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let mips = |t: &wgpu::Texture, count: usize| -> Vec<wgpu::TextureView> {
            (0..count as u32)
                .map(|m| {
                    t.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("bloom level"),
                        base_mip_level: m,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect()
        };
        let down_texture = texture("bloom down", levels);
        let down_views = mips(&down_texture, levels);
        let up_texture = (levels > 1).then(|| texture("bloom up", levels - 1));
        let up_views = up_texture
            .as_ref()
            .map_or_else(Vec::new, |t| mips(t, levels - 1));
        let bind_group = |layout: &wgpu::BindGroupLayout, views: &[(u32, &wgpu::TextureView)]| {
            let mut entries: Vec<wgpu::BindGroupEntry> = views
                .iter()
                .map(|&(binding, view)| wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom bg"),
                layout,
                entries: &entries,
            })
        };
        // d_k from d_{k−1} (level 1 from hdr).
        let down = (0..levels)
            .map(|k| {
                let src = if k == 0 {
                    &targets.hdr_view
                } else {
                    &down_views[k - 1]
                };
                (
                    bind_group(&self.down_layout, &[(0, src), (1, &down_views[k])]),
                    sizes[k],
                )
            })
            .collect();
        // u_k = d_k + tent(u_{k+1}), from u_{L−1} (whose lower level is d_L) down to u₁.
        let up = (0..levels.saturating_sub(1))
            .rev()
            .map(|k| {
                let low = if k + 2 == levels {
                    &down_views[k + 1]
                } else {
                    &up_views[k + 1]
                };
                let views = [(0, low), (1, &up_views[k]), (3, &down_views[k])];
                (bind_group(&self.up_layout, &views), sizes[k])
            })
            .collect();
        let top = up_views.first().unwrap_or(&down_views[0]);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom params"),
            contents: bytemuck::bytes_of(&BloomParamsGpu {
                scale: [1.0 / levels as f32, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom composite bg"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(top),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        Chain {
            hdr_view: targets.hdr_view.clone(),
            down,
            up,
            composite,
            _resources: (down_texture, up_texture, params),
        }
    }

    /// Rebuilds the pyramid if it was made for another `hdr` view (or not yet).
    fn ensure_chain(&mut self, device: &wgpu::Device, targets: &FrameTargets) {
        if self
            .chain
            .as_ref()
            .is_none_or(|c| c.hdr_view != targets.hdr_view)
        {
            self.chain = Some(self.build_chain(device, targets));
        }
    }
}

/// Group 1 of the compute passes: 0 the source (sampled, or read texel by texel on the first
/// level), 1 the destination level (storage), 2 the bilinear sampler, and for the upsamples 3 the
/// same-level downsample.
fn compute_layout_entries(up: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let mut entries = vec![
        texture(0),
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: PYRAMID_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ];
    if up {
        entries.push(texture(3));
    }
    entries
}

/// Group 1 of the composite: 0 u₁, 2 the sampler, 4 [`BloomParamsGpu`].
fn composite_layout_entries() -> [wgpu::BindGroupLayoutEntry; 3] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<BloomParamsGpu>() as u64
                ),
            },
            count: None,
        },
    ]
}

impl RenderFeature for Bloom {
    fn name(&self) -> &'static str {
        NAME
    }

    /// Reads the strength (clamped to [0, 1]; NaN turns bloom off) and builds the pyramid for
    /// the current targets if needed.
    fn prepare(&mut self, gpu: &GpuContext, _ctx: &FrameCtx, targets: &FrameTargets) {
        let s = self.strength.get();
        self.weight = if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
        if self.weight > 0.0 {
            self.ensure_chain(&gpu.device, targets);
        }
    }

    /// The pyramid in one compute pass (dispatches run in order), then the composite. Nothing
    /// at strength 0.
    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if self.weight <= 0.0 {
            return;
        }
        self.ensure_chain(&gpu.device, targets);
        let Some(chain) = self.chain.as_ref() else {
            return;
        };
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(NAME),
                timestamp_writes: timer.compute_writes(NAME),
            });
            pass.set_bind_group(CAMERA_GROUP, frame.camera, &[]);
            let groups = |size: UVec2| {
                (
                    size.x.div_ceil(WORKGROUP_EDGE),
                    size.y.div_ceil(WORKGROUP_EDGE),
                )
            };
            for (k, (bind_group, size)) in chain.down.iter().enumerate() {
                pass.set_pipeline(if k == 0 { &self.down_first } else { &self.down });
                pass.set_bind_group(BLOOM_GROUP, bind_group, &[]);
                let (x, y) = groups(*size);
                pass.dispatch_workgroups(x, y, 1);
            }
            pass.set_pipeline(&self.up);
            for (bind_group, size) in &chain.up {
                pass.set_bind_group(BLOOM_GROUP, bind_group, &[]);
                let (x, y) = groups(*size);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(COMPOSITE_NAME),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.hdr_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timer.render_writes(COMPOSITE_NAME),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(CAMERA_GROUP, frame.camera, &[]);
        pass.set_bind_group(BLOOM_GROUP, &chain.composite, &[]);
        let w = f64::from(self.weight);
        pass.set_blend_constant(wgpu::Color {
            r: w,
            g: w,
            b: w,
            a: w,
        });
        pass.draw(0..3, 0..1);
    }

    /// Drops the pyramid of the old size; the next frame builds the new one.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {
        self.chain = None;
    }
}

/// The first downsample (full resolution → level 1), with the input clamp and Karis average
/// switchable for the tests.
#[cfg(test)]
fn downsample_first(input: &[Vec3], size: UVec2, clamp: bool, karis: bool) -> Vec<Vec3> {
    let src = Image::new(size, input.to_vec());
    let level = bloom_level_sizes(size)[0];
    Image::from_fn(level, |i, j| first_level_texel(&src, i, j, clamp, karis)).px
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A black `size` frame with `value` at each of `points`.
    fn impulses(size: UVec2, points: &[(u32, u32, Vec3)]) -> Vec<Vec3> {
        let mut img = vec![Vec3::ZERO; (size.x * size.y) as usize];
        for &(x, y, v) in points {
            img[(y * size.x + x) as usize] = v;
        }
        img
    }

    fn sum(img: &[Vec3]) -> Vec3 {
        img.iter().copied().sum()
    }

    #[test]
    fn bloom_level_sizes_halve() {
        assert_eq!(
            bloom_level_sizes(UVec2::new(1920, 1080)),
            [
                (960, 540),
                (480, 270),
                (240, 135),
                (120, 67),
                (60, 33),
                (30, 16)
            ]
            .map(|(x, y)| UVec2::new(x, y))
        );
        assert_eq!(
            bloom_level_sizes(UVec2::new(1728, 1080)).len(),
            BLOOM_LEVELS
        );
        // Too small to halve six times: as many levels as the level-1 size has mips.
        assert_eq!(
            bloom_level_sizes(UVec2::new(16, 4)),
            [(8, 2), (4, 1), (2, 1), (1, 1)].map(|(x, y)| UVec2::new(x, y))
        );
        assert_eq!(bloom_level_sizes(UVec2::new(1, 1)), [UVec2::ONE]);
    }

    #[test]
    fn bloom_impulse_energy_conserved() {
        // Review m4. Karis average disabled: a delta through the 6-level down/up chain keeps its
        // energy (each 13-tap level and each tent upsample is normalised, the levels are
        // averaged), so hdr = mix(hdr, bloom, s) moves light but neither adds nor removes it.
        let size = UVec2::new(512, 512);
        for (x, y) in [(256, 256), (301, 207)] {
            let v = Vec3::new(1.0, 0.5, 0.25);
            let input = impulses(size, &[(x, y, v)]);
            let out = bloom_reference(&input, size, false);
            let (si, so) = (sum(&input), sum(&out));
            assert!(
                ((so - si) / si).abs().max_element() <= 0.01,
                "({x}, {y}): Σ in {si}, Σ out {so}"
            );
            // The glow is centred on the impulse and falls off away from it.
            let at = |dx: i32, dy: i32| {
                out[((y as i32 + dy) as u32 * size.x + (x as i32 + dx) as u32) as usize].x
            };
            assert!(
                at(0, 0) > at(8, 0)
                    && at(8, 0) > at(64, 0)
                    && at(64, 0) > at(200, 0)
                    && at(200, 0) > 0.0
            );
        }
    }

    #[test]
    fn karis_average_suppresses_firefly() {
        // One 60 000 texel among 1.0 texels. The first level (input clamp + Karis average) stays
        // below 2× the plain 13-tap average of the raw input / 10.
        let size = UVec2::new(32, 32);
        let mut input = vec![Vec3::ONE; 32 * 32];
        input[16 * 32 + 16] = Vec3::splat(60_000.0);
        let first = downsample_first(&input, size, true, true);
        let plain = downsample_first(&input, size, false, false);
        let max = |img: &[Vec3]| img.iter().map(|v| v.x).fold(0.0f32, f32::max);
        assert!(
            max(&first) < 2.0 * max(&plain) / 10.0,
            "{} vs {}",
            max(&first),
            max(&plain)
        );
        // The Karis average itself (Jimenez 2014's partial form, per block of four taps) cuts
        // the light the firefly feeds the chain: by a third after the clamp, by three quarters
        // without it. (The texel whose five blocks all contain the firefly keeps the plain
        // average, so the clamp is what bounds it.)
        let excess = |img: &[Vec3]| img.iter().map(|v| v.x - 1.0).sum::<f32>();
        let clamped = downsample_first(&input, size, true, false);
        assert!(
            excess(&first) < 0.7 * excess(&clamped),
            "{} vs {}",
            excess(&first),
            excess(&clamped)
        );
        let karis_only = downsample_first(&input, size, false, true);
        assert!(
            excess(&karis_only) < 0.3 * excess(&plain),
            "{} vs {}",
            excess(&karis_only),
            excess(&plain)
        );
        // A uniform field is unchanged by either.
        let flat = vec![Vec3::splat(3.0); 32 * 32];
        for v in downsample_first(&flat, size, true, true) {
            assert!((v - Vec3::splat(3.0)).abs().max_element() < 1e-5);
        }
    }

    #[test]
    fn bloom_input_clamped() {
        // A 60 000 texel enters the chain as BLOOM_INPUT_MAX.
        assert_eq!(
            bloom_input(Vec3::splat(60_000.0)),
            Vec3::splat(BLOOM_INPUT_MAX)
        );
        // Below the clamp, unchanged (Sirius at 35 pre-exposed, plan B6); negatives to 0.
        assert_eq!(bloom_input(Vec3::splat(35.0)), Vec3::splat(35.0));
        assert_eq!(
            bloom_input(Vec3::new(-1.0, 2.0, 3.0)),
            Vec3::new(0.0, 2.0, 3.0)
        );
        // Scaled, so a red star keeps its colour.
        let red = bloom_input(Vec3::new(300.0, 120.0, 40.0));
        assert!(
            (red - Vec3::new(64.0, 25.6, 64.0 * 40.0 / 300.0))
                .abs()
                .max_element()
                < 1e-4
        );
        // Through the first level: a 60 000 texel on black feeds exactly what a 64 one does.
        let size = UVec2::new(16, 16);
        let hot = impulses(size, &[(8, 8, Vec3::splat(60_000.0))]);
        let cap = impulses(size, &[(8, 8, Vec3::splat(BLOOM_INPUT_MAX))]);
        assert_eq!(
            downsample_first(&hot, size, true, true),
            downsample_first(&cap, size, true, true)
        );
        assert_eq!(
            bloom_reference(&hot, size, true),
            bloom_reference(&cap, size, true)
        );
    }
}
