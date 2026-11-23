//! HDR taps: a render feature that copies the pre-exposed `hdr` target into a read-back buffer
//! at its place in the pass order, on the frames it is armed for (captures only).
//!
//! Two taps sit in the capture stack (M1 Task B9 Step 2):
//! - **pre-bloom** ([`TAP_PREBLOOM`]), between the last sky pass and bloom (pass 15): the
//!   `--hdr` sidecar, so the photometric checks read unbloomed energy (re-review r3 minor 1:
//!   bloom otherwise moves 3–4 % of a point's energy outside a star's footprint);
//! - **world** ([`TAP_WORLD`]), right after the exposure meter (pass 9), before any sky light or
//!   point source is written: the world light the meter read, for the capture JSON's `lit_px`.
//!
//! Disarmed, a tap encodes nothing, so the window and `--bench` pay nothing for it.
//!
//! The module also holds two small timing passes (B9t): [`ClearMark`], the stack's first pass,
//! which ends when the renderer's untimed target clear ends, and [`FrameTag`], which lets the
//! sustained bench attach each timing record to its frame. These passes, like the taps, bind
//! their own resources at group 0 and never the renderer's camera group (the exposure
//! resolve's exemption from the frame contract's "features use groups ≥ 1", review minor 9).
//!
//! **Timing (review B9 r0 Important 1, lead fix).** An armed tap's copy is its own timed pass,
//! named [`TAP_WORLD`] / [`TAP_PREBLOOM`] in the timer and in the capture JSON's `gpu_ms`, so the
//! copy is not charged to the next pass (a `copy_texture_to_buffer` is not a pass and cannot
//! carry timestamps: `TIMESTAMP_QUERY_INSIDE_ENCODERS` is unavailable on the M3). The copy is a
//! render pass, like the passes around it, because `scene::pass_durations` turns the timer's
//! per-pass values into durations by differencing them in the render passes' sense (lead
//! ruling 7): one fullscreen triangle whose fragments each read one `hdr` texel and write it as
//! four f32 (exact: the texture holds f16) into a storage buffer. (`pack2x16float` would halve
//! the bytes but needs naga's `SHADER_FLOAT16_IN_FLOAT32` capability.) Its colour attachment is a throw-away R8 target (cleared, never stored), since
//! a render pass needs one. Metal orders the pass after the passes that wrote `hdr` (it reads the
//! texture) and before the next pass that writes it. The storage buffer is copied into a
//! mappable buffer after the frame, in [`HdrTap::take`]'s own command buffer, outside every
//! timed span. `gpu_frame_ms` of the captured frame still includes the taps' passes; `gpu_ms`
//! lists them, so a reader can subtract them.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Context as _;
use glam::UVec2;
use uni_gfx::{FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature};

/// Timer label and pass name of the tap after the meter (the world light, `lit_px`).
pub const TAP_WORLD: &str = "tap-world";
/// Timer label and pass name of the tap before bloom (the `--hdr` sidecar).
pub const TAP_PREBLOOM: &str = "tap-prebloom";

/// Bytes per texel of the tap's copy: RGBA f32.
const TAP_BYTES_PER_TEXEL: u32 = 16;
/// Format of the throw-away colour attachment the copy pass renders into.
const DUMMY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The copy pass's shader: a fullscreen triangle; each fragment copies its `hdr` texel into
/// `texels` (row-major, rows top-down) as RGBA f32.
pub const TAP_WGSL: &str = r#"// tap.wgsl: copies the pre-exposed hdr target into a storage buffer (uni-app HDR taps).
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> texels: array<vec4<f32>>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = textureDimensions(hdr);
    let p = vec2<u32>(pos.xy);
    if (p.x < size.x && p.y < size.y) {
        texels[p.y * size.x + p.x] = textureLoad(hdr, p, 0);
    }
    return vec4<f32>(0.0);
}
"#;

/// A copy of `hdr` taken during a frame: linear pre-exposed RGBA, rows top-down.
#[derive(Clone, Debug, PartialEq)]
pub struct HdrImage {
    pub size: UVec2,
    /// `size.x × size.y` texels, RGBA f32 (converted exactly from the target's f16).
    pub pixels: Vec<[f32; 4]>,
}

impl HdrImage {
    /// Texel (x, y), or `None` outside the image.
    pub fn at(&self, x: i64, y: i64) -> Option<[f32; 4]> {
        if x < 0 || y < 0 || x >= i64::from(self.size.x) || y >= i64::from(self.size.y) {
            return None;
        }
        self.pixels
            .get(y as usize * self.size.x as usize + x as usize)
            .copied()
    }

    /// Little-endian f32 RGBA bytes, rows top-down: the `.hdr.bin` sidecar layout.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 16);
        for p in &self.pixels {
            for c in p {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out
    }

    /// Decodes the tap's copy: `size.x × size.y` texels of four little-endian f32 each (the
    /// sidecar's own layout, [`Self::to_le_bytes`]).
    fn from_le_bytes(size: UVec2, bytes: &[u8]) -> Self {
        let pixels = bytes
            .chunks_exact(TAP_BYTES_PER_TEXEL as usize)
            .take((size.x * size.y) as usize)
            .map(|t| {
                std::array::from_fn(|c| {
                    f32::from_le_bytes([t[4 * c], t[4 * c + 1], t[4 * c + 2], t[4 * c + 3]])
                })
            })
            .collect();
        Self { size, pixels }
    }
}

/// What the feature and its owner share.
#[derive(Default)]
struct TapState {
    armed: bool,
    /// The storage buffer of the last copy and its size.
    pending: Option<(wgpu::Buffer, UVec2)>,
}

/// The app's handle on a tap in the renderer.
#[derive(Clone)]
pub struct HdrTap {
    state: Rc<RefCell<TapState>>,
}

impl HdrTap {
    /// A tap named `name` ([`TAP_WORLD`] or [`TAP_PREBLOOM`]) and the feature to push into the
    /// renderer at its place in the pass order.
    pub fn new(gpu: &GpuContext, name: &'static str) -> anyhow::Result<(Self, TapFeature)> {
        let state = Rc::new(RefCell::new(TapState::default()));
        let feature = TapFeature::new(gpu, name, Rc::clone(&state))?;
        Ok((Self { state }, feature))
    }

    /// Copy `hdr` on the next frame(s) rendered until disarmed.
    pub fn arm(&self, on: bool) {
        self.state.borrow_mut().armed = on;
    }

    /// Reads the last copy back (blocking: waits for the GPU) and clears it; `None` if no copy
    /// was taken since the last call. The storage → mappable copy is encoded and submitted
    /// here, after the frame, so no timed pass pays for it.
    pub fn take(&self, gpu: &GpuContext) -> anyhow::Result<Option<HdrImage>> {
        let Some((storage, size)) = self.state.borrow_mut().pending.take() else {
            return Ok(None);
        };
        let bytes = storage.size();
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdr tap read-back"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hdr tap read-back"),
            });
        enc.copy_buffer_to_buffer(&storage, 0, &readback, 0, bytes);
        gpu.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            // The receiver outlives the wait below; a send error cannot be acted on here.
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("waiting for the HDR tap")?;
        rx.try_recv()
            .context("HDR tap map callback did not run")?
            .context("mapping the HDR tap buffer")?;
        let image = {
            let view = readback
                .get_mapped_range(..)
                .context("reading the mapped HDR tap buffer")?;
            HdrImage::from_le_bytes(size, &view)
        };
        readback.unmap();
        Ok(Some(image))
    }
}

/// The render feature of an [`HdrTap`]: the timed copy pass (module docs).
pub struct TapFeature {
    name: &'static str,
    state: Rc<RefCell<TapState>>,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The throw-away colour attachment, at the targets' size; made on the first armed frame.
    dummy: Option<(UVec2, wgpu::TextureView)>,
}

/// Group 0 of the copy pass: `hdr` (binding 0) and the packed texels (binding 1).
fn tap_layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
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
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
}

impl TapFeature {
    fn new(
        gpu: &GpuContext,
        name: &'static str,
        state: Rc<RefCell<TapState>>,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(TAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdr tap bgl"),
            entries: &tap_layout_entries(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdr tap layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
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
                    format: DUMMY_FORMAT,
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
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the {name} pass: {err}");
        }
        Ok(Self {
            name,
            state,
            pipeline,
            layout,
            dummy: None,
        })
    }

    /// The throw-away attachment at `size`, (re)made when the size changes.
    fn dummy_view(&mut self, device: &wgpu::Device, size: UVec2) -> wgpu::TextureView {
        if let Some((made, view)) = &self.dummy
            && *made == size
        {
            return view.clone();
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr tap attachment"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DUMMY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.dummy = Some((size, view.clone()));
        view
    }
}

/// Bytes of a tap's copy of a `size` frame.
fn copy_bytes(size: UVec2) -> u64 {
    u64::from(size.x) * u64::from(size.y) * u64::from(TAP_BYTES_PER_TEXEL)
}

impl RenderFeature for TapFeature {
    fn name(&self) -> &'static str {
        self.name
    }

    fn prepare(&mut self, _gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {}

    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        _frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if !self.state.borrow().armed {
            return;
        }
        let size = targets.size;
        let storage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.name),
            size: copy_bytes(size),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdr tap bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage.as_entire_binding(),
                },
            ],
        });
        let dummy = self.dummy_view(&gpu.device, size);
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(self.name),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dummy,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timer.render_writes(self.name),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.state.borrow_mut().pending = Some((storage, size));
    }
}

/// Number of distinct frame tags ([`FRAME_TAGS`]).
pub const FRAME_TAG_COUNT: usize = 64;

/// Timer labels of the frame tags, one per frame index modulo [`FRAME_TAG_COUNT`].
pub const FRAME_TAGS: [&str; FRAME_TAG_COUNT] = [
    "frame-tag-00",
    "frame-tag-01",
    "frame-tag-02",
    "frame-tag-03",
    "frame-tag-04",
    "frame-tag-05",
    "frame-tag-06",
    "frame-tag-07",
    "frame-tag-08",
    "frame-tag-09",
    "frame-tag-10",
    "frame-tag-11",
    "frame-tag-12",
    "frame-tag-13",
    "frame-tag-14",
    "frame-tag-15",
    "frame-tag-16",
    "frame-tag-17",
    "frame-tag-18",
    "frame-tag-19",
    "frame-tag-20",
    "frame-tag-21",
    "frame-tag-22",
    "frame-tag-23",
    "frame-tag-24",
    "frame-tag-25",
    "frame-tag-26",
    "frame-tag-27",
    "frame-tag-28",
    "frame-tag-29",
    "frame-tag-30",
    "frame-tag-31",
    "frame-tag-32",
    "frame-tag-33",
    "frame-tag-34",
    "frame-tag-35",
    "frame-tag-36",
    "frame-tag-37",
    "frame-tag-38",
    "frame-tag-39",
    "frame-tag-40",
    "frame-tag-41",
    "frame-tag-42",
    "frame-tag-43",
    "frame-tag-44",
    "frame-tag-45",
    "frame-tag-46",
    "frame-tag-47",
    "frame-tag-48",
    "frame-tag-49",
    "frame-tag-50",
    "frame-tag-51",
    "frame-tag-52",
    "frame-tag-53",
    "frame-tag-54",
    "frame-tag-55",
    "frame-tag-56",
    "frame-tag-57",
    "frame-tag-58",
    "frame-tag-59",
    "frame-tag-60",
    "frame-tag-61",
    "frame-tag-62",
    "frame-tag-63",
];

/// The tag number of timer label `label`, if it is one of [`FRAME_TAGS`].
pub fn frame_tag_of(label: &str) -> Option<usize> {
    FRAME_TAGS.iter().position(|t| *t == label)
}

/// The bench's handle on the frame tag: which tag the next frame carries (`None`: no tag).
#[derive(Clone, Default)]
pub struct FrameTag(Rc<std::cell::Cell<Option<usize>>>);

impl FrameTag {
    /// A handle and the feature to push into the renderer (first, before every pass).
    pub fn new() -> (Self, FrameTagFeature) {
        let tag = Self::default();
        let feature = FrameTagFeature(tag.clone());
        (tag, feature)
    }

    /// Tag the next frames rendered with `tag % FRAME_TAG_COUNT`, or stop tagging (`None`).
    pub fn set(&self, tag: Option<usize>) {
        self.0.set(tag.map(|t| t % FRAME_TAG_COUNT));
    }
}

/// The render feature of a [`FrameTag`] (B9t, sustained bench): while a tag is set, one empty
/// compute pass whose timer label is that tag. The timer reports labels with every frame's
/// timings, so a record that arrives frames later names the frame it came from (modulo
/// [`FRAME_TAG_COUNT`]), whichever records the timer's ring dropped. The pass does no work and
/// touches no resource; its own times are dropped before durations are taken. Untagged, it
/// encodes nothing.
pub struct FrameTagFeature(FrameTag);

impl RenderFeature for FrameTagFeature {
    fn name(&self) -> &'static str {
        "frame tag"
    }

    fn prepare(&mut self, _gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {}

    fn encode(
        &mut self,
        _gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        _frame: &FrameBindings<'_>,
        _targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        let Some(tag) = (self.0).0.get() else {
            return;
        };
        let label = FRAME_TAGS[tag % FRAME_TAG_COUNT];
        let _pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: timer.compute_writes(label),
        });
    }
}

/// Timer label of the clear mark ([`ClearMark`]).
pub const CLEAR_MARK: &str = "clear targets";

/// The clear mark's shader: one thread writes 0 to texel (0, 0) of `hdr`, which the renderer's
/// clear has just set to 0.
pub const CLEAR_MARK_WGSL: &str = r#"// clear_mark.wgsl: a one-thread pass ordered after the target clear (uni-app, B9t).
@group(0) @binding(0) var hdr: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(1, 1, 1)
fn cs_mark() {
    textureStore(hdr, vec2<u32>(0u, 0u), vec4<f32>(0.0));
}
"#;

/// The first pass of the stack (B9t): a timed one-thread compute pass named [`CLEAR_MARK`]
/// that writes texel (0, 0) of `hdr`, as uni-gfx's frame brackets do. The renderer clears the
/// targets in an untimed render pass right before the features encode; the mark writes the
/// texture the clear wrote, so Metal orders it after the clear, and its end marks the clear's
/// end. With durations from end times (`scene::pass_durations`) the clear's ≈ 0.4 ms is then
/// its own, not the next timed pass's ("sun", B9p). The texel it writes is 0 already.
pub struct ClearMark {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// The bind group over the `hdr` view it was made for.
    binding: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

impl ClearMark {
    pub fn new(gpu: &GpuContext) -> anyhow::Result<Self> {
        let device = &gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(CLEAR_MARK),
            source: wgpu::ShaderSource::Wgsl(CLEAR_MARK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clear mark bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: uni_gfx::HDR_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clear mark layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(CLEAR_MARK),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_mark"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the {CLEAR_MARK} mark: {err}");
        }
        Ok(Self {
            pipeline,
            layout,
            binding: None,
        })
    }
}

impl RenderFeature for ClearMark {
    fn name(&self) -> &'static str {
        CLEAR_MARK
    }

    fn prepare(&mut self, _gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {}

    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        _frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if self
            .binding
            .as_ref()
            .is_none_or(|(view, _)| *view != targets.hdr_view)
        {
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clear mark bg"),
                layout: &self.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
                }],
            });
            self.binding = Some((targets.hdr_view.clone(), bind_group));
        }
        let Some((_, bind_group)) = self.binding.as_ref() else {
            return;
        };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(CLEAR_MARK),
            timestamp_writes: timer.compute_writes(CLEAR_MARK),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {
        self.binding = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_copy_decodes_and_round_trips_the_sidecar_bytes() {
        assert_eq!(copy_bytes(UVec2::new(1920, 1080)), 33_177_600);
        let img = HdrImage {
            size: UVec2::new(2, 1),
            pixels: vec![[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -1.0, 65504.0]],
        };
        let b = img.to_le_bytes();
        assert_eq!(b.len(), 32);
        assert_eq!(f32::from_le_bytes(b[4..8].try_into().unwrap()), 2.0);
        assert_eq!(f32::from_le_bytes(b[28..32].try_into().unwrap()), 65504.0);
        assert_eq!(HdrImage::from_le_bytes(img.size, &b), img);
        assert_eq!(img.at(1, 0), Some([0.5, 0.0, -1.0, 65504.0]));
        assert_eq!(img.at(2, 0), None);
        assert_eq!(img.at(0, -1), None);
    }

    /// CLAUDE.md §3: the clear mark's shader validates, writes `hdr` (rgba16float) at group 0,
    /// binding 0, one thread.
    #[test]
    fn clear_mark_shader_validates() {
        let module = uni_gfx::validate_wgsl(CLEAR_MARK_WGSL).expect("clear_mark.wgsl validates");
        let ep = module
            .entry_points
            .iter()
            .find(|e| e.name == "cs_mark")
            .expect("cs_mark");
        assert_eq!(ep.workgroup_size, [1, 1, 1]);
        let bindings: Vec<(u32, u32)> = module
            .global_variables
            .iter()
            .filter_map(|(_, g)| g.binding.as_ref().map(|b| (b.group, b.binding)))
            .collect();
        assert_eq!(bindings, [(0, 0)]);
        assert_eq!(uni_gfx::HDR_FORMAT, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn frame_tags_are_distinct_and_found() {
        let set: std::collections::BTreeSet<&str> = FRAME_TAGS.iter().copied().collect();
        assert_eq!(set.len(), FRAME_TAG_COUNT);
        for (k, t) in FRAME_TAGS.iter().enumerate() {
            assert_eq!(frame_tag_of(t), Some(k));
        }
        assert_eq!(frame_tag_of("bloom"), None);
        let (tag, _feature) = FrameTag::new();
        tag.set(Some(FRAME_TAG_COUNT + 3));
        assert_eq!(tag.0.get(), Some(3));
        tag.set(None);
        assert_eq!(tag.0.get(), None);
    }

    /// CLAUDE.md §3: the copy shader validates with naga, and its group 0 is the layout the
    /// feature binds (a texture at 0, a read-write storage buffer of 16-byte texels at 1).
    #[test]
    fn tap_shader_validates_and_matches_its_layout() {
        let module = uni_gfx::validate_wgsl(TAP_WGSL).expect("tap.wgsl validates");
        let mut bindings: Vec<(u32, String)> = module
            .global_variables
            .iter()
            .filter_map(|(_, g)| {
                let b = g.binding.as_ref()?;
                assert_eq!(b.group, 0);
                Some((b.binding, g.name.clone().unwrap_or_default()))
            })
            .collect();
        bindings.sort();
        assert_eq!(bindings, [(0, "hdr".to_owned()), (1, "texels".to_owned())]);
        let entries = tap_layout_entries();
        assert!(matches!(entries[0].ty, wgpu::BindingType::Texture { .. }));
        assert!(matches!(
            entries[1].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                ..
            }
        ));
        let texel = module
            .types
            .iter()
            .find_map(|(_, t)| match t.inner {
                wgpu::naga::TypeInner::Array { stride, .. } => Some(stride),
                _ => None,
            })
            .expect("the texel array");
        assert_eq!(texel, TAP_BYTES_PER_TEXEL);
        assert_ne!(TAP_WORLD, TAP_PREBLOOM);
    }
}
