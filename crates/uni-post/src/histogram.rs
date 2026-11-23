//! The exposure meter, pass "histogram" (Task B6).
//!
//! Pass 9 of the M1 frame contract: after the atmospheres, before any sky light or point source
//! is written, so it meters world light only. One compute pass with two dispatches:
//! 1. `cs_histogram`: each pixel's world luminance, Rec. 709 Y of `hdr` ÷ the pre-exposure the
//!    pixel was written with [cd/m²] ([`meter_pixel`]: `scene_pre()` for plain world light; a
//!    pixel clamped at `HDR_MAX` counts as at least `HDR_MAX / pre`), skipped below
//!    [`METER_MIN_CD_M2`] (not lit), binned into
//!    [`METER_BINS`] log₂ bins over [[`METER_LOG2_MIN`], [`METER_LOG2_MAX`]] (values outside go
//!    to the end bins) with workgroup-shared atomics flushed once per workgroup (bruop 2019,
//!    "Automatic Exposure Using a Luminance Histogram", <https://bruop.github.io/exposure/>).
//! 2. `cs_reduce`: one workgroup reads and clears the bins and reduces them to a [`Meter`] in
//!    the exposure state, which the next frame's resolve pass ([`crate::exposure`]) turns into
//!    its target.
//!
//! Sub-bin position (B6 decision): a bin is 0.164 EV wide, which alone would put the
//! highlight rule's 99.5th-percentile pixel up to ±6 % off its 4.0. Each bin therefore also sums
//! its pixels' position inside the bin in 1/[`METER_BIN_STEPS`] steps, and stands for its
//! pixels' mean log₂ L instead of its centre (the GPU packs count and step sum into one shared
//! atomic). [`Histogram`] is the exact CPU twin; [`meter_histogram`] (counts only, bin centres)
//! differs from it by at most half a bin.
//!
//! ## Star share in `hdr` alpha (fix round 1, review r0 Important 1; needs the lead's ruling)
//!
//! The plan's rule divides every pixel by `scene_pre`. A body whose projected radius is in the
//! 1.5–4 px LOD band is written at pre = mix(star_pre, scene_pre, w) (`uni_planet::
//! lod_exposure_mix`; body.wgsl, rings.wgsl, star_surface.wgsl), so ÷ `scene_pre` reads it
//! (1 − w)·2^(EV_scene − EV_star) + w times too bright. That reading grows with the scene EV, a
//! positive feedback: an Earth at 12 px with the Moon at 3.2 px ran the exposure to EV 40
//! (`earth_moon_band_pair_settles`). The meter therefore reads the pixel's **star share**
//! s = 1 − w from `hdr` alpha and divides by [`meter_pre`] = mix(scene_pre, star_pre, s):
//! - s = 0 is the clear value (`Renderer` clears `hdr` to 0) and what every pass leaves while
//!   its colour target masks alpha out (all of them at fix round 1). Such pixels read exactly
//!   as the plan's rule, so nothing changes until a writer stores s.
//! - The encoding is 1 − w, not w, for two reasons. Zero (the clear value) then means scene
//!   exposure. And f16's relative precision lands where the feedback is sensitive (w → 1):
//!   a stored s is off by ≤ 2⁻¹¹ relative, and so is the pre the meter recovers.
//! - Mixed pixels: if the writers composite s exactly like their premultiplied colour
//!   (s_out = coverage·s + (1 − coverage)·s_dst), the meter reads the pre-weighted mean of the
//!   luminances in the pixel, within their range at every EV, so no feedback
//!   (`mixed_band_pixel_reads_within_its_luminances`).
//! - Not built here: the writers (B1 body, B2 rings and star surfaces) must store s, and the
//!   frame contract (uni-gfx `targets.rs`) must state it. Until they do, a band body next to a
//!   metered one still runs the loop away. The fix round 1 report gives the exact change.

use anyhow::Context as _;
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_MAX, PassTimer, RenderFeature, ShaderLib,
};

use crate::exposure::{ExposureStateGpu, NIGHT_EV100, clamp_ev100};

/// A world-light pixel below this luminance [cd/m²] is not "lit" (plan B6).
pub const METER_MIN_CD_M2: f32 = 1.0e-3;
/// Below this many lit pixels (normalised to 1920 × 1080; a disk of radius 4.0 px, where
/// `uni_planet::lod_exposure_mix` has finished) the target is [`crate::NIGHT_EV100`] (plan B6).
pub const METER_MIN_LIT_PX: f32 = 50.0;
/// At and above this many lit pixels (0.019 % of the frame, a disk of radius 11.3 px) the target
/// is fully metered; in between it blends by smoothstep in ln(lit_px) (plan B6).
pub const METER_FULL_LIT_PX: f32 = 400.0;
/// Percentile of the lit pixels the highlight rule protects (plan B6).
pub const HIGHLIGHT_PERCENTILE: f32 = 0.995;
/// Pre-exposed value the [`HIGHLIGHT_PERCENTILE`] pixel lands at: 2.03 stops under AgX white
/// (2^4.026 = 16.29; uni-post `agx`), so a small bright subject next to a large dim one is not
/// clipped (plan B6).
pub const HIGHLIGHT_PRE: f32 = 4.0;

/// Number of log₂-luminance bins (plan B6).
pub const METER_BINS: usize = 256;
/// Lower edge of bin 0 [log₂ cd/m²] (plan B6).
pub const METER_LOG2_MIN: f32 = -10.0;
/// Upper edge of bin 255 [log₂ cd/m²] (plan B6).
pub const METER_LOG2_MAX: f32 = 32.0;
/// Width of one bin [EV]: 42/256 = 0.1640625, exact in binary.
pub const METER_BIN_EV: f32 = (METER_LOG2_MAX - METER_LOG2_MIN) / METER_BINS as f32;
/// Sub-bin steps recorded per pixel (B6 decision, see the module docs): a bin's mean is known to
/// 1/32 of a bin (0.005 EV). 16 also keeps a bin's step sum below 2³² for any texture the device
/// allows (16384² px × 15).
pub const METER_BIN_STEPS: u32 = 16;
/// The frame `Meter::lit_px` is normalised to: 1920 × 1080 px (plan B6).
pub const METER_REFERENCE_PX: f32 = 2_073_600.0;
/// The lit-pixel percentile band whose mean log₂ L is `L_avg` (plan B6: 50th–95th).
pub const AVERAGE_BAND: [f32; 2] = [0.50, 0.95];
/// log₂(S/K) with S = 100 (ISO 100) and K = 12.5, the reflected-light meter calibration constant
/// (Lagarde & de Rousiers 2014, "Moving Frostbite to PBR 3.0", §4.1.1, after ISO 2720:1974):
/// EV100 = log₂(L_avg·S/K).
pub const LOG2_S_OVER_K: f32 = 3.0;
/// L_max = 1.2·2^EV100 [cd/m²], the saturation-based exposure of `uni_gfx::exposure_scale`
/// (Lagarde & de Rousiers 2014 §4.1.1, after ISO 12232:2006): a pixel of luminance L is written
/// as L / (1.2·2^EV).
const SATURATION_FACTOR: f32 = 1.2;
/// Relative luminance weights of linear Rec. 709: ITU-R BT.709-6 (2015), item 3.2 (the same as
/// `luminance` in common/color.wgsl).
const REC709_Y: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// What the meter read from one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Meter {
    /// Lit pixels, normalised to 1920 × 1080.
    pub lit_px: f32,
    /// log₂(L_avg·S/K), L_avg over the 50th–95th percentile of lit pixels, S = 100, K = 12.5
    /// (Lagarde & de Rousiers 2014 §4.1.1, after ISO 2720).
    pub ev_avg: f32,
    /// log₂(L_p99.5 / (1.2 · [`HIGHLIGHT_PRE`])).
    pub ev_hi: f32,
}

/// Twin of `struct MeterBins` in `histogram.wgsl`: the global bins `cs_histogram` accumulates
/// and `cs_reduce` reads and clears (2 048 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeterBinsGpu {
    /// Lit pixels per bin.
    pub(crate) counts: [u32; METER_BINS],
    /// Per bin, the sum of its pixels' sub-bin steps.
    pub(crate) steps: [u32; METER_BINS],
}

/// The world luminance [cd/m²] the meter reads from a pre-exposed `hdr` pixel `pre_rgb` written
/// at pre-exposure `pre` ([`meter_pre`]; `scene_pre` for plain world light): Rec. 709 Y ÷ `pre`,
/// each channel clamped to [0, `HDR_MAX`] first. A pixel with any channel at `HDR_MAX` (clamped
/// by its writer) counts as at least `HDR_MAX / pre` (plan B6). That is a lower bound of its true
/// value only when all three channels clipped; a pixel clipped in one channel can read up to
/// log₂(1/0.0722) = 3.8 EV high (blue alone). The loop still converges: the next frame, exposed
/// for the high reading, no longer clips. Twin of `meter_luminance` in `histogram.wgsl`.
pub fn meter_luminance(pre_rgb: [f32; 3], pre: f32) -> f32 {
    let c = pre_rgb.map(|v| v.clamp(0.0, HDR_MAX));
    let l = (REC709_Y[0] * c[0] + REC709_Y[1] * c[1] + REC709_Y[2] * c[2]) / pre;
    if c[0].max(c[1]).max(c[2]) >= HDR_MAX {
        l.max(HDR_MAX / pre)
    } else {
        l
    }
}

/// The pre-exposure the light in an `hdr` pixel was written with, from the pixel's star share
/// `star_share` (its alpha at the meter, see the module docs): mix(`scene_pre`, `star_pre`, s) =
/// `scene_pre`·(1 − s) + `star_pre`·s (WGSL `mix`, same order), with s clamped to [0, 1]. s = 0
/// gives `scene_pre` exactly. A NaN share gives NaN (the pixel is not metered). Twin of
/// `meter_pre` in `histogram.wgsl`.
pub fn meter_pre(scene_pre: f32, star_pre: f32, star_share: f32) -> f32 {
    // f32::clamp keeps NaN, so a NaN share stays NaN here.
    let s = star_share.clamp(0.0, 1.0);
    scene_pre * (1.0 - s) + star_pre * s
}

/// What `cs_histogram` reads from one pre-exposed `hdr` texel `rgba`: its world luminance
/// [cd/m²], [`meter_luminance`] at [`meter_pre`]`(scene_pre, star_pre, rgba[3])`. A NaN alpha
/// returns NaN, which [`meter_bin`] treats as not lit (the GPU skips the pixel on its bits).
pub fn meter_pixel(rgba: [f32; 4], scene_pre: f32, star_pre: f32) -> f32 {
    let [r, g, b, s] = rgba;
    if s.is_nan() {
        return f32::NAN;
    }
    meter_luminance([r, g, b], meter_pre(scene_pre, star_pre, s))
}

/// The bin and sub-bin step (0..[`METER_BIN_STEPS`]) of world luminance `l_cd_m2`, or `None`
/// when the pixel is not lit (below [`METER_MIN_CD_M2`], or NaN). Values beyond the range go to
/// the end bins. Twin of `meter_bin` in `histogram.wgsl`.
pub fn meter_bin(l_cd_m2: f32) -> Option<(usize, u32)> {
    // `!(l >= min)` is also true for NaN.
    if !(l_cd_m2 >= METER_MIN_CD_M2) {
        return None;
    }
    // The bin coordinate, clamped just inside the top edge so 2^32 and beyond (and +∞) land in
    // bin 255's last step.
    let x = ((l_cd_m2.log2() - METER_LOG2_MIN) / METER_BIN_EV)
        .clamp(0.0, METER_BINS as f32 - BIN_COORD_MARGIN);
    let bin = (x as usize).min(METER_BINS - 1);
    let step = (((x - bin as f32) * METER_BIN_STEPS as f32) as u32).min(METER_BIN_STEPS - 1);
    Some((bin, step))
}

/// How far inside the top edge [`meter_bin`] clamps the bin coordinate (bins): in f32 at 256 it
/// is ~65 ulp, so the clamped coordinate is in bin 255 (`BIN_COORD_MARGIN` in histogram.wgsl).
pub(crate) const BIN_COORD_MARGIN: f32 = 1.0e-3;

/// The meter's histogram: per bin, the lit-pixel count and the sum of their sub-bin steps. The
/// CPU twin of what `cs_histogram` accumulates and `cs_reduce` reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram {
    /// Lit pixels per bin.
    pub counts: [u32; METER_BINS],
    /// Per bin, the sum of its pixels' sub-bin steps (each 0..[`METER_BIN_STEPS`]).
    pub steps: [u32; METER_BINS],
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            counts: [0; METER_BINS],
            steps: [0; METER_BINS],
        }
    }
}

impl Histogram {
    /// Bins one pixel of world luminance `l_cd_m2` (skipped when not lit).
    pub fn add(&mut self, l_cd_m2: f32) {
        if let Some((bin, step)) = meter_bin(l_cd_m2) {
            self.counts[bin] += 1;
            self.steps[bin] += step;
        }
    }

    /// The histogram of world luminances [cd/m²].
    pub fn from_luminances(lums: impl IntoIterator<Item = f32>) -> Self {
        let mut h = Self::default();
        for l in lums {
            h.add(l);
        }
        h
    }

    /// Reduces the histogram of a frame of `total_px` pixels to a [`Meter`], each bin standing
    /// for its pixels' mean log₂ L; `None` when nothing is lit. Exact CPU twin of `cs_reduce`.
    pub fn meter(&self, total_px: u32) -> Option<Meter> {
        reduce(
            &self.counts,
            |bin| {
                // The bin's pixels' mean position: their step sum / count, +½ step for the
                // truncation in `meter_bin`.
                let steps = self.steps[bin] as f32 / self.counts[bin] as f32;
                METER_LOG2_MIN
                    + (bin as f32 + (steps + 0.5) / METER_BIN_STEPS as f32) * METER_BIN_EV
            },
            total_px,
        )
    }
}

/// Reduces the 256-bin histogram (lit-pixel counts) of `total_px` pixels to a [`Meter`]; `None`
/// when nothing is lit. Counts only, so each bin stands for its centre: within half a bin
/// (0.082 EV) of [`Histogram::meter`], which also uses the sub-bin steps and is the exact twin of
/// `histogram.wgsl`'s reduction.
pub fn meter_histogram(bins: &[u32; 256], total_px: u32) -> Option<Meter> {
    reduce(
        bins,
        |bin| METER_LOG2_MIN + (bin as f32 + 0.5) * METER_BIN_EV,
        total_px,
    )
}

/// The reduction of `cs_reduce` (`meter_reduce` in histogram.wgsl, same order of operations, in
/// f32): `log2_l(bin)` is the log₂ luminance a non-empty bin stands for.
/// - lit_px = lit count × 1920·1080 / `total_px`;
/// - ev_avg = the mean log₂ L of the lit pixels ranked in the [`AVERAGE_BAND`] percentiles (a bin
///   straddling a band edge counts with the part inside), + [`LOG2_S_OVER_K`];
/// - ev_hi = log₂ L of the bin holding the [`HIGHLIGHT_PERCENTILE`] rank − log₂(1.2 ·
///   [`HIGHLIGHT_PRE`]), so that pixel lands at [`HIGHLIGHT_PRE`] at EV = ev_hi.
fn reduce(
    counts: &[u32; METER_BINS],
    log2_l: impl Fn(usize) -> f32,
    total_px: u32,
) -> Option<Meter> {
    let lit: u32 = counts.iter().sum();
    if lit == 0 {
        return None;
    }
    let n = lit as f32;
    let (lo, hi) = (AVERAGE_BAND[0] * n, AVERAGE_BAND[1] * n);
    let rank = HIGHLIGHT_PERCENTILE * n;
    let (mut below, mut sum, mut p_hi, mut found) = (0.0f32, 0.0f32, METER_LOG2_MIN, false);
    for (bin, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let l = log2_l(bin);
        let above = below + count as f32;
        sum += (above.min(hi) - below.max(lo)).max(0.0) * l;
        if !found && above >= rank {
            p_hi = l;
            found = true;
        }
        below = above;
    }
    Some(Meter {
        lit_px: n * (METER_REFERENCE_PX / total_px.max(1) as f32),
        ev_avg: sum / (hi - lo) + LOG2_S_OVER_K,
        ev_hi: p_hi - (SATURATION_FACTOR * HIGHLIGHT_PRE).log2(),
    })
}

/// B6's target rule: [`crate::NIGHT_EV100`] below [`METER_MIN_LIT_PX`], max(ev_avg, ev_hi) +
/// `compensation_ev` from [`METER_FULL_LIT_PX`], smoothstep in ln(lit_px) between; clamped to
/// [[`crate::EV100_MIN`], [`crate::EV100_MAX`]].
pub fn target_ev(m: Option<Meter>, compensation_ev: f32) -> f32 {
    let Some(m) = m else {
        return NIGHT_EV100;
    };
    // `!(x >= min)` is also true for NaN.
    if !(m.lit_px >= METER_MIN_LIT_PX) {
        return NIGHT_EV100;
    }
    let metered = m.ev_avg.max(m.ev_hi) + compensation_ev;
    let t = ((m.lit_px.ln() - METER_MIN_LIT_PX.ln())
        / (METER_FULL_LIT_PX.ln() - METER_MIN_LIT_PX.ln()))
    .clamp(0.0, 1.0);
    let w = t * t * (3.0 - 2.0 * t);
    // Written so w = 1 gives `metered` and w = 0 gives NIGHT exactly.
    clamp_ev100(metered * w + NIGHT_EV100 * (1.0 - w))
}

/// Feature name, also the GPU timing label.
const NAME: &str = "histogram";
/// Workgroup edge of `cs_histogram`: 16 × 16 threads, one per bin when flushing.
const WORKGROUP_EDGE: u32 = 16;
/// Bind group indices: the renderer's group 0 (`scene_pre()`) and the meter's own.
const CAMERA_GROUP: u32 = 0;
const METER_GROUP: u32 = 1;

/// The meter pass (`RenderFeature` named "histogram"), made by [`crate::auto_exposure`]. Push it
/// after the atmosphere pass and before any sky light (M1 frame contract, pass 9).
pub struct ExposureMeter {
    histogram: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    /// Layout of [`METER_GROUP`].
    layout: wgpu::BindGroupLayout,
    /// [`MeterBinsGpu`]: accumulated by `cs_histogram`, read and cleared by `cs_reduce`.
    bins: wgpu::Buffer,
    /// The auto-exposure state ([`ExposureStateGpu`]) shared with the resolve pass.
    state: wgpu::Buffer,
    /// The bind group over the current `targets.hdr_view`; `None` until the first `encode` and
    /// after a resize.
    binding: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

impl ExposureMeter {
    /// Builds both pipelines; `state` is the auto-exposure state buffer the resolve reads.
    pub(crate) fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        state: &wgpu::Buffer,
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let device = &gpu.device;
        let module = lib
            .module(device, crate::HISTOGRAM_WGSL)
            .context("compiling the exposure meter shader")?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meter bgl"),
            entries: &meter_layout_entries(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meter layout"),
            // Index = group: CAMERA_GROUP (0), METER_GROUP (1).
            bind_group_layouts: &[Some(camera_bgl), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let histogram = pipeline("cs_histogram");
        let reduce = pipeline("cs_reduce");
        // wgpu zero-initialises buffers, so the first frame starts from empty bins.
        let bins = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meter bins"),
            size: std::mem::size_of::<MeterBinsGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the exposure meter pipelines: {err}");
        }
        Ok(Self {
            histogram,
            reduce,
            layout,
            bins,
            state: state.clone(),
            binding: None,
        })
    }
}

/// [`METER_GROUP`]: binding 0 the hdr target (read with `textureLoad`, so any float format
/// binds), 1 the bins, 2 the auto-exposure state.
fn meter_layout_entries() -> [wgpu::BindGroupLayoutEntry; 3] {
    let storage = |binding: u32, size: usize| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size as u64),
        },
        count: None,
    };
    [
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
        storage(1, std::mem::size_of::<MeterBinsGpu>()),
        storage(2, std::mem::size_of::<ExposureStateGpu>()),
    ]
}

/// The meter's bind group for `targets.hdr_view`, rebuilt if `binding` was made for another
/// view. A free function over the fields so the caller can keep borrowing the others.
fn meter_bind_group<'a>(
    binding: &'a mut Option<(wgpu::TextureView, wgpu::BindGroup)>,
    layout: &wgpu::BindGroupLayout,
    bins: &wgpu::Buffer,
    state: &wgpu::Buffer,
    device: &wgpu::Device,
    targets: &FrameTargets,
) -> &'a wgpu::BindGroup {
    if binding
        .as_ref()
        .is_some_and(|(view, _)| *view != targets.hdr_view)
    {
        *binding = None;
    }
    let (_, bind_group) = binding.get_or_insert_with(|| {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meter bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bins.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.as_entire_binding(),
                },
            ],
        });
        (targets.hdr_view.clone(), bind_group)
    });
    bind_group
}

impl RenderFeature for ExposureMeter {
    fn name(&self) -> &'static str {
        NAME
    }

    /// Nothing to upload: the meter reads `scene_pre()` and `star_pre()` from group 0 and `hdr`.
    fn prepare(&mut self, _gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {}

    /// One compute pass: `cs_histogram` over every pixel, then `cs_reduce` in one workgroup.
    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        let bind_group = meter_bind_group(
            &mut self.binding,
            &self.layout,
            &self.bins,
            &self.state,
            &gpu.device,
            targets,
        );
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(NAME),
            timestamp_writes: timer.compute_writes(NAME),
        });
        pass.set_bind_group(CAMERA_GROUP, frame.camera, &[]);
        pass.set_bind_group(METER_GROUP, bind_group, &[]);
        pass.set_pipeline(&self.histogram);
        pass.dispatch_workgroups(
            targets.size.x.div_ceil(WORKGROUP_EDGE),
            targets.size.y.div_ceil(WORKGROUP_EDGE),
            1,
        );
        // Dispatches in one pass run in order (wgpu orders them through the bins buffer).
        pass.set_pipeline(&self.reduce);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Drops the bind group over the old hdr view; the next `encode` binds the new one.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {
        self.binding = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exposure::next_adapted_ev;
    use crate::{EV100_MAX, ExposureMode, ExposureSettings, NIGHT_EV100};
    use glam::Vec3;
    use std::f32::consts::PI;
    use uni_gfx::{Exposure, HDR_MAX, exposure_scale};

    /// AgX white in pre-exposed units: 2^4.026069 = 16.29 (`AGX_MAX_EV` in uni-post `agx`).
    const AGX_WHITE: f32 = 16.29;
    /// The 1920 × 1080 frame every bookmark is captured at.
    const FRAME_PX: u32 = 1920 * 1080;

    /// The lit pixels of a sphere of radius `r_px` seen from +z, lit by a distant light at phase
    /// angle `phase_rad` (the angle light → sphere → camera): Lambert, radiance `l0`·cos i on
    /// the day side (i = incidence angle) [cd/m²]. Phase 0 is a face-on full disk with `l0` at its
    /// centre.
    fn lambert_sphere(r_px: f32, phase_rad: f32, l0: f32) -> Vec<f32> {
        let sun = Vec3::new(phase_rad.sin(), 0.0, phase_rad.cos());
        let n = r_px.ceil() as i32 + 1;
        let mut lit = Vec::new();
        for y in -n..n {
            for x in -n..n {
                let (dx, dy) = ((x as f32 + 0.5) / r_px, (y as f32 + 0.5) / r_px);
                let r2 = dx * dx + dy * dy;
                if r2 >= 1.0 {
                    continue;
                }
                let mu0 = Vec3::new(dx, dy, (1.0 - r2).sqrt()).dot(sun);
                if mu0 > 0.0 {
                    lit.push(l0 * mu0);
                }
            }
        }
        lit
    }

    /// Auto exposure with captures' settle (τ = 0), compensation 0.
    fn settle_settings() -> ExposureSettings {
        ExposureSettings {
            mode: ExposureMode::Auto,
            manual: Exposure {
                scene_ev100: NIGHT_EV100,
                star_ev100: NIGHT_EV100,
            },
            compensation_ev: 0.0,
            eye_adapted: true,
            eye_star_ev100: NIGHT_EV100,
            daylight: false,
            adapt_tau_s: [0.6, 1.2],
            settle: true,
        }
    }

    /// The meter's reading of world luminances `lums` [cd/m²] rendered at scene EV `ev`: each
    /// pixel pre-exposed and clamped at `HDR_MAX` as the frame writes it, then read back through
    /// [`meter_luminance`]. The rest of the `total_px` frame is black.
    fn meter_at(lums: &[f32], total_px: u32, ev: f32) -> Option<Meter> {
        let pre = exposure_scale(ev);
        Histogram::from_luminances(
            lums.iter()
                .map(|&l| meter_luminance([(l * pre).min(HDR_MAX); 3], pre)),
        )
        .meter(total_px)
    }

    /// Scene EV frame by frame from `ev0`, with the resolve's settle step (the meter reads the
    /// previous frame): `[ev0, ev1, …, ev_frames]`.
    fn settle(lums: &[f32], total_px: u32, ev0: f32, frames: usize) -> Vec<f32> {
        settle_with(|ev| meter_at(lums, total_px, ev), ev0, frames)
    }

    /// [`settle`] with any meter: `read(ev)` is the reading of the frame rendered at scene EV `ev`.
    fn settle_with(read: impl Fn(f32) -> Option<Meter>, ev0: f32, frames: usize) -> Vec<f32> {
        let s = settle_settings();
        let mut evs = vec![ev0];
        for _ in 0..frames {
            let ev = *evs.last().unwrap();
            let next = next_adapted_ev(&s, Some(read(ev)), Some(ev), 1.0 / 60.0)
                .expect("Auto with a reading adapts");
            evs.push(next);
        }
        evs
    }

    /// `uni_planet::lod_exposure_mix` (B0), copied because uni-post does not depend on
    /// uni-planet: 0 at and below 1.5 px, 1 at and above 4 px, smoothstep in ln(radius) between.
    fn lod_mix(r_px: f32) -> f32 {
        let t = ((r_px / 1.5).ln() / (4.0f32 / 1.5).ln()).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// An f32 stored in and read back from an f16 channel of `hdr` (Rgba16Float).
    fn f16(x: f32) -> f32 {
        crate::probes::f16_value(crate::probes::f16_bits(x))
    }

    /// One grey world-light pixel of luminance `l` [cd/m²] as the frame contract writes it: at
    /// pre = mix(star_pre, scene_pre, w) (body.wgsl, rings.wgsl, star_surface.wgsl; w =
    /// `lod_exposure_mix`, 1 for a body of 4 px or more), clamped at `HDR_MAX`, with its star
    /// share 1 − w in alpha, stored as f16.
    fn hdr_pixel(l: f32, w: f32, scene_pre: f32, star_pre: f32) -> [f32; 4] {
        let pre = star_pre * (1.0 - w) + scene_pre * w;
        let v = f16((l * pre).min(HDR_MAX));
        [v, v, v, f16(1.0 - w)]
    }

    /// The meter's reading of world-light pixels `(l, w)` ([`hdr_pixel`]) rendered at scene EV
    /// `ev` and star EV `star_ev`; the rest of the `total_px` frame is black.
    fn meter_frame(px: &[(f32, f32)], total_px: u32, ev: f32, star_ev: f32) -> Option<Meter> {
        let (scene_pre, star_pre) = (exposure_scale(ev), exposure_scale(star_ev));
        Histogram::from_luminances(px.iter().map(|&(l, w)| {
            meter_pixel(hdr_pixel(l, w, scene_pre, star_pre), scene_pre, star_pre)
        }))
        .meter(total_px)
    }

    #[test]
    fn ev_from_uniform_luminance() {
        // Every pixel at 1000 cd/m²: ev_avg = log2(1000·S/K) = log2(1000·100/12.5) = 12.97.
        let n = 10_000u32;
        let h = Histogram::from_luminances(std::iter::repeat_n(1000.0, n as usize));
        let m = h.meter(n).expect("lit");
        let expected = (1000.0f32 * 100.0 / 12.5).log2();
        assert!((m.ev_avg - expected).abs() <= 0.05, "{m:?} vs {expected}");
        assert!(
            (m.ev_hi - (1000.0f32 / (1.2 * HIGHLIGHT_PRE)).log2()).abs() <= 0.05,
            "{m:?}"
        );
        assert!((m.lit_px / METER_REFERENCE_PX - 1.0).abs() < 1e-6, "{m:?}");
        assert!((target_ev(Some(m), 0.0) - expected).abs() <= 0.05);
        assert!((target_ev(Some(m), 1.5) - (m.ev_avg + 1.5)).abs() < 1e-5);
        // Counts only: every bin at its centre, within half a bin.
        let c = meter_histogram(&h.counts, n).expect("lit");
        assert!(
            (c.ev_avg - expected).abs() <= METER_BIN_EV / 2.0 + 1e-4,
            "{c:?} vs {expected}"
        );
        assert_eq!(c.lit_px, m.lit_px);
    }

    #[test]
    fn small_disk_not_blown_out() {
        // Review Focus 6, review C3: a 40-px-radius Lambert disk with 1000 cd/m² at its centre
        // on black (5 027 lit px of 2 073 600, 0.24 %). Its radiance CDF is (L/L0)², so the
        // 50th–95th percentile mean is L0·2^−0.244 → EV 12.72; ev_hi is 7.7.
        let disk = lambert_sphere(40.0, 0.0, 1000.0);
        assert!(
            (disk.len() as f32 - PI * 1600.0).abs() < 0.01 * PI * 1600.0,
            "{}",
            disk.len()
        );
        let evs = settle(&disk, FRAME_PX, NIGHT_EV100, 4);
        let ev = evs[4];
        assert!((evs[3] - ev).abs() < 1e-4, "not settled: {evs:?}");
        assert!((ev - 12.72).abs() < 0.05, "{evs:?}");
        let pre = exposure_scale(ev);
        let brightest = disk.iter().copied().fold(0.0f32, f32::max) * pre;
        assert!(brightest < AGX_WHITE, "{brightest}");
        assert!(
            (1000.0 * pre - 0.12).abs() < 0.01,
            "centre {}",
            1000.0 * pre
        );
    }

    #[test]
    fn meter_ramp_between_thresholds() {
        // Review Focus 6: NIGHT below 50 lit px, metered from 400, smoothstep in ln(lit_px)
        // between, monotonic and continuous at both ends.
        let at = |lit: f32, avg: f32, hi: f32| {
            target_ev(
                Some(Meter {
                    lit_px: lit,
                    ev_avg: avg,
                    ev_hi: hi,
                }),
                0.0,
            )
        };
        assert_eq!(at(25.0, 13.0, 8.0), NIGHT_EV100);
        assert_eq!(at(49.9, 13.0, 8.0), NIGHT_EV100);
        for lit in [400.0, 401.0, 5_027.0, METER_REFERENCE_PX] {
            assert_eq!(at(lit, 13.0, 8.0), 13.0, "{lit}");
            assert_eq!(at(lit, 13.0, 15.0), 15.0, "{lit}");
        }
        assert!((at(50.0 * (1.0 + 1e-4), 13.0, 8.0) - NIGHT_EV100).abs() < 1e-3);
        assert!((at(400.0 * (1.0 - 1e-4), 13.0, 8.0) - 13.0).abs() < 1e-3);
        // Halfway in ln(lit_px) (the geometric mean, 141 px) is halfway in EV.
        let mid = (50.0f32 * 400.0).sqrt();
        assert!((at(mid, 13.0, 8.0) - (NIGHT_EV100 + 13.0) / 2.0).abs() < 1e-3);
        // Monotonic between the thresholds, toward a metered EV above NIGHT and below it.
        for (avg, hi) in [(13.0, 8.0), (-9.0, -12.0)] {
            let mut prev = at(50.0, avg, hi);
            for k in 1..=1000 {
                let lit = 50.0 * 8.0f32.powf(k as f32 / 1000.0);
                let t = at(lit, avg, hi);
                if avg > NIGHT_EV100 {
                    assert!(t >= prev, "{lit}: {t} < {prev}");
                } else {
                    assert!(t <= prev, "{lit}: {t} > {prev}");
                }
                prev = t;
            }
            assert_eq!(prev, avg);
        }
    }

    #[test]
    fn start_ev_stable_when_lit_area_changes() {
        // Review C3 (it asked ≤ 1 stop): `start`'s Earth fills 23 400 px at phase 60°, a disk of
        // 31 200 px (lit fraction (1 + cos 60°)/2 = 0.75), radius 99.7 px. Lambert with ρ = 1.5·p_V
        // (p_V 0.434, A3) under 1.259e5 lux at 0.988 AU; the EV change does not depend on L0.
        let r = (23_400.0f32 / 0.75 / PI).sqrt();
        let l0 = 1.5 * 0.434 * (1.259e5 / (0.988 * 0.988)) / PI;
        let at = |deg: f32| {
            let lit = lambert_sphere(r, deg.to_radians(), l0);
            let ev = target_ev(
                Histogram::from_luminances(lit.iter().copied()).meter(FRAME_PX),
                0.0,
            );
            (lit.len() as f32 / FRAME_PX as f32, ev)
        };
        let (f60, ev60) = at(60.0);
        let (f90, ev90) = at(90.0);
        assert!((f60 - 0.0113).abs() < 2e-4, "{f60}");
        assert!((f90 - 0.0075).abs() < 2e-4, "{f90}");
        let d = (ev90 - ev60).abs();
        // The rule gives 0.32, from the lit part's shifting radiance distribution (the old 1 %
        // rule jumped 21 EV, from 17.1 to −4).
        assert!(d < 0.5, "{ev60} → {ev90}");
        assert!((d - 0.32).abs() < 0.05, "{ev60} → {ev90}");
    }

    #[test]
    fn highlight_rule_caps_bright_minority() {
        // 99 % of the lit pixels at 10 cd/m², 1 % at 1e5: the highlight rule sets the target,
        // and the 99.5th-percentile pixel (1e5) lands at HIGHLIGHT_PRE = 4.0 ± 2 %.
        let lums = std::iter::repeat_n(10.0, 9_900).chain(std::iter::repeat_n(1e5, 100));
        let m = Histogram::from_luminances(lums).meter(10_000).expect("lit");
        assert!(m.ev_hi > m.ev_avg, "{m:?}");
        let t = target_ev(Some(m), 0.0);
        assert_eq!(t, m.ev_hi);
        let pre = 1e5 * exposure_scale(t);
        assert!((pre / HIGHLIGHT_PRE - 1.0).abs() <= 0.02, "{pre}");
        assert!(pre < AGX_WHITE);
    }

    #[test]
    fn settle_converges_from_clamped_sun() {
        // Review Focus 6: the Sun's disk filling the frame, mean 1.853e9 cd/m²
        // (star_illuminance_lux(−26.74) / Ω☉ = 1.259e5 lux / 6.794e-5 sr), starting at EV −4.
        // Frame 1 reads the clamped 6e4 as 4 500 cd/m² → 15.1; frame 2 reads it unclamped
        // (4.4e4) → log2(1.853e9·8) = 33.79.
        let l = 1.853e9f32;
        let evs = settle(&vec![l; 10_000], 10_000, NIGHT_EV100, 4);
        assert!((evs[1] - 15.13).abs() < 0.05, "{evs:?}");
        let k = evs
            .iter()
            .position(|e| (e - 33.8).abs() < 0.05)
            .unwrap_or_else(|| panic!("never within 0.05 EV of 33.8: {evs:?}"));
        assert!(k <= 4, "{evs:?}");
        assert!((evs[4] - 33.8).abs() < 0.05, "{evs:?}");
        assert!(evs[4] <= EV100_MAX);
    }

    #[test]
    fn empty_histogram_targets_night_ev() {
        assert_eq!(Histogram::default().meter(FRAME_PX), None);
        assert_eq!(meter_histogram(&[0; METER_BINS], FRAME_PX), None);
        assert_eq!(target_ev(None, 3.0), NIGHT_EV100);
        // Pixels below METER_MIN_CD_M2, negative or NaN are not lit.
        let h = Histogram::from_luminances([0.0, 5e-4, 9.99e-4, -1.0, f32::NAN]);
        assert_eq!(h, Histogram::default());
        assert_eq!(h.meter(FRAME_PX), None);
        assert_eq!(meter_luminance([0.0; 3], exposure_scale(0.0)), 0.0);
        assert_eq!(meter_bin(METER_MIN_CD_M2).map(|b| b.0), Some(0));
    }

    #[test]
    fn meter_luminance_and_bins() {
        let pre = exposure_scale(10.0);
        // Rec. 709 Y ÷ scene_pre.
        let l = meter_luminance([0.3 * pre, 0.5 * pre, 0.9 * pre], pre);
        let y = 0.2126 * 0.3 + 0.7152 * 0.5 + 0.0722 * 0.9;
        assert!((l / y - 1.0).abs() < 1e-5, "{l} vs {y}");
        // A pixel clamped at HDR_MAX in any channel counts as at least HDR_MAX / scene_pre.
        assert_eq!(meter_luminance([0.0, 0.0, HDR_MAX], pre), HDR_MAX / pre);
        let inf = meter_luminance([f32::INFINITY; 3], pre);
        assert!(
            inf >= HDR_MAX / pre && inf <= HDR_MAX / pre * (1.0 + 1e-6),
            "{inf}"
        );
        // Bin edges: bin i covers [−10 + i·w, −10 + (i + 1)·w), steps of w/16 inside it.
        assert_eq!(meter_bin(1.0), Some((60, 15))); // log2 1 = 0: x = 10/w = 60.95
        assert_eq!(
            meter_bin(2.0f32.powf(-10.0 + 100.53 * METER_BIN_EV)),
            Some((100, 8))
        );
        assert_eq!(
            meter_bin(2.0f32.powf(40.0)),
            Some((255, METER_BIN_STEPS - 1))
        );
        assert_eq!(meter_bin(f32::INFINITY), Some((255, METER_BIN_STEPS - 1)));
    }

    #[test]
    fn meter_pixel_divides_by_its_own_pre() {
        // Fix round 1 (review r0 Important 1): hdr alpha is the pixel's star share s, and the
        // meter divides by mix(scene_pre, star_pre, s).
        let (scene, star) = (exposure_scale(15.0), exposure_scale(NIGHT_EV100));
        let rgbs = [
            [0.3, 0.5, 0.9],
            [1e-6, 2.0, 7.5],
            [0.0, 0.0, HDR_MAX],
            [HDR_MAX; 3],
        ];
        for [r, g, b] in rgbs {
            // s = 0 (the clear value, and every writer that leaves alpha alone) is the brief's
            // rule exactly: Y ÷ scene_pre.
            let zero = meter_pixel([r, g, b, 0.0], scene, star);
            assert_eq!(zero.to_bits(), meter_luminance([r, g, b], scene).to_bits());
            assert_eq!(meter_pixel([r, g, b, -1.0], scene, star), zero);
            // s = 1: star exposure; s = 2 and +∞ clamp to it.
            let one = meter_luminance([r, g, b], star);
            assert_eq!(meter_pixel([r, g, b, 1.0], scene, star), one);
            assert_eq!(meter_pixel([r, g, b, 2.0], scene, star), one);
            assert_eq!(meter_pixel([r, g, b, f32::INFINITY], scene, star), one);
            // Between: the writer's mix(star_pre, scene_pre, w) with w = 1 − s.
            let quarter = meter_luminance([r, g, b], scene * 0.75 + star * 0.25);
            assert_eq!(meter_pixel([r, g, b, 0.25], scene, star), quarter);
            // A NaN share: the pixel's pre is unknown, so it is not metered.
            assert_eq!(meter_bin(meter_pixel([r, g, b, f32::NAN], scene, star)), None);
        }
        // The HDR_MAX bound is taken at the pixel's own pre.
        let half = scene * 0.5 + star * 0.5;
        assert_eq!(
            meter_pixel([0.0, 0.0, HDR_MAX, 0.5], scene, star),
            HDR_MAX / half
        );
        assert_eq!(meter_pre(scene, star, 0.0), scene);
        assert_eq!(meter_pre(scene, star, 1.0), star);
    }

    #[test]
    fn earth_moon_band_pair_settles() {
        // Review r0 Important 1, its worked case: a full-phase Earth at r = 12 px (metered) with
        // the Moon at r = 3.2 px (w = 0.87) or 3.8 px (w = 0.99), inside the 1.5–4 px LOD band,
        // so drawn at mix(star_pre, scene_pre, w); eye-adapted stars at EV −4; settle (τ = 0).
        // Read ÷ scene_pre, the Moon's pixels grow by 2^(EV − EV_star) with the scene EV and the
        // loop runs to EV100_MAX (review: −4 → 15.1 → 26.1 → 37.2 → 40, a black frame). Read ÷
        // their own pre (star share in alpha), they read their true luminance and it settles.
        //
        // Lambert stand-ins at 1 AU (E = 1.259e5 lux, star_illuminance_lux(−26.74)), for the
        // loop's stability rather than an absolute EV: Earth's disk composite ρ = 0.302 (ruling
        // W-B1, ground and clouds without the atmosphere); the Moon ρ = 1.5·p_V = 0.18 (p_V
        // 0.12, NSSDCA Moon fact sheet, data/baked/body_facts.ron; a Lambert sphere has
        // p = 2ρ/3).
        let e = 1.259e5f32;
        let earth = lambert_sphere(12.0, 0.0, 0.302 * e / PI);
        let earth_px: Vec<(f32, f32)> = earth.iter().map(|&l| (l, 1.0)).collect();
        let alone = settle_with(
            |ev| meter_frame(&earth_px, FRAME_PX, ev, NIGHT_EV100),
            NIGHT_EV100,
            8,
        );
        assert!(earth.len() >= 400, "Earth is fully metered: {}", earth.len());
        assert!((alone[8] - alone[3]).abs() < 1e-3, "{alone:?}");
        for (moon_r, w_expected) in [(3.2f32, 0.87f32), (3.8, 0.99)] {
            let w = lod_mix(moon_r);
            assert!((w - w_expected).abs() < 0.005, "{moon_r} px: w {w}");
            let moon = lambert_sphere(moon_r, 0.0, 0.18 * e / PI);
            let px: Vec<(f32, f32)> = earth_px
                .iter()
                .copied()
                .chain(moon.iter().map(|&l| (l, w)))
                .collect();
            let evs = settle_with(
                |ev| meter_frame(&px, FRAME_PX, ev, NIGHT_EV100),
                NIGHT_EV100,
                8,
            );
            let ev = evs[8];
            // Settled from frame 3 on (frame 1 reads Earth clamped at HDR_MAX).
            assert!(
                evs[3..].iter().all(|e| (e - ev).abs() < 1e-3),
                "Moon at {moon_r} px: not settled {evs:?}"
            );
            // The true-luminance meter of the same pixels, and Earth alone (the dimmer Moon
            // barely moves it).
            let truth = target_ev(
                Histogram::from_luminances(earth.iter().chain(&moon).copied()).meter(FRAME_PX),
                0.0,
            );
            assert!(
                (ev - truth).abs() < 0.05,
                "Moon at {moon_r} px: {ev} vs true {truth}; {evs:?}"
            );
            assert!(
                (ev - alone[8]).abs() < 0.25,
                "Moon at {moon_r} px: {ev} vs Earth alone {}",
                alone[8]
            );
            assert!(ev < 20.0, "Moon at {moon_r} px: ran away {evs:?}");
        }
    }

    #[test]
    fn mixed_band_pixel_reads_within_its_luminances() {
        // Fix round 1: a pixel part-covered (coverage c) by a body in the LOD band (luminance
        // lb, star share sb = 1 − w) over world light at scene exposure (ld, share 0), with the
        // share composited like the colour ("over", premultiplied alpha): colour
        // c·lb·pre(sb) + (1 − c)·ld·pre(0), share c·sb. pre(s) = mix(scene_pre, star_pre, s) is
        // affine in s, so pre(c·sb) = c·pre(sb) + (1 − c)·pre(0) and the meter reads the
        // pre-weighted mean of lb and ld: within [min, max] at every scene EV, so it cannot feed
        // back. Swept over the whole EV range, with f16 storage.
        let star = exposure_scale(NIGHT_EV100);
        let mut checked = 0;
        for ev in (-30..=40).map(|e| e as f32) {
            let scene = exposure_scale(ev);
            for c in [0.1f32, 0.5, 0.9, 1.0] {
                for w in [0.0f32, 0.5, 0.87, 0.99] {
                    for (lb, ld) in [(5000.0f32, 100.0f32), (100.0, 5000.0), (2.0, 3.0e4)] {
                        let pre_b = star * (1.0 - w) + scene * w;
                        let raw = c * lb * pre_b + (1.0 - c) * ld * scene;
                        let v = f16(raw.min(HDR_MAX));
                        let l = meter_pixel([v, v, v, f16(c * (1.0 - w))], scene, star);
                        let (lo, hi) = (lb.min(ld), lb.max(ld));
                        assert!(
                            l <= hi * 1.002,
                            "EV {ev}, c {c}, w {w}, ({lb}, {ld}): read {l} > {hi}"
                        );
                        // Unclipped, the reading is also not below the dimmer one.
                        if raw < HDR_MAX && v >= 6.2e-5 {
                            assert!(
                                l >= lo * 0.998,
                                "EV {ev}, c {c}, w {w}, ({lb}, {ld}): read {l} < {lo}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 71 * 4 * 4 * 3);
    }
}
