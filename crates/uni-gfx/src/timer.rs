//! GPU pass timing with timestamp queries; degrades to a no-op when `TIMESTAMP_QUERY` is absent.
//!
//! Each frame records into one slot of a [`RING_LEN`]-deep ring. A slot has its own `QuerySet`
//! of 2 × `max_passes` timestamps (pass *i* of the frame writes its begin and end times to
//! queries 2*i* and 2*i* + 1) and a mappable read-back buffer. A slot's life:
//! 1. **Recording**: taken by the frame's first timed pass. With no free slot the frame is not
//!    timed (its timings are dropped rather than stalling the CPU).
//! 2. **In flight**: [`PassTimer::resolve`] closes the record and maps the slot's read-back
//!    buffer on submit (`map_buffer_on_submit`). That map completes when the frame's command
//!    buffer has finished on the GPU; it carries no data, only the completion.
//! 3. **Completed**: [`PassTimer::read_back`] (non-blocking) sees the completion and holds the
//!    slot until a LATER frame's command buffer has completed too (final fix wave F3, below).
//! 4. **Resolving**: it then resolves the slot's queries into the read-back buffer in a small
//!    command buffer of its own, which it submits, and maps the buffer again.
//! 5. A later `read_back` converts the ticks with `queue.get_timestamp_period()` and frees the
//!    slot. [`PassTimer::read_back_blocking`] runs steps 3–5 to the end for headless runs,
//!    resolving a completed slot without waiting for a later frame (the caller has waited for
//!    the GPU, and serial runs read every sample).
//!
//! Why the resolve waits for the frame (measured on the M3, M0 final fix). Metal writes a
//! pass's stage-boundary samples into the counter buffer late. A resolve encoded in the same
//! command buffer, even one ordered after every pass, read the previous frame's values for the
//! tonemap's end and the closing frame bracket, and zeros on the first frame; resolving after
//! the command buffer completed read every sample. That, not the render pass itself, is why M0
//! reported "tonemap 0.00 ms" every frame.
//!
//! Why a completed frame waits for the next one (final fix wave F3, measured on the M3 with
//! `RUST_LOG=uni_gfx::timer=debug` in a sustained `--bench`). Even resolved after its command
//! buffer completed, a frame's LAST samples — the tonemap's end and both ticks of the closing
//! bracket — still held the values its query slot had two frames earlier in most frames (C6:
//! the tonemap valid in 0.1–25 % of frames), while every earlier sample was current. Metal
//! writes those samples some time after the completion callback. Serial runs, which resolve
//! after waiting for the GPU, read them all. So a slot is resolved only once a frame encoded
//! after it has completed, one frame of extra lag; the ring is [`RING_LEN`] deep so the wait
//! does not starve it.
//!
//! Validity (review I5). A pass whose end tick is not after its begin tick, or whose ticks are
//! unwritten (0 or all ones), has no measurement: its time is `None`, shown as "n/a", never as
//! 0 ms. Render passes are sampled at the start of the vertex stage and the end of the fragment
//! stage (wgpu-hal 30, Metal), so a render pass's time can include waiting for the pass before
//! it; compute passes are sampled at the start and end of the encoder. One exception: the
//! frame's last pass (the tonemap) often has no end sample yet in sustained runs (C6), so its end
//! is taken from the closing bracket's begin tick, an upper bound (`last_pass_end`).
//!
//! Whole-frame GPU time. [`PassTimer::begin_frame`] and [`PassTimer::end_frame`] encode a timed
//! one-thread compute pass at the start and end of the frame; the frame time is the end
//! bracket's end tick minus the begin bracket's begin tick (`TIMESTAMP_QUERY_INSIDE_ENCODERS` is
//! unavailable on the M3: docs/research/wgpu30-skeleton/m3-probe-output.txt). Metal orders
//! encoders only through the tracked resources they share (wgpu-hal 30 keeps Metal's default
//! tracked hazard mode, and its barriers are no-ops), so a bracket that touches nothing may run
//! alongside the frame: on the M3 an empty one measured a 0.07 ms "frame" around a 148 ms lens
//! pass. So each bracket writes texel (0, 0) of an *anchor* texture, the HDR target. The
//! opening bracket comes before the clear that overwrites the anchor (write after write, and
//! after the previous frame's last use of it); the closing bracket comes after every pass that
//! writes or reads it, the tonemap included (write after read). Nothing reads the HDR target
//! after the frame's last pass, and the next frame clears it, so the texel written last is
//! never seen.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::context::GpuContext;

/// Depth of the ring: frames whose timings can be in flight at once (recording, waiting for
/// the GPU, or being resolved).
const RING_LEN: usize = 6;

/// States of a map, written by its callback.
const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;
/// The command buffer carrying the map was dropped without being submitted.
const MAP_NEVER_SUBMITTED: u8 = 3;

/// Labels of the two frame-bracket passes. They are not reported as passes.
const FRAME_BEGIN: &str = "frame begin";
const FRAME_END: &str = "frame end";

/// The frame-bracket kernel: one thread that writes 0 to texel (0, 0) of the anchor texture
/// (see the module docs), so the bracket is a real compute encoder ordered against every pass
/// that uses the anchor.
const BRACKET_WGSL: &str = "\
@group(0) @binding(0) var anchor: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(1, 1, 1)
fn cs_bracket() {
    textureStore(anchor, vec2<u32>(0u, 0u), vec4<f32>(0.0));
}
";

/// The anchor texture's format: the HDR target's ([`crate::HDR_FORMAT`]).
const ANCHOR_FORMAT: wgpu::TextureFormat = crate::targets::HDR_FORMAT;

/// One frame's GPU timings, in milliseconds.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuTimings {
    /// Order of the measured frame among the frames this timer recorded (0, 1, 2, ...).
    pub seq: u64,
    /// (pass label, ms) in encode order; `None` where the device returned no valid
    /// measurement for that pass. The frame's last pass (the tonemap) may end at the closing
    /// bracket's begin tick instead of its own unwritten or stale end (`last_pass_end`): its
    /// time is then an upper bound, late by the bracket's launch latency.
    pub passes: Vec<(&'static str, Option<f32>)>,
    /// (pass label, ms) in encode order: each pass's END tick minus the opening frame bracket's
    /// begin tick; `None` where either tick is unwritten or the end is not after the frame's
    /// begin. Per-pass durations are differences of these (B9t). The spans in `passes` do not
    /// difference cleanly on Metal: a render pass's begin tick is sampled early (vertex stage),
    /// a compute pass's exactly, so overlapping spans made compute passes read 0.
    pub pass_end_ms: Vec<(&'static str, Option<f32>)>,
    /// Whole-frame GPU time between the frame brackets; `None` if unavailable or invalid.
    pub frame_ms: Option<f32>,
}

/// Collects per-pass GPU times. Results lag the frame by at least one read-back.
pub struct PassTimer {
    /// `None` when disabled (no `TIMESTAMP_QUERY`, or `max_passes == 0`).
    gpu: Option<TimerGpu>,
    /// The most recent completed frame's timings.
    last: Option<GpuTimings>,
}

struct TimerGpu {
    /// The frame-bracket kernel ([`BRACKET_WGSL`]); `None` if it could not be created, in
    /// which case no whole-frame time is measured.
    bracket: Option<Bracket>,
    /// `QUERY_RESOLVE | COPY_SRC`, 2 × max_passes × 8 bytes: where slots are resolved before
    /// the copy into their read-back buffers.
    resolve_buffer: wgpu::Buffer,
    max_passes: u32,
    /// Nanoseconds per timestamp tick.
    period_ns: f32,
    /// This frame's slot: `None` before its first timed pass, `Some(None)` when no slot was
    /// free (the frame is not timed), `Some(Some(i))` while recording into slot `i`.
    frame_slot: Option<Option<usize>>,
    /// Labels of the passes given timestamp writes so far this frame, in query-pair order.
    frame_labels: Vec<&'static str>,
    ring: [Slot; RING_LEN],
    next_seq: u64,
}

/// One ring slot (see the module docs).
struct Slot {
    query_set: wgpu::QuerySet,
    /// `MAP_READ | COPY_DST`, the same size as the resolve buffer.
    buffer: wgpu::Buffer,
    labels: Vec<&'static str>,
    /// Frame order, to keep the newest result when several finish together.
    seq: u64,
    state: SlotState,
}

enum SlotState {
    Free,
    Recording,
    /// Submitted with its frame; the map completes when the frame's GPU work has finished.
    InFlight(Arc<AtomicU8>),
    /// The frame's GPU work has finished; waiting for a later frame's to finish too before the
    /// resolve (see the module docs).
    Completed,
    /// Resolved in the timer's own submission; the map completes when the ticks are readable.
    Resolving(Arc<AtomicU8>),
}

/// Owned by a map callback. Stores the map result when called; if the callback is dropped
/// without running (its command buffer was never submitted) it stores that instead, so the
/// slot is freed rather than leaked.
struct MapSignal(Arc<AtomicU8>);

impl MapSignal {
    fn finish(self, ok: bool) {
        self.0
            .store(if ok { MAP_READY } else { MAP_FAILED }, Ordering::Release);
    }
}

impl Drop for MapSignal {
    fn drop(&mut self) {
        // Only changes the state if `finish` never ran.
        let _ = self.0.compare_exchange(
            MAP_PENDING,
            MAP_NEVER_SUBMITTED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

/// A fresh map state and the callback that completes it.
fn map_signal() -> (
    Arc<AtomicU8>,
    impl FnOnce(Result<(), wgpu::BufferAsyncError>) + Send,
) {
    let state = Arc::new(AtomicU8::new(MAP_PENDING));
    let signal = MapSignal(Arc::clone(&state));
    (state, move |r: Result<(), wgpu::BufferAsyncError>| {
        signal.finish(r.is_ok())
    })
}

fn readback_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pass timer read-back"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl TimerGpu {
    /// Reserves the next (begin, end) query pair for `label` in this frame's slot, taking a free
    /// slot at the frame's first timed pass. `None` when the frame is full or not timed.
    fn next_pair(&mut self, label: &'static str) -> Option<(&wgpu::QuerySet, u32, u32)> {
        let slot = *self.frame_slot.get_or_insert_with(|| {
            let free = self
                .ring
                .iter()
                .position(|s| matches!(s.state, SlotState::Free));
            if let Some(i) = free {
                self.ring[i].state = SlotState::Recording;
            }
            free
        });
        let slot = slot?;
        let i = u32::try_from(self.frame_labels.len()).ok()?;
        if i >= self.max_passes {
            return None;
        }
        self.frame_labels.push(label);
        Some((&self.ring[slot].query_set, 2 * i, 2 * i + 1))
    }

    /// Moves every slot whose frame has finished on the GPU to `Completed`, then resolves the
    /// queries of every completed slot that a later finished frame follows (all of them with
    /// `force`) into its read-back buffer, in one submission of the timer's own, and maps the
    /// buffers.
    fn resolve_finished(&mut self, gpu: &GpuContext, force: bool) {
        let mut newest_done: Option<u64> = None;
        for slot in &mut self.ring {
            let SlotState::InFlight(done) = &slot.state else {
                if matches!(slot.state, SlotState::Completed) {
                    newest_done = newest_done.max(Some(slot.seq));
                }
                continue;
            };
            match done.load(Ordering::Acquire) {
                MAP_PENDING => continue,
                MAP_READY => {
                    // The completion map carried no data; release it for the copy.
                    slot.buffer.unmap();
                    slot.state = SlotState::Completed;
                    newest_done = newest_done.max(Some(slot.seq));
                }
                MAP_FAILED => {
                    // A failed map leaves the buffer unusable for another map; replace it.
                    log::warn!("pass timer: frame completion map failed; recreating the buffer");
                    slot.buffer = readback_buffer(&gpu.device, slot.buffer.size());
                    slot.free();
                }
                _ => slot.free(), // MAP_NEVER_SUBMITTED: the frame was never submitted
            }
        }
        // Frames complete in submission order, so a completed slot older than the newest
        // completed one has had a later frame finish after it.
        let newest_done = newest_done.unwrap_or(0);
        let mut enc: Option<wgpu::CommandEncoder> = None;
        for slot in &mut self.ring {
            if !matches!(slot.state, SlotState::Completed)
                || !resolve_now(slot.seq, newest_done, force)
            {
                continue;
            }
            let enc = enc.get_or_insert_with(|| {
                gpu.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pass timer resolve"),
                    })
            });
            let queries = 2 * slot.labels.len() as u32; // ≤ 2 × max_passes
            let bytes = u64::from(queries) * u64::from(wgpu::QUERY_SIZE);
            enc.resolve_query_set(&slot.query_set, 0..queries, &self.resolve_buffer, 0);
            enc.copy_buffer_to_buffer(&self.resolve_buffer, 0, &slot.buffer, 0, bytes);
            let (state, callback) = map_signal();
            enc.map_buffer_on_submit(&slot.buffer, wgpu::MapMode::Read, 0..bytes, callback);
            slot.state = SlotState::Resolving(state);
        }
        if let Some(enc) = enc {
            gpu.queue.submit(std::iter::once(enc.finish()));
        }
    }

    /// Reads every resolved slot whose map has finished and frees it; returns the newest
    /// frame's timings among them.
    fn collect_resolved(&mut self, gpu: &GpuContext) -> Option<GpuTimings> {
        let mut newest: Option<GpuTimings> = None;
        for slot in &mut self.ring {
            let SlotState::Resolving(map) = &slot.state else {
                continue;
            };
            match map.load(Ordering::Acquire) {
                MAP_PENDING => continue,
                MAP_READY => {
                    let bytes = 2 * slot.labels.len() as u64 * u64::from(wgpu::QUERY_SIZE);
                    match slot.buffer.get_mapped_range(0..bytes) {
                        Ok(view) => {
                            let ticks: Vec<u64> = view
                                .chunks_exact(wgpu::QUERY_SIZE as usize)
                                .map(bytemuck::pod_read_unaligned::<u64>)
                                .collect();
                            drop(view);
                            if newest.as_ref().is_none_or(|t| slot.seq > t.seq) {
                                newest = Some(timings_from_ticks(
                                    slot.seq,
                                    &slot.labels,
                                    &ticks,
                                    self.period_ns,
                                ));
                            }
                        }
                        Err(e) => log::warn!("pass timer: mapped range unavailable: {e}"),
                    }
                    slot.buffer.unmap();
                }
                MAP_FAILED => {
                    log::warn!("pass timer: read-back map failed; recreating the buffer");
                    slot.buffer = readback_buffer(&gpu.device, slot.buffer.size());
                }
                _ => {} // MAP_NEVER_SUBMITTED: the resolve was never submitted
            }
            slot.free();
        }
        newest
    }

    fn any_resolving(&self) -> bool {
        self.ring
            .iter()
            .any(|s| matches!(s.state, SlotState::Resolving(_)))
    }
}

impl Slot {
    fn free(&mut self) {
        self.state = SlotState::Free;
        self.labels.clear();
    }
}

impl PassTimer {
    /// A timer for up to `max_passes` passes per frame (disabled if the device lacks timestamps).
    pub fn new(gpu: &GpuContext, max_passes: u32) -> Self {
        let max_passes = max_passes.min(wgpu::QUERY_SET_MAX_QUERIES / 2);
        if max_passes == 0 || !gpu.has_timestamps() {
            return Self::disabled();
        }
        let count = 2 * max_passes;
        let bytes = u64::from(count) * u64::from(wgpu::QUERY_SIZE);
        let resolve_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass timer resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bracket = Bracket::new(&gpu.device);
        let ring = std::array::from_fn(|_| Slot {
            query_set: gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timer"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            buffer: readback_buffer(&gpu.device, bytes),
            labels: Vec::new(),
            seq: 0,
            state: SlotState::Free,
        });
        Self {
            gpu: Some(TimerGpu {
                bracket,
                resolve_buffer,
                max_passes,
                period_ns: gpu.queue.get_timestamp_period(),
                frame_slot: None,
                frame_labels: Vec::with_capacity(max_passes as usize),
                ring,
                next_seq: 0,
            }),
            last: None,
        }
    }

    /// A timer that records nothing.
    pub fn disabled() -> Self {
        Self {
            gpu: None,
            last: None,
        }
    }

    /// Encodes the timed compute pass that opens the frame. Call once, before the frame's
    /// first pass; [`Self::end_frame`] closes it. `anchor` is the frame's HDR target view
    /// (`Rgba16Float`, storage-capable), whose texel (0, 0) the bracket overwrites: the frame
    /// must clear it afterwards (see the module docs).
    pub fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        anchor: &wgpu::TextureView,
    ) {
        self.bracket(device, enc, anchor, FRAME_BEGIN);
    }

    /// Encodes the timed compute pass that closes the frame. Call once, after the frame's last
    /// pass and before [`Self::resolve`], with the same `anchor` as [`Self::begin_frame`]. It
    /// overwrites texel (0, 0) of `anchor`, so nothing may read the anchor after it.
    pub fn end_frame(
        &mut self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        anchor: &wgpu::TextureView,
    ) {
        self.bracket(device, enc, anchor, FRAME_END);
    }

    /// One timed one-thread compute pass named `label`, writing texel (0, 0) of `anchor`.
    fn bracket(
        &mut self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        anchor: &wgpu::TextureView,
        label: &'static str,
    ) {
        let Some(g) = self.gpu.as_mut() else { return };
        let Some(bracket) = g.bracket.as_mut() else {
            return;
        };
        let bind_group = bracket.bind_group(device, anchor).clone();
        let pipeline = bracket.pipeline.clone();
        let Some((query_set, begin, end)) = g.next_pair(label) else {
            return;
        };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(begin),
                end_of_pass_write_index: Some(end),
            }),
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Timestamp writes for a compute pass named `label`, or `None` when disabled, full or
    /// when this frame is not timed.
    pub fn compute_writes(
        &mut self,
        label: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (query_set, begin, end) = self.gpu.as_mut()?.next_pair(label)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Timestamp writes for a render pass named `label`, or `None` when disabled, full or when
    /// this frame is not timed.
    pub fn render_writes(
        &mut self,
        label: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (query_set, begin, end) = self.gpu.as_mut()?.next_pair(label)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Closes this frame's record. Call once per frame after the last timed pass (and
    /// [`Self::end_frame`]) is encoded into `enc`, the frame's last encoder. The queries are
    /// resolved later, by [`Self::read_back`], once the frame's command buffer has finished on
    /// the GPU (see the module docs for why).
    pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder) {
        let Some(g) = self.gpu.as_mut() else { return };
        let labels = std::mem::take(&mut g.frame_labels);
        let Some(Some(i)) = g.frame_slot.take() else {
            return; // nothing timed this frame, or no free slot
        };
        let slot = &mut g.ring[i];
        if labels.is_empty() {
            slot.free();
            return;
        }
        // The map carries no data: its callback runs once this frame's GPU work has finished.
        let (state, callback) = map_signal();
        enc.map_buffer_on_submit(
            &slot.buffer,
            wgpu::MapMode::Read,
            0..u64::from(wgpu::QUERY_SIZE),
            callback,
        );
        slot.labels = labels;
        slot.seq = g.next_seq;
        g.next_seq += 1;
        slot.state = SlotState::InFlight(state);
    }

    /// Non-blocking read-back. Collects the timings of every frame whose resolve has finished,
    /// and submits the resolve of every frame that has finished on the GPU and that a later
    /// finished frame follows (module docs).
    pub fn read_back(&mut self, gpu: &GpuContext) {
        self.read_back_with(gpu, false);
    }

    fn read_back_with(&mut self, gpu: &GpuContext, force: bool) {
        let Some(g) = self.gpu.as_mut() else { return };
        if g.ring
            .iter()
            .all(|s| matches!(s.state, SlotState::Free | SlotState::Recording))
        {
            return;
        }
        // Never waits: only runs callbacks of work that has already finished.
        if let Err(e) = gpu.device.poll(wgpu::PollType::Poll) {
            log::warn!("pass timer: device poll failed: {e}");
        }
        let newest = g.collect_resolved(gpu);
        g.resolve_finished(gpu, force);
        // The queue completes submissions in order, so the highest `seq` is the newest frame.
        if let Some(t) = newest.filter(|t| self.last.as_ref().is_none_or(|l| t.seq > l.seq)) {
            self.last = Some(t);
        }
    }

    /// Blocking read-back for headless runs: after the caller has waited for its frames, this
    /// resolves every finished frame, waits for the resolves and collects them, so the latest
    /// timings are those of the last finished frame.
    pub fn read_back_blocking(&mut self, gpu: &GpuContext) {
        self.read_back_with(gpu, true);
        if self.gpu.as_ref().is_some_and(TimerGpu::any_resolving) {
            if let Err(e) = gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
                log::warn!("pass timer: waiting for the resolve failed: {e}");
            }
            self.read_back_with(gpu, true);
        }
    }

    /// The most recent completed frame's timings; `None` when disabled or before the first
    /// read-back.
    pub fn last(&self) -> Option<&GpuTimings> {
        self.last.as_ref()
    }

    /// The most recent (label, milliseconds) pairs, `None` for an invalid measurement; empty
    /// when disabled or before the first read-back.
    pub fn last_ms(&self) -> Vec<(&'static str, Option<f32>)> {
        self.last
            .as_ref()
            .map_or_else(Vec::new, |t| t.passes.clone())
    }

    /// The most recent (label, end time on the frame's origin [ms]) pairs
    /// ([`GpuTimings::pass_end_ms`]); empty when disabled or before the first read-back.
    pub fn last_end_ms(&self) -> Vec<(&'static str, Option<f32>)> {
        self.last
            .as_ref()
            .map_or_else(Vec::new, |t| t.pass_end_ms.clone())
    }

    /// The most recent whole-frame GPU time [ms], if measured and valid.
    pub fn last_frame_ms(&self) -> Option<f32> {
        self.last.as_ref().and_then(|t| t.frame_ms)
    }
}

/// The frame-bracket pipeline and its bind group over the current anchor view.
struct Bracket {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Bind group for the anchor view it was made for; rebuilt when the view changes (the
    /// frame targets were recreated).
    binding: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

impl Bracket {
    /// The bracket pipeline, or `None` (logged) if the device rejects it.
    fn new(device: &wgpu::Device) -> Option<Self> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pass timer frame bracket"),
            source: wgpu::ShaderSource::Wgsl(BRACKET_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pass timer frame bracket"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: ANCHOR_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pass timer frame bracket"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pass timer frame bracket"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_bracket"),
            compilation_options: Default::default(),
            cache: None,
        });
        match pollster::block_on(scope.pop()) {
            None => Some(Self {
                pipeline,
                layout,
                binding: None,
            }),
            Some(e) => {
                log::warn!("pass timer: no whole-frame GPU time ({e})");
                None
            }
        }
    }

    /// The bind group over `anchor`, rebuilt if the cached one was made for another view.
    fn bind_group(
        &mut self,
        device: &wgpu::Device,
        anchor: &wgpu::TextureView,
    ) -> &wgpu::BindGroup {
        if self
            .binding
            .as_ref()
            .is_some_and(|(view, _)| view != anchor)
        {
            self.binding = None;
        }
        let layout = &self.layout;
        let (_, bind_group) = self.binding.get_or_insert_with(|| {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pass timer frame bracket"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(anchor),
                }],
            });
            (anchor.clone(), bind_group)
        });
        bind_group
    }
}

/// Whether a completed frame `seq` may be resolved now: always with `force` (headless runs,
/// after waiting for the GPU), otherwise once a later frame has completed (`newest_done`, the
/// newest completed frame's seq, is greater; module docs).
fn resolve_now(seq: u64, newest_done: u64, force: bool) -> bool {
    force || seq < newest_done
}

/// Milliseconds from tick `begin` to tick `end`, or `None` when the pair is not a measurement:
/// either tick unwritten (0, or all ones as Metal's `MTLCounterErrorValue`), or `end ≤ begin`.
fn span_ms(begin: u64, end: u64, period_ns: f32) -> Option<f32> {
    let unwritten = |t: u64| t == 0 || t == u64::MAX;
    if unwritten(begin) || unwritten(end) || end <= begin {
        return None;
    }
    let ns = (end - begin) as f64 * f64::from(period_ns);
    Some((ns * 1e-6) as f32)
}

/// The end tick to use for the frame's last pass (the one encoded right before the closing
/// bracket), given its own `(begin, end)` ticks, the opening bracket's begin tick and the closing
/// bracket's begin tick (final review render I1).
///
/// In sustained runs Metal often has not written the end-of-fragment sample of the frame's last
/// render pass (the tonemap, which draws into the output view) when the slot is resolved: C6
/// logged it as 0, or as a stale tick from an earlier frame (the slot's query reused when the
/// label list shifts), in 75–99.9 % of frames. The closing bracket is ordered after that pass by
/// Metal's hazard tracking, because the last pass reads the anchor texel the bracket writes (the
/// tonemap samples `hdr`; write after read), and C6 found the bracket's ticks always written. So
/// when the last pass's end is unwritten, not after its own begin, or before the frame's begin,
/// the closing bracket's BEGIN tick stands in for it: an upper bound on the pass's end, late by
/// the bracket's launch latency (measured on the M3 in serial runs, where both ticks are valid:
/// final-fix-F3-report.md). A valid end is kept as it is. `None` leaves the pass unmeasured.
fn last_pass_end(own: (u64, u64), frame_begin: Option<u64>, bracket_begin: u64) -> Option<u64> {
    let unwritten = |t: u64| t == 0 || t == u64::MAX;
    let (begin, end) = own;
    let stale = unwritten(end) || end <= begin || frame_begin.is_some_and(|b| end < b);
    if !stale {
        return Some(end);
    }
    let usable = !unwritten(bracket_begin)
        && (unwritten(begin) || bracket_begin > begin)
        && frame_begin.is_none_or(|b| bracket_begin > b);
    usable.then_some(bracket_begin)
}

/// Pairs each label with its (begin, end) ticks: the passes' times, and the whole-frame time
/// from the begin bracket's begin tick to the end bracket's end tick. The last pass before the
/// closing bracket takes that bracket's begin tick as its end when its own is unwritten or stale
/// ([`last_pass_end`]); its time and end stamp are then upper bounds.
fn timings_from_ticks(
    seq: u64,
    labels: &[&'static str],
    ticks: &[u64],
    period_ns: f32,
) -> GpuTimings {
    let pairs: Vec<(&'static str, (u64, u64))> = labels
        .iter()
        .zip(ticks.chunks_exact(2))
        .map(|(&l, t)| (l, (t[0], t[1])))
        .collect();
    let frame_begin = pairs
        .iter()
        .find(|(l, _)| *l == FRAME_BEGIN)
        .map(|(_, t)| t.0);
    let end_at = pairs.iter().position(|(l, _)| *l == FRAME_END);
    let frame_end = end_at.map(|i| pairs[i].1.1);
    // The pass encoded right before the closing bracket (the tonemap in the app).
    let last = end_at
        .and_then(|e| e.checked_sub(1))
        .filter(|&i| !matches!(pairs[i].0, FRAME_BEGIN | FRAME_END));
    let end_of = |i: usize| -> u64 {
        let (_, own) = pairs[i];
        match (last, end_at) {
            (Some(l), Some(e)) if l == i => {
                let end = last_pass_end(own, frame_begin, pairs[e].1.0);
                if end != Some(own.1) {
                    log::debug!(
                        "pass timer frame {seq}: {} end {} taken from the closing bracket's \
                         begin {:?}",
                        pairs[i].0,
                        own.1,
                        end
                    );
                }
                // No usable stand-in: keep the invalid tick, which reads as no measurement.
                end.unwrap_or(own.1)
            }
            _ => own.1,
        }
    };
    let mut passes = Vec::with_capacity(labels.len());
    let mut pass_end_ms = Vec::with_capacity(labels.len());
    for (i, &(label, (begin, _))) in pairs.iter().enumerate() {
        if matches!(label, FRAME_BEGIN | FRAME_END) {
            continue;
        }
        let end = end_of(i);
        passes.push((label, span_ms(begin, end, period_ns)));
        pass_end_ms.push((label, frame_begin.and_then(|b| span_ms(b, end, period_ns))));
    }
    let frame_ms = match (frame_begin, frame_end) {
        (Some(b), Some(e)) => span_ms(b, e, period_ns),
        _ => None,
    };
    if log::log_enabled!(log::Level::Debug) {
        let pairs: Vec<String> = labels
            .iter()
            .zip(ticks.chunks_exact(2))
            .map(|(l, t)| format!("{l} [{}, {}]", t[0], t[1]))
            .collect();
        log::debug!("pass timer frame {seq} ticks: {}", pairs.join(", "));
    }
    GpuTimings {
        seq,
        passes,
        pass_end_ms,
        frame_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Review Focus 4: without TIMESTAMP_QUERY everything still renders and timing reads "n/a".
    #[test]
    fn disabled_timer_yields_none() {
        let mut t = PassTimer::disabled();
        assert!(t.compute_writes("x").is_none());
        assert!(t.render_writes("x").is_none());
        assert!(t.last_ms().is_empty());
        assert!(t.last().is_none());
        assert_eq!(t.last_frame_ms(), None);
    }

    #[test]
    fn ticks_convert_to_ms() {
        // period 1 ns/tick (Apple M3, docs/research/wgpu30-skeleton/m3-probe-output.txt).
        let t = timings_from_ticks(
            7,
            &["lens", "stars"],
            &[1_000, 2_501_000, 3_000_000, 3_250_000],
            1.0,
        );
        assert_eq!(t.seq, 7);
        assert_eq!(t.passes.len(), 2);
        assert_eq!(t.passes[0].0, "lens");
        assert!((t.passes[0].1.unwrap() - 2.5).abs() < 1e-6);
        assert_eq!(t.passes[1].0, "stars");
        assert!((t.passes[1].1.unwrap() - 0.25).abs() < 1e-6);
        assert_eq!(t.frame_ms, None, "no brackets, no frame time");
        // A non-unit period scales the result.
        let t = timings_from_ticks(0, &["a"], &[1, 1_001], 41.666_668);
        assert!((t.passes[0].1.unwrap() - 0.041_666_668).abs() < 1e-7);
    }

    /// B9t: each pass's end time on the frame origin, so that a compute pass that starts before
    /// the previous render pass's (early-sampled) end is not floored to 0 when durations are
    /// taken as differences. Ticks laid out as Metal samples them.
    #[test]
    fn pass_end_times_are_on_the_frame_origin() {
        let labels = [FRAME_BEGIN, "x", "y", FRAME_END];
        let ticks = [
            1_000, 1_010, // frame begin
            1_100, 900_000, // x (render)
            900_500, 2_900_000, // y (compute)
            2_900_100, 2_900_200, // frame end
        ];
        let t = timings_from_ticks(0, &labels, &ticks, 1.0);
        assert_eq!(t.pass_end_ms.len(), 2, "brackets are not passes");
        assert_eq!(t.pass_end_ms[0].0, "x");
        assert!((t.pass_end_ms[0].1.unwrap() - 0.899).abs() < 1e-6);
        assert_eq!(t.pass_end_ms[1].0, "y");
        assert!((t.pass_end_ms[1].1.unwrap() - 2.899).abs() < 1e-6);
        // Differences of the end times sum to the frame (to the end bracket's few ticks).
        let sum = t.pass_end_ms[1].1.unwrap();
        assert!((sum - t.frame_ms.unwrap()).abs() < 1e-3);
        // Without the opening bracket there is no origin: every end is None, never 0.
        let t = timings_from_ticks(0, &["x", "y"], &ticks[2..6], 1.0);
        assert!(t.pass_end_ms.iter().all(|(_, ms)| ms.is_none()));
    }

    /// Review I5: backwards or unwritten timestamps are not a measurement of 0 ms.
    #[test]
    fn invalid_ticks_are_none_not_zero() {
        let t = timings_from_ticks(
            0,
            &["backwards", "equal", "unwritten", "error", "ok"],
            &[5_000, 4_000, 7, 7, 0, 9, 3, u64::MAX, 10, 20],
            1.0,
        );
        let got: Vec<_> = t.passes.iter().map(|(l, ms)| (*l, ms.is_some())).collect();
        assert_eq!(
            got,
            [
                ("backwards", false),
                ("equal", false),
                ("unwritten", false),
                ("error", false),
                ("ok", true)
            ]
        );
    }

    #[test]
    fn frame_brackets_give_the_frame_time_and_are_not_passes() {
        let t = timings_from_ticks(
            3,
            &[FRAME_BEGIN, "lens", "tonemap", FRAME_END],
            &[
                1_000, 1_500, // begin bracket
                2_000, 9_000, // lens
                9_500, 9_400, // tonemap: end before begin
                9_900, 11_000, // end bracket
            ],
            1.0,
        );
        assert_eq!(
            t.passes.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
            ["lens", "tonemap"]
        );
        // The last pass's bad end is replaced by the closing bracket's begin (9 900 − 9 500).
        assert!((t.passes[1].1.unwrap() - 0.0004).abs() < 1e-9);
        // 11 000 − 1 000 ticks of 1 ns.
        assert!((t.frame_ms.unwrap() - 0.01).abs() < 1e-7);
        // A backwards bracket is no frame time.
        let t = timings_from_ticks(0, &[FRAME_BEGIN, FRAME_END], &[50, 60, 30, 40], 1.0);
        assert_eq!(t.frame_ms, None);
    }

    /// Final review render I1: the tonemap's end tick as C6 logged it in sustained runs (0, or
    /// a stale tick from an earlier frame) is replaced by the closing bracket's begin tick, in
    /// both the span and the end stamp; a valid end is kept; only the pass right before the
    /// closing bracket is patched; with no usable bracket tick it stays unmeasured.
    #[test]
    fn last_pass_end_falls_back_to_the_closing_bracket() {
        let labels = [FRAME_BEGIN, "bloom composite", "tonemap", FRAME_END];
        // Ticks [begin, end] per label, 1 ns each; the frame starts at 10 000 000.
        let frame = |tonemap_end: u64, bloom_end: u64| {
            [
                10_000_000,
                10_000_050, // frame begin
                10_100_000,
                bloom_end, // bloom composite
                10_200_000,
                tonemap_end, // tonemap
                10_550_000,
                10_550_080, // frame end
            ]
        };
        let ms = |t: &GpuTimings, i: usize| t.passes[i].1;
        let end = |t: &GpuTimings, i: usize| t.pass_end_ms[i].1;
        // Valid end: kept (0.3 ms).
        let t = timings_from_ticks(0, &labels, &frame(10_500_000, 10_190_000), 1.0);
        assert!((ms(&t, 1).unwrap() - 0.3).abs() < 1e-6);
        assert!((end(&t, 1).unwrap() - 0.5).abs() < 1e-6);
        // Unwritten (0) and Metal's error value: the bracket's begin, 10 550 000.
        for bad in [0, u64::MAX] {
            let t = timings_from_ticks(1, &labels, &frame(bad, 10_190_000), 1.0);
            assert!((ms(&t, 1).unwrap() - 0.35).abs() < 1e-6, "{bad}");
            assert!((end(&t, 1).unwrap() - 0.55).abs() < 1e-6, "{bad}");
            // The pass before it is untouched.
            assert!((ms(&t, 0).unwrap() - 0.09).abs() < 1e-6);
        }
        // Stale: frame N − 1's bloom-composite end (C6's example), before this frame began.
        let t = timings_from_ticks(2, &labels, &frame(9_190_000, 10_190_000), 1.0);
        assert!((ms(&t, 1).unwrap() - 0.35).abs() < 1e-6);
        assert!((end(&t, 1).unwrap() - 0.55).abs() < 1e-6);
        // An end not after its own begin is stale too.
        let t = timings_from_ticks(3, &labels, &frame(10_200_000, 10_190_000), 1.0);
        assert!((ms(&t, 1).unwrap() - 0.35).abs() < 1e-6);
        // An earlier pass with a bad end is not patched (only the last one is).
        let t = timings_from_ticks(4, &labels, &frame(10_500_000, 0), 1.0);
        assert_eq!(ms(&t, 0), None);
        assert_eq!(end(&t, 0), None);
        // No usable bracket tick: the last pass stays unmeasured, never 0.
        let mut ticks = frame(0, 10_190_000);
        ticks[6] = 0;
        let t = timings_from_ticks(5, &labels, &ticks, 1.0);
        assert_eq!(ms(&t, 1), None);
        assert_eq!(end(&t, 1), None);
        // Without a closing bracket nothing is patched.
        let t = timings_from_ticks(6, &labels[..3], &frame(0, 10_190_000)[..6], 1.0);
        assert_eq!(ms(&t, 1), None);
        // The frame time is unchanged by the patch.
        let t = timings_from_ticks(7, &labels, &frame(0, 10_190_000), 1.0);
        assert!((t.frame_ms.unwrap() - 0.55008).abs() < 1e-6);
    }

    /// F3: a completed frame waits for a later one before its resolve (its last samples are
    /// written late in sustained runs), except in headless runs that force it.
    #[test]
    fn completed_frame_waits_for_a_later_one() {
        assert!(
            !resolve_now(7, 7, false),
            "the newest completed frame waits"
        );
        assert!(resolve_now(6, 7, false));
        assert!(resolve_now(7, 7, true), "headless runs resolve at once");
        assert!(!resolve_now(0, 0, false));
    }

    #[test]
    fn bracket_kernel_validates_and_writes_an_hdr_format_anchor() {
        let m = crate::shader::validate_wgsl(BRACKET_WGSL).unwrap();
        let (_, var) = m
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("anchor"))
            .expect("var anchor");
        assert_eq!(
            var.binding,
            Some(naga::ResourceBinding {
                group: 0,
                binding: 0
            })
        );
        let naga::TypeInner::Image {
            class: naga::ImageClass::Storage { format, access },
            ..
        } = m.types[var.ty].inner
        else {
            panic!("anchor is not a storage texture");
        };
        assert_eq!(format, naga::StorageFormat::Rgba16Float);
        assert_eq!(access, naga::StorageAccess::STORE);
        assert_eq!(ANCHOR_FORMAT, wgpu::TextureFormat::Rgba16Float);
    }
}
