//! GPU pass timing with timestamp queries; degrades to a no-op when `TIMESTAMP_QUERY` is absent.
//!
//! One `QuerySet` holds 2 × `max_passes` timestamps: pass *i* of a frame writes its begin
//! and end times to queries 2*i* and 2*i* + 1. [`PassTimer::resolve`] resolves the queries
//! used this frame into a resolve buffer, copies them into a free buffer of a 3-deep
//! read-back ring and schedules a map of that buffer for when the command buffer is
//! submitted (`map_buffer_on_submit`). [`PassTimer::read_back`] polls the device without
//! blocking and converts every ring buffer whose map has finished, using
//! `queue.get_timestamp_period()`. When all three ring buffers are still in flight the
//! frame's timings are dropped rather than stalling the CPU.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::context::GpuContext;

/// Depth of the read-back ring (frames whose timings can be in flight at once).
const RING_LEN: usize = 3;

/// States of a ring buffer's map, written by its map callback.
const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;
/// The command buffer carrying the map was dropped without being submitted.
const MAP_NEVER_SUBMITTED: u8 = 3;

/// Collects per-pass GPU times. Results lag by at least one frame.
pub struct PassTimer {
    /// `None` when disabled (no `TIMESTAMP_QUERY`, or `max_passes == 0`).
    gpu: Option<TimerGpu>,
    /// The most recent completed frame's (label, ms) pairs.
    last: Vec<(&'static str, f32)>,
}

struct TimerGpu {
    query_set: wgpu::QuerySet,
    /// `QUERY_RESOLVE | COPY_SRC`, 2 × max_passes × 8 bytes.
    resolve_buffer: wgpu::Buffer,
    max_passes: u32,
    /// Nanoseconds per timestamp tick.
    period_ns: f32,
    /// Labels of the passes given timestamp writes so far this frame, in query-pair order.
    frame_labels: Vec<&'static str>,
    ring: [ReadbackSlot; RING_LEN],
    next_seq: u64,
}

struct ReadbackSlot {
    /// `MAP_READ | COPY_DST`, same size as the resolve buffer.
    buffer: wgpu::Buffer,
    labels: Vec<&'static str>,
    /// Frame order, to keep the newest result when several finish together.
    seq: u64,
    /// `None` while free; otherwise the state written by the map callback.
    map: Option<Arc<AtomicU8>>,
}

/// Owned by a map callback. Stores the map result when called; if the callback is dropped
/// without running (its command buffer was never submitted) it stores that instead, so the
/// ring buffer is freed rather than leaked.
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

fn readback_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pass timer read-back"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl TimerGpu {
    /// Reserves the next (begin, end) query pair for `label`, or `None` when the frame is full.
    fn next_pair(&mut self, label: &'static str) -> Option<(u32, u32)> {
        let i = u32::try_from(self.frame_labels.len()).ok()?;
        if i >= self.max_passes {
            return None;
        }
        self.frame_labels.push(label);
        Some((2 * i, 2 * i + 1))
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
        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pass timer"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass timer resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ring = std::array::from_fn(|_| ReadbackSlot {
            buffer: readback_buffer(&gpu.device, bytes),
            labels: Vec::new(),
            seq: 0,
            map: None,
        });
        Self {
            gpu: Some(TimerGpu {
                query_set,
                resolve_buffer,
                max_passes,
                period_ns: gpu.queue.get_timestamp_period(),
                frame_labels: Vec::with_capacity(max_passes as usize),
                ring,
                next_seq: 0,
            }),
            last: Vec::new(),
        }
    }

    /// A timer that records nothing.
    pub fn disabled() -> Self {
        Self {
            gpu: None,
            last: Vec::new(),
        }
    }

    /// Timestamp writes for a compute pass named `label`, or `None` when disabled or full.
    pub fn compute_writes(
        &mut self,
        label: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let g = self.gpu.as_mut()?;
        let (begin, end) = g.next_pair(label)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &g.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Timestamp writes for a render pass named `label`, or `None` when disabled or full.
    pub fn render_writes(
        &mut self,
        label: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let g = self.gpu.as_mut()?;
        let (begin, end) = g.next_pair(label)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &g.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Resolves this frame's queries into the read-back ring.
    ///
    /// Call once per frame after the last timed pass is encoded. The map of the ring buffer
    /// starts when the command buffer is submitted; a frame with no free ring buffer is
    /// skipped.
    pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder) {
        let Some(g) = self.gpu.as_mut() else { return };
        let labels = std::mem::take(&mut g.frame_labels);
        if labels.is_empty() {
            return;
        }
        let Some(slot) = g.ring.iter_mut().find(|s| s.map.is_none()) else {
            return; // every ring buffer is still in flight: drop this frame's timings
        };
        let queries = 2 * labels.len() as u32; // ≤ 2 × max_passes ≤ QUERY_SET_MAX_QUERIES
        let bytes = u64::from(queries) * u64::from(wgpu::QUERY_SIZE);
        enc.resolve_query_set(&g.query_set, 0..queries, &g.resolve_buffer, 0);
        enc.copy_buffer_to_buffer(&g.resolve_buffer, 0, &slot.buffer, 0, bytes);
        let state = Arc::new(AtomicU8::new(MAP_PENDING));
        let signal = MapSignal(Arc::clone(&state));
        enc.map_buffer_on_submit(&slot.buffer, wgpu::MapMode::Read, 0..bytes, move |r| {
            signal.finish(r.is_ok());
        });
        slot.labels = labels;
        slot.seq = g.next_seq;
        g.next_seq += 1;
        slot.map = Some(state);
    }

    /// Non-blocking read-back; results lag by ≥ 1 frame.
    pub fn read_back(&mut self, gpu: &GpuContext) {
        let Some(g) = self.gpu.as_mut() else { return };
        if g.ring.iter().all(|s| s.map.is_none()) {
            return;
        }
        // Never waits: only runs callbacks of maps whose GPU work has already finished.
        if let Err(e) = gpu.device.poll(wgpu::PollType::Poll) {
            log::warn!("pass timer: device poll failed: {e}");
        }
        // The queue completes submissions in order, so the highest `seq` is the newest frame.
        let mut newest: Option<(u64, Vec<(&'static str, f32)>)> = None;
        for slot in &mut g.ring {
            let Some(state) = &slot.map else { continue };
            match state.load(Ordering::Acquire) {
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
                            if newest.as_ref().is_none_or(|(seq, _)| slot.seq > *seq) {
                                newest =
                                    Some((slot.seq, pass_times(&slot.labels, &ticks, g.period_ns)));
                            }
                        }
                        Err(e) => log::warn!("pass timer: mapped range unavailable: {e}"),
                    }
                    slot.buffer.unmap();
                }
                MAP_FAILED => {
                    // A failed map leaves the buffer unusable for another map; replace it.
                    log::warn!("pass timer: read-back map failed; recreating the buffer");
                    slot.buffer = readback_buffer(&gpu.device, slot.buffer.size());
                }
                _ => {} // MAP_NEVER_SUBMITTED: the buffer was never touched
            }
            slot.map = None;
            slot.labels.clear();
        }
        if let Some((_, times)) = newest {
            self.last = times;
        }
    }

    /// The most recent (label, milliseconds) pairs; empty when disabled.
    pub fn last_ms(&self) -> Vec<(&'static str, f32)> {
        self.last.clone()
    }
}

/// Pairs each label with its (begin, end) ticks and converts the difference to milliseconds.
/// An end tick before its begin tick (an unwritten or reordered query) reads as 0 ms.
fn pass_times(labels: &[&'static str], ticks: &[u64], period_ns: f32) -> Vec<(&'static str, f32)> {
    labels
        .iter()
        .zip(ticks.chunks_exact(2))
        .map(|(&label, t)| {
            let ns = t[1].saturating_sub(t[0]) as f64 * f64::from(period_ns);
            (label, (ns * 1e-6) as f32)
        })
        .collect()
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
    }

    #[test]
    fn pass_times_converts_ticks_to_ms() {
        // period 1 ns/tick (Apple M3, docs/research/wgpu30-skeleton/m3-probe-output.txt).
        let t = pass_times(
            &["lens", "stars"],
            &[1_000, 2_501_000, 3_000_000, 3_250_000],
            1.0,
        );
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].0, "lens");
        assert!((t[0].1 - 2.5).abs() < 1e-6);
        assert_eq!(t[1].0, "stars");
        assert!((t[1].1 - 0.25).abs() < 1e-6);
        // A non-unit period scales the result.
        let t = pass_times(&["a"], &[0, 1_000], 41.666_668);
        assert!((t[0].1 - 0.041_666_668).abs() < 1e-7);
    }

    #[test]
    fn pass_times_clamps_backwards_ticks_to_zero() {
        let t = pass_times(&["a"], &[5_000, 4_000], 1.0);
        assert_eq!(t, vec![("a", 0.0)]);
    }
}
