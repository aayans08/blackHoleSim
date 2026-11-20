//! GPU pass timing with timestamp queries; degrades to a no-op when `TIMESTAMP_QUERY` is absent.

use crate::context::GpuContext;

/// Collects per-pass GPU times. Results lag by at least one frame.
pub struct PassTimer {
    // Placeholder; M0 Task 4 defines the query set and read-back ring.
    _private: (),
}

impl PassTimer {
    /// A timer for up to `max_passes` passes per frame (disabled if the device lacks timestamps).
    pub fn new(gpu: &GpuContext, max_passes: u32) -> Self {
        todo!()
    }

    /// A timer that records nothing.
    pub fn disabled() -> Self {
        todo!()
    }

    /// Timestamp writes for a compute pass named `label`, or `None` when disabled or full.
    pub fn compute_writes(
        &mut self,
        label: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        todo!()
    }

    /// Timestamp writes for a render pass named `label`, or `None` when disabled or full.
    pub fn render_writes(
        &mut self,
        label: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        todo!()
    }

    /// Resolves this frame's queries into the read-back ring.
    pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder) {
        todo!()
    }

    /// Non-blocking read-back; results lag by ≥ 1 frame.
    pub fn read_back(&mut self, gpu: &GpuContext) {
        todo!()
    }

    /// The most recent (label, milliseconds) pairs; empty when disabled.
    pub fn last_ms(&self) -> Vec<(&'static str, f32)> {
        todo!()
    }
}
