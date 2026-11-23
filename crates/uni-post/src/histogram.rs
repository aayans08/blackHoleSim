//! The exposure meter, pass "histogram" (Task B6; B0 stub).
//!
//! Pass 9 of the M1 frame contract: after the atmospheres, before any sky light. World luminance
//! (Rec. 709 Y) of `hdr` ÷ `scene_pre` [cd/m²] into 256 log₂ bins over [−10, 32] with
//! workgroup-shared atomics, then one reduction (bruop 2019, "Automatic Exposure Using a
//! Luminance Histogram"). Lit-pixel counts are normalised to 1920 × 1080.

use uni_gfx::{FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature};

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

/// Reduces the 256-bin histogram of `total_px` pixels to a [`Meter`]; `None` when nothing is lit.
/// CPU twin of `histogram.wgsl`'s reduction (B6).
pub fn meter_histogram(bins: &[u32; 256], total_px: u32) -> Option<Meter> {
    let _ = (bins, total_px);
    todo!("B6")
}

/// B6's target rule: [`crate::NIGHT_EV100`] below [`METER_MIN_LIT_PX`], max(ev_avg, ev_hi) +
/// `compensation_ev` from [`METER_FULL_LIT_PX`], smoothstep in ln(lit_px) between; clamped to
/// [[`crate::EV100_MIN`], [`crate::EV100_MAX`]].
pub fn target_ev(m: Option<Meter>, compensation_ev: f32) -> f32 {
    let _ = (m, compensation_ev);
    todo!("B6")
}

/// The meter pass (`RenderFeature` named "histogram"), made by [`crate::auto_exposure`].
pub struct ExposureMeter {
    _private: (),
}

impl RenderFeature for ExposureMeter {
    fn name(&self) -> &'static str {
        "histogram"
    }

    fn prepare(&mut self, _gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {}

    fn encode(
        &mut self,
        _gpu: &GpuContext,
        _enc: &mut wgpu::CommandEncoder,
        _frame: &FrameBindings<'_>,
        _targets: &FrameTargets,
        _output: &wgpu::TextureView,
        _timer: &mut PassTimer,
    ) {
    }
}
