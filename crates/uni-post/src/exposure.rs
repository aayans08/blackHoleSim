//! Auto-exposure and the dual star exposure: the resolve pass "exposure" and its settings (Task
//! B6).
//!
//! Pass 1 of the M1 frame contract: scene EV = adapted (Auto) or `manual.scene_ev100` (Manual);
//! star EV = `eye_star_ev100` while [`ExposureSettings::eye_adapted_in_effect`], else the scene
//! EV; written into the renderer's exposure buffer through a storage binding in the pass's own
//! bind group, importing only `common/exposure.wgsl`, never group 0 (review m14). The meter is
//! [`crate::histogram`].

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::Context as _;
use uni_gfx::{
    Exposure, ExposureUniform, FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer,
    RenderFeature, ShaderLib, SharedInput,
};

use crate::histogram::{ExposureMeter, Meter, target_ev};

/// How the scene EV is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExposureMode {
    /// Metered (B6's rule) and adapted over time.
    Auto,
    /// [`ExposureSettings::manual`] (the HUD sliders, bookmarks, and the display-referred M0 debug
    /// grid).
    Manual,
}

/// The exposure controls the app sets every frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureSettings {
    pub mode: ExposureMode,
    /// The one manual source (review m13): the scene EV in Manual mode, and the app copies it
    /// into `FrameCtx::exposure` too. The resolve does not read `manual.star_ev100`: the star EV
    /// is [`Self::eye_star_ev100`] while eye-adapted stars are in effect, else the scene EV.
    pub manual: Exposure,
    /// Added to the metered target [EV].
    pub compensation_ev: f32,
    /// The "Eye-adapted stars" option: stars and point sources at [`Self::eye_star_ev100`].
    pub eye_adapted: bool,
    /// Star EV100 while eye-adapted stars are in effect.
    pub eye_star_ev100: f32,
    /// Set by the app from `uni_atmo::camera_in_daylight`: a daylit sky hides the stars, so the
    /// eye-adapted exposure is off (review m12).
    pub daylight: bool,
    /// Adaptation time constants [s]: `[0]` while the scene brightens (the target EV is above
    /// the current one: light adaptation, the faster), `[1]` while it darkens (dark adaptation).
    /// [`DEFAULT_ADAPT_TAU_S`] by default.
    pub adapt_tau_s: [f32; 2],
    /// Jump straight to the target (τ = 0), for captures.
    pub settle: bool,
}

impl ExposureSettings {
    /// True when the eye-adapted star exposure is in effect: `eye_adapted && !daylight` (M1 frame
    /// contract). It drives the resolve pass and the HUD's amber "Eye-adapted stars" label
    /// (review m13), not `FrameCtx::exposure`.
    pub fn eye_adapted_in_effect(&self) -> bool {
        self.eye_adapted && !self.daylight
    }
}

/// Dark-adapted target with no metered world light [EV100]. Equals
/// `uni_sky::DEFAULT_STAR_EV100` (−4, the M0 gate's star sweep; checked by an app test, B9).
pub const NIGHT_EV100: f32 = -4.0;
/// Lowest exposure the resolve writes [EV100]; equals the start of `uni_app::scene::EV100_LIMITS`
/// (review m5: 2^EV never overflows; checked by an app test, B9).
pub const EV100_MIN: f32 = -30.0;
/// Highest exposure the resolve writes [EV100]; equals the end of `uni_app::scene::EV100_LIMITS`.
pub const EV100_MAX: f32 = 40.0;

/// Default [`ExposureSettings::adapt_tau_s`] [s]: 0.6 s while the scene brightens, 1.2 s while it
/// darkens (plan B6). The exponential form and the asymmetry (light adaptation is faster than
/// dark adaptation) follow Pattanaik, Tumblin, Yee & Greenberg 2000, "Time-Dependent Visual
/// Adaptation for Fast Realistic Image Display" (SIGGRAPH 2000), and Krawczyk, Myszkowski &
/// Seidel 2005, "Perceptual Effects in Real-time Tone Mapping" (SCCG 2005), whose per-frame step
/// L_a += (L − L_a)(1 − e^(−Δt/τ)) [`adapt`] applies in EV. The values themselves are the plan's
/// choice: slower than Krawczyk's cone and rod constants (0.1 s and 0.4 s), so the exposure does
/// not pump visibly while flying, and far faster than real dark adaptation (minutes).
pub const DEFAULT_ADAPT_TAU_S: [f32; 2] = [0.6, 1.2];

/// `ev` clamped to [[`EV100_MIN`], [`EV100_MAX`]]; NaN becomes [`NIGHT_EV100`].
pub fn clamp_ev100(ev: f32) -> f32 {
    if ev.is_nan() {
        NIGHT_EV100
    } else {
        ev.clamp(EV100_MIN, EV100_MAX)
    }
}

/// One adaptation step: ev_prev + (ev_target − ev_prev)(1 − e^(−dt/τ)), with τ = `tau_s[0]`
/// when the scene brightens (`ev_target > ev_prev`) and `tau_s[1]` when it darkens (see
/// [`DEFAULT_ADAPT_TAU_S`] for the source). τ ≤ 0 jumps to the target (settle); dt ≤ 0 holds.
pub fn adapt(ev_prev: f32, ev_target: f32, dt_s: f32, tau_s: [f32; 2]) -> f32 {
    let tau = if ev_target > ev_prev {
        tau_s[0]
    } else {
        tau_s[1]
    };
    // `!(x > 0)` is also true for NaN.
    if !(tau > 0.0) {
        return ev_target;
    }
    if !(dt_s > 0.0) {
        return ev_prev;
    }
    ev_prev + (ev_target - ev_prev) * (1.0 - (-dt_s / tau).exp())
}

/// The resolve pass's state step, CPU twin of `cs_resolve` in `exposure.wgsl`: the adapted scene
/// EV after one frame, from the previous one (`prev`, `None` before the first) and the meter's
/// latest `reading` (`None` before the meter has read a frame; `Some(None)` when it read nothing
/// lit).
///
/// Manual: the manual scene EV (so switching to Auto starts from it). Auto with a reading: adapt
/// from `prev` (or the manual EV) toward [`crate::target_ev`] over `dt_s`, τ = 0 with `settle`.
/// Auto without a reading: `prev`, held. Non-finite compensation, τ or dt count as 0.
pub fn next_adapted_ev(
    s: &ExposureSettings,
    reading: Option<Option<Meter>>,
    prev: Option<f32>,
    dt_s: f32,
) -> Option<f32> {
    let p = ResolveParamsGpu::new(s, dt_s);
    let manual_ev = p.ev[0];
    if s.mode == ExposureMode::Manual {
        return Some(manual_ev);
    }
    let Some(meter) = reading else {
        return prev;
    };
    let from = prev.unwrap_or(manual_ev);
    let goal = target_ev(meter, p.ev[2]);
    Some(clamp_ev100(adapt(
        from,
        goal,
        p.mode[3],
        [p.tau[0], p.tau[1]],
    )))
}

/// The exposure uniform the resolve pass writes for `s`, with the adapted scene EV (Auto) or
/// `None` (Manual, or no meter reading yet). CPU twin of `exposure.wgsl`. Every EV is clamped
/// with [`clamp_ev100`].
pub fn resolve_exposure(s: &ExposureSettings, adapted_scene_ev: Option<f32>) -> ExposureUniform {
    let scene = match (s.mode, adapted_scene_ev) {
        (ExposureMode::Auto, Some(ev)) => ev,
        _ => s.manual.scene_ev100,
    };
    let scene_ev100 = clamp_ev100(scene);
    let star_ev100 = if s.eye_adapted_in_effect() {
        clamp_ev100(s.eye_star_ev100)
    } else {
        scene_ev100
    };
    ExposureUniform::from_exposure(Exposure {
        scene_ev100,
        star_ev100,
    })
}

/// Twin of `struct ExposureState` in `exposure.wgsl` and `histogram.wgsl`: the auto-exposure
/// state that lives on the GPU across frames (32 bytes, two vec4s).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ExposureStateGpu {
    /// Written by the meter: lit_px (normalised to 1920 × 1080), ev_avg, ev_hi, 1 once it has
    /// read a frame (0 before).
    pub(crate) meter: [f32; 4],
    /// Written by the resolve: the adapted scene EV100, 1 once it holds a value (0 before), 0, 0.
    pub(crate) adapted: [f32; 4],
}

/// Twin of `struct ResolveParams` in `exposure.wgsl`: the settings and the frame time the
/// resolve reads (48 bytes, three vec4s; flags as 0/1 floats).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ResolveParamsGpu {
    /// 1 = Manual (0 = Auto), 1 = eye-adapted stars in effect, 0, dt [s].
    pub(crate) mode: [f32; 4],
    /// Manual scene EV100, eye-adapted star EV100 (both clamped), compensation [EV], 0.
    pub(crate) ev: [f32; 4],
    /// τ while the scene brightens, τ while it darkens [s] (0 with settle), 0, 0.
    pub(crate) tau: [f32; 4],
}

impl ResolveParamsGpu {
    /// The resolve's inputs for settings `s` and a frame of `dt_s` seconds. EVs are clamped
    /// ([`clamp_ev100`]); a non-finite compensation, τ or dt counts as 0 and a negative τ or dt
    /// as 0 too, so the GPU never sees a NaN (Metal's fast math does not keep NaN semantics).
    pub(crate) fn new(s: &ExposureSettings, dt_s: f32) -> Self {
        let finite_or_0 = |x: f32| if x.is_finite() { x } else { 0.0 };
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        let tau = if s.settle {
            [0.0; 2]
        } else {
            s.adapt_tau_s.map(|t| finite_or_0(t).max(0.0))
        };
        Self {
            mode: [
                flag(s.mode == ExposureMode::Manual),
                flag(s.eye_adapted_in_effect()),
                0.0,
                finite_or_0(dt_s).max(0.0),
            ],
            ev: [
                clamp_ev100(s.manual.scene_ev100),
                clamp_ev100(s.eye_star_ev100),
                finite_or_0(s.compensation_ev),
                0.0,
            ],
            tau: [tau[0], tau[1], 0.0, 0.0],
        }
    }
}

/// Feature name, also the GPU timing label.
const NAME: &str = "exposure";
/// The resolve's own bind group (group 0 stays empty: review m14).
const RESOLVE_GROUP: u32 = 1;

/// The resolve pass (`RenderFeature` named "exposure"), made by [`auto_exposure`]. Push it first
/// (M1 frame contract, pass 1): it overwrites the exposure buffer the `Renderer` wrote from
/// `FrameCtx::exposure`, before any feature reads it.
pub struct ExposureResolve {
    pipeline: wgpu::ComputePipeline,
    /// [`RESOLVE_GROUP`]: the renderer's exposure buffer (storage), the params, the state.
    bind_group: wgpu::BindGroup,
    /// [`ResolveParamsGpu`], written in `prepare`.
    params: wgpu::Buffer,
    /// The renderer's exposure buffer, copied into the read-back ring after the resolve.
    exposure: wgpu::Buffer,
    /// The auto-exposure state ([`ExposureStateGpu`]) shared with the meter.
    pub(crate) state: wgpu::Buffer,
    settings: SharedInput<ExposureSettings>,
    readback: ExposureReadback,
}

impl RenderFeature for ExposureResolve {
    fn name(&self) -> &'static str {
        NAME
    }

    /// Uploads this frame's settings and `dt_real_s`, and collects finished read-backs.
    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, _targets: &FrameTargets) {
        let params = ResolveParamsGpu::new(&self.settings.get(), ctx.dt_real_s as f32);
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        if self.readback.in_flight() {
            // Never waits: runs the map callbacks of frames that have finished. A failed poll
            // only delays the read-back.
            let _ = gpu.device.poll(wgpu::PollType::Poll);
        }
        self.readback.collect();
    }

    /// One single-thread dispatch, then a copy of the exposure buffer into the read-back ring.
    fn encode(
        &mut self,
        _gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        _frame: &FrameBindings<'_>,
        _targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(NAME),
                timestamp_writes: timer.compute_writes(NAME),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(RESOLVE_GROUP, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.readback.request(enc, &self.exposure);
    }
}

/// Read-back buffers in the ring: frames whose exposure can be in flight at once.
const READBACK_SLOTS: usize = 3;
/// Bytes read back per frame: the whole [`ExposureUniform`].
const EXPOSURE_BYTES: u64 = std::mem::size_of::<ExposureUniform>() as u64;

/// States of a slot's map, written by its callback.
const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;
/// The command buffer carrying the map was dropped without being submitted.
const MAP_DROPPED: u8 = 3;

/// Owned by a map callback: stores the map's result, or [`MAP_DROPPED`] if the callback is
/// dropped without running, so the slot is freed rather than leaked (as `PassTimer` does).
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
            MAP_DROPPED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

/// One `MAP_READ` buffer of the ring, with the frame order and map state while in flight.
struct ReadbackSlot {
    buffer: wgpu::Buffer,
    in_flight: Option<(u64, Arc<AtomicU8>)>,
}

/// The ring and the newest exposure it has read.
struct ReadbackRing {
    device: wgpu::Device,
    slots: Vec<ReadbackSlot>,
    next_seq: u64,
    /// (frame order, [scene, star] EV100) of the newest frame read.
    latest: Option<(u64, [f32; 2])>,
}

fn readback_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exposure read-back"),
        size: EXPOSURE_BYTES,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl ReadbackRing {
    /// Reads every slot whose map has finished, keeps the newest frame's EVs and frees the slot.
    fn collect(&mut self) {
        for slot in &mut self.slots {
            let Some((seq, state)) = &slot.in_flight else {
                continue;
            };
            let seq = *seq;
            match state.load(Ordering::Acquire) {
                MAP_PENDING => continue,
                MAP_READY => {
                    if let Ok(view) = slot.buffer.get_mapped_range(..) {
                        let u: ExposureUniform = bytemuck::pod_read_unaligned(&view[..]);
                        drop(view);
                        if self.latest.is_none_or(|(newest, _)| seq > newest) {
                            self.latest = Some((seq, [u.ev100[0], u.ev100[1]]));
                        }
                    }
                    slot.buffer.unmap();
                }
                // A failed map leaves the buffer unusable for another map; replace it.
                MAP_FAILED => slot.buffer = readback_buffer(&self.device),
                _ => {} // MAP_DROPPED: never submitted
            }
            slot.in_flight = None;
        }
    }

    /// Copies `exposure` into a free slot and maps it when the frame's work finishes; with no
    /// free slot the frame is skipped rather than stalling.
    fn request(&mut self, enc: &mut wgpu::CommandEncoder, exposure: &wgpu::Buffer) {
        let Some(slot) = self.slots.iter_mut().find(|s| s.in_flight.is_none()) else {
            return;
        };
        enc.copy_buffer_to_buffer(exposure, 0, &slot.buffer, 0, EXPOSURE_BYTES);
        let state = Arc::new(AtomicU8::new(MAP_PENDING));
        let signal = MapSignal(Arc::clone(&state));
        enc.map_buffer_on_submit(&slot.buffer, wgpu::MapMode::Read, .., move |r| {
            signal.finish(r.is_ok())
        });
        slot.in_flight = Some((self.next_seq, state));
        self.next_seq += 1;
    }
}

/// The CPU's view of the exposures the GPU resolved, read back through a ring of mapped buffers
/// (like `PassTimer`); clones share it.
#[derive(Clone)]
pub struct ExposureReadback {
    ring: Arc<Mutex<ReadbackRing>>,
}

impl ExposureReadback {
    fn new(device: &wgpu::Device) -> Self {
        let slots = (0..READBACK_SLOTS)
            .map(|_| ReadbackSlot {
                buffer: readback_buffer(device),
                in_flight: None,
            })
            .collect();
        Self {
            ring: Arc::new(Mutex::new(ReadbackRing {
                device: device.clone(),
                slots,
                next_seq: 0,
                latest: None,
            })),
        }
    }

    /// The ring, even if a holder panicked: each update replaces whole values.
    fn lock(&self) -> MutexGuard<'_, ReadbackRing> {
        self.ring.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn in_flight(&self) -> bool {
        self.lock().slots.iter().any(|s| s.in_flight.is_some())
    }

    fn collect(&self) {
        self.lock().collect();
    }

    fn request(&self, enc: &mut wgpu::CommandEncoder, exposure: &wgpu::Buffer) {
        let mut ring = self.lock();
        ring.collect();
        ring.request(enc, exposure);
    }

    /// The latest read-back [scene, star] EV100, at least one frame old; `None` before the first.
    /// The map callbacks run when the device is polled (each frame's `prepare` of the resolve
    /// does, as does any submit or wait).
    pub fn last_ev100(&self) -> Option<[f32; 2]> {
        let mut ring = self.lock();
        ring.collect();
        ring.latest.map(|(_, ev)| ev)
    }
}

/// Builds the auto-exposure passes: the resolve (push first) and the meter (push after the
/// atmosphere pass), and the read-back. `exposure_buffer` is `Renderer::exposure_buffer()`;
/// `camera_bgl` is `Renderer::camera_bind_group_layout()` (the meter binds group 0; the resolve
/// never does).
pub fn auto_exposure(
    gpu: &GpuContext,
    lib: &mut ShaderLib,
    camera_bgl: &wgpu::BindGroupLayout,
    exposure_buffer: &wgpu::Buffer,
    settings: SharedInput<ExposureSettings>,
) -> anyhow::Result<(ExposureResolve, ExposureMeter, ExposureReadback)> {
    crate::register_shaders(lib);
    let device = &gpu.device;
    let module = lib
        .module(device, crate::EXPOSURE_WGSL)
        .context("compiling the exposure resolve shader")?;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let buffer = |label: &str, size: usize, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        })
    };
    // Zero-initialised by wgpu: no meter reading, no adapted EV.
    let state = buffer(
        "exposure state",
        std::mem::size_of::<ExposureStateGpu>(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let params = buffer(
        "exposure params",
        std::mem::size_of::<ResolveParamsGpu>(),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("exposure resolve bgl"),
        entries: &resolve_layout_entries(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("exposure resolve layout"),
        // Group 0 stays empty: the resolve never binds the renderer's group (review m14).
        bind_group_layouts: &[None, Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cs_resolve"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("cs_resolve"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("exposure resolve bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: exposure_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.as_entire_binding(),
            },
        ],
    });
    let readback = ExposureReadback::new(device);
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("creating the exposure resolve pipeline: {err}");
    }
    let meter = ExposureMeter::new(gpu, lib, camera_bgl, &state)?;
    let resolve = ExposureResolve {
        pipeline,
        bind_group,
        params,
        exposure: exposure_buffer.clone(),
        state,
        settings,
        readback: readback.clone(),
    };
    Ok((resolve, meter, readback))
}

/// [`RESOLVE_GROUP`]: binding 0 the renderer's exposure buffer as storage (written), 1 the
/// params, 2 the auto-exposure state.
fn resolve_layout_entries() -> [wgpu::BindGroupLayoutEntry; 3] {
    let entry =
        |binding: u32, ty: wgpu::BufferBindingType, size: usize| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size as u64),
            },
            count: None,
        };
    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    [
        entry(0, storage, std::mem::size_of::<ExposureUniform>()),
        entry(
            1,
            wgpu::BufferBindingType::Uniform,
            std::mem::size_of::<ResolveParamsGpu>(),
        ),
        entry(2, storage, std::mem::size_of::<ExposureStateGpu>()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::{
        HIGHLIGHT_PERCENTILE, HIGHLIGHT_PRE, METER_BIN_EV, METER_BIN_STEPS, METER_BINS,
        METER_FULL_LIT_PX, METER_LOG2_MAX, METER_LOG2_MIN, METER_MIN_CD_M2, METER_MIN_LIT_PX,
        METER_REFERENCE_PX, MeterBinsGpu, meter_bin, target_ev,
    };
    use crate::{BLOOM_INPUT_MAX, BLOOM_WGSL, EXPOSURE_WGSL, HISTOGRAM_WGSL, TONEMAP_SHADER};
    use core::mem::{offset_of, size_of};
    use uni_gfx::{exposure_scale, validate_wgsl};
    use wgpu::naga;

    fn settings(mode: ExposureMode) -> ExposureSettings {
        ExposureSettings {
            mode,
            manual: Exposure {
                scene_ev100: 9.0,
                star_ev100: NIGHT_EV100,
            },
            compensation_ev: 0.0,
            eye_adapted: true,
            eye_star_ev100: NIGHT_EV100,
            daylight: false,
            adapt_tau_s: DEFAULT_ADAPT_TAU_S,
            settle: false,
        }
    }

    fn lit(ev_avg: f32) -> Option<Meter> {
        Some(Meter {
            lit_px: METER_REFERENCE_PX,
            ev_avg,
            ev_hi: ev_avg - 5.0,
        })
    }

    #[test]
    fn adapt_converges_and_settles() {
        let dt = 1.0 / 60.0;
        let tau = DEFAULT_ADAPT_TAU_S;
        // The scene brightens (target EV above): τ = 0.6 s, so 95 % after 1.8 s = 3τ.
        let mut ev = 0.0;
        for _ in 0..108 {
            ev = adapt(ev, 10.0, dt, tau);
        }
        let done = ev / 10.0;
        assert!(done >= 0.95, "{done}");
        assert!((done - (1.0 - (-3.0f32).exp())).abs() < 1e-3, "{done}");
        // The step is exact for any dt: one 1.8 s step lands at the same place.
        assert!((adapt(0.0, 10.0, 1.8, tau) - ev).abs() < 1e-3);
        // The scene darkens (target EV below): τ = 1.2 s, 1 − e^−1.5 = 78 % after 1.8 s.
        let mut ev = 0.0;
        for _ in 0..108 {
            ev = adapt(ev, -10.0, dt, tau);
        }
        assert!((ev / -10.0 - (1.0 - (-1.5f32).exp())).abs() < 1e-3, "{ev}");
        // Settle: τ = 0 reaches the target in one step; dt = 0 holds.
        assert_eq!(adapt(3.0, 12.0, dt, [0.0, 0.0]), 12.0);
        assert_eq!(adapt(3.0, 12.0, 0.0, tau), 3.0);
        assert_eq!(adapt(3.0, 12.0, f32::NAN, tau), 3.0);
        // Through the resolve's step: settle jumps to the metered target.
        let mut s = settings(ExposureMode::Auto);
        s.settle = true;
        let t = target_ev(lit(14.0), 0.0);
        assert_eq!(next_adapted_ev(&s, Some(lit(14.0)), Some(3.0), dt), Some(t));
        // Without settle it moves by 1 − e^(−dt/τ) of the way.
        s.settle = false;
        let one = next_adapted_ev(&s, Some(lit(14.0)), Some(3.0), dt).unwrap();
        assert!((one - adapt(3.0, t, dt, tau)).abs() < 1e-6);
        // No previous EV: it starts from the manual one; no reading: it holds.
        let first = next_adapted_ev(&s, Some(lit(14.0)), None, dt).unwrap();
        assert!((first - adapt(9.0, t, dt, tau)).abs() < 1e-6);
        assert_eq!(next_adapted_ev(&s, None, Some(5.0), dt), Some(5.0));
        assert_eq!(next_adapted_ev(&s, None, None, dt), None);
        // Nothing lit: toward NIGHT.
        let night = next_adapted_ev(&s, Some(None), Some(NIGHT_EV100), dt).unwrap();
        assert_eq!(night, NIGHT_EV100);
        // Manual: the manual EV, whatever was adapted.
        let m = settings(ExposureMode::Manual);
        assert_eq!(
            next_adapted_ev(&m, Some(lit(14.0)), Some(3.0), dt),
            Some(9.0)
        );
    }

    #[test]
    fn resolve_eye_adapted_vs_strict() {
        let mut s = settings(ExposureMode::Auto);
        // Eye-adapted: stars at eye_star_ev100, the scene at the adapted EV.
        let u = resolve_exposure(&s, Some(13.0));
        assert_eq!(
            u.pre,
            [exposure_scale(13.0), exposure_scale(NIGHT_EV100), 0.0, 0.0]
        );
        assert_eq!(u.ev100, [13.0, NIGHT_EV100, 0.0, 0.0]);
        // Strict: stars follow the scene.
        s.eye_adapted = false;
        let u = resolve_exposure(&s, Some(13.0));
        assert_eq!(u.pre[1], u.pre[0]);
        assert_eq!(u.ev100, [13.0, 13.0, 0.0, 0.0]);
        // Auto with no reading yet, and Manual: the manual scene EV.
        assert_eq!(resolve_exposure(&s, None).ev100[0], 9.0);
        let m = settings(ExposureMode::Manual);
        let u = resolve_exposure(&m, Some(13.0));
        assert_eq!(u.ev100, [9.0, NIGHT_EV100, 0.0, 0.0]);
        assert_eq!(u.pre[0], exposure_scale(9.0));
    }

    #[test]
    fn daylight_disables_eye_adapted() {
        // Review Focus 6, review m12: a daylit sky hides the stars.
        let mut s = settings(ExposureMode::Auto);
        s.daylight = true;
        assert!(!s.eye_adapted_in_effect());
        let u = resolve_exposure(&s, Some(14.0));
        assert_eq!(u.pre[1], u.pre[0]);
        assert_eq!(u.ev100, [14.0, 14.0, 0.0, 0.0]);
        let mut m = settings(ExposureMode::Manual);
        m.daylight = true;
        let u = resolve_exposure(&m, None);
        assert_eq!(u.ev100, [9.0, 9.0, 0.0, 0.0]);
    }

    #[test]
    fn ev_clamped_to_limits() {
        // Review m3.
        let mut m = settings(ExposureMode::Manual);
        m.manual.scene_ev100 = 100.0;
        m.eye_star_ev100 = 100.0;
        assert_eq!(resolve_exposure(&m, None).ev100, [40.0, 40.0, 0.0, 0.0]);
        m.manual.scene_ev100 = -100.0;
        m.eye_star_ev100 = -100.0;
        let u = resolve_exposure(&m, None);
        assert_eq!(u.ev100, [-30.0, -30.0, 0.0, 0.0]);
        assert!(u.pre.iter().all(|p| p.is_finite()));
        assert_eq!(
            resolve_exposure(&settings(ExposureMode::Auto), Some(100.0)).ev100[0],
            40.0
        );
        m.manual.scene_ev100 = f32::NAN;
        assert_eq!(resolve_exposure(&m, None).ev100[0], NIGHT_EV100);
        assert_eq!(
            (clamp_ev100(f32::INFINITY), clamp_ev100(f32::NEG_INFINITY)),
            (40.0, -30.0)
        );
        // The target: a Meter with ev_avg 50 → 40.
        assert_eq!(target_ev(lit(50.0), 0.0), EV100_MAX);
        assert_eq!(target_ev(lit(-50.0), 0.0), EV100_MIN);
        // 1e12 cd/m² lands in bin 255 (2^32 is the top edge), not out of range.
        assert_eq!(meter_bin(1e12).map(|b| b.0), Some(255));
        assert_eq!(meter_bin(f32::MAX).map(|b| b.0), Some(255));
        assert_eq!(meter_bin(4.0e9).map(|b| b.0), Some(255));
    }

    /// The composed module of `path`, validated.
    fn module(path: &str) -> naga::Module {
        let mut lib = uni_gfx::ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let src = lib.compose(path).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{path}: {e}\n{src}"))
    }

    /// (size, [(member, offset)]) of WGSL struct `name` in `m`.
    fn wgsl_struct(m: &naga::Module, name: &str) -> (usize, Vec<(String, usize)>) {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("struct {name}"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let members = members
            .iter()
            .map(|mb| (mb.name.clone().unwrap_or_default(), mb.offset as usize))
            .collect();
        (layouter[handle].size as usize, members)
    }

    fn named(pairs: &[(&str, usize)]) -> Vec<(String, usize)> {
        pairs.iter().map(|&(n, o)| (n.to_owned(), o)).collect()
    }

    /// The uniform-layout rule (Global Constraints): every GPU struct's size and offsets match
    /// the WGSL twin naga lays out.
    #[test]
    fn post_layouts_match_wgsl() {
        let state = (
            size_of::<ExposureStateGpu>(),
            named(&[
                ("meter", offset_of!(ExposureStateGpu, meter)),
                ("adapted", offset_of!(ExposureStateGpu, adapted)),
            ]),
        );
        assert_eq!(state.0, 32);
        assert_eq!(wgsl_struct(&module(EXPOSURE_WGSL), "ExposureState"), state);
        assert_eq!(wgsl_struct(&module(HISTOGRAM_WGSL), "ExposureState"), state);

        let params = (
            size_of::<ResolveParamsGpu>(),
            named(&[
                ("mode", offset_of!(ResolveParamsGpu, mode)),
                ("ev", offset_of!(ResolveParamsGpu, ev)),
                ("tau", offset_of!(ResolveParamsGpu, tau)),
            ]),
        );
        assert_eq!(params.0, 48);
        assert_eq!(wgsl_struct(&module(EXPOSURE_WGSL), "ResolveParams"), params);

        let bins = (
            size_of::<MeterBinsGpu>(),
            named(&[
                ("counts", offset_of!(MeterBinsGpu, counts)),
                ("steps", offset_of!(MeterBinsGpu, steps)),
            ]),
        );
        assert_eq!(bins.0, 2 * 4 * METER_BINS);
        assert_eq!(wgsl_struct(&module(HISTOGRAM_WGSL), "MeterBins"), bins);

        let bloom = (
            size_of::<crate::bloom::BloomParamsGpu>(),
            named(&[("scale", offset_of!(crate::bloom::BloomParamsGpu, scale))]),
        );
        assert_eq!(bloom.0, 16);
        assert_eq!(wgsl_struct(&module(BLOOM_WGSL), "BloomParams"), bloom);
    }

    /// The scalar value of WGSL constant `name` (an f32, u32 or i32 literal) as f64.
    fn wgsl_const(m: &naga::Module, name: &str) -> f64 {
        let (_, c) = m
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("const {name}"));
        match &m.global_expressions[c.init] {
            naga::Expression::Literal(naga::Literal::F32(v)) => f64::from(*v),
            naga::Expression::Literal(naga::Literal::U32(v)) => f64::from(*v),
            naga::Expression::Literal(naga::Literal::I32(v)) => f64::from(*v),
            other => panic!("const {name}: {other:?}"),
        }
    }

    /// Every uni-post Wave B shader validates with the entry points its pipelines name; the
    /// resolve imports only common/exposure.wgsl and binds no group-0 resource (review m14); and
    /// every constant shared with Rust matches its CPU twin.
    #[test]
    fn post_shaders_validate() {
        let stages = |m: &naga::Module| {
            m.entry_points
                .iter()
                .map(|e| (e.name.clone(), e.stage))
                .collect::<Vec<_>>()
        };
        use naga::ShaderStage::{Compute, Fragment, Vertex};
        let exposure = module(EXPOSURE_WGSL);
        assert_eq!(stages(&exposure), [("cs_resolve".to_owned(), Compute)]);
        let histogram = module(HISTOGRAM_WGSL);
        assert_eq!(
            stages(&histogram),
            [
                ("cs_histogram".to_owned(), Compute),
                ("cs_reduce".to_owned(), Compute)
            ]
        );
        let bloom = module(BLOOM_WGSL);
        for (name, stage) in [
            ("cs_down_first", Compute),
            ("cs_down", Compute),
            ("cs_up", Compute),
            ("vs_composite", Vertex),
            ("fs_composite", Fragment),
        ] {
            assert!(stages(&bloom).contains(&(name.to_owned(), stage)), "{name}");
        }

        // The resolve: common/exposure.wgsl and itself only, and nothing at group 0.
        let mut lib = uni_gfx::ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let src = lib.compose(EXPOSURE_WGSL).unwrap();
        let files: Vec<&str> = src
            .lines()
            .filter_map(|l| l.strip_prefix("// ---- ")?.strip_suffix(" ----"))
            .collect();
        assert_eq!(files, ["common/exposure.wgsl", EXPOSURE_WGSL]);
        assert!(
            exposure
                .global_variables
                .iter()
                .filter_map(|(_, v)| v.binding.as_ref())
                .all(|b| b.group == 1)
        );

        // Constants shared with Rust.
        let close = |m: &naga::Module, name: &str, cpu: f64| {
            let w = wgsl_const(m, name);
            // Decimal → f64 → f32 (naga) and decimal → f32 (rustc) may differ by one ulp.
            assert!(
                (w - cpu).abs() <= cpu.abs() * 2.5e-7,
                "{name}: {w} vs {cpu}"
            );
        };
        for (name, cpu) in [
            ("NIGHT_EV100", NIGHT_EV100),
            ("EV100_MIN", EV100_MIN),
            ("EV100_MAX", EV100_MAX),
            ("METER_MIN_LIT_PX", METER_MIN_LIT_PX),
            ("METER_FULL_LIT_PX", METER_FULL_LIT_PX),
        ] {
            close(&exposure, name, f64::from(cpu));
        }
        for (name, cpu) in [
            ("METER_MIN_CD_M2", METER_MIN_CD_M2),
            ("METER_LOG2_MIN", METER_LOG2_MIN),
            ("METER_LOG2_MAX", METER_LOG2_MAX),
            ("METER_BIN_EV", METER_BIN_EV),
            ("METER_REFERENCE_PX", METER_REFERENCE_PX),
            ("HIGHLIGHT_PERCENTILE", HIGHLIGHT_PERCENTILE),
            ("HIGHLIGHT_PRE", HIGHLIGHT_PRE),
            ("AVERAGE_BAND_LO", crate::histogram::AVERAGE_BAND[0]),
            ("AVERAGE_BAND_HI", crate::histogram::AVERAGE_BAND[1]),
            ("LOG2_S_OVER_K", crate::histogram::LOG2_S_OVER_K),
            ("BIN_COORD_MARGIN", crate::histogram::BIN_COORD_MARGIN),
        ] {
            close(&histogram, name, f64::from(cpu));
        }
        close(&histogram, "METER_BINS", METER_BINS as f64);
        close(&histogram, "METER_BIN_STEPS", f64::from(METER_BIN_STEPS));
        close(&bloom, "BLOOM_INPUT_MAX", f64::from(BLOOM_INPUT_MAX));
        // Review deferral T5: the tonemap's TPDF dither spans ±1 LSB of the 8-bit output.
        close(&module(TONEMAP_SHADER), "DITHER_AMPLITUDE", 1.0 / 255.0);
    }
}
