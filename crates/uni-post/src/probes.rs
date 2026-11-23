//! GPU probes of uni-post's WGSL against its CPU twins, run by `universe --selftest` (Task B6).
//!
//! Each runs the real feature (its pipelines, bind groups and dispatches) on a synthetic frame
//! and compares what the GPU wrote with the CPU twin:
//! - `meter_histogram`: the meter and the resolve on a 512² frame (a quarter black, an 8 × 8
//!   block clamped at `HDR_MAX`, the rest log-normal luminance) vs [`Histogram::meter`] and
//!   [`target_ev`]: lit_px ±0.1 %, ev_avg, ev_hi and the target ±0.05 EV.
//! - `exposure_resolve`: the resolve for 16 settings and states vs [`next_adapted_ev`] and
//!   [`resolve_exposure`] (±1e-6 relative), and the read-back ring against the last of them.
//! - `bloom_impulse`: the whole bloom on impulses (one above `BLOOM_INPUT_MAX`) at strength 1
//!   vs [`bloom_reference`]: every texel within 1 % of the peak, each channel's sum within 1 %.
//!
//! M0's `agx` check stays in the app's selftest.

use anyhow::Context as _;
use glam::{DQuat, DVec3, Mat4, UVec2, Vec2, Vec3};
use uni_core::{CameraView, Epoch, UniversePos, splitmix64};
use uni_gfx::probe::ProbeResult;
use uni_gfx::{
    DEPTH_FORMAT, Exposure, ExposureUniform, FrameBindings, FrameCtx, FrameTargets, GpuContext,
    HDR_FORMAT, HDR_MAX, LENS_MASK_FORMAT, PassTimer, RenderFeature, Renderer, ShaderLib,
    SharedInput, exposure_scale, reversed_z_infinite_proj,
};

use crate::bloom::{Bloom, bloom_reference};
use crate::exposure::{
    DEFAULT_ADAPT_TAU_S, ExposureMode, ExposureSettings, ExposureStateGpu, NIGHT_EV100,
    auto_exposure, next_adapted_ev, resolve_exposure,
};
use crate::histogram::{Histogram, Meter, meter_luminance, target_ev};

/// Every uni-post probe, registering its shaders in `lib`.
pub fn gpu_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    crate::register_shaders(lib);
    let result = |name: &str, r: anyhow::Result<(bool, String)>| {
        let (pass, detail) = r.unwrap_or_else(|e| (false, format!("{e:#}")));
        ProbeResult {
            name: name.to_owned(),
            pass,
            detail,
        }
    };
    vec![
        result("meter_histogram", check_meter(gpu, lib)),
        result("exposure_resolve", check_resolve(gpu, lib)),
        result("bloom_impulse", check_bloom(gpu, lib)),
    ]
}

/// Largest |GPU − CPU| of the meter's EVs [EV] (plan B6).
const METER_EV_TOLERANCE: f32 = 0.05;
/// Largest relative |GPU − CPU| of the meter's lit-pixel count.
const LIT_TOLERANCE: f32 = 1e-3;
/// Largest relative |GPU − CPU| of a resolved value (plan B6).
const RESOLVE_TOLERANCE: f32 = 1e-6;
/// Largest |GPU − CPU| of a bloom texel, relative to the channel's peak, and of a channel's sum
/// (plan B6).
const BLOOM_TOLERANCE: f32 = 0.01;

fn check_meter(gpu: &GpuContext, lib: &mut ShaderLib) -> anyhow::Result<(bool, String)> {
    let size = UVec2::new(512, 512);
    let scene_ev = 10.0;
    let pre = exposure_scale(scene_ev);
    let frame = lognormal_frame(size, pre);

    // CPU: the same f16 values, through the twins.
    let total = size.x * size.y;
    let cpu = Histogram::from_luminances(frame.iter().map(|c| meter_luminance(c.to_array(), pre)))
        .meter(total)
        .context("the synthetic frame has lit pixels")?;
    let cpu_target = target_ev(Some(cpu), 0.0);

    // GPU: the meter, then the resolve (Auto, settle) reading it in the same submission.
    let renderer = Renderer::new(gpu, size);
    let settings = ExposureSettings {
        settle: true,
        ..probe_settings(ExposureMode::Auto)
    };
    let (mut resolve, mut meter, _) = auto_exposure(
        gpu,
        lib,
        renderer.camera_bind_group_layout(),
        renderer.exposure_buffer(),
        SharedInput::new(settings),
    )?;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    gpu.queue.write_buffer(
        renderer.exposure_buffer(),
        0,
        bytemuck::bytes_of(&ExposureUniform::from_exposure(Exposure {
            scene_ev100: scene_ev,
            star_ev100: scene_ev,
        })),
    );
    let targets = probe_targets(&gpu.device, size);
    upload_hdr(gpu, &targets, &frame);
    let ctx = probe_ctx(size, 1.0 / 60.0);
    meter.prepare(gpu, &ctx, &targets);
    resolve.prepare(gpu, &ctx, &targets);
    let mut enc = encoder(gpu);
    let bindings = FrameBindings {
        camera: renderer.camera_bind_group(),
    };
    let mut timer = PassTimer::disabled();
    meter.encode(
        gpu,
        &mut enc,
        &bindings,
        &targets,
        &targets.hdr_view,
        &mut timer,
    );
    resolve.encode(
        gpu,
        &mut enc,
        &bindings,
        &targets,
        &targets.hdr_view,
        &mut timer,
    );
    gpu.queue.submit(std::iter::once(enc.finish()));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("meter probe: {err}");
    }
    let state: ExposureStateGpu = read_pod(gpu, &resolve.state)?;
    let uniform: ExposureUniform = read_pod(gpu, renderer.exposure_buffer())?;

    let [lit, ev_avg, ev_hi, read] = state.meter;
    let gpu_target = uniform.ev100[0];
    let pass = read == 1.0
        && (lit / cpu.lit_px - 1.0).abs() <= LIT_TOLERANCE
        && (ev_avg - cpu.ev_avg).abs() <= METER_EV_TOLERANCE
        && (ev_hi - cpu.ev_hi).abs() <= METER_EV_TOLERANCE
        && (gpu_target - cpu_target).abs() <= METER_EV_TOLERANCE;
    Ok((
        pass,
        format!(
            "512² log-normal frame: lit_px GPU {lit:.0} / CPU {:.0}; ev_avg {ev_avg:.4} / {:.4}; \
             ev_hi {ev_hi:.4} / {:.4}; target {gpu_target:.4} / {cpu_target:.4} (tolerance \
             {METER_EV_TOLERANCE} EV)",
            cpu.lit_px, cpu.ev_avg, cpu.ev_hi
        ),
    ))
}

fn check_resolve(gpu: &GpuContext, lib: &mut ShaderLib) -> anyhow::Result<(bool, String)> {
    let size = UVec2::new(8, 8);
    let renderer = Renderer::new(gpu, size);
    let settings = SharedInput::new(probe_settings(ExposureMode::Auto));
    let (mut resolve, _, readback) = auto_exposure(
        gpu,
        lib,
        renderer.camera_bind_group_layout(),
        renderer.exposure_buffer(),
        settings.clone(),
    )?;
    let targets = probe_targets(&gpu.device, size);
    let bindings = FrameBindings {
        camera: renderer.camera_bind_group(),
    };
    let mut worst = 0.0f32;
    let mut failed = Vec::new();
    let mut last = [0.0; 2];
    for (case, (s, state, dt)) in resolve_cases().into_iter().enumerate() {
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        settings.set(s);
        gpu.queue
            .write_buffer(&resolve.state, 0, bytemuck::bytes_of(&state));
        resolve.prepare(gpu, &probe_ctx(size, f64::from(dt)), &targets);
        let mut enc = encoder(gpu);
        resolve.encode(
            gpu,
            &mut enc,
            &bindings,
            &targets,
            &targets.hdr_view,
            &mut PassTimer::disabled(),
        );
        gpu.queue.submit(std::iter::once(enc.finish()));
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("resolve probe, case {case}: {err}");
        }
        let got_state: ExposureStateGpu = read_pod(gpu, &resolve.state)?;
        let got: ExposureUniform = read_pod(gpu, renderer.exposure_buffer())?;

        // CPU: the state step, then the uniform.
        let reading = (state.meter[3] > 0.5).then_some(Some(Meter {
            lit_px: state.meter[0],
            ev_avg: state.meter[1],
            ev_hi: state.meter[2],
        }));
        let prev = (state.adapted[1] > 0.5).then_some(state.adapted[0]);
        let adapted = next_adapted_ev(&s, reading, prev, dt);
        let want = resolve_exposure(&s, adapted);
        let want_adapted = adapted.map_or(state.adapted, |ev| [ev, 1.0, 0.0, 0.0]);
        let pairs = got
            .pre
            .iter()
            .chain(&got.ev100)
            .chain(&got_state.adapted)
            .zip(want.pre.iter().chain(&want.ev100).chain(&want_adapted));
        let err = pairs
            .map(|(&g, &w)| relative_error(g, w))
            .fold(0.0f32, f32::max);
        worst = worst.max(err);
        if !(err <= RESOLVE_TOLERANCE) {
            failed.push(format!(
                "case {case}: GPU {:?} {:?} vs CPU {:?} {:?}",
                got.ev100, got_state.adapted, want.ev100, want_adapted
            ));
        }
        last = [got.ev100[0], got.ev100[1]];
    }
    // The read-back ring: after the waits above, the newest frame read is the last case.
    let ring = readback.last_ev100();
    if ring != Some(last) {
        failed.push(format!("read-back ring {ring:?}, last resolved {last:?}"));
    }
    Ok((
        failed.is_empty(),
        format!(
            "16 settings: worst relative error {worst:.1e} (tolerance {RESOLVE_TOLERANCE:.0e}); \
             read-back {ring:?}{}{}",
            if failed.is_empty() { "" } else { "; " },
            failed.join("; ")
        ),
    ))
}

fn check_bloom(gpu: &GpuContext, lib: &mut ShaderLib) -> anyhow::Result<(bool, String)> {
    let size = UVec2::new(256, 256);
    let mut frame = vec![Vec3::ZERO; (size.x * size.y) as usize];
    for (x, y, v) in [
        (128, 128, Vec3::ONE),
        (60, 200, Vec3::new(0.5, 0.25, 2.0)),
        // Above BLOOM_INPUT_MAX: it enters the chain clamped.
        (200, 60, Vec3::splat(HDR_MAX)),
    ] {
        frame[(y * size.x + x) as usize] = v;
    }
    let cpu = bloom_reference(&frame, size, true);

    let mut renderer = Renderer::new(gpu, size);
    // Strength 1: hdr = bloom, so the read-back is the chain's output.
    let mut bloom = Bloom::new(
        gpu,
        lib,
        renderer.camera_bind_group_layout(),
        SharedInput::new(1.0),
    )?;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let targets = probe_targets(&gpu.device, size);
    upload_hdr(gpu, &targets, &frame);
    let ctx = probe_ctx(size, 1.0 / 60.0);
    // One empty renderer frame uploads the camera uniform (the composite reads its viewport).
    let upload = renderer.render(gpu, &ctx, &targets.hdr_view);
    gpu.queue.submit(std::iter::once(upload));
    bloom.prepare(gpu, &ctx, &targets);
    let mut enc = encoder(gpu);
    bloom.encode(
        gpu,
        &mut enc,
        &FrameBindings {
            camera: renderer.camera_bind_group(),
        },
        &targets,
        &targets.hdr_view,
        &mut PassTimer::disabled(),
    );
    gpu.queue.submit(std::iter::once(enc.finish()));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("bloom probe: {err}");
    }
    let rgba = uni_gfx::read_hdr(gpu, &targets)?;
    let out: Vec<Vec3> = rgba
        .chunks_exact(4)
        .map(|t| Vec3::new(t[0], t[1], t[2]))
        .collect();

    let peak = cpu.iter().fold(Vec3::ZERO, |m, v| m.max(*v));
    let worst = out
        .iter()
        .zip(&cpu)
        .map(|(g, c)| ((*g - *c).abs() / peak).max_element())
        .fold(0.0f32, |a, e| {
            if a.is_nan() || e.is_nan() {
                f32::NAN
            } else {
                a.max(e)
            }
        });
    let (sum_gpu, sum_cpu): (Vec3, Vec3) = (out.iter().copied().sum(), cpu.iter().copied().sum());
    let sum_err = ((sum_gpu - sum_cpu) / sum_cpu).abs().max_element();
    Ok((
        worst <= BLOOM_TOLERANCE && sum_err <= BLOOM_TOLERANCE,
        format!(
            "256² impulses (one at HDR_MAX): worst texel |GPU − CPU| {:.2} % of peak, channel sums \
             within {:.2} % (tolerance {} %)",
            worst * 100.0,
            sum_err * 100.0,
            BLOOM_TOLERANCE * 100.0
        ),
    ))
}

/// |got − want| / |want|, and |got| when `want` is 0; NaN stays NaN (and fails).
fn relative_error(got: f32, want: f32) -> f32 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// Exposure settings for the probes: eye-adapted stars at −4, manual scene EV 9.
fn probe_settings(mode: ExposureMode) -> ExposureSettings {
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

/// 16 resolve inputs: (settings, state before the frame, dt [s]), covering Manual and Auto, eye
/// adaptation on, off and in daylight, the clamps, no reading, nothing lit, the lit-pixel ramp,
/// compensation, settle and both adaptation directions.
fn resolve_cases() -> Vec<(ExposureSettings, ExposureStateGpu, f32)> {
    let base = probe_settings(ExposureMode::Auto);
    let manual = probe_settings(ExposureMode::Manual);
    let meter = |lit: f32, avg: f32, hi: f32| [lit, avg, hi, 1.0];
    let full = 2_073_600.0;
    let state = |m: [f32; 4], adapted: Option<f32>| ExposureStateGpu {
        meter: m,
        adapted: adapted.map_or([0.0; 4], |ev| [ev, 1.0, 0.0, 0.0]),
    };
    let none = [0.0; 4];
    let dt = 1.0 / 60.0;
    let with = |s: ExposureSettings, f: &dyn Fn(&mut ExposureSettings)| {
        let mut s = s;
        f(&mut s);
        s
    };
    vec![
        (manual, state(none, None), dt),
        (
            with(manual, &|s| s.daylight = true),
            state(none, Some(5.0)),
            dt,
        ),
        (
            with(manual, &|s| {
                s.manual.scene_ev100 = 100.0;
                s.eye_adapted = false;
            }),
            state(none, None),
            dt,
        ),
        (
            with(manual, &|s| {
                s.manual.scene_ev100 = -100.0;
                s.eye_star_ev100 = 100.0;
            }),
            state(none, None),
            dt,
        ),
        (base, state(none, None), dt),
        (base, state(none, Some(13.0)), dt),
        (
            with(base, &|s| s.settle = true),
            state(meter(25.0, 12.0, 8.0), Some(5.0)),
            dt,
        ),
        (
            with(base, &|s| s.settle = true),
            state(meter(full, 12.0, 8.0), None),
            dt,
        ),
        (
            with(base, &|s| {
                s.settle = true;
                s.compensation_ev = 1.5;
            }),
            state(meter(full, 12.0, 15.0), Some(2.0)),
            dt,
        ),
        (
            with(base, &|s| s.settle = true),
            state(meter(141.4, 12.0, 8.0), None),
            dt,
        ),
        (base, state(meter(full, 14.0, 9.0), Some(3.0)), dt),
        (base, state(meter(full, -2.0, -7.0), Some(10.0)), 0.1),
        (
            with(base, &|s| s.settle = true),
            state(meter(full, 50.0, 45.0), Some(10.0)),
            dt,
        ),
        (
            with(base, &|s| s.settle = true),
            state(meter(full, 33.79, 28.52), Some(NIGHT_EV100)),
            dt,
        ),
        (base, state(meter(full, 12.0, 8.0), Some(7.0)), 0.0),
        (
            with(base, &|s| {
                s.daylight = true;
                s.adapt_tau_s = [0.3, 2.0];
            }),
            state(meter(5_000.0, 11.0, 6.0), Some(14.0)),
            2.0,
        ),
    ]
}

/// A minimal frame context for a `size` frame (the features read only the sizes and dt).
fn probe_ctx(size: UVec2, dt_real_s: f64) -> FrameCtx {
    let fov_y = 60f32.to_radians();
    FrameCtx {
        frame_index: 0,
        epoch: Epoch::J2000,
        dt_real_s,
        camera: CameraView {
            pos: UniversePos::default(),
            orient: DQuat::IDENTITY,
            fov_y_rad: f64::from(fov_y),
            velocity_mps: DVec3::ZERO,
        },
        view_rot: Mat4::IDENTITY,
        proj: reversed_z_infinite_proj(fov_y, size.x as f32 / size.y as f32, 0.1),
        viewport: size,
        output_size: size,
        exposure: Exposure {
            scene_ev100: 0.0,
            star_ev100: 0.0,
        },
        lenses: Vec::new(),
        lights: Vec::new(),
        sim_dt_s: 0.0,
        jitter_px: Vec2::ZERO,
    }
}

/// Frame targets of `size` whose `hdr` the probes can write (`COPY_DST`) and read (`COPY_SRC`).
fn probe_targets(device: &wgpu::Device, size: UVec2) -> FrameTargets {
    let target = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let attach = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let (hdr, hdr_view) = target(
        "probe hdr",
        HDR_FORMAT,
        attach
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
    );
    let (depth, depth_view) = target("probe depth", DEPTH_FORMAT, attach);
    let (lens_mask, lens_mask_view) = target("probe lens mask", LENS_MASK_FORMAT, attach);
    FrameTargets {
        size,
        hdr,
        hdr_view,
        depth,
        depth_view,
        lens_mask,
        lens_mask_view,
    }
}

/// Writes `rgb` (row-major, alpha 0) into `targets.hdr` as f16.
fn upload_hdr(gpu: &GpuContext, targets: &FrameTargets, rgb: &[Vec3]) {
    let bytes: Vec<u8> = rgb
        .iter()
        .flat_map(|c| [c.x, c.y, c.z, 0.0])
        .flat_map(|v| f16_bits(v).to_le_bytes())
        .collect();
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &targets.hdr,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(targets.size.x * 8),
            rows_per_image: Some(targets.size.y),
        },
        wgpu::Extent3d {
            width: targets.size.x,
            height: targets.size.y,
            depth_or_array_layers: 1,
        },
    );
}

/// The meter probe's frame at scene pre-exposure `pre`, already rounded to f16: the left
/// quarter black (not lit), an 8 × 8 block at `HDR_MAX` in the top-right corner (clamped), the
/// rest a log-normal luminance (median 300 cd/m², σ = 1.5 EV; Box–Muller on `splitmix64`) in a
/// warm tint.
fn lognormal_frame(size: UVec2, pre: f32) -> Vec<Vec3> {
    let tint = Vec3::new(1.1, 1.0, 0.8);
    let tint = tint / tint.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let unit = |h: u64| ((h >> 40) as f32 + 1.0) / (1u64 << 24) as f32; // (0, 1]
    (0..size.y)
        .flat_map(|y| (0..size.x).map(move |x| (x, y)))
        .map(|(x, y)| {
            if x < size.x / 4 {
                return Vec3::ZERO;
            }
            if x >= size.x - 8 && y < 8 {
                return Vec3::splat(HDR_MAX);
            }
            let h1 = splitmix64(u64::from(y * size.x + x));
            let h2 = splitmix64(h1);
            let z = (-2.0 * unit(h1).ln()).sqrt() * (std::f32::consts::TAU * unit(h2)).cos();
            let l = 300.0 * (1.5 * z).exp2();
            // Inside the normal f16 range, where `f16_bits` is exact.
            (tint * l * pre)
                .clamp(Vec3::splat(F16_MIN_NORMAL), Vec3::splat(HDR_MAX))
                .to_array()
                .map(|v| f16_value(f16_bits(v)))
                .into()
        })
        .collect()
}

/// Smallest normal f16, 2^−14.
const F16_MIN_NORMAL: f32 = 6.103_515_6e-5;

/// IEEE 754 binary16 bits of `x`, rounded to nearest even, for 0 and every finite |x| in the
/// normal f16 range [2^−14, 65504], which is all the probes write. Smaller values flush to 0.
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;
    if abs < F16_MIN_NORMAL.to_bits() {
        return sign;
    }
    // Rebias the exponent (f32 bias 127, f16 bias 15); abs ≥ 2^−14 keeps it ≥ 1.
    let exponent = (abs >> 23) + 15 - 127;
    let mantissa = abs & 0x7f_ffff;
    let mut half = (exponent << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    // Round to nearest even; a carry into the exponent is correct.
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

/// The f32 value of f16 bits `h` (normal, subnormal or zero).
fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let mantissa = f32::from(h & 0x3ff);
    sign * if exponent == 0 {
        mantissa * (-24.0f32).exp2()
    } else {
        (1.0 + mantissa / 1024.0) * ((exponent - 15) as f32).exp2()
    }
}

fn encoder(gpu: &GpuContext) -> wgpu::CommandEncoder {
    gpu.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("uni-post probe"),
        })
}

/// Blocking read of a `T` from the start of `buffer` (which needs `COPY_SRC`), after the work
/// already submitted.
fn read_pod<T: bytemuck::Pod>(gpu: &GpuContext, buffer: &wgpu::Buffer) -> anyhow::Result<T> {
    let size = std::mem::size_of::<T>() as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uni-post probe read-back"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = encoder(gpu);
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gpu.queue.submit(std::iter::once(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        // The receiver outlives the wait below; a send error cannot be acted on here.
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("waiting for the probe read-back")?;
    rx.try_recv()
        .context("the probe read-back callback did not run")?
        .context("mapping the probe read-back")?;
    let value = {
        let view = staging
            .get_mapped_range(..)
            .context("reading the probe read-back")?;
        bytemuck::pod_read_unaligned(&view[..])
    };
    staging.unmap();
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_on_the_probe_values() {
        for v in [0.0, 1.0, 0.5, 0.25, 2.0, HDR_MAX, F16_MIN_NORMAL, 65504.0] {
            assert_eq!(f16_value(f16_bits(v)), v, "{v}");
        }
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(HDR_MAX), 0x7b53);
        // Round to nearest even: 1 + 2^-11 (a tie) stays 1, 1 + 3·2^-11 rounds up.
        assert_eq!(f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        // Within half an f16 ulp (2^-11 relative) over the probe's range.
        for k in 0..1000 {
            let v = F16_MIN_NORMAL * (HDR_MAX / F16_MIN_NORMAL).powf(k as f32 / 999.0);
            let r = f16_value(f16_bits(v));
            assert!((r / v - 1.0).abs() <= 2f32.powi(-11), "{v} → {r}");
        }
    }

    #[test]
    fn probe_inputs_are_meaningful() {
        // The meter frame: three quarters lit, its target from the log-normal part (not NIGHT).
        let size = UVec2::new(512, 512);
        let pre = exposure_scale(10.0);
        let frame = lognormal_frame(size, pre);
        let m =
            Histogram::from_luminances(frame.iter().map(|c| meter_luminance(c.to_array(), pre)))
                .meter(size.x * size.y)
                .unwrap();
        assert!((m.lit_px / 2_073_600.0 - 0.75).abs() < 0.01, "{m:?}");
        // Median 300 cd/m², σ 1.5 EV: the 50–95th percentile mean is above the median.
        assert!(
            m.ev_avg > (300.0f32 * 8.0).log2() && m.ev_avg < 16.0,
            "{m:?}"
        );
        // The resolve cases: 16, each with finite expectations.
        let cases = resolve_cases();
        assert_eq!(cases.len(), 16);
        for (s, state, dt) in cases {
            let reading = (state.meter[3] > 0.5).then_some(Some(Meter {
                lit_px: state.meter[0],
                ev_avg: state.meter[1],
                ev_hi: state.meter[2],
            }));
            let prev = (state.adapted[1] > 0.5).then_some(state.adapted[0]);
            let u = resolve_exposure(&s, next_adapted_ev(&s, reading, prev, dt));
            assert!(u.pre.iter().chain(&u.ev100).all(|v| v.is_finite()));
        }
    }
}
