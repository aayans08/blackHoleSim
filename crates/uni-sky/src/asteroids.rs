//! Real asteroids and the procedural belt, pass "asteroids" (Task C2; Task C0 wrote the stub,
//! its constants and signatures).
//!
//! - **Real objects** (`World::asteroid_records`, H ≤ 14): each frame the CPU computes every
//!   record's mean anomaly in f64, M₀ + n·(t − epoch), wraps it to [−π, π] and uploads it as a
//!   double-single pair of f32 (`real_anomalies`). The GPU solves Kepler's equation for the
//!   high part in f32 ([`kepler_f32`]), corrects sin E and cos E to first order for the full M,
//!   builds the heliocentric position from the packed orbit ([`AsteroidGpu`], whose axes the CPU
//!   rotated from the J2000 ecliptic to ICRS in f64) and subtracts the camera's heliocentric
//!   position passed as a double-single pair ([`split_f64`]; "GPU RTE", Cozzi & Ring 2011, *3D
//!   Engine Design for Virtual Globes*, ch. 5). **Precision (review I12):** the f32 orbit carries
//!   a few ulps of a (2⁻²⁴ × 3 au = 27 km per ulp), so a GPU-drawn asteroid is good to about
//!   100 km in the main belt, 6 × 10⁻⁴ px at 1 au and 1080 lines over 60°: invisible as a point.
//!   The CPU twin measures 94 km at most over the 16 675 records within 3.3 au of the Sun
//!   (median 23 km, `real_positions_match_uni_physics`); the probe measures the GPU. Two choices
//!   beyond the brief's wording buy that: M as one f32 wrapped to [0, 2π) (or [−π, π]) and the
//!   obliquity rotation in the shader gave 135 km (p99 90 km), since both the f32 M and the f32
//!   E near ±π are quantised to 1.2 × 10⁻⁷ rad (54 km at 3 au each) and the rotation rounds
//!   once more; the pair and the folded rotation remove those terms, leaving the f32 assembly.
//! - **Suppressed records** (`AsteroidInput::suppressed`, `World::suppressed_asteroid_records`:
//!   the registry asteroids, which uni-sim draws from f64 states, and SBDB's Pluto row) and
//!   records with unusable elements get the mean anomaly `SUPPRESSED_ANOMALY`, for which the
//!   vertex stage emits a zero-area quad, so nothing is drawn twice. The skip flags are built
//!   when the list changes (it is static, ruling A.4a), not every frame.
//! - **The procedural belt** ([`crate::belt::BeltModel`]): M = M₀ + n·t in f32 on the GPU
//!   (statistical objects; no targeting).
//! - **Brightness:** H, G (`photometry::asteroid_v`, Bowell et al. 1989) at the true phase angle,
//!   as illuminance (`asteroid_lux` in the shader); PSF splat at `star_pre()` ×
//!   [`AsteroidInput::exposure_scale`] (the reflected-point rule beside a resolved body, sheet
//!   §1.3) × [`BELT_HIGHLIGHT_GAIN`] while [`AsteroidInput::highlight`], times (1 − r), and
//!   (1 − g) only while the camera is inside an atmosphere (as `PointSprites`, through
//!   [`AsteroidField::with_star_input`]); depth test `Greater` at the true depth. The colour is
//!   sunlight's (5772 K Planck, luminance 1): no taxonomy is known per object.
//! - **Budget:** ≤ 1.0 ms GPU at `asteroid-belt-above` (ladder: belt 250 k → 100 k → 50 k, C6);
//!   mean anomalies ≤ 0.5 ms CPU ([`AsteroidField::cpu_ms`] reports the whole `prepare`).
//!
//! **Visibility** (the plan's arithmetic redone with the real catalogue and this belt at
//! 2026-11-20, star EV −6.5; C2 report): from 6 au above the Sun the mean glow over the main
//! belt's annulus is 3.4 × 10⁻⁸ pre-exposed from this pass, 1.0 × 10⁻⁷ with the registry
//! asteroids uni-sim draws: 12.3–10.7 stops below AgX black, invisible as the real belt is.
//! Highlighted (+15 mag) it is 3.4 × 10⁻² – 1.0 × 10⁻¹, 7.6–9.2 stops above black.

use std::f64::consts::TAU;
use std::time::Instant;

use anyhow::Context as _;
use bytemuck::Zeroable as _;
use glam::{DVec3, Vec3};
use uni_core::{AU, DAY, Epoch, TEFF_SUN_K, blackbody_linear_srgb, star_illuminance_lux};
use uni_data::AsteroidRecord;
use uni_gfx::probe::{ProbeResult, run_compute};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};
use uni_physics::bodies::GM_SUN_DE440_M3S2;
use uni_physics::frames::ecliptic_j2000_to_icrs;
use uni_physics::kepler::{perifocal_to_reference, solve_kepler};
use uni_physics::photometry::asteroid_v;
use uni_physics::smallbody::{conic_from_mean_anomaly, heliocentric_state_icrs};
use wgpu::util::DeviceExt as _;

use crate::ASTEROIDS_WGSL;
use crate::belt::{BELT_EPOCH, BELT_H_CUT, BELT_SEED, BeltModel, BeltObject};
use crate::points::points_params;
use crate::stars3d::{
    LENS_MASK_BINDING, QUAD_INDICES, StarInput, begin_splat_pass, lens_mask_entry,
    quad_index_buffer, splat_pipeline, vertex_buffer_entry,
};

/// The opt-in "Highlight asteroid belt" multiplies every asteroid's illuminance by this: 15 mag
/// (round 2, review I12). Labelled in amber (`uni_ui::BELT_HIGHLIGHT_LABEL`). C6 applies the same
/// gain to the registry asteroids' points. 10⁶ and not 10⁵: from 6 au above the ecliptic the
/// highlighted belt's mean glow (C2 report) must stand well above the band behind it.
pub const BELT_HIGHLIGHT_GAIN: f64 = 1.0e6;
/// The H–G slope where a record has none, and of every procedural object: 0.15, the value SBDB
/// documents as its default (Bowell et al. 1989's mean slope).
pub const DEFAULT_SLOPE_G: f64 = 0.15;
/// The mean anomaly the CPU writes for a real record the pass must not draw: outside [−π, π],
/// so the vertex stage can tell it from any real one without a NaN test (Metal's fast math).
pub(crate) const SUPPRESSED_ANOMALY: f32 = 16.0;
/// Danby's starter |E₀| = |M| + k e (Danby 1988, *Fundamentals of Celestial Mechanics*, 2nd ed.,
/// §6.6), which [`kepler_f32`] caps by two upper bounds of the root.
pub(crate) const DANBY_K: f64 = 0.85;
/// Steps of Danby's quartic correction (Danby & Burkardt 1983, Celest. Mech. 31, 95) after the
/// starter. The brief asks for Danby's starter and 3 Newton steps; those leave 0.8 % of r at
/// e = 0.89 near perihelion (f64 analysis over e < 0.9, |M| ≤ π: worst 7.7 × 10⁻³ at e = 0.892,
/// M = −0.0094) against the brief's own 1e-5 bound. Three quartic steps cost the same three
/// sin/cos pairs and leave 2.3 × 10⁻¹⁵ for e < 0.9, but from Danby's bare starter still 10⁻³ of
/// r at e = 0.989 and M = 10⁻⁴; capped by the root's upper bounds ∛(12|M|) and |M|/(1 − e) they
/// leave ≤ 5.5 × 10⁻¹³ of r for every e ≤ 0.9999 (f64 analysis, |M| ≤ π and |M| = 10⁻¹…10⁻⁸).
/// The catalogue needs it: 49 of its 33 678 records have e > 0.95 (up to 0.9984, record 33 565,
/// a = 3 053 au, at perihelion in 2026), 25 of them drawn by this pass.
pub(crate) const KEPLER_STEPS: u32 = 3;
/// Floor of the starter's cube-root argument and of 1 − e (the shader's fast math keeps no
/// infinities).
pub(crate) const KEPLER_TINY: f32 = 1e-30;
/// Group-1 binding of the orbits ([`AsteroidGpu`]). `asteroids.wgsl` imports `points.wgsl`
/// (for the point pass's sky factor), which brings the star and point passes' declarations at
/// bindings 0–7; this pass shares only the lens mask at [`LENS_MASK_BINDING`].
pub(crate) const ASTEROIDS_BINDING: u32 = 8;
/// Group-1 binding of the real records' mean anomalies (two double-single pairs per vec4).
pub(crate) const ANOMALIES_BINDING: u32 = 9;
/// Group-1 binding of the [`AsteroidParams`] uniform.
pub(crate) const ASTEROID_PARAMS_BINDING: u32 = 10;
/// Relative luminance weights of linear Rec. 709: ITU-R BT.709-6 (2015), item 3.2.
const REC709_Y: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);
/// Julian Date (TDB) of J2000.0 (IERS Conventions 2010).
const JD_J2000: f64 = 2_451_545.0;

/// What the asteroid pass draws this frame (a `SharedInput` the app sets before `render`).
#[derive(Clone, Debug)]
pub struct AsteroidInput {
    pub epoch: Epoch,
    /// The Sun relative to the camera [m], ICRS (`Snapshot::sun`'s `rel_pos_m`). Non-finite:
    /// nothing is drawn.
    pub sun_rel_m: DVec3,
    /// Draw the procedural belt (the app raises `HonestyFlag::ProceduralBelt` while it is drawn
    /// and visible, C6).
    pub show_belt: bool,
    /// "Highlight asteroid belt" is on: the pass multiplies by [`BELT_HIGHLIGHT_GAIN`] (the app
    /// does not).
    pub highlight: bool,
    /// `Snapshot::suppressed_asteroids`: record indices not drawn here (sorted, static).
    pub suppressed: Vec<u32>,
    /// The factor on reflected light beside a resolved lit body: the app's
    /// `Frame::reflected_scale`, `pre_mix(star_pre, scene_pre, W) / star_pre` (1 with no resolved
    /// body or without eye-adapted stars; lead ruling B.1). Not finite or ≤ 0: nothing is drawn.
    pub exposure_scale: f64,
}

/// `uni_core::Epoch` has no `Default`, so this is by hand: J2000, no Sun offset, no belt, no
/// highlight, nothing suppressed, exposure factor 1.
impl Default for AsteroidInput {
    fn default() -> Self {
        AsteroidInput {
            epoch: Epoch::J2000,
            sun_rel_m: DVec3::ZERO,
            show_belt: false,
            highlight: false,
            suppressed: Vec::new(),
            exposure_scale: 1.0,
        }
    }
}

/// One orbit as the GPU sees it: the twin of `struct AsteroidGpu` in `asteroids.wgsl` (48
/// bytes, three vec4s). The axes are built and rotated to ICRS in f64, then cast, so each
/// component is good to half an f32 ulp of a.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AsteroidGpu {
    /// xyz: a·P [m], P the unit vector to perihelion, ICRS axes; w: e.
    pub p_e: [f32; 4],
    /// xyz: b·Q [m], b = a√(1 − e²), Q = P turned 90° in the direction of motion, ICRS; w: H.
    pub q_h: [f32; 4],
    /// x: the H–G slope G; y: mean motion n [rad/s]; z: M₀ [rad] at
    /// [`crate::belt::BELT_EPOCH`]; w: 0. Real records have y = z = 0: their M comes from the
    /// CPU each frame.
    pub slope_motion: [f32; 4],
}

/// The pass's per-frame uniform: the twin of `struct AsteroidParams` in `asteroids.wgsl` (80
/// bytes, five vec4s).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AsteroidParams {
    /// xyz: camera − Sun [m], ICRS, the f32 high parts ([`split_f64`]); w: 0.
    pub camera_hi: [f32; 4],
    /// xyz: the low parts; w: 0.
    pub camera_lo: [f32; 4],
    /// x: the illuminance factor, `exposure_scale` × [`BELT_HIGHLIGHT_GAIN`] while highlighted;
    /// y: 1 while the camera is inside an atmosphere (× (1 − g)), else 0; z: the belt time
    /// [s since `BELT_EPOCH`]; w: 0.
    pub look: [f32; 4],
    /// rgb: the colour of sunlight ([`solar_colour`]); w: 0.
    pub colour: [f32; 4],
    /// x: real records (instances below it are real); yzw: 0.
    pub counts: [u32; 4],
}

/// A real record's mean anomaly at J2000 [rad], unwrapped, and mean motion [rad/s], so M(t) is
/// one multiply-add in f64.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RealOrbit {
    pub m_j2000_rad: f64,
    pub n_rad_s: f64,
}

/// The CPU half of record `r`'s orbit, or `None` when its elements are not a bound orbit (the
/// loader rejects those; a foreign slice may hold them). n = √(GM☉/a³) with DE440's GM☉, the
/// formula of `uni_physics::kepler::Conic::from_mean_anomaly`.
pub(crate) fn real_orbit(r: &AsteroidRecord) -> Option<RealOrbit> {
    let finite = [r.a_au, r.e, r.i_deg, r.node_deg, r.argp_deg, r.m0_deg, r.epoch_jd_tdb]
        .iter()
        .all(|x| x.is_finite())
        && r.h.is_finite();
    if !(finite && r.a_au > 0.0 && (0.0..1.0).contains(&r.e)) {
        return None;
    }
    let a = r.a_au * AU;
    let n = (GM_SUN_DE440_M3S2 / a).sqrt() / a;
    let epoch_s = (r.epoch_jd_tdb - JD_J2000) * DAY;
    Some(RealOrbit {
        m_j2000_rad: r.m0_deg.to_radians() - n * epoch_s,
        n_rad_s: n,
    })
}

/// `x` [rad] wrapped to [−π, π].
fn wrap_pi(x: f64) -> f64 {
    x - TAU * (x / TAU).round()
}

/// Every real record's mean anomaly at `t` into `out` as double-single pairs, two per vec4
/// (record k's (hi, lo) at `[k / 2][2(k % 2)..]`, [`anomaly_pair`]): M in f64, wrapped to
/// [−π, π], then [`split_f64`]; ([`SUPPRESSED_ANOMALY`], 0) where `skip` (or the orbit is
/// unusable). The per-frame CPU work of the pass.
pub(crate) fn real_anomalies(
    orbits: &[Option<RealOrbit>],
    skip: &[bool],
    t: Epoch,
    out: &mut Vec<[f32; 4]>,
) {
    let t_s = t.seconds_since(Epoch::J2000);
    out.clear();
    out.resize(
        orbits.len().div_ceil(2),
        [SUPPRESSED_ANOMALY, 0.0, SUPPRESSED_ANOMALY, 0.0],
    );
    for (k, (orbit, &skipped)) in orbits.iter().zip(skip).enumerate() {
        if let Some(o) = orbit.filter(|_| !skipped) {
            let (hi, lo) = split_f64(wrap_pi(o.m_j2000_rad + o.n_rad_s * t_s));
            let j = 2 * (k % 2);
            out[k / 2][j] = hi;
            out[k / 2][j + 1] = lo;
        }
    }
}

/// Real record `k`'s (hi, lo) mean anomaly pair in a buffer written by [`real_anomalies`].
pub(crate) fn anomaly_pair(anomalies: &[[f32; 4]], k: usize) -> (f32, f32) {
    let j = 2 * (k % 2);
    (anomalies[k / 2][j], anomalies[k / 2][j + 1])
}

/// Which real records the pass skips: those with unusable elements, and the `suppressed`
/// indices (indices past the slice are ignored).
pub(crate) fn skip_flags(orbits: &[Option<RealOrbit>], suppressed: &[u32]) -> Vec<bool> {
    let mut skip: Vec<bool> = orbits.iter().map(Option::is_none).collect();
    for &i in suppressed {
        if let Some(s) = skip.get_mut(i as usize) {
            *s = true;
        }
    }
    skip
}

/// The packed orbit of semi-major axis `a_au`, eccentricity `e` and angles [rad] (J2000
/// ecliptic), with H `h`, slope `g`, mean motion `n` [rad/s] and M₀ `m0` [rad]: a·P and b·Q from
/// `uni_physics::kepler::perifocal_to_reference`, rotated to ICRS by
/// `uni_physics::frames::ecliptic_j2000_to_icrs` (the J2000 obliquity), in f64, then f32.
#[allow(clippy::too_many_arguments)]
fn orbit_gpu(
    a_au: f64,
    e: f64,
    i: f64,
    node: f64,
    argp: f64,
    h: f32,
    g: f32,
    n: f64,
    m0: f64,
) -> AsteroidGpu {
    let rot = perifocal_to_reference(i, node, argp);
    let a = a_au * AU;
    let b = a * ((1.0 - e) * (1.0 + e)).sqrt();
    let p = ecliptic_j2000_to_icrs(rot.x_axis * a);
    let q = ecliptic_j2000_to_icrs(rot.y_axis * b);
    AsteroidGpu {
        p_e: p.as_vec3().extend(e as f32).to_array(),
        q_h: q.as_vec3().extend(h).to_array(),
        slope_motion: [g, n as f32, m0 as f32, 0.0],
    }
}

/// Real record `r` packed for the GPU (all zero when its elements are unusable: such a record is
/// always skipped). G falls back to [`DEFAULT_SLOPE_G`] where it is not finite.
pub(crate) fn record_gpu(r: &AsteroidRecord) -> AsteroidGpu {
    if real_orbit(r).is_none() {
        return AsteroidGpu::zeroed();
    }
    let g = if r.g.is_finite() {
        r.g
    } else {
        DEFAULT_SLOPE_G as f32
    };
    orbit_gpu(
        r.a_au,
        r.e,
        r.i_deg.to_radians(),
        r.node_deg.to_radians(),
        r.argp_deg.to_radians(),
        r.h,
        g,
        0.0,
        0.0,
    )
}

/// Procedural object `o` packed for the GPU, with its M₀ and n for the shader's belt anomaly.
pub(crate) fn belt_gpu(o: &BeltObject) -> AsteroidGpu {
    orbit_gpu(
        o.a_au,
        o.e,
        o.i_rad,
        o.node_rad,
        o.argp_rad,
        o.h,
        DEFAULT_SLOPE_G as f32,
        o.n_rad_s,
        o.m0_rad,
    )
}

/// The colour of sunlight on an asteroid: the Planck colour of the Sun's 5772 K (IAU 2015
/// Resolution B3) in linear Rec. 709, scaled to luminance 1.
pub(crate) fn solar_colour() -> Vec3 {
    let c = blackbody_linear_srgb(TEFF_SUN_K);
    c / c.dot(REC709_Y)
}

/// The pass's uniform for `input`, the star pass's input `stars` (for "inside an atmosphere",
/// the point pass's rule, [`points_params`]) and `n_real` real records.
pub(crate) fn asteroid_params(
    input: &AsteroidInput,
    stars: Option<&StarInput>,
    n_real: u32,
) -> AsteroidParams {
    let camera = -input.sun_rel_m;
    let drawable = camera.is_finite()
        && input.exposure_scale.is_finite()
        && input.exposure_scale > 0.0;
    let camera = if camera.is_finite() {
        camera
    } else {
        DVec3::ZERO
    };
    let gain = if input.highlight {
        BELT_HIGHLIGHT_GAIN
    } else {
        1.0
    };
    let factor = if drawable {
        (input.exposure_scale * gain).min(f64::from(f32::MAX)) as f32
    } else {
        0.0
    };
    let [x, y, z] = camera.to_array().map(split_f64);
    AsteroidParams {
        camera_hi: [x.0, y.0, z.0, 0.0],
        camera_lo: [x.1, y.1, z.1, 0.0],
        look: [
            factor,
            points_params(stars).sky[0],
            input.epoch.seconds_since(BELT_EPOCH) as f32,
            0.0,
        ],
        colour: solar_colour().extend(0.0).to_array(),
        counts: [n_real, 0, 0, 0],
    }
}

/// The pass's group-1 layout: the lens mask, the orbits, the mean anomalies, the uniform.
pub(crate) fn asteroid_bgl_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
    let storage = wgpu::BufferBindingType::Storage { read_only: true };
    [
        lens_mask_entry(),
        vertex_buffer_entry(ASTEROIDS_BINDING, storage, size_of::<AsteroidGpu>()),
        vertex_buffer_entry(ANOMALIES_BINDING, storage, size_of::<[f32; 4]>()),
        wgpu::BindGroupLayoutEntry {
            binding: ASTEROID_PARAMS_BINDING,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<AsteroidParams>() as u64),
            },
            count: None,
        },
    ]
}

/// The asteroid pass (`RenderFeature` named "asteroids"). The app builds it only once C2 has
/// landed (lead rule: no stubbed renderer is constructed).
///
/// Group 0 is the renderer's. Group 1 (`asteroid_bgl_entries`): binding 1 the lens mask, 8 the
/// orbits (real records in catalogue order, then the belt; static), 9 the real records' mean
/// anomaly pairs (rewritten every frame), 10 the `AsteroidParams` uniform. One indexed draw of a
/// quad per instance: the real records, plus the belt while `show_belt`.
pub struct AsteroidField {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    orbit_buffer: wgpu::Buffer,
    anomaly_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    /// Group 1 for the current targets. `None` until the first `prepare` or `resize`.
    bind_group: Option<wgpu::BindGroup>,
    input: SharedInput<AsteroidInput>,
    /// The star pass's input, read for "the camera is inside an atmosphere"; `None` until
    /// [`AsteroidField::with_star_input`] (the camera is then taken to be in space).
    stars: Option<SharedInput<StarInput>>,
    /// Each real record's CPU orbit (`None`: unusable, never drawn).
    orbits: Vec<Option<RealOrbit>>,
    /// The suppressed list `skip` was built from.
    suppressed: Vec<u32>,
    /// Real records not drawn ([`skip_flags`]).
    skip: Vec<bool>,
    /// Real records drawn: `skip`'s false entries.
    real_drawn: u32,
    /// This frame's mean anomalies (reused between frames).
    anomalies: Vec<[f32; 4]>,
    n_real: u32,
    n_belt: u32,
    /// Instances to draw this frame.
    instances: u32,
    /// (real, procedural) drawn this frame.
    counts: (u32, u32),
    /// CPU time of the last `prepare` [ms].
    cpu_ms: f32,
}

impl AsteroidField {
    /// Uploads the real records `real` (`World::asteroid_records`) and the belt `belt`, and
    /// builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        real: &[AsteroidRecord],
        belt: &BeltModel,
        input: SharedInput<AsteroidInput>,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        crate::register_shaders(lib);
        let module = lib.module(device, ASTEROIDS_WGSL)?;
        let n_real = u32::try_from(real.len()).context("more asteroid records than a u32")?;
        let n_belt = u32::try_from(belt.len()).context("more belt objects than a u32")?;
        n_real
            .checked_add(n_belt)
            .context("more asteroids than one draw counts")?;
        let mut packed: Vec<AsteroidGpu> = Vec::with_capacity(real.len() + belt.len());
        packed.extend(real.iter().map(record_gpu));
        packed.extend(belt.objects().iter().map(belt_gpu));
        if packed.is_empty() {
            // A binding cannot be empty; nothing reads this one.
            packed.push(AsteroidGpu::zeroed());
        }
        let orbits: Vec<Option<RealOrbit>> = real.iter().map(real_orbit).collect();
        let skip = skip_flags(&orbits, &[]);
        let real_drawn = skip.iter().filter(|s| !**s).count() as u32;

        // Validation errors in the objects below come back as an Err, not a device panic.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let orbit_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("asteroid orbits"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let anomaly_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asteroid mean anomalies"),
            size: (real.len().div_ceil(2).max(1) * size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asteroid params"),
            size: size_of::<AsteroidParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = quad_index_buffer(device);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("asteroids bgl"),
            entries: &asteroid_bgl_entries(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("asteroids layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&bgl)],
            immediate_size: 0,
        });
        // Asteroids sit at their true distance: a body in front (greater reversed-Z depth) hides
        // them, and they draw over the cleared far plane and the sky.
        let pipeline = splat_pipeline(
            device,
            &module,
            &layout,
            ("vs_asteroids", "fs_asteroids"),
            wgpu::CompareFunction::Greater,
        );
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the asteroid pipeline: {err}");
        }

        Ok(Self {
            pipeline,
            bgl,
            orbit_buffer,
            anomaly_buffer,
            params_buffer,
            index_buffer,
            bind_group: None,
            input,
            stars: None,
            orbits,
            suppressed: Vec::new(),
            skip,
            real_drawn,
            anomalies: Vec::new(),
            n_real,
            n_belt,
            instances: 0,
            counts: (0, 0),
            cpu_ms: 0.0,
        })
    }

    /// The star pass's input, for its scintillation: while it is > 0 (the camera inside an
    /// atmosphere) asteroids are also multiplied by (1 − g), as `PointSprites::with_star_input`
    /// (ruling B4 r0, sheet §1.4). The app calls `AsteroidField::new(..)?.with_star_input(..)`.
    pub fn with_star_input(mut self, stars: SharedInput<StarInput>) -> Self {
        self.stars = Some(stars);
        self
    }

    /// (real asteroids drawn, procedural asteroids drawn) last frame: the instances submitted,
    /// real ones less those skipped (suppressed or unusable); a submitted object fainter than
    /// the PSF's visibility cutoff is culled in the vertex stage and still counts.
    pub fn counts(&self) -> (u32, u32) {
        self.counts
    }

    /// CPU time of the last `prepare` [ms]: the mean anomalies, their upload and the uniform
    /// (budget ≤ 0.5 ms, for `--bench`).
    pub fn cpu_ms(&self) -> f32 {
        self.cpu_ms
    }

    /// (Re)creates group 1 against `targets`.
    fn bind_targets(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("asteroids bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: LENS_MASK_BINDING,
                    resource: wgpu::BindingResource::TextureView(&targets.lens_mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: ASTEROIDS_BINDING,
                    resource: self.orbit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: ANOMALIES_BINDING,
                    resource: self.anomaly_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: ASTEROID_PARAMS_BINDING,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        }));
    }
}

impl RenderFeature for AsteroidField {
    fn name(&self) -> &'static str {
        "asteroids"
    }

    /// Rebuilds the skip flags when the suppressed list changed, writes the real records' mean
    /// anomalies and the uniform, and sets this frame's instance count.
    fn prepare(&mut self, gpu: &GpuContext, _ctx: &FrameCtx, targets: &FrameTargets) {
        let start = Instant::now();
        if self.input.with(|i| i.suppressed != self.suppressed) {
            self.suppressed = self.input.with(|i| i.suppressed.clone());
            self.skip = skip_flags(&self.orbits, &self.suppressed);
            self.real_drawn = self.skip.iter().filter(|s| !**s).count() as u32;
        }
        let n_real = self.n_real;
        let params = match &self.stars {
            Some(stars) => stars.with(|s| self.input.with(|i| asteroid_params(i, Some(s), n_real))),
            None => self.input.with(|i| asteroid_params(i, None, n_real)),
        };
        let (epoch, show_belt) = self.input.with(|i| (i.epoch, i.show_belt));
        real_anomalies(&self.orbits, &self.skip, epoch, &mut self.anomalies);
        if !self.anomalies.is_empty() {
            gpu.queue
                .write_buffer(&self.anomaly_buffer, 0, bytemuck::cast_slice(&self.anomalies));
        }
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let belt = if show_belt { self.n_belt } else { 0 };
        self.counts = (self.real_drawn, belt);
        self.instances = if belt > 0 {
            self.n_real + belt
        } else if self.real_drawn > 0 {
            self.n_real
        } else {
            0
        };
        if params.look[0] == 0.0 {
            // No light to draw (no Sun or no exposure factor).
            self.instances = 0;
        }
        if self.bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        self.cpu_ms = start.elapsed().as_secs_f32() * 1e3;
    }

    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if self.instances == 0 {
            return;
        }
        if self.bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        let Some(bind_group) = &self.bind_group else {
            return;
        };
        let mut pass = begin_splat_pass(enc, targets, timer, "asteroids");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..self.instances);
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_targets(gpu, targets);
    }
}

/// The shader's Kepler solver in f32: Danby's starter |E₀| = |M| + 0.85 e, capped by the
/// root's upper bounds ∛(12|M|) and |M|/(1 − e), with the sign of M, then `KEPLER_STEPS` (3) of
/// Danby's quartic correction (see `kepler_f32` in `asteroids.wgsl` for the formulas and
/// sources): the eccentric anomaly [rad] for mean anomaly `m` ∈ [−π, π] (the pass reduces M)
/// and eccentricity `e` < 1. CPU twin of the WGSL, operation for operation.
pub fn kepler_f32(m: f32, e: f32) -> f32 {
    let k = DANBY_K as f32;
    let m_abs = m.abs();
    let start = (m_abs + k * e)
        .min((12.0 * m_abs).max(KEPLER_TINY).powf(1.0 / 3.0))
        .min(m_abs / (1.0 - e).max(KEPLER_TINY));
    let mut ea = if m >= 0.0 { start } else { -start };
    for _ in 0..KEPLER_STEPS {
        let (s, c) = ea.sin_cos();
        let f = ea - e * s - m;
        let f1 = 1.0 - e * c;
        let f2 = e * s;
        let f3 = e * c;
        let d1 = -f / f1;
        let d2 = -f / (f1 + 0.5 * d1 * f2);
        let d3 = -f / (f1 + 0.5 * d2 * f2 + d2 * d2 * f3 / 6.0);
        ea += d3;
    }
    ea
}

/// `x` as a double-single pair (hi, lo): hi = x rounded to f32, lo = the rest rounded to f32,
/// so hi + lo carries 48 significant bits (the camera's heliocentric position for "GPU RTE"; the
/// shader reconstructs offsets as (p − hi) − lo). CPU twin of the shader's reconstruction.
pub fn split_f64(x: f64) -> (f32, f32) {
    let hi = x as f32;
    (hi, (x - f64::from(hi)) as f32)
}

/// Apparent V of an asteroid of absolute magnitude `h` at `r_au` from the Sun and `delta_au`
/// from the camera at phase angle `alpha_rad` (`photometry::asteroid_v` with
/// [`DEFAULT_SLOPE_G`], the slope of every procedural object; the pass uses each record's own G),
/// brightened by [`BELT_HIGHLIGHT_GAIN`] (2.5 log₁₀ 10⁶ = 15 mag) while `highlight`.
pub fn belt_highlight_v(h: f64, r_au: f64, delta_au: f64, alpha_rad: f64, highlight: bool) -> f64 {
    let v = asteroid_v(h, DEFAULT_SLOPE_G, r_au, delta_au, alpha_rad);
    if highlight {
        v - 2.5 * BELT_HIGHLIGHT_GAIN.log10()
    } else {
        v
    }
}

// ---------------------------------------------------------------------------------------------
// GPU probe (`universe --selftest`).

/// Shader path of the asteroid probe.
const ASTEROID_PROBE: &str = "sky/probe_asteroids.wgsl";
const ASTEROID_PROBE_WGSL: &str = r#"// sky/probe_asteroids.wgsl: `universe --selftest` probe of
// the asteroid pass's position (Kepler solve, double-single anomaly, orbit, double-single camera),
// belt anomaly and photometry (crates/uni-sky/src/asteroids.rs). Group 2: asteroids.wgsl
// declares groups 0 and 1.
#import "sky/asteroids.wgsl"

// Twin of `asteroids::AsteroidProbeIn`.
struct AsteroidProbeIn {
    orbit: AsteroidGpu,
    // A real record: xy the mean anomaly pair (hi, lo) [rad]. A procedural object: x the belt
    // time [s since the belt epoch], y 0. z: 1 for a procedural object, else 0; w: 0.
    when: vec4<f32>,
    // xyz: camera − Sun [m], ICRS, high parts; w: 0.
    camera_hi: vec4<f32>,
    // xyz: low parts; w: 0.
    camera_lo: vec4<f32>,
}

// Twin of `asteroids::AsteroidProbeOut`.
struct AsteroidProbeOut {
    // xyz: object − camera [m]; w: the eccentric anomaly [rad].
    rel_ea: vec4<f32>,
    // x: illuminance at the camera [lx] (factor 1); y: the mean anomaly's high part [rad]; zw: 0.
    lux_m: vec4<f32>,
}

@group(2) @binding(0) var<storage, read> probe_in: array<AsteroidProbeIn>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<AsteroidProbeOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    var m = c.when.xy;
    if (c.when.z > 0.5) {
        m = vec2<f32>(belt_anomaly(c.orbit.slope_motion.z, c.orbit.slope_motion.y, c.when.x), 0.0);
    }
    let helio = asteroid_helio_icrs(c.orbit, m);
    let rel = rte_offset(helio, c.camera_hi.xyz, c.camera_lo.xyz);
    let r = length(helio);
    let d = length(rel);
    var out: AsteroidProbeOut;
    out.rel_ea = vec4<f32>(rel, kepler_f32(m.x, c.orbit.p_e.w));
    let tan_half = tan_half_phase(helio, rel, r, d);
    out.lux_m = vec4<f32>(
        asteroid_lux(c.orbit.q_h.w, c.orbit.slope_motion.x, r, d, tan_half), m.x, 0.0, 0.0);
    probe_out[gid.x] = out;
}
"#;

/// One input of the probe: twin of `AsteroidProbeIn` (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct AsteroidProbeIn {
    orbit: AsteroidGpu,
    when: [f32; 4],
    camera_hi: [f32; 4],
    camera_lo: [f32; 4],
}

/// One output of the probe: twin of `AsteroidProbeOut` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct AsteroidProbeOut {
    rel_ea: [f32; 4],
    lux_m: [f32; 4],
}

/// The probe's date: 2026-11-20 0h TDB, the app's default start [days since J2000].
const PROBE_DAY: f64 = 9_820.5;
/// Real records the probe checks (brief: 256), spread evenly through the catalogue.
const PROBE_REAL: usize = 256;
/// The brief's position bound for the GPU path [m]: ≤ 100 km, "the stated f32 bound" (a few
/// ulps of a). It is three f32 ulps in the main belt's binade (1.84–3.68 au: 32.8 km per ulp).
/// The probe holds every main-belt record ([`PROBE_MAIN_BELT_R_M`]) to it from any camera, and
/// [`position_tol_m`] every orbit to three ulps of its own aphelion distance, at least this.
pub(crate) const POSITION_TOL_M: f64 = 100.0e3;
/// The main belt for the probe's [`POSITION_TOL_M`] gate: records within 3.3 au of the Sun at the
/// probe's date (the outer edge of the brief's 2.1–3.3 au annulus).
const PROBE_MAIN_BELT_R_M: f64 = 3.3 * AU;
/// f32 ulps of the aphelion distance in [`position_tol_m`].
pub(crate) const POSITION_TOL_ULPS: f64 = 3.0;
/// The procedural path's bound: M = M₀ + n·t in f32 (brief: statistical objects) is off by a few
/// ulps of n·t (~33 rad at 2026 for the main belt: ~4 × 10⁻⁶ rad each), so 2 × 10⁻⁵ of r plus
/// the real path's bound; a wrong orbit or anomaly errs by ≥ 10⁻³ of r.
const PROBE_BELT_TOL_REL: f64 = 2.0e-5;
/// Illuminance: relative to the f64 photometry (exp2/exp/log2 under Metal's fast math, ~10⁻⁶;
/// a wrong phase law or distance errs by ≥ 10⁻²).
const PROBE_LUX_TOL_REL: f64 = 1.0e-3;

/// One probe case with its f64 expectation.
struct ProbeCase {
    input: AsteroidProbeIn,
    /// Object − camera [m] (uni_physics f64).
    want_rel_m: DVec3,
    /// The object's heliocentric position [m] (uni_physics f64).
    helio_m: DVec3,
    /// Heliocentric distance [m].
    r_m: f64,
    /// The case's position bound [m] ([`position_tol_m`], plus the belt's share).
    tol_m: f64,
    /// A procedural object.
    belt: bool,
    /// The f64 eccentric anomaly of the f32 mean anomaly's high part (real records; NaN for the
    /// belt): what `kepler_f32` solves for.
    want_ea: f64,
}

/// The camera of case `k` for an object at heliocentric `helio` [m]: near the Earth (1 au), 6 au
/// above the ecliptic (`asteroid-belt-above`), or 2 236 km from the object on its sunward side
/// (where the double-single camera matters most).
fn probe_camera(k: usize, helio: DVec3) -> DVec3 {
    match k % 3 {
        0 => DVec3::new(-0.18, 0.89, 0.39) * AU,
        1 => uni_physics::frames::ecliptic_j2000_to_icrs(DVec3::new(0.0, 0.0, 6.0 * AU)),
        _ => {
            let towards_sun = -helio.normalize();
            let side = towards_sun.any_orthonormal_vector();
            helio + towards_sun * 2.0e6 + side * 1.0e6
        }
    }
}

/// The f64 illuminance [lx] of an object with H `h`, slope `g` at heliocentric `helio` seen from
/// `camera` [m] (the phase angle from atan2 of the cross and dot products).
fn probe_lux(h: f32, g: f32, helio: DVec3, camera: DVec3) -> f64 {
    let rel = helio - camera;
    let alpha = helio.cross(rel).length().atan2(helio.dot(rel));
    star_illuminance_lux(asteroid_v(
        f64::from(h),
        f64::from(g),
        helio.length() / AU,
        rel.length() / AU,
        alpha,
    ))
}

/// 256 synthetic records for a probe without `asteroids.bin`: a from 1.9 to 48 au, e up to
/// 0.6, every angle, one epoch.
fn synthetic_records() -> Vec<AsteroidRecord> {
    (0..PROBE_REAL)
        .map(|k| {
            let x = k as f64 / (PROBE_REAL - 1) as f64;
            AsteroidRecord {
                a_au: 1.9 * (48.0f64 / 1.9).powf(x),
                e: 0.6 * ((k * 37) % 101) as f64 / 100.0,
                i_deg: ((k * 53) % 180) as f64,
                node_deg: ((k * 97) % 360) as f64,
                argp_deg: ((k * 131) % 360) as f64,
                m0_deg: ((k * 211) % 360) as f64,
                epoch_jd_tdb: 2_461_200.5,
                h: 12.0,
                g: DEFAULT_SLOPE_G as f32,
                diameter_km: f32::NAN,
                albedo: f32::NAN,
                spkid: 50_000_001 + k as u32,
                name_index: 0,
            }
        })
        .collect()
}

/// The probe's cases: 256 real records (every ⌊N/256⌋-th of `asteroids.bin`, or synthetic ones
/// when it cannot be read) and 64 objects of a belt drawn from them, each from one of the three
/// cameras of [`probe_camera`]. Returns the cases and where the records came from.
fn probe_cases() -> (Vec<ProbeCase>, String) {
    let dir = uni_data::data_dir().join("baked");
    let (records, source) = match uni_data::load_asteroids(
        &dir.join("asteroids.bin"),
        &dir.join("asteroid_names.txt"),
    ) {
        Ok((records, _)) if !records.is_empty() => {
            let n = records.len();
            let stride = (n / PROBE_REAL).max(1);
            let chosen: Vec<AsteroidRecord> =
                records.iter().step_by(stride).take(PROBE_REAL).copied().collect();
            let note = format!("{} of {n} catalogue records", chosen.len());
            (chosen, note)
        }
        Ok(_) => (synthetic_records(), "synthetic records (asteroids.bin is empty)".to_owned()),
        Err(e) => (synthetic_records(), format!("synthetic records (asteroids.bin: {e})")),
    };
    let t = Epoch::from_days_since_j2000(PROBE_DAY);
    let mut cases = Vec::new();
    for (k, r) in records.iter().enumerate() {
        let Some(orbit) = real_orbit(r) else { continue };
        let mut pairs = Vec::new();
        real_anomalies(&[Some(orbit)], &[false], t, &mut pairs);
        let (m_hi, m_lo) = anomaly_pair(&pairs, 0);
        let conic = conic_from_mean_anomaly(
            r.a_au,
            r.e,
            r.i_deg,
            r.node_deg,
            r.argp_deg,
            r.m0_deg,
            r.epoch_jd_tdb,
        );
        let helio = heliocentric_state_icrs(&conic, t).0;
        let camera = probe_camera(k, helio);
        let orbit_gpu = record_gpu(r);
        let tol_m = position_tol_m(r.a_au * AU, r.e, (helio - camera).length());
        cases.push(probe_case(
            orbit_gpu,
            [m_hi, m_lo, 0.0, 0.0],
            (helio, tol_m),
            camera,
            false,
            solve_kepler(f64::from(m_hi), r.e),
        ));
    }
    // A small belt drawn from the probe's own records (its histograms need only records in
    // each range).
    let belt = BeltModel::from_catalogue(&records, 48, 16, BELT_H_CUT, BELT_SEED);
    let belt_t = t.seconds_since(BELT_EPOCH) as f32;
    for (k, o) in belt.objects().iter().enumerate() {
        let helio = o.heliocentric_icrs_m(t);
        let camera = probe_camera(k, helio);
        let tol_m = PROBE_BELT_TOL_REL * helio.length()
            + position_tol_m(o.a_au * AU, o.e, (helio - camera).length());
        cases.push(probe_case(
            belt_gpu(o),
            [belt_t, 0.0, 1.0, 0.0],
            (helio, tol_m),
            camera,
            true,
            f64::NAN,
        ));
    }
    (cases, source)
}

/// A case for `orbit` at `when`, whose f64 heliocentric position [m] and position bound [m] are
/// `(helio, tol_m)`, seen from `camera`.
fn probe_case(
    orbit: AsteroidGpu,
    when: [f32; 4],
    (helio, tol_m): (DVec3, f64),
    camera: DVec3,
    belt: bool,
    want_ea: f64,
) -> ProbeCase {
    let [x, y, z] = camera.to_array().map(split_f64);
    ProbeCase {
        input: AsteroidProbeIn {
            orbit,
            when,
            camera_hi: [x.0, y.0, z.0, 0.0],
            camera_lo: [x.1, y.1, z.1, 0.0],
        },
        want_rel_m: helio - camera,
        helio_m: helio,
        r_m: helio.length(),
        tol_m,
        belt,
        want_ea,
    }
}

/// The f32 spacing at `x` > 0: 2^(⌊log₂ x⌋ − 23) (0 for x ≤ 0 or not finite).
pub(crate) fn f32_ulp(x: f64) -> f64 {
    if x > 0.0 && x.is_finite() {
        2f64.powi(x.log2().floor() as i32 - 23)
    } else {
        0.0
    }
}

/// The GPU path's position bound [m] for an orbit of semi-major axis `a_m` and eccentricity `e`
/// seen from `d_m`: [`POSITION_TOL_ULPS`] f32 ulps of the aphelion distance a(1 + e), the largest
/// coordinate the f32 orbit forms, at least the brief's [`POSITION_TOL_M`] (review I12: "a few
/// ulps of a"), plus one ulp of `d_m` for the f32 camera offset's own rounding. Main belt seen
/// from near the Earth: 100 km + 33 km; a Hilda at aphelion (4.37 au): 197 km + 65 km; a
/// Kuiper-belt orbit (Q 30–59 au): 1 573 km plus; the catalogue's a = 3 053 au orbit: 201 000 km.
pub(crate) fn position_tol_m(a_m: f64, e: f64, d_m: f64) -> f64 {
    (POSITION_TOL_ULPS * f32_ulp(a_m * (1.0 + e))).max(POSITION_TOL_M) + f32_ulp(d_m)
}

fn nan_max(a: f64, b: f64) -> f64 {
    if b.is_nan() || b > a { b } else { a }
}

/// Runs the probe on the GPU: one output per case.
fn run_probe(
    gpu: &GpuContext,
    lib: &ShaderLib,
    cases: &[ProbeCase],
) -> anyhow::Result<Vec<AsteroidProbeOut>> {
    let inputs: Vec<AsteroidProbeIn> = cases.iter().map(|c| c.input).collect();
    let n = u32::try_from(inputs.len()).context("too many probe inputs")?;
    let src = lib.compose(ASTEROID_PROBE)?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        bytemuck::cast_slice(&inputs),
        (inputs.len() * size_of::<AsteroidProbeOut>()) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<AsteroidProbeOut> = bytemuck::pod_collect_to_vec(&bytes);
    // A short read-back must not pass on fewer cases.
    anyhow::ensure!(
        out.len() == inputs.len(),
        "read back {} results for {n} cases",
        out.len()
    );
    Ok(out)
}

/// `asteroid_positions`: every main-belt record (within [`PROBE_MAIN_BELT_R_M`] of the Sun)
/// within the brief's [`POSITION_TOL_M`] (≤ 100 km) of uni_physics' f64 state, from whichever
/// camera (review r0 I1), and every real record within its [`position_tol_m`] (three f32 ulps of
/// the aphelion distance, at least 100 km, plus one ulp of the distance). The probe needs at
/// least one main-belt record, or it has not checked the brief's bound.
fn check_real_positions(
    cases: &[ProbeCase],
    out: &[AsteroidProbeOut],
    source: &str,
) -> (bool, String) {
    let (mut worst_main, mut worst_ratio, mut worst_ea) = (0.0f64, 0.0f64, 0.0f64);
    let (mut n, mut n_main) = (0, 0);
    for (c, o) in cases.iter().zip(out).filter(|(c, _)| !c.belt) {
        n += 1;
        let got = Vec3::from_slice(&o.rel_ea[..3]).as_dvec3();
        let err = (got - c.want_rel_m).length();
        if c.r_m <= PROBE_MAIN_BELT_R_M {
            n_main += 1;
            worst_main = nan_max(worst_main, err);
        }
        worst_ratio = nan_max(worst_ratio, err / c.tol_m);
        worst_ea = nan_max(worst_ea, (f64::from(o.rel_ea[3]) - c.want_ea).abs());
    }
    (
        n_main > 0 && worst_main <= POSITION_TOL_M && worst_ratio <= 1.0,
        format!(
            "{n} real records ({source}): {n_main} in the main belt (r ≤ 3.3 au), max {:.1} km \
             (bound {:.0} km); all: worst error / bound {worst_ratio:.2} (bound: 3 f32 ulps of \
             a(1 + e), ≥ 100 km, + 1 ulp of the distance); \
             eccentric anomaly max |GPU − f64| {worst_ea:.1e} rad",
            worst_main / 1e3,
            POSITION_TOL_M / 1e3
        ),
    )
}

/// `asteroid_belt_lux`: the procedural objects' GPU positions (M = M₀ + n·t in f32) within
/// [`PROBE_BELT_TOL_REL`] of r plus [`position_tol_m`] of the f64 orbit, and every case's
/// illuminance within [`PROBE_LUX_TOL_REL`] of the f64 photometry at the GPU's own geometry (the
/// f64 object and a camera at object − the GPU's offset), so the position check's f32 error,
/// 4 % of Δ for the near camera, is not counted twice.
fn check_belt_and_lux(cases: &[ProbeCase], out: &[AsteroidProbeOut]) -> (bool, String) {
    let (mut worst_belt, mut worst_lux) = (0.0f64, 0.0f64);
    let mut n_belt = 0;
    for (c, o) in cases.iter().zip(out) {
        if c.belt {
            n_belt += 1;
            let got = Vec3::from_slice(&o.rel_ea[..3]).as_dvec3();
            worst_belt = nan_max(worst_belt, (got - c.want_rel_m).length() / c.tol_m);
        }
        let rel = Vec3::from_slice(&o.rel_ea[..3]).as_dvec3();
        let want_lux = probe_lux(
            c.input.orbit.q_h[3],
            c.input.orbit.slope_motion[0],
            c.helio_m,
            c.helio_m - rel,
        );
        worst_lux = nan_max(worst_lux, (f64::from(o.lux_m[0]) / want_lux - 1.0).abs());
    }
    (
        worst_belt <= 1.0 && worst_lux <= PROBE_LUX_TOL_REL,
        format!(
            "{n_belt} belt objects: worst position error / bound {worst_belt:.2} (bound \
             {PROBE_BELT_TOL_REL:.0e} r + the real path's); {} cases: illuminance max relative \
             error \
             {worst_lux:.1e} (tolerance {PROBE_LUX_TOL_REL:.0e})",
            cases.len()
        ),
    )
}

/// C2's probes, called from [`crate::gpu_probes`]: `asteroid_positions` (256 real asteroids'
/// GPU positions against `uni_physics::smallbody` f64 states, ≤ 100 km in the main belt, the
/// stated f32 bound) and `asteroid_belt_lux` (the procedural path and the photometry).
pub(crate) fn asteroid_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    lib.add(ASTEROID_PROBE, ASTEROID_PROBE_WGSL);
    let (cases, source) = probe_cases();
    let outcome = run_probe(gpu, lib, &cases);
    let result = |name: &str, check: &dyn Fn(&[AsteroidProbeOut]) -> (bool, String)| {
        let (pass, detail) = match &outcome {
            Ok(out) => check(out),
            Err(e) => (false, format!("{e:#}")),
        };
        ProbeResult {
            name: name.to_owned(),
            pass,
            detail,
        }
    };
    vec![
        result("asteroid_positions", &|out| check_real_positions(&cases, out, &source)),
        result("asteroid_belt_lux", &|out| check_belt_and_lux(&cases, out)),
    ]
}

/// CPU twins of the shader (test only): what `asteroids.wgsl` computes, operation for
/// operation, in f32.
#[cfg(test)]
pub(crate) mod twin {
    use super::*;
    use crate::stars3d::star_lux;
    use crate::stars3d::twin::{Splat, ViewTwin, splat};

    /// A₁, A₂ and B₁, B₂ of the H, G phase function Φᵢ = exp(−Aᵢ tan^Bᵢ(α/2)) (Bowell et al.
    /// 1989, *Asteroids II*, 524; the values of `uni_physics::photometry::hg_phase`), the
    /// shader's `HG_A1` … `HG_B2`.
    pub const HG_A: [f64; 2] = [3.33, 1.87];
    pub const HG_B: [f64; 2] = [0.63, 1.22];
    /// The shader's `HG_TAN_MIN`: floor of tan(α/2) before its log2 (fast math keeps no
    /// infinities), so Φ(0) = 1 to 1e-12.
    pub const HG_TAN_MIN: f32 = 1e-20;

    /// Twin of `asteroid_helio_icrs`, for the mean anomaly pair `(hi, lo)`.
    pub fn asteroid_helio_icrs(o: &AsteroidGpu, (hi, lo): (f32, f32)) -> Vec3 {
        let e = o.p_e[3];
        let ea = kepler_f32(hi, e);
        let (s, c) = (ea.sin(), ea.cos());
        let delta = -(((ea - hi) - e * s) - lo) / (1.0 - e * c);
        Vec3::from_slice(&o.p_e[..3]) * ((c - s * delta) - e)
            + Vec3::from_slice(&o.q_h[..3]) * (s + c * delta)
    }

    /// Twin of `rte_offset`.
    pub fn rte_offset(p: Vec3, hi: Vec3, lo: Vec3) -> Vec3 {
        (p - hi) - lo
    }

    /// Twin of `belt_anomaly`.
    pub fn belt_anomaly(m0: f32, n: f32, t: f32) -> f32 {
        let tau = std::f32::consts::TAU;
        let m = m0 + n * t;
        m - tau * (m / tau + 0.5).floor()
    }

    /// Twin of `tan_half_phase`.
    pub fn tan_half_phase(p: Vec3, rel: Vec3, r: f32, d: f32) -> f32 {
        let (u, v) = (p / r, rel / d);
        let denom = 1.0 + u.dot(v);
        if !(denom > 0.0) {
            return -1.0;
        }
        u.cross(v).length() / denom
    }

    /// Twin of `hg_phase`.
    pub fn hg_phase(tan_half: f32, g: f32) -> f32 {
        if !(tan_half >= 0.0) {
            return 0.0;
        }
        let lt = tan_half.max(HG_TAN_MIN).log2();
        let p1 = (-(HG_A[0] as f32) * (HG_B[0] as f32 * lt).exp2()).exp();
        let p2 = (-(HG_A[1] as f32) * (HG_B[1] as f32 * lt).exp2()).exp();
        p1 + g * (p2 - p1)
    }

    /// Twin of `asteroid_lux`.
    pub fn asteroid_lux(h: f32, g: f32, r_m: f32, d_m: f32, tan_half: f32) -> f32 {
        let au = AU as f32;
        let q = (r_m / au) * (d_m / au);
        star_lux(h) * hg_phase(tan_half, g) / (q * q)
    }

    /// What the probe's `cs_main` writes for `c`. Twin of `cs_main` in the probe shader.
    pub(super) fn probe_out(c: &AsteroidProbeIn) -> AsteroidProbeOut {
        let o = &c.orbit;
        let m = if c.when[2] > 0.5 {
            (belt_anomaly(o.slope_motion[2], o.slope_motion[1], c.when[0]), 0.0)
        } else {
            (c.when[0], c.when[1])
        };
        let helio = asteroid_helio_icrs(o, m);
        let rel = rte_offset(
            helio,
            Vec3::from_slice(&c.camera_hi[..3]),
            Vec3::from_slice(&c.camera_lo[..3]),
        );
        let (r, d) = (helio.length(), rel.length());
        AsteroidProbeOut {
            rel_ea: rel.extend(kepler_f32(m.0, o.p_e[3])).to_array(),
            lux_m: [
                asteroid_lux(
                    o.q_h[3],
                    o.slope_motion[0],
                    r,
                    d,
                    tan_half_phase(helio, rel, r, d),
                ),
                m.0,
                0.0,
                0.0,
            ],
        }
    }

    /// The splat of instance `ii`, or `None` where `vs_asteroids` culls it (suppressed, no
    /// light, behind the camera or below the PSF's cutoff). Twin of `vs_asteroids`.
    pub fn asteroid_splat(
        view: &ViewTwin,
        params: &AsteroidParams,
        records: &[AsteroidGpu],
        anomalies: &[[f32; 4]],
        ii: u32,
    ) -> Option<Splat> {
        let o = &records[ii as usize];
        let m = if ii < params.counts[0] {
            anomaly_pair(anomalies, ii as usize)
        } else {
            (belt_anomaly(o.slope_motion[2], o.slope_motion[1], params.look[2]), 0.0)
        };
        if m.0 >= SUPPRESSED_ANOMALY {
            return None;
        }
        let helio = asteroid_helio_icrs(o, m);
        let rel = rte_offset(
            helio,
            Vec3::from_slice(&params.camera_hi[..3]),
            Vec3::from_slice(&params.camera_lo[..3]),
        );
        let (d, r) = (rel.length(), helio.length());
        if !(d > 0.0) || !(r > 0.0) {
            return None;
        }
        let lux = asteroid_lux(o.q_h[3], o.slope_motion[0], r, d, tan_half_phase(helio, rel, r, d))
            * params.look[0];
        let clip = view.proj * (view.view_rot * rel.extend(0.0)).truncate().extend(1.0);
        let depth = clip.z / clip.w;
        if !(depth > 0.0 && depth <= 1.0) {
            return None;
        }
        splat(view, rel / d, lux, 1.0, Vec3::from_slice(&params.colour[..3]), depth)
    }
}

#[cfg(test)]
mod tests {
    use super::twin::*;
    use super::*;
    use crate::ASTEROIDS_WGSL;
    use crate::stars3d::tests_support::{calls, entry, func, module, wgsl_struct};
    use crate::stars3d::twin::{CULLED, ViewTwin, vertex};
    use core::mem::{offset_of, size_of};
    use glam::{DVec2, Vec2};
    use std::sync::OnceLock;
    use uni_core::{AU, star_illuminance_lux};
    use uni_gfx::validate_wgsl;
    use uni_physics::kepler::solve_kepler;
    use uni_physics::photometry::asteroid_v;
    use uni_physics::smallbody::{conic_from_mean_anomaly, heliocentric_state_icrs};
    use wgpu::naga;

    fn catalogue() -> &'static [AsteroidRecord] {
        static CATALOGUE: OnceLock<Vec<AsteroidRecord>> = OnceLock::new();
        CATALOGUE.get_or_init(|| {
            let dir = uni_data::data_dir().join("baked");
            uni_data::load_asteroids(
                &dir.join("asteroids.bin"),
                &dir.join("asteroid_names.txt"),
            )
            .expect("committed asteroids.bin")
            .0
        })
    }

    /// 2026-11-20 0h TDB (the app's default start date), days since J2000.
    const DAY_2026_11_20: f64 = 9_820.5;

    /// The f64 heliocentric ICRS position of record `r` at `t` (uni_physics: the ground truth).
    fn helio_f64(r: &AsteroidRecord, t: Epoch) -> DVec3 {
        let c = conic_from_mean_anomaly(
            r.a_au,
            r.e,
            r.i_deg,
            r.node_deg,
            r.argp_deg,
            r.m0_deg,
            r.epoch_jd_tdb,
        );
        heliocentric_state_icrs(&c, t).0
    }

    /// The perifocal position (cos E − e, √(1 − e²) sin E) per unit a, in f64.
    fn perifocal(ea: f64, e: f64) -> DVec2 {
        DVec2::new(ea.cos() - e, (1.0 - e * e).sqrt() * ea.sin())
    }

    /// The f32 solver against uni_physics' f64 one (brief): the position from each eccentric
    /// anomaly (computed in f64, so only the solver differs) agrees to 1e-6 of its length for
    /// e < 0.5 and to 1e-5 for e < 0.9, over the whole range of M the pass uploads, near
    /// perihelion included. Up to e = 0.999 it stays finite.
    #[test]
    fn kepler_f32_vs_f64() {
        let mut ms: Vec<f32> = (0..=1440)
            .map(|k| -std::f32::consts::PI + std::f32::consts::TAU * k as f32 / 1440.0)
            .collect();
        for s in [0.0, 1e-7, 1e-5, 1e-3, 0.01, 0.05, 0.2, 3.0, 3.14] {
            ms.extend([s, -s]);
        }
        let mut worst = [(0.0f64, 0.0f32, 0.0f32); 2];
        for k in 0..900 {
            let e = k as f32 * 1e-3;
            for &m in &ms {
                let ea32 = kepler_f32(m, e);
                let (e64, m64) = (f64::from(e), f64::from(m));
                let want = perifocal(solve_kepler(m64, e64), e64);
                let err = (perifocal(f64::from(ea32), e64) - want).length() / want.length();
                let band = usize::from(e >= 0.5);
                if !(err <= worst[band].0) {
                    worst[band] = (err, e, m);
                }
            }
        }
        assert!(worst[0].0 < 1e-6, "e < 0.5: {:?}", worst[0]);
        assert!(worst[1].0 < 1e-5, "e < 0.9: {:?}", worst[1]);
        // Near-parabolic orbits (the catalogue reaches e = 0.9984, at perihelion in 2026): the
        // position to 1e-6 of a, finite everywhere.
        for e in [0.9f32, 0.95, 0.99, 0.9984, 0.999] {
            for &m in &ms {
                let ea32 = kepler_f32(m, e);
                assert!(ea32.is_finite(), "e {e} M {m}");
                let (e64, m64) = (f64::from(e), f64::from(m));
                let want = perifocal(solve_kepler(m64, e64), e64);
                let err = (perifocal(f64::from(ea32), e64) - want).length();
                assert!(err < 1e-6, "e {e} M {m}: {err} of a");
            }
        }
    }

    /// Brief: hi + lo reproduces 4.5e12 m to < 1 mm; any value to the pair's 48 significant
    /// bits. The shader's (p − hi) − lo then gives a camera-relative offset to centimetres where
    /// a plain f32 camera position is off by up to 262 km (half an f32 ulp at 4.5e12 m).
    #[test]
    fn double_single_split() {
        let x = 4.5e12;
        let (hi, lo) = split_f64(x);
        assert!((f64::from(hi) + f64::from(lo) - x).abs() < 1e-3);
        for x in [
            4.5e12 + 0.3721,
            -7.48e12 + 123.456,
            1.495_978_707e11,
            6.0e12 / 7.0,
            -2.0e14 + 0.5,
            1e-3,
            0.0,
        ] {
            let (hi, lo) = split_f64(x);
            assert_eq!(hi, x as f32);
            let err = (f64::from(hi) + f64::from(lo) - x).abs();
            assert!(err <= x.abs() * 2f64.powi(-48), "{x}: {err}");
        }
        // Camera at x, object 1 000 km further along the axis (an f32 heliocentric position).
        let cam = 4.5e12 + 0.3721;
        let obj = (cam + 1.0e6) as f32;
        let want = f64::from(obj) - cam;
        let (hi, lo) = split_f64(cam);
        let rte = rte_offset(
            Vec3::new(obj, 0.0, 0.0),
            Vec3::new(hi, 0.0, 0.0),
            Vec3::new(lo, 0.0, 0.0),
        );
        assert!((f64::from(rte.x) - want).abs() < 0.1, "{} vs {want}", rte.x);
        let naive = obj - cam as f32;
        assert!((f64::from(naive) - want).abs() > 1.0e3);
    }

    /// Review I12: a suppressed record (a registry asteroid uni-sim draws, or SBDB's Pluto row)
    /// emits a zero-area quad: every vertex is the culled position. The same record drawn
    /// unsuppressed is a real splat. Records with unusable elements are skipped too.
    #[test]
    fn suppressed_records_not_drawn() {
        let records = &catalogue()[..3];
        let orbits: Vec<Option<RealOrbit>> = records.iter().map(real_orbit).collect();
        let gpu: Vec<AsteroidGpu> = records.iter().map(record_gpu).collect();
        let t = Epoch::from_days_since_j2000(DAY_2026_11_20);
        // Camera 1 AU from the Sun, looking at record 0 (Ceres), highlight on so all are bright.
        let camera = DVec3::new(AU, 0.0, 0.0);
        let input = AsteroidInput {
            epoch: t,
            sun_rel_m: -camera,
            highlight: true,
            suppressed: vec![0, 2, 99_999],
            ..AsteroidInput::default()
        };
        let params = asteroid_params(&input, None, records.len() as u32);
        let draw = |skip: &[bool], ii: u32, axis: DVec3| {
            let mut anomalies = Vec::new();
            real_anomalies(&orbits, skip, t, &mut anomalies);
            let view = ViewTwin::looking(axis.as_vec3(), 60.0, Vec2::new(1920.0, 1080.0), -6.5);
            let s = asteroid_splat(&view, &params, &gpu, &anomalies, ii);
            (s, (0..4).map(|vi| vertex(&view, s, vi)).collect::<Vec<_>>())
        };
        let skip = skip_flags(&orbits, &input.suppressed);
        assert_eq!(skip, vec![true, false, true]);
        for ii in 0..3u32 {
            let axis = helio_f64(&records[ii as usize], t) - camera;
            let (s, quad) = draw(&skip, ii, axis);
            if skip[ii as usize] {
                assert!(s.is_none(), "record {ii} drawn");
                assert!(quad.iter().all(|v| *v == CULLED), "record {ii}: {quad:?}");
            } else {
                let s = s.expect("record 1 on screen");
                assert!(s.radius_px > 0.0);
                assert!(quad[0] != quad[3]);
            }
            // Unsuppressed, the same record draws.
            let (s, quad) = draw(&[false; 3], ii, axis);
            assert!(s.is_some(), "record {ii} not drawn when unsuppressed");
            assert!(quad[0] != quad[3]);
        }
        // An unusable record is skipped whatever the list says.
        let mut bad = records[1];
        bad.e = f64::NAN;
        assert_eq!(real_orbit(&bad), None);
        assert_eq!(skip_flags(&[None, orbits[1]], &[]), vec![true, false]);
    }

    /// Review I12: `belt_highlight_v` for H 14 at r 2.8 AU, Δ 6.62 AU, α 65° is 22.64 without the
    /// highlight and 7.64 with it (±0.01; G = 0.15). The pass multiplies by the gain only while
    /// the highlight is on, on top of the app's exposure factor.
    #[test]
    fn highlight_gain_applied_only_when_on() {
        let alpha = 65f64.to_radians();
        let off = belt_highlight_v(14.0, 2.8, 6.62, alpha, false);
        let on = belt_highlight_v(14.0, 2.8, 6.62, alpha, true);
        assert!((off - 22.64).abs() <= 0.01, "{off}");
        assert!((on - 7.64).abs() <= 0.01, "{on}");
        assert!((off - on - 15.0).abs() < 1e-12);
        assert_eq!(off, asteroid_v(14.0, DEFAULT_SLOPE_G, 2.8, 6.62, alpha));
        let factor = |highlight: bool, exposure_scale: f64| {
            let input = AsteroidInput {
                highlight,
                exposure_scale,
                ..AsteroidInput::default()
            };
            asteroid_params(&input, None, 0).look[0]
        };
        assert_eq!(factor(false, 1.0), 1.0);
        assert_eq!(factor(true, 1.0), 1.0e6);
        assert_eq!(factor(false, 0.25), 0.25);
        assert_eq!(factor(true, 0.25), 0.25e6);
        for bad in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert_eq!(factor(true, bad), 0.0, "{bad}");
        }
        // The shader's illuminance with that factor is the photometry's V, 15 mag brighter.
        let r = 2.8 * AU;
        let d = 6.62 * AU;
        let lux = asteroid_lux(14.0, 0.15, r as f32, d as f32, (0.5 * alpha).tan() as f32);
        let rel = |a: f64, b: f64| (a / b - 1.0).abs();
        assert!(rel(f64::from(lux), star_illuminance_lux(off)) < 1e-4);
        assert!(rel(f64::from(lux) * 1.0e6, star_illuminance_lux(on)) < 1e-4);
    }

    /// Every GPU struct against naga's layout (the uniform layout rule), and the pass's group-1
    /// layout against what its entry points use.
    #[test]
    fn asteroid_layouts_match_wgsl() {
        let m = module(ASTEROIDS_WGSL);
        let (size, offsets, vec4_only) = wgsl_struct(&m, "AsteroidGpu");
        assert_eq!(size, 48);
        assert_eq!(size, size_of::<AsteroidGpu>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                ("p_e".to_owned(), offset_of!(AsteroidGpu, p_e)),
                ("q_h".to_owned(), offset_of!(AsteroidGpu, q_h)),
                ("slope_motion".to_owned(), offset_of!(AsteroidGpu, slope_motion)),
            ]
        );
        let (size, offsets, vec4_only) = wgsl_struct(&m, "AsteroidParams");
        assert_eq!(size, 80);
        assert_eq!(size, size_of::<AsteroidParams>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                ("camera_hi".to_owned(), offset_of!(AsteroidParams, camera_hi)),
                ("camera_lo".to_owned(), offset_of!(AsteroidParams, camera_lo)),
                ("look".to_owned(), offset_of!(AsteroidParams, look)),
                ("colour".to_owned(), offset_of!(AsteroidParams, colour)),
                ("counts".to_owned(), offset_of!(AsteroidParams, counts)),
            ]
        );
        let p = probe_module();
        let (size, offsets, vec4_only) = wgsl_struct(&p, "AsteroidProbeIn");
        assert_eq!(size, size_of::<AsteroidProbeIn>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                ("orbit".to_owned(), offset_of!(AsteroidProbeIn, orbit)),
                ("when".to_owned(), offset_of!(AsteroidProbeIn, when)),
                ("camera_hi".to_owned(), offset_of!(AsteroidProbeIn, camera_hi)),
                ("camera_lo".to_owned(), offset_of!(AsteroidProbeIn, camera_lo)),
            ]
        );
        let (size, offsets, vec4_only) = wgsl_struct(&p, "AsteroidProbeOut");
        assert_eq!(size, size_of::<AsteroidProbeOut>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                ("rel_ea".to_owned(), offset_of!(AsteroidProbeOut, rel_ea)),
                ("lux_m".to_owned(), offset_of!(AsteroidProbeOut, lux_m)),
            ]
        );

        // Group-1 bindings the entry points use, with their stages, against the layout.
        let caps = naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16;
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&m)
            .unwrap();
        let mut used = std::collections::BTreeMap::new();
        for (i, ep) in m.entry_points.iter().enumerate() {
            let stage = match ep.name.as_str() {
                "vs_asteroids" => wgpu::ShaderStages::VERTEX,
                "fs_asteroids" => wgpu::ShaderStages::FRAGMENT,
                _ => continue,
            };
            let ep_info = info.get_entry_point(i);
            for (h, v) in m.global_variables.iter() {
                let Some(b) = &v.binding else { continue };
                if ep_info[h].is_empty() {
                    continue;
                }
                assert!(b.group <= 1, "{} uses group {}", ep.name, b.group);
                if b.group == 1 {
                    let entry = used
                        .entry(b.binding)
                        .or_insert((wgpu::ShaderStages::NONE, v.space));
                    entry.0 |= stage;
                }
            }
        }
        let entries = asteroid_bgl_entries();
        let want: Vec<u32> = entries.iter().map(|e| e.binding).collect();
        assert_eq!(used.keys().copied().collect::<Vec<_>>(), {
            let mut w = want.clone();
            w.sort_unstable();
            w
        });
        for e in &entries {
            let (stages, space) = used[&e.binding];
            assert_eq!(e.visibility, stages, "binding {}", e.binding);
            match (&e.ty, space) {
                (wgpu::BindingType::Texture { .. }, naga::AddressSpace::Handle) => {}
                (
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size,
                        ..
                    },
                    naga::AddressSpace::Uniform,
                ) => assert_eq!(
                    min_binding_size.map(|s| s.get()),
                    Some(size_of::<AsteroidParams>() as u64)
                ),
                (
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        ..
                    },
                    naga::AddressSpace::Storage { access },
                ) => assert_eq!(access, naga::StorageAccess::LOAD),
                (ty, space) => panic!("binding {}: {ty:?} vs {space:?}", e.binding),
            }
        }
        assert_eq!(
            (ASTEROIDS_BINDING, ANOMALIES_BINDING, ASTEROID_PARAMS_BINDING),
            (8, 9, 10)
        );
    }

    /// The probe's cases and checks on the CPU: the twin's outputs (the GPU's arithmetic without
    /// fast math) pass both checks, so the bounds hold the f32 path itself; a real record 1 000 km
    /// off, a main-belt record 110 km off (inside its generic bound), a belt object 10⁻³ of r
    /// off, an illuminance 1 % off or a NaN fail.
    #[test]
    fn asteroid_probe_passes_on_the_twin_and_fails_on_errors() {
        let (cases, source) = probe_cases();
        assert!(source.contains("catalogue records"), "{source}");
        let real = cases.iter().filter(|c| !c.belt).count();
        assert_eq!(real, PROBE_REAL);
        assert!(cases.iter().filter(|c| c.belt).count() >= 32);
        let twin: Vec<AsteroidProbeOut> = cases.iter().map(|c| probe_out(&c.input)).collect();
        let (pass, detail) = check_real_positions(&cases, &twin, &source);
        assert!(pass, "{detail}");
        let (pass, detail) = check_belt_and_lux(&cases, &twin);
        assert!(pass, "{detail}");
        // Each of the three cameras is used.
        assert!(cases.iter().any(|c| c.want_rel_m.length() < 1.0e7));
        assert!(cases.iter().any(|c| c.want_rel_m.length() > 5.0 * AU));
        let corrupt = |k: usize, f: &dyn Fn(&mut AsteroidProbeOut)| {
            let mut out = twin.clone();
            f(&mut out[k]);
            out
        };
        let belt_k = cases.iter().position(|c| c.belt).unwrap();
        // 1 000 km off fails a real record; 10⁻³ of r (a wrong orbit or anomaly) a belt object.
        let off = corrupt(0, &|o| o.rel_ea[0] += 1.0e6);
        assert!(!check_real_positions(&cases, &off, &source).0);
        // A main-belt record 110 km off fails (review r0 I1: the brief's ≤ 100 km) although its
        // generic bound accepts it (100 km + one f32 ulp of a distance beyond 0.92 au: 116–165 km),
        // so only the main-belt gate can fail it. Of the main-belt records whose bound exceeds
        // 115 km, the one that the f32 rounding of the shifted offset leaves closest to 110 km off.
        let (main_k, wrong, err) = cases
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.belt && c.r_m <= PROBE_MAIN_BELT_R_M && c.tol_m > 115.0e3)
            .map(|(k, c)| {
                let wrong = (c.want_rel_m + c.want_rel_m.normalize() * 110.0e3).as_vec3();
                (k, wrong, (wrong.as_dvec3() - c.want_rel_m).length())
            })
            .min_by(|a, b| (a.2 - 110.0e3).abs().total_cmp(&(b.2 - 110.0e3).abs()))
            .expect("a main-belt record whose bound exceeds 115 km");
        assert!((err - 110.0e3).abs() < 2.0e3, "{err} m");
        assert!(err < cases[main_k].tol_m, "{err} m vs {} m", cases[main_k].tol_m);
        let off = corrupt(main_k, &|o| o.rel_ea[..3].copy_from_slice(&wrong.to_array()));
        let (pass, detail) = check_real_positions(&cases, &off, &source);
        assert!(!pass, "a main-belt record {err:.0} m off passed: {detail}");
        let shift = (1.0e-3 * cases[belt_k].r_m) as f32;
        let off = corrupt(belt_k, &|o| o.rel_ea[0] += shift);
        assert!(!check_belt_and_lux(&cases, &off).0);
        let dim = corrupt(3, &|o| o.lux_m[0] *= 1.01);
        assert!(!check_belt_and_lux(&cases, &dim).0);
        let nan = corrupt(5, &|o| o.rel_ea[1] = f32::NAN);
        assert!(!check_real_positions(&cases, &nan, &source).0);
        assert!(!check_belt_and_lux(&cases, &corrupt(5, &|o| o.lux_m[0] = f32::NAN)).0);
        // The fallback without asteroids.bin: 256 usable synthetic orbits.
        assert_eq!(synthetic_records().len(), PROBE_REAL);
        assert!(synthetic_records().iter().all(|r| real_orbit(r).is_some()));
    }

    /// The probe shader composed the way `asteroid_probes` composes it.
    fn probe_module() -> naga::Module {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        lib.add(ASTEROID_PROBE, ASTEROID_PROBE_WGSL);
        let src = lib.compose(ASTEROID_PROBE).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"))
    }

    /// The pass's shader and its probe compose and validate; the vertex stage runs the solver,
    /// the photometry and the shared splat, the fragment stage the shared PSF and the point
    /// pass's sky factor; the probe calls the same functions and binds only its own group.
    #[test]
    fn asteroid_shader_validates() {
        let m = module(ASTEROIDS_WGSL);
        let vs = entry(&m, "vs_asteroids");
        for f in ["asteroid_helio_icrs", "rte_offset", "asteroid_lux", "splat_vertex"] {
            assert!(calls(&vs.body, func(&m, f)), "vs_asteroids does not call {f}");
        }
        let helio = &m.functions[func(&m, "asteroid_helio_icrs")];
        assert!(calls(&helio.body, func(&m, "kepler_f32")));
        let fs = entry(&m, "fs_asteroids");
        for f in ["psf_weight", "point_sky_factor"] {
            assert!(calls(&fs.body, func(&m, f)), "fs_asteroids does not call {f}");
        }
        let p = probe_module();
        let cs = entry(&p, "cs_main");
        for f in [
            "asteroid_helio_icrs",
            "rte_offset",
            "asteroid_lux",
            "belt_anomaly",
            "kepler_f32",
        ] {
            assert!(calls(&cs.body, func(&p, f)), "probe does not call {f}");
        }
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16,
        )
        .validate(&p)
        .unwrap();
        let ep = p.entry_points.iter().position(|e| e.name == "cs_main").unwrap();
        let mut bound: Vec<(u32, u32)> = p
            .global_variables
            .iter()
            .filter(|(h, _)| !info.get_entry_point(ep)[*h].is_empty())
            .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
            .collect();
        bound.sort_unstable();
        assert_eq!(bound, vec![(2, 0), (2, 1)]);
    }

    /// The f32 path the vertex stage runs (its CPU twin: the double-single M from the CPU, the
    /// f32 solver and correction, the packed ICRS orbit, the double-single camera) against
    /// uni_physics' f64 states for every catalogue record, from a camera near the Earth: within
    /// 100 km for every record within 3.3 au of the Sun (brief), and within the probe's bound
    /// ([`position_tol_m`]: three f32 ulps of the aphelion distance, at least 100 km, plus one ulp
    /// of the distance) for all.
    #[test]
    fn real_positions_match_uni_physics() {
        let t = Epoch::from_days_since_j2000(DAY_2026_11_20);
        let camera = DVec3::new(-0.18, 0.89, 0.39) * AU;
        let orbits: Vec<Option<RealOrbit>> = catalogue().iter().map(real_orbit).collect();
        let skip = vec![false; orbits.len()];
        let mut anomalies = Vec::new();
        real_anomalies(&orbits, &skip, t, &mut anomalies);
        let cam = camera.to_array().map(split_f64);
        let hi = Vec3::new(cam[0].0, cam[1].0, cam[2].0);
        let lo = Vec3::new(cam[0].1, cam[1].1, cam[2].1);
        let mut worst_main = 0.0f64;
        let mut worst_ratio = (0.0f64, 0usize);
        for (k, r) in catalogue().iter().enumerate() {
            let m = anomaly_pair(&anomalies, k);
            let got = rte_offset(asteroid_helio_icrs(&record_gpu(r), m), hi, lo).as_dvec3();
            let want = helio_f64(r, t) - camera;
            let err = (got - want).length();
            let r_helio = helio_f64(r, t).length();
            if r_helio <= 3.3 * AU {
                worst_main = worst_main.max(err);
            }
            let ratio = err / position_tol_m(r.a_au * AU, r.e, want.length());
            if !(ratio <= worst_ratio.0) {
                worst_ratio = (ratio, k);
            }
        }
        assert!(worst_main < 100.0e3, "main belt (r ≤ 3.3 au): {worst_main} m");
        assert!(worst_ratio.0 < 1.0, "error / bound: {worst_ratio:?}");
    }

    /// The shader's illuminance (twin) is the H, G photometry of uni_physics through
    /// `star_illuminance_lux`, over phase angles 0–179° and the belt's distances.
    #[test]
    fn asteroid_lux_matches_photometry() {
        for (h, g) in [(14.0f32, 0.15f32), (3.34, 0.12), (16.5, 0.4), (10.0, -0.1)] {
            for alpha_deg in [0.0f64, 0.01, 0.5, 5.0, 30.0, 65.0, 120.0, 150.0, 170.0, 179.0] {
                for (r_au, d_au) in [(2.8, 6.62), (1.0, 1e-4), (45.0, 44.0), (3.0, 2.5)] {
                    let alpha = alpha_deg.to_radians();
                    let want = star_illuminance_lux(asteroid_v(
                        f64::from(h),
                        f64::from(g),
                        r_au,
                        d_au,
                        alpha,
                    ));
                    let got = asteroid_lux(
                        h,
                        g,
                        (r_au * AU) as f32,
                        (d_au * AU) as f32,
                        (0.5 * alpha).tan() as f32,
                    );
                    if want < 1e-36 {
                        // Below f32's normal range (α = 179° at 2.8 au): no light to speak of.
                        assert!((0.0..1e-36).contains(&got), "{got}");
                        continue;
                    }
                    assert!(
                        (f64::from(got) / want - 1.0).abs() < 1e-4,
                        "H {h} G {g} α {alpha_deg}° r {r_au} Δ {d_au}: {got} vs {want}"
                    );
                }
            }
        }
        // Unlit (α = 180°: no tan(α/2), the vertex stage's -1) gives no light, never NaN.
        assert_eq!(asteroid_lux(14.0, 0.15, 3.0e11, 1.0e11, -1.0), 0.0);
        // tan(α/2) from the geometry keeps an opposition angle cos α cannot (0.084°, a Kuiper
        // belt object seen from the Earth: f32 cos α was 6 × 10⁻⁴ off in Φ).
        let alpha = 0.084f64.to_radians();
        let (r, d) = (41.989 * AU, 41.0 * AU);
        let p = DVec3::new(r, 0.0, 0.0);
        let rel = DVec3::new(alpha.cos(), alpha.sin(), 0.0) * d;
        let t = tan_half_phase(p.as_vec3(), rel.as_vec3(), r as f32, d as f32);
        assert!((f64::from(t) / (0.5 * alpha).tan() - 1.0).abs() < 1e-5, "{t}");
        let behind = tan_half_phase(p.as_vec3(), -p.as_vec3(), r as f32, r as f32);
        assert!(behind < 0.0);
    }

    fn wgsl_const(m: &naga::Module, name: &str) -> naga::Literal {
        let (_, c) = m
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no const {name}"));
        match m.global_expressions[c.init] {
            naga::Expression::Literal(l) => l,
            ref e => panic!("const {name} is not a literal: {e:?}"),
        }
    }

    #[test]
    fn shader_constants_match_cpu() {
        let m = module(ASTEROIDS_WGSL);
        for (name, cpu) in [
            ("SUPPRESSED_ANOMALY", SUPPRESSED_ANOMALY),
            ("AU_M", AU as f32),
            ("DANBY_K", DANBY_K as f32),
            ("KEPLER_TINY", KEPLER_TINY),
            ("HG_A1", HG_A[0] as f32),
            ("HG_A2", HG_A[1] as f32),
            ("HG_B1", HG_B[0] as f32),
            ("HG_B2", HG_B[1] as f32),
            ("HG_TAN_MIN", HG_TAN_MIN),
        ] {
            assert_eq!(wgsl_const(&m, name), naga::Literal::F32(cpu), "{name}");
        }
        assert_eq!(wgsl_const(&m, "KEPLER_STEPS"), naga::Literal::U32(KEPLER_STEPS));
    }

    /// The CPU's mean anomalies: M₀ + n·(t − epoch) in f64 for every record, wrapped to
    /// [−π, π], as double-single pairs (two per vec4), the sentinel where skipped; equal to the
    /// anomaly of uni_physics' conic.
    #[test]
    fn mean_anomalies_wrap_and_match_conics() {
        let records = &catalogue()[..1001];
        let orbits: Vec<Option<RealOrbit>> = records.iter().map(real_orbit).collect();
        let mut skip = vec![false; records.len()];
        skip[5] = true;
        for days in [DAY_2026_11_20, -36_525.0 * 40.0, 36_525.0 * 900.0] {
            let t = Epoch::from_days_since_j2000(days);
            let mut out = Vec::new();
            real_anomalies(&orbits, &skip, t, &mut out);
            assert_eq!(out.len(), records.len().div_ceil(2));
            for (k, r) in records.iter().enumerate() {
                let (hi, lo) = anomaly_pair(&out, k);
                if skip[k] {
                    assert_eq!((hi, lo), (SUPPRESSED_ANOMALY, 0.0));
                    continue;
                }
                assert!((-std::f32::consts::PI..=std::f32::consts::PI).contains(&hi), "{hi}");
                assert!(lo.abs() <= 2.4e-7, "{lo}");
                let m = f64::from(hi) + f64::from(lo);
                let n = (uni_physics::bodies::GM_SUN_DE440_M3S2 / (r.a_au * AU).powi(3)).sqrt();
                let dt = (t.jd_tdb() - r.epoch_jd_tdb) * uni_core::DAY;
                let want = r.m0_deg.to_radians() + n * dt;
                let diff = m - want;
                let diff = diff - std::f64::consts::TAU * (diff / std::f64::consts::TAU).round();
                // The pair carries M to ~1e-14 rad; f64 M itself is good to ~1e-9 rad at
                // 900 centuries (n·Δt ~ 10⁵ rad).
                assert!(diff.abs() < 1e-8, "record {k} day {days}: {m} vs {want}");
            }
        }
    }

    /// The per-frame CPU cost of the mean anomalies for 50 000 records (brief: ≤ 0.5 ms,
    /// measured). Print-only (run with `--ignored --nocapture`): the test build is unoptimised.
    #[test]
    #[ignore]
    fn mean_anomaly_cost() {
        let base: Vec<Option<RealOrbit>> = catalogue().iter().map(real_orbit).collect();
        let orbits: Vec<Option<RealOrbit>> = base.iter().cycle().take(50_000).copied().collect();
        let skip = vec![false; orbits.len()];
        let mut out = Vec::new();
        let frames = 200;
        let start = std::time::Instant::now();
        for f in 0..frames {
            let t = Epoch::from_days_since_j2000(DAY_2026_11_20 + f as f64 * 0.01);
            real_anomalies(&orbits, &skip, t, &mut out);
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(frames);
        println!("mean anomalies, 50 000 records: {ms:.3} ms per frame (test profile)");
    }
}

