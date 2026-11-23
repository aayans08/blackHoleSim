//! Real asteroids and the procedural belt, pass "asteroids" (Task C2; Task C0 wrote this stub,
//! its constants and signatures).
//!
//! - **Real objects** (`World::asteroid_records`, H ≤ 14): the CPU computes each mean anomaly in
//!   f64 per frame (wrapped to [0, 2π)) and uploads it as f32; the GPU solves Kepler in f32
//!   ([`kepler_f32`]), builds the heliocentric ecliptic position, rotates it to ICRS and
//!   subtracts the camera's heliocentric position passed as a double-single pair
//!   ([`split_f64`]; Cesium "GPU RTE"). Good to ≈ 100 km (review I12): invisible as a point.
//! - **Suppressed records** (`AsteroidInput::suppressed`, `World::suppressed_asteroid_records`:
//!   the registry asteroids, which uni-sim draws from f64 states, and SBDB's Pluto row) emit
//!   zero-area quads, so nothing is drawn twice. The list is the same every frame (ruling A.4a):
//!   it may be uploaded once and again only when it changes.
//! - **The procedural belt** ([`crate::belt::BeltModel`]), M = M₀ + n·t in f32 on the GPU.
//! - **Brightness:** H, G (`photometry::asteroid_v`) at the true phase angle; PSF splat at
//!   `star_pre()` × [`AsteroidInput::exposure_scale`] (the reflected-point rule beside a resolved
//!   body, sheet §1.3) × [`BELT_HIGHLIGHT_GAIN`] while [`AsteroidInput::highlight`], times
//!   (1 − r), and (1 − g) only while the camera is inside an atmosphere (as `PointSprites`,
//!   through [`AsteroidField::with_star_input`]); depth test `Greater` at true depth.
//! - Budget: ≤ 1.0 ms GPU at `asteroid-belt-above` (ladder: belt 250 k → 100 k → 50 k, C6).

use glam::DVec3;
use uni_core::Epoch;
use uni_data::AsteroidRecord;
use uni_gfx::probe::ProbeResult;
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};

use crate::belt::BeltModel;
use crate::stars3d::StarInput;

/// The opt-in "Highlight asteroid belt" multiplies every asteroid's illuminance by this: 15 mag
/// (round 2, review I12; the arithmetic is C2's). Labelled in amber
/// (`uni_ui::BELT_HIGHLIGHT_LABEL`). C6 applies the same gain to the registry asteroids' points.
pub const BELT_HIGHLIGHT_GAIN: f64 = 1.0e6;

/// What the asteroid pass draws this frame (a `SharedInput` the app sets before `render`).
#[derive(Clone, Debug)]
pub struct AsteroidInput {
    pub epoch: Epoch,
    /// The Sun relative to the camera [m], ICRS (`Snapshot::sun`'s `rel_pos_m`).
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
    /// body or without eye-adapted stars; lead ruling B.1).
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

/// The asteroid pass (`RenderFeature` named "asteroids"). The app builds it only once C2 has
/// landed (lead rule: no stubbed renderer is constructed).
pub struct AsteroidField {
    _private: (),
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
        let _ = (gpu, lib, camera_bgl, real, belt, input);
        todo!("C2")
    }

    /// The star pass's input, for its scintillation: while it is > 0 (the camera inside an
    /// atmosphere) asteroids are also multiplied by (1 − g), as `PointSprites::with_star_input`
    /// (ruling B4 r0, sheet §1.4).
    pub fn with_star_input(self, stars: SharedInput<StarInput>) -> Self {
        let _ = stars;
        todo!("C2")
    }

    /// (real asteroids drawn, procedural asteroids drawn) last frame.
    pub fn counts(&self) -> (u32, u32) {
        todo!("C2")
    }
}

impl RenderFeature for AsteroidField {
    fn name(&self) -> &'static str {
        "asteroids"
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

/// The shader's Kepler solver in f32 (Danby's starter, then three Newton steps): the eccentric
/// anomaly [rad] for mean anomaly `m` [rad] and eccentricity `e` < 1. CPU twin of the WGSL.
pub fn kepler_f32(m: f32, e: f32) -> f32 {
    let _ = (m, e);
    todo!("C2")
}

/// `x` as a double-single pair (hi, lo) with hi + lo ≈ x (the camera's heliocentric position
/// for "GPU RTE"). CPU twin of the shader's reconstruction.
pub fn split_f64(x: f64) -> (f32, f32) {
    let _ = x;
    todo!("C2")
}

/// Apparent V of an asteroid of absolute magnitude `h` (G from the record, the pass's) at
/// `r_au` from the Sun and `delta_au` from the camera at phase angle `alpha_rad`, brightened by
/// [`BELT_HIGHLIGHT_GAIN`] (15 mag) while `highlight`.
pub fn belt_highlight_v(h: f64, r_au: f64, delta_au: f64, alpha_rad: f64, highlight: bool) -> f64 {
    let _ = (h, r_au, delta_au, alpha_rad, highlight);
    todo!("C2")
}

/// C2's probe (256 real asteroids' GPU positions against `uni_physics::smallbody` f64 states,
/// ≤ 100 km), called from [`crate::gpu_probes`]. Task C0 stub: none (the lead's selftest runs it
/// before C2 lands).
pub(crate) fn asteroid_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    let _ = (gpu, lib);
    Vec::new()
}
