//! The render stack and per-frame data shared by the window, `--capture` and `--bench`.
//!
//! [`Scene`] owns the [`Renderer`] with its features in their fixed order (design §5.3;
//! uni-gfx `targets` documents the contract). Two stacks exist, chosen by `--sky`
//! ([`SkyMode`]):
//! - **M0** (the default, unchanged): the debug-grid background (only while the lens is traced),
//!   the lens pass (only its traced pixels), then the BSC5 stars, then the tonemap (which
//!   upscales the internal resolution to the output).
//! - **Sky v1** (`--sky v1`, an early slice of Task B9): the Wave B sky in the M1 frame
//!   contract's order without the body passes: exposure resolve → band-update → histogram meter
//!   → lens (band environment) → band → AT-HYG stars in 3-D → bloom → tonemap ([`Scene::new_v1`]).
//!
//! The HUD draws after them. The pure helpers below turn the world and a camera view into the
//! [`FrameCtx`] the features read and the [`HudData`] the HUD shows.

use std::ops::RangeInclusive;

use anyhow::Context as _;
use glam::{DMat4, DVec3, UVec2, Vec2};
use uni_core::{CameraView, DAY, Epoch, PC, UniversePos};
use uni_data::{CatalogStar, Star3};
use uni_gfx::{
    Exposure, FrameCtx, GpuContext, RenderFeature, Renderer, ShaderLib, SharedInput,
    reversed_z_infinite_proj,
};
use uni_nav::{Bookmark, ExposureChoice};
use uni_post::{ExposureMode, ExposureReadback, ExposureSettings};
use uni_rel::{DebugSkyBackground, LensCoverage, LensEnv, LensScale, LensTracer};
use uni_sim::honesty::HonestyFlag;
use uni_sky::{BandInput, StarInput};
use uni_ui::{HudControls, HudData};

use crate::world::World;

// Task B0 adds B9's Wave B dependencies (ruling B.8), so no wave agent edits the manifest. These
// lines keep `cargo::unused_dependencies` quiet until B9 wires each crate in; B9 deletes each one
// then. (uni-procgen is used below for the band's galaxy model, uni-sim for the sky v1 stars and
// its honesty labels.)
use uni_atmo as _;
use uni_physics as _;
use uni_planet as _;

/// Near plane of the reversed-Z infinite projection [m]. M0 draws nothing closer than
/// infinity, so this only fixes the depth scale for later geometry.
pub const NEAR_PLANE_M: f32 = 0.1;
/// Scene exposure when nothing sets it [EV100]. The scene's only light in M0 is the debug grid,
/// which is display-referred (pre-exposed to the same values at any scene EV).
pub const DEFAULT_SCENE_EV100: f32 = 0.0;
/// Lens tracer resolution scale in the window (the HUD slider); the tracer's own default.
pub const INTERACTIVE_LENS_SCALE: f32 = uni_rel::DEFAULT_RESOLUTION_SCALE;
/// Lens tracer resolution scale in captures, unless a bookmark sets one.
pub const CAPTURE_LENS_SCALE: f32 = 1.0;
/// Lines of the window's internal render resolution (design §6: "1080p internal").
pub const INTERNAL_LINES: u32 = 1080;
/// The exposures the renderer accepts [EV100] (review m5): a typed slider value outside this is
/// clamped, so 2^EV never overflows and no inf or NaN reaches the HDR target. It spans a
/// moonless sky (about −6) to the Sun's disk (about 30) with margin.
pub const EV100_LIMITS: RangeInclusive<f32> = -30.0..=40.0;
/// Faintest V the sky v1 star pass draws from AT-HYG: the bake keeps V ≤ 10 (uni-data `athyg`,
/// the `athyg_40_reduced_m10` subset). Fainter starlight goes into the band
/// (`uni_sky::BandInput::v_lim`).
pub const ATHYG_V_LIM: f32 = 10.0;
/// Faintest V of the BSC5 fallback: the Yale Bright Star Catalogue lists the stars brighter than
/// about V 6.5 (Hoffleit & Warren 1991, 5th revised ed., CDS V/50).
pub const BSC5_V_LIM: f32 = 6.5;
/// A Julian year [s] (IAU: 365.25 d), the unit of `StarInput::years_since_epoch`.
const JULIAN_YEAR_S: f64 = 365.25 * DAY;
/// Shown while the lens is traced in sky v1: the stars are drawn unlensed over the traced pixels
/// until M3 (the plan's M1 wording; C4 moves the string into `uni_ui::theme`).
pub const UNLENSED_STARS_LABEL: &str = "Stars near the hole are not yet lensed (M3)";

/// Which sky stack the renderer runs (`--sky`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SkyMode {
    /// The M0 stack: debug-grid lens environment, BSC5 stars, manual exposure. The default, and
    /// what every run without `--sky` renders, unchanged.
    #[default]
    M0,
    /// `--sky v1`: the Milky Way band (B5), the AT-HYG stars in 3-D (B4), auto-exposure with the
    /// eye-adapted star exposure and bloom (B6), and the lens over the band environment (B7).
    V1,
}

impl SkyMode {
    /// The value of `--sky`: `v1`, or `m0` for the default.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "m0" => Ok(SkyMode::M0),
            "v1" => Ok(SkyMode::V1),
            _ => Err(format!("--sky takes v1 or m0, not {s:?}")),
        }
    }
}

/// The star catalogue a scene draws.
pub enum SkyCatalog {
    /// The Yale Bright Star Catalogue (M0's `uni_sky::StarField`): the M0 stack, or sky v1's
    /// fallback when AT-HYG is missing.
    Bsc5(Vec<CatalogStar>),
    /// AT-HYG in 3-D (`uni_sim::World::star_data()`), drawn by `uni_sky::StarField3` in sky v1.
    AtHyg(Vec<Star3>),
}

/// Sky v1's exposure choices beyond the HUD's two EV sliders, which a bookmark can set (uni-nav's
/// M1 fields `exposure` and `eye_adapted`): Auto exposure with eye-adapted stars by default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyExposure {
    pub mode: ExposureMode,
    /// The "Eye-adapted stars" option: stars at the star EV slider's value.
    pub eye_adapted: bool,
}

impl Default for SkyExposure {
    fn default() -> Self {
        Self {
            mode: ExposureMode::Auto,
            eye_adapted: true,
        }
    }
}

impl SkyExposure {
    /// The defaults, overridden by what `b` names.
    pub fn from_bookmark(b: &Bookmark) -> Self {
        let d = Self::default();
        Self {
            mode: match b.exposure {
                Some(ExposureChoice::Auto) => ExposureMode::Auto,
                Some(ExposureChoice::Manual) => ExposureMode::Manual,
                None => d.mode,
            },
            eye_adapted: b.eye_adapted.unwrap_or(d.eye_adapted),
        }
    }
}

/// B6's settings for one frame of sky v1 from the HUD controls (clamped as [`exposure`] does)
/// and the bookmark's choices: the scene slider is the manual scene EV (Auto starts from it), the
/// star slider the eye-adapted star EV. `settle` jumps straight to the metered target (captures).
/// `daylight` is false: sky v1 draws no atmosphere, so the camera is never under a daylit sky
/// (the M1 frame contract sets it from `uni_atmo::camera_in_daylight`, B9).
pub fn exposure_settings(
    controls: &HudControls,
    choice: SkyExposure,
    settle: bool,
) -> ExposureSettings {
    let e = exposure(controls);
    let mut s = ExposureSettings {
        mode: choice.mode,
        manual: e,
        compensation_ev: 0.0,
        eye_adapted: choice.eye_adapted,
        eye_star_ev100: e.star_ev100,
        daylight: false,
        adapt_tau_s: uni_post::exposure::DEFAULT_ADAPT_TAU_S,
        settle,
    };
    // B6 decision D4: the resolve takes the star EV from `eye_star_ev100` while eye-adapted stars
    // are in effect and from the scene EV otherwise, so `manual` (which becomes
    // `FrameCtx::exposure`) says the same.
    if !s.eye_adapted_in_effect() {
        s.manual.star_ev100 = s.manual.scene_ev100;
    }
    s
}

/// The exposure of one frame: what goes into `FrameCtx::exposure`, and sky v1's settings (which
/// the HUD's eye-adapted label follows); `None` for the M0 stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameExposure {
    pub exposure: Exposure,
    pub settings: Option<ExposureSettings>,
}

/// The star pass's input for `ctx` (sky v1): the camera from the SSB in parsecs, the years since
/// the catalogue epoch (J2000.0, `uni_data::STAR3_EPOCH_JD_TDB`, which is `Epoch::J2000`), vivid
/// colours on (owner feedback, labelled) and no twinkle (in space; scintillation is B3's, under
/// an atmosphere). Nothing is suppressed: sky v1 draws no star surfaces or point sprites.
pub fn star_input(ctx: &FrameCtx) -> StarInput {
    StarInput {
        years_since_epoch: ctx.epoch.seconds_since(Epoch::J2000) / JULIAN_YEAR_S,
        camera_pc: camera_icrs_pc(&ctx.camera.pos),
        suppressed: Vec::new(),
        vivid_colours: true,
        scintillation: 0.0,
    }
}

/// The band passes' input for `ctx` (sky v1): on, with stars fainter than `v_lim` in the band.
pub fn band_input(ctx: &FrameCtx, v_lim: f32) -> BandInput {
    BandInput {
        camera_icrs_pc: camera_icrs_pc(&ctx.camera.pos),
        enabled: true,
        v_lim,
    }
}

/// Camera − SSB [pc], ICRS axes (f64).
fn camera_icrs_pc(pos: &UniversePos) -> DVec3 {
    pos.delta_from(UniversePos::default()) / PC
}

/// Sky v1's shared inputs and read-back, set each frame before the renderer runs.
struct SkyV1 {
    stars: SharedInput<StarInput>,
    band: SharedInput<BandInput>,
    /// The band's star limit: [`ATHYG_V_LIM`], or [`BSC5_V_LIM`] with the fallback catalogue.
    v_lim: f32,
    exposure: SharedInput<ExposureSettings>,
    readback: ExposureReadback,
}

/// The renderer with its feature stack, plus the handles the app keeps to steer it.
pub struct Scene {
    pub renderer: Renderer,
    /// Shared with the lens pass and the debug background inside `renderer`.
    lens_scale: LensScale,
    /// Catalogue stars the star feature draws.
    pub star_count: u32,
    /// `None` for the M0 stack.
    sky_v1: Option<SkyV1>,
}

impl Scene {
    /// The stack for `mode` over `catalog`: [`Scene::new`] for M0 (which needs BSC5),
    /// [`Scene::new_v1`] for sky v1.
    pub fn for_sky(
        gpu: &GpuContext,
        internal_size: UVec2,
        output_format: wgpu::TextureFormat,
        mode: SkyMode,
        catalog: &SkyCatalog,
        lens_scale: f32,
    ) -> anyhow::Result<Self> {
        match (mode, catalog) {
            (SkyMode::M0, SkyCatalog::Bsc5(stars)) => {
                Scene::new(gpu, internal_size, output_format, stars, lens_scale)
            }
            (SkyMode::M0, SkyCatalog::AtHyg(_)) => {
                anyhow::bail!("the M0 sky draws the BSC5 catalogue, not AT-HYG")
            }
            (SkyMode::V1, catalog) => {
                Scene::new_v1(gpu, internal_size, output_format, catalog, lens_scale)
            }
        }
    }

    /// Builds the renderer at the internal resolution `internal_size` and pushes the debug
    /// background, the lens pass, the stars and the tonemap, in that order. `output_format` is
    /// the sRGB format the tonemap writes (the window's sRGB view, or
    /// `uni_gfx::CAPTURE_FORMAT`).
    pub fn new(
        gpu: &GpuContext,
        internal_size: UVec2,
        output_format: wgpu::TextureFormat,
        stars: &[CatalogStar],
        lens_scale: f32,
    ) -> anyhow::Result<Self> {
        let mut renderer = Renderer::new(gpu, internal_size);
        let mut lib = ShaderLib::with_common();
        let camera_bgl = renderer.camera_bind_group_layout();
        let lens_scale = LensScale::new(lens_scale);
        // M1 constructors (B0 rulings A.4, A.5): the lens pass is given the band's environment
        // cubemap and the lens environment. This M0 stack keeps the debug grid, and the band
        // passes are not pushed yet (B9 does both), so the cubemap stays black and unused.
        let band = uni_sky::BandCubemap::new(
            gpu,
            &mut lib,
            &uni_procgen::galaxy::GalaxyModel::milky_way(),
            SharedInput::new(uni_sky::BandInput::default()),
        )
        .context("creating the environment cubemap")?;
        let lens_env = SharedInput::new(LensEnv::DebugGrid);
        let background = DebugSkyBackground::new(
            gpu,
            &mut lib,
            camera_bgl,
            lens_scale.clone(),
            lens_env.clone(),
        )
        .context("creating the debug background pass")?;
        let tracer = LensTracer::new(
            gpu,
            &mut lib,
            camera_bgl,
            lens_scale.clone(),
            band.env_map(),
            lens_env,
        )
        .context("creating the lens pass")?;
        let star_field = uni_sky::StarField::new(gpu, &mut lib, camera_bgl, stars)
            .context("creating the star pass")?;
        let tonemap = uni_post::TonemapFeature::new(gpu, &mut lib, camera_bgl, output_format)
            .context("creating the tonemap pass")?;
        let star_count = star_field.count();

        renderer.push(Box::new(background));
        renderer.push(Box::new(tracer));
        renderer.push(Box::new(star_field));
        renderer.push(Box::new(tonemap));
        Ok(Self {
            renderer,
            lens_scale,
            star_count,
            sky_v1: None,
        })
    }

    /// Builds the sky v1 stack at the internal resolution `internal_size`, in the M1 frame
    /// contract's order (CLAUDE.md §2) without the passes sky v1 does not draw (debug background,
    /// Sun, bodies, rings, comets, atmosphere, points, asteroids):
    /// 1. `exposure`: B6's resolve, which overwrites the exposure buffer on the GPU;
    /// 2. `band-update`: B5's environment cubemap (all six faces on the first frame; it never
    ///    refreshes inside the Solar System);
    /// 3. `histogram`: B6's meter, before any sky light is written, so it meters world light only
    ///    (here there is none, and the scene EV settles at `uni_post::NIGHT_EV100`);
    /// 4. `lens`: B7's tracer with [`LensEnv::Band`] (only while a lens is traced);
    /// 5. `band`: B5's band at star exposure × (1 − r)(1 − g)(1 − b);
    /// 6. `stars`: B4's AT-HYG field (`catalog`, normally `World::star_data()`), or M0's BSC5
    ///    field when `catalog` is the BSC5 fallback;
    /// 7. `bloom` (+ `bloom composite`): B6, at `uni_post::DEFAULT_BLOOM_STRENGTH`;
    /// 8. `tonemap` (upscale + AgX + dither), writing `output_format`.
    pub fn new_v1(
        gpu: &GpuContext,
        internal_size: UVec2,
        output_format: wgpu::TextureFormat,
        catalog: &SkyCatalog,
        lens_scale: f32,
    ) -> anyhow::Result<Self> {
        let mut renderer = Renderer::new(gpu, internal_size);
        let mut lib = ShaderLib::with_common();
        let camera_bgl = renderer.camera_bind_group_layout();
        let lens_scale = LensScale::new(lens_scale);
        let v_lim = match catalog {
            SkyCatalog::AtHyg(_) => ATHYG_V_LIM,
            SkyCatalog::Bsc5(_) => BSC5_V_LIM,
        };
        let star_input = SharedInput::new(StarInput::default());
        let band_input = SharedInput::new(BandInput {
            camera_icrs_pc: DVec3::ZERO,
            enabled: true,
            v_lim,
        });
        let settings = SharedInput::new(exposure_settings(
            &default_controls(lens_scale.get()),
            SkyExposure::default(),
            false,
        ));

        let (resolve, meter, readback) = uni_post::auto_exposure(
            gpu,
            &mut lib,
            camera_bgl,
            renderer.exposure_buffer(),
            settings.clone(),
        )
        .context("creating the auto-exposure passes")?;
        let band = uni_sky::BandCubemap::new(
            gpu,
            &mut lib,
            &uni_procgen::galaxy::GalaxyModel::milky_way(),
            band_input.clone(),
        )
        .context("creating the band cubemap pass")?;
        let tracer = LensTracer::new(
            gpu,
            &mut lib,
            camera_bgl,
            lens_scale.clone(),
            band.env_map(),
            SharedInput::new(LensEnv::Band),
        )
        .context("creating the lens pass")?;
        let band_draw = uni_sky::BandDraw::new(
            gpu,
            &mut lib,
            camera_bgl,
            band.env_map(),
            band_input.clone(),
        )
        .context("creating the band pass")?;
        let (star_field, star_count): (Box<dyn RenderFeature>, u32) = match catalog {
            SkyCatalog::AtHyg(stars) => {
                let f =
                    uni_sky::StarField3::new(gpu, &mut lib, camera_bgl, stars, star_input.clone())
                        .context("creating the 3-D star pass")?;
                let n = f.count();
                (Box::new(f), n)
            }
            SkyCatalog::Bsc5(stars) => {
                let f = uni_sky::StarField::new(gpu, &mut lib, camera_bgl, stars)
                    .context("creating the BSC5 star pass")?;
                let n = f.count();
                (Box::new(f), n)
            }
        };
        let bloom = uni_post::Bloom::new(
            gpu,
            &mut lib,
            camera_bgl,
            SharedInput::new(uni_post::DEFAULT_BLOOM_STRENGTH),
        )
        .context("creating the bloom pass")?;
        let tonemap = uni_post::TonemapFeature::new(gpu, &mut lib, camera_bgl, output_format)
            .context("creating the tonemap pass")?;

        renderer.push(Box::new(resolve));
        renderer.push(Box::new(band));
        renderer.push(Box::new(meter));
        renderer.push(Box::new(tracer));
        renderer.push(Box::new(band_draw));
        renderer.push(star_field);
        renderer.push(Box::new(bloom));
        renderer.push(Box::new(tonemap));
        Ok(Self {
            renderer,
            lens_scale,
            star_count,
            sky_v1: Some(SkyV1 {
                stars: star_input,
                band: band_input,
                v_lim,
                exposure: settings,
                readback,
            }),
        })
    }

    /// The stack this scene runs.
    pub fn sky(&self) -> SkyMode {
        if self.sky_v1.is_some() {
            SkyMode::V1
        } else {
            SkyMode::M0
        }
    }

    /// Lens tracer resolution scale from the next frame on (clamped to 0.25..=1).
    pub fn set_lens_scale(&self, s: f32) {
        self.lens_scale.set(s);
    }

    /// The lens tracer resolution scale in use.
    pub fn lens_scale(&self) -> f32 {
        self.lens_scale.get()
    }

    /// The exposure of the next frame. M0: the HUD controls, clamped ([`exposure`]). Sky v1: B6's
    /// settings from the controls and `choice` ([`exposure_settings`]), handed to the resolve
    /// pass, whose `manual` part goes into `FrameCtx::exposure`. Call it before [`Scene::render`].
    pub fn set_exposure(
        &self,
        controls: &HudControls,
        choice: SkyExposure,
        settle: bool,
    ) -> FrameExposure {
        match &self.sky_v1 {
            None => FrameExposure {
                exposure: exposure(controls),
                settings: None,
            },
            Some(v1) => {
                let s = exposure_settings(controls, choice, settle);
                v1.exposure.set(s);
                FrameExposure {
                    exposure: s.manual,
                    settings: Some(s),
                }
            }
        }
    }

    /// The [scene, star] EV100 the GPU resolved, read back at least one frame late (after a
    /// headless frame's wait: that frame's); `None` for the M0 stack (whose exposure is the HUD's)
    /// and before the first read-back.
    pub fn resolved_ev100(&self) -> Option<[f32; 2]> {
        self.sky_v1.as_ref()?.readback.last_ev100()
    }

    /// Records one frame into `output` (see [`Renderer::render`]). Sky v1 first sets the star and
    /// band inputs from `ctx` (the M1 frame contract: every shared input before the renderer).
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        ctx: &FrameCtx,
        output: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        if let Some(v1) = &self.sky_v1 {
            v1.stars.set(star_input(ctx));
            v1.band.set(band_input(ctx, v1.v_lim));
        }
        self.renderer.render(gpu, ctx, output)
    }
}

/// The internal render resolution for an output (window) of `output` pixels: about
/// [`INTERNAL_LINES`] lines at the output's aspect ratio, never more pixels than the output
/// (a smaller window renders 1:1). The tonemap upscales it to the output.
pub fn internal_size(output: UVec2) -> UVec2 {
    let out = output.max(UVec2::ONE);
    let lines = INTERNAL_LINES.min(out.y);
    let width = (f64::from(lines) * f64::from(out.x) / f64::from(out.y)).round() as u32;
    UVec2::new(width.clamp(1, out.x), lines)
}

/// Root → camera rotation for a camera whose `orient` turns camera axes into root axes.
pub fn view_rot(orient: glam::DQuat) -> glam::Mat4 {
    DMat4::from_quat(orient.inverse()).as_mat4()
}

/// The frame context for `view`, rendered at the internal resolution `viewport` and shown on an
/// output of `output_size` pixels. The renderer keeps its targets at `viewport`, so the lens
/// pass (which plans with the targets) and the HUD's lens status (`uni_rel::lens_coverage`, with
/// this viewport) agree.
pub fn frame_ctx(
    world: &World,
    view: &CameraView,
    viewport: UVec2,
    output_size: UVec2,
    exposure: Exposure,
    frame_index: u64,
    dt_real_s: f64,
) -> FrameCtx {
    let viewport = viewport.max(UVec2::ONE);
    let aspect = viewport.x as f32 / viewport.y as f32;
    // What `World::advance` moved the epoch by for this `dt_real_s` (it skips a non-finite step).
    let sim_dt_s = dt_real_s * world.time_rate;
    FrameCtx {
        frame_index,
        epoch: world.epoch,
        dt_real_s,
        camera: view.clone(),
        view_rot: view_rot(view.orient),
        proj: reversed_z_infinite_proj(view.fov_y_rad as f32, aspect, NEAR_PLANE_M),
        viewport,
        output_size: output_size.max(UVec2::ONE),
        exposure,
        lenses: world.lens_views(&view.pos),
        // M0's world has no light sources yet (the Sun becomes one in Wave A).
        lights: Vec::new(),
        sim_dt_s: if sim_dt_s.is_finite() { sim_dt_s } else { 0.0 },
        jitter_px: Vec2::ZERO,
    }
}

/// HUD controls at their start values, with the lens tracer at `lens_scale`.
pub fn default_controls(lens_scale: f32) -> HudControls {
    HudControls {
        scene_ev100: DEFAULT_SCENE_EV100,
        star_ev100: uni_sky::DEFAULT_STAR_EV100,
        lens_scale,
        visible: true,
        show_help: false,
    }
}

/// The exposures the HUD controls hold, clamped to [`EV100_LIMITS`] (a NaN gives the default).
pub fn exposure(controls: &HudControls) -> Exposure {
    Exposure {
        scene_ev100: clamp_ev(controls.scene_ev100, DEFAULT_SCENE_EV100),
        star_ev100: clamp_ev(controls.star_ev100, uni_sky::DEFAULT_STAR_EV100),
    }
}

/// `ev` clamped to [`EV100_LIMITS`], or `fallback` if it is NaN.
fn clamp_ev(ev: f32, fallback: f32) -> f32 {
    if ev.is_nan() {
        fallback
    } else {
        ev.clamp(*EV100_LIMITS.start(), *EV100_LIMITS.end())
    }
}

/// Applies `b`: its epoch to `world`, its exposure and lens scale to `controls` (fields it leaves
/// out keep their current values), and returns its view with the position normalised.
/// Errors name the bookmark: a bad epoch string, or a view that would put NaN on the GPU.
pub fn apply_bookmark(
    b: &Bookmark,
    world: &mut World,
    controls: &mut HudControls,
) -> anyhow::Result<CameraView> {
    let view = checked_view(b)?;
    if let Some(s) = &b.epoch_utc {
        world.epoch = Epoch::parse_iso_utc(s).with_context(|| {
            format!(
                "bookmark {:?}: epoch_utc {s:?} is not YYYY-MM-DDThh:mm:ss[.fff]Z",
                b.name
            )
        })?;
    }
    if let Some(ev) = b.scene_ev100 {
        controls.scene_ev100 = ev;
    }
    if let Some(ev) = b.star_ev100 {
        controls.star_ev100 = ev;
    }
    if let Some(s) = b.lens_scale {
        controls.lens_scale = s;
    }
    Ok(view)
}

/// `b`'s view with its position normalised and its orientation renormalised, or an error if its
/// position or velocity is not finite, the orientation is degenerate, or the field of view is
/// outside (0°, 180°).
fn checked_view(b: &Bookmark) -> anyhow::Result<CameraView> {
    let v = &b.view;
    let pos_ok = v.pos.local.is_finite() && v.velocity_mps.is_finite();
    let len = v.orient.length();
    let orient = (len.is_finite() && len > 1e-12).then(|| v.orient / len);
    let fov_ok = v.fov_y_rad > 0.0 && v.fov_y_rad < std::f64::consts::PI;
    match (pos_ok, orient, fov_ok) {
        (true, Some(orient), true) => Ok(CameraView {
            pos: v.pos.normalized(),
            orient,
            fov_y_rad: v.fov_y_rad,
            velocity_mps: v.velocity_mps,
        }),
        _ => anyhow::bail!(
            "bookmark {:?}: the view needs a finite position and velocity, a non-zero \
             orientation and 0 < fov_y_rad < π (got {:?})",
            b.name,
            v
        ),
    }
}

/// Frame timings for the HUD.
pub struct Perf {
    pub fps: f32,
    pub cpu_ms: f32,
    /// Per pass; `None` for an invalid measurement.
    pub gpu_ms: Vec<(&'static str, Option<f32>)>,
    pub gpu_frame_ms: Option<f32>,
    pub gpu_timing_available: bool,
}

impl Perf {
    /// The renderer's latest GPU timings, with the window's (or capture's) fps and CPU time.
    pub fn from_renderer(renderer: &Renderer, gpu: &GpuContext, fps: f32, cpu_ms: f32) -> Self {
        Self {
            fps,
            cpu_ms,
            gpu_ms: renderer.gpu_ms(),
            gpu_frame_ms: renderer.gpu_frame_ms(),
            gpu_timing_available: gpu.has_timestamps(),
        }
    }
}

/// What the HUD shows for this frame. `notices` (catalogue warnings and the like) come first,
/// then the lens labels for `ctx` ([`lens_warnings`]), then, in sky v1 (`sky` is its exposure
/// settings; `None` for M0), its honesty labels ([`sky_v1_labels`]).
pub fn hud_data(
    world: &World,
    ctx: &FrameCtx,
    perf: Perf,
    speed_mps: f64,
    star_count: u32,
    notices: &[String],
    sky: Option<&ExposureSettings>,
) -> HudData {
    let (nearest_name, nearest_dist_m) = world.nearest(&ctx.camera.pos);
    let mut warnings = notices.to_vec();
    warnings.extend(lens_warnings(ctx));
    // One source for the lens state, so the HUD and the lens pass agree (review m7).
    let coverage = uni_rel::lens_coverage(ctx);
    let (lens_on_screen, eye_adapted_stars) = match sky {
        // uni-ui shows its two M0 honesty lines ("Lens background: M0 debug grid", "Stars near
        // the hole are not yet lensed (M0)") only when the debug grid is actually visible: the
        // lens pass traced this frame (and the debug background drew; the same condition,
        // `uni_rel::debug_background_draws`). From inside the horizon the frame is all black, so
        // no grid is shown and the labels would be false (lead ruling, M0 gate, Task 10b). The M0
        // stack always runs the debug grid (B0 ruling A.5). uni-ui's amber "Eye-adapted stars
        // (dual exposure, non-physical)" shows while the stars' exposure differs from the
        // scene's (review I4, design §2 and §7.4).
        None => (
            uni_rel::debug_background_draws(coverage, LensEnv::DebugGrid),
            ctx.exposure.eye_adapted_stars(),
        ),
        // Sky v1 traces the band environment, which shows no grid: uni-ui's flag would print both
        // M0 lines, so it stays off and the one line that applies is in `warnings`
        // ([`sky_v1_labels`]). The eye-adapted label follows the condition the resolve pass uses
        // (M1 frame contract, review m13), not `FrameCtx::exposure`.
        Some(s) => {
            warnings.extend(sky_v1_labels(coverage));
            (false, s.eye_adapted_in_effect())
        }
    };
    HudData {
        fps: perf.fps,
        cpu_ms: perf.cpu_ms,
        gpu_ms: perf.gpu_ms,
        gpu_frame_ms: perf.gpu_frame_ms,
        gpu_timing_available: perf.gpu_timing_available,
        render_size: ctx.viewport.to_array(),
        output_size: ctx.output_size.to_array(),
        dist_from_sun_m: ctx
            .camera
            .pos
            .delta_from(uni_core::UniversePos::default())
            .length(),
        nearest_name,
        nearest_dist_m,
        speed_mps,
        epoch: world.epoch,
        time_rate: world.time_rate,
        star_count,
        warnings,
        inside_horizon: coverage == LensCoverage::InsideHorizon,
        lens_on_screen,
        eye_adapted_stars,
    }
}

/// Sky v1's honesty lines (shown in amber with the HUD's warnings; the plan colours the band
/// line grey, which uni-ui's HUD v0 cannot yet): "Stars near the hole are not yet lensed (M3)"
/// while the lens is traced (`uni_rel::lens_labels` with the band environment: the star pass
/// draws unlensed stars over the traced pixels), "Vivid star colours · non-physical" (the
/// owner's default, always on here), and uni-sim's band label (the band is always drawn here).
pub fn sky_v1_labels(coverage: LensCoverage) -> Vec<String> {
    let (_debug_grid, unlensed_stars) = uni_rel::lens_labels(coverage, LensEnv::Band);
    let mut out = Vec::new();
    if unlensed_stars {
        out.push(UNLENSED_STARS_LABEL.to_owned());
    }
    out.push(uni_ui::VIVID_STAR_COLOURS_LABEL.to_owned());
    out.push(uni_sim::honesty::label(HonestyFlag::BandModel).to_owned());
    out
}

/// uni-rel's coverage warning for `ctx` (a lens too far away for the tracer's step budget), if
/// any. The M0 honesty lines are uni-ui's, driven by `HudData::lens_on_screen`.
pub fn lens_warnings(ctx: &FrameCtx) -> Vec<String> {
    uni_rel::lens_coverage(ctx)
        .warning()
        .map(str::to_owned)
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, DVec3, Vec3, Vec4};
    use uni_core::{M_SUN, UniversePos, gravitational_radius};

    fn view_at(pos: UniversePos, orient: DQuat) -> CameraView {
        CameraView {
            pos,
            orient,
            fov_y_rad: 60_f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        }
    }

    fn exposure0() -> Exposure {
        exposure(&default_controls(1.0))
    }

    /// A camera `r_m` [M] sunward of TestBH-1, looking at it.
    fn ctx_near_hole(r_m: f64) -> (World, FrameCtx) {
        let world = World::new(Epoch::J2000);
        let m = gravitational_radius(world.lenses[0].mass_kg);
        let pos = world.lenses[0].pos.offset(DVec3::new(-r_m * m, 0.0, 0.0));
        // Looking along +x with +z up: camera −Z → +x, +Y → +z.
        let orient = DQuat::from_mat3(&glam::DMat3::from_cols(
            DVec3::NEG_Y,
            DVec3::Z,
            DVec3::NEG_X,
        ));
        let ctx = frame_ctx(
            &world,
            &view_at(pos, orient),
            UVec2::new(1920, 1080),
            UVec2::new(1920, 1080),
            exposure0(),
            0,
            0.0,
        );
        (world, ctx)
    }

    #[test]
    fn view_rot_maps_camera_forward_to_minus_z() {
        let orient = DQuat::from_rotation_y(0.7) * DQuat::from_rotation_x(-0.3);
        let fwd_world = (orient * DVec3::NEG_Z).as_vec3();
        let up_world = (orient * DVec3::Y).as_vec3();
        let r = view_rot(orient);
        let f = r.transform_vector3(fwd_world);
        let u = r.transform_vector3(up_world);
        assert!(f.abs_diff_eq(Vec3::NEG_Z, 1e-6), "{f}");
        assert!(u.abs_diff_eq(Vec3::Y, 1e-6), "{u}");
    }

    #[test]
    fn frame_ctx_centre_pixel_looks_along_the_camera_forward() {
        // pixel_dir's convention (uni-gfx camera.wgsl): the centre of a 2×2 frame is NDC (0, 0);
        // unprojecting it through inv(proj · view_rot) gives the forward direction.
        let world = World::new(Epoch::J2000);
        let orient = DQuat::from_rotation_z(1.1) * DQuat::from_rotation_x(0.4);
        let ctx = frame_ctx(
            &world,
            &view_at(UniversePos::default(), orient),
            UVec2::new(2, 2),
            UVec2::new(4, 4),
            exposure0(),
            7,
            0.016,
        );
        let p = (ctx.proj * ctx.view_rot).inverse() * Vec4::new(0.0, 0.0, 1.0, 1.0);
        let dir = (p.truncate() / p.w).normalize();
        let fwd = (orient * DVec3::NEG_Z).as_vec3();
        assert!(dir.abs_diff_eq(fwd, 1e-5), "{dir} vs {fwd}");
        assert_eq!(ctx.frame_index, 7);
        assert_eq!(ctx.viewport, UVec2::new(2, 2));
        assert_eq!(ctx.output_size, UVec2::new(4, 4));
        assert_eq!(ctx.lenses.len(), 1);
    }

    #[test]
    fn frame_ctx_survives_a_zero_viewport() {
        let world = World::new(Epoch::J2000);
        let ctx = frame_ctx(
            &world,
            &view_at(UniversePos::default(), DQuat::IDENTITY),
            UVec2::ZERO,
            UVec2::ZERO,
            exposure0(),
            0,
            0.0,
        );
        assert_eq!(ctx.viewport, UVec2::ONE);
        assert_eq!(ctx.output_size, UVec2::ONE);
        assert!(ctx.proj.is_finite());
    }

    /// Review I6: the window renders at about 1080 lines at its own aspect, never above the
    /// output's size, and the tonemap upscales.
    #[test]
    fn internal_size_is_about_1080_lines_at_the_output_aspect() {
        // The M3 Air's Retina window at 1280×800 points.
        assert_eq!(
            internal_size(UVec2::new(2560, 1600)),
            UVec2::new(1728, 1080)
        );
        assert_eq!(
            internal_size(UVec2::new(3840, 2160)),
            UVec2::new(1920, 1080)
        );
        // Smaller than 1080 lines: 1:1.
        assert_eq!(internal_size(UVec2::new(1280, 800)), UVec2::new(1280, 800));
        assert_eq!(
            internal_size(UVec2::new(1920, 1080)),
            UVec2::new(1920, 1080)
        );
        // Degenerate outputs stay valid.
        assert_eq!(internal_size(UVec2::ZERO), UVec2::ONE);
        assert_eq!(internal_size(UVec2::new(1, 5000)), UVec2::new(1, 1080));
    }

    /// Review m5: a typed EV outside the slider range cannot overflow 2^EV.
    #[test]
    fn exposure_is_clamped_to_finite_limits() {
        let mut c = default_controls(1.0);
        c.scene_ev100 = 1e30;
        c.star_ev100 = f32::NEG_INFINITY;
        let e = exposure(&c);
        assert_eq!(e.scene_ev100, *EV100_LIMITS.end());
        assert_eq!(e.star_ev100, *EV100_LIMITS.start());
        let u = uni_gfx::ExposureUniform::from_exposure(e);
        assert!(u.pre.iter().all(|x| x.is_finite() && *x >= 0.0), "{u:?}");
        assert!(u.pre[0] > 0.0 && u.pre[1] > 0.0, "{u:?}");
        c.scene_ev100 = f32::NAN;
        c.star_ev100 = f32::NAN;
        let e = exposure(&c);
        assert_eq!(e.scene_ev100, DEFAULT_SCENE_EV100);
        assert_eq!(e.star_ev100, uni_sky::DEFAULT_STAR_EV100);
        // In range, untouched.
        c.scene_ev100 = 12.5;
        assert_eq!(exposure(&c).scene_ev100, 12.5);
    }

    #[test]
    fn near_the_hole_the_lens_runs_and_the_hud_shows_the_m0_labels() {
        let (world, ctx) = ctx_near_hole(30.0);
        assert_eq!(uni_rel::lens_coverage(&ctx), LensCoverage::Traced);
        assert!(uni_rel::debug_background_draws(
            uni_rel::lens_coverage(&ctx),
            LensEnv::DebugGrid
        ));
        let perf = Perf {
            fps: 60.0,
            cpu_ms: 1.0,
            gpu_ms: vec![("lens", Some(2.0)), ("tonemap", None)],
            gpu_frame_ms: Some(3.0),
            gpu_timing_available: true,
        };
        let data = hud_data(&world, &ctx, perf, 0.0, 9096, &["sky ok?".to_owned()], None);
        assert_eq!(data.warnings, ["sky ok?"]);
        assert!(data.lens_on_screen);
        assert_eq!(data.gpu_ms, [("lens", Some(2.0)), ("tonemap", None)]);
        assert_eq!(data.gpu_frame_ms, Some(3.0));
        assert_eq!(data.render_size, [1920, 1080]);
        assert_eq!(data.output_size, [1920, 1080]);
        // The default exposures differ (star EV −4, scene EV 0): the dual exposure is labelled.
        assert!(data.eye_adapted_stars);
        assert_eq!(data.nearest_name, "TestBH-1");
        let m = gravitational_radius(10.0 * M_SUN);
        assert!((data.nearest_dist_m - 30.0 * m).abs() < 1.0);
        assert!(!data.inside_horizon);
        assert_eq!(data.star_count, 9096);
    }

    #[test]
    fn beyond_the_trace_reach_the_hud_warns_and_the_lens_does_not_run() {
        let (_, ctx) = ctx_near_hole(1e5);
        assert_eq!(uni_rel::lens_coverage(&ctx), LensCoverage::BeyondReach);
        assert!(!uni_rel::debug_background_draws(
            uni_rel::lens_coverage(&ctx),
            LensEnv::DebugGrid
        ));
        assert_eq!(lens_warnings(&ctx), [uni_rel::BEYOND_REACH_WARNING]);
    }

    #[test]
    fn from_the_ssb_no_lens_is_on_screen() {
        // TestBH-1 is ~1e10 M away: its light bending is far under half a pixel, so the lens
        // pass is skipped and the sky shows black space with stars (lead ruling, wave 2).
        let world = World::new(Epoch::J2000);
        for orient in [
            DQuat::IDENTITY,
            DQuat::from_rotation_y(-std::f64::consts::FRAC_PI_2),
        ] {
            let ctx = frame_ctx(
                &world,
                &view_at(UniversePos::default(), orient),
                UVec2::new(1920, 1080),
                UVec2::new(1920, 1080),
                exposure0(),
                0,
                0.0,
            );
            assert_eq!(uni_rel::lens_coverage(&ctx), LensCoverage::Unlensed);
            assert!(lens_warnings(&ctx).is_empty());
        }
    }

    #[test]
    fn inside_the_horizon_the_hud_says_so() {
        // Review Focus 2: inside, exactly on the horizon (r = 2 M) and at the centre.
        for r in [1.5, 2.0, 0.0] {
            let (world, ctx) = ctx_near_hole(r);
            assert_eq!(
                uni_rel::lens_coverage(&ctx),
                LensCoverage::InsideHorizon,
                "r = {r} M"
            );
            let perf = Perf {
                fps: 0.0,
                cpu_ms: 0.0,
                gpu_ms: Vec::new(),
                gpu_frame_ms: None,
                gpu_timing_available: false,
            };
            let data = hud_data(&world, &ctx, perf, 0.0, 0, &[], None);
            assert!(data.inside_horizon, "r = {r} M");
            // All black: no grid, so no grid labels.
            assert!(!data.lens_on_screen, "r = {r} M");
            // Default exposure: the dual-exposure label is up.
            assert!(data.eye_adapted_stars);
            assert!(data.warnings.is_empty(), "{:?}", data.warnings);
            assert!(data.nearest_dist_m.is_finite() && ctx.view_rot.is_finite());
        }
    }

    /// Review m7: the HUD's horizon and lens labels come from the same lens the lens pass
    /// traces (the first), so they cannot disagree with the frame when there are two.
    #[test]
    fn hud_lens_state_follows_the_traced_lens() {
        let (world, mut ctx) = ctx_near_hole(30.0);
        let mut inside = ctx.lenses[0].clone();
        inside.rel_pos_m = DVec3::ZERO; // the camera at a second hole's centre
        ctx.lenses.push(inside);
        let perf = || Perf {
            fps: 0.0,
            cpu_ms: 0.0,
            gpu_ms: Vec::new(),
            gpu_frame_ms: None,
            gpu_timing_available: false,
        };
        let data = hud_data(&world, &ctx, perf(), 0.0, 0, &[], None);
        assert_eq!(uni_rel::lens_coverage(&ctx), LensCoverage::Traced);
        assert!(!data.inside_horizon);
        assert!(data.lens_on_screen);
        ctx.lenses.swap(0, 1);
        let data = hud_data(&world, &ctx, perf(), 0.0, 0, &[], None);
        assert_eq!(uni_rel::lens_coverage(&ctx), LensCoverage::InsideHorizon);
        assert!(data.inside_horizon);
        assert!(!data.lens_on_screen);
    }

    fn bookmark(view: CameraView) -> Bookmark {
        Bookmark {
            name: "b".to_owned(),
            view,
            epoch_utc: None,
            scene_ev100: None,
            star_ev100: None,
            lens_scale: None,
            anchor: None,
            look_at: None,
            up: None,
            time_rate: None,
            exposure: None,
            eye_adapted: None,
            target: None,
            lens_env: None,
            cruise: None,
            highlight_belt: None,
        }
    }

    #[test]
    fn apply_bookmark_sets_only_what_it_names() {
        let mut world = World::new(Epoch::J2000);
        let mut controls = default_controls(0.5);
        let far = UniversePos::from_meters(DVec3::new(1e15, 0.0, 0.0));
        let unnormalised = UniversePos {
            cell: glam::I64Vec3::ZERO,
            local: far.delta_from(UniversePos::default()),
        };
        let mut b = bookmark(view_at(unnormalised, DQuat::from_xyzw(0.0, 0.0, 0.0, 2.0)));
        let view = apply_bookmark(&b, &mut world, &mut controls).unwrap();
        assert_eq!(view.pos, far);
        assert_eq!(view.orient, DQuat::IDENTITY);
        assert_eq!(controls, default_controls(0.5));
        assert_eq!(world.epoch, Epoch::J2000);

        b.epoch_utc = Some("2026-11-20T00:00:00Z".to_owned());
        b.scene_ev100 = Some(3.0);
        b.star_ev100 = Some(-1.0);
        b.lens_scale = Some(1.0);
        apply_bookmark(&b, &mut world, &mut controls).unwrap();
        assert_eq!(
            world.epoch,
            Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap()
        );
        assert_eq!(
            (
                controls.scene_ev100,
                controls.star_ev100,
                controls.lens_scale
            ),
            (3.0, -1.0, 1.0)
        );
    }

    #[test]
    fn apply_bookmark_rejects_views_that_would_put_nan_on_the_gpu() {
        let mut world = World::new(Epoch::J2000);
        let mut controls = default_controls(0.5);
        let ok = view_at(UniversePos::default(), DQuat::IDENTITY);
        let mut bad = Vec::new();
        bad.push(bookmark(CameraView {
            orient: DQuat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            ..ok.clone()
        }));
        bad.push(bookmark(CameraView {
            fov_y_rad: 0.0,
            ..ok.clone()
        }));
        bad.push(bookmark(CameraView {
            fov_y_rad: f64::NAN,
            ..ok.clone()
        }));
        let mut nan_pos = ok.clone();
        nan_pos.pos.local.x = f64::NAN;
        bad.push(bookmark(nan_pos));
        bad.push(bookmark(CameraView {
            velocity_mps: DVec3::new(0.0, f64::INFINITY, 0.0),
            ..ok.clone()
        }));
        // A finite velocity (a travel mode's, saved with the view) is kept.
        let moving = CameraView {
            velocity_mps: DVec3::new(0.0, 3.0e4, 0.0),
            ..ok.clone()
        };
        let applied = apply_bookmark(&bookmark(moving.clone()), &mut world, &mut controls);
        assert_eq!(applied.unwrap().velocity_mps, moving.velocity_mps);
        let mut bad_epoch = bookmark(ok);
        bad_epoch.epoch_utc = Some("yesterday".to_owned());
        bad.push(bad_epoch);
        for b in &bad {
            let err = apply_bookmark(b, &mut world, &mut controls).unwrap_err();
            assert!(format!("{err:#}").contains("bookmark \"b\""), "{err:#}");
        }
        assert_eq!(world.epoch, Epoch::J2000);
    }

    // ---- sky v1 (`--sky v1`) ---------------------------------------------------------

    #[test]
    fn sky_mode_parses_and_defaults_to_m0() {
        assert_eq!(SkyMode::default(), SkyMode::M0);
        assert_eq!(SkyMode::parse("v1"), Ok(SkyMode::V1));
        assert_eq!(SkyMode::parse("m0"), Ok(SkyMode::M0));
        for bad in ["", "V1", "v2", "band"] {
            assert!(SkyMode::parse(bad).is_err(), "{bad:?}");
        }
    }

    /// B9 Step 1's app tests: the night EV the meter falls back to is the default star EV, and
    /// the resolve's clamp is the app's.
    #[test]
    fn auto_exposure_constants_match_the_app() {
        assert_eq!(uni_post::NIGHT_EV100, uni_sky::DEFAULT_STAR_EV100);
        assert_eq!(uni_post::EV100_MIN..=uni_post::EV100_MAX, EV100_LIMITS);
    }

    #[test]
    fn star_and_band_inputs_follow_the_camera_and_epoch() {
        // The catalogue epoch is J2000.0 TDB (JD 2451545.0), which is `Epoch::J2000`.
        assert_eq!(uni_data::STAR3_EPOCH_JD_TDB, 2_451_545.0);
        let world = World::new(Epoch::J2000.add_seconds(10.0 * JULIAN_YEAR_S));
        let pos = UniversePos::from_meters(DVec3::new(2.0 * PC, -PC, 0.5 * PC));
        let ctx = frame_ctx(
            &world,
            &view_at(pos, DQuat::IDENTITY),
            UVec2::new(64, 36),
            UVec2::new(64, 36),
            exposure0(),
            0,
            0.0,
        );
        let s = star_input(&ctx);
        assert!(
            (s.years_since_epoch - 10.0).abs() < 1e-12,
            "{}",
            s.years_since_epoch
        );
        assert!(
            s.camera_pc.abs_diff_eq(DVec3::new(2.0, -1.0, 0.5), 1e-12),
            "{}",
            s.camera_pc
        );
        // Owner feedback: vivid colours on by default; no twinkle in space; nothing suppressed.
        assert!(s.vivid_colours);
        assert_eq!(s.scintillation, 0.0);
        assert!(s.suppressed.is_empty());
        let b = band_input(&ctx, ATHYG_V_LIM);
        assert!(b.enabled);
        assert_eq!(b.v_lim, 10.0);
        assert_eq!(b.camera_icrs_pc, s.camera_pc);
    }

    #[test]
    fn sky_exposure_comes_from_the_bookmark_with_auto_and_eye_adapted_defaults() {
        let mut b = bookmark(view_at(UniversePos::default(), DQuat::IDENTITY));
        assert_eq!(SkyExposure::from_bookmark(&b), SkyExposure::default());
        assert_eq!(
            SkyExposure::default(),
            SkyExposure {
                mode: ExposureMode::Auto,
                eye_adapted: true
            }
        );
        b.exposure = Some(ExposureChoice::Manual);
        b.eye_adapted = Some(false);
        assert_eq!(
            SkyExposure::from_bookmark(&b),
            SkyExposure {
                mode: ExposureMode::Manual,
                eye_adapted: false
            }
        );
    }

    #[test]
    fn exposure_settings_follow_the_controls_and_choices() {
        let mut c = default_controls(1.0);
        c.scene_ev100 = 3.0;
        c.star_ev100 = -5.0;
        let s = exposure_settings(&c, SkyExposure::default(), true);
        assert_eq!(s.mode, ExposureMode::Auto);
        assert!(s.eye_adapted && !s.daylight && s.settle);
        assert!(s.eye_adapted_in_effect());
        assert_eq!(s.eye_star_ev100, -5.0);
        assert_eq!(s.compensation_ev, 0.0);
        assert_eq!(s.adapt_tau_s, uni_post::exposure::DEFAULT_ADAPT_TAU_S);
        assert_eq!(
            s.manual,
            Exposure {
                scene_ev100: 3.0,
                star_ev100: -5.0
            }
        );
        // With eye-adapted stars off, the stars take the scene EV (B6 decision D4), and so does
        // `FrameCtx::exposure`.
        let off = SkyExposure {
            mode: ExposureMode::Manual,
            eye_adapted: false,
        };
        let s = exposure_settings(&c, off, false);
        assert_eq!(s.mode, ExposureMode::Manual);
        assert!(!s.eye_adapted_in_effect() && !s.settle);
        assert_eq!(s.manual.star_ev100, 3.0);
        // Out-of-range sliders are clamped as in M0 (review m5).
        c.scene_ev100 = 1e30;
        c.star_ev100 = f32::NAN;
        let s = exposure_settings(&c, SkyExposure::default(), false);
        assert_eq!(s.manual.scene_ev100, *EV100_LIMITS.end());
        assert_eq!(s.eye_star_ev100, uni_sky::DEFAULT_STAR_EV100);
        // The resolve's CPU twin gives the uniform the GPU should write: star EV −4 at night.
        let u = uni_post::resolve_exposure(&s, Some(uni_post::NIGHT_EV100));
        assert_eq!(u.ev100[0], uni_post::NIGHT_EV100);
        assert_eq!(u.ev100[1], uni_sky::DEFAULT_STAR_EV100);
    }

    #[test]
    fn sky_v1_hud_shows_its_labels_and_no_grid_lines() {
        let perf = || Perf {
            fps: 0.0,
            cpu_ms: 0.0,
            gpu_ms: Vec::new(),
            gpu_frame_ms: None,
            gpu_timing_available: false,
        };
        let band = uni_sim::honesty::label(HonestyFlag::BandModel).to_owned();
        let vivid = uni_ui::VIVID_STAR_COLOURS_LABEL.to_owned();
        let on = exposure_settings(&default_controls(1.0), SkyExposure::default(), false);
        // Near the hole: the lens is traced over the band; the stars over it are unlensed.
        let (world, ctx) = ctx_near_hole(30.0);
        let data = hud_data(&world, &ctx, perf(), 0.0, 1, &["n".to_owned()], Some(&on));
        assert!(!data.lens_on_screen);
        assert!(data.eye_adapted_stars);
        assert_eq!(
            data.warnings,
            [
                "n".to_owned(),
                UNLENSED_STARS_LABEL.to_owned(),
                vivid.clone(),
                band.clone()
            ]
        );
        // From the SSB: no lens on screen, so no lens line.
        let ctx = frame_ctx(
            &world,
            &view_at(UniversePos::default(), DQuat::IDENTITY),
            UVec2::new(1920, 1080),
            UVec2::new(1920, 1080),
            on.manual,
            0,
            0.0,
        );
        let data = hud_data(&world, &ctx, perf(), 0.0, 1, &[], Some(&on));
        assert_eq!(data.warnings, [vivid, band]);
        // Eye-adapted off: no label, even though `FrameCtx::exposure` is the same EV pair.
        let off = exposure_settings(
            &default_controls(1.0),
            SkyExposure {
                mode: ExposureMode::Auto,
                eye_adapted: false,
            },
            false,
        );
        assert!(!hud_data(&world, &ctx, perf(), 0.0, 1, &[], Some(&off)).eye_adapted_stars);
    }
}
