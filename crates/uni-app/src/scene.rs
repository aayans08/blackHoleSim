//! The render stack and per-frame data shared by the window, `--capture` and `--bench`
//! (M1 Task B9).
//!
//! [`Scene`] owns the [`Renderer`] with every Wave B feature in the M1 frame contract's order
//! (CLAUDE.md §2; uni-gfx `targets` documents it):
//! 1. `exposure` (B6 resolve) → 2. `band-update` (B5) → 3. `background` (B7, the M0 debug grid,
//!    only with `LensEnv::DebugGrid` and a traced lens) → 4. `sun` (B2 star surfaces) →
//!    5. `bodies` (B1) → 6. `rings` (B2; only when Saturn's ring profile loaded, ruling A.3) →
//!    (7. comets: Wave C) → 8. `atmosphere` (B3) → 9. `histogram` (B6 meter) → 10. `lens` (B7)
//!    → 11. `band` (B5) → 12. `stars` (B4 AT-HYG in 3-D, or M0's BSC5 field without AT-HYG) →
//!    13. `points` (B4) → (14. asteroids: Wave C) → 15. `bloom` (B6) → 16. `tonemap`.
//!
//! Two capture taps ([`crate::tap`]) sit after the meter and before bloom; they encode nothing
//! unless a capture arms them.
//!
//! Every frame, [`Scene::prepare`] takes the world's [`Snapshot`] for the camera and sets every
//! shared input before [`Scene::render`] runs the renderer (frame contract, "Shared inputs"):
//! the bodies and star surfaces ([`BodyScene`]), the atmospheres ([`AtmoScene`], each with its
//! body's exposure weight), the star pass ([`StarInput`]: camera, epoch, suppressed near stars,
//! twinkle inside an atmosphere), the point sprites, the band, the lens environment and B6's
//! [`ExposureSettings`] (`daylight` from the camera's atmosphere). The Sun becomes
//! `FrameCtx::lights`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::rc::Rc;
use std::time::Instant;

use anyhow::Context as _;
use glam::{DMat4, DVec3, UVec2, Vec2, Vec3};
use uni_core::{CameraView, DAY, Epoch, PC, UniversePos};
use uni_data::{CatalogStar, Star3};
use uni_gfx::{
    Exposure, FrameBindings, FrameCtx, FrameTargets, GpuContext, LightView, PassTimer,
    RenderFeature, Renderer, ShaderLib, SharedInput, reversed_z_infinite_proj,
};
use uni_nav::{Bookmark, ExposureChoice, LensEnvChoice};
use uni_planet::{BodyRenderer, BodyScene, RingRenderer, StarSurfaceRenderer};
use uni_post::{ExposureMode, ExposureReadback, ExposureSettings};
use uni_rel::{DebugSkyBackground, LensCoverage, LensEnv, LensScale, LensTracer};
use uni_sim::honesty::HonestyFlag;
use uni_sim::registry::ObjectKind;
use uni_sim::snapshot::{PointView, Snapshot};
use uni_sky::{BandInput, PointSprite, StarInput};
use uni_ui::{HudControls, HudData};

use crate::tap::HdrTap;
use crate::world::{Assets, World};

/// Near plane of the reversed-Z infinite projection [m]. B1's cap mesh is fine above it
/// (B1 report §7 item 4: a camera less than 0.1 m above a surface clips the cap's apex).
pub const NEAR_PLANE_M: f32 = 0.1;
/// Manual scene exposure when nothing sets it [EV100]; Auto starts from it.
pub const DEFAULT_SCENE_EV100: f32 = 0.0;
/// The eye-adapted star exposure [EV100] (lead ruling "B4L exposure"): a typical Milky Way
/// photograph (ISO 3200, f/2.8, 20 s: EV100 = log2(2.8²/20) − log2(3200/100) ≈ −6.35). M0's
/// value, `uni_sky::DEFAULT_STAR_EV100` (−4), is kept for the M0 debug-grid bookmark.
pub const DEFAULT_STAR_EV100: f32 = -6.5;
/// Lens tracer resolution scale in the window (the HUD slider); the tracer's own default.
pub const INTERACTIVE_LENS_SCALE: f32 = uni_rel::DEFAULT_RESOLUTION_SCALE;
/// Lens tracer resolution scale in captures, unless a bookmark sets one.
pub const CAPTURE_LENS_SCALE: f32 = 1.0;
/// Lines of the window's internal render resolution (design §6: "1080p internal").
pub const INTERNAL_LINES: u32 = 1080;
/// The exposures the renderer accepts [EV100] (review m5): a typed slider value outside this is
/// clamped, so 2^EV never overflows and no inf or NaN reaches the HDR target.
pub const EV100_LIMITS: RangeInclusive<f32> = -30.0..=40.0;
/// Faintest V the star pass draws from AT-HYG: the bake keeps V ≤ 10 (uni-data `athyg`, the
/// `athyg_40_reduced_m10` subset). Fainter starlight goes into the band (`BandInput::v_lim`).
pub const ATHYG_V_LIM: f32 = 10.0;
/// Faintest V of the BSC5 fallback: the Yale Bright Star Catalogue lists the stars brighter than
/// about V 6.5 (Hoffleit & Warren 1991, 5th revised ed., CDS V/50).
pub const BSC5_V_LIM: f32 = 6.5;
/// A Julian year [s] (IAU: 365.25 d), the unit of `StarInput::years_since_epoch`.
pub const JULIAN_YEAR_S: f64 = 365.25 * DAY;
/// While a lens is traced: the stars over it are drawn unlensed until M3 (the plan's M1
/// wording; C4 moves the string into `uni_ui::theme`).
pub const UNLENSED_STARS_LABEL: &str = "Stars near the hole are not yet lensed (M3)";
/// While the debug grid shows through a traced lens (ruling 14; m0.1's string).
pub const DEBUG_GRID_LABEL: &str = "Lens background: M0 debug grid";
/// While the camera is inside an atmosphere and the stars twinkle (B4 report, label request for
/// C4: Young's law extrapolated below its aperture range, every star at the zenith).
pub const TWINKLE_LABEL: &str = "Twinkling: approximate model";
/// While one of Saturn's major moons is on screen (A2 ruling I3, ruling 10: Meeus ch. 46 puts
/// Mimas up to 2.45 px and Dione 1.16 px off at the Saturn bookmarks).
pub const SATURN_MOONS_LABEL: &str = "Saturn's major moons: approximate position (Meeus 46)";
/// NAIF ids of Saturn's eight major moons (Mimas … Iapetus), Meeus ch. 46's.
pub const SATURN_MAJOR_MOONS: [i32; 8] = [601, 602, 603, 604, 605, 606, 607, 608];

/// The star catalogue a scene draws.
pub enum StarCatalog<'a> {
    /// AT-HYG in 3-D (`uni_sim::World::star_data()`), drawn by `uni_sky::StarField3`.
    AtHyg(&'a [Star3]),
    /// The Yale Bright Star Catalogue (M0's `uni_sky::StarField`): the window's fallback when
    /// AT-HYG is missing (B9 Step 1), with a HUD notice.
    Bsc5(&'a [CatalogStar]),
}

/// A view's render choices beyond the HUD's sliders, which a bookmark can set (uni-nav's M1
/// fields `exposure`, `eye_adapted` and `lens_env`): Auto exposure with eye-adapted stars and the
/// band behind a lens by default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewChoices {
    pub exposure: ExposureMode,
    /// The "Eye-adapted stars" option: stars at the star EV slider's value.
    pub eye_adapted: bool,
    pub lens_env: LensEnv,
}

impl Default for ViewChoices {
    fn default() -> Self {
        Self {
            exposure: ExposureMode::Auto,
            eye_adapted: true,
            lens_env: LensEnv::Band,
        }
    }
}

impl ViewChoices {
    /// The defaults, overridden by what `b` names.
    pub fn from_bookmark(b: &Bookmark) -> Self {
        let d = Self::default();
        Self {
            exposure: match b.exposure {
                Some(ExposureChoice::Auto) => ExposureMode::Auto,
                Some(ExposureChoice::Manual) => ExposureMode::Manual,
                None => d.exposure,
            },
            eye_adapted: b.eye_adapted.unwrap_or(d.eye_adapted),
            lens_env: match b.lens_env {
                Some(LensEnvChoice::DebugGrid) => LensEnv::DebugGrid,
                Some(LensEnvChoice::Band) | None => d.lens_env,
            },
        }
    }

    /// The exposure mode in effect: Manual whenever the debug grid is selected, because the
    /// grid is display-referred and must not be metered (frame contract, pass 3).
    pub fn mode(&self) -> ExposureMode {
        if self.lens_env == LensEnv::DebugGrid {
            ExposureMode::Manual
        } else {
            self.exposure
        }
    }
}

/// B6's settings for one frame from the HUD controls (clamped as [`exposure`] does), the view's
/// choices and the camera's daylight: the scene slider is the manual scene EV (Auto starts from
/// it), the star slider the eye-adapted star EV. `settle` jumps straight to the metered target
/// (captures).
pub fn exposure_settings(
    controls: &HudControls,
    choice: ViewChoices,
    daylight: bool,
    settle: bool,
) -> ExposureSettings {
    let e = exposure(controls);
    let mut s = ExposureSettings {
        mode: choice.mode(),
        manual: e,
        compensation_ev: 0.0,
        eye_adapted: choice.eye_adapted,
        eye_star_ev100: e.star_ev100,
        daylight,
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

/// Camera − SSB [pc], ICRS axes (f64).
pub fn camera_icrs_pc(pos: &UniversePos) -> DVec3 {
    pos.delta_from(UniversePos::default()) / PC
}

/// Rec. 709 luminance (ITU-R BT.709-6 item 3.2).
pub fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// `c` scaled to luminance 1 (white if `c` has none).
fn unit_luminance(c: Vec3) -> Vec3 {
    let y = luminance(c);
    if y > 0.0 && y.is_finite() {
        c / y
    } else {
        Vec3::ONE
    }
}

/// Lets the app keep a handle on a feature it pushed into the renderer (for its warnings).
struct Shared<F>(Rc<RefCell<F>>);

impl<F: RenderFeature> RenderFeature for Shared<F> {
    fn name(&self) -> &'static str {
        self.0.borrow().name()
    }
    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets) {
        self.0.borrow_mut().prepare(gpu, ctx, targets);
    }
    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        self.0
            .borrow_mut()
            .encode(gpu, enc, frame, targets, output, timer);
    }
    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.0.borrow_mut().resize(gpu, targets);
    }
}

/// One frame's inputs, built by [`Scene::prepare`].
pub struct Frame {
    pub ctx: FrameCtx,
    pub snapshot: Snapshot,
    pub settings: ExposureSettings,
    /// The honesty labels in effect (HUD and capture JSON `honesty`), in a fixed order.
    pub labels: Vec<String>,
    /// CPU time of `World::snapshot` [ms].
    pub snapshot_cpu_ms: f32,
    /// The lens coverage this frame.
    pub coverage: LensCoverage,
    /// The factor on reflected-light points' illuminance ([`Scene::prepare`]'s exposure rule
    /// for points beside a resolved body); 1 when none is resolved.
    pub reflected_scale: f64,
}

/// W of the reflected-point rule: the largest `lod_exposure_mix(radius_px)` among the sunlit
/// resolved bodies of `snapshot` (0 with none).
pub fn resolved_weight(snapshot: &Snapshot) -> f32 {
    snapshot
        .bodies
        .iter()
        .filter(|b| b.sun_illuminance_lux > 0.0)
        .map(|b| uni_planet::lod_exposure_mix(b.proj_radius_px))
        .fold(0.0_f32, f32::max)
}

/// `pre_mix(star_pre, scene_pre, w) / star_pre` for the EVs `star_ev100`, `scene_ev100`
/// (clamped as the resolve does): the factor on a reflected point's illuminance that the points
/// pass, which multiplies by `star_pre`, turns into the blended exposure.
pub fn point_exposure_factor(w: f32, star_ev100: f32, scene_ev100: f32) -> f64 {
    let star_pre = uni_gfx::exposure_scale(clamp_ev(star_ev100, DEFAULT_STAR_EV100));
    let scene_pre = uni_gfx::exposure_scale(clamp_ev(scene_ev100, DEFAULT_SCENE_EV100));
    f64::from(uni_gfx::frame::pre_mix(star_pre, scene_pre, w) / star_pre)
}

/// Whether an object drawn as a point shines by reflected sunlight (a planet, moon, dwarf
/// planet, asteroid or comet), not by its own light.
pub fn is_reflected(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::Planet
            | ObjectKind::DwarfPlanet
            | ObjectKind::Moon
            | ObjectKind::Asteroid
            | ObjectKind::Comet
    )
}

/// The renderer with its feature stack, plus the handles the app keeps to steer it.
pub struct Scene {
    pub renderer: Renderer,
    /// Shared with the lens pass and the debug background inside `renderer`.
    lens_scale: LensScale,
    /// Catalogue stars the star pass holds.
    pub star_count: u32,
    /// AT-HYG (true) or the BSC5 fallback.
    athyg: bool,
    /// The band's star limit: [`ATHYG_V_LIM`], or [`BSC5_V_LIM`] with the fallback catalogue.
    v_lim: f32,
    body: SharedInput<BodyScene>,
    atmo: SharedInput<uni_atmo::AtmoScene>,
    stars: SharedInput<StarInput>,
    points: SharedInput<Vec<PointSprite>>,
    band: SharedInput<BandInput>,
    lens_env: SharedInput<LensEnv>,
    exposure: SharedInput<ExposureSettings>,
    /// Bloom's mix weight: `DEFAULT_BLOOM_STRENGTH`, 0 with the debug grid.
    bloom_strength: SharedInput<f32>,
    readback: ExposureReadback,
    /// Mean albedo colour of each mapped body, for tinting its point sprite (carry-over 6).
    albedo_rgb: HashMap<i32, Vec3>,
    /// The atmosphere presets drawn (NAIF id, parameters).
    presets: Vec<(i32, uni_atmo::AtmosphereParams)>,
    bodies_pass: Rc<RefCell<BodyRenderer>>,
    sun_pass: Rc<RefCell<StarSurfaceRenderer>>,
    /// Warnings from building the stack (no ring pass, a fallback catalogue, ...).
    pub notices: Vec<String>,
    /// The pre-bloom HDR tap (the `--hdr` sidecar).
    pub hdr_tap: HdrTap,
    /// The HDR tap right after the meter (world light only, for `lit_px`).
    pub world_tap: HdrTap,
}

impl Scene {
    /// Builds the stack at the internal resolution `internal_size` (see the module docs).
    /// `output_format` is the sRGB format the tonemap writes (the window's sRGB view, or
    /// `uni_gfx::CAPTURE_FORMAT`).
    pub fn new(
        gpu: &GpuContext,
        internal_size: UVec2,
        output_format: wgpu::TextureFormat,
        catalog: StarCatalog<'_>,
        assets: &Assets,
        lens_scale: f32,
    ) -> anyhow::Result<Self> {
        let mut renderer = Renderer::new(gpu, internal_size);
        let mut lib = ShaderLib::with_common();
        let camera_bgl = renderer.camera_bind_group_layout();
        let lens_scale = LensScale::new(lens_scale);
        let mut notices = Vec::new();
        let (athyg, v_lim) = match catalog {
            StarCatalog::AtHyg(_) => (true, ATHYG_V_LIM),
            StarCatalog::Bsc5(_) => (false, BSC5_V_LIM),
        };

        let body = SharedInput::new(BodyScene::default());
        let atmo = SharedInput::new(uni_atmo::AtmoScene::default());
        let stars = SharedInput::new(StarInput::default());
        let points = SharedInput::new(Vec::<PointSprite>::new());
        let band = SharedInput::new(BandInput {
            camera_icrs_pc: DVec3::ZERO,
            enabled: true,
            v_lim,
        });
        let lens_env = SharedInput::new(LensEnv::Band);
        let exposure_in = SharedInput::new(exposure_settings(
            &default_controls(lens_scale.get()),
            ViewChoices::default(),
            false,
            false,
        ));

        // 1, 9: the resolve and the meter.
        let (resolve, meter, readback) = uni_post::auto_exposure(
            gpu,
            &mut lib,
            camera_bgl,
            renderer.exposure_buffer(),
            exposure_in.clone(),
        )
        .context("creating the auto-exposure passes")?;
        // 2, 11: the band cubemap and its draw.
        let band_cube = uni_sky::BandCubemap::new(
            gpu,
            &mut lib,
            &uni_procgen::galaxy::GalaxyModel::milky_way(),
            band.clone(),
        )
        .context("creating the band cubemap pass")?;
        let band_draw =
            uni_sky::BandDraw::new(gpu, &mut lib, camera_bgl, band_cube.env_map(), band.clone())
                .context("creating the band pass")?;
        // 3, 10: the debug background and the lens.
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
            band_cube.env_map(),
            lens_env.clone(),
        )
        .context("creating the lens pass")?;
        // 8: atmospheres (their LUTs are built here), whose ground lighting the bodies share.
        let presets: Vec<(i32, uni_atmo::AtmosphereParams)> = [299, 399, 499, 606]
            .into_iter()
            .filter_map(|n| uni_atmo::preset_for(n).map(|p| (n, p)))
            .collect();
        let atmo_set = uni_atmo::AtmosphereSet::new(gpu, &mut lib, &presets)
            .context("building the atmosphere LUTs")?;
        let atmosphere = uni_atmo::AtmosphereFeature::new(
            gpu,
            &mut lib,
            camera_bgl,
            &atmo_set,
            atmo.clone(),
            gpu.has_dual_source_blending(),
        )
        .context("creating the atmosphere pass")?;
        // 4, 5, 6: star surfaces, bodies, rings.
        let sun_pass = Rc::new(RefCell::new(
            StarSurfaceRenderer::new(gpu, &mut lib, camera_bgl, body.clone())
                .context("creating the star-surface pass")?,
        ));
        let bodies_pass = Rc::new(RefCell::new(
            BodyRenderer::new(
                gpu,
                &mut lib,
                camera_bgl,
                &assets.maps,
                &assets.map_dir,
                &assets.facts,
                assets.rings.as_ref(),
                atmo_set.ground_lighting(),
                body.clone(),
            )
            .context("creating the bodies pass")?,
        ));
        let rings = match &assets.rings {
            Some(profile) => Some(
                RingRenderer::new(gpu, &mut lib, camera_bgl, profile, body.clone())
                    .context("creating the ring pass")?,
            ),
            // `world::load` already put "Saturn's ring profile missing" in `Assets::warnings`,
            // which the HUD and the capture notes show beside `notices` (review B9 r0 minor 3:
            // the line appeared twice).
            None => None,
        };
        let albedo_rgb = {
            let b = bodies_pass.borrow();
            assets
                .maps
                .iter()
                .filter_map(|m| b.mean_albedo_rgb(m.naif).map(|c| (m.naif, c)))
                .collect()
        };
        // 12, 13: stars and points.
        let (star_field, star_count): (Box<dyn RenderFeature>, u32) = match catalog {
            StarCatalog::AtHyg(list) => {
                let f = uni_sky::StarField3::new(gpu, &mut lib, camera_bgl, list, stars.clone())
                    .context("creating the 3-D star pass")?;
                let n = f.count();
                (Box::new(f), n)
            }
            StarCatalog::Bsc5(list) => {
                notices.push("No AT-HYG stars: the sky shows the BSC5 stars instead".to_owned());
                let f = uni_sky::StarField::new(gpu, &mut lib, camera_bgl, list)
                    .context("creating the BSC5 star pass")?;
                let n = f.count();
                (Box::new(f), n)
            }
        };
        let point_pass = uni_sky::PointSprites::new(gpu, &mut lib, camera_bgl, points.clone())
            .context("creating the point-sprite pass")?
            .with_star_input(stars.clone());
        // 15, 16.
        let bloom_strength = SharedInput::new(uni_post::DEFAULT_BLOOM_STRENGTH);
        let bloom = uni_post::Bloom::new(gpu, &mut lib, camera_bgl, bloom_strength.clone())
            .context("creating the bloom pass")?;
        let tonemap = uni_post::TonemapFeature::new(gpu, &mut lib, camera_bgl, output_format)
            .context("creating the tonemap pass")?;
        let (world_tap, world_tap_pass) =
            HdrTap::new(gpu, crate::tap::TAP_WORLD).context("creating the world HDR tap")?;
        let (hdr_tap, hdr_tap_pass) =
            HdrTap::new(gpu, crate::tap::TAP_PREBLOOM).context("creating the pre-bloom HDR tap")?;

        renderer.push(Box::new(resolve));
        renderer.push(Box::new(band_cube));
        renderer.push(Box::new(background));
        renderer.push(Box::new(Shared(Rc::clone(&sun_pass))));
        renderer.push(Box::new(Shared(Rc::clone(&bodies_pass))));
        if let Some(r) = rings {
            renderer.push(Box::new(r));
        }
        renderer.push(Box::new(atmosphere));
        renderer.push(Box::new(meter));
        renderer.push(Box::new(world_tap_pass));
        renderer.push(Box::new(tracer));
        renderer.push(Box::new(band_draw));
        renderer.push(star_field);
        renderer.push(Box::new(point_pass));
        renderer.push(Box::new(hdr_tap_pass));
        renderer.push(Box::new(bloom));
        renderer.push(Box::new(tonemap));
        Ok(Self {
            renderer,
            lens_scale,
            star_count,
            athyg,
            v_lim,
            body,
            atmo,
            stars,
            points,
            band,
            lens_env,
            exposure: exposure_in,
            bloom_strength,
            readback,
            albedo_rgb,
            presets,
            bodies_pass,
            sun_pass,
            notices,
            hdr_tap,
            world_tap,
        })
    }

    /// Lens tracer resolution scale from the next frame on (clamped to 0.25..=1).
    pub fn set_lens_scale(&self, s: f32) {
        self.lens_scale.set(s);
    }

    /// The lens tracer resolution scale in use.
    pub fn lens_scale(&self) -> f32 {
        self.lens_scale.get()
    }

    /// The band's faint-star limit [V].
    pub fn v_lim(&self) -> f32 {
        self.v_lim
    }

    /// The [scene, star] EV100 the GPU resolved, read back at least one frame late (after a
    /// headless frame's wait: that frame's); `None` before the first read-back.
    pub fn resolved_ev100(&self) -> Option<[f32; 2]> {
        self.readback.last_ev100()
    }

    /// The bodies' and star surfaces' warnings from the last prepared frame (a missing map, a
    /// camera inside a body or star).
    pub fn pass_warnings(&self) -> Vec<String> {
        let mut w = self.bodies_pass.borrow().warnings();
        w.extend(self.sun_pass.borrow().warnings());
        w
    }

    /// Builds the frame for `view` and sets every shared input (see the module docs). Call it
    /// right before [`Scene::render`].
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        world: &World,
        view: &CameraView,
        viewport: UVec2,
        output_size: UVec2,
        controls: &HudControls,
        choice: ViewChoices,
        settle: bool,
        frame_index: u64,
        dt_real_s: f64,
    ) -> Frame {
        let viewport = viewport.max(UVec2::ONE);
        let t0 = Instant::now();
        let snapshot = world.sim.snapshot(view, viewport);
        let snapshot_cpu_ms = t0.elapsed().as_secs_f32() * 1e3;

        // Atmospheres of resolved bodies (every body in the snapshot is ≥ POINT_LOD_PX, B10
        // carry), and the camera's daylight and twinkle from them.
        let mut atmos: Vec<uni_atmo::AtmoDraw> = snapshot
            .bodies
            .iter()
            .filter(|b| b.has_atmosphere && b.proj_radius_px >= uni_sim::snapshot::POINT_LOD_PX)
            .filter_map(|b| {
                let naif = b.naif?;
                self.presets.iter().any(|(n, _)| *n == naif).then(|| {
                    uni_atmo::AtmoDraw {
                        naif,
                        rel_pos_m: b.rel_pos_m,
                        orient: b.orient,
                        sun_rel_m: b.sun_rel_m,
                        sun_illuminance_lux: b.sun_illuminance_lux,
                        exposure_mix: 1.0, // set below, once `daylight` is known
                    }
                })
            })
            .collect();
        let preset = |naif: i32| self.presets.iter().find(|(n, _)| *n == naif).map(|p| &p.1);
        let daylight = atmos
            .iter()
            .any(|a| preset(a.naif).is_some_and(|p| a.camera_in_daylight(p)));
        let scintillation = atmos
            .iter()
            .filter_map(|a| preset(a.naif).map(|p| a.scintillation_strength(p)))
            .fold(0.0_f32, f32::max);

        let lens_env = choice.lens_env;
        let settings = exposure_settings(controls, choice, daylight, settle);
        let eye = settings.eye_adapted_in_effect();
        for a in &mut atmos {
            let px = snapshot
                .bodies
                .iter()
                .find(|b| b.naif == Some(a.naif))
                .map_or(f64::INFINITY, |b| b.proj_radius_px);
            // Lead ruling B6 I1 / B10: the atmosphere shares its body's exposure weight.
            a.exposure_mix = uni_planet::body::exposure_weight(px, eye);
        }

        let ctx = frame_ctx(
            world,
            view,
            viewport,
            output_size,
            settings.manual,
            frame_index,
            dt_real_s,
            &snapshot,
        );
        let coverage = uni_rel::lens_coverage(&ctx);
        // The band is the lens environment; with the display-referred debug grid it is not
        // drawn (B7 report: it would add star-exposure light over the grid).
        let band_on = lens_env == LensEnv::Band;

        let names = snapshot
            .bodies
            .iter()
            .map(|b| b.id)
            .chain(snapshot.star_surfaces.iter().map(|s| s.id))
            .map(|id| (id, world.name(id).to_owned()))
            .collect();
        self.body.set(BodyScene {
            bodies: snapshot.bodies.clone(),
            stars: snapshot.star_surfaces.clone(),
            eye_adapted: eye,
            names,
        });
        self.atmo.set(uni_atmo::AtmoScene { atmos });
        let vivid = true; // owner feedback: on by default, labelled
        self.stars.set(StarInput {
            years_since_epoch: ctx.epoch.seconds_since(Epoch::J2000) / JULIAN_YEAR_S,
            camera_pc: camera_icrs_pc(&view.pos),
            suppressed: if self.athyg {
                snapshot.suppressed_stars.clone()
            } else {
                Vec::new()
            },
            vivid_colours: vivid,
            scintillation,
        });
        let reflected_scale = self.reflected_point_scale(&snapshot, &settings);
        self.points.set(
            snapshot
                .points
                .iter()
                .map(|p| self.point_sprite(world, p, vivid, reflected_scale))
                .collect(),
        );
        self.band.set(BandInput {
            camera_icrs_pc: camera_icrs_pc(&view.pos),
            enabled: band_on,
            v_lim: self.v_lim,
        });
        self.lens_env.set(lens_env);
        self.exposure.set(settings);
        // The debug grid is display-referred: like the meter, bloom must not act on it (its
        // glow would fill the shadow the M0 regression measures, bookmark 18).
        self.bloom_strength.set(if lens_env == LensEnv::DebugGrid {
            0.0
        } else {
            uni_post::DEFAULT_BLOOM_STRENGTH
        });

        let labels = honesty_labels(
            world,
            view,
            &snapshot,
            &settings,
            lens_env,
            coverage,
            band_on,
            scintillation,
            vivid,
        );
        Frame {
            ctx,
            snapshot,
            settings,
            labels,
            snapshot_cpu_ms,
            coverage,
            reflected_scale,
        }
    }

    /// The factor on a reflected-light point's illuminance that draws it at
    /// `pre_mix(star_pre, scene_pre, W)` instead of `star_pre` (lead ruling at B9: a moon beside
    /// a resolved, metered planet is otherwise drawn 15+ stops overexposed): W is the largest
    /// `lod_exposure_mix(radius_px)` among the resolved bodies (0 with none, so a lone point
    /// stays at star exposure like a star). The points pass multiplies by `star_pre`, so the
    /// factor is `pre_mix(star_pre, scene_pre, W) / star_pre`; with eye-adapted stars not in
    /// effect the two exposures are equal and it is 1. The exposures are this frame's in Manual
    /// mode, and the last read-back scene EV in Auto (one frame late while adapting).
    fn reflected_point_scale(&self, snapshot: &Snapshot, settings: &ExposureSettings) -> f64 {
        if !settings.eye_adapted_in_effect() {
            return 1.0;
        }
        let w = resolved_weight(snapshot);
        if !(w > 0.0) {
            return 1.0;
        }
        let scene_ev = match settings.mode {
            ExposureMode::Manual => settings.manual.scene_ev100,
            ExposureMode::Auto => self
                .readback
                .last_ev100()
                .map_or(settings.manual.scene_ev100, |e| e[0]),
        };
        point_exposure_factor(w, settings.eye_star_ev100, scene_ev)
    }

    /// The point sprite of `p`: reflected light (a body) tinted by its map's mean albedo colour
    /// with its luminance kept (carry-over 6); starlight (the Sun, catalogue stars) in the star
    /// pass's colour and through `vivid_color` while vivid colours are on (B4 notes for B9).
    fn point_sprite(
        &self,
        world: &World,
        p: &PointView,
        vivid: bool,
        reflected_scale: f64,
    ) -> PointSprite {
        let info = p.id.and_then(|id| world.sim.objects().get(id as usize));
        let scale = if info.is_some_and(|o| is_reflected(o.kind)) {
            reflected_scale
        } else {
            1.0
        };
        let solar = unit_luminance(p.color);
        let color = match info {
            Some(o) if matches!(o.kind, ObjectKind::Sun) => {
                if vivid {
                    uni_sky::vivid_color(solar)
                } else {
                    solar
                }
            }
            Some(o) if matches!(o.kind, ObjectKind::Star | ObjectKind::ExoHost) => {
                let c = o
                    .catalog_index
                    .and_then(|i| world.sim.star_data().get(i as usize))
                    .map(|s| s.bv())
                    .filter(|bv| bv.is_finite())
                    .map_or(solar, uni_sky::stars3d::bv_colour);
                if vivid { uni_sky::vivid_color(c) } else { c }
            }
            Some(o) => o
                .naif
                .and_then(|n| self.albedo_rgb.get(&n))
                .map_or(solar, |a| unit_luminance(solar * *a)),
            // An unnamed near star in its first frame (A4: no id yet).
            None => {
                if vivid {
                    uni_sky::vivid_color(solar)
                } else {
                    solar
                }
            }
        };
        PointSprite {
            rel_pos_m: p.rel_pos_m,
            illuminance_lux: p.illuminance_lux * scale,
            color,
        }
    }

    /// Records `frame` into `output` (see [`Renderer::render`]).
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        frame: &Frame,
        output: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        self.renderer.render(gpu, &frame.ctx, output)
    }
}

/// The honesty labels in effect for a frame, in a fixed order: uni-sim's world-state labels
/// (the band's two while it is drawn: ruling, B4L), the eye-adapted and vivid-colour labels,
/// the lens labels, the twinkle label and Saturn's moons' label.
#[allow(clippy::too_many_arguments)]
pub fn honesty_labels(
    world: &World,
    view: &CameraView,
    snapshot: &Snapshot,
    settings: &ExposureSettings,
    lens_env: LensEnv,
    coverage: LensCoverage,
    band_on: bool,
    scintillation: f32,
    vivid: bool,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for f in world.sim.honesty(view) {
        if f == HonestyFlag::BandModel && !band_on {
            continue;
        }
        out.push(uni_sim::honesty::label(f).to_owned());
        if f == HonestyFlag::BandModel {
            out.push(uni_sim::honesty::label(HonestyFlag::MilkyWayDetailProcedural).to_owned());
        }
    }
    if settings.eye_adapted_in_effect() {
        out.push(uni_ui::EYE_ADAPTED_STARS_LABEL.to_owned());
    }
    if vivid {
        out.push(uni_ui::VIVID_STAR_COLOURS_LABEL.to_owned());
    }
    let (grid, unlensed) = uni_rel::lens_labels(coverage, lens_env);
    if grid {
        out.push(DEBUG_GRID_LABEL.to_owned());
    }
    if unlensed {
        out.push(UNLENSED_STARS_LABEL.to_owned());
    }
    if scintillation > 0.0 {
        out.push(TWINKLE_LABEL.to_owned());
    }
    let saturn_moon_shown = snapshot
        .bodies
        .iter()
        .map(|b| b.id)
        .chain(snapshot.points.iter().filter_map(|p| p.id))
        .filter_map(|id| world.sim.objects().get(id as usize)?.naif)
        .any(|n| SATURN_MAJOR_MOONS.contains(&n));
    if saturn_moon_shown {
        out.push(SATURN_MOONS_LABEL.to_owned());
    }
    out
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
/// output of `output_size` pixels, with the Sun of `snapshot` as the light.
#[allow(clippy::too_many_arguments)]
pub fn frame_ctx(
    world: &World,
    view: &CameraView,
    viewport: UVec2,
    output_size: UVec2,
    exposure: Exposure,
    frame_index: u64,
    dt_real_s: f64,
    snapshot: &Snapshot,
) -> FrameCtx {
    let viewport = viewport.max(UVec2::ONE);
    let aspect = viewport.x as f32 / viewport.y as f32;
    let sim_dt_s = dt_real_s * world.time_rate();
    FrameCtx {
        frame_index,
        epoch: world.epoch(),
        dt_real_s,
        camera: view.clone(),
        view_rot: view_rot(view.orient),
        proj: reversed_z_infinite_proj(view.fov_y_rad as f32, aspect, NEAR_PLANE_M),
        viewport,
        output_size: output_size.max(UVec2::ONE),
        exposure,
        lenses: world.lens_views(&view.pos),
        lights: snapshot
            .sun
            .iter()
            .map(|s| LightView {
                rel_pos_m: s.rel_pos_m,
                radius_m: s.radius_m,
                luminous_intensity_cd: s.intensity_cd,
                color: s.color,
            })
            .collect(),
        sim_dt_s: if sim_dt_s.is_finite() { sim_dt_s } else { 0.0 },
        jitter_px: Vec2::ZERO,
    }
}

/// HUD controls at their start values, with the lens tracer at `lens_scale`.
pub fn default_controls(lens_scale: f32) -> HudControls {
    HudControls {
        scene_ev100: DEFAULT_SCENE_EV100,
        star_ev100: DEFAULT_STAR_EV100,
        lens_scale,
        visible: true,
        show_help: false,
    }
}

/// The exposures the HUD controls hold, clamped to [`EV100_LIMITS`] (a NaN gives the default).
pub fn exposure(controls: &HudControls) -> Exposure {
    Exposure {
        scene_ev100: clamp_ev(controls.scene_ev100, DEFAULT_SCENE_EV100),
        star_ev100: clamp_ev(controls.star_ev100, DEFAULT_STAR_EV100),
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

/// A bookmark applied to the world: its view (anchor and `look_at` resolved, A6) and the frame
/// to anchor the camera to.
pub struct Applied {
    pub view: CameraView,
    pub anchor: uni_core::FrameId,
    pub choice: ViewChoices,
}

/// Applies `b`: its epoch and time rate to `world`, its exposure and lens scale to `controls`
/// (fields it leaves out keep their current values); resolves its anchor and `look_at` against
/// the world at that epoch (uni-nav `resolve_bookmark`), and returns the view with the position
/// normalised. Errors name the bookmark: a bad epoch string, an unknown object, or a view that
/// would put NaN on the GPU.
pub fn apply_bookmark(
    b: &Bookmark,
    world: &mut World,
    controls: &mut HudControls,
) -> anyhow::Result<Applied> {
    if let Some(s) = &b.epoch_utc {
        let e = Epoch::parse_iso_utc(s).with_context(|| {
            format!(
                "bookmark {:?}: epoch_utc {s:?} is not YYYY-MM-DDThh:mm:ss[.fff]Z",
                b.name
            )
        })?;
        world.set_epoch(e);
    }
    if let Some(r) = b.time_rate {
        world.set_time_rate(r);
    }
    let (view, anchor) = uni_nav::resolve_bookmark(b, world)?;
    let view = checked_view(&b.name, &view)?;
    if let Some(ev) = b.scene_ev100 {
        controls.scene_ev100 = ev;
    }
    if let Some(ev) = b.star_ev100 {
        controls.star_ev100 = ev;
    }
    if let Some(s) = b.lens_scale {
        controls.lens_scale = s;
    }
    Ok(Applied {
        view,
        anchor,
        choice: ViewChoices::from_bookmark(b),
    })
}

/// `v` with its position normalised and its orientation renormalised, or an error naming
/// bookmark `name` if its position or velocity is not finite, the orientation is degenerate, or
/// the field of view is outside (0°, 180°).
fn checked_view(name: &str, v: &CameraView) -> anyhow::Result<CameraView> {
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
            "bookmark {name:?}: the view needs a finite position and velocity, a non-zero \
             orientation and 0 < fov_y_rad < π (got {v:?})"
        ),
    }
}

/// Per-pass GPU durations from the timer's per-pass values, which on Metal are effectively
/// cumulative end stamps (B4L report §5: a render pass is sampled from the start of its vertex
/// stage, which runs ahead, to the end of its fragment stage; lead ruling: report the
/// differences of consecutive stamps). Pass i's duration is its stamp minus the latest valid
/// stamp before it (0 for the first), floored at 0 where passes overlapped; an invalid stamp
/// stays `None` ("n/a", JSON `null`) and does not move the running stamp.
pub fn pass_durations(stamps: &[(&'static str, Option<f32>)]) -> Vec<(&'static str, Option<f32>)> {
    let mut prev = 0.0_f32;
    stamps
        .iter()
        .map(|&(name, ms)| {
            let d = ms.map(|t| {
                let d = (t - prev).max(0.0);
                prev = prev.max(t);
                d
            });
            (name, d)
        })
        .collect()
}

/// A `notes` line for every timer label that appears more than once in one frame's `timings`
/// (their `gpu_ms` entries would collide); empty when every label is unique.
pub fn duplicate_pass_names(timings: &[(&'static str, Option<f32>)]) -> Vec<String> {
    let mut seen = std::collections::BTreeMap::<&str, usize>::new();
    for (name, _) in timings {
        *seen.entry(name).or_default() += 1;
    }
    seen.into_iter()
        .filter(|&(_, n)| n > 1)
        .map(|(name, n)| {
            format!("GPU pass label {name:?} timed {n} times in one frame: its gpu_ms entry is ambiguous")
        })
        .collect()
}

/// Frame timings for the HUD.
pub struct Perf {
    pub fps: f32,
    pub cpu_ms: f32,
    /// Per pass durations; `None` for an invalid measurement.
    pub gpu_ms: Vec<(&'static str, Option<f32>)>,
    pub gpu_frame_ms: Option<f32>,
    pub gpu_timing_available: bool,
}

impl Perf {
    /// The renderer's latest GPU timings (as durations), with the window's (or capture's) fps
    /// and CPU time.
    pub fn from_renderer(renderer: &Renderer, gpu: &GpuContext, fps: f32, cpu_ms: f32) -> Self {
        Self {
            fps,
            cpu_ms,
            gpu_ms: pass_durations(&renderer.gpu_ms()),
            gpu_frame_ms: renderer.gpu_frame_ms(),
            gpu_timing_available: gpu.has_timestamps(),
        }
    }
}

/// What the HUD shows for this frame: `notices` (data and GPU warnings) first, then the lens
/// coverage warning, then the frame's honesty labels. The amber eye-adapted label follows the
/// resolve's own condition (frame contract, review m13); uni-ui's M0 lens lines stay off
/// (`lens_on_screen` false), since the lens labels are in the frame's labels with the M1
/// wording.
pub fn hud_data(
    world: &World,
    frame: &Frame,
    perf: Perf,
    speed_mps: f64,
    star_count: u32,
    notices: &[String],
) -> HudData {
    let ctx = &frame.ctx;
    let (nearest_name, nearest_dist_m) = world.nearest(&ctx.camera.pos);
    let mut warnings = notices.to_vec();
    warnings.extend(frame.coverage.warning().map(str::to_owned));
    warnings.extend(
        frame
            .labels
            .iter()
            .filter(|l| l.as_str() != uni_ui::EYE_ADAPTED_STARS_LABEL)
            .cloned(),
    );
    HudData {
        fps: perf.fps,
        cpu_ms: perf.cpu_ms,
        gpu_ms: perf.gpu_ms,
        gpu_frame_ms: perf.gpu_frame_ms,
        gpu_timing_available: perf.gpu_timing_available,
        render_size: ctx.viewport.to_array(),
        output_size: ctx.output_size.to_array(),
        dist_from_sun_m: world
            .position_of("Sun")
            .map_or(f64::NAN, |s| ctx.camera.pos.delta_from(s).length()),
        nearest_name,
        nearest_dist_m,
        speed_mps,
        epoch: world.epoch(),
        time_rate: world.time_rate(),
        star_count,
        warnings,
        inside_horizon: frame.coverage == LensCoverage::InsideHorizon,
        lens_on_screen: false,
        eye_adapted_stars: frame.settings.eye_adapted_in_effect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, Vec4};
    use uni_core::gravitational_radius;

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

    fn ctx(world: &World, view: &CameraView, viewport: UVec2) -> FrameCtx {
        let snap = world.sim.snapshot(view, viewport);
        frame_ctx(world, view, viewport, viewport, exposure0(), 0, 0.0, &snap)
    }

    /// A camera `r_m` [M] sunward of TestBH-1, looking at it.
    fn ctx_near_hole(r_m: f64) -> (World, FrameCtx) {
        let world = World::bare(Epoch::J2000);
        let lens = world.sim.lenses()[0].clone();
        let m = gravitational_radius(lens.mass_kg);
        let pos = lens.pos.offset(DVec3::new(-r_m * m, 0.0, 0.0));
        // Looking along +x with +z up: camera −Z → +x, +Y → +z.
        let orient = DQuat::from_mat3(&glam::DMat3::from_cols(
            DVec3::NEG_Y,
            DVec3::Z,
            DVec3::NEG_X,
        ));
        let c = ctx(&world, &view_at(pos, orient), UVec2::new(1920, 1080));
        (world, c)
    }

    #[test]
    fn view_rot_maps_camera_forward_to_minus_z() {
        let orient = DQuat::from_rotation_y(0.7) * DQuat::from_rotation_x(-0.3);
        let r = view_rot(orient);
        let f = r.transform_vector3((orient * DVec3::NEG_Z).as_vec3());
        let u = r.transform_vector3((orient * DVec3::Y).as_vec3());
        assert!(f.abs_diff_eq(Vec3::NEG_Z, 1e-6), "{f}");
        assert!(u.abs_diff_eq(Vec3::Y, 1e-6), "{u}");
    }

    #[test]
    fn frame_ctx_centre_pixel_looks_forward_and_the_sun_is_the_light() {
        let world = World::bare(Epoch::J2000);
        let orient = DQuat::from_rotation_z(1.1) * DQuat::from_rotation_x(0.4);
        let view = view_at(UniversePos::from_meters(DVec3::new(2e11, 0.0, 0.0)), orient);
        let snap = world.sim.snapshot(&view, UVec2::new(2, 2));
        let c = frame_ctx(
            &world,
            &view,
            UVec2::new(2, 2),
            UVec2::new(4, 4),
            exposure0(),
            7,
            0.016,
            &snap,
        );
        let p = (c.proj * c.view_rot).inverse() * Vec4::new(0.0, 0.0, 1.0, 1.0);
        let dir = (p.truncate() / p.w).normalize();
        let fwd = (orient * DVec3::NEG_Z).as_vec3();
        assert!(dir.abs_diff_eq(fwd, 1e-5), "{dir} vs {fwd}");
        assert_eq!((c.frame_index, c.viewport), (7, UVec2::new(2, 2)));
        assert_eq!(c.output_size, UVec2::new(4, 4));
        assert_eq!(c.lenses.len(), 1);
        // The Sun is the frame's one light, at its true offset from the camera.
        assert_eq!(c.lights.len(), 1);
        let sun = world.position_of("Sun").unwrap().delta_from(view.pos);
        assert!((c.lights[0].rel_pos_m - sun).length() < 1e-3);
        assert!(c.lights[0].luminous_intensity_cd > 1e27);
    }

    #[test]
    fn frame_ctx_survives_a_zero_viewport() {
        let world = World::bare(Epoch::J2000);
        let view = view_at(
            UniversePos::from_meters(DVec3::splat(1e12)),
            DQuat::IDENTITY,
        );
        let snap = world.sim.snapshot(&view, UVec2::ONE);
        let c = frame_ctx(
            &world,
            &view,
            UVec2::ZERO,
            UVec2::ZERO,
            exposure0(),
            0,
            0.0,
            &snap,
        );
        assert_eq!((c.viewport, c.output_size), (UVec2::ONE, UVec2::ONE));
        assert!(c.proj.is_finite());
    }

    /// Review I6: the window renders at about 1080 lines at its own aspect, never above the
    /// output's size, and the tonemap upscales.
    #[test]
    fn internal_size_is_about_1080_lines_at_the_output_aspect() {
        assert_eq!(
            internal_size(UVec2::new(2560, 1600)),
            UVec2::new(1728, 1080)
        );
        assert_eq!(
            internal_size(UVec2::new(3840, 2160)),
            UVec2::new(1920, 1080)
        );
        assert_eq!(internal_size(UVec2::new(1280, 800)), UVec2::new(1280, 800));
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
        c.scene_ev100 = f32::NAN;
        c.star_ev100 = f32::NAN;
        let e = exposure(&c);
        assert_eq!(e.scene_ev100, DEFAULT_SCENE_EV100);
        assert_eq!(e.star_ev100, DEFAULT_STAR_EV100);
    }

    /// B9 Step 1's app tests. The meter's night EV and uni-sky's star EV are both M0's −4 (the
    /// pair B6 and B4 were built against; the debug-grid bookmark keeps it); the app's
    /// eye-adapted default is −6.5 (lead ruling "B4L exposure"): a lower EV is more exposure, so
    /// the stars get 2.5 stops more exposure than at the night scene EV (`exposure_scale(−6.5)`
    /// is 2^2.5 = 5.7 × `exposure_scale(−4)`), never less. The resolve's clamp is the app's.
    #[test]
    fn auto_exposure_constants_match_the_app() {
        assert_eq!(uni_post::NIGHT_EV100, uni_sky::DEFAULT_STAR_EV100);
        assert_eq!(DEFAULT_STAR_EV100, -6.5);
        assert!(DEFAULT_STAR_EV100 <= uni_post::NIGHT_EV100);
        assert_eq!(uni_post::EV100_MIN..=uni_post::EV100_MAX, EV100_LIMITS);
    }

    #[test]
    fn view_choices_come_from_the_bookmark_and_the_grid_forces_manual() {
        let b = |exposure, eye_adapted, lens_env| Bookmark {
            name: "b".to_owned(),
            view: view_at(UniversePos::default(), DQuat::IDENTITY),
            epoch_utc: None,
            scene_ev100: None,
            star_ev100: None,
            lens_scale: None,
            anchor: None,
            look_at: None,
            up: None,
            time_rate: None,
            exposure,
            eye_adapted,
            target: None,
            lens_env,
            cruise: None,
            highlight_belt: None,
        };
        assert_eq!(
            ViewChoices::from_bookmark(&b(None, None, None)),
            ViewChoices::default()
        );
        let d = ViewChoices::default();
        assert_eq!(
            (d.exposure, d.eye_adapted, d.lens_env),
            (ExposureMode::Auto, true, LensEnv::Band)
        );
        let grid = ViewChoices::from_bookmark(&b(
            Some(ExposureChoice::Auto),
            Some(false),
            Some(LensEnvChoice::DebugGrid),
        ));
        assert_eq!(grid.lens_env, LensEnv::DebugGrid);
        assert!(!grid.eye_adapted);
        // Auto asked, but the debug grid is display-referred: Manual (frame contract, pass 3).
        assert_eq!(grid.mode(), ExposureMode::Manual);
        let s = exposure_settings(&default_controls(1.0), grid, false, true);
        assert_eq!(s.mode, ExposureMode::Manual);
    }

    #[test]
    fn exposure_settings_follow_the_controls_choices_and_daylight() {
        let mut c = default_controls(1.0);
        c.scene_ev100 = 3.0;
        c.star_ev100 = -5.0;
        let s = exposure_settings(&c, ViewChoices::default(), false, true);
        assert_eq!(s.mode, ExposureMode::Auto);
        assert!(s.eye_adapted && !s.daylight && s.settle && s.eye_adapted_in_effect());
        assert_eq!(s.eye_star_ev100, -5.0);
        assert_eq!(
            s.manual,
            Exposure {
                scene_ev100: 3.0,
                star_ev100: -5.0
            }
        );
        // Daylight (review m12): no eye-adapted stars; the stars take the scene EV (D4).
        let day = exposure_settings(&c, ViewChoices::default(), true, false);
        assert!(!day.eye_adapted_in_effect());
        assert_eq!(day.manual.star_ev100, 3.0);
        let u = uni_post::resolve_exposure(&day, Some(12.0));
        assert_eq!(u.ev100[0], u.ev100[1]);
        // At night with nothing lit: the scene at NIGHT_EV100, the stars at the app's −6.5.
        let night = exposure_settings(&default_controls(1.0), ViewChoices::default(), false, true);
        let u = uni_post::resolve_exposure(&night, Some(uni_post::NIGHT_EV100));
        assert_eq!(u.ev100[0], uni_post::NIGHT_EV100);
        assert_eq!(u.ev100[1], DEFAULT_STAR_EV100);
    }

    /// Lead ruling at B9: a reflected point beside a resolved body takes the body's blended
    /// exposure (W = 1 from 4 px: scene exposure), and stays at star exposure with nothing
    /// resolved.
    #[test]
    fn reflected_points_follow_the_resolved_bodies_exposure() {
        assert_eq!(point_exposure_factor(0.0, -6.5, 13.0), 1.0);
        let full = point_exposure_factor(1.0, -6.5, 13.0);
        assert!((full.log2() - (-19.5)).abs() < 1e-4, "{}", full.log2());
        let half = point_exposure_factor(0.5, -6.5, 13.0);
        assert!((half.log2() - (-9.75)).abs() < 1e-4, "{}", half.log2());
        // Jupiter resolved at 28.8 px: W = 1; from the SSB nothing is resolved: W = 0.
        let mut world = World::bare(Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap());
        world.set_epoch(Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap());
        let j = world.position_of("Jupiter").unwrap();
        let sun = world.position_of("Sun").unwrap().delta_from(j).normalize();
        let pos = j.offset(sun * 5.0e9);
        let fwd = -sun;
        let right = fwd.cross(DVec3::Z).normalize();
        let orient = DQuat::from_mat3(&glam::DMat3::from_cols(right, right.cross(fwd), -fwd));
        let mut v = view_at(pos, orient);
        v.fov_y_rad = 30_f64.to_radians();
        let snap = world.sim.snapshot(&v, UVec2::new(1920, 1080));
        assert_eq!(resolved_weight(&snap), 1.0);
        let far = world.sim.snapshot(
            &view_at(
                UniversePos::from_meters(DVec3::new(0.0, 0.0, 1e15)),
                DQuat::IDENTITY,
            ),
            UVec2::new(1920, 1080),
        );
        assert_eq!(resolved_weight(&far), 0.0);
        assert!(is_reflected(ObjectKind::Moon) && !is_reflected(ObjectKind::Star));
        assert!(!is_reflected(ObjectKind::Sun));
    }

    #[test]
    fn pass_durations_are_stamp_differences_and_keep_na() {
        let stamps = [
            ("exposure", Some(0.05)),
            ("band-update", None),
            ("bodies", Some(1.05)),
            ("atmosphere", Some(3.05)),
            ("histogram", Some(2.9)), // overlapped: floored at 0
            ("stars", Some(4.0)),
        ];
        let d = pass_durations(&stamps);
        let names: Vec<&str> = d.iter().map(|p| p.0).collect();
        assert_eq!(
            names,
            [
                "exposure",
                "band-update",
                "bodies",
                "atmosphere",
                "histogram",
                "stars"
            ]
        );
        let ms: Vec<Option<f32>> = d.iter().map(|p| p.1).collect();
        assert_eq!(ms[1], None);
        for (got, want) in ms.iter().zip([
            Some(0.05),
            None,
            Some(1.0),
            Some(2.0),
            Some(0.0),
            Some(0.95),
        ]) {
            match (got, want) {
                (Some(g), Some(w)) => assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}"),
                _ => assert_eq!(*got, want),
            }
        }
        assert!(pass_durations(&[]).is_empty());
    }

    /// The timer labels of the M1 stack in encode order (the frame contract's pass order, with the
    /// sub-passes some features time on their own and the two capture taps), for the uniqueness
    /// test below: `gpu_ms` in the capture JSON is keyed by label, so two passes with one label
    /// would share one entry. A tap times its copy only on the frames it is armed for (captures).
    /// The live check is `duplicate_pass_names` on the timer's own list, run on every capture.
    const TIMED_PASSES: [&str; 18] = [
        "exposure",
        "band-update",
        "background",
        "sun",
        "bodies",
        "rings",
        "atmosphere",
        "atmosphere mask",
        "histogram",
        crate::tap::TAP_WORLD,
        "lens",
        "band",
        "stars",
        "points",
        crate::tap::TAP_PREBLOOM,
        "bloom",
        "bloom composite",
        "tonemap",
    ];

    /// Review B9 r0 Important 1: each armed tap is its own timed span, named "tap-world" and
    /// "tap-prebloom", right after the meter and right before bloom, and every timer label of
    /// the stack is unique (so `gpu_ms` has one entry per pass). With the tap's span in the
    /// frame, the pass after it is charged its own time only, not the copy's.
    #[test]
    fn tap_spans_exist_and_pass_names_stay_unique() {
        let at = |name: &str| TIMED_PASSES.iter().position(|p| *p == name).unwrap();
        assert_eq!(at(crate::tap::TAP_WORLD), at("histogram") + 1);
        assert_eq!(at(crate::tap::TAP_WORLD) + 1, at("lens"));
        assert_eq!(at(crate::tap::TAP_PREBLOOM) + 1, at("bloom"));
        assert_eq!(at(crate::tap::TAP_PREBLOOM), at("points") + 1);
        assert_eq!(crate::tap::TAP_WORLD, "tap-world");
        assert_eq!(crate::tap::TAP_PREBLOOM, "tap-prebloom");
        let all: Vec<(&'static str, Option<f32>)> =
            TIMED_PASSES.iter().map(|p| (*p, Some(1.0))).collect();
        assert!(duplicate_pass_names(&all).is_empty());
        let mut twice = all.clone();
        twice.push(("band", None));
        let notes = duplicate_pass_names(&twice);
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("\"band\""), "{notes:?}");

        // A 0.3 ms copy after the meter: charged to the tap, not to the lens or band pass.
        let stamps = [
            ("histogram", Some(3.0)),
            (crate::tap::TAP_WORLD, Some(3.3)),
            ("band", Some(4.3)),
            ("points", Some(4.5)),
            (crate::tap::TAP_PREBLOOM, Some(4.8)),
            ("bloom", Some(5.8)),
        ];
        let d: std::collections::BTreeMap<&str, f32> = pass_durations(&stamps)
            .into_iter()
            .map(|(n, ms)| (n, ms.unwrap()))
            .collect();
        for (name, want) in [
            (crate::tap::TAP_WORLD, 0.3),
            ("band", 1.0),
            (crate::tap::TAP_PREBLOOM, 0.3),
            ("bloom", 1.0),
        ] {
            assert!((d[name] - want).abs() < 1e-5, "{name}: {}", d[name]);
        }
    }

    #[test]
    fn near_the_hole_the_lens_is_traced_and_labelled() {
        let (world, c) = ctx_near_hole(30.0);
        assert_eq!(uni_rel::lens_coverage(&c), LensCoverage::Traced);
        let snap = world.sim.snapshot(&c.camera, c.viewport);
        for (env, grid) in [(LensEnv::Band, false), (LensEnv::DebugGrid, true)] {
            let choice = ViewChoices {
                lens_env: env,
                ..ViewChoices::default()
            };
            let s = exposure_settings(&default_controls(1.0), choice, false, false);
            let labels = honesty_labels(
                &world,
                &c.camera,
                &snap,
                &s,
                env,
                LensCoverage::Traced,
                env == LensEnv::Band,
                0.0,
                true,
            );
            assert!(labels.contains(&UNLENSED_STARS_LABEL.to_owned()));
            assert_eq!(labels.contains(&DEBUG_GRID_LABEL.to_owned()), grid);
            let band = uni_sim::honesty::label(HonestyFlag::BandModel).to_owned();
            let detail = uni_sim::honesty::label(HonestyFlag::MilkyWayDetailProcedural).to_owned();
            // B4L: the detail label rides with the band label, only while the band is drawn.
            assert_eq!(labels.contains(&band), !grid, "{labels:?}");
            assert_eq!(labels.contains(&detail), !grid, "{labels:?}");
            assert!(labels.contains(&uni_ui::VIVID_STAR_COLOURS_LABEL.to_owned()));
        }
    }

    #[test]
    fn beyond_the_trace_reach_the_hud_warns() {
        let (_, c) = ctx_near_hole(1e5);
        assert_eq!(uni_rel::lens_coverage(&c), LensCoverage::BeyondReach);
        assert_eq!(
            uni_rel::lens_coverage(&c).warning(),
            Some(uni_rel::BEYOND_REACH_WARNING)
        );
    }

    #[test]
    fn inside_the_horizon_the_hud_says_so() {
        for r in [1.5, 2.0, 0.0] {
            let (_, c) = ctx_near_hole(r);
            assert_eq!(
                uni_rel::lens_coverage(&c),
                LensCoverage::InsideHorizon,
                "r = {r} M"
            );
            assert!(c.view_rot.is_finite());
        }
    }

    #[test]
    fn apply_bookmark_resolves_anchors_and_sets_only_what_it_names() {
        let mut world = World::bare(Epoch::J2000);
        let mut controls = default_controls(0.5);
        let mut b = Bookmark {
            name: "b".to_owned(),
            view: CameraView {
                pos: UniversePos::from_meters(DVec3::new(0.0, 0.0, 6.0e7)),
                orient: DQuat::from_xyzw(0.0, 0.0, 0.0, 2.0),
                fov_y_rad: 0.5,
                velocity_mps: DVec3::ZERO,
            },
            epoch_utc: Some("2026-11-20T00:00:00Z".to_owned()),
            scene_ev100: None,
            star_ev100: None,
            lens_scale: None,
            anchor: Some("Earth".to_owned()),
            look_at: Some("Earth".to_owned()),
            up: None,
            time_rate: None,
            exposure: None,
            eye_adapted: None,
            target: None,
            lens_env: None,
            cruise: None,
            highlight_belt: None,
        };
        let a = apply_bookmark(&b, &mut world, &mut controls).unwrap();
        assert_eq!(
            world.epoch(),
            Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap()
        );
        let earth = world.position_of("Earth").unwrap();
        let rel = a.view.pos.delta_from(earth);
        assert!((rel - DVec3::new(0.0, 0.0, 6.0e7)).length() < 1e-3, "{rel}");
        // look_at overrides the orientation: −Z toward Earth.
        let fwd = a.view.orient * DVec3::NEG_Z;
        assert!(fwd.abs_diff_eq(DVec3::NEG_Z, 1e-9), "{fwd}");
        assert_eq!(world.frame_body_name(a.anchor).as_deref(), Some("Earth"));
        assert_eq!(controls, default_controls(0.5));
        b.scene_ev100 = Some(3.0);
        b.star_ev100 = Some(-1.0);
        b.lens_scale = Some(1.0);
        b.anchor = Some("Vulcan".to_owned());
        let err = apply_bookmark(&b, &mut world, &mut controls).err().unwrap();
        assert!(format!("{err:#}").contains("Vulcan"), "{err:#}");
        b.anchor = None;
        b.look_at = None;
        b.view.pos = UniversePos::default();
        apply_bookmark(&b, &mut world, &mut controls).unwrap();
        assert_eq!(
            (
                controls.scene_ev100,
                controls.star_ev100,
                controls.lens_scale
            ),
            (3.0, -1.0, 1.0)
        );
        // A degenerate view is refused with the bookmark's name.
        b.view.fov_y_rad = 0.0;
        let err = apply_bookmark(&b, &mut world, &mut controls).err().unwrap();
        assert!(format!("{err:#}").contains("bookmark \"b\""), "{err:#}");
        b.view.fov_y_rad = 0.5;
        b.epoch_utc = Some("yesterday".to_owned());
        assert!(apply_bookmark(&b, &mut world, &mut controls).is_err());
    }
}
