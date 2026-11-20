//! The render stack and per-frame data shared by the window and `--capture` (M0 Task 11).
//!
//! [`Scene`] owns the [`Renderer`] with the M0 features in their fixed order: the lens pass,
//! then stars, then the tonemap (design §5.3; the HUD draws after them). The pure helpers
//! below turn the world and a camera view into the [`FrameCtx`] the features read and the
//! [`HudData`] the HUD shows.

use std::cell::Cell;
use std::rc::Rc;

use anyhow::Context as _;
use glam::{DMat4, UVec2};
use uni_core::{CameraView, Epoch};
use uni_data::CatalogStar;
use uni_gfx::{
    Exposure, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, Renderer, ShaderLib,
    reversed_z_infinite_proj,
};
use uni_nav::Bookmark;
use uni_rel::{LensCoverage, LensTracer};
use uni_ui::{HudControls, HudData};

use crate::world::World;

/// Near plane of the reversed-Z infinite projection [m]. M0 draws nothing closer than
/// infinity, so this only fixes the depth scale for later geometry.
pub const NEAR_PLANE_M: f32 = 0.1;
/// Scene exposure when nothing sets it [EV100]. The scene's only light in M0 is the debug grid,
/// which uni-rel scales to sit at this exposure.
pub const DEFAULT_SCENE_EV100: f32 = 0.0;
/// Lens tracer resolution scale in the window (the HUD slider); the tracer's own default.
pub const INTERACTIVE_LENS_SCALE: f32 = 0.5;
/// Lens tracer resolution scale in captures, unless a bookmark sets one.
pub const CAPTURE_LENS_SCALE: f32 = 1.0;

/// The renderer with the M0 feature stack, plus the handles the app keeps to steer it.
pub struct Scene {
    pub renderer: Renderer,
    /// Shared with the lens feature inside `renderer`, which reads it in `prepare`.
    lens_scale: Rc<Cell<f32>>,
    /// Catalogue stars the star feature draws.
    pub star_count: u32,
}

impl Scene {
    /// Builds the renderer at `size` and pushes the lens pass, the stars and the tonemap, in
    /// that order. `output_format` is the sRGB format the tonemap writes (the window's sRGB
    /// view, or `uni_gfx::CAPTURE_FORMAT`).
    pub fn new(
        gpu: &GpuContext,
        size: UVec2,
        output_format: wgpu::TextureFormat,
        stars: &[CatalogStar],
        lens_scale: f32,
    ) -> anyhow::Result<Self> {
        let mut renderer = Renderer::new(gpu, size);
        let mut lib = ShaderLib::with_common();
        let camera_bgl = renderer.camera_bind_group_layout();
        let tracer =
            LensTracer::new(gpu, &mut lib, camera_bgl).context("creating the lens pass")?;
        let star_field = uni_sky::StarField::new(gpu, &mut lib, camera_bgl, stars)
            .context("creating the star pass")?;
        let tonemap = uni_post::TonemapFeature::new(gpu, &mut lib, camera_bgl, output_format)
            .context("creating the tonemap pass")?;
        let star_count = star_field.count();

        let lens_scale = Rc::new(Cell::new(lens_scale));
        renderer.push(Box::new(LensGate {
            tracer,
            scale: Rc::clone(&lens_scale),
            runs: false,
        }));
        renderer.push(Box::new(star_field));
        renderer.push(Box::new(tonemap));
        Ok(Self {
            renderer,
            lens_scale,
            star_count,
        })
    }

    /// Lens tracer resolution scale from the next frame on (the tracer clamps it to 0.25..=1).
    pub fn set_lens_scale(&self, s: f32) {
        self.lens_scale.set(s);
    }

    /// Records one frame into `output` (see [`Renderer::render`]).
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        ctx: &FrameCtx,
        output: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        self.renderer.render(gpu, ctx, output)
    }
}

/// Whether the lens pass draws this frame. It runs only where it traces geodesics, or to black
/// out the frame from inside the horizon. Elsewhere it would paint the M0 debug grid over the
/// whole frame (uni-rel draws the unlensed grid outside its rectangle, even with no lens), so
/// the cleared black stays and the sky shows the real stars (lead ruling, wave 2).
pub fn lens_pass_runs(coverage: LensCoverage) -> bool {
    match coverage {
        LensCoverage::Traced | LensCoverage::InsideHorizon => true,
        LensCoverage::Unlensed | LensCoverage::BeyondReach => false,
    }
}

/// The lens tracer, run only when [`lens_pass_runs`], at the resolution scale the app sets.
/// `Renderer` owns its features, so the app reaches the tracer through the shared `scale`.
struct LensGate {
    tracer: LensTracer,
    scale: Rc<Cell<f32>>,
    /// Decided in `prepare` for this frame's `encode`.
    runs: bool,
}

impl RenderFeature for LensGate {
    fn name(&self) -> &'static str {
        self.tracer.name()
    }

    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets) {
        self.tracer.set_resolution_scale(self.scale.get());
        self.runs = lens_pass_runs(uni_rel::lens_coverage(ctx));
        if self.runs {
            self.tracer.prepare(gpu, ctx, targets);
        }
    }

    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        targets: &FrameTargets,
        output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if self.runs {
            self.tracer.encode(gpu, enc, targets, output, timer);
        }
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.tracer.resize(gpu, targets);
    }
}

/// Root → camera rotation for a camera whose `orient` turns camera axes into root axes.
pub fn view_rot(orient: glam::DQuat) -> glam::Mat4 {
    DMat4::from_quat(orient.inverse()).as_mat4()
}

/// The frame context for `view` on a `viewport`-sized frame. `viewport` must equal the
/// renderer's target size: the lens pass plans with the targets and the HUD's lens status with
/// this viewport (`uni_rel::lens_coverage`).
pub fn frame_ctx(
    world: &World,
    view: &CameraView,
    viewport: UVec2,
    exposure: Exposure,
    frame_index: u64,
    dt_real_s: f64,
) -> FrameCtx {
    let viewport = viewport.max(UVec2::ONE);
    let aspect = viewport.x as f32 / viewport.y as f32;
    FrameCtx {
        frame_index,
        epoch: world.epoch,
        dt_real_s,
        camera: view.clone(),
        view_rot: view_rot(view.orient),
        proj: reversed_z_infinite_proj(view.fov_y_rad as f32, aspect, NEAR_PLANE_M),
        viewport,
        exposure,
        lenses: world.lens_views(&view.pos),
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

/// The exposures the HUD controls hold.
pub fn exposure(controls: &HudControls) -> Exposure {
    Exposure {
        scene_ev100: controls.scene_ev100,
        star_ev100: controls.star_ev100,
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

/// `b`'s view with its position normalised and its orientation renormalised, or an error if it
/// is not finite, the orientation is degenerate, or the field of view is outside (0°, 180°).
fn checked_view(b: &Bookmark) -> anyhow::Result<CameraView> {
    let v = &b.view;
    let pos_ok = v.pos.local.is_finite();
    let len = v.orient.length();
    let orient = (len.is_finite() && len > 1e-12).then(|| v.orient / len);
    let fov_ok = v.fov_y_rad > 0.0 && v.fov_y_rad < std::f64::consts::PI;
    match (pos_ok, orient, fov_ok) {
        (true, Some(orient), true) => Ok(CameraView {
            pos: v.pos.normalized(),
            orient,
            fov_y_rad: v.fov_y_rad,
        }),
        _ => anyhow::bail!(
            "bookmark {:?}: the view needs a finite position, a non-zero orientation and \
             0 < fov_y_rad < π (got {:?})",
            b.name,
            v
        ),
    }
}

/// Frame timings for the HUD.
pub struct Perf {
    pub fps: f32,
    pub cpu_ms: f32,
    pub gpu_ms: Vec<(&'static str, f32)>,
    pub gpu_timing_available: bool,
}

/// What the HUD shows for this frame. `notices` (catalogue warnings and the like) come first,
/// then the lens labels for `ctx` ([`lens_warnings`]).
pub fn hud_data(
    world: &World,
    ctx: &FrameCtx,
    perf: Perf,
    speed_mps: f64,
    star_count: u32,
    notices: &[String],
) -> HudData {
    let (nearest_name, nearest_dist_m) = world.nearest(&ctx.camera.pos);
    let mut warnings = notices.to_vec();
    warnings.extend(lens_warnings(ctx));
    HudData {
        fps: perf.fps,
        cpu_ms: perf.cpu_ms,
        gpu_ms: perf.gpu_ms,
        gpu_timing_available: perf.gpu_timing_available,
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
        inside_horizon: ctx.lenses.iter().any(uni_rel::inside_horizon),
        // uni-ui shows its two M0 honesty lines ("Lens background: M0 debug grid", "Stars near
        // the hole are not yet lensed (M0)") only when the debug grid is actually visible: the
        // lens pass traced this frame. From inside the horizon the frame is all black, so no
        // grid is shown and the labels would be false (lead ruling, M0 gate, Task 10b).
        lens_on_screen: matches!(uni_rel::lens_coverage(ctx), LensCoverage::Traced),
    }
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
        assert_eq!(ctx.lenses.len(), 1);
    }

    #[test]
    fn frame_ctx_survives_a_zero_viewport() {
        let world = World::new(Epoch::J2000);
        let ctx = frame_ctx(
            &world,
            &view_at(UniversePos::default(), DQuat::IDENTITY),
            UVec2::ZERO,
            exposure0(),
            0,
            0.0,
        );
        assert_eq!(ctx.viewport, UVec2::ONE);
        assert!(ctx.proj.is_finite());
    }

    #[test]
    fn lens_pass_runs_only_where_it_traces_or_blacks_out() {
        assert!(lens_pass_runs(LensCoverage::Traced));
        assert!(lens_pass_runs(LensCoverage::InsideHorizon));
        assert!(!lens_pass_runs(LensCoverage::Unlensed));
        assert!(!lens_pass_runs(LensCoverage::BeyondReach));
    }

    #[test]
    fn near_the_hole_the_lens_runs_and_the_hud_shows_the_m0_labels() {
        let (world, ctx) = ctx_near_hole(30.0);
        assert_eq!(uni_rel::lens_coverage(&ctx), LensCoverage::Traced);
        assert!(lens_pass_runs(uni_rel::lens_coverage(&ctx)));
        let perf = Perf {
            fps: 60.0,
            cpu_ms: 1.0,
            gpu_ms: vec![("lens", 2.0)],
            gpu_timing_available: true,
        };
        let data = hud_data(&world, &ctx, perf, 0.0, 9096, &["sky ok?".to_owned()]);
        assert_eq!(data.warnings, ["sky ok?"]);
        assert!(data.lens_on_screen);
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
        assert!(!lens_pass_runs(uni_rel::lens_coverage(&ctx)));
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
            assert!(lens_pass_runs(uni_rel::lens_coverage(&ctx)));
            let perf = Perf {
                fps: 0.0,
                cpu_ms: 0.0,
                gpu_ms: Vec::new(),
                gpu_timing_available: false,
            };
            let data = hud_data(&world, &ctx, perf, 0.0, 0, &[]);
            assert!(data.inside_horizon, "r = {r} M");
            assert!(data.warnings.is_empty(), "{:?}", data.warnings);
            assert!(data.nearest_dist_m.is_finite() && ctx.view_rot.is_finite());
        }
    }

    fn bookmark(view: CameraView) -> Bookmark {
        Bookmark {
            name: "b".to_owned(),
            view,
            epoch_utc: None,
            scene_ev100: None,
            star_ev100: None,
            lens_scale: None,
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
        let mut bad_epoch = bookmark(ok);
        bad_epoch.epoch_utc = Some("yesterday".to_owned());
        bad.push(bad_epoch);
        for b in &bad {
            let err = apply_bookmark(b, &mut world, &mut controls).unwrap_err();
            assert!(format!("{err:#}").contains("bookmark \"b\""), "{err:#}");
        }
        assert_eq!(world.epoch, Epoch::J2000);
    }
}
