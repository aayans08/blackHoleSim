//! Analytic ellipsoid bodies with surface maps, pass "bodies" (Task B1; B0 stub).
//!
//! Per resolved body per frame the CPU builds the camera-facing cap ([`crate::cap_mesh`],
//! [`crate::CAP_RINGS`] × [`crate::CAP_SEGMENTS`]) in the body's unit-sphere space; the
//! fragment shader re-intersects the exact ellipsoid ([`ray_ellipsoid`]) and the mesh supplies
//! only coverage and vertex depth. Sunlight is E☉ × [`crate::sun_visible_fraction`] over the
//! snapshot's occluders × [`ring_transmittance`] (Saturn) × `atmo_sun_transmittance`
//! (uni-atmo's group 2), plus `atmo_sky_irradiance` × albedo for atmosphere bodies. The exposure
//! factor blends from `star_pre` to `scene_pre` with [`crate::lod_exposure_mix`] while eye-adapted
//! stars are in effect.
//!
//! Rules B1 must keep (B0 context, carry-overs):
//! - Photometry comes from the view: [`BodyView::brdf`], [`BodyView::geometric_albedo`] and
//!   [`BodyView::mean_radius_m`] are the point path's own values (ruling A.1), and the per-pixel
//!   phase correction is `planet_phase_correction(..)` clamped to
//!   `uni_sim::snapshot::MAX_PHASE_CORRECTION`, the same clamp as the point path. Its hold beyond
//!   each law's fitted range (from 170° for Earth and Mars) is A3's choice, accepted by the lead;
//!   the code using it says so.
//! - No Earth- or Venus-specific calibration constant lives here (constraints §4a, W-B1..W-B3):
//!   the albedo stays a per-body value from the map index or the view.
//! - `World::objects()` grows at runtime, so nothing here is sized per `ObjectId` at
//!   construction: caches are keyed by NAIF id or kept in a map.

use std::path::Path;
use std::sync::Arc;

use glam::{Vec2, Vec3};
use uni_atmo::GroundLighting;
use uni_data::{BodyFacts, MapInfo, RingProfile};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{BodyView, StarSurfaceView};

/// What the bodies, star-surface and ring passes draw this frame (set by the app before
/// `Renderer::render`; the three passes share one input).
#[derive(Clone, Debug, Default)]
pub struct BodyScene {
    /// Resolved bodies (`Snapshot::bodies`).
    pub bodies: Vec<BodyView>,
    /// Resolved stars, the Sun first when drawn (`Snapshot::star_surfaces`).
    pub stars: Vec<StarSurfaceView>,
    /// True while "Eye-adapted stars" is in effect (`ExposureSettings::eye_adapted_in_effect`):
    /// small disks then blend their exposure from `star_pre` with [`crate::lod_exposure_mix`].
    pub eye_adapted: bool,
    /// Display names of the views drawn this frame, for warnings such as "Inside Earth"
    /// (ruling A.2; the app fills it from `World::objects`, only for the ids drawn).
    pub names: Vec<(ObjectId, String)>,
}

/// The bodies pass (`RenderFeature` named "bodies").
pub struct BodyRenderer {
    _private: (),
}

impl BodyRenderer {
    /// Loads the maps (`map_dir` holds `maps/index.ron`'s files: the BC file when
    /// `GpuContext::has_bc_compression()`, else the 2k `fallback_file`), builds the pipeline and
    /// keeps `ground` (group 2) and `input`. A body without a map draws uniform albedo from its
    /// view's p_V and reports it in [`BodyRenderer::warnings`]. `ring` is Saturn's ring profile,
    /// for the rings' shadow on the globe.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        maps: &[MapInfo],
        map_dir: &Path,
        facts: &[BodyFacts],
        ring: Option<&RingProfile>,
        ground: Arc<GroundLighting>,
        input: SharedInput<BodyScene>,
    ) -> anyhow::Result<Self> {
        let _ = (
            gpu, lib, camera_bgl, maps, map_dir, facts, ring, ground, input,
        );
        todo!("B1")
    }

    /// Warnings for the HUD from the last prepared frame, e.g. "No surface map: uniform albedo
    /// (Titania)", "Inside Earth".
    pub fn warnings(&self) -> Vec<String> {
        todo!("B1")
    }

    /// Mean reflectance colour of body `naif`'s albedo map: linear Rec. 709, normalised to
    /// luminance 1; `None` for a body without a map. Computed on the CPU from the 2k map loaded
    /// at construction. The app tints a reflected `uni_sky::PointSprite::color` by it, keeping
    /// the sprite's luminance (carry-over 6); bodies without a map stay solar-coloured.
    pub fn mean_albedo_rgb(&self, naif: i32) -> Option<Vec3> {
        let _ = naif;
        todo!("B1")
    }
}

impl RenderFeature for BodyRenderer {
    fn name(&self) -> &'static str {
        "bodies"
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

/// Ray–ellipsoid intersection in the body's unit-sphere space (the ellipsoid scaled to a unit
/// sphere): origin `o_unit`, direction `d_unit` (not necessarily unit length), and
/// c = |o_unit|² − 1 computed in f64 on the CPU (review I5). Returns the entry and exit ray
/// parameters (t₀ ≤ t₁) or `None` on a miss. Numerically stable form (Haines et al. 2019, *Ray
/// Tracing Gems* ch. 7). Twin of the WGSL in `body.wgsl`.
pub fn ray_ellipsoid(o_unit: Vec3, d_unit: Vec3, c: f32) -> Option<(f32, f32)> {
    let _ = (o_unit, d_unit, c);
    todo!("B1")
}

/// Texture coordinates of planetocentric (east) longitude `lon_rad` and latitude `lat_rad` in an
/// equirectangular map whose left edge is at east longitude `lon0_deg` (`uni_data::MapInfo`):
/// u runs east through 360°, v from +90° (0) to −90° (1). Twin of the WGSL in `body.wgsl`.
pub fn equirect_uv(lon_rad: f32, lat_rad: f32, lon0_deg: f32) -> Vec2 {
    let _ = (lon_rad, lat_rad, lon0_deg);
    todo!("B1")
}

/// Transmittance of Saturn's rings for sunlight reaching the globe point `p_body` (body axes,
/// ring plane z = 0) from the unit direction `sun_dir`: 1 when the ray to the Sun misses the
/// annulus [`inner`, `outer`], else 1 − `opacity_at(radius)` at the crossing. Twin of the WGSL in
/// `body.wgsl`.
pub fn ring_transmittance(
    p_body: Vec3,
    sun_dir: Vec3,
    inner: f32,
    outer: f32,
    opacity_at: impl Fn(f32) -> f32,
) -> f32 {
    let _ = (p_body, sun_dir, inner, outer, opacity_at);
    todo!("B1")
}

/// East-longitude offset [deg] of Venus's cloud-top map at epoch `t`: retrograde super-rotation
/// with a 4.0-day period at the cloud tops (Sánchez-Lavega et al. 2017, Space Sci. Rev. 212, 1541),
/// not the 243-day surface rate (review m22).
pub fn venus_cloud_longitude_offset_deg(t: uni_core::Epoch) -> f64 {
    let _ = t;
    todo!("B1")
}
