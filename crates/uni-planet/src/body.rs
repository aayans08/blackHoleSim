//! Analytic ellipsoid bodies with surface maps, pass "bodies" (Task B1).
//!
//! Per resolved body per frame the CPU builds the camera-facing cap ([`crate::cap_mesh`],
//! [`crate::CAP_RINGS`] × [`crate::CAP_SEGMENTS`], circumscribed) in the body's unit-sphere space,
//! makes it camera-relative in f64 and only then casts to f32. The fragment shader
//! (`shaders/body.wgsl`) re-intersects the exact ellipsoid ([`ray_ellipsoid`]); the mesh supplies
//! only coverage and vertex depth (never `frag_depth`). A camera inside a body's solid radii: the
//! body draws nothing and [`BodyRenderer::warnings`] reports "Inside <name>". A camera between
//! Venus's surface and its cloud deck sees the solid radii drawn, with no warning ([`placement`]).
//!
//! **Near-surface precision (review I5).** The CPU passes c = |o_unit|² − 1 from f64 with the
//! unit-space camera o_unit; the shader solves the quadratic in the stable form (Haines et al.
//! 2019, *Ray Tracing Gems* ch. 7). At 1 m above Earth c = 3.1 × 10⁻⁷, exact to f32's relative
//! 6 × 10⁻⁸, where the naive f32 |o|² − 1 is off by tens of percent
//! (`ray_ellipsoid_f32_near_surface`). The remaining f32 error of the hit point, 6 × 10⁻⁸ × R ≈
//! 0.4 m at Earth, is 10⁻⁴ of an 8k texel (4.9 km); M2's terrain adds a surface-anchored origin.
//!
//! **Silhouette AA.** coverage = clamp(0.5 − d_px, 0, 1) from the signed angle between the pixel
//! ray and the analytic limb, turned into pixels by its screen-space gradient; below 1/255 the
//! fragment is discarded; premultiplied blend into `hdr`; `lens_mask.g` gets the coverage and
//! `lens_mask.a` the star share 1 − w ([`star_share`]), both with `Max`. The mesh's outline ring
//! is pushed [`AA_MARGIN_PX`] out so the rim outside the limb is
//! rasterised. Edge pixels write depth, so the band and stars (depth-tested at the far plane,
//! ruling 4) miss their (1 − coverage) share on the 1-px rim (decision N1).
//!
//! **Shading.** Albedo from the map (`albedo_scale` applied, sRGB decoded by the format) or a
//! uniform ρ from p_V ([`crate::material`]); Lambert or lunar-Lambert L = 0.9 (McEwen 1991) as
//! [`BodyView::brdf`] says (A8's calibration); sunlight E☉ × [`crate::sun_visible_fraction`]
//! over the snapshot's occluders × [`ring_transmittance`] (Saturn) × `atmo_sun_transmittance`,
//! plus `atmo_sky_irradiance` × albedo (uni-atmo's group 2). Earth's cloud map is composited over
//! the ground; Venus is its cloud-top map on a sphere [`uni_atmo::VENUS_CLOUD_TOP_M`] above the
//! surface, super-rotating ([`venus_cloud_longitude_offset_deg`]).
//!
//! **Photometry is the point path's** (B0 ruling A.1): [`BodyView::brdf`],
//! [`BodyView::geometric_albedo`] and [`BodyView::mean_radius_m`]; every pixel's radiance is
//! multiplied by Mallama & Hilton's phase correction C clamped to
//! `uni_sim::snapshot::MAX_PHASE_CORRECTION`, the same clamp as the point path (A3 ruling). For
//! the bodies drawn with an atmosphere the point flux cross-fades over the 1.5–4 px band to the
//! whole rendered disk, this pass's ground as uni-atmo's pass dims it plus that pass's
//! in-scatter (ruling W-B1, review I3: [`body_light`], [`flux_match`]; B3's disk share from its
//! CPU twins, [`atmo_handover`]). The exposure factor blends from `star_pre` to `scene_pre` with
//! the weight w = [`crate::lod_exposure_mix`] while eye-adapted stars are in effect
//! ([`exposure_weight`]), in log space: pre = `pre_mix(star_pre, scene_pre, w)` =
//! star_pre^(1 − w)·scene_pre^w (uni-gfx `common/exposure.wgsl`, CPU twin
//! `uni_gfx::frame::pre_mix`; task B11), so the exposure moves by equal stops per step of w and
//! the 1.5–4 px band spreads the star/scene gap evenly instead of dropping it just below 4 px.
//!
//! **Star share (lead ruling B6 I1, task B10).** Every uni-planet pass that draws at
//! pre = pre_mix(star_pre, scene_pre, w) with w < 1 (bodies, rings, star surfaces in the 1.5–4 px
//! band) stores s = 1 − w in `lens_mask.a` on each fragment that covers anything, blended with
//! `Max` ([`lens_mask_target`], [`star_share`]); the clear value 0 means scene exposure. The meter
//! (uni-post `histogram.wgsl`, pass 9) divides each pixel by pre_mix(scene_pre, star_pre, s),
//! which is the writer's pre, so a body in the band is metered at its true luminance and the
//! exposure loop cannot run away. w is snapped to 1 when 1 − w < [`STAR_SHARE_MIN`] = 2⁻¹⁴, f16's
//! smallest normal, which Metal may otherwise flush to 0 on store ([`exposure_weight`]). The app
//! gives a body's atmosphere the same w (`uni_atmo::AtmoDraw::exposure_mix`), so its in-scatter
//! shares the body's exposure and its share.
//!
//! No Earth- or Venus-specific calibration constant lives here (constraints §4a, W-B1..W-B3);
//! `World::objects()` grows at run time, so nothing is sized per `ObjectId`: materials are keyed
//! by NAIF id and the per-frame buffers grow with the number of bodies drawn.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use anyhow::Context as _;
use glam::{DMat3, DVec3, Vec2, Vec3};
use uni_atmo::{GroundLighting, VENUS_CLOUD_TOP_M};
use uni_core::{DAY, R_SUN_M, TEFF_SUN_K, blackbody_linear_srgb};
use uni_data::{BodyFacts, MapInfo, RingProfile, TexFormat, f32_to_f16};
use uni_gfx::{
    DEPTH_FORMAT, FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, LENS_MASK_FORMAT,
    PassTimer, RenderFeature, ShaderLib, SharedInput,
};
use uni_physics::photometry::{
    Brdf, disk_phase_function, planet_phase_correction, rho_from_geometric_albedo,
};
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{BodyView, MAX_PHASE_CORRECTION, StarSurfaceView};

use crate::capmesh::{CAP_RINGS, CAP_SEGMENTS, cap_indices, cap_mesh};
use crate::lod::lod_exposure_mix;
use crate::material::{
    BT709, GpuMaps, Loaded, MaterialTable, SurfaceParams, load_pair, solid_texture,
};

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

/// Cloud-top super-rotation period of Venus [days]: ~100 m/s westward zonal wind at the cloud
/// tops (65–70 km), a ~4-day rotation against the 243-day solid body (Sánchez-Lavega et al.
/// 2017, "The Atmospheric Dynamics of Venus", Space Sci. Rev. 212, 1541; checked against its
/// abstract, 2026-11-23). At the 70 km deck, 2π·6122 km / 4.0 d = 111 m/s.
pub const VENUS_CLOUD_TOP_PERIOD_D: f64 = 4.0;

/// NAIF id of Venus, whose drawn ellipsoid is its cloud deck.
const VENUS: i32 = 299;
/// NAIF id of Saturn, whose rings shade its globe.
const SATURN: i32 = 699;

/// The outline ring of each body's cap is pushed this many pixels (on-axis pixel angles, the
/// largest) away from the body's centre, so every pixel whose centre lies within the 0.5-px AA
/// rim outside the limb is rasterised. B0's cap hugs the limb (≤ 0.255 px off), which would cut
/// the rim's outer half.
pub(crate) const AA_MARGIN_PX: f64 = 1.0;

/// Bindings of group 1 (`body.wgsl`): the uniform, albedo, clouds, map sampler, ring, ring
/// sampler.
pub(crate) const BODY_BINDINGS: u32 = 6;

/// Smallest non-zero star share a uni-planet pass stores in `lens_mask.a`: 2⁻¹⁴, the smallest
/// normal binary16 (IEEE 754-2019 §3.6; `lens_mask` is Rgba16Float). Metal does not promise to
/// keep f16 subnormals when it stores a render target, and a share flushed to 0 reads as scene
/// exposure while the light was written at pre_mix(star_pre, scene_pre, w): the meter then reads
/// it 2^(s·(EV_scene − EV_star)) too bright, under 70·2⁻¹⁴ ≈ 0.004 stop for any s < 2⁻¹⁴ over
/// the whole EV100 range −30…40 (uni-post `band_share_flushed_below_f16_normal_stays_bounded`;
/// with B10's linear blend it was 1 + s·(2^(EV_scene − EV_star) − 1), which ran the loop away
/// for a scene EV well above the star EV). [`exposure_weight`] snaps w to 1 anyway (lead ruling
/// B6 I1), so a stored share is 0 or an f16 normal and the meter reads back exactly the writer's
/// pre.
pub const STAR_SHARE_MIN: f32 = 1.0 / 16384.0;

/// The exposure weight w with which the uni-planet passes draw a resolved body, its rings or a
/// star surface of projected radius `proj_radius_px` [px]: pre = pre_mix(star_pre, scene_pre, w),
/// the log-space blend (M1 frame contract, "Exposure split"; task B11).
///
/// While eye-adapted stars are in effect it is [`crate::lod_exposure_mix`] snapped by
/// [`snap_exposure_weight`] (0 at ≤ 1.5 px, 1 from about 3.98 px on), otherwise 1 (scene
/// exposure; `star_pre == scene_pre` then). The star share the pass stores is 1 − w
/// ([`star_share`]). The app must give the body's atmosphere this same value
/// (`uni_atmo::AtmoDraw::exposure_mix`, lead ruling B6 I1): its in-scatter then shares the body's
/// exposure, which ruling W-B1's handover ([`flux_match`] with σ = 1) relies on.
pub fn exposure_weight(proj_radius_px: f64, eye_adapted: bool) -> f32 {
    if eye_adapted {
        snap_exposure_weight(lod_exposure_mix(proj_radius_px))
    } else {
        1.0
    }
}

/// `w`, or 1 when 1 − w < [`STAR_SHARE_MIN`], so the share 1 − w is either 0 or an f16 normal.
/// 1 − smoothstep(t) ≈ 3(1 − t)², so the snap acts only for radii in about (3.982, 4) px, keeps w
/// non-decreasing in the radius, and moves it by less than 2⁻¹⁴. With the log-space blend
/// (task B11) that moves pre by less than 2⁻¹⁴·(EV_scene − EV_star) stops, 0.0012 stop at a
/// 19-stop gap: no visible step (with B10's linear blend the same snap stepped the brightness by
/// 1 + 2⁻¹⁴·(2^ΔEV − 1), 33× at 19 stops). The CPU applies it before the upload, so the pass's pre
/// and its share use the same w (in f32, 1 − w is exact for w ≥ ½). Kept in uni-planet rather
/// than in [`crate::lod_exposure_mix`]: uni-atmo snaps the w it is given with its own twin
/// (`uni_atmo::feature::atmo_exposure_mix`, which cannot depend on uni-planet), so moving the snap
/// would not remove that duplicate.
pub(crate) fn snap_exposure_weight(w: f32) -> f32 {
    if 1.0 - w < STAR_SHARE_MIN { 1.0 } else { w }
}

/// CPU twin of `star_share` in `body.wgsl`, `rings.wgsl` and `star_surface.wgsl`: the star share a
/// fragment drawn with exposure weight `w` and coverage (the rings: opacity) `coverage` stores in
/// `lens_mask.a`, blended with `Max` (lead ruling B6 I1): 1 − w where it covers anything, else 0.
/// The meter (uni-post `histogram.wgsl`, `meter_pre`) reads it back as the pixel's pre,
/// pre_mix(scene_pre, star_pre, s) = pre_mix(star_pre, scene_pre, w), the writer's own (0 =
/// scene exposure, the clear value).
pub fn star_share(w: f32, coverage: f32) -> f32 {
    if coverage > 0.0 { 1.0 - w } else { 0.0 }
}

/// The `lens_mask` target of the bodies, rings and star-surface passes: green (sky occlusion) and
/// alpha (the star share, [`star_share`]) written, both blended with `Max` (M1 frame contract;
/// WebGPU requires factor One for `Max`); red and blue are left to the lens pass.
pub(crate) fn lens_mask_target() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: LENS_MASK_FORMAT,
        blend: Some(wgpu::BlendState {
            color: MAX_BLEND,
            alpha: MAX_BLEND,
        }),
        write_mask: LENS_MASK_WRITES,
    }
}

/// The `lens_mask` channels the world writers own: `g` (sky occlusion) and `a` (star share).
pub(crate) const LENS_MASK_WRITES: wgpu::ColorWrites =
    wgpu::ColorWrites::GREEN.union(wgpu::ColorWrites::ALPHA);

/// Twin of `struct BodyUniform` in `body.wgsl` (vec4-sized members only; the field docs are in
/// the WGSL). One per body drawn, at a dynamic offset.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BodyUniform {
    /// Rows of the ICRS → body-fixed rotation; w 0.
    pub to_body_x: [f32; 4],
    pub to_body_y: [f32; 4],
    pub to_body_z: [f32; 4],
    /// Radii of the drawn ellipsoid [m]; w 0.
    pub radii: [f32; 4],
    /// Camera in unit-sphere space; w: c = |o|² − 1 (f64 on the CPU).
    pub cam_unit: [f32; 4],
    /// Sun − body centre, body-fixed [m]; w: the Sun's radius [m].
    pub sun: [f32; 4],
    /// E☉ × Sun colour × C; w: exposure weight.
    pub light: [f32; 4],
    /// Reflectance multiplier; w: lunar-Lambert L.
    pub albedo: [f32; 4],
    /// Has map, cloud scale, albedo lon0 [deg], cloud lon0 [deg].
    pub maps: [f32; 4],
    /// Longitude offset [rad], ring inner, ring outer [m], occluder count.
    pub extra: [f32; 4],
    /// Ruling W-B1's handover: w, A, B, k_max ([`flux_match`]).
    pub handover: [f32; 4],
    /// Occluder − body centre, body-fixed [m]; w radius [m].
    pub occluders: [[f32; 4]; 4],
}

/// The camera in a body's unit-sphere space, in f64.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraUnit {
    /// Camera position (body-fixed, divided by the radii).
    pub o: DVec3,
    /// |o|² − 1, > 0 outside (review I5: computed here in f64, not in the shader).
    pub c: f64,
}

/// The ellipsoid the pass draws from outside it: the body's radii, or for Venus its cloud deck
/// [`VENUS_CLOUD_TOP_M`] (70 km, Titov et al. 2018) above them, where the cloud-top map lives.
/// Under the deck [`placement`] draws the solid radii instead.
pub(crate) fn drawn_radii(view: &BodyView) -> DVec3 {
    if view.naif == Some(VENUS) {
        view.radii_m + DVec3::splat(VENUS_CLOUD_TOP_M)
    } else {
        view.radii_m
    }
}

/// The camera in the unit-sphere space of the ellipsoid `radii`, or `None` when it is inside
/// that ellipsoid (or on it, or anything is non-finite). [`placement`] decides what is drawn.
pub(crate) fn camera_unit(view: &BodyView, radii: DVec3) -> Option<CameraUnit> {
    let o = (view.orient.inverse() * -view.rel_pos_m) / radii;
    let c = o.length_squared() - 1.0;
    (o.is_finite() && c > 0.0 && c.is_finite()).then_some(CameraUnit { o, c })
}

/// What the pass does with one body this frame.
#[derive(Clone, Debug)]
pub(crate) enum Placement {
    /// Draw the ellipsoid of `radii`; the camera `cam` is outside it.
    Draw { radii: DVec3, cam: CameraUnit },
    /// Draw nothing; the HUD warning says why.
    Skip(String),
}

/// Where the camera is relative to `view` (named `name` in warnings), and so what is drawn
/// (review I2, lead ruling of fix round 1; Review Focus 2: a body never vanishes because the
/// camera crossed its proxy):
/// - outside the drawn ellipsoid ([`drawn_radii`]): that ellipsoid;
/// - between the solid radii and the drawn ones (Venus, 0–70 km up, under its cloud deck): the
///   solid radii, the surface under the deck, with no warning. Venus has no surface map, so the
///   solid sphere carries the cloud-top map, and it is lit as at the deck (uni-atmo clamps the
///   ground-lighting radius to its bottom radius, the deck). uni-atmo's pass draws nothing for
///   Venus there yet (its `camera_origin` is `None` more than 1 m below the deck), so no clouds
///   or haze surround the camera (review r1 N1, N2; a lead decision);
/// - inside (or on) the solid body: nothing, "Inside <name>";
/// - non-finite view data: nothing, and a warning that does not claim the camera is inside.
pub(crate) fn placement(view: &BodyView, name: &str) -> Placement {
    let drawn = drawn_radii(view);
    if let Some(cam) = camera_unit(view, drawn) {
        return Placement::Draw { radii: drawn, cam };
    }
    if drawn != view.radii_m
        && let Some(cam) = camera_unit(view, view.radii_m)
    {
        return Placement::Draw {
            radii: view.radii_m,
            cam,
        };
    }
    let o = (view.orient.inverse() * -view.rel_pos_m) / view.radii_m;
    let c = o.length_squared() - 1.0;
    if o.is_finite() && c.is_finite() && c <= 0.0 {
        Placement::Skip(format!("Inside {name}"))
    } else {
        Placement::Skip(format!("Not drawn: non-finite position or radii ({name})"))
    }
}

/// `x` (camera-relative) turned `delta` [rad] further from the unit direction `axis`, at the same
/// distance from the camera.
fn push_from_axis(x: DVec3, axis: DVec3, delta: f64) -> DVec3 {
    let along = x.dot(axis);
    let perp = x - axis * along;
    let p_len = perp.length();
    if !(p_len > 0.0) {
        return x;
    }
    let angle = p_len.atan2(along) + delta;
    (axis * angle.cos() + perp * (angle.sin() / p_len)) * x.length()
}

/// The body's cap as camera-relative f32 vertices [m] in root axes: [`cap_mesh`] around `cam`,
/// scaled by `radii`, rotated by the body's orientation and offset by its position, all in f64,
/// then cast. The outline ring is turned `margin_rad` away from the body's centre direction
/// ([`AA_MARGIN_PX`]). Indices: [`cap_indices`]`(CAP_RINGS, CAP_SEGMENTS)`.
pub(crate) fn mesh_vertices(
    view: &BodyView,
    radii: DVec3,
    cam: &CameraUnit,
    margin_rad: f64,
) -> Vec<[f32; 3]> {
    let unit = cap_mesh(cam.o, CAP_RINGS, CAP_SEGMENTS);
    let outline = 1 + (CAP_RINGS as usize - 1) * CAP_SEGMENTS as usize;
    let axis = view.rel_pos_m.normalize_or_zero();
    unit.iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = view.orient * (v * radii) + view.rel_pos_m;
            let x = if i >= outline {
                push_from_axis(x, axis, margin_rad)
            } else {
                x
            };
            x.as_vec3().to_array()
        })
        .collect()
}

/// Whether a sphere of `radius_m` at `rel_m` (camera-relative, root axes) can touch the view:
/// not entirely behind the camera and not entirely outside one of the four side planes of the
/// symmetric frustum of `camera` (fov_y, `aspect` = w/h). Conservative: `radius_m` should carry
/// the AA rim's margin.
pub(crate) fn sphere_in_view(
    camera: &uni_core::CameraView,
    aspect: f64,
    rel_m: DVec3,
    radius_m: f64,
) -> bool {
    let p = camera.orient.inverse() * rel_m;
    let tan_y = (0.5 * camera.fov_y_rad).tan();
    let (hy, hx) = (tan_y.atan(), (aspect * tan_y).atan());
    // Outward unit normals of the side planes through the camera (it looks along −Z).
    let planes = [
        DVec3::new(hx.cos(), 0.0, hx.sin()),
        DVec3::new(-hx.cos(), 0.0, hx.sin()),
        DVec3::new(0.0, hy.cos(), hy.sin()),
        DVec3::new(0.0, -hy.cos(), hy.sin()),
        DVec3::Z,
    ];
    planes.iter().all(|n| n.dot(p) <= radius_m)
}

/// Angle between two vectors [rad], accurate near 0 and π.
fn angle(a: DVec3, b: DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b))
}

/// The per-body photometric factors (the same at every pixel).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BodyLight {
    /// Phase angle Sun – body – camera [rad].
    pub phase_rad: f64,
    /// Mallama & Hilton's C(α), clamped as in the point path.
    pub phase_correction: f64,
    /// Ruling W-B1's flux weight w = [`lod_exposure_mix`], snapped ([`snap_exposure_weight`]): 0
    /// at 1.5 px, 1 from about 3.98 px on.
    pub flux_weight: f64,
    /// A: the point's flux over this pass's own disk flux at k = 1 (1 when there is nothing to
    /// match).
    pub point_share: f64,
    /// B: the atmosphere pass's in-scatter flux over this pass's disk flux at k = 1 (0 without
    /// an atmosphere).
    pub inscatter_share: f64,
    /// Exposure weight: 0 `star_pre` … 1 `scene_pre` ([`exposure_weight`]; the shader draws at
    /// `pre_mix(star_pre, scene_pre, w)`, task B11); the pass stores the star share 1 − this in
    /// `lens_mask.a`.
    pub exposure_mix: f32,
}

impl BodyLight {
    /// The uniform's `handover` vec4: (w, A, B, [`FLUX_MATCH_MAX`]).
    pub fn handover(&self) -> [f32; 4] {
        [
            self.flux_weight as f32,
            self.point_share as f32,
            self.inscatter_share as f32,
            FLUX_MATCH_MAX,
        ]
    }

    /// Ruling W-B1's k at σ ([`flux_match`]); the shader passes σ = 1 (the atmosphere draws at
    /// this body's exposure).
    #[cfg(test)]
    pub fn flux_match(&self, sigma: f64) -> f64 {
        f64::from(flux_match(self.handover(), sigma as f32))
    }
}

/// The factors [`body_uniform`] uploads for `view` drawn with `radii`. `disk_rho` is the disk-
/// averaged reflectance (the map's ρ̄, or the uniform ρ; `None` when unknown), `disk_colour` the
/// map's mean colour normalised to luminance 1 (1 without a map), `sun_color` the Sun's colour
/// (luminance 1). `atmo` gives B3's atmosphere over the body at a phase angle
/// ([`atmo_handover`]); it is called only for a body with an atmosphere inside the 1.5–4 px band.
///
/// - C = `planet_phase_correction(naif, brdf, α)` clamped to ≤ `MAX_PHASE_CORRECTION`, the
///   point path's clamp (A3 ruling), 1 without a NAIF id or for a NaN. C holds its value beyond
///   each law's fitted range; for Earth and Mars that hold starts at 170°, A3's choice, accepted
///   by the lead (A3 ruling).
/// - Ruling W-B1: a body drawn with an atmosphere shows its real ground (Earth's ρ̄ 0.302 against
///   ρ(p_V) 0.651) under B3's atmosphere, which adds its in-scatter and dims the ground. Its
///   point flux cross-fades to that whole rendering over the 1.5–4 px band of
///   [`lod_exposure_mix`]: this pass's light is scaled by k = [`flux_match`]`((w, A, B, k_max),
///   1)`, so at 1.5 px this pass and the atmosphere pass together send the point's flux, and from
///   4 px on k = 1. σ = 1 because B3 draws its in-scatter at every size, unfaded, at this body's
///   own exposure weight (lead ruling B6 I1: the app sets `uni_atmo::AtmoDraw::exposure_mix` to
///   [`exposure_weight`] of the same body; before task B10 it drew at scene_pre and σ was
///   scene_pre/pre). k, A and B are physical ratios: none depends on how the exposure blends
///   (task B11 changed it to the log-space `pre_mix`; body and atmosphere share it, so σ stays
///   1). In units of E·R_mean²/d², with ρ_p = ρ(p_V), R_d the drawn mean radius, R_b the preset's
///   bottom radius and lum = BT.709 ⊙ the Sun's colour:
///   - the point (uni-sim `reflected_lux`, Lambert): P = ρ_p·⅔Φ(α)·C;
///   - this pass at k = 1: D = C·ρ̄·⅔Φ(α)·(R_d/R_mean)²·lum·(colour ⊙ ground factor);
///   - B3's in-scatter: I = lum·in-scatter·(R_b/R_mean)²;
///   - A = P/D, B = I/D (bounded by [`HANDOVER_SHARE_MAX`]).
///
///   Without B3's table (a body with no uni-atmo preset) A is the bare-ground ratio
///   (ρ_p/ρ̄)·(R_mean/R_d)² and B = 0, fix round 0's k₀. A ground factor below
///   [`HANDOVER_GROUND_MIN`] (Titan) leaves A = 1, B = 0. Airless bodies, w = 1, or no map
///   statistics (`disk_rho` `None`, a 2k copy missing; the HUD says so, review I1) give A = 1,
///   B = 0, so k = 1.
/// - w is snapped ([`snap_exposure_weight`]); the exposure weight is [`exposure_weight`]: w while
///   eye-adapted stars are in effect, else 1 (scene exposure).
pub(crate) fn body_light(
    view: &BodyView,
    radii: DVec3,
    disk_rho: Option<f64>,
    disk_colour: Vec3,
    sun_color: Vec3,
    eye_adapted: bool,
    atmo: impl FnOnce(f64) -> Option<AtmoDisk>,
) -> BodyLight {
    let phase = angle(view.sun_rel_m, -view.rel_pos_m);
    let raw = view
        .naif
        .map_or(1.0, |n| planet_phase_correction(n, view.brdf, phase));
    let phase_correction = if raw.is_nan() {
        1.0
    } else {
        raw.min(MAX_PHASE_CORRECTION)
    };
    let w = f64::from(snap_exposure_weight(lod_exposure_mix(view.proj_radius_px)));
    let drawn_mean = (radii.x * radii.y * radii.z).cbrt();
    let (mut a, mut b) = (1.0, 0.0);
    if let Some(rho) = disk_rho
        && view.has_atmosphere
        && rho > 0.0
        && drawn_mean > 0.0
        && w < 1.0
    {
        let r_mean = view.mean_radius_m;
        let rho_p = rho_from_geometric_albedo(view.brdf, view.geometric_albedo);
        // The bare-ground ratio: a Lambert sphere of ρ̄ drawn at R_d against the point.
        a = rho_p / rho * (r_mean / drawn_mean).powi(2);
        if view.brdf == Brdf::Lambert
            && let Some(disk) = atmo(phase)
        {
            let lum = BT709 * sun_color.as_dvec3();
            let ground = (lum * disk_colour.as_dvec3() * disk.ground_factor).element_sum();
            if ground >= HANDOVER_GROUND_MIN {
                a /= ground;
                let d = phase_correction
                    * rho
                    * (2.0 / 3.0)
                    * disk_phase_function(Brdf::Lambert, phase)
                    * (drawn_mean / r_mean).powi(2)
                    * ground;
                let i = lum.dot(disk.inscatter) * (disk.bottom_m / r_mean).powi(2);
                b = (i / d).min(HANDOVER_SHARE_MAX);
            } else {
                a = 1.0;
            }
        }
    }
    if !(a.is_finite() && a >= 0.0) {
        a = 1.0;
    }
    if !(b.is_finite() && b >= 0.0) {
        b = 0.0;
    }
    BodyLight {
        phase_rad: phase,
        phase_correction,
        flux_weight: w,
        point_share: a,
        inscatter_share: b,
        exposure_mix: exposure_weight(view.proj_radius_px, eye_adapted),
    }
}

/// Packs one body's uniform. `ring` is Saturn's ring shadow (inner, outer radius [m]);
/// `lon_offset_rad` shifts the map in longitude (Venus's cloud tops).
#[allow(clippy::too_many_arguments)]
pub(crate) fn body_uniform(
    view: &BodyView,
    radii: DVec3,
    cam: &CameraUnit,
    light: &BodyLight,
    surface: &SurfaceParams,
    sun_color: Vec3,
    ring: Option<(f64, f64)>,
    lon_offset_rad: f64,
) -> BodyUniform {
    let to_body = DMat3::from_quat(view.orient.inverse());
    let rows = to_body.transpose();
    let row = |v: DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    let vec4 = |v: DVec3, w: f64| [v.x as f32, v.y as f32, v.z as f32, w as f32];
    let lambert_l = match view.brdf {
        Brdf::Lambert => 0.0,
        Brdf::LunarLambert { l } => l as f32,
    };
    let e = view.sun_illuminance_lux * light.phase_correction;
    let rgb = sun_color.as_dvec3() * e;
    let mut occluders = [[0.0; 4]; 4];
    let mut count = 0;
    for (slot, occ) in occluders.iter_mut().zip(&view.occluders) {
        *slot = vec4(to_body * (occ.rel_pos_m - view.rel_pos_m), occ.radius_m);
        count += 1;
    }
    let (inner, outer) = ring.unwrap_or((0.0, 0.0));
    let (cloud_scale, cloud_lon0) = surface.clouds.unwrap_or((0.0, 0.0));
    BodyUniform {
        to_body_x: row(rows.x_axis),
        to_body_y: row(rows.y_axis),
        to_body_z: row(rows.z_axis),
        radii: vec4(radii, 0.0),
        cam_unit: vec4(cam.o, cam.c),
        sun: vec4(to_body * view.sun_rel_m, R_SUN_M),
        light: [rgb.x as f32, rgb.y as f32, rgb.z as f32, light.exposure_mix],
        albedo: [surface.albedo, surface.albedo, surface.albedo, lambert_l],
        maps: [
            if surface.has_map { 1.0 } else { 0.0 },
            cloud_scale,
            surface.lon0_deg,
            cloud_lon0,
        ],
        extra: [
            lon_offset_rad as f32,
            inner as f32,
            outer as f32,
            count as f32,
        ],
        handover: light.handover(),
        occluders,
    }
}

/// One body drawn this frame.
struct Draw {
    slot: usize,
    naif: Option<i32>,
    uniform_offset: u32,
    base_vertex: i32,
}

/// One material's texture views (slot 0: the uniform default).
struct SlotViews {
    albedo: wgpu::TextureView,
    clouds: wgpu::TextureView,
}

/// Saturn's ring profile on the GPU, for the ring shadow on the globe.
struct RingShadow {
    inner_m: f64,
    outer_m: f64,
}

/// The bodies pass (`RenderFeature` named "bodies").
pub struct BodyRenderer {
    pipeline: wgpu::RenderPipeline,
    body_bgl: wgpu::BindGroupLayout,
    /// Every texture the slots and the ring view use, held for the renderer's lifetime.
    _textures: Vec<wgpu::Texture>,
    slots: Vec<SlotViews>,
    ring_view: wgpu::TextureView,
    map_sampler: wgpu::Sampler,
    ring_sampler: wgpu::Sampler,
    /// One per slot, over the current uniform buffer.
    bind_groups: Vec<wgpu::BindGroup>,
    uniform_buffer: wgpu::Buffer,
    /// Bytes between two bodies' uniforms (the device's dynamic-offset alignment).
    uniform_stride: u64,
    /// Bodies the uniform buffer holds.
    uniform_capacity: usize,
    vertex_buffer: wgpu::Buffer,
    /// Vertices the vertex buffer holds.
    vertex_capacity: usize,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    ground: Arc<GroundLighting>,
    input: SharedInput<BodyScene>,
    table: MaterialTable,
    ring: Option<RingShadow>,
    /// The Sun's colour, linear Rec. 709 with luminance 1 (the point path's `sun_color`).
    sun_color: Vec3,
    draws: Vec<Draw>,
    warnings: Vec<String>,
    /// This frame's uniforms (`uniform_stride` bytes per draw) and cap vertices, kept between
    /// frames so they are not reallocated.
    frame_uniforms: Vec<u8>,
    frame_vertices: Vec<[f32; 3]>,
}

/// Vertices of one body's cap.
const CAP_VERTICES: usize = 1 + (CAP_RINGS * CAP_SEGMENTS) as usize;
/// Bodies the buffers hold before they first grow.
const INITIAL_BODIES: usize = 16;

impl BodyRenderer {
    /// Loads the maps (`map_dir` holds `maps/index.ron`'s files: the BC file when
    /// `GpuContext::has_bc_compression()`, else the 2k `fallback_file`), builds the pipeline and
    /// keeps `ground` (group 2) and `input`. A body without a map draws uniform albedo from its
    /// view's p_V and reports it in [`BodyRenderer::warnings`]. `ring` is Saturn's ring profile,
    /// for the rings' shadow on the globe. `facts` supplies display names.
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
        let device = &gpu.device;
        crate::register_shaders(lib);
        let module = lib.module(device, crate::BODY_WGSL)?;

        // Maps first: a failed map is a warning, never an error.
        let bc = gpu.has_bc_compression();
        let mut gpu_maps: Vec<GpuMaps> = Vec::new();
        let table = MaterialTable::build(maps, facts, |_, pair| {
            let loaded = load_pair(gpu, map_dir, pair, bc)?;
            let clouds = loaded.maps.clouds.is_some();
            gpu_maps.push(loaded.maps);
            Ok(Loaded {
                slot: gpu_maps.len(),
                stats: loaded.stats,
                clouds,
                note: loaded.note,
            })
        });
        let white = solid_texture(
            gpu,
            "body albedo (uniform)",
            TexFormat::Rgba8UnormSrgb,
            &[255; 4],
        )?;
        let clear = solid_texture(gpu, "body clouds (none)", TexFormat::R8Unorm, &[0])?;
        let view_of = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let mut slots = vec![SlotViews {
            albedo: view_of(&white),
            clouds: view_of(&clear),
        }];
        slots.extend(gpu_maps.iter().map(|m| SlotViews {
            albedo: view_of(&m.albedo),
            clouds: m.clouds.as_ref().map_or_else(|| view_of(&clear), view_of),
        }));
        let (ring_texture, ring) = ring_texture(gpu, ring)?;
        let ring_view = view_of(&ring_texture);
        let mut textures = vec![white, clear, ring_texture];
        for m in gpu_maps {
            textures.push(m.albedo);
            textures.extend(m.clouds);
        }

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let map_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("body maps"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let ring_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("body ring profile"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let body_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bodies bgl"),
            entries: &body_layout_entries(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bodies layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&body_bgl), Some(&ground.bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bodies"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<[f32; 3]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    // Premultiplied "over" (the factor is the fragment's own coverage), colour
                    // only: hdr's alpha is left alone, as the star pass leaves it.
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    // lens_mask.g = max(g, coverage), lens_mask.a = max(a, 1 − w) (M1 frame
                    // contract; lead ruling B6 I1).
                    Some(lens_mask_target()),
                ],
            }),
            // Every cap triangle faces the camera; no culling needed.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let indices = cap_indices(CAP_RINGS, CAP_SEGMENTS);
        let index_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("body cap indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        );
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the bodies pipeline: {err}");
        }
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment.max(1));
        let uniform_stride = (size_of::<BodyUniform>() as u64).div_ceil(alignment) * alignment;
        let mut renderer = Self {
            pipeline,
            body_bgl,
            _textures: textures,
            slots,
            ring_view,
            map_sampler,
            ring_sampler,
            bind_groups: Vec::new(),
            uniform_buffer: uniform_buffer(gpu, uniform_stride, INITIAL_BODIES),
            uniform_stride,
            uniform_capacity: INITIAL_BODIES,
            vertex_buffer: vertex_buffer(gpu, INITIAL_BODIES * CAP_VERTICES),
            vertex_capacity: INITIAL_BODIES * CAP_VERTICES,
            index_buffer,
            index_count: u32::try_from(indices.len()).context("cap index count")?,
            ground,
            input,
            table,
            ring,
            sun_color: blackbody_linear_srgb(TEFF_SUN_K),
            draws: Vec::new(),
            warnings: Vec::new(),
            frame_uniforms: Vec::new(),
            frame_vertices: Vec::new(),
        };
        renderer.rebuild_bind_groups(gpu);
        // Ruling W-B1's handover tables of the bodies with an atmosphere (review I3), built in
        // the background before any of them reaches the 1.5–4 px band.
        warm_atmo_handover(
            facts
                .iter()
                .map(|f| f.naif)
                .filter(|&n| uni_atmo::preset_for(n).is_some())
                .collect(),
        );
        Ok(renderer)
    }

    /// Warnings for the HUD from the last prepared frame, e.g. "No surface map: uniform albedo
    /// (Titania)", "Inside Earth".
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// Mean reflectance colour of body `naif`'s albedo map: linear Rec. 709, normalised to
    /// luminance 1; `None` for a body without a map. Computed on the CPU from the 2k map loaded
    /// at construction (its rotation-averaged phase-0 disk average, Earth's clouds composited).
    /// The app tints a reflected `uni_sky::PointSprite::color` by it, keeping the sprite's
    /// luminance (carry-over 6); bodies without a map stay solar-coloured.
    pub fn mean_albedo_rgb(&self, naif: i32) -> Option<Vec3> {
        self.table.mean_albedo_rgb(naif)
    }

    /// One bind group per material slot over the current uniform buffer.
    fn rebuild_bind_groups(&mut self, gpu: &GpuContext) {
        let uniform = wgpu::BufferBinding {
            buffer: &self.uniform_buffer,
            offset: 0,
            size: wgpu::BufferSize::new(size_of::<BodyUniform>() as u64),
        };
        self.bind_groups = self
            .slots
            .iter()
            .map(|slot| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bodies bg"),
                    layout: &self.body_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(uniform.clone()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&slot.albedo),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&slot.clouds),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.map_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.ring_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&self.ring_sampler),
                        },
                    ],
                })
            })
            .collect();
    }

    /// The display name of `view`: the app's, else the fact sheets' or the map index's.
    fn name_of(&self, scene: &BodyScene, view: &BodyView) -> String {
        scene
            .names
            .iter()
            .find(|(id, _)| *id == view.id)
            .map(|(_, n)| n.clone())
            .or_else(|| {
                view.naif
                    .and_then(|n| self.table.name(n))
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| format!("object {}", view.id))
    }

    /// Uploads this frame's uniforms and vertices, growing the buffers (and rebuilding the bind
    /// groups over a new uniform buffer) when more bodies are drawn than they hold.
    fn upload_frame(&mut self, gpu: &GpuContext, uniforms: &[u8], vertices: &[[f32; 3]]) {
        if self.draws.is_empty() {
            return;
        }
        if self.draws.len() > self.uniform_capacity {
            self.uniform_capacity = self.draws.len().next_power_of_two();
            self.uniform_buffer = uniform_buffer(gpu, self.uniform_stride, self.uniform_capacity);
            self.rebuild_bind_groups(gpu);
        }
        if vertices.len() > self.vertex_capacity {
            self.vertex_capacity = self.draws.len().next_power_of_two() * CAP_VERTICES;
            self.vertex_buffer = vertex_buffer(gpu, self.vertex_capacity);
        }
        gpu.queue.write_buffer(&self.uniform_buffer, 0, uniforms);
        gpu.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
    }

    /// Builds this frame's draws, uniforms and vertices from `scene` (far to near, so a nearer
    /// body's AA rim blends over a farther body).
    fn build_frame(
        &mut self,
        scene: &BodyScene,
        ctx: &FrameCtx,
        uniforms: &mut Vec<u8>,
        vertices: &mut Vec<[f32; 3]>,
    ) {
        let pixel_rad = 2.0 * (0.5 * ctx.camera.fov_y_rad).tan() / f64::from(ctx.viewport.y.max(1));
        let aspect = f64::from(ctx.viewport.x.max(1)) / f64::from(ctx.viewport.y.max(1));
        let venus_offset = venus_cloud_longitude_offset_deg(ctx.epoch).to_radians();
        let mut order: Vec<&BodyView> = scene.bodies.iter().collect();
        order.sort_by(|a, b| {
            b.rel_pos_m
                .length_squared()
                .total_cmp(&a.rel_pos_m.length_squared())
        });
        for view in order {
            let name = self.name_of(scene, view);
            let (radii, cam) = match placement(view, &name) {
                Placement::Draw { radii, cam } => (radii, cam),
                Placement::Skip(warning) => {
                    self.warnings.push(warning);
                    continue;
                }
            };
            // Off screen (the snapshot lists every resolved body): skip it, with two pixels of
            // margin at its distance for the AA rim and the pushed outline.
            let margin = 2.0 * AA_MARGIN_PX * pixel_rad * view.rel_pos_m.length();
            if !sphere_in_view(
                &ctx.camera,
                aspect,
                view.rel_pos_m,
                radii.max_element() + margin,
            ) {
                continue;
            }
            let (surface, notes) =
                self.table
                    .surface_for(view.naif, view.brdf, view.geometric_albedo, &name);
            self.warnings.extend(notes);
            let colour = view
                .naif
                .and_then(|n| self.table.mean_albedo_rgb(n))
                .unwrap_or(Vec3::ONE);
            let light = body_light(
                view,
                radii,
                surface.disk_rho,
                colour,
                self.sun_color,
                scene.eye_adapted,
                |phase| Some(atmo_handover(view.naif?)?.at(phase)),
            );
            let ring = self
                .ring
                .as_ref()
                .filter(|_| view.naif == Some(SATURN) && view.ring.is_some())
                .map(|r| (r.inner_m, r.outer_m));
            let offset = if surface.cloud_tops {
                venus_offset
            } else {
                0.0
            };
            let uniform = body_uniform(
                view,
                radii,
                &cam,
                &light,
                &surface,
                self.sun_color,
                ring,
                offset,
            );
            let mesh = mesh_vertices(view, radii, &cam, AA_MARGIN_PX * pixel_rad);
            if mesh.len() != CAP_VERTICES || !mesh.iter().flatten().all(|v| v.is_finite()) {
                continue;
            }
            let index = self.draws.len();
            let start = uniforms.len();
            uniforms.resize(start + self.uniform_stride as usize, 0);
            uniforms[start..start + size_of::<BodyUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
            self.draws.push(Draw {
                slot: surface.slot,
                naif: view.naif,
                uniform_offset: (index as u64 * self.uniform_stride) as u32,
                base_vertex: (index * CAP_VERTICES) as i32,
            });
            vertices.extend(mesh);
        }
    }
}

/// `BlendOperation::Max` (its factors must be One).
const MAX_BLEND: wgpu::BlendComponent = wgpu::BlendComponent {
    src_factor: wgpu::BlendFactor::One,
    dst_factor: wgpu::BlendFactor::One,
    operation: wgpu::BlendOperation::Max,
};

/// Group 1 of the bodies pipeline (see `body.wgsl`).
fn body_layout_entries() -> [wgpu::BindGroupLayoutEntry; BODY_BINDINGS as usize] {
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampler = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(size_of::<BodyUniform>() as u64),
            },
            count: None,
        },
        texture(1),
        texture(2),
        sampler(3),
        texture(4),
        sampler(5),
    ]
}

fn uniform_buffer(gpu: &GpuContext, stride: u64, bodies: usize) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("body uniforms"),
        size: stride * bodies as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn vertex_buffer(gpu: &GpuContext, vertices: usize) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("body caps"),
        size: (vertices * size_of::<[f32; 3]>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Saturn's ring opacity as one row of R16Float texels (sample i at u = (i + ½)/n, as the
/// profile's sample centres), or a 1 × 1 zero texel without a profile.
fn ring_texture(
    gpu: &GpuContext,
    profile: Option<&RingProfile>,
) -> anyhow::Result<(wgpu::Texture, Option<RingShadow>)> {
    let max = gpu.device.limits().max_texture_dimension_2d as usize;
    let usable = profile.filter(|p| !p.samples.is_empty() && p.samples.len() <= max);
    let texels: Vec<u8> = match usable {
        Some(p) => p
            .samples
            .iter()
            .flat_map(|s| f32_to_f16(s[3]).to_le_bytes())
            .collect(),
        None => vec![0; 2],
    };
    let width = (texels.len() / 2) as u32;
    let device = &gpu.device;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("body ring profile"),
        size: wgpu::Extent3d {
            width,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(2 * width),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("creating the ring-profile texture: {err}");
    }
    let shadow = usable.map(|p| RingShadow {
        inner_m: p.inner_m,
        outer_m: p.outer_m,
    });
    Ok((texture, shadow))
}

impl RenderFeature for BodyRenderer {
    fn name(&self) -> &'static str {
        "bodies"
    }

    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, _targets: &FrameTargets) {
        self.draws.clear();
        self.warnings.clear();
        let mut uniforms = std::mem::take(&mut self.frame_uniforms);
        let mut vertices = std::mem::take(&mut self.frame_vertices);
        uniforms.clear();
        vertices.clear();
        let input = self.input.clone();
        input.with(|scene| self.build_frame(scene, ctx, &mut uniforms, &mut vertices));
        self.upload_frame(gpu, &uniforms, &vertices);
        self.frame_uniforms = uniforms;
        self.frame_vertices = vertices;
    }

    fn encode(
        &mut self,
        _gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if self.draws.is_empty() {
            return;
        }
        let load = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bodies"),
            color_attachments: &[load(&targets.hdr_view), load(&targets.lens_mask_view)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: timer.render_writes("bodies"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for draw in &self.draws {
            let Some(group) = self.bind_groups.get(draw.slot) else {
                continue;
            };
            pass.set_bind_group(1, group, &[draw.uniform_offset]);
            pass.set_bind_group(2, self.ground.group(draw.naif), &[]);
            pass.draw_indexed(0..self.index_count, draw.base_vertex, 0..1);
        }
    }
}

/// Ray–ellipsoid intersection in the body's unit-sphere space (the ellipsoid scaled to a unit
/// sphere): origin `o_unit`, direction `d_unit` (not necessarily unit length), and
/// c = |o_unit|² − 1 computed in f64 on the CPU (review I5). Returns the entry and exit ray
/// parameters (t₀ ≤ t₁) or `None` on a miss (a zero direction misses). Numerically stable form
/// (Haines et al. 2019, *Ray Tracing Gems* ch. 7): the discriminant b² − ac is formed as
/// a·(1 − |o − (b/a)·d|²), from the ray's distance to the centre, and q = −(b + sign(b)√disc)
/// gives the roots q/a and c/q without cancellation. Twin of the WGSL in `body.wgsl` (same order
/// of operations).
pub fn ray_ellipsoid(o_unit: Vec3, d_unit: Vec3, c: f32) -> Option<(f32, f32)> {
    let a = d_unit.dot(d_unit);
    let b = o_unit.dot(d_unit);
    let l = o_unit - d_unit * (b / a);
    let disc = a * (1.0 - l.dot(l));
    if !(disc >= 0.0) {
        return None;
    }
    let q = -(b + if b >= 0.0 { 1.0 } else { -1.0 } * disc.sqrt());
    if q == 0.0 {
        return Some((0.0, 0.0));
    }
    let (t0, t1) = (q / a, c / q);
    Some((t0.min(t1), t0.max(t1)))
}

/// Texture coordinates of planetocentric (east) longitude `lon_rad` and latitude `lat_rad` in an
/// equirectangular map whose left edge is at east longitude `lon0_deg` (`uni_data::MapInfo`):
/// u runs east through 360°, v from +90° (0) to −90° (1). Twin of the WGSL in `body.wgsl`.
pub fn equirect_uv(lon_rad: f32, lat_rad: f32, lon0_deg: f32) -> Vec2 {
    use std::f32::consts::{PI, TAU};
    let x = (lon_rad - lon0_deg.to_radians()) / TAU;
    Vec2::new(x - x.floor(), 0.5 - lat_rad / PI)
}

/// Radius where the ray from `p` toward the Sun (unit `s`) crosses the ring plane z = 0 of the
/// body's axes inside [`inner`, `outer`]; `None` when the Sun is in the plane, the crossing is
/// behind `p`, or it falls outside the annulus. Twin of the WGSL in `body.wgsl`.
pub(crate) fn ring_crossing_radius(p: Vec3, s: Vec3, inner: f32, outer: f32) -> Option<f32> {
    if s.z == 0.0 {
        return None;
    }
    let t = -p.z / s.z;
    if !(t > 0.0) {
        return None;
    }
    let r = (p.truncate() + t * s.truncate()).length();
    (r >= inner && r <= outer).then_some(r)
}

/// Transmittance of Saturn's rings for sunlight reaching the globe point `p_body` (body axes,
/// ring plane z = 0) from the unit direction `sun_dir`: 1 when the ray to the Sun misses the
/// annulus [`inner`, `outer`], else 1 − `opacity_at(radius)` at the crossing. The opacity is used
/// as the profile gives it, with no slant-path correction, as the ring pass (B2) uses it. Twin of
/// the WGSL in `body.wgsl`, whose opacity is [`ring_profile_opacity`] of the loaded profile.
pub fn ring_transmittance(
    p_body: Vec3,
    sun_dir: Vec3,
    inner: f32,
    outer: f32,
    opacity_at: impl Fn(f32) -> f32,
) -> f32 {
    ring_crossing_radius(p_body, sun_dir, inner, outer).map_or(1.0, |r| 1.0 - opacity_at(r))
}

/// The profile's opacity at radius `r_m`, linear between sample centres inner + (i + ½)Δ and
/// held beyond the first and last: what the shader's linear, clamped sampling of the ring
/// texture gives (up to its f16 rounding).
pub fn ring_profile_opacity(profile: &RingProfile, r_m: f32) -> f32 {
    let n = profile.samples.len();
    if n == 0 {
        return 0.0;
    }
    let u = (f64::from(r_m) - profile.inner_m) / (profile.outer_m - profile.inner_m);
    let x = (u * n as f64 - 0.5).clamp(0.0, (n - 1) as f64);
    let i = x.floor() as usize;
    let j = (i + 1).min(n - 1);
    let f = (x - i as f64) as f32;
    profile.samples[i][3] * (1.0 - f) + profile.samples[j][3] * f
}

// ---------------------------------------------------------------------------------------------
// Ruling W-B1's handover with B3's atmosphere drawn (review I3)

/// Upper bound of ruling W-B1's flux factor k ([`flux_match`]). k grows as this pass's ground
/// carries less of the disk's light; at phase 0 the point's share A is 1.6–2.9 for Venus, Mars
/// and Earth and k at most A (`flux_match_cases`), and the bound keeps a thin crescent's ground
/// from being scaled without limit.
pub(crate) const FLUX_MATCH_MAX: f32 = 10.0;

/// Below this ground factor (the ground's light left by the atmosphere, against a bare Lambert
/// ground at the same phase) the ground cannot carry the point's flux: k stays 1 and the
/// handover is the atmosphere pass's alone (Titan: its ground sits under τ ≈ 11 of haze, factor
/// ~10⁻⁵).
const HANDOVER_GROUND_MIN: f64 = 0.05;

/// Bound of the in-scatter share B, so a Φ(α) → 0 near phase 180° gives a large finite share
/// (k then clamps to 0), never ∞ or NaN.
const HANDOVER_SHARE_MAX: f64 = 1.0e6;

/// Phase-angle step of [`AtmoHandover`]'s tables [deg]: 0°, 10°, …, 180°.
const HANDOVER_PHASE_STEP_DEG: f64 = 10.0;
/// Entries of [`AtmoHandover`]'s tables.
const HANDOVER_PHASES: usize = 19;
/// Impact-parameter intervals of the disk quadrature over the ground (b < bottom radius), spaced
/// b = R·sin(π/2·s) so they crowd toward the limb.
const HANDOVER_GROUND_RINGS: usize = 24;
/// Impact-parameter intervals over the shell beyond the limb, b = R + (top − R)·s², crowded
/// toward the bottom where the air is densest.
const HANDOVER_SHELL_RINGS: usize = 24;
/// Azimuth intervals over the half disk (the integrands are mirror-symmetric about the plane of
/// the camera and the Sun).
const HANDOVER_AZIMUTHS: usize = 16;
/// Nodes of the sun-transmittance table in √μ_s over [0, 1].
const HANDOVER_SUN_T_NODES: usize = 64;
/// Nodes of the sky-irradiance table in μ_s over [[`HANDOVER_SKY_MU_MIN`], 1] (steps of 0.05).
const HANDOVER_SKY_NODES: usize = 29;
/// Lowest Sun zenith cosine of the sky-irradiance table (the Sun 23.6° below the horizon; the
/// sky is dark well before).
const HANDOVER_SKY_MU_MIN: f64 = -0.4;

/// One sample of the disk quadrature: the far camera on +x looks back along −x through the point
/// (y, z) of the projected disk; `weight` is the sample's area [m²].
#[derive(Clone, Copy, Debug)]
struct DiskSample {
    y: f64,
    z: f64,
    weight: f64,
}

/// A midpoint quadrature over the projected disk of a sphere of radius `bottom` and the shell up
/// to `top`: `ground_rings` rings at b = bottom·sin(π/2·s) (crowded toward the limb, where the
/// view path through the air lengthens), `shell_rings` at b = bottom + (top − bottom)·s²
/// (crowded toward the dense bottom of the shell), each split into `azimuths` sectors of the half
/// disk ψ ∈ [0, π] and weighted for both mirror halves.
fn disk_quadrature(
    bottom: f64,
    top: f64,
    ground_rings: usize,
    shell_rings: usize,
    azimuths: usize,
) -> Vec<DiskSample> {
    use std::f64::consts::{FRAC_PI_2, PI};
    let ground = (0..ground_rings).map(|k| {
        let s = (k as f64 + 0.5) / ground_rings as f64;
        let b = bottom * (FRAC_PI_2 * s).sin();
        (
            b,
            b * bottom * FRAC_PI_2 * (FRAC_PI_2 * s).cos() / ground_rings as f64,
        )
    });
    let shell = (0..shell_rings).map(|k| {
        let s = (k as f64 + 0.5) / shell_rings as f64;
        let b = bottom + (top - bottom) * s * s;
        (b, b * 2.0 * (top - bottom) * s / shell_rings as f64)
    });
    let d_psi = 2.0 * PI / azimuths as f64;
    ground
        .chain(shell)
        .flat_map(|(b, b_db)| {
            (0..azimuths).map(move |m| {
                let psi = PI * (m as f64 + 0.5) / azimuths as f64;
                DiskSample {
                    y: b * psi.cos(),
                    z: b * psi.sin(),
                    weight: b_db * d_psi,
                }
            })
        })
        .collect()
}

/// The ground lighting of one atmosphere at its bottom radius against the Sun's zenith cosine,
/// tabulated from uni-atmo's CPU twins (`transmittance`, `sky_irradiance`: the functions behind
/// `atmo_sun_transmittance` and `atmo_sky_irradiance`). The shader evaluates them at the same
/// radius (B3's contract: no terrain in M1).
struct GroundLightTable {
    /// Sun transmittance at √μ_s = i/(n − 1).
    sun_t: Vec<DVec3>,
    /// Sky irradiance per unit sun illuminance at μ_s = min + i·step.
    sky: Vec<DVec3>,
    sky_step: f64,
}

impl GroundLightTable {
    fn new(params: &uni_atmo::AtmosphereParams) -> Self {
        let r = params.bottom_radius_m;
        let sky_step = (1.0 - HANDOVER_SKY_MU_MIN) / (HANDOVER_SKY_NODES - 1) as f64;
        Self {
            sun_t: (0..HANDOVER_SUN_T_NODES)
                .map(|i| {
                    let s = i as f64 / (HANDOVER_SUN_T_NODES - 1) as f64;
                    uni_atmo::transmittance(params, r, s * s).as_dvec3()
                })
                .collect(),
            sky: (0..HANDOVER_SKY_NODES)
                .map(|i| {
                    let mu_s = HANDOVER_SKY_MU_MIN + i as f64 * sky_step;
                    uni_atmo::sky_irradiance(params, r, mu_s).as_dvec3()
                })
                .collect(),
            sky_step,
        }
    }

    /// μ₀·T_sun(μ₀) + E_sky(μ₀): what a Lambert ground of unit ρ multiplies by E/π (the shader's
    /// `refl * sun_t + sky`), linear between the nodes.
    fn lit(&self, mu0: f64) -> DVec3 {
        let direct = if mu0 > 0.0 {
            let x = mu0.min(1.0).sqrt() * (HANDOVER_SUN_T_NODES - 1) as f64;
            let i = (x.floor() as usize).min(HANDOVER_SUN_T_NODES - 2);
            self.sun_t[i].lerp(self.sun_t[i + 1], x - i as f64) * mu0
        } else {
            DVec3::ZERO
        };
        let x = ((mu0 - HANDOVER_SKY_MU_MIN) / self.sky_step).max(0.0);
        let i = (x.floor() as usize).min(HANDOVER_SKY_NODES - 2);
        direct + self.sky[i].lerp(self.sky[i + 1], (x - i as f64).min(1.0))
    }
}

/// B3's atmosphere over one body at one phase angle, per channel at (680, 550, 440) nm (B3's
/// channels, drawn as linear R, G, B).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AtmoDisk {
    /// The light this pass's Lambert ground sends after B3's pass, over a bare Lambert ground's
    /// at the same phase: ∫ T_view·(μ₀·T_sun + E_sky)/π dA ÷ ∫ max(μ₀, 0)/π dA on the same
    /// quadrature (T_view: the view ray's transmittance down to the ground, B3's dual-source
    /// blend dst × T; T_sun and E_sky: `atmo_sun_transmittance`, `atmo_sky_irradiance`).
    pub ground_factor: DVec3,
    /// B3's in-scatter over the disk and the shell beyond the limb per lux of sunlight, ÷ the
    /// bottom radius² (geometric-albedo units: Earth's is 0.063 at 550 nm and phase 0, B3's
    /// report).
    pub inscatter: DVec3,
    /// The preset's bottom radius [m] (Venus: its cloud tops).
    pub bottom_m: f64,
}

/// The two integrals of [`AtmoDisk`] for `params` at phase `phase_rad` on the quadrature
/// `samples` (from [`disk_quadrature`] with the preset's radii): (in-scatter, ground factor), the
/// factor `None` where the bare ground is unlit (phase 180°).
fn atmo_disk(
    params: &uni_atmo::AtmosphereParams,
    light: &GroundLightTable,
    samples: &[DiskSample],
    phase_rad: f64,
) -> (DVec3, Option<DVec3>) {
    use std::f64::consts::PI;
    let r = params.bottom_radius_m;
    // The camera far along +x, the Sun at phase α from it.
    let sun = DVec3::new(phase_rad.cos(), phase_rad.sin(), 0.0);
    let far = 2.0 * params.top_radius_m;
    let (mut ground, mut bare, mut inscatter) = (DVec3::ZERO, 0.0, DVec3::ZERO);
    for s in samples {
        let (scattered, t_view) = uni_atmo::segment_radiance(
            params,
            DVec3::new(far, s.y, s.z),
            DVec3::NEG_X,
            f64::INFINITY,
            sun,
        );
        inscatter += scattered.as_dvec3() * s.weight;
        let b2 = s.y * s.y + s.z * s.z;
        if b2 < r * r {
            let n = DVec3::new((r * r - b2).sqrt(), s.y, s.z) / r;
            let mu0 = n.dot(sun);
            ground += t_view.as_dvec3() * light.lit(mu0) * (s.weight / PI);
            bare += mu0.max(0.0) * s.weight / PI;
        }
    }
    let factor = (bare > 0.0).then(|| ground / bare);
    (inscatter / (r * r), factor)
}

/// [`AtmoDisk`] tables of one atmosphere body, against phase angle in
/// [`HANDOVER_PHASE_STEP_DEG`] steps. Built once per body (about 15 000 view marches, plus B3's
/// CPU LUTs on their first use).
#[derive(Clone, Debug)]
pub(crate) struct AtmoHandover {
    bottom_m: f64,
    ground_factor: [DVec3; HANDOVER_PHASES],
    inscatter: [DVec3; HANDOVER_PHASES],
}

impl AtmoHandover {
    /// The tables for `params` on the default quadrature.
    pub fn compute(params: &uni_atmo::AtmosphereParams) -> Self {
        let samples = disk_quadrature(
            params.bottom_radius_m,
            params.top_radius_m,
            HANDOVER_GROUND_RINGS,
            HANDOVER_SHELL_RINGS,
            HANDOVER_AZIMUTHS,
        );
        let light = GroundLightTable::new(params);
        let mut ground_factor = [DVec3::ONE; HANDOVER_PHASES];
        let mut inscatter = [DVec3::ZERO; HANDOVER_PHASES];
        for j in 0..HANDOVER_PHASES {
            let phase = (j as f64 * HANDOVER_PHASE_STEP_DEG).to_radians();
            let (i, g) = atmo_disk(params, &light, &samples, phase);
            inscatter[j] = i;
            // At 180° the bare ground is unlit: hold the factor of the previous phase.
            ground_factor[j] = g.unwrap_or(if j > 0 {
                ground_factor[j - 1]
            } else {
                DVec3::ONE
            });
        }
        Self {
            bottom_m: params.bottom_radius_m,
            ground_factor,
            inscatter,
        }
    }

    /// The atmosphere at phase `phase_rad`, linear between the tabulated phases.
    pub fn at(&self, phase_rad: f64) -> AtmoDisk {
        let x = (phase_rad.abs().to_degrees() / HANDOVER_PHASE_STEP_DEG)
            .clamp(0.0, (HANDOVER_PHASES - 1) as f64);
        let i = (x.floor() as usize).min(HANDOVER_PHASES - 2);
        let f = x - i as f64;
        AtmoDisk {
            ground_factor: self.ground_factor[i].lerp(self.ground_factor[i + 1], f),
            inscatter: self.inscatter[i].lerp(self.inscatter[i + 1], f),
            bottom_m: self.bottom_m,
        }
    }
}

/// The [`AtmoHandover`] of body `naif`, built on first use and shared (every renderer, every
/// thread); `None` for a body without a uni-atmo preset. A caller that finds it being built by
/// another thread ([`warm_atmo_handover`]) waits for it, so the result never depends on timing.
pub(crate) fn atmo_handover(naif: i32) -> Option<Arc<AtmoHandover>> {
    type Cell = Arc<OnceLock<Option<Arc<AtmoHandover>>>>;
    static CELLS: Mutex<Vec<(i32, Cell)>> = Mutex::new(Vec::new());
    let cell = {
        let mut cells = CELLS.lock().unwrap_or_else(PoisonError::into_inner);
        match cells.iter().find(|(n, _)| *n == naif) {
            Some((_, cell)) => Arc::clone(cell),
            None => {
                let cell = Cell::default();
                cells.push((naif, Arc::clone(&cell)));
                cell
            }
        }
    };
    cell.get_or_init(|| uni_atmo::preset_for(naif).map(|p| Arc::new(AtmoHandover::compute(&p))))
        .clone()
}

/// Builds the [`AtmoHandover`] of each of `naifs` on a background thread, so a body entering the
/// 1.5–4 px band seldom waits for its table (about 1.4 s per body in the unoptimised test build,
/// most of it B3's CPU LUTs). If the thread cannot start, each table is built on first use.
fn warm_atmo_handover(naifs: Vec<i32>) {
    if naifs.is_empty() {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("uni-planet handover".into())
        .spawn(move || {
            for naif in naifs {
                atmo_handover(naif);
            }
        });
}

/// Twin of `flux_match` in `body.wgsl`: ruling W-B1's factor k on this pass's light, from the
/// uniform's `handover` = (w, A, B, k_max) and σ = pre_atmo / pre (pre: this pass's exposure,
/// pre_atmo: the atmosphere pass's for the same body).
///
/// k = (1 − w)·A + w + B·(w − σ), clamped to [0, k_max]. With this pass's own disk flux D (at
/// k = 1), the point's P = A·D and the atmosphere pass's in-scatter I = B·D (written at
/// pre_atmo), the total k·D·pre + I·pre_atmo equals pre·[(1 − w)·P + w·(D + I)]: the point's
/// flux at 1.5 px (w = 0, pre = star_pre while eye-adapted stars are in effect), the real
/// rendering from 4 px (w = 1, k = 1, pre = scene_pre), and a linear cross-fade of the flux
/// between.
///
/// Since task B10 the atmosphere draws at its body's exposure weight (lead ruling B6 I1,
/// [`exposure_weight`]), so pre_atmo = pre and the shader passes σ = 1: k = (1 − w)·A + w +
/// B·(w − 1), linear in w in both exposure modes. (Before, B3 drew at scene_pre and σ was
/// scene_pre/pre.) Task B11 made pre itself the log-space blend pre_mix(star_pre, scene_pre, w)
/// (uni-gfx); since the body and its atmosphere use the same pre, σ = 1 and k are unchanged,
/// and the exposed total is pre_mix(star_pre, scene_pre, w)·[(1 − w)·P + w·(D + I)]: its
/// exposure falls by equal stops per step of w while the flux cross-fades
/// (`point_disk_handover_flux_continuous`).
pub(crate) fn flux_match(handover: [f32; 4], sigma: f32) -> f32 {
    let [w, a, b, k_max] = handover;
    ((1.0 - w) * a + w + b * (w - sigma)).clamp(0.0, k_max)
}

/// East-longitude offset [deg, in 0..360] of Venus's cloud-top map at epoch `t`: retrograde
/// super-rotation with a [`VENUS_CLOUD_TOP_PERIOD_D`] (4.0-day) period relative to the surface
/// (Sánchez-Lavega et al. 2017, Space Sci. Rev. 212, 1541), not the 243-day surface rate
/// (review m22); 0 at J2000. The shader looks the map up at body longitude − offset. The whole
/// days are reduced modulo the period first, so the offset stays exact at any epoch. On the card:
/// `Cloud tops: static map, super-rotating (approximate)`.
pub fn venus_cloud_longitude_offset_deg(t: uni_core::Epoch) -> f64 {
    let period = VENUS_CLOUD_TOP_PERIOD_D;
    let turns = ((t.day as f64).rem_euclid(period) + t.sec / DAY) / period;
    (-360.0 * turns).rem_euclid(360.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;
    use glam::{DQuat, DVec3};
    use uni_core::{AU, Epoch};
    use uni_gfx::frame::pre_mix;
    use uni_gfx::{ShaderLib, exposure_scale, validate_wgsl};
    use uni_physics::photometry::{
        Brdf, disk_illuminance_lux, disk_phase_function, planet_phase_correction,
        rho_from_geometric_albedo,
    };
    use uni_sim::snapshot::{MAX_PHASE_CORRECTION, RingView, proj_radius_px};
    use wgpu::naga;

    use crate::material::SurfaceParams;

    /// WGS 84 equatorial and polar radii [m] (NIMA TR8350.2, 3rd ed., 2000, Table 3.1).
    const EARTH_A: f64 = 6_378_137.0;
    const EARTH_C: f64 = 6_356_752.314_2;
    /// Saturn's 1-bar equatorial and polar radii [m] (Archinal et al. 2018, CMDA 130:22, Table 5).
    const SATURN_A: f64 = 60_268.0e3;
    const SATURN_C: f64 = 54_364.0e3;
    const LUNAR_LAMBERT: Brdf = Brdf::LunarLambert { l: 0.9 };

    /// `n` near-uniform unit directions (Fibonacci sphere).
    fn fibonacci_dirs(n: usize) -> Vec<DVec3> {
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        (0..n)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let r = (1.0 - z * z).sqrt();
                let phi = golden * i as f64;
                DVec3::new(r * phi.cos(), r * phi.sin(), z)
            })
            .collect()
    }

    /// The textbook-stable quadratic in f64, the reference for the f32 twin.
    fn ray_unit_sphere_f64(o: DVec3, d: DVec3, c: f64) -> Option<(f64, f64)> {
        let (a, b) = (d.length_squared(), o.dot(d));
        let disc = b * b - a * c;
        if disc < 0.0 {
            return None;
        }
        let q = -(b + b.signum() * disc.sqrt());
        let (t0, t1) = (q / a, c / q);
        Some((t0.min(t1), t0.max(t1)))
    }

    /// Review Focus 2: from the centre the exit distance is the radius along each axis; from
    /// inside the entry is behind (< 0) and the exit ahead; no direction gives a NaN, from the
    /// centre, inside, on the surface or outside.
    #[test]
    fn ray_ellipsoid_from_inside_and_centre() {
        for radii in [
            DVec3::new(EARTH_A, EARTH_A, EARTH_C),
            DVec3::new(SATURN_A, SATURN_A, SATURN_C),
        ] {
            for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                let r = radii.dot(axis);
                let d = (axis / radii).as_vec3();
                let (t0, t1) = ray_ellipsoid(Vec3::ZERO, d, -1.0).expect("hit from the centre");
                assert!(
                    ((f64::from(t1) - r) / r).abs() < 1e-6,
                    "{axis}: {t1} vs {r}"
                );
                assert!(
                    ((f64::from(t0) + r) / r).abs() < 1e-6,
                    "{axis}: {t0} vs −{r}"
                );
            }
            let inside = DVec3::new(0.3, -0.2, 0.5);
            let surface = DVec3::new(0.6, 0.0, 0.8);
            let outside = DVec3::new(2.0, 0.5, -1.0);
            for o in [DVec3::ZERO, inside, surface, outside] {
                let c = (o.length_squared() - 1.0) as f32;
                for dir in fibonacci_dirs(2000) {
                    let d = (dir / radii).as_vec3();
                    let hit = ray_ellipsoid(o.as_vec3(), d, c);
                    if let Some((t0, t1)) = hit {
                        assert!(t0.is_finite() && t1.is_finite() && t0 <= t1, "{o} {dir}");
                        // Both roots lie on the surface.
                        for t in [t0, t1] {
                            let p = o.as_vec3() + t * d;
                            assert!((p.length_squared() - 1.0).abs() < 1e-4, "{o} {dir} {t}");
                        }
                    }
                    if o.length() < 1.0 {
                        let (t0, t1) = hit.expect("every ray from inside hits");
                        assert!(t0 < 0.0 && t1 > 0.0, "{o} {dir}: {t0} {t1}");
                    }
                }
            }
        }
        // A zero direction is a miss, not a NaN.
        assert_eq!(ray_ellipsoid(Vec3::X * 2.0, Vec3::ZERO, 3.0), None);
    }

    /// Review Focus 2, review I5: 1 m to 10 km above Earth's ellipsoid, at nadir and 45°, the
    /// f32 twin with c from f64 gives the hit distance within 1 % of f64 (it is far better), and c
    /// never changes sign. The naive f32 c = |o|² − 1 is off by tens of percent at 1 m, which is
    /// why the CPU passes c.
    #[test]
    fn ray_ellipsoid_f32_near_surface() {
        let radii = DVec3::new(EARTH_A, EARTH_A, EARTH_C);
        let (lat, lon) = (37f64.to_radians(), 20f64.to_radians());
        let on_sphere = DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
        let surface = on_sphere * radii;
        let normal = (on_sphere / radii).normalize();
        let east = DVec3::Z.cross(normal).normalize();
        for h in [1.0, 10.0, 100.0, 1.0e4] {
            let o = (surface + normal * h) / radii;
            let c = o.length_squared() - 1.0;
            assert!(c > 0.0 && (c as f32) > 0.0, "h {h}: c {c}");
            for (label, dir) in [("nadir", -normal), ("45°", (east - normal).normalize())] {
                let d = dir / radii;
                let (want, _) = ray_unit_sphere_f64(o, d, c).unwrap();
                let (got, _) = ray_ellipsoid(o.as_vec3(), d.as_vec3(), c as f32).unwrap();
                let err = (f64::from(got) - want).abs() / want;
                eprintln!("h {h} m, {label}: t = {want:.6} m, f32 {got:.6} m, error {err:.1e}");
                assert!(err < 0.01, "h {h}, {label}: {got} vs {want}");
                if h == 1.0 {
                    // The naive form: c from the f32 camera position.
                    let o32 = o.as_vec3();
                    let naive_c = o32.dot(o32) - 1.0;
                    let naive = ray_ellipsoid(o32, d.as_vec3(), naive_c).map(|t| t.0);
                    let naive_err =
                        naive.map_or(f64::INFINITY, |t| (f64::from(t) - want).abs() / want);
                    eprintln!("  naive f32 c {naive_c:.3e} (f64 {c:.3e}): error {naive_err:.2}");
                    assert!(naive_err > 0.1, "the naive form should fail at 1 m");
                }
            }
        }
    }

    #[test]
    fn equirect_uv_lon0() {
        let uv = equirect_uv(0.0, 0.0, -180.0);
        assert!(
            (uv - Vec2::new(0.5, 0.5)).abs().max_element() < 1e-7,
            "{uv}"
        );
        let east = equirect_uv(179.9f32.to_radians(), 0.0, -180.0);
        let west = equirect_uv((-179.9f32).to_radians(), 0.0, -180.0);
        assert!((east.x - 359.9 / 360.0).abs() < 1e-6, "{east}");
        assert!((west.x - 0.1 / 360.0).abs() < 1e-6, "{west}");
        // North up: +90° at v = 0, −90° at v = 1; a map starting at 0° E.
        assert!(
            equirect_uv(0.0, std::f32::consts::FRAC_PI_2, -180.0)
                .y
                .abs()
                < 1e-7
        );
        assert!((equirect_uv(0.0, -std::f32::consts::FRAC_PI_2, -180.0).y - 1.0).abs() < 1e-7);
        assert!((equirect_uv(90f32.to_radians(), 0.0, 0.0).x - 0.25).abs() < 1e-7);
        assert!((equirect_uv((-90f32).to_radians(), 0.0, 0.0).x - 0.75).abs() < 1e-6);
    }

    /// Saturn's B ring (92 000–117 580 km; NASA NSSDCA Saturnian Rings Fact Sheet) shades the
    /// southern hemisphere when the Sun is 20° north of the ring plane, and the north is clear.
    #[test]
    fn saturn_ring_shadow_on_globe() {
        let path = uni_data::data_dir().join("baked/saturn_rings.ron");
        let profile = uni_data::load_ring_profile(&path).expect("the committed saturn_rings.ron");
        let (inner, outer) = (profile.inner_m as f32, profile.outer_m as f32);
        let elevation = 20f32.to_radians();
        let sun = Vec3::new(elevation.cos(), 0.0, elevation.sin());
        let lat = (-16f32).to_radians();
        let r = SATURN_A as f32;
        let south = Vec3::new(r * lat.cos(), 0.0, r * lat.sin());
        let crossing = ring_crossing_radius(south, sun, inner, outer).expect("crosses the rings");
        assert!((92.0e6..117.58e6).contains(&crossing), "{crossing}");
        let opacity = |r: f32| ring_profile_opacity(&profile, r);
        let t = ring_transmittance(south, sun, inner, outer, opacity);
        assert!(t < 0.5, "B ring transmittance {t}");
        let north = Vec3::new(south.x, 0.0, -south.z);
        assert_eq!(ring_transmittance(north, sun, inner, outer, opacity), 1.0);
        // The Sun in the ring plane: no crossing.
        assert_eq!(ring_crossing_radius(south, Vec3::X, inner, outer), None);
        // Opacity follows the profile's samples (sample centres at inner + (i + ½)Δ).
        let n = profile.samples.len();
        let step = (profile.outer_m - profile.inner_m) / n as f64;
        let mid = (profile.inner_m + (n as f64 / 2.0 + 0.5) * step) as f32;
        assert!((opacity(mid) - profile.samples[n / 2][3]).abs() < 1e-3);
    }

    /// A body of mean radius `radius_m` at `dist_m`, phase `phase_rad`, projecting to `px`.
    #[allow(clippy::too_many_arguments)]
    fn view(
        naif: Option<i32>,
        brdf: Brdf,
        p_v: f64,
        radius_m: f64,
        has_atmosphere: bool,
        dist_m: f64,
        phase_rad: f64,
        px: f64,
    ) -> BodyView {
        let to_camera = DVec3::new(0.3, -0.4, 0.866).normalize();
        let axis = to_camera.any_orthogonal_vector().normalize();
        let to_sun = DQuat::from_axis_angle(axis, phase_rad) * to_camera;
        BodyView {
            id: 7,
            naif,
            rel_pos_m: -to_camera * dist_m,
            orient: DQuat::from_rotation_z(0.3) * DQuat::from_rotation_x(0.2),
            radii_m: DVec3::splat(radius_m),
            proj_radius_px: px,
            sun_rel_m: to_sun * AU,
            sun_illuminance_lux: 1.2e5,
            occluders: Vec::new(),
            ring: None,
            has_atmosphere,
            brdf,
            geometric_albedo: p_v,
            mean_radius_m: radius_m,
        }
    }

    /// The disk integrals of B3's atmosphere for `naif` at `phase` on a quadrature finer than
    /// the table's in every direction (64 + 64 rings × 48 azimuths against 24 + 24 × 16), at
    /// the exact phase: the reference the table and the handover are checked against.
    fn fine_atmo(naif: i32, phase: f64) -> AtmoDisk {
        let p = uni_atmo::preset_for(naif).unwrap();
        let samples = disk_quadrature(p.bottom_radius_m, p.top_radius_m, 64, 64, 48);
        let light = GroundLightTable::new(&p);
        let (inscatter, factor) = atmo_disk(&p, &light, &samples, phase);
        AtmoDisk {
            ground_factor: factor.unwrap(),
            inscatter,
            bottom_m: p.bottom_radius_m,
        }
    }

    /// Review Focus 4 and review I3: at the 1.5 px handover the body carries the point's energy.
    /// The point path's flux is the closed form ρ·[L + 2(1 − L)/3]·E·(R/d)²·Φ(α)·C (uni-sim
    /// `reflected_lux`) at star exposure. The disk's is what this pass writes
    /// (`disk_illuminance_lux` with the map's disk-averaged ρ̄ and the drawn radius, times C and
    /// k) at the blended exposure, plus, for the bodies with an atmosphere, what B3's pass does
    /// to it: the ground dimmed by its ground factor and B3's in-scatter added at the body's own
    /// blended exposure (lead ruling B6 I1, task B10: the atmosphere gets the body's
    /// [`exposure_weight`], so the shader's σ is 1), both from [`fine_atmo`] at the exact phase
    /// (finer than the table [`body_light`] reads, and between its 10° nodes). Checked with
    /// eye-adapted stars (star EV −4 against scene EV 10) and without (one exposure). Earth, Mars
    /// and Venus exercise the cross-fade (Venus drawn at its cloud tops); the airless bodies have
    /// ρ̄ = ρ(p_V) and k = 1. Titan's ground is hidden under its haze, so k stays 1 there and the
    /// handover is B3's in-scatter alone: its gap is printed and bounded below, now in both
    /// exposure modes (before B10 the in-scatter stayed at scene exposure and Titan's disk fell
    /// the full star/scene gap short with eye-adapted stars). Also: k reaches 1 by 4 px, where
    /// the exposed disk is the real rendering at scene exposure, and the cross-fade is exact and
    /// smooth through the band.
    ///
    /// Task B11: the exposure is the log-space blend pre = `pre_mix(star_pre, scene_pre, w)`
    /// (uni-gfx), as body.wgsl and the atmosphere write it. Between neighbouring sizes (0.025 px
    /// apart) the exposure moves by exactly ΔEV·Δw stops and the exposed total by at most
    /// `MAX_STEP_STOPS` (measured: 0.22 stop with eye-adapted stars). B10's linear blend kept
    /// the total within 0.5 stop of the point's to 2 px and 3.1 stops to 3 px, then dropped the
    /// other ~11 of the 14 stops between 3 and 4 px, in steps of up to 1.57 stops per 0.025 px
    /// (printed as "linear" for the report, with the exposed total against the point at 1.5, 2,
    /// 3 and 4 px for both blends). k itself does not depend on the blend: the atmosphere uses
    /// the body's own pre, so σ = 1 for any blend ([`flux_match`]).
    #[test]
    fn point_disk_handover_flux_continuous() {
        /// Largest change of the exposed (disk + atmosphere) total between neighbouring sizes
        /// [stops]: ΔEV·Δw ≤ 14 × 0.017 from the exposure, ≤ 0.05 from the growing disk.
        const MAX_STEP_STOPS: f64 = 0.35;
        /// One size's numbers; energies are exposed [lx × pre].
        struct At {
            /// Disk + atmosphere at the log blend (what the passes write).
            disk: f64,
            /// The point at star exposure.
            point: f64,
            /// The cross-fade's target, pre·[(1 − w)·point + w·(disk + atmosphere at k = 1)].
            target: f64,
            light: BodyLight,
            k: f64,
            /// Disk + atmosphere with fix round 1's k₀ (no atmosphere table).
            old: f64,
            /// Disk + atmosphere at B10's linear blend (report only).
            linear: f64,
            /// The real rendering (k = 1) at scene exposure.
            real: f64,
            /// log₂ of the exposure.
            log2_pre: f64,
        }
        let fov = 60f64.to_radians();
        let sun_color = blackbody_linear_srgb(TEFF_SUN_K);
        // Mars's map is red: a luminance-1 colour, so the ground's channels weigh as drawn.
        let red = Vec3::new(1.8, 0.85, 0.55) / Vec3::new(1.8, 0.85, 0.55).dot(BT709.as_vec3());
        // (name, naif, brdf, p_V, mean radius [m], atmosphere, disk-averaged ρ̄ of the map or
        // None for no map, map colour, phase [deg]). ρ̄ values: A8's index (`source` strings).
        let cases = [
            (
                "Moon",
                Some(301),
                LUNAR_LAMBERT,
                0.12,
                1_737_400.0,
                false,
                Some(0.1241),
                Vec3::ONE,
                60.0,
            ),
            (
                "Mars",
                Some(499),
                Brdf::Lambert,
                0.17,
                3_389_500.0,
                true,
                Some(0.2550),
                red,
                45.0,
            ),
            (
                "Earth",
                Some(399),
                Brdf::Lambert,
                0.434,
                6_371_000.0,
                true,
                Some(0.3019),
                Vec3::ONE,
                0.0,
            ),
            (
                "Earth",
                Some(399),
                Brdf::Lambert,
                0.434,
                6_371_000.0,
                true,
                Some(0.3019),
                Vec3::ONE,
                85.0,
            ),
            (
                "Venus",
                Some(299),
                Brdf::Lambert,
                0.689,
                6_051_800.0,
                true,
                Some(0.95),
                Vec3::ONE,
                105.0,
            ),
            (
                "Titan",
                Some(606),
                Brdf::Lambert,
                0.22,
                2_574_730.0,
                true,
                None,
                Vec3::ONE,
                30.0,
            ),
            (
                "Jupiter",
                Some(599),
                Brdf::Lambert,
                0.538,
                69_911e3,
                false,
                Some(0.8072),
                Vec3::ONE,
                10.0,
            ),
            (
                "Titania",
                Some(703),
                LUNAR_LAMBERT,
                0.35,
                788_900.0,
                false,
                None,
                Vec3::ONE,
                30.0,
            ),
        ];
        let lum = BT709 * sun_color.as_dvec3();
        for (name, naif, brdf, p_v, radius, atmo, map_rho, colour, phase_deg) in cases {
            let phase = f64::to_radians(phase_deg);
            let rho_point = rho_from_geometric_albedo(brdf, p_v);
            let disk_rho = map_rho.unwrap_or(rho_point);
            let truth = atmo.then(|| fine_atmo(naif.unwrap(), phase));
            for eye_adapted in [true, false] {
                let (star_ev, scene_ev) = (if eye_adapted { -4.0 } else { 10.0 }, 10.0);
                let star_pre = f64::from(exposure_scale(star_ev));
                let scene_pre = f64::from(exposure_scale(scene_ev));
                let gap_stops = f64::from(scene_ev - star_ev);
                // The numbers at a projected radius of `px`.
                let energy_at = |px: f64| {
                    let dist = radius / (px * 2.0 * (0.5 * fov).tan() / 1080.0).atan().sin();
                    let v = view(naif, brdf, p_v, radius, atmo, dist, phase, px);
                    assert!((proj_radius_px(radius, dist, fov, 1080.0) - px).abs() < 1e-6);
                    let radii = drawn_radii(&v);
                    let light = body_light(
                        &v,
                        radii,
                        Some(disk_rho),
                        colour,
                        sun_color,
                        eye_adapted,
                        |p| Some(atmo_handover(naif?)?.at(p)),
                    );
                    let drawn_r = (radii.x * radii.y * radii.z).cbrt();
                    let bare = disk_illuminance_lux(brdf, disk_rho, drawn_r, 1.2e5, phase, dist)
                        * light.phase_correction;
                    // This pass at k = 1 as B3's pass leaves it, and B3's in-scatter [lx].
                    let (own, inscatter) = match &truth {
                        Some(t) => (
                            bare * (lum * colour.as_dvec3() * t.ground_factor).element_sum(),
                            lum.dot(t.inscatter) * 1.2e5 * (t.bottom_m / dist).powi(2),
                        ),
                        None => (bare, 0.0),
                    };
                    // body.wgsl and the atmosphere: pre_mix(star_pre(), scene_pre(), w).
                    let pre = f64::from(pre_mix(
                        star_pre as f32,
                        scene_pre as f32,
                        light.exposure_mix,
                    ));
                    // B10's linear blend, for the report's before/after numbers only.
                    let pre_linear =
                        star_pre + (scene_pre - star_pre) * f64::from(light.exposure_mix);
                    // σ = 1: B3's in-scatter is written at this same pre (body.wgsl fs_main).
                    let k = light.flux_match(1.0);
                    // The point path (uni-sim `reflected_lux`): C clamped to ≤ 10 (A3 ruling).
                    let c = naif
                        .map_or(1.0, |n| planet_phase_correction(n, brdf, phase))
                        .min(MAX_PHASE_CORRECTION);
                    assert!((light.phase_correction - c).abs() < 1e-12, "{name}");
                    let weight = match brdf {
                        Brdf::Lambert => 2.0 / 3.0,
                        Brdf::LunarLambert { l } => l + 2.0 * (1.0 - l) / 3.0,
                    };
                    let ratio = radius / dist;
                    let point = rho_point
                        * weight
                        * 1.2e5
                        * ratio
                        * ratio
                        * disk_phase_function(brdf, phase)
                        * c;
                    let w = light.flux_weight;
                    let target = pre * ((1.0 - w) * point + w * (own + inscatter));
                    // Fix round 1's k₀ = (ρ_p/ρ̄)·(R_mean/R_d)², blind to B3: the same factors
                    // without the atmosphere table (printed for the report).
                    let old = body_light(
                        &v,
                        radii,
                        Some(disk_rho),
                        colour,
                        sun_color,
                        eye_adapted,
                        |_| None,
                    )
                    .flux_match(1.0);
                    At {
                        disk: (k * own + inscatter) * pre,
                        point: point * star_pre,
                        target,
                        light,
                        k,
                        old: (old * own + inscatter) * pre,
                        linear: (k * own + inscatter) * pre_linear,
                        real: (own + inscatter) * scene_pre,
                        log2_pre: pre.log2(),
                    }
                };
                let at15 = energy_at(1.5);
                let At {
                    disk,
                    point,
                    light,
                    k: k0,
                    old,
                    ..
                } = at15;
                let gap = disk / point - 1.0;
                let mag = |x: f64| -2.5 * x.log10();
                // The report's table: the exposed total against the point at 1.5, 2, 3 and 4 px
                // (the point at the same distance), log blend and B10's linear blend.
                let row = [1.5, 2.0, 3.0, 4.0].map(|px| {
                    let a = energy_at(px);
                    (
                        px,
                        a.light.exposure_mix,
                        a.disk / a.point,
                        a.linear / a.point,
                    )
                });
                eprintln!(
                    "{name} {phase_deg}° eye-adapted {eye_adapted}: exposed (disk + atmosphere) \
                     / point, log blend [linear blend]: {}",
                    row.map(|(px, w, log, lin)| format!(
                        "{px} px (w {w:.4}) {log:.4e} = {:+.2} stops [{lin:.4e} = {:+.2} stops]",
                        log.log2(),
                        lin.log2()
                    ))
                    .join("; ")
                );
                eprintln!(
                    "{name} {phase_deg}° eye-adapted {eye_adapted}: (disk + atmosphere)/point at \
                     1.5 px = {:.4} ({:+.3} mag; fix round 1's k₀ gave {:.4}, {:+.3} mag); \
                     A {:.3}, B {:.3}, k {k0:.3}",
                    disk / point,
                    mag(disk / point),
                    old / point,
                    mag(old / point),
                    light.point_share,
                    light.inscatter_share,
                );
                assert_eq!(light.flux_weight, 0.0, "{name}");
                if name == "Titan" {
                    // Titan's ground is hidden (ground factor < HANDOVER_GROUND_MIN): k = 1, and
                    // the step at 1.5 px is the atmosphere pass's in-scatter against the point
                    // (it does not fade; review I3, out of this pass's reach). It is drawn at
                    // the body's exposure since B10, so the gap is the same in both modes.
                    let t = truth.as_ref().unwrap();
                    assert!((lum * t.ground_factor).element_sum() < HANDOVER_GROUND_MIN);
                    assert_eq!((light.point_share, light.inscatter_share), (1.0, 0.0));
                    assert_eq!(k0, 1.0);
                    assert!(gap.abs() < 0.3, "Titan, eye-adapted {eye_adapted}: {gap}");
                } else {
                    assert!(gap.abs() < 0.03, "{name}: disk {disk} vs point {point}");
                }
                assert_eq!(
                    light.exposure_mix,
                    if eye_adapted { 0.0 } else { 1.0 },
                    "{name}"
                );
                // By 4 px: scene exposure and the real rendering (no cross-fade left), exactly.
                let at4 = energy_at(4.0);
                let (light4, k4) = (at4.light, at4.k);
                assert_eq!(light4.exposure_mix, 1.0, "{name}");
                assert_eq!(light4.flux_weight, 1.0, "{name}");
                assert_eq!(k4, 1.0, "{name}: {light4:?}");
                assert!(
                    (at4.disk / at4.real - 1.0).abs() < 1e-12,
                    "{name}: {} vs {}",
                    at4.disk,
                    at4.real
                );
                // Through the band the total stays within 3 % of the cross-fade of the point
                // into the real rendering (k from the table, the rendering from fine_atmo), k
                // stays in [0, k_max], and the exposure weight steps by at most 3 % between
                // neighbouring sizes (smoothstep in ln r: at most 1.5 × the mean slope, 1.5 %
                // here). k is linear in w in both exposure modes (σ = 1: B3's in-scatter shares
                // this pass's exposure since B10), so it steps by at most 3 % of |k₀ − 1|. With
                // eye-adapted stars the exposed energy itself falls by the star/scene gap across
                // the band, which is the eye-adapted blend's purpose; the log blend spreads it
                // evenly, ΔEV·Δw stops per step (B11), so no step exceeds MAX_STEP_STOPS.
                let (mut prev, mut prev_k) = (light, k0);
                let (mut prev_e, mut prev_lin, mut prev_log2_pre) =
                    (at15.disk, at15.linear, at15.log2_pre);
                let (mut max_step, mut max_step_linear) = (0.0_f64, 0.0_f64);
                for step in 1..=100 {
                    let a = energy_at(1.5 + 2.5 * f64::from(step) / 100.0);
                    let (e, target, l, k) = (a.disk, a.target, a.light, a.k);
                    assert!(e.is_finite() && e > 0.0, "{name} step {step}");
                    // The exposure in stops is linear in w: it moves by ΔEV·Δw exactly.
                    let step_w = f64::from(l.exposure_mix - prev.exposure_mix);
                    let d_pre = a.log2_pre - prev_log2_pre;
                    assert!(
                        (d_pre + gap_stops * step_w).abs() < 1e-4,
                        "{name} step {step}: exposure moved {d_pre} stops, Δw {step_w}"
                    );
                    let jump = (e / prev_e).log2().abs();
                    assert!(
                        jump <= MAX_STEP_STOPS,
                        "{name} step {step} eye-adapted {eye_adapted}: {jump} stops"
                    );
                    max_step = max_step.max(jump);
                    max_step_linear = max_step_linear.max((a.linear / prev_lin).log2().abs());
                    (prev_e, prev_lin, prev_log2_pre) = (e, a.linear, a.log2_pre);
                    assert!(
                        (0.0..=f64::from(FLUX_MATCH_MAX)).contains(&k),
                        "{name} {step}"
                    );
                    // (The airless bodies have k = 1 by design: their maps are solved for p_V.)
                    let fades = light.point_share != 1.0 || light.inscatter_share != 0.0;
                    if fades && k > 0.0 && k < f64::from(FLUX_MATCH_MAX) {
                        let off = e / target - 1.0;
                        assert!(off.abs() < 0.03, "{name} step {step}: {off}");
                    }
                    let dk = (k - prev_k).abs();
                    assert!(
                        dk <= 0.03 * (k0 - 1.0).abs() + 1e-6,
                        "{name} {step} eye-adapted {eye_adapted}: Δk {dk}"
                    );
                    let dw = (l.exposure_mix - prev.exposure_mix).abs();
                    assert!((0.0..=0.03).contains(&dw), "{name} step {step}: Δw {dw}");
                    (prev, prev_k) = (l, k);
                }
                eprintln!(
                    "{name} {phase_deg}° eye-adapted {eye_adapted}: largest step of the exposed \
                     total between sizes 0.025 px apart: {max_step:.3} stops (B10's linear \
                     blend: {max_step_linear:.3} stops)"
                );
            }
        }
    }

    /// The disk quadrature: its weights cover the disk and shell exactly (π·top²), and over a
    /// bare Lambert ground it reproduces ⅔Φ(α) (Russell 1916) to 1 % up to 150° (the ground
    /// factor is a ratio on this same quadrature, so its error largely cancels).
    #[test]
    fn disk_quadrature_matches_lambert() {
        use std::f64::consts::PI;
        let (r, top) = (6.371e6, 6.471e6);
        let samples = disk_quadrature(r, top, HANDOVER_GROUND_RINGS, HANDOVER_SHELL_RINGS, 16);
        let area: f64 = samples.iter().map(|s| s.weight).sum();
        assert!((area / (PI * top * top) - 1.0).abs() < 1e-3, "{area}");
        for deg in (0..=150).step_by(10) {
            let a = f64::from(deg).to_radians();
            let sun = DVec3::new(a.cos(), a.sin(), 0.0);
            let lambert: f64 = samples
                .iter()
                .filter(|s| s.y * s.y + s.z * s.z < r * r)
                .map(|s| {
                    let n = DVec3::new((r * r - s.y * s.y - s.z * s.z).sqrt(), s.y, s.z) / r;
                    n.dot(sun).max(0.0) / PI * s.weight
                })
                .sum::<f64>()
                / (r * r);
            let want = 2.0 / 3.0 * disk_phase_function(Brdf::Lambert, a);
            assert!(
                (lambert / want - 1.0).abs() < 0.01,
                "{deg}°: {lambert} vs {want}"
            );
        }
    }

    /// The handover table against B3: Earth's disk with A8's 0.302 composite at 550 nm and phase
    /// 0 is B3's p₅₅₀ 0.210 (`earth_disk_albedo_w_b1`, same bounds) with in-scatter 0.063 (B3's
    /// report §6); the ground-light table matches `uni_atmo::ground::ground_lighting`; and the
    /// table read between its nodes is within 3 % of [`fine_atmo`] at the exact phase.
    #[test]
    fn handover_table_matches_b3() {
        let earth = atmo_handover(399).unwrap();
        let e0 = earth.at(0.0);
        let p550 = 0.302 * 2.0 / 3.0 * e0.ground_factor.y + e0.inscatter.y;
        eprintln!("Earth p550 = {p550:.4} (in-scatter {:.4})", e0.inscatter.y);
        assert!((0.20..=0.22).contains(&p550), "p550 = {p550}");
        assert!((e0.inscatter.y - 0.063).abs() < 0.004, "{e0:?}");
        let p = uni_atmo::earth();
        let table = GroundLightTable::new(&p);
        // Within 2 % of the value plus 0.5 % of the overhead Sun's (the scale of every disk
        // sample): linear interpolation runs above the convex twilight curve near μ_s = 0.
        let scale = table.lit(1.0).max_element();
        for mu in [-0.3, -0.05, 0.03, 0.2, 0.5, 0.9] {
            let (t, sky) = uni_atmo::ground::ground_lighting(Some(&p), p.bottom_radius_m, mu);
            let want = t.as_dvec3() * mu.max(0.0) + sky.as_dvec3();
            let got = table.lit(mu);
            let tol = 0.02 * want.max_element() + 0.005 * scale;
            assert!(
                (got - want).abs().max_element() < tol,
                "μ {mu}: {got} vs {want}"
            );
        }
        assert!(atmo_handover(301).is_none(), "the Moon has no preset");
        for (naif, deg) in [(399, 45.0), (499, 125.0), (299, 75.0)] {
            let phase = f64::to_radians(deg);
            let (table, fine) = (
                atmo_handover(naif).unwrap().at(phase),
                fine_atmo(naif, phase),
            );
            let rel = |a: DVec3, b: DVec3| ((a - b) / b).abs().max_element();
            eprintln!(
                "{naif} at {deg}°: ground factor {:.4} vs {:.4}, in-scatter {:.4} vs {:.4}",
                table.ground_factor.y, fine.ground_factor.y, table.inscatter.y, fine.inscatter.y
            );
            assert!(
                rel(table.ground_factor, fine.ground_factor) < 0.03,
                "{naif} {deg}°"
            );
            assert!(rel(table.inscatter, fine.inscatter) < 0.03, "{naif} {deg}°");
        }
    }

    /// [`flux_match`] (the shader's formula): k = 1 from 4 px on; the point's share minus the
    /// in-scatter's at 1.5 px (σ = 1, what the shader passes since B10); clamped to [0, k_max];
    /// and the shares of the bodies with an atmosphere at phase 0 (printed for the report): A
    /// 1.6–3, so the bound never binds there.
    #[test]
    fn flux_match_cases() {
        let k = |w: f32, a: f32, b: f32, sigma: f32| flux_match([w, a, b, FLUX_MATCH_MAX], sigma);
        assert_eq!(k(1.0, 2.5, 0.4, 1.0), 1.0);
        assert_eq!(k(1.0, 0.0, 0.0, 1.0), 1.0);
        assert!((k(0.0, 2.5, 0.4, 1.0) - 2.1).abs() < 1e-6);
        assert!((k(0.0, 2.5, 0.4, 1.0 / 16.0) - 2.475).abs() < 1e-6);
        assert!((k(0.5, 2.5, 0.4, 1.0) - 1.55).abs() < 1e-6);
        assert_eq!(k(0.0, 0.5, 3.0, 1.0), 0.0);
        assert_eq!(k(0.0, 50.0, 0.0, 1.0), FLUX_MATCH_MAX);
        let sun_color = blackbody_linear_srgb(TEFF_SUN_K);
        for (naif, p_v, radius, rho_bar) in [
            (399, 0.434, 6_371_000.0, 0.3019),
            (499, 0.17, 3_389_500.0, 0.2550),
            (299, 0.689, 6_051_800.0, 0.95),
        ] {
            let mut v = view(
                Some(naif),
                Brdf::Lambert,
                p_v,
                radius,
                true,
                1.0e10,
                0.0,
                1.5,
            );
            v.sun_rel_m = -v.rel_pos_m.normalize() * AU;
            let radii = drawn_radii(&v);
            let light = body_light(&v, radii, Some(rho_bar), Vec3::ONE, sun_color, true, |p| {
                Some(atmo_handover(naif)?.at(p))
            });
            let k1 = light.flux_match(1.0);
            eprintln!(
                "{naif} phase 0: A {:.3}, B {:.3}, k at 1.5 px {k1:.3} (σ = 1, both exposure \
                 modes since B10), {:.3} (σ = 1/16, the general formula)",
                light.point_share,
                light.inscatter_share,
                light.flux_match(1.0 / 16.0)
            );
            assert!(
                (1.5..=3.2).contains(&light.point_share),
                "{naif}: {light:?}"
            );
            assert!(k1 > 1.0 && k1 < f64::from(FLUX_MATCH_MAX), "{naif}: {k1}");
        }
    }

    /// Review Focus 5: with the camera 1e6 km from Neptune and the body 30 AU away, the body's
    /// centre as the GPU sees it (the f32 mesh, and the f32 unit-space camera of the ray test)
    /// is within 0.01 px of f64, at fov 60° and zoomed to 1°.
    #[test]
    fn body_position_f32_error_subpixel_at_30au() {
        let camera = DVec3::new(30.07 * AU, 0.0, 0.0) + DVec3::new(0.6, 0.8, 0.0) * 1.0e9;
        let body = DVec3::new(-0.3 * AU, 0.95 * AU, 0.01 * AU);
        let rel = body - camera;
        assert!(rel.length() > 29.0 * AU);
        let mut v = view(
            Some(399),
            Brdf::Lambert,
            0.434,
            6_371_000.0,
            true,
            1.0,
            0.5,
            2.0,
        );
        v.rel_pos_m = rel;
        v.radii_m = DVec3::new(EARTH_A, EARTH_A, EARTH_C);
        let radii = drawn_radii(&v);
        let cam = camera_unit(&v, radii).expect("outside");
        let axis = rel.normalize();
        // The centre the shader implies: the ray direction d with to_body(d)/radii ∥ −o (f32).
        let uniform = body_uniform(
            &v,
            radii,
            &cam,
            &body_light(&v, radii, None, Vec3::ONE, Vec3::ONE, false, |_| None),
            &SurfaceParams {
                slot: 0,
                albedo: 0.3,
                has_map: false,
                lon0_deg: 0.0,
                clouds: None,
                disk_rho: None,
                cloud_tops: false,
            },
            Vec3::ONE,
            None,
            0.0,
        );
        let row = |r: [f32; 4]| DVec3::new(r[0].into(), r[1].into(), r[2].into());
        let to_body_t = glam::DMat3::from_cols(
            row(uniform.to_body_x),
            row(uniform.to_body_y),
            row(uniform.to_body_z),
        );
        let o32 = row(uniform.cam_unit);
        let implied = (to_body_t * (-o32 * row(uniform.radii))).normalize();
        // atan2(|a × b|, a·b): acos loses angles below ~1e-8 rad.
        let angle = |a: DVec3, b: DVec3| a.cross(b).length().atan2(a.dot(b));
        let shader_err = angle(implied, axis);
        // The mesh: the mean of its outline ring, f32 → f64.
        let verts = mesh_vertices(&v, radii, &cam, 1e-9);
        let n = (CAP_SEGMENTS) as usize;
        let outline = &verts[verts.len() - n..];
        let mean = outline
            .iter()
            .map(|p| DVec3::new(p[0].into(), p[1].into(), p[2].into()))
            .sum::<DVec3>()
            / n as f64;
        let mesh_err = angle(mean, axis);
        for fov_deg in [60.0_f64, 1.0] {
            let px = 2.0 * (0.5 * fov_deg.to_radians()).tan() / 1080.0;
            eprintln!(
                "fov {fov_deg}°: shader centre {:.2e} px, mesh centre {:.2e} px",
                shader_err / px,
                mesh_err / px
            );
            assert!(shader_err / px < 0.01 && mesh_err / px < 0.01);
        }
    }

    /// The frustum cull: straight ahead is in view, behind the camera or beyond a side plane is
    /// not, and a sphere straddling a plane is kept.
    #[test]
    fn sphere_in_view_culls_off_screen_bodies() {
        let camera = uni_core::CameraView {
            pos: uni_core::UniversePos::default(),
            orient: DQuat::from_rotation_y(0.7),
            fov_y_rad: 60f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let world = |p: DVec3| camera.orient * p;
        let aspect = 16.0 / 9.0;
        assert!(sphere_in_view(
            &camera,
            aspect,
            world(DVec3::new(0.0, 0.0, -1e7)),
            1e6
        ));
        assert!(!sphere_in_view(
            &camera,
            aspect,
            world(DVec3::new(0.0, 0.0, 1e7)),
            1e6
        ));
        // 45° up is outside the 30° half-height: 1e7·(cos 30° − sin 30°) = 3.66e6 m past the
        // top plane, unless the sphere reaches back in.
        let up = DVec3::new(0.0, 1e7, -1e7);
        assert!(!sphere_in_view(&camera, aspect, world(up), 3.6e6));
        assert!(sphere_in_view(&camera, aspect, world(up), 3.7e6));
        // 45° right is inside the 16:9 frame's 45.8° half-width.
        assert!(sphere_in_view(
            &camera,
            aspect,
            world(DVec3::new(1e7, 0.0, -1e7)),
            1.0
        ));
        // A camera just above a huge body: always in view.
        assert!(sphere_in_view(
            &camera,
            aspect,
            world(DVec3::new(0.0, -6.4e6, 0.0)),
            6.4e6
        ));
    }

    /// Review m22: the cloud tops super-rotate retrograde with a 4.0-day period: −90° per day,
    /// −360° (≡ 0°) per 4 days, exactly, at any epoch.
    #[test]
    fn venus_clouds_super_rotate() {
        let wrap = |a: f64| (a + 180.0).rem_euclid(360.0) - 180.0;
        assert_eq!(venus_cloud_longitude_offset_deg(Epoch::J2000), 0.0);
        for start in [
            Epoch::J2000,
            Epoch::from_tdb_seconds(8.123e8),
            Epoch {
                day: -365_250_000,
                sec: 1234.5,
            },
            Epoch {
                day: 365_250_000,
                sec: 86_000.0,
            },
        ] {
            let off = venus_cloud_longitude_offset_deg(start);
            assert!((0.0..360.0).contains(&off), "{off}");
            for (days, want) in [(1.0, -90.0), (0.5, -45.0), (4.0, 0.0), (2.0, 180.0)] {
                let later = venus_cloud_longitude_offset_deg(start.add_seconds(days * 86_400.0));
                let step = wrap(later - off);
                assert!(
                    (wrap(step - want)).abs() < 1e-9,
                    "{start:?} +{days} d: {step}"
                );
            }
            // Retrograde: an hour later the offset is 3.75° smaller.
            let hour = venus_cloud_longitude_offset_deg(start.add_seconds(3600.0));
            assert!((wrap(hour - off) + 3.75).abs() < 1e-9);
        }
    }

    /// Global constraints: the uniform's size and every member offset match naga's layout of
    /// `struct BodyUniform` in `body.wgsl`.
    #[test]
    fn body_uniforms_layout_match_wgsl() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let m = validate_wgsl(&lib.compose(crate::BODY_WGSL).unwrap()).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("BodyUniform"))
            .expect("struct BodyUniform");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("BodyUniform is not a struct");
        };
        assert_eq!(layouter[handle].size as usize, size_of::<BodyUniform>());
        assert_eq!(size_of::<BodyUniform>(), 240);
        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        let rust = [
            ("to_body_x", offset_of!(BodyUniform, to_body_x)),
            ("to_body_y", offset_of!(BodyUniform, to_body_y)),
            ("to_body_z", offset_of!(BodyUniform, to_body_z)),
            ("radii", offset_of!(BodyUniform, radii)),
            ("cam_unit", offset_of!(BodyUniform, cam_unit)),
            ("sun", offset_of!(BodyUniform, sun)),
            ("light", offset_of!(BodyUniform, light)),
            ("albedo", offset_of!(BodyUniform, albedo)),
            ("maps", offset_of!(BodyUniform, maps)),
            ("extra", offset_of!(BodyUniform, extra)),
            ("handover", offset_of!(BodyUniform, handover)),
            ("occluders", offset_of!(BodyUniform, occluders)),
        ]
        .map(|(n, o)| (n.to_owned(), o));
        assert_eq!(wgsl, rust);
    }

    /// `body.wgsl` composes and validates, has the pass's entry points, binds group 1 as the
    /// Rust layout does, and carries the WGSL twins with the CPU twins' arities.
    #[test]
    fn body_shader_validates() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let src = lib.compose(crate::BODY_WGSL).unwrap();
        let m = validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let stages: Vec<(&str, naga::ShaderStage)> = m
            .entry_points
            .iter()
            .map(|e| (e.name.as_str(), e.stage))
            .collect();
        assert!(stages.contains(&("vs_main", naga::ShaderStage::Vertex)));
        assert!(stages.contains(&("fs_main", naga::ShaderStage::Fragment)));
        let arity = |name: &str| {
            m.functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some(name))
                .map(|(_, f)| f.arguments.len())
        };
        assert_eq!(arity("ray_ellipsoid"), Some(3));
        assert_eq!(arity("equirect_uv"), Some(3));
        assert_eq!(arity("ring_crossing_radius"), Some(4));
        assert_eq!(arity("ring_transmittance"), Some(4));
        assert_eq!(arity("lunar_lambert"), Some(3));
        assert_eq!(arity("sun_visible_fraction"), Some(3));
        assert_eq!(arity("flux_match"), Some(2));
        let mut group1: Vec<u32> = m
            .global_variables
            .iter()
            .filter_map(|(_, v)| v.binding.as_ref())
            .filter(|b| b.group == 1)
            .map(|b| b.binding)
            .collect();
        group1.sort_unstable();
        assert_eq!(group1, (0..BODY_BINDINGS).collect::<Vec<_>>());
    }

    /// The ring parameter in [`body_uniform`] and Venus's cloud-top radius: the drawn ellipsoid
    /// is the cloud deck, 70 km up (uni-atmo `VENUS_CLOUD_TOP_M`), every other body its radii.
    #[test]
    fn drawn_radii_and_inside_warning_geometry() {
        let mut v = view(
            Some(299),
            Brdf::Lambert,
            0.689,
            6_051_800.0,
            true,
            1e9,
            0.3,
            10.0,
        );
        let r = drawn_radii(&v);
        assert_eq!(r, DVec3::splat(6_051_800.0 + uni_atmo::VENUS_CLOUD_TOP_M));
        v.naif = Some(399);
        assert_eq!(drawn_radii(&v), v.radii_m);
        // Inside the drawn ellipsoid (or at its centre): no camera, so nothing is drawn.
        v.rel_pos_m = DVec3::new(1.0e6, 0.0, 0.0);
        assert!(camera_unit(&v, v.radii_m).is_none());
        v.rel_pos_m = DVec3::ZERO;
        assert!(camera_unit(&v, v.radii_m).is_none());
        // Saturn's ring reaches the uniform; others have none.
        let mut s = view(
            Some(699),
            Brdf::Lambert,
            0.499,
            58_232e3,
            false,
            1e10,
            0.3,
            50.0,
        );
        s.ring = Some(RingView {
            inner_m: 7.0e7,
            outer_m: 1.4e8,
            normal: DVec3::Z,
        });
        let radii = drawn_radii(&s);
        let cam = camera_unit(&s, radii).unwrap();
        let surface = SurfaceParams {
            slot: 0,
            albedo: 0.7,
            has_map: false,
            lon0_deg: 0.0,
            clouds: None,
            disk_rho: None,
            cloud_tops: false,
        };
        let light = body_light(&s, radii, None, Vec3::ONE, Vec3::ONE, false, |_| None);
        let u = body_uniform(
            &s,
            radii,
            &cam,
            &light,
            &surface,
            Vec3::ONE,
            Some((7.0e7, 1.4e8)),
            0.0,
        );
        assert_eq!([u.extra[1], u.extra[2]], [7.0e7, 1.4e8]);
        assert_eq!(u.albedo, [0.7, 0.7, 0.7, 0.0]);
        assert!((f64::from(u.cam_unit[3]) - cam.c).abs() <= cam.c * 1e-7);
    }

    /// Review I2 (lead ruling, fix round 1) and Review Focus 2: "inside" means inside the SOLID
    /// body. A camera 35 km above Venus's solid surface, under its 70 km cloud deck, still sees
    /// Venus: the pass draws the solid radii (the surface under the deck) with no warning. Above
    /// the deck the deck is drawn; inside the solid body nothing is, with "Inside Venus". A view
    /// with non-finite data is skipped without claiming the camera is inside anything (m7).
    #[test]
    fn camera_under_the_venus_deck_draws_the_solid_body() {
        let r = 6_051_800.0;
        let at = |naif: i32, altitude_m: f64| {
            view(
                Some(naif),
                Brdf::Lambert,
                0.689,
                r,
                true,
                r + altitude_m,
                0.3,
                1e4,
            )
        };
        let deck = DVec3::splat(r + uni_atmo::VENUS_CLOUD_TOP_M);
        // 35 km up: between the surface and the deck. Drawn at the solid radii, no warning.
        let v = at(VENUS, 35e3);
        let Placement::Draw { radii, cam } = placement(&v, "Venus") else {
            panic!("35 km above Venus: {:?}", placement(&v, "Venus"));
        };
        assert_eq!(radii, v.radii_m);
        let want_c = ((r + 35e3) / r).powi(2) - 1.0;
        assert!(
            (cam.c / want_c - 1.0).abs() < 1e-9,
            "c {} vs {want_c}",
            cam.c
        );
        // It is a real draw: finite mesh, camera outside the drawn ellipsoid.
        let mesh = mesh_vertices(&v, radii, &cam, 1e-4);
        assert_eq!(mesh.len(), CAP_VERTICES);
        assert!(mesh.iter().flatten().all(|x| x.is_finite()));
        // Just above the surface (1 m) and just below the deck (69.9 km): the same.
        for h in [1.0, 69.9e3] {
            let v = at(VENUS, h);
            assert!(
                matches!(placement(&v, "Venus"), Placement::Draw { radii, .. } if radii == v.radii_m),
                "{h} m"
            );
        }
        // Above the deck (100 km): the deck is drawn.
        let v = at(VENUS, 100e3);
        assert!(matches!(placement(&v, "Venus"), Placement::Draw { radii, .. } if radii == deck));
        // Inside the solid body (10 km down, and at the centre): nothing, "Inside Venus".
        for h in [-10e3, -r] {
            match placement(&at(VENUS, h), "Venus") {
                Placement::Skip(w) => assert_eq!(w, "Inside Venus", "{h} m"),
                p => panic!("{h} m: {p:?}"),
            }
        }
        // Earth has no deck: 35 km up draws its radii, 10 km down is inside.
        let e = at(399, 35e3);
        assert!(
            matches!(placement(&e, "Earth"), Placement::Draw { radii, .. } if radii == e.radii_m)
        );
        assert!(
            matches!(placement(&at(399, -10e3), "Earth"), Placement::Skip(w) if w == "Inside Earth")
        );
        // Non-finite view data: skipped, and the warning does not say "Inside".
        let mut bad = at(VENUS, 35e3);
        bad.rel_pos_m = DVec3::NAN;
        match placement(&bad, "Venus") {
            Placement::Skip(w) => assert!(!w.contains("Inside") && w.contains("Venus"), "{w}"),
            p => panic!("NaN position drawn: {p:?}"),
        }
    }

    /// Lead ruling B6 I1 (task B10): the star share a uni-planet pass stores in `lens_mask.a`.
    /// - One exposure: w = 1 and the share is 0 at every size (`star_pre == scene_pre` then).
    /// - Eye-adapted: share 1 at ≤ 1.5 px (star exposure), 0 from 4 px on, 1 − w between, and 0
    ///   wherever the fragment covers nothing.
    /// - The snap: every non-zero share is ≥ 2⁻¹⁴, f16's smallest normal, so the Rgba16Float store
    ///   cannot flush it even where Metal flushes subnormals. The snap acts only just below 4 px
    ///   (3.995 px snapped, 3.95 px not), moves w by < 2⁻¹⁴ and keeps it non-decreasing.
    /// - The meter's encoding (uni-post `histogram.wgsl` `meter_pre`: pre_mix(scene_pre,
    ///   star_pre, s), 0 = scene exposure) recovers the writer's pre pre_mix(star_pre,
    ///   scene_pre, w) (task B11, both the uni-gfx helper) from the f16-stored share to within
    ///   its rounding, 2⁻¹¹·s·|ΔEV| stops, at star EV −4 against scene EV 15, 0 and −10.
    /// - The snap's effect on brightness (B11): pre moves by less than |ΔEV|·2⁻¹⁴ stops.
    #[test]
    fn star_share_snapped_and_read_back_by_the_meter() {
        use uni_data::{f16_to_f32, f32_to_f16};
        use uni_sim::snapshot::{BLEND_LOD_PX, POINT_LOD_PX};
        assert_eq!(STAR_SHARE_MIN, 2f32.powi(-14));
        // 0x0400 is binary16's smallest normal (IEEE 754-2019 §3.6).
        assert_eq!(f16_to_f32(0x0400), STAR_SHARE_MIN);
        for r in [
            0.0,
            1.0,
            1.5,
            2.5,
            3.99,
            4.0,
            100.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert_eq!(exposure_weight(r, false), 1.0, "{r}");
            assert_eq!(star_share(exposure_weight(r, false), 1.0), 0.0, "{r}");
        }
        for r in [0.0, 1.0, POINT_LOD_PX] {
            assert_eq!(star_share(exposure_weight(r, true), 1.0), 1.0, "{r}");
        }
        for r in [BLEND_LOD_PX, 10.0, f64::INFINITY, f64::NAN] {
            assert_eq!(star_share(exposure_weight(r, true), 0.5), 0.0, "{r}");
        }
        // Coverage: none (or NaN) stores nothing; any coverage stores the whole share (Max).
        assert_eq!(star_share(0.25, 0.0), 0.0);
        assert_eq!(star_share(0.25, f32::NAN), 0.0);
        assert_eq!(star_share(0.25, 1.0e-3), 0.75);
        // The snap acts just below 4 px only.
        assert!(lod_exposure_mix(3.995) < 1.0 && 1.0 - lod_exposure_mix(3.995) < STAR_SHARE_MIN);
        assert_eq!(exposure_weight(3.995, true), 1.0);
        assert!(1.0 - lod_exposure_mix(3.95) > STAR_SHARE_MIN);
        assert_eq!(exposure_weight(3.95, true), lod_exposure_mix(3.95));

        let mut radii: Vec<f64> = (0..=100_000)
            .map(|i| POINT_LOD_PX + (BLEND_LOD_PX - POINT_LOD_PX) * f64::from(i) / 1.0e5)
            .chain((0..=10_000).map(|i| 3.97 + 0.03 * f64::from(i) / 1.0e4))
            .collect();
        radii.sort_by(f64::total_cmp);
        let pres = [(15.0, -4.0), (0.0, -4.0), (-10.0, -4.0)]
            .map(|(scene, star)| (exposure_scale(scene), exposure_scale(star)));
        let (mut prev, mut snapped, mut lowest_snapped) = (0.0_f32, 0, f64::INFINITY);
        for r in radii {
            let lod = lod_exposure_mix(r);
            let w = exposure_weight(r, true);
            assert!((w - lod).abs() < STAR_SHARE_MIN, "r {r}: w {w}, lod {lod}");
            assert!(w >= prev, "r {r}: {w} < {prev}");
            prev = w;
            if w == 1.0 && lod < 1.0 {
                snapped += 1;
                lowest_snapped = lowest_snapped.min(r);
            }
            let s = star_share(w, 1.0);
            assert!(s == 0.0 || s >= STAR_SHARE_MIN, "r {r}: share {s}");
            // What lens_mask holds after the store (round to nearest f16).
            let stored = f16_to_f32(f32_to_f16(s));
            assert_eq!(stored == 0.0, s == 0.0, "r {r}: {s} stored as {stored}");
            assert!(stored == 0.0 || stored >= STAR_SHARE_MIN, "r {r}: {stored}");
            assert!(
                (stored - s).abs() <= s * 2f32.powi(-11),
                "r {r}: {s} → {stored}"
            );
            for (scene_pre, star_pre) in pres {
                // The writer: pre_mix(star_pre, scene_pre, w). The meter: pre_mix(scene_pre,
                // star_pre, s), s clamped to [0, 1].
                let writer = pre_mix(star_pre, scene_pre, w);
                let meter = pre_mix(scene_pre, star_pre, stored.clamp(0.0, 1.0));
                let gap = f64::from(star_pre / scene_pre).log2().abs();
                // f16 rounding of s, plus f32 rounding of the logs (|log₂ pre| < 16 here).
                let bound = 2f64.powi(-11) * f64::from(s) * gap + 1e-5;
                let off = f64::from(meter / writer).log2().abs();
                assert!(
                    off <= bound,
                    "r {r}, pres {scene_pre} {star_pre}: meter {meter} vs writer {writer}"
                );
                if s == 0.0 {
                    assert_eq!(meter.to_bits(), scene_pre.to_bits(), "r {r}");
                    assert_eq!(writer.to_bits(), scene_pre.to_bits(), "r {r}");
                }
                // The snap moves pre by less than ΔEV·2⁻¹⁴ stops: no visible step.
                let unsnapped = pre_mix(star_pre, scene_pre, lod_exposure_mix(r));
                let moved = f64::from(writer / unsnapped).log2().abs();
                assert!(moved <= gap * 2f64.powi(-14) + 1e-5, "r {r}: {moved} stops");
            }
        }
        assert!(snapped > 0, "the snap never acted");
        assert!(lowest_snapped > 3.98, "snapped from {lowest_snapped} px");
    }

    /// Lead ruling B6 I1: a body's atmosphere uses the body's own exposure weight. uni-atmo
    /// cannot depend on uni-planet, so it keeps a twin of the snap
    /// (`uni_atmo::feature::atmo_exposure_mix`): same threshold, the same result on every raw
    /// `lod_exposure_mix` value, and a body's [`exposure_weight`] (what the app copies into
    /// `AtmoDraw::exposure_mix`) passes through it unchanged, so the pair share one pre and one
    /// star share.
    #[test]
    fn atmosphere_gets_the_bodys_exposure_weight() {
        use uni_atmo::feature::{STAR_SHARE_MIN as ATMO_MIN, atmo_exposure_mix, atmo_star_share};
        assert_eq!(ATMO_MIN, STAR_SHARE_MIN);
        for i in 0..=100_000 {
            let r = 1.0 + 3.5 * f64::from(i) / 1.0e5;
            let lod = lod_exposure_mix(r);
            assert_eq!(atmo_exposure_mix(lod), snap_exposure_weight(lod), "{r} px");
            for eye_adapted in [true, false] {
                let w = exposure_weight(r, eye_adapted);
                assert_eq!(atmo_exposure_mix(w), w, "{r} px");
                assert_eq!(atmo_star_share(w), star_share(w, 1.0), "{r} px");
            }
        }
    }

    /// The bodies, rings and star-surface passes share one `lens_mask` target (lead ruling B6 I1):
    /// green (sky occlusion) and alpha (the star share) written, red and blue left to the lens
    /// pass, both blended with `Max` (factors One), Rgba16Float.
    #[test]
    fn lens_mask_target_writes_green_and_alpha_with_max() {
        let t = lens_mask_target();
        assert_eq!(t.format, LENS_MASK_FORMAT);
        assert_eq!(
            t.write_mask,
            wgpu::ColorWrites::GREEN | wgpu::ColorWrites::ALPHA
        );
        let blend = t.blend.unwrap();
        for c in [blend.color, blend.alpha] {
            assert_eq!(c.operation, wgpu::BlendOperation::Max);
            assert_eq!(
                (c.src_factor, c.dst_factor),
                (wgpu::BlendFactor::One, wgpu::BlendFactor::One)
            );
        }
    }

    /// The three world writers' shaders store the star share (lead ruling B6 I1): each defines
    /// `star_share(w, coverage)` (twin of [`star_share`]) and writes it into the mask's alpha with
    /// the same w its pre uses, per instance for the star surfaces. Each draws at the log-space
    /// blend `pre_mix(star_pre(), scene_pre(), w)` (task B11, uni-gfx `common/exposure.wgsl`) and
    /// no linear mix of the two pre-exposures is left. body.wgsl passes σ = 1 to `flux_match`,
    /// because the atmosphere now draws at the body's exposure weight.
    #[test]
    fn world_writer_shaders_store_the_star_share() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        for (path, pre, mask) in [
            (
                crate::BODY_WGSL,
                "let pre = pre_mix(star_pre(), scene_pre(), body_u.light.w);",
                "out.mask = vec4<f32>(0.0, coverage, 0.0, star_share(body_u.light.w, coverage));",
            ),
            (
                crate::RINGS_WGSL,
                "let pre = pre_mix(star_pre(), scene_pre(), ring.centre.w);",
                "out.mask = vec4<f32>(0.0, alpha_view, 0.0, star_share(ring.centre.w, alpha_view));",
            ),
            (
                crate::STAR_SURFACE_WGSL,
                "let pre = pre_mix(star_pre(), scene_pre(), s.l0.w);",
                "out.mask = vec4<f32>(0.0, cov, 0.0, star_share(s.l0.w, cov));",
            ),
        ] {
            let src = lib.compose(path).unwrap();
            let m = validate_wgsl(&src).unwrap_or_else(|e| panic!("{path}: {e}"));
            let arity = m
                .functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some("star_share"))
                .map(|(_, f)| f.arguments.len());
            assert_eq!(arity, Some(2), "{path}");
            assert_eq!(
                src.matches("return select(0.0, 1.0 - w, coverage > 0.0);")
                    .count(),
                1,
                "{path}"
            );
            assert_eq!(src.matches(pre).count(), 1, "{path}: pre");
            // Every mix of the two pre-exposures is the log-space pre_mix.
            for arg in ["mix(star_pre()", "mix(scene_pre()"] {
                let log = format!("pre_{arg}");
                assert_eq!(
                    src.matches(arg).count(),
                    src.matches(log.as_str()).count(),
                    "{path}: a linear {arg}"
                );
            }
            assert_eq!(src.matches(mask).count(), 1, "{path}: mask");
        }
        let body = lib.compose(crate::BODY_WGSL).unwrap();
        assert!(body.contains("let k = flux_match(body_u.handover, 1.0);"));
    }
}
