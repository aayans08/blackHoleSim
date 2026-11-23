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
//! fragment is discarded; premultiplied blend into `hdr`; `lens_mask.g` gets the coverage with
//! `Max`. The mesh's outline ring is pushed [`AA_MARGIN_PX`] out so the rim outside the limb is
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
//! the bodies drawn with an atmosphere the point flux cross-fades to the rendered disk over the
//! 1.5–4 px band (ruling W-B1, [`body_light`]). The exposure factor blends from `star_pre` to
//! `scene_pre` with [`crate::lod_exposure_mix`] while eye-adapted stars are in effect.
//!
//! No Earth- or Venus-specific calibration constant lives here (constraints §4a, W-B1..W-B3);
//! `World::objects()` grows at run time, so nothing is sized per `ObjectId`: materials are keyed
//! by NAIF id and the per-frame buffers grow with the number of bodies drawn.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use glam::{DMat3, DVec3, Vec2, Vec3};
use uni_atmo::{GroundLighting, VENUS_CLOUD_TOP_M};
use uni_core::{DAY, R_SUN_M, TEFF_SUN_K, blackbody_linear_srgb};
use uni_data::{BodyFacts, MapInfo, RingProfile, TexFormat, f32_to_f16};
use uni_gfx::{
    DEPTH_FORMAT, FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, LENS_MASK_FORMAT,
    PassTimer, RenderFeature, ShaderLib, SharedInput,
};
use uni_physics::photometry::{Brdf, planet_phase_correction, rho_from_geometric_albedo};
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{BodyView, MAX_PHASE_CORRECTION, StarSurfaceView};

use crate::capmesh::{CAP_RINGS, CAP_SEGMENTS, cap_indices, cap_mesh};
use crate::lod::lod_exposure_mix;
use crate::material::{GpuMaps, Loaded, MaterialTable, SurfaceParams, load_pair, solid_texture};

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
    /// E☉ × Sun colour × C × cross-fade; w: exposure weight.
    pub light: [f32; 4],
    /// Reflectance multiplier; w: lunar-Lambert L.
    pub albedo: [f32; 4],
    /// Has map, cloud scale, albedo lon0 [deg], cloud lon0 [deg].
    pub maps: [f32; 4],
    /// Longitude offset [rad], ring inner, ring outer [m], occluder count.
    pub extra: [f32; 4],
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
///   solid radii, the surface under the deck, with no warning; the atmosphere pass draws the
///   clouds and haze around the camera. Venus has no surface map, so the solid sphere carries
///   the cloud-top map;
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
    /// Ruling W-B1's point → disk flux factor: k₀ at 1.5 px, 1 from 4 px on.
    pub flux_match: f64,
    /// Exposure weight: 0 `star_pre` … 1 `scene_pre`.
    pub exposure_mix: f32,
}

/// The factors [`body_uniform`] uploads for `view` drawn with `radii` and disk-averaged
/// reflectance `disk_rho` (the map's ρ̄, or the uniform ρ; `None` when unknown).
///
/// - C = `planet_phase_correction(naif, brdf, α)` clamped to ≤ `MAX_PHASE_CORRECTION`, the
///   point path's clamp (A3 ruling), 1 without a NAIF id or for a NaN. C holds its value beyond
///   each law's fitted range; for Earth and Mars that hold starts at 170°, A3's choice, accepted
///   by the lead (A3 ruling).
/// - Ruling W-B1: a body drawn with an atmosphere shows its real ground (Earth's ρ̄ 0.302 against
///   ρ(p_V) 0.651; the atmosphere pass supplies the rest), so its disk is scaled by
///   k = k₀ + (1 − k₀)·w, w = [`lod_exposure_mix`] (0 at 1.5 px, 1 at 4 px), with
///   k₀ = (ρ(p_V)/ρ̄)·(R_mean/R_drawn)²: at 1.5 px the disk sends the point's flux (Venus is drawn
///   at its cloud tops, 1.2 % larger), and from 4 px on its real rendering. Airless bodies keep
///   k = 1 (their maps are solved for p_V). Without the map statistics (`disk_rho` `None`, a 2k
///   copy missing; the HUD says so, review I1) k₀ = 1.
///   **Precondition (review I3, open, awaiting a lead ruling):** k₀ makes this pass alone carry the
///   point's whole flux at 1.5 px, so the total is continuous only if the atmosphere pass (B3)
///   fades its own light by the same w. uni-atmo does not do that yet: once B3's in-scatter is
///   calibrated to W-B1's target (ground + atmosphere = point), Earth at 1.5 px is
///   (0.651 + 0.349)/0.651 ≈ 1.54× the point (+0.47 mag). Not fixable inside this pass: it does
///   not know B3's disk share.
/// - The exposure weight is w while eye-adapted stars are in effect, else 1 (scene exposure).
pub(crate) fn body_light(
    view: &BodyView,
    radii: DVec3,
    disk_rho: Option<f64>,
    eye_adapted: bool,
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
    let w = f64::from(lod_exposure_mix(view.proj_radius_px));
    let drawn_mean = (radii.x * radii.y * radii.z).cbrt();
    let k0 = match disk_rho {
        Some(rho) if view.has_atmosphere && rho > 0.0 && drawn_mean > 0.0 => {
            let area = (view.mean_radius_m / drawn_mean).powi(2);
            rho_from_geometric_albedo(view.brdf, view.geometric_albedo) / rho * area
        }
        _ => 1.0,
    };
    let k0 = if k0.is_finite() { k0 } else { 1.0 };
    BodyLight {
        phase_rad: phase,
        phase_correction,
        flux_match: k0 + (1.0 - k0) * w,
        exposure_mix: if eye_adapted { w as f32 } else { 1.0 },
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
    let e = view.sun_illuminance_lux * light.phase_correction * light.flux_match;
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
                    // lens_mask.g = max(g, coverage) (M1 frame contract).
                    Some(wgpu::ColorTargetState {
                        format: LENS_MASK_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: MAX_BLEND,
                            alpha: MAX_BLEND,
                        }),
                        write_mask: wgpu::ColorWrites::GREEN,
                    }),
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
            let light = body_light(view, radii, surface.disk_rho, scene.eye_adapted);
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

    /// Review Focus 4: at the 1.5 px handover the resolved disk carries the point's energy. The
    /// point path's flux is the closed form ρ·[L + 2(1 − L)/3]·E·(R/d)²·Φ(α)·C (uni-sim
    /// `reflected_lux`) at star exposure; the disk's is the integral over what the shader writes
    /// (`disk_illuminance_lux` with the map's disk-averaged ρ̄ and the drawn radius) times the
    /// per-body factors [`body_light`] uploads, at the blended exposure. Earth and Venus
    /// exercise the W-B1 cross-fade (ρ̄ ≠ ρ(p_V), Venus drawn at its cloud tops); the others
    /// have ρ̄ = ρ(p_V). Also: the factors reach the real rendering by 4 px and change smoothly.
    #[test]
    fn point_disk_handover_flux_continuous() {
        let (star_pre, scene_pre) = (exposure_scale(-4.0), exposure_scale(10.0));
        let fov = 60f64.to_radians();
        // (name, naif, brdf, p_V, mean radius [m], atmosphere, disk-averaged ρ̄ of the map or
        // None for no map, phase [deg]). ρ̄ values: A8's index (`source` strings).
        let cases = [
            (
                "Moon",
                Some(301),
                LUNAR_LAMBERT,
                0.12,
                1_737_400.0,
                false,
                Some(0.1241),
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
                40.0,
            ),
            (
                "Earth",
                Some(399),
                Brdf::Lambert,
                0.434,
                6_371_000.0,
                true,
                Some(0.3019),
                90.0,
            ),
            (
                "Venus",
                Some(299),
                Brdf::Lambert,
                0.689,
                6_051_800.0,
                true,
                Some(0.95),
                100.0,
            ),
            (
                "Jupiter",
                Some(599),
                Brdf::Lambert,
                0.538,
                69_911e3,
                false,
                Some(0.8072),
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
                30.0,
            ),
        ];
        for (name, naif, brdf, p_v, radius, atmo, map_rho, phase_deg) in cases {
            let phase = f64::to_radians(phase_deg);
            let rho_point = rho_from_geometric_albedo(brdf, p_v);
            let disk_rho = map_rho.unwrap_or(rho_point);
            let energy_at = |px: f64| {
                let dist = radius / (px * 2.0 * (0.5 * fov).tan() / 1080.0).atan().sin();
                let v = view(naif, brdf, p_v, radius, atmo, dist, phase, px);
                assert!((proj_radius_px(radius, dist, fov, 1080.0) - px).abs() < 1e-6);
                let radii = drawn_radii(&v);
                let light = body_light(&v, radii, Some(disk_rho), true);
                let drawn_r = (radii.x * radii.y * radii.z).cbrt();
                let disk = disk_illuminance_lux(brdf, disk_rho, drawn_r, 1.2e5, phase, dist)
                    * light.phase_correction
                    * light.flux_match;
                let pre = star_pre + (scene_pre - star_pre) * light.exposure_mix;
                // The point path (uni-sim `reflected_lux`): C clamped to ≤ 10 (A3 ruling).
                let c = naif
                    .map_or(1.0, |n| planet_phase_correction(n, brdf, phase))
                    .min(MAX_PHASE_CORRECTION);
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
                assert!((light.phase_correction - c).abs() < 1e-12, "{name}");
                (disk * f64::from(pre), point * f64::from(star_pre), light)
            };
            let (disk, point, light) = energy_at(1.5);
            let gap = disk / point - 1.0;
            eprintln!("{name}: disk/point at 1.5 px = {:.5}", disk / point);
            assert!(gap.abs() < 0.05, "{name}: disk {disk} vs point {point}");
            assert_eq!(light.exposure_mix, 0.0, "{name}");
            // By 4 px: scene exposure and the real rendering (no cross-fade left).
            let (_, _, light4) = energy_at(4.0);
            assert_eq!(light4.exposure_mix, 1.0, "{name}");
            assert!(
                (light4.flux_match - 1.0).abs() < 1e-12,
                "{name}: {light4:?}"
            );
            // Continuous through the band: the cross-fade stays between k₀ and 1 and the exposure
            // weight between 0 and 1, and neither steps by more than 3 % of its range between
            // neighbouring sizes (smoothstep in ln r: at most 1.5 × the mean slope, 1.5 % here).
            // The exposed energy itself changes by orders of magnitude across the band, from
            // star_pre to scene_pre: that is the eye-adapted blend's purpose.
            let k0 = light.flux_match;
            let mut prev = light;
            for k in 1..=100 {
                let (e, _, l) = energy_at(1.5 + 2.5 * f64::from(k) / 100.0);
                assert!(e.is_finite() && e > 0.0, "{name} step {k}");
                let (lo, hi) = (k0.min(1.0), k0.max(1.0));
                assert!(
                    (lo - 1e-12..=hi + 1e-12).contains(&l.flux_match),
                    "{name} step {k}"
                );
                let dk = (l.flux_match - prev.flux_match).abs();
                assert!(
                    dk <= 0.03 * (k0 - 1.0).abs() + 1e-12,
                    "{name} step {k}: Δk {dk}"
                );
                let dw = (l.exposure_mix - prev.exposure_mix).abs();
                assert!((0.0..=0.03).contains(&dw), "{name} step {k}: Δw {dw}");
                prev = l;
            }
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
            &body_light(&v, radii, None, false),
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
        assert_eq!(size_of::<BodyUniform>(), 224);
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
        let light = body_light(&s, radii, None, false);
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
}
