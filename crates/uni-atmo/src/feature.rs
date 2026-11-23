//! The atmosphere render feature, pass "atmosphere" (Task B3).
//!
//! Pass 8 of the M1 frame contract: after the opaque passes, before the meter. Per pixel it
//! intersects each atmosphere's shell, clips by the scene depth and the planet's ground, and
//! marches the inscattered light (world light × the body's exposure, `scene_pre()` from 4 px
//! on), writing `lens_mask.g = 1 − luminance(T)` and `lens_mask.a` = the star share 1 − w, both
//! with `Max`. Two variants (review I7): dual-source blending (composite + mask pipelines,
//! `atmosphere_dual.wgsl` + `atmosphere_mask.wgsl`) or grey transmittance (one composite
//! pipeline, `atmosphere_grey.wgsl`); both march at half resolution first (`fs_march`, task B9p).
//!
//! **Exposure and star share (lead ruling B6 I1, task B10).** A body in the 1.5–4 px LOD band is
//! drawn at pre = pre_mix(star_pre, scene_pre, w), the log-space blend star_pre^(1 − w)·
//! scene_pre^w of uni-gfx (`common/exposure.wgsl`, CPU twin `uni_gfx::frame::pre_mix`; task
//! B11), with w = `uni_planet::lod_exposure_mix` while eye-adapted stars are in effect. Its
//! atmosphere uses the same w ([`AtmoDraw::exposure_mix`], which the app sets to
//! `uni_planet::body::exposure_weight` of the same body) and the same `pre_mix`, so the
//! in-scatter sits at the body's exposure (closing Titan's star/scene gap at the 1.5 px handover;
//! uni-planet `flux_match` then takes σ = 1), and every fragment the march does not discard
//! stores s = 1 − w in `lens_mask.a` with `Max`. The meter (uni-post `histogram.wgsl`, pass 9)
//! divides the pixel by pre_mix(scene_pre, star_pre, s), the writer's pre. w is snapped to 1
//! when 1 − w < [`STAR_SHARE_MIN`] = 2⁻¹⁴ ([`atmo_exposure_mix`]), so a share is never an f16
//! subnormal that Metal could flush to 0; with the log blend the snap moves pre by less than
//! 2⁻¹⁴·(EV_scene − EV_star) stops (0.0012 stop at a 19-stop gap).
//!
//! **Recorded deviation (frame contract):** a view march through Hillaire's transmittance and
//! multiple-scattering LUTs, not his sky-view and aerial-perspective LUTs (design §5.1); those are
//! the M2 upgrade.
//!
//! **Performance ladder, all three steps applied (task B9p; budget 3.0 ms; the pass measured
//! 11.2 ms at `earth-map-closeup`):**
//! 1. The dual variant's `g` comes from the transmittance-LUT ratio, not a second march
//!    ([`segment_transmittance_lut`]).
//! 2. The march takes 16 steps, up to 24 on long limb paths (it took 32 and 48). The quality cost
//!    is documented at `lut::VIEW_MARCH_STEPS`: a few % on a dawn limb's low red layer.
//! 3. The march runs at half resolution; the composite upsamples where the samples share the
//!    pixel's ray geometry and agree, and marches every other pixel exactly
//!    ([`upsample_weights`]).
//!
//! Result: 2.6 ms there.
//!
//! **The shell follows the body's ellipsoid.** Each draw maps root axes into the body-fixed frame
//! (`AtmoDraw::orient`) and scales it by R/(a, b, c) ([`atmosphere_frame`]), so the body's
//! reference ellipsoid (WGCCRE 2015 radii, [`body_shape_m`]) becomes the sphere of its mean
//! radius R, which is the atmosphere's bottom (Venus: R plus the cloud-top height). Heights are
//! then measured from the real ground at every latitude. Earth's ground is 21.4 km closer to the
//! centre at the poles than at the equator: with a sphere of radius 6 371 km the polar ground
//! would lie 14.2 km below the bottom (its lowest 14 km of air missing) and the equatorial ground
//! 7.1 km above it. Distances along a ray stay true; angles are distorted by at most the
//! flattening (0.6 % for Mars). The camera-height helpers ([`AtmoDraw::camera_in_daylight`],
//! [`AtmoDraw::scintillation_strength`]) use the same frame (review I1).

use glam::{DMat3, DQuat, DVec3, Mat4, Vec3};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, LENS_MASK_FORMAT, PassTimer,
    RenderFeature, ShaderLib, SharedInput,
};

use crate::lut::{
    AtmosphereSet, CpuLuts, March, cpu_luts, lut_transmittance, march_steps, ray_hits_ground,
    ray_sphere, view_step_count,
};
use crate::params::AtmosphereParams;

/// The atmospheres to draw this frame (set by the app before `Renderer::render`).
#[derive(Clone, Debug, Default)]
pub struct AtmoScene {
    pub atmos: Vec<AtmoDraw>,
}

/// One atmosphere to draw.
#[derive(Clone, Debug)]
pub struct AtmoDraw {
    /// NAIF id of the body; selects its preset and LUTs.
    pub naif: i32,
    /// Body centre − camera [m], root (ICRS) axes.
    pub rel_pos_m: DVec3,
    /// Body-fixed → ICRS rotation.
    pub orient: DQuat,
    /// Sun − body [m], root axes.
    pub sun_rel_m: DVec3,
    /// Solar illuminance at the body [lx].
    pub sun_illuminance_lux: f64,
    /// The body's exposure weight w: 0 = `star_pre` … 1 = `scene_pre` (lead ruling B6 I1). The
    /// app sets it to `uni_planet::body::exposure_weight(proj_radius_px, eye_adapted)` of the
    /// same body, so the atmosphere and its body share one exposure in the 1.5–4 px LOD band
    /// (1 from 4 px on, and whenever eye-adapted stars are not in effect). The pass writes its
    /// in-scatter at pre_mix(star_pre, scene_pre, w) and stores the star share 1 − w in
    /// `lens_mask.a`; [`atmo_exposure_mix`] sanitises and snaps it first.
    pub exposure_mix: f32,
}

impl AtmoDraw {
    /// Camera − body centre in the atmosphere's frame [m] ([`atmosphere_frame`]). Its length
    /// minus the preset's bottom radius is the camera's height above the body's reference
    /// ellipsoid (to the flattening: ×R/a on the equator, ×R/c at the poles).
    pub fn camera_in_atmosphere_frame(&self) -> DVec3 {
        atmosphere_frame(self.naif, self.orient) * -self.rel_pos_m
    }

    /// [`crate::camera_in_daylight`] for this draw's camera (review I1): inside the shell that
    /// follows the ellipsoid, with the Sun, seen from the camera, above the horizon plane of the
    /// atmosphere's frame (on the ground, the ellipsoid's tangent plane). B9 sets
    /// `uni_post::ExposureSettings::daylight` from it.
    pub fn camera_in_daylight(&self, params: &AtmosphereParams) -> bool {
        let m = atmosphere_frame(self.naif, self.orient);
        // Sun − camera = (Sun − body) + (body − camera).
        let to_sun = m * (self.sun_rel_m + self.rel_pos_m);
        crate::camera_in_daylight(params, m * -self.rel_pos_m, to_sun.normalize_or_zero())
    }

    /// [`crate::scintillation_strength`] for this draw's camera (review I1): the height is
    /// measured from the reference ellipsoid, so the strength at sea level is the same at every
    /// latitude. B9 copies it into `uni_sky::StarInput::scintillation`.
    pub fn scintillation_strength(&self, params: &AtmosphereParams) -> f32 {
        crate::scintillation_strength(params, self.camera_in_atmosphere_frame())
    }
}

/// Most atmospheres drawn per frame (M1 has four); extra draws are skipped.
pub const MAX_DRAWS: usize = 8;

/// Triaxial radii a ≥ b ≥ c [m] of the bodies with atmospheres: WGCCRE 2015 (Archinal et al.
/// 2018, Celest. Mech. Dyn. Astr. 130:22) as in NAIF `pck00011.tpc`, the values uni-physics's
/// registry holds for the bodies pass. `None` for other bodies (their shells stay spheres).
pub fn body_shape_m(naif: i32) -> Option<DVec3> {
    match naif {
        299 => Some(DVec3::splat(6_051.8e3)),
        399 => Some(DVec3::new(6_378.1366e3, 6_378.1366e3, 6_356.7519e3)),
        499 => Some(DVec3::new(3_396.19e3, 3_396.19e3, 3_376.20e3)),
        606 => Some(DVec3::new(2_575.15e3, 2_574.78e3, 2_574.47e3)),
        _ => None,
    }
}

/// Twin of `struct AtmoParamsGpu` in `atmosphere_common.wgsl` (nine vec4, 144 B;
/// `atmo_uniforms_layout_match_wgsl`). Coefficients [1/m] at (680, 550, 440) nm.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmoParamsGpu {
    /// bottom radius, top radius, Rayleigh scale height, Mie scale height [m].
    pub radii: [f32; 4],
    /// xyz Rayleigh scattering; w layer-2 tent centre [m].
    pub rayleigh_scattering: [f32; 4],
    /// xyz Mie scattering; w layer-2 tent full width [m].
    pub mie_scattering: [f32; 4],
    /// xyz Mie extinction; w 0.
    pub mie_extinction: [f32; 4],
    /// xyz Mie Cornette–Shanks g; w 0.
    pub mie_g: [f32; 4],
    /// xyz layer-2 scattering at the peak; w 0.
    pub layer2_scattering: [f32; 4],
    /// xyz layer-2 extinction at the peak; w 0.
    pub layer2_extinction: [f32; 4],
    /// xyz layer-2 Cornette–Shanks g; w 0.
    pub layer2_g: [f32; 4],
    /// xyz ground albedo; w 0.
    pub ground_albedo: [f32; 4],
}

impl AtmoParamsGpu {
    /// Packs `p` (radii and heights as f32: exact to 0.5 m at Earth's radius).
    pub fn from_params(p: &AtmosphereParams) -> Self {
        let v = |v: Vec3, w: f32| v.extend(w).to_array();
        Self {
            radii: [
                p.bottom_radius_m as f32,
                p.top_radius_m as f32,
                p.rayleigh_scale_height_m,
                p.mie_scale_height_m,
            ],
            rayleigh_scattering: v(p.rayleigh_scattering, p.layer2_center_m),
            mie_scattering: v(p.mie_scattering, p.layer2_width_m),
            mie_extinction: v(p.mie_extinction, 0.0),
            mie_g: v(p.mie_g, 0.0),
            layer2_scattering: v(p.layer2_scattering, 0.0),
            layer2_extinction: v(p.layer2_extinction, 0.0),
            layer2_g: v(p.layer2_g, 0.0),
            ground_albedo: v(p.ground_albedo, 0.0),
        }
    }

    #[cfg(test)]
    fn field_offsets() -> Vec<(String, usize)> {
        use core::mem::offset_of;
        [
            ("radii", offset_of!(Self, radii)),
            ("rayleigh_scattering", offset_of!(Self, rayleigh_scattering)),
            ("mie_scattering", offset_of!(Self, mie_scattering)),
            ("mie_extinction", offset_of!(Self, mie_extinction)),
            ("mie_g", offset_of!(Self, mie_g)),
            ("layer2_scattering", offset_of!(Self, layer2_scattering)),
            ("layer2_extinction", offset_of!(Self, layer2_extinction)),
            ("layer2_g", offset_of!(Self, layer2_g)),
            ("ground_albedo", offset_of!(Self, ground_albedo)),
        ]
        .map(|(n, o)| (n.to_owned(), o))
        .to_vec()
    }
}

/// Twin of `struct AtmoDrawGpu` in `atmosphere_common.wgsl` (seven vec4, 112 B), one per draw at
/// a 256-byte-aligned dynamic offset.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmoDrawGpu {
    /// Columns x, y, z of the root → scaled body frame matrix (w 0).
    pub to_body_x: [f32; 4],
    pub to_body_y: [f32; 4],
    pub to_body_z: [f32; 4],
    /// xyz camera − body centre in the scaled body frame [m]; w the camera's height above the
    /// bottom [m].
    pub origin: [f32; 4],
    /// xyz unit direction to the Sun in the scaled body frame; w sun illuminance [lx].
    pub sun: [f32; 4],
    /// NDC x0, y0, x1, y1 of the screen quad.
    pub rect: [f32; 4],
    /// x exposure mix ([`atmo_exposure_mix`] of [`AtmoDraw::exposure_mix`]; 0 = `star_pre`,
    /// 1 = `scene_pre`); y 1 = fullscreen triangle; z the own-ground depth tolerance [m]; w 0.
    pub misc: [f32; 4],
}

impl AtmoDrawGpu {
    #[cfg(test)]
    fn field_offsets() -> Vec<(String, usize)> {
        use core::mem::offset_of;
        [
            ("to_body_x", offset_of!(Self, to_body_x)),
            ("to_body_y", offset_of!(Self, to_body_y)),
            ("to_body_z", offset_of!(Self, to_body_z)),
            ("origin", offset_of!(Self, origin)),
            ("sun", offset_of!(Self, sun)),
            ("rect", offset_of!(Self, rect)),
            ("misc", offset_of!(Self, misc)),
        ]
        .map(|(n, o)| (n.to_owned(), o))
        .to_vec()
    }
}

/// The camera position the march starts from: `ro` (camera − body centre), lifted to 1 m above
/// the bottom when it lies within 1 m of it (f32 cannot resolve less at planetary radii);
/// `None` when the camera is more than 1 m below the bottom (inside the body, or under Venus's
/// cloud deck: the pass draws nothing for that atmosphere) or `ro` is not finite.
pub(crate) fn camera_origin(params: &AtmosphereParams, ro: DVec3) -> Option<DVec3> {
    let r = ro.length();
    let b = params.bottom_radius_m;
    if !r.is_finite() || r < b - 1.0 {
        return None;
    }
    if r < b + 1.0 {
        let up = if r > 0.0 { ro / r } else { DVec3::Z };
        return Some(up * (b + 1.0));
    }
    Some(ro)
}

/// The march bounds of a ray from `ro` (already through [`camera_origin`]) along the unit `rd`:
/// from the camera or the shell entry to the shell exit, the ground or `t_max`, whichever comes
/// first. `None` when the segment is empty. Twin of `atmo_view_ray` with the depth given as a
/// distance.
pub(crate) fn segment_bounds(
    p: &AtmosphereParams,
    ro: DVec3,
    rd: DVec3,
    t_max: f64,
) -> Option<(f64, f64)> {
    let (near, far) = ray_sphere(ro, rd, p.top_radius_m)?;
    if far <= 0.0 {
        return None;
    }
    let t0 = near.max(0.0);
    let mut t1 = far.min(t_max);
    if let Some((g0, g1)) = ray_sphere(ro, rd, p.bottom_radius_m)
        && g1 > 0.0
    {
        if g0 > 0.0 {
            t1 = t1.min(g0);
        } else if ro.dot(rd) < 0.0 {
            t1 = t0;
        }
    }
    (t1 > t0).then_some((t0, t1))
}

/// The light a ray picks up through the atmosphere: from `ro_m` (camera − body centre [m]) along
/// the unit direction `rd` for at most `t_max_m` metres (the scene depth), with the Sun in the
/// unit direction `sun_dir` (all in the same axes; a sphere of the preset's radii, without the
/// ellipsoid scaling the pass applies). Returns (inscattered radiance per lux of sun illuminance
/// [cd/m² per lx], transmittance along the segment), per channel. CPU twin of the view march in
/// `atmosphere_common.wgsl` (B3). Non-finite input, an empty segment or a camera inside the body
/// gives (0, 1).
pub fn segment_radiance(
    params: &AtmosphereParams,
    ro_m: DVec3,
    rd: DVec3,
    t_max_m: f64,
    sun_dir: DVec3,
) -> (Vec3, Vec3) {
    let empty = (Vec3::ZERO, Vec3::ONE);
    let (rd, sun) = (rd.normalize_or_zero(), sun_dir.normalize_or_zero());
    if rd == DVec3::ZERO || sun == DVec3::ZERO || t_max_m.is_nan() {
        return empty;
    }
    let Some(ro) = camera_origin(params, ro_m) else {
        return empty;
    };
    let Some((t0, t1)) = segment_bounds(params, ro, rd, t_max_m) else {
        return empty;
    };
    let luts = cpu_luts(params);
    // Re-centred at the entry point, with the view march's step count, as the shader does.
    let n = view_step_count(params, t1 - t0);
    let m: March = march_steps(&luts, ro + rd * t0, rd, 0.0, t1 - t0, sun, n);
    (m.inscatter.as_vec3(), m.transmittance.as_vec3())
}

/// The transmittance along the same segment as [`segment_radiance`] from the transmittance LUT,
/// without a march (Bruneton 2017, `GetTransmittance`): T(r₀, μ₀)/T(r₁, μ₁) for a ray that
/// misses the ground, T(r₁, −μ₁)/T(r₀, −μ₀) for one that hits it, clamped to 1. The mask pass's
/// `g` (plan fallback ladder, atmosphere step 1; task B9p). Twin of
/// `atmo_segment_lut_transmittance`. With the camera outside the shell it is one LUT read; from
/// inside, near-horizon rays divide two small numbers (see the test). Non-finite input, an empty
/// segment or a camera inside the body gives 1.
pub fn segment_transmittance_lut(
    params: &AtmosphereParams,
    ro_m: DVec3,
    rd: DVec3,
    t_max_m: f64,
) -> Vec3 {
    let rd = rd.normalize_or_zero();
    if rd == DVec3::ZERO || t_max_m.is_nan() {
        return Vec3::ONE;
    }
    let Some(ro) = camera_origin(params, ro_m) else {
        return Vec3::ONE;
    };
    let Some((t0, t1)) = segment_bounds(params, ro, rd, t_max_m) else {
        return Vec3::ONE;
    };
    let luts = cpu_luts(params);
    lut_ratio_transmittance(&luts, ro + rd * t0, rd, t1 - t0).as_vec3()
}

/// The LUT ratio itself, from the segment's start `o` (body-centred) along `d` for `len`
/// metres, as `atmo_segment_lut_transmittance(o, d, len)` takes it (the probe compares the two
/// on the same inputs).
pub(crate) fn lut_ratio_transmittance(luts: &CpuLuts, o: DVec3, d: DVec3, len: f64) -> DVec3 {
    let (num, den) = lut_ratio_parts(luts, o, d, len);
    (num / den.max(DVec3::splat(1e-30))).min(DVec3::ONE)
}

/// The numerator and denominator of [`lut_ratio_transmittance`] (the probe scales its tolerance
/// by the denominator: a LUT error ε moves the ratio by about ε·(1 + T)/den).
pub(crate) fn lut_ratio_parts(luts: &CpuLuts, o: DVec3, d: DVec3, len: f64) -> (DVec3, DVec3) {
    let dn = d.normalize_or_zero();
    let e = o + d * len;
    let (r0, r1) = (o.length(), e.length());
    let (mu0, mu1) = (o.dot(dn) / r0, e.dot(dn) / r1);
    if ray_hits_ground(luts.params(), r0, mu0) {
        (
            lut_transmittance(luts, r1, -mu1),
            lut_transmittance(luts, r0, -mu0),
        )
    } else {
        (
            lut_transmittance(luts, r0, mu0),
            lut_transmittance(luts, r1, mu1),
        )
    }
}

/// Disk-integrated reflectance of a body with this atmosphere over a uniform Lambert ground of
/// albedo `ground_albedo`, seen from far away at solar phase angle `phase_rad` (Sun–body–observer),
/// per channel, in units of geometric albedo.
///
/// It is the body's flux divided by that of a Lambert disk of albedo 1 and radius R =
/// `bottom_radius_m`, seen face-on under the same illuminance from the same distance. At phase 0
/// it is the geometric albedo p, and at phase α it is p·Φ(α), the phase function as defined by
/// Russell 1916 (ApJ 43, 173). F/F_ref = (1/R²)·∫ (L/E) dA over the projected disk out to the
/// shell top, where L/E is:
/// - [`segment_radiance`]'s inscatter along the ray; plus,
/// - where the ray reaches the ground, the view transmittance × ρ/π × (T_sun·μ₀ + sky
///   irradiance), using [`crate::ground::ground_lighting`] (the bodies pass's twin).
///
/// Limits and conventions:
/// - The body is the preset's sphere: no ellipsoid, no terrain, no clouds beyond what the
///   ground albedo holds. Sun and observer are at infinity.
/// - The integral uses `rings` annuli of impact parameter in total, split between the ground
///   disk and the shell annulus so that none straddles the ground's limb. Within each part the
///   rings are midpoints in θ, with b = R·sin θ, so they crowd toward the limb.
/// - For α > 0 it adds about `sectors` azimuth cells over the half disk, because the light is
///   mirror-symmetric about the plane of Sun and observer. A ring that crosses the terminator is
///   split at it.
/// - Each sample is one view march (about 5 ms in an unoptimised build). With 32 rings × 16
///   sectors a bare Lambert sphere is within 1 % of Russell's closed form at 0°, 60° and 120°
///   (`disk_reflectance_follows_lambert_phase_law`).
/// - For Venus, R is the cloud-top sphere. Multiply by (R/R_ref)² to refer the result to another
///   radius.
///
/// Uses: CPU only. It gives B9's expected disk flux for an atmosphere body (the W-B1 check,
/// review I4) and the report's albedo numbers. A non-finite phase or radius gives 0.
pub fn disk_reflectance(
    params: &AtmosphereParams,
    ground_albedo: Vec3,
    phase_rad: f64,
    rings: usize,
    sectors: usize,
) -> Vec3 {
    use std::f64::consts::{FRAC_PI_2, PI};
    let (r, top) = (params.bottom_radius_m, params.top_radius_m);
    if !(phase_rad.is_finite() && r.is_finite() && top.is_finite() && r > 0.0 && top >= r) {
        return Vec3::ZERO;
    }
    let alpha = phase_rad.abs().min(PI);
    let rings = rings.max(1);
    let sectors = if alpha == 0.0 { 1 } else { sectors.max(1) };
    // Observer on +x; the Sun at phase α in the x–y plane; parallel rays toward −x from outside
    // the shell.
    let sun = DVec3::new(alpha.cos(), alpha.sin(), 0.0);
    let far = 1e9 + 10.0 * top;
    let rho = ground_albedo.as_dvec3();
    // L/E along the ray at impact parameter b and azimuth φ (from +y toward +z).
    let radiance = |b: f64, phi: f64| -> DVec3 {
        let (y, z) = (b * phi.cos(), b * phi.sin());
        let (l, t) = segment_radiance(params, DVec3::new(far, y, z), -DVec3::X, f64::INFINITY, sun);
        let mut radiance = l.as_dvec3();
        if b < r {
            let normal = DVec3::new((r * r - b * b).sqrt(), y, z) / r;
            let mu_s = normal.dot(sun);
            let (t_sun, sky) = crate::ground::ground_lighting(Some(params), r, mu_s);
            radiance +=
                t.as_dvec3() * rho / PI * (t_sun.as_dvec3() * mu_s.max(0.0) + sky.as_dvec3());
        }
        radiance
    };
    // Rings: b = R_m·sin θ with the midpoint rule in θ, whose area element b·db·dφ =
    // R_m²·sin θ·cos θ·dθ·dφ is smooth where the view cosine μ = √(1 − b²/R²) is not. The
    // ground disk [0, R] (R_m = R, θ ∈ [0, π/2]) and the shell annulus [R, top] (R_m = top,
    // θ ∈ [asin(R/top), π/2]) get separate rings, so no ring straddles the limb of the ground.
    let theta_r = (r / top).asin();
    let shell_rings = if top > r {
        ((rings as f64 * (1.0 - theta_r / FRAC_PI_2)).round() as usize).clamp(1, rings)
    } else {
        0
    };
    let ground_rings = (rings - shell_rings).max(1);
    let ranges = [
        (r, 0.0, FRAC_PI_2, ground_rings),
        (top, theta_r, FRAC_PI_2, shell_rings),
    ];
    let mut sum = DVec3::ZERO;
    for (r_m, theta0, theta1, n_rings) in ranges {
        let d_theta = (theta1 - theta0) / n_rings.max(1) as f64;
        for k in 0..n_rings {
            let theta = theta0 + d_theta * (k as f64 + 0.5);
            let b = r_m * theta.sin();
            let ring_area_per_rad = r_m * r_m * theta.sin() * theta.cos() * d_theta;
            // Azimuth arcs over the half disk φ ∈ [0, π]. A ring that crosses the terminator is
            // split there, at the kink of the sunlit ground term (μ₀ = 0 at azimuth φ_t,
            // cos φ_t = −cos α·√(R² − b²)/(b·sin α)), so the midpoint rule sees a smooth
            // integrand on each arc.
            let mut arcs = [(0.0, PI, sectors), (PI, PI, 0)];
            if sectors > 1 && b < r && alpha.sin() > 1e-9 {
                let c = -alpha.cos() * (r * r - b * b).sqrt() / (b * alpha.sin());
                if c.abs() < 1.0 {
                    let phi_t = c.acos();
                    let lit =
                        ((sectors as f64 * phi_t / PI).round() as usize).clamp(1, sectors - 1);
                    arcs = [(0.0, phi_t, lit), (phi_t, PI, sectors - lit)];
                }
            }
            for (start, end, n) in arcs {
                let d_phi = (end - start) / n.max(1) as f64;
                for j in 0..n {
                    // Doubled for the mirrored half disk (2π in all).
                    let phi = start + d_phi * (j as f64 + 0.5);
                    sum += radiance(b, phi) * (ring_area_per_rad * 2.0 * d_phi);
                }
            }
        }
    }
    (sum / (r * r)).as_vec3()
}

/// The atmosphere's frame of body `naif` with body-fixed → ICRS rotation `orient`: the matrix
/// R/(a, b, c) × (body-fixed ← ICRS), R = (abc)^⅓, from root (ICRS) axes to the body-fixed
/// axes scaled so the WGCCRE reference ellipsoid ([`body_shape_m`]) becomes the sphere of radius
/// R. R equals the preset's bottom radius to within 1 m for Earth, 14 m for Mars and 70 m for
/// Titan. Venus is a sphere, so only the rotation applies and its bottom stays at the cloud
/// tops. Positions relative to the
/// body centre and directions mapped by it are what the march, the LUT twins,
/// [`crate::camera_in_daylight`] and [`crate::scintillation_strength`] take. Bodies without a
/// [`body_shape_m`] get the rotation only.
pub fn atmosphere_frame(naif: i32, orient: DQuat) -> DMat3 {
    let to_fixed = DMat3::from_quat(orient.normalize()).transpose();
    match body_shape_m(naif) {
        Some(abc) => {
            let mean = (abc.x * abc.y * abc.z).cbrt();
            DMat3::from_diagonal(DVec3::splat(mean) / abc) * to_fixed
        }
        None => to_fixed,
    }
}

/// The screen rectangle [NDC x0, y0, x1, y1] bounding a sphere of radius `radius` centred at
/// `center_view` (view space: camera at the origin looking down −Z), under a projection with
/// x_ndc = sx·x/(−z), y_ndc = sy·y/(−z). Exact per axis: the planes through the camera tangent
/// to the sphere (Mara & McGuire 2013, "2D Polyhedral Bounds of a Clipped, Perspective-Projected
/// 3D Sphere", JCGT 2(2)). `None` when the sphere reaches the camera plane (z ≥ −radius): draw
/// the full screen then.
pub(crate) fn sphere_screen_rect(
    center_view: DVec3,
    radius: f64,
    sx: f64,
    sy: f64,
) -> Option<[f64; 4]> {
    let depth = -center_view.z;
    if !(depth > radius) || !center_view.is_finite() {
        return None;
    }
    let bounds = |a: f64, scale: f64| {
        let l = (a * a + depth * depth).sqrt();
        let (centre, half) = (a.atan2(depth), (radius / l).asin());
        (scale * (centre - half).tan(), scale * (centre + half).tan())
    };
    let (x0, x1) = bounds(center_view.x, sx);
    let (y0, y1) = bounds(center_view.y, sy);
    Some([x0, y0, x1, y1])
}

/// The GPU data of one draw for this frame, or `None` when it draws nothing (camera inside the
/// body, shell off screen, bad input).
pub(crate) fn draw_uniform(
    params: &AtmosphereParams,
    draw: &AtmoDraw,
    view_rot: Mat4,
    proj: Mat4,
) -> Option<AtmoDrawGpu> {
    let finite = draw.rel_pos_m.is_finite()
        && draw.orient.is_finite()
        && draw.sun_rel_m.is_finite()
        && draw.sun_illuminance_lux.is_finite()
        && draw.sun_illuminance_lux >= 0.0;
    let sun_dir = draw.sun_rel_m.normalize_or_zero();
    if !finite || sun_dir == DVec3::ZERO {
        return None;
    }
    let m = atmosphere_frame(draw.naif, draw.orient);
    let origin = camera_origin(params, m * -draw.rel_pos_m)?;
    let sun = (m * sun_dir).normalize();
    let height = origin.length() - params.bottom_radius_m;
    // The shell in root space lies within top × max(a, b, c)/R of the centre.
    let stretch = body_shape_m(draw.naif).map_or(1.0, |abc| {
        abc.max_element() / (abc.x * abc.y * abc.z).cbrt()
    });
    let bound = params.top_radius_m * stretch * 1.001;
    let center_view = view_rot.as_dmat4().transform_vector3(draw.rel_pos_m);
    if center_view.z - bound >= 0.0 {
        // Entirely behind the camera.
        return None;
    }
    let (sx, sy) = (f64::from(proj.x_axis.x), f64::from(proj.y_axis.y));
    let (rect, full) = match sphere_screen_rect(center_view, bound, sx, sy) {
        Some([x0, y0, x1, y1]) => {
            if x0 >= 1.0 || x1 <= -1.0 || y0 >= 1.0 || y1 <= -1.0 {
                return None;
            }
            ([x0.max(-1.0), y0.max(-1.0), x1.min(1.0), y1.min(1.0)], 0.0)
        }
        None => ([-1.0, -1.0, 1.0, 1.0], 1.0),
    };
    // The bodies pass draws the ground on a proxy mesh up to ~1e-3 × the camera's height
    // outside the true surface (B0's cap-mesh gap; 2e-3 for the 32 × 64 ladder step); a depth
    // surface below this is the body's own ground. Capped at 2 % of the radius, far below any
    // moon.
    let own_ground = (200.0 + 2e-3 * height.max(0.0)).min(0.02 * params.bottom_radius_m);
    let col = |c: DVec3| c.as_vec3().extend(0.0).to_array();
    Some(AtmoDrawGpu {
        to_body_x: col(m.x_axis),
        to_body_y: col(m.y_axis),
        to_body_z: col(m.z_axis),
        origin: origin.as_vec3().extend(height as f32).to_array(),
        sun: sun
            .as_vec3()
            .extend(draw.sun_illuminance_lux as f32)
            .to_array(),
        rect: rect.map(|v| v as f32),
        // The body's exposure weight, snapped (lead ruling B6 I1).
        misc: [
            atmo_exposure_mix(draw.exposure_mix),
            full,
            own_ground as f32,
            0.0,
        ],
    })
}

/// Smallest non-zero star share the pass stores in `lens_mask.a`: 2⁻¹⁴, the smallest normal
/// binary16 (IEEE 754-2019 §3.6; `lens_mask` is Rgba16Float). Metal does not promise to keep f16
/// subnormals when it stores a render target; a share flushed to 0 would be metered at scene
/// exposure while the light was written nearer `star_pre`. Twin of
/// `uni_planet::body::STAR_SHARE_MIN` (uni-planet depends on this crate, not the reverse;
/// uni-planet's tests check the two agree).
pub const STAR_SHARE_MIN: f32 = 1.0 / 16384.0;

/// The exposure weight the pass uploads (`AtmoDrawGpu::misc[0]`) for [`AtmoDraw::exposure_mix`]
/// `w`: NaN gives 1 (scene exposure, the physical one), w is clamped to [0, 1], and snapped to 1
/// when 1 − w < [`STAR_SHARE_MIN`], exactly as `uni_planet::body::exposure_weight` snaps the
/// body's own w (idempotent: a w the app copied from there passes through unchanged).
pub fn atmo_exposure_mix(w: f32) -> f32 {
    if w.is_nan() {
        return 1.0;
    }
    let w = w.clamp(0.0, 1.0);
    if 1.0 - w < STAR_SHARE_MIN { 1.0 } else { w }
}

/// CPU twin of `atmo_star_share` in `atmosphere_common.wgsl`: the star share 1 − w that every
/// fragment of a draw with uploaded exposure weight `w` (`misc[0]`) stores in `lens_mask.a`,
/// blended with `Max`. Every fragment the march does not discard adds in-scatter at that
/// exposure, so it covers its pixel for this purpose (discarded pixels store nothing).
pub fn atmo_star_share(w: f32) -> f32 {
    1.0 - w
}

/// The pass's `lens_mask` target: green (1 − luminance(T)) and alpha (the star share,
/// [`atmo_star_share`]) written, both blended with `Max` (M1 frame contract, lead ruling B6 I1;
/// WebGPU requires factor One for `Max`); red and blue are left to the lens pass.
pub(crate) fn lens_mask_target() -> wgpu::ColorTargetState {
    let max = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Max,
    };
    wgpu::ColorTargetState {
        format: LENS_MASK_FORMAT,
        blend: Some(wgpu::BlendState {
            color: max,
            alpha: max,
        }),
        write_mask: wgpu::ColorWrites::GREEN.union(wgpu::ColorWrites::ALPHA),
    }
}

/// The pipelines of one variant. Both start with the half-resolution march into the scratch
/// targets (`fs_march`, task B9p).
enum Pipelines {
    /// The march; the dual-source composite into `hdr`; the `lens_mask` (g and a) pipeline.
    Dual {
        march: wgpu::RenderPipeline,
        color: wgpu::RenderPipeline,
        mask: wgpu::RenderPipeline,
    },
    /// The march; one composite into `hdr` (grey transmittance) and `lens_mask` (g and a).
    Grey {
        march: wgpu::RenderPipeline,
        composite: wgpu::RenderPipeline,
    },
}

/// Format of the half-resolution scratch targets (light + segment length [km]; transmittance +
/// draw tag), task B9p.
const SCRATCH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Largest relative difference between a half-resolution sample's segment length and the pixel's
/// own for the composite to use the sample (twin of `ATMO_UPSAMPLE_LEN_TOL`). A ray that grazes
/// the ground and one that just misses it differ by a factor of about 2, and a ray stopped by a
/// body in front by more; within one geometry the length changes by under 10 % per sample except
/// in the outermost kilometres of the shell, where the light is faint and the pixel is marched
/// instead.
pub const UPSAMPLE_LEN_TOL: f32 = 0.1;

/// The half-resolution size the march runs at for an internal size `size`: half, rounded up, so
/// full-resolution pixel p reads samples ⌊p/2⌋ and ⌊p/2⌋ + 1 (clamped).
pub fn half_size(size: glam::UVec2) -> glam::UVec2 {
    glam::UVec2::new(size.x.div_ceil(2), size.y.div_ceil(2)).max(glam::UVec2::ONE)
}

/// Largest spread of the samples' light (largest channel, pre-exposed) relative to the brightest
/// above [`UPSAMPLE_LIGHT_FLOOR`] for a pixel to be interpolated (twin of
/// `ATMO_UPSAMPLE_LIGHT_STEP`). A haze ring one or two pixels wide (Venus's night limb at
/// 150 000 km: 48 km of tangent height per pixel, the haze 28 km thick) differs by far more
/// between samples two pixels apart; the smooth gradients of a disk or a wide limb arc by less.
pub const UPSAMPLE_LIGHT_STEP: f32 = 0.5;
/// Light below this (pre-exposed; AgX's black is 1.75e-4) is never a reason to march.
pub const UPSAMPLE_LIGHT_FLOOR: f32 = 1e-4;
/// Largest spread of the samples' transmittance (largest channel) for a pixel to be interpolated
/// (twin of `ATMO_UPSAMPLE_T_STEP`).
pub const UPSAMPLE_T_STEP: f32 = 0.25;

/// One half-resolution sample as the composite reads it: segment length [km], draw tag (index +
/// 1; 0 = not marched), the light's largest channel (pre-exposed) and the transmittance's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfSample {
    pub len_km: f32,
    pub tag: f32,
    pub light_max: f32,
    pub trans_max: f32,
}

/// CPU twin of the composite's choice (`atmo_composite`): for full-resolution pixel `pix` with
/// segment length `len_km` in draw `tag`, the bilinear weights of samples ⌊pix/2⌋ + (0,0),
/// (1,0), (0,1), (1,1) (even coordinates sit on a sample, odd ones halfway). `None`, and the
/// composite marches the pixel itself, unless every weighted sample is of this draw, within
/// [`UPSAMPLE_LEN_TOL`] of `len_km` (the same ray geometry: a body, the ground or the shell ends
/// it), and the samples' light and transmittance are alike ([`UPSAMPLE_LIGHT_STEP`],
/// [`UPSAMPLE_T_STEP`]).
pub fn upsample_weights(
    pix: glam::UVec2,
    len_km: f32,
    tag: f32,
    samples: [HalfSample; 4],
) -> Option<[f32; 4]> {
    let odd = (pix % 2).as_vec2();
    let mut w = [0.0_f32; 4];
    let mut alike = true;
    let (mut l_lo, mut l_hi, mut t_lo, mut t_hi) = (f32::MAX, 0.0_f32, f32::MAX, 0.0_f32);
    for (k, s) in samples.into_iter().enumerate() {
        let wx = if k & 1 == 1 {
            0.5 * odd.x
        } else {
            1.0 - 0.5 * odd.x
        };
        let wy = if k >> 1 == 1 {
            0.5 * odd.y
        } else {
            1.0 - 0.5 * odd.y
        };
        if wx * wy > 0.0 {
            alike &= s.tag == tag && (s.len_km - len_km).abs() <= UPSAMPLE_LEN_TOL * len_km;
            (l_lo, l_hi) = (l_lo.min(s.light_max), l_hi.max(s.light_max));
            (t_lo, t_hi) = (t_lo.min(s.trans_max), t_hi.max(s.trans_max));
            w[k] = wx * wy;
        }
    }
    let smooth = !(l_hi - l_lo > UPSAMPLE_LIGHT_STEP * l_hi && l_hi > UPSAMPLE_LIGHT_FLOOR)
        && t_hi - t_lo <= UPSAMPLE_T_STEP;
    (alike && smooth).then_some(w)
}

/// One draw recorded in `prepare`.
struct DrawCmd {
    /// Byte offset of its `AtmoDrawGpu` in the draw buffer.
    offset: u32,
    /// Index into `AtmosphereFeature::atmos`.
    atmo: usize,
    /// 3 (fullscreen triangle) or 6 (quad).
    vertices: u32,
}

/// The atmosphere pass (`RenderFeature` named "atmosphere").
pub struct AtmosphereFeature {
    input: SharedInput<AtmoScene>,
    /// (NAIF id, parameters, group 2) per atmosphere of the set.
    atmos: Vec<(i32, AtmosphereParams, wgpu::BindGroup)>,
    pipelines: Pipelines,
    /// Group 1 of the march and the mask: draw uniform + depth.
    draw_bgl: wgpu::BindGroupLayout,
    /// Group 1 of the composite: draw uniform + depth + the two scratch targets.
    composite_bgl: wgpu::BindGroupLayout,
    draw_buffer: wgpu::Buffer,
    /// Byte stride between draws in `draw_buffer` (the device's uniform offset alignment).
    stride: u32,
    /// Group 1 bind groups and scratch targets for the current targets; rebuilt on resize.
    bound: Option<Bound>,
    draws: Vec<DrawCmd>,
}

/// What [`AtmosphereFeature`] binds for one set of frame targets.
struct Bound {
    /// Group 1 of the march and the mask.
    draw_group: wgpu::BindGroup,
    /// Group 1 of the composite.
    composite_group: wgpu::BindGroup,
    /// The half-resolution scratch targets: light + length, transmittance + tag.
    light_view: wgpu::TextureView,
    trans_view: wgpu::TextureView,
    _textures: [wgpu::Texture; 2],
}

impl AtmosphereFeature {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`; `set` holds the
    /// LUTs; `dual_source` is `GpuContext::has_dual_source_blending()` and picks the variant.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        set: &AtmosphereSet,
        input: SharedInput<AtmoScene>,
        dual_source: bool,
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let device = &gpu.device;
        let dual_source = dual_source && gpu.has_dual_source_blending();
        let modules = if dual_source {
            (
                Some(lib.module(device, crate::ATMOSPHERE_DUAL_WGSL)?),
                Some(lib.module(device, crate::ATMOSPHERE_MASK_WGSL)?),
                None,
            )
        } else {
            (
                None,
                None,
                Some(lib.module(device, crate::ATMOSPHERE_GREY_WGSL)?),
            )
        };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride = (size_of::<AtmoDrawGpu>() as u32).div_ceil(align) * align;
        let draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmo draws"),
            size: u64::from(stride) * MAX_DRAWS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_entries = [
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(size_of::<AtmoDrawGpu>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ];
        let scratch_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo draw bgl"),
            entries: &draw_entries,
        });
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo composite bgl"),
            entries: &[
                draw_entries[0],
                draw_entries[1],
                scratch_entry(2),
                scratch_entry(3),
            ],
        });
        let layout_with = |label: &str, group1: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(camera_bgl), Some(group1), Some(set.view_bgl())],
                immediate_size: 0,
            })
        };
        let layout = layout_with("atmo layout", &draw_bgl);
        let composite_layout = layout_with("atmo composite layout", &composite_bgl);
        let pipeline = |label: &str,
                        layout: &wgpu::PipelineLayout,
                        module: &wgpu::ShaderModule,
                        fs: &str,
                        targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_atmosphere"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets,
                }),
                primitive: wgpu::PrimitiveState::default(),
                // No depth attachment: the pass reads the depth target as a texture.
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let mask_target = Some(lens_mask_target());
        // hdr's alpha is never touched.
        let keep_alpha = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        // The half-resolution march writes both scratch targets, no blending.
        let scratch = [
            Some(wgpu::ColorTargetState {
                format: SCRATCH_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: SCRATCH_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let pipelines = match modules {
            (Some(dual), Some(mask), _) => Pipelines::Dual {
                march: pipeline("atmosphere march", &layout, &dual, "fs_march", &scratch),
                // dst·T + src: source 1 is the per-channel transmittance.
                color: pipeline(
                    "atmosphere",
                    &composite_layout,
                    &dual,
                    "fs_dual",
                    &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::Src1,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: keep_alpha,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                ),
                mask: pipeline("atmosphere mask", &layout, &mask, "fs_mask", &[mask_target]),
            },
            (_, _, Some(grey)) => Pipelines::Grey {
                march: pipeline("atmosphere march", &layout, &grey, "fs_march", &scratch),
                composite: pipeline(
                    "atmosphere",
                    &composite_layout,
                    &grey,
                    "fs_grey",
                    &[
                        // dst·(1 − α) + L.
                        Some(wgpu::ColorTargetState {
                            format: HDR_FORMAT,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: keep_alpha,
                            }),
                            write_mask: wgpu::ColorWrites::COLOR,
                        }),
                        mask_target,
                    ],
                ),
            },
            _ => anyhow::bail!("atmosphere: no shader variant"),
        };
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the atmosphere pipelines: {err}");
        }
        let atmos = set
            .entries()
            .iter()
            .map(|e| (e.naif, e.params.clone(), e.view_group.clone()))
            .collect();
        Ok(Self {
            input,
            atmos,
            pipelines,
            draw_bgl,
            composite_bgl,
            draw_buffer,
            stride,
            bound: None,
            draws: Vec::new(),
        })
    }

    /// (Re)creates the half-resolution scratch targets and the group 1 bind groups against
    /// `targets` (the depth view and the size change with the targets).
    fn bind_targets(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        let device = &gpu.device;
        let half = half_size(targets.size);
        let scratch = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: half.x,
                    height: half.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SCRATCH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let light = scratch("atmo half light");
        let trans = scratch("atmo half transmittance");
        let light_view = light.create_view(&Default::default());
        let trans_view = trans.create_view(&Default::default());
        let uniform = wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.draw_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(size_of::<AtmoDrawGpu>() as u64),
            }),
        };
        let depth = wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&targets.depth_view),
        };
        let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmo draw bg"),
            layout: &self.draw_bgl,
            entries: &[uniform.clone(), depth.clone()],
        });
        let composite_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmo composite bg"),
            layout: &self.composite_bgl,
            entries: &[
                uniform,
                depth,
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&light_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&trans_view),
                },
            ],
        });
        self.bound = Some(Bound {
            draw_group,
            composite_group,
            light_view,
            trans_view,
            _textures: [light, trans],
        });
    }

    /// Records the draws of one pass (pipeline and group 0 already set).
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, group1: &wgpu::BindGroup) {
        for d in &self.draws {
            pass.set_bind_group(1, group1, &[d.offset]);
            pass.set_bind_group(2, &self.atmos[d.atmo].2, &[]);
            pass.draw(0..d.vertices, 0..1);
        }
    }
}

impl RenderFeature for AtmosphereFeature {
    fn name(&self) -> &'static str {
        "atmosphere"
    }

    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets) {
        if self.bound.is_none() {
            self.bind_targets(gpu, targets);
        }
        self.draws.clear();
        let mut uniforms: Vec<(u32, AtmoDrawGpu)> = Vec::new();
        self.input.with(|scene| {
            for draw in &scene.atmos {
                if uniforms.len() >= MAX_DRAWS {
                    break;
                }
                let Some(atmo) = self.atmos.iter().position(|a| a.0 == draw.naif) else {
                    continue;
                };
                let Some(mut u) = draw_uniform(&self.atmos[atmo].1, draw, ctx.view_rot, ctx.proj)
                else {
                    continue;
                };
                // The tag of this draw's half-resolution samples (task B9p).
                u.misc[3] = (uniforms.len() + 1) as f32;
                let offset = self.stride * uniforms.len() as u32;
                let vertices = if u.misc[1] > 0.5 { 3 } else { 6 };
                self.draws.push(DrawCmd {
                    offset,
                    atmo,
                    vertices,
                });
                uniforms.push((offset, u));
            }
        });
        for (offset, u) in uniforms {
            gpu.queue
                .write_buffer(&self.draw_buffer, u64::from(offset), bytemuck::bytes_of(&u));
        }
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
        if self.draws.is_empty() {
            return;
        }
        if self.bound.is_none() {
            self.bind_targets(gpu, targets);
        }
        let Some(bound) = self.bound.as_ref() else {
            return;
        };
        let attachment = |view, load| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        // The half-resolution march (task B9p): both scratch targets cleared to 0 (tag 0 = no
        // sample), every draw in order (a later draw overwrites an earlier one's samples where
        // they overlap; the composite then marches those pixels itself, by the tag).
        let (march, composite, mask) = match &self.pipelines {
            Pipelines::Dual { march, color, mask } => (march, color, Some(mask)),
            Pipelines::Grey { march, composite } => (march, composite, None),
        };
        let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("atmosphere march"),
                color_attachments: &[
                    attachment(&bound.light_view, clear),
                    attachment(&bound.trans_view, clear),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: timer.render_writes("atmosphere"),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(march);
            pass.set_bind_group(0, frame.camera, &[]);
            self.record(&mut pass, &bound.draw_group);
        }
        // The full-resolution composite into hdr (and, grey variant, lens_mask).
        {
            let load = wgpu::LoadOp::Load;
            let views: &[_] = if mask.is_some() {
                &[attachment(&targets.hdr_view, load)]
            } else {
                &[
                    attachment(&targets.hdr_view, load),
                    attachment(&targets.lens_mask_view, load),
                ]
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("atmosphere composite"),
                color_attachments: views,
                depth_stencil_attachment: None,
                timestamp_writes: timer.render_writes("atmosphere composite"),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(composite);
            pass.set_bind_group(0, frame.camera, &[]);
            self.record(&mut pass, &bound.composite_group);
        }
        // Dual variant: lens_mask g and a (g from the transmittance-LUT ratio).
        if let Some(mask) = mask {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("atmosphere mask"),
                color_attachments: &[attachment(&targets.lens_mask_view, wgpu::LoadOp::Load)],
                depth_stencil_attachment: None,
                timestamp_writes: timer.render_writes("atmosphere mask"),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(mask);
            pass.set_bind_group(0, frame.camera, &[]);
            self.record(&mut pass, &bound.draw_group);
        }
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_targets(gpu, targets);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut::cpu_luts;
    use crate::params::{earth, mars, titan, venus};

    fn dir_elev(elev_deg: f64, azimuth_deg: f64) -> DVec3 {
        let (e, a) = (elev_deg.to_radians(), azimuth_deg.to_radians());
        DVec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin())
    }

    /// Zenith sky from the ground with the Sun 60° up: blue (B > G > R).
    #[test]
    fn earth_sky_blue_at_noon() {
        let p = earth();
        let ro = DVec3::new(0.0, 0.0, p.bottom_radius_m + 1.0);
        let (l, t) = segment_radiance(&p, ro, DVec3::Z, f64::INFINITY, dir_elev(60.0, 0.0));
        assert!(l.z > l.y && l.y > l.x, "zenith radiance per lux {l}");
        assert!(
            t.is_finite() && t.min_element() > 0.0 && t.max_element() < 1.0,
            "{t}"
        );
    }

    /// Review I6: with the per-channel dust g, the sky within 5° of a setting Sun is blue
    /// (B/R > 1: here 1–4° above a Sun 2° up) and the noon zenith is red (R/B > 1).
    #[test]
    fn mars_blue_sunset() {
        let p = mars();
        let ro = DVec3::new(0.0, 0.0, p.bottom_radius_m + 1.0);
        let sun = dir_elev(2.0, 0.0);
        for above in [1.0, 2.0, 3.0, 4.0] {
            let (near_sun, _) =
                segment_radiance(&p, ro, dir_elev(2.0 + above, 0.0), f64::INFINITY, sun);
            assert!(
                near_sun.z / near_sun.x > 1.0,
                "{above}° above the setting Sun: {near_sun}"
            );
        }
        let (zenith, _) = segment_radiance(&p, ro, DVec3::Z, f64::INFINITY, dir_elev(60.0, 0.0));
        assert!(zenith.x / zenith.z > 1.0, "noon zenith {zenith}");
    }

    /// Review Focus 2: a camera inside the shell (10 km up) marches from the camera (t_start 0),
    /// an upward ray is finite, a downward ray stops at the ground, a horizontal ray is fine.
    #[test]
    fn segment_from_inside_shell() {
        let p = earth();
        let h = 10e3;
        let ro = DVec3::new(0.0, 0.0, p.bottom_radius_m + h);
        let sun = dir_elev(30.0, 0.0);
        for rd in [DVec3::Z, -DVec3::Z, DVec3::X, dir_elev(-1.0, 45.0)] {
            let (l, t) = segment_radiance(&p, ro, rd, f64::INFINITY, sun);
            assert!(l.is_finite() && t.is_finite(), "{rd}: {l} {t}");
            assert!(l.min_element() > 0.0, "{rd}: {l}");
            assert!(t.min_element() > 0.0 && t.max_element() < 1.0, "{rd}: {t}");
        }
        // Downward: the ground at 10 km ends the march, so a longer t_max changes nothing.
        let ground = segment_radiance(&p, ro, -DVec3::Z, h, sun);
        let beyond = segment_radiance(&p, ro, -DVec3::Z, 1e9, sun);
        assert!((ground.0 - beyond.0).abs().max_element() <= 1e-6 * ground.0.max_element());
        assert!((ground.1 - beyond.1).abs().max_element() <= 1e-9);
        // t_start is the camera: a 100 m segment has the transmittance of 100 m of air there.
        let (short_l, short_t) = segment_radiance(&p, ro, DVec3::X, 100.0, sun);
        let ext = crate::lut::extinction_at(&p, h);
        for c in 0..3 {
            let expect = (-ext[c] * 100.0).exp();
            assert!(
                (f64::from(short_t[c]) - expect).abs() < 1e-6,
                "{c}: {short_t} vs {expect}"
            );
        }
        assert!(short_l.min_element() > 0.0);
        // Degenerate segments add nothing.
        assert_eq!(
            segment_radiance(&p, ro, DVec3::X, 0.0, sun),
            (Vec3::ZERO, Vec3::ONE)
        );
        let (l, t) = segment_radiance(&p, DVec3::NAN, DVec3::X, 1e3, sun);
        assert_eq!((l, t), (Vec3::ZERO, Vec3::ONE));
    }

    /// Every uniform struct matches its WGSL twin (size and member offsets, naga).
    #[test]
    fn atmo_uniforms_layout_match_wgsl() {
        use uni_gfx::{ShaderLib, validate_wgsl};
        use wgpu::naga;
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let m = validate_wgsl(&lib.compose(crate::ATMOSPHERE_COMMON_WGSL).unwrap()).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let wgsl = |name: &str| -> (usize, Vec<(String, usize)>) {
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
                .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
                .collect();
            (layouter[handle].size as usize, members)
        };
        let (size, members) = wgsl("AtmoParamsGpu");
        assert_eq!(size, size_of::<AtmoParamsGpu>());
        let rust = AtmoParamsGpu::field_offsets();
        assert_eq!(members, rust);
        let (size, members) = wgsl("AtmoDrawGpu");
        assert_eq!(size, size_of::<AtmoDrawGpu>());
        assert_eq!(members, AtmoDrawGpu::field_offsets());
    }

    /// Review I7: the dual-source and mask variants validate with DUAL_SOURCE_BLENDING, the grey
    /// variant, the LUT passes and the probe kernels with m0.1's capabilities; every entry point
    /// the feature and the LUT builder use exists.
    #[test]
    fn atmo_shaders_validate() {
        use uni_gfx::{ShaderLib, validate_wgsl, validate_wgsl_with};
        use wgpu::naga::valid::Capabilities as Caps;
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        crate::probes::register_probe_shaders(&mut lib);
        let caps = Caps::default() | Caps::SHADER_FLOAT16 | Caps::DUAL_SOURCE_BLENDING;
        let has = |m: &wgpu::naga::Module, entry: &str| {
            assert!(
                m.entry_points.iter().any(|e| e.name == entry),
                "missing entry point {entry}"
            );
        };
        for (path, entries) in [
            (
                crate::ATMOSPHERE_DUAL_WGSL,
                &["vs_atmosphere", "fs_dual", "fs_march"][..],
            ),
            (
                crate::ATMOSPHERE_MASK_WGSL,
                &["vs_atmosphere", "fs_mask"][..],
            ),
        ] {
            let m = validate_wgsl_with(&lib.compose(path).unwrap(), caps)
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            entries.iter().for_each(|e| has(&m, e));
        }
        for (path, entries) in [
            (
                crate::ATMOSPHERE_GREY_WGSL,
                &["vs_atmosphere", "fs_grey", "fs_march"][..],
            ),
            (crate::TRANSMITTANCE_LUT_WGSL, &["cs_transmittance"][..]),
            (crate::MULTISCATTER_LUT_WGSL, &["cs_multiscatter"][..]),
            (crate::IRRADIANCE_LUT_WGSL, &["cs_irradiance"][..]),
            (crate::probes::PROBE_WGSL, &crate::probes::PROBE_ENTRIES[..]),
        ] {
            let m = validate_wgsl(&lib.compose(path).unwrap())
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            entries.iter().for_each(|e| has(&m, e));
        }
    }

    /// Rec. 709 luminance of a linear colour.
    fn lum(v: Vec3) -> f32 {
        0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z
    }

    /// Solar illuminance at 1 au [lx]: 1361 W/m² (Kopp & Lean 2011) × ≈ 93 lm/W.
    const E_SUN_1AU_LX: f32 = 1.27e5;

    /// The view-direction toward a limb point of a body seen from `cam` far along +x: the point
    /// at angle `phi` round the limb (0 toward +y, 90° toward +z) and tangent height `h`.
    fn limb_ray(p: &AtmosphereParams, cam: DVec3, phi_deg: f64, h: f64) -> DVec3 {
        let phi = phi_deg.to_radians();
        let target = DVec3::new(0.0, phi.cos(), phi.sin()) * (p.bottom_radius_m + h);
        (target - cam).normalize()
    }

    /// The composite's upsample (ladder step 3, task B9p): even pixels copy their own sample,
    /// odd ones average their neighbours, a neighbour of another geometry (length off by more
    /// than 10 %: the ground's edge, a body in front) or another draw is left out, and with none
    /// left the pixel is marched. The WGSL constant matches.
    #[test]
    fn upsample_weights_follow_the_ray_geometry() {
        use glam::UVec2;
        let s = |len_km: f32, tag: f32, light_max: f32, trans_max: f32| HalfSample {
            len_km,
            tag,
            light_max,
            trans_max,
        };
        let same = [s(100.0, 1.0, 0.2, 0.8); 4];
        assert_eq!(
            upsample_weights(UVec2::new(4, 6), 100.0, 1.0, same),
            Some([1.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            upsample_weights(UVec2::new(5, 6), 100.0, 1.0, same),
            Some([0.5, 0.5, 0.0, 0.0])
        );
        assert_eq!(
            upsample_weights(UVec2::new(5, 7), 104.0, 1.0, same),
            Some([0.25; 4])
        );
        // A smooth gradient (light 0.15..0.2, T 0.7..0.8) is interpolated.
        let gradient = [
            s(100.0, 1.0, 0.15, 0.7),
            s(101.0, 1.0, 0.2, 0.8),
            s(99.0, 1.0, 0.17, 0.75),
            s(100.0, 1.0, 0.19, 0.78),
        ];
        assert_eq!(
            upsample_weights(UVec2::new(5, 7), 100.0, 1.0, gradient),
            Some([0.25; 4])
        );
        // Marched: a neighbour past the horizon (twice as long), one of another draw, an
        // unmarched one (tag 0), a one-pixel haze ring (light 0.03 next to 0.2), a step in T. An
        // even pixel only looks at its own sample.
        let mut cases = vec![];
        for (k, bad) in [
            s(200.0, 1.0, 0.2, 0.8),
            s(100.0, 2.0, 0.2, 0.8),
            s(100.0, 0.0, 0.0, 0.0),
            s(100.0, 1.0, 0.03, 0.8),
            s(100.0, 1.0, 0.2, 0.4),
        ]
        .into_iter()
        .enumerate()
        {
            let mut v = same;
            v[k % 4] = bad;
            cases.push((k, v));
        }
        for (k, v) in cases {
            assert_eq!(
                upsample_weights(UVec2::new(5, 7), 100.0, 1.0, v),
                None,
                "{k}"
            );
        }
        let mut far = same;
        far[3] = s(200.0, 1.0, 0.2, 0.8);
        assert!(upsample_weights(UVec2::new(4, 6), 100.0, 1.0, far).is_some());
        // Light under the floor never forces a march.
        let faint = [
            s(100.0, 1.0, 1e-5, 0.8),
            s(100.0, 1.0, 9e-5, 0.8),
            s(100.0, 1.0, 5e-5, 0.8),
            s(100.0, 1.0, 1e-5, 0.8),
        ];
        assert!(upsample_weights(UVec2::new(5, 7), 100.0, 1.0, faint).is_some());
        assert_eq!(half_size(UVec2::new(1920, 1080)), UVec2::new(960, 540));
        assert_eq!(half_size(UVec2::new(1727, 1)), UVec2::new(864, 1));
        let src = include_str!("../shaders/atmosphere_common.wgsl");
        for (name, value) in [
            ("ATMO_UPSAMPLE_LEN_TOL", UPSAMPLE_LEN_TOL),
            ("ATMO_UPSAMPLE_LIGHT_STEP", UPSAMPLE_LIGHT_STEP),
            ("ATMO_UPSAMPLE_LIGHT_FLOOR", UPSAMPLE_LIGHT_FLOOR),
            ("ATMO_UPSAMPLE_T_STEP", UPSAMPLE_T_STEP),
        ] {
            assert!(
                src.contains(&format!("const {name}: f32 = {value:?};")),
                "{name}"
            );
        }
    }

    /// The mask pass's LUT-ratio transmittance (ladder step 1, task B9p) against the march's.
    /// `g` = 1 − luminance(T) is the sky occlusion, and it only matters where the ray reaches the
    /// sky: where it ends on the ground or on a body, that surface's own coverage (g = 1, Max)
    /// hides the sky anyway. Sky rays (from space through the limb arc at 5, 20 and 50 % of the
    /// shell, from inside up and 10° up): |Δg| ≤ 0.005. Ground and body rays from space (one LUT
    /// read): |Δg| ≤ 0.005. Ground rays from inside divide two small LUT values: Titan 10° down
    /// is off by 0.036 in g (blue T 6e-10 against 0.083), hidden by the ground's g = 1; they are
    /// only checked to be finite and in [0, 1].
    #[test]
    fn lut_ratio_transmittance_matches_the_march() {
        for (name, p) in [
            ("Earth", earth()),
            ("Mars", mars()),
            ("Venus", venus()),
            ("Titan", titan()),
        ] {
            let (b, top) = (p.bottom_radius_m, p.top_radius_m);
            let space = DVec3::new(3.0 * b, 0.0, 0.0);
            let inside = DVec3::new(0.0, 0.0, b + 1e3);
            let sun = DVec3::new(0.3, 0.9, 0.3).normalize();
            // (camera, direction, depth, checked to 0.005).
            let cases: [(DVec3, DVec3, f64, bool); 9] = [
                (space, -DVec3::X, f64::INFINITY, true),
                (
                    space,
                    limb_ray(&p, space, 30.0, -0.02 * b),
                    f64::INFINITY,
                    true,
                ),
                (
                    space,
                    limb_ray(&p, space, 30.0, 0.05 * (top - b)),
                    f64::INFINITY,
                    true,
                ),
                (
                    space,
                    limb_ray(&p, space, 30.0, 0.2 * (top - b)),
                    f64::INFINITY,
                    true,
                ),
                (
                    space,
                    limb_ray(&p, space, 30.0, 0.5 * (top - b)),
                    f64::INFINITY,
                    true,
                ),
                (space, -DVec3::X, 2.0 * b - 0.3 * (top - b), true),
                (inside, dir_elev(90.0, 0.0), f64::INFINITY, true),
                (inside, dir_elev(10.0, 0.0), f64::INFINITY, true),
                (inside, dir_elev(-10.0, 0.0), f64::INFINITY, false),
            ];
            for (ro, rd, t_max, checked) in cases {
                let (_, t_march) = segment_radiance(&p, ro, rd, t_max, sun);
                let t_lut = segment_transmittance_lut(&p, ro, rd, t_max);
                let (g_march, g_lut) = (1.0 - lum(t_march), 1.0 - lum(t_lut));
                let in_range =
                    t_lut.is_finite() && t_lut.min_element() >= 0.0 && t_lut.max_element() <= 1.0;
                assert!(
                    in_range && (!checked || (g_lut - g_march).abs() <= 0.005),
                    "{name}: ro {ro}, rd {rd}: g LUT {g_lut} vs march {g_march} (T {t_lut} vs \
                     {t_march})"
                );
            }
        }
        // Rays that miss the shell, and a camera inside the body, give T = 1.
        let p = earth();
        let far = DVec3::new(3.0 * p.bottom_radius_m, 0.0, 0.0);
        assert_eq!(
            segment_transmittance_lut(&p, far, DVec3::Y, f64::INFINITY),
            Vec3::ONE
        );
        assert_eq!(
            segment_transmittance_lut(&p, DVec3::ZERO, DVec3::Y, f64::INFINITY),
            Vec3::ONE
        );
    }

    /// Ladder step 2's cost where it is largest: `earth-limb-dawn` (420 km up, looking at the
    /// limb, the Sun 1.8° below it), the low red layer (tangent heights 1–12 km) and the high
    /// blue arc (30–60 km), against a 1024-step reference, per channel relative to that channel.
    /// Measured (B9p fix round 1):
    /// - Low layer: 16–24 steps 0.3–3.9 % (R/B 0.6–3.5 % low), against 0.05–0.75 % at 32–48.
    /// - High arc: 16–24 steps 0.19–0.57 %, against 0.06–0.15 % at 32–48.
    ///
    /// The captured `limb_arc_low` R/B went from 33.7 to 31.4.
    #[test]
    fn view_march_at_the_dawn_terminator() {
        let p = earth();
        let b = p.bottom_radius_m;
        let luts = cpu_luts(&p);
        let alt = 420e3;
        let ro = DVec3::new(0.0, 0.0, b + alt);
        let dip_limb = (b / (b + alt)).acos().to_degrees();
        let sun = dir_elev(-(dip_limb + 1.8), 0.0);
        for (h, view_tol, lut_tol) in [
            (1e3, 0.05, 0.01),
            (3e3, 0.05, 0.01),
            (6e3, 0.05, 0.01),
            (9e3, 0.05, 0.01),
            (12e3, 0.05, 0.01),
            (30e3, 0.01, 0.002),
            (40e3, 0.01, 0.002),
            (50e3, 0.01, 0.002),
            (60e3, 0.01, 0.002),
        ] {
            let elev = -((b + h) / (b + alt)).acos().to_degrees();
            let rd = dir_elev(elev, 0.0);
            let (t0, t1) = segment_bounds(&p, ro, rd, f64::INFINITY).unwrap();
            let (o, len) = (ro + rd * t0, t1 - t0);
            let reference = march_steps(&luts, o, rd, 0.0, len, sun, 1024.0);
            let rel = |m: March| {
                ((m.inscatter - reference.inscatter).abs() / reference.inscatter).max_element()
            };
            let view = rel(march_steps(
                &luts,
                o,
                rd,
                0.0,
                len,
                sun,
                view_step_count(&p, len),
            ));
            let lut = rel(crate::lut::march(&luts, o, rd, 0.0, len, sun));
            assert!(
                view <= view_tol && lut <= lut_tol,
                "tangent height {h} m: 16–24 steps {view:.2e}, 32–48 steps {lut:.2e}"
            );
        }
    }

    /// The 32–48-step march (LUT passes, probes) is within 1 % of a 1024-step reference
    /// (inscatter, relative to its largest channel) and within 1e-3 in transmittance (1 % where
    /// T > 0.01), on vertical, horizontal and limb paths; the 16–24-step view march within 1 % in
    /// both.
    #[test]
    fn march_converges() {
        // The 16–24-step view march (ladder step 2, task B9p) measured 2.3e-3..6.5e-3 in
        // inscatter and 1.8e-3..3.2e-3 in T on these paths.
        const VIEW_MARCH_TOLERANCE: f64 = 0.01;
        let p = earth();
        let b = p.bottom_radius_m;
        let luts = cpu_luts(&p);
        for (ro, rd, sun) in [
            (DVec3::new(0.0, 0.0, b + 1.0), DVec3::Z, dir_elev(30.0, 0.0)),
            (
                DVec3::new(0.0, 0.0, b + 1.0),
                dir_elev(1.0, 90.0),
                dir_elev(10.0, 0.0),
            ),
            (
                DVec3::new(0.0, 0.0, b + 400e3),
                -DVec3::Z,
                dir_elev(40.0, 0.0),
            ),
            (
                DVec3::new(0.0, 0.0, b + 400e3),
                dir_elev(-19.5, 0.0),
                dir_elev(10.0, 0.0),
            ),
            (
                DVec3::new(0.0, 0.0, b + 400e3),
                dir_elev(-20.1, 0.0),
                dir_elev(10.0, 0.0),
            ),
        ] {
            let ro = camera_origin(&p, ro).unwrap();
            let (t0, t1) = segment_bounds(&p, ro, rd, f64::INFINITY).unwrap();
            let o = ro + rd * t0;
            let fast = crate::lut::march(&luts, o, rd, 0.0, t1 - t0, sun);
            let reference = crate::lut::march_steps(&luts, o, rd, 0.0, t1 - t0, sun, 1024.0);
            let l_err = (fast.inscatter - reference.inscatter).abs().max_element()
                / reference.inscatter.max_element();
            let t_err = (fast.transmittance - reference.transmittance).abs();
            let t_ok = (0..3).all(|c| {
                t_err[c] < 1e-3
                    && (reference.transmittance[c] < 0.01
                        || t_err[c] < 0.01 * reference.transmittance[c])
            });
            assert!(
                l_err < 0.01 && t_ok,
                "{rd}: inscatter {l_err:.2e}, T {t_err}"
            );
            // The view march (16–24 steps, ladder step 2): segment_radiance's own count.
            let view = march_steps(
                &luts,
                o,
                rd,
                0.0,
                t1 - t0,
                sun,
                view_step_count(&p, t1 - t0),
            );
            let lv_err = (view.inscatter - reference.inscatter).abs().max_element()
                / reference.inscatter.max_element();
            let tv_err = (view.transmittance - reference.transmittance)
                .abs()
                .max_element();
            println!("{rd:.3}: view march inscatter {lv_err:.2e}, T {tv_err:.2e}");
            assert!(
                lv_err < VIEW_MARCH_TOLERANCE && tv_err < VIEW_MARCH_TOLERANCE,
                "{rd}: view march inscatter {lv_err:.2e}, T {tv_err:.2e}"
            );
        }
    }

    /// B9's `earth-limb-dawn` check on the twin: 420 km up, looking at the limb toward a Sun
    /// 1.8° below it; tangent heights 30–60 km are blue (B/R ≥ 1.5), 0–12 km red (R/B ≥ 1.5).
    #[test]
    fn earth_limb_dawn_arc_colours() {
        let p = earth();
        let b = p.bottom_radius_m;
        let cam = DVec3::new(0.0, 0.0, b + 420e3);
        let dip = (b / (b + 420e3)).acos().to_degrees();
        let sun = dir_elev(-dip - 1.8, 0.0);
        let at = |h: f64| {
            let elev = -((b + h) / (b + 420e3)).acos().to_degrees();
            segment_radiance(&p, cam, dir_elev(elev, 0.0), f64::INFINITY, sun).0
        };
        for h in [30e3, 45e3, 60e3] {
            let l = at(h);
            assert!(l.z / l.x >= 1.5, "{} km: {l}", h / 1e3);
        }
        for h in [0.5e3, 6e3, 12e3] {
            let l = at(h);
            assert!(l.x / l.z >= 1.5, "{} km: {l}", h / 1e3);
        }
    }

    /// B9's `venus-crescent` check (phase 155°, 150 000 km): the haze above the cloud tops is
    /// brighter than 183 cd/m² (3 stops above AgX black at EV 16.7) on the lit limb and where the
    /// limb crosses the terminator.
    #[test]
    fn venus_haze_ring_visible() {
        let p = venus();
        let e = E_SUN_1AU_LX / (0.7233_f32 * 0.7233);
        let phase = 155.0_f64.to_radians();
        let sun = DVec3::new(phase.cos(), phase.sin(), 0.0);
        let cam = DVec3::new(150_000e3, 0.0, 0.0);
        for phi in [0.0, 45.0, 90.0] {
            let brightest = [2e3, 8e3, 15e3, 20e3, 30e3]
                .into_iter()
                .map(|h| {
                    lum(segment_radiance(&p, cam, limb_ray(&p, cam, phi, h), f64::INFINITY, sun).0)
                        * e
                })
                .fold(0.0, f32::max);
            assert!(brightest >= 183.0, "limb at {phi}°: {brightest} cd/m²");
        }
    }

    /// B9's `titan-backlit` check (phase 160°, 60 000 km): the main haze at the limb is orange
    /// (R/B ≥ 1.2) and brighter than 0.27 cd/m² (visible at EV 7.3).
    #[test]
    fn titan_backlit_haze_orange() {
        let p = titan();
        let e = E_SUN_1AU_LX / (9.58_f32 * 9.58);
        let phase = 160.0_f64.to_radians();
        let sun = DVec3::new(phase.cos(), phase.sin(), 0.0);
        let cam = DVec3::new(60_000e3, 0.0, 0.0);
        for phi in [0.0, 90.0] {
            for h in [50e3, 100e3, 150e3] {
                let (l, _) =
                    segment_radiance(&p, cam, limb_ray(&p, cam, phi, h), f64::INFINITY, sun);
                assert!(
                    l.x / l.z >= 1.2 && lum(l) * e >= 0.27,
                    "limb {phi}° at {} km: {l}",
                    h / 1e3
                );
            }
        }
    }

    /// B9's `mars-reference-angle` check (phase 45°, 8 210 km): the limb haze is red (R/B ≥ 1).
    #[test]
    fn mars_limb_haze_red() {
        let p = mars();
        let phase = 45.0_f64.to_radians();
        let sun = DVec3::new(phase.cos(), phase.sin(), 0.0);
        let cam = DVec3::new(8_210e3, 0.0, 0.0);
        for phi in [0.0, 90.0, 180.0] {
            for h in [5e3, 20e3, 40e3] {
                let (l, _) =
                    segment_radiance(&p, cam, limb_ray(&p, cam, phi, h), f64::INFINITY, sun);
                assert!(
                    l.x / l.z >= 1.0 && l.min_element() > 0.0,
                    "limb {phi}° at {} km: {l}",
                    h / 1e3
                );
            }
        }
    }

    /// Ruling W-B1: Earth's disk-integrated geometric albedo at 550 nm (≈ V) with the
    /// atmosphere over A8's Lambert composite 0.302 is p = 0.210 (600-sample value; 0.201
    /// without the atmosphere), 0.79 mag below the catalogue p_V 0.434: the atmosphere adds
    /// light (0.063 over a black ground) but also dims the ground. Pinned so B9's measurement has
    /// a reference. p = (2π/R²) ∫ (L/E) b db over impact parameters out to the top (a Lambert
    /// sphere gives 2A/3).
    #[test]
    fn earth_disk_albedo_w_b1() {
        let albedo = disk_albedo(&earth(), Vec3::splat(0.302), 200).y;
        assert!((0.20..=0.22).contains(&albedo), "p_550 = {albedo}");
    }

    /// Review I3: Titan's haze albedo in green and blue is fitted to the disk photometry (see
    /// `params::titan`). At phase 0 over a 0.15 ground the disk has p_V 0.22 (NSSDCA) and
    /// Harris 1961's p_B/p_V = 0.12/0.21. Red uses DISR's measured albedo, not fitted, and lands
    /// near Younkin 1974's 0.325 (rescaled to 2 575 km) and Harris's broad-band R of 0.32.
    #[test]
    fn titan_disk_matches_its_photometry() {
        let p = disk_albedo(&titan(), Vec3::splat(0.15), 300);
        assert!((p.y - 0.22).abs() < 0.005, "{p}");
        assert!((p.z - 0.22 * 0.12 / 0.21).abs() < 0.005, "{p}");
        assert!((0.30..0.37).contains(&p.x), "{p}");
    }

    /// The screen rectangle contains the projection of every point of the sphere and touches
    /// it on all four sides.
    #[test]
    fn sphere_screen_rect_bounds_the_sphere() {
        let (fov, aspect) = (60f64.to_radians(), 16.0 / 9.0);
        let sy = 1.0 / (0.5 * fov).tan();
        let sx = sy / aspect;
        for (c, radius) in [
            (DVec3::new(0.0, 0.0, -10.0), 3.0),
            (DVec3::new(4.0, -2.0, -12.0), 2.0),
            (DVec3::new(-30.0, 9.0, -20.0), 5.0),
            (DVec3::new(1e7, 3e6, -4e7), 6.4e6),
        ] {
            let [x0, y0, x1, y1] = sphere_screen_rect(c, radius, sx, sy).unwrap();
            let (mut lo, mut hi) = (glam::DVec2::splat(f64::MAX), glam::DVec2::splat(f64::MIN));
            for i in 0..200 {
                for j in 0..400 {
                    let (th, ph) = (
                        std::f64::consts::PI * (i as f64 + 0.5) / 200.0,
                        std::f64::consts::TAU * j as f64 / 400.0,
                    );
                    let q =
                        c + DVec3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()) * radius;
                    let ndc = glam::DVec2::new(sx * q.x / -q.z, sy * q.y / -q.z);
                    lo = lo.min(ndc);
                    hi = hi.max(ndc);
                }
            }
            let tol = 1e-9 * (x1 - x0).abs().max(1.0);
            assert!(lo.x >= x0 - tol && lo.y >= y0 - tol && hi.x <= x1 + tol && hi.y <= y1 + tol);
            let slack = 1e-3 * (x1 - x0).max(y1 - y0);
            assert!(
                lo.x - x0 < slack && lo.y - y0 < slack && x1 - hi.x < slack && y1 - hi.y < slack
            );
        }
        assert!(sphere_screen_rect(DVec3::new(0.0, 0.0, -2.0), 3.0, sx, sy).is_none());
        assert!(sphere_screen_rect(DVec3::new(0.0, 0.0, 5.0), 3.0, sx, sy).is_none());
    }

    /// The draw's frame maps Earth's ground to the bottom radius at the pole and the equator,
    /// the camera inside the shell gets the fullscreen triangle, and a camera inside the body, a
    /// shell behind the camera or bad input draws nothing.
    #[test]
    fn draw_uniform_follows_the_ellipsoid() {
        let p = earth();
        let orient = DQuat::from_rotation_x(0.4) * DQuat::from_rotation_z(1.1);
        let m = atmosphere_frame(399, orient);
        let abc = body_shape_m(399).unwrap();
        for fixed in [
            DVec3::new(0.0, 0.0, abc.z),
            DVec3::new(abc.x, 0.0, 0.0),
            DVec3::new(0.0, -abc.y, 0.0),
        ] {
            let r = (m * (orient * fixed)).length();
            assert!((r - p.bottom_radius_m).abs() < 2.0, "{fixed}: {r}");
        }
        let proj = uni_gfx::reversed_z_infinite_proj(60f32.to_radians(), 16.0 / 9.0, 0.1);
        let draw = |rel: DVec3| AtmoDraw {
            naif: 399,
            rel_pos_m: rel,
            orient,
            sun_rel_m: DVec3::new(1.5e11, 0.0, 0.0),
            sun_illuminance_lux: 1.27e5,
            exposure_mix: 1.0,
        };
        // Body 5 radii ahead (−Z): a quad inside the screen, origin height ≈ 4 radii.
        let u = draw_uniform(
            &p,
            &draw(DVec3::new(0.0, 0.0, -5.0 * p.bottom_radius_m)),
            Mat4::IDENTITY,
            proj,
        )
        .unwrap();
        assert_eq!(u.misc[1], 0.0);
        assert!(
            u.rect[0] > -1.0 && u.rect[2] < 1.0 && u.rect[0] < u.rect[2] && u.rect[1] < u.rect[3]
        );
        assert!((u.origin[3] / p.bottom_radius_m as f32 - 4.0).abs() < 0.01);
        assert_eq!(u.sun[3], 1.27e5);
        // Inside the shell, 10 km up: fullscreen.
        let inside = draw(DVec3::new(0.0, -(p.bottom_radius_m + 10e3), 0.0));
        assert_eq!(
            draw_uniform(&p, &inside, Mat4::IDENTITY, proj)
                .unwrap()
                .misc[1],
            1.0
        );
        // Inside the body; behind the camera; off screen; non-finite.
        assert!(
            draw_uniform(&p, &draw(DVec3::new(0.0, -1e6, 0.0)), Mat4::IDENTITY, proj).is_none()
        );
        assert!(
            draw_uniform(
                &p,
                &draw(DVec3::new(0.0, 0.0, 5.0 * p.bottom_radius_m)),
                Mat4::IDENTITY,
                proj
            )
            .is_none()
        );
        assert!(
            draw_uniform(
                &p,
                &draw(DVec3::new(
                    9.0 * p.bottom_radius_m,
                    0.0,
                    -2.0 * p.bottom_radius_m
                )),
                Mat4::IDENTITY,
                proj
            )
            .is_none()
        );
        assert!(draw_uniform(&p, &draw(DVec3::NAN), Mat4::IDENTITY, proj).is_none());
    }

    /// Lead ruling B6 I1 (task B10): the draw uploads its body's exposure weight, sanitised and
    /// snapped ([`atmo_exposure_mix`]): NaN → 1 (scene exposure), clamped to [0, 1], and 1 when
    /// 1 − w < 2⁻¹⁴, so the star share 1 − w it stores in `lens_mask.a` ([`atmo_star_share`]) is
    /// 0 or an f16 normal (≥ 2⁻¹⁴), never a subnormal Metal could flush. A w already snapped by
    /// uni-planet passes through unchanged.
    #[test]
    fn exposure_mix_snapped_into_the_draw() {
        assert_eq!(STAR_SHARE_MIN, 2f32.powi(-14));
        let p = earth();
        let proj = uni_gfx::reversed_z_infinite_proj(60f32.to_radians(), 16.0 / 9.0, 0.1);
        let draw = |w: f32| AtmoDraw {
            naif: 399,
            rel_pos_m: DVec3::new(0.0, 0.0, -5.0 * p.bottom_radius_m),
            orient: DQuat::IDENTITY,
            sun_rel_m: DVec3::new(1.5e11, 0.0, 0.0),
            sun_illuminance_lux: 1.27e5,
            exposure_mix: w,
        };
        let snap_edge = 1.0 - STAR_SHARE_MIN;
        for (w, want) in [
            (0.0, 0.0),
            (0.3, 0.3),
            (0.99, 0.99),
            (1.0 - 1.0e-4, 1.0 - 1.0e-4),
            (snap_edge, snap_edge),
            (1.0 - 3.0e-5, 1.0),
            (1.0, 1.0),
            (f32::NAN, 1.0),
            (-0.5, 0.0),
            (2.0, 1.0),
        ] {
            let u = draw_uniform(&p, &draw(w), Mat4::IDENTITY, proj).unwrap();
            assert_eq!(u.misc[0], want, "w {w}");
            assert_eq!(atmo_exposure_mix(want), want, "idempotent at {w}");
            let s = atmo_star_share(u.misc[0]);
            assert!(
                s == 0.0 || (STAR_SHARE_MIN..=1.0).contains(&s),
                "w {w}: share {s}"
            );
            assert_eq!(s, 1.0 - want);
        }
        // A sweep through the top of the band: the share is never in (0, 2⁻¹⁴).
        for i in 0..=100_000 {
            let w = 1.0 - 1.0e-3 * i as f32 / 1.0e5;
            let s = atmo_star_share(atmo_exposure_mix(w));
            assert!(s == 0.0 || s >= STAR_SHARE_MIN, "w {w}: share {s}");
        }
    }

    /// The `lens_mask` target of both variants (lead ruling B6 I1): green (1 − luminance(T)) and
    /// alpha (the star share) written, red and blue left to the lens pass, both blended with
    /// `Max` (factors One), Rgba16Float.
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

    /// Both lens-mask writers (the dual variant's mask pipeline and the grey variant) store
    /// `atmo_star_share()` in alpha, which is 1 − the same `misc.x` that `atmo_light` exposes the
    /// in-scatter with (lead ruling B6 I1), through the log-space `pre_mix` of uni-gfx (task B11),
    /// as the body pass does, and no linear mix of the two pre-exposures is left.
    #[test]
    fn atmosphere_shaders_store_the_star_share() {
        use uni_gfx::{ShaderLib, validate_wgsl};
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let common = lib.compose(crate::ATMOSPHERE_COMMON_WGSL).unwrap();
        let m = validate_wgsl(&common).unwrap();
        let arity = m
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("atmo_star_share"))
            .map(|(_, f)| f.arguments.len());
        assert_eq!(arity, Some(0));
        assert_eq!(common.matches("return 1.0 - atmo_draw.misc.x;").count(), 1);
        assert_eq!(
            common
                .matches("let pre = pre_mix(star_pre(), scene_pre(), atmo_draw.misc.x);")
                .count(),
            1
        );
        for arg in ["mix(star_pre()", "mix(scene_pre()"] {
            let log = format!("pre_{arg}");
            assert_eq!(
                common.matches(arg).count(),
                common.matches(log.as_str()).count(),
                "a linear {arg}"
            );
        }
        for (path, line) in [
            (
                crate::ATMOSPHERE_MASK_WGSL,
                "return vec4<f32>(0.0, clamp(1.0 - luminance(t), 0.0, 1.0), 0.0, atmo_star_share());",
            ),
            (
                crate::ATMOSPHERE_GREY_WGSL,
                "vec4<f32>(0.0, alpha, 0.0, atmo_star_share()),",
            ),
        ] {
            let src = lib.compose(path).unwrap();
            assert_eq!(src.matches(line).count(), 1, "{path}");
        }
    }

    /// Review I1: [`AtmoDraw::camera_in_daylight`] and [`AtmoDraw::scintillation_strength`]
    /// measure height from the reference ellipsoid, as the shell does. At sea level on the
    /// equator (two longitudes) and at the pole of Earth and Mars the strength is the ground
    /// value (with a sphere, Earth's equator gave exp(−7.1/8) ≈ 0.41 of it); with the Sun
    /// overhead it is daylight 1 km below the shell's top at every latitude and not 1 km above
    /// it; with the Sun 5° below the local horizon it is not daylight.
    #[test]
    fn daylight_and_scintillation_follow_the_ellipsoid() {
        let orient = DQuat::from_rotation_x(0.4) * DQuat::from_rotation_z(1.1);
        for (naif, p) in [(399, earth()), (499, mars())] {
            let abc = body_shape_m(naif).unwrap();
            let ground = crate::scintillation_strength(&p, DVec3::new(0.0, 0.0, p.bottom_radius_m));
            let thick = p.top_radius_m - p.bottom_radius_m;
            for (up, along) in [
                (DVec3::X, DVec3::Y),
                (DVec3::Y, DVec3::Z),
                (DVec3::Z, DVec3::X),
            ] {
                // Body-fixed: the ground point on this axis, `h` above it, the Sun 1 au away
                // at elevation `elev` toward `along`.
                let at = |h: f64, elev_deg: f64| {
                    let cam = up * abc + up * h;
                    let e = elev_deg.to_radians();
                    let sun = cam + (up * e.sin() + along * e.cos()) * 1.496e11;
                    AtmoDraw {
                        naif,
                        rel_pos_m: orient * -cam,
                        orient,
                        sun_rel_m: orient * sun,
                        sun_illuminance_lux: 1.0,
                        exposure_mix: 1.0,
                    }
                };
                let s = at(1.0, 90.0).scintillation_strength(&p);
                assert!(
                    (s - ground).abs() <= 3e-3 * ground,
                    "NAIF {naif} {up}: {s} against the ground value {ground}"
                );
                assert!(at(1.0, 30.0).camera_in_daylight(&p), "NAIF {naif} {up}");
                assert!(!at(1.0, -5.0).camera_in_daylight(&p), "NAIF {naif} {up}");
                assert!(
                    at(thick - 1e3, 90.0).camera_in_daylight(&p),
                    "NAIF {naif} {up}"
                );
                assert!(
                    !at(thick + 1e3, 90.0).camera_in_daylight(&p),
                    "NAIF {naif} {up}"
                );
                assert_eq!(at(thick + 1e3, 90.0).scintillation_strength(&p), 0.0);
            }
        }
    }

    /// Geometric albedo (phase 0) per channel over a Lambert ground, with `n` impact-parameter
    /// rings: [`disk_reflectance`] at phase 0.
    fn disk_albedo(p: &AtmosphereParams, ground: Vec3, n: usize) -> Vec3 {
        disk_reflectance(p, ground, 0.0, n, 1)
    }

    /// Spherical albedo A = p·q = 2∫₀^π p·Φ(α)·sin α dα (Russell 1916, ApJ 43, 173) per channel
    /// over a Lambert ground, with `phases` midpoints in α and [`disk_reflectance`]'s
    /// `rings` × `sectors` grid at each.
    fn spherical_albedo(
        p: &AtmosphereParams,
        ground: Vec3,
        phases: usize,
        rings: usize,
        sectors: usize,
    ) -> DVec3 {
        let d_alpha = std::f64::consts::PI / phases as f64;
        (0..phases)
            .map(|i| {
                let a = d_alpha * (i as f64 + 0.5);
                disk_reflectance(p, ground, a, rings, sectors).as_dvec3()
                    * (2.0 * a.sin() * d_alpha)
            })
            .sum()
    }

    /// The same preset with no air at all: every coefficient 0, so a bare Lambert sphere.
    fn airless(p: AtmosphereParams) -> AtmosphereParams {
        AtmosphereParams {
            rayleigh_scattering: Vec3::ZERO,
            mie_scattering: Vec3::ZERO,
            mie_extinction: Vec3::ZERO,
            layer2_scattering: Vec3::ZERO,
            layer2_extinction: Vec3::ZERO,
            ..p
        }
    }

    /// Review I4: `disk_reflectance` against the closed form for a Lambert sphere of albedo ρ
    /// (no air): p·Φ(α) = (2ρ/3)·(sin α + (π − α)·cos α)/π (Russell 1916, "On the albedo of the
    /// planets and their satellites", ApJ 43, 173). Phase 0 takes the rotationally symmetric
    /// path; 60° and 120° the azimuth grid, with the terminator crossing it.
    #[test]
    fn disk_reflectance_follows_lambert_phase_law() {
        use std::f64::consts::PI;
        let (p, rho) = (airless(earth()), 0.3);
        for deg in [0.0f64, 60.0, 120.0] {
            let a = deg.to_radians();
            let want = 2.0 * rho / 3.0 * (a.sin() + (PI - a) * a.cos()) / PI;
            let got = disk_reflectance(&p, Vec3::splat(rho as f32), a, 32, 16);
            for c in 0..3 {
                assert!(
                    (got[c] as f64 / want - 1.0).abs() < 0.01,
                    "{deg}°: {got} against {want:.5}"
                );
            }
        }
    }

    /// Review I4 (ruling W-B1): physical air cannot bring Earth's disk up to the catalogue p_V
    /// 0.434 (NASA NSSDCA Earth fact sheet, baked in `data/baked/body_facts.ron`).
    ///
    /// Terms: p is the geometric albedo and A = p·q the spherical albedo, with q the phase
    /// integral (Russell 1916). The model's values are per wavelength, here 550 nm, its green
    /// channel, near V. A Bond albedo is A integrated over the solar spectrum.
    /// - Over a uniform Lambert ground of albedo ρ the model's reflectance is affine in ρ: the
    ///   in-scatter and the LUTs do not depend on the map's ρ. So p₅₅₀ reaches 0.434 at one ρ*,
    ///   ≈ 0.76.
    /// - Integrated over phase at that ρ*, the model's A₅₅₀ is 0.609 (q 1.40). No catalogue
    ///   gives Earth's spherical albedo at 550 nm. The check therefore uses a deliberately
    ///   generous visible-band ceiling, A_VIS_MAX = 0.35, above every measured value to hand:
    ///   - NSSDCA's Bond albedo 0.294 (bolometric);
    ///   - 0.297 ± 0.005 from optical earthshine photometry (Goode et al. 2001, GRL 28(9),
    ///     1671, doi:10.1029/2000GL012580).
    ///
    ///   Comparing 550 nm with band-integrated values is approximate, but the gap is not: A₅₅₀(ρ*)
    ///   is about twice the measured values, and the test requires more than 1.5 × the ceiling
    ///   (0.525).
    /// - So a Lambert Earth can match p_V or Earth's spherical albedo, not both. The real
    ///   Earth's phase integral is q = A/p_V ≤ 0.35/0.434 = 0.81, against ≈ 1.4–1.5 for this
    ///   model (a bare Lambert sphere has 3/2). The gap is in the phase law of the clouds and
    ///   ground (backscatter toward phase 0), not in the air.
    /// - Nor does more air help. With ×2 and ×3 the whole column (Rayleigh, aerosol, ozone),
    ///   p₅₅₀ over A8's 0.302 composite stays ≤ 0.22, because the thicker air dims the ground as
    ///   much as it adds.
    ///
    /// Phase quadrature: 8 midpoints of 22.5°, 16 rings × 8 sectors. On the same grid a bare
    /// Lambert sphere of albedo ρ* gives A = 1.014·ρ* (closed form ρ*; `report_numbers`).
    #[test]
    fn w_b1_p_v_needs_a_non_lambert_earth() {
        const P_V: f64 = 0.434;
        const A_VIS_MAX: f64 = 0.35;
        let e = earth();
        let p0 = disk_albedo(&e, Vec3::ZERO, 200).y as f64;
        let p1 = disk_albedo(&e, Vec3::ONE, 200).y as f64;
        let rho_needed = (P_V - p0) / (p1 - p0);
        let a_550 = spherical_albedo(&e, Vec3::splat(rho_needed as f32), 8, 16, 8).y;
        assert!(
            a_550 > 1.5 * A_VIS_MAX,
            "Lambert ground reaching p_V: ρ* {rho_needed:.3}, A₅₅₀ {a_550:.3}, q {:.2}",
            a_550 / P_V
        );
        for k in [2.0f32, 3.0] {
            let thick = AtmosphereParams {
                rayleigh_scattering: e.rayleigh_scattering * k,
                mie_scattering: e.mie_scattering * k,
                mie_extinction: e.mie_extinction * k,
                layer2_extinction: e.layer2_extinction * k,
                ..earth()
            };
            let p = disk_albedo(&thick, Vec3::splat(0.302), 200).y;
            assert!(p < 0.22, "air ×{k}: p₅₅₀ {p}");
        }
    }

    /// Prints every number the B3 report quotes (`cargo test -p uni-atmo --lib report_numbers
    /// -- --ignored --nocapture`; about 20 s unoptimised).
    #[test]
    #[ignore = "print-only: the numbers the B3 task report quotes"]
    fn report_numbers() {
        let e = earth();
        let b = e.bottom_radius_m;
        println!(
            "Earth T(ground, zenith) = {}",
            crate::transmittance(&e, b, 1.0)
        );
        let t = crate::transmittance(&e, b, 0.02);
        println!("Earth T(ground, mu_s 0.02) = {t}, R/B {:.1}", t.x / t.z);
        let mu = 60f64.to_radians().sin();
        let (l, _) = segment_radiance(
            &e,
            DVec3::new(0.0, 0.0, b + 1.0),
            DVec3::Z,
            f64::INFINITY,
            dir_elev(60.0, 0.0),
        );
        let sky = crate::sky_irradiance(&e, b, mu);
        let direct = crate::transmittance(&e, b, mu) * mu as f32;
        println!(
            "Earth Sun 60°: zenith L/E {l} = {:.0} cd/m² at {E_SUN_1AU_LX} lx; sky E/E0 {sky}; diffuse fraction (G) {:.3}",
            lum(l) * E_SUN_1AU_LX,
            sky.y / (sky.y + direct.y)
        );
        let m = mars();
        let ro = DVec3::new(0.0, 0.0, m.bottom_radius_m + 1.0);
        for above in [1.0, 2.0, 3.0, 4.0, 6.0, 10.0] {
            let (l, _) = segment_radiance(
                &m,
                ro,
                dir_elev(2.0 + above, 0.0),
                f64::INFINITY,
                dir_elev(2.0, 0.0),
            );
            println!("Mars {above}° above a Sun 2° up: B/R {:.3}", l.z / l.x);
        }
        for sun in [30.0, 45.0, 60.0] {
            let (l, _) = segment_radiance(&m, ro, DVec3::Z, f64::INFINITY, dir_elev(sun, 0.0));
            println!("Mars zenith, Sun {sun}° up: R/B {:.3}", l.x / l.z);
        }
        let cam = DVec3::new(0.0, 0.0, b + 420e3);
        let dip = (b / (b + 420e3)).acos().to_degrees();
        for h in [60e3, 45e3, 30e3, 20e3, 12e3, 6e3, 0.5e3] {
            let elev = -((b + h) / (b + 420e3)).acos().to_degrees();
            let (l, _) = segment_radiance(
                &e,
                cam,
                dir_elev(elev, 0.0),
                f64::INFINITY,
                dir_elev(-dip - 1.8, 0.0),
            );
            println!(
                "earth-limb-dawn tangent {:>4.1} km: B/R {:.2}, R/B {:.2}, {:.0} cd/m²",
                h / 1e3,
                l.z / l.x,
                l.x / l.z,
                lum(l) * E_SUN_1AU_LX
            );
        }
        let v = venus();
        let ev = E_SUN_1AU_LX / (0.7233 * 0.7233);
        let phase = 155f64.to_radians();
        let vc = DVec3::new(150_000e3, 0.0, 0.0);
        for phi in [0.0, 45.0, 90.0, 100.0, 110.0] {
            for h in [2e3, 8e3, 15e3, 20e3, 30e3] {
                let (l, _) = segment_radiance(
                    &v,
                    vc,
                    limb_ray(&v, vc, phi, h),
                    f64::INFINITY,
                    DVec3::new(phase.cos(), phase.sin(), 0.0),
                );
                println!(
                    "venus-crescent limb {phi:>5.1}° tangent {:>4.1} km: {:.3e} cd/m² (needs ≥ 183)",
                    h / 1e3,
                    lum(l) * ev
                );
            }
        }
        let ti = titan();
        let et = E_SUN_1AU_LX / (9.58 * 9.58);
        let phase = 160f64.to_radians();
        let tc = DVec3::new(60_000e3, 0.0, 0.0);
        for phi in [0.0, 90.0, 180.0] {
            for h in [50e3, 100e3, 150e3, 300e3, 500e3] {
                let (l, _) = segment_radiance(
                    &ti,
                    tc,
                    limb_ray(&ti, tc, phi, h),
                    f64::INFINITY,
                    DVec3::new(phase.cos(), phase.sin(), 0.0),
                );
                println!(
                    "titan-backlit limb {phi:>5.1}° tangent {:>5.1} km: {:.3} cd/m² (needs ≥ 0.27), R/B {:.2}",
                    h / 1e3,
                    lum(l) * et,
                    l.x / l.z
                );
            }
        }
        let phase = 45f64.to_radians();
        let mc = DVec3::new(8_210e3, 0.0, 0.0);
        for phi in [0.0, 90.0, 180.0] {
            let (l, _) = segment_radiance(
                &m,
                mc,
                limb_ray(&m, mc, phi, 20e3),
                f64::INFINITY,
                DVec3::new(phase.cos(), phase.sin(), 0.0),
            );
            println!(
                "mars-reference-angle limb {phi:>5.1}° tangent 20 km: R/B {:.2}",
                l.x / l.z
            );
        }
        for (name, p, ground) in [
            ("Earth, ground 0.302", earth(), Vec3::splat(0.302)),
            ("Earth, black ground", earth(), Vec3::ZERO),
            (
                "Earth, no atmosphere",
                AtmosphereParams {
                    rayleigh_scattering: Vec3::ZERO,
                    mie_scattering: Vec3::ZERO,
                    mie_extinction: Vec3::ZERO,
                    layer2_extinction: Vec3::ZERO,
                    ..earth()
                },
                Vec3::splat(0.302),
            ),
            ("Venus, cloud deck 0.76", venus(), Vec3::splat(0.76)),
            (
                "Mars, ground (0.25, 0.15, 0.08)",
                mars(),
                Vec3::new(0.25, 0.15, 0.08),
            ),
            // A Lambert ground calibrated to the catalogue p_V 0.17 without an atmosphere
            // (2A/3 = 0.17), as A8 calibrates Mars's map: what the dust does to that disk.
            (
                "Mars, ground 0.255 (p_V-calibrated)",
                mars(),
                Vec3::splat(0.255),
            ),
            ("Titan, ground 0.15", titan(), Vec3::splat(0.15)),
        ] {
            println!("geometric albedo, {name}: {}", disk_albedo(&p, ground, 600));
        }
        for (name, p) in [
            ("Earth", earth()),
            ("Mars", mars()),
            ("Venus", venus()),
            ("Titan", titan()),
        ] {
            let s = |h: f64| {
                crate::scintillation_strength(&p, DVec3::new(0.0, 0.0, p.bottom_radius_m + h))
            };
            println!(
                "scintillation {name}: ground {:.3}, 5 km {:.3}, 30 km {:.3}",
                s(0.0),
                s(5e3),
                s(30e3)
            );
        }
        // Review I1: sea level on the equator and at the pole, through AtmoDraw (ellipsoid)
        // and, for comparison, the plain root-axes vector (sphere).
        for (naif, p) in [(399, earth()), (499, mars())] {
            let abc = body_shape_m(naif).unwrap();
            for (where_, cam) in [
                ("equator", DVec3::new(abc.x, 0.0, 0.0)),
                ("pole", DVec3::new(0.0, 0.0, abc.z)),
            ] {
                let draw = AtmoDraw {
                    naif,
                    rel_pos_m: -cam,
                    orient: DQuat::IDENTITY,
                    sun_rel_m: DVec3::new(0.0, 0.0, 1.5e11),
                    sun_illuminance_lux: 1.0,
                    exposure_mix: 1.0,
                };
                println!(
                    "scintillation NAIF {naif} sea level, {where_}: ellipsoid {:.3}, sphere {:.3}",
                    draw.scintillation_strength(&p),
                    crate::scintillation_strength(&p, cam)
                );
            }
        }
        // Review I2 (ruled: r_m is the bottom radius): ground lighting at the bottom against the
        // equatorial ground's distance from the centre, bottom + (a − bottom), which the
        // contract excludes.
        for (name, p, a) in [
            ("Earth", earth(), body_shape_m(399).unwrap().x),
            ("Mars", mars(), body_shape_m(499).unwrap().x),
        ] {
            let b = p.bottom_radius_m;
            for mu_s in [1.0, 0.5, 0.1, 0.05, 0.02] {
                let (t0, e0) = crate::ground::ground_lighting(Some(&p), b, mu_s);
                let (t1, e1) = crate::ground::ground_lighting(Some(&p), a, mu_s);
                println!(
                    "ground lighting {name} equator mu_s {mu_s}: sun T {t0} (r_m bottom) vs {t1} \
                     (|p| = bottom + {:.1} km, not the contract); sky E/E0 {e0} vs {e1}",
                    (a - b) / 1e3
                );
            }
        }
        // Review I4 (W-B1): what physical air can and cannot do for Earth's p₅₅₀.
        let lambert =
            |a: f64| (a.sin() + (std::f64::consts::PI - a) * a.cos()) / std::f64::consts::PI;
        for deg in [0.0f64, 60.0, 120.0, 150.0] {
            let a = deg.to_radians();
            let got = disk_reflectance(&airless(earth()), Vec3::splat(0.3), a, 32, 16).y as f64;
            println!(
                "W-B1 quadrature: bare Lambert 0.3 at {deg}°: {got:.5} against Russell {:.5} ({:+.2} %)",
                0.2 * lambert(a),
                (got / (0.2 * lambert(a)) - 1.0) * 100.0
            );
        }
        let (p0, p1) = (
            disk_albedo(&e, Vec3::ZERO, 600),
            disk_albedo(&e, Vec3::ONE, 600),
        );
        println!(
            "W-B1 Earth p over ground 0 / 1: {p0} / {p1}; Lambert ground reaching p_V 0.434: {}",
            (Vec3::splat(0.434) - p0) / (p1 - p0)
        );
        let lut_ground = AtmosphereParams {
            ground_albedo: Vec3::splat(0.302),
            ..earth()
        };
        println!(
            "W-B1 Earth p over 0.302 with the LUT's ground albedo 0.302 (preset 0.1): {}",
            disk_albedo(&lut_ground, Vec3::splat(0.302), 600)
        );
        for k in [2.0f32, 3.0, 5.0, 10.0] {
            let column = AtmosphereParams {
                rayleigh_scattering: e.rayleigh_scattering * k,
                mie_scattering: e.mie_scattering * k,
                mie_extinction: e.mie_extinction * k,
                layer2_extinction: e.layer2_extinction * k,
                ..earth()
            };
            let rayleigh = AtmosphereParams {
                rayleigh_scattering: e.rayleigh_scattering * k,
                ..earth()
            };
            println!(
                "W-B1 Earth p over 0.302, air column ×{k}: {}; Rayleigh only ×{k}: {}",
                disk_albedo(&column, Vec3::splat(0.302), 300),
                disk_albedo(&rayleigh, Vec3::splat(0.302), 300)
            );
        }
        // Spherical albedo A = p·q per channel (550 nm in the middle; not a Bond albedo, which is
        // band-integrated), 12 midpoints of 15°, 32 rings × 12 sectors.
        for (name, p) in [("Earth", earth()), ("bare Lambert", airless(earth()))] {
            let a = spherical_albedo(&p, Vec3::splat(0.302), 12, 32, 12);
            let p0 = disk_albedo(&p, Vec3::splat(0.302), 600).as_dvec3();
            println!(
                "W-B1 {name} over 0.302: spherical albedo {a:.4}, p {p0:.4}, phase integral q {:.3}",
                a.y / p0.y
            );
        }
        // `w_b1_p_v_needs_a_non_lambert_earth`'s grid (8 phases, 16 × 8): Earth at the ρ* that
        // reaches p_V, and the bare Lambert control (closed form A = ρ).
        let (q0, q1) = (
            disk_albedo(&e, Vec3::ZERO, 200).y as f64,
            disk_albedo(&e, Vec3::ONE, 200).y as f64,
        );
        let rho_star = (0.434 - q0) / (q1 - q0);
        let a_star = spherical_albedo(&e, Vec3::splat(rho_star as f32), 8, 16, 8);
        let a_bare = spherical_albedo(&airless(earth()), Vec3::splat(rho_star as f32), 8, 16, 8);
        println!(
            "W-B1 test grid: Earth ρ* {rho_star:.4}: A {a_star:.4} (q₅₅₀ {:.3}); bare Lambert at ρ*: \
             A {a_bare:.4} against {rho_star:.4} ({:+.2} %)",
            a_star.y / 0.434,
            (a_bare.y / rho_star - 1.0) * 100.0
        );
    }
}
