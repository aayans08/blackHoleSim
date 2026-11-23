//! The per-atmosphere LUTs and their CPU twins (Task B3).
//!
//! Hillaire 2020 ("A Scalable and Production Ready Sky and Atmosphere Rendering Technique", CGF
//! 39(4); sample code MIT, used as a reference for the algorithm, see `ATTRIBUTION.md`) with
//! Bruneton 2017's (r, μ) parameterisation ("Precomputed Atmospheric Scattering: a New
//! Implementation", `functions.glsl`): transmittance 256 × 64, multiple scattering 32 × 32,
//! ground irradiance 64 × 16, all Rgba16Float, computed once per atmosphere at start-up by the
//! compute passes in `shaders/*_lut.wgsl`. Radii are metres from the body's centre; μ is the
//! cosine of the zenith angle of the view (or of the Sun, μ_s).
//!
//! The CPU twins compute the same quantities in f64 with the same quadratures. The view march
//! and the multiple-scattering and irradiance integrals sample CPU copies of the LUTs
//! ([`Table`], bilinear and clamped exactly like the GPU's linear clamp-to-edge sampler), built
//! once per parameter set and cached, so a twin differs from its shader only by f16 storage and
//! f32 arithmetic.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use anyhow::Context as _;
use glam::{DVec2, DVec3, Vec3};
use uni_gfx::{GpuContext, ShaderLib};
use wgpu::util::DeviceExt as _;

use crate::ground::{GroundLighting, LUT_FORMAT};
use crate::params::AtmosphereParams;

/// Transmittance LUT size: 256 (μ) × 64 (r) (Hillaire 2020 §4, Bruneton 2017).
pub const TRANSMITTANCE_SIZE: [u32; 2] = [256, 64];
/// Multiple-scattering LUT size: 32 (μ_s) × 32 (r) (Hillaire 2020 §5.5).
pub const MULTISCATTER_SIZE: [u32; 2] = [32, 32];
/// Ground-irradiance LUT size: 64 (μ_s) × 16 (r) (Bruneton 2017's irradiance texture).
pub const IRRADIANCE_SIZE: [u32; 2] = [64, 16];
/// Trapezoid intervals of the optical-depth integral (Bruneton 2017,
/// `ComputeOpticalLengthToTopAtmosphereBoundary`, SAMPLE_COUNT 500).
pub const TRANSMITTANCE_STEPS: u32 = 500;
/// Directions per multiple-scattering texel: 8 × 8 on the sphere (Hillaire 2020's sample,
/// SQRTSAMPLECOUNT 8).
pub const MULTISCATTER_SQRT_DIRS: u32 = 8;
/// March steps per multiple-scattering direction (Hillaire 2020's sample: 20).
pub const MULTISCATTER_STEPS: u32 = 20;
/// Cosine-weighted rings × azimuths of the sky-irradiance hemisphere integral.
pub const IRRADIANCE_DIRS: [u32; 2] = [16, 16];
/// March steps of the irradiance LUT and the probes: 32, rising smoothly to 48 on long limb
/// paths (B3 brief; the per-pixel view march used these until task B9p).
pub const MARCH_STEPS: f64 = 32.0;
/// See [`MARCH_STEPS`].
pub const MARCH_STEPS_LIMB: f64 = 48.0;
/// Per-pixel view-march steps: 16, rising smoothly to 24 on long limb paths (plan fallback
/// ladder, atmosphere step 2, task B9p: the pass measured 8.2 ms at `earth-map-closeup` with
/// 32–48).
///
/// **Quality cost, against a 1024-step reference.**
/// - Most paths: under 1 % (`march_converges`), as at 32–48.
/// - Where step 2 costs most, the low red layer of a dawn limb (`earth-limb-dawn`, tangent
///   heights 1–12 km; `view_march_at_the_dawn_terminator`):
///   - inscatter off by up to 3.9 % per channel, against 0.75 % at 32–48;
///   - R/B up to 3.5 % low;
///   - in the capture, `limb_arc_low` R/B 33.7 → 31.4 (the check needs ≥ 1.5), and up to 11
///     code values on some 5 500 channels of that layer.
/// - The high blue arc: 0.6 % (against 0.15 %).
pub const VIEW_MARCH_STEPS: f64 = 16.0;
/// See [`VIEW_MARCH_STEPS`].
pub const VIEW_MARCH_STEPS_LIMB: f64 = 24.0;
/// LUT texels for the ground row are evaluated this far above the bottom radius [m], so the
/// integrals never start exactly on the sphere (Hillaire's PLANET_RADIUS_OFFSET plays this
/// role).
pub const LUT_GROUND_OFFSET_M: f64 = 1.0;
/// Upper bound of Hillaire's f_ms, so the series 1/(1 − f_ms) stays finite in optically thick
/// atmospheres (Titan).
pub const MAX_F_MS: f64 = 0.99;

// ---------------------------------------------------------------------------------------------
// The medium

/// Scattering and extinction coefficients [1/m] at one point.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Medium {
    pub rayleigh: DVec3,
    pub mie: DVec3,
    pub layer2: DVec3,
    pub extinction: DVec3,
}

impl Medium {
    pub fn scattering(&self) -> DVec3 {
        self.rayleigh + self.mie + self.layer2
    }
}

fn dv(v: Vec3) -> DVec3 {
    v.as_dvec3()
}

/// The medium at altitude `h_m` above the bottom radius (clamped to ≥ 0): Rayleigh and Mie
/// densities exp(−h/H), layer 2 a tent of full width w around its centre (Hillaire 2020 §4,
/// Bruneton 2017 `DensityProfileLayer`).
pub(crate) fn medium(p: &AtmosphereParams, h_m: f64) -> Medium {
    let h = h_m.max(0.0);
    let d_r = (-h / f64::from(p.rayleigh_scale_height_m)).exp();
    let d_m = (-h / f64::from(p.mie_scale_height_m)).exp();
    let half = 0.5 * f64::from(p.layer2_width_m);
    let d_2 = if half > 0.0 {
        (1.0 - (h - f64::from(p.layer2_center_m)).abs() / half).max(0.0)
    } else {
        0.0
    };
    let rayleigh = dv(p.rayleigh_scattering) * d_r;
    let mie = dv(p.mie_scattering) * d_m;
    let layer2 = dv(p.layer2_scattering) * d_2;
    let extinction = rayleigh + dv(p.mie_extinction) * d_m + dv(p.layer2_extinction) * d_2;
    Medium {
        rayleigh,
        mie,
        layer2,
        extinction,
    }
}

/// Extinction coefficient [1/m] at altitude `h_m` above the bottom radius.
pub fn extinction_at(params: &AtmosphereParams, h_m: f64) -> DVec3 {
    medium(params, h_m).extinction
}

/// ∫₀^dt e^(−σ s) ds per channel: Hillaire 2020's energy-conserving step integral (Hillaire 2015,
/// "Physically Based and Unified Volumetric Rendering in Frostbite", slide 28), with its series
/// dt·(1 − x/2 + x²/6) below x = σ·dt = 1e-3 (where the f32 shader would lose digits). Twin of
/// `atmo_step_integral`.
pub(crate) fn step_integral(ext: DVec3, dt: f64) -> DVec3 {
    let f = |s: f64| {
        let x = s * dt;
        if x < 1e-3 {
            dt * (1.0 - 0.5 * x + x * x / 6.0)
        } else {
            (1.0 - (-x).exp()) / s
        }
    };
    DVec3::new(f(ext.x), f(ext.y), f(ext.z))
}

/// Rayleigh phase function 3(1 + ν²)/(16π) [1/sr] (Rayleigh 1871; Bruneton 2017).
pub(crate) fn rayleigh_phase(nu: f64) -> f64 {
    3.0 * (1.0 + nu * nu) / (16.0 * std::f64::consts::PI)
}

/// Cornette–Shanks phase function [1/sr] per channel (Cornette & Shanks 1992; Bruneton
/// 2017's `MiePhaseFunction`), ν = cos of the angle between the view ray and the direction to
/// the Sun (1 looking at the Sun: forward scattering). g = 0 gives the Rayleigh phase.
pub(crate) fn cornette_shanks(g: DVec3, nu: f64) -> DVec3 {
    let f = |g: f64| {
        let k = 3.0 / (8.0 * std::f64::consts::PI) * (1.0 - g * g) / (2.0 + g * g);
        k * (1.0 + nu * nu) / (1.0 + g * g - 2.0 * g * nu).powf(1.5)
    };
    DVec3::new(f(g.x), f(g.y), f(g.z))
}

// ---------------------------------------------------------------------------------------------
// Geometry

/// Distance from radius r along zenith cosine μ to the top of the shell (Bruneton 2017,
/// `DistanceToTopAtmosphereBoundary`).
pub(crate) fn distance_to_top(p: &AtmosphereParams, r: f64, mu: f64) -> f64 {
    let t = p.top_radius_m;
    let disc = (t - r) * (t + r) + r * r * mu * mu;
    (-r * mu + disc.max(0.0).sqrt()).max(0.0)
}

/// True when the ray from radius r with zenith cosine μ hits the bottom sphere (Bruneton 2017,
/// `RayIntersectsGround`).
pub(crate) fn ray_hits_ground(p: &AtmosphereParams, r: f64, mu: f64) -> bool {
    let b = p.bottom_radius_m;
    mu < 0.0 && (b - r) * (b + r) + r * r * mu * mu >= 0.0
}

/// Ray–sphere intersection distances (near, far) for origin `o`, unit direction `d`, radius
/// `radius`, or `None` when the line misses. Uses the perpendicular-distance discriminant for
/// precision far from the sphere (Haines et al. 2019, "Precision Improvements for Ray/Sphere
/// Intersection", Ray Tracing Gems ch. 7). Twin of `atmo_ray_sphere` (with |d| = 1).
pub(crate) fn ray_sphere(o: DVec3, d: DVec3, radius: f64) -> Option<(f64, f64)> {
    let a = d.length_squared();
    let b = o.dot(d);
    let perp = o - d * (b / a);
    let disc = a * (radius - perp.length()) * (radius + perp.length());
    if !(disc >= 0.0) {
        return None;
    }
    let s = disc.sqrt();
    Some(((-b - s) / a, (-b + s) / a))
}

// ---------------------------------------------------------------------------------------------
// Transmittance

/// Optical depth from radius r along zenith cosine μ to the top of the shell (ignoring the
/// ground), per channel: trapezoid rule with [`TRANSMITTANCE_STEPS`] intervals (Bruneton 2017,
/// `ComputeOpticalLengthToTopAtmosphereBoundary`).
pub(crate) fn optical_depth_to_top(p: &AtmosphereParams, r: f64, mu: f64) -> DVec3 {
    let d = distance_to_top(p, r, mu);
    let n = TRANSMITTANCE_STEPS;
    let dx = d / f64::from(n);
    let mut tau = DVec3::ZERO;
    for i in 0..=n {
        let di = f64::from(i) * dx;
        let ri = (di * di + 2.0 * r * mu * di + r * r).sqrt();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        tau += extinction_at(p, ri - p.bottom_radius_m) * (w * dx);
    }
    tau
}

/// exp(−optical depth to the top), ignoring the ground: what a transmittance-LUT texel holds.
pub(crate) fn transmittance_to_top(p: &AtmosphereParams, r: f64, mu: f64) -> DVec3 {
    let tau = optical_depth_to_top(p, r, mu);
    DVec3::new((-tau.x).exp(), (-tau.y).exp(), (-tau.z).exp())
}

/// Transmittance from radius `r_m` along a ray with zenith cosine `mu` to the top of the shell
/// (0 where the ray hits the ground), per channel at (680, 550, 440) nm. CPU twin of the
/// transmittance LUT (B3).
///
/// Below the bottom radius r is taken as the bottom. Above the top, a ray that misses the shell
/// or leaves it gives 1, and one that enters it gives the transmittance through it from the
/// entry point. Non-finite input gives 0.
pub fn transmittance(params: &AtmosphereParams, r_m: f64, mu: f64) -> Vec3 {
    if !(r_m.is_finite() && mu.is_finite()) {
        return Vec3::ZERO;
    }
    let mu = mu.clamp(-1.0, 1.0);
    let top = params.top_radius_m;
    let (r, mu) = if r_m > top {
        let disc = (top - r_m) * (top + r_m) + r_m * r_m * mu * mu;
        if mu >= 0.0 || disc < 0.0 {
            return Vec3::ONE;
        }
        let t = -r_m * mu - disc.sqrt();
        (top, ((r_m * mu + t) / top).clamp(-1.0, 1.0))
    } else {
        (r_m.max(params.bottom_radius_m), mu)
    };
    if ray_hits_ground(params, r, mu) {
        return Vec3::ZERO;
    }
    transmittance_to_top(params, r, mu).as_vec3()
}

// ---------------------------------------------------------------------------------------------
// LUT parameterisations

fn unit_to_uv(x: f64, n: u32) -> f64 {
    0.5 / f64::from(n) + x * (1.0 - 1.0 / f64::from(n))
}

fn uv_to_unit(u: f64, n: u32) -> f64 {
    (u - 0.5 / f64::from(n)) / (1.0 - 1.0 / f64::from(n))
}

/// Bruneton 2017's transmittance-LUT texture coordinates of (r, μ)
/// (`GetTransmittanceTextureUvFromRMu`): u from the distance to the top between its minimum and
/// maximum for this r, v from ρ/H, ρ = √(r² − r_b²), H = √(r_t² − r_b²); both mapped to texel
/// centres. Twin of `atmo_transmittance_uv` in `ground_lighting.wgsl`.
pub fn transmittance_uv(params: &AtmosphereParams, r_m: f64, mu: f64) -> DVec2 {
    let (b, t) = (params.bottom_radius_m, params.top_radius_m);
    let h = ((t - b) * (t + b)).sqrt();
    let rho = ((r_m - b) * (r_m + b)).max(0.0).sqrt();
    let d = distance_to_top(params, r_m, mu);
    let (d_min, d_max) = (t - r_m, rho + h);
    let x_mu = if d_max > d_min {
        (d - d_min) / (d_max - d_min)
    } else {
        0.0
    };
    let [w, hh] = TRANSMITTANCE_SIZE;
    DVec2::new(unit_to_uv(x_mu, w), unit_to_uv(rho / h, hh))
}

/// Inverse of [`transmittance_uv`] (Bruneton 2017, `GetRMuFromTransmittanceTextureUv`).
pub fn transmittance_r_mu(params: &AtmosphereParams, uv: DVec2) -> (f64, f64) {
    let (b, t) = (params.bottom_radius_m, params.top_radius_m);
    let [w, hh] = TRANSMITTANCE_SIZE;
    let x_mu = uv_to_unit(uv.x, w);
    let x_r = uv_to_unit(uv.y, hh);
    let h = ((t - b) * (t + b)).sqrt();
    let rho = h * x_r;
    let r = (rho * rho + b * b).sqrt();
    let (d_min, d_max) = (t - r, rho + h);
    let d = d_min + x_mu * (d_max - d_min);
    let mu = if d == 0.0 {
        1.0
    } else {
        (h * h - rho * rho - d * d) / (2.0 * r * d)
    };
    (r, mu.clamp(-1.0, 1.0))
}

/// Texture coordinates of (r, μ_s) in the multiple-scattering or irradiance LUT of `size`:
/// x = μ_s/2 + 1/2, y = (r − r_b)/(r_t − r_b), mapped to texel centres (Hillaire 2020's sample,
/// `fromUnitToSubUvs`; Bruneton 2017's irradiance texture). Twin of `atmo_rmus_uv`.
pub fn rmus_uv(params: &AtmosphereParams, r_m: f64, mu_s: f64, size: [u32; 2]) -> DVec2 {
    let (b, t) = (params.bottom_radius_m, params.top_radius_m);
    let x = (mu_s * 0.5 + 0.5).clamp(0.0, 1.0);
    let y = ((r_m - b) / (t - b)).clamp(0.0, 1.0);
    DVec2::new(unit_to_uv(x, size[0]), unit_to_uv(y, size[1]))
}

/// Inverse of [`rmus_uv`]: (r, μ_s) of a point in the LUT of `size`.
pub fn rmus_from_uv(params: &AtmosphereParams, uv: DVec2, size: [u32; 2]) -> (f64, f64) {
    let (b, t) = (params.bottom_radius_m, params.top_radius_m);
    let mu_s = (uv_to_unit(uv.x, size[0]) * 2.0 - 1.0).clamp(-1.0, 1.0);
    let r = b + uv_to_unit(uv.y, size[1]).clamp(0.0, 1.0) * (t - b);
    (r, mu_s)
}

/// The (r, μ_s) a multiple-scattering or irradiance texel is evaluated at: [`rmus_from_uv`] of
/// its centre, lifted to at least [`LUT_GROUND_OFFSET_M`] above the bottom. Twin of
/// `atmo_texel_rmus`.
pub(crate) fn texel_rmus(p: &AtmosphereParams, uv: DVec2, size: [u32; 2]) -> (f64, f64) {
    let (r, mu_s) = rmus_from_uv(p, uv, size);
    (r.max(p.bottom_radius_m + LUT_GROUND_OFFSET_M), mu_s)
}

// ---------------------------------------------------------------------------------------------
// CPU copies of the LUTs

/// A LUT on the CPU, sampled like the GPU's linear clamp-to-edge sampler.
pub(crate) struct Table {
    size: [u32; 2],
    texels: Vec<DVec3>,
}

impl Table {
    /// Evaluates `f` at every texel centre's uv.
    fn build(size: [u32; 2], f: impl Fn(DVec2) -> DVec3) -> Self {
        let [w, h] = size;
        let mut texels = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                let uv = DVec2::new(
                    (f64::from(i) + 0.5) / f64::from(w),
                    (f64::from(j) + 0.5) / f64::from(h),
                );
                texels.push(f(uv));
            }
        }
        Self { size, texels }
    }

    /// Bilinear sample at `uv` with clamp-to-edge (texel i covers [i, i + 1)/w; its centre is
    /// exact). Non-finite uv gives 0.
    pub fn sample(&self, uv: DVec2) -> DVec3 {
        if !uv.is_finite() {
            return DVec3::ZERO;
        }
        let [w, h] = self.size.map(|n| n as i64);
        let x = uv.x * w as f64 - 0.5;
        let y = uv.y * h as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |i: f64, j: f64| {
            let i = (i as i64).clamp(0, w - 1);
            let j = (j as i64).clamp(0, h - 1);
            self.texels[(j * w + i) as usize]
        };
        let top = at(x0, y0).lerp(at(x0 + 1.0, y0), fx);
        let bottom = at(x0, y0 + 1.0).lerp(at(x0 + 1.0, y0 + 1.0), fx);
        top.lerp(bottom, fy)
    }
}

/// The CPU LUTs of one parameter set, built on first use.
pub(crate) struct CpuLuts {
    params: AtmosphereParams,
    transmittance: OnceLock<Table>,
    multiscatter: OnceLock<Table>,
}

impl CpuLuts {
    pub fn params(&self) -> &AtmosphereParams {
        &self.params
    }

    pub fn transmittance(&self) -> &Table {
        self.transmittance.get_or_init(|| {
            let p = &self.params;
            Table::build(TRANSMITTANCE_SIZE, |uv| {
                let (r, mu) = transmittance_r_mu(p, uv);
                transmittance_to_top(p, r, mu)
            })
        })
    }

    pub fn multiscatter(&self) -> &Table {
        self.multiscatter.get_or_init(|| {
            let t = self.transmittance();
            Table::build(MULTISCATTER_SIZE, |uv| {
                let (r, mu_s) = texel_rmus(&self.params, uv, MULTISCATTER_SIZE);
                multiscatter_with(&self.params, t, r, mu_s)
            })
        })
    }
}

/// The LUT transmittance toward the Sun at radius r: 0 in the planet's shadow (Hillaire 2020's
/// planet-shadow test), else the transmittance LUT at (clamped r, μ_s). Twin of
/// `atmo_sun_lut_transmittance`.
fn sun_transmittance_with(p: &AtmosphereParams, t: &Table, r: f64, mu_s: f64) -> DVec3 {
    if ray_hits_ground(p, r, mu_s) {
        return DVec3::ZERO;
    }
    let r = r.clamp(p.bottom_radius_m, p.top_radius_m);
    t.sample(transmittance_uv(p, r, mu_s))
}

/// The transmittance LUT at (r clamped to the shell, μ), no ground test. Twin of
/// `atmo_lut_transmittance`.
pub(crate) fn lut_transmittance(luts: &CpuLuts, r: f64, mu: f64) -> DVec3 {
    let p = luts.params();
    let r = r.clamp(p.bottom_radius_m, p.top_radius_m);
    luts.transmittance().sample(transmittance_uv(p, r, mu))
}

/// The CPU LUTs for `params`, shared by every caller with equal parameters (a small cache: the
/// four presets plus a few test variants).
pub(crate) fn cpu_luts(params: &AtmosphereParams) -> Arc<CpuLuts> {
    static CACHE: Mutex<Vec<Arc<CpuLuts>>> = Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(luts) = cache.iter().find(|l| l.params == *params) {
        return Arc::clone(luts);
    }
    if cache.len() >= 16 {
        cache.remove(0);
    }
    let luts = Arc::new(CpuLuts {
        params: params.clone(),
        transmittance: OnceLock::new(),
        multiscatter: OnceLock::new(),
    });
    cache.push(Arc::clone(&luts));
    luts
}

// ---------------------------------------------------------------------------------------------
// Multiple scattering (Hillaire 2020 §5.5)

/// Hillaire 2020's Ψ_ms at (r, μ_s) using the transmittance table `t` for the Sun (as the GPU
/// pass samples the transmittance LUT): over 8 × 8 directions on the sphere, march 20 midpoint
/// steps with an isotropic phase to get L₂ (second-order light, eq. 5 as his sample
/// code numbers it) and f_ms (the transfer fraction, eq. 7), plus the sunlit ground's Lambert bounce; Ψ_ms = L₂ / (1 − f_ms) (eq. 10).
/// Twin of `cs_multiscatter`.
fn multiscatter_with(p: &AtmosphereParams, t: &Table, r: f64, mu_s: f64) -> DVec3 {
    let r = r.clamp(p.bottom_radius_m + LUT_GROUND_OFFSET_M, p.top_radius_m);
    let mu_s = mu_s.clamp(-1.0, 1.0);
    let o = DVec3::new(0.0, 0.0, r);
    let sun = DVec3::new((1.0 - mu_s * mu_s).max(0.0).sqrt(), 0.0, mu_s);
    let n = MULTISCATTER_SQRT_DIRS;
    let mut l2 = DVec3::ZERO;
    let mut f_ms = DVec3::ZERO;
    for a in 0..n {
        for b in 0..n {
            let rand_a = (f64::from(a) + 0.5) / f64::from(n);
            let rand_b = (f64::from(b) + 0.5) / f64::from(n);
            let theta = std::f64::consts::TAU * rand_a;
            let phi = (1.0 - 2.0 * rand_b).acos();
            let dir = DVec3::new(theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos());
            let (l, f) = multiscatter_ray(p, t, o, dir, sun);
            l2 += l;
            f_ms += f;
        }
    }
    let count = f64::from(n * n);
    let (l2, f_ms) = (l2 / count, (f_ms / count).min(DVec3::splat(MAX_F_MS)));
    l2 / (DVec3::ONE - f_ms)
}

/// One direction of [`multiscatter_with`]: (L, f) along the ray from `o` (inside the shell).
fn multiscatter_ray(
    p: &AtmosphereParams,
    t: &Table,
    o: DVec3,
    d: DVec3,
    sun: DVec3,
) -> (DVec3, DVec3) {
    let r = o.length();
    let mu = o.dot(d) / r;
    let hits = ray_hits_ground(p, r, mu);
    let b = p.bottom_radius_m;
    let t_max = if hits {
        let disc = (b - r) * (b + r) + r * r * mu * mu;
        (-r * mu - disc.max(0.0).sqrt()).max(0.0)
    } else {
        distance_to_top(p, r, mu)
    };
    let n = MULTISCATTER_STEPS;
    let dt = t_max / f64::from(n);
    let iso = 1.0 / (4.0 * std::f64::consts::PI);
    let mut throughput = DVec3::ONE;
    let mut l = DVec3::ZERO;
    let mut f = DVec3::ZERO;
    for s in 0..n {
        let ts = (f64::from(s) + 0.5) * dt;
        let pos = o + d * ts;
        let rr = pos.length();
        let m = medium(p, rr - b);
        let mu_s = pos.dot(sun) / rr;
        let t_sun = sun_transmittance_with(p, t, rr, mu_s);
        let scat = m.scattering();
        let integ = step_integral(m.extinction, dt);
        l += throughput * t_sun * scat * iso * integ;
        f += throughput * scat * integ;
        throughput *= DVec3::new(
            (-m.extinction.x * dt).exp(),
            (-m.extinction.y * dt).exp(),
            (-m.extinction.z * dt).exp(),
        );
    }
    if hits {
        let pos = o + d * t_max;
        let rr = pos.length();
        let mu_s = pos.dot(sun) / rr;
        let t_sun = t.sample(transmittance_uv(p, b, mu_s));
        l += t_sun * throughput * mu_s.max(0.0) * dv(p.ground_albedo) / std::f64::consts::PI;
    }
    (l, f)
}

/// Hillaire 2020's multiple-scattering term Ψ_ms at radius `r_m` for a Sun at zenith cosine
/// `mu_s` [1/sr, per lux of sun illuminance]: multiply by the scattering coefficient to get the
/// multiply-scattered light a point adds. CPU twin of the multiple-scattering LUT (B3).
pub fn multiscatter(params: &AtmosphereParams, r_m: f64, mu_s: f64) -> Vec3 {
    if !(r_m.is_finite() && mu_s.is_finite()) {
        return Vec3::ZERO;
    }
    let luts = cpu_luts(params);
    multiscatter_with(params, luts.transmittance(), r_m, mu_s).as_vec3()
}

// ---------------------------------------------------------------------------------------------
// The view march (shared by the feature's twin `segment_radiance` and the irradiance integral)

/// What a march returns: inscattered light per lux of sun illuminance [1/sr] and the
/// transmittance along the segment, per channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct March {
    pub inscatter: DVec3,
    pub transmittance: DVec3,
}

impl March {
    pub const EMPTY: Self = Self {
        inscatter: DVec3::ZERO,
        transmittance: DVec3::ONE,
    };
}

/// Step count of a segment of length `len`: [`MARCH_STEPS`], rising linearly to
/// [`MARCH_STEPS_LIMB`] as `len` goes from H to 2H, H = √(r_t² − r_b²) (the horizontal distance
/// from the ground to the top). Continuous in `len`, so no seam appears where it changes. Twin
/// of `atmo_step_count`.
pub(crate) fn step_count(p: &AtmosphereParams, len: f64) -> f64 {
    step_count_between(p, len, MARCH_STEPS, MARCH_STEPS_LIMB)
}

/// The view march's step count: [`VIEW_MARCH_STEPS`], rising to [`VIEW_MARCH_STEPS_LIMB`] like
/// [`step_count`]. Twin of `atmo_view_step_count`.
pub(crate) fn view_step_count(p: &AtmosphereParams, len: f64) -> f64 {
    step_count_between(p, len, VIEW_MARCH_STEPS, VIEW_MARCH_STEPS_LIMB)
}

/// `base` steps, rising linearly to `limb` as `len` goes from H to 2H. Twin of
/// `atmo_step_count_between`.
fn step_count_between(p: &AtmosphereParams, len: f64, base: f64, limb: f64) -> f64 {
    let (b, t) = (p.bottom_radius_m, p.top_radius_m);
    let h = ((t - b) * (t + b)).sqrt();
    base + (limb - base) * ((len - h) / h).clamp(0.0, 1.0)
}

/// Positions of a half-segment's steps, measured from its dense end: q_k = L·min((k/c)², 1),
/// k = 0 ..= ⌈c⌉ (c > 0). Quadratic spacing puts short steps where the air is densest (Hillaire
/// 2020's sample, "VariableSampleCount"); the clamp makes the spacing continuous in c (a new
/// step appears with zero length).
fn half_step(len: f64, c: f64, k: f64) -> f64 {
    len * ((k / c) * (k / c)).min(1.0)
}

/// Marches a ray from `o` (body-centred position) along the unit direction `d` over
/// [t0, t1], with the Sun in the unit direction `sun`. Single scattering through the
/// transmittance LUT (zero in the planet's shadow) with Rayleigh and per-channel
/// Cornette–Shanks phases, plus Hillaire's Ψ_ms × σ_s (Hillaire 2020 §5.5), integrated per step
/// with [`step_integral`].
///
/// The segment is split at the ray's closest approach to the centre (its densest point), and
/// each half's steps are spaced quadratically from that point ([`half_step`]), [`step_count`]
/// steps in all, shared by length. Each step is sampled at its midpoint: with the quadratic
/// spacing that is within 0.2 % of a 1024-step reference, where Hillaire's SampleSegmentT 0.3
/// (meant for uniform steps) overestimated by 3 % (`march_converges`). Twin of `atmo_march`.
pub(crate) fn march(luts: &CpuLuts, o: DVec3, d: DVec3, t0: f64, t1: f64, sun: DVec3) -> March {
    march_steps(luts, o, d, t0, t1, sun, step_count(luts.params(), t1 - t0))
}

/// [`march`] with `n` steps instead of [`step_count`]'s (the convergence test's reference).
pub(crate) fn march_steps(
    luts: &CpuLuts,
    o: DVec3,
    d: DVec3,
    t0: f64,
    t1: f64,
    sun: DVec3,
    n: f64,
) -> March {
    let p = luts.params();
    let len = t1 - t0;
    if !(len > 0.0) || !(n > 0.0) || !o.is_finite() || !d.is_finite() || !sun.is_finite() {
        return March::EMPTY;
    }
    let t_tab = luts.transmittance();
    let ms_tab = luts.multiscatter();
    let nu = d.dot(sun);
    let phase_r = rayleigh_phase(nu);
    let phase_m = cornette_shanks(dv(p.mie_g), nu);
    let phase_2 = cornette_shanks(dv(p.layer2_g), nu);
    let tc = (-o.dot(d)).clamp(t0, t1);
    let (len_a, len_b) = (tc - t0, t1 - tc);
    let (c_a, c_b) = (n * len_a / len, n * len_b / len);
    let mut acc = March::EMPTY;
    let mut step = |ts: f64, dt: f64| {
        let pos = o + d * ts;
        let r = pos.length();
        let m = medium(p, r - p.bottom_radius_m);
        let mu_s = pos.dot(sun) / r;
        let t_sun = sun_transmittance_with(p, t_tab, r, mu_s);
        let psi = ms_tab.sample(rmus_uv(p, r, mu_s, MULTISCATTER_SIZE));
        let single = m.rayleigh * phase_r + m.mie * phase_m + m.layer2 * phase_2;
        let s = t_sun * single + psi * m.scattering();
        acc.inscatter += acc.transmittance * s * step_integral(m.extinction, dt);
        acc.transmittance *= DVec3::new(
            (-m.extinction.x * dt).exp(),
            (-m.extinction.y * dt).exp(),
            (-m.extinction.z * dt).exp(),
        );
    };
    // Half A, camera side: from t0 toward tc, the dense end at tc.
    if len_a > 0.0 {
        let steps = c_a.ceil();
        let mut k = steps;
        while k >= 1.0 {
            let (q_far, q_near) = (half_step(len_a, c_a, k), half_step(len_a, c_a, k - 1.0));
            let dt = q_far - q_near;
            if dt > 0.0 {
                step(tc - q_near - 0.5 * dt, dt);
            }
            k -= 1.0;
        }
    }
    // Half B: from tc outward, the dense end at tc.
    if len_b > 0.0 {
        let steps = c_b.ceil();
        let mut k = 0.0;
        while k < steps {
            let (q0, q1) = (half_step(len_b, c_b, k), half_step(len_b, c_b, k + 1.0));
            let dt = q1 - q0;
            if dt > 0.0 {
                step(tc + q0 + 0.5 * dt, dt);
            }
            k += 1.0;
        }
    }
    acc
}

// ---------------------------------------------------------------------------------------------
// Sky irradiance

/// Sky irradiance on a horizontal surface at radius `r_m` per unit sun illuminance, for a Sun at
/// zenith cosine `mu_s` (dimensionless: lux of skylight per lux of sunlight above the
/// atmosphere). E = ∫ L(ω) cos θ dω over the upper hemisphere, with L from the view march
/// (single plus multiple scattering), by a cosine-weighted 16 × 16 quadrature: E = π·mean(L).
/// CPU twin of the irradiance LUT and of `atmo_sky_irradiance` (B3; B9's shading samples use it).
/// r is clamped to [bottom + 1 m, top]; non-finite input gives 0.
pub fn sky_irradiance(params: &AtmosphereParams, r_m: f64, mu_s: f64) -> Vec3 {
    if !(r_m.is_finite() && mu_s.is_finite()) {
        return Vec3::ZERO;
    }
    let luts = cpu_luts(params);
    sky_irradiance_with(&luts, r_m, mu_s).as_vec3()
}

pub(crate) fn sky_irradiance_with(luts: &CpuLuts, r_m: f64, mu_s: f64) -> DVec3 {
    let p = luts.params();
    let r = r_m.clamp(p.bottom_radius_m + LUT_GROUND_OFFSET_M, p.top_radius_m);
    let mu_s = mu_s.clamp(-1.0, 1.0);
    let o = DVec3::new(0.0, 0.0, r);
    let sun = DVec3::new((1.0 - mu_s * mu_s).max(0.0).sqrt(), 0.0, mu_s);
    let [nu_rings, n_az] = IRRADIANCE_DIRS;
    let mut sum = DVec3::ZERO;
    for i in 0..nu_rings {
        let u = (f64::from(i) + 0.5) / f64::from(nu_rings);
        let (sin_t, cos_t) = (u.sqrt(), (1.0 - u).sqrt());
        for j in 0..n_az {
            let phi = std::f64::consts::TAU * (f64::from(j) + 0.5) / f64::from(n_az);
            let d = DVec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t);
            let t_top = distance_to_top(p, r, cos_t);
            sum += march(luts, o, d, 0.0, t_top, sun).inscatter;
        }
    }
    sum * (std::f64::consts::PI / f64::from(nu_rings * n_az))
}

// ---------------------------------------------------------------------------------------------
// The GPU LUTs

/// One atmosphere's GPU resources.
pub(crate) struct AtmoEntry {
    pub naif: i32,
    pub params: AtmosphereParams,
    /// Group 2 of the view pipelines and of the irradiance pass: binding 1 the transmittance
    /// LUT, 3 the sampler, 4 the multiple-scattering LUT, 5 `AtmoParamsGpu`.
    pub view_group: wgpu::BindGroup,
}

/// The LUTs of every atmosphere in the scene, built once, and the ground-lighting groups over
/// them.
pub struct AtmosphereSet {
    ground: Arc<GroundLighting>,
    view_bgl: wgpu::BindGroupLayout,
    atmos: Vec<AtmoEntry>,
}

impl AtmosphereSet {
    /// Builds the LUTs for `presets` (NAIF id, parameters) and the ground-lighting groups. The
    /// three LUT passes of every atmosphere are recorded into one command buffer and submitted
    /// here; later passes that sample the LUTs are ordered after them by wgpu. A repeated NAIF id
    /// keeps its first preset. Invalid parameters ([`AtmosphereParams::validate`]) and GPU
    /// validation errors come back as an `Err`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        presets: &[(i32, AtmosphereParams)],
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let mut ground = GroundLighting::neutral(gpu)?;
        let view_bgl = view_bind_group_layout(&gpu.device);
        let mut atmos: Vec<AtmoEntry> = Vec::new();
        if !presets.is_empty() {
            let builder = LutBuilder::new(gpu, lib, &view_bgl)?;
            for (naif, params) in presets {
                if atmos.iter().any(|a| a.naif == *naif) {
                    continue;
                }
                params
                    .validate()
                    .with_context(|| format!("atmosphere preset for NAIF {naif}"))?;
                let built = builder
                    .build(gpu, &view_bgl, params)
                    .with_context(|| format!("building the LUTs of NAIF {naif}"))?;
                ground.add(gpu, *naif, params, &built.transmittance, &built.irradiance)?;
                atmos.push(AtmoEntry {
                    naif: *naif,
                    params: params.clone(),
                    view_group: built.view_group,
                });
            }
        }
        Ok(Self {
            ground: Arc::new(ground),
            view_bgl,
            atmos,
        })
    }

    /// The ground-lighting groups, shared with the bodies pass (`uni_planet::BodyRenderer::new`).
    pub fn ground_lighting(&self) -> Arc<GroundLighting> {
        Arc::clone(&self.ground)
    }

    /// Layout of the per-atmosphere group 2 of the view pipelines.
    pub(crate) fn view_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.view_bgl
    }

    /// The atmosphere of body `naif`, if the set has one.
    pub(crate) fn entry(&self, naif: i32) -> Option<&AtmoEntry> {
        self.atmos.iter().find(|a| a.naif == naif)
    }

    /// Every atmosphere in the set.
    pub(crate) fn entries(&self) -> &[AtmoEntry] {
        &self.atmos
    }
}

/// A texture binding of a filterable float 2D texture.
pub(crate) fn lut_texture_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn params_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(
                size_of::<crate::feature::AtmoParamsGpu>() as u64
            ),
        },
        count: None,
    }
}

/// Group 2 of the view pipelines, the irradiance pass and the march probe: binding 1 the
/// transmittance LUT, 3 its sampler (both declared in `ground_lighting.wgsl`), 4 the
/// multiple-scattering LUT, 5 `AtmoParamsGpu` (`atmosphere_common.wgsl`).
fn view_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let vis = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("atmo view bgl"),
        entries: &[
            lut_texture_entry(1, vis),
            sampler_entry(3, vis),
            lut_texture_entry(4, vis),
            params_entry(5, vis),
        ],
    })
}

/// A linear, clamp-to-edge sampler (the CPU [`Table::sample`] twin).
pub(crate) fn lut_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("atmo lut sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// The three LUT pipelines, shared by every atmosphere.
struct LutBuilder {
    params_bgl: wgpu::BindGroupLayout,
    t_bgl: wgpu::BindGroupLayout,
    out_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    transmittance: wgpu::ComputePipeline,
    multiscatter: wgpu::ComputePipeline,
    irradiance: wgpu::ComputePipeline,
}

/// One atmosphere's LUT views and its view group.
struct BuiltLuts {
    transmittance: wgpu::TextureView,
    irradiance: wgpu::TextureView,
    view_group: wgpu::BindGroup,
}

impl LutBuilder {
    fn new(
        gpu: &GpuContext,
        lib: &ShaderLib,
        view_bgl: &wgpu::BindGroupLayout,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        let cs = wgpu::ShaderStages::COMPUTE;
        let vis = wgpu::ShaderStages::FRAGMENT | cs;
        let t_module = lib.module(device, crate::TRANSMITTANCE_LUT_WGSL)?;
        let ms_module = lib.module(device, crate::MULTISCATTER_LUT_WGSL)?;
        let irr_module = lib.module(device, crate::IRRADIANCE_LUT_WGSL)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo lut params bgl"),
            entries: &[params_entry(5, vis)],
        });
        let t_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo lut transmittance bgl"),
            entries: &[
                lut_texture_entry(1, vis),
                sampler_entry(3, vis),
                params_entry(5, vis),
            ],
        });
        let out_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo lut out bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: cs,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: LUT_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let pipeline = |label: &str,
                        module: &wgpu::ShaderModule,
                        entry: &str,
                        group2: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[None, None, Some(group2), Some(&out_bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let transmittance = pipeline(
            "atmo transmittance lut",
            &t_module,
            "cs_transmittance",
            &params_bgl,
        );
        let multiscatter = pipeline(
            "atmo multiscatter lut",
            &ms_module,
            "cs_multiscatter",
            &t_bgl,
        );
        let irradiance = pipeline(
            "atmo irradiance lut",
            &irr_module,
            "cs_irradiance",
            view_bgl,
        );
        let sampler = lut_sampler(device);
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the atmosphere LUT pipelines: {err}");
        }
        Ok(Self {
            params_bgl,
            t_bgl,
            out_bgl,
            sampler,
            transmittance,
            multiscatter,
            irradiance,
        })
    }

    /// Creates the three LUTs of `params`, records and submits their passes.
    fn build(
        &self,
        gpu: &GpuContext,
        view_bgl: &wgpu::BindGroupLayout,
        params: &AtmosphereParams,
    ) -> anyhow::Result<BuiltLuts> {
        let device = &gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("atmo params"),
            contents: bytemuck::bytes_of(&crate::feature::AtmoParamsGpu::from_params(params)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let lut = |label: &str, [w, h]: [u32; 2]| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: LUT_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let t_view = lut("atmo transmittance lut", TRANSMITTANCE_SIZE);
        let ms_view = lut("atmo multiscatter lut", MULTISCATTER_SIZE);
        let irr_view = lut("atmo irradiance lut", IRRADIANCE_SIZE);
        fn tex(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            }
        }
        let sampler = wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        };
        let params_entry = wgpu::BindGroupEntry {
            binding: 5,
            resource: params_buffer.as_entire_binding(),
        };
        let group =
            |label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries,
                })
            };
        let g_params = group("atmo lut params", &self.params_bgl, &[params_entry.clone()]);
        let g_t = group(
            "atmo lut transmittance",
            &self.t_bgl,
            &[tex(1, &t_view), sampler.clone(), params_entry.clone()],
        );
        let view_group = group(
            "atmo view",
            view_bgl,
            &[tex(1, &t_view), sampler, tex(4, &ms_view), params_entry],
        );
        let out = |view: &wgpu::TextureView| group("atmo lut out", &self.out_bgl, &[tex(0, view)]);
        let (out_t, out_ms, out_irr) = (out(&t_view), out(&ms_view), out(&irr_view));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atmo luts"),
        });
        for (label, pipeline, g2, g3, [w, h]) in [
            (
                "atmo transmittance lut",
                &self.transmittance,
                &g_params,
                &out_t,
                TRANSMITTANCE_SIZE,
            ),
            (
                "atmo multiscatter lut",
                &self.multiscatter,
                &g_t,
                &out_ms,
                MULTISCATTER_SIZE,
            ),
            (
                "atmo irradiance lut",
                &self.irradiance,
                &view_group,
                &out_irr,
                IRRADIANCE_SIZE,
            ),
        ] {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(2, g2, &[]);
            pass.set_bind_group(3, g3, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("{err}");
        }
        Ok(BuiltLuts {
            transmittance: t_view,
            irradiance: irr_view,
            view_group,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{earth, mars, titan, venus};

    fn rayleigh_only() -> AtmosphereParams {
        AtmosphereParams {
            mie_scattering: Vec3::ZERO,
            mie_extinction: Vec3::ZERO,
            layer2_scattering: Vec3::ZERO,
            layer2_extinction: Vec3::ZERO,
            ..earth()
        }
    }

    /// The WGSL step counts are the Rust ones: the view march's 16/24 (`atmosphere_common.wgsl`
    /// `ATMO_VIEW_MARCH_STEPS(_LIMB)`, ladder step 2) and the LUT passes' 32/48, so
    /// `segment_radiance` and the pass cannot drift apart (B9p review I1).
    #[test]
    fn march_step_constants_match_wgsl() {
        let src = include_str!("../shaders/atmosphere_common.wgsl");
        for (name, value) in [
            ("ATMO_MARCH_STEPS", MARCH_STEPS),
            ("ATMO_MARCH_STEPS_LIMB", MARCH_STEPS_LIMB),
            ("ATMO_VIEW_MARCH_STEPS", VIEW_MARCH_STEPS),
            ("ATMO_VIEW_MARCH_STEPS_LIMB", VIEW_MARCH_STEPS_LIMB),
        ] {
            let line = format!("const {name}: f32 = {:?};", value as f32);
            assert_eq!(src.matches(line.as_str()).count(), 1, "{line}");
        }
        assert!(src.contains("return atmo_step_count_between(len, ATMO_VIEW_MARCH_STEPS, ATMO_VIEW_MARCH_STEPS_LIMB);"));
    }

    /// Earth Rayleigh only, r = bottom, μ = 1: T = exp(−β·H·(1 − e^{−100 km/H})) ±1e-4.
    #[test]
    fn rayleigh_only_vertical_transmittance() {
        let p = rayleigh_only();
        let h = f64::from(p.rayleigh_scale_height_m);
        let thick = p.top_radius_m - p.bottom_radius_m;
        let t = transmittance(&p, p.bottom_radius_m, 1.0);
        for c in 0..3 {
            let beta = f64::from(p.rayleigh_scattering[c]);
            let expect = (-beta * h * (1.0 - (-thick / h).exp())).exp();
            assert!(
                (f64::from(t[c]) - expect).abs() <= 1e-4,
                "channel {c}: {} vs {expect}",
                t[c]
            );
        }
    }

    /// The full Earth preset: T_550 at the ground toward the zenith in [0.80, 0.90] (the plan
    /// review computed 0.868).
    #[test]
    fn earth_zenith_transmittance_550() {
        let p = earth();
        let t = transmittance(&p, p.bottom_radius_m, 1.0);
        assert!((0.80..=0.90).contains(&t.y), "T_550 = {}", t.y);
    }

    /// The Sun's light at μ_s = 0.02 from the ground is red: R/B > 5 (review computed ≈ 186).
    #[test]
    fn earth_sunset_reddens() {
        let p = earth();
        let t = transmittance(&p, p.bottom_radius_m, 0.02);
        assert!(t.x / t.z > 5.0, "R/B = {} ({t})", t.x / t.z);
    }

    /// uv ↔ (r, μ) are inverse to ±1e-5 for every preset, over the whole LUT (texel centre of
    /// the first texel to that of the last), both ways; the same for the (r, μ_s) LUTs.
    #[test]
    fn lut_param_roundtrip() {
        for p in [earth(), mars(), venus(), titan()] {
            let [w, h] = TRANSMITTANCE_SIZE;
            for i in 0..=40 {
                for j in 0..=40 {
                    let x = DVec2::new(f64::from(i) / 40.0, f64::from(j) / 40.0);
                    let uv = DVec2::new(unit_to_uv(x.x, w), unit_to_uv(x.y, h));
                    let (r, mu) = transmittance_r_mu(&p, uv);
                    let back = transmittance_uv(&p, r, mu);
                    assert!(
                        (back - uv).abs().max_element() <= 1e-5,
                        "{uv} -> ({r}, {mu}) -> {back}"
                    );
                    let (r2, mu2) = transmittance_r_mu(&p, back);
                    let span = p.top_radius_m - p.bottom_radius_m;
                    assert!((r2 - r).abs() <= 1e-5 * span && (mu2 - mu).abs() <= 1e-5);
                    for size in [MULTISCATTER_SIZE, IRRADIANCE_SIZE] {
                        let uv = DVec2::new(unit_to_uv(x.x, size[0]), unit_to_uv(x.y, size[1]));
                        let (r, mu_s) = rmus_from_uv(&p, uv, size);
                        let back = rmus_uv(&p, r, mu_s, size);
                        assert!((back - uv).abs().max_element() <= 1e-5, "{uv} -> {back}");
                    }
                }
            }
            // (r, μ) → uv → (r, μ) for rays that do not hit the ground, below the top (at the
            // top every upward ray has d = 0, so μ is not recoverable there; T = 1 for all of
            // them, so nothing is lost).
            for k in 0..20 {
                let r =
                    p.bottom_radius_m + (p.top_radius_m - p.bottom_radius_m) * f64::from(k) / 20.0;
                let mu_h = -(1.0 - (p.bottom_radius_m / r).powi(2)).max(0.0).sqrt();
                for m in 0..=20 {
                    let mu = mu_h + (1.0 - mu_h) * f64::from(m) / 20.0;
                    let (r2, mu2) = transmittance_r_mu(&p, transmittance_uv(&p, r, mu));
                    assert!(
                        (r2 - r).abs() <= 1e-5 * (p.top_radius_m - p.bottom_radius_m),
                        "r {r} -> {r2}"
                    );
                    assert!((mu2 - mu).abs() <= 1e-5, "mu {mu} -> {mu2} at r {r}");
                }
            }
        }
    }

    /// Rayleigh and Cornette–Shanks integrate to 1 over the sphere, and Cornette–Shanks with
    /// g = 0 is the Rayleigh phase.
    #[test]
    fn phase_functions_normalised() {
        let n = 20_000;
        let integral = |f: &dyn Fn(f64) -> f64| {
            // ∫ P dΩ = 2π ∫₋₁¹ P(ν) dν, midpoint rule.
            (0..n)
                .map(|i| f(-1.0 + 2.0 * (f64::from(i) + 0.5) / f64::from(n)))
                .sum::<f64>()
                * 2.0
                / f64::from(n)
                * std::f64::consts::TAU
        };
        assert!((integral(&rayleigh_phase) - 1.0).abs() < 1e-6);
        for g in [0.0, 0.5, 0.65, 0.8] {
            let total = integral(&|nu| cornette_shanks(DVec3::splat(g), nu).x);
            assert!((total - 1.0).abs() < 1e-4, "g {g}: {total}");
        }
        for nu in [-1.0, -0.3, 0.0, 0.7, 1.0] {
            assert!((cornette_shanks(DVec3::ZERO, nu).x - rayleigh_phase(nu)).abs() < 1e-15);
        }
        // Cornette–Shanks's mean cosine is 3g(4 + g²)/(5(2 + g²)) (params.rs module docs).
        for g in [0.5, 0.73, 0.8] {
            let mean_cos = integral(&|nu| nu * cornette_shanks(DVec3::splat(g), nu).x);
            let formula = 3.0 * g * (4.0 + g * g) / (5.0 * (2.0 + g * g));
            assert!(
                (mean_cos - formula).abs() < 1e-4,
                "g {g}: {mean_cos} vs {formula}"
            );
        }
    }

    /// Sky irradiance: Earth's diffuse fraction with the Sun 60° up is that of a clean,
    /// Rayleigh-dominated sky (3–20 %), it rises with the Sun, vanishes at night, and blue
    /// dominates; Ψ_ms is finite and non-negative for every preset, also in Titan's thick haze.
    #[test]
    fn sky_irradiance_and_multiscatter_sane() {
        let p = earth();
        let b = p.bottom_radius_m;
        let mu = 60f64.to_radians().sin();
        let sky = sky_irradiance(&p, b, mu);
        let direct = transmittance(&p, b, mu) * mu as f32;
        let diffuse = sky.y / (sky.y + direct.y);
        assert!((0.03..0.2).contains(&diffuse), "diffuse fraction {diffuse}");
        assert!(sky.z > sky.y && sky.y > sky.x, "{sky}");
        let mut last = Vec3::ZERO;
        for mu_s in [-0.2, -0.05, 0.0, 0.1, 0.3, 0.6, 1.0] {
            let e = sky_irradiance(&p, b, mu_s);
            assert!(e.cmpge(last).all(), "{mu_s}: {e} < {last}");
            last = e;
        }
        assert!(sky_irradiance(&p, b, -0.5).max_element() < 1e-4 * sky.max_element());
        for q in [earth(), mars(), venus(), titan()] {
            for k in 0..=4 {
                let r =
                    q.bottom_radius_m + (q.top_radius_m - q.bottom_radius_m) * f64::from(k) / 4.0;
                for mu_s in [-1.0, -0.2, 0.0, 0.2, 1.0] {
                    let psi = multiscatter(&q, r, mu_s);
                    assert!(
                        psi.is_finite() && psi.min_element() >= 0.0,
                        "{}: {psi}",
                        q.source
                    );
                }
            }
        }
    }

    /// The transmittance twin's edges: from above the shell (missing it, leaving it, entering
    /// it), below the ground (clamped), down into the ground (0) and non-finite input (0).
    #[test]
    fn transmittance_edge_cases() {
        let p = earth();
        let (b, t) = (p.bottom_radius_m, p.top_radius_m);
        assert_eq!(transmittance(&p, 2.0 * t, 0.5), Vec3::ONE);
        assert_eq!(transmittance(&p, 2.0 * t, -0.1), Vec3::ONE);
        // From 2·top straight down hits the ground; at a grazing angle it crosses the shell
        // and equals the transmittance from the entry point.
        assert_eq!(transmittance(&p, 2.0 * t, -1.0), Vec3::ZERO);
        let r = 2.0 * t;
        let sin_g = (b + 50e3) / r;
        let mu = -(1.0 - sin_g * sin_g).sqrt();
        let disc = (t - r) * (t + r) + r * r * mu * mu;
        let entry_mu = (r * mu + (-r * mu - disc.sqrt())) / t;
        assert!(
            (transmittance(&p, r, mu) - transmittance(&p, t, entry_mu))
                .abs()
                .max_element()
                < 1e-6
        );
        assert_eq!(transmittance(&p, b - 500.0, 1.0), transmittance(&p, b, 1.0));
        assert_eq!(transmittance(&p, b, -0.1), Vec3::ZERO);
        assert_eq!(transmittance(&p, f64::NAN, 1.0), Vec3::ZERO);
        assert_eq!(transmittance(&p, b, f64::INFINITY), Vec3::ZERO);
    }

    /// The CPU LUT copy returns the exact function at texel centres and interpolates between
    /// them, clamping at the edges, as the GPU's linear clamp-to-edge sampler does.
    #[test]
    fn cpu_table_samples_like_the_gpu() {
        let size = [4, 3];
        let table = Table::build(size, |uv| DVec3::new(uv.x, uv.y, uv.x * uv.y));
        let centre =
            |i: u32, j: u32| DVec2::new((f64::from(i) + 0.5) / 4.0, (f64::from(j) + 0.5) / 3.0);
        assert_eq!(table.sample(centre(2, 1)), DVec3::new(0.625, 0.5, 0.3125));
        let mid = (centre(1, 0) + centre(2, 0)) * 0.5;
        assert!((table.sample(mid) - DVec3::new(0.5, 1.0 / 6.0, 1.0 / 12.0)).length() < 1e-12);
        assert_eq!(
            table.sample(DVec2::new(-1.0, 5.0)),
            table.sample(centre(0, 2))
        );
        assert_eq!(table.sample(DVec2::NAN), DVec3::ZERO);
    }
}
