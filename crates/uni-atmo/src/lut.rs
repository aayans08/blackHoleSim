//! The per-atmosphere LUTs and their CPU twins (Task B3; B0 stub).
//!
//! Hillaire 2020 (CGF 39(4)) with Bruneton 2017's (r, μ) parameterisation: transmittance
//! 256 × 64, multiple scattering 32 × 32, ground irradiance 64 × 16, all Rgba16Float, computed
//! once per atmosphere at start-up. Radii are metres from the body's centre; μ is the cosine of the
//! zenith angle of the view (or of the Sun, μ_s).

use std::sync::Arc;

use glam::Vec3;
use uni_gfx::{GpuContext, ShaderLib};

use crate::ground::GroundLighting;
use crate::params::AtmosphereParams;

/// The LUTs of every atmosphere in the scene, built once, and the ground-lighting groups over
/// them.
pub struct AtmosphereSet {
    ground: Arc<GroundLighting>,
}

impl AtmosphereSet {
    /// Builds the LUTs for `presets` (NAIF id, parameters) and the ground-lighting groups.
    ///
    /// B0 stub: registers the uni-atmo shaders and builds only the neutral ground-lighting group
    /// ([`GroundLighting::group`] returns it for every body), so the bodies pass can run before
    /// B3 lands; `presets` is not used yet.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        presets: &[(i32, AtmosphereParams)],
    ) -> anyhow::Result<Self> {
        let _ = presets;
        crate::register_shaders(lib);
        Ok(Self {
            ground: Arc::new(GroundLighting::neutral(gpu)?),
        })
    }

    /// The ground-lighting groups, shared with the bodies pass (`uni_planet::BodyRenderer::new`).
    pub fn ground_lighting(&self) -> Arc<GroundLighting> {
        Arc::clone(&self.ground)
    }
}

/// Transmittance from radius `r_m` along a ray with zenith cosine `mu` to the top of the shell
/// (0 where the ray hits the ground), per channel at (680, 550, 440) nm. CPU twin of the
/// transmittance LUT (B3).
pub fn transmittance(params: &AtmosphereParams, r_m: f64, mu: f64) -> Vec3 {
    let _ = (params, r_m, mu);
    todo!("B3")
}

/// Hillaire 2020's multiple-scattering term Ψ_ms at radius `r_m` for a Sun at zenith cosine
/// `mu_s`. CPU twin of the multiple-scattering LUT (B3).
pub fn multiscatter(params: &AtmosphereParams, r_m: f64, mu_s: f64) -> Vec3 {
    let _ = (params, r_m, mu_s);
    todo!("B3")
}

/// Sky irradiance on a horizontal surface at radius `r_m` per unit sun illuminance, for a Sun at
/// zenith cosine `mu_s`. CPU twin of the irradiance LUT and of `atmo_sky_irradiance` (B3; B9's
/// shading samples use it).
pub fn sky_irradiance(params: &AtmosphereParams, r_m: f64, mu_s: f64) -> Vec3 {
    let _ = (params, r_m, mu_s);
    todo!("B3")
}
