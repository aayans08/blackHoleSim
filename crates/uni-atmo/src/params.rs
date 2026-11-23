//! Atmosphere parameters and the four M1 presets (Task B3; B0 stub).
//!
//! Coefficients are per metre at the three wavelengths the renderer's R, G and B channels stand
//! for, (680, 550, 440) nm. Layer 2 is a tent-profile layer (Hillaire 2020, "A Scalable and
//! Production Ready Sky and Atmosphere Rendering Technique", CGF 39(4)): Earth's ozone (scattering
//! 0) or a haze layer. Every preset coefficient is cited in B3's code, and every "verify" value is
//! checked against its source before B3 merges (ruling 17).

use glam::{DVec3, Vec3};

/// One planet's atmosphere.
#[derive(Clone, Debug, PartialEq)]
pub struct AtmosphereParams {
    /// Radius of the atmosphere's lower boundary [m] (the ground; Venus: the cloud tops).
    pub bottom_radius_m: f64,
    /// Radius of the top of the shell [m]; nothing scatters above it.
    pub top_radius_m: f64,
    /// Rayleigh scattering coefficient at the bottom [1/m] at (680, 550, 440) nm.
    pub rayleigh_scattering: Vec3,
    /// Rayleigh density scale height [m].
    pub rayleigh_scale_height_m: f32,
    /// Mie (aerosol) scattering coefficient at the bottom [1/m].
    pub mie_scattering: Vec3,
    /// Mie extinction coefficient at the bottom [1/m] (scattering + absorption).
    pub mie_extinction: Vec3,
    /// Mie density scale height [m].
    pub mie_scale_height_m: f32,
    /// Cornette–Shanks asymmetry g per channel (review I6: Mars's blue aureole needs a stronger
    /// blue forward lobe).
    pub mie_g: Vec3,
    /// Layer-2 scattering coefficient at the tent's peak [1/m] (0 for Earth's ozone).
    pub layer2_scattering: Vec3,
    /// Layer-2 extinction coefficient at the tent's peak [1/m].
    pub layer2_extinction: Vec3,
    /// Altitude of the tent's peak [m].
    pub layer2_center_m: f32,
    /// Full width of the tent [m].
    pub layer2_width_m: f32,
    /// Layer-2 phase asymmetry g per channel.
    pub layer2_g: Vec3,
    /// Lambert albedo of the ground under the atmosphere, for the multiple-scattering LUT.
    pub ground_albedo: Vec3,
    /// Where the coefficients come from (papers and tables), shown on the info card.
    pub source: &'static str,
}

/// Altitude of Venus's visible cloud tops above its solid-surface radius [m]: about 70 km at low
/// latitudes (Titov et al. 2018, "Clouds and Hazes of Venus", Space Sci. Rev. 214, 126). The
/// bodies pass (B1) draws Venus's `CloudTops` map at R + this, and the Venus preset's
/// `bottom_radius_m` is the same sphere.
pub const VENUS_CLOUD_TOP_M: f64 = 70_000.0;

/// Earth: Hillaire 2020 / Bruneton 2017 defaults with bottom radius 6 371 km (B3).
pub fn earth() -> AtmosphereParams {
    todo!("B3")
}

/// Mars: CO₂ Rayleigh plus dust with per-channel asymmetry (B3).
pub fn mars() -> AtmosphereParams {
    todo!("B3")
}

/// Venus above the cloud tops: CO₂ Rayleigh plus the upper haze (B3).
pub fn venus() -> AtmosphereParams {
    todo!("B3")
}

/// Titan: N₂ Rayleigh plus the orange haze, detached layer as layer 2 (B3).
pub fn titan() -> AtmosphereParams {
    todo!("B3")
}

/// The preset for a body by NAIF id: Venus 299, Earth 399, Mars 499, Titan 606; `None` for every
/// other body. The same set as uni-sim's `has_atmosphere` (registry `ATMOSPHERE_BODIES`).
pub fn preset_for(naif: i32) -> Option<AtmosphereParams> {
    match naif {
        299 => Some(venus()),
        399 => Some(earth()),
        499 => Some(mars()),
        606 => Some(titan()),
        _ => None,
    }
}

/// True when the camera is inside the shell (`|cam_rel_body_m|` below `top_radius_m`) and the
/// Sun is above its local geometric horizon (`sun_dir`, a unit vector from the camera toward the
/// Sun, has a positive component along the local up). The app sets
/// `uni_post::ExposureSettings::daylight` from it (B6's rule; review m12). `cam_rel_body_m` is
/// camera − body centre [m] in any axes `sun_dir` shares.
pub fn camera_in_daylight(
    params: &AtmosphereParams,
    cam_rel_body_m: DVec3,
    sun_dir: DVec3,
) -> bool {
    let _ = (params, cam_rel_body_m, sun_dir);
    todo!("B3")
}

/// Strength of stellar scintillation (twinkle) for a camera at `cam_rel_body_m` (camera − body
/// centre [m]), in [0, 1]: 0 outside the shell. The app copies it into
/// `uni_sky::StarInput::scintillation`, and the star pass (B4) drives the twinkle from it
/// deterministically (owner feedback, 2026-11-22: stars twinkle only under an atmosphere).
///
/// B0 stub: 0 everywhere (no twinkle). B3 implements the inside-the-shell value.
pub fn scintillation_strength(params: &AtmosphereParams, cam_rel_body_m: DVec3) -> f32 {
    let _ = (params, cam_rel_body_m);
    0.0
}
