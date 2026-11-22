//! Photometry (M1 Task A3): asteroid H, G magnitudes (Bowell et al. 1989), comet total
//! magnitudes, Lambert and lunar-Lambert (McEwen 1991) reflectance, disk illuminance, and star
//! radii from bolometric corrections (Torres 2010). A0b stub; A3 implements.

/// A surface reflectance model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Brdf {
    Lambert,
    /// Lunar-Lambert with limb-darkening weight `l` (1 = Lommel–Seeliger, 0 = Lambert).
    LunarLambert {
        l: f64,
    },
}

/// The H, G phase function (Bowell et al. 1989) at phase angle `alpha_rad`.
pub fn hg_phase(_alpha_rad: f64, _g: f64) -> f64 {
    todo!("A3")
}

/// Asteroid apparent V magnitude from H, G at heliocentric `r_au`, observer `delta_au` and
/// phase `alpha_rad`.
pub fn asteroid_v(_h: f64, _g: f64, _r_au: f64, _delta_au: f64, _alpha_rad: f64) -> f64 {
    todo!("A3")
}

/// Comet total magnitude m = M1 + 5 log₁₀Δ + K1 log₁₀r (JPL SBDB convention).
pub fn comet_total_v(_m1: f64, _k1: f64, _r_au: f64, _delta_au: f64) -> f64 {
    todo!("A3")
}

/// ρ-normalised reflectance factor at incidence cosine `mu0` and emission cosine `mu`; the CPU
/// twin of `body.wgsl`.
pub fn reflectance(_brdf: Brdf, _mu0: f64, _mu: f64) -> f64 {
    todo!("A3")
}

/// Normal reflectance ρ that gives V geometric albedo `p_v`: Lambert ρ = 1.5 p; lunar-Lambert
/// p = ρ[L + 2(1 − L)/3]; capped at 1 (review I4).
pub fn rho_from_geometric_albedo(_brdf: Brdf, _p_v: f64) -> f64 {
    todo!("A3")
}

/// Lambert-sphere phase function Φ(α) = [sin α + (π − α) cos α]/π.
pub fn lambert_phase(_alpha_rad: f64) -> f64 {
    todo!("A3")
}

/// Illuminance [lx] at `dist_m` from a sphere of `radius_m` with normal reflectance `albedo`,
/// lit by `sun_lux` at phase `phase_rad` (numerical integral over the lit disk).
pub fn disk_illuminance_lux(
    _brdf: Brdf,
    _albedo: f64,
    _radius_m: f64,
    _sun_lux: f64,
    _phase_rad: f64,
    _dist_m: f64,
) -> f64 {
    todo!("A3")
}

/// Bolometric correction BC_V at effective temperature `teff_k` (Torres 2010, AJ 140, 1158).
pub fn bolometric_correction_v(_teff_k: f64) -> f64 {
    todo!("A3")
}

/// Photosphere radius [m] from absolute V magnitude and effective temperature.
pub fn star_radius_m(_abs_mag_v: f64, _teff_k: f64) -> f64 {
    todo!("A3")
}
