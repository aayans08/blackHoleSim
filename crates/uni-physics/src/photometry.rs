//! Photometry (M1 Task A3): asteroid H, G magnitudes (Bowell et al. 1989), comet total
//! magnitudes, Lambert and lunar-Lambert (McEwen 1991) reflectance, disk illuminance, and star
//! radii from bolometric corrections (Torres 2010). A0b stub; A3 implements.
//!
//! [`rho_from_geometric_albedo`] is already implemented, and this file already ends with a
//! `#[cfg(test)] mod tests`: A3 adds its tests inside it (a second `mod tests` does not compile).

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
///
/// Implemented by the lead's A0b (fix round 1, review I3) ahead of A3, because A8's bake
/// (`albedo_scale`) calls it before A3 runs. A3 owns this module and keeps or extends it.
///
/// Derivation: at phase 0 (μ₀ = μ) the lunar-Lambert reflectance factor
/// ρ[2Lμ₀/(μ₀ + μ) + (1 − L)μ₀] (McEwen 1991, Icarus 92, 298) is ρ[L + (1 − L)μ]. The geometric
/// albedo is its mean over the projected disk, where ⟨μ⟩ = 2/3, so p = ρ[L + 2(1 − L)/3]. Lambert
/// is L = 0, giving the classical Lambert-sphere p = 2ρ/3 (Russell 1916, ApJ 43, 173). A NaN
/// `p_v` stays NaN (no silent default).
pub fn rho_from_geometric_albedo(brdf: Brdf, p_v: f64) -> f64 {
    let l = match brdf {
        Brdf::Lambert => 0.0,
        Brdf::LunarLambert { l } => l,
    };
    let rho = p_v / (l + 2.0 * (1.0 - l) / 3.0);
    // Cap at 1 (review I4: Venus, p_V = 0.689, would need ρ = 1.03; the cap costs 0.036 mag).
    if rho > 1.0 { 1.0 } else { rho }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The values A3's brief and A8's `albedo_scale_hits_geometric_albedo` expect (review I4).
    /// A3 may fold this into its `rho_from_geometric_albedo_cases`.
    #[test]
    fn rho_from_geometric_albedo_brief_values() {
        // Lambert: ρ = 1.5 p.
        assert!((rho_from_geometric_albedo(Brdf::Lambert, 0.538) - 0.807).abs() < 1e-12);
        // Lunar-Lambert L = 0.9: ρ = p / (0.9 + 0.2/3) = p / 0.96667.
        let r = rho_from_geometric_albedo(Brdf::LunarLambert { l: 0.9 }, 0.12);
        assert!((r - 0.1241).abs() < 1e-4, "{r}");
        // Venus-like p = 0.689 > 2/3 would need ρ = 1.0335: capped at 1.
        assert_eq!(rho_from_geometric_albedo(Brdf::Lambert, 0.689), 1.0);
        // L = 0 is Lambert and L = 1 is Lommel–Seeliger (ρ = p).
        assert_eq!(
            rho_from_geometric_albedo(Brdf::LunarLambert { l: 0.0 }, 0.3),
            rho_from_geometric_albedo(Brdf::Lambert, 0.3)
        );
        assert!(
            (rho_from_geometric_albedo(Brdf::LunarLambert { l: 1.0 }, 0.3) - 0.3).abs() < 1e-15
        );
        // No silent replacement of a missing albedo.
        assert!(rho_from_geometric_albedo(Brdf::Lambert, f64::NAN).is_nan());
    }
}
