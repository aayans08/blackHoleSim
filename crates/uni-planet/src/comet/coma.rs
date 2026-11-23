//! The coma (Task C1; Task C0 stub).
//!
//! Total luminous intensity from SBDB's magnitude law, I = E·Δ² with E =
//! `star_illuminance_lux(comet_total_v(M1, K1, r, Δ))`; surface brightness S(ρ) = K/ρ · e^(−ρ/ℓ)
//! with ℓ = 10⁵ km · r_h² (photodissociation scale, Haser 1957; scaling from Combi et al. 2004,
//! Comets II), normalised so ∫S dA = I (K = I/(2πℓ)). Colour: half C₂ Swan-band green, half
//! sunlight (approximate; card note). Camera-facing quad, additive, world exposure, depth test.

/// Coma surface brightness [cd/m²] at projected radius `rho_m` [m] for a coma of total luminous
/// intensity `intensity_cd` [cd] and scale length `scale_m` [m] (module docs). CPU twin of the
/// shader's; probed on the GPU.
pub fn coma_surface_brightness(rho_m: f64, intensity_cd: f64, scale_m: f64) -> f64 {
    let _ = (rho_m, intensity_cd, scale_m);
    todo!("C1")
}
