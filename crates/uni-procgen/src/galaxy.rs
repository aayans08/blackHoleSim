//! Milky Way galaxy model v0: thin and thick disks, halo, bar, spiral arms and dust, with a WGSL
//! twin in `shaders/galaxy.wgsl` (M1 Task A5). A0b stub; A5 implements.
//!
//! Galactocentric frame: origin at Sgr A*, +x from the Sun's projection toward the centre, +z
//! toward the north Galactic pole; distances [pc].

use glam::DVec3;

/// The model's parameters.
pub struct GalaxyModel {
    /// Placeholder for the parameters; A5 replaces it.
    _private: (),
}

/// The model's parameters for the GPU: vec4-sized members only, laid out like `GalaxyGpu` in
/// `galaxy.wgsl` (CLAUDE.md §3 uniform layout rule).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GalaxyGpu {
    /// A0b placeholder member; A5 replaces it with the model's vec4 parameters.
    pub placeholder: [f32; 4],
}

/// `ShaderLib` path of the WGSL twin (B5 imports it and binds `GalaxyGpu` at its own group).
pub const GALAXY_WGSL_PATH: &str = "procgen/galaxy.wgsl";
/// Source of the WGSL twin.
pub const GALAXY_WGSL_SRC: &str = include_str!("../shaders/galaxy.wgsl");

impl GalaxyModel {
    pub fn milky_way() -> Self {
        todo!("A5")
    }

    /// The Sun's galactocentric position [pc].
    pub fn sun_galactocentric_pc(&self) -> DVec3 {
        todo!("A5")
    }

    /// Galactocentric position [pc] of an ICRS position [pc, origin SSB].
    pub fn icrs_pc_to_galactocentric(&self, _p_icrs_pc: DVec3) -> DVec3 {
        todo!("A5")
    }

    /// Star density [stars/pc³].
    pub fn star_density(&self, _p_gc_pc: DVec3) -> f64 {
        todo!("A5")
    }

    /// V-band luminosity density [L☉,V/pc³].
    pub fn v_luminosity_density(&self, _p_gc_pc: DVec3) -> f64 {
        todo!("A5")
    }

    /// Dust extinction A_V per unit length [mag/pc].
    pub fn extinction_v_per_pc(&self, _p_gc_pc: DVec3) -> f64 {
        todo!("A5")
    }

    /// Band radiance [cd/m², linear Rec.709] seen from `cam_gc_pc` along `dir_gc`, from stars
    /// fainter than `v_lim`; the CPU twin of `gal_band_radiance`.
    pub fn band_radiance(&self, _cam_gc_pc: DVec3, _dir_gc: DVec3, _v_lim: f64) -> DVec3 {
        todo!("A5")
    }

    /// The parameters packed for the GPU.
    pub fn gpu(&self) -> GalaxyGpu {
        todo!("A5")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CLAUDE.md §3: every WGSL file is validated by a naga test in the crate that owns it.
    #[test]
    fn galaxy_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(GALAXY_WGSL_SRC)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(GALAXY_WGSL_SRC)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(GALAXY_WGSL_SRC)));
        for name in [
            "gal_density",
            "gal_luminosity",
            "gal_extinction",
            "gal_band_radiance",
        ] {
            assert!(
                module
                    .functions
                    .iter()
                    .any(|(_, f)| f.name.as_deref() == Some(name)),
                "galaxy.wgsl has no fn {name}"
            );
        }
    }
}
