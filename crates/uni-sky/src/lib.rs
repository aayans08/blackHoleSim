//! uni-sky: everything drawn at infinity or as point sources. M0: PSF star splats with Planck colours.
//!
//! [`StarField`] draws catalogue stars as instanced point-spread-function splats (see
//! `shaders/stars.wgsl`). [`psf`] holds the CPU twins of the shader's PSF maths.

pub mod psf;
pub mod stars;

pub use psf::psf_weights;
pub use stars::{
    DEFAULT_STAR_EV100, STARS_SHADER, StarField, StarInstance, register_shaders, star_instance,
};
