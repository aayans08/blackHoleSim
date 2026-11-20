//! uni-core: the GPU-free foundation every other crate builds on.
//!
//! Positions ([`UniversePos`]), reference frames ([`FrameTree`]), time ([`Epoch`]),
//! SI constants, integer hashes (Rust twins of `shaders/common/hash.wgsl`) and
//! blackbody colour science. CPU maths is SI units in `f64`.

pub mod camera;
pub mod color;
pub mod epoch;
pub mod frame;
pub mod galactic;
pub mod hash;
pub mod pos;
pub mod units;

pub use camera::{CameraView, Lens};
pub use color::{
    blackbody_linear_srgb, blackbody_xyz, bv_to_teff, star_illuminance_lux, xyz_to_linear_srgb,
};
pub use epoch::Epoch;
pub use frame::{FrameId, FrameMotion, FrameTree, MotionFn, ROOT_FRAME};
pub use galactic::{ICRS_TO_GALACTIC, icrs_to_galactic, radec_to_icrs};
pub use hash::{hash_u64s, pcg3d, splitmix64};
pub use pos::{CELL_SIZE_M, UniversePos};
pub use units::{
    AU, C, DAY, G, GM_SUN, LY, M_SUN, PC, gravitational_radius, schwarzschild_radius,
};
