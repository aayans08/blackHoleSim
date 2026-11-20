//! Camera view and gravitational-lens descriptions shared by navigation, rendering and the HUD.

use glam::DQuat;

use crate::pos::UniversePos;

/// Where the camera is and where it looks.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CameraView {
    /// Position in the root frame.
    pub pos: UniversePos,
    /// Camera→root rotation; the camera looks along −Z with +Y up.
    pub orient: DQuat,
    /// Vertical field of view [rad].
    pub fov_y_rad: f64,
}

/// A massive object that bends light.
#[derive(Clone, Debug)]
pub struct Lens {
    pub name: String,
    /// Position in the root frame.
    pub pos: UniversePos,
    /// Mass [kg].
    pub mass_kg: f64,
    /// Dimensionless spin a/M (0 = Schwarzschild).
    pub spin: f64,
}
