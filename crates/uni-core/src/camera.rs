//! Camera view and gravitational-lens descriptions shared by navigation, rendering and the HUD.

use glam::{DQuat, DVec3};

use crate::pos::UniversePos;

/// Where the camera is, where it looks and how fast it moves.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CameraView {
    /// Position in the root frame.
    pub pos: UniversePos,
    /// Camera→root rotation; the camera looks along −Z with +Y up.
    pub orient: DQuat,
    /// Vertical field of view [rad].
    pub fov_y_rad: f64,
    /// Velocity [m/s] in root axes. 0 unless a travel mode sets it; the HUD shows it, and M3's
    /// aberration will use it. Bookmarks saved before M1 have no such field and load at rest.
    #[serde(default)]
    pub velocity_mps: DVec3,
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    /// Bookmarks written before M1 have no velocity; they must still load, at rest.
    #[test]
    fn camera_view_ron_without_velocity_parses() {
        // The "start" view of bookmarks/m0.ron, verbatim.
        let m0_view = r#"(
            pos: (cell: (0, 0, 0), local: (149597869814024.97, 0.0, 0.0)),
            orient: (0.5609855267969309, -0.4304593345768794, -0.4304593345768794, 0.560985526796931),
            fov_y_rad: 1.0471975511965976, // 60°
        )"#;
        let view: CameraView = ron::from_str(m0_view).unwrap();
        assert_eq!(view.velocity_mps, DVec3::ZERO);
        assert_eq!(view.pos.local.x, 149_597_869_814_024.97);
        assert_eq!(view.fov_y_rad, 1.047_197_551_196_597_6);

        // A velocity, once set, survives a round trip bit-exactly.
        let moving = CameraView {
            velocity_mps: DVec3::new(1.5, -2.0e3, 2.9e8),
            ..view
        };
        let text = ron::to_string(&moving).unwrap();
        assert_eq!(ron::from_str::<CameraView>(&text).unwrap(), moving);
    }
}
