//! Projection matrices: reversed-Z with an infinite far plane.

use glam::Mat4;

/// Right-handed perspective, camera looking −Z, mapping `near` to depth 1 and infinity to 0.
pub fn reversed_z_infinite_proj(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    todo!()
}
