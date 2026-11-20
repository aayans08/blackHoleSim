//! Projection matrices: reversed-Z with an infinite far plane.

use glam::{Mat4, Vec4};

/// Right-handed perspective, camera looking −Z, mapping `near` to depth 1 and infinity to 0.
///
/// `fov_y` is the full vertical field of view in radians and `aspect` is width / height.
/// With f = 1 / tan(fov_y / 2), a camera-space point (x, y, z, 1) goes to clip space
/// (f·x / aspect, f·y, near, −z), so NDC depth is near / −z: 1 at z = −near, falling
/// towards 0 as z → −∞. Reversed-Z spends f32 precision evenly in log depth; see
/// Reed 2015, "Depth Precision Visualized" (NVIDIA Developer Blog), and Upchurch &
/// Desbrun 2012, "Tightening the Precision of Perspective Rendering", JGT 16(1).
pub fn reversed_z_infinite_proj(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (0.5 * fov_y).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ndc(p: Mat4, v: Vec4) -> glam::Vec3 {
        let c = p * v;
        c.truncate() / c.w
    }

    #[test]
    fn reversed_z_maps_near_to_one_far_to_zero() {
        let p = reversed_z_infinite_proj(1.0, 1.5, 0.1);
        let near = ndc(p, Vec4::new(0.0, 0.0, -0.1, 1.0));
        assert!((near.z - 1.0).abs() < 1e-6, "near z = {}", near.z);
        let far = ndc(p, Vec4::new(0.0, 0.0, -1e12, 1.0));
        assert!(far.z > 0.0 && far.z < 1e-9, "far z = {}", far.z);
    }

    #[test]
    fn fov_edges_map_to_ndc_edges() {
        let (fov_y, aspect) = (60f32.to_radians(), 16.0 / 9.0);
        let p = reversed_z_infinite_proj(fov_y, aspect, 0.1);
        let t = (0.5 * fov_y).tan();
        // A point on the top edge of the frustum, 10 m ahead (camera looks along −Z, +Y up).
        let top = ndc(p, Vec4::new(0.0, 10.0 * t, -10.0, 1.0));
        assert!((top.y - 1.0).abs() < 1e-5, "top y = {}", top.y);
        let right = ndc(p, Vec4::new(10.0 * t * aspect, 0.0, -10.0, 1.0));
        assert!((right.x - 1.0).abs() < 1e-5, "right x = {}", right.x);
    }
}
