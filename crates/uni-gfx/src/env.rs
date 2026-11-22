//! The environment cubemap and its face convention.
//!
//! The environment is the sky behind everything (the Milky Way band in M1): band-update renders
//! it into a cubemap, and the lens pass samples it for its escaped rays, so the lensed sky and
//! the band drawn around it match.
//!
//! Directions are root (ICRS) axes: the texel at `cube_dir(face, uv)` holds the sky in that
//! root-axes direction, and `textureSample(env, s, dir)` with a root-axes `dir` reads it back.
//! Texels hold absolute luminance in units of [`ENV_UNIT_CD_M2`], not pre-exposed, so an
//! exposure change never invalidates the map; readers multiply by `ENV_UNIT_CD_M2` and then
//! by their exposure factor (`star_pre()` for sky light).

use glam::{Vec2, Vec3};

/// Luminance [cd/m²] of one unit in an environment texel. The faintest band light kept,
/// 1e-6 cd/m², is then 0.01, a normal f16 (≥ 2^-14 ≈ 6.1e-5), and f16's largest finite value
/// (65504) is 6.55 cd/m². Writers clamp to 60000 units (`uni_gfx::HDR_MAX`).
pub const ENV_UNIT_CD_M2: f32 = 1.0e-4;

/// The environment cubemap.
pub struct EnvMap {
    /// Six-layer 2D texture, `Rgba16Float`, texels in [`ENV_UNIT_CD_M2`] units.
    pub texture: wgpu::Texture,
    /// The `Cube` view of `texture`.
    pub view: wgpu::TextureView,
    /// The sampler readers use with `view`.
    pub sampler: wgpu::Sampler,
}

/// Unit direction through the point `uv` of cube face `face`, in the WebGPU convention:
/// layers [+X, −X, +Y, −Y, +Z, −Z] (WebGPU, `GPUTextureViewDimension` `"cube"`), `uv` in
/// [0, 1]² with (0, 0) the top-left corner of the face image, +U right and +V down.
///
/// It inverts the face selection a cube sample uses (the Vulkan specification's "Cube Map Face
/// Selection" table, which WebGPU's face orientations follow), so a texel written at
/// `cube_dir(face, uv)` is what a cube sample in that direction returns. Faces above 5 are
/// treated as −Z. Twin of `cube_dir` in `shaders/common/cube.wgsl` (the B5 band and the B7
/// lens share it, review I9); `universe --selftest` checks the two (probe `cube_dir`).
pub fn cube_dir(face: u32, uv: Vec2) -> Vec3 {
    let a = 2.0 * uv.x - 1.0;
    let b = 2.0 * uv.y - 1.0;
    let d = match face {
        0 => Vec3::new(1.0, -b, -a),
        1 => Vec3::new(-1.0, -b, a),
        2 => Vec3::new(a, 1.0, b),
        3 => Vec3::new(a, -1.0, -b),
        4 => Vec3::new(a, -b, 1.0),
        _ => Vec3::new(-a, -b, -1.0),
    };
    d.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::{ShaderLib, validate_wgsl};
    use glam::{Vec2, Vec3};

    #[test]
    fn cube_face_dirs_match_webgpu() {
        // Face centres, in the WebGPU layer order [+X, −X, +Y, −Y, +Z, −Z].
        let centres = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        for (face, want) in centres.into_iter().enumerate() {
            let got = cube_dir(face as u32, Vec2::splat(0.5));
            assert!((got - want).length() < 1e-7, "face {face}: {got}");
        }
        // The (0, 0) corner (top left, +U right, +V down) of each face, from the face selection
        // table (s_c, t_c, r_c) = +X (−z, −y, x), −X (z, −y, x), +Y (x, z, y), −Y (x, −z, y),
        // +Z (x, −y, z), −Z (−x, −y, z): u = (s_c/|r_c| + 1)/2, v = (t_c/|r_c| + 1)/2.
        let corners = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
        ];
        for (face, want) in corners.into_iter().enumerate() {
            let got = cube_dir(face as u32, Vec2::ZERO);
            assert!(
                (got - want.normalize()).length() < 1e-6,
                "face {face}: {got}"
            );
        }
        // +U and +V on +Z (the face straight ahead from inside: +X right, +Y up).
        let right = cube_dir(4, Vec2::new(0.75, 0.5));
        let down = cube_dir(4, Vec2::new(0.5, 0.75));
        assert!(right.x > 0.0 && right.y.abs() < 1e-7, "{right}");
        assert!(down.y < 0.0 && down.x.abs() < 1e-7, "{down}");

        // Each direction is selected back onto its own face at its own (u, v): the inverse of
        // the face selection table, over a grid that includes the edges.
        for face in 0..6 {
            for i in 0..=8 {
                for j in 0..=8 {
                    let uv = Vec2::new(i as f32, j as f32) / 8.0;
                    let d = cube_dir(face, uv);
                    assert!((d.length() - 1.0).abs() < 1e-6);
                    let (f, back) = select_face(d);
                    let on_edge = i == 0 || i == 8 || j == 0 || j == 8;
                    if !on_edge {
                        assert_eq!(f, face, "{face} {uv}");
                    }
                    if f == face {
                        assert!((back - uv).length() < 1e-6, "{face} {uv} -> {back}");
                    }
                }
            }
        }

        // The WGSL twin exists and validates (the GPU check is the `cube_dir` probe).
        let lib = ShaderLib::with_common();
        let module = validate_wgsl(&lib.compose("common/cube.wgsl").unwrap()).unwrap();
        assert!(
            module
                .functions
                .iter()
                .any(|(_, f)| f.name.as_deref() == Some("cube_dir"))
        );
    }

    /// Face selection as the GPU samples a cube (WebGPU `"cube"` view; the Vulkan
    /// specification's "Cube Map Face Selection" table): the major axis picks the face.
    fn select_face(d: Vec3) -> (u32, Vec2) {
        let a = d.abs();
        let (face, sc, tc, rc) = if a.x >= a.y && a.x >= a.z {
            if d.x > 0.0 {
                (0, -d.z, -d.y, a.x)
            } else {
                (1, d.z, -d.y, a.x)
            }
        } else if a.y >= a.z {
            if d.y > 0.0 {
                (2, d.x, d.z, a.y)
            } else {
                (3, d.x, -d.z, a.y)
            }
        } else if d.z > 0.0 {
            (4, d.x, -d.y, a.z)
        } else {
            (5, -d.x, -d.y, a.z)
        };
        (face, (Vec2::new(sc, tc) / rc + 1.0) * 0.5)
    }

    #[test]
    fn env_unit_keeps_faint_sky_normal_with_headroom() {
        // The faintest band light we keep, 1e-6 cd/m², is a normal f16 (≥ 2^-14) ...
        assert!(1.0e-6 / ENV_UNIT_CD_M2 >= 2_f32.powi(-14));
        assert!((1.0e-6 / ENV_UNIT_CD_M2 - 0.01).abs() < 1e-9);
        // ... and f16's largest finite value is 6.55 cd/m².
        assert!((65504.0 * ENV_UNIT_CD_M2 - 6.5504).abs() < 1e-5);
    }
}
