//! The comet nucleus mesh (Task C1; Task C0 stub).
//!
//! An icosphere (subdivision [`NUCLEUS_SUBDIVISIONS`], ≈ 20 k triangles, smooth normals)
//! displaced by integer-lattice value noise seeded from the SPK-ID (two octaves, ±25 % of the
//! radius), with a two-lobe blend for seeds whose low bit is set; mean radius from SBDB's
//! diameter, else 2 km ("size assumed"); lunar-Lambert, albedo 0.04 (Lamy et al. 2004, Comets
//! II). Card note `theme::NOTE_NUCLEUS_SHAPE` (review m11).

use glam::Vec3;

/// Icosphere subdivisions of the nucleus mesh (plan C1).
pub const NUCLEUS_SUBDIVISIONS: u32 = 5;

/// The nucleus mesh of `seed` (the SPK-ID) with mean radius `radius_m` [m] at `subdiv`
/// subdivisions: vertices [m, body axes] and triangle indices. Deterministic and closed.
pub fn nucleus_mesh(seed: u64, radius_m: f64, subdiv: u32) -> (Vec<Vec3>, Vec<u32>) {
    let _ = (seed, radius_m, subdiv);
    todo!("C1")
}
