//! The comet nucleus: a procedural mesh and its photometry (Task C1).
//!
//! **Shape.** An icosphere ([`NUCLEUS_SUBDIVISIONS`] = 5: 10 242 vertices, 20 480 triangles,
//! smooth normals) displaced radially by integer-lattice value noise seeded from the SPK-ID (two
//! octaves, ±25 % of the radius; `uni_core::pcg3d` on the lattice, `uni_core::splitmix64` for the
//! seed lanes), with a two-lobe base shape (a smooth union of two overlapping spheres, a contact
//! binary like 67P or 8P) for seeds whose low bit is set. The mesh is scaled so its mean
//! cross-section is πR², R = SBDB's diameter / 2, else uni-sim's assumed 2 km
//! (`CometView::radius_assumed`; card "size assumed"): R is read as the effective radius, the
//! radius of the sphere with the same cross-section, which is how nucleus radii are measured
//! (Lamy et al. 2004, *Comets II*, 223, derive them from the cross-section with p_V = 0.04) and
//! what uni-sim's point reflects (πR²), so the point → mesh handover keeps its flux. Scaled by
//! its mean vertex distance instead, the mesh would show 8 % (one lobe) to 27 % (two lobes) more
//! cross-section than the point (measured over the tests' seeds). The mean vertex distance is
//! 3–4 % (one lobe) or about 11 % (two lobes) below R. It is
//! procedural, not a measured shape: card note `uni_ui::theme::NOTE_NUCLEUS_SHAPE` (review m11).
//! It does not rotate (no rotation states are in the catalogue); its fixed orientation comes
//! from the same seed ([`nucleus_orientation`]).
//!
//! **Photometry** is the point path's (lead ruling "comet double light"): lunar-Lambert
//! L = 0.9 (McEwen 1991, Icarus 92, 298; uni-sim's `registry::brdf_for` gives every body without
//! an atmosphere that is not a giant this BRDF), V geometric albedo `uni_sim::COMET_ALBEDO` = 0.04
//! (Lamy et al. 2004, *Comets II*, 223). [`nucleus_illuminance_lux`] is the flux an unresolved
//! nucleus sends to the camera; the comet pass's coma carries the SBDB total minus this
//! (`coma::coma_intensity_cd`), and the app's nucleus point should use this same function so the
//! point → mesh handover at `POINT_LOD_PX` keeps its flux. No self-shadowing: the neck of a
//! two-lobe nucleus is lit as if nothing blocked the Sun (approximate).

use std::collections::HashMap;

use glam::{DQuat, DVec3, IVec3, UVec3, Vec3};
use uni_core::{pcg3d, splitmix64};
use uni_physics::photometry::{Brdf, disk_phase_function, rho_from_geometric_albedo};
use uni_sim::snapshot::CometView;

/// Icosphere subdivisions of the nucleus mesh (plan C1): 20·4⁵ = 20 480 triangles.
pub const NUCLEUS_SUBDIVISIONS: u32 = 5;

/// Subdivisions above this are clamped (7 is 327 680 triangles; more would only cost memory).
const MAX_SUBDIVISIONS: u32 = 7;

/// The nucleus BRDF: lunar-Lambert with L = 0.9 (McEwen 1991, Icarus 92, 298), as uni-sim's point
/// path and the bodies pass use for airless small bodies.
pub const NUCLEUS_BRDF: Brdf = Brdf::LunarLambert { l: 0.9 };

/// V geometric albedo of a comet nucleus, 0.04 (Lamy et al. 2004, *Comets II*, 223): uni-sim's
/// `COMET_ALBEDO`, so the point and the mesh agree.
pub const NUCLEUS_ALBEDO: f64 = uni_sim::COMET_ALBEDO;

/// Relief of the value noise: ±25 % of the radius (plan C1).
const RELIEF: f64 = 0.25;
/// Lattice frequencies of the two noise octaves (cells per unit radius) and their weights,
/// which sum to 1 so the noise stays in [−1, 1].
const OCTAVES: [(f64, f64); 2] = [(2.0, 2.0 / 3.0), (4.0, 1.0 / 3.0)];

/// The two lobes of a contact-binary nucleus, in units of the shape's scale: centres along the
/// lobe axis at +0.45 and −0.40 with radii 0.70 and 0.55, so the origin lies inside both and
/// every ray from it leaves each lobe once (the shape stays star-shaped and can displace an
/// icosphere). Chosen by eye for a 67P-like bilobe (lobe ratio ~1.3, neck ~half the length);
/// not a fit to any comet.
const LOBES: [(f64, f64); 2] = [(0.45, 0.70), (-0.40, 0.55)];
/// Width of the smooth union at the neck (polynomial smooth maximum, Quílez 2013, "smooth
/// minimum", iquilezles.org/articles/smin).
const NECK_BLEND: f64 = 0.2;

/// An icosphere: unit directions and CCW-outward triangle indices.
pub(crate) struct Icosphere {
    pub(crate) dirs: Vec<DVec3>,
    pub(crate) indices: Vec<u32>,
}

/// The unit icosphere with `subdiv` subdivisions (at most [`MAX_SUBDIVISIONS`]): the regular
/// icosahedron, each triangle split into four at its edge midpoints (pushed out to the sphere)
/// `subdiv` times. Shared vertices, so the mesh is closed: 10·4ⁿ + 2 vertices, 20·4ⁿ triangles.
pub(crate) fn icosphere(subdiv: u32) -> Icosphere {
    // The icosahedron's 12 vertices are the cyclic permutations of (0, ±1, ±φ) (golden ratio φ).
    let phi = 0.5 * (1.0 + 5f64.sqrt());
    let mut dirs: Vec<DVec3> = [
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| DVec3::new(x, y, z).normalize())
    .collect();
    // Counter-clockwise seen from outside.
    let mut indices: Vec<u32> = vec![
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ];
    for _ in 0..subdiv.min(MAX_SUBDIVISIONS) {
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut next = Vec::with_capacity(indices.len() * 4);
        let mut midpoint = |a: u32, b: u32, dirs: &mut Vec<DVec3>| -> u32 {
            *midpoints.entry((a.min(b), a.max(b))).or_insert_with(|| {
                dirs.push((dirs[a as usize] + dirs[b as usize]).normalize());
                (dirs.len() - 1) as u32
            })
        };
        for t in indices.chunks(3) {
            let (a, b, c) = (t[0], t[1], t[2]);
            let ab = midpoint(a, b, &mut dirs);
            let bc = midpoint(b, c, &mut dirs);
            let ca = midpoint(c, a, &mut dirs);
            next.extend_from_slice(&[a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
        }
        indices = next;
    }
    Icosphere { dirs, indices }
}

/// The nucleus mesh of `seed` (the SPK-ID) with mean radius `radius_m` [m] (its mean
/// cross-section is π·radius², module docs) at `subdiv` subdivisions: vertices [m, body axes] and
/// triangle indices (CCW seen from outside).
/// Deterministic and closed. A radius that is not positive and finite gives an empty mesh.
pub fn nucleus_mesh(seed: u64, radius_m: f64, subdiv: u32) -> (Vec<Vec3>, Vec<u32>) {
    let ico = icosphere(subdiv);
    let vertices = nucleus_vertices(&ico, seed, radius_m);
    if vertices.is_empty() {
        return (Vec::new(), Vec::new());
    }
    (vertices, ico.indices)
}

/// [`nucleus_mesh`]'s vertices on a prebuilt icosphere (the renderer builds the icosphere once).
pub(crate) fn nucleus_vertices(ico: &Icosphere, seed: u64, radius_m: f64) -> Vec<Vec3> {
    if !(radius_m.is_finite() && radius_m > 0.0) || ico.dirs.is_empty() {
        return Vec::new();
    }
    let shaped: Vec<DVec3> = ico
        .dirs
        .iter()
        .map(|&d| d * base_radius(d, seed) * (1.0 + RELIEF * relief_noise(d, seed)))
        .collect();
    // Mean cross-section → π·radius² (the effective radius; module docs).
    let axes = silhouette_axes();
    let area = mean_silhouette_area(&shaped, &ico.indices, &axes, SILHOUETTE_GRID);
    if !(area > 0.0 && area.is_finite()) {
        return Vec::new();
    }
    let scale = radius_m / (area / std::f64::consts::PI).sqrt();
    shaped.iter().map(|&p| (p * scale).as_vec3()).collect()
}

/// Cells per side of the grid each silhouette is rasterised on. With the six
/// [`silhouette_axes`], the mean cross-section is within 0.7 % of a 21-direction 400² reference
/// for every tested seed (measured 2026-11-23), ≈ 10 ms per mesh in a debug build.
const SILHOUETTE_GRID: usize = 96;

/// The icosahedron's six vertex axes (one of each antipodal pair: a silhouette seen from d and
/// −d has the same area), spread evenly over the sphere of directions.
pub(crate) fn silhouette_axes() -> Vec<DVec3> {
    icosphere(0)
        .dirs
        .into_iter()
        .filter(|d| (d.z, d.y, d.x) > (0.0, 0.0, 0.0))
        .collect()
}

/// Mean silhouette (cross-section) area of a closed mesh over view directions `dirs` [units²]:
/// each projection is rasterised on an n × n grid spanning the mesh (cell centres inside a
/// front-facing triangle count; overlapping triangles count once).
pub(crate) fn mean_silhouette_area(
    points: &[DVec3],
    indices: &[u32],
    dirs: &[DVec3],
    n: usize,
) -> f64 {
    let reach = points.iter().map(|p| p.length()).fold(0.0, f64::max) * 1.01;
    if !(reach > 0.0) || dirs.is_empty() || n == 0 {
        return 0.0;
    }
    let cell = 2.0 * reach / n as f64;
    let mut grid = vec![false; n * n];
    let mut total = 0.0;
    for &d in dirs {
        let e1 = d.any_orthonormal_vector();
        let e2 = d.cross(e1);
        let to_grid = |p: DVec3| ((p.dot(e1) + reach) / cell, (p.dot(e2) + reach) / cell);
        grid.fill(false);
        for t in indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| to_grid(points[i as usize]));
            // (e1, e2, d) is right-handed, so a triangle facing d is counter-clockwise here;
            // back faces are skipped (a front face covers the same silhouette).
            let area2 = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
            if area2 <= 0.0 {
                continue;
            }
            let lo = |x: f64, y: f64, z: f64| x.min(y).min(z).floor().max(0.0) as usize;
            let hi = |x: f64, y: f64, z: f64| (x.max(y).max(z).ceil().max(0.0) as usize).min(n);
            let edge = |p: (f64, f64), q: (f64, f64), x: f64, y: f64| {
                (q.0 - p.0) * (y - p.1) - (x - p.0) * (q.1 - p.1)
            };
            for y in lo(a.1, b.1, c.1)..hi(a.1, b.1, c.1) {
                for x in lo(a.0, b.0, c.0)..hi(a.0, b.0, c.0) {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    if edge(a, b, px, py) >= 0.0
                        && edge(b, c, px, py) >= 0.0
                        && edge(c, a, px, py) >= 0.0
                    {
                        grid[y * n + x] = true;
                    }
                }
            }
        }
        total += grid.iter().filter(|&&covered| covered).count() as f64 * cell * cell;
    }
    total / dirs.len() as f64
}

/// The noise-free base radius of the shape along unit direction `d` (scale units): 1 for a seed
/// whose low bit is clear, else the smooth union of the two [`LOBES`] along the seed's lobe axis.
pub(crate) fn base_radius(d: DVec3, seed: u64) -> f64 {
    if seed & 1 == 0 {
        return 1.0;
    }
    let axis = lobe_axis(seed);
    // Distance from the origin (inside the sphere) along d to the sphere of centre c, radius r:
    // the positive root of |t·d − c|² = r².
    let exit = |(offset, r): (f64, f64)| {
        let c = axis * offset;
        let b = d.dot(c);
        b + (b * b - c.length_squared() + r * r).sqrt()
    };
    smooth_max(exit(LOBES[0]), exit(LOBES[1]), NECK_BLEND)
}

/// The polynomial smooth maximum: max(a, b) + h²k/4, h = max(k − |a − b|, 0)/k (Quílez 2013;
/// continuous with a continuous first derivative, equal to max(a, b) once |a − b| ≥ k).
fn smooth_max(a: f64, b: f64, k: f64) -> f64 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.max(b) + h * h * k * 0.25
}

/// The two-lobe axis of `seed`: a uniformly distributed unit vector (z = 2u − 1, φ = 2πv: by
/// Archimedes' hat-box theorem, z of a uniform point on the sphere is uniform).
fn lobe_axis(seed: u64) -> DVec3 {
    let (u, v) = (unit_f64(seed, 101), unit_f64(seed, 102));
    let z = 2.0 * u - 1.0;
    let s = (1.0 - z * z).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * v;
    DVec3::new(s * phi.cos(), s * phi.sin(), z)
}

/// A uniform number in [0, 1) from `seed` and a stream `lane` (SplitMix64; the top 53 bits).
fn unit_f64(seed: u64, lane: u64) -> f64 {
    let h = splitmix64(splitmix64(seed) ^ lane.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (h >> 11) as f64 / (1_u64 << 53) as f64
}

/// Two-octave value noise in [−1, 1] on the unit direction `d` for `seed`: Σ wₒ·value(d·fₒ)
/// over [`OCTAVES`], each octave on its own seeded lattice (integer lattice + fraction inputs,
/// CLAUDE.md §6).
pub(crate) fn relief_noise(d: DVec3, seed: u64) -> f64 {
    OCTAVES
        .iter()
        .enumerate()
        .map(|(o, &(freq, weight))| {
            let lane = splitmix64(seed ^ (o as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let offset = UVec3::new(lane as u32, (lane >> 32) as u32, splitmix64(lane) as u32);
            weight * value_noise(d * freq, offset)
        })
        .sum()
}

/// Value noise in [−1, 1] at `p` (lattice units): pcg3d values at the 8 corners of p's lattice
/// cell (the lattice shifted by `offset`), blended trilinearly with the quintic fade
/// 6t⁵ − 15t⁴ + 10t³ (Perlin 2002, "Improving noise", SIGGRAPH).
fn value_noise(p: DVec3, offset: UVec3) -> f64 {
    let cell = p.floor();
    let f = p - cell;
    let base = IVec3::new(cell.x as i32, cell.y as i32, cell.z as i32);
    let fade = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
    let corner = |dx: i32, dy: i32, dz: i32| {
        let c = (base + IVec3::new(dx, dy, dz)).as_uvec3();
        let h = pcg3d(c.wrapping_add(offset));
        f64::from(h.x) / f64::from(u32::MAX) * 2.0 - 1.0
    };
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

/// Smooth vertex normals of a closed mesh: the area-weighted mean of the adjacent faces' normals.
pub fn vertex_normals(vertices: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut sum = vec![DVec3::ZERO; vertices.len()];
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| i as usize);
        if a.max(b).max(c) >= vertices.len() {
            continue;
        }
        let (pa, pb, pc) = (
            vertices[a].as_dvec3(),
            vertices[b].as_dvec3(),
            vertices[c].as_dvec3(),
        );
        // |cross| is twice the area: area-weighted.
        let n = (pb - pa).cross(pc - pa);
        for i in [a, b, c] {
            sum[i] += n;
        }
    }
    sum.iter()
        .map(|n| n.normalize_or_zero().as_vec3())
        .collect()
}

/// The nucleus's fixed orientation (body → ICRS) for `seed`: a uniformly distributed rotation
/// (Shoemake 1992, "Uniform random rotations", *Graphics Gems III*, 124).
pub fn nucleus_orientation(seed: u64) -> DQuat {
    let (u1, u2, u3) = (
        unit_f64(seed, 201),
        unit_f64(seed, 202),
        unit_f64(seed, 203),
    );
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    let (t2, t3) = (std::f64::consts::TAU * u2, std::f64::consts::TAU * u3);
    DQuat::from_xyzw(a * t2.sin(), a * t2.cos(), b * t3.sin(), b * t3.cos()).normalize()
}

/// Illuminance [lx] at the camera from comet `view`'s nucleus alone, unresolved:
/// p_V·E☉(r_h)·(R/Δ)²·Φ(α) with R its mean radius, Φ the lunar-Lambert disk phase function
/// (`photometry::disk_phase_function`) and ρ from p_V = 0.04 (the closed form of
/// `photometry::disk_illuminance_lux`, as uni-sim's `reflected_lux`; comets get no planetary
/// phase correction). 0 when the geometry is degenerate.
pub fn nucleus_illuminance_lux(view: &CometView) -> f64 {
    let (d, r) = (view.nucleus_rel_m.length(), view.sun_rel_m.length());
    if !(d > 0.0 && r > 0.0 && d.is_finite() && r.is_finite()) {
        return 0.0;
    }
    let l = match NUCLEUS_BRDF {
        Brdf::Lambert => 0.0,
        Brdf::LunarLambert { l } => l,
    };
    // ρ·[L + 2(1 − L)/3] is the disk integral at α = 0 (`rho_from_geometric_albedo`'s doc).
    let rho = rho_from_geometric_albedo(NUCLEUS_BRDF, NUCLEUS_ALBEDO);
    let at_zero_phase = rho * (l + 2.0 * (1.0 - l) / 3.0);
    // Phase angle at the nucleus between the Sun and the camera.
    let to_camera = -view.nucleus_rel_m;
    let phase = view
        .sun_rel_m
        .cross(to_camera)
        .length()
        .atan2(view.sun_rel_m.dot(to_camera));
    let ratio = view.nucleus_radius_m / d;
    at_zero_phase
        * (super::sun_intensity_cd() / (r * r))
        * ratio
        * ratio
        * disk_phase_function(NUCLEUS_BRDF, phase)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use uni_core::AU;
    use uni_physics::photometry::disk_illuminance_lux;

    use super::*;
    use crate::comet::sun_intensity_cd;
    use crate::comet::test_support::static_view;

    /// Signed volume [m³] of a closed mesh (divergence theorem; positive for outward winding).
    fn volume(v: &[Vec3], idx: &[u32]) -> f64 {
        idx.chunks(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| v[t[k] as usize].as_dvec3());
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    fn mean_radius(v: &[Vec3]) -> f64 {
        v.iter().map(|p| f64::from(p.length())).sum::<f64>() / v.len() as f64
    }

    /// Real SPK-IDs, both low-bit cases: 1P/Halley 1000036 (even: one lobe), 67P 1000012
    /// (even), 2P/Encke 1000025 (odd: two lobes), C/2020 F3 NEOWISE 1003690 (even), and some
    /// odd ones.
    const SEEDS: [u64; 6] = [1_000_036, 1_000_012, 1_000_025, 1_003_690, 1_000_001, 7];

    /// Same seed → identical mesh; closed and consistently wound (every directed edge once, its
    /// reverse once); Euler characteristic V − E + F = 2; ≈ 20 k triangles; no zero-area
    /// triangle; positive (outward) volume; different seeds give different shapes.
    #[test]
    fn nucleus_mesh_deterministic_and_closed() {
        for seed in SEEDS {
            let clock = std::time::Instant::now();
            let (v, idx) = nucleus_mesh(seed, 2_000.0, NUCLEUS_SUBDIVISIONS);
            println!(
                "seed {seed}: mesh built in {:.1} ms (this build)",
                clock.elapsed().as_secs_f64() * 1e3
            );
            let (v2, idx2) = nucleus_mesh(seed, 2_000.0, NUCLEUS_SUBDIVISIONS);
            assert_eq!(v, v2, "seed {seed}: vertices not deterministic");
            assert_eq!(idx, idx2, "seed {seed}: indices not deterministic");
            assert_eq!(idx.len() % 3, 0);
            let faces = idx.len() / 3;
            assert_eq!(faces, 20_480);
            assert_eq!(v.len(), 10_242);
            let mut directed = HashSet::new();
            for t in idx.chunks(3) {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    assert!(directed.insert((a, b)), "seed {seed}: edge {a}→{b} twice");
                }
            }
            for &(a, b) in &directed {
                assert!(directed.contains(&(b, a)), "seed {seed}: open edge {a}→{b}");
            }
            let edges = directed.len() / 2;
            assert_eq!(v.len() as i64 - edges as i64 + faces as i64, 2);
            let used: HashSet<u32> = idx.iter().copied().collect();
            assert_eq!(used.len(), v.len(), "seed {seed}: unused vertices");
            let min_area = idx
                .chunks(3)
                .map(|t| {
                    let [a, b, c] = [0, 1, 2].map(|k| v[t[k] as usize].as_dvec3());
                    0.5 * (b - a).cross(c - a).length()
                })
                .fold(f64::INFINITY, f64::min);
            // A mean triangle is 4π(2 km)²/20 480 ≈ 2 450 m²; none is degenerate.
            assert!(
                min_area > 100.0,
                "seed {seed}: smallest triangle {min_area} m²"
            );
            assert!(volume(&v, &idx) > 0.0, "seed {seed}: inward winding");
            assert!(v.iter().all(|p| p.is_finite()));
        }
        assert_ne!(
            nucleus_mesh(1_000_036, 2_000.0, 5).0,
            nucleus_mesh(1_000_038, 2_000.0, 5).0
        );
        // A bad radius gives nothing rather than NaN vertices.
        assert!(nucleus_mesh(1, f64::NAN, 5).0.is_empty());
        assert!(nucleus_mesh(1, 0.0, 5).1.is_empty());
    }

    /// The mean radius is SBDB's diameter / 2 (±5 %, plan C1), read as the effective radius
    /// √(⟨cross-section⟩/π) (module docs): checked against a 21-direction, 160² rasterisation
    /// (finer than the build's), within 2 %, for both shape families. The mean vertex distance
    /// stays within 5 % of R for one-lobe and 15 % for two-lobe shapes; the volume-equivalent
    /// radius within 10 % (both printed). The plan's culling bounds hold.
    #[test]
    fn nucleus_mean_radius_matches_diameter() {
        let half: Vec<DVec3> = icosphere(1)
            .dirs
            .into_iter()
            .filter(|d| (d.z, d.y, d.x) > (0.0, 0.0, 0.0))
            .collect();
        assert_eq!(half.len(), 21);
        for seed in SEEDS {
            for radius in [2_000.0, 5_500.0] {
                let (v, idx) = nucleus_mesh(seed, radius, NUCLEUS_SUBDIVISIONS);
                let points: Vec<DVec3> = v.iter().map(|p| p.as_dvec3()).collect();
                let area = mean_silhouette_area(&points, &idx, &half, 160);
                let r_eff = (area / std::f64::consts::PI).sqrt();
                let mean = mean_radius(&v);
                let r_vol = (3.0 * volume(&v, &idx) / (4.0 * std::f64::consts::PI)).cbrt();
                println!(
                    "seed {seed} R {radius}: effective {r_eff:.1} m ({:+.2} %), mean vertex \
                     distance {mean:.1} m ({:+.2} %), volume-equivalent {r_vol:.1} m ({:+.2} %)",
                    100.0 * (r_eff / radius - 1.0),
                    100.0 * (mean / radius - 1.0),
                    100.0 * (r_vol / radius - 1.0)
                );
                assert!(
                    (r_eff / radius - 1.0).abs() < 0.02,
                    "seed {seed}: {r_eff} m"
                );
                let mean_tol = if seed & 1 == 0 { 0.05 } else { 0.15 };
                assert!(
                    (mean / radius - 1.0).abs() < mean_tol,
                    "seed {seed}: mean {mean} m"
                );
                assert!(
                    (r_vol / radius - 1.0).abs() < 0.10,
                    "seed {seed}: {r_vol} m"
                );
                let (lo, hi) = v.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), p| {
                    let r = f64::from(p.length()) / radius;
                    (lo.min(r), hi.max(r))
                });
                assert!(
                    hi < crate::comet::NUCLEUS_BOUND_RADII,
                    "seed {seed}: reaches {hi} R"
                );
                assert!(
                    lo > crate::comet::NUCLEUS_INSIDE_RADII,
                    "seed {seed}: dips to {lo} R"
                );
            }
        }
    }

    /// The relief noise stays in [−1, 1] (so ±25 % of the base radius) and uses both signs;
    /// one-lobe seeds (low bit clear) have a spherical base, so their max/min radius is at most
    /// (1 + 0.25)/(1 − 0.25) = 5/3; two-lobe seeds (low bit set) have a base about twice as long
    /// along the lobe axis as across the neck.
    #[test]
    fn nucleus_lobes_follow_the_seed_low_bit() {
        let ico = icosphere(NUCLEUS_SUBDIVISIONS);
        for seed in SEEDS {
            let n: Vec<f64> = ico.dirs.iter().map(|&d| relief_noise(d, seed)).collect();
            let (lo, hi) = n
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
                    (a.min(x), b.max(x))
                });
            assert!(
                -1.0 <= lo && hi <= 1.0,
                "seed {seed}: noise in [{lo}, {hi}]"
            );
            assert!(
                lo < -0.2 && hi > 0.2,
                "seed {seed}: noise only in [{lo}, {hi}]"
            );
        }
        for seed in [1_000_036_u64, 1_000_012, 1_003_690, 2] {
            assert!(ico.dirs.iter().all(|&d| base_radius(d, seed) == 1.0));
            let (v, _) = nucleus_mesh(seed, 1.0, NUCLEUS_SUBDIVISIONS);
            let r: Vec<f64> = v.iter().map(|p| f64::from(p.length())).collect();
            let ratio = r.iter().copied().fold(0.0, f64::max)
                / r.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                ratio <= 5.0 / 3.0 + 1e-5,
                "one-lobe seed {seed}: max/min {ratio}"
            );
        }
        for seed in [1_000_025_u64, 1_000_001, 7, 3] {
            let b: Vec<f64> = ico.dirs.iter().map(|&d| base_radius(d, seed)).collect();
            let ratio = b.iter().copied().fold(0.0, f64::max)
                / b.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(ratio > 1.9, "two-lobe seed {seed}: base max/min {ratio}");
        }
    }

    /// Smooth normals are unit, point outward, and on a sphere-like mesh agree with the radial
    /// direction to within the relief's slope.
    #[test]
    fn nucleus_normals_point_outward() {
        let (v, idx) = nucleus_mesh(1_000_036, 2_000.0, NUCLEUS_SUBDIVISIONS);
        let n = vertex_normals(&v, &idx);
        assert_eq!(n.len(), v.len());
        for (p, n) in v.iter().zip(&n) {
            assert!((n.length() - 1.0).abs() < 1e-5);
            assert!(n.dot(p.normalize()) > 0.2, "normal {n} at {p}");
        }
        let q = nucleus_orientation(1_000_036);
        assert!((q.length() - 1.0).abs() < 1e-12);
        assert_eq!(q, nucleus_orientation(1_000_036));
        assert_ne!(q, nucleus_orientation(1_000_037));
    }

    /// The nucleus flux is `disk_illuminance_lux`'s integral for ρ from p_V = 0.04 (the point
    /// path's closed form), at several phases.
    #[test]
    fn nucleus_flux_is_the_disk_integral() {
        for phase_deg in [0.0_f64, 30.0, 90.0, 150.0] {
            let a = phase_deg.to_radians();
            // Sun 1 AU from the nucleus along −x; camera 0.2 AU away at phase a.
            let sun_rel = DVec3::new(-AU, 0.0, 0.0);
            let rel = -0.2 * AU * DVec3::new(-a.cos(), a.sin(), 0.0);
            let view = static_view(1, rel, sun_rel, Some(10.0), Some(10.0));
            let got = nucleus_illuminance_lux(&view);
            let rho = rho_from_geometric_albedo(NUCLEUS_BRDF, NUCLEUS_ALBEDO);
            let e_sun = sun_intensity_cd() / (AU * AU);
            let want = disk_illuminance_lux(NUCLEUS_BRDF, rho, 2_000.0, e_sun, a, 0.2 * AU);
            assert!(
                (got / want - 1.0).abs() < 2e-3,
                "phase {phase_deg}°: {got} lx vs integral {want} lx"
            );
        }
    }
}
