//! The camera-facing proxy mesh of a resolved ellipsoid (Task B0, shared by B1 and B2).
//!
//! Everything here is in the body's **unit-sphere space**: body-fixed coordinates divided by the
//! ellipsoid's radii, so the surface is the unit sphere. The caller scales the vertices back by
//! the radii, rotates them to ICRS, makes them camera-relative in f64 and only then casts to
//! f32. Scaling is affine, so a mesh that lies outside the unit sphere lies outside the
//! ellipsoid.

use std::f64::consts::{PI, TAU};

use glam::DVec3;

/// Rings of the bodies and star-surface caps (B1 brief: `N_RINGS = 48`). The performance
/// ladder's step is 32.
pub const CAP_RINGS: u32 = 48;
/// Segments per ring of the bodies and star-surface caps (B1 brief: `N_SEG = 96`). The
/// performance ladder's step is 64.
pub const CAP_SEGMENTS: u32 = 96;

/// Shape of the ring spacing: θ_k = θ_max · s(k/n), s(t) = t − b·sin(2πt)/(2π) with this b.
/// The rings are 1 − b = 0.4 times the uniform spacing apart at both ends of the cap and 1.6
/// times in the middle (s′(t) = 1 − b·cos 2πt). Chosen from a sweep of b over altitudes 1 m to
/// 1e9 m at Earth's radius (B0 report): near the apex a low camera sees the gap almost
/// broadside about one altitude from the point beneath it, and near the limb a camera one tenth
/// to one radius up sees the outer rings broadside, while the middle of the cap is seen at a
/// slant and has room to spare.
const RING_BEND: f64 = 0.6;

/// Ring spacing s(t) for t = k/n in [0, 1]: increasing (s′ ≥ 1 − [`RING_BEND`] > 0), s(0) = 0,
/// s(1) = 1.
fn ring_spacing(t: f64) -> f64 {
    t - RING_BEND * (TAU * t).sin() / TAU
}

/// Vertex count 1 + n_rings·n_seg when every index fits in a `u32`.
fn vertex_count(n_rings: u32, n_seg: u32) -> Option<usize> {
    let n = n_rings.checked_mul(n_seg)?.checked_add(1)?;
    usize::try_from(n).ok()
}

/// The visible cap of the unit sphere seen from `cam_unit` (the camera in unit-sphere space),
/// as a mesh that lies outside the sphere (circumscribed) and whose outline, seen from the
/// camera, contains the sphere's silhouette. Triangles: [`cap_indices`]. The bodies pass draws
/// it for coverage and vertex depth and re-intersects the exact ellipsoid per pixel (B1).
///
/// **Vertex order.** Index 0 is the apex, on the camera's direction ĉ = `cam_unit`/|`cam_unit`|.
/// Then `n_rings` rings of `n_seg` vertices, from the apex outward: ring k (1-based) vertex j is
/// index 1 + (k − 1)·`n_seg` + j, at polar angle θ_k from ĉ and azimuth φ_j = 2πj/`n_seg`,
/// measured from û = normalize(ĉ.any_orthogonal_vector()) toward ŵ = ĉ × û, i.e.
/// counter-clockwise as seen from the camera. The last ring is the outline.
///
/// **Rings.** θ_k = θ_max·s(k/`n_rings`) with s(t) = t − 0.6·sin(2πt)/(2π) ([`RING_BEND`]):
/// denser toward the limb (and the apex) than in the middle. The last ring sits exactly at
/// θ_max = acos(1/|`cam_unit`|), the polar angle of the silhouette circle.
///
/// **Circumscribed.** Every quad (and fan triangle) has a centre direction m̂, at the mean polar
/// angle of its two rings and the middle azimuth of its segment, and each ring's radius
/// ρ_k ≥ 1 is the smallest for which every vertex of every quad touching it satisfies
/// x·m̂ ≥ 1: ρ = 1/cos(angle between the vertex direction and m̂). A triangle whose vertices lie
/// in the half-space x·m̂ ≥ 1 lies in it entirely, and that half-space is outside the open unit
/// ball, so every triangle lies outside the sphere. The outer ring's edges then lie on or beyond
/// the planes tangent to the sphere at the silhouette, so the outline contains the silhouette.
///
/// **Gap** (the angle the segment from a mesh point down to the sphere subtends at the camera,
/// rulings B.3/B.4; measured by `cap_mesh_sagitta_below_altitude` at Earth's radius, 1080 lines
/// and fov_y 60°): for 48 × 96, 0.0072 px at 1 m altitude, 0.0052 px at 100 m, 0.029 px at
/// 10 km, 0.221 px at 1e6 m and 0.0038 px at 1e9 m, and the radial gap stays under 1.1e-3 × the
/// altitude. An outline with n corners must stand (π/n)²/4·sin 2β off the limb somewhere
/// (β the silhouette's half-angle), whatever the ring spacing: 0.250 px at most for 96 corners
/// (where the body fills 90°), 0.563 px for 64. So 48 × 96 exceeds 0.25 px slightly, at most
/// 0.255 px, while the body subtends about 79°–101° (camera 1.30–1.57 radii from the centre);
/// the ladder's 32 × 64 exceeds it from about 1.03 to 4.9 radii, by up to 0.57 px (0.496 px at
/// 1e6 m above Earth). `cap_mesh_gap_between_the_test_altitudes` pins both.
///
/// Returns an empty `Vec` when the camera is on or inside the sphere (|`cam_unit`| ≤ 1), is not
/// finite, or `n_rings` = 0, `n_seg` < 3, or the vertex count does not fit a `u32` index. Work
/// in f64; the caller makes the vertices camera-relative before casting to f32.
pub fn cap_mesh(cam_unit: DVec3, n_rings: u32, n_seg: u32) -> Vec<DVec3> {
    let d2 = cam_unit.length_squared();
    if !(d2 > 1.0 && d2.is_finite()) || n_rings == 0 || n_seg < 3 {
        return Vec::new();
    }
    let Some(count) = vertex_count(n_rings, n_seg) else {
        return Vec::new();
    };
    let dist = d2.sqrt();
    let c_hat = cam_unit / dist;
    let u = c_hat.any_orthogonal_vector().normalize();
    let w = c_hat.cross(u);
    // acos(1/|c|), stable for a camera just above the surface.
    let theta_max = (d2 - 1.0).sqrt().atan2(1.0);
    let n = n_rings as usize;
    let theta: Vec<f64> = (0..=n)
        .map(|k| {
            if k == n {
                theta_max
            } else {
                theta_max * ring_spacing(k as f64 / n as f64)
            }
        })
        .collect();

    // Ring radii: ρ_k = max over the bands touching ring k of 1/cos(vertex ↔ m̂). The angle
    // between (θ_k, φ_j) and m̂ = (θ_m, φ_j + π/n_seg) has cosine
    // cos θ_m cos θ_k + sin θ_m sin θ_k cos(π/n_seg) (spherical law of cosines), which is > 0
    // for θ ≤ θ_max < π/2 and n_seg ≥ 3.
    let cos_half = (PI / f64::from(n_seg)).cos();
    let mut rho = vec![1.0_f64; n + 1];
    for band in 0..n {
        let (sin_m, cos_m) = (0.5 * (theta[band] + theta[band + 1])).sin_cos();
        for k in [band, band + 1] {
            let (sin_k, cos_k) = theta[k].sin_cos();
            let cos_angle = cos_m * cos_k + sin_m * sin_k * cos_half;
            rho[k] = rho[k].max(1.0 / cos_angle);
        }
    }

    let spokes: Vec<DVec3> = (0..n_seg)
        .map(|j| {
            let (s, c) = (TAU * f64::from(j) / f64::from(n_seg)).sin_cos();
            u * c + w * s
        })
        .collect();
    let mut v = Vec::with_capacity(count);
    v.push(c_hat * rho[0]);
    for k in 1..=n {
        let (sin_k, cos_k) = theta[k].sin_cos();
        v.extend(spokes.iter().map(|&s| (c_hat * cos_k + s * sin_k) * rho[k]));
    }
    v
}

/// Triangle-list indices of [`cap_mesh`]`(_, n_rings, n_seg)`, counter-clockwise as seen from
/// the camera (every triangle faces it): first the apex fan, (0, r(1, j), r(1, j + 1)) for each
/// j, then for each band k = 1 … n_rings − 1 and segment j the two triangles (a, c, d) and
/// (a, d, b), with a = r(k, j), b = r(k, j + 1), c = r(k + 1, j), d = r(k + 1, j + 1), where
/// r(k, j) = 1 + (k − 1)·n_seg + (j mod n_seg). 3·n_seg·(2·n_rings − 1) indices; empty for the
/// sizes [`cap_mesh`] rejects.
pub fn cap_indices(n_rings: u32, n_seg: u32) -> Vec<u32> {
    if n_rings == 0 || n_seg < 3 || vertex_count(n_rings, n_seg).is_none() {
        return Vec::new();
    }
    let ring = |k: u32, j: u32| 1 + (k - 1) * n_seg + j % n_seg;
    let mut idx = Vec::with_capacity(3 * n_seg as usize * (2 * n_rings as usize - 1));
    for j in 0..n_seg {
        idx.extend([0, ring(1, j), ring(1, j + 1)]);
    }
    for k in 1..n_rings {
        for j in 0..n_seg {
            let (a, b) = (ring(k, j), ring(k, j + 1));
            let (c, d) = (ring(k + 1, j), ring(k + 1, j + 1));
            idx.extend([a, c, d, a, d, b]);
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Earth's mean radius [m] (IUGG mean radius R₁ of GRS 80: Moritz 2000, "Geodetic Reference
    /// System 1980", J. Geodesy 74, 128).
    const EARTH_R_M: f64 = 6_371_008.8;
    /// The altitudes of Review Focus 2's test [m].
    const ALTITUDES_M: [f64; 5] = [1.0, 100.0, 1.0e4, 1.0e6, 1.0e9];
    /// The mesh the passes use and the performance ladder's step (rings, segments).
    const LADDER_MESH: (u32, u32) = (32, 64);

    /// One pixel's angle on the optical axis at 1080 lines and fov_y 60° [rad]: 2·tan(30°)/1080.
    fn pixel_rad() -> f64 {
        2.0 * 30_f64.to_radians().tan() / 1080.0
    }

    /// A camera direction off every axis, so the mesh's own azimuth basis is exercised.
    fn cam_dir() -> DVec3 {
        DVec3::new(0.3, -0.5, 0.8).normalize()
    }

    /// The camera at `alt_m` above Earth, in unit-sphere space.
    fn camera(alt_m: f64) -> DVec3 {
        cam_dir() * (1.0 + alt_m / EARTH_R_M)
    }

    /// The angle [rad] the segment from mesh point `q` down to the sphere point under it,
    /// q/|q|, subtends at the camera `c` (B0 ruling B.3/B.4: the gap is the angle at the camera).
    fn subtended(q: DVec3, c: DVec3) -> f64 {
        let (a, b) = (q / q.length() - c, q - c);
        a.cross(b).length().atan2(a.dot(b))
    }

    /// What the gap test measures for one mesh.
    struct Gap {
        /// Largest angle the mesh–sphere gap subtends at the camera [rad].
        angle_rad: f64,
        /// Largest radial gap |q| − 1 over the mesh, in metres at Earth's radius.
        gap_m: f64,
        /// Largest vertex radius (unit-sphere space).
        max_vertex_r: f64,
        /// Smallest distance of a triangle's plane from the centre (≥ 1: circumscribed).
        min_plane: f64,
        /// Smallest distance of the camera in front of a triangle's plane (> 0: every triangle
        /// faces the camera).
        min_facing: f64,
    }

    /// Samples every triangle of the mesh: a barycentric grid of order 4, and for the apex fan
    /// log-spaced points toward the apex, where a camera near the surface sees the gap at its
    /// widest (the angle peaks about one altitude from the point under the camera).
    fn measure(c: DVec3, (n_rings, n_seg): (u32, u32)) -> Gap {
        let v = cap_mesh(c, n_rings, n_seg);
        let idx = cap_indices(n_rings, n_seg);
        assert_eq!(v.len(), 1 + (n_rings * n_seg) as usize);
        let mut g = Gap {
            angle_rad: 0.0,
            gap_m: 0.0,
            max_vertex_r: v.iter().map(|p| p.length()).fold(0.0, f64::max),
            min_plane: f64::INFINITY,
            min_facing: f64::INFINITY,
        };
        let mut sample = |q: DVec3| {
            g.angle_rad = g.angle_rad.max(subtended(q, c));
            g.gap_m = g.gap_m.max((q.length() - 1.0) * EARTH_R_M);
        };
        const ORDER: u32 = 4;
        for t in idx.chunks_exact(3) {
            let [a, b, d] = [t[0], t[1], t[2]].map(|i| v[i as usize]);
            for i in 0..=ORDER {
                for j in 0..=ORDER - i {
                    let order = f64::from(ORDER);
                    let (u, w) = (f64::from(i) / order, f64::from(j) / order);
                    sample(a * (1.0 - u - w) + b * u + d * w);
                }
            }
            if t[0] == 0 {
                for k in 0..=120 {
                    let f = 10_f64.powf(-f64::from(k) / 20.0);
                    for p in [b, d, 0.5 * (b + d)] {
                        sample(a + (p - a) * f);
                    }
                }
            }
            let n = (b - a).cross(d - a).normalize();
            g.min_plane = g.min_plane.min(n.dot(a));
            g.min_facing = g.min_facing.min(n.dot(c - a));
        }
        g
    }

    /// The least angle [rad] any mesh whose outline is an `n_seg`-gon must subtend somewhere:
    /// the regular polygon circumscribing the silhouette (half-angle β = asin(1/|c|)) reaches
    /// angle β' with tan β' = tan β / cos(π/n) at its corners, so a corner stands β' − β ≈
    /// (π/n)²/4 · sin 2β off the limb, whatever the ring spacing.
    fn azimuthal_floor_rad(c: DVec3, n_seg: u32) -> f64 {
        let beta = (1.0 / c.length()).asin();
        let beta_corner = (beta.tan() / (std::f64::consts::PI / f64::from(n_seg)).cos()).atan();
        beta_corner - beta
    }

    /// Review Focus 2, rulings B.3/B.4: at Earth's radius and altitudes 1 m … 1e9 m, the largest
    /// mesh–sphere gap is below 0.1 × the altitude (in metres) and below 0.25 px at 1080p/60°
    /// (as the angle it subtends at the camera); every vertex is outside the sphere (every
    /// triangle, in fact); the camera is outside the mesh's hull and sees every triangle's
    /// front. 48 × 96 meets all of it. The ladder's 32 × 64 meets it except at 1e6 m, where its
    /// 64-gon outline alone stands 0.49 px off the limb: no ring spacing can fix that (the
    /// lead records that the ladder step gives up the bound there).
    #[test]
    fn cap_mesh_sagitta_below_altitude() {
        let bound = 0.25 * pixel_rad();
        for mesh in [(CAP_RINGS, CAP_SEGMENTS), LADDER_MESH] {
            for alt in ALTITUDES_M {
                let c = camera(alt);
                let g = measure(c, mesh);
                let px = g.angle_rad / pixel_rad();
                let floor_px = azimuthal_floor_rad(c, mesh.1) / pixel_rad();
                eprintln!(
                    "cap {}x{} alt {alt:e} m: gap {px:.4} px (outline floor {floor_px:.4} px), \
                     {:.4e} m = {:.2e} of the altitude; plane margin {:.1e}, facing {:.1e}",
                    mesh.0,
                    mesh.1,
                    g.gap_m,
                    g.gap_m / alt,
                    g.min_plane - 1.0,
                    g.min_facing
                );
                let what = format!("{}x{} at {alt:e} m", mesh.0, mesh.1);
                assert!(g.gap_m < 0.1 * alt, "{what}: gap {} m", g.gap_m);
                assert!(
                    g.min_plane >= 1.0 - 1e-10,
                    "{what}: a triangle cuts the sphere"
                );
                assert!(
                    g.max_vertex_r < c.length(),
                    "{what}: camera inside the hull"
                );
                assert!(g.min_facing > 0.0, "{what}: a triangle faces away");
                // No mesh beats its outline's floor (a check on the measurement).
                assert!(
                    g.angle_rad >= azimuthal_floor_rad(c, mesh.1) * (1.0 - 1e-6),
                    "{what}"
                );
                if mesh == LADDER_MESH && alt == 1.0e6 {
                    // The shortfall, pinned: over the bound, within 2 % of the outline's floor.
                    assert!(g.angle_rad > bound, "{what}: {px} px");
                    assert!(px < 1.02 * floor_px, "{what}: {px} px vs floor {floor_px}");
                } else {
                    assert!(g.angle_rad < bound, "{what}: gap {px} px");
                }
            }
        }
    }

    /// Between the test altitudes (B0 report): the gap over a sweep of camera distances from
    /// 1 + 1e-8 to 1 + 1e4 radii. 48 × 96 exceeds 0.25 px only while the body fills about
    /// 79°–101° (1.30–1.57 radii), by at most 0.255 px, within 2 % of its outline's floor; the
    /// ladder's 32 × 64 from about 1.03 to 4.9 radii, by up to 0.573 px, within 12 % of its floor.
    #[test]
    fn cap_mesh_gap_between_the_test_altitudes() {
        let bound = 0.25 * pixel_rad();
        for (mesh, worst_px, band, floor_ratio) in [
            ((CAP_RINGS, CAP_SEGMENTS), 0.256, (1.29, 1.58), 1.02),
            (LADDER_MESH, 0.574, (1.02, 5.0), 1.12),
        ] {
            let mut worst = (0.0_f64, 0.0_f64);
            // 4 per decade of altitude, and 2 % steps through the worst region.
            let dists = (-32..=16)
                .map(|e| 1.0 + 10_f64.powf(f64::from(e) / 4.0))
                .chain((0..=40).map(|i| 1.25 * 1.02_f64.powi(i)));
            for dist in dists {
                let c = cam_dir() * dist;
                let g = measure(c, mesh);
                let floor = azimuthal_floor_rad(c, mesh.1);
                if g.angle_rad >= bound {
                    let px = g.angle_rad / pixel_rad();
                    assert!(
                        (band.0..=band.1).contains(&dist),
                        "{mesh:?} at {dist} radii: {px} px"
                    );
                    assert!(
                        g.angle_rad < floor_ratio * floor,
                        "{mesh:?} at {dist} radii: {px} px, {} × the floor",
                        g.angle_rad / floor
                    );
                }
                if g.angle_rad > worst.0 {
                    worst = (g.angle_rad, dist);
                }
            }
            let px = worst.0 / pixel_rad();
            eprintln!("cap {mesh:?}: worst {px:.4} px at {:.3} radii", worst.1);
            assert!(px < worst_px, "{mesh:?}: {px} px");
        }
    }

    /// 1024 points of the silhouette project inside the mesh outline (the last ring, a convex
    /// polygon on the image plane), for both meshes, at every altitude of the gap test and where
    /// the sphere fills 90° of the view.
    #[test]
    fn cap_mesh_covers_silhouette() {
        let alts = ALTITUDES_M
            .into_iter()
            .chain([(std::f64::consts::SQRT_2 - 1.0) * EARTH_R_M]);
        for alt in alts {
            let c = camera(alt);
            let c_hat = c.normalize();
            let u = c_hat.any_orthogonal_vector().normalize();
            let w = c_hat.cross(u);
            // Image-plane coordinates of a point seen from the camera.
            let project = |x: DVec3| {
                let r = x - c;
                glam::DVec2::new(r.dot(u), r.dot(w)) / -r.dot(c_hat)
            };
            let cos_t = 1.0 / c.length();
            let sin_t = (1.0 - cos_t * cos_t).sqrt();
            for (n_rings, n_seg) in [(CAP_RINGS, CAP_SEGMENTS), LADDER_MESH] {
                let v = cap_mesh(c, n_rings, n_seg);
                let start = 1 + ((n_rings - 1) * n_seg) as usize;
                let outline: Vec<glam::DVec2> = v[start..].iter().map(|&p| project(p)).collect();
                let n = outline.len();
                let edge = |j: usize| (outline[j], outline[(j + 1) % n]);
                // Counter-clockwise and convex.
                for j in 0..n {
                    let ((a, b), (_, d)) = (edge(j), edge((j + 1) % n));
                    assert!(
                        (b - a).perp_dot(d - b) > 0.0,
                        "{alt:e} m: not convex at {j}"
                    );
                }
                let mut margin = f64::INFINITY;
                for i in 0..1024 {
                    let phi = std::f64::consts::TAU * (f64::from(i) + 0.5) / 1024.0;
                    let limb = cos_t * c_hat + sin_t * (phi.cos() * u + phi.sin() * w);
                    let p = project(limb);
                    for j in 0..n {
                        let (a, b) = edge(j);
                        margin = margin.min((b - a).perp_dot(p - a) / (b - a).length());
                    }
                }
                // Margin in image-plane units (tan of the angle from the axis).
                assert!(
                    margin > 0.0,
                    "{n_rings}x{n_seg} at {alt:e} m: a limb point is outside ({margin:e})"
                );
            }
        }
    }

    /// The documented vertex order and triangle topology.
    #[test]
    fn cap_indices_topology() {
        for (n_rings, n_seg) in [(CAP_RINGS, CAP_SEGMENTS), LADDER_MESH, (1, 3), (2, 5)] {
            let idx = cap_indices(n_rings, n_seg);
            assert_eq!(idx.len() as u32, 3 * n_seg * (2 * n_rings - 1));
            let count = 1 + n_rings * n_seg;
            assert!(idx.iter().all(|&i| i < count));
            // The fan comes first, around the apex.
            assert!(idx[..(3 * n_seg) as usize].chunks(3).all(|t| t[0] == 0));
            assert!(idx[(3 * n_seg) as usize..].iter().all(|&i| i != 0));
            // Every vertex is used.
            let mut used = vec![false; count as usize];
            idx.iter().for_each(|&i| used[i as usize] = true);
            assert!(used.into_iter().all(|u| u));
        }
        let v = cap_mesh(camera(1.0e5), 3, 8);
        // The apex lies on the camera direction; ring 1 vertex 0 comes next.
        assert!(v[0].normalize().abs_diff_eq(cam_dir(), 1e-12));
        assert_eq!(v.len(), 25);
    }

    /// A camera on or inside the sphere, a non-finite camera and degenerate sizes give no mesh.
    #[test]
    fn cap_mesh_empty_when_inside_or_degenerate() {
        for c in [
            DVec3::ZERO,
            DVec3::X * 0.5,
            DVec3::X,
            DVec3::new(0.6, 0.8, 0.0),
            DVec3::NAN,
            DVec3::X * f64::INFINITY,
        ] {
            assert!(cap_mesh(c, CAP_RINGS, CAP_SEGMENTS).is_empty(), "{c}");
        }
        let outside = DVec3::X * 2.0;
        assert!(cap_mesh(outside, 0, CAP_SEGMENTS).is_empty());
        assert!(cap_mesh(outside, CAP_RINGS, 2).is_empty());
        assert!(cap_indices(0, CAP_SEGMENTS).is_empty());
        assert!(cap_indices(CAP_RINGS, 2).is_empty());
        assert!(cap_indices(u32::MAX, u32::MAX).is_empty());
    }
}
