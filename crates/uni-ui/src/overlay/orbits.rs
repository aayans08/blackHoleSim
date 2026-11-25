//! Orbit lines (Task C5; Task C0 wrote the type and the `exo_orbit_points` stub).
//!
//! 256-point polylines of the current osculating conic (`Conic::from_state` of the body's state
//! about its parent), projected in f64 from camera-relative positions and split at the near
//! plane, 1 pt grey at 35 % alpha (selected: cyan). Which orbits (coordinator ruling at Task C6:
//! orbit lines off by default, soft and uncluttered): the selected object's and the goto
//! target's own orbit (planet, dwarf planet, moon, comet or asteroid) and, for a planet or dwarf
//! planet, its major moons' orbits; with `OverlaySettings::planet_orbits` on (off in the app),
//! also every planet's and dwarf planet's about the Sun and a selected moon's siblings'. No
//! orbit is drawn over a body's disk: [`clip_out_disks`] cuts every line where it crosses the
//! projected disk of a resolved body or star (so a body also hides the orbits behind it).
//!
//! **Exoplanet orbits** (ruling 13, review m9) use the sky-plane frame of Winn 2010
//! ("Transits and Occultations", arXiv:1001.2010, §2.1 eqs. 1–4, read 2026-11-23): +Z toward the
//! observer (from the host toward the Sun), X along the node line, Y completing the frame;
//! X = −r cos(ω + f), Y = −r sin(ω + f) cos i, Z = r sin(ω + f) sin i with
//! r = a(1 − e²)/(1 + e cos f), so the planet transits (Z > 0, in front of the star) at
//! f = 90° − ω. Winn puts the descending node on +X ("Ω = 180°" in his convention); here +X lies
//! at position angle Ω_sky from celestial north toward east. Lead ruling B.2: the archive's
//! `argp_deg` is taken as the PLANET's argument of periastron ω (the archive does not say whose;
//! card note `theme::NOTE_PERIASTRON_CONVENTION`); ω = 90° when it is missing or e < 0.01, so the
//! planet is at f = 0 at `t_transit_jd`. The archive has no Ω, so every M1 exoplanet orbit is
//! drawn with Ω_sky = 0° and dashed (with `theme::NOTE_NODE_UNKNOWN`), and i = 90° when the
//! inclination is missing (`theme::NOTE_INCLINATION_UNKNOWN`). A missing eccentricity is drawn
//! circular; a planet without a semi-major axis is not drawn (no host mass to get it from the
//! period). The solid case is tested on a synthetic system.

use std::f64::consts::{FRAC_PI_2, TAU};

use egui::{Painter, Pos2, Shape, Stroke};
use glam::{DMat3, DVec2, DVec3, Vec2};
use uni_core::{AU, Epoch, PC, UniversePos};
use uni_data::{ExoPlanet, ExoSystem};
use uni_physics::kepler::{Conic, perifocal_to_reference, solve_kepler};
use uni_sim::registry::{ObjectId, ObjectKind};
use uni_sim::snapshot::{MAJOR_MOONS, Snapshot};
use uni_sim::world::{ObjectDetails, World};

use super::ScreenMap;
use crate::theme::{self, LabelTone};

/// One orbit polyline, in output points. A line of one point is a dot: an exoplanet where its
/// transit time and period place it now.
#[derive(Clone, Debug)]
pub struct OrbitLine {
    pub pts: Vec<Vec2>,
    /// Drawn dashed: the orientation is assumed (exoplanets: Ω or i unknown).
    pub dashed: bool,
    pub tone: LabelTone,
}

/// The near plane [m] lines are clipped at: the renderer's (`uni-app` `scene::NEAR_PLANE_M`).
const NEAR_PLANE_M: f64 = 0.1;
/// An open conic (e ≥ 1) is drawn out to max(this × q, [`OPEN_ORBIT_REACH_AU`]) from its focus.
const OPEN_ORBIT_REACH_Q: f64 = 10.0;
const OPEN_ORBIT_REACH_AU: f64 = 50.0;
/// Dashes of an assumed orbit [pt].
const DASH_PT: f32 = 5.0;
const GAP_PT: f32 = 4.0;
/// An exoplanet's dot: radius [pt] and alpha (the orbit's own 35 % would hide it).
const EXO_DOT_RADIUS_PT: f32 = 2.0;
const EXO_DOT_ALPHA: f32 = 0.8;
/// Exoplanet orbits are drawn around a selected host or a host within this distance [m]
/// (0.05 pc, plan C5).
pub const EXO_ORBIT_RANGE_M: f64 = 0.05 * PC;
/// Below this eccentricity the archive's ω is ignored and ω = 90° (lead ruling B.2).
const CIRCULAR_E: f64 = 0.01;
/// Julian date of J2000.0 (TDB; IERS Conventions 2010).
const JD_J2000: f64 = 2_451_545.0;

/// Eccentric anomalies [rad] of `n − 1` points around an ellipse of eccentricity `e` < 1,
/// starting at periapsis, at equal steps of the tangent's direction ψ: each segment turns the
/// same angle, 2π/(n − 1), so the sharp apsides of an eccentric orbit get the points they need
/// and the near-straight flanks few (uniform E would leave chords as long as the curvature
/// radius at the apsides of a long-period comet). The tangent (−a sin E, b cos E) has direction
/// ψ where tan E = −(b/a) cos ψ / sin ψ, so E = atan2(−√(1 − e²) cos ψ, sin ψ); ψ = 90° at
/// periapsis. A circle gets uniform E.
fn ellipse_anomalies(e: f64, n: usize) -> impl Iterator<Item = f64> {
    let b_over_a = ((1.0 - e) * (1.0 + e)).sqrt();
    (0..n - 1).map(move |k| {
        let psi = FRAC_PI_2 + TAU * k as f64 / (n - 1) as f64;
        (-b_over_a * psi.cos()).atan2(psi.sin())
    })
}

/// `n` points [m] of conic `c` relative to its focus, in the conic's reference axes. For e < 1
/// the polyline is closed (`n − 1` points at equal turning of the tangent, from periapsis, then
/// the first again); for e ≥ 1 it runs uniformly in true anomaly out to max(10 q, 50 AU). Empty
/// for a conic with no periapsis (q ≤ 0) or non-finite elements; `n` is at least 3.
pub fn conic_points(c: &Conic, n: usize) -> Vec<DVec3> {
    let n = n.max(3);
    let (q, e) = (c.q_m, c.e);
    if !(q > 0.0
        && e >= 0.0
        && e.is_finite()
        && c.i_rad.is_finite()
        && c.node_rad.is_finite()
        && c.argp_rad.is_finite())
    {
        return Vec::new();
    }
    let rot = perifocal_to_reference(c.i_rad, c.node_rad, c.argp_rad);
    if e < 1.0 {
        // Perifocal ellipse (Vallado 2013 §2.2): x = a(cos E − e) = q − 2a sin²(E/2), the form
        // without cancellation near periapsis, y = b sin E with b = √(a q (1 + e)).
        let a = q / (1.0 - e);
        let b = (a * q * (1.0 + e)).sqrt();
        let mut pts: Vec<DVec3> = ellipse_anomalies(e, n)
            .map(|ea| {
                let s = (0.5 * ea).sin();
                rot * DVec3::new(q - 2.0 * a * s * s, b * ea.sin(), 0.0)
            })
            .collect();
        pts.push(pts[0]);
        pts
    } else {
        // Open conic (parabola or hyperbola), r = p/(1 + e cos f) with p = q(1 + e), out to
        // r_max: cos f_max = (p/r_max − 1)/e, inside the asymptotes (cos f > −1/e).
        let p = q * (1.0 + e);
        let r_max = (OPEN_ORBIT_REACH_Q * q).max(OPEN_ORBIT_REACH_AU * AU);
        let f_max = ((p / r_max - 1.0) / e).clamp(-1.0, 1.0).acos();
        (0..n)
            .map(|k| {
                let f = f_max * (2.0 * k as f64 / (n - 1) as f64 - 1.0);
                let r = p / (1.0 + e * f.cos());
                rot * DVec3::new(r * f.cos(), r * f.sin(), 0.0)
            })
            .collect()
    }
}

/// The part of segment `a`–`b` [camera axes] in front of the near plane (z ≤ −0.1 m), and
/// whether each end was cut; `None` when none of it is.
fn clip_near(a: DVec3, b: DVec3) -> Option<(DVec3, DVec3, bool, bool)> {
    let z = -NEAR_PLANE_M;
    match (a.z <= z, b.z <= z) {
        (true, true) => Some((a, b, false, false)),
        (true, false) => Some((a, a + (b - a) * ((z - a.z) / (b.z - a.z)), false, true)),
        (false, true) => Some((a + (b - a) * ((z - a.z) / (b.z - a.z)), b, true, false)),
        // Both behind, or a NaN.
        (false, false) => None,
    }
}

/// The part of segment `a`–`b` [pt] inside [0, size] (Liang & Barsky 1984, "A new concept and
/// method for line clipping", ACM TOG 3, 1), and whether each end was cut; `None` outside.
fn clip_frame(a: DVec2, b: DVec2, size: DVec2) -> Option<(DVec2, DVec2, bool, bool)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (p, q) in [
        (-d.x, a.x),
        (d.x, size.x - a.x),
        (-d.y, a.y),
        (d.y, size.y - a.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1).then(|| (a + d * t0, a + d * t1, t0 > 0.0, t1 < 1.0))
}

/// Projects the polyline `pts` [m from `origin`; `origin` relative to the camera, root axes] to
/// output points. Each segment is clipped to the space in front of the near plane, then
/// projected in f64 and clipped to the frame, so no drawn segment crosses the screen. Each run
/// of connected segments is one polyline of at least two points.
pub(crate) fn project_polyline(map: &ScreenMap, origin: DVec3, pts: &[DVec3]) -> Vec<Vec<Vec2>> {
    let origin = map.camera(origin);
    let cam: Vec<DVec3> = pts.iter().map(|&p| origin + map.camera(p)).collect();
    let mut runs: Vec<Vec<Vec2>> = Vec::new();
    // Whether the last run ends on the previous segment's own (uncut) end point.
    let mut open = false;
    for w in cam.windows(2) {
        let clipped = clip_near(w[0], w[1]).and_then(|(a, b, cut_a, cut_b)| {
            let (pa, pb) = (map.project(a)?, map.project(b)?);
            let (qa, qb, cut_qa, cut_qb) = clip_frame(pa, pb, map.output)?;
            Some((qa, qb, cut_a || cut_qa, cut_b || cut_qb))
        });
        let Some((a, b, cut_a, cut_b)) = clipped else {
            open = false;
            continue;
        };
        match runs.last_mut() {
            Some(run) if open && !cut_a => run.push(b.as_vec2()),
            _ => runs.push(vec![a.as_vec2(), b.as_vec2()]),
        }
        open = !cut_b;
    }
    runs
}

/// An exoplanet's orbit as drawn: the archive's elements with the M1 defaults (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SkyOrbit {
    /// Semi-major axis [m].
    pub a_m: f64,
    pub e: f64,
    /// Inclination to the sky plane [rad] (90° = edge-on).
    pub i_rad: f64,
    /// The planet's argument of periastron ω [rad].
    pub omega_rad: f64,
    /// Position angle of Winn's +X axis from celestial north toward east [rad].
    pub node_pa_rad: f64,
    /// Ω or i was unknown and assumed: drawn dashed.
    pub assumed: bool,
    /// Orbital period [d].
    pub period_d: Option<f64>,
    /// Time of transit (conjunction) as published [JD].
    pub t_transit_jd: Option<f64>,
}

impl SkyOrbit {
    /// The orbit of `p` with the node's position angle `node_pa_deg` (`None`: unknown, 0°; the
    /// archive never gives it). `None` without a semi-major axis, or for e ≥ 1.
    pub(crate) fn new(p: &ExoPlanet, node_pa_deg: Option<f64>) -> Option<Self> {
        let a_m = p.a_au.filter(|a| *a > 0.0)? * AU;
        // A missing eccentricity is drawn circular (module docs).
        let e = p.e.unwrap_or(0.0).max(0.0);
        if !(e < 1.0) {
            return None;
        }
        // Lead ruling B.2: the archive's ω is the planet's; 90° when missing or e < 0.01.
        let omega_deg = p.argp_deg.filter(|_| e >= CIRCULAR_E).unwrap_or(90.0);
        // Ruling 13: i = 90° (edge-on) and Ω = 0° when unknown, drawn dashed.
        let i_deg = p.i_deg.unwrap_or(90.0);
        Some(SkyOrbit {
            a_m,
            e,
            i_rad: i_deg.to_radians(),
            omega_rad: omega_deg.to_radians(),
            node_pa_rad: node_pa_deg.unwrap_or(0.0).to_radians(),
            assumed: node_pa_deg.is_none() || p.i_deg.is_none(),
            period_d: p.period_d.filter(|d| *d > 0.0),
            t_transit_jd: p.t_transit_jd,
        })
    }

    /// The sky-plane axes at a host in direction `host_dir` from the Sun, as columns of a matrix
    /// (ICRS): X at position angle Ω_sky from celestial north toward east, Z toward the Sun,
    /// Y = Z × X. North is the celestial pole's direction projected on the sky; at a pole, where
    /// it is undefined, the ICRS x axis's (RA 0h) stands in. `None` for a zero or NaN direction.
    fn axes(&self, host_dir: DVec3) -> Option<DMat3> {
        let s = host_dir.try_normalize()?;
        let north = (DVec3::Z - DVec3::Z.dot(s) * s)
            .try_normalize()
            .or_else(|| (DVec3::X - DVec3::X.dot(s) * s).try_normalize())?;
        let east = north.cross(s);
        let (sin, cos) = self.node_pa_rad.sin_cos();
        let x = cos * north + sin * east;
        let z = -s;
        Some(DMat3::from_cols(x, z.cross(x), z))
    }

    /// The planet at true anomaly `f` in sky-plane coordinates (Winn 2010 eqs. 1–4) [m].
    fn sky_xyz(&self, f: f64) -> DVec3 {
        let r = self.a_m * (1.0 - self.e * self.e) / (1.0 + self.e * f.cos());
        let (sin_u, cos_u) = (self.omega_rad + f).sin_cos();
        let (sin_i, cos_i) = self.i_rad.sin_cos();
        DVec3::new(-r * cos_u, -r * sin_u * cos_i, r * sin_u * sin_i)
    }

    /// True anomaly of eccentric anomaly `ea` (Meeus 1998, *Astronomical Algorithms* 2nd ed.,
    /// eq. 30.1): tan(f/2) = √((1 + e)/(1 − e)) tan(E/2), quadrant kept by atan2.
    fn true_anomaly(&self, ea: f64) -> f64 {
        let half = 0.5 * ea;
        2.0 * ((1.0 + self.e).sqrt() * half.sin()).atan2((1.0 - self.e).sqrt() * half.cos())
    }

    /// `n` points [m] of the orbit relative to the host, ICRS axes, closed (module docs);
    /// `host_dir` points from the Sun to the host. Empty for a zero or NaN `host_dir`.
    pub(crate) fn points(&self, host_dir: DVec3, n: usize) -> Vec<DVec3> {
        let n = n.max(3);
        let Some(axes) = self.axes(host_dir) else {
            return Vec::new();
        };
        let mut pts: Vec<DVec3> = ellipse_anomalies(self.e, n)
            .map(|ea| axes * self.sky_xyz(self.true_anomaly(ea)))
            .collect();
        pts.push(pts[0]);
        pts
    }

    /// The planet relative to the host at `t` [m], ICRS axes: transit at `t_transit_jd`, then
    /// the mean motion 2π/P. `None` without a period or a transit time.
    pub(crate) fn position_at(&self, host_dir: DVec3, t: Epoch) -> Option<DVec3> {
        let (period, t_transit) = (self.period_d?, self.t_transit_jd?);
        let axes = self.axes(host_dir)?;
        // The transit is at f = 90° − ω (Winn 2010 §2.1, f_tra = π/2 − ω); its mean anomaly by
        // Kepler's equation M = E − e sin E with tan(E/2) = √((1 − e)/(1 + e)) tan(f/2).
        let f_transit = FRAC_PI_2 - self.omega_rad;
        let half = 0.5 * f_transit;
        let e_transit =
            2.0 * ((1.0 - self.e).sqrt() * half.sin()).atan2((1.0 + self.e).sqrt() * half.cos());
        let m_transit = e_transit - self.e * e_transit.sin();
        // Days since the transit, taken from J2000 so the JD's large offset cancels exactly.
        let days = t.days_since_j2000() - (t_transit - JD_J2000);
        let m = m_transit + TAU * (days / period).fract();
        let f = self.true_anomaly(solve_kepler(m, self.e));
        Some(axes * self.sky_xyz(f))
    }
}

/// `n` points [m] of planet `planet` of `sys`'s orbit around its host, relative to the host,
/// ICRS axes, with the sky-plane +Z along `-host_icrs_dir` (toward the Sun); and whether its
/// orientation is assumed (module docs). Closed: the last point repeats the first. No points for
/// a planet the archive gives no semi-major axis (or e ≥ 1); every archive orbit is assumed,
/// since the archive has no Ω.
pub fn exo_orbit_points(
    sys: &uni_data::ExoSystem,
    planet: usize,
    host_icrs_dir: DVec3,
    n: usize,
) -> (Vec<DVec3>, bool) {
    match sys.planets.get(planet).and_then(|p| SkyOrbit::new(p, None)) {
        Some(orbit) => (orbit.points(host_icrs_dir, n), orbit.assumed),
        None => (Vec::new(), true),
    }
}

/// Where planet `planet` of `sys` is at `t` relative to its host [m], ICRS axes (as
/// [`exo_orbit_points`]): at f = 90° − ω, in front of the star, at its published transit time
/// `t_transit_jd`, moved on by its period. The archive's transit times are mostly BJD (TDB),
/// times when an observer at the Solar System barycentre saw the transit, so the dot shows the
/// planet as seen from the Sun, light time included. `None` without a semi-major axis, a period
/// or a transit time.
pub fn exo_planet_position(
    sys: &ExoSystem,
    planet: usize,
    host_icrs_dir: DVec3,
    t: Epoch,
) -> Option<DVec3> {
    SkyOrbit::new(sys.planets.get(planet)?, None)?.position_at(host_icrs_dir, t)
}

/// The smallest projected semi-major axis [pt] an exoplanet orbit is drawn with (Task C6 fix round
/// 1, I2; context item 7): a smaller one is a ring of dots stacked on the host's glyph.
pub(crate) const EXO_ORBIT_MIN_SEMI_MAJOR_PT: f64 = 3.0;

/// The lines of one exoplanet orbit around a host at `host_rel` [m from the camera]: the orbit
/// (dashed when assumed) and the planet's dot at `t` when it is in the frame. None when the
/// orbit's semi-major axis projects under [`EXO_ORBIT_MIN_SEMI_MAJOR_PT`] (on the frame's axis).
pub(crate) fn exo_orbit_lines(
    map: &ScreenMap,
    host_rel: DVec3,
    host_dir: DVec3,
    orbit: &SkyOrbit,
    t: Epoch,
) -> Vec<OrbitLine> {
    let line = |pts| OrbitLine {
        pts,
        dashed: orbit.assumed,
        tone: LabelTone::Neutral,
    };
    if !map.may_see_sphere(map.camera(host_rel), orbit.a_m * (1.0 + orbit.e))
        || map.size_pt(orbit.a_m, host_rel.length()) < EXO_ORBIT_MIN_SEMI_MAJOR_PT
    {
        return Vec::new();
    }
    let pts = orbit.points(host_dir, theme::ORBIT_POINTS);
    let mut lines: Vec<OrbitLine> = project_polyline(map, host_rel, &pts)
        .into_iter()
        .map(line)
        .collect();
    let dot = orbit
        .position_at(host_dir, t)
        .and_then(|p| map.project(map.camera(host_rel) + map.camera(p)))
        .filter(|p| p.x >= 0.0 && p.y >= 0.0 && p.x <= map.output.x && p.y <= map.output.y);
    if let Some(p) = dot {
        lines.push(line(vec![p.as_vec2()]));
    }
    lines
}

/// The Solar System orbits (module docs) seen from `cam`: each the osculating conic of the
/// body's state about its parent, with μ = G(M + m) of the pair (relative two-body motion;
/// Murray & Dermott 1999, *Solar System Dynamics* §2.2). The planets and dwarf planets come from
/// `snap.destinations` (always destinations, lead ruling B.3); the registry is searched for
/// moons only when a planet or moon is selected. An orbit whose bounding sphere (apoapsis
/// distance about its parent) is wholly outside the view is skipped before it is sampled.
pub(crate) fn solar_system_orbits(
    world: &World,
    snap: &Snapshot,
    cam: UniversePos,
    map: &ScreenMap,
    selected: Option<ObjectId>,
    planet_orbits: bool,
    out: &mut Vec<OrbitLine>,
) {
    let objects = world.objects();
    let selected_info = selected.and_then(|id| objects.get(id as usize));
    let mut drawn: Vec<ObjectId> = Vec::new();
    if let Some(info) = selected_info
        && matches!(
            info.kind,
            ObjectKind::Planet
                | ObjectKind::DwarfPlanet
                | ObjectKind::Moon
                | ObjectKind::Comet
                | ObjectKind::Asteroid
        )
    {
        drawn.push(info.id);
    }
    // The major moons of a selected planet always; of a selected moon's planet (its siblings)
    // only with every planet orbit on.
    let moon_host = selected_info.and_then(|info| match info.kind {
        ObjectKind::Planet | ObjectKind::DwarfPlanet => Some(info.id),
        ObjectKind::Moon if planet_orbits => info.parent,
        _ => None,
    });
    {
        let planets = snap
            .destinations
            .iter()
            .filter(|d| {
                planet_orbits && matches!(d.kind, ObjectKind::Planet | ObjectKind::DwarfPlanet)
            })
            .map(|d| d.id);
        let moons: Vec<ObjectId> = match moon_host {
            Some(host) => objects
                .iter()
                .filter(|info| {
                    info.kind == ObjectKind::Moon
                        && info.parent == Some(host)
                        && info.naif.is_some_and(|n| MAJOR_MOONS.contains(&n))
                })
                .map(|info| info.id)
                .collect(),
            None => Vec::new(),
        };
        for id in planets.chain(moons) {
            if !drawn.contains(&id) {
                drawn.push(id);
            }
        }
    }
    let epoch = world.epoch();
    // Each parent's position and velocity, once (the Sun is the parent of most).
    let mut parents: Vec<(ObjectId, UniversePos, DVec3)> = Vec::new();
    for id in drawn {
        let Some(info) = objects.get(id as usize) else {
            continue;
        };
        let Some(parent) = info.parent.and_then(|p| objects.get(p as usize)) else {
            continue;
        };
        let (parent_pos, parent_vel) = match parents.iter().find(|p| p.0 == parent.id) {
            Some(&(_, pos, vel)) => (pos, vel),
            None => {
                let state = (world.position(parent.id), world.velocity(parent.id));
                parents.push((parent.id, state.0, state.1));
                state
            }
        };
        let r = world.position(id).delta_from(parent_pos);
        let v = world.velocity(id) - parent_vel;
        let mu = parent.gm_m3s2 + info.gm_m3s2;
        if !(mu > 0.0 && r.is_finite() && v.is_finite()) {
            continue;
        }
        let conic = Conic::from_state(r, v, mu, epoch);
        let origin = parent_pos.delta_from(cam);
        if !map.may_see_sphere(map.camera(origin), conic_reach(&conic)) {
            continue;
        }
        let pts = conic_points(&conic, theme::ORBIT_POINTS);
        let tone = if selected == Some(id) {
            LabelTone::Nav
        } else {
            LabelTone::Neutral
        };
        out.extend(
            project_polyline(map, origin, &pts)
                .into_iter()
                .map(|pts| OrbitLine {
                    pts,
                    dashed: false,
                    tone,
                }),
        );
    }
}

/// The farthest any point of [`conic_points`]'s polyline lies from the focus [m]: the apoapsis
/// distance q(1 + e)/(1 − e) of an ellipse, the drawn reach of an open conic.
fn conic_reach(c: &Conic) -> f64 {
    if c.e < 1.0 {
        c.q_m * (1.0 + c.e) / (1.0 - c.e)
    } else {
        (OPEN_ORBIT_REACH_Q * c.q_m).max(OPEN_ORBIT_REACH_AU * AU)
    }
}

/// The orbits of every planet around a selected exoplanet host and, with `nearby`
/// (`OverlaySettings::exo_orbits`, off in the app: coordinator ruling at Task C6), around each
/// host within [`EXO_ORBIT_RANGE_M`] of `cam`. A host that near is in `snap.star_surfaces` when
/// resolved, or a point in `snap.points` while on screen (within uni-sim's 64 px point margin): a
/// host drawn as a point just beyond that margin loses its orbits, which then reach at most a
/// few degrees into the frame.
pub(crate) fn exo_orbits(
    world: &World,
    snap: &Snapshot,
    cam: UniversePos,
    map: &ScreenMap,
    selected: Option<ObjectId>,
    nearby: bool,
    out: &mut Vec<OrbitLine>,
) {
    let objects = world.objects();
    let is_host = |id: ObjectId| {
        objects
            .get(id as usize)
            .is_some_and(|o| o.kind == ObjectKind::ExoHost)
    };
    let near = |id: ObjectId| world.position(id).delta_from(cam).length() <= EXO_ORBIT_RANGE_M;
    let mut hosts: Vec<ObjectId> = selected.filter(|&id| is_host(id)).into_iter().collect();
    let candidates = snap
        .star_surfaces
        .iter()
        .map(|s| s.id)
        .chain(snap.points.iter().filter_map(|p| p.id))
        .chain(snap.destinations.iter().map(|d| d.id))
        .filter(|_| nearby);
    for id in candidates {
        if !hosts.contains(&id) && is_host(id) && near(id) {
            hosts.push(id);
        }
    }
    let epoch = world.epoch();
    for host in hosts {
        let Some(ObjectDetails::ExoHost { system, .. }) = world.details(host) else {
            continue;
        };
        let pos = world.position(host);
        let host_rel = pos.delta_from(cam);
        // Sun → host: the SSB stands in for the Sun (≤ 0.01 AU apart, below 10⁻⁷ rad at a
        // parsec).
        let host_dir = pos.delta_from(UniversePos::default());
        for planet in &system.planets {
            if let Some(orbit) = SkyOrbit::new(planet, None) {
                out.extend(exo_orbit_lines(map, host_rel, host_dir, &orbit, epoch));
            }
        }
    }
}

/// Margin around a disk inside which no orbit is drawn [pt].
pub(crate) const DISK_CLIP_MARGIN_PT: f32 = 2.0;

/// `lines` with every part inside a disk of `disks` ((centre, radius) [pt]) cut out (coordinator
/// ruling at Task C6: never over a body's disk; a body hides the orbits behind it). A line cut
/// in its middle becomes two; a dot inside a disk goes.
pub(crate) fn clip_out_disks(lines: Vec<OrbitLine>, disks: &[(Vec2, f32)]) -> Vec<OrbitLine> {
    if disks.is_empty() {
        return lines;
    }
    cut_lines(
        lines,
        |p| disks.iter().any(|&(c, r)| p.distance(c) < r),
        |a, b| outside_intervals(a, b, disks),
    )
}

/// `lines` with every part inside a rect of `rects` cut out (lead ruling C4w r3: the overlay
/// never draws over a HUD panel; the caller grows the panels' rects by a margin); a dot inside
/// one goes.
pub(crate) fn clip_out_rects(lines: Vec<OrbitLine>, rects: &[egui::Rect]) -> Vec<OrbitLine> {
    if rects.is_empty() {
        return lines;
    }
    cut_lines(
        lines,
        |p| rects.iter().any(|r| r.contains(egui::pos2(p.x, p.y))),
        |a, b| outside_rect_intervals(a, b, rects),
    )
}

/// Splits each line into the runs `keep` leaves of its segments; a one-point line stays unless
/// `inside` holds there.
fn cut_lines(
    lines: Vec<OrbitLine>,
    inside: impl Fn(Vec2) -> bool,
    keep: impl Fn(Vec2, Vec2) -> Vec<(f32, f32)>,
) -> Vec<OrbitLine> {
    let mut out = Vec::new();
    for line in lines {
        let OrbitLine { pts, dashed, tone } = line;
        let mut runs: Vec<Vec<Vec2>> = Vec::new();
        let mut run: Vec<Vec2> = Vec::new();
        if pts.len() == 1 {
            if !inside(pts[0]) {
                runs.push(pts);
            }
        } else {
            for seg in pts.windows(2) {
                let (a, b) = (seg[0], seg[1]);
                for (t0, t1) in keep(a, b) {
                    let (p0, p1) = (a + (b - a) * t0, a + (b - a) * t1);
                    let joins = run.last().is_some_and(|&q| q.distance(p0) < 1e-4);
                    if !joins {
                        if run.len() >= 2 {
                            runs.push(std::mem::take(&mut run));
                        }
                        run.clear();
                        run.push(p0);
                    }
                    run.push(p1);
                }
            }
            if run.len() >= 2 {
                runs.push(run);
            }
        }
        out.extend(runs.into_iter().map(|pts| OrbitLine { pts, dashed, tone }));
    }
    out
}

/// The parameter intervals [t0, t1] ⊂ [0, 1] of segment a → b outside every disk, in order.
fn outside_intervals(a: Vec2, b: Vec2, disks: &[(Vec2, f32)]) -> Vec<(f32, f32)> {
    let d = b - a;
    let dd = d.length_squared();
    // The inside intervals, merged.
    let mut cuts: Vec<(f32, f32)> = Vec::new();
    for &(c, r) in disks {
        let f = a - c;
        if !(dd > 0.0) {
            if f.length() < r {
                return Vec::new();
            }
            continue;
        }
        // |f + t d|² = r²: t² dd + 2t f·d + |f|² − r² = 0.
        let bh = f.dot(d);
        let disc = bh * bh - dd * (f.length_squared() - r * r);
        if disc <= 0.0 {
            continue;
        }
        let s = disc.sqrt();
        let (t0, t1) = (((-bh - s) / dd).max(0.0), ((-bh + s) / dd).min(1.0));
        if t0 < t1 {
            cuts.push((t0, t1));
        }
    }
    complement(cuts)
}

/// The parameter intervals [t0, t1] ⊂ [0, 1] of segment a → b outside every rect (Liang and
/// Barsky 1984's parametric clip gives the part inside each), in order.
fn outside_rect_intervals(a: Vec2, b: Vec2, rects: &[egui::Rect]) -> Vec<(f32, f32)> {
    let d = b - a;
    let mut cuts = Vec::new();
    for r in rects {
        let (mut t0, mut t1) = (0.0_f32, 1.0_f32);
        let mut hit = true;
        for (p, q) in [
            (-d.x, a.x - r.min.x),
            (d.x, r.max.x - a.x),
            (-d.y, a.y - r.min.y),
            (d.y, r.max.y - a.y),
        ] {
            if p == 0.0 {
                if q < 0.0 {
                    hit = false;
                    break;
                }
            } else if p < 0.0 {
                t0 = t0.max(q / p);
            } else {
                t1 = t1.min(q / p);
            }
        }
        if hit && t0 < t1 {
            cuts.push((t0, t1));
        }
    }
    complement(cuts)
}

/// [0, 1] less the union of `cuts`, in order.
fn complement(mut cuts: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut keep = Vec::new();
    let mut from = 0.0_f32;
    for (t0, t1) in cuts {
        if t0 > from {
            keep.push((from, t0));
        }
        from = from.max(t1);
    }
    if from < 1.0 {
        keep.push((from, 1.0));
    }
    keep
}

/// Paints `lines`: antialiased 1 pt polylines in their tone at 35 % alpha, dashed where
/// assumed; one-point lines as dots.
pub(crate) fn draw(painter: &Painter, lines: &[OrbitLine]) {
    for line in lines {
        let color = theme::tone_color(line.tone);
        let pts: Vec<Pos2> = line.pts.iter().map(|p| egui::pos2(p.x, p.y)).collect();
        match pts.as_slice() {
            [] => {}
            [dot] => {
                painter.circle_filled(*dot, EXO_DOT_RADIUS_PT, color.gamma_multiply(EXO_DOT_ALPHA));
            }
            _ => {
                let stroke = Stroke::new(
                    theme::ORBIT_STROKE,
                    color.gamma_multiply(theme::ORBIT_ALPHA),
                );
                if line.dashed {
                    painter.extend(Shape::dashed_line(&pts, stroke, DASH_PT, GAP_PT));
                } else {
                    painter.add(Shape::line(pts, stroke));
                }
            }
        }
    }
}

#[cfg(test)]
mod clip_tests {
    use super::*;

    /// Lead ruling C4w r3: an orbit line is cut out of a HUD panel's rect where it crosses it
    /// (straight through, from a corner, along an edge), a line or dot wholly inside goes, and
    /// one beside the rect stays whole.
    #[test]
    fn orbit_lines_are_cut_out_of_panels() {
        let line = |pts: Vec<Vec2>| OrbitLine {
            pts,
            dashed: true,
            tone: LabelTone::NonPhysical,
        };
        let card = egui::Rect::from_min_max(egui::pos2(100.0, 0.0), egui::pos2(200.0, 50.0));
        // Through: two runs, ending on the rect's sides; dashing and tone kept.
        let out = clip_out_rects(
            vec![line(vec![Vec2::new(0.0, 20.0), Vec2::new(300.0, 20.0)])],
            &[card],
        );
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].pts, [Vec2::new(0.0, 20.0), Vec2::new(100.0, 20.0)]);
        assert_eq!(out[1].pts, [Vec2::new(200.0, 20.0), Vec2::new(300.0, 20.0)]);
        assert!(
            out.iter()
                .all(|l| l.dashed && l.tone == LabelTone::NonPhysical)
        );
        // A polyline entering through the bottom: cut where it enters, the rest kept joined.
        let out = clip_out_rects(
            vec![line(vec![
                Vec2::new(150.0, 25.0),
                Vec2::new(150.0, 100.0),
                Vec2::new(250.0, 100.0),
            ])],
            &[card],
        );
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(
            out[0].pts,
            [
                Vec2::new(150.0, 50.0),
                Vec2::new(150.0, 100.0),
                Vec2::new(250.0, 100.0)
            ]
        );
        // Inside, or a dot inside: gone. Beside: whole.
        let inside = line(vec![Vec2::new(120.0, 10.0), Vec2::new(180.0, 40.0)]);
        let dot = line(vec![Vec2::new(150.0, 25.0)]);
        let beside = line(vec![Vec2::new(0.0, 60.0), Vec2::new(300.0, 60.0)]);
        let out = clip_out_rects(vec![inside, dot, beside.clone()], &[card]);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].pts, beside.pts);
    }

    /// Coordinator ruling at Task C6: an orbit line crossing a disk is cut where it enters and
    /// leaves it (with the margin the caller adds), a line wholly inside goes, a dot inside goes.
    #[test]
    fn orbit_lines_are_cut_out_of_disks() {
        let line = |pts: Vec<Vec2>| OrbitLine {
            pts,
            dashed: false,
            tone: LabelTone::Neutral,
        };
        let disks = [(Vec2::new(50.0, 0.0), 10.0)];
        let out = clip_out_disks(
            vec![line(vec![Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)])],
            &disks,
        );
        assert_eq!(out.len(), 2);
        assert!((out[0].pts[1].x - 40.0).abs() < 1e-3, "{:?}", out[0].pts);
        assert!((out[1].pts[0].x - 60.0).abs() < 1e-3, "{:?}", out[1].pts);
        // A polyline through the disk keeps its outside vertices, joined.
        let out = clip_out_disks(
            vec![line(vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(20.0, 0.0),
                Vec2::new(45.0, 0.0),
                Vec2::new(70.0, 5.0),
                Vec2::new(90.0, 30.0),
            ])],
            &disks,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].pts.len(), 3);
        assert!(
            out.iter()
                .all(|l| l.pts.iter().all(|p| p.distance(disks[0].0) >= 10.0 - 1e-3))
        );
        // Wholly inside, and a dot inside: nothing left; a line missing the disk: unchanged.
        assert!(
            clip_out_disks(
                vec![line(vec![Vec2::new(48.0, 0.0), Vec2::new(52.0, 1.0)])],
                &disks
            )
            .is_empty()
        );
        assert!(clip_out_disks(vec![line(vec![Vec2::new(50.0, 1.0)])], &disks).is_empty());
        let miss = vec![Vec2::new(0.0, 20.0), Vec2::new(100.0, 20.0)];
        let out = clip_out_disks(vec![line(miss.clone())], &disks);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pts, miss);
    }
}

#[cfg(test)]
mod tests {
    use glam::{DQuat, UVec2};
    use uni_core::{CameraView, DAY, GM_SUN};

    use super::*;

    const INTERNAL: UVec2 = UVec2::new(1662, 1080);
    const OUTPUT: Vec2 = Vec2::new(1470.0, 956.0);

    fn conic(q_au: f64, e: f64, i_deg: f64, node_deg: f64, argp_deg: f64) -> Conic {
        Conic {
            q_m: q_au * AU,
            e,
            i_rad: i_deg.to_radians(),
            node_rad: node_deg.to_radians(),
            argp_rad: argp_deg.to_radians(),
            tp: Epoch::J2000,
            mu_m3s2: GM_SUN,
        }
    }

    /// The conic's radius at the true anomaly of `p`: r = q(1 + e)/(1 + e cos f).
    fn conic_radius_at(c: &Conic, p: DVec3) -> f64 {
        let back = perifocal_to_reference(c.i_rad, c.node_rad, c.argp_rad).transpose() * p;
        let f = back.y.atan2(back.x);
        c.q_m * (1.0 + c.e) / (1.0 + c.e * f.cos())
    }

    /// A camera at the root origin looking along −Z, +Y up, fov 60°.
    fn camera_at_origin() -> CameraView {
        CameraView {
            pos: UniversePos::default(),
            orient: DQuat::IDENTITY,
            fov_y_rad: 60f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        }
    }

    fn planet(i_deg: Option<f64>, e: Option<f64>, argp_deg: Option<f64>) -> ExoPlanet {
        ExoPlanet {
            name: "Test b".into(),
            a_au: Some(0.05),
            e,
            i_deg,
            argp_deg,
            period_d: Some(4.0),
            t_transit_jd: Some(2_457_000.25),
            mass_mjup: None,
            radius_rjup: None,
            teq_k: None,
            method: "Transit".into(),
            year: 2020,
        }
    }

    fn system(planets: Vec<ExoPlanet>) -> ExoSystem {
        ExoSystem {
            host: "Test".into(),
            star_index: None,
            ra_deg: 0.0,
            dec_deg: 0.0,
            dist_pc: 10.0,
            v_mag: None,
            teff_k: None,
            radius_rsun: None,
            planets,
        }
    }

    /// Celestial north and east on the sky at a host in direction `dir` from the Sun.
    fn north_east(dir: DVec3) -> (DVec3, DVec3) {
        let north = (DVec3::Z - DVec3::Z.dot(dir) * dir).normalize();
        (north, north.cross(dir))
    }

    /// The distance [pt] from `p` to the nearest segment of any of `runs`.
    fn distance_to_runs(runs: &[Vec<Vec2>], p: Vec2) -> f32 {
        let mut best = f32::INFINITY;
        for run in runs {
            for w in run.windows(2) {
                let (a, b) = (w[0], w[1]);
                let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
                best = best.min(p.distance(a + (b - a) * t));
            }
        }
        best
    }

    /// Plan C5 step 1: the polyline is the conic. Closed for e < 1; every point on the conic's
    /// radius at its true anomaly to 1e-9 (relative) and in its plane; open conics too.
    #[test]
    fn orbit_polyline_matches_conic() {
        let closed = [
            conic(0.983, 0.0167, 0.0, 0.0, 102.9), // Earth-like, equatorial
            conic(1.381, 0.0934, 1.85, 49.6, 286.5), // Mars-like
            conic(0.586, 0.967, 162.3, 58.4, 111.3), // Halley-like, retrograde
            conic(4.95, 0.0, 30.0, 10.0, 0.0),     // circular
            conic(0.5, 0.9995, 70.0, 120.0, 200.0), // long-period comet, Q ≈ 2000 AU
        ];
        let open = [
            conic(1.0, 1.0, 20.0, 0.0, 0.0),
            conic(0.3, 1.2, 50.0, 40.0, 30.0),
        ];
        for (c, is_closed) in closed
            .iter()
            .map(|c| (c, true))
            .chain(open.iter().map(|c| (c, false)))
        {
            let pts = conic_points(c, 256);
            assert_eq!(pts.len(), 256);
            assert_eq!(pts[0] == pts[255], is_closed, "closed iff e < 1: {c:?}");
            let normal = perifocal_to_reference(c.i_rad, c.node_rad, c.argp_rad) * DVec3::Z;
            for p in &pts {
                let want = conic_radius_at(c, *p);
                assert!(
                    ((p.length() - want) / want).abs() < 1e-9,
                    "{c:?}: {} vs {want}",
                    p.length()
                );
                assert!(
                    p.dot(normal).abs() < 1e-9 * p.length(),
                    "out of the orbit plane"
                );
            }
            let nearest = pts.iter().map(|p| p.length()).fold(f64::INFINITY, f64::min);
            assert!(
                (nearest - c.q_m).abs() < 1e-3 * c.q_m,
                "periapsis drawn: {nearest} vs {}",
                c.q_m
            );
            if is_closed {
                // Every segment turns the same 2π/255 (equal turning of the tangent), so the
                // apsides of the e = 0.9995 orbit are as smooth as a circle's.
                let step = TAU / 255.0;
                for w in pts.windows(3) {
                    let turn = (w[1] - w[0]).angle_between(w[2] - w[1]);
                    assert!(turn < 1.5 * step, "{c:?}: segment turns {turn} rad");
                }
            }
        }
        assert!(
            conic_points(
                &Conic {
                    q_m: 0.0,
                    ..closed[0]
                },
                256
            )
            .is_empty()
        );
    }

    /// Plan C5 step 1: an orbit around the camera, half of it behind: no drawn segment is longer
    /// than the screen diagonal or leaves the frame, and the half in front is drawn where it
    /// projects. Without the split, a segment from in front to behind would cross the screen.
    #[test]
    fn near_plane_split() {
        let view = camera_at_origin();
        let map = ScreenMap::new(&view, INTERNAL, OUTPUT);
        // A 1 AU circle in the x–z plane, 0.05 AU below the camera.
        let c = conic(1.0, 0.0, 90.0, 0.0, 0.0);
        let pts = conic_points(&c, 256);
        let origin = DVec3::new(0.0, -0.05 * AU, 0.0);
        let runs = project_polyline(&map, origin, &pts);
        assert!(!runs.is_empty());
        let diagonal = OUTPUT.length();
        for run in &runs {
            assert!(run.len() >= 2);
            for p in run {
                assert!(p.is_finite());
                assert!(
                    p.x >= -1e-3
                        && p.y >= -1e-3
                        && p.x <= OUTPUT.x + 1e-3
                        && p.y <= OUTPUT.y + 1e-3
                );
            }
            for w in run.windows(2) {
                assert!(
                    w[0].distance(w[1]) <= diagonal * (1.0 + 1e-6),
                    "{} pt",
                    w[0].distance(w[1])
                );
            }
        }
        // The point straight ahead, 1 AU out and 0.05 AU down: 735 pt across, and
        // 0.5·(1 + 0.05/tan 30°)·1080 px · (956/1080) = 519.4 pt down.
        let ahead = Vec2::new(
            735.0,
            (0.5 * (1.0 + 0.05 / 30f64.to_radians().tan()) * 956.0) as f32,
        );
        assert!(
            distance_to_runs(&runs, ahead) < 0.5,
            "{}",
            distance_to_runs(&runs, ahead)
        );
        // The same points projected without the split (behind the camera mirrored through it).
        let naive: Vec<Vec2> = pts
            .iter()
            .map(|&p| {
                let q = origin + p;
                let ndc =
                    DVec2::new(q.x / (1662.0 / 1080.0), q.y) / (-q.z * 30f64.to_radians().tan());
                (DVec2::new(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * OUTPUT.as_dvec2()).as_vec2()
            })
            .collect();
        let longest = naive
            .windows(2)
            .map(|w| w[0].distance(w[1]))
            .fold(0.0, f32::max);
        assert!(longest > diagonal, "the case is real: {longest} pt");
    }

    /// Plan C5 step 1 (Winn 2010 §2.1): Ω 0, i 90°: the orbit plane holds the line of sight and
    /// celestial north; i 0: the plane is the sky plane, ⟂ the line of sight.
    #[test]
    fn exo_sky_plane_geometry() {
        let dir = DVec3::new(0.3, -0.8, 0.52).normalize();
        let (north, east) = north_east(dir);
        let edge_on = system(vec![planet(Some(90.0), Some(0.2), Some(40.0))]);
        let (pts, assumed) = exo_orbit_points(&edge_on, 0, dir, 256);
        assert!(assumed, "the archive has no Ω");
        assert_eq!(pts.len(), 256);
        assert_eq!(pts[0], pts[255]);
        let a = 0.05 * AU;
        for p in &pts {
            assert!(
                p.dot(east).abs() < 1e-9 * p.length(),
                "out of the north–line-of-sight plane"
            );
            assert!(p.length() > 0.8 * a * (1.0 - 1e-9) && p.length() < 1.2 * a * (1.0 + 1e-9));
        }
        assert!(
            pts.iter().any(|p| p.dot(north).abs() > 0.9 * p.length()),
            "reaches north"
        );
        assert!(
            pts.iter().any(|p| p.dot(dir).abs() > 0.9 * p.length()),
            "reaches the line of sight"
        );
        let face_on = system(vec![planet(Some(0.0), Some(0.2), Some(40.0))]);
        let (pts, _) = exo_orbit_points(&face_on, 0, dir, 256);
        assert_eq!(pts.len(), 256);
        for p in &pts {
            assert!(p.dot(dir).abs() < 1e-9 * p.length(), "out of the sky plane");
        }
        // No semi-major axis: nothing drawn. An index past the planets: nothing either.
        let no_a = system(vec![ExoPlanet {
            a_au: None,
            ..planet(Some(90.0), None, None)
        }]);
        assert!(exo_orbit_points(&no_a, 0, dir, 256).0.is_empty());
        assert!(exo_orbit_points(&edge_on, 3, dir, 256).0.is_empty());
    }

    /// Plan C5 step 1, review m9: at t_transit the planet is between its star and the Sun (Z
    /// toward the observer > 0), offset on the sky by the impact parameter r cos i; half a period
    /// later it is behind the star (Z < 0); a thousand periods later it transits again.
    #[test]
    fn exo_transit_in_front_of_star() {
        let dir = DVec3::new(-0.6, 0.2, -0.77).normalize();
        let toward_sun = -dir;
        let t_transit = Epoch::from_days_since_j2000(2_457_000.25 - JD_J2000);
        for (e, argp) in [
            (None, None),
            (Some(0.3), Some(40.0)),
            (Some(0.1), Some(250.0)),
            (Some(0.005), Some(200.0)),
        ] {
            let sys = system(vec![planet(Some(88.0), e, argp)]);
            let at = exo_planet_position(&sys, 0, dir, t_transit).expect("placed");
            let z = at.dot(toward_sun);
            assert!(z > 0.0, "e {e:?} ω {argp:?}: in front, z = {z}");
            let r = at.length();
            let sky = (at - z * toward_sun).length();
            assert!(
                (sky - r * 88f64.to_radians().cos()).abs() < 1e-9 * r,
                "impact parameter: {sky}"
            );
            let later = exo_planet_position(&sys, 0, dir, t_transit.add_seconds(2.0 * DAY))
                .expect("placed");
            assert!(
                later.dot(toward_sun) < 0.0,
                "e {e:?} ω {argp:?}: behind half a period later"
            );
            let again =
                exo_planet_position(&sys, 0, dir, t_transit.add_seconds(1000.0 * 4.0 * DAY))
                    .expect("placed");
            assert!(
                (again - at).length() < 1e-6 * r,
                "periodic: {}",
                (again - at).length()
            );
            // The dot lies on the drawn orbit's ellipse: between periastron and apastron.
            let (a, ecc) = (0.05 * AU, e.unwrap_or(0.0));
            assert!(r >= a * (1.0 - ecc) * (1.0 - 1e-9) && r <= a * (1.0 + ecc) * (1.0 + 1e-9));
        }
        // Lead ruling B.2: ω = 90° when missing or e < 0.01, else the archive's value, read as
        // the planet's.
        let omega = |e, argp| {
            SkyOrbit::new(&planet(Some(88.0), e, argp), None)
                .expect("orbit")
                .omega_rad
                .to_degrees()
        };
        assert!((omega(Some(0.005), Some(200.0)) - 90.0).abs() < 1e-12);
        assert!((omega(Some(0.3), None) - 90.0).abs() < 1e-12);
        assert!((omega(Some(0.3), Some(200.0)) - 200.0).abs() < 1e-12);
        // No transit time or no period: no dot.
        let no_t = system(vec![ExoPlanet {
            t_transit_jd: None,
            ..planet(Some(88.0), None, None)
        }]);
        assert!(exo_planet_position(&no_t, 0, dir, t_transit).is_none());
    }

    /// Task C6 fix round 1 (I2, context item 7): an exoplanet orbit whose semi-major axis
    /// projects under 3 pt is not drawn, nor its dot (they would stack on the host's glyph);
    /// just over 3 pt it is. The scale is the 60° frame's: 540 internal px per tan 30°, at
    /// 956/1080 pt per px.
    #[test]
    fn sub_three_point_exo_orbits_are_skipped() {
        let orbit =
            SkyOrbit::new(&planet(Some(85.0), Some(0.1), Some(30.0)), Some(30.0)).expect("orbit");
        let map = ScreenMap::new(&camera_at_origin(), INTERNAL, OUTPUT);
        let pt_per_rad = 540.0 / 30f64.to_radians().tan() * 956.0 / 1080.0;
        let host_dir = DVec3::new(0.6, 0.0, -0.8);
        let t = Epoch::from_days_since_j2000(2_457_000.25 - JD_J2000);
        let lines_at = |a_pt: f64| {
            let dist = orbit.a_m * pt_per_rad / a_pt;
            assert!((map.size_pt(orbit.a_m, dist) - a_pt).abs() < 1e-9 * a_pt);
            exo_orbit_lines(&map, DVec3::new(0.0, 0.0, -dist), host_dir, &orbit, t)
        };
        for a_pt in [0.2, 1.0, 2.9] {
            assert!(lines_at(a_pt).is_empty(), "{a_pt} pt: nothing drawn");
        }
        for a_pt in [3.1, 40.0] {
            let lines = lines_at(a_pt);
            assert!(
                lines.iter().any(|l| l.pts.len() >= 2),
                "{a_pt} pt: orbit drawn"
            );
            assert_eq!(
                lines.iter().filter(|l| l.pts.len() == 1).count(),
                1,
                "{a_pt} pt: dot"
            );
        }
    }

    /// Plan C5 step 1 (ruling 13, lead ruling B.2): Ω missing → dashed (every archive orbit), i
    /// missing → dashed and edge-on; Ω and i known (a synthetic system) → solid. The overlay's
    /// lines carry the flag.
    #[test]
    fn exo_dashed_when_assumed() {
        let dir = DVec3::new(0.0, 0.6, 0.8);
        let known_i = planet(Some(85.0), Some(0.1), Some(30.0));
        let no_i = planet(None, Some(0.1), Some(30.0));
        assert!(
            exo_orbit_points(&system(vec![known_i.clone()]), 0, dir, 64).1,
            "Ω missing"
        );
        let (pts, assumed) = exo_orbit_points(&system(vec![no_i.clone()]), 0, dir, 64);
        assert!(assumed, "i missing");
        let (_, east) = north_east(dir);
        assert!(
            pts.iter().all(|p| p.dot(east).abs() < 1e-9 * p.length()),
            "i = 90° when missing"
        );
        let solid = SkyOrbit::new(&known_i, Some(30.0)).expect("orbit");
        assert!(!solid.assumed);
        assert!(SkyOrbit::new(&no_i, Some(30.0)).expect("orbit").assumed);
        assert!(SkyOrbit::new(&known_i, None).expect("orbit").assumed);
        // A camera 0.3 AU from the host, looking at it (the orbit, 0.05 AU across, fills ±10°).
        let view = camera_at_origin();
        let map = ScreenMap::new(&view, INTERNAL, OUTPUT);
        let host_rel = DVec3::new(0.0, 0.0, -0.3 * AU);
        let host_dir = DVec3::new(0.6, 0.0, -0.8);
        let t = Epoch::from_days_since_j2000(2_457_000.25 - JD_J2000);
        for (orbit, dashed) in [
            (solid, false),
            (SkyOrbit::new(&known_i, None).expect("orbit"), true),
        ] {
            let lines = exo_orbit_lines(&map, host_rel, host_dir, &orbit, t);
            assert!(lines.iter().any(|l| l.pts.len() >= 2), "orbit drawn");
            assert_eq!(
                lines.iter().filter(|l| l.pts.len() == 1).count(),
                1,
                "one planet dot"
            );
            assert!(
                lines
                    .iter()
                    .all(|l| l.dashed == dashed && l.tone == LabelTone::Neutral)
            );
        }
        // The solid orbit's node line lies at position angle 30°, along cos 30° N + sin 30° E:
        // where it crosses the sky plane (Z = 0, so sin(ω + f) = 0 and Y = 0, Winn eqs. 2–4).
        let (north, east) = north_east(host_dir.normalize());
        let x_axis = 30f64.to_radians().cos() * north + 30f64.to_radians().sin() * east;
        let toward_sun = -host_dir.normalize();
        let pts = solid.points(host_dir, 512);
        let mut crossings = 0;
        for w in pts.windows(2) {
            let (za, zb) = (w[0].dot(toward_sun), w[1].dot(toward_sun));
            if za * zb < 0.0 {
                let node = w[0] + (w[1] - w[0]) * (za / (za - zb));
                assert!(
                    node.normalize().dot(x_axis).abs() > 0.999,
                    "{}",
                    node.normalize().dot(x_axis)
                );
                crossings += 1;
            }
        }
        assert_eq!(crossings, 2);
    }
}
