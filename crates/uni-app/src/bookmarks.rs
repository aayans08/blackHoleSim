//! `bookmarks/m1.ron` and `bookmarks/m1-bench.ron`: the M1 capture bookmarks (plan, table "M1
//! capture bookmarks", entries 1–19), how they were computed from the World, and their intent
//! tests (M1 Task B9 Step 5).
//!
//! [`write_m1_bookmarks`] (an ignored test, run by hand) computes every placement numerically
//! from the World at its epoch and prints the RON; the committed files are its output. The
//! intent tests then load the committed files and check each stated geometry from the World
//! (distance, phase angle, apparent radius, ring-plane elevation, Sun angle) to ±1 %, the
//! lit-pixel count against the meter thresholds and the bookmark's "Exposure on paper" row
//! (±0.5 EV, with the CPU twins: `crate::paper`), and every moon in frame other than the anchor
//! against its Horizons spot as seen from the bookmark camera (< 1 px; Saturn's major moons
//! against the measured ceilings, ruling 10).

use std::f64::consts::PI;
use std::fmt::Write as _;

use glam::{DMat3, DQuat, DVec3};
use uni_core::{AU, CameraView, Epoch, PC, UniversePos, gravitational_radius, radec_to_icrs};
use uni_nav::Bookmark;

use crate::world::World;
use crate::world::tests::full_world;

/// The spot epoch of every moon check (ruling 10): 2026-11-20 00:00 UTC.
pub const SPOT_EPOCH_UTC: &str = "2026-11-20T00:00:00Z";
/// Rows of a capture.
const H: f64 = 1080.0;
/// Columns of a capture.
const W: f64 = 1920.0;

fn deg(x: f64) -> f64 {
    x.to_radians()
}

fn epoch(s: &str) -> Epoch {
    Epoch::parse_iso_utc(s).unwrap()
}

/// Centre of `name` (root).
fn pos(world: &World, name: &str) -> UniversePos {
    world
        .position_of(name)
        .unwrap_or_else(|| panic!("no object {name}"))
}

/// `to` − `from` [m].
fn rel(world: &World, from: &str, to: &str) -> DVec3 {
    pos(world, to).delta_from(pos(world, from))
}

/// NAIF id of `name`.
fn naif(world: &World, name: &str) -> i32 {
    let id = world.sim.find(name).unwrap();
    world.sim.objects()[id as usize].naif.unwrap()
}

/// Body `name`'s north pole [ICRS unit].
fn pole(world: &World, name: &str) -> DVec3 {
    uni_physics::rotation::pole_icrs(naif(world, name), world.epoch()).unwrap()
}

/// Two unit vectors completing `a` to an orthonormal basis, the first as close as possible to
/// `hint`.
fn basis(a: DVec3, hint: DVec3) -> (DVec3, DVec3) {
    let e1 = (hint - a * hint.dot(a)).normalize();
    (e1, a.cross(e1))
}

/// Camera → root rotation for a camera looking along `fwd` with its +Y as close as possible to
/// `up`.
fn look(fwd: DVec3, up: DVec3) -> DQuat {
    let f = fwd.normalize();
    let right = f.cross(up).normalize();
    let up = right.cross(f);
    DQuat::from_mat3(&DMat3::from_cols(right, up, -f)).normalize()
}

/// The angle between two vectors [rad].
fn angle(a: DVec3, b: DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b))
}

/// Pixel of the point at `rel` [m from the camera] for a camera `orient` with `fov`.
fn project(orient: DQuat, fov: f64, rel: DVec3) -> Option<(f64, f64)> {
    let c = orient.conjugate() * rel;
    if !(c.z < 0.0) {
        return None;
    }
    let t = (0.5 * fov).tan();
    let x = c.x / (W / H) / (-c.z * t);
    let y = c.y / (-c.z * t);
    Some((0.5 * (x + 1.0) * W, 0.5 * (1.0 - y) * H))
}

/// Projected radius [px] of a sphere (the snapshot's rule).
fn radius_px(r: f64, d: f64, fov: f64) -> f64 {
    uni_sim::snapshot::proj_radius_px(r, d, fov, H)
}

/// One bookmark of the generated file, with the comment above it.
struct Mark {
    comment: String,
    b: Bookmark,
}

fn blank(name: &str, fov: f64, epoch_utc: &str) -> Bookmark {
    Bookmark {
        name: name.to_owned(),
        view: CameraView {
            pos: UniversePos::default(),
            orient: DQuat::IDENTITY,
            fov_y_rad: fov,
            velocity_mps: DVec3::ZERO,
        },
        epoch_utc: Some(epoch_utc.to_owned()),
        scene_ev100: None,
        star_ev100: None,
        lens_scale: None,
        anchor: None,
        look_at: None,
        up: None,
        time_rate: None,
        exposure: None,
        eye_adapted: None,
        target: None,
        lens_env: None,
        cruise: None,
        highlight_belt: None,
    }
}

/// A camera at `offset` [m] from `anchor` looking at it with `up` as its up hint.
fn anchored(
    name: &str,
    fov: f64,
    epoch_utc: &str,
    anchor: &str,
    offset: DVec3,
    up: DVec3,
) -> Bookmark {
    let mut b = blank(name, fov, epoch_utc);
    b.view.pos = UniversePos {
        cell: glam::I64Vec3::ZERO,
        local: offset,
    };
    b.anchor = Some(anchor.to_owned());
    b.look_at = Some(anchor.to_owned());
    b.up = Some(up);
    b
}

/// RON text of `b` (implicit_some; the form `bookmarks/m0.ron` uses).
fn ron(b: &Bookmark) -> String {
    let v = &b.view;
    let mut s = String::new();
    let _ = writeln!(s, "    (");
    let _ = writeln!(s, "        name: {:?},", b.name);
    let _ = writeln!(s, "        view: (");
    let _ = writeln!(
        s,
        "            pos: (cell: ({}, {}, {}), local: ({:?}, {:?}, {:?})),",
        v.pos.cell.x, v.pos.cell.y, v.pos.cell.z, v.pos.local.x, v.pos.local.y, v.pos.local.z
    );
    let _ = writeln!(
        s,
        "            orient: ({:?}, {:?}, {:?}, {:?}),",
        v.orient.x, v.orient.y, v.orient.z, v.orient.w
    );
    let _ = writeln!(
        s,
        "            fov_y_rad: {:?}, // {}°",
        v.fov_y_rad,
        (v.fov_y_rad.to_degrees() * 1e6).round() / 1e6
    );
    let _ = writeln!(s, "        ),");
    if let Some(e) = &b.epoch_utc {
        let _ = writeln!(s, "        epoch_utc: {e:?},");
    }
    if let Some(x) = b.scene_ev100 {
        let _ = writeln!(s, "        scene_ev100: {x:?},");
    }
    if let Some(x) = b.star_ev100 {
        let _ = writeln!(s, "        star_ev100: {x:?},");
    }
    if let Some(x) = b.lens_scale {
        let _ = writeln!(s, "        lens_scale: {x:?},");
    }
    if let Some(x) = &b.anchor {
        let _ = writeln!(s, "        anchor: {x:?},");
    }
    if let Some(x) = &b.look_at {
        let _ = writeln!(s, "        look_at: {x:?},");
    }
    if let Some(u) = b.up {
        let _ = writeln!(s, "        up: ({:?}, {:?}, {:?}),", u.x, u.y, u.z);
    }
    if let Some(x) = b.exposure {
        let _ = writeln!(s, "        exposure: {x:?},");
    }
    if let Some(x) = b.eye_adapted {
        let _ = writeln!(s, "        eye_adapted: {x:?},");
    }
    if let Some(x) = b.lens_env {
        let _ = writeln!(s, "        lens_env: {x:?},");
    }
    if let Some(x) = &b.target {
        let _ = writeln!(s, "        target: {x:?},");
    }
    if let Some(x) = b.cruise {
        let _ = writeln!(s, "        cruise: {x:?},");
    }
    if let Some(x) = b.highlight_belt {
        let _ = writeln!(s, "        highlight_belt: {x:?},");
    }
    let _ = writeln!(s, "    ),");
    s
}

/// The moons theory-vs-Horizons check: for each moon of `moons` in frame of a camera at `cam`
/// [root] with `orient` and `fov`, the pixel distance between its theory position and its
/// Horizons spot (its planet's theory position + the spot's planet-relative vector) at the
/// world's epoch. `None` where no spot exists at that epoch.
fn moon_errors_px(
    world: &World,
    cam: UniversePos,
    orient: DQuat,
    fov: f64,
    planet: &str,
    moons: &[&str],
) -> Vec<(String, Option<f64>)> {
    let spots = uni_data::load_ephem_spots(&uni_data::data_dir().join("golden/horizons_spots.ron"))
        .unwrap();
    let jd = world.epoch().jd_tdb();
    let p = pos(world, planet);
    let p_naif = naif(world, planet);
    let mut out = Vec::new();
    for &m in moons {
        let theory = pos(world, m).delta_from(cam);
        let Some((x, y)) = project(orient, fov, theory) else {
            continue;
        };
        if !(0.0..W).contains(&x) || !(0.0..H).contains(&y) {
            continue;
        }
        let spot = spots.iter().find(|s| {
            s.target == naif(world, m) && s.center == p_naif && (s.jd_tdb - jd).abs() < 1e-3
        });
        let err = spot.and_then(|s| {
            let truth = p.offset(DVec3::from_array(s.pos_km) * 1e3).delta_from(cam);
            let (xs, ys) = project(orient, fov, truth)?;
            Some((xs - x).hypot(ys - y))
        });
        out.push((m.to_owned(), err));
    }
    out
}

/// Whether any of `moons` (or a moon's shadow on the planet) lies on or within `margin_px` of
/// `planet`'s disk for the camera at `cam` with `orient`, `fov`.
fn moon_near_disk(
    world: &World,
    cam: UniversePos,
    orient: DQuat,
    fov: f64,
    planet: &str,
    moons: &[&str],
    margin_px: f64,
) -> Vec<String> {
    let pc = pos(world, planet).delta_from(cam);
    let r_planet = world.sim.objects()[world.sim.find(planet).unwrap() as usize]
        .radii_m
        .x;
    let Some((cx, cy)) = project(orient, fov, pc) else {
        return Vec::new();
    };
    let rp = radius_px(r_planet, pc.length(), fov);
    let sun = rel(world, planet, "Sun").normalize();
    let mut hits = Vec::new();
    for &m in moons {
        let mrel = pos(world, m).delta_from(cam);
        if let Some((x, y)) = project(orient, fov, mrel) {
            let r_m = world.sim.objects()[world.sim.find(m).unwrap() as usize]
                .radii_m
                .x;
            let rm = radius_px(r_m, mrel.length(), fov);
            if (x - cx).hypot(y - cy) < rp + rm + margin_px {
                hits.push(format!("{m} on/near the disk"));
            }
        }
        // The moon's shadow: from the moon along −sun until it meets the planet's sphere.
        let mp = pos(world, m).delta_from(pos(world, planet));
        let b = mp.dot(-sun);
        let c = mp.length_squared() - r_planet * r_planet;
        let disc = b * b - c;
        if disc >= 0.0 && -b - disc.sqrt() > 0.0 {
            let t = -b - disc.sqrt();
            let hit = mp - sun * t;
            // Visible from the camera (on the near hemisphere)?
            if hit.dot(-pc) > 0.0 {
                hits.push(format!("{m}'s shadow on the disk"));
            }
        }
    }
    hits
}

/// The planet's solar ring-plane elevation B′ [rad] (Sun above the equator plane).
fn sun_elevation(world: &World, planet: &str) -> f64 {
    rel(world, planet, "Sun")
        .normalize()
        .dot(pole(world, planet))
        .asin()
}

/// Generates the 19 bookmarks (see the module docs).
fn generate(world: &mut World) -> Vec<Mark> {
    let mut marks = Vec::new();
    let t0 = SPOT_EPOCH_UTC;

    // 1. start: Earth from 60 000 km at phase 60°, the Moon in frame and clear of the disk.
    world.set_epoch(epoch(t0));
    {
        let fov = deg(60.0);
        let s = rel(world, "Earth", "Sun").normalize();
        let n = pole(world, "Earth");
        let (e1, e2) = basis(s, n);
        let r_e: f64 = 6_378_136.6;
        let mut best = (f64::NEG_INFINITY, DVec3::ZERO);
        for k in 0..1440 {
            let psi = deg(0.25 * f64::from(k));
            let c = s * deg(60.0).cos() + (e1 * psi.cos() + e2 * psi.sin()) * deg(60.0).sin();
            let cam = pos(world, "Earth").offset(c * 6.0e7);
            let o = look(-c, n);
            let m = pos(world, "Moon").delta_from(cam);
            let Some((x, y)) = project(o, fov, m) else {
                continue;
            };
            let edge = x.min(W - x).min(y).min(H - y);
            let sep = angle(m, -c) - (r_e / 6.0e7).asin();
            let sep_px = sep / (2.0 * (0.5 * fov).tan() / H);
            let score = edge.min(sep_px - 40.0);
            if score > best.0 {
                best = (score, c);
            }
        }
        let c = best.1;
        let b = anchored("start", fov, t0, "Earth", c * 6.0e7, n);
        marks.push(Mark {
            comment: format!(
                "#1 Earth from 60 000 km (99.9 px), phase 60°, north up; the Moon in frame and \
                 clear of the disk (margin {:.0} px). Window start.",
                best.0
            ),
            b,
        });
    }

    // 2. earth-limb-dawn: 420 km up, 22° of arc into the night side, looking horizontally
    // toward the Sun's azimuth.
    {
        let s = rel(world, "Earth", "Sun").normalize();
        let n = pole(world, "Earth");
        let p = s.cross(n).normalize();
        let u = s * deg(112.0).cos() + p * deg(112.0).sin();
        let id = world.sim.find("Earth").unwrap();
        let radii = world.sim.objects()[id as usize].radii_m;
        let q = uni_physics::rotation::body_to_icrs(399, world.epoch()).unwrap();
        let r = crate::measure::ellipsoid_radius(u, q, radii);
        let f = (s - u * s.dot(u)).normalize();
        let mut b = blank("earth-limb-dawn", deg(60.0), t0);
        b.anchor = Some("Earth".to_owned());
        b.view.pos.local = u * (r + 420.0e3);
        b.view.orient = look(f, u);
        marks.push(Mark {
            comment: "#2 420 km above Earth, 22° of arc past the terminator into the night, \
                      looking horizontally toward the Sun's azimuth: the limb 20.26° below \
                      horizontal at 2 351 km, the Sun ≈ 1.8° below it."
                .to_owned(),
            b,
        });
    }

    // 3. earth-map-closeup: 1 500 km above the Strait of Gibraltar, 30° off nadir toward the
    // Iberian coast (north), fov 30°, at 12:00 UTC.
    {
        let t = "2026-11-20T12:00:00Z";
        world.set_epoch(epoch(t));
        let id = world.sim.find("Earth").unwrap();
        let radii = world.sim.objects()[id as usize].radii_m;
        let q = uni_physics::rotation::body_to_icrs(399, world.epoch()).unwrap();
        let (lat, lon) = (deg(36.0), deg(-5.6));
        let (a, c) = (radii.x, radii.z);
        // Geodetic → body-fixed on the spheroid (e.g. Hofmann-Wellenhof et al., GNSS, §5.6.1).
        let nrad = a * a / (a * a * lat.cos().powi(2) + c * c * lat.sin().powi(2)).sqrt();
        let normal = DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
        let surface = DVec3::new(
            nrad * lat.cos() * lon.cos(),
            nrad * lat.cos() * lon.sin(),
            (c * c / (a * a)) * nrad * lat.sin(),
        );
        let north = DVec3::new(-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos());
        let cam_b = surface + normal * 1.5e6;
        let f_b = -normal * deg(30.0).cos() + north * deg(30.0).sin();
        let mut b = blank("earth-map-closeup", deg(30.0), t);
        b.anchor = Some("Earth".to_owned());
        b.view.pos.local = q * cam_b;
        b.view.orient = look(q * f_b, q * north);
        marks.push(Mark {
            comment: "#3 1 500 km above the Strait of Gibraltar (36.0°N, 5.6°W), 30° off nadir \
                      toward the Iberian coast (north), fov 30°, 12:00 UTC (Sun ≈ 34° up)."
                .to_owned(),
            b,
        });
    }

    // 4. moon-full: phase 1°, 20 000 km, fov 20°, strict exposure; Earth out of frame.
    world.set_epoch(epoch(t0));
    {
        let fov = deg(20.0);
        let s = rel(world, "Moon", "Sun").normalize();
        let n = pole(world, "Moon");
        let (e1, e2) = basis(s, n);
        let earth = rel(world, "Moon", "Earth");
        let mut best = (f64::NEG_INFINITY, DVec3::ZERO);
        for k in 0..360 {
            let psi = deg(f64::from(k));
            let c = s * deg(1.0).cos() + (e1 * psi.cos() + e2 * psi.sin()) * deg(1.0).sin();
            let score = angle(earth - c * 2.0e7, -c);
            if score > best.0 {
                best = (score, c);
            }
        }
        let c = best.1;
        let mut b = anchored("moon-full", fov, t0, "Moon", c * 2.0e7, n);
        b.eye_adapted = Some(false);
        marks.push(Mark {
            comment: format!(
                "#4 the Moon at phase 1° from 20 000 km, fov 20° (a = 267.0 px), strict \
                 exposure; Earth {:.1}° off-axis.",
                best.0.to_degrees()
            ),
            b,
        });
    }

    // 5. mars-reference-angle: a = 425 px at fov 60° (8 210 km), phase 45°, lit limb left,
    // north up.
    {
        let fov = deg(60.0);
        let s = rel(world, "Mars", "Sun").normalize();
        let n = pole(world, "Mars");
        let (e1, e2) = basis(s, n);
        let mut best = (f64::INFINITY, DVec3::ZERO);
        for k in 0..1440 {
            let psi = deg(0.25 * f64::from(k));
            let c = s * deg(45.0).cos() + (e1 * psi.cos() + e2 * psi.sin()) * deg(45.0).sin();
            let o = look(-c, n);
            let sc = o.conjugate() * s;
            let ang = sc.y.atan2(sc.x).abs(); // 0 = right, π = left
            let off = PI - ang;
            if off < best.0 {
                best = (off, c);
            }
        }
        let b = anchored("mars-reference-angle", fov, t0, "Mars", best.1 * 8.21e6, n);
        marks.push(Mark {
            comment: format!(
                "#5 Mars a = 425 px at fov 60° (8 210 km), phase 45°, north up, the Sun {:.1}° \
                 from straight left (lit limb left).",
                best.0.to_degrees()
            ),
            b,
        });
    }

    // 6. jupiter-galileans: 5.0e6 km in Jupiter's equatorial plane on the Sun's side, pole up,
    // fov 30°.
    {
        let fov = deg(30.0);
        let s = rel(world, "Jupiter", "Sun").normalize();
        let n = pole(world, "Jupiter");
        let c = (s - n * s.dot(n)).normalize();
        let b = anchored("jupiter-galileans", fov, t0, "Jupiter", c * 5.0e9, n);
        marks.push(Mark {
            comment: format!(
                "#6 Jupiter from 5.0e6 km (28.8 px) in its equatorial plane, phase {:.2}°, pole \
                 up, fov 30°; all four Galileans in frame.",
                angle(s, c).to_degrees()
            ),
            b,
        });
    }

    // 7. jupiter-disk: 3.0e6 km near the Jupiter–Sun line (phase 0.5°), fov 3.5°, strict.
    {
        let fov = deg(3.5);
        let mut t = epoch(t0);
        let moons = ["Io", "Europa", "Ganymede", "Callisto"];
        let mut note = String::new();
        for h in 0..48 {
            // 0, +1 h, −1 h, +2 h, ...
            let dh = if h % 2 == 0 { h / 2 } else { -(h / 2 + 1) };
            t = epoch(t0).add_seconds(3600.0 * f64::from(dh));
            world.set_epoch(t);
            let s = rel(world, "Jupiter", "Sun").normalize();
            let n = pole(world, "Jupiter");
            let (e1, _) = basis(s, n);
            let c = s * deg(0.5).cos() + e1 * deg(0.5).sin();
            let cam = pos(world, "Jupiter").offset(c * 3.0e9);
            let hits = moon_near_disk(world, cam, look(-c, n), fov, "Jupiter", &moons, 5.0);
            if hits.is_empty() {
                note = format!("{dh:+} h from 2026-11-20 00:00 UTC");
                break;
            }
        }
        let iso = crate::world::iso_utc(t);
        let s = rel(world, "Jupiter", "Sun").normalize();
        let n = pole(world, "Jupiter");
        let (e1, _) = basis(s, n);
        let c = s * deg(0.5).cos() + e1 * deg(0.5).sin();
        let mut b = anchored("jupiter-disk", fov, &iso, "Jupiter", c * 3.0e9, n);
        b.eye_adapted = Some(false);
        marks.push(Mark {
            comment: format!(
                "#7 Jupiter from 3.0e6 km at phase 0.5°, fov 3.5° (a = 421.3 px), pole up, strict \
                 exposure; no moon or moon shadow within 5 px of the disk ({note})."
            ),
            b,
        });
    }

    // 8. saturn-ring-plane: 1.5e6 km, 4° above the lit (south) ring face, on the Sun's azimuth.
    world.set_epoch(epoch(t0));
    {
        let fov = deg(20.0);
        let s = rel(world, "Saturn", "Sun").normalize();
        let n = pole(world, "Saturn");
        let bp = sun_elevation(world, "Saturn");
        let sh = (s - n * s.dot(n)).normalize();
        let el = deg(4.0).copysign(bp);
        let c = sh * el.cos() + n * el.sin();
        let b = anchored("saturn-ring-plane", fov, t0, "Saturn", c * 1.5e9, n);
        marks.push(Mark {
            comment: format!(
                "#8 Saturn from 1.5e6 km, 4° above the lit ring face (B′ = {:.2}°), on the Sun's \
                 azimuth (phase {:.2}°), fov 20° (a = 123.1 px), pole up.",
                bp.to_degrees(),
                angle(s, c).to_degrees()
            ),
            b,
        });
    }

    // 9. saturn-edge-on: in the ring plane at the 2025 equinox (B′ crosses 0), 2.0e6 km, fov
    // 4.3°, strict; no moon or moon shadow within 5 px of the disk.
    {
        let fov = deg(4.3);
        let (mut lo, mut hi) = (epoch("2025-03-01T00:00:00Z"), epoch("2025-07-01T00:00:00Z"));
        world.set_epoch(lo);
        let s_lo = sun_elevation(world, "Saturn").signum();
        for _ in 0..60 {
            let mid = lo.add_seconds(0.5 * hi.seconds_since(lo));
            world.set_epoch(mid);
            if sun_elevation(world, "Saturn").signum() == s_lo {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let crossing = lo;
        let moons = [
            "Mimas",
            "Enceladus",
            "Tethys",
            "Dione",
            "Rhea",
            "Titan",
            "Hyperion",
            "Iapetus",
        ];
        let mut chosen = None;
        for h in 0..(2 * 24 * 30) {
            let dh = if h % 2 == 0 { h / 2 } else { -(h / 2 + 1) };
            let t = crossing.add_seconds(3600.0 * f64::from(dh));
            // Whole hours.
            let t = epoch(&crate::world::iso_utc(t));
            world.set_epoch(t);
            if sun_elevation(world, "Saturn").abs() >= deg(0.5) {
                continue;
            }
            let s = rel(world, "Saturn", "Sun").normalize();
            let n = pole(world, "Saturn");
            let c = (s - n * s.dot(n)).normalize();
            let cam = pos(world, "Saturn").offset(c * 2.0e9);
            if moon_near_disk(world, cam, look(-c, n), fov, "Saturn", &moons, 5.0).is_empty() {
                chosen = Some(t);
                break;
            }
        }
        let t = chosen.expect("an equinox hour with no moon near the disk");
        world.set_epoch(t);
        let iso = crate::world::iso_utc(t);
        let s = rel(world, "Saturn", "Sun").normalize();
        let n = pole(world, "Saturn");
        let c = (s - n * s.dot(n)).normalize();
        let mut b = anchored("saturn-edge-on", fov, &iso, "Saturn", c * 2.0e9, n);
        b.eye_adapted = Some(false);
        marks.push(Mark {
            comment: format!(
                "#9 Saturn edge-on at its 2025 equinox (B′ = 0 at {}; here B′ = {:.3}°), in the \
                 ring plane on the Sun's azimuth (phase {:.3}°), 2.0e6 km, fov 4.3° (a = 433.6 \
                 px), strict; no moon or moon shadow within 5 px of the disk.",
                crate::world::iso_utc(crossing),
                sun_elevation(world, "Saturn").to_degrees(),
                angle(s, c).to_degrees()
            ),
            b,
        });
    }

    // 10. saturn-night-side: phase 150°, 25° above the unlit (north) face, 1.5e6 km, fov 20°.
    world.set_epoch(epoch(t0));
    {
        let fov = deg(20.0);
        let s = rel(world, "Saturn", "Sun").normalize();
        let n = pole(world, "Saturn");
        let bp = sun_elevation(world, "Saturn");
        let el = deg(25.0).copysign(-bp);
        let sh = (s - n * s.dot(n)).normalize();
        let e2 = n.cross(sh);
        // cos 150° = sin B′ sin el + cos B′ cos el cos φ.
        let cos_phi = (deg(150.0).cos() - bp.sin() * el.sin()) / (bp.cos() * el.cos());
        let phi = cos_phi.clamp(-1.0, 1.0).acos();
        let moons = [
            "Mimas",
            "Enceladus",
            "Tethys",
            "Dione",
            "Rhea",
            "Titan",
            "Hyperion",
            "Iapetus",
        ];
        let mut best: Option<(f64, DVec3)> = None;
        for sign in [1.0, -1.0] {
            let h = sh * phi.cos() + e2 * (sign * phi).sin();
            let c = h * el.cos() + n * el.sin();
            let cam = pos(world, "Saturn").offset(c * 1.5e9);
            let worst = moon_errors_px(world, cam, look(-c, n), fov, "Saturn", &moons)
                .into_iter()
                .map(|(m, e)| e.unwrap_or(0.0) / ceiling_px(&m))
                .fold(0.0, f64::max);
            if best.is_none_or(|(w, _)| worst < w) {
                best = Some((worst, c));
            }
        }
        let c = best.unwrap().1;
        let b = anchored("saturn-night-side", fov, t0, "Saturn", c * 1.5e9, n);
        marks.push(Mark {
            comment: format!(
                "#10 Saturn from 1.5e6 km, 25° above the unlit ring face, phase {:.2}°, fov 20°, \
                 pole up.",
                angle(s, c).to_degrees()
            ),
            b,
        });
    }

    // 11, 12: the Sun from the side facing Earth.
    {
        let c = rel(world, "Sun", "Earth").normalize();
        let n = DVec3::Z;
        let b = anchored("sun-closeup", deg(30.0), t0, "Sun", c * 0.02 * AU, n);
        marks.push(Mark {
            comment: "#11 the Sun from 0.02 AU on its Earth-facing side, fov 30° (disk radius \
                      481.8 px), north up."
                .to_owned(),
            b,
        });
        let r_sun = world.sim.objects()[world.sim.find("Sun").unwrap() as usize]
            .radii_m
            .x;
        let b = anchored(
            "sun-granulation",
            deg(30.0),
            t0,
            "Sun",
            c * (r_sun + 3.0e8),
            n,
        );
        marks.push(Mark {
            comment: "#12 3.0e8 m above the photosphere looking at the Sun's centre, fov 30°: \
                      granules 8.7 px at the centre."
                .to_owned(),
            b,
        });
    }

    // 13. venus-crescent: phase 155°, 150 000 km, fov 20°; the Sun above the frame.
    {
        let fov = deg(20.0);
        let s = rel(world, "Venus", "Sun").normalize();
        let n = DVec3::Z;
        let (e1, _) = basis(s, n);
        let c = s * deg(155.0).cos() - e1 * deg(155.0).sin();
        let b = anchored("venus-crescent", fov, t0, "Venus", c * 1.5e8, n);
        marks.push(Mark {
            comment: "#13 Venus at phase 155° from 150 000 km, fov 20° (cloud-top radius 125.1 \
                      px), north up; the Sun 25° off-axis, above the frame."
                .to_owned(),
            b,
        });
    }

    // 14. titan-backlit: phase 160°, 60 000 km, fov 20°; Saturn and the Sun out of frame, no
    // moon over 1 px off Horizons.
    {
        let fov = deg(20.0);
        let s = rel(world, "Titan", "Sun").normalize();
        let n = pole(world, "Titan");
        let (e1, e2) = basis(s, n);
        let saturn = rel(world, "Titan", "Saturn");
        let moons = [
            "Mimas",
            "Enceladus",
            "Tethys",
            "Dione",
            "Rhea",
            "Hyperion",
            "Iapetus",
        ];
        let mut best = (f64::NEG_INFINITY, DVec3::ZERO);
        for k in 0..720 {
            let psi = deg(0.5 * f64::from(k));
            let c = s * deg(160.0).cos() + (e1 * psi.cos() + e2 * psi.sin()) * deg(160.0).sin();
            let o = look(-c, n);
            let cam = pos(world, "Titan").offset(c * 6.0e7);
            // The Sun's image direction: vertical is best (it is 20° off-axis, the frame's
            // half-height is 10°).
            let sc = o.conjugate() * s;
            let vertical = sc.y.abs() / sc.truncate().length();
            let sat_off = angle(saturn - c * 6.0e7, -c).to_degrees();
            let moons_ok = moon_errors_px(world, cam, o, fov, "Saturn", &moons)
                .iter()
                .all(|(_, e)| e.is_none_or(|e| e < 1.0));
            if !moons_ok || sat_off < 25.0 {
                continue;
            }
            let score = vertical + 0.001 * sat_off;
            if score > best.0 {
                best = (score, c);
            }
        }
        let c = best.1;
        let b = anchored("titan-backlit", fov, t0, "Titan", c * 6.0e7, n);
        marks.push(Mark {
            comment: format!(
                "#14 Titan at phase 160° from 60 000 km, fov 20° (131.6 px), pole up; the Sun \
                 above or below the frame, Saturn {:.0}° off-axis.",
                angle(saturn - c * 6.0e7, -c).to_degrees()
            ),
            b,
        });
    }

    // 15, 16: 100 AU from the Sun along the view direction, looking away from it, north up.
    for (name, ra, dec, what) in [
        (
            "sky-orion-3d",
            83.8,
            -1.2,
            "#15 Orion (RA 83.8°, Dec −1.2°)",
        ),
        (
            "sky-galactic-center",
            266.405,
            -28.936,
            "#16 the Galactic Centre (RA 266.405°, Dec −28.936°)",
        ),
    ] {
        let v = radec_to_icrs(deg(ra), deg(dec));
        let mut b = blank(name, deg(60.0), t0);
        b.anchor = Some("Sun".to_owned());
        b.view.pos.local = v * 100.0 * AU;
        b.view.orient = look(v, DVec3::Z);
        marks.push(Mark {
            comment: format!(
                "{what}: the camera 100 AU from the Sun along the view direction (the Sun \
                 directly behind), fov 60°, north up."
            ),
            b,
        });
    }

    // 17. alpha-cen-sun: 0.05 pc from α Cen A toward the Sun, looking at the Sun, fov 20°.
    {
        let a = "Rigil Kentaurus";
        let to_sun = rel(world, a, "Sun").normalize();
        let mut b = blank("alpha-cen-sun", deg(20.0), t0);
        b.anchor = Some(a.to_owned());
        b.view.pos.local = to_sun * 0.05 * PC;
        b.look_at = Some("Sun".to_owned());
        b.up = Some(DVec3::Z);
        marks.push(Mark {
            comment: "#17 0.05 pc from α Cen A (Rigil Kentaurus) toward the Sun, on the Sun–A \
                      line, looking at the Sun (RA 39.9°, Dec +60.8°), fov 20°, north up; α Cen A \
                      and B 180° behind."
                .to_owned(),
            b,
        });
    }

    // 18. bh-shadow-30M-grid: m0's bh-shadow-30M with the debug grid, Manual at M0's EVs.
    {
        let m0 = uni_nav::load_bookmarks(&crate::world::m0_bookmarks_path()).unwrap();
        let mut b = m0.into_iter().find(|b| b.name == "bh-shadow-30M").unwrap();
        b.name = "bh-shadow-30M-grid".to_owned();
        b.exposure = Some(uni_nav::ExposureChoice::Manual);
        b.lens_env = Some(uni_nav::LensEnvChoice::DebugGrid);
        b.scene_ev100 = Some(0.0);
        b.star_ev100 = Some(uni_sky::DEFAULT_STAR_EV100);
        b.eye_adapted = Some(true);
        marks.push(Mark {
            comment: "#18 m0.1's bh-shadow-30M (30 M sunward of TestBH-1, looking at it) with the \
                      M0 debug grid, Manual at M0's EVs (scene 0, stars −4): the M0 shadow \
                      regression (158.745 px)."
                .to_owned(),
            b,
        });
    }

    // 19. bh-lens-band: 30 M from TestBH-1 on the side away from the Galactic Centre, looking
    // through the hole toward it.
    {
        let g = radec_to_icrs(deg(266.405), deg(-28.936));
        let lens = &world.sim.lenses()[0];
        let m = gravitational_radius(lens.mass_kg);
        let mut b = blank("bh-lens-band", deg(60.0), t0);
        b.anchor = Some(crate::world::TEST_BH_NAME.to_owned());
        b.view.pos.local = -g * 30.0 * m;
        b.view.orient = look(g, DVec3::Z);
        b.lens_scale = Some(1.0);
        marks.push(Mark {
            comment: "#19 30 M from TestBH-1 on the side away from the Galactic Centre, looking \
                      through the hole toward it (the Sun 86.9° off-axis), north up, the band \
                      lensed, auto exposure. Checks: visual and seam only; no numeric shadow \
                      check (lead ruling \"row 19 shadow\": #18 covers the geometry)."
                .to_owned(),
            b,
        });
    }
    marks.extend(generate_wave_c(world, &marks));
    marks
}

/// The comet the plan's row 20 looks at (SBDB's name in the registry).
pub const NEOWISE_NAME: &str = "C/2020 F3 (NEOWISE)";
/// Row 21's comet.
pub const HALLEY_NAME: &str = "1P/Halley";
/// Row 23's exoplanet host.
pub const TRAPPIST_NAME: &str = "TRAPPIST-1";
/// Row 26's epoch: row 15's date 50 000 years later (ISO 8601 expanded year,
/// `scene::parse_bookmark_epoch`).
pub const ORION_50KYR_UTC: &str = "+52026-11-20T00:00:00Z";
/// The seam bookmark's distance from TestBH-1 [M] and field of view [deg] (lead ruling "row 19
/// seam"): at fov 100° the lens cone (deflection ≥ ½ px) is 53.6° across from 4 500 M, so the
/// traced rectangle spans the frame's height and ≈ 64 % of its width, and its left and right
/// edges are seams; within `uni_rel::TRACE_REACH` (5 000 M). At fov 60° no distance within the
/// reach gives a partial rectangle (b_max = 7 484 M > 5 000 M).
pub const SEAM_DISTANCE_M: f64 = 4_500.0;
pub const SEAM_FOV_DEG: f64 = 100.0;

/// The Wave C bookmarks, rows 20–26 (plan, table "M1 capture bookmarks"), and #27, the seam
/// bookmark of the lead's ruling "row 19 seam" (`marks` holds rows 1–19).
fn generate_wave_c(world: &mut World, marks: &[Mark]) -> Vec<Mark> {
    let mut out = Vec::new();
    // 20. comet-neowise-2020: 7 000 km from Earth's centre toward the Sun, looking at NEOWISE.
    {
        let t = "2020-07-15T03:00:00Z";
        world.set_epoch(epoch(t));
        world.sim.settle_comets();
        let s = rel(world, "Earth", "Sun").normalize();
        let mut b = blank("comet-neowise-2020", deg(20.0), t);
        b.anchor = Some("Earth".to_owned());
        b.view.pos.local = s * 7.0e6;
        b.look_at = Some(NEOWISE_NAME.to_owned());
        b.up = Some(DVec3::Z);
        let cam = pos(world, "Earth").offset(s * 7.0e6);
        let c = pos(world, NEOWISE_NAME).delta_from(cam);
        out.push(Mark {
            comment: format!(
                "#20 C/2020 F3 (NEOWISE) on 2020-07-15 03:00 UTC from 7 000 km sunward of \
                 Earth's centre (629 km up, outside the atmosphere), fov 20°, north up; the Sun \
                 {:.1}° off-axis, the comet {:.3} AU away.",
                angle(c, s).to_degrees(),
                c.length() / AU
            ),
            b,
        });
    }
    // 21. comet-halley-1986: 2e6 km from 1P, looking along its orbit's normal (phase 90°, the
    // dust tail's curve face-on), the Sun to the left; fov 20° (653 km/px).
    {
        let t = "1986-03-13T00:00:00Z";
        world.set_epoch(epoch(t));
        world.sim.settle_comets();
        let id = world.sim.find(HALLEY_NAME).unwrap();
        let s = rel(world, HALLEY_NAME, "Sun").normalize();
        let v = world.sim.velocity(id) - world.sim.velocity(world.sim.find("Sun").unwrap());
        let n = s.cross(v).normalize();
        let c = (n - s * n.dot(s)).normalize();
        let a = -s;
        let mut b = anchored(
            "comet-halley-1986",
            deg(20.0),
            t,
            HALLEY_NAME,
            c * 2.0e9,
            c.cross(a),
        );
        b.look_at = Some(HALLEY_NAME.to_owned());
        out.push(Mark {
            comment: format!(
                "#21 1P/Halley on 1986-03-13 from 2.0e6 km along its orbit's normal (phase \
                 {:.1}°), the Sun to the left, fov 20° (653 km/px); r_h {:.3} AU.",
                angle(s, c).to_degrees(),
                rel(world, HALLEY_NAME, "Sun").length() / AU
            ),
            b,
        });
    }
    // 22. asteroid-belt-above: 6 AU above the ecliptic over the Sun, looking at it, fov 70°,
    // the belt highlighted; the vernal equinox up.
    world.set_epoch(epoch(SPOT_EPOCH_UTC));
    world.sim.settle_comets();
    {
        let pole = uni_physics::frames::ecliptic_j2000_to_icrs(DVec3::Z);
        let mut b = anchored(
            "asteroid-belt-above",
            deg(70.0),
            SPOT_EPOCH_UTC,
            "Sun",
            pole * 6.0 * AU,
            DVec3::X,
        );
        b.highlight_belt = Some(true);
        out.push(Mark {
            comment: "#22 6 AU above the ecliptic over the Sun, looking down at it, fov 70° \
                      (128.5 px/AU at the Sun), the vernal equinox up; \"Highlight asteroid \
                      belt\" on (amber label). The belt check reads the 3:1 Kirkwood dip in \
                      semi-major axis, not in projected distance (lead ruling, C6 fix round 1: \
                      eccentricities smear the gaps in distance from the Sun)."
                .to_owned(),
            b,
        });
    }
    // 23. exo-trappist1: 0.2 AU from TRAPPIST-1, 30° off its (assumed) orbit normal, `--hud`
    // with the host as the target.
    {
        let id = world.sim.find(TRAPPIST_NAME).unwrap();
        let host = world.sim.position(id);
        let dir = host.delta_from(UniversePos::default()).normalize();
        let Some(uni_sim::world::ObjectDetails::ExoHost { system, .. }) = world.sim.details(id)
        else {
            panic!("{TRAPPIST_NAME} is not an exoplanet host");
        };
        let (pts, _) = uni_ui::exo_orbit_points(&system, 0, dir, 64);
        let normal = (pts[16] - pts[0]).cross(pts[32] - pts[0]).normalize();
        let (e1, _) = basis(normal, dir);
        let c = (normal * deg(30.0).cos() + e1 * deg(30.0).sin()).normalize();
        let mut b = anchored(
            "exo-trappist1",
            deg(60.0),
            SPOT_EPOCH_UTC,
            TRAPPIST_NAME,
            c * 0.2 * AU,
            normal,
        );
        b.target = Some(TRAPPIST_NAME.to_owned());
        let sun = pos(world, "Sun").delta_from(host.offset(c * 0.2 * AU));
        out.push(Mark {
            comment: format!(
                "#23 0.2 AU from TRAPPIST-1 ({} planets), 30° off the normal of its (assumed, \
                 dashed) orbit plane, fov 60°; target TRAPPIST-1 (capture with --hud); the Sun \
                 {:.1}° off-axis.",
                system.planets.len(),
                angle(sun, -c).to_degrees()
            ),
            b,
        });
    }
    // 24. hud-target-mars: #5 with Mars as the target (`--hud`).
    {
        let mut b = marks
            .iter()
            .find(|m| m.b.name == "mars-reference-angle")
            .unwrap()
            .b
            .clone();
        b.name = "hud-target-mars".to_owned();
        b.target = Some("Mars".to_owned());
        out.push(Mark {
            comment: "#24 #5 (mars-reference-angle) with Mars as the target: the card (capture \
                      with --hud)."
                .to_owned(),
            b,
        });
    }
    // 25. hud-cruise: halfway between Earth and Mars, looking at Mars, cruising (`--hud`).
    {
        let half = rel(world, "Earth", "Mars") * 0.5;
        let mut b = blank("hud-cruise", deg(60.0), SPOT_EPOCH_UTC);
        b.anchor = Some("Earth".to_owned());
        b.view.pos.local = half;
        b.look_at = Some("Mars".to_owned());
        b.up = Some(DVec3::Z);
        b.cruise = Some(true);
        out.push(Mark {
            comment: format!(
                "#25 halfway between Earth and Mars ({:.3} AU from each), looking at Mars, \
                 cruise on: the amber cruise label (capture with --hud).",
                half.length() / AU
            ),
            b,
        });
    }
    // 26. sky-orion-50kyr: #15 at 2026 + 50 000 yr.
    {
        let mut b = marks
            .iter()
            .find(|m| m.b.name == "sky-orion-3d")
            .unwrap()
            .b
            .clone();
        b.name = "sky-orion-50kyr".to_owned();
        b.epoch_utc = Some(ORION_50KYR_UTC.to_owned());
        out.push(Mark {
            comment: "#26 #15 (sky-orion-3d) 50 000 years later: proper motion over 50 kyr; the \
                      planets-extrapolated label, not the stars one (capture with --hud)."
                .to_owned(),
            b,
        });
    }
    // 27. bh-lens-seam: TestBH-1's traced rectangle smaller than the frame (lead ruling "row 19
    // seam"): 4 500 M on the side away from the Galactic Centre, looking through the hole toward
    // it, fov 100°.
    {
        let g = radec_to_icrs(deg(266.405), deg(-28.936));
        let lens = &world.sim.lenses()[0];
        let m = gravitational_radius(lens.mass_kg);
        let mut b = blank("bh-lens-seam", deg(SEAM_FOV_DEG), SPOT_EPOCH_UTC);
        b.anchor = Some(crate::world::TEST_BH_NAME.to_owned());
        b.view.pos.local = -g * SEAM_DISTANCE_M * m;
        b.view.orient = look(g, DVec3::Z);
        b.lens_scale = Some(1.0);
        out.push(Mark {
            comment: "#27 (lead ruling \"row 19 seam\") 4 500 M from TestBH-1 on the side away \
                      from the Galactic Centre, looking through it, fov 100°: the traced \
                      rectangle spans the height and the middle 64 % of the width, so its left \
                      and right edges are seams (HDR ratio inner / outer edge = 1 ± 0.1)."
                .to_owned(),
            b,
        });
    }
    out
}

/// Ruling 10's ceiling [px] for a moon's theory-vs-Horizons error at the Saturn bookmarks
/// (A2 ruling I3: Meeus ch. 46, measured in e539535's test); 1 px for every other moon.
pub fn ceiling_px(moon: &str) -> f64 {
    match moon {
        "Mimas" => 2.5,
        "Dione" => 1.2,
        _ => 1.0,
    }
}

const HEADER: &str = "#![enable(implicit_some)]
// M1 capture bookmarks (plan, table \"M1 capture bookmarks\": entries 1–19, Wave B; 20–26, Wave C;
// 27, the seam bookmark of the lead's ruling \"row 19 seam\").
// Generated by `cargo test -p uni-app --bins -- --ignored write_m1_bookmarks --nocapture`
// (crates/uni-app/src/bookmarks.rs) from the World at each epoch; the intent tests there check
// every placement.
//
// Axes: ICRS/J2000, metres. With `anchor`, `pos` is the offset from that object's centre (cell
// 0, root axes); `look_at` turns the camera's −Z toward the object with +Y as close as possible
// to `up` (A6). Fields left out keep their defaults: Auto exposure, eye-adapted stars at
// star EV100 −6.5, the band behind a lens, lens scale 1.0 in captures, nothing selected, free
// flight, the belt not highlighted. Rows 23–26 are drawn with the HUD by the gate's `--hud` run.
[
";

/// Prints `bookmarks/m1.ron` and `bookmarks/m1-bench.ron` (run by hand; module docs).
#[test]
#[ignore = "generator: prints bookmarks/m1.ron and m1-bench.ron"]
fn write_m1_bookmarks() {
    let mut guard = full_world();
    let (world, _) = &mut *guard;
    let marks = generate(world);
    let mut m1 = String::from(HEADER);
    for m in &marks {
        let _ = writeln!(m1, "    // {}", m.comment);
        m1.push_str(&ron(&m.b));
    }
    m1.push_str("]\n");
    println!("===== m1.ron\n{m1}===== end");
    let bench = BENCH_NAMES;
    let mut b = String::from(
        "#![enable(implicit_some)]
// M1 bench bookmarks (plan: the Wave B gate's subset of bookmarks/m1.ron, plus Wave C's
// comet-neowise-2020 and asteroid-belt-above, copied unchanged). `universe --bench
// bookmarks/m1-bench.ron --sustained 60 --json <out>`.
[
",
    );
    for name in bench {
        let m = marks.iter().find(|m| m.b.name == name).unwrap();
        let _ = writeln!(b, "    // {}", m.comment);
        b.push_str(&ron(&m.b));
    }
    b.push_str("]\n");
    println!("===== m1-bench.ron\n{b}===== end");
}

/// `bookmarks/m1-bench.ron`'s bookmarks (plan C6: the Wave B gate's eight and Wave C's two).
const BENCH_NAMES: [&str; 10] = [
    "start",
    "earth-limb-dawn",
    "earth-map-closeup",
    "mars-reference-angle",
    "jupiter-galileans",
    "saturn-ring-plane",
    "sun-closeup",
    "sky-orion-3d",
    "comet-neowise-2020",
    "asteroid-belt-above",
];

/// The committed `bookmarks/m1.ron`.
fn committed() -> Vec<Bookmark> {
    uni_nav::load_bookmarks(&crate::world::m1_bookmarks_path()).expect("bookmarks/m1.ron loads")
}

/// Bookmark `name` of the committed file, applied to `world` (its epoch, anchor and look_at).
fn applied(world: &mut World, name: &str) -> (Bookmark, crate::scene::Applied) {
    let b = committed()
        .into_iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("no bookmark {name}"));
    let mut controls = crate::scene::default_controls(1.0);
    // As a capture: the comets settled at the bookmark's epoch first, where the view depends on
    // a comet (anchor or `look_at`). Elsewhere it changes no geometry these tests check, and a
    // debug-build settle at row 26's +50 kyr (the 200-year perturbed span of 512 comets) takes
    // hours.
    let on_comet = [&b.anchor, &b.look_at].into_iter().flatten().any(|n| {
        world.sim.find(n).is_some_and(|id| {
            world.sim.objects()[id as usize].kind == uni_sim::registry::ObjectKind::Comet
        })
    });
    if on_comet {
        crate::capture::settle_at_bookmark_epoch(&b, world).unwrap();
    }
    let a = crate::scene::apply_bookmark(&b, world, &mut controls).unwrap();
    // The selection as a capture sets it (the shared world keeps state between tests).
    crate::capture::bookmark_ui(&b, world).unwrap();
    (b, a)
}

/// Prints every bookmark's paper exposure (diagnostic).
#[test]
#[ignore = "diagnostic: prints the paper exposure of every bookmark"]
fn print_paper() {
    let mut guard = full_world();
    let (world, assets) = &mut *guard;
    for b in committed() {
        let (_, a) = applied(world, &b.name);
        let p = crate::paper::on_paper(
            world,
            &a.view,
            glam::UVec2::new(1920, 1080),
            assets.rings.as_ref(),
        );
        println!("{:24} {:?}", b.name, p);
    }
}

// ---- intent tests (B9 Step 5) ----------------------------------------------------------------

/// The capture size every check assumes.
const SIZE: glam::UVec2 = glam::UVec2::new(1920, 1080);

/// `got` within `rel` (a fraction) of `want`.
#[track_caller]
fn near(what: &str, got: f64, want: f64, rel: f64) {
    assert!(
        (got - want).abs() <= rel * want.abs(),
        "{what}: {got} vs {want} (±{} %)",
        rel * 100.0
    );
}

/// The view of bookmark `name` and its snapshot body `body` (resolved), with geometry helpers.
struct Case {
    view: CameraView,
    snap: uni_sim::snapshot::Snapshot,
}

impl Case {
    fn new(world: &mut World, name: &str) -> Self {
        let (_, a) = applied(world, name);
        let snap = world.sim.snapshot(&a.view, SIZE);
        Case { view: a.view, snap }
    }

    fn body(&self, world: &World, name: &str) -> &uni_sim::snapshot::BodyView {
        let id = world.sim.find(name).unwrap();
        self.snap
            .bodies
            .iter()
            .find(|b| b.id == id)
            .unwrap_or_else(|| panic!("{name} is not resolved"))
    }

    fn fwd(&self) -> DVec3 {
        self.view.orient * DVec3::NEG_Z
    }

    /// Pixel of the point `rel` [m from the camera].
    fn px(&self, rel: DVec3) -> Option<(f64, f64)> {
        project(self.view.orient, self.view.fov_y_rad, rel)
    }
}

/// Every moon of `moons` in frame other than the anchor is within its ceiling of its Horizons
/// spot (ruling 10); returns how many were in frame.
#[track_caller]
fn assert_moons(world: &World, c: &Case, planet: &str, moons: &[&str]) -> usize {
    let errs = moon_errors_px(
        world,
        c.view.pos,
        c.view.orient,
        c.view.fov_y_rad,
        planet,
        moons,
    );
    for (m, e) in &errs {
        let e = e.unwrap_or_else(|| panic!("{m}: no Horizons spot at this epoch"));
        assert!(
            e < ceiling_px(m),
            "{m}: {e:.3} px from Horizons (ceiling {})",
            ceiling_px(m)
        );
    }
    errs.len()
}

/// The paper exposure of `c` against its "Exposure on paper" row: EV within ±0.5 and lit pixels
/// within ±10 % of the row (or, for the sky views, under the meter's 50 px).
#[track_caller]
fn assert_paper(
    world: &World,
    rings: Option<&uni_data::RingProfile>,
    c: &Case,
    ev: f32,
    lit: Option<f64>,
) {
    let p = crate::paper::on_paper(world, &c.view, SIZE, rings);
    assert!(
        (p.target_ev - ev).abs() <= 0.5,
        "paper EV {} vs the row's {ev} (±0.5): {p:?}",
        p.target_ev
    );
    match lit {
        Some(n) => near("lit px", f64::from(p.lit_px), n, 0.10),
        None => assert!(
            f64::from(p.lit_px) < f64::from(uni_post::METER_MIN_LIT_PX),
            "{p:?}"
        ),
    }
    if lit.is_some() {
        assert!(
            f64::from(p.lit_px) >= f64::from(uni_post::METER_FULL_LIT_PX),
            "{p:?}"
        );
    }
}

/// The committed files are the generator's output (names and order), and `m1-bench.ron` holds
/// the Wave B gate's subset unchanged.
#[test]
fn committed_bookmarks_are_the_plan_set() {
    let names: Vec<String> = committed().into_iter().map(|b| b.name).collect();
    assert_eq!(
        names,
        [
            "start",
            "earth-limb-dawn",
            "earth-map-closeup",
            "moon-full",
            "mars-reference-angle",
            "jupiter-galileans",
            "jupiter-disk",
            "saturn-ring-plane",
            "saturn-edge-on",
            "saturn-night-side",
            "sun-closeup",
            "sun-granulation",
            "venus-crescent",
            "titan-backlit",
            "sky-orion-3d",
            "sky-galactic-center",
            "alpha-cen-sun",
            "bh-shadow-30M-grid",
            "bh-lens-band",
            "comet-neowise-2020",
            "comet-halley-1986",
            "asteroid-belt-above",
            "exo-trappist1",
            "hud-target-mars",
            "hud-cruise",
            "sky-orion-50kyr",
            "bh-lens-seam",
        ]
    );
    let bench = uni_nav::load_bookmarks(&crate::world::bookmarks_path("m1-bench.ron")).unwrap();
    let all = committed();
    let bench_names: Vec<&str> = bench.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(bench_names, BENCH_NAMES);
    for b in &bench {
        assert_eq!(Some(b), all.iter().find(|a| a.name == b.name), "{}", b.name);
    }
    // Strict exposure where the disk fits need it (re-review r1 NP2); manual debug grid at 18.
    for b in &all {
        let strict = matches!(
            b.name.as_str(),
            "moon-full" | "jupiter-disk" | "saturn-edge-on"
        );
        assert_eq!(b.eye_adapted == Some(false), strict, "{}", b.name);
    }
}

#[test]
fn intent_01_start() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "start");
    let e = c.body(world, "Earth");
    near("distance", e.rel_pos_m.length(), 6.0e7, 0.01);
    near(
        "phase",
        crate::measure::phase_rad(e).to_degrees(),
        60.0,
        0.01,
    );
    near("Earth radius px", e.proj_radius_px, 99.9, 0.01);
    near(
        "look at Earth",
        angle(c.fwd(), e.rel_pos_m) + 1.0,
        1.0,
        1e-9,
    );
    // The Moon in frame and resolved (≈ 4.2 px), clear of Earth's disk.
    let m = c.body(world, "Moon");
    near("Moon radius px", m.proj_radius_px, 4.2, 0.15);
    let (x, y) = c.px(m.rel_pos_m).unwrap();
    assert!(
        x > 20.0 && x < W - 20.0 && y > 20.0 && y < H - 20.0,
        "Moon at {x}, {y}"
    );
    assert!(angle(m.rel_pos_m, e.rel_pos_m) > (6.4e6 / 6.0e7_f64).asin() + deg(2.0));
    assert_eq!(assert_moons(world, &c, "Earth", &["Moon"]), 1);
    // Re-derived row (ruling W-B1/B3 I4): the composite ρ 0.302 plus B3's sky, 15.9 (the plan's
    // 17.1 took p_V 0.434), 24 450 lit px.
    assert_paper(world, assets.rings.as_ref(), &c, 15.92, Some(24_450.0));
}

#[test]
fn intent_02_earth_limb_dawn() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    let c = Case::new(world, "earth-limb-dawn");
    let e = c.body(world, "Earth");
    let (_, alt) = world.sim.nearest_surface(&c.view.pos).unwrap();
    near("altitude", alt, 420.0e3, 0.01);
    let up = (-e.rel_pos_m).normalize();
    let sun = c.snap.sun.as_ref().unwrap().rel_pos_m.normalize();
    // Horizontal, toward the Sun's azimuth.
    assert!(c.fwd().dot(up).abs() < 1e-9);
    let sun_h = (sun - up * sun.dot(up)).normalize();
    assert!(angle(c.fwd(), sun_h) < 1e-9);
    // 22° of arc into the night: the Sun 22° below horizontal; the limb 20.26° below it at
    // 2 351 km; the Sun ≈ 1.8° below the limb.
    let sun_dep = sun.dot(up).asin().to_degrees();
    near("Sun depression", -sun_dep, 22.0, 0.01);
    let r = e.rel_pos_m.length();
    let r_ground = r - alt;
    let limb_dep = (r_ground / r).acos().to_degrees();
    near("limb depression", limb_dep, 20.26, 0.01);
    near(
        "limb distance",
        (r * r - r_ground * r_ground).sqrt(),
        2.351e6,
        0.01,
    );
    near("Sun below the limb", -sun_dep - limb_dep, 1.8, 0.1);
    // Not metered on paper: only the atmosphere's arc is lit (the ground is night).
}

#[test]
fn intent_03_earth_map_closeup() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "earth-map-closeup");
    let e = c.body(world, "Earth");
    let (_, alt) = world.sim.nearest_surface(&c.view.pos).unwrap();
    near("altitude", alt, 1.5e6, 0.01);
    let nadir = e.rel_pos_m.normalize();
    near("off nadir", angle(c.fwd(), nadir).to_degrees(), 30.0, 0.01);
    near("fov", c.view.fov_y_rad.to_degrees(), 30.0, 1e-9);
    // The Sun's elevation at the sub-camera point (36.0°N, 5.6°W) at 12:00 UTC: ≈ 34°.
    let up = -nadir;
    let sun = (e.sun_rel_m + e.rel_pos_m - up * e.radii_m.x).normalize();
    let el = sun.dot(up).asin().to_degrees();
    assert!((el - 34.0).abs() < 1.5, "Sun elevation {el}°");
    // One 8k texel (4.9 km) ≈ 6.6 px at nadir.
    let px_per_rad = (H / 2.0) / (c.view.fov_y_rad / 2.0).tan();
    near("texel px", 4.9e3 / alt * px_per_rad, 6.6, 0.02);
    assert_paper(world, assets.rings.as_ref(), &c, 15.0, Some(2_073_600.0));
}

#[test]
fn intent_04_moon_full() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "moon-full");
    let m = c.body(world, "Moon");
    near("distance", m.rel_pos_m.length(), 2.0e7, 0.01);
    let phase = crate::measure::phase_rad(m).to_degrees();
    assert!(phase <= 2.0, "phase {phase}");
    near("radius px", m.proj_radius_px, 267.0, 0.01);
    // Not eclipsed: Earth is not between the Moon and the Sun.
    let earth = rel(world, "Moon", "Earth");
    assert!(angle(earth, m.sun_rel_m) > deg(5.0));
    assert_paper(world, assets.rings.as_ref(), &c, 15.3, Some(224_000.0));
}

#[test]
fn intent_05_mars_reference_angle() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "mars-reference-angle");
    let m = c.body(world, "Mars");
    near("radius px", m.proj_radius_px, 425.0, 0.01);
    near("distance", m.rel_pos_m.length(), 8.21e6, 0.01);
    near(
        "phase",
        crate::measure::phase_rad(m).to_degrees(),
        45.0,
        0.01,
    );
    // Lit limb left: the Sun projects within 18° of the image's −x axis; north up.
    let sc = c.view.orient.conjugate() * m.sun_rel_m;
    assert!(sc.y.atan2(-sc.x).abs() < deg(18.0), "{sc}");
    assert!((c.view.orient * DVec3::Y).dot(pole(world, "Mars")) > 0.9);
    assert_moons(world, &c, "Mars", &["Phobos", "Deimos"]);
    // Re-derived row: B1 applies A3's phase correction C per pixel (lead ruling after A3),
    // which the plan's 14.6 left out: 14.03, 545 400 lit px (the camera at 2.4 radii sees more
    // of the lit cap than (1 + cos α)/2 of the disk).
    assert_paper(world, assets.rings.as_ref(), &c, 14.03, Some(545_400.0));
}

#[test]
fn intent_06_jupiter_galileans() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "jupiter-galileans");
    let j = c.body(world, "Jupiter");
    near("distance", j.rel_pos_m.length(), 5.0e9, 0.01);
    near("radius px", j.proj_radius_px, 28.8, 0.01);
    assert!(crate::measure::phase_rad(j).to_degrees() < 10.0);
    let el = (-j.rel_pos_m)
        .normalize()
        .dot(pole(world, "Jupiter"))
        .asin()
        .to_degrees();
    assert!(el.abs() <= 2.0, "elevation {el}°");
    let moons = ["Io", "Europa", "Ganymede", "Callisto"];
    for m in moons {
        let r = pos(world, m).delta_from(c.view.pos);
        let (x, y) = c.px(r).unwrap_or_else(|| panic!("{m} behind"));
        assert!(
            (0.0..W).contains(&x) && (0.0..H).contains(&y),
            "{m} at {x}, {y}"
        );
    }
    let callisto = angle(rel(world, "Jupiter", "Callisto") + j.rel_pos_m, j.rel_pos_m);
    assert!(
        callisto.to_degrees() <= 20.6,
        "Callisto {}°",
        callisto.to_degrees()
    );
    assert_eq!(assert_moons(world, &c, "Jupiter", &moons), 4);
    assert_paper(world, assets.rings.as_ref(), &c, 12.9, Some(2_440.0));
}

#[test]
fn intent_07_jupiter_disk() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "jupiter-disk");
    let j = c.body(world, "Jupiter");
    near("distance", j.rel_pos_m.length(), 3.0e9, 0.01);
    assert!(crate::measure::phase_rad(j).to_degrees() <= 1.0);
    near("radius px", j.proj_radius_px, 421.3, 0.01);
    let moons = ["Io", "Europa", "Ganymede", "Callisto"];
    let hits = moon_near_disk(
        world,
        c.view.pos,
        c.view.orient,
        c.view.fov_y_rad,
        "Jupiter",
        &moons,
        5.0,
    );
    assert!(hits.is_empty(), "{hits:?}");
    assert_moons(world, &c, "Jupiter", &moons);
    assert_paper(world, assets.rings.as_ref(), &c, 12.9, Some(521_000.0));
}

#[test]
fn intent_08_saturn_ring_plane() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "saturn-ring-plane");
    let s = c.body(world, "Saturn");
    near("distance", s.rel_pos_m.length(), 1.5e9, 0.01);
    near("radius px", s.proj_radius_px, 123.1, 0.01);
    let n = pole(world, "Saturn");
    let bp = sun_elevation(world, "Saturn").to_degrees();
    assert!((bp + 8.5).abs() < 0.5, "B′ {bp}° (≈ 8.5° S)");
    let el = (-s.rel_pos_m).normalize().dot(n).asin().to_degrees();
    near("elevation above the lit (south) face", -el, 4.0, 0.01);
    assert!(crate::measure::phase_rad(s).to_degrees() < 10.0);
    let moons = [
        "Mimas",
        "Enceladus",
        "Tethys",
        "Dione",
        "Rhea",
        "Titan",
        "Hyperion",
        "Iapetus",
    ];
    assert_moons(world, &c, "Saturn", &moons);
    // The Cassini division at 240–249 px from the centre on the ring's major axis.
    let px_per_m = s.proj_radius_px / s.radii_m.x;
    near("Cassini inner px", 117_580.0e3 * px_per_m, 240.0, 0.02);
    assert_paper(world, assets.rings.as_ref(), &c, 11.1, Some(51_200.0));
}

#[test]
fn intent_09_saturn_edge_on() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "saturn-edge-on");
    let s = c.body(world, "Saturn");
    assert!(sun_elevation(world, "Saturn").to_degrees().abs() < 0.5);
    let n = pole(world, "Saturn");
    let el = (-s.rel_pos_m).normalize().dot(n).asin().to_degrees();
    assert!(el.abs() < 0.5, "camera elevation {el}°");
    assert!(crate::measure::phase_rad(s).to_degrees() <= 1.0);
    near("distance", s.rel_pos_m.length(), 2.0e9, 0.01);
    near("radius px", s.proj_radius_px, 433.6, 0.01);
    let moons = [
        "Mimas",
        "Enceladus",
        "Tethys",
        "Dione",
        "Rhea",
        "Titan",
        "Hyperion",
        "Iapetus",
    ];
    let hits = moon_near_disk(
        world,
        c.view.pos,
        c.view.orient,
        c.view.fov_y_rad,
        "Saturn",
        &moons,
        5.0,
    );
    assert!(hits.is_empty(), "{hits:?}");
    // 2025 has no Horizons spot (ruling 10 uses the 2026-11-20 spot epoch).
    assert_paper(world, assets.rings.as_ref(), &c, 11.1, Some(533_000.0));
}

#[test]
fn intent_10_saturn_night_side() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "saturn-night-side");
    let s = c.body(world, "Saturn");
    near(
        "phase",
        crate::measure::phase_rad(s).to_degrees(),
        150.0,
        0.01,
    );
    near("distance", s.rel_pos_m.length(), 1.5e9, 0.01);
    let n = pole(world, "Saturn");
    let el = (-s.rel_pos_m).normalize().dot(n).asin().to_degrees();
    // 25° above the unlit face: the north face, while the Sun lights the south (B′ < 0).
    assert!(sun_elevation(world, "Saturn") < 0.0);
    near("elevation above the unlit face", el, 25.0, 0.01);
    let moons = [
        "Mimas",
        "Enceladus",
        "Tethys",
        "Dione",
        "Rhea",
        "Titan",
        "Hyperion",
        "Iapetus",
    ];
    assert_moons(world, &c, "Saturn", &moons);
    // Re-derived row (ruling B2 I2: the rings' unlit face is 0.0036–0.032 of E/π, not 0.35):
    // 6.41, 56 460 lit px (the plan's 10.3 and 75 600).
    assert_paper(world, assets.rings.as_ref(), &c, 6.41, Some(56_460.0));
}

#[test]
fn intent_11_12_sun() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "sun-closeup");
    let s = &c.snap.star_surfaces[0];
    near("distance", s.rel_pos_m.length(), 0.02 * AU, 0.01);
    near("disk radius px", s.proj_radius_px, 481.8, 0.01);
    assert_paper(world, assets.rings.as_ref(), &c, 33.8, Some(728_000.0));
    let c = Case::new(world, "sun-granulation");
    let s = &c.snap.star_surfaces[0];
    near("altitude", s.rel_pos_m.length() - s.radius_m, 3.0e8, 0.01);
    // Granules 8.7 px at the frame centre (B2's cell size at the Sun's T_eff and radius).
    let cell = uni_planet::star_surface::granule_cell_m(s.teff_k, s.radius_m);
    let px_per_rad = (H / 2.0) / (c.view.fov_y_rad / 2.0).tan();
    near("granule px", cell / 3.0e8 * px_per_rad, 8.7, 0.02);
    // The limb is out of frame: 44.3° off-axis, past the 30° field's half-diagonal.
    let limb = (s.radius_m / s.rel_pos_m.length()).asin().to_degrees();
    near("limb off-axis", limb, 44.3, 0.01);
    let half_diag = (((H / 2.0).hypot(W / 2.0)) / px_per_rad)
        .atan()
        .to_degrees();
    assert!(limb > half_diag);
    assert_paper(world, assets.rings.as_ref(), &c, 33.8, Some(2_073_600.0));
}

#[test]
fn intent_13_venus_crescent() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "venus-crescent");
    let v = c.body(world, "Venus");
    near(
        "phase",
        crate::measure::phase_rad(v).to_degrees(),
        155.0,
        0.01,
    );
    near("distance", v.rel_pos_m.length(), 1.5e8, 0.01);
    let cloud_top = v.radii_m.x + uni_atmo::VENUS_CLOUD_TOP_M;
    near(
        "cloud-top radius px",
        radius_px(cloud_top, v.rel_pos_m.length(), c.view.fov_y_rad),
        125.1,
        0.01,
    );
    // Re-derived row (task B9g, lead ruling "row 13 Venus haze"): 17.00 EV, 5 436 lit px. The
    // plan's 16.7 left out A3's C (clamped at 10 at 155°); B9's 18.93 / 1 844 px added C to a
    // bare Lambert deck but left out B3's haze, which the passes draw: through τ ≈ 0.36 at
    // grazing angles the deck is dimmed and the haze's twilight glow lights pixels far past the
    // hard crescent, and the meter's 50th–95th percentile band takes them in (the capture at
    // 6cd81e9: 17.25 EV, 5 180 px). With the haze twin alone the paper reads 16.74 / 5 698 px;
    // the `hdr` target's f16 floor (the meter cannot see l·pre < 2⁻²⁴) removes the faintest
    // glow: 17.00 / 5 436.
    assert_paper(world, assets.rings.as_ref(), &c, 17.0, Some(5_436.0));
    // B9g fix round 1 (review minor 1): with the `hdr` target's f16 subnormals flushed the
    // paper reads 18.32 EV / 3 568 px, 1.07 EV above the capture and with 1 600 lit pixels
    // fewer than the capture's meter saw; kept (the model, `paper::Subnormals`), 17.00 / 5 436.
    let flushed = crate::paper::on_paper_model(
        world,
        &c.view,
        SIZE,
        assets.rings.as_ref(),
        true,
        crate::paper::Subnormals::Flush,
    );
    println!("venus-crescent paper with subnormals flushed: {flushed:?}");
    near(
        "flushed paper EV",
        f64::from(flushed.target_ev),
        18.32,
        0.002,
    );
    near("flushed lit px", f64::from(flushed.lit_px), 3_568.0, 0.01);
    let bare = crate::paper::on_paper_with(world, &c.view, SIZE, assets.rings.as_ref(), false);
    near(
        "bare-deck paper EV (B9's row)",
        f64::from(bare.target_ev),
        18.93,
        0.001,
    );
}

/// Task B9g: the rendered Venus (the passes' model, `paper::VenusAtmo`) sends at 155° nearly the
/// light of the real planet, its p_V point flux with Mallama & Hilton's C (clamped at 10, A3):
/// 0.89 of it, 0.13 mag fainter, inside ruling W-B1's 0.3 mag. So the 1.7 EV gap of row 13 was
/// the paper's (a bare Lambert deck), not a renderer defect. (A bare deck with ρ(p_V) reads
/// 0.74 of the point flux: the camera at 24.5 radii sees a narrower crescent than the point
/// formula's observer at infinity.) And the plan's `night_limb_haze` region measured the sky:
/// there, past ~25° beyond the terminator, the haze shell is in Venus's shadow.
#[test]
fn venus_disk_flux_and_night_limb_twin() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    let c = Case::new(world, "venus-crescent");
    let v = c.body(world, "Venus");
    let cam = crate::measure::Cam::new(&c.view, SIZE);
    let atmo = crate::paper::VenusAtmo::get().expect("Venus preset");
    let mut lux = 0.0;
    for p in cam.pixels() {
        if let Some((l, _)) = atmo.luminance(v, cam.ray(p)) {
            lux += l * cam.pixel_solid_angle(p);
        }
    }
    let point = crate::paper::point_lux(v);
    let gap = -2.5 * (lux / point).log10();
    println!("Venus at 155°: rendered {lux:.4e} lx, point {point:.4e} lx, gap {gap:+.3} mag");
    assert!((0.0..0.3).contains(&gap), "{gap}");
    // B9's capture JSON (git-ignored; skipped when absent): its night_limb_haze pixels hold
    // twin light below 1e-4 of the 1.28e-3 pre-exposed B9 measured there.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../captures/b9-dev/venus-crescent.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        println!("{} absent: region part skipped", path.display());
        return;
    };
    let json: serde_json::Value = serde_json::from_str(&text).expect("b9-dev capture parses");
    let pre = json["hdr"]["pre"][0].as_f64().expect("pre");
    let region = json["regions"]["night_limb_haze"]
        .as_array()
        .expect("B9's night_limb_haze region");
    let mut worst: f64 = 0.0;
    for p in region {
        let (x, y) = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
        let l = atmo
            .luminance(v, cam.ray(glam::DVec2::new(x, y)))
            .map_or(0.0, |r| r.0);
        worst = worst.max(l * pre);
    }
    println!(
        "night_limb_haze: {} px, brightest twin pixel {worst:.3e} pre-exposed",
        region.len()
    );
    assert!(worst < 1.28e-3 * 1e-4, "{worst}");
}

#[test]
fn intent_14_titan_backlit() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "titan-backlit");
    let t = c.body(world, "Titan");
    near(
        "phase",
        crate::measure::phase_rad(t).to_degrees(),
        160.0,
        0.01,
    );
    near("distance", t.rel_pos_m.length(), 6.0e7, 0.01);
    near("radius px", t.proj_radius_px, 131.6, 0.01);
    // Saturn out of frame; every Saturnian moon in frame within 1 px of Horizons.
    let saturn = pos(world, "Saturn").delta_from(c.view.pos);
    assert!(angle(saturn, c.fwd()).to_degrees() > 25.0);
    let moons = [
        "Mimas",
        "Enceladus",
        "Tethys",
        "Dione",
        "Rhea",
        "Hyperion",
        "Iapetus",
    ];
    for (m, e) in moon_errors_px(
        world,
        c.view.pos,
        c.view.orient,
        c.view.fov_y_rad,
        "Saturn",
        &moons,
    ) {
        assert!(e.unwrap() < 1.0, "{m}");
    }
    // Surface only (the haze raises the metered EV): 7.17 vs the row's 7.3; lit 1 191 px (the
    // row's "≥ 1 680" assumed a camera at infinity).
    assert_paper(world, assets.rings.as_ref(), &c, 7.3, Some(1_191.0));
}

/// Sum of the illuminance of every catalogue star within `radius_px` of pixel (x, y) other than
/// `skip`, for the camera of `c` [lx].
fn neighbours_lux(
    world: &World,
    c: &Case,
    x: f64,
    y: f64,
    radius_px: f64,
    skip: Option<u32>,
) -> f64 {
    let years = world.epoch().seconds_since(Epoch::J2000) / crate::scene::JULIAN_YEAR_S;
    let cam_pc = crate::scene::camera_icrs_pc(&c.view.pos);
    let mut sum = 0.0;
    for (i, s) in world.sim.star_data().iter().enumerate() {
        if Some(i as u32) == skip || c.snap.suppressed_stars.contains(&(i as u32)) {
            continue;
        }
        let rel = crate::measure::star_pos_pc(s, years) - cam_pc;
        let Some((sx, sy)) = c.px(rel) else { continue };
        if (sx - x).hypot(sy - y) < radius_px {
            let d = rel.length();
            sum +=
                uni_core::star_illuminance_lux(f64::from(s.abs_mag_v) + 5.0 * (d / 10.0).log10());
        }
    }
    sum
}

/// Rigel's flux radius at star EV `ev` for the camera of `c`, its pixel and catalogue index.
fn star_footprint(world: &World, c: &Case, name: &str, ev: f32) -> (f64, f64, f64, f64, u32) {
    let years = world.epoch().seconds_since(Epoch::J2000) / crate::scene::JULIAN_YEAR_S;
    let cam_pc = crate::scene::camera_icrs_pc(&c.view.pos);
    let id = world.sim.find(name).unwrap();
    let i = world.sim.objects()[id as usize].catalog_index.unwrap();
    let s = &world.sim.star_data()[i as usize];
    let rel = crate::measure::star_pos_pc(s, years) - cam_pc;
    let lux = uni_core::star_illuminance_lux(
        f64::from(s.abs_mag_v) + 5.0 * (rel.length() / 10.0).log10(),
    );
    let (x, y) = c.px(rel).unwrap();
    let cos = rel.normalize().dot(c.fwd()) as f32;
    let e = uni_sky::stars::star_display_energy_off_axis(
        lux as f32,
        c.view.fov_y_rad as f32,
        H as f32,
        ev,
        cos,
    );
    let r = f64::from(uni_sky::stars3d::footprint_radius_px(
        e,
        uni_sky::stars3d::STARS3D_HALO_FRACTION,
    ));
    (x, y, r, lux, i)
}

#[test]
fn intent_15_16_sky_views() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let bh = pos(world, crate::world::TEST_BH_NAME);
    for (name, ra, dec, bh_off) in [
        ("sky-orion-3d", 83.8, -1.2, 89.5),
        ("sky-galactic-center", 266.405, -28.936, 98.8),
    ] {
        let c = Case::new(world, name);
        let v = radec_to_icrs(deg(ra), deg(dec));
        assert!(angle(c.fwd(), v) < 1e-9, "{name}");
        let sun = pos(world, "Sun").delta_from(c.view.pos);
        near("Sun distance [AU]", sun.length() / AU, 100.0, 1e-6);
        assert!(angle(sun, c.fwd()).to_degrees() >= 90.0);
        let off = angle(bh.delta_from(c.view.pos), c.fwd()).to_degrees();
        near("TestBH-1 off-axis", off, bh_off, 0.01);
        assert!(off > 49.66, "{name}: TestBH-1 inside the half-diagonal");
        assert!((c.view.orient * DVec3::Y).z > 0.0, "north up");
        assert_paper(
            world,
            assets.rings.as_ref(),
            &c,
            uni_post::NIGHT_EV100,
            None,
        );
    }
    // The band twin: the Galactic Centre over the anticentre ≥ 2 (from 100 AU, V ≤ 10 drawn).
    let model = uni_procgen::galaxy::GalaxyModel::milky_way();
    let cam = model.icrs_pc_to_galactocentric(DVec3::ZERO);
    let y = |d: DVec3| {
        model
            .band_radiance(cam, d, 10.0, deg(1.0))
            .dot(DVec3::new(0.2126, 0.7152, 0.0722))
    };
    let ratio = y(DVec3::X) / y(DVec3::NEG_X);
    assert!(ratio >= 2.0, "{ratio}");
}

/// The other catalogue stars inside Rigel's flux radius (re-review r2 NPA). At the plan's star
/// EV −4 they stay under 1 % of Rigel only if the footprint is small; this records the measured
/// shares at −4 and at the new default −6.5 (B4L's halo widened every footprint) and asserts
/// the plan's 1 % only where it holds.
#[test]
fn intent_15_rigel_flux_radius() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "sky-orion-3d");
    let mut shares = Vec::new();
    for ev in [-4.0_f32, crate::scene::DEFAULT_STAR_EV100] {
        let (x, y, r, lux, i) = star_footprint(world, &c, "Rigel", ev);
        let share = neighbours_lux(world, &c, x, y, r, Some(i)) / lux;
        shares.push((ev, r, share));
    }
    println!("Rigel flux radius and neighbour share: {shares:?}");
    // The footprint grows with the star EV; the share with it.
    assert!(shares[1].1 > shares[0].1 && shares[1].2 >= shares[0].2);
    // Lead ruling "row 15 flux": the capture JSON's `neighbour_lux` (what B8's check adds to
    // Rigel's own illuminance) is this sum over the stars the star pass draws (its visibility
    // cut at the default star EV); review B9 r0 minor 13.
    let ev = crate::scene::DEFAULT_STAR_EV100;
    let (x, y, r, lux, i) = star_footprint(world, &c, "Rigel", ev);
    let drawn = drawn_neighbours_lux(world, &c, x, y, r, Some(i), ev);
    let m = measured(world, assets, "sky-orion-3d");
    let f = &m.flux[0];
    near("Rigel expected_lux", f.expected_lux, lux, 1e-9);
    near("Rigel R", f.radius_px, r, 1e-6);
    near("neighbour_lux", f.neighbour_lux, drawn, 1e-9);
    println!(
        "Rigel neighbour_lux {:.4e} lx = {:.1} % of its own {:.4e} lx",
        f.neighbour_lux,
        100.0 * f.neighbour_lux / f.expected_lux,
        f.expected_lux
    );
}

/// [`neighbours_lux`] over the stars the star pass draws at star EV `ev`
/// (`uni_sky::stars3d::star_is_visible`) and in front of the camera, as `measure` counts them.
fn drawn_neighbours_lux(
    world: &World,
    c: &Case,
    x: f64,
    y: f64,
    radius_px: f64,
    skip: Option<u32>,
    ev: f32,
) -> f64 {
    let years = world.epoch().seconds_since(Epoch::J2000) / crate::scene::JULIAN_YEAR_S;
    let cam_pc = crate::scene::camera_icrs_pc(&c.view.pos);
    let mut sum = 0.0;
    for (i, s) in world.sim.star_data().iter().enumerate() {
        if Some(i as u32) == skip || c.snap.suppressed_stars.contains(&(i as u32)) {
            continue;
        }
        let rel = crate::measure::star_pos_pc(s, years) - cam_pc;
        if rel.normalize_or_zero().dot(c.fwd()) < 0.1 {
            continue;
        }
        let Some((sx, sy)) = c.px(rel) else { continue };
        if (sx - x).hypot(sy - y) >= radius_px {
            continue;
        }
        let lux = uni_core::star_illuminance_lux(
            f64::from(s.abs_mag_v) + 5.0 * (rel.length() / 10.0).log10(),
        );
        let e = uni_sky::stars::star_display_energy_off_axis(
            lux as f32,
            c.view.fov_y_rad as f32,
            H as f32,
            ev,
            rel.normalize().dot(c.fwd()) as f32,
        );
        if uni_sky::stars3d::star_is_visible(e, uni_sky::stars3d::STARS3D_HALO_FRACTION) {
            sum += lux;
        }
    }
    sum
}

#[test]
fn intent_17_alpha_cen_sun() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    let c = Case::new(world, "alpha-cen-sun");
    let a = pos(world, "Rigil Kentaurus").delta_from(c.view.pos);
    let b = pos(world, "Toliman").delta_from(c.view.pos);
    near("0.05 pc from A", a.length() / PC, 0.05, 0.01);
    assert!(angle(a, c.fwd()).to_degrees() >= 90.0 && angle(b, c.fwd()).to_degrees() >= 90.0);
    let sun = pos(world, "Sun").delta_from(c.view.pos);
    assert!(angle(sun, c.fwd()) < 1e-9);
    // Looking at RA 39.9°, Dec +60.8° (α Cen's antipode).
    let v = radec_to_icrs(deg(39.902), deg(60.834));
    assert!(
        angle(c.fwd(), v).to_degrees() < 0.5,
        "{}",
        angle(c.fwd(), v).to_degrees()
    );
    let d_pc = sun.length() / PC;
    let v_sun = uni_core::M_V_SUN + 5.0 * (d_pc / 10.0).log10();
    assert!((v_sun - 0.40).abs() < 0.1, "Sun V {v_sun} at {d_pc} pc");
    // The Sun is a point sprite here (a 1.3 pc star is far below 1.5 px).
    let sun_id = world.sim.find("Sun").unwrap();
    let pt = c
        .snap
        .points
        .iter()
        .find(|p| p.id == Some(sun_id))
        .expect("the Sun as a point");
    near(
        "Sun point lux",
        pt.illuminance_lux,
        uni_core::star_illuminance_lux(v_sun),
        0.01,
    );
    // No catalogue star brighter than the Sun within 64 px.
    let years = world.epoch().seconds_since(Epoch::J2000) / crate::scene::JULIAN_YEAR_S;
    let cam_pc = crate::scene::camera_icrs_pc(&c.view.pos);
    for s in world.sim.star_data() {
        let rel = crate::measure::star_pos_pc(s, years) - cam_pc;
        let Some((x, y)) = c.px(rel) else { continue };
        if (x - W / 2.0).hypot(y - H / 2.0) < 64.0 {
            let v = f64::from(s.abs_mag_v) + 5.0 * (rel.length() / 10.0).log10();
            assert!(v > v_sun, "a V {v} star within 64 px of the Sun");
        }
    }
    // The other stars inside the Sun's flux radius (the plan: under 1 % of the Sun, with its
    // 21.0-px footprint). B4L's halo widened the footprint, so this records the share at the
    // plan's star EV −4 and at the default −6.5 and asserts only that it grows with the radius.
    let mut shares = Vec::new();
    for ev in [-4.0_f32, crate::scene::DEFAULT_STAR_EV100] {
        let e = uni_sky::stars::star_display_energy_off_axis(
            pt.illuminance_lux as f32,
            c.view.fov_y_rad as f32,
            H as f32,
            ev,
            1.0,
        );
        let r = f64::from(uni_sky::stars3d::footprint_radius_px(
            e,
            uni_sky::stars3d::STARS3D_HALO_FRACTION,
        ));
        let share = neighbours_lux(world, &c, W / 2.0, H / 2.0, r, None) / pt.illuminance_lux;
        shares.push((ev, r, share));
    }
    println!("Sun flux radius and neighbour share: {shares:?}");
    assert!(shares[1].1 >= shares[0].1 && shares[1].2 >= shares[0].2);
}

#[test]
fn intent_18_19_lens_bookmarks() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    // 18: m0.1's bh-shadow-30M exactly, with the debug grid in Manual at M0's EVs.
    let m0 = uni_nav::load_bookmarks(&crate::world::m0_bookmarks_path()).unwrap();
    let m0 = m0.iter().find(|b| b.name == "bh-shadow-30M").unwrap();
    let b18 = committed()
        .into_iter()
        .find(|b| b.name == "bh-shadow-30M-grid")
        .unwrap();
    assert_eq!(b18.view, m0.view);
    assert_eq!(b18.epoch_utc, m0.epoch_utc);
    assert_eq!(b18.lens_env, Some(uni_nav::LensEnvChoice::DebugGrid));
    assert_eq!(b18.exposure, Some(uni_nav::ExposureChoice::Manual));
    assert_eq!((b18.scene_ev100, b18.star_ev100), (Some(0.0), Some(-4.0)));
    assert_eq!(b18.lens_scale, Some(1.0));
    // 19: 30 M from TestBH-1, looking through it at the Galactic Centre; the Sun 86.9° off-axis.
    let c = Case::new(world, "bh-lens-band");
    let lens = &world.sim.lenses()[0];
    let m = gravitational_radius(lens.mass_kg);
    let hole = lens.pos.delta_from(c.view.pos);
    near("30 M", hole.length() / m, 30.0, 1e-6);
    assert!(angle(hole, c.fwd()) < 1e-7);
    let g = radec_to_icrs(deg(266.405), deg(-28.936));
    assert!(angle(c.fwd(), g) < 1e-9);
    let sun = pos(world, "Sun").delta_from(c.view.pos);
    near("Sun off-axis", angle(sun, c.fwd()).to_degrees(), 86.9, 0.01);
    assert_paper(
        world,
        assets.rings.as_ref(),
        &c,
        uni_post::NIGHT_EV100,
        None,
    );
}

// ---- Wave C intent tests (Task C6) -----------------------------------------------------------

/// The comet `name` of `c`'s snapshot (the comet pass draws it).
fn comet_view<'a>(world: &World, c: &'a Case, name: &str) -> &'a uni_sim::snapshot::CometView {
    let id = world.sim.find(name).unwrap();
    c.snap
        .comets
        .iter()
        .find(|v| v.id == id)
        .unwrap_or_else(|| panic!("{name} is not in the snapshot's comets"))
}

/// Row 20's paper EV, derived with the dust tail (C1L fix round 1, lead ruling C1L-b): the coma
/// twin (`paper::coma_luminance`, NEOWISE's cited visual law, V 2.1) lights only 44 pixels over
/// the meter's 10⁻³ cd/m², but the comet pass's dust-tail twin (`uni_planet::dust_tail_luminance`)
/// lights 5 550, whose 50th–95th percentile average is EV −5.10 (`intent_20_comet_neowise` prints
/// the paper). That is with NEOWISE's measured dust law (C1L fix round 2, lead ruling C1L-R1:
/// A(0)fρ ∝ r_h^−4.0, Biver et al. 2022; the tail 11.5 × the coma); with the K1 slope (fix
/// round 1) it was 1 219 pixels at −5.37. Before C1L the paper rested at the night EV (−4) with the
/// tails left out; the plan's "≈ −2 ± 2" assumed a larger, brighter coma.
pub const NEOWISE_PAPER_EV: f32 = -5.10;
/// Row 21's re-derived paper EV (Task C6): Halley's coma lights 173 000 pixels at 653 km/px;
/// the meter's 50th–95th percentile average sits at −4.56 (the plan's "≈ −2 ± 2" was an
/// estimate before the coma twin existed). With the dust-tail twin (C1L fix round 1) the paper
/// reads −4.51: from 2 × 10⁶ km the pass draws 6 % of the dust's light (its 64 px cap; the HUD
/// says so), which barely moves the meter.
pub const HALLEY_PAPER_EV: f32 = -4.56;

/// The paper exposure of a comet bookmark (with the coma twin, `paper::coma_luminance`, and the
/// dust-tail twin) against its re-derived row `row_ev`, ±0.5 EV as every intent test.
#[track_caller]
fn assert_comet_paper(world: &World, c: &Case, row_ev: f32) -> crate::paper::Paper {
    let p = crate::paper::on_paper(world, &c.view, SIZE, None);
    println!("paper: {p:?}");
    assert!((p.target_ev - row_ev).abs() <= 0.5, "{p:?} vs {row_ev}");
    p
}

#[test]
fn intent_20_comet_neowise() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    let c = Case::new(world, "comet-neowise-2020");
    near("fov", c.view.fov_y_rad.to_degrees(), 20.0, 1e-9);
    // 7 000 km from Earth's centre toward the Sun (629 km up, outside the 100 km atmosphere).
    let earth = pos(world, "Earth");
    let off = c.view.pos.delta_from(earth);
    near("7 000 km", off.length(), 7.0e6, 1e-9);
    assert!(angle(off, rel(world, "Earth", "Sun")) < 1e-9);
    let top = world.sim.objects()[world.sim.find("Earth").unwrap() as usize]
        .atmosphere_top_m
        .unwrap();
    assert!(off.length() > top, "outside the atmosphere");
    // Looking at the settled comet, which the comet pass draws (active, r_h < 5 AU).
    let v = comet_view(world, &c, NEOWISE_NAME);
    assert!(angle(c.fwd(), v.nucleus_rel_m) < 1e-9);
    // Its world position within the check's 5 px of the Horizons RA/Dec (sky_spots.ron).
    let cam = crate::measure::Cam::new(&c.view, SIZE);
    let spots =
        uni_data::load_sky_spots(&uni_data::data_dir().join("golden/sky_spots.ron")).unwrap();
    let spot = spots.iter().find(|s| s.target == "C/2020 F3").unwrap();
    let truth = cam
        .project(radec_to_icrs(deg(spot.ra_deg), deg(spot.dec_deg)))
        .unwrap();
    let drawn = cam.project(v.nucleus_rel_m).unwrap();
    let err = truth.distance(drawn);
    println!("NEOWISE theory vs Horizons: {err:.3} px");
    assert!(err < 5.0, "{err}");
    // The Sun (along `off`, 25.3° away) and Earth (behind) out of frame (half-diagonal 19.8°).
    assert!(angle(c.fwd(), off) > deg(25.0) && angle(c.fwd(), -off) > deg(90.0));
    assert_comet_paper(world, &c, NEOWISE_PAPER_EV);
    // The cited visual law (`uni_sim::world::COMET_VISUAL_LAWS`, fix round 1 I1): the observed
    // V 2.1 on the date (COBS median, 2020-07-14–16).
    let total = v.total_v.expect("NEOWISE has a magnitude law");
    println!(
        "NEOWISE total V {total:.2} (r_h {:.3} AU, Δ {:.3} AU)",
        v.r_au, v.delta_au
    );
    assert!((total - 2.1).abs() < 0.05, "{total}");
}

#[test]
fn intent_21_comet_halley() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    let c = Case::new(world, "comet-halley-1986");
    let v = comet_view(world, &c, HALLEY_NAME);
    near("2e6 km", v.nucleus_rel_m.length(), 2.0e9, 1e-6);
    // Side view: phase 90° (the camera on the orbit's normal), 653 km/px at fov 20°.
    let phase = angle(v.sun_rel_m, -v.nucleus_rel_m).to_degrees();
    near("phase", phase, 90.0, 0.01);
    let km_per_px = 2.0e6 * 2.0 * deg(10.0).tan() / H;
    near("km/px", km_per_px, 653.0, 0.01);
    assert!(angle(c.fwd(), v.nucleus_rel_m) < 1e-9);
    // The Sun to the left: anti-sunward points to +x on the image.
    let tail = c.view.orient.conjugate() * -v.sun_rel_m;
    assert!(tail.x > 0.99 * tail.length(), "{tail}");
    // The coma's expected slope (−1 − ⟨ρ/ℓ⟩ over 8–40 px) and its feature at the frame centre.
    let p = crate::measure::Cam::new(&c.view, SIZE)
        .project(v.nucleus_rel_m)
        .unwrap();
    assert!(p.distance(glam::DVec2::new(W / 2.0, H / 2.0)) < 1e-6, "{p}");
    let scale_km = uni_planet::comet::coma::coma_scale_m(v.r_au) / 1e3;
    let mean_rho_over_l = 24.0 * km_per_px / scale_km;
    println!("ℓ = {scale_km:.0} km, ⟨ρ/ℓ⟩ ≈ {mean_rho_over_l:.3}");
    assert!((0.05..0.5).contains(&mean_rho_over_l));
    assert_comet_paper(world, &c, HALLEY_PAPER_EV);
}

#[test]
fn intent_22_asteroid_belt_above() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    let c = Case::new(world, "asteroid-belt-above");
    let (b, _) = applied(world, "asteroid-belt-above");
    assert_eq!(b.highlight_belt, Some(true));
    near("fov", c.view.fov_y_rad.to_degrees(), 70.0, 1e-9);
    let sun = pos(world, "Sun").delta_from(c.view.pos);
    near("6 AU", sun.length() / AU, 6.0, 1e-9);
    let pole = uni_physics::frames::ecliptic_j2000_to_icrs(DVec3::Z);
    assert!(angle(-sun, pole) < 1e-9, "above the ecliptic");
    assert!(angle(c.fwd(), sun) < 1e-9);
    // 128.5 px/AU at the Sun (rectilinear: f / 6 AU, f = 540 / tan 35°).
    near("px/AU", 540.0 / deg(35.0).tan() / 6.0, 128.5, 0.01);
    // The belt from the CPU positions: the 3:1 Kirkwood gap in a (lead ruling, C6 fix round 1:
    // a, not the plan's projected r, where eccentricities smear the gap), and the Trojan clouds.
    let belt = uni_sky::BeltModel::from_catalogue(
        world.sim.asteroid_records(),
        crate::scene::BELT_MAIN,
        crate::scene::BELT_KUIPER,
        uni_sky::belt::BELT_H_CUT,
        uni_sky::belt::BELT_SEED,
    );
    let assets = crate::world::Assets {
        facts: Vec::new(),
        rings: None,
        maps: Vec::new(),
        map_dir: std::path::PathBuf::new(),
        warnings: Vec::new(),
    };
    let m = measured_with(world, &assets, "asteroid-belt-above", Some(&belt));
    let h = m
        .metrics
        .belt_radial_hist
        .expect("belt_radial_hist with the highlight");
    println!(
        "belt: Kirkwood 3:1 dip by a {:.3}, by projected r {:.3}; L4 {:.1}×, L5 {:.1}× the \
         background",
        h.kirkwood_dip_a, h.kirkwood_dip_r, h.l4_over_bg, h.l5_over_bg
    );
    assert!(h.kirkwood_dip_a < 0.3, "{}", h.kirkwood_dip_a);
    assert!(h.l4_over_bg > 3.0 && h.l5_over_bg > 3.0);
    // Every drawn asteroid within 6 AU: most of the 33 678 records and the belt's
    // `BELT_MAIN` main-belt and Trojan objects (80 000 since the ladder step, §7 of the report).
    let within: u64 = h.counts.iter().map(|&n| u64::from(n)).sum();
    assert!(within > u64::from(crate::scene::BELT_MAIN), "{within}");
    for r in ["belt_annulus", "belt_outer"] {
        assert!(m.regions.get(r).is_some_and(|v| !v.is_empty()), "{r}");
    }
    // No world light on screen: the night EV, as the plan's row for 22 says.
    assert_paper(world, None, &c, uni_post::NIGHT_EV100, None);
}

#[test]
fn intent_23_exo_trappist1() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    let c = Case::new(world, "exo-trappist1");
    let id = world.sim.find(TRAPPIST_NAME).unwrap();
    let host = world.sim.position(id).delta_from(c.view.pos);
    near("0.2 AU", host.length() / AU, 0.2, 1e-9);
    assert!(angle(c.fwd(), host) < 1e-9);
    // The host is drawn (an archive star, review I11): a star surface or a point.
    let drawn = c.snap.star_surfaces.iter().any(|s| s.id == id)
        || c.snap.points.iter().any(|p| p.id == Some(id));
    assert!(drawn, "TRAPPIST-1 not drawn");
    // Selected (the bookmark's target): seven dashed orbits on the overlay.
    let (b, _) = applied(world, "exo-trappist1");
    assert_eq!(b.target.as_deref(), Some(TRAPPIST_NAME));
    let snap = world.sim.snapshot(&c.view, SIZE);
    let o = uni_ui::build_overlay(
        &world.sim,
        &snap,
        &c.view,
        SIZE,
        SIZE.as_vec2(),
        Some(id),
        None,
        None,
        &crate::scene::OVERLAY_SETTINGS,
    );
    let n = crate::measure::overlay_counts(&o);
    assert_eq!(n.dashed, 7, "{n:?}");
    // The Sun out of frame (half-diagonal 49.66°).
    let sun = pos(world, "Sun").delta_from(c.view.pos);
    assert!(angle(sun, c.fwd()).to_degrees() > 49.66);
    // The host's disk is the only world light (2.6 px across): paper against the re-derived row.
    let p = crate::paper::on_paper(world, &c.view, SIZE, None);
    println!("paper: {p:?}");
    assert!((p.target_ev - TRAPPIST_PAPER_EV).abs() <= 0.5, "{p:?}");
}

/// Row 23's re-derived paper EV (Task C6): TRAPPIST-1's resolved disk (2.6 px in radius, ≈ 76
/// px over the meter's threshold) is world light, so the meter leaves the night EV and ramps
/// toward the disk's ev_avg (22.8) between `METER_MIN_LIT_PX` and `METER_FULL_LIT_PX` (the plan's
/// table lumped row 23 with the sky views at the night EV). A few pixels move it by about an EV
/// each there (re-review r1 minor 6's ramp), so the capture's EV can differ by more than 0.5.
pub const TRAPPIST_PAPER_EV: f32 = -1.18;

#[test]
fn intent_24_25_hud_bookmarks() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    // 24: #5's view with Mars as the target; its card in the mockup's order.
    let all = committed();
    let five = all
        .iter()
        .find(|b| b.name == "mars-reference-angle")
        .unwrap();
    let b24 = all.iter().find(|b| b.name == "hud-target-mars").unwrap();
    assert_eq!((&b24.view, &b24.epoch_utc), (&five.view, &five.epoch_utc));
    assert_eq!(b24.target.as_deref(), Some("Mars"));
    let c = Case::new(world, "hud-target-mars");
    let mars = world.sim.find("Mars").unwrap();
    let card = uni_ui::build_card(&world.sim, mars, &c.view).unwrap();
    let rows: Vec<&str> = card
        .live
        .iter()
        .chain(card.facts.iter())
        .map(|r| r.label.as_str())
        .collect();
    assert_eq!(rows, MARS_CARD_ROWS);
    // 25: halfway between Earth and Mars, looking at Mars, cruising: the cruise label.
    let c = Case::new(world, "hud-cruise");
    let e = pos(world, "Earth").delta_from(c.view.pos);
    let m = pos(world, "Mars").delta_from(c.view.pos);
    near("halfway", e.length() / m.length(), 1.0, 1e-9);
    assert!(angle(c.fwd(), m) < 1e-9);
    let (b, a) = applied(world, "hud-cruise");
    assert_eq!(b.cruise, Some(true));
    let ui = crate::capture::bookmark_ui(&b, world).unwrap();
    assert_eq!(ui.flight, uni_ui::FlightLabel::Cruise);
    let s = crate::scene::exposure_settings(
        &crate::scene::default_controls(1.0),
        a.choice,
        false,
        true,
    );
    let labels = crate::scene::honesty_labels(
        world,
        &c.view,
        SIZE,
        &c.snap,
        &s,
        &crate::scene::LabelInputs {
            lens_env: uni_rel::LensEnv::Band,
            coverage: uni_rel::LensCoverage::Unlensed,
            band_on: true,
            scintillation: 0.0,
            vivid: true,
            belt_visible: false,
            ui,
        },
    );
    assert_eq!(
        labels[0],
        (
            uni_ui::CRUISE_LABEL.to_owned(),
            uni_ui::LabelTone::NonPhysical
        )
    );
}

/// The coordinator's rulings at Task C6 on the `--hud` views: no orbit line without a selection;
/// the selected planet's orbit and its major moons' only, none over a body's disk; markers for
/// the Sun, the planets, the Moon (from Earth) and TestBH-1 always, and at most three "others"
/// (dwarf planets, comets, asteroids, hosts), with no two labels overlapping.
#[test]
fn hud_overlay_is_soft_and_uncluttered() {
    let mut g = full_world();
    let (world, _) = &mut *g;
    use uni_sim::registry::ObjectKind;
    let overlay = |world: &World, c: &Case, selected| {
        uni_ui::build_overlay(
            &world.sim,
            &c.snap,
            &c.view,
            SIZE,
            SIZE.as_vec2(),
            selected,
            None,
            None,
            &crate::scene::OVERLAY_SETTINGS,
        )
    };
    // `start`: Earth from 60 000 km, nothing selected.
    let c = Case::new(world, "start");
    let o = overlay(world, &c, None);
    assert!(o.orbits.is_empty(), "{} orbit lines", o.orbits.len());
    let kinds: Vec<ObjectKind> = o
        .markers
        .iter()
        .map(|m| world.sim.objects()[m.id as usize].kind)
        .collect();
    let others = kinds
        .iter()
        .filter(|k| {
            matches!(
                k,
                ObjectKind::DwarfPlanet
                    | ObjectKind::Comet
                    | ObjectKind::Asteroid
                    | ObjectKind::ExoHost
                    | ObjectKind::Star
            )
        })
        .count();
    println!(
        "start: {} markers ({others} others): {kinds:?}",
        o.markers.len()
    );
    assert!(others <= uni_sim::snapshot::MAX_BRIGHT_DESTINATIONS);
    let labels: Vec<(f32, f32, String)> = o
        .markers
        .iter()
        .filter_map(|m| m.label.clone().map(|l| (m.pt.x, m.pt.y, l)))
        .collect();
    for (i, a) in labels.iter().enumerate() {
        for b in &labels[i + 1..] {
            // Same row band and horizontally close: labels would overlap.
            assert!(
                (a.1 - b.1).abs() > 12.0 || (a.0 - b.0).abs() > 30.0,
                "{a:?} and {b:?}"
            );
        }
    }
    // `hud-target-mars`: Mars selected; its orbit and Phobos's and Deimos's, none over the disk.
    let c = Case::new(world, "hud-target-mars");
    let mars = world.sim.find("Mars").unwrap();
    let o = overlay(world, &c, Some(mars));
    assert!(!o.orbits.is_empty());
    let b = c.body(world, "Mars");
    let centre = crate::measure::Cam::new(&c.view, SIZE)
        .project(b.rel_pos_m)
        .unwrap()
        .as_vec2();
    let r = b.proj_radius_px as f32;
    for l in &o.orbits {
        assert!(
            l.pts.iter().all(|p| p.distance(centre) >= r),
            "an orbit point over Mars's disk"
        );
    }
}

/// The mockup's Mars card rows in order (C4's `mars_card_rows_in_mockup_order`): the live rows,
/// then the facts.
pub const MARS_CARD_ROWS: [&str; 15] = [
    "Distance",
    "Closing speed",
    "Light time",
    "Mean radius",
    "Mass",
    "Surface gravity",
    "Escape velocity",
    "Sidereal day",
    "Orbital period",
    "Semi-major axis",
    "Axial tilt",
    "Surface pressure",
    "Atmosphere",
    "Mean temperature",
    "Moons",
];

#[test]
fn intent_26_sky_orion_50kyr() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let all = committed();
    let fifteen = all.iter().find(|b| b.name == "sky-orion-3d").unwrap();
    let b26 = all.iter().find(|b| b.name == "sky-orion-50kyr").unwrap();
    assert_eq!(b26.view, fifteen.view);
    // 50 000 Gregorian years after row 15's date: 50 000 × 365.2425 days.
    let t15 = epoch(SPOT_EPOCH_UTC);
    let t26 = crate::scene::parse_bookmark_epoch(ORION_50KYR_UTC).unwrap();
    near(
        "50 kyr [days]",
        t26.seconds_since(t15) / uni_core::DAY,
        50_000.0 * 365.2425,
        1e-12,
    );
    let c = Case::new(world, "sky-orion-50kyr");
    assert_eq!(world.epoch(), t26);
    // The four stars in frame at their +50 kyr positions.
    let m = measured(world, assets, "sky-orion-50kyr");
    for n in ["Betelgeuse", "Rigel", "Bellatrix", "Alnilam"] {
        let p = m.projected.iter().find(|p| p.name == n);
        assert!(
            p.is_some_and(|p| (0.0..W).contains(&p.x) && (0.0..H).contains(&p.y)),
            "{n}"
        );
    }
    // Planets extrapolated (outside 3000 BC – AD 3000), stars not (|Δt| < 100 kyr).
    let flags = world.sim.honesty(&c.view, SIZE);
    use uni_sim::honesty::HonestyFlag;
    assert!(flags.contains(&HonestyFlag::PlanetsExtrapolated));
    assert!(!flags.contains(&HonestyFlag::StarsExtrapolated));
    assert_paper(world, None, &c, uni_post::NIGHT_EV100, None);
}

#[test]
fn intent_27_bh_lens_seam() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    let c = Case::new(world, "bh-lens-seam");
    let lens = &world.sim.lenses()[0];
    let m_len = gravitational_radius(lens.mass_kg);
    let hole = lens.pos.delta_from(c.view.pos);
    near("4 500 M", hole.length() / m_len, SEAM_DISTANCE_M, 1e-6);
    assert!(angle(hole, c.fwd()) < 1e-7);
    near("fov", c.view.fov_y_rad.to_degrees(), SEAM_FOV_DEG, 1e-9);
    // Traced, and the rectangle is smaller than the frame: the full height, part of the width.
    let ctx = crate::scene::frame_ctx(
        world,
        &c.view,
        SIZE,
        SIZE,
        uni_gfx::Exposure {
            scene_ev100: 0.0,
            star_ev100: -6.5,
        },
        0,
        0.0,
        &c.snap,
    );
    assert_eq!(uni_rel::lens_coverage(&ctx), uni_rel::LensCoverage::Traced);
    let cam = crate::measure::Cam::new(&c.view, SIZE);
    let (lo, hi) = crate::measure::lens_rect(&cam, &ctx.lenses[0]).unwrap();
    println!("lens rectangle: x {}–{}, y {}–{}", lo.x, hi.x, lo.y, hi.y);
    // 1 234 of 1 920 columns (64 %): both side edges well inside the frame.
    assert!(
        lo.x > 0.1 * W && hi.x < 0.9 * W && hi.x - lo.x > 0.5 * W,
        "{lo} {hi}"
    );
    assert_eq!((lo.y, hi.y), (0.0, H));
    let m = measured(world, assets, "bh-lens-seam");
    for r in ["lens_rect_inner_edge", "lens_rect_outer_edge"] {
        let n = m.regions.get(r).map_or(0, Vec::len);
        assert!(n > 0 && n <= crate::measure::MAX_REGION_SAMPLES, "{r}: {n}");
    }
    // The Sun out of frame (half-diagonal 67.7° at fov 100°).
    let sun = pos(world, "Sun").delta_from(c.view.pos);
    assert!(angle(sun, c.fwd()).to_degrees() > 70.0);
    assert_paper(world, None, &c, uni_post::NIGHT_EV100, None);
}

/// The capture JSON's CPU-side measurements for bookmark `name` (no GPU: no taps).
fn measured(
    world: &mut World,
    assets: &crate::world::Assets,
    name: &str,
) -> crate::measure::Measurements {
    measured_with(world, assets, name, None)
}

/// [`measured`] with the procedural belt `belt` (for `belt_radial_hist`).
fn measured_with(
    world: &mut World,
    assets: &crate::world::Assets,
    name: &str,
    belt: Option<&uni_sky::BeltModel>,
) -> crate::measure::Measurements {
    let (b, a) = applied(world, name);
    let controls = crate::scene::default_controls(1.0);
    let settings = crate::scene::exposure_settings(&controls, a.choice, false, true);
    let snapshot = world.sim.snapshot(&a.view, SIZE);
    let ctx = crate::scene::frame_ctx(
        world,
        &a.view,
        SIZE,
        SIZE,
        settings.manual,
        0,
        0.0,
        &snapshot,
    );
    let frame = crate::scene::Frame {
        coverage: uni_rel::lens_coverage(&ctx),
        ctx,
        snapshot,
        settings,
        labels: Vec::new(),
        ui: crate::capture::bookmark_ui(&b, world).unwrap(),
        snapshot_cpu_ms: 0.0,
        reflected_scale: 1.0,
    };
    crate::measure::measure(&crate::measure::Input {
        name,
        world,
        frame: &frame,
        ev100: [10.0, -6.5],
        world_hdr: None,
        v_lim: 10.0,
        comets_settle_ms: 0.0,
        rings: assets.rings.as_ref(),
        maps: &assets.maps,
        map_dir: &assets.map_dir,
        belt,
        overlay: None,
        comets_cpu_ms: None,
        asteroids_cpu_ms: None,
        asteroid_counts: None,
    })
}

/// Rows 3–5's 16 map samples and every bookmark's regions exist (B9 Step 2: the JSON the gate's
/// HDR checks read), and each region has samples.
#[test]
fn capture_measurements_cover_the_plan_checks() {
    let mut g = full_world();
    let (world, assets) = &mut *g;
    for name in ["earth-map-closeup", "moon-full", "mars-reference-angle"] {
        let m = measured(world, assets, name);
        assert_eq!(m.samples.len(), 16, "{name}");
        assert!(
            m.samples
                .iter()
                .all(|s| s.expected_cd_m2.iter().all(|c| *c > 0.0)),
            "{name}"
        );
    }
    for (name, regions) in [
        ("earth-limb-dawn", &["limb_arc_high", "limb_arc_low"][..]),
        ("mars-reference-angle", &["limb_haze"]),
        (
            "saturn-ring-plane",
            &[
                "ring_cassini",
                "ring_b",
                "globe_ring_shadow",
                "globe_lit_ref",
            ],
        ),
        (
            "saturn-night-side",
            &["ring_in_planet_shadow", "ring_lit_same_radius"],
        ),
        ("sun-closeup", &["mu_0p5", "disk_centre"]),
        ("sun-granulation", &["granulation_patch"]),
        ("titan-backlit", &["haze_ring"]),
        ("sky-galactic-center", &["band_plane", "band_high_lat"]),
    ] {
        let m = measured(world, assets, name);
        for r in regions {
            let n = m.regions.get(*r).map_or(0, Vec::len);
            assert!(
                n > 0 && n <= crate::measure::MAX_REGION_SAMPLES,
                "{name} {r}: {n}"
            );
        }
    }
    let m = measured(world, assets, "sky-orion-3d");
    assert_eq!(m.flux.len(), 1);
    assert_eq!(m.flux[0].name, "Rigel");
    for n in ["Betelgeuse", "Rigel", "Bellatrix", "Alnilam"] {
        assert!(m.projected.iter().any(|p| p.name == n), "{n}");
    }
    let m = measured(world, assets, "alpha-cen-sun");
    assert_eq!(m.flux.len(), 1);
    assert_eq!(m.flux[0].name, "Sun");
    // Row 13 (lead ruling "row 13 Venus haze"; fix round 1, review I1): the night limb across
    // the haze, for B8's `limb`, in the lit azimuth bins only (the two cusp arcs), each with
    // its own sky.
    let m = measured(world, assets, "venus-crescent");
    let limb = m
        .limbs
        .iter()
        .find(|l| l.name == "venus_night_limb")
        .expect("venus_night_limb");
    let cam = crate::measure::Cam::new(&applied(world, "venus-crescent").1.view, SIZE);
    let omega = |q: &[f64; 4]| cam.pixel_solid_angle(glam::DVec2::new(q[0], q[1]));
    let (mut twin, mut deck, mut sky_px, mut strip_px) = (0.0, 0.0, 0, 0);
    for bin in &limb.bins {
        let lux: f64 = bin.strip.iter().map(|q| q[2] * omega(q)).sum();
        twin += lux;
        deck += bin
            .strip
            .iter()
            .filter(|q| q[3] == 0.0)
            .map(|q| q[2] * omega(q))
            .sum::<f64>();
        sky_px += bin.sky.len();
        strip_px += bin.strip.len();
        // Every kept bin lies on the night half, near a cusp (|ψ| 90°–130°).
        let mid = 0.5 * (bin.psi_deg[0] + bin.psi_deg[1]);
        assert!((88.0..=130.0).contains(&mid.abs()), "{:?}", bin.psi_deg);
        assert!(bin.strip.iter().all(|q| (0.0..=1.0).contains(&q[3])));
    }
    println!(
        "venus_night_limb: {} bins, {strip_px} strip px, {sky_px} sky px, twin {twin:.4e} lx \
         ({:.1} % on the deck)",
        limb.bins.len(),
        100.0 * deck / twin
    );
    assert!(
        limb.bins.len() >= 4 && limb.bins.len() <= 40,
        "{}",
        limb.bins.len()
    );
    assert!(twin > 0.0 && sky_px > 100 && strip_px > 50);
    assert!(!m.regions.contains_key("night_limb_haze"));
    // Row 19 (lead ruling "row 19 shadow"): visual and seam checks only, so the JSON carries no
    // regions, samples or flux for it (the `lens_rect_*_edge` regions cannot exist either: at
    // 30 M the traced rectangle is the whole frame; #27 `bh-lens-seam` has them).
    let m = measured(world, assets, "bh-lens-band");
    assert!(m.regions.is_empty() && m.samples.is_empty() && m.flux.is_empty());
    // Wave C: the comets' features (row 20's ion-tail direction, row 21's coma profile) and
    // NEOWISE's Horizons position.
    let m = measured(world, assets, "comet-neowise-2020");
    let f = &m.features["ion_tail"];
    assert!(f.angle_deg.is_some_and(f64::is_finite) && f.r_max > f.r_min);
    assert!(m.projected.iter().any(|p| p.name == "C/2020 F3 Horizons"));
    let m = measured(world, assets, "comet-halley-1986");
    let f = &m.features["coma"];
    assert!((f.r_min, f.r_max) == (8.0, 40.0));
    assert!(
        f.expected_slope.is_some_and(|s| (-1.6..-1.0).contains(&s)),
        "{f:?}"
    );
}
