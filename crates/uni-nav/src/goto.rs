//! The animated goto (Task C3; Task C0 stub).
//!
//! Lead ruling on the C0 context (owner point 2), after the owner-approved preview
//! (`72-nav.js`):
//! - **Turn first** to face the target, with the quintic ease x³(6x² − 15x + 10) over
//!   [`turn_duration_s`] = 0.40 + 0.55·turn/π s (it replaces the plan's
//!   `GOTO_TURN_RATE_DEG_S`). The camera holds its place meanwhile.
//! - **Then travel** for [`GOTO_DURATION_S`] = 4.5 s from any distance: ln(d − d_stop) moves along
//!   the distance share of [`goto_profile`], whose speed share is
//!   smoothstep(0, [`GOTO_ACCEL_FRAC`], x)·(1 − smoothstep(1 − [`GOTO_DECEL_FRAC`], 1, x)) and whose
//!   distance share is its integral, normalised (accelerate, cruise, decelerate, with parallax).
//!   It runs from the gap at the start of the travel to [`ARRIVAL_FRAC`]·d_stop at x = 1, so the
//!   travel always takes the whole 4.5 s. Short hops (< 10 × d_stop) use the same profile. A
//!   camera inside the stand-off backs out along the same profile (on |d − d_stop|). The camera
//!   moves straight along the line to the target, looking at it. The whole goto takes
//!   `turn_duration_s(turn) + GOTO_DURATION_S`, independent of the distance
//!   (`goto_duration_independent_of_distance`, review m20).
//! - **Arrival** when d − d_stop < [`ARRIVAL_FRAC`]·d_stop: free flight, `DropOut::Arrived`. That
//!   is the end of the travel (x = 1), where the camera is put at d_stop exactly.
//! - **Stand-off** ([`standoff_distance`], plan C3): bodies 4 × their largest radius, or 1.5 × the
//!   atmosphere top if larger; the Sun and stars 1 AU × √(L/L☉) but at least 10 R★; comets
//!   10⁵ km; asteroids 10 × their radius (at least 50 km); exoplanet hosts as stars; TestBH-1
//!   100 GM/c².
//! - **Following the target** (lead ruling A.5). A target with a frame: the camera is reanchored
//!   to it when the goto starts, so the app's `FlyCamera::sync` carries the camera with it. A
//!   frameless target (comet, asteroid, star, exoplanet host): each frame the step adds the
//!   target's motion relative to the camera's anchor over the coming tick (its
//!   `World::velocity` × the sim step, less the anchor frame's own motion from the frame tree),
//!   so the camera is where it should be relative to the target after the app's tick and sync.

use std::f64::consts::PI;

use glam::{DQuat, DVec3};
use uni_core::{AU, C, FrameId, R_SUN_M, TEFF_SUN_K};
use uni_sim::registry::{ObjectId, ObjectInfo, ObjectKind};
use uni_sim::world::World;

use crate::camera::FlyCamera;

/// Travel time of a goto after its turn [s], independent of the distance (lead ruling, owner
/// point 2).
pub const GOTO_DURATION_S: f64 = 4.5;
/// Share of the travel spent accelerating (the preview's smoothstep(0, 0.30, x)).
pub const GOTO_ACCEL_FRAC: f64 = 0.30;
/// Share of the travel spent decelerating (the preview's 1 − smoothstep(0.64, 1, x)).
pub const GOTO_DECEL_FRAC: f64 = 0.36;
/// The turn's fixed part [s] (0.40 + 0.55·turn/π).
pub const GOTO_TURN_BASE_S: f64 = 0.40;
/// The turn's part per π of turn [s].
pub const GOTO_TURN_PER_PI_S: f64 = 0.55;
/// A goto has arrived when the distance left to the stand-off point is below this share of the
/// stand-off distance (plan C3).
pub const ARRIVAL_FRAC: f64 = 1.0e-3;

/// Bodies (planets, dwarf planets, moons): stand-off in largest radii (plan C3).
pub const BODY_STANDOFF_RADII: f64 = 4.0;
/// Bodies with an atmosphere: at least this many times its top radius (plan C3), so a goto stops
/// outside it (Review Focus 2).
pub const ATMOSPHERE_STANDOFF_FACTOR: f64 = 1.5;
/// Stars: the stand-off is at least this many photosphere radii (plan C3).
pub const STAR_MIN_STANDOFF_RADII: f64 = 10.0;
/// Comets: stand-off [m] (plan C3: 10⁵ km, outside the inner coma).
pub const COMET_STANDOFF_M: f64 = 1.0e8;
/// Asteroids: stand-off in radii (plan C3) ...
pub const ASTEROID_STANDOFF_RADII: f64 = 10.0;
/// ... but at least this far [m] (plan C3: 50 km).
pub const ASTEROID_MIN_STANDOFF_M: f64 = 50.0e3;
/// Black holes: stand-off in units of M = GM/c² (plan C3: TestBH-1's 100 GM/c², the mockup's
/// exit point; geometrised units, Misner, Thorne & Wheeler 1973).
pub const BLACK_HOLE_STANDOFF_M: f64 = 100.0;

// The profile below is a product of a rising and a falling smoothstep that do not overlap.
const _: () = assert!(GOTO_ACCEL_FRAC + GOTO_DECEL_FRAC <= 1.0);
/// ∫₀¹ v(x) dx of [`goto_profile`]'s speed share: each smoothstep ramp holds half its width
/// (∫₀¹ (3u² − 2u³) du = ½), so 1 − a/2 − d/2 = 0.67.
const PROFILE_INTEGRAL: f64 = 1.0 - 0.5 * GOTO_ACCEL_FRAC - 0.5 * GOTO_DECEL_FRAC;

/// Duration [s] of the turn through `turn_rad` (0 ..= π) before the travel: 0.40 + 0.55·turn/π.
/// A turn that is not a number counts as π.
pub fn turn_duration_s(turn_rad: f64) -> f64 {
    let turn = if turn_rad.is_finite() {
        turn_rad.clamp(0.0, PI)
    } else {
        PI
    };
    GOTO_TURN_BASE_S + GOTO_TURN_PER_PI_S * turn / PI
}

/// The quintic ease x³(6x² − 15x + 10) of the turn, for x in 0 ..= 1 (clamped; not a number
/// gives 0): zero speed and acceleration at both ends (Perlin 2002, "Improving noise").
pub fn quintic_ease(x: f64) -> f64 {
    if !(x > 0.0) {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    x * x * x * (x * (6.0 * x - 15.0) + 10.0)
}

/// The cubic smoothstep 3u² − 2u³ for u in 0 ..= 1 (the preview's `smooth`).
fn smoothstep(u: f64) -> f64 {
    u * u * (3.0 - 2.0 * u)
}

/// ∫₀ᵘ smoothstep = u³ − u⁴/2.
fn smoothstep_integral(u: f64) -> f64 {
    u * u * u * (1.0 - 0.5 * u)
}

/// The travel profile at x = elapsed ÷ [`GOTO_DURATION_S`] (0 ..= 1): (distance share 0 ..= 1,
/// speed share 0 ..= 1). See the module docs. The speed share peaks at 1 between the ramps, so
/// it is also the speed ÷ the top speed; the distance share grows at speed share ÷
/// `PROFILE_INTEGRAL` (0.67) per unit of x. Clamped outside 0 ..= 1; not a number gives 0.
pub fn goto_profile(x: f64) -> (f64, f64) {
    let (a, d) = (GOTO_ACCEL_FRAC, GOTO_DECEL_FRAC);
    if !(x > 0.0) {
        return (0.0, 0.0);
    }
    if x >= 1.0 {
        return (1.0, 0.0);
    }
    let (integral, speed) = if x < a {
        let u = x / a;
        (a * smoothstep_integral(u), smoothstep(u))
    } else if x <= 1.0 - d {
        (0.5 * a + (x - a), 1.0)
    } else {
        let w = (x - (1.0 - d)) / d;
        (
            0.5 * a + (1.0 - d - a) + d * (w - smoothstep_integral(w)),
            1.0 - smoothstep(w),
        )
    };
    ((integral / PROFILE_INTEGRAL).min(1.0), speed)
}

/// Stand-off [m] of a star of photosphere radius `radius_m` and `teff_k` (module docs): 1 AU ×
/// √(L/L☉), the distance at which it gives the flux the Sun gives at 1 AU (inverse-square law),
/// at least [`STAR_MIN_STANDOFF_RADII`] radii. L/L☉ = (R/R☉)²(T/T☉)⁴ (Stefan–Boltzmann law, with
/// the IAU 2015 B3 nominal R☉ and T☉), so √(L/L☉) = (R/R☉)(T/T☉)². A temperature that is not a
/// positive number counts as T☉.
pub fn star_standoff_m(radius_m: f64, teff_k: f64) -> f64 {
    let t = if teff_k.is_finite() && teff_k > 0.0 {
        teff_k / TEFF_SUN_K
    } else {
        1.0
    };
    let flux_match = AU * (radius_m / R_SUN_M) * t * t;
    flux_match.max(STAR_MIN_STANDOFF_RADII * radius_m)
}

/// The distance [m] from `info`'s centre at which a goto stops (module docs). Never below the
/// object's largest radius; 4 × that radius (or 1 km) if a rule gives no positive number.
pub fn standoff_distance(info: &ObjectInfo, world: &World) -> f64 {
    let r_max = info.radii_m.max_element();
    let s = match info.kind {
        ObjectKind::Planet | ObjectKind::DwarfPlanet | ObjectKind::Moon => {
            let atmosphere = info
                .atmosphere_top_m
                .map_or(0.0, |top| ATMOSPHERE_STANDOFF_FACTOR * top);
            (BODY_STANDOFF_RADII * r_max).max(atmosphere)
        }
        // L = L☉ by definition (IAU 2015 B3).
        ObjectKind::Sun => AU.max(STAR_MIN_STANDOFF_RADII * info.radii_m.x),
        ObjectKind::Star | ObjectKind::ExoHost => match info.catalog_index {
            // The radius and temperature the renderer draws the star with.
            Some(i) => star_standoff_m(world.star_radius_m(i), world.star_teff_k(i)),
            None => star_standoff_m(info.radii_m.x, TEFF_SUN_K),
        },
        ObjectKind::Comet => COMET_STANDOFF_M,
        ObjectKind::Asteroid => (ASTEROID_STANDOFF_RADII * r_max).max(ASTEROID_MIN_STANDOFF_M),
        ObjectKind::BlackHole => BLACK_HOLE_STANDOFF_M * info.gm_m3s2 / (C * C),
    };
    if s.is_finite() && s > 0.0 {
        s.max(r_max)
    } else {
        (BODY_STANDOFF_RADII * r_max).max(1.0e3)
    }
}

/// A goto in flight: the navigator's state between frames.
#[derive(Clone, Debug)]
pub(crate) struct GotoFlight {
    pub(crate) target: ObjectId,
    /// The target's frame, which the camera rides during the goto (`None`: frameless).
    frame: Option<FrameId>,
    standoff_m: f64,
    /// Wall time since the goto started [s].
    elapsed_s: f64,
    turn_s: f64,
    /// The camera's orientation when the goto started.
    q0: DQuat,
    /// Set on the first frame of the travel.
    travel: Option<Travel>,
}

/// The travel's start: ln |d − d_stop| and its sign (−1: backing out to the stand-off).
#[derive(Clone, Copy, Debug)]
struct Travel {
    ln_gap0: f64,
    sign: f64,
}

/// What one frame of a goto did.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GotoStep {
    pub(crate) arrived: bool,
    /// Closing speed toward the stand-off point [m/s].
    pub(crate) speed_mps: f64,
    /// Share of the whole goto done, 0 ..= 1.
    pub(crate) progress: f64,
    /// |d − d_stop| after the step [m].
    pub(crate) to_go_m: f64,
    /// Time left [s].
    pub(crate) eta_s: f64,
    /// The camera's velocity relative to the target this frame [m/s, root axes]: zero in the
    /// turn and on arrival (the last 10⁻³ of the stand-off is a placement, not motion).
    pub(crate) velocity_mps: DVec3,
}

impl GotoFlight {
    /// Starts a goto to `target` from where `cam` is and looks now; reanchors `cam` to the
    /// target's frame if it has one (plan C3). `None` if `target` is not in the registry.
    pub(crate) fn start(target: ObjectId, cam: &mut FlyCamera, world: &World) -> Option<Self> {
        let info = world.objects().get(target as usize)?;
        if let Some(frame) = info.frame {
            cam.reanchor(world.frames(), world.epoch(), frame);
        }
        let to_target = world.position(target).delta_from(cam.view.pos);
        let turn = forward(cam).angle_between(to_target);
        Some(GotoFlight {
            target,
            frame: info.frame,
            standoff_m: standoff_distance(info, world),
            elapsed_s: 0.0,
            turn_s: turn_duration_s(if turn.is_finite() { turn } else { 0.0 }),
            q0: cam.view.orient,
            travel: None,
        })
    }

    /// The frame the app anchors the camera to during this goto (`NavStatus::anchor`).
    pub(crate) fn anchor(&self) -> Option<FrameId> {
        self.frame
    }

    /// The whole goto's duration [s]: the turn plus the travel.
    fn total_s(&self) -> f64 {
        self.turn_s + GOTO_DURATION_S
    }

    /// Advances `dt` wall seconds: turns, then moves `cam` (by `FlyCamera::translate`) so that
    /// after the app's tick and sync it is on the profile, looking at the target.
    pub(crate) fn step(&mut self, cam: &mut FlyCamera, world: &World, dt: f64) -> GotoStep {
        self.elapsed_s += dt;
        let drift = self.tick_drift(cam, world, dt);
        // Camera − target, now and (before this step's move) after the tick.
        let rel = cam.view.pos.delta_from(world.position(self.target));
        let rel_after = rel - drift;
        let d = rel.length();
        let progress = (self.elapsed_s / self.total_s()).min(1.0);
        let eta_s = (self.total_s() - self.elapsed_s).max(0.0);

        if self.elapsed_s < self.turn_s {
            let to_target = -rel_after;
            let fwd0 = self.q0 * DVec3::NEG_Z;
            if to_target.length_squared() > 0.0 && to_target.is_finite() {
                let face = DQuat::from_rotation_arc(fwd0, to_target.normalize()) * self.q0;
                let e = quintic_ease(self.elapsed_s / self.turn_s);
                cam.view.orient = self.q0.slerp(face, e).normalize();
            }
            return GotoStep {
                arrived: false,
                speed_mps: 0.0,
                progress,
                to_go_m: (d - self.standoff_m).abs(),
                eta_s,
                velocity_mps: DVec3::ZERO,
            };
        }

        // From the target toward the camera: the line the camera moves along.
        let dir = if d > 0.0 && rel.is_finite() {
            rel / d
        } else {
            -forward(cam)
        };
        let end_gap = ARRIVAL_FRAC * self.standoff_m;
        let travel = *self.travel.get_or_insert_with(|| {
            let gap0 = d - self.standoff_m;
            Travel {
                ln_gap0: gap0.abs().max(end_gap).ln(),
                sign: if gap0 < 0.0 { -1.0 } else { 1.0 },
            }
        });
        let x = ((self.elapsed_s - self.turn_s) / GOTO_DURATION_S).min(1.0);
        let ln_span = end_gap.ln() - travel.ln_gap0;
        let arrived = x >= 1.0 || ln_span >= 0.0;
        let (s, v) = goto_profile(x);
        let gap = if arrived {
            0.0
        } else {
            travel.sign * (travel.ln_gap0 + s * ln_span).exp()
        };
        // Where the camera must be relative to the target after the tick.
        let goal = dir * (self.standoff_m + gap);
        cam.translate(goal - rel_after);
        let fwd = forward(cam);
        cam.view.orient = (DQuat::from_rotation_arc(fwd, -dir) * cam.view.orient).normalize();
        // d|gap|/dt = |gap|·|ln span|·(ds/dx)/T with ds/dx = v / PROFILE_INTEGRAL.
        let speed_mps = if arrived {
            0.0
        } else {
            gap.abs() * ln_span.abs() * v / (PROFILE_INTEGRAL * GOTO_DURATION_S)
        };
        let velocity_mps = if arrived || !(dt > 0.0) {
            DVec3::ZERO
        } else {
            (goal - rel) / dt
        };
        GotoStep {
            arrived,
            speed_mps,
            progress: if arrived { 1.0 } else { progress },
            to_go_m: gap.abs(),
            eta_s: if arrived { 0.0 } else { eta_s },
            velocity_mps,
        }
    }

    /// How far the target moves relative to the camera's anchor during the app's coming tick
    /// of `dt` wall seconds [m, root axes]: zero while the camera rides the target's frame.
    fn tick_drift(&self, cam: &FlyCamera, world: &World, dt: f64) -> DVec3 {
        let time = &world.time;
        let sim_s = if time.is_paused() {
            0.0
        } else {
            time.rate() * dt
        };
        if !(sim_s != 0.0 && sim_s.is_finite()) || self.frame == Some(cam.anchor) {
            return DVec3::ZERO;
        }
        let frames = world.frames();
        let (t0, t1) = (world.epoch(), time.epoch().add_seconds(sim_s));
        let moved = |frame: FrameId| {
            frames
                .origin_in_root(frame, t1)
                .delta_from(frames.origin_in_root(frame, t0))
        };
        let target = match self.frame {
            Some(frame) => moved(frame),
            None => world.velocity(self.target) * sim_s,
        };
        target - moved(cam.anchor)
    }
}

/// The camera's direction of view (−Z), root axes.
fn forward(cam: &FlyCamera) -> DVec3 {
    cam.view.orient * DVec3::NEG_Z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav::testkit;
    use std::f64::consts::PI;
    use uni_core::{AU, C, GM_SUN};

    #[test]
    fn turn_duration_and_quintic_ease() {
        assert!((turn_duration_s(0.0) - 0.40).abs() < 1e-12);
        assert!((turn_duration_s(PI) - 0.95).abs() < 1e-12);
        assert!((turn_duration_s(0.5 * PI) - 0.675).abs() < 1e-12);
        assert_eq!(quintic_ease(0.0), 0.0);
        assert_eq!(quintic_ease(1.0), 1.0);
        assert!((quintic_ease(0.5) - 0.5).abs() < 1e-12);
        // Clamped outside 0 ..= 1, monotonic inside.
        assert_eq!(quintic_ease(-1.0), 0.0);
        assert_eq!(quintic_ease(2.0), 1.0);
        let mut last = 0.0;
        for i in 1..=100 {
            let e = quintic_ease(f64::from(i) / 100.0);
            assert!(e > last, "not increasing at {i}");
            last = e;
        }
    }

    #[test]
    fn goto_profile_accelerates_cruises_decelerates() {
        assert_eq!(goto_profile(0.0), (0.0, 0.0));
        let (s1, v1) = goto_profile(1.0);
        assert!((s1 - 1.0).abs() < 1e-12 && v1.abs() < 1e-12, "{s1} {v1}");
        // Full speed between the acceleration and the deceleration.
        for x in [0.30, 0.45, 0.64] {
            assert!((goto_profile(x).1 - 1.0).abs() < 1e-12, "{x}");
        }
        // The distance share is the integral of the speed share, normalised (trapezoids).
        let n = 20_000;
        let mut integral = 0.0;
        let mut last_s = 0.0;
        for i in 1..=n {
            let (x0, x1) = (f64::from(i - 1) / f64::from(n), f64::from(i) / f64::from(n));
            integral += 0.5 * (goto_profile(x0).1 + goto_profile(x1).1) * (x1 - x0);
            let s = goto_profile(x1).0;
            assert!(s >= last_s, "distance share went back at {x1}");
            last_s = s;
        }
        // ∫v = 1 − a/2 − d/2 = 0.67, so s(x) = (∫₀ˣ v) / 0.67.
        assert!((integral - 0.67).abs() < 1e-6, "{integral}");
        let x = 0.5;
        let mut part = 0.0;
        let m = 10_000;
        for i in 1..=m {
            let (x0, x1) = (x * f64::from(i - 1) / f64::from(m), x * f64::from(i) / f64::from(m));
            part += 0.5 * (goto_profile(x0).1 + goto_profile(x1).1) * (x1 - x0);
        }
        assert!((goto_profile(x).0 - part / 0.67).abs() < 1e-6);
        // Clamped outside 0 ..= 1.
        assert_eq!(goto_profile(-0.5), (0.0, 0.0));
        assert_eq!(goto_profile(1.5).0, 1.0);
    }

    #[test]
    fn standoff_per_kind() {
        let world = testkit::world_with_asteroid();
        let info = |name: &str| world.objects()[testkit::id(&world, name) as usize].clone();
        // Bodies: 4 × the largest radius (Earth's equatorial 6378.137 km), above the atmosphere.
        let earth = info("Earth");
        let s = standoff_distance(&earth, &world);
        assert!((s - 4.0 * earth.radii_m.x).abs() < 1e-6, "{s}");
        assert!(s > 1.5 * earth.atmosphere_top_m.unwrap_or(f64::INFINITY));
        // The Sun: 1 AU × √(L/L☉) = 1 AU (more than 10 R☉).
        assert!((standoff_distance(&info("Sun"), &world) - AU).abs() < 1.0);
        // TestBH-1: 100 GM/c² for 10 M☉ = 1 476.6 km.
        let bh = standoff_distance(&info("TestBH-1"), &world);
        assert!((bh - 100.0 * 10.0 * GM_SUN / (C * C)).abs() < 1e-6, "{bh}");
        assert!((bh - 1.4766e6).abs() < 1e2, "{bh}");
        // Asteroids: 10 × the radius, at least 50 km (the test asteroid is 100 km across).
        let rock = info(testkit::ASTEROID);
        assert!((standoff_distance(&rock, &world) - 500.0e3).abs() < 1e-6);
        let mut pebble = rock.clone();
        pebble.radii_m = glam::DVec3::splat(1.0e3);
        assert!((standoff_distance(&pebble, &world) - 50.0e3).abs() < 1e-6);
        // Comets: 10⁵ km.
        let mut comet = rock.clone();
        comet.kind = uni_sim::registry::ObjectKind::Comet;
        assert!((standoff_distance(&comet, &world) - 1.0e8).abs() < 1e-6);
        // Stars as the Sun's rule from their own L: a star with R = 2 R☉ and T = T☉ has
        // L = 4 L☉, so 2 AU. 10 R★ wins only below T ≈ 0.216 T☉ (1245 K), e.g. a brown dwarf of
        // 0.1 R☉ at 1000 K: √L AU = 4.5e8 m < 10 R★ = 7.0e8 m.
        let (r, t) = (2.0 * uni_core::R_SUN_M, uni_core::TEFF_SUN_K);
        assert!((star_standoff_m(r, t) - 2.0 * AU).abs() < 1e-9 * AU);
        let dwarf = 0.1 * uni_core::R_SUN_M;
        assert!((star_standoff_m(dwarf, 1000.0) - 10.0 * dwarf).abs() < 1e-6);
    }
}
