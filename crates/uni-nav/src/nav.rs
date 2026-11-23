//! The navigator: selection, the aim ring, cruise, the animated goto and the time-warp keys
//! (Task C3; Task C0 wrote this stub, its types, constants and signatures).
//!
//! The owner's point-and-fly navigation (owner feedback 2026-11-22; lead rulings on the C0
//! context, owner points 1–3 and A.5):
//! - **Aim.** The candidate is the destination (`Snapshot::destinations`) nearest the view
//!   centre within [`AIM_CONE_DEG`], kept while it stays the nearest and within
//!   [`AIM_KEEP_DEG`]. A destination the camera is already at (within
//!   1 + [`AIM_HERE_FRAC`] of its stand-off) is never a candidate. While the aim is armed (a
//!   mouse-look drag over [`AIM_ARM_DRAG_PX`] or a movement-key press arms it; arrival, a launch
//!   and Esc disarm it) its ring fills over [`AIM_FILL_S`] (at full rate with the view still,
//!   [`AIM_STEADY_LOOKING`] of it during mouse look, [`AIM_STEADY_MOVING`] while a movement or
//!   roll key is held), and outside mouse look the view eases onto it by
//!   1 − e^(−[`AIM_EASE_PER_S`]·dt) per frame; off target it drains at [`AIM_DRAIN_PER_S`] per
//!   second. A full ring selects the candidate and starts the goto. Only in free flight.
//! - **Click.** A click selects (`Navigator::pick`); if the object is a destination the goto
//!   starts at once. F flies to any selection (`Navigator::goto_selected`).
//! - **Cancel.** Esc cancels an aim, else aborts a goto (mode Free), else the app collapses the
//!   card, else deselects: the navigator takes the Esc press only when it cancelled something.
//!   Any movement-key press or a mouse-look drag over [`AIM_ARM_DRAG_PX`] during a goto aborts
//!   it. Both count from the launch: the input of the launch's own frame (with the F press,
//!   or with the click that `pick` handled before the update) does not abort it.
//! - **Goto** ([`crate::goto`]): turn first, then accelerate–cruise–decelerate. Every drop-out
//!   (cruise's and a goto's arrival) leaves free flight at `DROPOUT_SPEED_MPS`, 1 km/s.
//! - **Anchor.** A goto target with a frame is followed by reanchoring to it; a frameless one
//!   (comet, asteroid, star) by `World::position` + `World::velocity` each frame. While
//!   [`FlightMode::Goto`] is active the app skips its dominant-frame reanchor and applies
//!   [`NavStatus::anchor`] instead (`None`: keep the camera's anchor). [`Navigator::update`]
//!   replaces the app's `FlyCamera::update` in every mode.
//! - **Velocity.** Every frame sets `CameraView::velocity_mps` (root axes; "0 unless a travel
//!   mode sets it", uni-core): the frame's motion ÷ dt relative to the camera's anchor in free
//!   flight and cruise, relative to the target during a goto (zero in its turn and on arrival).
//!   [`NavStatus::speed_mps`] is the speed the HUD shows.
//!
//! **The app's frame** (C6 wires it): on a [`crate::Action::Select`] press, [`Navigator::pick`]
//! at the cursor (`InputState::cursor_px` × internal ÷ output size) with the last snapshot; then
//! [`Navigator::update`] with the last snapshot and the frame's dt; [`handle_time_keys`]; then
//! any Esc press the navigator left (collapse the card, else [`Navigator::deselect`]);
//! `World::set_selected` and `World::set_goto_target` from the status; the world's tick with
//! the same dt (a goto predicts that tick, so a different dt leaves the camera one frame's
//! difference off the profile, corrected on the next frame); `FlyCamera::sync`; the reanchor
//! above; the new snapshot.

use glam::{DQuat, DVec3, Vec2};
use uni_core::{C, FrameId};
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::Snapshot;
use uni_sim::time::TimeController;
use uni_sim::world::World;

use crate::camera::{FlyCamera, ROLL_RATE_RAD_PER_S, SPEED_STEP_LOG10};
use crate::cruise::{self, CRUISE_LOG10_DEFAULT, CRUISE_LOG10_RANGE, DROPOUT_SPEED_MPS};
use crate::goto::{GotoFlight, standoff_distance};
use crate::input::{Action, InputState};
use crate::pick::nearest_pick;

/// A destination within this angle of the view centre [deg] becomes the aim candidate (the
/// owner-approved preview, `74-hud.js`).
pub const AIM_CONE_DEG: f64 = 5.0;
/// The aim candidate is kept while it stays within this angle of the view centre [deg].
pub const AIM_KEEP_DEG: f64 = 6.5;
/// Time [s] for the aim ring to fill with the view held still.
pub const AIM_FILL_S: f64 = 1.0;
/// Rate [1/s] at which the ring drains once the view leaves the candidate.
pub const AIM_DRAIN_PER_S: f64 = 3.0;
/// While the ring fills, the view eases onto the candidate by 1 − e^(−rate·dt) per frame [1/s].
pub const AIM_EASE_PER_S: f64 = 2.2;
/// Share of the fill rate while mouse look is held (the preview's 0.6 "while a pointer is
/// held").
pub const AIM_STEADY_LOOKING: f64 = 0.6;
/// Share of the fill rate while a movement or roll key is held (the preview's 0.3 "while
/// turning").
pub const AIM_STEADY_MOVING: f64 = 0.3;
/// A mouse-look drag longer than this [raw mouse units, about px] arms the aim and aborts a goto
/// (the preview's "dragDist > 4").
pub const AIM_ARM_DRAG_PX: f64 = 4.0;
/// A destination is "here", and not aimed at, while the camera is within (1 + this) × its
/// stand-off distance: a goto there would hardly move.
pub const AIM_HERE_FRAC: f64 = 0.1;

/// The movement keys: a press of one arms the aim and aborts a goto.
const MOVEMENT: [Action; 8] = [
    Action::Forward,
    Action::Back,
    Action::Left,
    Action::Right,
    Action::Up,
    Action::Down,
    Action::RollLeft,
    Action::RollRight,
];

/// How the camera moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlightMode {
    /// Free flight (FlyCamera's keys and mouse look).
    Free,
    /// Cruise: speed proportional to the distance to the nearest surface, along the view.
    Cruise,
    /// The animated goto toward `target`.
    Goto { target: ObjectId },
}

/// Why cruise or goto handed the camera back to free flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropOut {
    /// Crossed the top of an atmosphere (`ObjectInfo::atmosphere_top_m`).
    Atmosphere,
    /// Within 10 photosphere radii of the Sun or a catalogue star.
    NearStar,
    /// Within 100 Schwarzschild radii of a black hole.
    BlackHole,
    /// A goto reached its stand-off distance.
    Arrived,
}

/// The aim ring: its candidate and how full it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aim {
    pub target: ObjectId,
    /// 0 ..= 1; the goto starts at 1.
    pub fill: f32,
}

/// What the navigator did this frame, for the HUD, the overlay and the app.
#[derive(Clone, Debug)]
pub struct NavStatus {
    /// The mode after this frame (Free on the frame of a drop-out).
    pub mode: FlightMode,
    /// The camera's speed this frame [m/s] (the HUD's speed and closing speed): free flight's
    /// set speed, cruise's step ÷ dt, a goto's closing speed toward its stand-off point.
    pub speed_mps: f64,
    /// `speed_mps` exceeds c (non-physical: labelled).
    pub ftl: bool,
    /// Set on the frame cruise or goto dropped out.
    pub dropout: Option<DropOut>,
    /// Time left of a goto [s].
    pub eta_s: Option<f64>,
    pub selected: Option<ObjectId>,
    /// The aim ring, while there is a candidate.
    pub aim: Option<Aim>,
    /// Share of the goto done, 0 ..= 1, while a goto runs.
    pub goto_progress: Option<f32>,
    /// Distance left to the stand-off point [m], while a goto runs.
    pub to_go_m: Option<f64>,
    /// While a goto runs: the frame the app anchors the camera to instead of the dominant frame
    /// (the target's frame, or `None` for a frameless target, followed by position + velocity).
    pub anchor: Option<FrameId>,
}

/// Selection, flight mode and cruise setting.
pub struct Navigator {
    pub selected: Option<ObjectId>,
    pub mode: FlightMode,
    /// log10 of cruise speed ÷ distance to the nearest surface [1/s], within
    /// `crate::CRUISE_LOG10_RANGE`; the wheel moves it while cruising.
    pub cruise_log10: f64,
    /// The aim ring's candidate and fill (0 ..= 1).
    aim: Option<(ObjectId, f64)>,
    /// A drag or a movement key armed the aim (module docs).
    armed: bool,
    /// Raw mouse motion since mouse look began, or since the last launch [raw units].
    drag_px: f64,
    /// Set by every launch, cleared at the end of every [`Navigator::update`] (review r0 I1). A
    /// launch made before an update (a click through `pick`, or `goto_selected` called by the
    /// app) is not aborted by that update's presses and mouse motion, which came with it. A
    /// launch inside an update (F, a full aim ring) comes after that update's abort check, and
    /// the next update's input aborts it as usual.
    launched: bool,
    /// The goto in flight; `None` until the first update after one is asked for.
    flight: Option<GotoFlight>,
}

impl Default for Navigator {
    fn default() -> Self {
        Self::new()
    }
}

impl Navigator {
    /// Nothing selected, free flight, cruise at `cruise::CRUISE_LOG10_DEFAULT`.
    pub fn new() -> Self {
        Navigator {
            selected: None,
            mode: FlightMode::Free,
            cruise_log10: CRUISE_LOG10_DEFAULT,
            aim: None,
            armed: false,
            drag_px: 0.0,
            launched: false,
            flight: None,
        }
    }

    /// The click: selects the pick item of `snap` nearest `cursor_internal_px` [internal px]
    /// within `radius_px`, by priority on ties (`crate::pick`); returns it (and keeps it
    /// selected), or `None` (the selection is then unchanged). If the object is one of
    /// `snap.destinations` (it has a marker), the goto to it starts at once (lead ruling, owner
    /// point 1).
    pub fn pick(
        &mut self,
        snap: &Snapshot,
        cursor_internal_px: Vec2,
        radius_px: f32,
    ) -> Option<ObjectId> {
        let id = nearest_pick(&snap.pick, cursor_internal_px, radius_px)?;
        self.selected = Some(id);
        if snap.destinations.iter().any(|d| d.id == id) {
            self.begin_goto(id);
        }
        Some(id)
    }

    /// Free ↔ cruise (a goto is aborted into cruise). Clears the aim.
    pub fn toggle_cruise(&mut self) {
        self.mode = match self.mode {
            FlightMode::Cruise => FlightMode::Free,
            FlightMode::Free | FlightMode::Goto { .. } => FlightMode::Cruise,
        };
        self.flight = None;
        self.aim = None;
    }

    /// Starts a goto to the selection; false without one. A goto already flying there goes on.
    pub fn goto_selected(&mut self) -> bool {
        match self.selected {
            Some(id) => {
                self.begin_goto(id);
                true
            }
            None => false,
        }
    }

    /// Clears the selection.
    pub fn deselect(&mut self) {
        self.selected = None;
    }

    /// Advances one frame of `dt_real_s` wall seconds: takes the navigation actions from
    /// `input`, runs the aim against `snap` (last frame's snapshot: its destinations), and moves
    /// `cam` by free flight, cruise or the goto (`FlyCamera::translate`). Replaces the app's
    /// `FlyCamera::update` in every mode (module docs: the app's frame).
    ///
    /// Takes the movement presses, F, C and the Esc presses it uses; the mouse delta, the wheel
    /// and the speed presses go to free flight's camera, to cruise (the wheel and speed keys
    /// move `cruise_log10`, mouse look and roll steer) or, during a goto, nowhere.
    pub fn update(
        &mut self,
        cam: &mut FlyCamera,
        world: &World,
        snap: &Snapshot,
        input: &mut InputState,
        dt_real_s: f64,
    ) -> NavStatus {
        // A NaN or negative frame time moves nothing.
        let dt = if dt_real_s.is_finite() {
            dt_real_s.max(0.0)
        } else {
            0.0
        };
        let mut moved = false;
        for a in MOVEMENT {
            while input.take_pressed_once(a) {
                moved = true;
            }
        }
        // Measure the mouse-look drag without taking the motion from the camera.
        let look = input.take_mouse_delta();
        input.add_mouse_delta(look);
        if input.is_looking() {
            self.drag_px += look.length();
        } else {
            self.drag_px = 0.0;
        }
        let dragged = self.drag_px > AIM_ARM_DRAG_PX;

        // The abort and arming check runs before F, C and Esc, so it acts only on a goto that
        // was flying before this frame's presses (review r0 I1). A goto launched since the last
        // update by a click (`pick`, which the app runs before this) came with this frame's
        // input: that input neither aborts it nor re-arms the aim, and its drag counts from
        // the next frame. F below launches after this check, and `begin_goto` resets the drag.
        if self.launched {
            self.drag_px = 0.0;
        } else if moved || dragged {
            if matches!(self.mode, FlightMode::Goto { .. }) {
                self.stop_flight();
            }
            self.armed = true;
        }

        if input.take_pressed_once(Action::Cruise) {
            self.toggle_cruise();
        }
        if input.take_pressed_once(Action::Goto) {
            self.goto_selected();
        }
        if input.has_pending_press(Action::Deselect) {
            if self.aim.is_some() {
                input.take_pressed_once(Action::Deselect);
                self.aim = None;
            } else if matches!(self.mode, FlightMode::Goto { .. }) {
                input.take_pressed_once(Action::Deselect);
                self.stop_flight();
            }
            self.armed = false;
        }

        let mut speed_mps = cam.speed_mps();
        let mut dropout = None;
        let mut flight_step = None;
        let before = cam.view.pos;
        // Free flight and cruise move relative to the camera's anchor, a goto relative to its
        // target (`CameraView::velocity_mps`: set by the travel modes for the HUD and M3).
        let per_s = |d: DVec3| if dt > 0.0 { d / dt } else { DVec3::ZERO };
        match self.mode {
            FlightMode::Free => {
                cam.update(input, dt);
                cam.view.velocity_mps = per_s(cam.view.pos.delta_from(before));
                speed_mps = cam.speed_mps();
                self.run_aim(cam, world, snap, input, dt);
            }
            FlightMode::Cruise => {
                let notches = take_speed_notches(input);
                if notches.is_finite() {
                    self.cruise_log10 += notches * SPEED_STEP_LOG10;
                }
                self.cruise_log10 = self
                    .cruise_log10
                    .clamp(CRUISE_LOG10_RANGE.0, CRUISE_LOG10_RANGE.1);
                steer(cam, input, dt);
                let step = cruise::advance(cam, world, snap, self.cruise_log10, dt);
                cam.view.velocity_mps = per_s(cam.view.pos.delta_from(before));
                speed_mps = step.speed_mps;
                dropout = step.dropout;
            }
            FlightMode::Goto { .. } => {
                // The free-flight inputs would act after the goto: drop them.
                take_speed_notches(input);
                input.take_mouse_delta();
                cam.view.velocity_mps = DVec3::ZERO;
                if self.ensure_flight(cam, world)
                    && let Some(flight) = self.flight.as_mut()
                {
                    let step = flight.step(cam, world, dt);
                    cam.view.velocity_mps = step.velocity_mps;
                    speed_mps = step.speed_mps;
                    flight_step = Some(step);
                    if step.arrived {
                        dropout = Some(DropOut::Arrived);
                    }
                }
            }
        }
        // An aim that completed this frame started a goto: set it up now, so the status
        // carries its anchor.
        if self.flight.is_none() && matches!(self.mode, FlightMode::Goto { .. }) {
            self.ensure_flight(cam, world);
        }
        if dropout.is_some() {
            self.stop_flight();
            self.armed = false;
            cam.speed_log10 = DROPOUT_SPEED_MPS.log10();
            speed_mps = cam.speed_mps();
        }

        // From the next frame on, input comes after any launch made so far.
        self.launched = false;

        let flying = matches!(self.mode, FlightMode::Goto { .. });
        let progress = flight_step.filter(|_| flying);
        NavStatus {
            mode: self.mode,
            speed_mps,
            ftl: speed_mps > C,
            dropout,
            eta_s: progress.map(|s| s.eta_s),
            selected: self.selected,
            aim: self.aim.map(|(target, fill)| Aim {
                target,
                fill: fill as f32,
            }),
            goto_progress: progress.map(|s| s.progress as f32),
            to_go_m: progress.map(|s| s.to_go_m),
            anchor: self.flight.as_ref().and_then(GotoFlight::anchor),
        }
    }

    /// Switches to a goto to `id` (one already flying there goes on); clears and disarms the
    /// aim, and counts the mouse-look drag (and, for a launch before [`Self::update`], the
    /// presses) from here. The flight itself is set up by the next [`Self::update`].
    fn begin_goto(&mut self, id: ObjectId) {
        if self.mode != (FlightMode::Goto { target: id }) {
            self.mode = FlightMode::Goto { target: id };
            self.flight = None;
        }
        self.aim = None;
        self.armed = false;
        self.drag_px = 0.0;
        self.launched = true;
    }

    /// Back to free flight, the goto (if any) dropped.
    fn stop_flight(&mut self) {
        self.mode = FlightMode::Free;
        self.flight = None;
    }

    /// Sets up the flight of the goto mode if it is not yet; false (and free flight) if its
    /// target is not in the registry.
    fn ensure_flight(&mut self, cam: &mut FlyCamera, world: &World) -> bool {
        let FlightMode::Goto { target } = self.mode else {
            return false;
        };
        if self.flight.as_ref().is_some_and(|f| f.target == target) {
            return true;
        }
        self.flight = GotoFlight::start(target, cam, world);
        if self.flight.is_none() {
            self.stop_flight();
        }
        self.flight.is_some()
    }

    /// The aim ring for one free-flight frame (module docs); a full ring starts the goto.
    fn run_aim(
        &mut self,
        cam: &mut FlyCamera,
        world: &World,
        snap: &Snapshot,
        input: &InputState,
        dt: f64,
    ) {
        let fwd = cam.view.orient * DVec3::NEG_Z;
        let mut best: Option<(ObjectId, f64)> = None;
        for d in &snap.destinations {
            let Some(info) = world.objects().get(d.id as usize) else {
                continue;
            };
            let to = world.position(d.id).delta_from(cam.view.pos);
            let dist = to.length();
            if !(dist.is_finite() && dist > (1.0 + AIM_HERE_FRAC) * standoff_distance(info, world))
            {
                continue;
            }
            let angle = fwd.angle_between(to);
            if best.is_none_or(|(_, a)| angle < a) {
                best = Some((d.id, angle));
            }
        }
        let held = self.aim.map(|(id, _)| id);
        let candidate = match best {
            Some((id, a)) if a < AIM_CONE_DEG.to_radians() => Some(id),
            Some((id, a)) if Some(id) == held && a < AIM_KEEP_DEG.to_radians() => Some(id),
            _ => None,
        };
        match candidate {
            Some(id) if self.armed => {
                let fill = match self.aim {
                    Some((held, fill)) if held == id => fill,
                    _ => 0.0,
                };
                let steady = if input.is_looking() {
                    AIM_STEADY_LOOKING
                } else if MOVEMENT.iter().any(|&a| input.is_down(a)) {
                    AIM_STEADY_MOVING
                } else {
                    1.0
                };
                let fill = fill + dt * steady / AIM_FILL_S;
                if !input.is_looking() {
                    let to = world.position(id).delta_from(cam.view.pos);
                    if to.length_squared() > 0.0 && to.is_finite() {
                        let face = DQuat::from_rotation_arc(fwd, to.normalize()) * cam.view.orient;
                        let k = 1.0 - (-AIM_EASE_PER_S * dt).exp();
                        cam.view.orient = cam.view.orient.slerp(face, k).normalize();
                    }
                }
                if fill >= 1.0 {
                    self.selected = Some(id);
                    self.begin_goto(id);
                } else {
                    self.aim = Some((id, fill));
                }
            }
            _ => {
                self.aim = self
                    .aim
                    .map(|(id, fill)| (id, fill - dt * AIM_DRAIN_PER_S))
                    .filter(|&(_, fill)| fill > 0.0);
            }
        }
    }
}

/// Takes the wheel notches and the speed-key presses, as `FlyCamera::update` counts them.
fn take_speed_notches(input: &mut InputState) -> f64 {
    let mut notches = input.take_wheel();
    while input.take_pressed_once(Action::SpeedUp) {
        notches += 1.0;
    }
    while input.take_pressed_once(Action::SpeedDown) {
        notches -= 1.0;
    }
    notches
}

/// Mouse look and roll without moving: cruise's steering. The same turn as `FlyCamera::update`
/// (yaw about +Y, pitch about +X, roll about −Z, in camera axes), which a test checks.
fn steer(cam: &mut FlyCamera, input: &mut InputState, dt: f64) {
    let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
    let look = input.take_mouse_delta() * cam.look_sensitivity;
    let roll_axis = f64::from(
        i8::from(input.is_down(Action::RollRight)) - i8::from(input.is_down(Action::RollLeft)),
    );
    let roll = roll_axis * ROLL_RATE_RAD_PER_S * dt;
    let turn = DQuat::from_rotation_y(-look.x)
        * DQuat::from_rotation_x(-look.y)
        * DQuat::from_axis_angle(DVec3::NEG_Z, roll);
    let orient = (cam.view.orient * turn).normalize();
    cam.view.orient = if orient.is_finite() {
        orient
    } else {
        DQuat::IDENTITY
    };
}

/// The time keys (ruling 7): P pauses, `.` and `,` step through `uni_sim::time::WARP_PRESETS`.
/// T (the time-warp panel) is UI state; the app takes it.
pub fn handle_time_keys(input: &mut InputState, time: &mut TimeController) {
    while input.take_pressed_once(Action::TimePause) {
        time.toggle_pause();
    }
    while input.take_pressed_once(Action::TimeFaster) {
        time.step_up();
    }
    while input.take_pressed_once(Action::TimeSlower) {
        time.step_down();
    }
}

/// Test helpers shared by the uni-nav tests: worlds and the app's frame loop (lead ruling A.5).
#[cfg(test)]
pub(crate) mod testkit {
    use glam::{DQuat, DVec3, UVec2};
    use uni_core::{CameraView, Epoch, UniversePos};
    use uni_data::{AsteroidRecord, Star3};
    use uni_sim::registry::ObjectId;
    use uni_sim::snapshot::Snapshot;
    use uni_sim::world::{World, WorldData};

    use super::{FlightMode, NavStatus, Navigator};
    use crate::camera::FlyCamera;
    use crate::input::InputState;

    /// The name of the test asteroid of [`world_with_asteroid`].
    pub const ASTEROID: &str = "Testroid";
    /// The internal size the tests render at.
    pub const VIEWPORT: UVec2 = UVec2::new(1920, 1080);
    /// One frame at 60 fps [s].
    pub const DT: f64 = 1.0 / 60.0;

    /// The M1 start date.
    pub fn date() -> Epoch {
        Epoch::parse_iso_utc("2026-11-20T00:00:00Z").expect("ISO date")
    }

    fn data(asteroids: Vec<AsteroidRecord>, names: Vec<String>, stars: Vec<Star3>) -> WorldData {
        WorldData {
            stars,
            star_names: None,
            sats: Vec::new(),
            comets: Vec::new(),
            asteroids,
            asteroid_names: names,
            exo: Vec::new(),
            facts: Vec::new(),
            rings: None,
            warnings: Vec::new(),
        }
    }

    /// The Solar System from the ephemeris and TestBH-1, no catalogue data.
    pub fn world() -> World {
        World::new(data(Vec::new(), Vec::new(), Vec::new()), date())
    }

    /// [`world`] plus a 100 km asteroid (H 5, so a registry object without a frame) on a 2 AU
    /// orbit, moving at about 21 km/s.
    pub fn world_with_asteroid() -> World {
        let rock = AsteroidRecord {
            a_au: 2.0,
            e: 0.1,
            i_deg: 5.0,
            node_deg: 80.0,
            argp_deg: 70.0,
            m0_deg: 10.0,
            epoch_jd_tdb: 2_451_545.0,
            h: 5.0,
            g: 0.15,
            diameter_km: 100.0,
            albedo: 0.1,
            spkid: 29_999_999,
            name_index: 0,
        };
        World::new(data(vec![rock], vec![ASTEROID.to_owned()], Vec::new()), date())
    }

    /// [`world`] plus one unnamed Sun-like catalogue star at 1 pc on +z (ICRS), at rest.
    pub fn world_with_star() -> World {
        let star = Star3::new([0.0, 0.0, 1.0], 4.83, [0.0; 3], 0.65, 1, 0);
        World::new(data(Vec::new(), Vec::new(), vec![star]), date())
    }

    pub fn id(world: &World, name: &str) -> ObjectId {
        world
            .find(name)
            .unwrap_or_else(|| panic!("{name} not in the registry"))
    }

    /// A camera at `pos` whose −Z axis points along `dir`, joined to the dominant frame.
    pub fn camera(world: &World, pos: UniversePos, dir: DVec3) -> FlyCamera {
        let mut cam = FlyCamera::new(CameraView {
            pos,
            orient: DQuat::from_rotation_arc(DVec3::NEG_Z, dir.normalize()),
            fov_y_rad: 60_f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        });
        let dominant = world.dominant_frame(&pos);
        cam.reanchor(world.frames(), world.epoch(), dominant);
        cam
    }

    /// The camera's direction of view (−Z), root axes.
    pub fn forward(cam: &FlyCamera) -> DVec3 {
        cam.view.orient * DVec3::NEG_Z
    }

    /// Distance [m] from the camera to object `id`'s centre.
    pub fn dist(world: &World, cam: &FlyCamera, id: ObjectId) -> f64 {
        cam.view.pos.delta_from(world.position(id)).length()
    }

    /// One frame of the app's loop (lead ruling A.5, wired by C6): the navigator moves the
    /// camera, the world ticks, the camera follows its anchor, then joins the dominant frame,
    /// or during a goto the navigator's anchor (none for a frameless target: it keeps its
    /// own). With `snapshots`, `snap` becomes this frame's snapshot for the next frame.
    pub fn frame(
        nav: &mut Navigator,
        cam: &mut FlyCamera,
        world: &mut World,
        input: &mut InputState,
        snap: &mut Snapshot,
        snapshots: bool,
        dt: f64,
    ) -> NavStatus {
        let status = nav.update(cam, world, snap, input, dt);
        let target = match status.mode {
            FlightMode::Goto { target } => Some(target),
            _ => None,
        };
        world.set_goto_target(target);
        world.set_selected(status.selected);
        world.tick(dt);
        cam.sync(world.frames(), world.epoch());
        let anchor = match status.mode {
            FlightMode::Goto { .. } => status.anchor,
            _ => Some(world.dominant_frame(&cam.view.pos)),
        };
        if let Some(anchor) = anchor {
            cam.reanchor(world.frames(), world.epoch(), anchor);
        }
        if snapshots {
            *snap = world.snapshot(&cam.view, VIEWPORT);
        }
        status
    }

    /// Runs frames of `dt` until `done` holds for a frame's status (that frame included) or
    /// `max_frames` pass; returns every status.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        nav: &mut Navigator,
        cam: &mut FlyCamera,
        world: &mut World,
        input: &mut InputState,
        snapshots: bool,
        dt: f64,
        max_frames: usize,
        done: impl Fn(&NavStatus) -> bool,
    ) -> Vec<NavStatus> {
        let mut snap = if snapshots {
            world.snapshot(&cam.view, VIEWPORT)
        } else {
            Snapshot::default()
        };
        let mut out = Vec::new();
        for _ in 0..max_frames {
            let status = frame(nav, cam, world, input, &mut snap, snapshots, dt);
            let stop = done(&status);
            out.push(status);
            if stop {
                break;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use glam::{DQuat, DVec2, DVec3};
    use uni_core::{AU, C, PC, R_SUN_M};
    use uni_sim::registry::ObjectKind;
    use uni_sim::snapshot::{Destination, PickItem};
    use winit::keyboard::KeyCode;

    use super::testkit::{self, DT, camera, dist, forward, id, run};
    use super::*;
    use crate::cruise::{CRUISE_LOG10_RANGE, DROPOUT_SPEED_MPS};
    use crate::goto::{GOTO_DURATION_S, standoff_distance, turn_duration_s};
    use crate::input::Action;

    fn cruising() -> Navigator {
        let mut nav = Navigator::new();
        nav.toggle_cruise();
        assert_eq!(nav.mode, FlightMode::Cruise);
        nav
    }

    fn not_cruising(s: &NavStatus) -> bool {
        s.mode != FlightMode::Cruise
    }

    fn not_flying(s: &NavStatus) -> bool {
        matches!(s.mode, FlightMode::Free)
    }

    /// Unit vector toward the north ecliptic pole in ICRS (obliquity 23.439°, IAU 2006): off
    /// the planets' paths.
    fn ecliptic_north() -> DVec3 {
        let eps = 23.439_f64.to_radians();
        DVec3::new(0.0, -eps.sin(), eps.cos())
    }

    #[test]
    fn cruise_drops_out_at_atmosphere_top() {
        // Review Focus 2. From 10⁵ km above Earth's pole, looking down, at the default cruise
        // (a tenth of the distance per second, 60 fps) and at the fastest (10× per second, 10 fps).
        for (log10, dt) in [(CRUISE_LOG10_DEFAULT, DT), (1.0, 0.1)] {
            let mut world = testkit::world();
            let earth = id(&world, "Earth");
            let top = world.objects()[earth as usize].atmosphere_top_m.expect("Earth's top");
            let start = world.position(earth).offset(DVec3::Z * 1.0e8);
            let mut cam = camera(&world, start, DVec3::NEG_Z);
            let mut nav = cruising();
            nav.cruise_log10 = log10;
            let mut input = InputState::new();
            let out = run(&mut nav, &mut cam, &mut world, &mut input, false, dt, 20_000, not_cruising);
            let last = out.last().expect("frames");
            assert_eq!(last.dropout, Some(DropOut::Atmosphere), "cruise {log10}");
            assert_eq!(last.mode, FlightMode::Free);
            let d = dist(&world, &cam, earth);
            assert!(d >= top - 1.0, "cruise {log10}: {d} m from the centre, top {top}");
            assert!(d <= top + 1.0, "cruise {log10}: stopped early, {d} m vs top {top}");
            // Free flight at 1 km/s afterwards.
            assert!((cam.speed_mps() - DROPOUT_SPEED_MPS).abs() < 1e-6);
            assert!((last.speed_mps - DROPOUT_SPEED_MPS).abs() < 1e-6);
            if log10 == 1.0 {
                // Each step covers half the altitude (SURFACE_STEP_FRAC), so the last one, from
                // below 200 km, would cross the 100 km top: it ends on it instead (above).
                assert!(out.len() < 30, "{} frames", out.len());
            }
        }
    }

    #[test]
    fn cruise_drops_out_near_sun_and_black_hole() {
        let mut world = testkit::world();
        // Toward the Sun from 0.3 AU above the ecliptic, at the fastest cruise.
        let sun = id(&world, "Sun");
        let r_sun = world.objects()[sun as usize].radii_m.x;
        assert!((r_sun - R_SUN_M).abs() < 1e6, "{r_sun}");
        let start = world.position(sun).offset(ecliptic_north() * 0.3 * AU);
        let mut cam = camera(&world, start, -ecliptic_north());
        let mut nav = cruising();
        nav.cruise_log10 = 1.0;
        let mut input = InputState::new();
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, 0.1, 1000, not_cruising);
        assert_eq!(out.last().and_then(|s| s.dropout), Some(DropOut::NearStar));
        let d = dist(&world, &cam, sun);
        assert!(d >= 10.0 * r_sun - 1.0 && d < 10.0 * r_sun + 1.0, "{} R☉", d / r_sun);

        // Toward TestBH-1 from 1 AU.
        let bh = id(&world, "TestBH-1");
        let r_s = 2.0 * world.objects()[bh as usize].gm_m3s2 / (C * C);
        let start = world.position(bh).offset(DVec3::Z * AU);
        let mut cam = camera(&world, start, DVec3::NEG_Z);
        let mut nav = cruising();
        nav.cruise_log10 = 1.0;
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, 0.1, 1000, not_cruising);
        assert_eq!(out.last().and_then(|s| s.dropout), Some(DropOut::BlackHole));
        let d = dist(&world, &cam, bh);
        assert!(d >= 100.0 * r_s - 1.0 && d < 100.0 * r_s + 1.0, "{} r_s", d / r_s);
    }

    #[test]
    fn cruise_drops_out_near_a_catalogue_star() {
        // An unnamed star 0.05 pc ahead: the snapshot finds it (as a point without an id), the
        // next tick registers it, and cruise stops at 10 of its radii.
        let mut world = testkit::world_with_star();
        let star_pos = uni_core::UniversePos::from_meters(DVec3::Z * PC);
        let start = star_pos.offset(DVec3::NEG_Z * 0.05 * PC);
        let mut cam = camera(&world, start, DVec3::Z);
        let mut nav = cruising();
        nav.cruise_log10 = 1.0;
        let mut input = InputState::new();
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, 0.1, 1000, not_cruising);
        assert_eq!(out.last().and_then(|s| s.dropout), Some(DropOut::NearStar));
        let star = world
            .objects()
            .iter()
            .find(|o| o.kind == ObjectKind::Star)
            .expect("the star was registered");
        let r = star.radii_m.x;
        let d = cam.view.pos.delta_from(star_pos).length();
        assert!(d >= 10.0 * r - 1.0 && d < 10.0 * r + 1.0, "{} R★", d / r);
    }

    #[test]
    fn cruise_inside_a_boundary_only_leaves() {
        // 50 km up, inside Earth's atmosphere top (100 km): looking down drops out at once
        // without moving; looking up cruises out.
        let mut world = testkit::world();
        let earth = id(&world, "Earth");
        let pos = world.position(earth).offset(DVec3::X * 6.43e6);
        let mut input = InputState::new();
        let mut cam = camera(&world, pos, DVec3::NEG_X);
        let mut nav = cruising();
        let out = run(&mut nav, &mut cam, &mut world, &mut input, false, DT, 1, |_| true);
        assert_eq!(out[0].dropout, Some(DropOut::Atmosphere));
        assert!((dist(&world, &cam, earth) - 6.43e6).abs() < 1e-3);

        let mut cam = camera(&world, world.position(earth).offset(DVec3::X * 6.43e6), DVec3::X);
        let mut nav = cruising();
        let out = run(&mut nav, &mut cam, &mut world, &mut input, false, DT, 30, |_| false);
        assert!(out.iter().all(|s| s.mode == FlightMode::Cruise && s.dropout.is_none()));
        assert!(dist(&world, &cam, earth) > 6.43e6 + 1.0e3);
    }

    #[test]
    fn cruise_wheel_moves_cruise_log10_within_range() {
        let mut world = testkit::world();
        let earth = id(&world, "Earth");
        let mut cam = camera(&world, world.position(earth).offset(DVec3::Z * 1e9), DVec3::Z);
        let speed = cam.speed_log10;
        let mut nav = cruising();
        let mut input = InputState::new();
        let mut step = |input: &mut InputState, nav: &mut Navigator| {
            run(nav, &mut cam, &mut world, input, false, DT, 1, |_| true);
            nav.cruise_log10
        };
        input.add_wheel(5.0);
        assert!((step(&mut input, &mut nav) + 0.5).abs() < 1e-12);
        input.add_wheel(100.0);
        assert_eq!(step(&mut input, &mut nav), CRUISE_LOG10_RANGE.1);
        input.add_wheel(-100.0);
        assert_eq!(step(&mut input, &mut nav), CRUISE_LOG10_RANGE.0);
        drop(step);
        assert_eq!(cam.speed_log10, speed, "the wheel moved the free-flight speed");
    }

    #[test]
    fn steering_turns_as_free_flight_does() {
        let world = testkit::world();
        let pos = world.position(id(&world, "Earth")).offset(DVec3::Z * 1e9);
        let mut a = camera(&world, pos, DVec3::new(1.0, 2.0, 3.0));
        let mut b = a.clone();
        let mut input_a = InputState::new();
        let mut input_b = InputState::new();
        for input in [&mut input_a, &mut input_b] {
            input.add_mouse_delta(DVec2::new(30.0, -12.0));
            input.set_down(Action::RollRight, true);
        }
        a.update(&mut input_a, 0.2);
        steer(&mut b, &mut input_b, 0.2);
        assert!(a.view.orient.abs_diff_eq(b.view.orient, 1e-12), "{} vs {}", a.view.orient, b.view.orient);
        assert_eq!(b.view.pos, pos, "steering moved the camera");
    }

    /// Runs a goto to `target` from `start` (looking along `look`) at `dt`; returns the
    /// statuses and the camera.
    fn fly_to(
        world: &mut uni_sim::world::World,
        target: ObjectId,
        start: uni_core::UniversePos,
        look: DVec3,
    ) -> (Vec<NavStatus>, crate::camera::FlyCamera) {
        let mut cam = camera(world, start, look);
        let mut nav = Navigator::new();
        nav.selected = Some(target);
        let mut input = InputState::new();
        input.on_key(KeyCode::KeyF, true, false);
        input.on_key(KeyCode::KeyF, false, false);
        let out = run(&mut nav, &mut cam, world, &mut input, false, DT, 2000, not_flying);
        (out, cam)
    }

    #[test]
    fn goto_stops_outside_atmosphere() {
        // Review Focus 2.
        let mut world = testkit::world();
        let earth = id(&world, "Earth");
        let info = world.objects()[earth as usize].clone();
        let standoff = standoff_distance(&info, &world);
        let top = info.atmosphere_top_m.expect("Earth's top");
        assert!(standoff > top);
        let start = world.position(earth).offset(DVec3::new(1.0, -2.0, 0.5).normalize() * 1.0e9);
        let mut cam = camera(&world, start, DVec3::Z);
        let mut nav = Navigator::new();
        nav.selected = Some(earth);
        assert!(nav.goto_selected());
        let mut input = InputState::new();
        // Mid-travel the camera's velocity is its closing velocity: toward Earth, at the
        // status's speed (a frame's average against the profile's instantaneous speed).
        let out = run(&mut nav, &mut cam, &mut world, &mut input, false, DT, 210, |_| false);
        let mid = out.last().expect("frames");
        assert!(matches!(mid.mode, FlightMode::Goto { .. }) && mid.speed_mps > 0.0);
        let v = cam.view.velocity_mps;
        let to_earth = world.position(earth).delta_from(cam.view.pos).normalize();
        assert!(v.normalize().dot(to_earth) > 0.999_999, "{v}");
        assert!((v.length() / mid.speed_mps - 1.0).abs() < 0.05, "{} vs {}", v.length(), mid.speed_mps);
        let out = run(&mut nav, &mut cam, &mut world, &mut input, false, DT, 2000, not_flying);
        let last = out.last().expect("frames");
        assert_eq!(last.dropout, Some(DropOut::Arrived));
        assert_eq!(cam.view.velocity_mps, DVec3::ZERO, "at rest on arrival");
        let d = dist(&world, &cam, earth);
        assert!((d / standoff - 1.0).abs() < 1e-3, "{d} vs {standoff}");
        assert!(d > top);
        // It ends looking at the target.
        let to_earth = world.position(earth).delta_from(cam.view.pos).normalize();
        assert!(forward(&cam).dot(to_earth) > (0.01_f64).to_radians().cos());
    }

    #[test]
    fn goto_duration_independent_of_distance() {
        // Review m20, as the lead ruled: the turn (0.40 + 0.55·turn/π s) plus 4.5 s of travel,
        // from 1 AU and from 1 pc, facing the target (no turn) or facing away (180°).
        let dir = DVec3::new(1.0, 2.0, 3.0).normalize();
        for distance in [AU, PC] {
            for facing in [true, false] {
                let mut world = testkit::world();
                let earth = id(&world, "Earth");
                let standoff = standoff_distance(&world.objects()[earth as usize], &world);
                let start = world.position(earth).offset(dir * distance);
                let look = if facing { -dir } else { dir };
                let (out, cam) = fly_to(&mut world, earth, start, look);
                let turn = if facing { 0.0 } else { PI };
                let expected = turn_duration_s(turn) + GOTO_DURATION_S;
                let took = out.len() as f64 * DT;
                let case = format!("{} m, facing {facing}", distance);
                assert_eq!(out.last().and_then(|s| s.dropout), Some(DropOut::Arrived), "{case}");
                assert!((took - expected).abs() <= 0.25, "{case}: {took} s vs {expected} s");
                let d = dist(&world, &cam, earth);
                assert!((d / standoff - 1.0).abs() < 1e-3, "{case}: {d}");
                // The camera really moves (parallax): the distance left only shrinks (up to the
                // rounding of a distance this large while the camera holds still in the turn),
                // and the progress only grows.
                let to_go: Vec<f64> = out.iter().filter_map(|s| s.to_go_m).collect();
                assert!(to_go.windows(2).all(|w| w[1] <= w[0] * (1.0 + 1e-14)), "{case}");
                assert!(to_go.last().is_some_and(|&g| g < 1e-3 * standoff), "{case}");
                let progress: Vec<f32> = out.iter().filter_map(|s| s.goto_progress).collect();
                assert!(progress.windows(2).all(|w| w[1] >= w[0]), "{case}");
                if distance == PC {
                    assert!(out.iter().any(|s| s.ftl), "{case}: 1 pc in 4.5 s is above c");
                }
            }
        }
    }

    #[test]
    fn goto_follows_moving_target() {
        // Earth (a frame, 30 km/s) and a frameless asteroid (21 km/s), both at time warp ×10³,
        // so each moves 10⁵ km per second of the flight.
        for name in ["Earth", testkit::ASTEROID] {
            let mut world = testkit::world_with_asteroid();
            world.time.set_rate(1.0e3);
            let target = id(&world, name);
            let info = world.objects()[target as usize].clone();
            assert_eq!(info.frame.is_some(), name == "Earth");
            let standoff = standoff_distance(&info, &world);
            let start = world.position(target).offset(DVec3::new(-1.0, 0.5, 0.2).normalize() * 1e9);
            let (out, cam) = fly_to(&mut world, target, start, DVec3::X);
            assert_eq!(out.last().and_then(|s| s.dropout), Some(DropOut::Arrived), "{name}");
            let d = dist(&world, &cam, target);
            assert!((d / standoff - 1.0).abs() < 0.01, "{name}: {d} vs {standoff}");
            let anchors: Vec<_> = out.iter().filter(|s| matches!(s.mode, FlightMode::Goto { .. })).map(|s| s.anchor).collect();
            assert!(anchors.iter().all(|&a| a == info.frame), "{name}");
        }
    }

    #[test]
    fn ftl_flag_above_c() {
        // Cruise at 10⁰ × the distance to TestBH-1's horizon, looking away from it.
        for (d_surface, ftl) in [(2.9e8, false), (3.1e8, true)] {
            let mut world = testkit::world();
            let bh = id(&world, "TestBH-1");
            let horizon = world.objects()[bh as usize].radii_m.x;
            let mut cam = camera(&world, world.position(bh).offset(DVec3::Z * (d_surface + horizon)), DVec3::Z);
            let mut nav = cruising();
            nav.cruise_log10 = 0.0;
            let mut input = InputState::new();
            let out = run(&mut nav, &mut cam, &mut world, &mut input, false, DT, 1, |_| true);
            assert!((out[0].speed_mps / d_surface - 1.0).abs() < 1e-9, "{}", out[0].speed_mps);
            assert_eq!(out[0].ftl, ftl, "{d_surface}");
            // The camera's velocity (for the HUD, and M3's aberration) is the cruise along −Z.
            let v = cam.view.velocity_mps;
            assert!((v - forward(&cam) * out[0].speed_mps).length() < 1e-6 * d_surface, "{v}");
        }
    }

    #[test]
    fn time_keys() {
        let mut time = uni_sim::time::TimeController::new(testkit::date());
        let mut input = InputState::new();
        let tap = |input: &mut InputState, code: KeyCode| {
            input.on_key(code, true, false);
            input.on_key(code, false, false);
        };
        tap(&mut input, KeyCode::KeyP);
        handle_time_keys(&mut input, &mut time);
        assert!(time.is_paused());
        tap(&mut input, KeyCode::KeyP);
        handle_time_keys(&mut input, &mut time);
        assert!(!time.is_paused());
        tap(&mut input, KeyCode::Period);
        handle_time_keys(&mut input, &mut time);
        assert_eq!(time.rate(), 1.0e3);
        tap(&mut input, KeyCode::Comma);
        handle_time_keys(&mut input, &mut time);
        assert_eq!(time.rate(), 1.0);
        // T is the app's (the panel), and stays for it.
        tap(&mut input, KeyCode::KeyT);
        handle_time_keys(&mut input, &mut time);
        assert!(input.take_pressed_once(Action::TimePanel));
    }

    /// A camera 6 × 10⁷ m from Earth looking 3° off its centre, its snapshot, and a navigator.
    fn aiming_at_earth() -> (uni_sim::world::World, crate::camera::FlyCamera, ObjectId) {
        let world = testkit::world();
        let earth = id(&world, "Earth");
        let out = DVec3::new(0.3, -1.0, 0.4).normalize();
        let pos = world.position(earth).offset(out * 6.0e7);
        let off_axis = DQuat::from_axis_angle(out.any_orthonormal_vector(), 3_f64.to_radians());
        let cam = camera(&world, pos, off_axis * -out);
        (world, cam, earth)
    }

    fn angle_to(world: &uni_sim::world::World, cam: &crate::camera::FlyCamera, id: ObjectId) -> f64 {
        let to = world.position(id).delta_from(cam.view.pos);
        forward(cam).angle_between(to)
    }

    #[test]
    fn aim_fills_when_armed_and_launches_goto() {
        let (mut world, mut cam, earth) = aiming_at_earth();
        let mut nav = Navigator::new();
        let mut input = InputState::new();
        // Not armed: the start view does not fly off by itself.
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, DT, 90, |_| false);
        assert!(out.iter().all(|s| s.aim.is_none() && s.mode == FlightMode::Free));
        // A tap of a movement key arms it; the ring fills over 1 s and the view eases on.
        input.on_key(KeyCode::KeyD, true, false);
        input.on_key(KeyCode::KeyD, false, false);
        let before = angle_to(&world, &cam, earth);
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, DT, 120, |s| {
            matches!(s.mode, FlightMode::Goto { .. })
        });
        let launch = out.len();
        assert!((59..=62).contains(&launch), "launched after {launch} frames");
        let fills: Vec<f32> = out.iter().filter_map(|s| s.aim.map(|a| a.fill)).collect();
        assert!(fills.windows(2).all(|w| w[1] > w[0]), "{fills:?}");
        assert!(out[..launch - 1].iter().all(|s| s.aim.map(|a| a.target) == Some(earth)));
        assert!(angle_to(&world, &cam, earth) < 0.5 * before, "the view did not ease on");
        let last = out.last().expect("frames");
        assert_eq!(last.mode, FlightMode::Goto { target: earth });
        assert_eq!(last.selected, Some(earth));
        assert!(last.aim.is_none());
        // The launch came after its frame's input, so a press in the next frame aborts it.
        input.on_key(KeyCode::KeyD, true, false);
        input.on_key(KeyCode::KeyD, false, false);
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, DT, 1, |_| true);
        assert_eq!((out[0].mode, out[0].dropout), (FlightMode::Free, None));
    }

    #[test]
    fn esc_cancels_an_aim_then_a_goto_then_leaves_it_for_the_app() {
        let (mut world, mut cam, earth) = aiming_at_earth();
        let mut nav = Navigator::new();
        let mut input = InputState::new();
        input.on_key(KeyCode::KeyD, true, false);
        input.on_key(KeyCode::KeyD, false, false);
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, DT, 10, |_| false);
        assert!(out.last().and_then(|s| s.aim).is_some());
        // Esc cancels the aim (and disarms: the ring does not come back).
        input.on_key(KeyCode::Escape, true, false);
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, DT, 90, |_| false);
        assert!(out.iter().all(|s| s.aim.is_none() && s.mode == FlightMode::Free));
        assert!(!input.has_pending_press(Action::Deselect), "the navigator took it");
        // Esc aborts a goto.
        nav.selected = Some(earth);
        assert!(nav.goto_selected());
        run(&mut nav, &mut cam, &mut world, &mut input, true, DT, 5, |_| false);
        input.on_key(KeyCode::Escape, true, false);
        let out = run(&mut nav, &mut cam, &mut world, &mut input, true, DT, 1, |_| true);
        assert_eq!(out[0].mode, FlightMode::Free);
        assert_eq!(out[0].dropout, None);
        assert!(!input.has_pending_press(Action::Deselect));
        // Nothing left to cancel: the press stays for the app (card, then deselect).
        input.on_key(KeyCode::Escape, true, false);
        run(&mut nav, &mut cam, &mut world, &mut input, true, DT, 1, |_| true);
        assert!(input.take_pressed_once(Action::Deselect));
        nav.deselect();
        assert_eq!(nav.selected, None);
    }

    #[test]
    fn movement_or_mouse_look_aborts_goto() {
        let mut world = testkit::world();
        let earth = id(&world, "Earth");
        let start = world.position(earth).offset(DVec3::Z * 1e9);
        for abort in ["key", "look"] {
            let mut cam = camera(&world, start, DVec3::X);
            let mut nav = Navigator::new();
            nav.selected = Some(earth);
            assert!(nav.goto_selected());
            let mut input = InputState::new();
            run(&mut nav, &mut cam, &mut world, &mut input, false, DT, 30, |_| false);
            assert_eq!(nav.mode, FlightMode::Goto { target: earth });
            if abort == "key" {
                input.on_key(KeyCode::KeyW, true, false);
            } else {
                input.set_looking(true);
                input.add_mouse_delta(DVec2::new(10.0, 0.0));
            }
            let out = run(&mut nav, &mut cam, &mut world, &mut input, false, DT, 1, |_| true);
            assert_eq!(out[0].mode, FlightMode::Free, "{abort}");
            assert_eq!(out[0].dropout, None, "{abort}");
        }
    }

    #[test]
    fn goto_launch_survives_the_input_of_its_own_frame() {
        // Review r0 I1. A goto launched by F, or by a click on a destination (`pick`, which the
        // app runs before `update`), is not aborted by the input that came in the same frame:
        // mouse look held with more than 4 units of motion (the drag of the whole hold, and this
        // frame's), or a movement-key tap. A drag counts from the launch, and a press after the
        // launch frame still aborts.
        let mut world = testkit::world();
        let earth = id(&world, "Earth");
        let start = world.position(earth).offset(DVec3::Z * 1e9);
        let at = glam::Vec2::new(100.0, 500.0);
        let snap = uni_sim::snapshot::Snapshot {
            pick: vec![PickItem {
                id: earth,
                screen_px: at,
                dist_m: 1e9,
                priority: 3,
            }],
            destinations: vec![Destination {
                id: earth,
                kind: ObjectKind::Planet,
                rel_pos_m: DVec3::NEG_Z * 1e9,
                dist_m: 1e9,
                screen_px: Some(at),
                proj_radius_px: 2.0,
            }],
            ..Default::default()
        };
        let tap = |input: &mut InputState, code: KeyCode| {
            input.on_key(code, true, false);
            input.on_key(code, false, false);
        };
        let one = |nav: &mut Navigator,
                   cam: &mut crate::camera::FlyCamera,
                   world: &mut uni_sim::world::World,
                   input: &mut InputState| {
            run(nav, cam, world, input, false, DT, 1, |_| true).remove(0)
        };
        let flying = FlightMode::Goto { target: earth };
        for launch in ["F", "click"] {
            for extra in ["look", "key"] {
                let case = format!("{launch} + {extra}");
                let mut cam = camera(&world, start, DVec3::X);
                let mut nav = Navigator::new();
                let mut input = InputState::new();
                if extra == "look" {
                    // Steering with the right button held: the drag of the hold is over 4 units
                    // before the launch.
                    input.set_looking(true);
                    input.add_mouse_delta(DVec2::new(10.0, 0.0));
                    let s = one(&mut nav, &mut cam, &mut world, &mut input);
                    assert_eq!(s.mode, FlightMode::Free, "{case}");
                    input.add_mouse_delta(DVec2::new(10.0, 0.0));
                } else {
                    tap(&mut input, KeyCode::KeyW);
                }
                if launch == "F" {
                    nav.selected = Some(earth);
                    tap(&mut input, KeyCode::KeyF);
                } else {
                    assert_eq!(nav.pick(&snap, at, 12.0), Some(earth), "{case}");
                }
                let s = one(&mut nav, &mut cam, &mut world, &mut input);
                assert_eq!((s.mode, s.dropout), (flying, None), "{case}: the launch frame");
                assert!(s.to_go_m.is_some(), "{case}: the flight runs");
                let s = one(&mut nav, &mut cam, &mut world, &mut input);
                assert_eq!(s.mode, flying, "{case}: the frame after the launch");
                if extra == "look" {
                    // The drag counts from the launch: 3 units keep the goto, 2 more abort it.
                    input.add_mouse_delta(DVec2::new(3.0, 0.0));
                    let s = one(&mut nav, &mut cam, &mut world, &mut input);
                    assert_eq!(s.mode, flying, "{case}: 3 units since the launch");
                    input.add_mouse_delta(DVec2::new(0.0, 2.0));
                } else {
                    tap(&mut input, KeyCode::KeyW);
                }
                let s = one(&mut nav, &mut cam, &mut world, &mut input);
                assert_eq!((s.mode, s.dropout), (FlightMode::Free, None), "{case}: aborted after");
            }
        }
    }

    #[test]
    fn click_selects_and_flies_to_a_destination() {
        let item = |id, x| PickItem {
            id,
            screen_px: glam::Vec2::new(x, 500.0),
            dist_m: 1e11,
            priority: 3,
        };
        let dest = Destination {
            id: 7,
            kind: ObjectKind::Planet,
            rel_pos_m: DVec3::X * 1e11,
            dist_m: 1e11,
            screen_px: Some(glam::Vec2::new(100.0, 500.0)),
            proj_radius_px: 2.0,
        };
        let snap = uni_sim::snapshot::Snapshot {
            pick: vec![item(7, 100.0), item(9, 300.0)],
            destinations: vec![dest],
            ..Default::default()
        };
        let mut nav = Navigator::new();
        let at = |x: f32| glam::Vec2::new(x, 505.0);
        // Not a destination: selected only.
        assert_eq!(nav.pick(&snap, at(300.0), 12.0), Some(9));
        assert_eq!((nav.selected, nav.mode), (Some(9), FlightMode::Free));
        // Nothing under the cursor: the selection stays.
        assert_eq!(nav.pick(&snap, at(600.0), 12.0), None);
        assert_eq!(nav.selected, Some(9));
        // A destination: selected, and the goto starts at once.
        assert_eq!(nav.pick(&snap, at(100.0), 12.0), Some(7));
        assert_eq!((nav.selected, nav.mode), (Some(7), FlightMode::Goto { target: 7 }));
        // C toggles: a goto is aborted into cruise; then back to free flight.
        nav.toggle_cruise();
        assert_eq!(nav.mode, FlightMode::Cruise);
        nav.toggle_cruise();
        assert_eq!(nav.mode, FlightMode::Free);
        // F without a selection does nothing.
        nav.deselect();
        assert!(!nav.goto_selected());
        assert_eq!(nav.mode, FlightMode::Free);
    }
}
