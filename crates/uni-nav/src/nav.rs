//! The navigator: selection, the aim ring, cruise, the animated goto and the time-warp keys
//! (Task C3; Task C0 wrote this stub, its types, constants and signatures).
//!
//! The owner's point-and-fly navigation (owner feedback 2026-11-22; lead rulings on the C0
//! context, owner points 1–3 and A.5):
//! - **Aim.** The candidate is the destination (`Snapshot::destinations`) nearest the view
//!   centre within [`AIM_CONE_DEG`], kept while within [`AIM_KEEP_DEG`]. While the view is still
//!   and the aim is armed (a drag or an arrow key arms it; arrival, a launch and Esc disarm it),
//!   its ring fills over [`AIM_FILL_S`] and the view eases onto it by 1 − e^(−[`AIM_EASE_PER_S`]·dt)
//!   per frame; off target it drains at [`AIM_DRAIN_PER_S`] per second. A full ring starts the
//!   goto.
//! - **Click.** A click selects (`Navigator::pick`); if the object is a destination the goto
//!   starts at once. F flies to any selection (`Navigator::goto_selected`).
//! - **Cancel.** Esc cancels an aim, else aborts a goto (mode Free), else the app collapses the
//!   card, else deselects. Any movement key or mouse look during a goto aborts it.
//! - **Goto** ([`crate::goto`]): turn first, then accelerate–cruise–decelerate.
//! - **Anchor.** A goto target with a frame is followed by reanchoring to it; a frameless one
//!   (comet, asteroid, star) by `World::position` + `World::velocity` each frame. While
//!   [`FlightMode::Goto`] is active the app skips its dominant-frame reanchor and applies
//!   [`NavStatus::anchor`] instead. [`Navigator::update`] replaces the app's `FlyCamera::update`
//!   call in every mode.

use glam::Vec2;
use uni_core::FrameId;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::Snapshot;
use uni_sim::time::TimeController;
use uni_sim::world::World;

use crate::camera::FlyCamera;
use crate::cruise::CRUISE_LOG10_DEFAULT;
use crate::input::InputState;

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
    pub mode: FlightMode,
    /// The camera's speed this frame [m/s] (the HUD's speed and closing speed).
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
        }
    }

    /// Selects the pick item of `snap` nearest `cursor_internal_px` [internal px] within
    /// `radius_px`, by priority on ties (`crate::pick`); returns it (and keeps it selected), or
    /// `None` (the selection is then unchanged).
    pub fn pick(
        &mut self,
        snap: &Snapshot,
        cursor_internal_px: Vec2,
        radius_px: f32,
    ) -> Option<ObjectId> {
        let _ = (snap, cursor_internal_px, radius_px);
        todo!("C3")
    }

    /// Free ↔ cruise (a goto is aborted into cruise).
    pub fn toggle_cruise(&mut self) {
        todo!("C3")
    }

    /// Starts a goto to the selection; false without one.
    pub fn goto_selected(&mut self) -> bool {
        todo!("C3")
    }

    /// Clears the selection.
    pub fn deselect(&mut self) {
        todo!("C3")
    }

    /// Advances one frame of `dt_real_s` wall seconds: takes the navigation actions from
    /// `input`, runs the aim against `snap` (last frame's snapshot: its destinations), and moves
    /// `cam` by free flight, cruise or the goto (`FlyCamera::translate`). Replaces the app's
    /// `FlyCamera::update` in every mode.
    pub fn update(
        &mut self,
        cam: &mut FlyCamera,
        world: &World,
        snap: &Snapshot,
        input: &mut InputState,
        dt_real_s: f64,
    ) -> NavStatus {
        let _ = (cam, world, snap, input, dt_real_s);
        todo!("C3")
    }
}

/// The time keys (ruling 7): P pauses, `.` and `,` step through `uni_sim::time::WARP_PRESETS`.
/// T (the time-warp panel) is UI state; the app takes it.
pub fn handle_time_keys(input: &mut InputState, time: &mut TimeController) {
    let _ = (input, time);
    todo!("C3")
}
