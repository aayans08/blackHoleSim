//! Quaternion 6-DoF free camera with logarithmic speed control (M0 Task 6), parented to a
//! frame (M1 Task A6).
//!
//! Free flight ignores gravity and reaches 10^26 m/s, far above c. Speeds above c are
//! non-physical, and [`FlyCamera::is_ftl`] tells the HUD to label them so.
//!
//! The camera rides with its anchor frame (design §5.2: it is parented to the nearest dominant
//! frame). Each frame the app calls [`FlyCamera::update`] (input moves `rel_m` and `view.pos`
//! together), advances time, calls [`FlyCamera::sync`] (`view.pos` = the anchor's origin now +
//! `rel_m`), then [`FlyCamera::reanchor`] to the dominant frame. Time warp therefore carries
//! the camera with its body.

use std::f64::consts::FRAC_PI_2;

use glam::{DQuat, DVec3, I64Vec3};
use uni_core::{C, CameraView, Epoch, FrameId, FrameTree, ROOT_FRAME, UniversePos};

use crate::anchor::{add_m, delta_m};
use crate::input::{Action, InputState};

/// Lowest `speed_log10`: 10^−1 m/s.
pub const SPEED_LOG10_MIN: f64 = -1.0;
/// Highest `speed_log10`: 10^26 m/s, about 3 Gpc/s.
pub const SPEED_LOG10_MAX: f64 = 26.0;
/// Change in `speed_log10` per wheel notch or `=`/`-` press.
pub const SPEED_STEP_LOG10: f64 = 0.1;
/// `speed_log10` of a new camera: 10^3 m/s = 1 km/s.
pub const INITIAL_SPEED_LOG10: f64 = 3.0;
/// Look sensitivity of a new camera [rad per raw mouse-motion unit].
pub const DEFAULT_LOOK_SENSITIVITY: f64 = 0.002;
/// Roll rate while Q or E is held [rad/s] (90°/s).
pub const ROLL_RATE_RAD_PER_S: f64 = FRAC_PI_2;

/// Free-flight camera: a [`CameraView`] plus logarithmic speed and mouse-look sensitivity.
#[derive(Clone, Debug)]
pub struct FlyCamera {
    pub view: CameraView,
    /// log10 of the flight speed in m/s, kept in [`SPEED_LOG10_MIN`, `SPEED_LOG10_MAX`].
    pub speed_log10: f64,
    /// Mouse-look turn per raw mouse-motion unit [rad].
    pub look_sensitivity: f64,
    /// The frame the camera moves with (M1 Task A6). Change it with [`FlyCamera::reanchor`].
    pub anchor: FrameId,
    /// Offset from the anchor frame's origin [m], root axes (so the camera follows the frame's
    /// origin, not its spin); [`FlyCamera::sync`] sets `view.pos` from it.
    ///
    /// Anchored to [`ROOT_FRAME`], which never moves, `view.pos` is the camera's position and
    /// `rel_m` only mirrors it in f64 metres from the SSB (4.3e9 m apart at 1 Gpc), so `sync`
    /// keeps `view.pos` there instead of rebuilding it from `rel_m`.
    pub rel_m: DVec3,
}

impl FlyCamera {
    /// Wraps `view`, starting at 1 km/s, anchored to the root frame.
    pub fn new(view: CameraView) -> Self {
        let rel_m = view.pos.delta_from(UniversePos::default());
        FlyCamera {
            view,
            speed_log10: INITIAL_SPEED_LOG10,
            look_sensitivity: DEFAULT_LOOK_SENSITIVITY,
            anchor: ROOT_FRAME,
            rel_m,
        }
    }

    /// Moves the camera into `new_anchor`, keeping its root position exactly (M1 Task A6).
    ///
    /// `view.pos` is taken as the camera's root position at `t` and left untouched; `rel_m`
    /// becomes its offset from `new_anchor`'s origin at `t` in f64 metres (`UniversePos::
    /// delta_from` with the cell-local difference kept exact, so a later `sync` lands within a
    /// spacing of `rel_m` plus 2^-11 m even at a cell's edge). Call [`Self::sync`] first when
    /// time has moved since the last one. To place the camera by hand, set `view.pos` and call
    /// this.
    ///
    /// Reanchoring to the current anchor when `rel_m` already reproduces `view.pos` at `t`
    /// (the app's every-frame call after `sync`) keeps `rel_m` bit for bit, so repeated calls
    /// never walk the camera off its body by rounding.
    ///
    /// # Panics
    /// If `new_anchor` is not a frame of `frames`.
    pub fn reanchor(&mut self, frames: &FrameTree, t: Epoch, new_anchor: FrameId) {
        let origin = frames.origin_in_root(new_anchor, t);
        if new_anchor == self.anchor && add_m(origin, self.rel_m) == self.view.pos {
            return;
        }
        self.anchor = new_anchor;
        self.rel_m = delta_m(self.view.pos, origin);
    }

    /// Sets `view.pos` = origin(anchor) + `rel_m` at `t` (M1 Task A6), rounding once.
    ///
    /// For [`ROOT_FRAME`] (whose origin is the SSB at every epoch) `view.pos` is kept, since
    /// rebuilding it from f64 metres would cost up to 2^-53 of the distance from the SSB, and
    /// `rel_m` is refreshed from it.
    ///
    /// # Panics
    /// If `anchor` is not a frame of `frames`.
    pub fn sync(&mut self, frames: &FrameTree, t: Epoch) {
        if self.anchor == ROOT_FRAME {
            self.rel_m = self.view.pos.delta_from(UniversePos::default());
        } else {
            self.view.pos = add_m(frames.origin_in_root(self.anchor, t), self.rel_m);
        }
    }

    /// The view as F12 saves it in a Bookmark v2 (M1 Task A6): with `anchor` = the anchor
    /// body's name, `pos` is `rel_m` in cell 0 (metres from the body's centre, root axes); while
    /// anchored to [`ROOT_FRAME`], it is the root position itself, saved without `anchor`.
    pub fn anchored_view(&self) -> CameraView {
        let mut view = self.view.clone();
        if self.anchor != ROOT_FRAME {
            view.pos = UniversePos {
                cell: I64Vec3::ZERO,
                local: self.rel_m,
            };
        }
        view
    }

    /// Flight speed [m/s] = 10^speed_log10.
    pub fn speed_mps(&self) -> f64 {
        10_f64.powf(self.speed_log10)
    }

    /// Whether the speed exceeds the speed of light, which is non-physical.
    pub fn is_ftl(&self) -> bool {
        self.speed_mps() > C
    }

    /// Advances one frame of `dt_s` seconds.
    ///
    /// Takes the wheel notches and speed presses (±0.1 in log10 each), then the mouse-look
    /// delta, then turns and moves. Rotations are about camera-local axes: yaw about +Y, pitch
    /// about +X, roll about −Z (the view direction). Movement is along camera-local axes at
    /// [`Self::speed_mps`]; diagonal input is normalised, so it is no faster. The step moves
    /// `rel_m` and `view.pos` alike, so the camera is in place before the next [`Self::sync`].
    pub fn update(&mut self, input: &mut InputState, dt_s: f64) {
        // A NaN or negative frame time must not move the camera backwards or poison it.
        let dt = if dt_s.is_finite() { dt_s.max(0.0) } else { 0.0 };

        let mut notches = input.take_wheel();
        while input.take_pressed_once(Action::SpeedUp) {
            notches += 1.0;
        }
        while input.take_pressed_once(Action::SpeedDown) {
            notches -= 1.0;
        }
        if notches.is_finite() {
            self.speed_log10 += notches * SPEED_STEP_LOG10;
        }
        self.speed_log10 = self.speed_log10.clamp(SPEED_LOG10_MIN, SPEED_LOG10_MAX);

        // Raw mouse +x is rightwards and +y downwards. Turning right is a negative rotation
        // about +Y and looking down a negative rotation about +X; rolling right is a positive
        // rotation about −Z.
        let look = input.take_mouse_delta() * self.look_sensitivity;
        let roll = axis(input, Action::RollRight, Action::RollLeft) * ROLL_RATE_RAD_PER_S * dt;
        let turn = DQuat::from_rotation_y(-look.x)
            * DQuat::from_rotation_x(-look.y)
            * DQuat::from_axis_angle(DVec3::NEG_Z, roll);
        // Right-multiplying applies the turn in camera-local axes. Renormalising every frame
        // stops rounding drift; a zero or non-finite orientation (say, from a hand-edited
        // bookmark) resets to identity so NaN never reaches the GPU.
        let orient = (self.view.orient * turn).normalize();
        self.view.orient = if orient.is_finite() {
            orient
        } else {
            DQuat::IDENTITY
        };

        let dir_local = DVec3::new(
            axis(input, Action::Right, Action::Left),
            axis(input, Action::Up, Action::Down),
            axis(input, Action::Back, Action::Forward),
        )
        .normalize_or_zero();
        let step = self.view.orient * dir_local * (self.speed_mps() * dt);
        self.rel_m += step;
        self.view.pos = self.view.pos.offset(step);
    }
}

/// +1 if only `pos` is held, −1 if only `neg` is, else 0.
fn axis(input: &InputState, pos: Action, neg: Action) -> f64 {
    f64::from(i8::from(input.is_down(pos)) - i8::from(input.is_down(neg)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;
    use uni_core::{AU, CELL_SIZE_M, FrameMotion, PC, UniversePos};

    fn camera_at_origin() -> FlyCamera {
        FlyCamera::new(CameraView {
            pos: UniversePos::default(),
            orient: DQuat::IDENTITY,
            fov_y_rad: 60_f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        })
    }

    #[test]
    fn forward_moves_along_minus_z() {
        let mut cam = camera_at_origin();
        assert_eq!(cam.speed_log10, 3.0);
        let start = cam.view.pos;
        let mut input = InputState::new();
        input.set_down(Action::Forward, true);
        cam.update(&mut input, 1.0);
        let moved = cam.view.pos.delta_from(start);
        assert!(
            (moved - DVec3::new(0.0, 0.0, -1000.0)).length() < 1e-6,
            "{moved}"
        );
    }

    #[test]
    fn movement_follows_orientation() {
        // Yawed 90° right, "forward" is world +X; "up" stays world +Y.
        let mut cam = camera_at_origin();
        cam.view.orient = DQuat::from_rotation_y(-FRAC_PI_2);
        let mut input = InputState::new();
        input.set_down(Action::Forward, true);
        input.set_down(Action::Up, true);
        let start = cam.view.pos;
        cam.update(&mut input, 1.0);
        let moved = cam.view.pos.delta_from(start);
        // Diagonal input is normalised, so the speed is still 1000 m/s.
        let expected = DVec3::new(1.0, 1.0, 0.0).normalize() * 1000.0;
        assert!((moved - expected).length() < 1e-6, "{moved}");
    }

    #[test]
    fn look_and_roll_turn_about_camera_axes() {
        let forward = |c: &FlyCamera| c.view.orient * DVec3::NEG_Z;
        let up = |c: &FlyCamera| c.view.orient * DVec3::Y;
        let mut input = InputState::new();

        // Mouse right (raw +x) turns right, towards camera +X.
        let mut cam = camera_at_origin();
        input.add_mouse_delta(DVec2::new(10.0, 0.0));
        cam.update(&mut input, 0.0);
        assert!(
            forward(&cam).x > 0.0 && forward(&cam).y.abs() < 1e-12,
            "{}",
            forward(&cam)
        );
        let yaw = forward(&cam).x.atan2(-forward(&cam).z);
        assert!((yaw - 10.0 * cam.look_sensitivity).abs() < 1e-12);

        // Mouse down (raw +y) looks down.
        let mut cam = camera_at_origin();
        input.add_mouse_delta(DVec2::new(0.0, 10.0));
        cam.update(&mut input, 0.0);
        assert!(
            forward(&cam).y < 0.0 && forward(&cam).x.abs() < 1e-12,
            "{}",
            forward(&cam)
        );

        // E rolls right: up tilts towards camera +X; the view direction is unchanged.
        let mut cam = camera_at_origin();
        input.set_down(Action::RollRight, true);
        cam.update(&mut input, 0.1);
        input.set_down(Action::RollRight, false);
        assert!(up(&cam).x > 0.0, "{}", up(&cam));
        assert!((forward(&cam) - DVec3::NEG_Z).length() < 1e-12);

        // Yaw happens about the camera's own +Y: after rolling 90° right, mouse right turns
        // the view towards world −Y (the camera's +X).
        let mut cam = camera_at_origin();
        cam.view.orient = DQuat::from_axis_angle(DVec3::NEG_Z, FRAC_PI_2);
        input.add_mouse_delta(DVec2::new(10.0, 0.0));
        cam.update(&mut input, 0.0);
        assert!(
            forward(&cam).y < 0.0 && forward(&cam).x.abs() < 1e-12,
            "{}",
            forward(&cam)
        );
    }

    #[test]
    fn orientation_stays_unit() {
        let mut cam = camera_at_origin();
        let mut input = InputState::new();
        input.set_down(Action::RollLeft, true);
        for _ in 0..100_000 {
            input.add_mouse_delta(DVec2::new(3.0, -2.0));
            cam.update(&mut input, 0.016);
        }
        let len = cam.view.orient.length();
        assert!((len - 1.0).abs() < 1e-9, "|orient| = {len}");
    }

    #[test]
    fn update_renormalises_orientation() {
        // f64 drift over the 1e5 steps above stays under 1e-9 even without renormalising, so
        // pin the renormalisation directly: one update brings a 0.1 %-long quaternion to unit.
        let mut cam = camera_at_origin();
        cam.view.orient = DQuat::from_xyzw(0.0, 0.0, 0.0, 1.001);
        cam.update(&mut InputState::new(), 0.016);
        let len = cam.view.orient.length();
        assert!((len - 1.0).abs() < 1e-15, "|orient| = {len}");
    }

    #[test]
    fn wheel_and_keys_change_speed_by_a_tenth_decade_per_notch() {
        let mut cam = camera_at_origin();
        let mut input = InputState::new();
        input.add_wheel(3.0);
        input.press(Action::SpeedUp);
        input.press(Action::SpeedDown);
        input.press(Action::SpeedDown);
        cam.update(&mut input, 0.0);
        assert!((cam.speed_log10 - 3.2).abs() < 1e-12, "{}", cam.speed_log10);
        assert!(
            !input.take_pressed_once(Action::SpeedUp),
            "update left a speed press untaken"
        );
        assert_eq!(input.take_wheel(), 0.0, "update left wheel notches untaken");
    }

    #[test]
    fn speed_clamped() {
        let mut cam = camera_at_origin();
        let mut input = InputState::new();
        for _ in 0..400 {
            input.press(Action::SpeedUp);
            cam.update(&mut input, 0.0);
        }
        assert_eq!(cam.speed_log10, 26.0);
        for _ in 0..400 {
            input.press(Action::SpeedDown);
            cam.update(&mut input, 0.0);
        }
        assert_eq!(cam.speed_log10, -1.0);
    }

    #[test]
    fn extreme_speed_stays_finite() {
        // Review Focus 5: 1000 frames at 10^26 m/s fly about 1.6e27 m (about 52 Gpc).
        let mut cam = camera_at_origin();
        cam.speed_log10 = 26.0;
        let start = cam.view.pos;
        let mut input = InputState::new();
        input.set_down(Action::Forward, true);
        for _ in 0..1000 {
            cam.update(&mut input, 0.016);
        }
        let pos = cam.view.pos;
        assert!(pos.local.is_finite(), "{pos:?}");
        assert!(
            pos.local.abs().max_element() <= CELL_SIZE_M / 2.0,
            "{pos:?}"
        );
        let max_cell = pos.cell.abs().max_element();
        assert!(
            max_cell > 0 && max_cell < i64::MAX,
            "cell saturated: {pos:?}"
        );
        assert!(cam.view.orient.is_finite());

        let moved = pos.delta_from(start);
        let expected = -1000.0 * 1e26 * 0.016;
        assert!(
            moved.is_finite() && moved.x == 0.0 && moved.y == 0.0,
            "{moved}"
        );
        assert!((moved.z / expected - 1.0).abs() < 1e-9, "{moved}");

        // Back at 1 m/s, a 1 m step out there is still mm-precise.
        cam.speed_log10 = 0.0;
        let before = cam.view.pos;
        cam.update(&mut input, 1.0);
        let step = cam.view.pos.delta_from(before);
        assert!(
            (step - DVec3::new(0.0, 0.0, -1.0)).length() < 1e-3,
            "{step}"
        );
    }

    #[test]
    fn ftl_flag() {
        let mut cam = camera_at_origin();
        cam.speed_log10 = 9.0;
        assert!(cam.is_ftl());
        cam.speed_log10 = 8.0;
        assert!(!cam.is_ftl());
        assert!((cam.speed_mps() - 1e8).abs() < 1e-3);
    }

    /// A frame that moves along root +Y at `speed_mps` from `start` (root axes, as M1's body
    /// frames are).
    fn moving(start: UniversePos, speed_mps: f64) -> FrameMotion {
        FrameMotion::Custom(Box::new(move |t: Epoch| {
            let y = speed_mps * t.seconds_since(Epoch::J2000);
            (start.offset(DVec3::new(0.0, y, 0.0)), DQuat::IDENTITY)
        }))
    }

    fn fixed_at(origin: UniversePos) -> FrameMotion {
        FrameMotion::Fixed {
            origin,
            rotation: DQuat::IDENTITY,
        }
    }

    fn camera_at(pos: UniversePos) -> FlyCamera {
        let mut cam = camera_at_origin();
        cam.view.pos = pos;
        cam.rel_m = pos.delta_from(UniversePos::default());
        cam
    }

    #[test]
    fn reanchor_preserves_root_position() {
        // Review Focus 5. Two anchors 1 AU apart, first near the SSB, then inside a system frame
        // 1 Gpc out; the camera sits 1.3e7 m from the first anchor.
        let t = Epoch::J2000.add_seconds(1.0e7);
        let far = UniversePos::from_meters(DVec3::new(1e9 * PC, -2e8 * PC, 3e8 * PC))
            .offset(DVec3::new(0.375, 0.0, 0.0));
        // The same 1 Gpc out, with the system frame placed so the camera and anchor b lie in
        // different cells (x and y), where cell-local metres are 2^-10 m apart.
        let half = CELL_SIZE_M / 2.0;
        let edge = UniversePos {
            cell: far.cell,
            local: DVec3::new(
                half - 0.8 * AU + 0.314_159_265_358_979_3,
                half - 3.3e6 - 0.271_828_182_845_904_5,
                -half + 4.0e6 + 0.577_215_664_901_532_9,
            ),
        };
        for base in [UniversePos::default(), far, edge] {
            let mut frames = FrameTree::new();
            let system = frames.add(ROOT_FRAME, "system", fixed_at(base));
            let a = frames.add(
                system,
                "a",
                moving(UniversePos::from_meters(DVec3::new(0.3 * AU, 0.0, 0.0)), 30e3),
            );
            let b = frames.add(
                system,
                "b",
                moving(UniversePos::from_meters(DVec3::new(1.3 * AU, 0.0, 0.0)), -20e3),
            );
            let offset = DVec3::new(
                3.0e6 + 0.123_456_789_1,
                -4.0e6 - 0.987_654_321,
                1.2e7 + 0.333_333_333_333,
            );
            let mut cam = camera_at(frames.origin_in_root(a, t).offset(offset));
            cam.reanchor(&frames, t, a);
            assert_eq!(cam.anchor, a);
            assert!((cam.rel_m - offset).length() < 1e-3, "{}", cam.rel_m);
            cam.sync(&frames, t);
            let before = cam.view.pos;

            for (to, expected_rel) in [
                (b, before.delta_from(frames.origin_in_root(b, t))),
                (a, offset),
                (system, before.delta_from(base)),
                (ROOT_FRAME, before.delta_from(UniversePos::default())),
                (a, offset),
            ] {
                let pos = cam.view.pos;
                cam.reanchor(&frames, t, to);
                assert_eq!(cam.anchor, to);
                // The root position is kept exactly, and syncing at the same epoch moves it by
                // at most a rounding of the cell-local metres (summed over every hop here).
                assert_eq!(cam.view.pos, pos, "reanchor to {to}");
                cam.sync(&frames, t);
                let moved = cam.view.pos.delta_from(before).length();
                assert!(moved < 1e-3, "reanchor to {to}: moved {moved} m");
                let rel_err = (cam.rel_m - expected_rel).length();
                assert!(rel_err <= 1e-3 + 1e-15 * expected_rel.length(), "{rel_err} m");
            }
        }
    }

    #[test]
    fn reanchor_every_frame_does_not_drift() {
        // The app reanchors to the dominant frame every frame (B9); with the same anchor that
        // must leave `rel_m` bit-for-bit alone, or rounding would walk the camera off its body.
        let mut frames = FrameTree::new();
        let body = frames.add(
            ROOT_FRAME,
            "body",
            moving(UniversePos::from_meters(DVec3::new(5.2 * AU, 0.0, 0.0)), 13e3),
        );
        let mut cam = camera_at(UniversePos::from_meters(DVec3::new(5.2 * AU, 0.0, 7.0e7)));
        cam.reanchor(&frames, Epoch::J2000, body);
        let rel = cam.rel_m;
        for i in 1..=10_000 {
            let t = Epoch::J2000.add_seconds(37.7 * f64::from(i));
            cam.sync(&frames, t);
            cam.reanchor(&frames, t, body);
        }
        assert_eq!(cam.rel_m, rel);

        // A camera moved by hand (view.pos set directly) is re-measured instead.
        let t = Epoch::J2000.add_seconds(377_000.0);
        cam.sync(&frames, t);
        cam.view.pos = cam.view.pos.offset(DVec3::new(1000.0, 0.0, 0.0));
        cam.reanchor(&frames, t, body);
        assert!((cam.rel_m - rel - DVec3::new(1000.0, 0.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn anchored_camera_follows_moving_frame() {
        // Review Focus 1: time warp carries the camera with its body. 30 km/s for 1e6 s, in
        // 1000 frames of 1/60 s at a warp of 6e4.
        let mut frames = FrameTree::new();
        let body = frames.add(
            ROOT_FRAME,
            "body",
            moving(UniversePos::from_meters(DVec3::new(AU, 0.0, 0.0)), 30e3),
        );
        let mut cam = camera_at(UniversePos::from_meters(DVec3::new(AU, 0.0, 1e7)));
        cam.reanchor(&frames, Epoch::J2000, body);
        cam.sync(&frames, Epoch::J2000);
        let rel = cam.rel_m;
        let start = cam.view.pos;
        let mut input = InputState::new();
        for i in 1..=1000 {
            cam.update(&mut input, 1.0 / 60.0);
            cam.sync(&frames, Epoch::J2000.add_seconds(1000.0 * f64::from(i)));
        }
        assert_eq!(cam.rel_m, rel);
        let moved = cam.view.pos.delta_from(start);
        assert!(
            (moved - DVec3::new(0.0, 3e10, 0.0)).length() < 1e-3,
            "{moved}"
        );
    }

    #[test]
    fn update_moves_rel_m_and_view_together() {
        let mut frames = FrameTree::new();
        let body = frames.add(
            ROOT_FRAME,
            "body",
            moving(UniversePos::from_meters(DVec3::new(AU, 0.0, 0.0)), 30e3),
        );
        let t0 = Epoch::J2000;
        let mut cam = camera_at(UniversePos::from_meters(DVec3::new(AU, 0.0, 1e7)));
        cam.reanchor(&frames, t0, body);
        cam.sync(&frames, t0);
        let (rel0, pos0) = (cam.rel_m, cam.view.pos);

        let mut input = InputState::new();
        input.set_down(Action::Forward, true);
        cam.update(&mut input, 1.0);
        let step = cam.rel_m - rel0;
        assert!((step - DVec3::new(0.0, 0.0, -1000.0)).length() < 1e-9, "{step}");
        // Before the next sync, view.pos has moved by the same step, at the old epoch.
        assert!((cam.view.pos.delta_from(pos0) - step).length() < 1e-3);
        let t1 = t0.add_seconds(1.0);
        cam.sync(&frames, t1);
        let expected = frames.origin_in_root(body, t1).offset(cam.rel_m);
        assert_eq!(cam.view.pos, expected);
        let carried = cam.view.pos.delta_from(pos0) - step;
        assert!((carried - DVec3::new(0.0, 30e3, 0.0)).length() < 1e-3, "{carried}");
    }

    #[test]
    fn root_anchor_keeps_cell_precision_far_away() {
        // Review Focus 5. 1 Gpc out, f64 metres from the SSB are 4.3e9 m apart; anchored to the
        // root frame (which never moves), a 1 m step must survive `sync` all the same.
        let frames = FrameTree::new();
        let far = UniversePos::from_meters(DVec3::new(1e9 * PC, 0.0, 0.0));
        let mut cam = FlyCamera::new(CameraView {
            pos: far,
            orient: DQuat::IDENTITY,
            fov_y_rad: 1.0,
            velocity_mps: DVec3::ZERO,
        });
        assert_eq!(cam.anchor, ROOT_FRAME);
        cam.speed_log10 = 0.0;
        let mut input = InputState::new();
        input.set_down(Action::Forward, true);
        for i in 1..=10 {
            cam.update(&mut input, 1.0);
            cam.sync(&frames, Epoch::J2000.add_seconds(f64::from(i)));
        }
        let moved = cam.view.pos.delta_from(far);
        assert!(
            (moved - DVec3::new(0.0, 0.0, -10.0)).length() < 1e-3,
            "{moved}"
        );
        // rel_m mirrors the root position as well as f64 metres can.
        assert_eq!(cam.rel_m, cam.view.pos.delta_from(UniversePos::default()));
    }

    #[test]
    fn degenerate_orientation_resets_instead_of_nan() {
        // A hand-edited bookmark could hold a zero quaternion; NaN must never reach the GPU.
        let mut cam = camera_at_origin();
        cam.view.orient = DQuat::from_xyzw(0.0, 0.0, 0.0, 0.0);
        let mut input = InputState::new();
        input.set_down(Action::Forward, true);
        cam.update(&mut input, 1.0);
        assert!(
            cam.view.orient.is_finite() && cam.view.pos.local.is_finite(),
            "{:?}",
            cam.view
        );
        assert!((cam.view.orient.length() - 1.0).abs() < 1e-12);
    }
}
