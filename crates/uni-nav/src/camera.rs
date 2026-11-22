//! Quaternion 6-DoF free camera with logarithmic speed control (M0 Task 6).
//!
//! Free flight ignores gravity and reaches 10^26 m/s, far above c. Speeds above c are
//! non-physical, and [`FlyCamera::is_ftl`] tells the HUD to label them so.

use std::f64::consts::FRAC_PI_2;

use glam::{DQuat, DVec3};
use uni_core::{C, CameraView};

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
}

impl FlyCamera {
    /// Wraps `view`, starting at 1 km/s.
    pub fn new(view: CameraView) -> Self {
        FlyCamera {
            view,
            speed_log10: INITIAL_SPEED_LOG10,
            look_sensitivity: DEFAULT_LOOK_SENSITIVITY,
        }
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
    /// [`Self::speed_mps`]; diagonal input is normalised, so it is no faster.
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
    use uni_core::{CELL_SIZE_M, UniversePos};

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
