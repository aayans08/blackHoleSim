//! Keyboard and mouse state from winit events, mapped to navigation actions (M0 Task 6).
//!
//! Keys are mapped by physical position ([`KeyCode`]), so the keyboard layout does not matter.
//! Raw mouse motion counts only while the right button is held (mouse look).

use std::collections::{HashMap, HashSet};

use glam::DVec2;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Something the user can ask the navigation layer to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Forward,
    Back,
    Left,
    Right,
    Up,
    Down,
    RollLeft,
    RollRight,
    SpeedUp,
    SpeedDown,
    ToggleHud,
    Screenshot,
    Help,
}

impl Action {
    /// Whether the OS key-repeat of a held key counts as another press. Only the speed keys
    /// repeat, so holding `=` keeps speeding up while holding Tab toggles the HUD once.
    fn repeats(self) -> bool {
        matches!(self, Action::SpeedUp | Action::SpeedDown)
    }
}

/// Physical pixels of pixel-precise (trackpad) scrolling that count as one wheel notch.
pub const PIXELS_PER_NOTCH: f64 = 50.0;

/// Maps a physical key to its action (the default key map, design §5.6).
pub fn action_for(code: KeyCode) -> Option<Action> {
    let action = match code {
        KeyCode::KeyW => Action::Forward,
        KeyCode::KeyS => Action::Back,
        KeyCode::KeyA => Action::Left,
        KeyCode::KeyD => Action::Right,
        KeyCode::Space => Action::Up,
        KeyCode::ControlLeft => Action::Down,
        KeyCode::KeyQ => Action::RollLeft,
        KeyCode::KeyE => Action::RollRight,
        KeyCode::Equal => Action::SpeedUp,
        KeyCode::Minus => Action::SpeedDown,
        KeyCode::Tab => Action::ToggleHud,
        KeyCode::F12 => Action::Screenshot,
        KeyCode::F1 => Action::Help,
        _ => return None,
    };
    Some(action)
}

/// Keys held, presses not yet taken, and mouse motion and wheel accumulated since the last take.
///
/// [`FlyCamera::update`](crate::FlyCamera::update) takes the mouse delta, the wheel and the
/// speed presses; the app takes [`Action::ToggleHud`], [`Action::Screenshot`] and
/// [`Action::Help`].
#[derive(Debug, Default)]
pub struct InputState {
    /// Actions whose key is held.
    down: HashSet<Action>,
    /// Presses of each action not yet returned by `take_pressed_once`.
    presses: HashMap<Action, u32>,
    /// The right mouse button is held, so raw mouse motion turns the camera.
    looking: bool,
    /// Raw mouse motion while looking, since the last `take_mouse_delta`.
    mouse_delta: DVec2,
    /// Wheel notches since the last `take_wheel`; positive speeds up.
    wheel: f64,
}

impl InputState {
    /// Nothing held, nothing pending.
    pub fn new() -> Self {
        Self::default()
    }

    /// Handles keys, the mouse wheel, the right (look) button and focus loss; ignores the rest.
    ///
    /// The app may withhold events the UI consumed, but should still forward key and button
    /// releases, or a key or mouse look can stick.
    pub fn on_window_event(&mut self, e: &WindowEvent) {
        match e {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.on_key(code, event.state == ElementState::Pressed, event.repeat);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Positive y is scrolling up (wheel away from the user), as the OS reports it.
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(*y),
                    MouseScrollDelta::PixelDelta(p) => p.y / PIXELS_PER_NOTCH,
                };
                self.add_wheel(notches);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => self.set_looking(*state == ElementState::Pressed),
            // Releases made while another window has focus never arrive, so drop everything
            // held rather than let a key or mouse look stick.
            WindowEvent::Focused(false) => {
                self.down.clear();
                self.looking = false;
            }
            _ => {}
        }
    }

    /// Accumulates raw mouse motion, but only while mouse look is on.
    pub fn on_device_event(&mut self, e: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = e
            && self.looking
        {
            self.add_mouse_delta(DVec2::new(*dx, *dy));
        }
    }

    /// The keyboard path of [`Self::on_window_event`], for a key code already in hand (winit's
    /// `KeyEvent` cannot be built outside winit). `repeat` marks an OS key-repeat of a held key.
    pub fn on_key(&mut self, code: KeyCode, pressed: bool, repeat: bool) {
        let Some(a) = action_for(code) else {
            return;
        };
        if pressed && (!repeat || a.repeats()) {
            self.press(a);
        }
        self.set_down(a, pressed);
    }

    /// Marks `a` held or released. Counts no press.
    pub fn set_down(&mut self, a: Action, down: bool) {
        if down {
            self.down.insert(a);
        } else {
            self.down.remove(&a);
        }
    }

    /// Counts one press of `a` for [`Self::take_pressed_once`]. Does not hold it.
    pub fn press(&mut self, a: Action) {
        let n = self.presses.entry(a).or_insert(0);
        *n = n.saturating_add(1);
    }

    /// Turns mouse look on or off.
    pub fn set_looking(&mut self, looking: bool) {
        self.looking = looking;
    }

    /// Whether mouse look is on; the app hides and confines the cursor while it is.
    pub fn is_looking(&self) -> bool {
        self.looking
    }

    /// Adds raw mouse motion whether or not mouse look is on.
    pub fn add_mouse_delta(&mut self, d: DVec2) {
        self.mouse_delta += d;
    }

    /// Adds wheel notches; positive speeds up.
    pub fn add_wheel(&mut self, notches: f64) {
        self.wheel += notches;
    }

    /// Whether the key for `a` is held.
    pub fn is_down(&self, a: Action) -> bool {
        self.down.contains(&a)
    }

    /// Returns the raw mouse motion since the last call and resets it.
    pub fn take_mouse_delta(&mut self) -> DVec2 {
        std::mem::take(&mut self.mouse_delta)
    }

    /// Returns the wheel notches since the last call and resets them.
    pub fn take_wheel(&mut self) -> f64 {
        std::mem::take(&mut self.wheel)
    }

    /// Returns true once for each press of `a` not yet taken.
    pub fn take_pressed_once(&mut self, a: Action) -> bool {
        match self.presses.get_mut(&a) {
            Some(n) if *n > 0 => {
                *n -= 1;
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::dpi::PhysicalPosition;
    use winit::event::{DeviceId, TouchPhase};

    fn right_button(state: ElementState) -> WindowEvent {
        WindowEvent::MouseInput {
            device_id: DeviceId::dummy(),
            state,
            button: MouseButton::Right,
        }
    }

    fn motion(dx: f64, dy: f64) -> DeviceEvent {
        DeviceEvent::MouseMotion { delta: (dx, dy) }
    }

    fn wheel(delta: MouseScrollDelta) -> WindowEvent {
        WindowEvent::MouseWheel {
            device_id: DeviceId::dummy(),
            delta,
            phase: TouchPhase::Moved,
        }
    }

    #[test]
    fn key_map_matches_brief() {
        let expected = [
            (KeyCode::KeyW, Action::Forward),
            (KeyCode::KeyS, Action::Back),
            (KeyCode::KeyA, Action::Left),
            (KeyCode::KeyD, Action::Right),
            (KeyCode::Space, Action::Up),
            (KeyCode::ControlLeft, Action::Down),
            (KeyCode::KeyQ, Action::RollLeft),
            (KeyCode::KeyE, Action::RollRight),
            (KeyCode::Equal, Action::SpeedUp),
            (KeyCode::Minus, Action::SpeedDown),
            (KeyCode::Tab, Action::ToggleHud),
            (KeyCode::F12, Action::Screenshot),
            (KeyCode::F1, Action::Help),
        ];
        for (code, action) in expected {
            assert_eq!(action_for(code), Some(action), "{code:?}");
        }
        for unmapped in [KeyCode::KeyZ, KeyCode::ControlRight, KeyCode::Escape] {
            assert_eq!(action_for(unmapped), None, "{unmapped:?}");
        }
    }

    #[test]
    fn key_press_is_held_until_release_and_taken_once() {
        let mut input = InputState::new();
        input.on_key(KeyCode::KeyW, true, false);
        assert!(input.is_down(Action::Forward));
        assert!(input.take_pressed_once(Action::Forward));
        assert!(!input.take_pressed_once(Action::Forward));
        assert!(
            input.is_down(Action::Forward),
            "taking a press must not release the key"
        );
        input.on_key(KeyCode::KeyW, false, false);
        assert!(!input.is_down(Action::Forward));
        input.on_key(KeyCode::KeyZ, true, false);
        assert!(
            ALL_ACTIONS.iter().all(|&a| !input.is_down(a)),
            "unmapped key changed state"
        );
    }

    #[test]
    fn key_repeat_counts_only_for_speed() {
        let mut input = InputState::new();
        input.on_key(KeyCode::Equal, true, false);
        input.on_key(KeyCode::Equal, true, true);
        input.on_key(KeyCode::Equal, true, true);
        input.on_key(KeyCode::Tab, true, false);
        input.on_key(KeyCode::Tab, true, true);
        input.on_key(KeyCode::Tab, true, true);
        let mut speed_presses = 0;
        while input.take_pressed_once(Action::SpeedUp) {
            speed_presses += 1;
        }
        assert_eq!(speed_presses, 3);
        assert!(input.take_pressed_once(Action::ToggleHud));
        assert!(
            !input.take_pressed_once(Action::ToggleHud),
            "a held Tab toggled the HUD twice"
        );
    }

    #[test]
    fn mouse_motion_counts_only_while_right_button_held() {
        let mut input = InputState::new();
        input.on_device_event(&motion(5.0, 5.0));
        assert_eq!(input.take_mouse_delta(), DVec2::ZERO);

        input.on_window_event(&right_button(ElementState::Pressed));
        assert!(input.is_looking());
        input.on_device_event(&motion(3.0, -2.0));
        input.on_device_event(&motion(3.0, -2.0));
        assert_eq!(input.take_mouse_delta(), DVec2::new(6.0, -4.0));
        assert_eq!(input.take_mouse_delta(), DVec2::ZERO);

        input.on_window_event(&right_button(ElementState::Released));
        assert!(!input.is_looking());
        input.on_device_event(&motion(7.0, 1.0));
        assert_eq!(input.take_mouse_delta(), DVec2::ZERO);
    }

    #[test]
    fn wheel_lines_and_pixels_become_notches() {
        let mut input = InputState::new();
        input.on_window_event(&wheel(MouseScrollDelta::LineDelta(0.0, 2.0)));
        input.on_window_event(&wheel(MouseScrollDelta::LineDelta(0.0, -0.5)));
        assert_eq!(input.take_wheel(), 1.5);
        assert_eq!(input.take_wheel(), 0.0);
        let pixels = PhysicalPosition::new(0.0, -2.0 * PIXELS_PER_NOTCH);
        input.on_window_event(&wheel(MouseScrollDelta::PixelDelta(pixels)));
        assert_eq!(input.take_wheel(), -2.0);
    }

    #[test]
    fn focus_loss_releases_keys_and_look() {
        let mut input = InputState::new();
        input.on_key(KeyCode::KeyW, true, false);
        input.on_window_event(&right_button(ElementState::Pressed));
        input.on_window_event(&WindowEvent::Focused(false));
        assert!(!input.is_down(Action::Forward));
        assert!(!input.is_looking());
    }

    const ALL_ACTIONS: [Action; 13] = [
        Action::Forward,
        Action::Back,
        Action::Left,
        Action::Right,
        Action::Up,
        Action::Down,
        Action::RollLeft,
        Action::RollRight,
        Action::SpeedUp,
        Action::SpeedDown,
        Action::ToggleHud,
        Action::Screenshot,
        Action::Help,
    ];
}
