//! Keyboard and mouse state from winit events, mapped to navigation actions (M0 Task 6; M1
//! Task C3 adds selection, goto, cruise, the time keys and the cursor).
//!
//! Keys are mapped by physical position ([`KeyCode`]), so the keyboard layout does not matter.
//! Raw mouse motion counts only while the right button is held (mouse look). A left click (a
//! press and release of the left button within [`CLICK_SLOP_PX`] of each other) is one press
//! of [`Action::Select`]; the app picks at [`InputState::cursor_px`].

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
    // M1 Task C3 (names frozen by the C0 context, ruling C.10 and the owner-feedback mapping).
    /// A left click: the app picks the object under [`InputState::cursor_px`].
    Select,
    /// F: fly to the selection (`Navigator::goto_selected`).
    Goto,
    /// C: free flight ↔ cruise.
    Cruise,
    /// T: show or hide the time-warp panel (UI state; the app takes it).
    TimePanel,
    /// P: pause or resume time (ruling 7; `handle_time_keys`).
    TimePause,
    /// `.`: the next faster warp preset.
    TimeFaster,
    /// `,`: the next slower warp preset.
    TimeSlower,
    /// Esc: cancel an aim, else abort a goto (the navigator), else collapse the card, else
    /// deselect (the app).
    Deselect,
    /// I: expand or collapse the target card (the app).
    ToggleCard,
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
/// A left press and release at most this far apart [window physical px] is a click; further is
/// a drag (the owner-approved preview's tap threshold, `74-hud.js`: "dragDist < 6").
pub const CLICK_SLOP_PX: f64 = 6.0;

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
        // M1 Task C3 (design §5.6 "F: focus/goto", "T: time-warp controls"; ruling 7: P pauses,
        // Space stays "up").
        KeyCode::KeyF => Action::Goto,
        KeyCode::KeyC => Action::Cruise,
        KeyCode::KeyT => Action::TimePanel,
        KeyCode::KeyP => Action::TimePause,
        KeyCode::Period => Action::TimeFaster,
        KeyCode::Comma => Action::TimeSlower,
        KeyCode::Escape => Action::Deselect,
        KeyCode::KeyI => Action::ToggleCard,
        _ => return None,
    };
    Some(action)
}

/// Keys held, presses not yet taken, mouse motion and wheel accumulated since the last take, and
/// the cursor.
///
/// [`Navigator::update`](crate::Navigator::update) takes the mouse delta, the wheel, the speed
/// and movement presses, [`Action::Goto`], [`Action::Cruise`] and the [`Action::Deselect`]
/// presses it uses (cancelling an aim or a goto); [`crate::handle_time_keys`] takes the time
/// presses; the app takes [`Action::ToggleHud`], [`Action::Screenshot`], [`Action::Help`],
/// [`Action::Select`], [`Action::TimePanel`], [`Action::ToggleCard`] and the Esc presses the
/// navigator left.
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
    /// The cursor [window physical px], once it has moved over the window.
    cursor: Option<DVec2>,
    /// The left button went down over the window (where the cursor was, if known), and has not
    /// come up yet.
    left_press: Option<Option<DVec2>>,
}

impl InputState {
    /// Nothing held, nothing pending.
    pub fn new() -> Self {
        Self::default()
    }

    /// Handles keys, the mouse wheel, the right (look) and left (click) buttons, the cursor and
    /// focus loss; ignores the rest.
    ///
    /// The app may withhold events the UI consumed, but should still forward key and button
    /// releases, or a key or mouse look can stick. A left release whose press was withheld is
    /// not a click.
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
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => self.left_press = Some(self.cursor),
                ElementState::Released => {
                    if let Some(from) = self.left_press.take() {
                        let moved = match (from, self.cursor) {
                            (Some(a), Some(b)) => a.distance(b),
                            _ => 0.0,
                        };
                        if moved <= CLICK_SLOP_PX {
                            self.press(Action::Select);
                        }
                    }
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Some(DVec2::new(position.x, position.y));
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            // Releases made while another window has focus never arrive, so drop everything
            // held rather than let a key, mouse look or click stick.
            WindowEvent::Focused(false) => {
                self.down.clear();
                self.looking = false;
                self.left_press = None;
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

    /// The cursor [window physical px, origin top left], `None` before it first moves over the
    /// window or after it leaves. The app scales it by internal ÷ output size for
    /// `Navigator::pick` (ruling C.10).
    pub fn cursor_px(&self) -> Option<DVec2> {
        self.cursor
    }

    /// Whether a press of `a` waits to be taken; does not take it (the navigator takes an Esc
    /// only when it has an aim or a goto to cancel, and leaves it for the app otherwise).
    pub fn has_pending_press(&self, a: Action) -> bool {
        self.presses.get(&a).is_some_and(|&n| n > 0)
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

    fn left_button(state: ElementState) -> WindowEvent {
        WindowEvent::MouseInput {
            device_id: DeviceId::dummy(),
            state,
            button: MouseButton::Left,
        }
    }

    fn cursor(x: f64, y: f64) -> WindowEvent {
        WindowEvent::CursorMoved {
            device_id: DeviceId::dummy(),
            position: PhysicalPosition::new(x, y),
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
        for unmapped in [KeyCode::KeyZ, KeyCode::ControlRight, KeyCode::KeyX] {
            assert_eq!(action_for(unmapped), None, "{unmapped:?}");
        }
    }

    #[test]
    fn new_actions_mapped() {
        let expected = [
            (KeyCode::KeyF, Action::Goto),
            (KeyCode::KeyC, Action::Cruise),
            (KeyCode::KeyT, Action::TimePanel),
            (KeyCode::KeyP, Action::TimePause),
            (KeyCode::Period, Action::TimeFaster),
            (KeyCode::Comma, Action::TimeSlower),
            (KeyCode::Escape, Action::Deselect),
            (KeyCode::KeyI, Action::ToggleCard),
        ];
        for (code, action) in expected {
            assert_eq!(action_for(code), Some(action), "{code:?}");
        }
        // Ruling 7: Space stays "up"; the old map is unchanged.
        assert_eq!(action_for(KeyCode::Space), Some(Action::Up));
        // Select has no key: it is a left click.
        let mut input = InputState::new();
        input.on_window_event(&cursor(100.0, 200.0));
        input.on_window_event(&left_button(ElementState::Pressed));
        input.on_window_event(&left_button(ElementState::Released));
        assert!(input.take_pressed_once(Action::Select));
        assert!(!input.take_pressed_once(Action::Select));
    }

    #[test]
    fn time_keys_do_not_repeat() {
        let mut input = InputState::new();
        input.on_key(KeyCode::Period, true, false);
        input.on_key(KeyCode::Period, true, true);
        assert!(input.take_pressed_once(Action::TimeFaster));
        assert!(!input.take_pressed_once(Action::TimeFaster));
    }

    #[test]
    fn cursor_is_tracked_in_physical_px() {
        let mut input = InputState::new();
        assert_eq!(input.cursor_px(), None);
        input.on_window_event(&cursor(12.5, 40.0));
        assert_eq!(input.cursor_px(), Some(DVec2::new(12.5, 40.0)));
        input.on_window_event(&WindowEvent::CursorLeft {
            device_id: DeviceId::dummy(),
        });
        assert_eq!(input.cursor_px(), None);
    }

    #[test]
    fn click_selects_but_a_drag_does_not() {
        let mut input = InputState::new();
        // A press and release within CLICK_SLOP_PX is a click.
        input.on_window_event(&cursor(100.0, 100.0));
        input.on_window_event(&left_button(ElementState::Pressed));
        input.on_window_event(&cursor(103.0, 104.0));
        input.on_window_event(&left_button(ElementState::Released));
        assert!(input.take_pressed_once(Action::Select));
        // Moved further than CLICK_SLOP_PX between press and release: a drag, no click.
        input.on_window_event(&left_button(ElementState::Pressed));
        input.on_window_event(&cursor(150.0, 104.0));
        input.on_window_event(&left_button(ElementState::Released));
        assert!(!input.take_pressed_once(Action::Select));
        // A release whose press the UI kept (never forwarded) is no click.
        input.on_window_event(&left_button(ElementState::Released));
        assert!(!input.take_pressed_once(Action::Select));
        // Focus loss forgets a held left button.
        input.on_window_event(&left_button(ElementState::Pressed));
        input.on_window_event(&WindowEvent::Focused(false));
        input.on_window_event(&left_button(ElementState::Released));
        assert!(!input.take_pressed_once(Action::Select));
    }

    #[test]
    fn pending_press_is_seen_without_taking_it() {
        let mut input = InputState::new();
        assert!(!input.has_pending_press(Action::Deselect));
        input.on_key(KeyCode::Escape, true, false);
        assert!(input.has_pending_press(Action::Deselect));
        assert!(input.has_pending_press(Action::Deselect), "peeking took it");
        assert!(input.take_pressed_once(Action::Deselect));
        assert!(!input.has_pending_press(Action::Deselect));
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

    const ALL_ACTIONS: [Action; 22] = [
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
        Action::Select,
        Action::Goto,
        Action::Cruise,
        Action::TimePanel,
        Action::TimePause,
        Action::TimeFaster,
        Action::TimeSlower,
        Action::Deselect,
        Action::ToggleCard,
    ];
}
