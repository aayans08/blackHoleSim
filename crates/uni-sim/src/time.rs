//! Simulation time: epoch, warp rate and pause (M1 Task A4). A0b stub; A4 implements.

use uni_core::Epoch;

/// Warp rates [sim s per real s] that `step_up`/`step_down` move between.
pub const WARP_PRESETS: [f64; 5] = [1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12];
/// Largest |rate| [sim s per real s].
pub const MAX_RATE: f64 = 1.0e12;

/// The simulation clock.
pub struct TimeController {
    /// Placeholder for the private state (epoch, rate, paused); A4 replaces it.
    _private: (),
}

impl TimeController {
    /// Starts at `epoch`, rate 1, running.
    pub fn new(_epoch: Epoch) -> Self {
        todo!("A4")
    }

    pub fn epoch(&self) -> Epoch {
        todo!("A4")
    }

    pub fn set_epoch(&mut self, _e: Epoch) {
        todo!("A4")
    }

    /// Warp rate [sim s per real s].
    pub fn rate(&self) -> f64 {
        todo!("A4")
    }

    /// Sets the rate, clamped to ±[`MAX_RATE`]; NaN is ignored.
    pub fn set_rate(&mut self, _r: f64) {
        todo!("A4")
    }

    /// Next faster preset.
    pub fn step_up(&mut self) {
        todo!("A4")
    }

    /// Next slower preset.
    pub fn step_down(&mut self) {
        todo!("A4")
    }

    pub fn toggle_pause(&mut self) {
        todo!("A4")
    }

    pub fn is_paused(&self) -> bool {
        todo!("A4")
    }

    /// Advances by `dt_real_s` of wall time; returns the simulation step [s] (0 while paused).
    pub fn advance(&mut self, _dt_real_s: f64) -> f64 {
        todo!("A4")
    }
}
