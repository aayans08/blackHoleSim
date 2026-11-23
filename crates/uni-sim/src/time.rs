//! Simulation time: epoch, warp rate and pause (M1 Task A4).

use uni_core::Epoch;

/// Warp rates [sim s per real s] that `step_up`/`step_down` move between.
pub const WARP_PRESETS: [f64; 5] = [1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12];
/// Largest |rate| [sim s per real s].
pub const MAX_RATE: f64 = 1.0e12;

/// The simulation clock.
pub struct TimeController {
    epoch: Epoch,
    /// Sim seconds per real second; negative runs time backwards. |rate| ≤ [`MAX_RATE`].
    rate: f64,
    paused: bool,
}

impl TimeController {
    /// Starts at `epoch`, rate 1, running.
    pub fn new(epoch: Epoch) -> Self {
        TimeController {
            epoch,
            rate: 1.0,
            paused: false,
        }
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    pub fn set_epoch(&mut self, e: Epoch) {
        self.epoch = e;
    }

    /// Warp rate [sim s per real s]; pausing does not change it.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Sets the rate, clamped to ±[`MAX_RATE`]; NaN is ignored.
    pub fn set_rate(&mut self, r: f64) {
        if !r.is_nan() {
            self.rate = r.clamp(-MAX_RATE, MAX_RATE);
        }
    }

    /// Next faster preset: the smallest one above |rate|, keeping the sign (reverse time stays
    /// reversed). The fastest preset stays.
    pub fn step_up(&mut self) {
        let size = self.rate.abs();
        if let Some(&next) = WARP_PRESETS.iter().find(|&&p| p > size) {
            self.rate = next.copysign(self.rate);
        }
    }

    /// Next slower preset: the largest one below |rate|, keeping the sign. Real time (±1)
    /// stays; stepping down never pauses or reverses time.
    pub fn step_down(&mut self) {
        let size = self.rate.abs();
        if let Some(&next) = WARP_PRESETS.iter().rev().find(|&&p| p < size) {
            self.rate = next.copysign(self.rate);
        }
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Advances by `dt_real_s` of wall time; returns the simulation step [s] (0 while paused).
    /// A wall-time step that is not a positive finite number moves nothing.
    pub fn advance(&mut self, dt_real_s: f64) -> f64 {
        if self.paused || !(dt_real_s > 0.0 && dt_real_s.is_finite()) {
            return 0.0;
        }
        let step = self.rate * dt_real_s;
        self.epoch = self.epoch.add_seconds(step);
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_presets_and_pause() {
        let mut time = TimeController::new(Epoch::J2000);
        assert_eq!(time.rate(), 1.0);
        assert!(!time.is_paused());
        for _ in 0..4 {
            time.step_up();
        }
        assert_eq!(time.rate(), 1.0e12);
        time.step_up();
        assert_eq!(time.rate(), 1.0e12, "the fastest preset stays");
        time.toggle_pause();
        assert!(time.is_paused());
        assert_eq!(time.advance(1.0 / 60.0), 0.0);
        assert_eq!(time.epoch(), Epoch::J2000, "paused time does not move");
        time.toggle_pause();
        let step = time.advance(0.5);
        assert_eq!(step, 0.5e12);
        assert_eq!(time.epoch(), Epoch::J2000.add_seconds(0.5e12));
    }

    #[test]
    fn step_down_walks_the_presets_and_stops_at_real_time() {
        let mut time = TimeController::new(Epoch::J2000);
        time.set_rate(5.0e9);
        time.step_down();
        assert_eq!(time.rate(), 1.0e9, "between presets: the next one below");
        time.set_rate(5.0e9);
        time.step_up();
        assert_eq!(time.rate(), 1.0e12, "between presets: the next one above");
        for _ in 0..10 {
            time.step_down();
        }
        assert_eq!(time.rate(), 1.0);
        // Reverse time keeps its sign while stepping.
        time.set_rate(-1.0e3);
        time.step_up();
        assert_eq!(time.rate(), -1.0e6);
        time.step_down();
        time.step_down();
        assert_eq!(time.rate(), -1.0);
    }

    #[test]
    fn set_rate_clamps_and_ignores_nan() {
        let mut time = TimeController::new(Epoch::J2000);
        time.set_rate(3.0e15);
        assert_eq!(time.rate(), MAX_RATE);
        time.set_rate(f64::NEG_INFINITY);
        assert_eq!(time.rate(), -MAX_RATE);
        time.set_rate(f64::NAN);
        assert_eq!(time.rate(), -MAX_RATE, "NaN leaves the rate unchanged");
        time.set_rate(-2.0);
        assert_eq!(time.advance(3.0), -6.0);
        assert_eq!(time.epoch(), Epoch::J2000.add_seconds(-6.0));
    }

    #[test]
    fn advance_ignores_bad_wall_time() {
        let mut time = TimeController::new(Epoch::J2000);
        time.set_rate(1.0e6);
        for dt in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert_eq!(time.advance(dt), 0.0, "dt {dt}");
        }
        assert_eq!(time.epoch(), Epoch::J2000);
        let later = Epoch::J2000.add_seconds(86_400.0);
        time.set_epoch(later);
        assert_eq!(time.epoch(), later);
    }
}
