//! When the window tries to present (Task C6w): the main loop's answer to a surface that gives
//! no texture.
//!
//! wgpu 30's Metal backend returns `CurrentSurfaceTexture::Occluded` at once while the window's
//! `occlusionState` is not visible (minimised, hidden, on another Space, or covered by another
//! app's window; wgpu-hal `metal/surface.rs`, the workaround for gfx-rs/wgpu#8309). Before this
//! module the loop asked for the next redraw straight away, so an occluded window spun one core
//! at about 270 000 acquire attempts per second, presented nothing and logged its smoothed
//! "fps" as about 3 × 10⁵ (the C6 window run's "stall", task-C6w-report.md). Now:
//! - **Occluded** (winit's `WindowEvent::Occluded(true)`, the acquire status, or a zero window
//!   size): the loop sleeps (`ControlFlow::WaitUntil`) and tries again every
//!   [`OCCLUDED_RETRY`], so a missed `Occluded(false)` costs at most that long;
//!   `Occluded(false)` resumes at once.
//! - **Timeout, a validation error, or a surface to reconfigure** (suboptimal, outdated, lost):
//!   the first retry is immediate, then the waits double from [`FIRST_BACKOFF`] up to
//!   [`OCCLUDED_RETRY`], so a surface that keeps failing neither spins nor floods the log.
//! - **A frame** resets everything.
//!
//! The simulation does not run while no frame is presented (as before): the first frame after
//! a pause moves nothing (dt = 0, `App::redraw`), so the hidden time is not simulated.

use std::time::{Duration, Instant};

/// How often an occluded window tries to present again (a missed `Occluded(false)` event, or a
/// platform that sends none, costs at most this long).
pub const OCCLUDED_RETRY: Duration = Duration::from_millis(250);
/// The first wait after a second failure in a row that is not an occlusion; it doubles per
/// failure up to [`OCCLUDED_RETRY`].
pub const FIRST_BACKOFF: Duration = Duration::from_millis(1);
/// A pause that is not an occlusion and lasts longer than this is logged as a warning.
pub const STUCK_WARNING: Duration = Duration::from_secs(1);

/// What one attempt to get a surface texture gave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attempt {
    /// A texture: the frame was rendered and presented.
    Frame,
    /// Minimised, hidden or covered (`Occluded`, or a zero window size).
    Occluded,
    /// No drawable in time (`Timeout`).
    Timeout,
    /// The surface was reconfigured (`Suboptimal`, `Outdated`, `Lost`).
    Reconfigured,
    /// A validation error, reported through the device's error handler (`Validation`).
    Validation,
}

impl Attempt {
    const SKIPS: [Attempt; 4] = [
        Attempt::Occluded,
        Attempt::Timeout,
        Attempt::Reconfigured,
        Attempt::Validation,
    ];

    fn index(self) -> Option<usize> {
        Self::SKIPS.iter().position(|&a| a == self)
    }

    fn name(self) -> &'static str {
        match self {
            Attempt::Frame => "frame",
            Attempt::Occluded => "occluded",
            Attempt::Timeout => "timeout",
            Attempt::Reconfigured => "reconfigured",
            Attempt::Validation => "validation error",
        }
    }
}

/// The surface status of `t` as an [`Attempt`] (any texture it holds is dropped).
pub fn attempt_of(t: &wgpu::CurrentSurfaceTexture) -> Attempt {
    match t {
        wgpu::CurrentSurfaceTexture::Success(_) => Attempt::Frame,
        wgpu::CurrentSurfaceTexture::Occluded => Attempt::Occluded,
        wgpu::CurrentSurfaceTexture::Timeout => Attempt::Timeout,
        wgpu::CurrentSurfaceTexture::Suboptimal(_)
        | wgpu::CurrentSurfaceTexture::Outdated
        | wgpu::CurrentSurfaceTexture::Lost => Attempt::Reconfigured,
        wgpu::CurrentSurfaceTexture::Validation => Attempt::Validation,
    }
}

/// A change the main loop logs.
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    /// Presenting stopped: the first attempt without a frame.
    Paused(Attempt),
    /// A pause that is not an occlusion has lasted [`STUCK_WARNING`] (logged once per pause).
    Stuck {
        for_s: f64,
        last: Attempt,
    },
    /// Presenting again, after `for_s` seconds and these attempts without a frame.
    Resumed {
        for_s: f64,
        skipped: String,
    },
}

/// The window's present state (module docs).
#[derive(Debug, Default)]
pub struct PresentGate {
    /// winit says the window is occluded, or the last attempt did.
    occluded: bool,
    /// Failures in a row that are not occlusions.
    failures: u32,
    /// No attempt before this.
    retry_at: Option<Instant>,
    /// When presenting stopped, while it is stopped.
    paused_since: Option<Instant>,
    /// Attempts without a frame in this pause, per [`Attempt::SKIPS`].
    skipped: [u64; 4],
    stuck_logged: bool,
}

impl PresentGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// winit's `WindowEvent::Occluded`. Becoming visible makes the next attempt immediate.
    pub fn on_occluded(&mut self, occluded: bool, now: Instant) {
        self.occluded = occluded;
        if occluded {
            self.retry_at = Some(now + OCCLUDED_RETRY);
        } else {
            self.failures = 0;
            self.retry_at = None;
        }
    }

    /// The outcome of one attempt; returns what to log, if anything.
    pub fn on_attempt(&mut self, attempt: Attempt, now: Instant) -> Option<Change> {
        let Some(k) = attempt.index() else {
            // A frame.
            let change = self.paused_since.map(|since| Change::Resumed {
                for_s: (now - since).as_secs_f64(),
                skipped: self.skipped_summary(),
            });
            *self = PresentGate::default();
            return change;
        };
        self.skipped[k] += 1;
        let wait = if attempt == Attempt::Occluded {
            self.occluded = true;
            OCCLUDED_RETRY
        } else {
            self.failures += 1;
            backoff(self.failures)
        };
        self.retry_at = (!wait.is_zero()).then_some(now + wait);
        match self.paused_since {
            None => {
                self.paused_since = Some(now);
                Some(Change::Paused(attempt))
            }
            Some(since)
                if !self.stuck_logged
                    && attempt != Attempt::Occluded
                    && now - since >= STUCK_WARNING =>
            {
                self.stuck_logged = true;
                Some(Change::Stuck {
                    for_s: (now - since).as_secs_f64(),
                    last: attempt,
                })
            }
            Some(_) => None,
        }
    }

    /// `None`: try to present now (the loop polls and asks for a redraw). `Some(t)`: sleep
    /// until `t` or the next event.
    pub fn next_attempt(&self, now: Instant) -> Option<Instant> {
        self.retry_at.filter(|&t| t > now)
    }

    /// Whether the window is presenting (no attempt without a frame since the last frame).
    pub fn presenting(&self) -> bool {
        self.paused_since.is_none()
    }

    /// "occluded 12, timeout 1": this pause's attempts without a frame.
    fn skipped_summary(&self) -> String {
        Attempt::SKIPS
            .iter()
            .zip(self.skipped)
            .filter(|(_, n)| *n > 0)
            .map(|(a, n)| format!("{} {n}", a.name()))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The wait after the `n`-th failure in a row that is not an occlusion: none after the first
/// (a reconfigured surface usually works at once), then [`FIRST_BACKOFF`] doubling up to
/// [`OCCLUDED_RETRY`].
fn backoff(n: u32) -> Duration {
    if n <= 1 {
        return Duration::ZERO;
    }
    let doublings = (n - 2).min(16);
    (FIRST_BACKOFF * (1u32 << doublings)).min(OCCLUDED_RETRY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(t: Instant, ms: u64) -> Instant {
        t + Duration::from_millis(ms)
    }

    /// The C6 stall: an occluded window used to retry at once, ~270 000 times a second. Now it
    /// waits OCCLUDED_RETRY between attempts, and a frame resumes with the pause reported.
    #[test]
    fn an_occluded_window_waits_between_attempts_instead_of_spinning() {
        let t0 = Instant::now();
        let mut g = PresentGate::new();
        assert_eq!(g.next_attempt(t0), None);
        assert_eq!(
            g.on_attempt(Attempt::Occluded, t0),
            Some(Change::Paused(Attempt::Occluded))
        );
        assert!(!g.presenting());
        assert_eq!(g.next_attempt(t0), Some(t0 + OCCLUDED_RETRY));
        assert_eq!(g.next_attempt(ms(t0, 249)), Some(t0 + OCCLUDED_RETRY));
        // Without any event, the next attempt comes OCCLUDED_RETRY later: at most four a second.
        let mut t = t0;
        let mut attempts = 1;
        while t < ms(t0, 10_000) {
            t = g.next_attempt(t).unwrap_or(t);
            assert_eq!(g.on_attempt(Attempt::Occluded, t), None, "logged once per pause");
            attempts += 1;
        }
        assert!(attempts <= 42, "{attempts} attempts in 10 s");
        let back = ms(t0, 10_100);
        match g.on_attempt(Attempt::Frame, back) {
            Some(Change::Resumed { for_s, skipped }) => {
                assert!((for_s - 10.1).abs() < 1e-9, "{for_s}");
                assert_eq!(skipped, format!("occluded {attempts}"));
            }
            other => panic!("{other:?}"),
        }
        assert!(g.presenting());
        assert_eq!(g.next_attempt(back), None);
        assert_eq!(g.on_attempt(Attempt::Frame, back), None);
    }

    /// winit's Occluded(true) stops the attempts before any fails; Occluded(false) resumes at
    /// once, without waiting out the retry.
    #[test]
    fn visibility_events_pause_and_resume_at_once() {
        let t0 = Instant::now();
        let mut g = PresentGate::new();
        g.on_occluded(true, t0);
        assert_eq!(g.next_attempt(ms(t0, 1)), Some(t0 + OCCLUDED_RETRY));
        g.on_occluded(false, ms(t0, 5));
        assert_eq!(g.next_attempt(ms(t0, 5)), None);
        // A spurious Occluded(true) costs one retry interval, then the acquire decides.
        g.on_occluded(true, ms(t0, 10));
        assert_eq!(g.next_attempt(ms(t0, 260)), None);
        assert_eq!(g.on_attempt(Attempt::Frame, ms(t0, 260)), None);
        assert_eq!(g.next_attempt(ms(t0, 260)), None);
    }

    /// A surface that keeps failing (timeouts, validation errors, reconfigures) retries once at
    /// once, then backs off by doubling to OCCLUDED_RETRY, and warns once after STUCK_WARNING.
    #[test]
    fn repeated_failures_back_off_to_the_cap_and_warn_once() {
        let t0 = Instant::now();
        let mut g = PresentGate::new();
        assert_eq!(
            g.on_attempt(Attempt::Reconfigured, t0),
            Some(Change::Paused(Attempt::Reconfigured))
        );
        assert_eq!(g.next_attempt(t0), None, "a reconfigured surface retries at once");
        let mut t = t0;
        let mut waits = Vec::new();
        let mut stuck = Vec::new();
        for _ in 0..40 {
            if let Some(c) = g.on_attempt(Attempt::Timeout, t) {
                stuck.push(c);
            }
            let next = g.next_attempt(t).unwrap_or(t);
            waits.push(next - t);
            t = next;
        }
        assert_eq!(waits[0], FIRST_BACKOFF);
        assert_eq!(waits[1], FIRST_BACKOFF * 2);
        assert!(waits.windows(2).all(|w| w[1] >= w[0]), "{waits:?}");
        assert!(waits.iter().all(|&w| w <= OCCLUDED_RETRY));
        assert_eq!(*waits.last().unwrap(), OCCLUDED_RETRY);
        assert_eq!(stuck.len(), 1, "{stuck:?}");
        assert!(matches!(
            stuck[0],
            Change::Stuck { for_s, last: Attempt::Timeout } if for_s >= STUCK_WARNING.as_secs_f64()
        ));
        // A frame clears the back-off.
        assert!(matches!(
            g.on_attempt(Attempt::Frame, t),
            Some(Change::Resumed { .. })
        ));
        assert_eq!(g.next_attempt(t), None);
        assert_eq!(
            g.on_attempt(Attempt::Validation, t),
            Some(Change::Paused(Attempt::Validation))
        );
        assert_eq!(g.next_attempt(t), None);
    }

    #[test]
    fn backoff_doubles_from_the_second_failure_and_is_capped() {
        assert_eq!(backoff(0), Duration::ZERO);
        assert_eq!(backoff(1), Duration::ZERO);
        assert_eq!(backoff(2), Duration::from_millis(1));
        assert_eq!(backoff(5), Duration::from_millis(8));
        assert_eq!(backoff(9), OCCLUDED_RETRY.min(Duration::from_millis(128)));
        assert_eq!(backoff(10), OCCLUDED_RETRY);
        assert_eq!(backoff(u32::MAX), OCCLUDED_RETRY);
    }

    #[test]
    fn every_surface_status_maps_to_an_attempt() {
        assert_eq!(
            attempt_of(&wgpu::CurrentSurfaceTexture::Occluded),
            Attempt::Occluded
        );
        assert_eq!(
            attempt_of(&wgpu::CurrentSurfaceTexture::Timeout),
            Attempt::Timeout
        );
        assert_eq!(
            attempt_of(&wgpu::CurrentSurfaceTexture::Outdated),
            Attempt::Reconfigured
        );
        assert_eq!(
            attempt_of(&wgpu::CurrentSurfaceTexture::Lost),
            Attempt::Reconfigured
        );
        assert_eq!(
            attempt_of(&wgpu::CurrentSurfaceTexture::Validation),
            Attempt::Validation
        );
    }
}
