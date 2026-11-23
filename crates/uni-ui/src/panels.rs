//! The HUD's panels: what the app fills for each (Task C4; Task C0 wrote the types).
//!
//! Collapsed by default (owner feedback 2026-11-22; lead ruling on the C0 context, owner point
//! 4): a muted place line with an "i" button at the top left ([`NavLine`]), the muted notes line
//! at the bottom left (which holds the flight label, ruling 6), one mono time line at the top
//! centre (ruling 15), the perf line at the bottom right. The Location, Gravity and Speed panels
//! show only while the card is expanded (`HudControls::card_expanded`).

use uni_core::Epoch;

use crate::theme::LabelTone;

/// Location (top left): kicker LOCATION, the region, then From Sun, Light time and Frame.
#[derive(Clone, Debug)]
pub struct LocationPanel {
    /// "Near Mars", "Interstellar space", …
    pub region: String,
    pub from_sun_m: f64,
    /// One-way light time from the Sun [s].
    pub light_time_s: f64,
    /// The camera's anchor frame, by its body's name.
    pub frame: String,
}

/// Time (top right; one mono line at top centre while a card is open, ruling 15). Whether the
/// warp buttons show is `HudControls::show_warp` (ruling C.4).
#[derive(Clone, Debug)]
pub struct TimePanel {
    pub epoch: Epoch,
    /// Simulated seconds per real second.
    pub rate: f64,
    pub paused: bool,
}

/// Gravity (bottom left), kicker `theme::gravity_kicker(body)`: calculators only, never applied
/// (ruling 6).
#[derive(Clone, Debug)]
pub struct GravityPanel {
    pub body: String,
    /// Local gravitational acceleration [m/s²] (`uni_physics::calc::local_g`).
    pub local_g: f64,
    pub escape_mps: f64,
    pub circular_mps: f64,
}

/// Speed (bottom centre): the header in its tone (amber `FREE FLIGHT · GRAVITY NOT SIMULATED`
/// or `CRUISE · FTL · NON-PHYSICAL · GRAVITY IGNORED`), the speed, its sub-line.
#[derive(Clone, Debug)]
pub struct SpeedPanel {
    pub header: String,
    pub tone: LabelTone,
    pub speed_mps: f64,
    /// E.g. "auto drop-out near stars, atmospheres and black holes" while cruising.
    pub note: Option<String>,
}

/// The collapsed HUD's one muted place line (top left, beside the "i" button), after the
/// owner-approved preview (`74-hud.js`); C4 owns the strings.
#[derive(Clone, Debug, PartialEq)]
pub enum NavLine {
    /// Nothing selected or aimed at: the place the camera is at.
    Here { name: String, dist_m: f64 },
    /// An object is selected: its name, distance and closing speed (> 0 approaching).
    Selected {
        name: String,
        dist_m: f64,
        closing_mps: f64,
    },
    /// The aim ring is filling on a destination ("→ <name>", "<dist> · hold to fly there, or
    /// click it").
    Aiming {
        name: String,
        dist_m: f64,
        fill: f32,
    },
    /// A goto is running ("En route · <name>", "<dist> to go · travel time compressed").
    EnRoute { name: String, to_go_m: f64 },
}
