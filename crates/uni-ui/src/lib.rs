//! uni-ui: the egui overlay. M0: HUD v0 (FPS, GPU ms per pass, position, speed) and unit formatting.
//! M1 Wave C (stubs by Task C0): the approved HUD ([`hud`], [`theme`], [`panels`], [`cards`];
//! Task C4) and the overlays: target brackets, markers and orbits ([`overlay`]; Task C5).
//!
//! Only C0 and the lead edit this file (module list, re-exports); C4 and C5 do not.

pub mod cards;
pub mod format;
pub mod hud;
pub mod overlay;
pub mod panels;
pub mod theme;

pub use cards::{CardAction, CardRow, TargetCard, build_card};
pub use format::{format_distance, format_epoch, format_speed};
pub use hud::{Hud, HudControls, HudData, HudOutput};
pub use overlay::{
    Bracket, Marker, MarkerKind, OrbitLine, Overlay, OverlaySettings, build_overlay, draw_overlay,
    exo_orbit_points,
};
pub use panels::{GravityPanel, LocationPanel, NavLine, SpeedPanel, TimePanel};
pub use theme::{
    BELT_HIGHLIGHT_LABEL, CRUISE_LABEL, DEBUG_GRID_LABEL, EYE_ADAPTED_STARS_LABEL,
    FREE_FLIGHT_LABEL, FlightLabel, LabelTone, SATURN_MOONS_LABEL, TWINKLE_LABEL,
    UNLENSED_STARS_LABEL, VIVID_STAR_COLOURS_LABEL, gravity_kicker, tone_color, ui_labels,
};
