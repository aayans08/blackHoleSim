//! The HUD's look and its UI-state honesty labels (Task C4; Task C0 wrote this file with the
//! mockup's values, so C5 draws in the right colours before C4 merges; C4 owns it afterwards).
//!
//! Values: the owner-approved HUD mockup (four artboards at 1440 × 900 pt; egui points equal
//! CSS px; plan C4 "The design") and the owner-approved look preview's tokens
//! (`.superpowers/look-preview/v2/parts/00-head.html`). Colour carries meaning only: cyan =
//! selected target and navigation, amber = non-physical or hypothetical, red = danger, every
//! other thing neutral grey on translucent glass. Alphas are given as fractions (the mockup's
//! CSS) and rounded to the nearest of 255 steps in the `Color32`s.
//!
//! Labels: every UI-state label is defined here once, character for character from the plan's
//! table "Honesty labels in M1"; world-state labels live in `uni_sim::honesty`.

use egui::Color32;

/// `a` (0 ..= 1) as the nearest 8-bit alpha.
const fn alpha(a: f32) -> u8 {
    (a * 255.0 + 0.5) as u8
}

// ---- Colours (mockup palette and preview tokens) ----

/// Values and titles (preview `--ink`).
pub const INK: Color32 = Color32::from_rgb(0xE6, 0xEA, 0xF0);
/// Row labels (preview `--ink-2`).
pub const INK_2: Color32 = Color32::from_rgb(0x9A, 0xA4, 0xB2);
/// Section kickers and muted lines (preview `--ink-3`).
pub const INK_3: Color32 = Color32::from_rgb(0x8D, 0x97, 0xA6);
/// The muted grey of the collapsed HUD's lines and of neutral labels: [`INK_3`].
pub const MUTED: Color32 = INK_3;
/// Alpha of [`INK_SOFT`] (preview `--ink-soft`, rgba(214, 222, 234, 0.62)).
pub const INK_SOFT_ALPHA: f32 = 0.62;
pub const INK_SOFT: Color32 =
    Color32::from_rgba_unmultiplied_const(214, 222, 234, alpha(INK_SOFT_ALPHA));
/// Selected target and navigation (mockup cyan, preview `--target`).
pub const TARGET: Color32 = Color32::from_rgb(0x7D, 0xD3, 0xFC);
/// Alpha of [`TARGET_LINE`] (preview `--target-line`; the target card's border).
pub const TARGET_LINE_ALPHA: f32 = 0.35;
pub const TARGET_LINE: Color32 =
    Color32::from_rgba_unmultiplied_const(125, 211, 252, alpha(TARGET_LINE_ALPHA));
/// Non-physical or hypothetical (mockup amber, preview `--nonphys`).
pub const NON_PHYSICAL: Color32 = Color32::from_rgb(0xF5, 0xB0, 0x4C);
/// Alpha of [`NON_PHYSICAL_SOFT`] (preview `--nonphys-soft`).
pub const NON_PHYSICAL_SOFT_ALPHA: f32 = 0.80;
pub const NON_PHYSICAL_SOFT: Color32 =
    Color32::from_rgba_unmultiplied_const(245, 176, 76, alpha(NON_PHYSICAL_SOFT_ALPHA));
/// Danger (mockup red).
pub const DANGER: Color32 = Color32::from_rgb(0xF8, 0x71, 0x71);
/// Alerts in running text (preview `--alert`).
pub const ALERT: Color32 = Color32::from_rgb(0xFC, 0xA5, 0xA5);

// ---- Glass panels ----

/// Alpha of [`GLASS`], the panels' fill rgba(10, 14, 22, 0.55). egui has no backdrop blur, so
/// the glass is translucency only.
pub const GLASS_ALPHA: f32 = 0.55;
pub const GLASS: Color32 = Color32::from_rgba_unmultiplied_const(10, 14, 22, alpha(GLASS_ALPHA));
/// Alpha of [`CARD_GLASS`], the target card's fill rgba(10, 14, 22, 0.62).
pub const CARD_GLASS_ALPHA: f32 = 0.62;
pub const CARD_GLASS: Color32 =
    Color32::from_rgba_unmultiplied_const(10, 14, 22, alpha(CARD_GLASS_ALPHA));
/// Alpha of [`HAIRLINE`], the panels' 1 pt border rgba(255, 255, 255, 0.10).
pub const HAIRLINE_ALPHA: f32 = 0.10;
pub const HAIRLINE: Color32 =
    Color32::from_rgba_unmultiplied_const(255, 255, 255, alpha(HAIRLINE_ALPHA));
/// Alpha of [`CRUISE_BORDER`], the Speed panel's border while cruising rgba(245, 176, 76, 0.40).
pub const CRUISE_BORDER_ALPHA: f32 = 0.40;
pub const CRUISE_BORDER: Color32 =
    Color32::from_rgba_unmultiplied_const(245, 176, 76, alpha(CRUISE_BORDER_ALPHA));
/// Border width of panels and the card [pt].
pub const BORDER_PT: f32 = 1.0;
/// Panel corner radius [pt].
pub const PANEL_RADIUS: f32 = 10.0;
/// Panel padding [pt]: 14 top and bottom, 16 left and right.
pub const PANEL_PADDING_Y: f32 = 14.0;
pub const PANEL_PADDING_X: f32 = 16.0;
/// Target card corner radius [pt].
pub const CARD_RADIUS: f32 = 12.0;
/// Target card padding [pt]: 20 top and bottom, 22 left and right.
pub const CARD_PADDING_Y: f32 = 20.0;
pub const CARD_PADDING_X: f32 = 22.0;
/// Margin between the panels and the screen edges [pt].
pub const SCREEN_MARGIN: f32 = 24.0;
/// Panel widths [pt]: Location, Time (380 with the warp buttons), Gravity, Speed, target card.
pub const LOCATION_WIDTH: f32 = 300.0;
pub const TIME_WIDTH: f32 = 280.0;
pub const TIME_WARP_WIDTH: f32 = 380.0;
pub const GRAVITY_WIDTH: f32 = 260.0;
pub const SPEED_WIDTH: f32 = 520.0;
pub const CARD_WIDTH: f32 = 420.0;
/// Smallest button height [pt] (touch-size targets).
pub const BUTTON_MIN_HEIGHT: f32 = 44.0;

// ---- Type ----

/// Text family (Inter 300/400/500/600) and number family (IBM Plex Mono 300/400/500), loaded
/// from `data/fonts` (A9).
pub const FONT_TEXT: &str = "Inter";
pub const FONT_MONO: &str = "IBM Plex Mono";
/// Section kickers: 11 pt, weight 500, letter-spacing 0.12 em, uppercase, [`INK_3`].
pub const KICKER_SIZE: f32 = 11.0;
pub const KICKER_TRACKING_EM: f32 = 0.12;
/// Body rows [pt]: labels [`INK_2`], values [`INK`] in mono, right-aligned.
pub const BODY_SIZE: f32 = 13.0;
/// Location's region line [pt].
pub const REGION_SIZE: f32 = 18.0;
/// Time panel's epoch, mono [pt].
pub const EPOCH_SIZE: f32 = 15.0;
/// Speed readout, mono weight 300 [pt].
pub const SPEED_SIZE: f32 = 40.0;
/// Target card title, light [pt].
pub const TITLE_SIZE: f32 = 30.0;
/// Notes (honesty and warnings) [pt].
pub const NOTES_SIZE: f32 = 12.0;
/// Sources line, perf line and hints [pt].
pub const SMALL_SIZE: f32 = 11.0;
/// The speed bar spans 0.1 m/s to 10¹⁸ c on a log scale, with an amber tick at c.
pub const SPEED_BAR_MIN_MPS: f64 = 0.1;
pub const SPEED_BAR_MAX_C: f64 = 1.0e18;

// ---- Overlay (C5) ----

/// Target bracket: arm length and stroke [pt]; half-size = max(projected radius + pad, min).
pub const BRACKET_ARM: f32 = 18.0;
pub const BRACKET_STROKE: f32 = 1.5;
pub const BRACKET_PAD: f32 = 12.0;
pub const BRACKET_MIN_HALF: f32 = 20.0;
/// Bracket name (12 pt, weight 600, letter-spacing 0.14 em, cyan) and sub-line (mono 13 pt).
pub const BRACKET_NAME_SIZE: f32 = 12.0;
pub const BRACKET_NAME_TRACKING_EM: f32 = 0.14;
pub const BRACKET_SUB_SIZE: f32 = 13.0;
/// Marker ring size [pt] and alpha (grey rings; the selected marker is [`TARGET`]).
pub const MARKER_RING_PT: f32 = 6.0;
pub const MARKER_ALPHA: f32 = 0.70;
/// No marker ring over a disk larger than this [pt] (the label only).
pub const MARKER_MAX_DISK_PT: f32 = 20.0;
/// Orbit polylines: stroke [pt], alpha of the grey line, points per orbit.
pub const ORBIT_STROKE: f32 = 1.0;
pub const ORBIT_ALPHA: f32 = 0.35;
pub const ORBIT_POINTS: usize = 256;
/// The aim ring (preview): radius and stroke [pt], the track's white alpha; the arc is
/// [`TARGET`], from 12 o'clock.
pub const AIM_RING_RADIUS: f32 = 18.0;
pub const AIM_RING_STROKE: f32 = 1.6;
pub const AIM_TRACK_ALPHA: f32 = 0.16;

// ---- Tones ----

/// What a line of text means, which sets its colour ([`tone_color`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelTone {
    /// Approximate but physical, or plain information: muted grey.
    Neutral,
    /// The selected target and navigation: cyan.
    Nav,
    /// Non-physical or hypothetical: amber.
    NonPhysical,
    /// Danger: red.
    Danger,
}

/// The colour of `t`: [`MUTED`], [`TARGET`], [`NON_PHYSICAL`] or [`DANGER`].
pub fn tone_color(t: LabelTone) -> Color32 {
    match t {
        LabelTone::Neutral => MUTED,
        LabelTone::Nav => TARGET,
        LabelTone::NonPhysical => NON_PHYSICAL,
        LabelTone::Danger => DANGER,
    }
}

// ---- UI-state honesty labels (plan "Honesty labels in M1"; exact strings) ----

/// Shown in amber while stars are exposed differently from the scene (design §7 item 4; §2
/// anti-goal 2), exactly while `ExposureSettings::eye_adapted_in_effect()` (review m13): the
/// "Eye-adapted stars" dual exposure is not what a camera would record (m0.1).
pub const EYE_ADAPTED_STARS_LABEL: &str = "Eye-adapted stars (dual exposure, non-physical)";
/// Shown in amber while the "Vivid star colours" option is on (owner feedback, 2026-11-22; on by
/// default): star colours are saturated beyond their Planck chromaticity, keeping their
/// luminance (`uni_sky::vivid_color`), which a camera would not record.
pub const VIVID_STAR_COLOURS_LABEL: &str = "Vivid star colours · non-physical";
/// Free flight: M1 has no ship gravity (ruling 6), amber.
pub const FREE_FLIGHT_LABEL: &str = "FREE FLIGHT · GRAVITY NOT SIMULATED";
/// Cruise at any speed, or a goto above c (plan table), amber.
pub const CRUISE_LABEL: &str = "CRUISE · FTL · NON-PHYSICAL · GRAVITY IGNORED";
/// While "Highlight asteroid belt" is on (round 2, review I12), amber.
pub const BELT_HIGHLIGHT_LABEL: &str = "Asteroid belt brightness exaggerated (non-physical)";
/// While the debug grid shows through a traced lens (ruling 14; m0.1's string), amber.
pub const DEBUG_GRID_LABEL: &str = "Lens background: M0 debug grid";
/// While a lens is traced: the stars over it are drawn unlensed until M3 (the plan's M1
/// wording, replacing m0.1's "(M0)"), amber.
pub const UNLENSED_STARS_LABEL: &str = "Stars near the hole are not yet lensed (M3)";
/// While the camera is inside an atmosphere and the stars twinkle (B4 report: Young's law
/// extrapolated below its aperture range, every star at the zenith), grey.
pub const TWINKLE_LABEL: &str = "Twinkling: approximate model";
/// While one of Saturn's major moons is on screen, and on their cards (A2 ruling I3, ruling 10:
/// Meeus ch. 46 puts Mimas up to 2.45 px and Dione 1.16 px off at the Saturn bookmarks), grey.
pub const SATURN_MOONS_LABEL: &str = "Saturn's major moons: approximate position (Meeus 46)";

/// The Gravity panel's kicker for `body` (ruling 6): `GRAVITY · <BODY> · NOT APPLIED`, the
/// body's name in capitals, amber.
pub fn gravity_kicker(body: &str) -> String {
    format!("GRAVITY · {} · NOT APPLIED", body.to_uppercase())
}

/// Which flight label applies (plan table): [`FlightLabel::Cruise`] while cruising at any speed
/// or in a goto above c, [`FlightLabel::Free`] otherwise (free flight, or a goto below c:
/// gravity is not simulated either way).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlightLabel {
    Free,
    Cruise,
}

/// The UI-state labels in effect, with their tones, in the order the notes line shows them: the
/// flight label ([`FREE_FLIGHT_LABEL`] or [`CRUISE_LABEL`]), [`BELT_HIGHLIGHT_LABEL`] while
/// `highlight_belt`, [`EYE_ADAPTED_STARS_LABEL`] while `eye_adapted` (C4 implements and tests
/// it; the app calls it for both the HUD notes and the capture JSON, ruling C.5).
pub fn ui_labels(
    flight: FlightLabel,
    highlight_belt: bool,
    eye_adapted: bool,
) -> Vec<(&'static str, LabelTone)> {
    let _ = (flight, highlight_belt, eye_adapted);
    todo!("C4")
}

// ---- Info-card notes (plan table, grey; exact strings) ----

/// A planet or moon on JPL mean elements.
pub const NOTE_ORBIT_MEAN_ELEMENTS: &str = "Orbit: JPL mean elements (approximate)";

/// A comet or asteroid moved two-body from its osculation epoch `date_ymd` ("YYYY-MM-DD").
pub fn note_orbit_two_body(date_ymd: &str) -> String {
    format!("Orbit: two-body from epoch {date_ymd} (approximate far from epoch)")
}

/// A body without a surface map.
pub const NOTE_NO_SURFACE_MAP: &str = "No surface map: uniform albedo";
/// A comet's tails.
pub const NOTE_TAILS: &str = "Tails: Finson\u{2013}Probstein dust model; brightness approximate";
/// A comet's nucleus (review m11).
pub const NOTE_NUCLEUS_SHAPE: &str = "Nucleus shape: procedural";
/// An exoplanet orbit with no Ω (every M1 orbit: the archive has no Ω; ruling 13), drawn dashed.
pub const NOTE_NODE_UNKNOWN: &str =
    "Orbit orientation on the sky unknown: drawn with \u{3a9} = 0\u{b0}";
/// An exoplanet orbit with no inclination (ruling 13), drawn dashed.
pub const NOTE_INCLINATION_UNKNOWN: &str = "Inclination unknown: drawn edge-on (i = 90\u{b0})";
/// An exoplanet orbit's phase: the archive's `pl_orblper` is read as the PLANET's argument of
/// periastron (lead ruling B.2; the archive does not say whose it is).
pub const NOTE_PERIASTRON_CONVENTION: &str =
    "Periastron: the archive's \u{3c9} read as the planet's (the archive does not say)";
/// The Sun's surface.
pub const NOTE_GRANULATION: &str = "Granulation: procedural (measured cell size and lifetime)";
/// Venus and Titan.
pub const NOTE_HAZE: &str = "Atmosphere: simplified haze model (approximate)";
/// Saturn.
pub const NOTE_RINGS: &str = "Rings: Solar System Scope texture; phase function approximate";
/// Earth.
pub const NOTE_EARTH_CLOUDS: &str = "Clouds: static map (no weather)";
/// Venus.
pub const NOTE_VENUS_CLOUD_TOPS: &str = "Cloud tops: static map, super-rotating (approximate)";
/// A star flagged `FLAG_NO_RV`.
pub const NOTE_TANGENTIAL_MOTION: &str = "Star motion: tangential only (no radial velocity)";
/// An exoplanet host built from the archive (`FLAG_FROM_ARCHIVE`).
pub const NOTE_ARCHIVE_HOST: &str = "Host star from NASA Exoplanet Archive (not in AT-HYG)";

#[cfg(test)]
mod tests {
    use super::*;

    /// The alphas are the mockup's fractions to the nearest 8-bit step.
    #[test]
    fn alphas_round_to_the_nearest_step() {
        assert_eq!(alpha(0.55), 140);
        assert_eq!(alpha(0.10), 26);
        assert_eq!(alpha(1.0), 255);
        assert_eq!(GLASS.a(), 140);
        assert_eq!(tone_color(LabelTone::NonPhysical), NON_PHYSICAL);
    }

    /// The plan table's strings, character for character, and the kicker's form.
    #[test]
    fn labels_are_the_plan_strings() {
        assert_eq!(
            FREE_FLIGHT_LABEL,
            "FREE FLIGHT \u{b7} GRAVITY NOT SIMULATED"
        );
        assert_eq!(
            CRUISE_LABEL,
            "CRUISE \u{b7} FTL \u{b7} NON-PHYSICAL \u{b7} GRAVITY IGNORED"
        );
        assert_eq!(
            gravity_kicker("Mars"),
            "GRAVITY \u{b7} MARS \u{b7} NOT APPLIED"
        );
        assert_eq!(
            note_orbit_two_body("2026-11-20"),
            "Orbit: two-body from epoch 2026-11-20 (approximate far from epoch)"
        );
        assert_eq!(
            NOTE_NODE_UNKNOWN,
            "Orbit orientation on the sky unknown: drawn with Ω = 0°"
        );
        assert_eq!(
            NOTE_INCLINATION_UNKNOWN,
            "Inclination unknown: drawn edge-on (i = 90°)"
        );
        assert_eq!(
            NOTE_TAILS,
            "Tails: Finson–Probstein dust model; brightness approximate"
        );
    }
}
