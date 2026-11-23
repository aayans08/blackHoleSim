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
    let flight = match flight {
        FlightLabel::Free => FREE_FLIGHT_LABEL,
        FlightLabel::Cruise => CRUISE_LABEL,
    };
    let mut labels = vec![(flight, LabelTone::NonPhysical)];
    if highlight_belt {
        labels.push((BELT_HIGHLIGHT_LABEL, LabelTone::NonPhysical));
    }
    if eye_adapted {
        labels.push((EYE_ADAPTED_STARS_LABEL, LabelTone::NonPhysical));
    }
    labels
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
/// TestBH-1's card (plan C4): M1 shows its mass, Schwarzschild radius and distance only.
pub const NOTE_BLACK_HOLE_M3: &str = "Full black-hole card in M3";

// ---- More mockup and preview tokens (Task C4) ----

/// Alpha of [`CARD_HAIRLINE`], the target card's inner hairlines rgba(255, 255, 255, 0.08).
pub const CARD_HAIRLINE_ALPHA: f32 = 0.08;
pub const CARD_HAIRLINE: Color32 =
    Color32::from_rgba_unmultiplied_const(255, 255, 255, alpha(CARD_HAIRLINE_ALPHA));
/// Alpha of [`HAIRLINE_STRONG`] (preview `--hairline-strong`): the "i" button's ring.
pub const HAIRLINE_STRONG_ALPHA: f32 = 0.18;
pub const HAIRLINE_STRONG: Color32 =
    Color32::from_rgba_unmultiplied_const(255, 255, 255, alpha(HAIRLINE_STRONG_ALPHA));
/// Alpha of [`INFO_FILL`], the closed "i" button's fill rgba(10, 14, 22, 0.35) (preview).
pub const INFO_FILL_ALPHA: f32 = 0.35;
pub const INFO_FILL: Color32 =
    Color32::from_rgba_unmultiplied_const(10, 14, 22, alpha(INFO_FILL_ALPHA));
/// The "i" button's diameter [pt] (preview `.info-btn`).
pub const INFO_BUTTON_SIZE: f32 = 26.0;
/// Buttons: corner radius [pt]; the cyan "Go to" button's fill rgba(125, 211, 252, 0.12) and
/// border rgba(125, 211, 252, 0.60); a neutral button's border rgba(255, 255, 255, 0.16).
pub const BUTTON_RADIUS: f32 = 8.0;
pub const NAV_BUTTON_FILL_ALPHA: f32 = 0.12;
pub const NAV_BUTTON_FILL: Color32 =
    Color32::from_rgba_unmultiplied_const(125, 211, 252, alpha(NAV_BUTTON_FILL_ALPHA));
pub const NAV_BUTTON_BORDER_ALPHA: f32 = 0.60;
pub const NAV_BUTTON_BORDER: Color32 =
    Color32::from_rgba_unmultiplied_const(125, 211, 252, alpha(NAV_BUTTON_BORDER_ALPHA));
pub const BUTTON_BORDER_ALPHA: f32 = 0.16;
pub const BUTTON_BORDER: Color32 =
    Color32::from_rgba_unmultiplied_const(255, 255, 255, alpha(BUTTON_BORDER_ALPHA));
/// Time-warp buttons: the selected one fill rgba(125, 211, 252, 0.14) and border
/// rgba(125, 211, 252, 0.70); the others a rgba(255, 255, 255, 0.14) border.
pub const WARP_SELECTED_FILL_ALPHA: f32 = 0.14;
pub const WARP_SELECTED_FILL: Color32 =
    Color32::from_rgba_unmultiplied_const(125, 211, 252, alpha(WARP_SELECTED_FILL_ALPHA));
pub const WARP_SELECTED_BORDER_ALPHA: f32 = 0.70;
pub const WARP_SELECTED_BORDER: Color32 =
    Color32::from_rgba_unmultiplied_const(125, 211, 252, alpha(WARP_SELECTED_BORDER_ALPHA));
pub const WARP_BORDER_ALPHA: f32 = 0.14;
pub const WARP_BORDER: Color32 =
    Color32::from_rgba_unmultiplied_const(255, 255, 255, alpha(WARP_BORDER_ALPHA));
/// The speed bar: track rgba(255, 255, 255, 0.08), fill rgba(230, 234, 240, 0.55), 6 pt high
/// with a 3 pt radius; the amber tick at c is 1 pt wide and 12 pt high.
pub const SPEED_BAR_TRACK_ALPHA: f32 = 0.08;
pub const SPEED_BAR_TRACK: Color32 =
    Color32::from_rgba_unmultiplied_const(255, 255, 255, alpha(SPEED_BAR_TRACK_ALPHA));
pub const SPEED_BAR_FILL_ALPHA: f32 = 0.55;
pub const SPEED_BAR_FILL: Color32 =
    Color32::from_rgba_unmultiplied_const(230, 234, 240, alpha(SPEED_BAR_FILL_ALPHA));
pub const SPEED_BAR_HEIGHT: f32 = 6.0;
pub const SPEED_TICK_HEIGHT: f32 = 12.0;
/// The target card's kicker: 11 pt, weight 600, letter-spacing 0.14 em, [`TARGET`].
pub const CARD_KICKER_TRACKING_EM: f32 = 0.14;
/// The honesty line (preview `.np-line`): at least the preview's 560 pt wide, wider where the
/// screen leaves room beside the perf line (so fewer labels end in "…"); its amber dots 6 pt;
/// the "What is approximate here" pop-up 380 pt wide.
pub const NOTES_LINE_MAX_WIDTH: f32 = 560.0;
pub const NOTE_DOT: f32 = 6.0;
pub const NOTES_POPUP_WIDTH: f32 = 380.0;
/// The collapsed HUD's time line, mono [pt].
pub const TIME_LINE_SIZE: f32 = 13.0;

// ---- UI text (not honesty labels) ----

/// The honesty pop-up's title (preview).
pub const NOTES_POPUP_TITLE: &str = "What is approximate here";
/// Shown in red while the camera is inside an event horizon (m0.1).
pub const INSIDE_HORIZON_LABEL: &str = "Inside event horizon";
/// Bottom-right hints: collapsed (the mockup's "Tab hide HUD · F1 controls" plus the I key of
/// the owner's collapsed HUD) and while the card is open (the mockup's target artboard).
pub const HINT_COLLAPSED: &str = "I details · Tab hide HUD · F1 controls";
pub const HINT_CARD: &str = "Esc close card · Tab hide HUD";
/// The settings toggle (round 2, review I12) and its amber tag.
pub const BELT_TOGGLE_LABEL: &str = "Highlight asteroid belt";
/// The time-warp buttons (mockup): Pause, then the five `uni_sim::time::WARP_PRESETS`.
pub const WARP_BUTTONS: [&str; 6] = ["Pause", "1×", "10³", "10⁶", "10⁹", "10¹²"];
/// One explanatory line per warp button (mockup). 10³ s ≈ 17 min, 10⁶ s ≈ 11.6 d,
/// 10⁹ s ≈ 31.7 yr, 10¹² s ≈ 31 700 yr (Julian years of 365.25 d).
pub const WARP_NOTES: [&str; 6] = [
    "Time is paused. Flight still works.",
    "Real time.",
    "One second of yours is about 17 minutes of the universe. Planets visibly move.",
    "One second is about 12 days. Moons race around planets.",
    "One second is about 32 years. Comets swing through the inner Solar System.",
    "One second is about 32 000 years. Nearby stars visibly drift; stars do not evolve in \
     this version.",
];

// ---- Fonts ----

/// The light, regular, medium and semibold cuts of Inter, and the matching IBM Plex Mono
/// families (egui has no variable weights: each weight is its own family).
pub const FONT_TEXT_LIGHT: &str = "Inter Light";
pub const FONT_TEXT_MEDIUM: &str = "Inter Medium";
pub const FONT_TEXT_SEMIBOLD: &str = "Inter SemiBold";
pub const FONT_MONO_LIGHT: &str = "IBM Plex Mono Light";
pub const FONT_MONO_MEDIUM: &str = "IBM Plex Mono Medium";

/// A font weight of the mockup (CSS 300, 400, 500, 600).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    Light,
    Regular,
    Medium,
    SemiBold,
}

/// Inter at `size` [pt] and weight `w`.
pub fn text_font(size: f32, w: Weight) -> egui::FontId {
    let family = match w {
        Weight::Light => FONT_TEXT_LIGHT,
        Weight::Regular => FONT_TEXT,
        Weight::Medium => FONT_TEXT_MEDIUM,
        Weight::SemiBold => FONT_TEXT_SEMIBOLD,
    };
    egui::FontId::new(size, egui::FontFamily::Name(family.into()))
}

/// IBM Plex Mono at `size` [pt] and weight `w` (Plex Mono is baked without 600: SemiBold is
/// Medium).
pub fn mono_font(size: f32, w: Weight) -> egui::FontId {
    let family = match w {
        Weight::Light => FONT_MONO_LIGHT,
        Weight::Regular => FONT_MONO,
        Weight::Medium | Weight::SemiBold => FONT_MONO_MEDIUM,
    };
    egui::FontId::new(size, egui::FontFamily::Name(family.into()))
}

/// The embedded font files: (font-data key, family, file). Inter 4.1 and IBM Plex Mono 2.5.0,
/// SIL OFL 1.1 (`data/fonts/OFL-*.txt`; manifest `fonts-inter`, `fonts-ibm-plex-mono`; A9).
const FONT_FILES: [(&str, &str, &[u8]); 7] = [
    (
        "Inter-Regular",
        FONT_TEXT,
        include_bytes!("../../../data/fonts/Inter-Regular.ttf"),
    ),
    (
        "Inter-Light",
        FONT_TEXT_LIGHT,
        include_bytes!("../../../data/fonts/Inter-Light.ttf"),
    ),
    (
        "Inter-Medium",
        FONT_TEXT_MEDIUM,
        include_bytes!("../../../data/fonts/Inter-Medium.ttf"),
    ),
    (
        "Inter-SemiBold",
        FONT_TEXT_SEMIBOLD,
        include_bytes!("../../../data/fonts/Inter-SemiBold.ttf"),
    ),
    (
        "IBMPlexMono-Regular",
        FONT_MONO,
        include_bytes!("../../../data/fonts/IBMPlexMono-Regular.ttf"),
    ),
    (
        "IBMPlexMono-Light",
        FONT_MONO_LIGHT,
        include_bytes!("../../../data/fonts/IBMPlexMono-Light.ttf"),
    ),
    (
        "IBMPlexMono-Medium",
        FONT_MONO_MEDIUM,
        include_bytes!("../../../data/fonts/IBMPlexMono-Medium.ttf"),
    ),
];

/// The HUD's fonts: Inter and IBM Plex Mono from `data/fonts` (A9; SIL OFL 1.1), embedded.
pub fn font_definitions() -> egui::FontDefinitions {
    use std::sync::Arc;

    use egui::{FontData, FontFamily};

    let mut defs = egui::FontDefinitions::default();
    // egui's own fonts stay as the last fallbacks (its icons, and any glyph ours lack).
    let egui_fallbacks = defs
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    for (key, _, bytes) in FONT_FILES {
        defs.font_data
            .insert(key.to_owned(), Arc::new(FontData::from_static(bytes)));
    }
    for (key, family, _) in FONT_FILES {
        let mut chain = vec![key.to_owned()];
        // Plex Mono has no superscript minus and no Greek: Inter of the same weight fills them.
        if family.starts_with(FONT_MONO) {
            let weight = key.trim_start_matches("IBMPlexMono-");
            chain.push(format!("Inter-{weight}"));
        }
        chain.extend(egui_fallbacks.iter().cloned());
        defs.families.insert(FontFamily::Name(family.into()), chain);
    }
    for (family, key) in [
        (FontFamily::Proportional, "Inter-Regular"),
        (FontFamily::Monospace, "IBMPlexMono-Regular"),
    ] {
        let chain = defs.families.entry(family).or_default();
        chain.insert(0, key.to_owned());
        if key.starts_with("IBMPlexMono") {
            chain.insert(1, "Inter-Regular".to_owned());
        }
    }
    defs
}

/// The short form of a label for the collapsed honesty line: the text before its first " ("
/// (owner: "short items joined by ' · '"); the full sentence is in the pop-up.
pub fn short_label(full: &str) -> &str {
    full.split_once(" (").map_or(full, |(short, _)| short)
}

// ---- Drawing primitives (crate-internal; panels, card and HUD share them) ----

/// One run of text: `font`, `color` and letter spacing `tracking` [pt], wrapped at `wrap` [pt].
pub(crate) fn job(
    text: &str,
    font: egui::FontId,
    color: Color32,
    tracking: f32,
    wrap: f32,
) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::single_section(
        text.to_owned(),
        egui::TextFormat {
            font_id: font,
            color,
            extra_letter_spacing: tracking,
            ..Default::default()
        },
    );
    job.wrap.max_width = wrap;
    job
}

/// Lays out and paints `text` at the cursor, wrapped at the available width; not interactive.
pub(crate) fn text(ui: &mut egui::Ui, text: &str, font: egui::FontId, color: Color32) {
    text_tracked(ui, text, font, color, 0.0);
}

/// [`text`] with letter spacing `tracking` [pt].
pub(crate) fn text_tracked(
    ui: &mut egui::Ui,
    text: &str,
    font: egui::FontId,
    color: Color32,
    tracking: f32,
) {
    let galley = ui
        .painter()
        .layout_job(job(text, font, color, tracking, ui.available_width()));
    let (rect, _) = ui.allocate_exact_size(galley.size(), egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        ui.painter().galley(rect.min, galley, color);
    }
}

/// `text` centred in the available width.
pub(crate) fn text_centered(ui: &mut egui::Ui, text: &str, font: egui::FontId, color: Color32) {
    let width = ui.available_width();
    let galley = ui
        .painter()
        .layout_job(job(text, font.clone(), color, 0.0, width));
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(width, galley.size().y), egui::Sense::hover());
    // Centre each wrapped row, not just the block.
    let galley = if galley.rows.len() > 1 {
        let mut centred = job(text, font, color, 0.0, width);
        centred.halign = egui::Align::Center;
        ui.painter().layout_job(centred)
    } else {
        galley
    };
    let x = if galley.rows.len() > 1 {
        rect.center().x
    } else {
        rect.center().x - galley.size().x / 2.0
    };
    if ui.is_rect_visible(rect) {
        ui.painter()
            .galley(egui::pos2(x, rect.top()), galley, color);
    }
}

/// A section kicker: 11 pt, uppercase, letter-spaced `tracking_em`, weight `w`.
pub(crate) fn kicker(ui: &mut egui::Ui, text: &str, color: Color32, w: Weight, tracking_em: f32) {
    text_tracked(
        ui,
        &text.to_uppercase(),
        text_font(KICKER_SIZE, w),
        color,
        tracking_em * KICKER_SIZE,
    );
}

/// A label–value row (mockup): the label 13 pt [`INK_2`] on the left, the value right-aligned
/// in [`INK`] (or `value_color`), mono when `mono`. A value too wide to share the line goes on
/// its own line under the label.
pub(crate) fn row(ui: &mut egui::Ui, label: &str, value: &str, mono: bool, value_color: Color32) {
    /// Space between label and value [pt] (mockup grid gap 12–16).
    const GAP: f32 = 12.0;
    let width = ui.available_width();
    let value_font = if mono {
        mono_font(BODY_SIZE, Weight::Regular)
    } else {
        text_font(BODY_SIZE, Weight::Regular)
    };
    let label_font = text_font(BODY_SIZE, Weight::Regular);
    let value_galley = ui.painter().layout_job(job(
        value,
        value_font.clone(),
        value_color,
        0.0,
        f32::INFINITY,
    ));
    let label_min = (0.3 * width).min(110.0);
    if value_galley.size().x + GAP + label_min <= width {
        let label_galley = ui.painter().layout_job(job(
            label,
            label_font,
            INK_2,
            0.0,
            width - value_galley.size().x - GAP,
        ));
        let height = label_galley.size().y.max(value_galley.size().y);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
        if ui.is_rect_visible(rect) {
            ui.painter().galley(rect.min, label_galley, INK_2);
            let x = rect.right() - value_galley.size().x;
            ui.painter()
                .galley(egui::pos2(x, rect.top()), value_galley, value_color);
        }
    } else {
        text(ui, label, label_font, INK_2);
        text(ui, value, value_font, value_color);
    }
}

/// How a button looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ButtonKind {
    /// Navigation ("Go to · F"): cyan border and fill.
    Nav,
    /// Every other card button: a grey border.
    Neutral,
    /// A time-warp button: cyan when `selected`.
    Warp { selected: bool },
}

/// A touch-sized button (≥ [`BUTTON_MIN_HEIGHT`]) `width` wide (or its text plus 16 pt each
/// side), its label in `font`.
pub(crate) fn button(
    ui: &mut egui::Ui,
    label: &str,
    font: egui::FontId,
    width: Option<f32>,
    kind: ButtonKind,
) -> egui::Response {
    let galley = ui
        .painter()
        .layout_job(job(label, font, INK, 0.0, f32::INFINITY));
    let width = width.unwrap_or(galley.size().x + 32.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, BUTTON_MIN_HEIGHT), egui::Sense::click());
    let (fill, border) = match kind {
        ButtonKind::Nav => (NAV_BUTTON_FILL, NAV_BUTTON_BORDER),
        ButtonKind::Neutral => (Color32::TRANSPARENT, BUTTON_BORDER),
        ButtonKind::Warp { selected: true } => (WARP_SELECTED_FILL, WARP_SELECTED_BORDER),
        ButtonKind::Warp { selected: false } => (Color32::TRANSPARENT, WARP_BORDER),
    };
    // Hover lays a faint white over the fill (preview `--glass-hover`).
    let fill = if response.hovered() {
        fill.blend(Color32::from_white_alpha(alpha(0.06)))
    } else {
        fill
    };
    if !ui.is_rect_visible(rect) {
        return response;
    }
    ui.painter().rect(
        rect,
        BUTTON_RADIUS,
        fill,
        egui::Stroke::new(BORDER_PT, border),
        egui::StrokeKind::Inside,
    );
    let pos = rect.center() - galley.size() / 2.0;
    ui.painter().galley(pos, galley, INK);
    response
}

/// A horizontal hairline across the available width.
pub(crate) fn hairline(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), BORDER_PT),
        egui::Sense::hover(),
    );
    ui.painter().rect_filled(rect, 0.0, color);
}

/// A glass panel (mockup): [`GLASS`] fill, a 1 pt `border`, radius 10, padding 14 × 16.
pub(crate) fn panel_frame(border: Color32) -> egui::Frame {
    egui::Frame::new()
        .fill(GLASS)
        .stroke(egui::Stroke::new(BORDER_PT, border))
        .corner_radius(PANEL_RADIUS)
        .inner_margin(egui::Margin::symmetric(
            PANEL_PADDING_X as i8,
            PANEL_PADDING_Y as i8,
        ))
}

/// The target card's glass: [`CARD_GLASS`], a [`TARGET_LINE`] border, radius 12, padding
/// 20 × 22.
pub(crate) fn card_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD_GLASS)
        .stroke(egui::Stroke::new(BORDER_PT, TARGET_LINE))
        .corner_radius(CARD_RADIUS)
        .inner_margin(egui::Margin::symmetric(
            CARD_PADDING_X as i8,
            CARD_PADDING_Y as i8,
        ))
}

/// A soft dark halo behind text that has no panel (the preview's text-shadow): a blurred
/// rectangle, 40 % black at its core.
pub(crate) fn halo_shape(rect: egui::Rect) -> egui::Shape {
    let shadow = egui::Shadow {
        offset: [0, 1],
        blur: 14,
        spread: 0,
        color: Color32::from_black_alpha(alpha(0.40)),
    };
    shadow.as_shape(rect, 8.0).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::FontFamily;

    fn rgba(r: u8, g: u8, b: u8, a: f32) -> Color32 {
        Color32::from_rgba_unmultiplied(r, g, b, (a * 255.0).round() as u8)
    }

    /// Plan C4 "The design": every colour, alpha, radius, padding, width and type size is the
    /// mockup's (CSS values; egui points equal CSS px), the tokens the preview's.
    #[test]
    fn theme_matches_mockup() {
        // Colour carries meaning only.
        assert_eq!(TARGET, Color32::from_rgb(0x7D, 0xD3, 0xFC));
        assert_eq!(NON_PHYSICAL, Color32::from_rgb(0xF5, 0xB0, 0x4C));
        assert_eq!(DANGER, Color32::from_rgb(0xF8, 0x71, 0x71));
        assert_eq!(ALERT, Color32::from_rgb(0xFC, 0xA5, 0xA5));
        // Section kickers #8D97A6, labels #9AA4B2, values #E6EAF0.
        assert_eq!(INK_3, Color32::from_rgb(0x8D, 0x97, 0xA6));
        assert_eq!(INK_2, Color32::from_rgb(0x9A, 0xA4, 0xB2));
        assert_eq!(INK, Color32::from_rgb(0xE6, 0xEA, 0xF0));
        assert_eq!(MUTED, INK_3);
        assert_eq!(INK_SOFT, rgba(214, 222, 234, 0.62));
        // Glass: panels rgba(10,14,22,0.55) with a 1 pt rgba(255,255,255,0.10) border, radius
        // 10, padding 14 × 16; the card 0.62, border rgba(125,211,252,0.35), radius 12,
        // padding 20 × 22.
        assert_eq!(GLASS, rgba(10, 14, 22, 0.55));
        assert_eq!(HAIRLINE, rgba(255, 255, 255, 0.10));
        assert_eq!(BORDER_PT, 1.0);
        assert_eq!(
            (PANEL_RADIUS, PANEL_PADDING_Y, PANEL_PADDING_X),
            (10.0, 14.0, 16.0)
        );
        assert_eq!(CARD_GLASS, rgba(10, 14, 22, 0.62));
        assert_eq!(TARGET_LINE, rgba(125, 211, 252, 0.35));
        assert_eq!(
            (CARD_RADIUS, CARD_PADDING_Y, CARD_PADDING_X),
            (12.0, 20.0, 22.0)
        );
        assert_eq!(CARD_HAIRLINE, rgba(255, 255, 255, 0.08));
        // Cruise: the Speed panel's border rgba(245,176,76,0.40).
        assert_eq!(CRUISE_BORDER, rgba(245, 176, 76, 0.40));
        assert_eq!(NON_PHYSICAL_SOFT, rgba(245, 176, 76, 0.80));
        // Buttons and the speed bar.
        assert_eq!(BUTTON_MIN_HEIGHT, 44.0);
        assert_eq!(BUTTON_RADIUS, 8.0);
        assert_eq!(NAV_BUTTON_FILL, rgba(125, 211, 252, 0.12));
        assert_eq!(NAV_BUTTON_BORDER, rgba(125, 211, 252, 0.60));
        assert_eq!(BUTTON_BORDER, rgba(255, 255, 255, 0.16));
        assert_eq!(WARP_SELECTED_FILL, rgba(125, 211, 252, 0.14));
        assert_eq!(WARP_SELECTED_BORDER, rgba(125, 211, 252, 0.70));
        assert_eq!(WARP_BORDER, rgba(255, 255, 255, 0.14));
        assert_eq!(SPEED_BAR_TRACK, rgba(255, 255, 255, 0.08));
        assert_eq!(SPEED_BAR_FILL, rgba(230, 234, 240, 0.55));
        assert_eq!((SPEED_BAR_HEIGHT, SPEED_TICK_HEIGHT), (6.0, 12.0));
        assert_eq!((SPEED_BAR_MIN_MPS, SPEED_BAR_MAX_C), (0.1, 1.0e18));
        // Layout: 24 pt margins; widths Location 300, Time 280 (380 with warp), Gravity 260,
        // Speed 520, card 420.
        assert_eq!(SCREEN_MARGIN, 24.0);
        assert_eq!(
            [
                LOCATION_WIDTH,
                TIME_WIDTH,
                TIME_WARP_WIDTH,
                GRAVITY_WIDTH,
                SPEED_WIDTH,
                CARD_WIDTH
            ],
            [300.0, 280.0, 380.0, 260.0, 520.0, 420.0]
        );
        // Type: kickers 11 pt with 0.12 em (the card's 0.14 em), body 13, region 18, epoch 15,
        // speed 40, title 30, notes 12, small 11.
        assert_eq!((KICKER_SIZE, KICKER_TRACKING_EM), (11.0, 0.12));
        assert_eq!(CARD_KICKER_TRACKING_EM, 0.14);
        assert_eq!(
            [
                BODY_SIZE,
                REGION_SIZE,
                EPOCH_SIZE,
                SPEED_SIZE,
                TITLE_SIZE,
                NOTES_SIZE,
                SMALL_SIZE
            ],
            [13.0, 18.0, 15.0, 40.0, 30.0, 12.0, 11.0]
        );
        assert_eq!((FONT_TEXT, FONT_MONO), ("Inter", "IBM Plex Mono"));
        // Tones.
        assert_eq!(tone_color(LabelTone::Neutral), MUTED);
        assert_eq!(tone_color(LabelTone::Nav), TARGET);
        assert_eq!(tone_color(LabelTone::NonPhysical), NON_PHYSICAL);
        assert_eq!(tone_color(LabelTone::Danger), DANGER);
    }

    /// Inter (300/400/500/600) and IBM Plex Mono (300/400/500) are embedded and are egui's
    /// proportional and monospace faces.
    #[test]
    fn fonts_load() {
        let defs = font_definitions();
        for family in [
            FONT_TEXT,
            FONT_TEXT_LIGHT,
            FONT_TEXT_MEDIUM,
            FONT_TEXT_SEMIBOLD,
            FONT_MONO,
            FONT_MONO_LIGHT,
            FONT_MONO_MEDIUM,
        ] {
            let fonts = defs
                .families
                .get(&FontFamily::Name(family.into()))
                .unwrap_or_else(|| panic!("no family {family:?}"));
            let first = defs.font_data.get(&fonts[0]).expect("font data");
            // A TrueType file starts with the sfnt version 0x00010000.
            assert_eq!(first.bytes()[..4], [0, 1, 0, 0], "{family}");
        }
        assert_eq!(defs.families[&FontFamily::Proportional][0], "Inter-Regular");
        assert_eq!(
            defs.families[&FontFamily::Monospace][0],
            "IBMPlexMono-Regular"
        );
        assert_eq!(
            text_font(13.0, Weight::Medium).family,
            FontFamily::Name(FONT_TEXT_MEDIUM.into())
        );
        assert_eq!(
            mono_font(40.0, Weight::Light).family,
            FontFamily::Name(FONT_MONO_LIGHT.into())
        );
        assert_eq!(
            mono_font(11.0, Weight::SemiBold).family,
            FontFamily::Name(FONT_MONO_MEDIUM.into())
        );
        assert_eq!(text_font(30.0, Weight::Light).size, 30.0);
    }

    /// Every non-ASCII character the HUD writes has a glyph in every family's fallback chain
    /// (Plex Mono has no ⁻ or Greek: Inter is its first fallback; neither font has ☉, so the
    /// cards spell out "solar masses").
    #[test]
    fn hud_glyphs_exist() {
        let ctx = egui::Context::default();
        ctx.set_fonts(font_definitions());
        let mut out = ctx.run_ui(egui::RawInput::default(), |_| {});
        out.textures_delta.clear();
        let glyphs = "×⁰¹²³⁴⁵⁶⁷⁸⁹⁻−·°→–…≈±≥Ωωαβγδ₀₁₂₃₄₅₆₇₈₉";
        for family in [
            FONT_TEXT,
            FONT_TEXT_LIGHT,
            FONT_TEXT_MEDIUM,
            FONT_TEXT_SEMIBOLD,
            FONT_MONO,
            FONT_MONO_LIGHT,
            FONT_MONO_MEDIUM,
        ] {
            let font = egui::FontId::new(13.0, FontFamily::Name(family.into()));
            let missing: String =
                ctx.fonts_mut(|f| glyphs.chars().filter(|&c| !f.has_glyph(&font, c)).collect());
            assert!(missing.is_empty(), "{family}: no glyph for {missing:?}");
        }
    }

    /// The collapsed honesty line's items: the text before " (", the flight labels whole.
    #[test]
    fn short_labels() {
        assert_eq!(short_label(EYE_ADAPTED_STARS_LABEL), "Eye-adapted stars");
        assert_eq!(
            short_label(BELT_HIGHLIGHT_LABEL),
            "Asteroid belt brightness exaggerated"
        );
        assert_eq!(short_label(FREE_FLIGHT_LABEL), FREE_FLIGHT_LABEL);
        assert_eq!(short_label(CRUISE_LABEL), CRUISE_LABEL);
        assert_eq!(short_label(DEBUG_GRID_LABEL), DEBUG_GRID_LABEL);
    }

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
