//! The HUD's panels: what the app fills for each (Task C4; Task C0 wrote the types).
//!
//! Collapsed by default (owner feedback 2026-11-22; lead ruling on the C0 context, owner point
//! 4): a muted place line with an "i" button at the top left ([`NavLine`]), the muted notes line
//! at the bottom left (which holds the flight label, ruling 6), one mono time line at the top
//! centre (ruling 15), the perf line at the bottom right. The Location, Gravity and Speed panels
//! show only while the card is expanded (`HudControls::card_expanded`).

use uni_core::{C, Epoch};

use crate::format::{
    format_c_fraction, format_distance, format_duration, format_epoch, format_sig, format_speed,
    format_speed_value, format_time_rate,
};
use crate::theme::{
    self, BODY_SIZE, CRUISE_BORDER, CRUISE_LABEL, EPOCH_SIZE, GRAVITY_WIDTH, HAIRLINE, INK, INK_2,
    INK_3, INK_SOFT, KICKER_TRACKING_EM, LOCATION_WIDTH, LabelTone, NON_PHYSICAL, REGION_SIZE,
    SMALL_SIZE, SPEED_BAR_FILL, SPEED_BAR_HEIGHT, SPEED_BAR_MAX_C, SPEED_BAR_MIN_MPS,
    SPEED_BAR_TRACK, SPEED_SIZE, SPEED_TICK_HEIGHT, SPEED_WIDTH, TARGET, TIME_WARP_WIDTH,
    TIME_WIDTH, WARP_BUTTONS, WARP_NOTES, Weight, mono_font, text_font, tone_color,
};

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

/// Where `mps` sits on the Speed panel's log bar, 0 ..= 1: 0.1 m/s at 0, 10¹⁸ c at 1
/// (`theme::SPEED_BAR_MIN_MPS`, `theme::SPEED_BAR_MAX_C`).
pub fn speed_bar_frac(mps: f64) -> f64 {
    let lo = SPEED_BAR_MIN_MPS.log10();
    let hi = (SPEED_BAR_MAX_C * C).log10();
    if mps.is_nan() || mps <= SPEED_BAR_MIN_MPS {
        return 0.0; // At rest, below the bar, or NaN.
    }
    ((mps.log10() - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// The place line's text: a cyan part, the rest of the first line, and the muted sub-line.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NavText {
    pub cyan: String,
    pub rest: String,
    pub sub: String,
}

/// The strings of `n` (the preview's place line, `74-hud.js`).
pub(crate) fn nav_text(n: &NavLine) -> NavText {
    match n {
        NavLine::Here { name, dist_m } => NavText {
            cyan: String::new(),
            rest: name.clone(),
            sub: format!("{} away", format_distance(*dist_m)),
        },
        NavLine::Selected {
            name,
            dist_m,
            closing_mps,
        } => {
            let way = if *closing_mps < 0.0 {
                "receding"
            } else {
                "closing"
            };
            NavText {
                cyan: name.clone(),
                rest: String::new(),
                sub: format!(
                    "{} · {way} {}",
                    format_distance(*dist_m),
                    format_speed(closing_mps.abs())
                ),
            }
        }
        NavLine::Aiming { name, dist_m, .. } => NavText {
            cyan: format!("→ {name}"),
            rest: String::new(),
            sub: format!(
                "{} · hold to fly there, or click it",
                format_distance(*dist_m)
            ),
        },
        NavLine::EnRoute { name, to_go_m } => NavText {
            cyan: "En route".into(),
            rest: format!(" · {name}"),
            sub: format!(
                "{} to go · travel time compressed",
                format_distance(to_go_m.max(0.0))
            ),
        },
    }
}

/// The collapsed HUD's one-line time display (ruling 15): "2026-11-20 04:12:09 UTC · × 1 000".
pub(crate) fn time_line(t: &TimePanel) -> String {
    format!("{} · {}", format_epoch(t.epoch), rate_text(t))
}

/// The rate as the Time panel shows it: "× 1 000", or "paused · × 1 000".
fn rate_text(t: &TimePanel) -> String {
    if t.paused {
        format!("paused · {}", format_time_rate(t.rate))
    } else {
        format_time_rate(t.rate)
    }
}

/// The highlighted warp button: 0 (Pause) while paused, else 1 + the index of the preset equal
/// to the rate (`uni_sim::time::WARP_PRESETS`), `None` for any other rate.
pub(crate) fn warp_selected(t: &TimePanel) -> Option<usize> {
    if t.paused {
        return Some(0);
    }
    uni_sim::time::WARP_PRESETS
        .iter()
        .position(|&p| (t.rate - p).abs() <= 1e-9 * p)
        .map(|i| i + 1)
}

/// A press on a warp button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WarpPick {
    Pause,
    /// An index into `uni_sim::time::WARP_PRESETS`.
    Preset(usize),
}

/// The content width of a panel `outer` points wide with `pad_x` padding and a 1 pt border
/// (the mockup's CSS `box-sizing: border-box`).
pub(crate) fn inner_width(outer: f32, pad_x: f32) -> f32 {
    outer - 2.0 * pad_x - 2.0 * theme::BORDER_PT
}

/// The "i" button (preview `.info-btn`): a 26 pt ring with an italic i, its ring and fill
/// stronger while the card is open.
pub(crate) fn info_button(ui: &mut egui::Ui, open: bool) -> egui::Response {
    let size = egui::vec2(theme::INFO_BUTTON_SIZE, theme::INFO_BUTTON_SIZE);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let lit = open || response.hovered();
    let (fill, ring, ink) = if lit {
        (theme::GLASS, theme::TARGET_LINE, INK)
    } else {
        (theme::INFO_FILL, theme::HAIRLINE_STRONG, INK_2)
    };
    let painter = ui.painter();
    painter.circle(
        rect.center(),
        theme::INFO_BUTTON_SIZE / 2.0 - 0.5,
        fill,
        egui::Stroke::new(theme::BORDER_PT, ring),
    );
    let mut job = theme::job(
        "i",
        text_font(BODY_SIZE, Weight::Medium),
        ink,
        0.0,
        f32::INFINITY,
    );
    if let Some(section) = job.sections.first_mut() {
        section.format.italics = true;
    }
    let galley = painter.layout_job(job);
    painter.galley(rect.center() - galley.size() / 2.0, galley, ink);
    response
}

/// The collapsed place line (top left): the "i" button and the muted place text. Returns the
/// response of the whole line; a click on it opens the card (owner: "a click … expands").
pub(crate) fn show_place(ui: &mut egui::Ui, n: &NavLine, open: bool) -> egui::Response {
    let text = nav_text(n);
    let inner = ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(10.0, 1.0);
        let button = info_button(ui, open);
        let column = ui.vertical(|ui| {
            // The name: 13 pt, weight 500, ink-soft, its navigation part cyan.
            let font = text_font(BODY_SIZE, Weight::Medium);
            let mut job = egui::text::LayoutJob::default();
            if !text.cyan.is_empty() {
                job.append(
                    &text.cyan,
                    0.0,
                    egui::TextFormat::simple(font.clone(), TARGET),
                );
            }
            if !text.rest.is_empty() {
                job.append(&text.rest, 0.0, egui::TextFormat::simple(font, INK_SOFT));
            }
            let galley = ui.painter().layout_job(job);
            let (rect, _) = ui.allocate_exact_size(galley.size(), egui::Sense::hover());
            ui.painter().galley(rect.min, galley, INK_SOFT);
            // The sub-line: 11 pt mono, ink-3.
            theme::text(ui, &text.sub, mono_font(SMALL_SIZE, Weight::Regular), INK_3);
        });
        button.union(column.response)
    });
    let rect = inner.response.rect;
    let line = ui.interact(rect, ui.scope_id().with("place line"), egui::Sense::click());
    inner.inner.union(line)
}

/// The Location panel (top left, 300 pt): kicker LOCATION with the open "i", the region, then
/// From Sun, Light time and Frame. Returns true when its "i" was clicked (close the card).
pub(crate) fn show_location(ui: &mut egui::Ui, p: &LocationPanel) -> bool {
    let mut close = false;
    theme::panel_frame(HAIRLINE).show(ui, |ui| {
        ui.set_width(inner_width(LOCATION_WIDTH, theme::PANEL_PADDING_X));
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
        ui.horizontal(|ui| {
            theme::kicker(ui, "Location", INK_3, Weight::Medium, KICKER_TRACKING_EM);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                close = info_button(ui, true).clicked();
            });
        });
        ui.add_space(2.0);
        theme::text(ui, &p.region, text_font(REGION_SIZE, Weight::Regular), INK);
        ui.add_space(8.0);
        let rows = [
            ("From Sun", format_distance(p.from_sun_m), true),
            ("Light time", format_duration(p.light_time_s), true),
            // Body frames translate with their body and keep ICRS axes (uni-sim): non-rotating,
            // but not inertial (they accelerate with the body).
            ("Frame", format!("{} · non-rotating", p.frame), false),
        ];
        for (i, (label, value, mono)) in rows.iter().enumerate() {
            if i > 0 {
                ui.add_space(4.0);
            }
            theme::row(ui, label, value, *mono, INK);
        }
    });
    close
}

/// The Time panel (top right, 280 pt): kicker TIME, the epoch in mono 15 pt, the Rate row.
pub(crate) fn show_time(ui: &mut egui::Ui, t: &TimePanel) {
    theme::panel_frame(HAIRLINE).show(ui, |ui| {
        ui.set_width(inner_width(TIME_WIDTH, theme::PANEL_PADDING_X));
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 6.0);
        theme::kicker(ui, "Time", INK_3, Weight::Medium, KICKER_TRACKING_EM);
        theme::text(
            ui,
            &format_epoch(t.epoch),
            mono_font(EPOCH_SIZE, Weight::Regular),
            INK,
        );
        theme::row(ui, "Rate", &rate_text(t), true, INK);
    });
}

/// The time-warp panel (T; 380 pt): kicker TIME WARP and the epoch, six buttons (the current
/// rate's cyan), the mockup's line for the current rate. Returns the button pressed.
pub(crate) fn show_warp(ui: &mut egui::Ui, t: &TimePanel) -> Option<WarpPick> {
    let mut pick = None;
    theme::panel_frame(HAIRLINE).show(ui, |ui| {
        let width = inner_width(TIME_WARP_WIDTH, theme::PANEL_PADDING_X);
        ui.set_width(width);
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 10.0);
        ui.horizontal(|ui| {
            theme::kicker(ui, "Time warp", INK_3, Weight::Medium, KICKER_TRACKING_EM);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let epoch = format_epoch(t.epoch);
                theme::text(ui, &epoch, mono_font(BODY_SIZE, Weight::Regular), INK);
            });
        });
        let selected = warp_selected(t);
        ui.horizontal(|ui| {
            let gap = ui.spacing().item_spacing.x;
            let button_w = (width - 5.0 * gap) / 6.0;
            for (i, label) in WARP_BUTTONS.iter().enumerate() {
                let kind = theme::ButtonKind::Warp {
                    selected: selected == Some(i),
                };
                let font = mono_font(BODY_SIZE, Weight::Medium);
                if theme::button(ui, label, font, Some(button_w), kind).clicked() {
                    pick = Some(if i == 0 {
                        WarpPick::Pause
                    } else {
                        WarpPick::Preset(i - 1)
                    });
                }
            }
        });
        let note = match selected {
            Some(i) => WARP_NOTES[i].to_owned(),
            None => format!(
                "One second is about {} of the universe.",
                format_duration(t.rate.abs())
            ),
        };
        theme::text(
            ui,
            &note,
            text_font(theme::NOTES_SIZE, Weight::Regular),
            INK_2,
        );
    });
    pick
}

/// The Gravity panel (bottom left, 260 pt): the amber kicker `GRAVITY · <BODY> · NOT APPLIED`
/// (ruling 6), then Local g, Escape v and Circular v.
pub(crate) fn show_gravity(ui: &mut egui::Ui, g: &GravityPanel) {
    theme::panel_frame(HAIRLINE).show(ui, |ui| {
        ui.set_width(inner_width(GRAVITY_WIDTH, theme::PANEL_PADDING_X));
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
        theme::kicker(
            ui,
            &theme::gravity_kicker(&g.body),
            NON_PHYSICAL,
            Weight::Medium,
            KICKER_TRACKING_EM,
        );
        ui.add_space(4.0);
        theme::row(
            ui,
            "Local g",
            &format!("{} m/s²", format_sig(g.local_g, 3)),
            true,
            INK,
        );
        theme::row(ui, "Escape v", &format_speed(g.escape_mps), true, INK);
        theme::row(ui, "Circular v", &format_speed(g.circular_mps), true, INK);
    });
}

/// The Speed panel (bottom centre, 520 pt): the header in its tone, the speed in mono 40 pt
/// light, the sub-line, and the log bar from 0.1 m/s to 10¹⁸ c with an amber tick at c. In
/// cruise the border is amber and the sub-line gives the distance per hour and the note.
pub(crate) fn show_speed(ui: &mut egui::Ui, s: &SpeedPanel) {
    let cruise = s.header == CRUISE_LABEL;
    let border = if cruise { CRUISE_BORDER } else { HAIRLINE };
    // Mockup padding: 14 top, 20 sides, 16 bottom.
    let frame = theme::panel_frame(border).inner_margin(egui::Margin {
        left: 20,
        right: 20,
        top: 14,
        bottom: 16,
    });
    frame.show(ui, |ui| {
        let width = inner_width(SPEED_WIDTH, 20.0);
        ui.set_width(width);
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 8.0);
        let header = tone_color(s.tone);
        let weight = if cruise {
            Weight::SemiBold
        } else {
            Weight::Medium
        };
        centered_kicker(ui, &s.header, header, weight);
        theme::text_centered(
            ui,
            &format_speed_value(s.speed_mps),
            mono_font(SPEED_SIZE, Weight::Light),
            INK,
        );
        let ftl = s.speed_mps > C;
        let sub_font = mono_font(theme::NOTES_SIZE, Weight::Regular);
        if cruise {
            let per_hour = format!("{} per hour", format_distance(s.speed_mps * 3600.0));
            let sub = match &s.note {
                Some(note) => format!("{per_hour} · {note}"),
                None => per_hour,
            };
            theme::text_centered(ui, &sub, sub_font, INK_2);
        } else if ftl && !s.header.contains("FTL") {
            // Free flight above c: the M0 rule "speeds faster than c are labelled".
            theme::text_centered(ui, "FTL · non-physical", sub_font.clone(), NON_PHYSICAL);
            theme::text_centered(ui, &format_c_fraction(s.speed_mps), sub_font, INK_2);
        } else {
            theme::text_centered(ui, &format_c_fraction(s.speed_mps), sub_font, INK_2);
        }
        ui.add_space(4.0);
        speed_bar(ui, s.speed_mps);
    });
}

/// A kicker centred in the available width.
fn centered_kicker(ui: &mut egui::Ui, text: &str, color: egui::Color32, w: Weight) {
    let width = ui.available_width();
    let galley = ui.painter().layout_job(theme::job(
        &text.to_uppercase(),
        text_font(theme::KICKER_SIZE, w),
        color,
        KICKER_TRACKING_EM * theme::KICKER_SIZE,
        width,
    ));
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(width, galley.size().y), egui::Sense::hover());
    let x = rect.center().x - galley.size().x / 2.0;
    ui.painter()
        .galley(egui::pos2(x, rect.top()), galley, color);
}

/// The log speed bar and its scale: 0.1 m/s, 1 km/s, c (amber), 10⁶ c, 10¹⁸ c, each at its
/// true place on the bar (the mockup spaces them evenly; the tick at c is at 0.3449).
fn speed_bar(ui: &mut egui::Ui, mps: f64) {
    let width = ui.available_width();
    let font = mono_font(SMALL_SIZE, Weight::Regular);
    let label_h = ui.painter().fonts_mut(|f| f.row_height(&font));
    let height = SPEED_TICK_HEIGHT + 4.0 + label_h;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter();
    let top = rect.top() + (SPEED_TICK_HEIGHT - SPEED_BAR_HEIGHT) / 2.0;
    let track = egui::Rect::from_min_size(
        egui::pos2(rect.left(), top),
        egui::vec2(width, SPEED_BAR_HEIGHT),
    );
    let radius = SPEED_BAR_HEIGHT / 2.0;
    painter.rect_filled(track, radius, SPEED_BAR_TRACK);
    let fill_w = (speed_bar_frac(mps) as f32 * width).max(0.0);
    if fill_w > 0.0 {
        let fill = egui::Rect::from_min_size(track.min, egui::vec2(fill_w, SPEED_BAR_HEIGHT));
        painter.rect_filled(fill, radius, SPEED_BAR_FILL);
    }
    let x_of = |v: f64| rect.left() + speed_bar_frac(v) as f32 * width;
    let tick = egui::Rect::from_min_size(
        egui::pos2(x_of(C) - 0.5, rect.top()),
        egui::vec2(theme::BORDER_PT, SPEED_TICK_HEIGHT),
    );
    painter.rect_filled(tick, 0.0, NON_PHYSICAL);
    let y = rect.top() + SPEED_TICK_HEIGHT + 4.0;
    let marks = [
        ("0.1 m/s", SPEED_BAR_MIN_MPS, egui::Align::Min, INK_3),
        ("1 km/s", 1.0e3, egui::Align::Center, INK_3),
        ("c", C, egui::Align::Center, NON_PHYSICAL),
        ("10⁶ c", 1.0e6 * C, egui::Align::Center, INK_3),
        ("10¹⁸ c", SPEED_BAR_MAX_C * C, egui::Align::Max, INK_3),
    ];
    /// Least space between two scale labels [pt]: "1 km/s" sits only 70 pt from 0.1 m/s.
    const LABEL_GAP: f32 = 12.0;
    let mut prev_right = f32::NEG_INFINITY;
    for (label, v, align, color) in marks {
        let galley = painter.layout_job(theme::job(label, font.clone(), color, 0.0, f32::INFINITY));
        let w = galley.size().x;
        let x = match align {
            egui::Align::Min => x_of(v),
            egui::Align::Center => x_of(v) - w / 2.0,
            egui::Align::Max => x_of(v) - w,
        };
        let x = x.max(prev_right + LABEL_GAP);
        prev_right = x + w;
        painter.galley(egui::pos2(x, y), galley, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_core::C;

    /// The amber tick at c sits at (log₁₀ c + 1)/(log₁₀(10¹⁸ c) + 1) ≈ 0.3449 of the bar
    /// (the bar starts at 0.1 m/s, log₁₀ 0.1 = −1).
    #[test]
    fn speed_bar_c_tick() {
        let formula = (C.log10() + 1.0) / ((1.0e18 * C).log10() + 1.0);
        assert!(
            (speed_bar_frac(C) - formula).abs() < 1e-6,
            "{}",
            speed_bar_frac(C)
        );
        // 0.344902…: the brief's 0.3449 to its four decimals.
        assert!(
            (speed_bar_frac(C) - 0.3449).abs() < 5e-5,
            "{}",
            speed_bar_frac(C)
        );
        assert_eq!(speed_bar_frac(0.1), 0.0);
        assert!((speed_bar_frac(1.0e18 * C) - 1.0).abs() < 1e-12);
        // Clamped, and safe for 0 and NaN (at rest).
        for (v, f) in [(0.0, 0.0), (-5.0, 0.0), (f64::NAN, 0.0), (1.0e40, 1.0)] {
            assert_eq!(speed_bar_frac(v), f, "{v}");
        }
    }

    /// The owner-approved preview's place-line wording.
    #[test]
    fn nav_line_strings() {
        let text = |n: NavLine| nav_text(&n);
        assert_eq!(
            text(NavLine::Here {
                name: "Mars".into(),
                dist_m: 1.42e7
            }),
            NavText {
                cyan: String::new(),
                rest: "Mars".into(),
                sub: "14 200 km away".into()
            }
        );
        assert_eq!(
            text(NavLine::Selected {
                name: "Mars".into(),
                dist_m: 1.42e7,
                closing_mps: 2100.0
            }),
            NavText {
                cyan: "Mars".into(),
                rest: String::new(),
                sub: "14 200 km · closing 2.10 km/s".into()
            }
        );
        assert_eq!(
            text(NavLine::Selected {
                name: "Mars".into(),
                dist_m: 1.42e7,
                closing_mps: -2100.0
            })
            .sub,
            "14 200 km · receding 2.10 km/s"
        );
        assert_eq!(
            text(NavLine::Aiming {
                name: "Jupiter".into(),
                dist_m: 5.0 * uni_core::AU,
                fill: 0.4
            }),
            NavText {
                cyan: "→ Jupiter".into(),
                rest: String::new(),
                sub: "5.000 AU · hold to fly there, or click it".into()
            }
        );
        assert_eq!(
            text(NavLine::EnRoute {
                name: "Saturn".into(),
                to_go_m: 1.42e7
            }),
            NavText {
                cyan: "En route".into(),
                rest: " · Saturn".into(),
                sub: "14 200 km to go · travel time compressed".into()
            }
        );
    }
}
