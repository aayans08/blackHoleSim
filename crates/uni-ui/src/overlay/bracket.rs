//! The selected target's corner brackets (Task C5; Task C0 wrote the type).
//!
//! Four cyan corner brackets (`theme::BRACKET_ARM` 18 pt arms, `theme::BRACKET_STROKE` 1.5 pt),
//! half-size = max(projected radius + 12 pt, 20 pt); below them the name (12 pt, weight 600,
//! letter-spacing 0.14 em, cyan) and "distance · closing speed" (mono 13 pt). The mockup's
//! in-flight artboard sets the rest: the name in capitals ("MARS"), the label block's left edge
//! on the bracket's, 12 pt below it, the sub-line in `theme::INK` 2 pt under the name, worded
//! "14 200 km · closing 2.10 km/s" ("receding" when the distance grows).

use egui::text::{LayoutJob, TextFormat};
use egui::{FontId, Painter, Rect, Shape, Stroke};
use glam::Vec2;

use crate::format::{format_distance, format_speed};
use crate::theme;

/// The target bracket, in output points.
#[derive(Clone, Debug)]
pub struct Bracket {
    pub center_pt: Vec2,
    pub half_size_pt: f32,
    pub name: String,
    /// "distance · closing speed".
    pub sub: String,
}

/// Gap between the bracket and its label, and between the name and the sub-line [pt] (mockup).
const LABEL_GAP_PT: f32 = 12.0;
const SUB_GAP_PT: f32 = 2.0;

/// Half the bracket's side [pt] around a disk of projected radius `radius_pt`:
/// max(radius + `theme::BRACKET_PAD`, `theme::BRACKET_MIN_HALF`).
pub fn bracket_half_size(radius_pt: f32) -> f32 {
    // `f32::max` ignores a NaN radius: the smallest bracket.
    (radius_pt + theme::BRACKET_PAD).max(theme::BRACKET_MIN_HALF)
}

/// The sub-line: the distance to the centre [m] and the closing speed [m/s] (positive while the
/// distance shrinks), e.g. "1.524 AU · closing 2.10 km/s"; only the distance when the speed is
/// not finite.
pub(crate) fn bracket_sub(dist_m: f64, closing_mps: f64) -> String {
    let dist = format_distance(dist_m);
    if !closing_mps.is_finite() {
        dist
    } else if closing_mps >= 0.0 {
        format!("{dist} \u{b7} closing {}", format_speed(closing_mps))
    } else {
        format!("{dist} \u{b7} receding {}", format_speed(-closing_mps))
    }
}

/// Where the label block lies [pt], estimated without fonts (markers keep their labels off it):
/// 0.6 em per character, IBM Plex Mono's advance and above Inter's average.
pub(crate) fn label_rect(b: &Bracket) -> Rect {
    const ADVANCE_EM: f32 = 0.6;
    let name_w = b.name.chars().count() as f32
        * theme::BRACKET_NAME_SIZE
        * (ADVANCE_EM + theme::BRACKET_NAME_TRACKING_EM);
    let sub_w = b.sub.chars().count() as f32 * theme::BRACKET_SUB_SIZE * ADVANCE_EM;
    let min = egui::pos2(
        b.center_pt.x - b.half_size_pt,
        b.center_pt.y + b.half_size_pt + LABEL_GAP_PT,
    );
    // The two lines, their gap, and 4 pt for the lines' leading.
    let height = theme::BRACKET_NAME_SIZE + SUB_GAP_PT + theme::BRACKET_SUB_SIZE + 4.0;
    Rect::from_min_size(min, egui::vec2(name_w.max(sub_w), height))
}

/// Paints `b`: the four corners, then the name and the sub-line under the bracket.
pub(crate) fn draw(painter: &Painter, b: &Bracket) {
    let (c, h, arm) = (b.center_pt, b.half_size_pt, theme::BRACKET_ARM);
    let stroke = Stroke::new(theme::BRACKET_STROKE, theme::TARGET);
    for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let corner = egui::pos2(c.x + sx * h, c.y + sy * h);
        painter.add(Shape::line(
            vec![
                corner - egui::vec2(sx * arm, 0.0),
                corner,
                corner - egui::vec2(0.0, sy * arm),
            ],
            stroke,
        ));
    }
    let name = TextFormat {
        font_id: FontId::proportional(theme::BRACKET_NAME_SIZE),
        extra_letter_spacing: theme::BRACKET_NAME_TRACKING_EM * theme::BRACKET_NAME_SIZE,
        color: theme::TARGET,
        ..TextFormat::default()
    };
    let name = painter.layout_job(LayoutJob::single_section(b.name.to_uppercase(), name));
    let top = egui::pos2(c.x - h, c.y + h + LABEL_GAP_PT);
    let sub_top = top + egui::vec2(0.0, name.size().y + SUB_GAP_PT);
    let sub = painter.layout_no_wrap(
        b.sub.clone(),
        FontId::monospace(theme::BRACKET_SUB_SIZE),
        theme::INK,
    );
    super::markers::paint_text(painter, top, name, 1.0);
    super::markers::paint_text(painter, sub_top, sub, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plan C5 step 1: half-size = max(radius + 12 pt, 20 pt).
    #[test]
    fn bracket_size_from_radius() {
        assert_eq!(bracket_half_size(3.0), 20.0);
        assert_eq!(bracket_half_size(100.0), 112.0);
        assert_eq!(bracket_half_size(8.0), 20.0);
        assert_eq!(bracket_half_size(8.5), 20.5);
        assert_eq!(
            bracket_half_size(f32::NAN),
            20.0,
            "no disk size: the smallest bracket"
        );
    }

    /// The mockup's wording: "distance · closing speed", receding when the distance grows.
    #[test]
    fn sub_line_names_closing_or_receding() {
        // Built from the HUD's own formatters (C4 owns their number style, ruling 16).
        use crate::format::{format_distance, format_speed};
        let d = 1.524 * uni_core::AU;
        let (dist, speed) = (format_distance(d), format_speed(2100.0));
        assert_eq!(
            bracket_sub(d, 2100.0),
            format!("{dist} \u{b7} closing {speed}")
        );
        assert_eq!(
            bracket_sub(d, -2100.0),
            format!("{dist} \u{b7} receding {speed}")
        );
        assert_eq!(bracket_sub(d, f64::NAN), dist);
        let b = Bracket {
            center_pt: Vec2::new(500.0, 300.0),
            half_size_pt: 40.0,
            name: "Mars".into(),
            sub: "x".into(),
        };
        let r = label_rect(&b);
        assert_eq!(
            r.min,
            egui::pos2(460.0, 352.0),
            "left on the bracket's edge, 12 pt below it"
        );
        assert!(r.height() > 20.0 && r.width() > 10.0);
    }
}
