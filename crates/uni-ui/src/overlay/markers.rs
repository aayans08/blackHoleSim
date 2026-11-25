//! Destination markers (Task C5; Task C0 wrote the types).
//!
//! One marker per `Snapshot::destinations` entry (lead ruling B.3 decides which objects carry
//! one). The look is the owner-approved look preview's (lead ruling for C5;
//! `.superpowers/look-preview/v2/parts/74-hud.js:225-343`, `00-head.html:246-281`): a 20 pt
//! glyph per kind centred on the object, the name (11 pt, weight 500, letter-spacing 0.03 em)
//! 4 pt to its right and a mono distance (10 pt, `theme::INK_3`) 8 pt after the name; colour
//! `theme::INK_SOFT` at opacity 0.5, 0.32 when more than 30° off the view axis, 1 (in
//! `theme::INK`, the name cyan) when aimed at or the flight target; the selected marker is cyan
//! (plan C5). The aim ring: radius `theme::AIM_RING_RADIUS`, stroke `theme::AIM_RING_STROKE`, a
//! white track at `theme::AIM_TRACK_ALPHA` and a `theme::TARGET` arc from 12 o'clock, clockwise,
//! filled by `aim_fill`.
//!
//! Fonts: the name is Inter Medium, the distance IBM Plex Mono Regular (`theme::text_font`,
//! `theme::mono_font`), as installed by `theme::font_definitions`; in a context without them the
//! overlay falls back to egui's proportional and monospace families ([`installed_font`]).
//!
//! Hidden (preview `74-hud.js:256-279`, `:298`): the place you are at (the nearest surface,
//! within [`AT_PLACE_RADII`] of its radii), anything behind a resolved body nearer the camera
//! (the preview hides what is behind the body you are at; every resolved body is a superset),
//! all but the target during a flight (and the target once its disk is over 30 pt), and anything
//! within 14 pt of the sides, 24 pt of the top or 30 pt of the bottom. Over a disk larger than
//! `theme::MARKER_MAX_DISK_PT` there is no glyph, only the label (plan C5: "no marker over a disk
//! larger than 20 pt (label only)"; [`Marker::glyph`] is false); the aim ring is still drawn
//! there, since it counts down a goto (lead ruling for C5). The selected object's large disk has
//! no label either, its bracket names it. Labels are drawn only where they do not overlap (lead
//! ruling B.3): the flight target, the aimed and the selected marker first, then the Sun and
//! planets, TestBH-1, dwarf planets, moons, comets, asteroids, exoplanet hosts and stars, nearer
//! first.
//!
//! **Off-screen destinations** (lead ruling for C5): the selected object and the goto target,
//! when outside that inner frame or behind the camera, keep a marker whose `pt` lies outside the
//! inner frame, toward the object; [`draw`] paints it as a small muted chevron on the inner
//! frame's edge pointing that way, with no label.

use std::f32::consts::TAU;
use std::sync::Arc;

use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, FontFamily, FontId, Galley, Painter, Pos2, Rect, Shape, Stroke};
use glam::Vec2;
use uni_sim::registry::{ObjectId, ObjectKind};

use crate::format::format_distance;
use crate::theme;

/// What a marker stands for (its glyph). The Sun is a `Star`, dwarf planets are `Planet`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    Planet,
    Moon,
    Comet,
    Asteroid,
    /// An exoplanet host: ring and centre dot.
    ExoHost,
    Star,
    BlackHole,
}

/// One marker, in output points.
#[derive(Clone, Debug)]
pub struct Marker {
    /// The object it marks.
    pub id: ObjectId,
    /// The object's centre, where the glyph is drawn and the label hangs from. Outside the inner
    /// frame (module docs) for an off-screen destination: [`draw`] then paints an edge chevron
    /// pointing toward `pt`.
    pub pt: Vec2,
    /// Its glyph is drawn. False over a disk larger than `theme::MARKER_MAX_DISK_PT`: label only
    /// (plan C5), the label where it would be beside a glyph.
    pub glyph: bool,
    /// `None` where the label would overlap another, where the bracket already names the
    /// object, for an off-screen chevron, or with labels off.
    pub label: Option<String>,
    pub kind: MarkerKind,
    pub selected: bool,
    /// The mono sub-line (distance); present exactly when `label` is.
    pub sub: Option<String>,
    /// The aim ring's fill, 0 ..= 1 (0: no ring).
    pub aim_fill: f32,
    /// The goto's destination.
    pub target: bool,
    /// Angle from the view axis [rad] (markers far off axis are fainter).
    pub off_axis_rad: f32,
}

/// The place you are at: the nearest surface when the camera is within this many of its largest
/// radii of the centre (a goto stands off at 4, plan C3).
pub const AT_PLACE_RADII: f64 = 5.0;
/// Margins of the inner frame markers live in [pt] (preview `74-hud.js:298`).
const EDGE_SIDE_PT: f32 = 14.0;
const EDGE_TOP_PT: f32 = 24.0;
const EDGE_BOTTOM_PT: f32 = 30.0;
/// During a flight the target's marker hides once its disk is larger than this [pt]
/// (preview `74-hud.js:257`).
const FLIGHT_TARGET_MAX_PT: f32 = 30.0;
/// Opacities (preview `00-head.html:261`, `:269-271`) and the off-axis angle [deg] past which a
/// marker is faint (`74-hud.js:340`).
const OPACITY: f32 = 0.5;
const OPACITY_FAR: f32 = 0.32;
const OPACITY_FULL: f32 = 1.0;
const FAR_OFF_AXIS_DEG: f32 = 30.0;
/// The glyph's box [pt], the gap glyph → name and name → distance [pt], the type sizes [pt] and
/// the name's letter-spacing [em] (preview `00-head.html:255-276`).
const GLYPH_BOX_PT: f32 = 20.0;
const NAME_GAP_PT: f32 = 4.0;
const SUB_GAP_PT: f32 = 8.0;
const NAME_SIZE_PT: f32 = 11.0;
const SUB_SIZE_PT: f32 = 10.0;
const NAME_TRACKING_EM: f32 = 0.03;
/// Advance of a character [em] for the label-overlap estimate (build time has no fonts). Names
/// are Inter Medium at 11 pt: over all 17 821 registry names, measured with the HUD's fonts on
/// 2026-11-23, 0.538 em per character (median), 0.630 (99 %), 0.663 (99.9 %) and 0.733 at most
/// ("Homam", a star). 0.67 covers 99.9 %; the few wider names overrun by under 4 pt, which is
/// cheaper than reserving the widest name's advance for every label and dropping more of them.
const LABEL_ADVANCE_EM: f32 = 0.67;
/// IBM Plex Mono's advance, exactly 0.6 em (the distance line).
const MONO_ADVANCE_EM: f32 = 0.6;
/// The off-screen chevron: inset from the inner frame's edge, half its arm span and its depth
/// [pt], stroke [pt], opacity.
const CHEVRON_INSET_PT: f32 = 6.0;
const CHEVRON_HALF_PT: f32 = 5.0;
const CHEVRON_DEPTH_PT: f32 = 4.0;
const CHEVRON_STROKE_PT: f32 = 1.5;
const CHEVRON_OPACITY: f32 = 0.5;
/// Alpha of the labels' shadow (preview `00-head.html:266`: rgba(0, 0, 0, 0.75)).
const SHADOW_ALPHA: f32 = 0.75;
/// Segments of a full aim arc.
const AIM_ARC_SEGMENTS: f32 = 64.0;

/// The marker glyph of an object kind (the Sun is a star, dwarf planets are planets).
pub(crate) fn marker_kind(kind: ObjectKind) -> MarkerKind {
    match kind {
        ObjectKind::Sun | ObjectKind::Star => MarkerKind::Star,
        ObjectKind::Planet | ObjectKind::DwarfPlanet => MarkerKind::Planet,
        ObjectKind::Moon => MarkerKind::Moon,
        ObjectKind::Asteroid => MarkerKind::Asteroid,
        ObjectKind::Comet => MarkerKind::Comet,
        ObjectKind::ExoHost => MarkerKind::ExoHost,
        ObjectKind::BlackHole => MarkerKind::BlackHole,
    }
}

/// Label order within a tier (module docs): smaller first.
fn kind_rank(kind: ObjectKind) -> u8 {
    match kind {
        ObjectKind::Sun | ObjectKind::Planet => 0,
        ObjectKind::BlackHole => 1,
        ObjectKind::DwarfPlanet => 2,
        ObjectKind::Moon => 3,
        ObjectKind::Comet => 4,
        ObjectKind::Asteroid => 5,
        ObjectKind::ExoHost => 6,
        ObjectKind::Star => 7,
    }
}

/// One destination as the marker rules see it ([`super::build_overlay`] fills it).
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub id: ObjectId,
    pub kind: ObjectKind,
    pub name: String,
    /// Distance to the centre [m].
    pub dist_m: f64,
    /// Where the centre projects [pt], inside the frame or not; `None` behind the camera.
    pub center_pt: Option<Vec2>,
    /// A point outside the frame toward the object, for its off-screen chevron: `center_pt` in
    /// front of the camera, else far out along the object's direction in the image plane.
    pub toward_pt: Vec2,
    /// Projected radius of its disk [pt].
    pub disk_pt: f32,
    pub off_axis_rad: f32,
    /// Behind a resolved body nearer the camera.
    pub occluded: bool,
}

/// What decides which markers show and how.
#[derive(Clone, Debug)]
pub(crate) struct Rules {
    pub selected: Option<ObjectId>,
    pub aim: Option<(ObjectId, f32)>,
    pub goto_target: Option<ObjectId>,
    /// The body the camera is at.
    pub at: Option<ObjectId>,
    /// Marker labels on (`OverlaySettings::labels`).
    pub labels: bool,
    /// The output frame [pt].
    pub frame_pt: Vec2,
    /// The bracket is drawn on the selected object: it names it, so the marker does not.
    pub bracket_shown: bool,
    /// Rectangles other labels must not overlap (the bracket's label).
    pub reserved: Vec<Rect>,
}

/// The frame markers live in: `frame` without the side, top and bottom margins.
pub(crate) fn inner_frame(frame: Rect) -> Rect {
    Rect::from_min_max(
        frame.min + egui::vec2(EDGE_SIDE_PT, EDGE_TOP_PT),
        frame.max - egui::vec2(EDGE_SIDE_PT, EDGE_BOTTOM_PT),
    )
}

fn pos2(p: Vec2) -> Pos2 {
    egui::pos2(p.x, p.y)
}

/// The markers for `cands` under `rules` (module docs), most important first.
pub(crate) fn layout(cands: &[Candidate], rules: &Rules) -> Vec<Marker> {
    let frame = Rect::from_min_size(Pos2::ZERO, egui::vec2(rules.frame_pt.x, rules.frame_pt.y));
    let inner = inner_frame(frame);
    let in_flight = rules.goto_target.is_some();
    let aimed = |id| rules.aim.is_some_and(|(a, _)| a == id);
    let mut shown: Vec<(&Candidate, Marker)> = Vec::new();
    for c in cands {
        let target = rules.goto_target == Some(c.id);
        let selected = rules.selected == Some(c.id);
        if rules.at == Some(c.id)
            || (in_flight && !target)
            || (target && c.disk_pt > FLIGHT_TARGET_MAX_PT)
        {
            continue;
        }
        let aim_fill = rules
            .aim
            .filter(|&(id, _)| id == c.id)
            .map_or(0.0, |(_, fill)| fill.clamp(0.0, 1.0));
        let marker = Marker {
            id: c.id,
            pt: c.toward_pt,
            glyph: true,
            label: None,
            kind: marker_kind(c.kind),
            selected,
            sub: None,
            aim_fill,
            target,
            off_axis_rad: c.off_axis_rad,
        };
        let Some(center) = c.center_pt.filter(|&p| inner.contains(pos2(p))) else {
            // Off the inner frame or behind the camera: an edge chevron for a chosen object, unless
            // its bracket (centre still in the frame's margin) already marks it.
            if target || (selected && !rules.bracket_shown) {
                shown.push((
                    c,
                    Marker {
                        aim_fill: 0.0,
                        ..marker
                    },
                ));
            }
            continue;
        };
        if c.occluded {
            continue;
        }
        // Plan C5: no glyph over a disk larger than 20 pt, label only (a NaN size keeps it).
        let glyph = !(c.disk_pt > theme::MARKER_MAX_DISK_PT);
        let named = rules.labels && !(selected && rules.bracket_shown);
        if !glyph && !named && aim_fill == 0.0 {
            // Nothing left to draw (the selected disk's bracket names it, or labels are off).
            continue;
        }
        shown.push((
            c,
            Marker {
                pt: center,
                glyph,
                label: named.then(|| c.name.clone()),
                sub: named.then(|| format_distance(c.dist_m)),
                ..marker
            },
        ));
    }
    let tier = |c: &Candidate| {
        if rules.goto_target == Some(c.id) || aimed(c.id) {
            0u8
        } else if rules.selected == Some(c.id) {
            1
        } else {
            2
        }
    };
    shown.sort_by(|(a, _), (b, _)| {
        tier(a)
            .cmp(&tier(b))
            .then(kind_rank(a.kind).cmp(&kind_rank(b.kind)))
            .then(a.dist_m.total_cmp(&b.dist_m))
            .then(a.id.cmp(&b.id))
    });
    // Labels where they do not overlap one already placed (lead ruling B.3). An "other" object
    // (dwarf planet, comet, asteroid, exoplanet host or star) that is neither the goto target,
    // aimed at nor selected is marked only with its label: if the label would overlap, or labels
    // are off, the whole marker goes (coordinator ruling at Task C6: soft and uncluttered).
    let mut taken = rules.reserved.clone();
    let mut keep = vec![true; shown.len()];
    for (k, (c, m)) in shown.iter_mut().enumerate() {
        let other = tier(c) == 2 && is_other(c.kind);
        let Some(rect) = label_rect(m) else {
            keep[k] = !other || m.aim_fill > 0.0;
            continue;
        };
        if taken.iter().any(|t| t.intersects(rect)) {
            m.label = None;
            m.sub = None;
            keep[k] = !other;
        } else {
            taken.push(rect);
        }
    }
    shown
        .into_iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|((_, m), _)| m)
        .collect()
}

/// The kinds marked only by brightness (and only with a label that fits): the coordinator's
/// ruling at Task C6 ("others": dwarf planets, comets, asteroids, exoplanet hosts; stars, which
/// are destinations only when chosen, too).
fn is_other(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::DwarfPlanet
            | ObjectKind::Comet
            | ObjectKind::Asteroid
            | ObjectKind::ExoHost
            | ObjectKind::Star
    )
}

/// Where `m`'s label and distance lie [pt], estimated without fonts at
/// [`LABEL_ADVANCE_EM`] per character; `None` without a label.
fn label_rect(m: &Marker) -> Option<Rect> {
    let name = m.label.as_ref()?;
    let chars = |s: &str| s.chars().count() as f32;
    let name_w = chars(name) * NAME_SIZE_PT * (LABEL_ADVANCE_EM + NAME_TRACKING_EM);
    let sub_w = m.sub.as_ref().map_or(0.0, |s| {
        SUB_GAP_PT + chars(s) * SUB_SIZE_PT * MONO_ADVANCE_EM
    });
    let left = m.pt.x + 0.5 * GLYPH_BOX_PT + NAME_GAP_PT;
    Some(Rect::from_min_size(
        egui::pos2(left, m.pt.y - 0.5 * NAME_SIZE_PT - 1.0),
        egui::vec2(name_w + sub_w, NAME_SIZE_PT + 2.0),
    ))
}

/// The opacity of `m` (module docs).
pub(crate) fn opacity(m: &Marker) -> f32 {
    if m.target || m.aim_fill > 0.0 {
        OPACITY_FULL
    } else if m.off_axis_rad > FAR_OFF_AXIS_DEG.to_radians() {
        OPACITY_FAR
    } else {
        OPACITY
    }
}

/// Paints `markers` (most important last, so on top) in `frame` [pt].
///
/// A marker whose glyph, aim ring or label meets a rect of `keep_out` (the HUD's panels, grown
/// by a margin) is not drawn at all, nor an edge chevron there (lead ruling C4w r3: the overlay
/// never draws over a panel).
pub(crate) fn draw(painter: &Painter, frame: Rect, markers: &[Marker], keep_out: &[Rect]) {
    let inner = inner_frame(frame);
    let meets = |r: Rect| keep_out.iter().any(|k| k.intersects(r));
    for m in markers.iter().rev() {
        let p = pos2(m.pt);
        if !inner.contains(p) {
            if let Some((tip, dir)) = chevron_tip(frame, inner, p)
                && !meets(Rect::from_center_size(
                    tip,
                    egui::Vec2::splat(2.0 * CHEVRON_HALF_PT),
                ))
            {
                draw_chevron(painter, tip, dir);
            }
            continue;
        }
        let o = opacity(m);
        let lit = m.target || m.aim_fill > 0.0;
        let ink = if m.selected {
            theme::TARGET
        } else if lit {
            theme::INK
        } else {
            theme::INK_SOFT
        };
        let (name, sub) = label_galleys(painter, m, ink, lit, o);
        let mut footprint = Rect::from_center_size(p, egui::Vec2::splat(GLYPH_BOX_PT));
        if m.aim_fill > 0.0 {
            footprint = footprint.union(Rect::from_center_size(
                p,
                egui::Vec2::splat(2.0 * theme::AIM_RING_RADIUS + theme::AIM_RING_STROKE),
            ));
        }
        let left = p.x + 0.5 * GLYPH_BOX_PT + NAME_GAP_PT;
        let name_top_left = |g: &Arc<Galley>| egui::pos2(left, p.y - 0.5 * g.size().y);
        let sub_left = left + name.as_ref().map_or(0.0, |g| g.size().x) + SUB_GAP_PT;
        let sub_top_left = |g: &Arc<Galley>| egui::pos2(sub_left, p.y - 0.5 * g.size().y);
        if let Some(g) = &name {
            footprint = footprint.union(Rect::from_min_size(name_top_left(g), g.size()));
        }
        if let Some(g) = &sub {
            footprint = footprint.union(Rect::from_min_size(sub_top_left(g), g.size()));
        }
        if meets(footprint) {
            continue;
        }
        if m.glyph {
            draw_glyph(painter, p, m.kind, ink.gamma_multiply(o));
        }
        if m.aim_fill > 0.0 {
            draw_aim_ring(painter, p, m.aim_fill);
        }
        if let Some(g) = name {
            let at = name_top_left(&g);
            paint_text(painter, at, g, o);
        }
        if let Some(g) = sub {
            let at = sub_top_left(&g);
            paint_text(painter, at, g, o);
        }
    }
}

/// The name and sub-line galleys of `m`'s label (none without one).
fn label_galleys(
    painter: &Painter,
    m: &Marker,
    ink: Color32,
    lit: bool,
    o: f32,
) -> (Option<Arc<Galley>>, Option<Arc<Galley>>) {
    let Some(name) = &m.label else {
        return (None, None);
    };
    {
        let name_color = if m.selected || lit {
            theme::TARGET
        } else {
            ink
        };
        let format = TextFormat {
            font_id: installed_font(
                painter,
                theme::text_font(NAME_SIZE_PT, theme::Weight::Medium),
                FontFamily::Proportional,
            ),
            extra_letter_spacing: NAME_TRACKING_EM * NAME_SIZE_PT,
            color: name_color.gamma_multiply(o),
            ..TextFormat::default()
        };
        let name = painter.layout_job(LayoutJob::single_section(name.clone(), format));
        let sub = m.sub.as_ref().map(|sub| {
            painter.layout_no_wrap(
                sub.clone(),
                installed_font(
                    painter,
                    theme::mono_font(SUB_SIZE_PT, theme::Weight::Regular),
                    FontFamily::Monospace,
                ),
                theme::INK_3.gamma_multiply(o),
            )
        });
        (Some(name), sub)
    }
}

/// `want` when `painter`'s context has its family (the HUD's `theme::font_definitions` install
/// every weight), else the same size in `fallback`: egui panics on the first glyph of a family it
/// has no font for, and the overlay may be drawn in a context without the HUD's fonts.
pub(crate) fn installed_font(painter: &Painter, want: FontId, fallback: FontFamily) -> FontId {
    let have = painter
        .ctx()
        .fonts(|f| f.definitions().families.contains_key(&want.family));
    if have {
        want
    } else {
        FontId::new(want.size, fallback)
    }
}

/// Paints `galley` at `top_left` over a soft dark shadow 1 pt below it, so labels stay legible
/// over a bright body (the preview's text-shadow), the shadow faded with `opacity`.
pub(crate) fn paint_text(painter: &Painter, top_left: Pos2, galley: Arc<Galley>, opacity: f32) {
    let shadow = Color32::from_black_alpha((SHADOW_ALPHA * opacity * 255.0).round() as u8);
    painter.galley_with_override_text_color(
        top_left + egui::vec2(0.0, 1.0),
        galley.clone(),
        shadow,
    );
    painter.galley(top_left, galley, Color32::PLACEHOLDER);
}

/// The glyph of `kind` centred on `c`, in `color`. Planet, moon, comet and black hole are the
/// preview's 20 × 20 SVGs (`74-hud.js:226-232`, offsets from the box's centre); the exoplanet
/// host is the planet ring with a centre dot (plan C5); the star (a ring with four rays) and the
/// asteroid (a small diamond) are ours, in the same strokes.
fn draw_glyph(painter: &Painter, c: Pos2, kind: MarkerKind, color: Color32) {
    let ring = |r: f32, w: f32| {
        painter.circle_stroke(c, r, Stroke::new(w, color));
    };
    match kind {
        MarkerKind::Planet => ring(4.2, 1.3),
        MarkerKind::Moon => ring(2.8, 1.3),
        MarkerKind::BlackHole => {
            ring(6.0, 1.0);
            painter.circle_filled(c, 2.6, color);
        }
        MarkerKind::Comet => {
            painter.circle_stroke(c + egui::vec2(2.5, -2.5), 2.4, Stroke::new(1.3, color));
            painter.line_segment(
                [c + egui::vec2(0.8, -0.8), c + egui::vec2(-7.0, 7.0)],
                Stroke::new(1.1, color),
            );
        }
        MarkerKind::ExoHost => {
            ring(4.2, 1.3);
            painter.circle_filled(c, 1.3, color);
        }
        MarkerKind::Star => {
            ring(3.0, 1.3);
            for d in [
                egui::vec2(1.0, 0.0),
                egui::vec2(0.0, 1.0),
                egui::vec2(-1.0, 0.0),
                egui::vec2(0.0, -1.0),
            ] {
                painter.line_segment([c + d * 5.0, c + d * 7.5], Stroke::new(1.0, color));
            }
        }
        MarkerKind::Asteroid => {
            let r = 3.2;
            painter.add(Shape::closed_line(
                vec![
                    c + egui::vec2(0.0, -r),
                    c + egui::vec2(r, 0.0),
                    c + egui::vec2(0.0, r),
                    c + egui::vec2(-r, 0.0),
                ],
                Stroke::new(1.2, color),
            ));
        }
    }
}

/// The aim ring around `c`: the white track, then the cyan arc from 12 o'clock, clockwise over
/// `fill` of the turn.
fn draw_aim_ring(painter: &Painter, c: Pos2, fill: f32) {
    let track = Color32::WHITE.gamma_multiply(theme::AIM_TRACK_ALPHA);
    painter.circle_stroke(
        c,
        theme::AIM_RING_RADIUS,
        Stroke::new(theme::AIM_RING_STROKE, track),
    );
    let fill = fill.clamp(0.0, 1.0);
    let n = ((AIM_ARC_SEGMENTS * fill).ceil() as usize).max(1);
    let arc = (0..=n)
        .map(|k| {
            let a = TAU * fill * k as f32 / n as f32;
            c + egui::vec2(a.sin(), -a.cos()) * theme::AIM_RING_RADIUS
        })
        .collect();
    painter.add(Shape::line(
        arc,
        Stroke::new(theme::AIM_RING_STROKE, theme::TARGET),
    ));
}

/// Where the muted chevron for an off-screen destination sits, and its outward direction:
/// where the ray from the frame's centre toward `toward` leaves the inner frame, inset. `None`
/// at the centre itself.
fn chevron_tip(frame: Rect, inner: Rect, toward: Pos2) -> Option<(Pos2, egui::Vec2)> {
    let from = frame.center();
    let d = toward - from;
    let len = d.length();
    if !(len > 0.0) {
        return None;
    }
    let dir = d / len;
    // Distance along the ray to the inner frame's edge, per axis.
    let exit = |o: f32, v: f32, lo: f32, hi: f32| {
        if v > 0.0 {
            (hi - o) / v
        } else if v < 0.0 {
            (lo - o) / v
        } else {
            f32::INFINITY
        }
    };
    let t = exit(from.x, dir.x, inner.min.x, inner.max.x)
        .min(exit(from.y, dir.y, inner.min.y, inner.max.y))
        .max(0.0);
    let tip = (from + dir * t).clamp(
        inner.min + egui::vec2(CHEVRON_INSET_PT, CHEVRON_INSET_PT),
        inner.max - egui::vec2(CHEVRON_INSET_PT, CHEVRON_INSET_PT),
    );
    Some((tip, dir))
}

/// The edge chevron at `tip`, pointing along `dir`.
fn draw_chevron(painter: &Painter, tip: Pos2, dir: egui::Vec2) {
    let back = tip - dir * CHEVRON_DEPTH_PT;
    let side = egui::vec2(-dir.y, dir.x) * CHEVRON_HALF_PT;
    painter.add(Shape::line(
        vec![back + side, tip, back - side],
        Stroke::new(
            CHEVRON_STROKE_PT,
            theme::INK_SOFT.gamma_multiply(CHEVRON_OPACITY),
        ),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: Vec2 = Vec2::new(1470.0, 956.0);

    fn cand(
        id: ObjectId,
        kind: ObjectKind,
        center: (f32, f32),
        disk_pt: f32,
        dist_m: f64,
    ) -> Candidate {
        let c = Vec2::new(center.0, center.1);
        Candidate {
            id,
            kind,
            name: format!("Object {id}"),
            dist_m,
            center_pt: Some(c),
            toward_pt: c,
            disk_pt,
            off_axis_rad: 0.1,
            occluded: false,
        }
    }

    fn rules() -> Rules {
        Rules {
            selected: None,
            aim: None,
            goto_target: None,
            at: None,
            labels: true,
            frame_pt: FRAME,
            bracket_shown: false,
            reserved: Vec::new(),
        }
    }

    fn find(markers: &[Marker], id: ObjectId) -> Option<&Marker> {
        markers.iter().find(|m| m.id == id)
    }

    /// Plan C5 step 1: "body disk 25 pt → label, no ring; 15 pt → ring and label". Exactly
    /// 20 pt still has its glyph ("larger than 20 pt"). The selected large disk has neither (its
    /// bracket names it) unless it is aimed at, when only the aim ring is left; with labels off a
    /// large disk has no marker.
    #[test]
    fn markers_suppressed_over_large_disks() {
        let big = cand(1, ObjectKind::Planet, (600.0, 400.0), 25.0, 1.0e9);
        let small = cand(2, ObjectKind::Planet, (900.0, 400.0), 15.0, 1.0e9);
        let edge = cand(3, ObjectKind::Moon, (300.0, 400.0), 20.0, 1.0e9);
        let m = layout(&[big.clone(), small, edge], &rules());
        let big_m = find(&m, 1).expect("big disk labelled");
        assert!(!big_m.glyph, "no ring over a 25 pt disk");
        assert_eq!(big_m.label.as_deref(), Some("Object 1"));
        assert!(big_m.sub.is_some());
        assert_eq!(big_m.pt, Vec2::new(600.0, 400.0), "label from the centre");
        let small = find(&m, 2).expect("small disk marked");
        assert!(small.glyph, "ring on a 15 pt disk");
        assert_eq!(small.pt, Vec2::new(900.0, 400.0));
        assert_eq!(small.label.as_deref(), Some("Object 2"));
        assert!(find(&m, 3).expect("20 pt disk").glyph);
        // The selected object's large disk carries the bracket instead.
        let r = Rules {
            selected: Some(1),
            bracket_shown: true,
            ..rules()
        };
        assert!(layout(std::slice::from_ref(&big), &r).is_empty());
        // Aimed at, it keeps the aim ring (a goto counting down), still with no glyph or label.
        let aimed = layout(
            std::slice::from_ref(&big),
            &Rules {
                aim: Some((1, 0.5)),
                ..r
            },
        );
        assert_eq!(aimed.len(), 1);
        let a = &aimed[0];
        assert!(!a.glyph && a.label.is_none() && a.aim_fill == 0.5);
        let off = Rules {
            labels: false,
            ..rules()
        };
        assert!(
            layout(&[big], &off).is_empty(),
            "labels off: nothing to draw"
        );
    }

    /// Preview `74-hud.js:256-298`: no marker within 14 pt of the sides, 24 pt of the top or
    /// 30 pt of the bottom, none for the place you are at, none behind a nearer body.
    #[test]
    fn markers_hidden_at_edges_here_and_behind_bodies() {
        let cands = [
            cand(1, ObjectKind::Planet, (10.0, 400.0), 1.0, 1e9),
            cand(2, ObjectKind::Planet, (1460.0, 400.0), 1.0, 1e9),
            cand(3, ObjectKind::Planet, (700.0, 20.0), 1.0, 1e9),
            cand(4, ObjectKind::Planet, (700.0, 930.0), 1.0, 1e9),
            cand(5, ObjectKind::Planet, (700.0, 400.0), 1.0, 1e9),
            cand(6, ObjectKind::Moon, (300.0, 300.0), 1.0, 1e9),
            Candidate {
                occluded: true,
                ..cand(7, ObjectKind::Moon, (400.0, 300.0), 1.0, 1e9)
            },
            cand(8, ObjectKind::Moon, (15.0, 25.0), 1.0, 1e9),
        ];
        let m = layout(
            &cands,
            &Rules {
                at: Some(6),
                ..rules()
            },
        );
        let ids: Vec<ObjectId> = m.iter().map(|m| m.id).collect();
        assert_eq!(ids.len(), 2, "{ids:?}");
        assert!(ids.contains(&5) && ids.contains(&8), "{ids:?}");
    }

    /// Preview `74-hud.js:257`: during a flight only the target is marked, until its disk is
    /// larger than 30 pt; its marker is full opacity and flagged.
    #[test]
    fn flight_marks_only_the_target() {
        let cands = [
            cand(1, ObjectKind::Planet, (300.0, 300.0), 2.0, 1e9),
            cand(2, ObjectKind::Planet, (700.0, 400.0), 12.0, 1e9),
        ];
        let r = Rules {
            goto_target: Some(2),
            ..rules()
        };
        let m = layout(&cands, &r);
        assert_eq!(m.len(), 1);
        assert!(m[0].id == 2 && m[0].target);
        assert_eq!(opacity(&m[0]), 1.0);
        let near = [cand(2, ObjectKind::Planet, (700.0, 400.0), 31.0, 1e9)];
        assert!(layout(&near, &r).is_empty());
    }

    /// Preview `00-head.html:261-271`: 0.5, 0.32 beyond 30° off axis, 1 aimed or the target;
    /// the aim's fill goes to its marker only.
    #[test]
    fn opacity_follows_aim_target_and_off_axis() {
        let far = Candidate {
            off_axis_rad: 31f32.to_radians(),
            ..cand(1, ObjectKind::Moon, (300.0, 300.0), 1.0, 1e9)
        };
        let near = cand(2, ObjectKind::Moon, (600.0, 300.0), 1.0, 1e9);
        let aimed = Candidate {
            off_axis_rad: 40f32.to_radians(),
            ..cand(3, ObjectKind::Moon, (900.0, 300.0), 1.0, 1e9)
        };
        let m = layout(
            &[far, near, aimed],
            &Rules {
                aim: Some((3, 0.4)),
                ..rules()
            },
        );
        assert_eq!(opacity(find(&m, 1).expect("far")), 0.32);
        assert_eq!(opacity(find(&m, 2).expect("near")), 0.5);
        let aimed = find(&m, 3).expect("aimed");
        assert_eq!((aimed.aim_fill, opacity(aimed)), (0.4, 1.0));
        assert_eq!(find(&m, 2).expect("near").aim_fill, 0.0);
    }

    /// Lead ruling B.3: labels only where they do not overlap; the more important keeps its
    /// label (a planet over its moon; the selected object over a planet); labels can be off.
    #[test]
    fn overlapping_labels_keep_the_more_important() {
        let earth = cand(1, ObjectKind::Planet, (700.0, 400.0), 1.0, 2e11);
        let moon = cand(2, ObjectKind::Moon, (702.0, 401.0), 0.5, 2e11);
        let mars = cand(3, ObjectKind::Planet, (700.0, 600.0), 1.0, 1e11);
        let m = layout(&[moon.clone(), earth.clone(), mars.clone()], &rules());
        assert_eq!(m.len(), 3, "every glyph stays");
        assert!(find(&m, 1).expect("earth").label.is_some());
        let moon_m = find(&m, 2).expect("moon");
        assert!(moon_m.label.is_none() && moon_m.sub.is_none());
        assert!(
            find(&m, 3).expect("mars").label.is_some(),
            "no overlap, labelled"
        );
        assert_eq!(m[0].id, 3, "nearer planet first");
        let selected = layout(
            &[moon.clone(), earth.clone()],
            &Rules {
                selected: Some(2),
                ..rules()
            },
        );
        assert!(find(&selected, 2).expect("moon").label.is_some());
        assert!(find(&selected, 1).expect("earth").label.is_none());
        assert!(find(&selected, 2).expect("moon").selected);
        // The bracket's label is taken.
        let reserved = Rect::from_min_size(egui::pos2(690.0, 390.0), egui::vec2(300.0, 30.0));
        let m = layout(
            &[earth.clone()],
            &Rules {
                reserved: vec![reserved],
                ..rules()
            },
        );
        assert!(m[0].label.is_none());
        let m = layout(
            &[earth, mars],
            &Rules {
                labels: false,
                ..rules()
            },
        );
        assert!(m.iter().all(|m| m.label.is_none() && m.sub.is_none()));
    }

    /// Lead ruling for C5: the selected object and the goto target off screen keep a marker
    /// outside the inner frame (an edge chevron), unlabelled; other off-screen objects none.
    #[test]
    fn off_screen_choices_get_a_chevron() {
        let behind = Candidate {
            center_pt: None,
            toward_pt: Vec2::new(-5000.0, 478.0),
            ..cand(1, ObjectKind::Planet, (0.0, 0.0), 1.0, 1e11)
        };
        let off_right = cand(2, ObjectKind::Comet, (1700.0, 300.0), 0.1, 1e11);
        let other = cand(3, ObjectKind::Planet, (1800.0, 300.0), 0.1, 1e11);
        let r = Rules {
            selected: Some(1),
            goto_target: Some(2),
            ..rules()
        };
        // A flight shows only the target: the selected chevron hides.
        let m = layout(&[behind.clone(), off_right.clone(), other.clone()], &r);
        assert_eq!(m.len(), 1);
        assert_eq!((m[0].id, m[0].pt), (2, Vec2::new(1700.0, 300.0)));
        assert!(m[0].label.is_none() && m[0].target);
        let m = layout(
            &[behind, off_right, other],
            &Rules {
                goto_target: None,
                ..r
            },
        );
        assert_eq!(m.len(), 1);
        assert_eq!((m[0].id, m[0].pt), (1, Vec2::new(-5000.0, 478.0)));
        let inner = inner_frame(Rect::from_min_size(
            Pos2::ZERO,
            egui::vec2(FRAME.x, FRAME.y),
        ));
        assert!(!inner.contains(egui::pos2(m[0].pt.x, m[0].pt.y)));
        assert_eq!(
            inner,
            Rect::from_min_max(egui::pos2(14.0, 24.0), egui::pos2(1456.0, 926.0))
        );
    }

    /// The label estimate covers a marker's name as painted in Inter Medium with its tracking
    /// (lead ruling for C5: 11 px, 500), so the overlap rule sees the label's real extent: fully
    /// for ordinary names, within `OVERRUN_PT` for the widest registry names
    /// ([`LABEL_ADVANCE_EM`]).
    #[test]
    fn label_rect_covers_the_painted_name() {
        const OVERRUN_PT: f32 = 4.0;
        let ctx = crate::overlay::tests::fonts_context();
        for name in [
            "Mars",
            "Jupiter",
            "Mercury",
            "Ganymede",
            "Miranda",
            "TestBH-1",
            "TRAPPIST-1",
            "1P/Halley",
            "Wolf 359",
            "67P/Churyumov\u{2013}Gerasimenko",
            // The widest registry names per character.
            "Homam",
            "Emma",
            "Mneme",
            "Ogma",
        ] {
            let painted = ctx.fonts_mut(|f| {
                let format = TextFormat {
                    font_id: theme::text_font(NAME_SIZE_PT, theme::Weight::Medium),
                    extra_letter_spacing: NAME_TRACKING_EM * NAME_SIZE_PT,
                    ..TextFormat::default()
                };
                f.layout_job(LayoutJob::single_section(name.to_owned(), format))
                    .size()
                    .x
            });
            let m = Marker {
                id: 1,
                pt: Vec2::new(500.0, 300.0),
                glyph: true,
                label: Some(name.to_owned()),
                kind: MarkerKind::Planet,
                selected: false,
                sub: None,
                aim_fill: 0.0,
                target: false,
                off_axis_rad: 0.0,
            };
            let est = label_rect(&m).expect("labelled").width();
            let wide = ["Homam", "Emma", "Mneme", "Ogma"].contains(&name);
            let slack = if wide { OVERRUN_PT } else { 0.0 };
            assert!(
                est + slack >= painted,
                "{name}: estimate {est} < painted {painted}"
            );
        }
    }

    /// Every object kind has a glyph; the Sun is a star and dwarf planets are planets.
    #[test]
    fn kinds_map_to_glyphs() {
        assert_eq!(marker_kind(ObjectKind::Sun), MarkerKind::Star);
        assert_eq!(marker_kind(ObjectKind::DwarfPlanet), MarkerKind::Planet);
        assert_eq!(marker_kind(ObjectKind::ExoHost), MarkerKind::ExoHost);
        assert_eq!(marker_kind(ObjectKind::BlackHole), MarkerKind::BlackHole);
    }
}
