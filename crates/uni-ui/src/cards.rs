//! The target card (Task C4; Task C0 wrote the types and the `build_card` stub).
//!
//! Closed by default (owner point 4): the "i" button or I opens it; while a goto runs it
//! describes the destination (kicker "En route to"). Rows per `uni_sim::world::ObjectDetails`
//! kind (plan C4); the notes are the info-card notes of `crate::theme` (exact strings).

use glam::DVec3;
use uni_core::{AU, CameraView, DAY, Epoch, GM_SUN, M_SUN, PC, R_SUN_M, TEFF_SUN_K};
use uni_data::{
    AsteroidRecord, BodyFacts, Comet, ExoSystem, FLAG_NO_RV, FLAG_NO_VELOCITY, SatRecord, Star3,
    StarName,
};
use uni_physics::kepler::Conic;
use uni_sim::registry::{ObjectId, ObjectInfo, ObjectKind};
use uni_sim::world::{ObjectDetails, World};

use crate::format::{
    format_date_tdb, format_distance, format_grouped, format_hms, format_mass_kg,
    format_period_days, format_sig, format_speed, subscript_digits,
};
use crate::theme::{
    self, LabelTone, NOTE_ARCHIVE_HOST, NOTE_BLACK_HOLE_M3, NOTE_EARTH_CLOUDS, NOTE_GRANULATION,
    NOTE_HAZE, NOTE_INCLINATION_UNKNOWN, NOTE_NODE_UNKNOWN, NOTE_NUCLEUS_SHAPE,
    NOTE_ORBIT_MEAN_ELEMENTS, NOTE_PERIASTRON_CONVENTION, NOTE_RINGS, NOTE_TAILS,
    NOTE_TANGENTIAL_MOTION, NOTE_VENUS_CLOUD_TOPS, SATURN_MOONS_LABEL,
};

/// One label–value row.
#[derive(Clone, Debug)]
pub struct CardRow {
    pub label: String,
    pub value: String,
    /// The value is a number, set in the mono family.
    pub mono: bool,
    pub tone: LabelTone,
}

/// A card button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardAction {
    /// "Go to · F".
    Goto,
    /// "Moons": lists the moons; clicking one selects it (`HudOutput::selected_moon`).
    Moons,
    /// "Close · Esc".
    Close,
}

/// What the target card shows.
#[derive(Clone, Debug)]
pub struct TargetCard {
    /// "TARGET", "TARGET · BLACK HOLE", "EN ROUTE TO", …
    pub kicker: String,
    pub title: String,
    pub subtitle: String,
    /// Distance, closing speed, light time (between hairlines).
    pub live: Vec<CardRow>,
    pub facts: Vec<CardRow>,
    /// The info-card honesty notes (grey).
    pub notes: Vec<String>,
    pub sources: String,
    pub actions: Vec<CardAction>,
    /// Names of the moons the Moons button lists.
    pub moons: Vec<String>,
}

/// The card of object `id` seen from `view`, from `World::details`; `None` for an id the world
/// does not know (ruling A.3b).
pub fn build_card(world: &World, id: ObjectId, view: &CameraView) -> Option<TargetCard> {
    let info = world.objects().get(id as usize)?;
    let details = world.details(id)?;
    let pos = world.position(id);
    // Object − camera, and the object's velocity relative to the camera.
    let rel = pos.delta_from(view.pos);
    let rel_vel = world.velocity(id) - view.velocity_mps;
    let dist_m = rel.length();
    // d|r|/dt = r̂·v; closing is its negative (> 0 approaching).
    let closing_mps = if dist_m > 0.0 {
        -rel.dot(rel_vel) / dist_m
    } else {
        0.0
    };
    let sun = world
        .objects()
        .iter()
        .find(|o| o.kind == ObjectKind::Sun)
        .map(|o| world.position(o.id));
    let ctx = Ctx {
        epoch: world.epoch(),
        cam_dist_m: dist_m,
        from_sun_m: sun.map(|s| pos.delta_from(s)),
        parent: info
            .parent
            .and_then(|p| world.objects().get(p as usize))
            .map(|p| p.name.clone()),
    };
    let mut card = describe(info, &details, &ctx);
    card.live = vec![
        number_row("Distance", format_distance(dist_m)),
        number_row("Closing speed", signed_speed(closing_mps)),
        number_row(
            "Light time",
            crate::format::format_duration(uni_physics::calc::light_time_s(dist_m)),
        ),
    ];
    Some(card)
}

/// What [`describe`] needs beyond the object's records.
struct Ctx {
    /// The shown date.
    epoch: Epoch,
    /// Camera → object [m].
    cam_dist_m: f64,
    /// Object − Sun [m], if there is a Sun.
    from_sun_m: Option<DVec3>,
    /// The parent object's name (a moon's planet).
    parent: Option<String>,
}

/// A row whose value is a number (mono, neutral).
fn number_row(label: &str, value: String) -> CardRow {
    CardRow {
        label: label.to_owned(),
        value,
        mono: true,
        tone: LabelTone::Neutral,
    }
}

/// A row whose value is words (Inter, neutral), like the mockup's "Phobos, Deimos".
fn word_row(label: &str, value: String) -> CardRow {
    CardRow {
        mono: false,
        ..number_row(label, value)
    }
}

/// A closing speed, negative while receding.
fn signed_speed(mps: f64) -> String {
    if mps < 0.0 {
        format!("\u{2212}{}", format_speed(-mps))
    } else {
        format_speed(mps)
    }
}

/// Everything on the card except the live rows.
fn describe(info: &ObjectInfo, details: &ObjectDetails, ctx: &Ctx) -> TargetCard {
    let mut card = TargetCard {
        kicker: "TARGET".into(),
        title: info.name.clone(),
        subtitle: String::new(),
        live: Vec::new(),
        facts: Vec::new(),
        notes: card_notes(info, details),
        sources: String::new(),
        actions: vec![CardAction::Goto, CardAction::Close],
        moons: Vec::new(),
    };
    match details {
        ObjectDetails::Body { facts, sat } => {
            card.subtitle = body_subtitle(info, ctx);
            if let Some(f) = facts {
                card.facts = body_rows(info, f);
                card.moons = f.moons.clone();
            } else if let Some(sat) = sat {
                card.facts = sat_rows(sat);
            }
            card.sources = body_sources(info, facts.as_ref(), sat.as_ref());
        }
        ObjectDetails::Star {
            star,
            name,
            teff_k,
            radius_m,
        } => {
            card.subtitle = star_subtitle(info, name.as_ref(), None);
            card.facts = star_rows(star, name.as_ref(), *teff_k, *radius_m, ctx);
            card.sources = STAR_SOURCES.into();
        }
        ObjectDetails::Comet {
            record,
            conic,
            conic_epoch: _,
        } => {
            card.subtitle = comet_subtitle(conic);
            card.facts = comet_rows(record, conic, ctx);
            card.sources = "Data: JPL Small-Body Database · orbit re-osculated near the shown \
                            date with planetary perturbations"
                .into();
        }
        ObjectDetails::Asteroid { record, name } => {
            let kind = if info.kind == ObjectKind::DwarfPlanet {
                "Dwarf planet"
            } else {
                "Asteroid"
            };
            card.subtitle = match name {
                Some(full) if full != &info.name => format!("{kind} · {full} · real object"),
                _ => format!("{kind} · real object"),
            };
            card.facts = asteroid_rows(record);
            card.sources = "Data: JPL Small-Body Database".into();
        }
        ObjectDetails::ExoHost {
            system,
            star,
            name,
            archive_host,
            teff_k,
            radius_m,
        } => {
            card.subtitle = star_subtitle(info, name.as_ref(), Some(system.planets.len()));
            if let Some(star) = star {
                card.facts = star_rows(star, name.as_ref(), *teff_k, *radius_m, ctx);
            }
            card.facts.extend(system.planets.iter().map(|p| {
                let mut parts = Vec::new();
                if let Some(a) = p.a_au {
                    parts.push(format!("{} AU", format_sig(a, 3)));
                }
                if let Some(period) = p.period_d {
                    parts.push(format_period_days(period));
                }
                if let Some(m) = p.mass_mjup {
                    parts.push(format!("{} Jupiter masses", format_sig(m, 3)));
                } else if let Some(r) = p.radius_rjup {
                    parts.push(format!("{} Jupiter radii", format_sig(r, 3)));
                }
                parts.push(p.method.clone());
                parts.push(p.year.to_string());
                word_row(&p.name, parts.join(" · "))
            }));
            let star_source = if *archive_host {
                "star: NASA Exoplanet Archive"
            } else {
                "star: AT-HYG v4.0 (CC BY-SA 4.0)"
            };
            card.sources = format!(
                "Data: NASA Exoplanet Archive (pscomppars; masses are the archive's best mass, \
                 which may be M sin i) · {star_source}"
            );
        }
        ObjectDetails::BlackHole { mass_kg, spin } => {
            card.kicker = "TARGET · BLACK HOLE".into();
            card.subtitle = if *spin == 0.0 {
                "Black hole · non-rotating (Schwarzschild)".into()
            } else {
                format!("Black hole · spin a* = {}", format_sig(*spin, 3))
            };
            card.facts = vec![
                CardRow {
                    tone: LabelTone::NonPhysical,
                    ..word_row("Status", "Hypothetical test object".into())
                },
                number_row(
                    "Mass",
                    format!("{} solar masses", format_sig(mass_kg / M_SUN, 3)),
                ),
                // r_s = 2GM/c² (Schwarzschild 1916).
                number_row(
                    "Schwarzschild radius",
                    format!(
                        "{} km",
                        format_sig(uni_core::schwarzschild_radius(*mass_kg) / 1e3, 3)
                    ),
                ),
            ];
            if let Some(d) = ctx.from_sun_m {
                card.facts
                    .push(number_row("Distance from Sun", format_distance(d.length())));
            }
            card.sources = "Schwarzschild 1916 (r_s = 2GM/c²)".into();
        }
    }
    if !card.moons.is_empty() {
        card.actions.insert(1, CardAction::Moons);
    }
    card
}

/// What the card's buttons asked for this frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CardClicks {
    pub actions: Vec<CardAction>,
    pub selected_moon: Option<String>,
}

/// Paints `card` (mockup target artboard): kicker, title 30 pt light and subtitle; then, in a
/// scroll area at most `max_body_h` tall, the live rows between hairlines, the facts, the
/// buttons (≥ 44 pt), the moons list when open, the sources line and the notes.
pub(crate) fn show_card(ui: &mut egui::Ui, card: &TargetCard, max_body_h: f32) -> CardClicks {
    use crate::panels::inner_width;
    use crate::theme::{
        BODY_SIZE, CARD_HAIRLINE, CARD_KICKER_TRACKING_EM, CARD_PADDING_X, CARD_WIDTH, INK, INK_2,
        INK_3, MUTED, NOTES_SIZE, SMALL_SIZE, TARGET, TITLE_SIZE, Weight, text_font, tone_color,
    };

    let mut clicks = CardClicks::default();
    theme::card_frame().show(ui, |ui| {
        ui.set_width(inner_width(CARD_WIDTH, CARD_PADDING_X));
        ui.spacing_mut().item_spacing = egui::vec2(8.0, 0.0);
        theme::kicker(
            ui,
            &card.kicker,
            TARGET,
            Weight::SemiBold,
            CARD_KICKER_TRACKING_EM,
        );
        ui.add_space(4.0);
        theme::text(ui, &card.title, text_font(TITLE_SIZE, Weight::Light), INK);
        ui.add_space(4.0);
        theme::text(
            ui,
            &card.subtitle,
            text_font(BODY_SIZE, Weight::Regular),
            INK_2,
        );
        ui.add_space(16.0);
        let value_color = |r: &CardRow| match r.tone {
            LabelTone::Neutral => INK,
            tone => tone_color(tone),
        };
        egui::ScrollArea::vertical()
            .id_salt("hud card body")
            .max_height(max_body_h)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 0.0);
                if !card.live.is_empty() {
                    theme::hairline(ui, CARD_HAIRLINE);
                    ui.add_space(12.0);
                    for (i, r) in card.live.iter().enumerate() {
                        if i > 0 {
                            ui.add_space(6.0);
                        }
                        theme::row(ui, &r.label, &r.value, r.mono, value_color(r));
                    }
                    ui.add_space(12.0);
                    theme::hairline(ui, CARD_HAIRLINE);
                    ui.add_space(16.0);
                }
                for (i, r) in card.facts.iter().enumerate() {
                    if i > 0 {
                        ui.add_space(6.0);
                    }
                    theme::row(ui, &r.label, &r.value, r.mono, value_color(r));
                }
                ui.add_space(16.0);
                let moons_open_id = egui::Id::unique(("hud card moons", card.title.as_str()));
                let mut moons_open = ui
                    .data(|d| d.get_temp::<bool>(moons_open_id))
                    .unwrap_or(false);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                    let font = text_font(BODY_SIZE, Weight::Medium);
                    for action in &card.actions {
                        let (label, kind) = match action {
                            CardAction::Goto => ("Go to · F", theme::ButtonKind::Nav),
                            CardAction::Moons => ("Moons", theme::ButtonKind::Neutral),
                            CardAction::Close => ("Close · Esc", theme::ButtonKind::Neutral),
                        };
                        if theme::button(ui, label, font.clone(), None, kind).clicked() {
                            clicks.actions.push(*action);
                            if *action == CardAction::Moons {
                                moons_open = !moons_open;
                            }
                        }
                    }
                });
                ui.data_mut(|d| d.insert_temp(moons_open_id, moons_open));
                if moons_open && !card.moons.is_empty() {
                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                        let font = text_font(BODY_SIZE, Weight::Regular);
                        for moon in &card.moons {
                            let kind = theme::ButtonKind::Neutral;
                            if theme::button(ui, moon, font.clone(), None, kind).clicked() {
                                clicks.selected_moon = Some(moon.clone());
                            }
                        }
                    });
                }
                if !card.sources.is_empty() {
                    ui.add_space(16.0);
                    theme::text(
                        ui,
                        &card.sources,
                        text_font(SMALL_SIZE, Weight::Regular),
                        INK_3,
                    );
                }
                for note in &card.notes {
                    ui.add_space(6.0);
                    theme::text(ui, note, text_font(NOTES_SIZE, Weight::Regular), MUTED);
                }
            });
    });
    clicks
}

/// The info-card notes of an object (plan table "Honesty labels in M1", info cards; exact
/// strings from `crate::theme`).
pub(crate) fn card_notes(info: &ObjectInfo, details: &ObjectDetails) -> Vec<String> {
    let mut notes: Vec<String> = Vec::new();
    match details {
        ObjectDetails::Body { .. } => {
            notes.extend(
                info.naif
                    .map(body_notes)
                    .unwrap_or_default()
                    .iter()
                    .map(|n| (*n).into()),
            );
        }
        ObjectDetails::Star { star, .. } => {
            if star.flags() & FLAG_NO_RV != 0 {
                notes.push(NOTE_TANGENTIAL_MOTION.into());
            }
        }
        ObjectDetails::Comet { conic_epoch, .. } => {
            notes.push(theme::note_orbit_two_body(&format_date_tdb(*conic_epoch)));
            notes.push(NOTE_TAILS.into());
            notes.push(NOTE_NUCLEUS_SHAPE.into());
        }
        ObjectDetails::Asteroid { record, .. } => {
            notes.push(theme::note_orbit_two_body(&format_date_tdb(epoch_of_jd(
                record.epoch_jd_tdb,
            ))));
        }
        ObjectDetails::ExoHost {
            system,
            star,
            archive_host,
            ..
        } => {
            if *archive_host {
                notes.push(NOTE_ARCHIVE_HOST.into());
            }
            if star.is_some_and(|s| s.flags() & FLAG_NO_RV != 0) {
                notes.push(NOTE_TANGENTIAL_MOTION.into());
            }
            notes.extend(exo_orientation_notes(system).iter().map(|n| (*n).into()));
        }
        ObjectDetails::BlackHole { .. } => notes.push(NOTE_BLACK_HOLE_M3.into()),
    }
    notes
}

/// The notes on an exoplanet system's drawn orbits (ruling 13, lead ruling B.2): the archive
/// has no Ω, so every orbit is drawn with Ω = 0°; a planet without an inclination is drawn
/// edge-on; a planet whose ω is used (given, e ≥ 0.01) reads the archive's ω as the planet's.
fn exo_orientation_notes(system: &ExoSystem) -> Vec<&'static str> {
    let mut notes = Vec::new();
    if system.planets.is_empty() {
        return notes;
    }
    notes.push(NOTE_NODE_UNKNOWN);
    if system.planets.iter().any(|p| p.i_deg.is_none()) {
        notes.push(NOTE_INCLINATION_UNKNOWN);
    }
    if system
        .planets
        .iter()
        .any(|p| p.argp_deg.is_some() && p.e.is_some_and(|e| e >= 0.01))
    {
        notes.push(NOTE_PERIASTRON_CONVENTION);
    }
    notes
}

/// A body's notes by NAIF id: the theory that places it (uni-physics `ephemeris`), and what
/// the renderer approximates on it (plan table).
fn body_notes(naif: i32) -> Vec<&'static str> {
    match naif {
        // The Sun: its granulation (B2).
        10 => vec![NOTE_GRANULATION],
        // Mercury … Neptune: Standish's JPL mean elements.
        299 => vec![NOTE_ORBIT_MEAN_ELEMENTS, NOTE_VENUS_CLOUD_TOPS, NOTE_HAZE],
        399 => vec![NOTE_ORBIT_MEAN_ELEMENTS, NOTE_EARTH_CLOUDS],
        699 => vec![NOTE_ORBIT_MEAN_ELEMENTS, NOTE_RINGS],
        199 | 499 | 599 | 799 | 899 => vec![NOTE_ORBIT_MEAN_ELEMENTS],
        // Pluto (Meeus 37; the world raises its extrapolation flag), the Moon (Meeus 47) and the
        // Galileans (Meeus 44, E5): their own theories, measured under 1 px (A2).
        999 | 301 | 501..=504 => Vec::new(),
        // Saturn's major moons: Meeus 46 (A2 ruling I3); Titan's haze (B3).
        606 => vec![SATURN_MOONS_LABEL, NOTE_HAZE],
        601..=608 => vec![SATURN_MOONS_LABEL],
        // Every other moon: JPL mean elements.
        _ => vec![NOTE_ORBIT_MEAN_ELEMENTS],
    }
}

/// The epoch of a Julian Date (TDB).
fn epoch_of_jd(jd_tdb: f64) -> Epoch {
    /// JD of J2000.0 (IAU 1976: 2451545.0 TDB).
    const JD_J2000: f64 = 2_451_545.0;
    Epoch::from_days_since_j2000(jd_tdb - JD_J2000)
}

/// "Terrestrial planet · Sol IV · real object", "Moon of Mars · real object", …
fn body_subtitle(info: &ObjectInfo, ctx: &Ctx) -> String {
    let naif = info.naif.unwrap_or(0);
    let first = match info.kind {
        ObjectKind::Sun => "Star · G2V".to_owned(),
        ObjectKind::Planet => {
            // Planet NAIF ids are 100·n + 99, n = 1 (Mercury) … 8 (Neptune).
            const ROMAN: [&str; 8] = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"];
            let n = (naif / 100) as usize;
            let class = match naif {
                199 | 299 | 399 | 499 => "Terrestrial planet",
                599 | 699 => "Gas giant",
                _ => "Ice giant",
            };
            match ROMAN.get(n.wrapping_sub(1)) {
                Some(numeral) => format!("{class} · Sol {numeral}"),
                None => class.to_owned(),
            }
        }
        ObjectKind::DwarfPlanet => "Dwarf planet".to_owned(),
        ObjectKind::Moon => match &ctx.parent {
            Some(parent) => format!("Moon of {parent}"),
            None => "Moon".to_owned(),
        },
        _ => String::new(),
    };
    if first.is_empty() {
        "real object".into()
    } else {
        format!("{first} · real object")
    }
}

/// A fact sheet's rows in the mockup's order (Mars artboard); missing values are left out.
fn body_rows(info: &ObjectInfo, f: &BodyFacts) -> Vec<CardRow> {
    let mut rows = vec![
        number_row(
            "Mean radius",
            format!("{} km", format_grouped(f.mean_radius_km, 1)),
        ),
        number_row("Mass", format_mass_kg(f.mass_kg)),
    ];
    if let Some(g) = f.surface_gravity_ms2 {
        rows.push(number_row(
            "Surface gravity",
            format!("{} m/s²", format_sig(g, 3)),
        ));
    }
    if let Some(v) = f.escape_velocity_kms {
        rows.push(number_row(
            "Escape velocity",
            format!("{} km/s", format_sig(v, 3)),
        ));
    }
    if let Some(h) = f.sidereal_rotation_h {
        let label = match info.kind {
            ObjectKind::Planet | ObjectKind::DwarfPlanet => "Sidereal day",
            _ => "Rotation period",
        };
        let period = if h.abs() < 48.0 {
            format_hms(h.abs() * 3600.0)
        } else {
            format_period_days(h.abs() / 24.0)
        };
        let value = if h < 0.0 {
            format!("{period}, retrograde")
        } else {
            period
        };
        rows.push(number_row(label, value));
    }
    if let Some(d) = f.orbital_period_d {
        rows.push(number_row("Orbital period", format_period_days(d)));
    }
    if let Some(a) = f.semi_major_au {
        rows.push(number_row("Semi-major axis", format_distance(a * AU)));
    }
    if let Some(tilt) = f.axial_tilt_deg {
        rows.push(number_row("Axial tilt", format!("{tilt:.2}°")));
    }
    if let Some(p) = f.surface_pressure_kpa {
        rows.push(number_row(
            "Surface pressure",
            format!("{} kPa", format_sig(p, 3)),
        ));
    }
    if let Some(atm) = &f.atmosphere {
        rows.push(word_row("Atmosphere", main_constituent(atm)));
    }
    if let Some(t) = f.mean_temp_k {
        rows.push(number_row(
            "Mean temperature",
            format!("{} K", format_sig(t, 3)),
        ));
    }
    if !f.moons.is_empty() {
        rows.push(word_row("Moons", f.moons.join(", ")));
    }
    rows
}

/// The main constituent of a fact sheet's atmosphere list, as the mockup writes it:
/// "CO2 95.1%, N2 2.59%, …" → "95.1 % CO₂". Anything else is shown as given.
fn main_constituent(list: &str) -> String {
    let first = list.split(", ").next().unwrap_or(list);
    match first.rsplit_once(' ') {
        Some((gas, share)) if share.ends_with('%') && !gas.contains(':') => {
            format!(
                "{} % {}",
                share.trim_end_matches('%'),
                subscript_digits(gas)
            )
        }
        _ => list.to_owned(),
    }
}

/// A moon with JPL mean elements but no fact sheet.
fn sat_rows(sat: &SatRecord) -> Vec<CardRow> {
    let mut rows = Vec::new();
    if let Some(r) = sat.radius_km {
        rows.push(number_row(
            "Mean radius",
            format!("{} km", format_grouped(r, 1)),
        ));
    }
    if let Some(gm) = sat.gm_km3_s2 {
        // m = GM/G, GM in km³/s² (JPL SSD physical parameters), G from CODATA 2018.
        rows.push(number_row("Mass", format_mass_kg(gm * 1e9 / uni_core::G)));
    }
    rows.push(number_row(
        "Orbital period",
        format_period_days(360.0 / sat.n_deg_per_day),
    ));
    rows.push(number_row(
        "Semi-major axis",
        format_distance(sat.a_km * 1e3),
    ));
    rows.push(number_row("Eccentricity", format!("{:.4}", sat.e)));
    rows.push(number_row("Inclination", format!("{:.2}°", sat.i_deg)));
    rows
}

/// Where a body's numbers and orbit come from.
fn body_sources(info: &ObjectInfo, facts: Option<&BodyFacts>, sat: Option<&SatRecord>) -> String {
    let theory = match info.naif.unwrap_or(0) {
        10 => "position: barycentre offset of the Standish planets",
        199 | 299 | 399 | 499 | 599 | 699 | 799 | 899 => "orbit: JPL Keplerian elements (Standish)",
        999 => "orbit: Meeus ch. 37",
        301 => "orbit: Meeus ch. 47 (ELP-2000/82)",
        501..=504 => "orbit: Meeus ch. 44 (Lieske E5)",
        601..=608 => "orbit: Meeus ch. 46",
        _ => "orbit: JPL mean elements",
    };
    let data = match (facts, sat) {
        (Some(f), _) => format!("Data: NASA {} Fact Sheet (NSSDCA)", f.name),
        (None, Some(_)) => "Data: JPL SSD satellite elements and physical parameters".to_owned(),
        (None, None) => "Data: JPL".to_owned(),
    };
    format!("{data} · {theory} · rotation: IAU WGCCRE 2015")
}

/// Where a catalogue star's numbers come from.
const STAR_SOURCES: &str = "Data: AT-HYG v4.0 (D. Nash; CC BY-SA 4.0) · T_eff from B−V \
                            (Ballesteros 2012) · radius from M_V and T_eff (Torres 2010 BC)";

/// "Star · A1V · α CMa · HIP 32349 · real object" (or "· 3 known planets").
fn star_subtitle(info: &ObjectInfo, name: Option<&StarName>, planets: Option<usize>) -> String {
    let mut parts = vec!["Star".to_owned()];
    if let Some(n) = name {
        parts.extend(n.spect.clone());
        parts.extend(n.bayer_flamsteed.clone().filter(|b| b != &info.name));
        parts.extend(n.hip.map(|h| format!("HIP {h}")));
    }
    match planets {
        Some(1) => parts.push("1 known planet".into()),
        Some(k) => parts.push(format!("{k} known planets")),
        None => {}
    }
    parts.push("real object".into());
    parts.join(" · ")
}

/// A star's rows: distance, apparent and absolute V, spectral type, T_eff, luminosity, radius,
/// radial velocity and proper motion.
fn star_rows(
    star: &Star3,
    name: Option<&StarName>,
    teff_k: f64,
    radius_m: f64,
    ctx: &Ctx,
) -> Vec<CardRow> {
    let abs_v = f64::from(star.abs_mag_v);
    let mut rows = Vec::new();
    let dist = ctx.from_sun_m.map(|d| d.length());
    if let Some(d) = dist {
        rows.push(number_row("Distance from Sun", format_distance(d)));
    }
    // Distance modulus m − M = 5 log₁₀(d / 10 pc) (Pogson's magnitudes).
    let apparent = abs_v + 5.0 * (ctx.cam_dist_m / (10.0 * PC)).log10();
    if apparent.is_finite() {
        rows.push(number_row("Apparent V from here", format!("{apparent:.2}")));
    }
    rows.push(number_row("Absolute magnitude M_V", format!("{abs_v:.2}")));
    let spect = name
        .and_then(|n| n.spect.clone())
        .unwrap_or_else(|| "unknown".into());
    rows.push(word_row("Spectral type", spect));
    rows.push(number_row(
        "Effective temperature",
        format!("{} K", format_grouped(teff_k, 0)),
    ));
    // L/L☉ = (R/R☉)² (T/T☉)⁴ (Stefan–Boltzmann), with IAU 2015 B3's nominal R☉ and T☉.
    let r_sun = radius_m / R_SUN_M;
    let lum = r_sun * r_sun * (teff_k / TEFF_SUN_K).powi(4);
    rows.push(number_row(
        "Luminosity",
        format!("{} solar luminosities", format_sig(lum, 3)),
    ));
    rows.push(word_row(
        "Radius",
        format!(
            "{} solar radii · estimated (Torres 2010 BC)",
            format_sig(r_sun, 3)
        ),
    ));
    let flags = star.flags();
    let velocity = star.velocity_kms().as_dvec3();
    let dir = ctx
        .from_sun_m
        .map(DVec3::normalize_or_zero)
        .unwrap_or(DVec3::ZERO);
    let radial = velocity.dot(dir);
    let rv = if flags & (FLAG_NO_RV | FLAG_NO_VELOCITY) != 0 || dir == DVec3::ZERO {
        "unknown".to_owned()
    } else {
        format!("{} km/s", format_grouped(radial, 1))
    };
    rows.push(number_row("Radial velocity", rv));
    // μ ["/yr] = v_t / (A_v d [pc]) with A_v = 1 au per Julian year = 4.740 47 km/s (the
    // Hipparcos A_v, ESA 1997, vol. 1 §1.2.8; here from the IAU 2012 au and 365.25 d).
    let au_per_year_kms = AU / 1e3 / (365.25 * DAY);
    let pm = match dist {
        Some(d) if flags & FLAG_NO_VELOCITY == 0 && d > 0.0 => {
            let tangential = (velocity - dir * radial).length();
            let mas_per_yr = 1e3 * tangential / (au_per_year_kms * d / PC);
            format!("{} mas/yr", format_sig(mas_per_yr, 3))
        }
        _ => "unknown".to_owned(),
    };
    rows.push(number_row("Proper motion", pm));
    rows
}

/// "Periodic comet · real object" and its kin.
fn comet_subtitle(conic: &Conic) -> String {
    /// Comets with periods under 200 yr are "periodic" (IAU naming rules, the P/ prefix).
    const PERIODIC_MAX_S: f64 = 200.0 * 365.25 * DAY;
    let kind = match conic.period_s() {
        Some(p) if p < PERIODIC_MAX_S => "Periodic comet",
        Some(_) => "Long-period comet",
        None => "Comet on an unbound orbit",
    };
    format!("{kind} · real object")
}

/// A comet's rows: q, e, period, next perihelion, r_h, total magnitude from here, nucleus.
fn comet_rows(record: &Comet, conic: &Conic, ctx: &Ctx) -> Vec<CardRow> {
    let mut rows = vec![
        number_row("Perihelion distance", format_distance(conic.q_m)),
        number_row("Eccentricity", format!("{:.4}", conic.e)),
    ];
    let period = conic.period_s();
    rows.push(number_row(
        "Orbital period",
        period.map_or_else(
            || "none (unbound)".to_owned(),
            |p| format_period_days(p / DAY),
        ),
    ));
    let next = match period {
        Some(p) if p > 0.0 => {
            let laps = (ctx.epoch.seconds_since(conic.tp) / p).ceil();
            format_date_tdb(conic.tp.add_seconds(laps * p))
        }
        _ if ctx.epoch.seconds_since(conic.tp) > 0.0 => {
            format!("{} (passed)", format_date_tdb(conic.tp))
        }
        _ => format_date_tdb(conic.tp),
    };
    rows.push(number_row("Next perihelion", next));
    let r_h = ctx.from_sun_m.map(|d| d.length());
    if let Some(r) = r_h {
        rows.push(number_row("Distance from Sun", format_distance(r)));
    }
    // Total magnitude m = M1 + 5 log₁₀ Δ + K1 log₁₀ r [AU] (JPL SBDB's comet parameters).
    let magnitude = match (record.m1, record.k1, r_h) {
        (Some(m1), Some(k1), Some(r)) if ctx.cam_dist_m > 0.0 && r > 0.0 => {
            let m = m1 + 5.0 * (ctx.cam_dist_m / AU).log10() + k1 * (r / AU).log10();
            format!("{m:.1}")
        }
        _ => "unknown".to_owned(),
    };
    rows.push(number_row("Total magnitude from here", magnitude));
    rows.push(number_row(
        "Nucleus diameter",
        record.diameter_km.map_or_else(
            || "unknown".to_owned(),
            |d| format!("{} km", format_sig(d, 3)),
        ),
    ));
    rows
}

/// An asteroid's rows: a, e, i, H, diameter, albedo and the period (Kepler's third law).
fn asteroid_rows(r: &AsteroidRecord) -> Vec<CardRow> {
    let known = |x: f32| x.is_finite().then_some(f64::from(x));
    let a_m = r.a_au * AU;
    // P = 2π √(a³/GM☉) (Kepler's third law; GM☉ the IAU 2015 B3 nominal value).
    let period_d = std::f64::consts::TAU * (a_m * a_m * a_m / GM_SUN).sqrt() / DAY;
    vec![
        number_row("Semi-major axis", format_distance(a_m)),
        number_row("Eccentricity", format!("{:.4}", r.e)),
        number_row("Inclination", format!("{:.2}°", r.i_deg)),
        number_row("Absolute magnitude H", format!("{:.2}", r.h)),
        number_row(
            "Diameter",
            known(r.diameter_km)
                .map_or_else(|| "unknown".into(), |d| format!("{} km", format_sig(d, 3))),
        ),
        number_row(
            "Geometric albedo",
            known(r.albedo).map_or_else(|| "unknown".into(), |p| format!("{p:.3}")),
        ),
        number_row("Orbital period", format_period_days(period_d)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::{
        NOTE_ARCHIVE_HOST, NOTE_EARTH_CLOUDS, NOTE_GRANULATION, NOTE_HAZE,
        NOTE_INCLINATION_UNKNOWN, NOTE_NODE_UNKNOWN, NOTE_NUCLEUS_SHAPE, NOTE_ORBIT_MEAN_ELEMENTS,
        NOTE_RINGS, NOTE_TAILS, NOTE_TANGENTIAL_MOTION, NOTE_VENUS_CLOUD_TOPS, SATURN_MOONS_LABEL,
        note_orbit_two_body,
    };
    use glam::{DQuat, DVec3};
    use uni_core::Epoch;
    use uni_data::{
        BodyFacts, Comet, ExoPlanet, ExoSystem, FLAG_NAMED, FLAG_NO_RV, Star3, StarName, StarNames,
    };
    use uni_sim::registry::ObjectKind;
    use uni_sim::world::WorldData;

    /// Julian Date of J2000.0 (TDB).
    const JD_J2000: f64 = 2_451_545.0;

    fn empty_data() -> WorldData {
        WorldData {
            stars: Vec::new(),
            star_names: None,
            sats: Vec::new(),
            comets: Vec::new(),
            asteroids: Vec::new(),
            asteroid_names: Vec::new(),
            exo: Vec::new(),
            facts: Vec::new(),
            rings: None,
            warnings: Vec::new(),
        }
    }

    /// A fact sheet with the values of `data/baked/body_facts.ron` (NSSDCA fact sheets).
    fn facts(naif: i32, name: &str, moons: &[&str]) -> BodyFacts {
        let (radius, mass, g, esc, rot, period, a, tilt, p, atm, t) = match naif {
            499 => (
                3389.5,
                6.4169e23,
                3.73,
                5.03,
                24.6229,
                686.98,
                1.52366231,
                25.19,
                0.636,
                "CO2 95.1%, N2 2.59%, Ar 1.94%, O2 0.16%, CO 0.06%",
                208.15,
            ),
            _ => (
                6371.0,
                5.9722e24,
                9.82,
                11.186,
                23.9345,
                365.256,
                1.00000011,
                23.44,
                101.4,
                "N2 78.08%, O2 20.95%, Ar 9340 ppm, CO2 420 ppm",
                288.15,
            ),
        };
        BodyFacts {
            naif,
            name: name.into(),
            kind: "planet".into(),
            mean_radius_km: radius,
            mass_kg: mass,
            surface_gravity_ms2: Some(g),
            escape_velocity_kms: Some(esc),
            sidereal_rotation_h: Some(rot),
            orbital_period_d: Some(period),
            semi_major_au: Some(a),
            axial_tilt_deg: Some(tilt),
            surface_pressure_kpa: Some(p),
            atmosphere: Some(atm.into()),
            mean_temp_k: Some(t),
            bond_albedo: None,
            geometric_albedo: None,
            moons: moons.iter().map(|m| (*m).to_owned()).collect(),
            source: format!("https://nssdc.gsfc.nasa.gov/planetary/factsheet/{name}fact.html"),
        }
    }

    fn planet(name: &str, inclination: Option<f64>) -> ExoPlanet {
        ExoPlanet {
            name: name.into(),
            a_au: Some(0.05),
            e: Some(0.0),
            i_deg: inclination,
            argp_deg: None,
            period_d: Some(4.23),
            t_transit_jd: None,
            mass_mjup: Some(0.46),
            radius_rjup: None,
            teq_k: None,
            method: "Radial Velocity".into(),
            year: 1995,
        }
    }

    /// A host not in AT-HYG (`star_index: None`): the world builds its star from the archive.
    fn archive_system(host: &str, planets: Vec<ExoPlanet>) -> ExoSystem {
        ExoSystem {
            host: host.into(),
            star_index: None,
            ra_deg: 10.0,
            dec_deg: 5.0,
            dist_pc: 12.47,
            v_mag: Some(9.0),
            teff_k: Some(3500.0),
            radius_rsun: Some(0.4),
            planets,
        }
    }

    /// A world with Mars's and Earth's sheets, a named star without radial velocity, two
    /// archive hosts (one planet with an inclination, one without) and a comet, at J2000.
    fn test_world() -> World {
        let mut data = empty_data();
        data.facts = vec![
            facts(499, "Mars", &["Phobos", "Deimos"]),
            facts(399, "Earth", &["Moon"]),
        ];
        data.stars = vec![Star3::new(
            [2.0, 0.0, 0.0],
            1.0,
            [10.0, 0.0, 0.0],
            0.6,
            11,
            FLAG_NAMED | FLAG_NO_RV,
        )];
        data.star_names = Some(StarNames {
            entries: vec![StarName {
                index: 0,
                proper: Some("Testar".into()),
                bayer_flamsteed: None,
                con: None,
                hip: None,
                spect: Some("A1V".into()),
            }],
        });
        data.exo = vec![
            archive_system("Host-N", vec![planet("Host-N b", None)]),
            archive_system("Host-I", vec![planet("Host-I b", Some(89.0))]),
        ];
        data.comets = vec![Comet {
            name: "9P/Test".into(),
            spkid: 9_000_009,
            q_au: 1.5,
            e: 0.5,
            i_deg: 10.0,
            node_deg: 0.0,
            argp_deg: 0.0,
            tp_jd_tdb: JD_J2000,
            epoch_jd_tdb: JD_J2000,
            m1: Some(8.0),
            k1: Some(10.0),
            diameter_km: Some(6.0),
        }];
        World::new(data, Epoch::J2000)
    }

    /// A camera `offset_m` from object `id`, moving at `rel_vel_mps` relative to it.
    fn view_near(world: &World, id: ObjectId, offset_m: DVec3, rel_vel_mps: DVec3) -> CameraView {
        CameraView {
            pos: world.position(id).offset(offset_m),
            orient: DQuat::IDENTITY,
            fov_y_rad: 60_f64.to_radians(),
            velocity_mps: world.velocity(id) + rel_vel_mps,
        }
    }

    fn card_of(world: &World, name: &str) -> TargetCard {
        let id = world.find(name).unwrap_or_else(|| panic!("no {name}"));
        let view = view_near(world, id, DVec3::new(1.0e9, 0.0, 0.0), DVec3::ZERO);
        build_card(world, id, &view).unwrap_or_else(|| panic!("no card for {name}"))
    }

    /// The mockup's Mars card: its live rows, then its facts in the mockup's order and format.
    #[test]
    fn mars_card_rows_in_mockup_order() {
        let world = test_world();
        let mars = world.find("Mars").expect("Mars");
        // 14 200 km out on +x, closing at 2.10 km/s (moving −x relative to Mars).
        let view = view_near(
            &world,
            mars,
            DVec3::new(1.42e7, 0.0, 0.0),
            DVec3::new(-2100.0, 0.0, 0.0),
        );
        let card = build_card(&world, mars, &view).expect("Mars card");
        assert_eq!(card.kicker, "TARGET");
        assert_eq!(card.title, "Mars");
        assert_eq!(card.subtitle, "Terrestrial planet · Sol IV · real object");
        let pairs = |rows: &[CardRow]| -> Vec<(String, String)> {
            rows.iter()
                .map(|r| (r.label.clone(), r.value.clone()))
                .collect()
        };
        assert_eq!(
            pairs(&card.live),
            [
                ("Distance".into(), "14 200 km".into()),
                ("Closing speed".into(), "2.10 km/s".into()),
                ("Light time".into(), "0.047 s".into()),
            ]
        );
        let expected: [(&str, &str); 12] = [
            ("Mean radius", "3 389.5 km"),
            ("Mass", "6.417 × 10²³ kg"),
            ("Surface gravity", "3.73 m/s²"),
            ("Escape velocity", "5.03 km/s"),
            // 24.6229 h (the sheet's value) = 88 642.4 s.
            ("Sidereal day", "24 h 37 m 22 s"),
            ("Orbital period", "686.98 d"),
            ("Semi-major axis", "1.524 AU"),
            ("Axial tilt", "25.19°"),
            ("Surface pressure", "0.636 kPa"),
            ("Atmosphere", "95.1 % CO₂"),
            ("Mean temperature", "208 K"),
            ("Moons", "Phobos, Deimos"),
        ];
        let expected: Vec<(String, String)> = expected
            .iter()
            .map(|(l, v)| ((*l).into(), (*v).into()))
            .collect();
        assert_eq!(pairs(&card.facts), expected);
        // Numbers in mono, words not (mockup: "95 % CO₂" and "Phobos, Deimos" in Inter).
        let mono: Vec<bool> = card.facts.iter().map(|r| r.mono).collect();
        assert_eq!(
            mono,
            [
                true, true, true, true, true, true, true, true, true, false, true, false
            ]
        );
        assert_eq!(
            card.actions,
            [CardAction::Goto, CardAction::Moons, CardAction::Close]
        );
        assert_eq!(card.moons, ["Phobos", "Deimos"]);
        assert!(
            card.sources.contains("NASA Mars Fact Sheet"),
            "{}",
            card.sources
        );
        assert!(
            card.notes.iter().any(|n| n == NOTE_ORBIT_MEAN_ELEMENTS),
            "{:?}",
            card.notes
        );
        // An id the world does not know has no card.
        assert!(build_card(&world, ObjectId::MAX, &view).is_none());
    }

    /// Review m11: each kind's card carries its info-card notes, character for character.
    #[test]
    fn card_notes_from_details() {
        let world = test_world();
        let has = |card: &TargetCard, note: &str| card.notes.iter().any(|n| n == note);
        let earth = card_of(&world, "Earth");
        assert!(has(&earth, NOTE_EARTH_CLOUDS), "{:?}", earth.notes);
        let star = card_of(&world, "Testar");
        assert!(has(&star, NOTE_TANGENTIAL_MOTION), "{:?}", star.notes);
        assert!(!has(&star, NOTE_ARCHIVE_HOST), "{:?}", star.notes);
        let host = card_of(&world, "Host-N");
        assert!(has(&host, NOTE_ARCHIVE_HOST), "{:?}", host.notes);
        let comet = card_of(&world, "9P/Test");
        for note in [NOTE_NUCLEUS_SHAPE, NOTE_TAILS] {
            assert!(has(&comet, note), "{note:?} not in {:?}", comet.notes);
        }
        assert!(
            has(&comet, &note_orbit_two_body("2000-01-01")),
            "{:?}",
            comet.notes
        );
        // The Mars card has no Earth note and vice versa.
        assert!(!has(&card_of(&world, "Mars"), NOTE_EARTH_CLOUDS));
    }

    /// Ruling 13: the archive has no Ω, so every exoplanet orbit is drawn with Ω = 0°; a planet
    /// without an inclination is drawn edge-on. Both say so on the card.
    #[test]
    fn exo_card_notes_when_orientation_unknown() {
        let world = test_world();
        let unknown = card_of(&world, "Host-N");
        let both = [NOTE_NODE_UNKNOWN, NOTE_INCLINATION_UNKNOWN];
        for note in both {
            assert!(
                unknown.notes.iter().any(|n| n == note),
                "{note:?} not in {:?}",
                unknown.notes
            );
        }
        assert_eq!(
            both,
            [
                "Orbit orientation on the sky unknown: drawn with Ω = 0°",
                "Inclination unknown: drawn edge-on (i = 90°)",
            ]
        );
        let known_i = card_of(&world, "Host-I");
        assert!(known_i.notes.iter().any(|n| n == NOTE_NODE_UNKNOWN));
        assert!(!known_i.notes.iter().any(|n| n == NOTE_INCLINATION_UNKNOWN));
        // One row per planet.
        assert!(
            unknown.facts.iter().any(|r| r.label == "Host-N b"),
            "{:?}",
            unknown.facts
        );
    }

    /// The star, asteroid and black-hole rows from hand-checkable values.
    #[test]
    fn card_rows_per_kind() {
        use uni_core::{AU, PC, R_SUN_M, TEFF_SUN_K};
        use uni_data::AsteroidRecord;
        let info = |name: &str, kind: ObjectKind| ObjectInfo {
            id: 0,
            name: name.into(),
            kind,
            naif: None,
            catalog_index: None,
            parent: None,
            radii_m: DVec3::ONE,
            gm_m3s2: 1.0,
            frame: None,
            hill_radius_m: 1.0,
            atmosphere_top_m: None,
        };
        let value = |card: &TargetCard, label: &str| -> String {
            card.facts
                .iter()
                .find(|r| r.label == label)
                .map(|r| r.value.clone())
                .unwrap_or_else(|| panic!("no {label:?} in {:?}", card.facts))
        };
        // A Sun twin 10 pc out on +x moving 10 au per Julian year across the line of sight:
        // m = M at 10 pc, L = 1, R = 1, μ = 10 au / 10 pc per year = 1″/yr, no radial velocity.
        let au_per_year_kms = AU / 1e3 / (365.25 * 86_400.0);
        let star = Star3::new(
            [10.0, 0.0, 0.0],
            4.83,
            [0.0, (10.0 * au_per_year_kms) as f32, 0.0],
            0.65,
            1,
            0,
        );
        let name = StarName {
            index: 0,
            proper: Some("Twin".into()),
            bayer_flamsteed: None,
            con: None,
            hip: Some(1),
            spect: Some("G2V".into()),
        };
        let ctx = Ctx {
            epoch: Epoch::J2000,
            cam_dist_m: 10.0 * PC,
            from_sun_m: Some(DVec3::new(10.0 * PC, 0.0, 0.0)),
            parent: None,
        };
        let details = ObjectDetails::Star {
            star,
            name: Some(name),
            teff_k: TEFF_SUN_K,
            radius_m: R_SUN_M,
        };
        let card = describe(&info("Twin", ObjectKind::Star), &details, &ctx);
        assert_eq!(card.subtitle, "Star · G2V · HIP 1 · real object");
        for (label, expected) in [
            ("Distance from Sun", "10.00 pc (32.62 ly)"),
            ("Apparent V from here", "4.83"),
            ("Absolute magnitude M_V", "4.83"),
            ("Spectral type", "G2V"),
            ("Effective temperature", "5 772 K"),
            ("Luminosity", "1.00 solar luminosities"),
            ("Radius", "1.00 solar radii · estimated (Torres 2010 BC)"),
            ("Radial velocity", "0.0 km/s"),
            // The f16 velocity (47.406 km/s) is 3e-5 above 1 au/yr.
            ("Proper motion", "1 000 mas/yr"),
        ] {
            assert_eq!(value(&card, label), expected, "{label}");
        }
        assert!(card.notes.is_empty(), "{:?}", card.notes);
        // An asteroid at a = 2.7666 AU (Ceres): P = 2π √(a³/GM☉) = 4.60 yr.
        let ceres = AsteroidRecord {
            a_au: 2.7666,
            e: 0.0785,
            i_deg: 10.59,
            node_deg: 80.3,
            argp_deg: 73.6,
            m0_deg: 0.0,
            epoch_jd_tdb: 2_461_000.5,
            h: 3.34,
            g: 0.12,
            diameter_km: 939.4,
            albedo: f32::NAN,
            spkid: 20_000_001,
            name_index: 0,
        };
        let details = ObjectDetails::Asteroid {
            record: ceres,
            name: Some("1 Ceres (A801 AA)".into()),
        };
        let card = describe(&info("Ceres", ObjectKind::DwarfPlanet), &details, &ctx);
        assert_eq!(
            card.subtitle,
            "Dwarf planet · 1 Ceres (A801 AA) · real object"
        );
        assert_eq!(value(&card, "Semi-major axis"), "2.767 AU");
        assert_eq!(value(&card, "Orbital period"), "4.60 yr");
        assert_eq!(value(&card, "Diameter"), "939 km");
        assert_eq!(value(&card, "Geometric albedo"), "unknown");
        assert_eq!(card.notes, [note_orbit_two_body("2025-11-21")]);
        // TestBH-1: 10 solar masses, r_s = 2GM/c² = 29.5 km, labelled hypothetical in amber.
        let details = ObjectDetails::BlackHole {
            mass_kg: 10.0 * uni_core::M_SUN,
            spin: 0.0,
        };
        let card = describe(&info("TestBH-1", ObjectKind::BlackHole), &details, &ctx);
        assert_eq!(card.kicker, "TARGET · BLACK HOLE");
        assert_eq!(value(&card, "Mass"), "10.0 solar masses");
        assert_eq!(value(&card, "Schwarzschild radius"), "29.5 km");
        let status = card
            .facts
            .iter()
            .find(|r| r.label == "Status")
            .expect("status");
        assert_eq!(status.tone, LabelTone::NonPhysical);
        assert_eq!(card.notes, [crate::theme::NOTE_BLACK_HOLE_M3]);
        assert_eq!(card.actions, [CardAction::Goto, CardAction::Close]);
    }

    /// The per-body notes of the plan table, from the object alone.
    #[test]
    fn body_notes_per_body() {
        let info = |naif: i32, kind: ObjectKind| ObjectInfo {
            id: 0,
            name: format!("{naif}"),
            kind,
            naif: Some(naif),
            catalog_index: None,
            parent: None,
            radii_m: DVec3::ONE,
            gm_m3s2: 1.0,
            frame: None,
            hill_radius_m: 1.0,
            atmosphere_top_m: None,
        };
        let body = ObjectDetails::Body {
            facts: None,
            sat: None,
        };
        let notes = |naif: i32, kind: ObjectKind| card_notes(&info(naif, kind), &body);
        assert_eq!(notes(10, ObjectKind::Sun), [NOTE_GRANULATION]);
        assert_eq!(
            notes(299, ObjectKind::Planet),
            [NOTE_ORBIT_MEAN_ELEMENTS, NOTE_VENUS_CLOUD_TOPS, NOTE_HAZE]
        );
        assert_eq!(
            notes(699, ObjectKind::Planet),
            [NOTE_ORBIT_MEAN_ELEMENTS, NOTE_RINGS]
        );
        assert_eq!(
            notes(606, ObjectKind::Moon),
            [SATURN_MOONS_LABEL, NOTE_HAZE]
        );
        assert_eq!(notes(601, ObjectKind::Moon), [SATURN_MOONS_LABEL]);
        // JPL mean elements for the moons without their own theory; none for the Moon (Meeus
        // 47), the Galileans (E5) or Pluto (Meeus 37).
        assert_eq!(notes(401, ObjectKind::Moon), [NOTE_ORBIT_MEAN_ELEMENTS]);
        assert!(notes(301, ObjectKind::Moon).is_empty());
        assert!(notes(501, ObjectKind::Moon).is_empty());
        assert!(notes(999, ObjectKind::DwarfPlanet).is_empty());
    }
}
