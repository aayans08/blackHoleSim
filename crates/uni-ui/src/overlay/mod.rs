//! Overlays drawn under the HUD panels: the target bracket, destination markers and orbit lines
//! (Task C5; Task C0 wrote the types and stubs).
//!
//! [`build_overlay`] turns the snapshot into points [pt] (internal pixels × `output_pt /
//! internal_px`); [`draw_overlay`] paints them. C4's HUD calls `draw_overlay` inside a
//! full-screen, non-interactive `egui::Area` at `egui::Order::Background`, so the overlay sits
//! under the panels and markers never take clicks (ruling C.7). Which objects carry a marker is
//! uni-sim's `Snapshot::destinations` (lead ruling B.3); labels are drawn only where they do not
//! overlap. The aim and the goto target come in as plain values, so uni-ui needs no uni-nav.
//!
//! Every position is projected in f64 from camera-relative positions, exactly as uni-sim's
//! snapshot projects them to internal pixels, then scaled per axis to output points; radii use
//! the vertical scale. Paint order: orbits, the bracket, then the markers on top.

pub mod bracket;
pub mod markers;
pub mod orbits;

pub use bracket::Bracket;
pub use markers::{Marker, MarkerKind};
pub use orbits::{OrbitLine, exo_orbit_points};

use glam::{DQuat, DVec2, DVec3, UVec2, Vec2};
use markers::{Candidate, Rules};
use uni_core::CameraView;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{Destination, Snapshot, proj_radius_px};
use uni_sim::world::World;

/// Everything the overlay draws this frame, in output points.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    /// The selected object's corner brackets.
    pub bracket: Option<Bracket>,
    pub markers: Vec<Marker>,
    pub orbits: Vec<OrbitLine>,
}

/// What the overlay shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlaySettings {
    /// The planets' orbits about the Sun (and the moons' about the selected planet).
    pub planet_orbits: bool,
    /// Marker labels.
    pub labels: bool,
    /// Exoplanet orbits around the selected host, or a host within 0.05 pc.
    pub exo_orbits: bool,
}

/// The camera's projection from camera-relative positions [m, root axes] to output points [pt]:
/// to internal pixels as uni-sim's snapshot projects (origin top left, y down; pixel i spans
/// [i, i + 1), as `pixel_dir` in `common/camera.wgsl`), then × `output_pt / internal_px` per
/// axis (the tonemap's upscale of the internal frame to the output).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScreenMap {
    to_camera: DQuat,
    tan_half_y: f64,
    aspect: f64,
    internal: DVec2,
    /// Output points per internal pixel, per axis.
    scale: DVec2,
    /// The output frame [pt].
    pub output: DVec2,
}

impl ScreenMap {
    pub(crate) fn new(view: &CameraView, internal_px: UVec2, output_pt: Vec2) -> Self {
        let internal = internal_px.max(UVec2::ONE).as_dvec2();
        let output = output_pt.as_dvec2();
        ScreenMap {
            to_camera: view.orient.normalize().conjugate(),
            tan_half_y: (0.5 * view.fov_y_rad).tan(),
            aspect: internal.x / internal.y,
            internal,
            scale: output / internal,
            output,
        }
    }

    /// `rel` [m, root axes] in camera axes (the camera looks along −Z, +Y up).
    pub(crate) fn camera(&self, rel: DVec3) -> DVec3 {
        self.to_camera * rel
    }

    /// Where camera-space point `c` lands [pt], inside the frame or not; `None` unless it is in
    /// front of the camera (c.z < 0).
    pub(crate) fn project(&self, c: DVec3) -> Option<DVec2> {
        if !(c.z < 0.0) {
            return None;
        }
        let ndc = DVec2::new(c.x / self.aspect, c.y) / (-c.z * self.tan_half_y);
        let px = DVec2::new(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * self.internal;
        Some(px * self.scale)
    }

    /// Output points per internal pixel along y, for radii.
    pub(crate) fn radius_scale(&self) -> f64 {
        self.scale.y
    }

    /// Whether any of the sphere at camera-space `c` with radius `r` [m] may be in the frame:
    /// false only when it lies wholly outside the cone around the view frustum (half-angle to
    /// the frame's corner). Always true with the camera inside it, or for NaN.
    pub(crate) fn may_see_sphere(&self, c: DVec3, r: f64) -> bool {
        let d = c.length();
        if !(d > r) {
            return true;
        }
        let corner = (self.tan_half_y * (1.0 + self.aspect * self.aspect).sqrt()).atan();
        let off_axis = DVec2::new(c.x, c.y).length().atan2(-c.z);
        off_axis - (r / d).asin() <= corner
    }
}

/// Internal pixel position `p` as output points: `p × output_pt / internal_px` per axis.
pub(crate) fn internal_to_output(p: Vec2, internal_px: UVec2, output_pt: Vec2) -> Vec2 {
    (p.as_dvec2() * output_pt.as_dvec2() / internal_px.max(UVec2::ONE).as_dvec2()).as_vec2()
}

/// Whether the object at `rel` [m from the camera] is hidden behind one of `bodies` (id, centre
/// [m from the camera], radius [m]) nearer the camera than it: within the body's angular
/// radius asin(R/d) of its centre. An object never hides itself.
pub(crate) fn behind_a_body(id: ObjectId, rel: DVec3, bodies: &[(ObjectId, DVec3, f64)]) -> bool {
    let d = rel.length();
    bodies.iter().any(|&(body, centre, radius)| {
        let d_body = centre.length();
        // The angle between the two directions, from atan2 so it stays exact when tiny.
        let apart = rel.cross(centre).length().atan2(rel.dot(centre));
        body != id && d_body < d && apart < (radius / d_body).min(1.0).asin()
    })
}

/// The place the camera is at: the nearest surface, when the camera is within
/// [`markers::AT_PLACE_RADII`] of its largest radii of its centre.
fn place_at(world: &World, snap: &Snapshot) -> Option<ObjectId> {
    let (id, altitude) = snap.nearest_surface?;
    let radius = world.objects().get(id as usize)?.radii_m.max_element();
    (altitude + radius <= markers::AT_PLACE_RADII * radius).then_some(id)
}

/// The marker rules' view of destination `d` (`map`'s camera; `bodies` as for [`behind_a_body`]).
fn candidate(
    world: &World,
    d: &Destination,
    map: &ScreenMap,
    internal_px: UVec2,
    bodies: &[(ObjectId, DVec3, f64)],
) -> Candidate {
    let center_pt = d
        .screen_px
        .map(|p| internal_to_output(p, internal_px, map.output.as_vec2()));
    let c = map.camera(d.rel_pos_m);
    // An object behind the camera: far outside the frame, along its direction on the image
    // plane (x right, y down), straight down when it is dead behind.
    let toward_pt = center_pt.unwrap_or_else(|| {
        let dir = DVec2::new(c.x, -c.y).try_normalize().unwrap_or(DVec2::Y);
        (0.5 * map.output + dir * 4.0 * map.output.max_element()).as_vec2()
    });
    Candidate {
        id: d.id,
        kind: d.kind,
        name: world
            .objects()
            .get(d.id as usize)
            .map_or_else(String::new, |info| info.name.clone()),
        dist_m: d.dist_m,
        center_pt,
        toward_pt,
        disk_pt: (d.proj_radius_px * map.radius_scale()) as f32,
        // The angle from the view axis (camera −Z), from atan2 so it is exact near 0 and 180°.
        off_axis_rad: DVec2::new(c.x, c.y).length().atan2(-c.z) as f32,
        occluded: behind_a_body(d.id, d.rel_pos_m, bodies),
    }
}

/// The bracket on `id` when its centre is in front of the camera and inside the frame, unless
/// the camera is at it.
fn bracket_for(
    world: &World,
    snap: &Snapshot,
    view: &CameraView,
    map: &ScreenMap,
    internal_px: UVec2,
    id: ObjectId,
) -> Option<Bracket> {
    let info = world.objects().get(id as usize)?;
    let rel = snap
        .destinations
        .iter()
        .find(|d| d.id == id)
        .map_or_else(|| world.position(id).delta_from(view.pos), |d| d.rel_pos_m);
    let dist = rel.length();
    let center = map.project(map.camera(rel))?;
    if !(center.x >= 0.0 && center.y >= 0.0 && center.x <= map.output.x && center.y <= map.output.y)
    {
        return None;
    }
    let radius_px = proj_radius_px(
        info.radii_m.max_element(),
        dist,
        view.fov_y_rad,
        f64::from(internal_px.y.max(1)),
    );
    // Range rate d|r|/dt = r̂ · (v_object − v_camera); closing is its negative.
    let closing = -(rel / dist).dot(world.velocity(id) - view.velocity_mps);
    Some(Bracket {
        center_pt: center.as_vec2(),
        half_size_pt: bracket::bracket_half_size((radius_px * map.radius_scale()) as f32),
        name: info.name.clone(),
        sub: bracket::bracket_sub(dist, closing),
    })
}

/// The overlay for `snap` seen from `view`, at internal size `internal_px` [px] shown on
/// `output_pt` [pt]. `selected` gets the bracket, `aim` = (candidate, ring fill 0 ..= 1) its aim
/// ring, `goto_target` the target styling.
#[allow(clippy::too_many_arguments)]
pub fn build_overlay(
    world: &World,
    snap: &Snapshot,
    view: &CameraView,
    internal_px: UVec2,
    output_pt: Vec2,
    selected: Option<ObjectId>,
    aim: Option<(ObjectId, f32)>,
    goto_target: Option<ObjectId>,
    s: &OverlaySettings,
) -> Overlay {
    let map = ScreenMap::new(view, internal_px, output_pt);
    let at = place_at(world, snap);
    let bracket = selected
        .filter(|&id| Some(id) != at)
        .and_then(|id| bracket_for(world, snap, view, &map, internal_px, id));
    // The resolved bodies and star surfaces that hide what lies behind them.
    let bodies: Vec<(ObjectId, DVec3, f64)> = snap
        .bodies
        .iter()
        .map(|b| (b.id, b.rel_pos_m, b.radii_m.max_element()))
        .chain(
            snap.star_surfaces
                .iter()
                .map(|s| (s.id, s.rel_pos_m, s.radius_m)),
        )
        .collect();
    let cands: Vec<Candidate> = snap
        .destinations
        .iter()
        .map(|d| candidate(world, d, &map, internal_px, &bodies))
        .collect();
    let rules = Rules {
        selected,
        aim,
        goto_target,
        at,
        labels: s.labels,
        frame_pt: output_pt,
        bracket_shown: bracket.is_some(),
        reserved: bracket.iter().map(bracket::label_rect).collect(),
    };
    let markers = markers::layout(&cands, &rules);
    let mut orbits = Vec::new();
    orbits::solar_system_orbits(
        world,
        snap,
        view.pos,
        &map,
        selected,
        s.planet_orbits,
        &mut orbits,
    );
    if s.exo_orbits {
        orbits::exo_orbits(world, snap, view.pos, &map, selected, &mut orbits);
    }
    Overlay {
        bracket,
        markers,
        orbits,
    }
}

/// Paints `o` with `ui`'s painter: orbits, then the bracket, then the markers. The frame is the
/// context's content rectangle, the output the overlay was built for.
pub fn draw_overlay(ui: &egui::Ui, o: &Overlay) {
    let painter = ui.painter();
    orbits::draw(painter, &o.orbits);
    if let Some(b) = &o.bracket {
        bracket::draw(painter, b);
    }
    markers::draw(painter, ui.ctx().content_rect(), &o.markers);
}

#[cfg(test)]
mod tests {
    use egui::{Pos2, Shape};
    use uni_core::{AU, Epoch};
    use uni_sim::world::WorldData;

    use super::*;
    use crate::theme::LabelTone;

    const INTERNAL: UVec2 = UVec2::new(1662, 1080);
    const OUTPUT: Vec2 = Vec2::new(1470.0, 956.0);

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

    fn distance_to_lines<'a>(lines: impl Iterator<Item = &'a OrbitLine>, p: Vec2) -> f32 {
        let mut best = f32::INFINITY;
        for line in lines {
            for w in line.pts.windows(2) {
                let (a, b) = (w[0], w[1]);
                let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
                best = best.min(p.distance(a + (b - a) * t));
            }
        }
        best
    }

    /// Plan C5 step 1: internal 1662×1080 shown on 1470×956 pt maps internal (831, 540) to
    /// (735, 478); a point straight ahead projects there.
    #[test]
    fn overlay_scales_internal_to_output() {
        assert_eq!(
            internal_to_output(Vec2::new(831.0, 540.0), INTERNAL, OUTPUT),
            Vec2::new(735.0, 478.0)
        );
        assert_eq!(
            internal_to_output(Vec2::new(1662.0, 0.0), INTERNAL, OUTPUT),
            Vec2::new(1470.0, 0.0)
        );
        let view = CameraView {
            pos: uni_core::UniversePos::default(),
            orient: DQuat::from_rotation_y(0.7),
            fov_y_rad: 60f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let map = ScreenMap::new(&view, INTERNAL, OUTPUT);
        let ahead = view.orient * DVec3::new(0.0, 0.0, -1.0e12);
        let p = map.project(map.camera(ahead)).expect("in front");
        assert!((p - DVec2::new(735.0, 478.0)).length() < 1e-9, "{p}");
        assert!(map.project(map.camera(-ahead)).is_none(), "behind");
        // The top edge of the field (30° up) lands on y = 0.
        let up = view.orient * DVec3::new(0.0, 30f64.to_radians().tan(), -1.0);
        assert!(map.project(map.camera(up)).expect("in front").y.abs() < 1e-9);
        assert!((map.radius_scale() - 956.0 / 1080.0).abs() < 1e-15);
    }

    /// The orbit cull keeps any sphere that reaches into the cone around the frustum (corner
    /// half-angle atan(tan 30° · √(1 + (1662/1080)²)) = 46.66°) and drops one wholly outside.
    #[test]
    fn sphere_cull_keeps_what_may_show() {
        let view = CameraView {
            pos: uni_core::UniversePos::default(),
            orient: DQuat::IDENTITY,
            fov_y_rad: 60f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let map = ScreenMap::new(&view, INTERNAL, OUTPUT);
        assert!(
            map.may_see_sphere(DVec3::new(0.0, 0.0, -10.0), 1.0),
            "ahead"
        );
        assert!(
            !map.may_see_sphere(DVec3::new(0.0, 0.0, 10.0), 1.0),
            "behind"
        );
        assert!(
            map.may_see_sphere(DVec3::new(0.0, 0.0, 10.0), 20.0),
            "camera inside"
        );
        let corner =
            (30f64.to_radians().tan() * (1.0 + (1662.0f64 / 1080.0).powi(2)).sqrt()).atan();
        assert!((corner.to_degrees() - 46.66).abs() < 0.01);
        // A sphere centred 2° beyond the corner direction, of angular radius 1° and 3°.
        let dir = |deg: f64| DVec3::new(deg.to_radians().sin(), 0.0, -deg.to_radians().cos());
        let beyond = dir(corner.to_degrees() + 2.0) * 100.0;
        assert!(!map.may_see_sphere(beyond, 100.0 * 1f64.to_radians().sin()));
        assert!(map.may_see_sphere(beyond, 100.0 * 3f64.to_radians().sin()));
        assert!(map.may_see_sphere(DVec3::NAN, 1.0), "NaN is never culled");
    }

    /// Behind a nearer body within its angular radius: hidden; beside it, in front of it, or the
    /// body itself: not.
    #[test]
    fn hidden_only_behind_a_nearer_body() {
        let bodies = [(9, DVec3::new(0.0, 0.0, -1.0e9), 1.0e7)];
        assert!(behind_a_body(1, DVec3::new(0.0, 0.0, -2.0e9), &bodies));
        assert!(
            behind_a_body(1, DVec3::new(1.9e7, 0.0, -2.0e9), &bodies),
            "0.0095 rad < asin 0.01"
        );
        assert!(
            !behind_a_body(1, DVec3::new(2.1e7, 0.0, -2.0e9), &bodies),
            "0.0105 rad > asin 0.01"
        );
        assert!(
            !behind_a_body(1, DVec3::new(0.0, 0.0, -0.5e9), &bodies),
            "in front"
        );
        assert!(
            !behind_a_body(9, DVec3::new(0.0, 0.0, -1.0e9), &bodies),
            "itself"
        );
    }

    /// The real Solar System from 6 AU above the Sun, Mars selected: the bracket sits on Mars's
    /// projected centre, Mars's marker is cyan and unlabelled (the bracket names it), Mars's
    /// orbit is cyan and passes through it, Earth sits on a grey orbit; planet orbits off leaves
    /// only the selected orbit.
    #[test]
    fn overlay_of_the_real_solar_system() {
        let epoch = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").expect("date");
        let world = World::new(empty_data(), epoch);
        let id = |name: &str| world.find(name).unwrap_or_else(|| panic!("{name}"));
        let (sun, mars, earth) = (id("Sun"), id("Mars"), id("Earth"));
        // The ecliptic north pole in ICRS (obliquity 23.439 281°, IAU 2006 at J2000).
        let eps = 23.439_281f64.to_radians();
        let pole = DVec3::new(0.0, -eps.sin(), eps.cos());
        let view = CameraView {
            pos: world.position(sun).offset(pole * 6.0 * AU),
            orient: DQuat::from_rotation_arc(DVec3::NEG_Z, -pole),
            fov_y_rad: 60f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let snap = world.snapshot(&view, INTERNAL);
        let all = OverlaySettings {
            planet_orbits: true,
            labels: true,
            exo_orbits: true,
        };
        let o = build_overlay(
            &world,
            &snap,
            &view,
            INTERNAL,
            OUTPUT,
            Some(mars),
            None,
            None,
            &all,
        );

        let dest = snap
            .destinations
            .iter()
            .find(|d| d.id == mars)
            .expect("Mars is a destination");
        let mars_pt = internal_to_output(dest.screen_px.expect("in front"), INTERNAL, OUTPUT);
        let b = o.bracket.as_ref().expect("Mars bracketed");
        assert!(b.center_pt.distance(mars_pt) < 1e-3);
        assert_eq!((b.name.as_str(), b.half_size_pt), ("Mars", 20.0));
        let marker = |id| o.markers.iter().find(|m| m.id == id);
        let mars_m = marker(mars).expect("Mars marked");
        assert!(mars_m.selected && mars_m.label.is_none());
        assert_eq!(mars_m.pt, mars_pt);
        let sun_m = marker(sun).expect("Sun marked");
        assert_eq!(
            (sun_m.kind, sun_m.label.as_deref()),
            (MarkerKind::Star, Some("Sun"))
        );
        let earth_m = marker(earth).expect("Earth marked");
        assert_eq!(earth_m.kind, MarkerKind::Planet);

        assert!(o.orbits.iter().all(|l| l.pts.len() >= 2 && !l.dashed));
        for p in o.orbits.iter().flat_map(|l| &l.pts) {
            assert!(
                p.is_finite()
                    && p.x >= -1e-3
                    && p.y >= -1e-3
                    && p.x <= OUTPUT.x + 1e-3
                    && p.y <= OUTPUT.y + 1e-3
            );
        }
        let nav = || o.orbits.iter().filter(|l| l.tone == LabelTone::Nav);
        assert!(nav().count() >= 1);
        assert!(
            distance_to_lines(nav(), mars_pt) < 0.5,
            "{}",
            distance_to_lines(nav(), mars_pt)
        );
        let grey = o.orbits.iter().filter(|l| l.tone == LabelTone::Neutral);
        assert!(distance_to_lines(grey, earth_m.pt) < 0.5);

        let selected_only = OverlaySettings {
            planet_orbits: false,
            ..all
        };
        let o = build_overlay(
            &world,
            &snap,
            &view,
            INTERNAL,
            OUTPUT,
            Some(mars),
            None,
            None,
            &selected_only,
        );
        assert!(!o.orbits.is_empty() && o.orbits.iter().all(|l| l.tone == LabelTone::Nav));
        let o = build_overlay(
            &world,
            &snap,
            &view,
            INTERNAL,
            OUTPUT,
            None,
            None,
            None,
            &selected_only,
        );
        assert!(o.orbits.is_empty() && o.bracket.is_none());
    }

    /// Plan C5 "done when" on the CPU side, with every baked file: TRAPPIST-1 selected from
    /// 0.2 AU on the Sun's side shows its seven planets' orbits, all dashed (the archive has no
    /// Ω), seven planet dots, its bracket and its ring-and-dot marker; unselected, a host within
    /// 0.05 pc still shows them.
    #[test]
    fn trappist1_shows_seven_dashed_orbits() {
        let data = uni_sim::world::load_world_data(&uni_data::data_dir().join("baked"));
        let epoch = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").expect("date");
        let world = World::new(data, epoch);
        let host = world.find("TRAPPIST-1").expect("TRAPPIST-1 registered");
        let host_pos = world.position(host);
        let dir = host_pos
            .delta_from(uni_core::UniversePos::default())
            .normalize();
        let view = CameraView {
            pos: host_pos.offset(-dir * 0.2 * AU),
            orient: DQuat::from_rotation_arc(DVec3::NEG_Z, dir),
            fov_y_rad: 60f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let snap = world.snapshot(&view, INTERNAL);
        let exo_only = OverlaySettings {
            planet_orbits: false,
            labels: true,
            exo_orbits: true,
        };
        let o = build_overlay(
            &world,
            &snap,
            &view,
            INTERNAL,
            OUTPUT,
            Some(host),
            None,
            None,
            &exo_only,
        );
        let lines = o.orbits.iter().filter(|l| l.pts.len() >= 2).count();
        let dots = o.orbits.iter().filter(|l| l.pts.len() == 1).count();
        assert!(lines >= 7, "{lines} orbit lines");
        assert_eq!(dots, 7);
        assert!(
            o.orbits
                .iter()
                .all(|l| l.dashed && l.tone == LabelTone::Neutral)
        );
        let b = o.bracket.as_ref().expect("host bracketed");
        assert!(
            b.center_pt.distance(Vec2::new(735.0, 478.0)) < 0.5,
            "{}",
            b.center_pt
        );
        let marker = o
            .markers
            .iter()
            .find(|m| m.id == host)
            .expect("host marked");
        assert_eq!((marker.kind, marker.selected), (MarkerKind::ExoHost, true));
        // Within 0.05 pc and not selected: still drawn.
        let o = build_overlay(
            &world, &snap, &view, INTERNAL, OUTPUT, None, None, None, &exo_only,
        );
        assert_eq!(o.orbits.iter().filter(|l| l.pts.len() == 1).count(), 7);
    }

    /// Prints `build_overlay`'s CPU time with every baked file loaded (the registry's full object
    /// list is scanned for planets and moons), planet and exoplanet orbits on and Saturn selected
    /// (its eight major moons' orbits too), from 0.05 AU off Saturn
    /// (`cargo test -p uni-ui --lib -- --ignored --nocapture overlay_build_cost`).
    #[test]
    #[ignore = "print-only timing"]
    fn overlay_build_cost() {
        let data = uni_sim::world::load_world_data(&uni_data::data_dir().join("baked"));
        let epoch = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").expect("date");
        let world = World::new(data, epoch);
        let saturn = world.find("Saturn").expect("Saturn");
        let target = world.position(saturn);
        let pos = target.offset(DVec3::new(0.03, 0.02, 0.03) * AU);
        let view = CameraView {
            pos,
            orient: DQuat::from_rotation_arc(DVec3::NEG_Z, target.delta_from(pos).normalize()),
            fov_y_rad: 60f64.to_radians(),
            velocity_mps: DVec3::ZERO,
        };
        let snap = world.snapshot(&view, INTERNAL);
        let all = OverlaySettings {
            planet_orbits: true,
            labels: true,
            exo_orbits: true,
        };
        let n = 50;
        let t0 = std::time::Instant::now();
        let mut o = Overlay::default();
        for _ in 0..n {
            o = build_overlay(
                &world,
                &snap,
                &view,
                INTERNAL,
                OUTPUT,
                Some(saturn),
                None,
                None,
                &all,
            );
        }
        let ms = t0.elapsed().as_secs_f64() / f64::from(n) * 1e3;
        let points: usize = o.orbits.iter().map(|l| l.pts.len()).sum();
        println!(
            "build_overlay: {ms:.3} ms per call; {} objects, {} destinations, {} markers, \
             {} orbit lines ({points} points)",
            world.objects().len(),
            snap.destinations.len(),
            o.markers.len(),
            o.orbits.len()
        );
        let map = ScreenMap::new(&view, INTERNAL, OUTPUT);
        let per = |f: &mut dyn FnMut()| {
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            t0.elapsed().as_secs_f64() / f64::from(n) * 1e3
        };
        let mut out = Vec::new();
        let planets = per(&mut || {
            out.clear();
            orbits::solar_system_orbits(
                &world,
                &snap,
                view.pos,
                &map,
                Some(saturn),
                true,
                &mut out,
            );
        });
        let selected_only = per(&mut || {
            out.clear();
            orbits::solar_system_orbits(
                &world,
                &snap,
                view.pos,
                &map,
                Some(saturn),
                false,
                &mut out,
            );
        });
        let exo = per(&mut || {
            out.clear();
            orbits::exo_orbits(&world, &snap, view.pos, &map, Some(saturn), &mut out);
        });
        let velocity = per(&mut || {
            std::hint::black_box(world.velocity(saturn));
        });
        let scan = per(&mut || {
            let moons = world.objects().iter().filter(|o| {
                o.kind == uni_sim::registry::ObjectKind::Moon && o.parent == Some(saturn)
            });
            std::hint::black_box(moons.count());
        });
        println!(
            "split: planet orbits {planets:.3} ms (selected only {selected_only:.3}), exo \
             {exo:.3} ms, one World::velocity {velocity:.4} ms, registry scan {scan:.3} ms"
        );
    }

    /// Every shape painted, collected.
    fn painted(out: &egui::FullOutput) -> Vec<Shape> {
        fn walk(shape: &Shape, acc: &mut Vec<Shape>) {
            match shape {
                Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, acc)),
                s => acc.push(s.clone()),
            }
        }
        let mut acc = Vec::new();
        for clipped in &out.shapes {
            walk(&clipped.shape, &mut acc);
        }
        acc
    }

    /// The overlay of `draw_overlay_paints_every_part`: a bracket on Mars, Mars's selected marker,
    /// Earth aimed at, an off-screen comet, Jupiter's label-only marker over its large disk, and
    /// a solid orbit, a dashed one and an exoplanet dot.
    fn every_part() -> Overlay {
        let marker = |id, pt: Vec2, label: Option<&str>, kind| Marker {
            id,
            pt,
            glyph: true,
            label: label.map(str::to_owned),
            kind,
            selected: false,
            sub: label.map(|_| "1.000 AU".to_owned()),
            aim_fill: 0.0,
            target: false,
            off_axis_rad: 0.1,
        };
        Overlay {
            bracket: Some(Bracket {
                center_pt: Vec2::new(400.0, 300.0),
                half_size_pt: 40.0,
                name: "Mars".into(),
                sub: "1.524 AU \u{b7} closing 2.10 km/s".into(),
            }),
            markers: vec![
                Marker {
                    selected: true,
                    ..marker(1, Vec2::new(400.0, 300.0), None, MarkerKind::Planet)
                },
                Marker {
                    aim_fill: 0.6,
                    ..marker(
                        2,
                        Vec2::new(800.0, 500.0),
                        Some("Earth"),
                        MarkerKind::Planet,
                    )
                },
                marker(3, Vec2::new(-3000.0, 478.0), None, MarkerKind::Comet),
                Marker {
                    glyph: false,
                    ..marker(
                        4,
                        Vec2::new(1000.0, 200.0),
                        Some("Jupiter"),
                        MarkerKind::Planet,
                    )
                },
            ],
            orbits: vec![
                OrbitLine {
                    pts: vec![
                        Vec2::new(100.0, 100.0),
                        Vec2::new(300.0, 120.0),
                        Vec2::new(500.0, 200.0),
                    ],
                    dashed: false,
                    tone: LabelTone::Nav,
                },
                OrbitLine {
                    pts: vec![Vec2::new(100.0, 700.0), Vec2::new(600.0, 700.0)],
                    dashed: true,
                    tone: LabelTone::Neutral,
                },
                OrbitLine {
                    pts: vec![Vec2::new(650.0, 650.0)],
                    dashed: true,
                    tone: LabelTone::Neutral,
                },
            ],
        }
    }

    /// Paints `o` in a fresh egui context of the output's size, with the HUD's fonts when
    /// `hud_fonts` (as `hud.rs`'s context has them), and returns every shape.
    fn paint(o: &Overlay, hud_fonts: bool) -> Vec<Shape> {
        let ctx = egui::Context::default();
        if hud_fonts {
            ctx.set_fonts(crate::theme::font_definitions());
        }
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(OUTPUT.x, OUTPUT.y),
            )),
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw, |ui| draw_overlay(ui, o));
        // No GPU here to upload to; epaint asserts that deltas are not silently dropped.
        out.textures_delta.clear();
        painted(&out)
    }

    /// Every text painted, with the font family of its first section.
    fn texts(shapes: &[Shape]) -> Vec<(String, egui::FontFamily)> {
        shapes
            .iter()
            .filter_map(|s| match s {
                Shape::Text(t) => Some((
                    t.galley.text().to_owned(),
                    t.galley.job.sections[0].format.font_id.family.clone(),
                )),
                _ => None,
            })
            .collect()
    }

    /// `draw_overlay` paints each part where the overlay puts it: the bracket's capitalised name,
    /// a marker's label, a solid orbit as one path, a dashed one as many dashes, the exoplanet
    /// dot, an off-screen marker as a chevron on the inner frame's left edge, and a label-only
    /// marker as its label with no glyph. With the HUD's fonts, the bracket's name is Inter
    /// SemiBold (brief: 600), a marker's name Inter Medium (lead ruling for C5: 500), both
    /// sub-lines IBM Plex Mono Regular (review r0 I2).
    #[test]
    fn draw_overlay_paints_every_part() {
        use crate::theme::{FONT_MONO, FONT_TEXT_MEDIUM, FONT_TEXT_SEMIBOLD};
        use egui::FontFamily;
        let shapes = paint(&every_part(), true);
        let texts = texts(&shapes);
        let named = |f: &str| FontFamily::Name(f.into());
        for (want, family) in [
            ("MARS", named(FONT_TEXT_SEMIBOLD)),
            ("1.524 AU \u{b7} closing 2.10 km/s", named(FONT_MONO)),
            ("Earth", named(FONT_TEXT_MEDIUM)),
            ("1.000 AU", named(FONT_MONO)),
            ("Jupiter", named(FONT_TEXT_MEDIUM)),
        ] {
            assert!(
                texts.iter().any(|(t, f)| t == want && *f == family),
                "{want:?} in {family:?} not in {texts:?}"
            );
        }
        // The label-only marker: its label, but no glyph (no circle) at its centre.
        let jupiter = Pos2::new(1000.0, 200.0);
        assert!(
            !shapes
                .iter()
                .any(|s| matches!(s, Shape::Circle(c) if c.center == jupiter)),
            "no glyph over a large disk"
        );
        let earth = Pos2::new(800.0, 500.0);
        assert!(
            shapes
                .iter()
                .any(|s| matches!(s, Shape::Circle(c) if c.center == earth && c.radius == 4.2)),
            "a glyph-marker keeps its ring"
        );
        let solid = shapes.iter().any(|s| match s {
            Shape::Path(p) => p.points.len() == 3 && p.points[0] == Pos2::new(100.0, 100.0),
            _ => false,
        });
        assert!(solid, "solid orbit as one path");
        let dashes = shapes
            .iter()
            .filter(|s| matches!(s, Shape::LineSegment { points, .. } if points[0].y == 700.0))
            .count();
        assert!(dashes > 20, "dashed orbit: {dashes} dashes");
        let dot = shapes.iter().any(|s| match s {
            Shape::Circle(c) => c.center == Pos2::new(650.0, 650.0) && c.fill.a() > 0,
            _ => false,
        });
        assert!(dot, "exoplanet dot");
        // The chevron: on the inner frame's left edge (14 pt) plus its inset, at the marker's
        // height.
        let chevron = shapes.iter().any(|s| match s {
            Shape::Path(p) => {
                p.points.len() == 3
                    && p.points
                        .iter()
                        .all(|q| q.x > 14.0 && q.x < 34.0 && (q.y - 478.0).abs() < 8.0)
            }
            _ => false,
        });
        assert!(chevron, "edge chevron");
        // Nothing is painted for a marker's glyph outside the frame.
        assert!(
            shapes
                .iter()
                .all(|s| s.visual_bounding_rect().min.x > -100.0)
        );
    }

    /// In a context without the HUD's fonts the overlay still draws (egui panics on a family it
    /// has no font for): the names fall back to egui's proportional family, the sub-lines to its
    /// monospace one.
    #[test]
    fn draw_overlay_without_hud_fonts_falls_back() {
        use egui::FontFamily;
        let texts = texts(&paint(&every_part(), false));
        for (want, family) in [
            ("MARS", FontFamily::Proportional),
            ("1.524 AU \u{b7} closing 2.10 km/s", FontFamily::Monospace),
            ("Earth", FontFamily::Proportional),
            ("1.000 AU", FontFamily::Monospace),
        ] {
            assert!(
                texts.iter().any(|(t, f)| t == want && *f == family),
                "{want:?} in {family:?} not in {texts:?}"
            );
        }
    }

    /// An egui context with the HUD's fonts, after one pass (fonts take effect at a pass's
    /// start), for measuring real text.
    pub(super) fn fonts_context() -> egui::Context {
        let ctx = egui::Context::default();
        ctx.set_fonts(crate::theme::font_definitions());
        let mut out = ctx.run_ui(egui::RawInput::default(), |_| {});
        out.textures_delta.clear();
        ctx
    }
}
