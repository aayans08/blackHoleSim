//! Overlays drawn under the HUD panels: the target bracket, destination markers and orbit lines
//! (Task C5; Task C0 wrote the types and stubs).
//!
//! [`build_overlay`] turns the snapshot into points [pt] (internal pixels × `output_pt /
//! internal_px`); [`draw_overlay`] paints them. C4's HUD calls `draw_overlay` inside a
//! full-screen, non-interactive `egui::Area` at `egui::Order::Background`, so the overlay sits
//! under the panels and markers never take clicks (ruling C.7). Which objects carry a marker is
//! uni-sim's `Snapshot::destinations` (lead ruling B.3); labels are drawn only where they do not
//! overlap. The aim and the goto target come in as plain values, so uni-ui needs no uni-nav.

pub mod bracket;
pub mod markers;
pub mod orbits;

pub use bracket::Bracket;
pub use markers::{Marker, MarkerKind};
pub use orbits::{OrbitLine, exo_orbit_points};

use glam::{UVec2, Vec2};
use uni_core::CameraView;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::Snapshot;
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
    let _ = (
        world,
        snap,
        view,
        internal_px,
        output_pt,
        selected,
        aim,
        goto_target,
        s,
    );
    todo!("C5")
}

/// Paints `o` with `ui`'s painter. Task C0 stub: draws nothing (C4's HUD calls it before C5
/// lands, so it must not panic).
pub fn draw_overlay(ui: &egui::Ui, o: &Overlay) {
    let _ = (ui, o);
}
