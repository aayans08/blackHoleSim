//! The target card (Task C4; Task C0 wrote the types and the `build_card` stub).
//!
//! Closed by default (owner point 4): the "i" button or I opens it; while a goto runs it
//! describes the destination (kicker "En route to"). Rows per `uni_sim::world::ObjectDetails`
//! kind (plan C4); the notes are the info-card notes of `crate::theme` (exact strings).

use uni_core::CameraView;
use uni_sim::registry::ObjectId;
use uni_sim::world::World;

use crate::theme::LabelTone;

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
    let _ = (world, id, view);
    todo!("C4")
}
