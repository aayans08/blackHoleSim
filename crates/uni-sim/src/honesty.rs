//! World-state honesty labels (M1 Task A4). The exact strings are in the M1 plan's table
//! "Honesty labels in M1". A0b stub; A4 implements.

/// A world-state condition the HUD must label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HonestyFlag {
    PlanetsExtrapolated,
    PlutoExtrapolated,
    StarsExtrapolated,
    BandModel,
    ProceduralBelt,
}

/// The exact on-screen text of `f`.
pub fn label(_f: HonestyFlag) -> &'static str {
    todo!("A4")
}
