//! World-state honesty labels (M1 Task A4). The exact strings are in the M1 plan's table
//! "Honesty labels in M1"; [`crate::world::World::honesty`] raises the conditions.

/// A world-state condition the HUD must label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HonestyFlag {
    /// The epoch is outside 3000 BC – AD 3000, Standish's Table 2 window (amber).
    PlanetsExtrapolated,
    /// The epoch is outside Meeus ch. 37's 1885–2099 while Pluto or Charon is on screen or
    /// selected (amber).
    PlutoExtrapolated,
    /// The epoch is more than 100 000 yr from the star catalogue's (amber).
    StarsExtrapolated,
    /// The Milky Way band is drawn from galaxy model v0 (grey).
    BandModel,
    /// The procedural asteroid belt is drawn and visible; the app raises it (C6) (grey).
    ProceduralBelt,
}

/// The exact on-screen text of `f`.
pub fn label(f: HonestyFlag) -> &'static str {
    match f {
        HonestyFlag::PlanetsExtrapolated => {
            "Planet positions extrapolated outside 3000 BC \u{2013} AD 3000 (approximate)"
        }
        HonestyFlag::PlutoExtrapolated => {
            "Pluto position extrapolated outside 1885\u{2013}2099 (approximate)"
        }
        HonestyFlag::StarsExtrapolated => {
            "Star positions: linear motion beyond \u{b1}100 000 yr (approximate)"
        }
        HonestyFlag::BandModel => "Milky Way band: galaxy model v0 (real dust map in M4)",
        HonestyFlag::ProceduralBelt => "Faint asteroids: procedural, from catalogue statistics",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The M1 plan's table "Honesty labels in M1", character for character (en dashes U+2013,
    /// the plus-minus sign U+00B1, an ASCII space in "100 000").
    #[test]
    fn labels_are_the_plan_strings() {
        let expected = [
            (
                HonestyFlag::PlanetsExtrapolated,
                "Planet positions extrapolated outside 3000 BC \u{2013} AD 3000 (approximate)",
            ),
            (
                HonestyFlag::PlutoExtrapolated,
                "Pluto position extrapolated outside 1885\u{2013}2099 (approximate)",
            ),
            (
                HonestyFlag::StarsExtrapolated,
                "Star positions: linear motion beyond \u{b1}100 000 yr (approximate)",
            ),
            (
                HonestyFlag::BandModel,
                "Milky Way band: galaxy model v0 (real dust map in M4)",
            ),
            (
                HonestyFlag::ProceduralBelt,
                "Faint asteroids: procedural, from catalogue statistics",
            ),
        ];
        for (flag, text) in expected {
            assert_eq!(label(flag), text, "{flag:?}");
        }
    }
}
