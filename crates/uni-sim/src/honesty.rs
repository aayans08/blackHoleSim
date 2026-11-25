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
    /// The band's clumpy dust, star clouds and the shapes of its named dark clouds are procedural
    /// (uni-procgen galaxy model "Detail", Task B4L); raised with [`HonestyFlag::BandModel`]
    /// while the band is on screen (grey).
    MilkyWayDetailProcedural,
    /// The procedural asteroid belt is drawn and visible; the app raises it (C6) (grey).
    ProceduralBelt,
    /// The selected comet, or a comet the last snapshot drew, is still being re-osculated
    /// toward the shown date after a date change (amber).
    CometsCatchingUp,
    /// The epoch is outside the Meeus moon theories' J2000 ± 3000 yr (about 1000 BC – AD 5000)
    /// and the Moon, the Galileans or Saturn's eight major moons run on their JPL mean elements:
    /// the Moon's phase and node, and Saturn's moons' angles, are effectively unknown there, and
    /// the moons (and the Earth, placed from the Moon) jump at the window's edges (final review
    /// physics I1; uni-physics `EphemFlag::MoonTheoryFallback`) (amber).
    MoonsMeanElements,
    /// Outside that window, moons without mean elements (no `sat_elements.ron`) stay at their
    /// window-edge positions and stop orbiting (uni-physics `EphemFlag::MoonsFrozen`) (amber).
    MoonsFrozen,
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
        HonestyFlag::MilkyWayDetailProcedural => {
            "Milky Way detail: procedural (real dust map in M4)"
        }
        HonestyFlag::ProceduralBelt => "Faint asteroids: procedural, from catalogue statistics",
        HonestyFlag::CometsCatchingUp => {
            "Comet positions catching up after a date change (approximate)"
        }
        HonestyFlag::MoonsMeanElements => {
            "Moon positions: mean elements outside 1000 BC \u{2013} AD 5000 (approximate)"
        }
        HonestyFlag::MoonsFrozen => {
            "Moon positions frozen outside 1000 BC \u{2013} AD 5000 (no mean elements loaded)"
        }
    }
}

/// Whether `f` labels something non-physical or extrapolated (amber) rather than an
/// approximate-but-physical model (neutral grey), per the colour column of the M1 plan's table
/// "Honesty labels in M1" (Task C0, A4r minor "no severity map"): amber for the three
/// extrapolation flags and [`HonestyFlag::CometsCatchingUp`] (lead ruling A4 open 1), and for
/// the two moon flags (final review physics I1: positions effectively unknown, or frozen), grey
/// for the band model, the band's procedural detail and the procedural belt.
pub fn is_non_physical(f: HonestyFlag) -> bool {
    match f {
        HonestyFlag::PlanetsExtrapolated
        | HonestyFlag::PlutoExtrapolated
        | HonestyFlag::StarsExtrapolated
        | HonestyFlag::CometsCatchingUp
        | HonestyFlag::MoonsMeanElements
        | HonestyFlag::MoonsFrozen => true,
        HonestyFlag::BandModel
        | HonestyFlag::MilkyWayDetailProcedural
        | HonestyFlag::ProceduralBelt => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Task C0 carry-over 3: the tone of every flag is the plan table's colour (amber = true).
    #[test]
    fn honesty_tones_follow_the_plan_table() {
        let amber = [
            HonestyFlag::PlanetsExtrapolated,
            HonestyFlag::PlutoExtrapolated,
            HonestyFlag::StarsExtrapolated,
            HonestyFlag::CometsCatchingUp,
            HonestyFlag::MoonsMeanElements,
            HonestyFlag::MoonsFrozen,
        ];
        let grey = [
            HonestyFlag::BandModel,
            HonestyFlag::MilkyWayDetailProcedural,
            HonestyFlag::ProceduralBelt,
        ];
        for f in amber {
            assert!(is_non_physical(f), "{f:?} is amber");
        }
        for f in grey {
            assert!(!is_non_physical(f), "{f:?} is grey");
        }
    }

    /// The M1 plan's table "Honesty labels in M1", character for character (en dashes U+2013,
    /// the plus-minus sign U+00B1, an ASCII space in "100 000"), plus Task B4L's
    /// `MilkyWayDetailProcedural` (its string from the B4L context).
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
                HonestyFlag::MilkyWayDetailProcedural,
                "Milky Way detail: procedural (real dust map in M4)",
            ),
            (
                HonestyFlag::ProceduralBelt,
                "Faint asteroids: procedural, from catalogue statistics",
            ),
            (
                HonestyFlag::CometsCatchingUp,
                "Comet positions catching up after a date change (approximate)",
            ),
            // Final review physics I1 (not in the plan's table; the lead's triage asked for it).
            (
                HonestyFlag::MoonsMeanElements,
                "Moon positions: mean elements outside 1000 BC \u{2013} AD 5000 (approximate)",
            ),
            (
                HonestyFlag::MoonsFrozen,
                "Moon positions frozen outside 1000 BC \u{2013} AD 5000 (no mean elements loaded)",
            ),
        ];
        for (flag, text) in expected {
            assert_eq!(label(flag), text, "{flag:?}");
        }
    }
}
