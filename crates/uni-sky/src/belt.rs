//! The procedural asteroid belt (Task C2; Task C0 stub): "drawn from the catalogue's statistical
//! distributions" (design §5.4).
//!
//! 3-D histograms of (a, e, i) for the main belt (1.8–5.5 AU) and the Kuiper belt (30–50 AU)
//! from the real records, sampled with `splitmix64`; Jupiter Trojans with λ − λ_J drawn from the
//! real Trojans around ±60°; H > `h_cut` from N(<H) ∝ 10^(αH) with α fitted to the catalogue's
//! own counts in [h_cut − 2, h_cut] (Jedicke & Metcalfe 1998), so no procedural object
//! duplicates a catalogue one. Labelled `HonestyFlag::ProceduralBelt` (grey) while drawn and
//! visible. Catalogue facts for C2's arithmetic (counted from `data/baked/asteroids.bin` for the
//! C0 context): 33 678 records (H ≤ 14), 19 463 in the main belt 2.1–3.3 AU, 6 095 in 30–50 AU.

use uni_data::AsteroidRecord;

/// Procedural main-belt objects (ruling 11: 200 k main + 50 k Kuiper is the starting point; C6
/// lowers it if the pass breaks its budget).
pub const BELT_MAIN_COUNT: u32 = 200_000;
/// Procedural Kuiper-belt objects (ruling 11).
pub const BELT_KUIPER_COUNT: u32 = 50_000;

/// The sampled procedural elements.
#[derive(Clone, Debug)]
pub struct BeltModel {
    _private: (),
}

impl BeltModel {
    /// Samples `n_main` main-belt and `n_kuiper` Kuiper-belt objects (plus the Trojans) fainter
    /// than `h_cut` from the distributions of `real`, deterministically from `seed`.
    pub fn from_catalogue(
        real: &[AsteroidRecord],
        n_main: u32,
        n_kuiper: u32,
        h_cut: f32,
        seed: u64,
    ) -> Self {
        let _ = (real, n_main, n_kuiper, h_cut, seed);
        todo!("C2")
    }

    /// Number of procedural objects.
    pub fn len(&self) -> usize {
        todo!("C2")
    }

    /// No procedural objects.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
