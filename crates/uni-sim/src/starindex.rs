//! Nearest-star queries over the catalogue (M1 Task A4). A0b stub; A4 implements.

use glam::DVec3;
use uni_data::Star3;

/// A uniform grid of 4 pc cells over the catalogue stars.
pub struct StarIndex {
    /// Placeholder for the private grid; A4 replaces it.
    _private: (),
}

impl StarIndex {
    pub fn new(_stars: &[Star3]) -> Self {
        todo!("A4")
    }

    /// The `k` stars nearest `p_pc` [pc, ICRS] at `years` after the catalogue epoch, as
    /// (index, distance [pc]), nearest first.
    pub fn nearest(
        &self,
        _stars: &[Star3],
        _p_pc: DVec3,
        _years: f64,
        _k: usize,
    ) -> Vec<(u32, f64)> {
        todo!("A4")
    }
}
