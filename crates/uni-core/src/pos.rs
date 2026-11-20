//! Universe-scale positions: an integer cell plus an f64 offset inside the cell.

use glam::{DVec3, I64Vec3};

/// Edge length of one position cell: 2^44 m (about 117.6 au).
pub const CELL_SIZE_M: f64 = 17_592_186_044_416.0;

/// A position in the root frame: `cell * CELL_SIZE_M + local`, in metres.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct UniversePos {
    pub cell: I64Vec3,
    pub local: DVec3,
}

impl UniversePos {
    /// Builds a normalized position from absolute metres.
    pub fn from_meters(m: DVec3) -> Self {
        UniversePos {
            cell: I64Vec3::ZERO,
            local: m,
        }
        .normalized()
    }

    /// Moves whole cells out of `local` so each |local| component is ≤ CELL_SIZE_M / 2.
    ///
    /// Exact: `local / CELL_SIZE_M` is a power-of-two scaling, and subtracting the nearest whole
    /// cell loses no bits. Past ±2^63 cells (about ±1.6e32 m) the cell saturates instead of
    /// overflowing.
    pub fn normalized(self) -> Self {
        let whole_cells = (self.local / CELL_SIZE_M).round();
        UniversePos {
            cell: self.cell.saturating_add(whole_cells.as_i64vec3()),
            local: self.local - whole_cells * CELL_SIZE_M,
        }
    }

    /// Adds a displacement in metres; the result is normalized.
    pub fn offset(self, d: DVec3) -> Self {
        UniversePos {
            cell: self.cell,
            local: self.local + d,
        }
        .normalized()
    }

    /// Returns `self − origin` in metres.
    pub fn delta_from(self, origin: UniversePos) -> DVec3 {
        (self.cell - origin.cell).as_dvec3() * CELL_SIZE_M + (self.local - origin.local)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::PC;

    #[test]
    fn from_meters_roundtrip_1pc() {
        let p = UniversePos::from_meters(DVec3::new(PC, -PC, 0.5));
        assert!((p.delta_from(UniversePos::default()) - DVec3::new(PC, -PC, 0.5)).length() < 1e-3);
    }

    #[test]
    fn offset_precision_far_from_origin() {
        // Review Focus 5. At 1 Gpc, `far.local.x` is 772·2^32 m ≈ 3.3e12 m, where f64 values are
        // 2^-11 m apart. A 1 mm step therefore lands within half that spacing (≈0.24 mm) and no
        // closer (2.34e-5 m here). The brief's 1e-6 m bound is guaranteed only while the stepped
        // |local| (`local + step`, not `local`) stays below 2^34 m: from 2^34 m on, values are
        // 2^-18 m apart and one rounding can cost 1.9e-6 m, even for a `local` just under 2^34 m
        // that the step carries across. We assert the one-rounding bound and 0.1 mm, which is
        // "mm-precise" with a 10× margin over an absolute-metres `offset` (it drops the step, an
        // error of the full 1 mm). `offset_precision_across_cell_far_from_origin` sweeps the
        // same bound across the cell.
        let far = UniversePos::from_meters(DVec3::new(1e9 * PC, 0.0, 0.0));
        let near = far.offset(DVec3::new(0.001, 0.0, 0.0));
        let half_spacing = 0.5 * (far.local.x.abs().next_up() - far.local.x.abs());
        let err = (near.delta_from(far).x - 0.001).abs();
        assert!(err <= half_spacing && err < 1e-4, "error {err} m");
        // A step that is a multiple of the local spacing comes back exactly.
        let step = DVec3::new(0.5, -0.25, 1.0);
        assert_eq!(far.offset(step).delta_from(far), step);
    }

    #[test]
    fn offset_precision_across_cell_far_from_origin() {
        // Review Focus 5, swept across one cell 1 Gpc from the origin. A step's error depends
        // only on |local| (the distance to the cell centre), never on the distance from the
        // origin: it is at most one f64 rounding at |local|. That guarantees the brief's 1e-6 m
        // bound whenever the stepped |local| is below 2^34 m (≈0.11 au), so it is asserted for
        // every sample with k ≤ 33. Out to the CELL_SIZE_M / 2 = 2^43 m edge, f64 values are at
        // most 2^-10 m apart, so the error stays under 0.5 mm. The sweep stops at 2^42 m, because
        // a step from exactly ±2^43 m can cross into the next cell.
        let cell = UniversePos::from_meters(DVec3::new(1e9 * PC, 0.0, 0.0)).cell;
        let step = DVec3::new(0.001, 0.0, 0.0);
        for k in 0..=42 {
            for sign in [1.0, -1.0] {
                let far = UniversePos {
                    cell,
                    local: DVec3::new(sign * 2_f64.powi(k), 0.0, 0.0),
                };
                let err = (far.offset(step).delta_from(far).x - step.x).abs();
                let half_spacing = 0.5 * (far.local.x.abs().next_up() - far.local.x.abs());
                assert!(
                    err <= half_spacing && err < 1e-3,
                    "|local| 2^{k} m: error {err} m"
                );
                if k <= 33 {
                    assert!(err < 1e-6, "|local| 2^{k} m: error {err} m");
                }
            }
        }
    }

    #[test]
    fn normalized_local_within_half_cell() {
        for sign in [1.0, -1.0] {
            let d = DVec3::splat(sign * 3.7 * CELL_SIZE_M);
            let p = UniversePos::default().offset(d);
            assert!(p.local.abs().max_element() <= CELL_SIZE_M / 2.0, "{p:?}");
            assert_eq!(p.cell, I64Vec3::splat(4 * sign as i64));
            assert_eq!(p.delta_from(UniversePos::default()), d);
        }
    }

    #[test]
    fn delta_from_spans_cells() {
        let a = UniversePos {
            cell: I64Vec3::new(5, -3, 0),
            local: DVec3::new(1.0, 2.0, 3.0),
        };
        let b = UniversePos {
            cell: I64Vec3::new(2, -3, 1),
            local: DVec3::new(0.5, 0.0, -1.0),
        };
        let expected = DVec3::new(3.0 * CELL_SIZE_M + 0.5, 2.0, -CELL_SIZE_M + 4.0);
        assert_eq!(a.delta_from(b), expected);
    }

    #[test]
    fn offset_beyond_cell_range_does_not_panic() {
        // ±2^63 cells is about ±1.6e32 m; past that the cell saturates instead of overflowing.
        let p = UniversePos::default().offset(DVec3::splat(1e40));
        assert_eq!(p.cell, I64Vec3::splat(i64::MAX));
        assert!(p.local.is_finite());
    }
}
