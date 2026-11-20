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
        todo!()
    }

    /// Moves whole cells out of `local` so each |local| component is ≤ CELL_SIZE_M / 2.
    pub fn normalized(self) -> Self {
        todo!()
    }

    /// Adds a displacement in metres; the result is normalized.
    pub fn offset(self, d: DVec3) -> Self {
        todo!()
    }

    /// Returns `self − origin` in metres.
    pub fn delta_from(self, origin: UniversePos) -> DVec3 {
        todo!()
    }
}
