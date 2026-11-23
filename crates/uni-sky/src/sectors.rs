//! Star sectors (Task B4; B0 stub): 16-pc cubes, each with an f64 origin, so the GPU works with
//! f32 offsets from a nearby origin and stays precise far from the SSB (design §5.2; Review
//! Focus 5).

use glam::{DVec3, IVec3};

/// Edge of a star sector [pc] (design §5.2).
pub const SECTOR_SIZE_PC: f64 = 16.0;

/// The sector holding the ICRS position `pos_pc` [pc, origin SSB]: floor(pos / 16 pc) per axis.
pub fn sector_of(pos_pc: DVec3) -> IVec3 {
    let _ = pos_pc;
    todo!("B4")
}
