//! Reference-frame tree: frames placed relative to a parent, resolved to the root (ICRS axes, SSB origin).

use glam::{DQuat, DVec3};

use crate::epoch::Epoch;
use crate::pos::UniversePos;

/// Index of a frame in a [`FrameTree`].
pub type FrameId = u32;

/// The root frame: ICRS axes, origin at the Solar System barycentre.
pub const ROOT_FRAME: FrameId = 0;

/// Time-dependent placement of a frame relative to its parent: (origin, frame→parent rotation).
pub type MotionFn = Box<dyn Fn(Epoch) -> (UniversePos, DQuat) + Send + Sync>;

/// How a frame moves relative to its parent.
pub enum FrameMotion {
    /// Constant origin and rotation, relative to the parent.
    Fixed { origin: UniversePos, rotation: DQuat },
    /// Arbitrary motion relative to the parent; used by uni-physics later.
    Custom(MotionFn),
}

/// A tree of named frames rooted at [`ROOT_FRAME`].
pub struct FrameTree {
    // Placeholder; M0 Task 3 defines the real storage.
    _private: (),
}

impl FrameTree {
    /// Creates a tree holding only the root frame.
    pub fn new() -> Self {
        todo!()
    }

    /// Adds a child of `parent` and returns its id.
    pub fn add(&mut self, parent: FrameId, name: &str, motion: FrameMotion) -> FrameId {
        todo!()
    }

    /// Looks a frame up by name.
    pub fn find(&self, name: &str) -> Option<FrameId> {
        todo!()
    }

    /// The frame's origin in the root frame at time `t`.
    pub fn origin_in_root(&self, id: FrameId, t: Epoch) -> UniversePos {
        todo!()
    }

    /// The rotation taking this frame's axes to root axes at time `t`.
    pub fn rotation_to_root(&self, id: FrameId, t: Epoch) -> DQuat {
        todo!()
    }

    /// Converts a point given in metres in frame `id` to a root-frame position at time `t`.
    pub fn to_root(&self, id: FrameId, local_m: DVec3, t: Epoch) -> UniversePos {
        todo!()
    }
}
