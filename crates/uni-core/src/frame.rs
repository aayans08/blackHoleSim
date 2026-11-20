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
    Fixed {
        origin: UniversePos,
        rotation: DQuat,
    },
    /// Arbitrary motion relative to the parent; used by uni-physics later.
    Custom(MotionFn),
}

impl FrameMotion {
    /// (origin in the parent, frame→parent rotation) at time `t`.
    fn at(&self, t: Epoch) -> (UniversePos, DQuat) {
        match self {
            FrameMotion::Fixed { origin, rotation } => (*origin, *rotation),
            FrameMotion::Custom(motion) => motion(t),
        }
    }
}

/// A tree of named frames rooted at [`ROOT_FRAME`].
pub struct FrameTree {
    /// Indexed by [`FrameId`]. A parent always has a smaller id than its children, so the
    /// tree cannot contain a cycle.
    frames: Vec<Frame>,
}

struct Frame {
    name: String,
    /// `None` only for the root.
    parent: Option<FrameId>,
    motion: FrameMotion,
}

impl FrameTree {
    /// Creates a tree holding only the root frame, named `"root"`.
    pub fn new() -> Self {
        let root = Frame {
            name: "root".to_owned(),
            parent: None,
            motion: FrameMotion::Fixed {
                origin: UniversePos::default(),
                rotation: DQuat::IDENTITY,
            },
        };
        FrameTree { frames: vec![root] }
    }

    /// Adds a child of `parent` and returns its id.
    ///
    /// # Panics
    /// If `parent` is not a frame of this tree.
    pub fn add(&mut self, parent: FrameId, name: &str, motion: FrameMotion) -> FrameId {
        assert!(
            (parent as usize) < self.frames.len(),
            "FrameTree has no frame with id {parent}"
        );
        let id = self.frames.len() as FrameId;
        self.frames.push(Frame {
            name: name.to_owned(),
            parent: Some(parent),
            motion,
        });
        id
    }

    /// Looks a frame up by name. If several frames share a name, the first one added wins.
    pub fn find(&self, name: &str) -> Option<FrameId> {
        self.frames
            .iter()
            .position(|f| f.name == name)
            .map(|i| i as FrameId)
    }

    /// The frame's origin in the root frame at time `t`.
    pub fn origin_in_root(&self, id: FrameId, t: Epoch) -> UniversePos {
        self.resolve(id, t).0
    }

    /// The rotation taking this frame's axes to root axes at time `t`.
    pub fn rotation_to_root(&self, id: FrameId, t: Epoch) -> DQuat {
        self.resolve(id, t).1
    }

    /// Converts a point given in metres in frame `id` to a root-frame position at time `t`.
    pub fn to_root(&self, id: FrameId, local_m: DVec3, t: Epoch) -> UniversePos {
        let (origin, rotation) = self.resolve(id, t);
        origin.offset(rotation * local_m)
    }

    /// (origin in root, frame→root rotation) of frame `id` at `t`. Walks up to the root once,
    /// evaluating each ancestor's motion a single time.
    fn resolve(&self, id: FrameId, t: Epoch) -> (UniversePos, DQuat) {
        let frame = self.frame(id);
        let Some(parent) = frame.parent else {
            return frame.motion.at(t);
        };
        let (parent_origin, parent_rotation) = self.resolve(parent, t);
        let (origin, rotation) = frame.motion.at(t);
        let origin_in_root = if parent_rotation == DQuat::IDENTITY {
            // Root-aligned parent: add cell by cell, so origins far from the root keep full
            // precision instead of going through f64 metres.
            UniversePos {
                cell: parent_origin.cell.saturating_add(origin.cell),
                local: parent_origin.local + origin.local,
            }
            .normalized()
        } else {
            // A rotated parent (e.g. a spinning body) rotates the child's offset in f64 metres,
            // which keeps about 16 significant digits of it.
            parent_origin.offset(parent_rotation * origin.delta_from(UniversePos::default()))
        };
        (origin_in_root, parent_rotation * rotation)
    }

    fn frame(&self, id: FrameId) -> &Frame {
        self.frames
            .get(id as usize)
            .unwrap_or_else(|| panic!("FrameTree has no frame with id {id}"))
    }
}

impl Default for FrameTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{AU, PC};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn fixed(origin_m: DVec3, rotation: DQuat) -> FrameMotion {
        FrameMotion::Fixed {
            origin: UniversePos::from_meters(origin_m),
            rotation,
        }
    }

    fn meters(p: UniversePos) -> DVec3 {
        p.delta_from(UniversePos::default())
    }

    #[test]
    fn frame_tree_fixed_child() {
        let mut tree = FrameTree::new();
        let quarter_turn = DQuat::from_rotation_z(FRAC_PI_2);
        let child = tree.add(
            ROOT_FRAME,
            "child",
            fixed(DVec3::new(AU, 0.0, 0.0), quarter_turn),
        );
        let p = tree.to_root(child, DVec3::X, Epoch::J2000);
        assert!(
            (meters(p) - DVec3::new(AU, 1.0, 0.0)).length() < 1e-6,
            "{p:?}"
        );
    }

    #[test]
    fn new_tree_holds_only_the_root() {
        let tree = FrameTree::default();
        let t = Epoch::J2000;
        assert_eq!(tree.find("root"), Some(ROOT_FRAME));
        assert_eq!(tree.find("earth"), None);
        assert_eq!(tree.origin_in_root(ROOT_FRAME, t), UniversePos::default());
        assert_eq!(tree.rotation_to_root(ROOT_FRAME, t), DQuat::IDENTITY);
    }

    #[test]
    fn nested_frames_compose() {
        let mut tree = FrameTree::new();
        let quarter_turn = DQuat::from_rotation_z(FRAC_PI_2);
        let a = tree.add(
            ROOT_FRAME,
            "a",
            fixed(DVec3::new(AU, 0.0, 0.0), quarter_turn),
        );
        let b = tree.add(a, "b", fixed(DVec3::new(0.0, 2.0, 0.0), quarter_turn));
        let t = Epoch::J2000;
        assert_eq!(tree.find("b"), Some(b));
        // b sits 2 m along a's +y, which is root −x; b's axes are turned 180° about z.
        let origin = meters(tree.origin_in_root(b, t));
        assert!(
            (origin - DVec3::new(AU - 2.0, 0.0, 0.0)).length() < 1e-6,
            "{origin}"
        );
        assert!(
            tree.rotation_to_root(b, t)
                .abs_diff_eq(DQuat::from_rotation_z(PI), 1e-12)
        );
        let p = meters(tree.to_root(b, DVec3::X, t));
        assert!((p - DVec3::new(AU - 3.0, 0.0, 0.0)).length() < 1e-6, "{p}");
    }

    #[test]
    fn custom_motion_follows_the_epoch() {
        let mut tree = FrameTree::new();
        let probe = tree.add(
            ROOT_FRAME,
            "probe",
            FrameMotion::Custom(Box::new(|t: Epoch| {
                let x = 1000.0 * t.seconds_since(Epoch::J2000);
                (
                    UniversePos::from_meters(DVec3::new(x, 0.0, 0.0)),
                    DQuat::IDENTITY,
                )
            })),
        );
        let t = Epoch::J2000.add_seconds(10.0);
        assert_eq!(
            meters(tree.origin_in_root(probe, t)),
            DVec3::new(10_000.0, 0.0, 0.0)
        );
    }

    #[test]
    fn unrotated_parents_keep_full_precision() {
        // 8 kpc as plain f64 metres has 32 km spacing; frames with root-aligned axes must add
        // cell by cell and keep the 0.25 m that a round trip through f64 metres would drop.
        let galactic = UniversePos::from_meters(DVec3::new(8_000.0 * PC, 0.0, 0.0))
            .offset(DVec3::new(0.25, 0.0, 0.0));
        let mut tree = FrameTree::new();
        let system = tree.add(
            ROOT_FRAME,
            "system",
            FrameMotion::Fixed {
                origin: galactic,
                rotation: DQuat::IDENTITY,
            },
        );
        let body = tree.add(system, "body", fixed(DVec3::X, DQuat::IDENTITY));
        let t = Epoch::J2000;
        assert_eq!(tree.origin_in_root(system, t), galactic);
        let p = tree.to_root(body, DVec3::new(0.25, 0.0, 0.0), t);
        assert_eq!(p.delta_from(galactic), DVec3::new(1.25, 0.0, 0.0));
    }
}
