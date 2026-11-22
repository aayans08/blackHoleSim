//! Bookmark anchors: resolving a bookmark's `anchor`, `look_at` and `up` against the world's
//! objects (M1 Task A6). A0b stub; A6 implements.

use uni_core::{CameraView, FrameId, UniversePos};

use crate::bookmark::Bookmark;

/// Finds a named object: its centre in the root frame and its frame.
pub trait ObjectLookup {
    fn find_object(&self, name: &str) -> Option<(UniversePos, FrameId)>;
}

/// The camera view and anchor frame a bookmark describes. `anchor` makes `view.pos` an offset
/// from that object's centre; `look_at` overrides `view.orient`. An unknown name is an error that
/// names it.
pub fn resolve_bookmark(
    _b: &Bookmark,
    _world: &dyn ObjectLookup,
) -> anyhow::Result<(CameraView, FrameId)> {
    todo!("A6")
}
