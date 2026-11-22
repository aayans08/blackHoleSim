//! uni-nav: camera and ship control. M0: quaternion 6-DoF camera, logarithmic speed, input, bookmarks.
//! M1 (Task A6): camera parenting to a frame, Bookmark v2 and anchor resolution.

pub mod anchor;
pub mod bookmark;
pub mod camera;
pub mod input;

pub use anchor::{ObjectLookup, resolve_bookmark};
pub use bookmark::{Bookmark, ExposureChoice, LensEnvChoice, load_bookmarks, save_bookmarks};
pub use camera::FlyCamera;
pub use input::{Action, InputState, action_for};
