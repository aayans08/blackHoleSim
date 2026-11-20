//! uni-nav: camera and ship control. M0: quaternion 6-DoF camera, logarithmic speed, input, bookmarks.

pub mod bookmark;
pub mod camera;
pub mod input;

pub use bookmark::{Bookmark, load_bookmarks, save_bookmarks};
pub use camera::FlyCamera;
pub use input::{Action, InputState, action_for};
