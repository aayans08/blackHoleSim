//! uni-nav: camera and ship control. M0: quaternion 6-DoF camera, logarithmic speed, input, bookmarks.
//! M1 (Task A6): camera parenting to a frame, Bookmark v2 and anchor resolution.
//! M1 Wave C (Task C3; stubs by Task C0): selection and picking ([`pick`]), cruise ([`cruise`]),
//! the animated goto ([`goto`]), the aim ring and the [`Navigator`] that runs them ([`nav`]).
//! Only C0 and the lead edit this file (module list, re-exports).

pub mod anchor;
pub mod bookmark;
pub mod camera;
pub mod cruise;
pub mod goto;
pub mod input;
pub mod nav;
pub mod pick;

pub use anchor::{ObjectLookup, resolve_bookmark};
pub use bookmark::{Bookmark, ExposureChoice, LensEnvChoice, load_bookmarks, save_bookmarks};
pub use camera::FlyCamera;
pub use cruise::{CRUISE_LOG10_RANGE, cruise_speed};
pub use goto::{
    GOTO_ACCEL_FRAC, GOTO_DECEL_FRAC, GOTO_DURATION_S, goto_profile, standoff_distance,
    turn_duration_s,
};
pub use input::{Action, InputState, action_for};
pub use nav::{
    AIM_CONE_DEG, AIM_DRAIN_PER_S, AIM_EASE_PER_S, AIM_FILL_S, AIM_KEEP_DEG, Aim, DropOut,
    FlightMode, NavStatus, Navigator, handle_time_keys,
};
pub use pick::{PICK_RADIUS_PX_AT_1080, nearest_pick, pick_radius_px};

// Dependencies the stubs do not use yet (Task C0 adds every Wave C manifest entry, review I8).
// These lines keep `cargo::unused_dependencies` quiet until C3 uses them; delete each one then.
use uni_data as _;
use uni_physics as _;
