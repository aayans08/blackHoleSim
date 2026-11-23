//! Cruise (Task C3; Task C0 stub; design §5.6), labelled `CRUISE · FTL · NON-PHYSICAL · GRAVITY
//! IGNORED` (`uni_ui::theme::CRUISE_LABEL`).
//!
//! v = 10^cruise_log10 s⁻¹ × d_surface (at least [`MIN_CRUISE_SPEED_MPS`]) along the camera's −Z;
//! the wheel moves `cruise_log10` within [`CRUISE_LOG10_RANGE`]. Cruise drops out to free flight
//! at [`DROPOUT_SPEED_MPS`] on crossing an `ObjectInfo::atmosphere_top_m`, within
//! [`NEAR_STAR_DROPOUT_RADII`] photosphere radii of the Sun or a catalogue star, or within
//! [`BLACK_HOLE_DROPOUT_RS`] Schwarzschild radii of a black hole; one frame's step never crosses
//! a drop-out boundary.

/// Range of `Navigator::cruise_log10` [log10 of 1/s] (plan C3).
pub const CRUISE_LOG10_RANGE: (f64, f64) = (-3.0, 1.0);
/// `Navigator::cruise_log10` of a new navigator: a tenth of the distance per second.
pub const CRUISE_LOG10_DEFAULT: f64 = -1.0;
/// Cruise never goes slower than this [m/s] (plan C3).
pub const MIN_CRUISE_SPEED_MPS: f64 = 1.0;
/// Speed [m/s] the camera is left at when cruise drops out (plan C3: 1 km/s).
pub const DROPOUT_SPEED_MPS: f64 = 1.0e3;
/// Cruise drops out within this many photosphere radii of the Sun or a catalogue star.
pub const NEAR_STAR_DROPOUT_RADII: f64 = 10.0;
/// Cruise drops out within this many Schwarzschild radii of a black hole.
pub const BLACK_HOLE_DROPOUT_RS: f64 = 100.0;

/// Cruise speed [m/s] at `d_surface_m` from the nearest surface: 10^cruise_log10 × d_surface,
/// at least [`MIN_CRUISE_SPEED_MPS`].
pub fn cruise_speed(d_surface_m: f64, cruise_log10: f64) -> f64 {
    let _ = (d_surface_m, cruise_log10);
    todo!("C3")
}
