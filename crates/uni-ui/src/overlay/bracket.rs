//! The selected target's corner brackets (Task C5; Task C0 wrote the type).
//!
//! Four cyan corner brackets (`theme::BRACKET_ARM` 18 pt arms, `theme::BRACKET_STROKE` 1.5 pt),
//! half-size = max(projected radius + 12 pt, 20 pt); below them the name (12 pt, weight 600,
//! letter-spacing 0.14 em, cyan) and "distance · closing speed" (mono 13 pt).

use glam::Vec2;

/// The target bracket, in output points.
#[derive(Clone, Debug)]
pub struct Bracket {
    pub center_pt: Vec2,
    pub half_size_pt: f32,
    pub name: String,
    /// "distance · closing speed".
    pub sub: String,
}
