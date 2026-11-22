//! IEEE 754 binary16 conversion for the packed star velocities (M1 Task A7; no `half` crate).
//! A0b stub; A7 implements.

/// Rounds `x` to the nearest binary16 (ties to even).
pub fn f32_to_f16(_x: f32) -> u16 {
    todo!("A7")
}

/// The binary16 `h` as an f32 (exact).
pub fn f16_to_f32(_h: u16) -> f32 {
    todo!("A7")
}
