// common/constants.wgsl: shared mathematical constants.
//
// Keep this file to constants every shader may need; feature-specific constants live
// with the feature, so an `#import` of this file never collides with a local name.

// π and τ = 2π, rounded to f32.
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
