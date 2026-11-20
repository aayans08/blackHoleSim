// common/color.wgsl: colour helpers for scene-linear Rec. 709 / sRGB primaries.

// Relative luminance Y of linear Rec. 709 RGB.
// Weights: ITU-R BT.709-6 (2015), item 3.2 (Y = 0.2126 R + 0.7152 G + 0.0722 B).
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Linear → non-linear sRGB encoding (the sRGB OETF), per channel.
// Source: IEC 61966-2-1:1999, §5.2: 12.92·c for c ≤ 0.0031308, else 1.055·c^(1/2.4) − 0.055.
// Negative inputs are clamped to 0 so `pow` never sees a negative base.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    let lo = 12.92 * x;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}
