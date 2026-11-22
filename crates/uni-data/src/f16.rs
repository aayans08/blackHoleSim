//! IEEE 754 binary16 conversion for the packed star velocities (M1 Task A7; no `half` crate).
//!
//! Formats from IEEE 754-2019 §3.6: binary16 has 1 sign bit, 5 exponent bits (bias 15) and 10
//! fraction bits; binary32 has 8 exponent bits (bias 127) and 23 fraction bits. Every binary16
//! value is exactly representable in binary32, so only the narrowing direction rounds.

/// Rounds `x` to the nearest binary16 (ties to even). Values beyond the largest finite
/// binary16 (65504) round to ±infinity, values below half the smallest subnormal (2⁻²⁵) to ±0,
/// and NaN stays NaN (quiet, with the top fraction bits kept).
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;
    if exp == 0xFF {
        return if frac == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7E00 | (frac >> 13) as u16
        };
    }
    let e = exp - 127; // unbiased exponent (−127 for zero and f32 subnormals)
    if e > 15 {
        return sign | 0x7C00;
    }
    if e >= -14 {
        // Normal binary16: keep the top 10 fraction bits and round on the 13 dropped ones. A
        // carry out of the fraction correctly bumps the exponent (up to infinity at 0x7C00).
        let mut h = (((e + 15) as u32) << 10) | (frac >> 13);
        let rest = frac & 0x1FFF;
        if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    if e < -25 {
        return sign;
    }
    // Subnormal binary16, in units of 2⁻²⁴: the 24-bit significand (implicit bit included,
    // weight 2^(e − 23)) shifted right by −(e + 1) ∈ [14, 24]. A carry into bit 10 yields the
    // smallest normal, 0x0400, which is the right encoding.
    let significand = frac | 0x0080_0000;
    let shift = (-(e + 1)) as u32;
    let mut h = significand >> shift;
    let rest = significand & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if rest > half || (rest == half && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// The binary16 `h` as an f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1F);
    let frac = u32::from(h & 0x03FF);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: frac × 2⁻²⁴, exact in f32 (frac < 2¹⁰).
            let magnitude = frac as f32 * f32::from_bits((127 - 24) << 23);
            return if sign == 0 { magnitude } else { -magnitude };
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, _) => sign | 0x7FC0_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXP_MASK: u16 = 0x7C00;
    const MAN_MASK: u16 = 0x03FF;

    fn is_nan(h: u16) -> bool {
        h & EXP_MASK == EXP_MASK && h & MAN_MASK != 0
    }

    #[test]
    fn f16_roundtrip_exhaustive() {
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if is_nan(h) {
                // NaN payloads are not preserved bit for bit, but NaN stays NaN.
                assert!(x.is_nan(), "{h:#06x}");
                assert!(is_nan(f32_to_f16(x)), "{h:#06x}");
            } else {
                assert_eq!(f32_to_f16(x), h, "{h:#06x} -> {x:e}");
            }
        }
    }

    #[test]
    fn f16_reference_values() {
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF, "largest finite");
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        assert!(is_nan(f32_to_f16(f32::NAN)));
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14), "smallest normal");
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24), "smallest subnormal");
    }

    #[test]
    fn f16_subnormals_and_underflow() {
        // 1e-6 = 16.78 × 2⁻²⁴, a subnormal: rounds to 17 × 2⁻²⁴.
        assert_eq!(f32_to_f16(1e-6), 0x0011);
        assert_eq!(f32_to_f16(-1e-6), 0x8011);
        // 1e-8 is 0.17 × 2⁻²⁴, below half the smallest subnormal: round to nearest gives +0.
        assert_eq!(f32_to_f16(1e-8), 0x0000);
        // Exactly half the smallest subnormal is a tie and goes to even (0); just above it, to 1.
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(2f32.powi(-25) * 1.000_001), 0x0001);
        // 1.5 × 2⁻²⁴ is a tie between 1 and 2 subnormal units: even wins.
        assert_eq!(f32_to_f16(1.5 * 2f32.powi(-24)), 0x0002);
        // The largest subnormal rounds up into the smallest normal.
        assert_eq!(f32_to_f16(2f32.powi(-14) * (1.0 - 2f32.powi(-12))), 0x0400);
        // f32 subnormals are far below the f16 range.
        assert_eq!(f32_to_f16(f32::from_bits(1)), 0x0000);
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        let ulp_at_one = 2f32.powi(-10);
        // Half an ulp above 1.0 is a tie: 1.0 has an even mantissa and wins.
        assert_eq!(f32_to_f16(1.0 + 0.5 * ulp_at_one), 0x3C00);
        // Half an ulp above 1 + ulp is a tie: 1 + 2 ulp is even and wins.
        assert_eq!(f32_to_f16(1.0 + 1.5 * ulp_at_one), 0x3C02);
        // Just past the tie rounds up.
        assert_eq!(f32_to_f16(1.0 + 0.51 * ulp_at_one), 0x3C01);
        // 65519 rounds down to the largest finite value; 65520 is the tie with 2¹⁶ and overflows.
        assert_eq!(f32_to_f16(65519.0), 0x7BFF);
        assert_eq!(f32_to_f16(65520.0), 0x7C00);
        assert_eq!(f32_to_f16(1e10), 0x7C00);
        assert_eq!(f32_to_f16(-1e10), 0xFC00);
    }

    #[test]
    fn f16_star_velocity_precision() {
        // Relative error ≤ 2⁻¹¹ for normal values: 0.05 % of a stellar velocity.
        for v in [-617.3_f32, -21.943, 0.37, 8.59, 117.51, 1234.5] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!(
                (back - v).abs() <= v.abs() * 2f32.powi(-11),
                "{v} -> {back}"
            );
        }
    }
}
