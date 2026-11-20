//! AgX tonemap CPU twin: scene-linear Rec.709 in, display-linear out (M0 Task 5).
//!
//! Source: Troy Sobotka, AgX (<https://github.com/sobotka/AgX>), the "base" transform with
//! the default contrast look, in the real-time form published by Benjamin Wrensch, "Minimal
//! AgX Implementation" (iolite engine blog, 2023,
//! <https://iolite-engine.com/blog_posts/minimal_agx_implementation>). The inset and outset
//! matrices, the log2 range and the 6th-order polynomial fit of the default-contrast sigmoid
//! (mean squared error 3.67e-6 against Sobotka's curve) are taken from there verbatim.
//!
//! The pipeline is: inset matrix → log2 encoding over [−12.47393, +4.026069] EV → sigmoid →
//! outset matrix → decode to display-linear. The sigmoid's output is an sRGB-encoded signal.
//! The tonemap pass writes to an sRGB view, whose hardware encoder applies the IEC 61966-2-1
//! piecewise curve, so the decode uses that curve's exact inverse; the 8-bit code values on
//! screen are then exactly AgX's encoded output. (Wrensch decodes with a pure 2.2 power, which
//! the piecewise encoder does not invert and which darkens deep shadows.)
//!
//! [`agx`] must mirror `fn agx` in `shaders/tonemap.wgsl`: the same constants, the same order
//! of operations. `universe --selftest` compares the two on the GPU within 1e-3.

use glam::{Mat3, Vec3};

/// Inset matrix (columns), applied to linear Rec. 709 RGB before the log encoding. It mixes
/// a few percent of each primary into the others, so very bright saturated colours move
/// toward white instead of clipping per channel. Source: Wrensch 2023 (`agx_mat`).
#[allow(clippy::excessive_precision)] // digits kept verbatim from the source and the WGSL
const AGX_INSET: Mat3 = Mat3::from_cols(
    Vec3::new(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
    Vec3::new(0.0784335999999992, 0.878468636469772, 0.0784336),
    Vec3::new(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
);

/// Outset matrix (columns): the inverse of [`AGX_INSET`], applied after the sigmoid.
/// Source: Wrensch 2023 (`agx_mat_inv`).
#[allow(clippy::excessive_precision)] // digits kept verbatim from the source and the WGSL
const AGX_OUTSET: Mat3 = Mat3::from_cols(
    Vec3::new(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
    Vec3::new(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
    Vec3::new(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
);

/// Bottom of the log2 encoding range [EV relative to 1.0]: 10 stops below middle grey
/// (log2 0.18 = −2.47393). Source: Sobotka AgX; Wrensch 2023 (`min_ev`).
const AGX_MIN_EV: f32 = -12.47393;
/// Top of the log2 encoding range: 6.5 stops above middle grey. Source: as [`AGX_MIN_EV`].
const AGX_MAX_EV: f32 = 4.026069;
/// Floor applied before `log2` so zero or negative input stays finite. It lies far below
/// 2^AGX_MIN_EV, so it lands at the bottom of the range like any other near-black value.
const AGX_LOG_FLOOR: f32 = 1e-10;

/// AgX "base" with the default contrast look.
///
/// Input: scene-linear Rec. 709 RGB, already multiplied by the exposure scale (middle grey
/// at 0.18). Output: display-linear RGB in [0, 1]; the sRGB render target does the encoding.
/// Zero, negative and infinite inputs give finite results.
pub fn agx(rgb_linear: Vec3) -> Vec3 {
    // 1. Inset.
    let inset = AGX_INSET * rgb_linear;
    // 2. Log2 encoding, normalised so [AGX_MIN_EV, AGX_MAX_EV] maps to [0, 1].
    let ev = inset
        .max(Vec3::splat(AGX_LOG_FLOOR))
        .log2()
        .clamp(Vec3::splat(AGX_MIN_EV), Vec3::splat(AGX_MAX_EV));
    let x = (ev - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);
    // 3. Default-contrast sigmoid → sRGB-encoded signal.
    let encoded = agx_default_contrast(x);
    // 4. Outset. Bright saturated colours can overshoot [0, 1] here, so clamp to the display.
    let outset = (AGX_OUTSET * encoded).clamp(Vec3::ZERO, Vec3::ONE);
    // 5. Decode to display-linear.
    srgb_to_linear(outset)
}

/// 6th-order polynomial fit of AgX's default-contrast sigmoid on [0, 1].
/// Source: Wrensch 2023 (`agxDefaultContrastApprox`), coefficients and term order verbatim.
fn agx_default_contrast(x: Vec3) -> Vec3 {
    let x2 = x * x;
    let x4 = x2 * x2;
    15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
        - 0.00232
}

/// sRGB code value → display-linear, per channel (the inverse of the sRGB encoding).
/// Source: IEC 61966-2-1:1999, §5.2: c / 12.92 for c ≤ 0.04045, else ((c + 0.055) / 1.055)^2.4.
/// Negative inputs are clamped to 0. Twin of `srgb_to_linear` in `shaders/tonemap.wgsl`.
fn srgb_to_linear(c: Vec3) -> Vec3 {
    c.map(|v| {
        let x = v.max(0.0);
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::naga;

    #[test]
    fn agx_black_is_black() {
        assert!(agx(Vec3::ZERO).max_element() < 1e-3);
    }

    #[test]
    fn agx_monotonic_grey() {
        let mut last = -1.0;
        for i in 0..200 {
            let x = 1e-4 * 1.08f32.powi(i);
            let y = agx(Vec3::splat(x)).y;
            assert!(y >= last - 1e-6, "agx({x}).y = {y} < previous {last}");
            last = y;
        }
    }

    #[test]
    fn agx_bounded() {
        assert!(agx(Vec3::splat(1e6)).max_element() <= 1.0 + 1e-4);
    }

    #[test]
    fn agx_mid_grey_reasonable() {
        // Display-linear, so roughly 0.21 rather than the encoded ~0.5.
        let y = agx(Vec3::splat(0.18)).y;
        assert!(y > 0.10 && y < 0.30, "agx(0.18).y = {y}");
    }

    #[test]
    fn agx_extreme_inputs_stay_finite_and_in_range() {
        // `--selftest` compares 256 inputs with the GPU; none may produce NaN or leave [0, 1].
        // A primary near 20 puts its own channel at the top of the log range and the others
        // (mixed in by the inset) near 0.74, where the outset overshoots 1 unless clamped.
        // Far brighter primaries saturate every channel and come out as near-white.
        let inputs = [
            Vec3::ZERO,
            Vec3::splat(-1.0),
            Vec3::new(-5.0, 0.2, 3.0),
            Vec3::splat(1e-30),
            Vec3::splat(f32::INFINITY),
            Vec3::new(f32::INFINITY, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
            Vec3::new(0.0, 20.0, 0.0),
            Vec3::new(0.0, 0.0, 20.0),
            Vec3::new(1e6, 0.0, 0.0),
            Vec3::new(3e4, 2e4, 0.0),
        ];
        for c in inputs {
            let out = agx(c);
            assert!(out.is_finite(), "agx({c}) = {out}");
            assert!(
                out.min_element() >= 0.0 && out.max_element() <= 1.0,
                "agx({c}) = {out}"
            );
        }
    }

    #[test]
    fn srgb_to_linear_matches_iec_61966_2_1() {
        let s = |v: f32| srgb_to_linear(Vec3::splat(v)).x;
        assert_eq!(s(0.0), 0.0);
        assert!((s(1.0) - 1.0).abs() < 1e-6);
        // Linear segment: 0.02 / 12.92.
        assert!((s(0.02) - 0.02 / 12.92).abs() < 1e-9);
        // The two branches meet at code 0.04045 ↔ linear 0.0031308 (IEC 61966-2-1 §5.2).
        assert!((s(0.04045) - 0.003_130_8).abs() < 1e-7);
        assert!((s(0.040_46) - 0.003_130_8).abs() < 2e-6);
        // Power segment: ((0.5 + 0.055) / 1.055)^2.4 = 0.214041.
        assert!((s(0.5) - 0.214_041).abs() < 1e-6);
        // Negative code values clamp to 0 instead of going through `powf`.
        assert_eq!(s(-0.5), 0.0);
    }

    /// Flattens a constant's initialiser (nested `Compose` of f32 literals) in column order.
    fn const_f32s(module: &naga::Module, name: &str) -> Vec<f32> {
        fn flatten(module: &naga::Module, h: naga::Handle<naga::Expression>, out: &mut Vec<f32>) {
            match &module.global_expressions[h] {
                naga::Expression::Literal(naga::Literal::F32(v)) => out.push(*v),
                naga::Expression::Compose { components, .. } => {
                    for &c in components {
                        flatten(module, c, out);
                    }
                }
                other => panic!("unexpected constant expression {other:?}"),
            }
        }
        let (_, c) = module
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("const {name} missing from post/tonemap.wgsl"));
        let mut out = Vec::new();
        flatten(module, c.init, &mut out);
        out
    }

    #[test]
    fn wgsl_agx_constants_match_cpu_twin() {
        // `--selftest` checks the twins on the GPU; this catches a constant edited on one side
        // only, without a GPU.
        let mut lib = uni_gfx::ShaderLib::with_common();
        crate::tonemap::register_shaders(&mut lib);
        let src = lib.compose(crate::tonemap::TONEMAP_SHADER).unwrap();
        let module = uni_gfx::validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}"));

        let close = |wgsl: &[f32], cpu: &[f32], name: &str| {
            assert_eq!(wgsl.len(), cpu.len(), "{name}");
            for (w, c) in wgsl.iter().zip(cpu) {
                // Decimal → f64 → f32 (naga) and decimal → f32 (rustc) may differ by one ulp.
                assert!(
                    (w - c).abs() <= c.abs() * f32::EPSILON,
                    "{name}: {w} vs {c}"
                );
            }
        };
        close(
            &const_f32s(&module, "AGX_INSET"),
            &AGX_INSET.to_cols_array(),
            "AGX_INSET",
        );
        close(
            &const_f32s(&module, "AGX_OUTSET"),
            &AGX_OUTSET.to_cols_array(),
            "AGX_OUTSET",
        );
        close(
            &const_f32s(&module, "AGX_MIN_EV"),
            &[AGX_MIN_EV],
            "AGX_MIN_EV",
        );
        close(
            &const_f32s(&module, "AGX_MAX_EV"),
            &[AGX_MAX_EV],
            "AGX_MAX_EV",
        );
        close(
            &const_f32s(&module, "AGX_LOG_FLOOR"),
            &[AGX_LOG_FLOOR],
            "AGX_LOG_FLOOR",
        );
    }
}
