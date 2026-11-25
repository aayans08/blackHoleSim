//! AgX-style tonemap CPU twin: scene-linear Rec.709 in, display-linear out (M0 Task 5; constants
//! re-derived in the M1 final fix wave, F3).
//!
//! The idea is Troy Sobotka's AgX (<https://github.com/sobotka/AgX>): inset the primaries toward
//! white, encode log2 over a fixed range around middle grey, apply a sigmoid, then undo the inset.
//! Every constant here is derived in this file (final review: the previous ones were copied
//! verbatim from Benjamin Wrensch's 2023 "Minimal AgX Implementation" blog post, which states no
//! licence, "© 2024 Missing Deadlines. All rights reserved", and Sobotka's AgX repositories carry
//! no licence file either; both checked 2026-11-25):
//! - **Inset** ([`AGX_INSET`]): each Rec. 709 primary moved [`AGX_INSET_FRACTION`] = 20 % of the
//!   way to the D65 white point in CIE 1931 xy, white kept. A white-preserving matrix that moves
//!   every primary along its straight line to white is (1 − t)·I + t·𝟙wᵀ with Σw = 1 (its rows
//!   sum to 1); primary j then lands a fraction t·w_j·S/((1 − t)·S_j + t·w_j·S) of the way, with
//!   S_j = X + Y + Z of primary j at unit RGB = Y_j/y_j and S = ΣS_j, so the same fraction t for
//!   every primary needs w_j = S_j/S. Primaries and white: Rec. ITU-R BT.709-6, items 1.3–1.4;
//!   Y_j from the standard RGB→XYZ construction (SMPTE RP 177-1993). `inset_matrix_derivation`
//!   rebuilds it in f64. (The previous matrix agreed with it to 8e-5.)
//! - **Outset** ([`AGX_OUTSET`]): its exact inverse, (I − t·𝟙wᵀ)/(1 − t) (Sherman–Morrison).
//! - **Log2 range**: 10 stops below and 6.5 stops above middle grey 0.18 (AgX's base range, a
//!   parameter, written as log2 0.18 − 10 and log2 0.18 + 6.5).
//! - **Sigmoid** ([`agx_sigmoid`]): this file's own toe/shoulder curve through the pivot
//!   (middle grey's position, [`SIGMOID_PIVOT_OUT`]) with slope [`SIGMOID_SLOPE`], each side a
//!   rational power curve that meets 0 at x = 0 and 1 at x = 1 exactly. Its four parameters were
//!   fitted (minimax) to keep the M0 look: the curve is within 0.004 of the previous sigmoid
//!   on [0, 1] (`sigmoid_keeps_the_m0_look`), and the whole transform within 0.0056 in the encoded
//!   signal (1.4 code values of 255) over 49³ colours from black to 2¹¹ × middle grey (measured
//!   once, final-fix-F3-report.md).
//!
//! The pipeline is: inset matrix → log2 encoding over [AGX_MIN_EV, AGX_MAX_EV] → sigmoid →
//! outset matrix → decode to display-linear. The sigmoid's output is an sRGB-encoded signal.
//! The tonemap pass writes to an sRGB view, whose hardware encoder applies the IEC 61966-2-1
//! piecewise curve, so the decode uses that curve's exact inverse; the 8-bit code values on
//! screen are then exactly the encoded output.
//!
//! [`agx`] must mirror `fn agx` in `shaders/tonemap.wgsl`: the same constants, the same order
//! of operations. `universe --selftest` compares the two on the GPU within 1e-3.

use glam::{Mat3, Vec3};

/// The inset: each Rec. 709 primary moves this fraction of the way to D65 in CIE 1931 xy.
pub const AGX_INSET_FRACTION: f64 = 0.2;

/// Inset matrix (columns), applied to linear Rec. 709 RGB before the log encoding:
/// (1 − t)·I + t·𝟙wᵀ with t = [`AGX_INSET_FRACTION`] and w_j = S_j/ΣS (module docs), so very
/// bright saturated colours move toward white instead of clipping per channel. Derived in
/// `inset_matrix_derivation`.
#[allow(clippy::excessive_precision)] // f64 digits of the derivation, rounded to f32 by rustc
const AGX_INSET: Mat3 = Mat3::from_cols(
    Vec3::new(0.8423989290495315, 0.04239892904953144, 0.04239892904953144),
    Vec3::new(0.07843016510486392, 0.8784301651048639, 0.07843016510486392),
    Vec3::new(0.07917090584560466, 0.07917090584560466, 0.8791709058456048),
);

/// Outset matrix (columns): the exact inverse of [`AGX_INSET`], (I − t·𝟙wᵀ)/(1 − t), applied
/// after the sigmoid.
#[allow(clippy::excessive_precision)] // f64 digits of the derivation, rounded to f32 by rustc
const AGX_OUTSET: Mat3 = Mat3::from_cols(
    Vec3::new(1.197001338688086, -0.0529986613119143, -0.0529986613119143),
    Vec3::new(-0.09803770638107989, 1.15196229361892, -0.09803770638107989),
    Vec3::new(
        -0.09896363230700582,
        -0.09896363230700582,
        1.151036367692994,
    ),
);

/// Bottom of the log2 encoding range [EV relative to 1.0]: 10 stops below middle grey,
/// log2 0.18 − 10. Everything at or below it maps to display black.
#[allow(clippy::excessive_precision)]
pub const AGX_MIN_EV: f32 = -12.473931188332411;
/// Top of the log2 encoding range: 6.5 stops above middle grey, log2 0.18 + 6.5.
#[allow(clippy::excessive_precision)]
pub const AGX_MAX_EV: f32 = 4.026068811667588;
/// Floor applied before `log2` so zero or negative input stays finite. It lies far below
/// 2^AGX_MIN_EV, so it lands at the bottom of the range like any other near-black value.
const AGX_LOG_FLOOR: f32 = 1e-10;

/// The sigmoid's pivot input: middle grey's position in the log range, 10/16.5.
const SIGMOID_PIVOT_IN: f32 = 10.0 / 16.5;
/// The sigmoid's output at the pivot: middle grey at code value 0.5 (display-linear 0.214).
pub const SIGMOID_PIVOT_OUT: f32 = 0.5;
/// The sigmoid's slope at the pivot [encoded output per unit of the 16.5-stop input].
pub const SIGMOID_SLOPE: f32 = 2.02;
/// The toe's and the shoulder's power p in [`rational_curve`] (fitted, module docs).
const SIGMOID_TOE_POWER: f32 = 2.9;
const SIGMOID_SHOULDER_POWER: f32 = 2.8;

/// AgX-style transform (module docs).
///
/// Input: pre-exposed scene-linear Rec. 709 RGB, i.e. the HDR target's values (light ×
/// exposure, uni-gfx `camera.wgsl`; middle grey at 0.18). Output: display-linear RGB in
/// [0, 1]; the sRGB render target does the encoding.
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
    // 3. Sigmoid → sRGB-encoded signal.
    let encoded = x.map(agx_sigmoid);
    // 4. Outset. Bright saturated colours can overshoot [0, 1] here, so clamp to the display.
    let outset = (AGX_OUTSET * encoded).clamp(Vec3::ZERO, Vec3::ONE);
    // 5. Decode to display-linear.
    srgb_to_linear(outset)
}

/// S(v) = k·v / (1 + (k^p − 1)·v^p)^(1/p) on [0, 1]: S(0) = 0, S(1) = 1, S'(0) = k, increasing for
/// k ≥ 1 (S' = k·(1 + (k^p − 1)v^p)^(−(p+1)/p) > 0), and S''(0) = 0 for p > 1. Twin of
/// `rational_curve` in `tonemap.wgsl`.
fn rational_curve(v: f32, k: f32, p: f32) -> f32 {
    k * v / (1.0 + (k.powf(p) - 1.0) * v.powf(p)).powf(1.0 / p)
}

/// The sigmoid: x in [0, 1] (the log-encoded input) → the sRGB-encoded output in [0, 1]. Through
/// (SIGMOID_PIVOT_IN, SIGMOID_PIVOT_OUT) with slope SIGMOID_SLOPE; above the pivot a shoulder
/// out + (1 − out)·S((x − in)/(1 − in)), below it a toe out − out·S((in − x)/in), each S a
/// [`rational_curve`] whose k carries the pivot slope over. Ends: exactly 0 at x = 0, 1 at x = 1;
/// slope and curvature continuous at the pivot. Twin of `agx_sigmoid` in `tonemap.wgsl`.
pub fn agx_sigmoid(x: f32) -> f32 {
    let (xp, yp) = (SIGMOID_PIVOT_IN, SIGMOID_PIVOT_OUT);
    if x >= xp {
        let k = SIGMOID_SLOPE * (1.0 - xp) / (1.0 - yp);
        yp + (1.0 - yp) * rational_curve((x - xp) / (1.0 - xp), k, SIGMOID_SHOULDER_POWER)
    } else {
        let k = SIGMOID_SLOPE * xp / yp;
        yp - yp * rational_curve((xp - x) / xp, k, SIGMOID_TOE_POWER)
    }
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

    /// The inset matrix from its stated method, in f64: BT.709-6 primaries and D65, Y_j from
    /// M·(1, 1, 1) = XYZ(D65) (SMPTE RP 177-1993), w_j = S_j/ΣS with S_j = Y_j/y_j, M_in =
    /// (1 − t)I + t𝟙wᵀ; each primary then moves exactly t of the way to white in xy along the
    /// line to it, white is kept, and the outset is the inverse. The constants are its digits.
    #[test]
    fn inset_matrix_derivation() {
        use glam::{DMat3, DVec3};
        let prim = [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)];
        let white = (0.3127, 0.3290);
        let xyz = |(x, y): (f64, f64)| DVec3::new(x / y, 1.0, (1.0 - x - y) / y);
        let p = DMat3::from_cols(xyz(prim[0]), xyz(prim[1]), xyz(prim[2]));
        let y = p.inverse() * xyz(white); // the primaries' luminances Y_j
        assert!((y.x - 0.2126).abs() < 1e-4 && (y.y - 0.7152).abs() < 1e-4);
        let s = DVec3::new(y.x / prim[0].1, y.y / prim[1].1, y.z / prim[2].1);
        let w = s / s.element_sum();
        let t = AGX_INSET_FRACTION;
        let ones = DMat3::from_cols(w * t, w * t, w * t).transpose(); // row i = t·wᵀ
        let inset = DMat3::IDENTITY * (1.0 - t) + ones;
        let outset = (DMat3::IDENTITY - ones) * (1.0 / (1.0 - t));
        let close = |a: DMat3, b: Mat3, name: &str| {
            for (x, y) in a.to_cols_array().iter().zip(b.to_cols_array()) {
                assert!((*x as f32 - y).abs() <= 1e-7, "{name}: {x} vs {y}");
            }
        };
        close(inset, AGX_INSET, "AGX_INSET");
        close(outset, AGX_OUTSET, "AGX_OUTSET");
        let prod = inset * outset - DMat3::IDENTITY;
        assert!(
            prod.to_cols_array().iter().all(|v| v.abs() < 1e-12),
            "{prod}"
        );
        // White kept; each primary 20 % of the way to D65 in xy, on the line to it.
        assert!((inset * DVec3::ONE - DVec3::ONE).abs().max_element() < 1e-12);
        let rgb_to_xyz = p * DMat3::from_diagonal(y);
        let xy = |v: DVec3| {
            let c = rgb_to_xyz * v;
            (c.x / c.element_sum(), c.y / c.element_sum())
        };
        for j in 0..3 {
            let moved = xy(inset.col(j));
            let (dx, dy) = (white.0 - prim[j].0, white.1 - prim[j].1);
            let f = ((moved.0 - prim[j].0) * dx + (moved.1 - prim[j].1) * dy) / (dx * dx + dy * dy);
            let off_line = (moved.0 - prim[j].0) * dy - (moved.1 - prim[j].1) * dx;
            assert!(
                (f - t).abs() < 1e-12 && off_line.abs() < 1e-12,
                "primary {j}: {f}"
            );
        }
        // The log range: 10 stops below and 6.5 above middle grey.
        assert!((f64::from(AGX_MIN_EV) - (0.18f64.log2() - 10.0)).abs() < 1e-6);
        assert!((f64::from(AGX_MAX_EV) - (0.18f64.log2() + 6.5)).abs() < 1e-6);
    }

    /// The sigmoid's own properties: exact ends, the pivot and its slope, monotonic, C¹ at the
    /// pivot; and middle grey (0.18) lands at code value 0.5 through the whole transform.
    #[test]
    fn sigmoid_shape() {
        assert_eq!(agx_sigmoid(0.0), 0.0);
        assert!((agx_sigmoid(1.0) - 1.0).abs() < 1e-6);
        assert!((agx_sigmoid(SIGMOID_PIVOT_IN) - SIGMOID_PIVOT_OUT).abs() < 1e-7);
        let h = 1e-3;
        for side in [-1.0_f32, 1.0] {
            let x = SIGMOID_PIVOT_IN + side * h;
            let slope = (agx_sigmoid(x) - SIGMOID_PIVOT_OUT) / (side * h);
            assert!((slope - SIGMOID_SLOPE).abs() < 2e-3, "{side}: {slope}");
        }
        let mut last = -1.0;
        for i in 0..=1000 {
            let y = agx_sigmoid(i as f32 / 1000.0);
            assert!(y > last || i == 0, "{i}: {y}");
            assert!((0.0..=1.0 + 1e-6).contains(&y));
            last = y;
        }
        let grey = agx(Vec3::splat(0.18));
        let code = grey.map(|v| {
            if v <= 0.003_130_8 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        });
        assert!(
            (code - Vec3::splat(0.5)).abs().max_element() < 1e-4,
            "{code}"
        );
    }

    /// The look is kept: the previous (M0) sigmoid, sampled at x = k/16 and clamped to [0, 1]
    /// (values rounded to 4 decimals, final-fix-F3-report.md), against this file's curve: within
    /// 0.0045 everywhere (the minimax fit's error is 0.0040 on a 4001-point grid).
    #[test]
    fn sigmoid_keeps_the_m0_look() {
        const M0: [f32; 17] = [
            0.0, 0.0056, 0.0125, 0.0207, 0.0364, 0.0669, 0.1181, 0.1931, 0.2915, 0.4085, 0.5358,
            0.6625, 0.7769, 0.8688, 0.9325, 0.9706, 0.9986,
        ];
        for (k, m0) in M0.iter().enumerate() {
            let y = agx_sigmoid(k as f32 / 16.0);
            assert!((y - m0).abs() < 0.0045, "x = {k}/16: {y} vs {m0}");
        }
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
        for (name, cpu) in [
            ("SIGMOID_PIVOT_IN", SIGMOID_PIVOT_IN),
            ("SIGMOID_PIVOT_OUT", SIGMOID_PIVOT_OUT),
            ("SIGMOID_SLOPE", SIGMOID_SLOPE),
            ("SIGMOID_TOE_POWER", SIGMOID_TOE_POWER),
            ("SIGMOID_SHOULDER_POWER", SIGMOID_SHOULDER_POWER),
        ] {
            close(&const_f32s(&module, name), &[cpu], name);
        }
    }
}
