//! The local V-band luminosity function (M1 Task A5): how many stars of each absolute magnitude
//! M_V live in a cubic parsec near the Sun, and from it the fraction of a population's light that
//! comes from stars too faint to be in the star catalogue (the light that makes the Milky Way band).
//!
//! The table is a composite of three volume-complete samples, joined where each is best measured
//! (1-mag bins unless noted; no bin overlaps another):
//! - M_V −3.5…3.5, dwarfs and giants: Flynn et al. 2006, MNRAS 372, 1149, Table 3 (the Heidelberg
//!   luminosity function from the CNS4 50 pc sample), Φ in stars per 20 pc sphere (552 stars
//!   within 50 pc give Φ = 35.3 at M_V = 3, so the volume is the 20 pc sphere). The M_V = −2 row
//!   is empty in the source and is left out.
//! - M_V 3.5…9.5: Rybizki & Just 2015, MNRAS 447, 3880, Tables 1 (dwarfs) and 2 (giants), column
//!   N₂₅: observed counts rescaled to the 25 pc sphere.
//! - M_V 9.5…18: Reid, Gizis & Hawley 2002, AJ 124, 2721 (PMSU IV), Table 4, Φ_tot [stars pc⁻³
//!   × 10⁴] (primaries, single stars and companions; values from the authors' submitted version,
//!   STScI). Its bins are centred on half magnitudes, so only the faint half of its M_V 9.5 bin
//!   (9.5–10) is used.
//!
//! The table holds main-sequence stars and giants; white dwarfs and brown dwarfs, which add
//! almost no V light, are not in it. It gives 0.106 stars/pc³ and 0.0519 L☉,V/pc³ (measured;
//! `table_matches_its_sources` checks both against the literature). Within a bin the stars are
//! spread uniformly in M_V. The bins from M_V = 15 on rest on a few stars within 5 pc: they hold
//! 38 % of the count but under 10⁻⁴ of the light.

use std::f64::consts::{LN_10, PI};

use uni_core::M_V_SUN;

/// Local V-band luminosity density [L☉,V/pc³] of all stars, from the Tuorla luminosity function:
/// Flynn et al. 2006, MNRAS 372, 1149, §2.3 ("ρ_L = 0.056"). The galaxy model is checked against it.
pub const LOCAL_V_LUMINOSITY_DENSITY_FLYNN_2006: f64 = 0.056;

/// Number of luminosity-function bins (one `GalaxyGpu::lf` entry each).
pub(crate) const LF_BINS: usize = 21;

/// One luminosity-function bin: absolute magnitude range and stars per pc³ in it.
struct LfBin {
    m_bright: f64,
    m_faint: f64,
    stars_per_pc3: f64,
}

/// Volume of the 20 pc sphere [pc³], the unit of Flynn et al. 2006 Table 3.
const SPHERE_20PC: f64 = 4.0 / 3.0 * PI * 8_000.0;
/// Volume of the 25 pc sphere [pc³], the unit of Rybizki & Just 2015 N₂₅.
const SPHERE_25PC: f64 = 4.0 / 3.0 * PI * 15_625.0;

/// A 1-mag bin centred on `m`.
const fn bin(m: f64, stars_per_pc3: f64) -> LfBin {
    LfBin {
        m_bright: m - 0.5,
        m_faint: m + 0.5,
        stars_per_pc3,
    }
}

/// The composite local luminosity function (sources in the module doc).
const LOCAL_LF: [LfBin; LF_BINS] = [
    // Flynn et al. 2006 Table 3, column Φ (all stars), per 20 pc sphere.
    bin(-3.0, 0.06 / SPHERE_20PC),
    bin(-1.0, 1.1 / SPHERE_20PC),
    bin(0.0, 3.3 / SPHERE_20PC),
    bin(1.0, 11.3 / SPHERE_20PC),
    bin(2.0, 16.7 / SPHERE_20PC),
    bin(3.0, 35.3 / SPHERE_20PC),
    // Rybizki & Just 2015 Tables 1 + 2, N₂₅ (dwarfs 115.7 + giants 5.21 at M_V 4; dwarfs only
    // fainter, where Table 2 ends).
    bin(4.0, (115.7 + 5.21) / SPHERE_25PC),
    bin(5.0, 191.0 / SPHERE_25PC),
    bin(6.0, 198.0 / SPHERE_25PC),
    bin(7.0, 193.0 / SPHERE_25PC),
    bin(8.0, 207.0 / SPHERE_25PC),
    bin(9.0, 271.5 / SPHERE_25PC),
    // Reid, Gizis & Hawley 2002 Table 4, Φ_tot × 10⁻⁴; the faint half of the M_V 9.5 bin first.
    LfBin {
        m_bright: 9.5,
        m_faint: 10.0,
        stars_per_pc3: 0.5 * 58.26e-4,
    },
    bin(10.5, 88.16e-4),
    bin(11.5, 91.06e-4),
    bin(12.5, 107.30e-4),
    bin(13.5, 92.31e-4),
    bin(14.5, 50.93e-4),
    bin(15.5, 248.28e-4),
    bin(16.5, 76.39e-4),
    bin(17.5, 76.39e-4),
];

/// V luminosity [L☉,V] of a star of absolute magnitude `m_v` (M_V☉ = 4.83, `uni_core::M_V_SUN`).
fn luminosity(m_v: f64) -> f64 {
    10_f64.powf(-0.4 * (m_v - M_V_SUN))
}

/// V luminosity density [L☉,V/pc³] of one bin, its stars spread uniformly in M_V:
/// n · (mean of 10^(−0.4 (M − M_V☉)) over the bin).
fn bin_light(b: &LfBin) -> f64 {
    let width = b.m_faint - b.m_bright;
    b.stars_per_pc3 * (luminosity(b.m_bright) - luminosity(b.m_faint)) / (0.4 * LN_10 * width)
}

/// Stars per pc³ in the table (main sequence and giants near the Sun).
pub(crate) fn local_number_density() -> f64 {
    LOCAL_LF.iter().map(|b| b.stars_per_pc3).sum()
}

/// V luminosity density [L☉,V/pc³] of the table.
pub(crate) fn local_v_luminosity_density() -> f64 {
    LOCAL_LF.iter().map(bin_light).sum()
}

/// Per bin: `[x_bright, x_faint, weight]`, with x = 10^(−0.4 M_V) at the bin's bright and faint
/// edges and weight = the bin's share of the table's light. The GPU twin uses the same numbers.
///
/// Why x: a star of absolute magnitude M at distance d appears fainter than V_lim exactly when
/// 10^(−0.4 M) < y with y = 10^(−0.4 V_lim) · (d / 10 pc)² (distance modulus, no extinction). The
/// light of a bin's stars fainter than that limit is ∝ 10^(−0.4 M_lim) − 10^(−0.4 M_faint), which
/// is linear in y, so each bin adds weight · clamp((y − x_faint) / (x_bright − x_faint), 0, 1).
pub(crate) fn faint_bins() -> [[f64; 3]; LF_BINS] {
    let total = local_v_luminosity_density();
    std::array::from_fn(|i| {
        let b = &LOCAL_LF[i];
        [
            10_f64.powf(-0.4 * b.m_bright),
            10_f64.powf(-0.4 * b.m_faint),
            bin_light(b) / total,
        ]
    })
}

/// The flux limit y = 10^(−0.4 V_lim) · (d / 10 pc)² of [`faint_bins`].
pub(crate) fn flux_limit(dist_pc: f64, v_lim: f64) -> f64 {
    let d10 = dist_pc / 10.0;
    10_f64.powf(-0.4 * v_lim) * d10 * d10
}

/// The faint light fraction at flux limit `y` (see [`faint_bins`]).
pub(crate) fn faint_fraction_at(bins: &[[f64; 3]; LF_BINS], y: f64) -> f64 {
    bins.iter()
        .map(|&[x_bright, x_faint, w]| w * ((y - x_faint) / (x_bright - x_faint)).clamp(0.0, 1.0))
        .sum()
}

/// Fraction of the V-band light of a stellar population at `dist_pc` that comes from stars
/// fainter than apparent magnitude `v_lim`.
///
/// Uses the local luminosity function for every population and ignores extinction: a star dimmed
/// below `v_lim` by dust still counts as bright here, so where A_V ≳ 1 mag the band is somewhat
/// too faint (v0 approximation).
pub fn faint_light_fraction(dist_pc: f64, v_lim: f64) -> f64 {
    faint_fraction_at(&faint_bins(), flux_limit(dist_pc, v_lim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faint_fraction_limits() {
        assert!((faint_light_fraction(1e5, 10.0) - 1.0).abs() < 1e-12);
        let near = faint_light_fraction(1.0, 10.0);
        assert!((0.0..1e-3).contains(&near), "{near}");
        let mut prev = 0.0;
        for i in 0..=500 {
            let d = 10_f64.powf(i as f64 / 100.0);
            let f = faint_light_fraction(d, 10.0);
            assert!(
                f >= prev && f <= 1.0,
                "not monotonic at {d} pc: {f} < {prev}"
            );
            prev = f;
        }
    }

    /// The bins run from bright to faint without overlaps (the empty M_V −2 row of Flynn et al.
    /// 2006 is the only gap), and the weights sum to 1.
    #[test]
    fn bins_are_ordered_and_weights_sum_to_one() {
        for pair in LOCAL_LF.windows(2) {
            let gap = pair[1].m_bright - pair[0].m_faint;
            assert!(
                gap == 0.0 || (pair[0].m_faint == -2.5 && gap == 1.0),
                "gap {gap}"
            );
        }
        let sum: f64 = faint_bins().iter().map(|b| b[2]).sum();
        assert!((sum - 1.0).abs() < 1e-12, "{sum}");
    }

    /// Cross-checks of the table against what its sources state:
    /// - number: ≈ 0.1 stars/pc³ (the 10 pc census, Reylé et al. 2021, A&A 650, A201: 379 stars
    ///   incl. white dwarfs → 0.090 pc⁻³, with its 85 brown dwarfs → 0.111 pc⁻³);
    /// - light of the bins from M_V = −1 on: Flynn et al. 2006 §2.3 give 0.047 L☉,V/pc³ for the
    ///   Heidelberg function over M_V ≥ −1;
    /// - total light: within 10 % of their 0.056 (Tuorla, all stars).
    #[test]
    fn table_matches_its_sources() {
        let n = local_number_density();
        assert!((0.09..=0.115).contains(&n), "{n} stars/pc³");
        let from_m_v_minus_1: f64 = LOCAL_LF
            .iter()
            .filter(|b| b.m_bright >= -1.5)
            .map(bin_light)
            .sum();
        assert!(
            (from_m_v_minus_1 / 0.047 - 1.0).abs() < 0.10,
            "{from_m_v_minus_1}"
        );
        let total = local_v_luminosity_density();
        let cited = LOCAL_V_LUMINOSITY_DENSITY_FLYNN_2006;
        assert!((total / cited - 1.0).abs() < 0.10, "{total}");
    }
}
