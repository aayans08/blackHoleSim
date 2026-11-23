//! Body orientation from the IAU WGCCRE 2015 report (Archinal et al. 2018, CMDA 130:22, and
//! the 2019 erratum) (M1 Task A2).
//!
//! Each model gives the north pole's ICRF right ascension α₀ and declination δ₀ and the prime
//! meridian angle W (WGCCRE 2015 §2, Fig. 1):
//! α₀ = α₀,₀ + α̇·T + Σ aᵢ sin θᵢ, δ₀ = δ₀,₀ + δ̇·T + Σ bᵢ cos θᵢ, W = W₀ + Ẇ·d + Ẅ·d² + Σ cᵢ sin θᵢ,
//! T in Julian centuries and d in days (TDB) from J2000, θᵢ = θᵢ,₀ + θ̇ᵢ·T (+ θ̈ᵢ·T²) the
//! report's phase angles (E1–E13, M1–M10, Ja–Je, S1–S6, U1–U16, N–N7).
//!
//! Transcription source: NAIF `pck00011.tpc` (N. Bachman, 2022-12-27; read 2026-11-22 at
//! https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc), whose orientation
//! data are the 2015 report's [1], the Phobos prime meridian from its 2019 correction [2], and
//! the Earth and Moon from the 2009 report (the 2015 report gives none for them). The tables
//! below copy its `BODYnnn_POLE_RA`, `_POLE_DEC`, `_PM` and `_NUT_PREC_*` values, with the
//! phase angles' rates in degrees per Julian century as the kernel scales them (the report's
//! per-day rates × 36 525) and each periodic term listed as (1-based angle index, coefficient).
//! The report claims 0.1° accuracy in general; NAIF notes the Earth model's prime meridian is
//! off by ≥ 150″.
//!
//! Body-fixed → ICRS is R = R_z(α₀ + 90°)·R_x(90° − δ₀)·R_z(W) (the inverse of the report's
//! ICRF → body-fixed rotation): body +z is the pole, body +x the prime meridian.
//!
//! Outside 3000 BC – AD 3000 (review C1) α₀, δ₀, every periodic term and the quadratic W term
//! are evaluated with T (and d) clamped to [`TABLE2_T_MIN`, `TABLE2_T_MAX`]; only W's linear
//! term advances with the true d. The linear pole rates would otherwise carry Earth's pole
//! 55.7° from J2000 in 10 kyr, where the real pole precesses on a 23.4° circle; the Moon's
//! −1.4 × 10⁻¹² d² is −4.67° at T = −50 and −0.19° at T = +10 instead of −1.87 × 10⁷ ° at
//! 10⁷ yr. Phobos's 12.721 927 97°·T² is the only other quadratic W term, clamped the same way.

use glam::{DQuat, DVec3};
use uni_core::{DAY, Epoch};

use crate::NaifId;
use crate::standish::{TABLE2_T_MAX, TABLE2_T_MIN};

/// One Julian century [days].
const DAYS_PER_CENTURY: f64 = 36_525.0;

/// One phase angle θ = θ₀ + θ̇·T + θ̈·T² [°], T in Julian centuries TDB from J2000.
#[derive(Clone, Copy)]
struct Angle {
    at_j2000: f64,
    rate_per_cy: f64,
    accel_per_cy2: f64,
}

const fn ang(at_j2000: f64, rate_per_cy: f64) -> Angle {
    Angle {
        at_j2000,
        rate_per_cy,
        accel_per_cy2: 0.0,
    }
}

impl Angle {
    fn rad(&self, t_cy: f64) -> f64 {
        (self.at_j2000 + (self.rate_per_cy + self.accel_per_cy2 * t_cy) * t_cy).to_radians()
    }
}

/// One body's orientation model.
struct Model {
    naif: NaifId,
    /// The phase angles its periodic terms index (the kernel's `BODYn_NUT_PREC_ANGLES`).
    system: &'static [Angle],
    /// α₀ = ra[0] + ra[1]·T [°].
    ra: [f64; 2],
    /// δ₀ = dec[0] + dec[1]·T [°].
    dec: [f64; 2],
    /// W = pm[0] + pm[1]·d + pm[2]·d² [°], d in days.
    pm: [f64; 3],
    /// (angle index from 1, coefficient [°]) of the sin θ terms in α₀.
    ra_sin: &'static [(usize, f64)],
    /// cos θ terms in δ₀.
    dec_cos: &'static [(usize, f64)],
    /// sin θ terms in W.
    pm_sin: &'static [(usize, f64)],
}

const NO_TERMS: &[(usize, f64)] = &[];

#[allow(clippy::too_many_arguments)]
const fn periodic_model(
    naif: NaifId,
    system: &'static [Angle],
    ra: [f64; 2],
    dec: [f64; 2],
    pm: [f64; 3],
    ra_sin: &'static [(usize, f64)],
    dec_cos: &'static [(usize, f64)],
    pm_sin: &'static [(usize, f64)],
) -> Model {
    Model {
        naif,
        system,
        ra,
        dec,
        pm,
        ra_sin,
        dec_cos,
        pm_sin,
    }
}

/// A model without periodic terms.
const fn plain(naif: NaifId, ra: [f64; 2], dec: [f64; 2], pm: [f64; 3]) -> Model {
    periodic_model(naif, &[], ra, dec, pm, NO_TERMS, NO_TERMS, NO_TERMS)
}

/// Mercury's M1–M5 (`BODY1_NUT_PREC_ANGLES`; the report's 4.092335°/d … × 36 525).
#[rustfmt::skip]
static MERCURY_ANGLES: [Angle; 5] = [
    ang(174.791_085_7, 149_472.535_875_000_03),
    ang(349.582_171_4, 298_945.071_750_000_06),
    ang(164.373_257_1, 448_417.607_625_000_06),
    ang(339.164_342_9, 597_890.143_500_000_12),
    ang(153.955_428_6, 747_362.679_374_999_95),
];

/// The Earth–Moon E1–E13 (`BODY3_NUT_PREC_ANGLES`; the report's per-day rates × 36 525).
#[rustfmt::skip]
static EARTH_MOON_ANGLES: [Angle; 13] = [
    ang(125.045, -1_935.536_452_5),
    ang(250.089, -3_871.072_905),
    ang(260.008, 475_263.332_872_5),
    ang(176.625, 487_269.629_985),
    ang(357.529, 35_999.050_957_5),
    ang(311.589, 964_468.499_31),
    ang(134.963, 477_198.869_325),
    ang(276.617, 12_006.300_765),
    ang(34.226, 63_863.513_242_5),
    ang(15.134, -5_806.609_357_5),
    ang(119.743, 131.840_64),
    ang(239.961, 6_003.150_382_5),
    ang(25.053, 473_327.796_42),
];

/// Mars M1–M10 (Phobos and Deimos) and the Mars terms 11–26 (`BODY4_NUT_PREC_ANGLES`, with
/// the quadratic coefficient of Phobos's M5; `BODY4_MAX_PHASE_DEGREE = 2`).
#[rustfmt::skip]
static MARS_ANGLES: [Angle; 26] = [
    ang(190.726_466_43, 15_917.108_186_95),
    ang(21.468_924_70, 31_834.279_340_54),
    ang(332.860_827_93, 19_139.896_947_42),
    ang(394.932_564_37, 38_280.796_318_35),
    Angle { at_j2000: 189.632_715_60, rate_per_cy: 41_215_158.184_200_50, accel_per_cy2: 12.711_923_222 },
    ang(121.468_936_64, 660.228_034_74),
    ang(231.050_285_81, 660.991_235_40),
    ang(251.373_140_25, 1_320.501_452_45),
    ang(217.986_359_55, 38_279.961_255_50),
    ang(196.197_294_02, 19_139.836_286_08),
    ang(198.991_226, 19_139.481_998_5),
    ang(226.292_679, 38_280.851_128_1),
    ang(249.663_391, 57_420.725_159_3),
    ang(266.183_510, 76_560.636_795_0),
    ang(79.398_797, 0.504_261_5),
    ang(122.433_576, 19_139.940_747_6),
    ang(43.058_401, 38_280.875_327_2),
    ang(57.663_379, 57_420.751_720_5),
    ang(79.476_401, 76_560.649_500_4),
    ang(166.325_722, 0.504_261_5),
    ang(129.071_773, 19_140.032_824_4),
    ang(36.352_167, 38_281.047_359_1),
    ang(56.668_646, 57_420.929_536_0),
    ang(67.364_003, 76_560.255_221_5),
    ang(104.792_680, 95_700.438_757_8),
    ang(95.391_654, 0.504_261_5),
];

/// Jupiter J1–J10 and Ja–Je (`BODY5_NUT_PREC_ANGLES`; 9 and 10 are 2·J1 and 2·J2).
#[rustfmt::skip]
static JUPITER_ANGLES: [Angle; 15] = [
    ang(73.32, 91_472.9),
    ang(24.62, 45_137.2),
    ang(283.90, 4_850.7),
    ang(355.80, 1_191.3),
    ang(119.90, 262.1),
    ang(229.80, 64.3),
    ang(352.25, 2_382.6),
    ang(113.35, 6_070.0),
    ang(146.64, 182_945.8),
    ang(49.24, 90_274.4),
    ang(99.360_714, 4_850.404_6),
    ang(175.895_369, 1_191.960_5),
    ang(300.323_162, 262.547_5),
    ang(114.012_305, 6_070.247_6),
    ang(49.511_251, 64.300_0),
];

/// Saturn S1–S6, 2·S1, 2·S2 (`BODY6_NUT_PREC_ANGLES`).
#[rustfmt::skip]
static SATURN_ANGLES: [Angle; 8] = [
    ang(353.32, 75_706.7),
    ang(28.72, 75_706.7),
    ang(177.40, -36_505.5),
    ang(300.00, -7_225.9),
    ang(316.45, 506.2),
    ang(345.20, -1_016.3),
    ang(706.64, 151_413.4),
    ang(57.44, 151_413.4),
];

/// Uranus U1–U16, 2·U11, 2·U12 (`BODY7_NUT_PREC_ANGLES`).
#[rustfmt::skip]
static URANUS_ANGLES: [Angle; 18] = [
    ang(115.75, 54_991.87),
    ang(141.69, 41_887.66),
    ang(135.03, 29_927.35),
    ang(61.77, 25_733.59),
    ang(249.32, 24_471.46),
    ang(43.86, 22_278.41),
    ang(77.66, 20_289.42),
    ang(157.36, 16_652.76),
    ang(101.81, 12_872.63),
    ang(138.64, 8_061.81),
    ang(102.23, -2_024.22),
    ang(316.41, 2_863.96),
    ang(304.01, -51.94),
    ang(308.71, -93.17),
    ang(340.82, -75.32),
    ang(259.14, -504.81),
    ang(204.46, -4_048.44),
    ang(632.82, 5_727.92),
];

/// Neptune N, N1–N7, 2·N1, 2·N7 … 9·N7 (`BODY8_NUT_PREC_ANGLES`).
#[rustfmt::skip]
static NEPTUNE_ANGLES: [Angle; 17] = [
    ang(357.85, 52.316),
    ang(323.92, 62_606.6),
    ang(220.51, 55_064.2),
    ang(354.27, 46_564.5),
    ang(75.31, 26_109.4),
    ang(35.36, 14_325.4),
    ang(142.61, 2_824.6),
    ang(177.85, 52.316),
    ang(647.840, 125_213.200),
    ang(355.700, 104.632),
    ang(533.550, 156.948),
    ang(711.400, 209.264),
    ang(889.250, 261.580),
    ang(1_067.100, 313.896),
    ang(1_244.950, 366.212),
    ang(1_422.800, 418.528),
    ang(1_600.650, 470.844),
];

/// Saturn's moons share one pole polynomial (WGCCRE 2015; Titan and Iapetus have their own).
const SAT_RA: [f64; 2] = [40.66, -0.036];
const SAT_DEC: [f64; 2] = [83.52, -0.004];
/// The Uranian moons' pole (Miranda's δ₀ is −15.08°).
const URA_RA: [f64; 2] = [257.43, 0.0];
const URA_DEC: [f64; 2] = [-15.10, 0.0];

/// Every body with a model: the Sun, the planets, Pluto and the moons the M1 ephemeris places
/// by name (Hyperion has none: its rotation is chaotic).
#[rustfmt::skip]
static MODELS: [Model; 31] = [
    plain(10, [286.13, 0.0], [63.87, 0.0], [84.176, 14.184_40, 0.0]),
    periodic_model(199, &MERCURY_ANGLES, [281.0103, -0.0328], [61.4155, -0.0049], [329.5988, 6.138_510_8, 0.0],
        NO_TERMS, NO_TERMS,
        &[(1, 0.010_672_57), (2, -0.001_123_09), (3, -0.000_110_40), (4, -0.000_025_39), (5, -0.000_005_71)]),
    plain(299, [272.76, 0.0], [67.16, 0.0], [160.20, -1.481_368_8, 0.0]),
    plain(399, [0.0, -0.641], [90.0, -0.557], [190.147, 360.985_623_5, 0.0]),
    periodic_model(499, &MARS_ANGLES, [317.269_202, -0.109_275_47], [54.432_516, -0.058_271_05],
        [176.049_863, 350.891_982_443_297, 0.0],
        &[(11, 0.000_068), (12, 0.000_238), (13, 0.000_052), (14, 0.000_009), (15, 0.419_057)],
        &[(16, 0.000_051), (17, 0.000_141), (18, 0.000_031), (19, 0.000_005), (20, 1.591_274)],
        &[(21, 0.000_145), (22, 0.000_157), (23, 0.000_040), (24, 0.000_001), (25, 0.000_001), (26, 0.584_542)]),
    periodic_model(599, &JUPITER_ANGLES, [268.056_595, -0.006_499], [64.495_303, 0.002_413], [284.95, 870.536_000_0, 0.0],
        &[(11, 0.000_117), (12, 0.000_938), (13, 0.001_432), (14, 0.000_030), (15, 0.002_150)],
        &[(11, 0.000_050), (12, 0.000_404), (13, 0.000_617), (14, -0.000_013), (15, 0.000_926)],
        NO_TERMS),
    plain(699, [40.589, -0.036], [83.537, -0.004], [38.90, 810.793_902_4, 0.0]),
    plain(799, [257.311, 0.0], [-15.175, 0.0], [203.81, -501.160_092_8, 0.0]),
    periodic_model(899, &NEPTUNE_ANGLES, [299.36, 0.0], [43.46, 0.0], [249.978, 541.139_775_7, 0.0],
        &[(1, 0.70)], &[(1, -0.51)], &[(1, -0.48)]),
    plain(999, [132.993, 0.0], [-6.163, 0.0], [302.695, 56.362_522_5, 0.0]),
    periodic_model(301, &EARTH_MOON_ANGLES, [269.9949, 0.0031], [66.5392, 0.0130], [38.3213, 13.176_358_15, -1.4e-12],
        &[(1, -3.8787), (2, -0.1204), (3, 0.0700), (4, -0.0172), (6, 0.0072), (10, -0.0052), (13, 0.0043)],
        &[(1, 1.5419), (2, 0.0239), (3, -0.0278), (4, 0.0068), (6, -0.0029), (7, 0.0009), (10, 0.0008), (13, -0.0009)],
        &[(1, 3.5610), (2, 0.1208), (3, -0.0642), (4, 0.0158), (5, 0.0252), (6, -0.0066), (7, -0.0047),
          (8, -0.0046), (9, 0.0028), (10, 0.0052), (11, 0.0040), (12, 0.0019), (13, -0.0044)]),
    // Phobos: W from the 2019 correction; 12.721 927 97°·T² = 9.536 137 031 212 154e-9 °·d².
    periodic_model(401, &MARS_ANGLES, [317.670_716_57, -0.108_443_26], [52.886_272_66, -0.061_347_06],
        [35.187_744_40, 1_128.844_759_28, 9.536_137_031_212_154e-9],
        &[(1, -1.784_283_99), (2, 0.022_128_24), (3, -0.010_282_51), (4, -0.004_755_95)],
        &[(1, -1.075_165_37), (2, 0.006_686_26), (3, -0.006_487_40), (4, 0.002_815_76)],
        &[(1, 1.424_217_69), (2, -0.022_737_83), (3, 0.004_107_11), (4, 0.006_319_64), (5, -1.143)]),
    periodic_model(402, &MARS_ANGLES, [316.657_058_08, -0.105_180_14], [53.509_920_33, -0.059_790_94],
        [79.399_329_54, 285.161_888_99, 0.0],
        &[(6, 3.092_177_26), (7, 0.229_806_37), (8, 0.064_186_55), (9, 0.025_335_37), (10, 0.007_786_95)],
        &[(6, 1.839_360_04), (7, 0.143_253_20), (8, 0.019_114_09), (9, -0.014_825_90), (10, 0.001_924_30)],
        &[(6, -2.739_548_29), (7, -0.399_686_06), (8, -0.065_632_59), (9, -0.029_129_40), (10, 0.016_991_60)]),
    periodic_model(501, &JUPITER_ANGLES, [268.05, -0.009], [64.50, 0.003], [200.39, 203.488_953_8, 0.0],
        &[(3, 0.094), (4, 0.024)], &[(3, 0.040), (4, 0.011)], &[(3, -0.085), (4, -0.022)]),
    periodic_model(502, &JUPITER_ANGLES, [268.08, -0.009], [64.51, 0.003], [36.022, 101.374_723_5, 0.0],
        &[(4, 1.086), (5, 0.060), (6, 0.015), (7, 0.009)],
        &[(4, 0.468), (5, 0.026), (6, 0.007), (7, 0.002)],
        &[(4, -0.980), (5, -0.054), (6, -0.014), (7, -0.008)]),
    periodic_model(503, &JUPITER_ANGLES, [268.20, -0.009], [64.57, 0.003], [44.064, 50.317_608_1, 0.0],
        &[(4, -0.037), (5, 0.431), (6, 0.091)],
        &[(4, -0.016), (5, 0.186), (6, 0.039)],
        &[(4, 0.033), (5, -0.389), (6, -0.082)]),
    periodic_model(504, &JUPITER_ANGLES, [268.72, -0.009], [64.83, 0.003], [259.51, 21.571_071_5, 0.0],
        &[(5, -0.068), (6, 0.590), (8, 0.010)],
        &[(5, -0.029), (6, 0.254), (8, -0.004)],
        &[(5, 0.061), (6, -0.533), (8, -0.009)]),
    periodic_model(601, &SATURN_ANGLES, SAT_RA, SAT_DEC, [333.46, 381.994_555_0, 0.0],
        &[(3, 13.56)], &[(3, -1.53)], &[(3, -13.48), (5, -44.85)]),
    plain(602, SAT_RA, SAT_DEC, [6.32, 262.731_899_6, 0.0]),
    periodic_model(603, &SATURN_ANGLES, SAT_RA, SAT_DEC, [8.95, 190.697_908_5, 0.0],
        &[(4, 9.66)], &[(4, -1.09)], &[(4, -9.60), (5, 2.23)]),
    plain(604, SAT_RA, SAT_DEC, [357.6, 131.534_931_6, 0.0]),
    periodic_model(605, &SATURN_ANGLES, [40.38, -0.036], [83.55, -0.004], [235.16, 79.690_047_8, 0.0],
        &[(6, 3.10)], &[(6, -0.35)], &[(6, -3.08)]),
    plain(606, [39.4827, 0.0], [83.4279, 0.0], [186.5855, 22.576_976_8, 0.0]),
    plain(608, [318.16, -3.949], [75.03, -1.143], [355.2, 4.537_957_2, 0.0]),
    periodic_model(701, &URANUS_ANGLES, URA_RA, URA_DEC, [156.22, -142.835_668_1, 0.0],
        &[(13, 0.29)], &[(13, 0.28)], &[(12, 0.05), (13, 0.08)]),
    periodic_model(702, &URANUS_ANGLES, URA_RA, URA_DEC, [108.05, -86.868_892_3, 0.0],
        &[(14, 0.21)], &[(14, 0.20)], &[(12, -0.09), (14, 0.06)]),
    periodic_model(703, &URANUS_ANGLES, URA_RA, URA_DEC, [77.74, -41.351_431_6, 0.0],
        &[(15, 0.29)], &[(15, 0.28)], &[(15, 0.08)]),
    periodic_model(704, &URANUS_ANGLES, URA_RA, URA_DEC, [6.77, -26.739_493_2, 0.0],
        &[(16, 0.16)], &[(16, 0.16)], &[(16, 0.04)]),
    periodic_model(705, &URANUS_ANGLES, URA_RA, [-15.08, 0.0], [30.70, -254.690_689_2, 0.0],
        &[(11, 4.41), (17, -0.04)], &[(11, 4.25), (17, -0.02)],
        &[(11, 1.15), (12, -1.27), (17, -0.09), (18, 0.15)]),
    periodic_model(801, &NEPTUNE_ANGLES, [299.36, 0.0], [41.17, 0.0], [296.53, -61.257_263_7, 0.0],
        &[(8, -32.35), (10, -6.28), (11, -2.08), (12, -0.74), (13, -0.28), (14, -0.11), (15, -0.07), (16, -0.02), (17, -0.01)],
        &[(8, 22.55), (10, 2.10), (11, 0.55), (12, 0.16), (13, 0.05), (14, 0.02), (15, 0.01)],
        &[(8, 22.25), (10, 6.73), (11, 2.05), (12, 0.74), (13, 0.28), (14, 0.11), (15, 0.05), (16, 0.02), (17, 0.01)]),
    plain(901, [132.993, 0.0], [-6.163, 0.0], [122.695, 56.362_522_5, 0.0]),
];

fn model(naif: NaifId) -> Option<&'static Model> {
    MODELS.iter().find(|m| m.naif == naif)
}

/// T [Julian centuries] clamped to the window where the pole polynomials are used.
fn clamped_t(t: Epoch) -> f64 {
    t.julian_centuries().clamp(TABLE2_T_MIN, TABLE2_T_MAX)
}

/// Σ cᵢ·f(θᵢ(T)) over one list of periodic terms [°].
fn periodic(system: &[Angle], terms: &[(usize, f64)], t_cy: f64, f: fn(f64) -> f64) -> f64 {
    terms
        .iter()
        .map(|&(k, c)| c * f(system[k - 1].rad(t_cy)))
        .sum()
}

/// α₀, δ₀ [°] at `t` (clamped T).
fn pole_deg(m: &Model, t: Epoch) -> (f64, f64) {
    let tc = clamped_t(t);
    let ra = m.ra[0] + m.ra[1] * tc + periodic(m.system, m.ra_sin, tc, f64::sin);
    let dec = m.dec[0] + m.dec[1] * tc + periodic(m.system, m.dec_cos, tc, f64::cos);
    (ra, dec)
}

/// The three parts of W [°]: the linear term at the true d (reduced to [0, 360)), the
/// quadratic term at the clamped d, and the periodic terms at the clamped T.
struct WParts {
    linear: f64,
    quadratic: f64,
    periodic: f64,
}

fn w_parts(m: &Model, t: Epoch) -> WParts {
    let tc = clamped_t(t);
    let dc = tc * DAYS_PER_CENTURY;
    WParts {
        linear: linear_deg(m.pm[0], m.pm[1], t),
        quadratic: m.pm[2] * dc * dc,
        periodic: periodic(m.system, m.pm_sin, tc, f64::sin),
    }
}

/// W₀ + Ẇ·d mod 360° without losing the day fraction at large d: Ẇ·(whole days) is formed as
/// an exact two-product hi + lo, and hi is reduced exactly (`%` is exact in IEEE 754), so a
/// one-day step keeps ≈ 1e-13° at 10⁷ yr, where Ẇ·d alone would be rounded to ≈ 3e-4°.
fn linear_deg(w0: f64, rate_per_day: f64, t: Epoch) -> f64 {
    let days = t.day as f64; // exact below 2⁵³ days
    let hi = rate_per_day * days;
    let lo = rate_per_day.mul_add(days, -hi);
    (hi.rem_euclid(360.0) + lo + rate_per_day * (t.sec / DAY) + w0).rem_euclid(360.0)
}

/// Body-fixed → ICRS rotation, `None` for a body without a rotation model.
///
/// R = R_z(α₀ + 90°)·R_x(90° − δ₀)·R_z(W) (module doc).
pub fn body_to_icrs(naif: NaifId, t: Epoch) -> Option<DQuat> {
    let m = model(naif)?;
    let (ra, dec) = pole_deg(m, t);
    let w = w_deg(m, t);
    Some(
        DQuat::from_rotation_z((ra + 90.0).to_radians())
            * DQuat::from_rotation_x((90.0 - dec).to_radians())
            * DQuat::from_rotation_z(w.to_radians()),
    )
}

/// Prime-meridian angle W [deg], in [0, 360).
pub fn prime_meridian_deg(naif: NaifId, t: Epoch) -> Option<f64> {
    model(naif).map(|m| w_deg(m, t))
}

fn w_deg(m: &Model, t: Epoch) -> f64 {
    let p = w_parts(m, t);
    // `rem_euclid` can round up to exactly 360.0 for a tiny negative sum.
    let w = (p.linear + p.quadratic + p.periodic).rem_euclid(360.0);
    if w < 360.0 { w } else { 0.0 }
}

/// North pole (unit vector, ICRS axes).
pub fn pole_icrs(naif: NaifId, t: Epoch) -> Option<DVec3> {
    let m = model(naif)?;
    let (ra, dec) = pole_deg(m, t);
    let (sin_ra, cos_ra) = ra.to_radians().sin_cos();
    let (sin_dec, cos_dec) = dec.to_radians().sin_cos();
    Some(DVec3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_S: f64 = 86_400.0;
    const YEAR_S: f64 = 365.25 * DAY_S;

    fn years(y: f64) -> Epoch {
        Epoch::J2000.add_seconds(y * YEAR_S)
    }

    /// WGCCRE 2009 Earth (kept by the 2015 report's PCK): W = 190.147° + 360.985 623 5°·d.
    #[test]
    fn earth_rotation_rate() {
        let w0 = prime_meridian_deg(399, Epoch::J2000).expect("Earth model");
        assert!((w0 - 190.147).abs() < 1e-9, "{w0}");
        let w1 = prime_meridian_deg(399, Epoch::J2000.add_seconds(DAY_S)).expect("Earth");
        let rate = (w1 - w0).rem_euclid(360.0) + 360.0;
        assert!((rate - 360.985_623_5).abs() < 1e-7, "{rate}");
    }

    /// The Moon's mean (secular) W rate is 13.176 358 15°/d (WGCCRE): the linear term plus the
    /// −1.4 × 10⁻¹² d² term, whose slope is 0 at J2000. The full W adds the periodic terms,
    /// whose slope over a day is bounded by Σ|cᵢ·θ̇ᵢ|.
    #[test]
    fn moon_rotation_rate() {
        let model = model(301).expect("Moon model");
        let secular = |t: Epoch| {
            let p = w_parts(model, t);
            p.linear + p.quadratic
        };
        let t1 = Epoch::J2000.add_seconds(DAY_S);
        let mean_rate = (secular(t1) - secular(Epoch::J2000)).rem_euclid(360.0);
        assert!((mean_rate - 13.176_358_15).abs() < 1e-6, "{mean_rate}");
        let full_rate = (prime_meridian_deg(301, t1).expect("Moon")
            - prime_meridian_deg(301, Epoch::J2000).expect("Moon"))
        .rem_euclid(360.0);
        let bound: f64 = model
            .pm_sin
            .iter()
            .map(|&(k, c)| (c * model.system[k - 1].rate_per_cy / 36_525.0).abs())
            .sum::<f64>()
            .to_radians();
        assert!(
            (full_rate - mean_rate).abs() <= bound,
            "{full_rate} vs {mean_rate}"
        );
    }

    /// Earth's WGCCRE pole at J2000 is the ICRS pole (α₀ = 0°, δ₀ = 90°).
    #[test]
    fn earth_pole_at_j2000_is_icrs_z() {
        let pole = pole_icrs(399, Epoch::J2000).expect("Earth pole");
        assert!(pole.dot(DVec3::Z) > (1e-6_f64).cos(), "{pole}");
    }

    /// Body-fixed → ICRS = R_z(α₀ + 90°)·R_x(90° − δ₀)·R_z(W): body +z is the pole, and body +x
    /// lies W from the node of the body's equator on the ICRF equator (at RA α₀ + 90°).
    #[test]
    fn body_frame_matches_pole_and_meridian() {
        let t = years(26.9);
        for naif in [
            10, 199, 299, 399, 499, 599, 699, 799, 899, 999, 301, 401, 501, 606, 801,
        ] {
            let q = body_to_icrs(naif, t).expect("model");
            let pole = pole_icrs(naif, t).expect("pole");
            assert!((q * DVec3::Z - pole).length() < 1e-12, "{naif}");
            let node = DVec3::Z.cross(pole).normalize();
            let w = prime_meridian_deg(naif, t).expect("W").to_radians();
            let x = q * DVec3::X;
            let expected = w.cos() * node + w.sin() * pole.cross(node);
            assert!((x - expected).length() < 1e-9, "{naif}: {x} vs {expected}");
        }
        assert!(
            body_to_icrs(607, t).is_none(),
            "Hyperion has no model (chaotic)"
        );
        assert!(pole_icrs(123_456, t).is_none());
    }

    /// Review C1, Review Focus 1: outside 3000 BC – AD 3000 the poles (and their periodic
    /// terms) stop at the nearer window edge, while W's linear term keeps advancing.
    #[test]
    fn poles_frozen_outside_window() {
        let j2000 = pole_icrs(399, Epoch::J2000).expect("Earth");
        let edge_past = years(-5_000.0);
        let edge_future = years(1_000.0);
        for y in [-1e7, -1e6, 1e6, 1e7] {
            let t = years(y);
            let pole = pole_icrs(399, t).expect("Earth");
            let edge = pole_icrs(399, if y < 0.0 { edge_past } else { edge_future }).expect("edge");
            assert!(pole.angle_between(j2000).to_degrees() < 30.0, "{y}: {pole}");
            assert!((pole - edge).length() < 1e-9, "{y}: {pole} vs edge {edge}");
            let rate = (prime_meridian_deg(399, t.add_seconds(DAY_S)).expect("W")
                - prime_meridian_deg(399, t).expect("W"))
            .rem_euclid(360.0)
                + 360.0;
            assert!((rate - 360.985_623_5).abs() < 1e-6, "{y}: {rate}");
            for m in &MODELS {
                let p = pole_icrs(m.naif, t).expect("pole");
                assert!(
                    p.is_finite() && (p.length() - 1.0).abs() < 1e-12,
                    "{}: {p}",
                    m.naif
                );
                let q = body_to_icrs(m.naif, t).expect("frame");
                assert!(q.is_finite() && q.is_normalized(), "{}", m.naif);
                let w = prime_meridian_deg(m.naif, t).expect("W");
                assert!((0.0..360.0).contains(&w), "{}: {w}", m.naif);
            }
        }
        // Edge values (brief): Earth's δ₀ moves 0.557° per century: 27.85° at T = −50 and
        // 5.57° at T = +10.
        let angle = |t| {
            pole_icrs(399, t)
                .expect("Earth")
                .angle_between(j2000)
                .to_degrees()
        };
        assert!(
            (angle(years(-1e7)) - 27.85).abs() < 0.2,
            "{}",
            angle(years(-1e7))
        );
        assert!(
            (angle(years(1e7)) - 5.57).abs() < 0.05,
            "{}",
            angle(years(1e7))
        );
    }

    /// Re-review r1 minor 3: the Moon's −1.4 × 10⁻¹² d² term is evaluated at the clamped d:
    /// −4.67° at T = −50 and −0.19° at T = +10; W stays finite and its mean rate is unchanged.
    #[test]
    fn moon_w_quadratic_term_clamped() {
        let moon = model(301).expect("Moon model");
        for (y, edge_deg) in [(-1e7, -4.67), (1e7, -0.19), (-1e6, -4.67), (1e6, -0.19)] {
            let t = years(y);
            let parts = w_parts(moon, t);
            assert!(
                (parts.quadratic - edge_deg).abs() < 0.005,
                "{y}: {}",
                parts.quadratic
            );
            let edge = if y < 0.0 { -50.0 } else { 10.0 } * 36_525.0;
            assert_eq!(parts.quadratic, -1.4e-12 * edge * edge, "{y}");
            let w = prime_meridian_deg(301, t).expect("W");
            assert!(w.is_finite());
            let next = w_parts(moon, t.add_seconds(DAY_S));
            let rate =
                (next.linear + next.quadratic - parts.linear - parts.quadratic).rem_euclid(360.0);
            assert!((rate - 13.176_358_15).abs() < 1e-6, "{y}: {rate}");
        }
        // Quadratic terms in W elsewhere: Phobos's 12.721 927 97 T² (2019 erratum) is the only
        // other one, clamped the same way.
        let phobos = model(401).expect("Phobos");
        let far = w_parts(phobos, years(1e7)).quadratic;
        assert!((far - 12.721_927_97 * 100.0).abs() < 1e-6, "{far}");
    }
}
