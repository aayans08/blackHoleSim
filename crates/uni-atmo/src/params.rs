//! Atmosphere parameters and the four M1 presets (Task B3).
//!
//! Coefficients are per metre at the three wavelengths the renderer's R, G and B channels stand
//! for, (680, 550, 440) nm (Bruneton 2017's choice). Layer 2 is a tent-profile layer (Hillaire
//! 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique", CGF 39(4)):
//! Earth's ozone (scattering 0) or a haze layer.
//!
//! Every preset coefficient names its source below. Where a source could not be read in full
//! (Wolff et al. 2009's tables, Tomasko et al. 2008's tables), the comment says exactly which
//! numbers were checked, against what, and which are our own estimates; the task report lists
//! them. Venus and Titan carry the card label "Atmosphere: simplified haze model
//! (approximate)" (C4).
//!
//! Phase functions are Cornette–Shanks (Cornette & Shanks 1992, Applied Optics 31(16), 3152),
//! evaluated per channel. As Bruneton 2017 and Hillaire 2020 do for Earth's aerosol (g = 0.8),
//! Venus's and Titan's asymmetry parameters are used as the Cornette–Shanks g directly.
//! Cornette–Shanks's own mean cosine is then a little larger, 3g(4 + g²)/(5(2 + g²)) (0.78 for
//! g = 0.73; `phase_functions_normalised` checks the formula), and its forward peak
//! correspondingly higher: it stands in, roughly, for the diffraction peak that neither
//! Henyey–Greenstein nor Cornette–Shanks has. Mars's dust g is instead solved so that this mean
//! cosine equals the Mie asymmetry (see [`mars`]).

use glam::{DVec3, Vec3};

/// One planet's atmosphere.
#[derive(Clone, Debug, PartialEq)]
pub struct AtmosphereParams {
    /// Radius of the atmosphere's lower boundary [m] (the ground; Venus: the cloud tops).
    pub bottom_radius_m: f64,
    /// Radius of the top of the shell [m]; nothing scatters above it.
    pub top_radius_m: f64,
    /// Rayleigh scattering coefficient at the bottom [1/m] at (680, 550, 440) nm.
    pub rayleigh_scattering: Vec3,
    /// Rayleigh density scale height [m].
    pub rayleigh_scale_height_m: f32,
    /// Mie (aerosol) scattering coefficient at the bottom [1/m].
    pub mie_scattering: Vec3,
    /// Mie extinction coefficient at the bottom [1/m] (scattering + absorption).
    pub mie_extinction: Vec3,
    /// Mie density scale height [m].
    pub mie_scale_height_m: f32,
    /// Cornette–Shanks asymmetry g per channel (review I6: Mars's blue aureole needs a stronger
    /// blue forward lobe).
    pub mie_g: Vec3,
    /// Layer-2 scattering coefficient at the tent's peak [1/m] (0 for Earth's ozone).
    pub layer2_scattering: Vec3,
    /// Layer-2 extinction coefficient at the tent's peak [1/m].
    pub layer2_extinction: Vec3,
    /// Altitude of the tent's peak [m].
    pub layer2_center_m: f32,
    /// Full width of the tent [m].
    pub layer2_width_m: f32,
    /// Layer-2 phase asymmetry g per channel.
    pub layer2_g: Vec3,
    /// Lambert albedo of the ground under the atmosphere, for the multiple-scattering LUT.
    pub ground_albedo: Vec3,
    /// Where the coefficients come from (papers and tables), shown on the info card.
    pub source: &'static str,
}

/// Altitude of Venus's visible cloud tops above its solid-surface radius [m]: about 70 km at low
/// latitudes (Titov et al. 2018, "Clouds and Hazes of Venus", Space Sci. Rev. 214, 126). The
/// bodies pass (B1) draws Venus's `CloudTops` map at R + this, and the Venus preset's
/// `bottom_radius_m` is the same sphere.
pub const VENUS_CLOUD_TOP_M: f64 = 70_000.0;

/// Boltzmann constant [J/K] (exact since the 2019 SI redefinition; CODATA 2018).
const K_B: f64 = 1.380_649e-23;

/// The wavelengths [nm] the R, G and B channels stand for (Bruneton 2017).
const CHANNEL_NM: [f64; 3] = [680.0, 550.0, 440.0];

/// Rayleigh scattering coefficients [1/m] at (680, 550, 440) nm of a gas with number density
/// `n_m3` whose cross-section at `ref_nm` is `sigma_ref_cm2`, scaled as λ⁻⁴ (the dispersion of
/// the refractive index steepens real cross-sections slightly, to about λ⁻⁴·¹; neglected).
fn rayleigh_coefficients(n_m3: f64, sigma_ref_cm2: f64, ref_nm: f64) -> Vec3 {
    let beta = |nm: f64| (n_m3 * sigma_ref_cm2 * 1e-4 * (ref_nm / nm).powi(4)) as f32;
    Vec3::new(
        beta(CHANNEL_NM[0]),
        beta(CHANNEL_NM[1]),
        beta(CHANNEL_NM[2]),
    )
}

/// CO₂ Rayleigh cross-section at 532.24 nm: 12.4 × 10⁻²⁷ cm² (Sneep & Ubachs 2005, JQSRT 92,
/// 293, cavity ring-down; as tabulated in the A&A literature that cites it).
const CO2_SIGMA_532_CM2: f64 = 12.4e-27;
const SIGMA_REF_NM: f64 = 532.24;

/// Earth: Hillaire 2020 / Bruneton 2017 defaults with bottom radius 6 371 km (B3).
///
/// Values exactly as in Hillaire's sample code (`SetupEarthAtmosphere`, per km there) and
/// Bruneton's `demo.cc`: Rayleigh (5.802, 13.558, 33.1) × 10⁻⁶ m⁻¹, H_R 8 km; Mie scattering
/// 3.996 × 10⁻⁶ and extinction 4.440 × 10⁻⁶ m⁻¹ (Ångström β 5.328 × 10⁻³ over H_M 1.2 km,
/// single-scattering albedo 0.9), g 0.8; ozone absorption (0.650, 1.881, 0.085) × 10⁻⁶ m⁻¹ in a
/// tent from 10 to 40 km peaking at 25 km. Bottom 6 371 km (the mean radius, which is also
/// (a²c)^⅓ of the WGCCRE 2015 ellipsoid), top + 100 km. Ground albedo 0.1 (Bruneton 2017's
/// `kGroundAlbedo`; it only feeds the multiple-scattering LUT; the app may replace it with the
/// map's mean albedo).
pub fn earth() -> AtmosphereParams {
    AtmosphereParams {
        bottom_radius_m: 6_371_000.0,
        top_radius_m: 6_471_000.0,
        rayleigh_scattering: Vec3::new(5.802e-6, 13.558e-6, 33.1e-6),
        rayleigh_scale_height_m: 8_000.0,
        mie_scattering: Vec3::splat(3.996e-6),
        mie_extinction: Vec3::splat(4.440e-6),
        mie_scale_height_m: 1_200.0,
        mie_g: Vec3::splat(0.8),
        layer2_scattering: Vec3::ZERO,
        layer2_extinction: Vec3::new(0.650e-6, 1.881e-6, 0.085e-6),
        layer2_center_m: 25_000.0,
        layer2_width_m: 30_000.0,
        layer2_g: Vec3::ZERO,
        ground_albedo: Vec3::splat(0.1),
        source: "Hillaire 2020 (CGF 39(4)) and Bruneton 2017 Earth defaults; bottom 6 371 km",
    }
}

/// Mars: CO₂ Rayleigh plus dust with per-channel asymmetry (B3), the structure of Collienne et
/// al. 2013 ("Physically Based Rendering of the Martian Atmosphere", GI VR/AR workshop) with
/// measured coefficients.
///
/// - Radii: bottom 3 389.5 km (IAU mean radius, (a²c)^⅓ of WGCCRE 2015's 3 396.19 × 3 376.20 km),
///   top + 80 km.
/// - CO₂ Rayleigh: number density p/(k_B T) at the mean surface, p = 636 Pa and T = 210 K (NASA
///   NSSDCA Mars fact sheet), with Sneep & Ubachs 2005's CO₂ cross-section; scale height 11.1 km
///   (NSSDCA). β_550 = 2.4 × 10⁻⁷ m⁻¹, τ_550 ≈ 0.003: negligible beside the dust, as expected.
/// - Dust: visible optical depth τ = 0.5 (the brief's value; a clear-season level in the MER
///   optical-depth record, Lemmon et al. 2015, Icarus 251, 96) with scale height 11 km (the
///   brief's), grey in extinction (the Mie computation below gives extinction ratios
///   (1.03, 1, 0.97) at (680, 550, 440) nm, so grey to ±3 %).
/// - Single-scattering albedo (0.972, 0.846, 0.711) and Cornette–Shanks g (0.652, 0.711, 0.780)
///   at (680, 550, 440) nm (review I6), from our Mie computation (Bohren & Huffman 1983's
///   algorithm, checked against their reference case x = 5.213, m = 1.55: Q_ext 3.10543,
///   g 0.63314) for spheres with Ehlers et al. 2014's dust refractive index (Applied Optics 53,
///   1808, "Blue moons and Martian sunsets": n = 1.52, k = (0.001, 0.006, 0.013) at these
///   wavelengths, as tabulated by Schneegans et al. 2020, arXiv:2010.03534, Table 1) and
///   Wolff et al. 2009's size distribution (JGR 114, E00D04: r_eff 1.5 µm, v_eff 0.3, the v_eff
///   as quoted by Chen-Chen et al. 2019, arXiv:1905.01073). Mie gives the asymmetry ⟨cos Θ⟩
///   (0.714, 0.767, 0.827); each g is the Cornette–Shanks g with that mean cosine (module
///   docs). Checks at the red end: Wolff et al. 2009's measured single-scattering albedo 0.975
///   at ~650 nm and the MSL sky-radiance asymmetry 0.687 ± 0.081 at ~650 nm, "in good agreement
///   with Wolff et al. 2009" (both as Chen-Chen et al. 2019, arXiv:1905.01074, quote them).
///   Wolff et al. 2009's own 440–550 nm values (their Table 2) could not be read by this task;
///   the blue forward lobe g(440) > g(680) is what makes Mars's sunset aureole blue (Ehlers et
///   al. 2014).
/// - Ground albedo (0.25, 0.15, 0.08): Bond albedo 0.25 (NSSDCA) in red, falling toward blue as
///   Mars's reflectance spectrum does (approximate; it feeds only the multiple-scattering LUT).
pub fn mars() -> AtmosphereParams {
    let bottom = 3_389_500.0;
    let top = bottom + 80_000.0;
    let n_co2 = 636.0 / (K_B * 210.0);
    let dust_h: f64 = 11_000.0;
    let tau = 0.5;
    // Exponential column inside the shell: τ = β₀·H·(1 − e^(−80/11)).
    let beta_ext = (tau / (dust_h * (1.0 - (-(top - bottom) / dust_h).exp()))) as f32;
    let albedo = Vec3::new(0.972, 0.846, 0.711);
    AtmosphereParams {
        bottom_radius_m: bottom,
        top_radius_m: top,
        rayleigh_scattering: rayleigh_coefficients(n_co2, CO2_SIGMA_532_CM2, SIGMA_REF_NM),
        rayleigh_scale_height_m: 11_100.0,
        mie_scattering: albedo * beta_ext,
        mie_extinction: Vec3::splat(beta_ext),
        mie_scale_height_m: dust_h as f32,
        mie_g: Vec3::new(0.652, 0.711, 0.780),
        layer2_scattering: Vec3::ZERO,
        layer2_extinction: Vec3::ZERO,
        layer2_center_m: 0.0,
        layer2_width_m: 0.0,
        layer2_g: Vec3::ZERO,
        ground_albedo: Vec3::new(0.25, 0.15, 0.08),
        source: "Mars: CO2 Rayleigh (Sneep & Ubachs 2005; NSSDCA) + dust tau 0.5, H 11 km; \
                 dust albedo and g by Mie from Ehlers et al. 2014 indices, Wolff et al. 2009 sizes",
    }
}

/// Venus above the cloud tops: CO₂ Rayleigh plus the upper haze (B3).
///
/// - Radii: bottom at the cloud tops, 6 051.8 km (WGCCRE 2015) + [`VENUS_CLOUD_TOP_M`]; top
///   + 60 km.
/// - CO₂ Rayleigh: p ≈ 30 hPa, T ≈ 230 K at 70 km (Limaye et al. 2017, Icarus 294, 124, compare
///   the Venus Express profiles with VIRA: 10–30 mbar over 70–80 km), with Sneep & Ubachs 2005's
///   CO₂ cross-section (the 3.5 % N₂ is counted as CO₂). Gas scale height 4.5 km at the cloud
///   top (Titov et al. 2018, "Clouds and Hazes of Venus", Space Sci. Rev. 214:126, §2).
/// - Upper haze (Titov et al. 2018, Table 1 and §2–4): 70–90 km, optical depth 0.2–1.0 at
///   0.63 µm (we take 0.3), well mixed with the gas at low latitudes (aerosol scale height
///   ≈ gas scale height, 4.5 km), sulphuric-acid droplets with r_eff 0.25–0.29 µm and refractive
///   index 1.435–1.45 at 0.55 µm, so non-absorbing (single-scattering albedo 1). Per channel,
///   from our Mie computation (Bohren & Huffman 1983 algorithm, checked against their
///   reference case) for a gamma distribution r_eff 0.27 µm, v_eff 0.1, n = 1.44: extinction
///   relative to 630 nm (0.892, 1.188, 1.429) and asymmetry (0.716, 0.746, 0.760) at
///   (680, 550, 440) nm, used as the Cornette–Shanks g (module docs).
/// - Ground albedo 0.76: the cloud deck's Bond albedo (Haus et al. 2016, Icarus 272, 178), grey.
pub fn venus() -> AtmosphereParams {
    let bottom = 6_051_800.0 + VENUS_CLOUD_TOP_M;
    let top = bottom + 60_000.0;
    let h: f64 = 4_500.0;
    let n_co2 = 3_000.0 / (K_B * 230.0);
    let tau_630 = 0.3;
    let beta_630 = tau_630 / (h * (1.0 - (-(top - bottom) / h).exp()));
    let haze = Vec3::new(0.892, 1.188, 1.429) * beta_630 as f32;
    AtmosphereParams {
        bottom_radius_m: bottom,
        top_radius_m: top,
        rayleigh_scattering: rayleigh_coefficients(n_co2, CO2_SIGMA_532_CM2, SIGMA_REF_NM),
        rayleigh_scale_height_m: h as f32,
        mie_scattering: haze,
        mie_extinction: haze,
        mie_scale_height_m: h as f32,
        mie_g: Vec3::new(0.716, 0.746, 0.760),
        layer2_scattering: Vec3::ZERO,
        layer2_extinction: Vec3::ZERO,
        layer2_center_m: 0.0,
        layer2_width_m: 0.0,
        layer2_g: Vec3::ZERO,
        ground_albedo: Vec3::splat(0.76),
        source: "Venus above the cloud tops: CO2 Rayleigh + H2SO4 upper haze tau(0.63 um) 0.3, \
                 H 4.5 km (Titov et al. 2018); simplified haze model",
    }
}

/// Titan: N₂ Rayleigh plus the orange haze, detached layer as layer 2 (B3).
///
/// - Radii: bottom 2 574.73 km (IAU mean radius, Archinal et al. 2018), top + 600 km.
/// - N₂ Rayleigh: surface p = 1 467 hPa and T = 93.65 K (Huygens HASI, Fulchignoni et al. 2005,
///   Nature 438, 785), N₂'s cross-section taken equal to air's (within ≈ 2 % in the visible),
///   i.e. Earth's coefficients per molecule at Bruneton's sea-level 2.547 × 10²⁵ m⁻³; scale height
///   k_B T/(m g) = 20.6 km, which keeps the surface density and the column (≈ 11 × Earth's).
/// - Main haze (the Mie slot): scale height 65 km above 80 km (Tomasko et al. 2008, PSS 56, 669,
///   as quoted by Vincendon & Langevin 2010, arXiv:1103.3215), total optical depth 2.6 at
///   1.08 µm (Tomasko et al. 2008, same quote) carried to the visible with the near-IR spectral
///   exponent 2.2 (Vincendon & Langevin 2010; range 1.8–2.4): τ = (7.2, 11.5, 18.7). The haze
///   absorbs blue (orange colour): single-scattering albedo (0.93, 0.87, 0.70) and asymmetry
///   (0.60, 0.65, 0.70) are our estimates of Tomasko et al. 2008's wavelength dependence (their
///   tables were not accessible to this task; the task report lists these as unchecked).
/// - Detached haze layer (layer 2): centred at 500 km (Seignovert et al. 2017, arXiv:1704.00842:
///   500 ± 8 km in 2005–2007; West et al. 2011 saw it drop to 380 km by 2010), 40 km full width;
///   peak extinction 8.4 × 10⁻⁸ m⁻¹ at 338 nm (number density 2 × 10⁻² cm⁻³ × σ_ext 4.2 ×
///   10⁻¹² m², Seignovert et al. 2017) carried to the visible with the same exponent 2.2;
///   single-scattering albedo 0.69 at 338 nm there, taken as (0.90, 0.85, 0.75) in the visible
///   (our estimate), asymmetry as the main haze.
/// - Ground albedo 0.15 (approximate; under τ ≈ 10 it barely matters).
pub fn titan() -> AtmosphereParams {
    let bottom = 2_574_730.0;
    let top = bottom + 600_000.0;
    // Earth's preset per molecule (Bruneton's coefficients at 2.547e25 m⁻³), times Titan's
    // surface number density.
    let n_titan = 146_700.0 / (K_B * 93.65);
    let per_molecule = earth().rayleigh_scattering / 2.547e25_f32;
    let rayleigh = per_molecule * n_titan as f32;
    // k_B T / (m g): m = 28.0134 u (N₂), g = 1.352 m/s² (NSSDCA).
    let h_r = K_B * 93.65 / (28.0134 * 1.660_539_07e-27 * 1.352);
    let haze_h: f64 = 65_000.0;
    let tau = |um: f64| 2.6 * (1.08 / um).powf(2.2);
    let shell = 1.0 - (-(top - bottom) / haze_h).exp();
    let beta = |um: f64| (tau(um) / (haze_h * shell)) as f32;
    let haze_ext = Vec3::new(beta(0.68), beta(0.55), beta(0.44));
    let haze_albedo = Vec3::new(0.93, 0.87, 0.70);
    let dhl = |um: f64| (8.4e-8 * (0.338 / um).powf(2.2)) as f32;
    let dhl_ext = Vec3::new(dhl(0.68), dhl(0.55), dhl(0.44));
    let asym = Vec3::new(0.60, 0.65, 0.70);
    AtmosphereParams {
        bottom_radius_m: bottom,
        top_radius_m: top,
        rayleigh_scattering: rayleigh,
        rayleigh_scale_height_m: h_r as f32,
        mie_scattering: haze_ext * haze_albedo,
        mie_extinction: haze_ext,
        mie_scale_height_m: haze_h as f32,
        mie_g: asym,
        layer2_scattering: dhl_ext * Vec3::new(0.90, 0.85, 0.75),
        layer2_extinction: dhl_ext,
        layer2_center_m: 500_000.0,
        layer2_width_m: 40_000.0,
        layer2_g: asym,
        ground_albedo: Vec3::splat(0.15),
        source: "Titan: N2 Rayleigh (Fulchignoni et al. 2005) + haze H 65 km, tau(1.08 um) 2.6 \
                 (Tomasko et al. 2008) + detached layer at 500 km (Seignovert et al. 2017); \
                 simplified haze model",
    }
}

/// The preset for a body by NAIF id: Venus 299, Earth 399, Mars 499, Titan 606; `None` for every
/// other body. The same set as uni-sim's `has_atmosphere` (registry `ATMOSPHERE_BODIES`).
pub fn preset_for(naif: i32) -> Option<AtmosphereParams> {
    match naif {
        299 => Some(venus()),
        399 => Some(earth()),
        499 => Some(mars()),
        606 => Some(titan()),
        _ => None,
    }
}

impl AtmosphereParams {
    /// Checks what the LUTs and the march rely on: finite values, 0 < bottom < top, positive
    /// scale heights, non-negative coefficients, extinction ≥ scattering, |g| < 1 and a
    /// non-negative tent width.
    pub fn validate(&self) -> anyhow::Result<()> {
        let vecs = [
            ("rayleigh_scattering", self.rayleigh_scattering),
            ("mie_scattering", self.mie_scattering),
            ("mie_extinction", self.mie_extinction),
            ("layer2_scattering", self.layer2_scattering),
            ("layer2_extinction", self.layer2_extinction),
            ("ground_albedo", self.ground_albedo),
        ];
        for (name, v) in vecs {
            anyhow::ensure!(v.is_finite() && v.min_element() >= 0.0, "{name} = {v}");
        }
        for (name, g) in [("mie_g", self.mie_g), ("layer2_g", self.layer2_g)] {
            anyhow::ensure!(g.is_finite() && g.abs().max_element() < 1.0, "{name} = {g}");
        }
        anyhow::ensure!(
            self.mie_extinction.cmpge(self.mie_scattering).all()
                && self.layer2_extinction.cmpge(self.layer2_scattering).all(),
            "extinction below scattering"
        );
        anyhow::ensure!(
            self.bottom_radius_m.is_finite()
                && self.bottom_radius_m > 0.0
                && self.top_radius_m.is_finite()
                && self.top_radius_m > self.bottom_radius_m,
            "radii {} .. {}",
            self.bottom_radius_m,
            self.top_radius_m
        );
        anyhow::ensure!(
            self.rayleigh_scale_height_m > 0.0 && self.mie_scale_height_m > 0.0,
            "scale heights {} and {}",
            self.rayleigh_scale_height_m,
            self.mie_scale_height_m
        );
        anyhow::ensure!(
            self.layer2_center_m.is_finite()
                && self.layer2_width_m.is_finite()
                && self.layer2_width_m >= 0.0,
            "layer 2 at {} width {}",
            self.layer2_center_m,
            self.layer2_width_m
        );
        Ok(())
    }
}

/// True when the camera is inside the shell (`|cam_rel_body_m|` below `top_radius_m`) and the
/// Sun is above its local geometric horizon (`sun_dir`, a unit vector from the camera toward the
/// Sun, has a positive component along the local up). The app sets
/// `uni_post::ExposureSettings::daylight` from it (B6's rule; review m12). `cam_rel_body_m` is
/// camera − body centre [m] in any axes `sun_dir` shares. Non-finite input gives `false`.
pub fn camera_in_daylight(
    params: &AtmosphereParams,
    cam_rel_body_m: DVec3,
    sun_dir: DVec3,
) -> bool {
    let r = cam_rel_body_m.length();
    r.is_finite()
        && sun_dir.is_finite()
        && r > 0.0
        && r < params.top_radius_m
        && cam_rel_body_m.dot(sun_dir) > 0.0
}

/// Earth's sea-level vertical Rayleigh optical depth at 550 nm under [`earth`]'s coefficients:
/// β·H·(1 − e^(−100 km / 8 km)), the reference [`scintillation_strength`] is normalised to.
fn earth_sea_level_rayleigh_tau_550() -> f64 {
    let p = earth();
    let h = f64::from(p.rayleigh_scale_height_m);
    f64::from(p.rayleigh_scattering.y)
        * h
        * (1.0 - (-(p.top_radius_m - p.bottom_radius_m) / h).exp())
}

/// Strength of stellar scintillation (twinkle) for a camera at `cam_rel_body_m` (camera − body
/// centre [m]), in [0, 1]: 0 outside the shell. The app copies it into
/// `uni_sky::StarInput::scintillation`, and the star pass (B4) drives the twinkle from it
/// deterministically (owner feedback, 2026-11-22: stars twinkle only under an atmosphere).
///
/// Model (approximate): scintillation comes from refractive-index turbulence in the air above
/// the observer, and falls with the observer's altitude as exp(−h/H₀), H₀ = 8 km, the
/// atmosphere's scale height (Young 1967, AJ 72, 747, "Photometric error analysis. VI";
/// Dravins et al. 1997, PASP 109, 173). We take the Rayleigh optical depth at 550 nm
/// above the camera as the measure of "air above", normalised to Earth's at sea level, so Earth
/// gives Young's exp(−h/8 km) and 1 at the ground, a thin atmosphere (Mars ≈ 0.03, Venus above
/// its clouds ≈ 0.05) little, and Titan saturates at 1. "Air mass" here is the column above the
/// camera; the growth with each star's zenith angle (air mass^1.75, Young 1967) needs the star's
/// direction, which this function does not take, and the M1 star pass (B4) does not model it.
/// Non-finite input gives 0.
pub fn scintillation_strength(params: &AtmosphereParams, cam_rel_body_m: DVec3) -> f32 {
    let r = cam_rel_body_m.length();
    if !r.is_finite() || r >= params.top_radius_m {
        return 0.0;
    }
    let h_r = f64::from(params.rayleigh_scale_height_m);
    if !(h_r > 0.0) {
        return 0.0;
    }
    let h = (r - params.bottom_radius_m).max(0.0);
    let thick = params.top_radius_m - params.bottom_radius_m;
    let tau_above = f64::from(params.rayleigh_scattering.y)
        * h_r
        * ((-h / h_r).exp() - (-thick / h_r).exp()).max(0.0);
    (tau_above / earth_sea_level_rayleigh_tau_550()).clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B6's `daylight` rule (review m12): inside the shell with the Sun above the local horizon.
    #[test]
    fn camera_in_daylight_cases() {
        let p = earth();
        let cam = DVec3::new(0.0, 0.0, p.bottom_radius_m + 10e3);
        let sun = |elev_deg: f64| {
            let e = elev_deg.to_radians();
            DVec3::new(e.cos(), 0.0, e.sin())
        };
        assert!(camera_in_daylight(&p, cam, sun(30.0)));
        assert!(!camera_in_daylight(&p, cam, sun(-5.0)));
        let high = DVec3::new(0.0, 0.0, p.bottom_radius_m + 1_000e3);
        assert!(!camera_in_daylight(&p, high, sun(30.0)));
        assert!(!camera_in_daylight(&p, DVec3::NAN, sun(30.0)));
    }

    /// Owner feedback (stars twinkle only under an atmosphere): 0 outside the shell, rising
    /// toward the ground, never above 1.
    #[test]
    fn scintillation_strength_cases() {
        let p = earth();
        let at = |h: f64| scintillation_strength(&p, DVec3::new(0.0, p.bottom_radius_m + h, 0.0));
        assert_eq!(at(1_000e3), 0.0);
        assert_eq!(at(100e3 + 1.0), 0.0);
        assert!(at(0.0) > at(5e3) && at(5e3) > at(30e3) && at(30e3) > 0.0);
        assert!((at(0.0) - 1.0).abs() < 1e-3, "{}", at(0.0));
        assert_eq!(scintillation_strength(&p, DVec3::NAN), 0.0);
        let m = mars();
        let mars_ground = scintillation_strength(&m, DVec3::new(m.bottom_radius_m, 0.0, 0.0));
        assert!(mars_ground > 0.0 && mars_ground < 0.1, "{mars_ground}");
    }

    /// Vertical optical depth from the ground to the top of a coefficient profile.
    fn column(beta: f32, h: f32, thick: f64) -> f64 {
        f64::from(beta) * f64::from(h) * (1.0 - (-thick / f64::from(h)).exp())
    }

    /// Every preset validates and reproduces the numbers its comments cite: Mars's dust τ 0.5,
    /// Venus's haze τ(630 nm) 0.3 and its Mie extinction ratios, Titan's Rayleigh column ≈ 11 ×
    /// Earth's and haze τ(1.08 µm) 2.6 carried to 550 nm (11.5); `validate` rejects bad input.
    #[test]
    fn presets_validate_and_match_their_sources() {
        for naif in [299, 399, 499, 606] {
            let p = preset_for(naif).unwrap();
            p.validate().unwrap_or_else(|e| panic!("{naif}: {e}"));
        }
        assert!(preset_for(301).is_none());
        let m = mars();
        let thick = m.top_radius_m - m.bottom_radius_m;
        assert!((column(m.mie_extinction.y, m.mie_scale_height_m, thick) - 0.5).abs() < 1e-6);
        assert!(column(m.rayleigh_scattering.y, m.rayleigh_scale_height_m, thick) < 0.005);
        let v = venus();
        let thick = v.top_radius_m - v.bottom_radius_m;
        let tau_630 = column(v.mie_extinction.y, v.mie_scale_height_m, thick) / 1.188;
        assert!((tau_630 - 0.3).abs() < 1e-6, "{tau_630}");
        assert_eq!(v.mie_scattering, v.mie_extinction);
        let t = titan();
        let e = earth();
        let ratio = column(
            t.rayleigh_scattering.y,
            t.rayleigh_scale_height_m,
            t.top_radius_m - t.bottom_radius_m,
        ) / column(
            e.rayleigh_scattering.y,
            e.rayleigh_scale_height_m,
            e.top_radius_m - e.bottom_radius_m,
        );
        assert!(
            (10.0..12.0).contains(&ratio),
            "Titan/Earth Rayleigh column {ratio}"
        );
        let tau_550 = column(
            t.mie_extinction.y,
            t.mie_scale_height_m,
            t.top_radius_m - t.bottom_radius_m,
        );
        assert!(
            (tau_550 - 2.6 * (1.08_f64 / 0.55).powf(2.2)).abs() < 1e-4,
            "{tau_550}"
        );
        let bad = [
            AtmosphereParams {
                top_radius_m: e.bottom_radius_m,
                ..earth()
            },
            AtmosphereParams {
                mie_g: Vec3::splat(1.0),
                ..earth()
            },
            AtmosphereParams {
                mie_scattering: Vec3::splat(1e-5),
                ..earth()
            },
            AtmosphereParams {
                rayleigh_scale_height_m: 0.0,
                ..earth()
            },
            AtmosphereParams {
                layer2_extinction: Vec3::NAN,
                ..earth()
            },
        ];
        for p in bad {
            assert!(p.validate().is_err(), "{p:?}");
        }
    }
}
