//! The dust and ion tails (Task C1).
//!
//! - **Dust:** `uni_physics::dust::finson_probstein` (Finson & Probstein 1968, ApJ 154, 327) on
//!   `CometView::conic` (J2000 ecliptic, as it expects; lead ruling A.1: never rotate it to ICRS
//!   first), β ∈ [0.001, 0.9] in 32 log steps × ages up to 60 d in 64 steps = 2 048 grains, dust
//!   production ∝ r_h^(−K1/2.5) at emission (SBDB's magnitude law). Each grain is a soft Gaussian
//!   sprite whose width comes from the spacing of its grid neighbours, so the syndynes and
//!   synchrones join into a curved sheet. A grain's light ∝ its weight (production × √β) ×
//!   E☉(r) at its own heliocentric distance × a Henyey–Greenstein phase function with g = 0.6
//!   (the plan's single forward lobe, after Marcus 2007, ICQ 29, 39, whose model is a compound
//!   forward/backward HG; [`DUST_PHASE_G`]). The phase-averaged tail light is 0.5 × the coma's
//!   (v0 ratio; card note `uni_ui::theme::NOTE_TAILS`).
//! - **Ion:** straight along the aberrated anti-sunward direction û = normalize(400 km/s · r̂_h −
//!   v_comet) (Brandt 1968, in *Introduction to the Solar Wind*; ruling 8), 0.3 AU / r_h long (at
//!   most 1 AU), falling off exponentially along its length, a Gaussian 10⁵ km wide (FWHM)
//!   across it, CO⁺ blue (427 nm chromaticity, gamut-mapped, then desaturated 50 % toward white),
//!   0.2 × the coma's light (v0). Drawn as a ribbon that turns about its axis to face the camera,
//!   with the surface brightness of an optically thin cylinder ([`ion_tail_brightness`]).
//! - **Which comets, and how often** (ruling round 2, review I13): only the
//!   [`MAX_TAILED_COMETS`] most prominent comets on screen (brightest `total_v` from the camera)
//!   get tails ([`tailed_comets`]). A tail is recomputed when the epoch has moved more than
//!   [`TAIL_SIM_REFRESH_S`] of sim time since its last refresh but never more often than every
//!   [`TAIL_REFRESH_WALL_S`] of wall time ([`tail_schedule`]), and the grain work is spread over
//!   frames in chunks of whole β rows within ≤ [`TAIL_CPU_SHARE_MS`] of CPU per frame
//!   ([`TailManager`]). Refresh times are kept by `ObjectId`, because `Snapshot::comets` changes
//!   order and membership from frame to frame (ruling C.8).
//! - Only comets with r_h < 5 AU (`uni_sim::snapshot::COMET_ACTIVE_AU`) get tails, and only
//!   those with SBDB's M1 and K1 (no magnitude law, no light).
//!
//! Neither tail dims the sky behind it (optically thin), so neither writes `lens_mask`.

use std::collections::HashMap;

use glam::{DVec3, Vec3};
use uni_core::{AU, Epoch};
use uni_physics::dust::{DustGrain, DustParams, finson_probstein};
use uni_physics::kepler::Conic;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{COMET_ACTIVE_AU, CometView};

use super::{MAX_TAILED_COMETS, TAIL_REFRESH_WALL_S, line_colour};

/// Solar-wind speed [m/s] for the ion tail's aberration (ruling 8; Brandt 1968).
pub const SOLAR_WIND_MPS: f64 = 400.0e3;
/// Ion-tail length at r_h = 1 AU [m]; it scales as 1/r_h (plan C1).
pub const ION_TAIL_LENGTH_AT_1AU_M: f64 = 0.3 * AU;
/// Longest ion tail drawn [m] (plan C1).
pub const ION_TAIL_MAX_LENGTH_M: f64 = AU;
/// Ion-tail width [m], the FWHM of its Gaussian cross-section (plan C1: 10⁵ km).
pub const ION_TAIL_WIDTH_M: f64 = 1.0e8;
/// The ion tail's light as a fraction of the coma's (plan C1, v0).
pub const ION_TAIL_LIGHT: f64 = 0.2;
/// e-foldings of the ion tail's exponential fall-off over its length (a choice: the brightness
/// at the tip is e^(−3) = 5 % of the base's before the profile is brought to 0 there).
pub const ION_TAIL_EFOLDS: f64 = 3.0;
/// Segments of the ion-tail ribbon (twin of `ION_SEGMENTS` in `comet.wgsl`).
pub const ION_SEGMENTS: u32 = 32;
/// The ribbon reaches this many σ of the cross-section on each side (light beyond: 5 × 10⁻⁴).
pub const ION_HALF_WIDTH_SIGMAS: f64 = 3.5;

/// The dust tail's phase-averaged light as a fraction of the coma's (plan C1, v0).
pub const DUST_TAIL_LIGHT: f64 = 0.5;
/// Henyey–Greenstein asymmetry of the dust's phase function (plan C1). Checked 2026-11-23: the
/// plan cites Marcus 2007 as "Icarus 191"; the paper is Marcus 2007, ICQ 29, 39, and its model is
/// a compound HG with a strong forward lobe (k = 0.95, as Li & Jewitt 2015, arXiv:1503.00387,
/// quote it; they fit g_f = 0.926, g_b = −0.584 for C/2010 X1). The single g = 0.6 lobe is the
/// plan's simplification: it under-predicts the forward-scattering surge near phase 180°.
pub const DUST_PHASE_G: f64 = 0.6;
/// The Finson–Probstein grid (plan C1): β ∈ [0.001, 0.9], 32 log steps; ages to 60 d, 64 steps.
pub const DUST_PARAMS: DustParams = DustParams {
    beta_min: 0.001,
    beta_max: 0.9,
    n_beta: 32,
    max_age_days: 60.0,
    n_age: 64,
};
/// Grains per tail: 32 × 64.
pub const GRAINS_PER_TAIL: usize = 2_048;
/// A sprite's σ as a fraction of the larger spacing to its grid neighbours (a choice: Gaussians
/// spaced 1/0.6 σ apart blend into a smooth sheet without washing out the syndynes' curve).
pub const SPRITE_SPACING_FRACTION: f64 = 0.6;
/// Sprites are cut at this many σ (the profile is shifted to reach 0 there and renormalised).
pub const SPRITE_SIGMAS: f64 = 3.0;

/// A tail is recomputed once the epoch has moved this far [s of sim time] from its last
/// refresh (plan C1: 1 h).
pub const TAIL_SIM_REFRESH_S: f64 = 3_600.0;
/// CPU share of the tail work per frame [ms] (plan C1, performance budget).
pub const TAIL_CPU_SHARE_MS: f64 = 1.0;
/// The plan's cost of one Kepler solve [µs] (review I13: "≈ 0.5 µs each"), the nominal rate the
/// schedule is tested at; the renderer measures the real rate.
pub const KEPLER_SOLVE_US: f64 = 0.5;
/// Kepler solves the nominal per-frame budget allows: 1 ms / 0.5 µs.
pub const NOMINAL_SOLVES_PER_FRAME: usize = 2_000;

/// CIE 1931 2° colour-matching functions x̄, ȳ, z̄ at 427 nm, the CO⁺ comet-tail band (CIE 2018
/// data table, DOI 10.25039/CIE.DS.xvudnb9b; read 2026-11-23).
pub const CO_PLUS_CMF: [f64; 3] = [0.244_879_7, 0.008_908_72, 1.188_497_1];

/// Direction of the ion tail (unit, ICRS) from the Sun → nucleus geometry: `sun_rel` is the Sun
/// relative to the nucleus [m] (`CometView::sun_rel_m`), `helio_vel` the comet's heliocentric
/// velocity [m/s]. Anti-sunward, aberrated by the 400 km/s solar wind (module docs). Zero when
/// `sun_rel` is zero.
pub fn ion_tail_dir(sun_rel: DVec3, helio_vel: DVec3) -> DVec3 {
    let anti_sun = (-sun_rel).normalize_or_zero();
    if anti_sun == DVec3::ZERO {
        return DVec3::ZERO;
    }
    // The solar wind as the comet sees it: its radial velocity minus the comet's own.
    (SOLAR_WIND_MPS * anti_sun - helio_vel).normalize_or_zero()
}

/// The ion tail's length [m] at heliocentric distance `r_au`: 0.3 AU / r_h, at most 1 AU.
pub fn ion_tail_length_m(r_au: f64) -> f64 {
    (ION_TAIL_LENGTH_AT_1AU_M / r_au.max(0.0)).min(ION_TAIL_MAX_LENGTH_M)
}

/// The ion tail's surface brightness [cd/m²] on its ribbon at distance `s_m` along the axis from
/// the nucleus and `x_m` across it, for a tail of `length_m` carrying `intensity_cd`, seen at
/// angle ψ to its axis (`sin_psi`). The light per unit length falls as e^(−s/λ) − e^(−L/λ),
/// λ = L/[`ION_TAIL_EFOLDS`] (0 at the tip), the cross-section is a Gaussian of FWHM
/// [`ION_TAIL_WIDTH_M`], and the ribbon's brightness is divided by sin ψ (at least w/L) because an
/// optically thin cylinder's projected light is foreshortened into a shorter streak. Over the
/// ribbon ∫S sin ψ ds dx = I. CPU twin of the shader's; probed on the GPU.
pub fn ion_tail_brightness(
    s_m: f64,
    x_m: f64,
    length_m: f64,
    intensity_cd: f64,
    sin_psi: f64,
) -> f64 {
    if !(s_m >= 0.0 && s_m < length_m) {
        return 0.0;
    }
    let lambda = length_m / ION_TAIL_EFOLDS;
    let tip = (-ION_TAIL_EFOLDS).exp();
    // ∫₀^L (e^(−s/λ) − e^(−L/λ)) ds.
    let norm = lambda * (1.0 - tip) - length_m * tip;
    let along = ((-s_m / lambda).exp() - tip) / norm;
    let sigma = ion_sigma_m();
    let t = x_m / sigma;
    let across = (-0.5 * t * t).exp() / (std::f64::consts::TAU.sqrt() * sigma);
    let sin_eff = sin_psi.max(ION_TAIL_WIDTH_M / length_m);
    intensity_cd * along * across / sin_eff
}

/// σ of the ion tail's Gaussian cross-section [m]: FWHM / (2√(2 ln 2)).
pub(crate) fn ion_sigma_m() -> f64 {
    ION_TAIL_WIDTH_M / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt())
}

/// The ion tail's colour, linear Rec. 709 with luminance 1 (module docs).
pub fn ion_colour() -> Vec3 {
    0.5 * line_colour(CO_PLUS_CMF) + 0.5 * Vec3::ONE
}

/// Henyey & Greenstein 1941 (ApJ 93, 70) phase function with g = [`DUST_PHASE_G`], normalised
/// to a mean of 1 over the sphere, at scattering-angle cosine `cos_theta` (1 = forward). CPU twin
/// of the shader's; probed on the GPU.
pub fn dust_phase(cos_theta: f64) -> f64 {
    let g = DUST_PHASE_G;
    let denom = 1.0 + g * g - 2.0 * g * cos_theta.clamp(-1.0, 1.0);
    (1.0 - g * g) / (denom * denom.sqrt())
}

/// A grain sprite's normalised profile [1/m²] at distance `r` from its centre for width `sigma`:
/// a Gaussian cut at [`SPRITE_SIGMAS`]·σ, shifted to reach 0 there and renormalised so its
/// integral over the plane is 1. CPU twin of the shader's; probed on the GPU.
pub fn sprite_profile(r: f64, sigma: f64) -> f64 {
    if !(sigma > 0.0 && r.abs() < SPRITE_SIGMAS * sigma) {
        return 0.0;
    }
    let cut2 = SPRITE_SIGMAS * SPRITE_SIGMAS;
    let edge = (-0.5 * cut2).exp();
    // ∫ over the disk r < cut·σ of (e^(−r²/2σ²) − edge) dA = 2πσ²[1 − edge·(1 + cut²/2)].
    let norm = std::f64::consts::TAU * sigma * sigma * (1.0 - edge * (1.0 + 0.5 * cut2));
    let t = r / sigma;
    ((-0.5 * t * t).exp() - edge) / norm
}

/// Whether `view` can carry a coma and tails: active (r_h < 5 AU) with SBDB's M1 and K1.
pub(crate) fn is_active(view: &CometView) -> bool {
    view.r_au < COMET_ACTIVE_AU
        && view.m1.is_some()
        && view.k1.is_some()
        && view.total_v.is_some_and(f64::is_finite)
}

/// The comets that draw tails: indices into `comets` of the at most [`MAX_TAILED_COMETS`]
/// brightest (`total_v`, then id) active ones, brightest first. The caller passes only comets
/// whose bounds are on screen.
pub fn tailed_comets(comets: &[CometView]) -> Vec<usize> {
    let mut picked: Vec<usize> = (0..comets.len())
        .filter(|&i| is_active(&comets[i]))
        .collect();
    picked.sort_by(|&a, &b| {
        let v = |i: usize| comets[i].total_v.unwrap_or(f64::INFINITY);
        v(a).total_cmp(&v(b)).then(comets[a].id.cmp(&comets[b].id))
    });
    picked.truncate(MAX_TAILED_COMETS);
    picked
}

/// Which of `comets` (indices into it) get a tail refresh this frame at wall time `wall_s` [s],
/// given each comet's last refresh `last_refresh` [wall s] by id (module docs): the
/// [`tailed_comets`] whose last refresh is at least [`TAIL_REFRESH_WALL_S`] ago (or that have
/// none), never-refreshed first, then the longest-waiting. The renderer also skips those whose
/// tail is less than [`TAIL_SIM_REFRESH_S`] of sim time old, and works down the list within its
/// per-frame CPU share ([`TailManager`]).
pub fn tail_schedule(
    comets: &[CometView],
    last_refresh: &HashMap<ObjectId, f64>,
    wall_s: f64,
) -> Vec<usize> {
    let mut due: Vec<(usize, f64)> = tailed_comets(comets)
        .into_iter()
        .filter_map(|i| match last_refresh.get(&comets[i].id) {
            None => Some((i, f64::NEG_INFINITY)),
            Some(&t) if wall_s - t >= TAIL_REFRESH_WALL_S => Some((i, t)),
            Some(_) => None,
        })
        .collect();
    // Stable: equally old refreshes keep the brightness order.
    due.sort_by(|a, b| a.1.total_cmp(&b.1));
    due.into_iter().map(|(i, _)| i).collect()
}

/// The grains of β rows `rows` (indices 0..32 into [`DUST_PARAMS`]'s β grid) of `conic` at
/// `t_obs`, with dust production ∝ r_h^(−K1/2.5) (module docs). Row-major as `finson_probstein`
/// (β-major, ages within a row).
pub(crate) fn dust_rows(
    conic: &Conic,
    k1: f64,
    t_obs: Epoch,
    rows: std::ops::Range<u32>,
) -> Vec<DustGrain> {
    let n = DUST_PARAMS.n_beta;
    let (j0, j1) = (rows.start.min(n), rows.end.min(n));
    if j0 >= j1 {
        return Vec::new();
    }
    // The j-th β of the full grid, as `DustParams` spaces it (log-spaced, endpoints exact); a
    // sub-grid from β(j0) to β(j1 − 1) in j1 − j0 steps has the same ratio between rows.
    let beta = |j: u32| {
        let p = DUST_PARAMS;
        if j == 0 {
            p.beta_min
        } else if j + 1 == n {
            p.beta_max
        } else {
            p.beta_min * (p.beta_max / p.beta_min).powf(f64::from(j) / f64::from(n - 1))
        }
    };
    let p = DustParams {
        beta_min: beta(j0),
        beta_max: beta(j1 - 1),
        n_beta: j1 - j0,
        ..DUST_PARAMS
    };
    // Production ∝ r_h^(−K1/2.5): SBDB's total-magnitude law m = M1 + 5 log Δ + K1 log r_h
    // (plan C1).
    finson_probstein(conic, t_obs, &p, &|r_h| r_h.powf(-k1 / 2.5))
}

/// One grain as the dust pass draws it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TailGrain {
    /// Offset from the nucleus at the refresh epoch [m], ICRS.
    pub(crate) offset_m: DVec3,
    /// Sprite σ [m] (neighbour spacing).
    pub(crate) sigma_m: f64,
    /// Light weight, normalised so Σ wᵢ·(r_h/rᵢ)² = 1 at the refresh epoch (rᵢ the grain's
    /// heliocentric distance): the pass multiplies by the dust light × (r_h/rᵢ)² × phase.
    pub(crate) weight: f64,
}

/// A computed dust tail.
#[derive(Clone, Debug)]
pub(crate) struct TailState {
    /// The epoch the grains were computed for.
    pub(crate) epoch: Epoch,
    pub(crate) grains: Vec<TailGrain>,
    /// Largest |offset| + [`SPRITE_SIGMAS`]·σ [m]: the tail's reach from the nucleus.
    pub(crate) reach_m: f64,
    /// Bumped at every completed refresh (the renderer re-uploads on a change).
    pub(crate) version: u64,
    /// Wall time [s] the comet last drew a tail (eviction).
    pub(crate) last_seen_wall_s: f64,
}

/// The sprites of a complete β × age grid of grains (`grains` in `finson_probstein`'s order) seen
/// from a nucleus whose Sun is at `sun_rel` [m] (module docs).
///
/// σ = [`SPRITE_SPACING_FRACTION`] × the larger of the mean distances to the grain's β
/// neighbours (same age) and to its age neighbours (same β; the youngest grain's younger
/// neighbour is the nucleus), so neighbouring sprites always overlap. Weights are normalised so
/// that Σ wᵢ·(r_h/rᵢ)² = 1, rᵢ = |offsetᵢ − sun_rel| the grain's heliocentric distance.
pub(crate) fn tail_grains(grains: &[DustGrain], sun_rel: DVec3) -> Vec<TailGrain> {
    let n_age = DUST_PARAMS.n_age as usize;
    let n_beta = grains.len() / n_age;
    let at = |j: usize, k: usize| grains[j * n_age + k].pos_rel_nucleus_m;
    let mean_distance = |p: DVec3, others: &[DVec3]| {
        others.iter().map(|o| (*o - p).length()).sum::<f64>() / others.len().max(1) as f64
    };
    let r_h2 = sun_rel.length_squared();
    let lit = |g: &DustGrain| g.weight * r_h2 / (g.pos_rel_nucleus_m - sun_rel).length_squared();
    let total: f64 = grains[..n_beta * n_age].iter().map(lit).sum();
    let norm = if total > 0.0 && total.is_finite() {
        1.0 / total
    } else {
        0.0
    };
    let mut out = Vec::with_capacity(n_beta * n_age);
    for j in 0..n_beta {
        for k in 0..n_age {
            let p = at(j, k);
            let beta_nb: Vec<DVec3> = [j.checked_sub(1), (j + 1 < n_beta).then_some(j + 1)]
                .into_iter()
                .flatten()
                .map(|jj| at(jj, k))
                .collect();
            let age_nb: Vec<DVec3> = [
                Some(k.checked_sub(1).map_or(DVec3::ZERO, |kk| at(j, kk))),
                (k + 1 < n_age).then(|| at(j, k + 1)),
            ]
            .into_iter()
            .flatten()
            .collect();
            let spacing = mean_distance(p, &beta_nb).max(mean_distance(p, &age_nb));
            out.push(TailGrain {
                offset_m: p,
                // 1 m floor: a sprite is never degenerate.
                sigma_m: (SPRITE_SPACING_FRACTION * spacing).max(1.0),
                weight: grains[j * n_age + k].weight * norm,
            });
        }
    }
    out
}

/// The reach of a tail of `grains` from the nucleus [m]: its farthest sprite edge.
pub(crate) fn tail_reach_m(grains: &[TailGrain]) -> f64 {
    grains
        .iter()
        .map(|g| g.offset_m.length() + SPRITE_SIGMAS * g.sigma_m)
        .fold(0.0, f64::max)
}

/// A conservative reach [m] of a dust tail not computed yet, for culling: 1.5 × the drift
/// ½·β·g☉(r_h)·t² of the lightest grain (β = 0.9) at the oldest age (60 d), at most 3 AU.
/// Radiation pressure is β times the Sun's gravity g☉ = GM☉/r² (Burns, Lamy & Soter 1979, Icarus
/// 40, 1); a grain that drifts outward feels less, so ½βg☉t² bounds the drift; the factor 1.5
/// covers the orbital shear the formula leaves out.
pub(crate) fn dust_reach_estimate_m(r_au: f64) -> f64 {
    let r = r_au.max(1e-3) * AU;
    let t = DUST_PARAMS.max_age_days * uni_core::DAY;
    let drift = 0.5 * DUST_PARAMS.beta_max * uni_core::GM_SUN / (r * r) * t * t;
    (1.5 * drift).min(3.0 * AU)
}

/// What the tail work did in one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct TailWork {
    /// Kepler solves this frame (grains plus the nucleus's emission states).
    pub(crate) solves: usize,
    /// Comets whose refresh started this frame.
    pub(crate) started: Vec<ObjectId>,
    /// Comets whose refresh completed this frame.
    pub(crate) completed: Vec<ObjectId>,
}

/// A refresh in progress: the grid is filled row by row over frames, all for one epoch.
#[derive(Clone, Debug)]
struct TailJob {
    id: ObjectId,
    epoch: Epoch,
    conic: Conic,
    k1: f64,
    sun_rel: DVec3,
    grains: Vec<DustGrain>,
    rows_done: u32,
}

/// The dust tails' CPU side (no GPU objects, so it is tested on its own): per-comet tails keyed
/// by id, the schedule's refresh times, and the refresh in progress.
#[derive(Debug, Default)]
pub(crate) struct TailManager {
    pub(crate) tails: HashMap<ObjectId, TailState>,
    pub(crate) last_refresh: HashMap<ObjectId, f64>,
    job: Option<TailJob>,
    next_version: u64,
}

/// A tail not drawn for this long [wall s] is dropped (memory; it is recomputed on return).
pub(crate) const TAIL_EVICT_WALL_S: f64 = 30.0;

impl TailManager {
    /// One frame of tail work for `comets` (the comets whose bounds are on screen) at `epoch`
    /// and wall time `wall_s`, within `max_solves` Kepler solves (at least one β row is always
    /// done while a refresh is due, so the work never stalls).
    pub(crate) fn update(
        &mut self,
        comets: &[CometView],
        epoch: Epoch,
        wall_s: f64,
        max_solves: usize,
    ) -> TailWork {
        let mut work = TailWork::default();
        let tailed = tailed_comets(comets);
        let is_tailed = |id: ObjectId| tailed.iter().any(|&i| comets[i].id == id);
        for &i in &tailed {
            if let Some(t) = self.tails.get_mut(&comets[i].id) {
                t.last_seen_wall_s = wall_s;
            }
        }
        // A comet that no longer draws a tail loses its refresh in progress.
        if self.job.as_ref().is_some_and(|job| !is_tailed(job.id)) {
            self.job = None;
        }
        let n_age = DUST_PARAMS.n_age as usize;
        loop {
            if self.job.is_none() {
                // Starting a refresh only pays once a row fits (or nothing was done yet).
                if work.solves > 0 && max_solves.saturating_sub(work.solves) < 2 * n_age {
                    break;
                }
                let stale = |id: ObjectId| {
                    self.tails
                        .get(&id)
                        .is_none_or(|t| epoch.seconds_since(t.epoch).abs() > TAIL_SIM_REFRESH_S)
                };
                let next = tail_schedule(comets, &self.last_refresh, wall_s)
                    .into_iter()
                    .find(|&i| stale(comets[i].id));
                let Some((view, k1)) = next.and_then(|i| Some((&comets[i], comets[i].k1?))) else {
                    break;
                };
                self.last_refresh.insert(view.id, wall_s);
                work.started.push(view.id);
                self.job = Some(TailJob {
                    id: view.id,
                    epoch,
                    conic: view.conic,
                    k1,
                    sun_rel: view.sun_rel_m,
                    grains: Vec::with_capacity(GRAINS_PER_TAIL),
                    rows_done: 0,
                });
            }
            let Some(job) = self.job.as_mut() else {
                break;
            };
            // Each chunk solves the nucleus once per age, then one grain per β row and age.
            let rows_left = DUST_PARAMS.n_beta - job.rows_done;
            let fit = max_solves.saturating_sub(work.solves).saturating_sub(n_age) / n_age;
            let rows = match (fit, work.solves) {
                (0, 0) => 1,
                (0, _) => break,
                (fit, _) => u32::try_from(fit).unwrap_or(u32::MAX).min(rows_left),
            };
            let range = job.rows_done..job.rows_done + rows;
            job.grains
                .extend(dust_rows(&job.conic, job.k1, job.epoch, range));
            job.rows_done += rows;
            work.solves += (rows as usize + 1) * n_age;
            if let Some(done) = self
                .job
                .take_if(|j| j.rows_done >= DUST_PARAMS.n_beta)
            {
                let grains = tail_grains(&done.grains, done.sun_rel);
                self.next_version += 1;
                self.tails.insert(
                    done.id,
                    TailState {
                        epoch: done.epoch,
                        reach_m: tail_reach_m(&grains),
                        grains,
                        version: self.next_version,
                        last_seen_wall_s: wall_s,
                    },
                );
                work.completed.push(done.id);
            }
        }
        self.tails
            .retain(|_, t| wall_s - t.last_seen_wall_s <= TAIL_EVICT_WALL_S);
        let tails = &self.tails;
        self.last_refresh
            .retain(|id, t| tails.contains_key(id) || wall_s - *t <= TAIL_EVICT_WALL_S);
        work
    }
}

#[cfg(test)]
mod tests {
    use uni_core::DAY;
    use uni_physics::smallbody::{conic_from_perihelion, heliocentric_state_icrs};

    use super::*;
    use crate::comet::luminance;
    use crate::comet::test_support::{comet_on, static_view};

    fn angle_deg(a: DVec3, b: DVec3) -> f64 {
        a.cross(b).length().atan2(a.dot(b)).to_degrees()
    }

    /// Ruling 8: a comet moving at 40 km/s perpendicular to the Sun line has its ion tail
    /// atan(40/400) = 5.711° off anti-sunward, on the side it is moving away from; at rest the
    /// tail is exactly anti-sunward.
    #[test]
    fn ion_tail_aberration() {
        let want = (40.0_f64 / 400.0).atan().to_degrees();
        assert!((want - 5.711).abs() < 1e-3);
        for sun_rel in [
            DVec3::new(-AU, 0.0, 0.0),
            DVec3::new(0.3, -0.5, 0.8).normalize() * 0.7 * AU,
        ] {
            let anti = -sun_rel.normalize();
            let side = anti.any_orthonormal_vector();
            let v = 40.0e3 * side;
            let u = ion_tail_dir(sun_rel, v);
            assert!((u.length() - 1.0).abs() < 1e-12);
            let got = angle_deg(u, anti);
            assert!((got - want).abs() < 0.01, "{got}° vs {want}°");
            assert!(u.dot(side) < 0.0, "the tail lags the motion");
            let rest = ion_tail_dir(sun_rel, DVec3::ZERO);
            assert!(angle_deg(rest, anti) < 1e-9, "{rest} vs {anti}");
        }
        assert_eq!(ion_tail_dir(DVec3::ZERO, DVec3::X), DVec3::ZERO);
        assert!((ion_tail_length_m(1.0) - 0.3 * AU).abs() < 1.0);
        assert!((ion_tail_length_m(0.6) - 0.5 * AU).abs() < 1.0);
        assert_eq!(ion_tail_length_m(0.1), AU);
    }

    /// The ion ribbon's profile integrates to the tail's light, ∫∫ S sin ψ ds dx = I, and is 0 at
    /// the tip and beyond either end.
    #[test]
    fn ion_tail_profile_integrates_to_its_light() {
        let (l, i) = (0.4 * AU, 2.0e14);
        for sin_psi in [1.0, 0.5, 0.1] {
            let (ns, nx) = (4_000, 400);
            let half = ION_HALF_WIDTH_SIGMAS * ION_TAIL_WIDTH_M / 2.354_820_045;
            let (ds, dx) = (l / ns as f64, 2.0 * half / nx as f64);
            let mut sum = 0.0;
            for a in 0..ns {
                for b in 0..nx {
                    let s = (a as f64 + 0.5) * ds;
                    let x = -half + (b as f64 + 0.5) * dx;
                    sum += ion_tail_brightness(s, x, l, i, sin_psi) * sin_psi * ds * dx;
                }
            }
            assert!((sum / i - 1.0).abs() < 2e-3, "sin ψ {sin_psi}: {sum} of {i}");
        }
        assert_eq!(ion_tail_brightness(l, 0.0, l, i, 1.0), 0.0);
        assert_eq!(ion_tail_brightness(-1.0, 0.0, l, i, 1.0), 0.0);
        assert_eq!(ion_tail_brightness(l + 1.0, 0.0, l, i, 1.0), 0.0);
        // Looking along the axis the brightness is capped at the axial column (sin ψ ≥ w/L).
        assert_eq!(
            ion_tail_brightness(1e9, 0.0, l, i, 0.0),
            ion_tail_brightness(1e9, 0.0, l, i, ION_TAIL_WIDTH_M / l)
        );
        let c = ion_colour();
        assert!((luminance(c) - 1.0).abs() < 1e-5, "{c}");
        assert!(c.min_element() > 0.0 && c.z > c.y && c.z > c.x, "{c}");
    }

    /// The HG phase function has mean 1 over the sphere and favours forward scattering; the
    /// sprite profile integrates to 1 and reaches 0 at its cut.
    #[test]
    fn dust_phase_and_sprite_are_normalised() {
        let n = 20_000;
        let mean = (0..n)
            .map(|k| dust_phase(-1.0 + 2.0 * (k as f64 + 0.5) / n as f64))
            .sum::<f64>()
            / n as f64;
        assert!((mean - 1.0).abs() < 1e-4, "{mean}");
        assert!(dust_phase(1.0) > 10.0 * dust_phase(-1.0));
        let sigma = 3.0e7;
        let m = 20_000;
        let h = SPRITE_SIGMAS * sigma / m as f64;
        let total: f64 = (0..m)
            .map(|k| {
                let r = (k as f64 + 0.5) * h;
                2.0 * std::f64::consts::PI * r * sprite_profile(r, sigma) * h
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-6, "{total}");
        assert_eq!(sprite_profile(SPRITE_SIGMAS * sigma, sigma), 0.0);
        assert_eq!(sprite_profile(4.0 * sigma, sigma), 0.0);
    }

    /// A prograde short-period comet and 1P/Halley (retrograde) shortly after perihelion: the
    /// dust tail trails the motion (mean grain offset · v < 0) and points away from the Sun
    /// (· r̂_h > 0); the sprites cover every grain, and the light weights sum to 1.
    #[test]
    fn dust_tail_lags_motion() {
        let comets = [
            // 1P/Halley's SBDB elements (q, e, i, Ω, ω, T_p), 9 days after perihelion.
            (
                conic_from_perihelion(
                    0.574_863_831_374_341_3,
                    0.967_935_995_695_321_1,
                    162.190_530_043_912_9,
                    59.098_947_206_124_37,
                    112.241_431_463_776_4,
                    2_446_469.973_616_146_5,
                ),
                9.0,
            ),
            // A 67P-like prograde orbit (q 1.24 AU, e 0.64, i 7°), 20 days after perihelion.
            (
                conic_from_perihelion(1.24, 0.64, 7.04, 50.1, 12.8, 2_459_885.5),
                20.0,
            ),
        ];
        for (conic, days) in comets {
            let t = conic.tp.add_seconds(days * DAY);
            let (r, v) = heliocentric_state_icrs(&conic, t);
            let grains = dust_rows(&conic, 10.0, t, 0..DUST_PARAMS.n_beta);
            assert_eq!(grains.len(), GRAINS_PER_TAIL);
            let mean = grains
                .iter()
                .map(|g| g.pos_rel_nucleus_m)
                .sum::<DVec3>()
                / grains.len() as f64;
            assert!(mean.dot(v) < 0.0, "mean offset {mean} m leads the motion {v}");
            assert!(mean.dot(r) > 0.0, "mean offset {mean} m points sunward");
            let sprites = tail_grains(&grains, -r);
            assert_eq!(sprites.len(), GRAINS_PER_TAIL);
            assert!(sprites.iter().all(|g| g.sigma_m > 0.0 && g.weight >= 0.0));
            let lit: f64 = sprites
                .iter()
                .map(|g| g.weight * r.length_squared() / (r + g.offset_m).length_squared())
                .sum();
            assert!((lit - 1.0).abs() < 1e-9, "Σ w (r_h/r)² = {lit}");
        }
    }

    /// Review I13: with 20 active comets on screen only the 8 brightest by `total_v` get tails;
    /// a comet without a magnitude law or beyond 5 AU never does; refreshed comets wait
    /// `TAIL_REFRESH_WALL_S` and then come back oldest first.
    #[test]
    fn tails_capped_at_eight() {
        let mut views: Vec<CometView> = (0..20_u32)
            .map(|id| {
                let mut v = static_view(
                    id,
                    DVec3::new(0.0, 0.0, -0.5 * AU),
                    DVec3::new(-AU, 0.0, 0.0),
                    Some(8.0),
                    Some(10.0),
                );
                v.total_v = Some(5.0 + 0.37 * f64::from((id * 7) % 20));
                v
            })
            .collect();
        let picked = tail_schedule(&views, &HashMap::new(), 0.0);
        assert_eq!(picked.len(), MAX_TAILED_COMETS);
        let mut mags: Vec<f64> = views.iter().filter_map(|v| v.total_v).collect();
        mags.sort_by(f64::total_cmp);
        let mut got: Vec<f64> = picked.iter().filter_map(|&i| views[i].total_v).collect();
        got.sort_by(f64::total_cmp);
        assert_eq!(got, mags[..MAX_TAILED_COMETS].to_vec());
        assert_eq!(tailed_comets(&views).len(), MAX_TAILED_COMETS);

        // The brightest comet loses its law; another moves beyond 5 AU: neither is picked.
        let brightest = picked[0];
        views[brightest].m1 = None;
        views[brightest].total_v = None;
        let second = tailed_comets(&views)[0];
        views[second].r_au = 5.1;
        let picked = tail_schedule(&views, &HashMap::new(), 0.0);
        assert_eq!(picked.len(), MAX_TAILED_COMETS);
        assert!(!picked.contains(&brightest) && !picked.contains(&second));

        // All refreshed at wall 0: none due at 0.1 s; all due again at 0.3 s, oldest first.
        let mut last: HashMap<ObjectId, f64> = picked
            .iter()
            .map(|&i| (views[i].id, 0.0))
            .collect();
        assert!(tail_schedule(&views, &last, 0.1).is_empty());
        last.insert(views[picked[5]].id, -1.0);
        let again = tail_schedule(&views, &last, 0.3);
        assert_eq!(again.len(), MAX_TAILED_COMETS);
        assert_eq!(again[0], picked[5]);
    }

    /// Review I13: under warp 10⁹ for 2 s of wall time at 60 fps (every frame moves the epoch
    /// 193 days), each tailed comet is refreshed at most 9 times (every 0.25 s of wall time) and
    /// at least 6 (the round robin reaches all eight), comets beyond the eight never, and no frame
    /// does more than the 1 ms share of Kepler solves at the plan's 0.5 µs each.
    #[test]
    fn tail_refresh_rate_limited_under_warp() {
        let conic = conic_from_perihelion(1.24, 0.64, 7.04, 50.1, 12.8, 2_459_885.5);
        let t0 = conic.tp;
        let views: Vec<CometView> = (0..10_u32)
            .map(|id| {
                let mut v = comet_on(id, conic, t0, DVec3::new(0.0, AU, 0.0), 8.0, 10.0);
                v.total_v = Some(6.0 + f64::from(id));
                v
            })
            .collect();
        let tailed: Vec<ObjectId> = tailed_comets(&views).iter().map(|&i| views[i].id).collect();
        assert_eq!(tailed.len(), 8);
        let mut manager = TailManager::default();
        let mut started: HashMap<ObjectId, usize> = HashMap::new();
        let mut completed: HashMap<ObjectId, usize> = HashMap::new();
        let (mut worst, mut total) = (0, 0);
        let clock = std::time::Instant::now();
        for frame in 0..120_u32 {
            let wall = f64::from(frame) / 60.0;
            let epoch = t0.add_seconds(1.0e9 * wall);
            let work = manager.update(&views, epoch, wall, NOMINAL_SOLVES_PER_FRAME);
            worst = worst.max(work.solves);
            total += work.solves;
            for id in work.started {
                *started.entry(id).or_default() += 1;
            }
            for id in work.completed {
                *completed.entry(id).or_default() += 1;
            }
        }
        println!(
            "120 frames: worst {worst} solves/frame, {total} in all, {:.2} µs per solve (this \
             build); started {started:?}, completed {completed:?}",
            clock.elapsed().as_secs_f64() * 1e6 / total as f64
        );
        assert!(
            worst as f64 * KEPLER_SOLVE_US <= 1_000.0 * TAIL_CPU_SHARE_MS,
            "{worst} solves in one frame"
        );
        for id in &tailed {
            let n = started.get(id).copied().unwrap_or(0);
            assert!(n <= 9, "comet {id} refreshed {n} times in 2 s");
            let done = completed.get(id).copied().unwrap_or(0);
            assert!(done >= 6, "comet {id} completed only {done} refreshes");
        }
        assert!(started.keys().all(|id| tailed.contains(id)));
        assert_eq!(manager.tails.len(), 8);
        for (id, tail) in &manager.tails {
            assert!(tailed.contains(id));
            assert_eq!(tail.grains.len(), GRAINS_PER_TAIL);
            assert!(tail.reach_m > 0.0 && tail.reach_m.is_finite());
        }
    }

    /// At real time a tail is not recomputed until an hour of sim time has passed, however much
    /// wall time goes by; a date jump refreshes it; a comet gone from the screen is dropped after
    /// `TAIL_EVICT_WALL_S`.
    #[test]
    fn tail_refresh_waits_for_an_hour_of_sim_time() {
        let conic = conic_from_perihelion(1.24, 0.64, 7.04, 50.1, 12.8, 2_459_885.5);
        let t0 = conic.tp;
        let views = vec![comet_on(1, conic, t0, DVec3::new(0.0, AU, 0.0), 8.0, 10.0)];
        let mut manager = TailManager::default();
        let work = manager.update(&views, t0, 0.0, usize::MAX);
        assert_eq!((work.started.clone(), work.completed.clone()), (vec![1], vec![1]));
        // 50 minutes of sim time over 3 000 s of wall time: no refresh.
        for k in 1..=10 {
            let wall = 300.0 * f64::from(k);
            let work = manager.update(&views, t0.add_seconds(wall), wall, usize::MAX);
            if wall < TAIL_SIM_REFRESH_S {
                assert!(work.started.is_empty(), "refreshed at {wall} s");
            }
        }
        let work = manager.update(&views, t0.add_seconds(-5.0 * DAY), 3_100.0, usize::MAX);
        assert_eq!(work.completed, vec![1]);
        // Off screen for longer than the eviction time: dropped.
        manager.update(&[], t0, 3_100.0 + TAIL_EVICT_WALL_S + 1.0, usize::MAX);
        assert!(manager.tails.is_empty() && manager.last_refresh.is_empty());
    }
}
