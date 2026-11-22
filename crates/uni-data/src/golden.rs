//! JPL Horizons spot values used as test goldens (M1 Task A9, review I9):
//! `data/golden/horizons_spots.ron` (geometric state vectors) and `data/golden/sky_spots.ron`
//! (observer tables). `uni-bake horizons` fetches and writes both. Horizons served DE441 for the
//! Sun, planets and Moon and the JPL satellite solutions for the other moons; each file's header
//! names the source of every answer.
//!
//! Format: RON, a `//` provenance comment block (query, fetch time, ephemeris source per target)
//! and then one sequence of compact structs, one spot per line. Field names are the struct's.

use std::path::Path;

use crate::DataError;

/// One geometric state vector (Horizons `VEC_CORR=NONE`), ICRF axes, `REF_PLANE=FRAME`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EphemSpot {
    /// Horizons target (NAIF id; small bodies use their SBDB SPK-ID, e.g. 2000001 for Ceres).
    pub target: i32,
    /// Horizons centre (NAIF id), the body or barycentre the vector is measured from.
    pub center: i32,
    /// Julian Date, TDB.
    pub jd_tdb: f64,
    /// Position of the target relative to the centre [km].
    pub pos_km: [f64; 3],
    /// Velocity of the target relative to the centre [km/s].
    pub vel_km_s: [f64; 3],
}

/// One observer-table row (astrometric; Horizons quantities 1, 9, 19, 20, 24).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SkySpot {
    /// Horizons target: a NAIF id ("599") or a small-body designation ("C/2020 F3").
    pub target: String,
    /// NAIF id of the body at whose centre the observer sits (Horizons `CENTER='500@<id>'`).
    pub observer: i32,
    /// Julian Date, UTC.
    pub jd_utc: f64,
    /// Astrometric right ascension, ICRF [deg] (quantity 1).
    pub ra_deg: f64,
    /// Astrometric declination, ICRF [deg] (quantity 1).
    pub dec_deg: f64,
    /// Observer–target distance Δ [au] (quantity 20).
    pub delta_au: f64,
    /// Apparent visual magnitude APmag (quantity 9), when asked for.
    pub apmag: Option<f64>,
    /// Sun–target distance r [au] (quantity 19), when asked for.
    pub r_au: Option<f64>,
    /// Phase angle, Sun–target–observer [deg] (quantity 24), when asked for.
    pub phase_deg: Option<f64>,
}

/// Reads `horizons_spots.ron`.
///
/// A missing file is [`DataError::NotFound`]; a file that is not a RON sequence of
/// [`EphemSpot`], or that holds a non-finite number, is [`DataError::BadFormat`].
pub fn load_ephem_spots(path: &Path) -> Result<Vec<EphemSpot>, DataError> {
    let spots: Vec<EphemSpot> = read_ron(path)?;
    for (index, spot) in spots.iter().enumerate() {
        let finite = [spot.jd_tdb]
            .into_iter()
            .chain(spot.pos_km)
            .chain(spot.vel_km_s)
            .all(f64::is_finite);
        if !finite {
            return Err(bad_spot(path, index, "non-finite number"));
        }
    }
    Ok(spots)
}

/// Reads `sky_spots.ron`.
///
/// A missing file is [`DataError::NotFound`]; a file that is not a RON sequence of [`SkySpot`],
/// or that holds a non-finite number, an RA outside [0°, 360°), a Dec outside [−90°, 90°] or a
/// distance Δ ≤ 0, is [`DataError::BadFormat`].
pub fn load_sky_spots(path: &Path) -> Result<Vec<SkySpot>, DataError> {
    let spots: Vec<SkySpot> = read_ron(path)?;
    for (index, spot) in spots.iter().enumerate() {
        let finite = [spot.jd_utc, spot.ra_deg, spot.dec_deg, spot.delta_au]
            .into_iter()
            .chain(spot.apmag)
            .chain(spot.r_au)
            .chain(spot.phase_deg)
            .all(f64::is_finite);
        let why = if !finite {
            "non-finite number"
        } else if !(0.0..360.0).contains(&spot.ra_deg) {
            "RA outside [0, 360) deg"
        } else if !(-90.0..=90.0).contains(&spot.dec_deg) {
            "Dec outside [-90, 90] deg"
        } else if spot.delta_au <= 0.0 {
            "distance delta_au <= 0"
        } else {
            continue;
        };
        return Err(bad_spot(path, index, why));
    }
    Ok(spots)
}

/// Reads and parses a whole RON file.
fn read_ron<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, DataError> {
    let text = std::fs::read_to_string(path).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    })?;
    ron::from_str(&text).map_err(|err| DataError::BadFormat(format!("{}: {err}", path.display())))
}

fn bad_spot(path: &Path, index: usize, why: &str) -> DataError {
    DataError::BadFormat(format!("{}: spot {index}: {why}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // The epochs of A9 Step 1, as Julian Dates: TDB for the state vectors, UTC for the observer
    // tables. Horizons reads every date before 1582-10-15 as Julian, so the plan names JDs only.
    const JD_1000: f64 = 2_086_307.5; // AD 1000-01-01 0h (Julian calendar)
    const JD_1800: f64 = 2_378_496.5; // 1800-01-01 0h
    const JD_1900: f64 = 2_415_020.5; // 1900-01-01 0h
    const JD_1950: f64 = 2_433_282.5; // 1950-01-01 0h
    const JD_2000: f64 = 2_451_545.0; // J2000.0 = 2000-01-01 12h
    /// 2026-11-20 00:00 UTC as TDB: UTC + 37 s (leap seconds) + 32.184 s (TT − TAI) = +69.184 s
    /// = +0.000 800 7 d, which the plan rounds to 0.0008 d.
    const JD_2026_TDB: f64 = 2_461_364.500_8;
    const JD_2026_UTC: f64 = 2_461_364.5; // 2026-11-20 00:00 UTC
    const JD_2050: f64 = 2_469_807.5; // 2050-01-01 0h
    const JD_2090: f64 = 2_484_417.5; // 2090-01-01 0h
    const JD_2500: f64 = 2_634_166.5; // AD 2500-01-01 0h
    const JD_HALLEY: f64 = 2_446_502.5; // 1986-03-13 0h
    const JD_NEOWISE_UTC: f64 = 2_459_045.625; // 2020-07-15 03:00 UTC

    /// SBDB SPK-IDs (NAIF small-body ids) of 1P/Halley and (1) Ceres.
    const HALLEY: i32 = 1_000_036;
    const CERES: i32 = 2_000_001;

    /// Every (target, center, JD TDB) of A9 Step 1, restated from the brief.
    fn planned_vectors() -> Vec<(i32, i32, f64)> {
        let mut plan = Vec::new();
        let standish = [
            JD_1800,
            JD_1900,
            JD_1950,
            JD_2000,
            JD_2026_TDB,
            JD_2050,
            JD_1000,
            JD_2500,
        ];
        for target in 1..=8 {
            plan.extend(standish.iter().map(|&jd| (target, 10, jd)));
        }
        let pluto = [JD_1900, JD_1950, JD_2000, JD_2026_TDB, JD_2050, JD_2090];
        plan.extend(pluto.iter().map(|&jd| (9, 10, jd)));
        plan.extend(
            [JD_1950, JD_2000, JD_2026_TDB]
                .iter()
                .map(|&jd| (10, 0, jd)),
        );
        let moons = [
            (301, 399),
            (401, 499),
            (402, 499),
            (501, 599),
            (502, 599),
            (503, 599),
            (504, 599),
            (601, 699),
            (602, 699),
            (603, 699),
            (604, 699),
            (605, 699),
            (606, 699),
            (607, 699),
            (608, 699),
            (703, 799),
            (704, 799),
            (801, 899),
            (901, 999),
        ];
        for (moon, planet) in moons {
            plan.extend(
                [JD_1950, JD_2000, JD_2026_TDB, JD_2050]
                    .iter()
                    .map(|&jd| (moon, planet, jd)),
            );
        }
        plan.push((HALLEY, 10, JD_HALLEY));
        plan.push((CERES, 10, JD_2026_TDB));
        plan
    }

    /// The planets' observer spots (quantities 1, 9, 19, 20, 24). Jupiter's '1,20' spot of the
    /// brief is the same (target, observer, time), so this row covers it.
    const PLANET_SKY: [&str; 7] = ["199", "299", "499", "599", "699", "799", "899"];

    /// Every (target, observer, JD UTC) of A9 Step 1's observer tables.
    fn planned_sky() -> Vec<(&'static str, i32, f64)> {
        let mut plan: Vec<_> = PLANET_SKY.iter().map(|&t| (t, 399, JD_2026_UTC)).collect();
        plan.push(("C/2020 F3", 399, JD_NEOWISE_UTC));
        plan
    }

    fn golden_path(name: &str) -> PathBuf {
        crate::data_dir().join("golden").join(name)
    }

    /// A per-process scratch path (no tempfile crate).
    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-golden-{}-{name}", std::process::id()))
    }

    fn ephem_bits(s: &EphemSpot) -> (i32, i32, u64, [u64; 3], [u64; 3]) {
        (
            s.target,
            s.center,
            s.jd_tdb.to_bits(),
            s.pos_km.map(f64::to_bits),
            s.vel_km_s.map(f64::to_bits),
        )
    }

    type SkyBits = (String, i32, [u64; 4], [Option<u64>; 3]);

    fn sky_bits(s: &SkySpot) -> SkyBits {
        (
            s.target.clone(),
            s.observer,
            [s.jd_utc, s.ra_deg, s.dec_deg, s.delta_au].map(f64::to_bits),
            [s.apmag, s.r_au, s.phase_deg].map(|v| v.map(f64::to_bits)),
        )
    }

    #[test]
    fn golden_ron_roundtrip() {
        let ephem = vec![
            EphemSpot {
                target: 5,
                center: 10,
                jd_tdb: JD_2026_TDB,
                pos_km: [-5.640_475_942_392_98e8, 0.1 + 0.2, -0.0],
                vel_km_s: [-9.381_734_722_133_515, f64::MIN_POSITIVE, 1.5e-300],
            },
            EphemSpot {
                target: HALLEY,
                center: 10,
                jd_tdb: JD_HALLEY,
                pos_km: [1.0 / 3.0, 6.022_140_76e23, -2.0e-7],
                vel_km_s: [42.0, -1.0e10, 7.0],
            },
        ];
        let sky = vec![
            SkySpot {
                target: "699".into(),
                observer: 399,
                jd_utc: JD_2026_UTC,
                ra_deg: 8.331_486_146,
                dec_deg: -0.734_877_993,
                delta_au: 8.746_533_393_790_41,
                apmag: Some(0.555),
                r_au: Some(9.419_475_012_161),
                phase_deg: Some(4.5638),
            },
            SkySpot {
                target: "C/2020 F3".into(),
                observer: 399,
                jd_utc: JD_NEOWISE_UTC,
                ra_deg: 359.999_999_999,
                dec_deg: 90.0,
                delta_au: 0.784_344_098_586_66,
                apmag: None,
                r_au: None,
                phase_deg: None,
            },
        ];
        let config = ron::ser::PrettyConfig::new().compact_structs(true);
        let ephem_path = scratch("roundtrip-ephem.ron");
        let sky_path = scratch("roundtrip-sky.ron");
        let ephem_text = ron::ser::to_string_pretty(&ephem, config.clone()).expect("serialise");
        let sky_text = ron::ser::to_string_pretty(&sky, config).expect("serialise");
        // The baked files open with a provenance comment block.
        std::fs::write(&ephem_path, format!("// provenance\n{ephem_text}\n")).expect("write");
        std::fs::write(&sky_path, format!("// provenance\n{sky_text}\n")).expect("write");
        let ephem_back = load_ephem_spots(&ephem_path).expect("load ephem");
        let sky_back = load_sky_spots(&sky_path).expect("load sky");
        let _ = std::fs::remove_file(&ephem_path);
        let _ = std::fs::remove_file(&sky_path);
        let bits = |v: &[EphemSpot]| v.iter().map(ephem_bits).collect::<Vec<_>>();
        assert_eq!(bits(&ephem_back), bits(&ephem));
        let bits = |v: &[SkySpot]| v.iter().map(sky_bits).collect::<Vec<_>>();
        assert_eq!(bits(&sky_back), bits(&sky));
    }

    #[test]
    fn golden_missing_file_is_not_found() {
        let path = scratch("no-such-golden.ron");
        assert!(matches!(load_ephem_spots(&path), Err(DataError::NotFound(p)) if p == path));
        assert!(matches!(load_sky_spots(&path), Err(DataError::NotFound(p)) if p == path));
    }

    #[test]
    fn golden_rejects_malformed_or_impossible_values() {
        let path = scratch("bad-golden.ron");
        let spot = "(target: 5, center: 10, jd_tdb: 2451545.0, pos_km: (1.0, 2.0, 3.0), vel_km_s: (1.0, 2.0, 3.0))";
        let sky = |ra: &str, dec: &str, delta: &str, apmag: &str| {
            format!(
                "[(target: \"599\", observer: 399, jd_utc: 2461364.5, ra_deg: {ra}, dec_deg: {dec}, \
                 delta_au: {delta}, apmag: {apmag}, r_au: None, phase_deg: None)]"
            )
        };
        let ephem_cases = [
            "not ron at all".to_string(),
            format!("[{spot}"),
            format!("[{}]", spot.replace("2.0, 3.0)", "NaN, 3.0)")),
            format!("[{}]", spot.replace("2451545.0", "inf")),
        ];
        for text in ephem_cases {
            std::fs::write(&path, &text).expect("write");
            let result = load_ephem_spots(&path);
            assert!(
                matches!(result, Err(DataError::BadFormat(_))),
                "{text}: {result:?}"
            );
        }
        let sky_cases = [
            sky("360.0", "0.0", "1.0", "None"),
            sky("-0.5", "0.0", "1.0", "None"),
            sky("10.0", "90.5", "1.0", "None"),
            sky("10.0", "0.0", "0.0", "None"),
            sky("10.0", "0.0", "1.0", "Some(NaN)"),
        ];
        for text in sky_cases {
            std::fs::write(&path, &text).expect("write");
            let result = load_sky_spots(&path);
            assert!(
                matches!(result, Err(DataError::BadFormat(_))),
                "{text}: {result:?}"
            );
        }
        std::fs::write(&path, format!("[{spot}]")).expect("write");
        assert_eq!(load_ephem_spots(&path).expect("valid spot").len(), 1);
        std::fs::write(&path, sky("10.0", "-90.0", "1.0", "Some(-2.5)")).expect("write");
        assert_eq!(load_sky_spots(&path).expect("valid sky spot").len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn golden_spots_cover_plan() {
        let path = golden_path("horizons_spots.ron");
        let spots = load_ephem_spots(&path)
            .unwrap_or_else(|e| panic!("golden file missing or unreadable: {e} (A9 Step 1)"));
        let plan = planned_vectors();
        assert_eq!(plan.len(), 151);
        for &(target, center, jd) in &plan {
            let n = spots
                .iter()
                .filter(|s| {
                    s.target == target && s.center == center && (s.jd_tdb - jd).abs() < 1e-6
                })
                .count();
            assert_eq!(
                n, 1,
                "{target}@{center} at JD {jd}: {n} spots, want exactly 1"
            );
        }
        assert_eq!(spots.len(), plan.len(), "spots outside the plan");

        let path = golden_path("sky_spots.ron");
        let sky = load_sky_spots(&path)
            .unwrap_or_else(|e| panic!("golden file missing or unreadable: {e} (A9 Step 1)"));
        let plan = planned_sky();
        for &(target, observer, jd) in &plan {
            let rows: Vec<_> = sky
                .iter()
                .filter(|s| {
                    s.target == target && s.observer == observer && (s.jd_utc - jd).abs() < 1e-6
                })
                .collect();
            assert_eq!(
                rows.len(),
                1,
                "{target}@{observer} at JD {jd} UTC: want exactly 1"
            );
            if PLANET_SKY.contains(&target) {
                let row = rows[0];
                for (name, value) in [
                    ("apmag", row.apmag),
                    ("r_au", row.r_au),
                    ("phase_deg", row.phase_deg),
                ] {
                    assert!(
                        value.is_some_and(f64::is_finite),
                        "{target}: {name} = {value:?}"
                    );
                }
            }
        }
        assert_eq!(sky.len(), plan.len(), "sky spots outside the plan");
    }

    /// Cheap physical checks of the fetched numbers, so a column mix-up or a unit slip in the bake
    /// cannot pass: planet barycentres lie between their perihelion and aphelion and move at the
    /// vis-viva speed; each planet's observer row closes the Sun–Earth–planet triangle.
    #[test]
    fn golden_spots_are_physically_consistent() {
        // a [au] and e at J2000 (Standish & Williams 1992, "Approximate Positions of the Planets",
        // JPL SSD, Table 1, 1800–2050). Orbits drift little over AD 1000–2500 and the Sun sits up
        // to 0.01 au from the SSB, hence the 2 % margin on r and 3 % on the vis-viva speed.
        const ORBITS: [(i32, f64, f64); 8] = [
            (1, 0.387_099_27, 0.205_635_93),
            (2, 0.723_335_66, 0.006_776_72),
            (3, 1.000_002_61, 0.016_711_23),
            (4, 1.523_710_34, 0.093_394_10),
            (5, 5.202_887_00, 0.048_386_24),
            (6, 9.536_675_94, 0.053_861_79),
            (7, 19.189_164_64, 0.047_257_44),
            (8, 30.069_922_76, 0.008_590_48),
        ];
        const AU_KM: f64 = 1.495_978_707e8; // IAU 2012 Resolution B2
        let gm_sun_km3_s2 = uni_physics::bodies::by_naif(10).expect("Sun").gm_m3s2 * 1e-9;
        let spots = load_ephem_spots(&golden_path("horizons_spots.ron")).expect("golden file");
        let norm = |v: [f64; 3]| v.iter().map(|c| c * c).sum::<f64>().sqrt();
        for (target, a, e) in ORBITS {
            for s in spots
                .iter()
                .filter(|s| s.target == target && s.center == 10)
            {
                let r_km = norm(s.pos_km);
                let r = r_km / AU_KM;
                let (q, big_q) = (a * (1.0 - e) - 0.02 * a, a * (1.0 + e) + 0.02 * a);
                assert!(
                    (q..=big_q).contains(&r),
                    "{target} at JD {}: r = {r} au",
                    s.jd_tdb
                );
                let vis_viva = (gm_sun_km3_s2 * (2.0 / r_km - 1.0 / (a * AU_KM))).sqrt();
                let v = norm(s.vel_km_s);
                assert!(
                    (v / vis_viva - 1.0).abs() < 0.03,
                    "{target} at JD {}: v = {v} km/s, vis-viva {vis_viva}",
                    s.jd_tdb
                );
            }
        }
        // Earth–Sun distance from r, Δ and the phase angle (law of cosines) lies within Earth's
        // perihelion–aphelion range a(1 ∓ e) of the EMB row above, ±0.001 au.
        let sky = load_sky_spots(&golden_path("sky_spots.ron")).expect("golden file");
        for row in sky
            .iter()
            .filter(|s| PLANET_SKY.contains(&s.target.as_str()))
        {
            let (r, delta, phase) = (
                row.r_au.expect("r"),
                row.delta_au,
                row.phase_deg.expect("phase").to_radians(),
            );
            let sun_earth = (r * r + delta * delta - 2.0 * r * delta * phase.cos()).sqrt();
            assert!(
                (0.982..=1.018).contains(&sun_earth),
                "{}: Sun–Earth {sun_earth} au",
                row.target
            );
        }
    }
}
