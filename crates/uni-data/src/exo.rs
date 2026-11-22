//! Exoplanet systems from the NASA Exoplanet Archive `pscomppars` table (`exoplanets.ron`)
//! (M1 Task A7). `uni-bake exo` writes the file: one [`ExoSystem`] per archive host, sorted by
//! host name, each with its planets sorted by name. Column definitions quoted below are the
//! archive's (https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html, read
//! 2026-11-22).

use std::path::Path;

use crate::DataError;

/// One planet.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExoPlanet {
    pub name: String,
    /// `pl_orbsmax` [au].
    pub a_au: Option<f64>,
    /// `pl_orbeccen`.
    pub e: Option<f64>,
    /// `pl_orbincl` [deg].
    pub i_deg: Option<f64>,
    /// Archive `pl_orblper` [deg], "the angular separation between the orbit's ascending node
    /// and periastron". The archive does not standardise which body it refers to: "some
    /// publications refer the planet's orbit, others to the host star's reflex orbit, which
    /// differs by 180 deg. The values in the Exoplanet Archive are not corrected to a
    /// standardized system, but are as-reported for each publication." So it may be ω of the
    /// planet or of the star; nothing in the table says which.
    pub argp_deg: Option<f64>,
    /// `pl_orbper` [days].
    pub period_d: Option<f64>,
    /// `pl_tranmid` [days], the time of conjunction (transit midpoint), as published: mostly
    /// BJD (TDB), but some rows are HJD or geocentric JD, in UTC or TDB
    /// (`pl_tranmid_systemref`). The standards differ by at most ≈ 0.006 d (8.3 min of light
    /// time across the Earth's orbit plus 69 s of TT − UTC).
    pub t_transit_jd: Option<f64>,
    /// `pl_bmassj` [Jupiter masses]: the mass, or M sin i when only that is known.
    pub mass_mjup: Option<f64>,
    /// `pl_radj` [Jupiter radii].
    pub radius_rjup: Option<f64>,
    /// `pl_eqt` [K].
    pub teq_k: Option<f64>,
    /// Discovery method.
    pub method: String,
    /// Discovery year.
    pub year: u16,
}

/// One host star and its planets.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExoSystem {
    /// `hostname`.
    pub host: String,
    /// Index of the matched star in `athyg_m10.bin`; `None` if the host is not in AT-HYG.
    pub star_index: Option<u32>,
    /// Archive `ra` [deg].
    pub ra_deg: f64,
    /// Archive `dec` [deg].
    pub dec_deg: f64,
    /// Archive `sy_dist` [pc]; for the few hosts without one, the matched star's distance.
    pub dist_pc: f64,
    /// Archive `sy_vmag`.
    pub v_mag: Option<f64>,
    /// `st_teff` [K].
    pub teff_k: Option<f64>,
    /// `st_rad` [solar radii].
    pub radius_rsun: Option<f64>,
    pub planets: Vec<ExoPlanet>,
}

/// Reads `exoplanets.ron`. Missing → [`DataError::NotFound`]; unparsable, or a system with a
/// non-finite value, a distance ≤ 0 or no planets → [`DataError::BadFormat`].
pub fn load_exoplanets(path: &Path) -> Result<Vec<ExoSystem>, DataError> {
    let text = std::fs::read_to_string(path).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    })?;
    let systems: Vec<ExoSystem> = ron::from_str(&text)
        .map_err(|err| DataError::BadFormat(format!("{}: {err}", path.display())))?;
    if let Some(bad) = systems.iter().find(|sys| !is_valid(sys)) {
        return Err(DataError::BadFormat(format!(
            "{}: invalid system {:?}",
            path.display(),
            bad.host
        )));
    }
    Ok(systems)
}

fn is_valid(sys: &ExoSystem) -> bool {
    let finite = |x: Option<f64>| x.is_none_or(f64::is_finite);
    let planet_ok = |p: &ExoPlanet| {
        [
            p.a_au,
            p.e,
            p.i_deg,
            p.argp_deg,
            p.period_d,
            p.t_transit_jd,
            p.mass_mjup,
            p.radius_rjup,
            p.teq_k,
        ]
        .into_iter()
        .all(finite)
    };
    sys.ra_deg.is_finite()
        && sys.dec_deg.is_finite()
        && sys.dist_pc.is_finite()
        && sys.dist_pc > 0.0
        && [sys.v_mag, sys.teff_k, sys.radius_rsun]
            .into_iter()
            .all(finite)
        && !sys.planets.is_empty()
        && sys.planets.iter().all(planet_ok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-exo-{}-{}", std::process::id(), name))
    }

    fn planet(name: &str) -> ExoPlanet {
        ExoPlanet {
            name: name.to_owned(),
            a_au: Some(0.0527),
            e: Some(0.01),
            i_deg: None,
            argp_deg: Some(58.0),
            period_d: Some(4.2308),
            t_transit_jd: None,
            mass_mjup: Some(0.46),
            radius_rjup: None,
            teq_k: Some(1260.0),
            method: "Radial Velocity".to_owned(),
            year: 1995,
        }
    }

    fn system() -> ExoSystem {
        ExoSystem {
            host: "51 Peg".to_owned(),
            star_index: Some(321_000),
            ra_deg: 344.3665854,
            dec_deg: 20.7688425,
            dist_pc: 15.4614,
            v_mag: Some(5.46),
            teff_k: Some(5758.0),
            radius_rsun: Some(1.15),
            planets: vec![planet("51 Peg b")],
        }
    }

    fn write_ron(path: &Path, systems: &[ExoSystem]) {
        let text =
            ron::ser::to_string_pretty(systems, ron::ser::PrettyConfig::new()).expect("serialise");
        std::fs::write(path, text).expect("write");
    }

    #[test]
    fn exoplanets_roundtrip() {
        let mut archive_only = system();
        archive_only.host = "TRAPPIST-1".to_owned();
        archive_only.star_index = None;
        archive_only.v_mag = Some(18.8);
        let path = temp_path("roundtrip.ron");
        write_ron(&path, &[system(), archive_only]);
        let loaded = load_exoplanets(&path);
        let _ = std::fs::remove_file(&path);
        let systems = loaded.expect("load");
        assert_eq!(systems.len(), 2);
        assert_eq!(systems[0].host, "51 Peg");
        assert_eq!(systems[0].star_index, Some(321_000));
        assert_eq!(systems[0].planets[0].name, "51 Peg b");
        assert_eq!(systems[0].planets[0].argp_deg, Some(58.0));
        assert_eq!(systems[1].star_index, None);
        assert_eq!(
            systems[1].v_mag,
            Some(18.8),
            "unmatched hosts keep their archive V"
        );
    }

    #[test]
    fn committed_exoplanets_match_host_flags() {
        // data/baked/exoplanets.ron (pscomppars of 2026-11-22) and athyg_m10.bin (M1 Task A7).
        let dir = crate::data_dir_from(None).join("baked");
        let systems =
            load_exoplanets(&dir.join("exoplanets.ron")).expect("committed exoplanets.ron");
        let stars = crate::load_athyg(&dir.join("athyg_m10.bin")).expect("committed athyg_m10.bin");
        assert_eq!(systems.len(), 4839);
        assert_eq!(systems.iter().map(|s| s.planets.len()).sum::<usize>(), 6447);
        let mut hosts = vec![false; stars.len()];
        for sys in &systems {
            if let Some(index) = sys.star_index {
                hosts[index as usize] = true;
            }
        }
        for (index, star) in stars.iter().enumerate() {
            assert_eq!(star.is_exo_host(), hosts[index], "star {index}");
        }
        let peg = systems.iter().find(|s| s.host == "51 Peg").expect("51 Peg");
        let star = stars[peg.star_index.expect("51 Peg is in AT-HYG") as usize];
        assert!((glam::Vec3::from(star.pos_pc).length() as f64 - peg.dist_pc).abs() < 0.1);
        let trappist = systems
            .iter()
            .find(|s| s.host == "TRAPPIST-1")
            .expect("TRAPPIST-1");
        assert_eq!(trappist.star_index, None, "V ≈ 17: not in AT-HYG m10");
        assert!(trappist.v_mag.is_some());
    }

    #[test]
    fn load_missing_exoplanets_is_not_found() {
        assert!(matches!(
            load_exoplanets(Path::new("/nonexistent/exoplanets.ron")),
            Err(DataError::NotFound(_))
        ));
    }

    #[test]
    fn load_garbage_exoplanets_is_bad_format() {
        let path = temp_path("garbage.ron");
        std::fs::write(&path, "[(host: \"51 Peg\",").expect("write");
        let loaded = load_exoplanets(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }

    #[test]
    fn load_invalid_system_is_bad_format() {
        // Review Focus 3: never NaN, never a host without a usable distance.
        let mut nan_dec = system();
        nan_dec.dec_deg = f64::NAN;
        let mut zero_dist = system();
        zero_dist.dist_pc = 0.0;
        let mut nan_planet = system();
        nan_planet.planets[0].e = Some(f64::NAN);
        let mut no_planets = system();
        no_planets.planets.clear();
        let path = temp_path("invalid.ron");
        for (what, sys) in [
            ("dec", nan_dec),
            ("dist", zero_dist),
            ("planet", nan_planet),
            ("empty", no_planets),
        ] {
            // RON writes NaN as `NaN`, which it reads back, so the loader must check.
            write_ron(&path, &[sys]);
            let loaded = load_exoplanets(&path);
            assert!(
                matches!(loaded, Err(DataError::BadFormat(_))),
                "{what}: {loaded:?}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}
