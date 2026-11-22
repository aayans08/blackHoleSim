//! Saturn's ring radial profile `saturn_rings.ron` (M1 Task A8), written by `uni-bake rings` from
//! Solar System Scope's `8k_saturn_ring_alpha.png` (CC BY 4.0).
//!
//! Sample `i` of `n` covers radii [inner + i·Δ, inner + (i + 1)·Δ), Δ = (outer − inner)/n, so its
//! centre is at inner + (i + ½)·Δ. Its colour is the opacity-weighted mean of the PNG's colour,
//! decoded from sRGB to linear, and its opacity the mean of the PNG's alpha.

use std::path::Path;

use crate::DataError;

/// Ring colour and opacity sampled uniformly in radius from `inner_m` to `outer_m`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RingProfile {
    pub inner_m: f64,
    pub outer_m: f64,
    /// Linear RGB reflectance and opacity per sample.
    pub samples: Vec<[f32; 4]>,
}

/// Reads `saturn_rings.ron`. A missing file is [`DataError::NotFound`]; a file that does not
/// parse, has fewer than two samples, `inner_m ≥ outer_m`, a non-finite value, a negative colour
/// or an opacity outside [0, 1] is [`DataError::BadFormat`].
pub fn load_ring_profile(path: &Path) -> Result<RingProfile, DataError> {
    let text = std::fs::read_to_string(path).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    })?;
    let bad = |why: String| DataError::BadFormat(format!("{}: {why}", path.display()));
    let profile: RingProfile = ron::from_str(&text).map_err(|err| bad(err.to_string()))?;
    check(&profile).map_err(bad)?;
    Ok(profile)
}

fn check(profile: &RingProfile) -> Result<(), String> {
    let (inner, outer) = (profile.inner_m, profile.outer_m);
    if !(inner.is_finite() && outer.is_finite() && 0.0 <= inner && inner < outer) {
        return Err(format!("radii {inner} m – {outer} m"));
    }
    if profile.samples.len() < 2 {
        return Err(format!(
            "{} samples; at least 2 needed",
            profile.samples.len()
        ));
    }
    for (i, [r, g, b, opacity]) in profile.samples.iter().enumerate() {
        let colour_ok = [r, g, b].iter().all(|c| c.is_finite() && **c >= 0.0);
        if !colour_ok || !(0.0..=1.0).contains(opacity) {
            return Err(format!("sample {i} is [{r}, {g}, {b}, {opacity}]"));
        }
    }
    Ok(())
}

/// Writes `profile` as RON, one sample per line, replacing any existing file. An invalid profile
/// (as [`load_ring_profile`] would reject it) is [`DataError::BadFormat`] and nothing is written.
pub fn write_ring_profile(path: &Path, profile: &RingProfile) -> Result<(), DataError> {
    check(profile).map_err(DataError::BadFormat)?;
    let mut text = format!(
        "(\n  inner_m: {},\n  outer_m: {},\n  samples: [\n",
        to_ron(&profile.inner_m)?,
        to_ron(&profile.outer_m)?
    );
    for sample in &profile.samples {
        text.push_str(&format!("    {},\n", to_ron(sample)?));
    }
    text.push_str("  ],\n)\n");
    std::fs::write(path, text)?;
    Ok(())
}

fn to_ron<T: serde::Serialize>(value: &T) -> Result<String, DataError> {
    ron::to_string(value)
        .map_err(|err| DataError::BadFormat(format!("serialising the ring profile: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-rings-{}-{}", std::process::id(), name))
    }

    fn sample() -> RingProfile {
        RingProfile {
            inner_m: 7.4e7,
            outer_m: 1.4e8,
            samples: vec![
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.45, 0.4, 0.9],
                [0.3, 0.3, 0.3, 0.25],
            ],
        }
    }

    /// Mean opacity of the samples whose centres lie in [r0_km, r1_km].
    fn mean_opacity(profile: &RingProfile, r0_km: f64, r1_km: f64) -> f64 {
        let n = profile.samples.len() as f64;
        let step = (profile.outer_m - profile.inner_m) / n;
        let picked: Vec<f64> = profile
            .samples
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let r_km = (profile.inner_m + (*i as f64 + 0.5) * step) / 1e3;
                (r0_km..=r1_km).contains(&r_km)
            })
            .map(|(_, s)| f64::from(s[3]))
            .collect();
        assert!(!picked.is_empty(), "no samples in {r0_km}–{r1_km} km");
        picked.iter().sum::<f64>() / picked.len() as f64
    }

    #[test]
    fn ring_profile_roundtrip() {
        let path = temp_path("rings.ron");
        write_ring_profile(&path, &sample()).expect("write");
        let loaded = load_ring_profile(&path);
        let _ = std::fs::remove_file(&path);
        let loaded = loaded.expect("load");
        assert_eq!(loaded.inner_m, sample().inner_m);
        assert_eq!(loaded.outer_m, sample().outer_m);
        assert_eq!(loaded.samples, sample().samples);
    }

    #[test]
    fn load_ring_profile_missing_is_not_found() {
        assert!(matches!(
            load_ring_profile(Path::new("/nonexistent/saturn_rings.ron")),
            Err(DataError::NotFound(_))
        ));
    }

    #[test]
    fn load_ring_profile_rejects_bad_profiles() {
        let path = temp_path("bad_rings.ron");
        let mut cases = vec!["(inner_m: 1.0)".to_string()];
        for edit in 0..5 {
            let mut p = sample();
            match edit {
                0 => p.outer_m = p.inner_m,
                1 => p.samples.truncate(1),
                2 => p.samples[1][3] = 1.5,
                3 => p.samples[1][0] = -0.1,
                _ => p.samples[2][1] = f32::NAN,
            }
            cases.push(ron::to_string(&p).expect("serialise"));
        }
        for (i, text) in cases.iter().enumerate() {
            std::fs::write(&path, text).expect("write");
            let loaded = load_ring_profile(&path);
            assert!(
                matches!(loaded, Err(DataError::BadFormat(_))),
                "case {i}: {loaded:?}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The committed bake shows the Cassini division where Saturn's rings have it: between the B
    /// ring's outer edge (117 507 km) and the A ring's inner edge (122 340 km), NSSDCA Saturnian
    /// Rings Fact Sheet (https://nssdc.gsfc.nasa.gov/planetary/factsheet/satringfact.html).
    #[test]
    fn ring_profile_cassini_division() {
        let path = crate::data_dir().join("baked/saturn_rings.ron");
        let profile = load_ring_profile(&path).expect("committed saturn_rings.ron");
        assert_eq!(profile.samples.len(), 1024);
        let division = mean_opacity(&profile, 118_000.0, 121_000.0);
        let b_ring = mean_opacity(&profile, 100_000.0, 110_000.0);
        assert!(
            division < 0.5 * b_ring,
            "Cassini division opacity {division:.3} vs B ring {b_ring:.3}"
        );
        // The A ring is more opaque than the division again, and beyond its outer edge (136 780 km,
        // same fact sheet) only a faint haze is left before the F ring (139 826 km).
        assert!(mean_opacity(&profile, 126_000.0, 133_000.0) > division);
        assert!(mean_opacity(&profile, 137_500.0, 139_000.0) < 0.1);
    }
}
