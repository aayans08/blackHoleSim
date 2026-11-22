//! Comets (`comets.ron`) and asteroids (`asteroids.bin` + `asteroid_names.txt`) from the JPL
//! Small-Body Database Query API (<https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html>), baked by
//! `uni-bake sbdb` (M1 Task A9).
//!
//! The elements are SBDB's heliocentric osculating elements, referred to the ecliptic and mean
//! equinox of J2000 (SBDB `equinox` J2000), with angles in degrees and the osculation epoch and
//! perihelion time as TDB Julian Dates. Each object has its own epoch.
//!
//! - `comets.ron`: a `//` provenance header, then a RON sequence of [`Comet`], one per line.
//! - `asteroids.bin`: a 16-byte header (the magic `b"UNIAST1\0"`, the record count as a u32 and
//!   the record size as a u32, 80), then the [`AsteroidRecord`]s. Every number is little-endian,
//!   field by field in declaration order, which is the `#[repr(C)]` layout (no padding).
//! - `asteroid_names.txt`: UTF-8, one name per line (SBDB `full_name`, trimmed). A record's
//!   `name_index` is its line, counted from 0.
//!
//! This file ends with a `#[cfg(test)] mod tests` that holds `asteroid_record_is_80_bytes` (lead
//! ruling, review I2) and A9's tests.

use std::path::Path;

use crate::DataError;

/// One comet's osculating elements (J2000 ecliptic), as SBDB gives them.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Comet {
    /// SBDB full name, trimmed: "1P/Halley", "C/1995 O1 (Hale-Bopp)".
    pub name: String,
    /// SBDB primary SPK-ID (1000036 is 1P/Halley).
    pub spkid: i32,
    /// Perihelion distance [au].
    pub q_au: f64,
    /// Eccentricity (≥ 1 for the hyperbolic long-period comets).
    pub e: f64,
    /// Inclination to the J2000 ecliptic [deg].
    pub i_deg: f64,
    /// Longitude of the ascending node [deg].
    pub node_deg: f64,
    /// Argument of perihelion [deg].
    pub argp_deg: f64,
    /// Time of perihelion passage, Julian Date (TDB).
    pub tp_jd_tdb: f64,
    /// Osculation epoch, Julian Date (TDB).
    pub epoch_jd_tdb: f64,
    /// Total-magnitude parameters (m = M1 + 5 log₁₀Δ + K1 log₁₀r).
    pub m1: Option<f64>,
    pub k1: Option<f64>,
    /// Effective diameter [km], where SBDB has one.
    pub diameter_km: Option<f64>,
}

/// Reads `comets.ron`.
///
/// A missing file is [`DataError::NotFound`]. A file that is not a RON sequence of [`Comet`], or
/// that holds an impossible orbit (q ≤ 0, e < 0, i outside [0°, 180°], a non-finite number, an
/// empty name or a diameter ≤ 0), is [`DataError::BadFormat`].
pub fn load_comets(path: &Path) -> Result<Vec<Comet>, DataError> {
    let comets: Vec<Comet> = read_ron(path)?;
    for (index, comet) in comets.iter().enumerate() {
        if let Some(why) = comet_problem(comet) {
            return Err(DataError::BadFormat(format!(
                "{}: comet {index} ({:?}): {why}",
                path.display(),
                comet.name
            )));
        }
    }
    Ok(comets)
}

fn comet_problem(c: &Comet) -> Option<&'static str> {
    let angles_and_times = [c.i_deg, c.node_deg, c.argp_deg, c.tp_jd_tdb, c.epoch_jd_tdb];
    if c.name.trim().is_empty() {
        Some("empty name")
    } else if !(c.q_au.is_finite() && c.q_au > 0.0) {
        Some("perihelion distance q is not > 0")
    } else if !(c.e.is_finite() && c.e >= 0.0) {
        Some("eccentricity is not >= 0")
    } else if !angles_and_times.iter().all(|v| v.is_finite()) {
        Some("non-finite angle or time")
    } else if !(0.0..=180.0).contains(&c.i_deg) {
        Some("inclination outside [0, 180] deg")
    } else if ![c.m1, c.k1].into_iter().flatten().all(f64::is_finite) {
        Some("non-finite magnitude parameter")
    } else if c.diameter_km.is_some_and(|d| !(d.is_finite() && d > 0.0)) {
        Some("diameter is not > 0")
    } else {
        None
    }
}

/// One asteroid's osculating elements (J2000 ecliptic) and photometry, as stored on disk.
///
/// **80 B, no padding:** 7 × f64 + 4 × f32 + 2 × u32. Lead ruling (A0b fix round 1, review
/// I2): the contracted fields stand, so the record is 80 B. The plan's "72 B; ≈ 3.6 MB at 50 k"
/// was a miscount: 80 B gives ≈ 4.0 MB at 50 k, still far below the 10 MB commit limit. The test
/// `asteroid_record_is_80_bytes` below replaces A9's brief test `asteroid_record_is_72_bytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AsteroidRecord {
    /// Semi-major axis [au].
    pub a_au: f64,
    pub e: f64,
    /// Inclination to the J2000 ecliptic [deg].
    pub i_deg: f64,
    /// Longitude of the ascending node [deg].
    pub node_deg: f64,
    /// Argument of perihelion [deg].
    pub argp_deg: f64,
    /// Mean anomaly at `epoch_jd_tdb` [deg].
    pub m0_deg: f64,
    /// Osculation epoch, Julian Date (TDB).
    pub epoch_jd_tdb: f64,
    /// Absolute magnitude H.
    pub h: f32,
    /// H–G slope parameter; 0.15 where SBDB has none (SBDB documents 0.15 as the default).
    pub g: f32,
    /// NaN when unknown.
    pub diameter_km: f32,
    /// V-band geometric albedo. NaN when unknown.
    pub albedo: f32,
    /// SBDB primary SPK-ID. Numbered asteroids are 20 000 000 + their number, so (1) Ceres is
    /// 20000001; the Horizons goldens name Ceres by its older SPK-ID, 2000001.
    pub spkid: u32,
    /// Line of the name in `asteroid_names.txt`.
    pub name_index: u32,
}

const MAGIC: [u8; 8] = *b"UNIAST1\0";
const HEADER_LEN: usize = 16;
const RECORD_LEN: usize = std::mem::size_of::<AsteroidRecord>();

/// Reads `asteroids.bin` (magic `b"UNIAST1\0"`) and its names file.
///
/// A missing file is [`DataError::NotFound`]. A wrong magic, record size or length, a name index
/// past the names file, or an impossible orbit (a ≤ 0, e outside [0, 1), i outside [0°, 180°],
/// a non-finite element, H or G, a diameter that is neither NaN nor > 0, an albedo that is
/// neither NaN nor ≥ 0) is [`DataError::BadFormat`].
pub fn load_asteroids(
    bin: &Path,
    names: &Path,
) -> Result<(Vec<AsteroidRecord>, Vec<String>), DataError> {
    let bytes = std::fs::read(bin).map_err(|err| io_error(bin, err))?;
    let text = std::fs::read_to_string(names).map_err(|err| io_error(names, err))?;
    let names: Vec<String> = text.lines().map(str::to_owned).collect();
    let records = decode_asteroids(&bytes, names.len())
        .map_err(|why| DataError::BadFormat(format!("{}: {why}", bin.display())))?;
    Ok((records, names))
}

/// Writes `asteroids.bin` and its names file (one name per line, in `names` order).
///
/// Refuses, with [`DataError::BadFormat`] and before writing anything, a record that
/// [`load_asteroids`] would reject or a name that is empty or holds a line break.
pub fn write_asteroids(
    bin: &Path,
    names_path: &Path,
    records: &[AsteroidRecord],
    names: &[String],
) -> Result<(), DataError> {
    let count = u32::try_from(records.len()).map_err(|_| {
        DataError::BadFormat(format!(
            "{} asteroids do not fit a u32 count",
            records.len()
        ))
    })?;
    if let Some(name) = names
        .iter()
        .find(|n| n.is_empty() || n.contains(['\n', '\r']))
    {
        return Err(DataError::BadFormat(format!(
            "asteroid name {name:?} is empty or holds a line break"
        )));
    }
    for (index, record) in records.iter().enumerate() {
        if let Some(why) = asteroid_problem(record, names.len()) {
            return Err(DataError::BadFormat(format!(
                "asteroid {index} (SPK-ID {}): {why}",
                record.spkid
            )));
        }
    }
    let mut bytes = Vec::with_capacity(HEADER_LEN + RECORD_LEN * records.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&(RECORD_LEN as u32).to_le_bytes());
    for r in records {
        let wide = [
            r.a_au,
            r.e,
            r.i_deg,
            r.node_deg,
            r.argp_deg,
            r.m0_deg,
            r.epoch_jd_tdb,
        ];
        for value in wide {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [r.h, r.g, r.diameter_km, r.albedo] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [r.spkid, r.name_index] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(bin, bytes)?;
    let mut text = names.join("\n");
    if !names.is_empty() {
        text.push('\n');
    }
    std::fs::write(names_path, text)?;
    Ok(())
}

fn decode_asteroids(bytes: &[u8], names: usize) -> Result<Vec<AsteroidRecord>, String> {
    let (header, body) = bytes.split_at_checked(HEADER_LEN).ok_or_else(|| {
        format!(
            "{} bytes is shorter than the {HEADER_LEN}-byte header",
            bytes.len()
        )
    })?;
    if header[..8] != MAGIC {
        return Err("bad magic: not a UNIAST1 file".into());
    }
    let count = le_u32(header, 8) as usize;
    let record_len = le_u32(header, 12) as usize;
    if record_len != RECORD_LEN {
        return Err(format!(
            "record size {record_len} B; this build reads {RECORD_LEN} B"
        ));
    }
    if Some(body.len()) != count.checked_mul(RECORD_LEN) {
        return Err(format!(
            "header says {count} asteroids ({} B) but {} B follow it",
            count.saturating_mul(RECORD_LEN),
            body.len()
        ));
    }
    body.chunks_exact(RECORD_LEN)
        .enumerate()
        .map(|(index, chunk)| {
            let f64_at = |i: usize| f64::from_bits(le_u64(chunk, 8 * i));
            let f32_at = |i: usize| f32::from_bits(le_u32(chunk, 56 + 4 * i));
            let record = AsteroidRecord {
                a_au: f64_at(0),
                e: f64_at(1),
                i_deg: f64_at(2),
                node_deg: f64_at(3),
                argp_deg: f64_at(4),
                m0_deg: f64_at(5),
                epoch_jd_tdb: f64_at(6),
                h: f32_at(0),
                g: f32_at(1),
                diameter_km: f32_at(2),
                albedo: f32_at(3),
                spkid: le_u32(chunk, 72),
                name_index: le_u32(chunk, 76),
            };
            match asteroid_problem(&record, names) {
                Some(why) => Err(format!("asteroid {index} (SPK-ID {}): {why}", record.spkid)),
                None => Ok(record),
            }
        })
        .collect()
}

fn asteroid_problem(r: &AsteroidRecord, names: usize) -> Option<&'static str> {
    let angles_and_epoch = [r.i_deg, r.node_deg, r.argp_deg, r.m0_deg, r.epoch_jd_tdb];
    let unknown_or_positive = |v: f32| v.is_nan() || (v.is_finite() && v > 0.0);
    if !(r.a_au.is_finite() && r.a_au > 0.0) {
        Some("semi-major axis is not > 0")
    } else if !(r.e.is_finite() && (0.0..1.0).contains(&r.e)) {
        Some("eccentricity outside [0, 1)")
    } else if !angles_and_epoch.iter().all(|v| v.is_finite()) {
        Some("non-finite angle or epoch")
    } else if !(0.0..=180.0).contains(&r.i_deg) {
        Some("inclination outside [0, 180] deg")
    } else if !(r.h.is_finite() && r.g.is_finite()) {
        Some("non-finite H or G")
    } else if !unknown_or_positive(r.diameter_km) {
        Some("diameter is neither NaN nor > 0")
    } else if !(r.albedo.is_nan() || (r.albedo.is_finite() && r.albedo >= 0.0)) {
        Some("albedo is neither NaN nor >= 0")
    } else if r.name_index as usize >= names {
        Some("name_index is past the end of the names file")
    } else {
        None
    }
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(std::array::from_fn(|i| bytes[offset + i]))
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(std::array::from_fn(|i| bytes[offset + i]))
}

/// A missing file is [`DataError::NotFound`]; any other I/O failure is [`DataError::Io`].
fn io_error(path: &Path, err: std::io::Error) -> DataError {
    match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    }
}

/// Reads and parses a whole RON file (shared by the A9 loaders in `sats` and `facts`). A missing
/// file is [`DataError::NotFound`]; text that is not RON of `T` is [`DataError::BadFormat`].
pub(crate) fn read_ron<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, DataError> {
    let text = std::fs::read_to_string(path).map_err(|err| io_error(path, err))?;
    ron::from_str(&text).map_err(|err| DataError::BadFormat(format!("{}: {err}", path.display())))
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};
    use std::path::PathBuf;

    use super::*;

    /// Lead ruling (review I2): the on-disk record is 80 B. A9 keeps this layout.
    #[test]
    fn asteroid_record_is_80_bytes() {
        assert_eq!(size_of::<AsteroidRecord>(), 80);
        assert_eq!(align_of::<AsteroidRecord>(), 8);
        assert_eq!(offset_of!(AsteroidRecord, epoch_jd_tdb), 48);
        assert_eq!(offset_of!(AsteroidRecord, h), 56);
        assert_eq!(offset_of!(AsteroidRecord, albedo), 68);
        assert_eq!(offset_of!(AsteroidRecord, spkid), 72);
        assert_eq!(offset_of!(AsteroidRecord, name_index), 76);
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-a9-{}-{name}", std::process::id()))
    }

    fn baked(file: &str) -> PathBuf {
        crate::data_dir().join("baked").join(file)
    }

    /// (1) Ceres as SBDB served it on 2026-11-22, and an object with no diameter or albedo.
    fn sample_asteroids() -> (Vec<AsteroidRecord>, Vec<String>) {
        let ceres = AsteroidRecord {
            a_au: 2.765_552_595_034_094,
            e: 0.079_692_295_148_165_86,
            i_deg: 10.588_027_801_834_62,
            node_deg: 80.248_626_820_432_21,
            argp_deg: 73.294_214_530_215_87,
            m0_deg: 274.419_346_376_134_2,
            epoch_jd_tdb: 2_461_200.5,
            h: 3.34,
            g: 0.12,
            diameter_km: 939.4,
            albedo: 0.09,
            spkid: 20_000_001,
            name_index: 0,
        };
        let unknown_size = AsteroidRecord {
            a_au: 3.1,
            e: 0.25,
            i_deg: 179.5,
            node_deg: 0.0,
            argp_deg: 359.9,
            m0_deg: -0.0,
            epoch_jd_tdb: 2_461_200.5,
            h: 14.0,
            g: 0.15,
            diameter_km: f32::NAN,
            albedo: f32::NAN,
            spkid: 54_321_098,
            name_index: 1,
        };
        let names = vec!["1 Ceres (A801 AA)".to_owned(), "(2001 XY123)".to_owned()];
        (vec![ceres, unknown_size], names)
    }

    fn bytes(records: &[AsteroidRecord]) -> &[u8] {
        bytemuck::cast_slice(records)
    }

    #[test]
    fn asteroid_roundtrip() {
        let (records, names) = sample_asteroids();
        let (bin, txt) = (temp_path("rt.bin"), temp_path("rt.txt"));
        write_asteroids(&bin, &txt, &records, &names).expect("write");
        let len = std::fs::metadata(&bin).map(|m| m.len());
        let loaded = load_asteroids(&bin, &txt);
        let _ = (std::fs::remove_file(&bin), std::fs::remove_file(&txt));
        assert_eq!(len.expect("metadata"), 16 + 80 * 2);
        let (back, back_names) = loaded.expect("load");
        // Bitwise, so the NaN "unknown" markers and −0.0 count too.
        assert_eq!(bytes(&back), bytes(&records));
        assert_eq!(back_names, names);
    }

    #[test]
    fn asteroid_loader_rejects_bad_files() {
        let (records, names) = sample_asteroids();
        let (bin, txt) = (temp_path("bad.bin"), temp_path("bad.txt"));
        write_asteroids(&bin, &txt, &records, &names).expect("write");
        let good = std::fs::read(&bin).expect("read back");
        let check = |what: &str, body: &[u8], names: &str| {
            std::fs::write(&bin, body).expect("write bin");
            std::fs::write(&txt, names).expect("write names");
            let result = load_asteroids(&bin, &txt);
            assert!(
                matches!(result, Err(DataError::BadFormat(_))),
                "{what}: {result:?}"
            );
        };
        let names_ok = "1 Ceres (A801 AA)\n(2001 XY123)\n";
        let mut bad_magic = good.clone();
        bad_magic[3] = b'X';
        check("magic", &bad_magic, names_ok);
        check("truncated", &good[..good.len() - 1], names_ok);
        check("short header", &good[..10], names_ok);
        let mut bad_count = good.clone();
        bad_count[8] = 3;
        check("count", &bad_count, names_ok);
        let mut bad_size = good.clone();
        bad_size[12] = 72;
        check("record size", &bad_size, names_ok);
        check("name index past the names", &good, "1 Ceres (A801 AA)\n");
        let mut nan_a = good.clone();
        nan_a[16..24].copy_from_slice(&f64::NAN.to_le_bytes());
        check("NaN a", &nan_a, names_ok);
        let mut hyperbolic = good.clone();
        hyperbolic[24..32].copy_from_slice(&1.2_f64.to_le_bytes());
        check("e >= 1", &hyperbolic, names_ok);
        let _ = (std::fs::remove_file(&bin), std::fs::remove_file(&txt));

        let missing = load_asteroids(Path::new("/nonexistent/a.bin"), Path::new("/x.txt"));
        assert!(
            matches!(missing, Err(DataError::NotFound(_))),
            "{missing:?}"
        );
        let mut bad = records.clone();
        bad[1].name_index = 2;
        let unwritable = write_asteroids(&temp_path("w.bin"), &temp_path("w.txt"), &bad, &names);
        assert!(unwritable.is_err(), "writer accepted a dangling name index");
        let two_lines = vec!["a\nb".to_owned(), "c".to_owned()];
        let unwritable = write_asteroids(
            &temp_path("w.bin"),
            &temp_path("w.txt"),
            &records,
            &two_lines,
        );
        assert!(
            unwritable.is_err(),
            "writer accepted a name with a line break"
        );
    }

    fn sample_comets() -> Vec<Comet> {
        vec![
            Comet {
                name: "1P/Halley".to_owned(),
                spkid: 1_000_036,
                q_au: 0.574_863_831_374_341_3,
                e: 0.967_935_995_695_321_1,
                i_deg: 162.190_530_043_912_9,
                node_deg: 59.098_947_206_124_37,
                argp_deg: 112.241_431_463_776_4,
                tp_jd_tdb: 2_446_469.973_616_146_677,
                epoch_jd_tdb: 2_439_875.5,
                m1: Some(5.5),
                k1: Some(8.0),
                diameter_km: Some(11.0),
            },
            Comet {
                name: "C/2023 A3 (Tsuchinshan-ATLAS)".to_owned(),
                spkid: 1_004_083,
                q_au: 0.391,
                e: 1.000_1,
                i_deg: 139.1,
                node_deg: 21.6,
                argp_deg: 308.5,
                tp_jd_tdb: 2_460_581.2,
                epoch_jd_tdb: 2_460_600.5,
                m1: None,
                k1: None,
                diameter_km: None,
            },
        ]
    }

    fn comet_bits(c: &Comet) -> (String, i32, Vec<u64>, [Option<u64>; 3]) {
        let numbers = [
            c.q_au,
            c.e,
            c.i_deg,
            c.node_deg,
            c.argp_deg,
            c.tp_jd_tdb,
            c.epoch_jd_tdb,
        ];
        let opt = |v: Option<f64>| v.map(f64::to_bits);
        (
            c.name.clone(),
            c.spkid,
            numbers.iter().map(|v| v.to_bits()).collect(),
            [opt(c.m1), opt(c.k1), opt(c.diameter_km)],
        )
    }

    #[test]
    fn comet_ron_roundtrip() {
        let comets = sample_comets();
        let path = temp_path("comets.ron");
        let body =
            ron::ser::to_string_pretty(&comets, ron::ser::PrettyConfig::new()).expect("serialise");
        std::fs::write(&path, format!("// provenance comment\n{body}\n")).expect("write");
        let loaded = load_comets(&path);
        let _ = std::fs::remove_file(&path);
        let loaded = loaded.expect("load");
        let bits = |list: &[Comet]| list.iter().map(comet_bits).collect::<Vec<_>>();
        assert_eq!(bits(&loaded), bits(&comets));
    }

    #[test]
    fn comet_loader_rejects_bad_values() {
        let path = temp_path("bad-comets.ron");
        let check = |what: &str, edit: fn(&mut Comet)| {
            let mut comets = sample_comets();
            edit(&mut comets[0]);
            let body = ron::ser::to_string(&comets).expect("serialise");
            std::fs::write(&path, body).expect("write");
            let result = load_comets(&path);
            assert!(
                matches!(result, Err(DataError::BadFormat(_))),
                "{what}: {result:?}"
            );
        };
        check("q <= 0", |c| c.q_au = 0.0);
        check("e < 0", |c| c.e = -0.1);
        check("NaN node", |c| c.node_deg = f64::NAN);
        check("i > 180", |c| c.i_deg = 181.0);
        check("empty name", |c| c.name.clear());
        check("NaN M1", |c| c.m1 = Some(f64::NAN));
        check("diameter <= 0", |c| c.diameter_km = Some(0.0));
        std::fs::write(&path, "[(name: \"x\")]").expect("write");
        assert!(matches!(load_comets(&path), Err(DataError::BadFormat(_))));
        let _ = std::fs::remove_file(&path);
        let missing = load_comets(Path::new("/nonexistent/comets.ron"));
        assert!(
            matches!(missing, Err(DataError::NotFound(_))),
            "{missing:?}"
        );
    }

    fn baked_asteroids() -> (Vec<AsteroidRecord>, Vec<String>) {
        load_asteroids(&baked("asteroids.bin"), &baked("asteroid_names.txt")).unwrap_or_else(
            |err| panic!("baked asteroids missing or unreadable: {err} (A9 Step 2)"),
        )
    }

    fn baked_comets() -> Vec<Comet> {
        load_comets(&baked("comets.ron"))
            .unwrap_or_else(|err| panic!("baked comets missing or unreadable: {err} (A9 Step 2)"))
    }

    /// SBDB's primary SPK-ID of (1) Ceres (numbered asteroids are 20 000 000 + number).
    const CERES_SPKID: u32 = 20_000_001;

    #[test]
    fn ceres_row() {
        let (records, names) = baked_asteroids();
        let ceres: Vec<_> = records.iter().filter(|r| r.spkid == CERES_SPKID).collect();
        assert_eq!(ceres.len(), 1, "Ceres must appear exactly once");
        let ceres = ceres[0];
        assert_eq!(names[ceres.name_index as usize], "1 Ceres (A801 AA)");
        // Ceres: a ≈ 2.77 au, H ≈ 3.3 (brief; SBDB gives a = 2.7656 au, H = 3.34).
        assert!((ceres.a_au - 2.77).abs() <= 0.01, "a = {}", ceres.a_au);
        assert!((ceres.h - 3.3).abs() <= 0.2, "H = {}", ceres.h);
        assert!(ceres.diameter_km > 900.0 && ceres.albedo > 0.0, "{ceres:?}");
    }

    #[test]
    fn halley_row() {
        let comets = baked_comets();
        let halley: Vec<_> = comets.iter().filter(|c| c.name == "1P/Halley").collect();
        assert_eq!(halley.len(), 1, "1P/Halley must appear exactly once");
        let halley = halley[0];
        assert_eq!(halley.spkid, 1_000_036);
        // 1P/Halley: q ≈ 0.575 au, e ≈ 0.967 (brief; SBDB JPL#75 gives 0.5749 au, 0.96794).
        assert!((halley.q_au - 0.575).abs() <= 0.01, "q = {}", halley.q_au);
        assert!((halley.e - 0.967).abs() <= 0.002, "e = {}", halley.e);
    }

    /// The baked files hold what A9 Step 2 asked for: every asteroid with H ≤ 14 and the numbered
    /// periodic comets plus the six named long-period ones, each object once.
    #[test]
    fn baked_small_bodies_match_the_query() {
        let (records, names) = baked_asteroids();
        // N(H ≤ 14) was 33 681 on 2026-11-22; the catalogue only grows.
        assert!(records.len() >= 33_000, "{} asteroids", records.len());
        assert_eq!(records.len(), names.len());
        let mut spkids: Vec<u32> = records.iter().map(|r| r.spkid).collect();
        spkids.sort_unstable();
        spkids.dedup();
        assert_eq!(spkids.len(), records.len(), "duplicate asteroid SPK-IDs");
        for (index, r) in records.iter().enumerate() {
            assert!(r.h <= 14.0, "record {index}: H = {}", r.h);
            assert_eq!(r.name_index as usize, index, "names are in record order");
        }
        assert!(records.iter().any(|r| r.diameter_km.is_nan()));
        assert!(records.iter().filter(|r| r.albedo.is_finite()).count() > 10_000);

        let comets = baked_comets();
        let numbered = comets
            .iter()
            .filter(|c| c.name.starts_with(|ch: char| ch.is_ascii_digit()))
            .count();
        assert!(numbered >= 500, "{numbered} numbered periodic comets");
        for designation in [
            "C/1995 O1",
            "C/2020 F3",
            "C/1996 B2",
            "C/2006 P1",
            "C/2023 A3",
            "C/2011 L4",
        ] {
            let found = comets
                .iter()
                .filter(|c| {
                    c.name == designation || c.name.starts_with(&format!("{designation} ("))
                })
                .count();
            assert_eq!(found, 1, "{designation}");
        }
        assert_eq!(
            comets.len(),
            numbered + 6,
            "only numbered and the named six"
        );
        // A fragment's designation carries a letter after a hyphen: "73P-B/Schwassmann-Wachmann".
        let fragment = |c: &&Comet| c.name.split('/').next().is_some_and(|d| d.contains('-'));
        let fragments: Vec<_> = comets.iter().filter(fragment).map(|c| &c.name).collect();
        assert!(fragments.is_empty(), "comet fragments: {fragments:?}");
    }
}
