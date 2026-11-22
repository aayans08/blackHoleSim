//! AT-HYG v4.0 stars with V ≤ 10 (D. Nash, astronexus; CC BY-SA 4.0) (M1 Task A7).
//! `uni-bake athyg` writes the file from the `athyg_40_reduced_m10` subset.
//!
//! Binary format v1, little-endian: magic `b"UNIATHY\0"`, `u32` version 1, `u32` count, then
//! `count` × 28-byte [`Star3`] records, each 7 little-endian 32-bit words in field order:
//! `pos_pc` x, y, z (f32), `abs_mag_v` (f32), `vel_bv[0]` = f16 vx | f16 vy << 16,
//! `vel_bv[1]` = f16 vz | f16 B−V << 16, `id_flags`.
//!
//! Known source limitation: AT-HYG v4.0 takes the positions of Tycho-2 supplement-1 stars
//! (bright Hipparcos stars such as Sirius, Vega and Arcturus) at their epoch J1991.25, not
//! J2000, so those stars are off by 8.75 yr of proper motion (Sirius 11.7″, Arcturus 20″).

use std::path::Path;

use crate::DataError;
use crate::f16::{f16_to_f32, f32_to_f16};

/// One star as stored on disk (B4 builds its own vec4-only GPU record).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Star3 {
    /// Position [pc], ICRS, origin SSB, at [`STAR3_EPOCH_JD_TDB`].
    pub pos_pc: [f32; 3],
    /// Absolute magnitude, Johnson V (Tycho rows converted).
    pub abs_mag_v: f32,
    /// f16 pairs: (vx, vy), (vz, B−V); velocities [km/s], ICRS.
    pub vel_bv: [u32; 2],
    /// Low 22 bits: AT-HYG id (or `ExoSystem` index for archive hosts); flag bits above.
    pub id_flags: u32,
}

/// Size of one [`Star3`] record [bytes].
pub const STAR3_SIZE: usize = 28;
/// Epoch of [`Star3::pos_pc`] [JD TDB]. AT-HYG: "epoch + equinox 2000.0" (version-info.md).
pub const STAR3_EPOCH_JD_TDB: f64 = 2451545.0;
/// The star has a proper name or Bayer/Flamsteed designation in `athyg_names.ron`.
pub const FLAG_NAMED: u32 = 1 << 22;
/// The star hosts known exoplanets.
pub const FLAG_EXO_HOST: u32 = 1 << 23;
/// AT-HYG computed no Cartesian velocity (missing distance or proper motion): stored as 0.
pub const FLAG_NO_VELOCITY: u32 = 1 << 24;
/// Radial velocity missing: tangential motion only.
pub const FLAG_NO_RV: u32 = 1 << 25;
/// Built by uni-sim from the NASA Exoplanet Archive (review I11); never in the baked file.
pub const FLAG_FROM_ARCHIVE: u32 = 1 << 26;

/// Mask of the id field (low 22 bits) of [`Star3::id_flags`].
const ID_MASK: u32 = (1 << 22) - 1;
/// Bits 27..=31 of [`Star3::id_flags`] are unused; a file that sets them is corrupt.
const RESERVED_MASK: u32 =
    !(ID_MASK | FLAG_NAMED | FLAG_EXO_HOST | FLAG_NO_VELOCITY | FLAG_NO_RV | FLAG_FROM_ARCHIVE);

const MAGIC: [u8; 8] = *b"UNIATHY\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 16;

impl Star3 {
    /// Packs one record. `vel_kms` and `bv` are rounded to binary16 (11 significant bits,
    /// 0.05 %); `id` is masked to the low 22 bits and `flags` to the bits above them.
    pub fn new(
        pos_pc: [f32; 3],
        abs_mag_v: f32,
        vel_kms: [f32; 3],
        bv: f32,
        id: u32,
        flags: u32,
    ) -> Self {
        let pack = |lo: f32, hi: f32| u32::from(f32_to_f16(lo)) | u32::from(f32_to_f16(hi)) << 16;
        Self {
            pos_pc,
            abs_mag_v,
            vel_bv: [pack(vel_kms[0], vel_kms[1]), pack(vel_kms[2], bv)],
            id_flags: (id & ID_MASK) | (flags & !ID_MASK),
        }
    }

    /// Velocity [km/s], ICRS.
    pub fn velocity_kms(&self) -> glam::Vec3 {
        let [x, y] = unpack(self.vel_bv[0]);
        let [z, _] = unpack(self.vel_bv[1]);
        glam::Vec3::new(x, y, z)
    }

    /// Johnson B−V colour index.
    pub fn bv(&self) -> f32 {
        unpack(self.vel_bv[1])[1]
    }

    /// AT-HYG id (low 22 bits of `id_flags`).
    pub fn athyg_id(&self) -> u32 {
        self.id_flags & ID_MASK
    }

    /// Flag bits of `id_flags`.
    pub fn flags(&self) -> u32 {
        self.id_flags & !ID_MASK
    }

    pub fn is_named(&self) -> bool {
        self.id_flags & FLAG_NAMED != 0
    }

    pub fn is_exo_host(&self) -> bool {
        self.id_flags & FLAG_EXO_HOST != 0
    }
}

/// The two binary16 halves of `word`, low half first (the order of WGSL `unpack2x16float`).
fn unpack(word: u32) -> [f32; 2] {
    [f16_to_f32(word as u16), f16_to_f32((word >> 16) as u16)]
}

/// Reads a baked `athyg_m10.bin`. Missing → [`DataError::NotFound`]; wrong magic, version or
/// length → [`DataError::BadFormat`]. A record with a non-finite value or a reserved flag bit
/// is also [`DataError::BadFormat`], so a NULL catalogue value can never reach the GPU as NaN.
pub fn load_athyg(path: &Path) -> Result<Vec<Star3>, DataError> {
    let bytes = std::fs::read(path).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    })?;
    decode(&bytes).map_err(|why| DataError::BadFormat(format!("{}: {why}", path.display())))
}

/// Writes `s` as a baked `athyg_m10.bin`, replacing any existing file.
pub fn write_athyg(path: &Path, s: &[Star3]) -> Result<(), DataError> {
    let count = u32::try_from(s.len())
        .map_err(|_| DataError::BadFormat(format!("{} stars do not fit a u32 count", s.len())))?;
    let mut bytes = Vec::with_capacity(HEADER_LEN + STAR3_SIZE * s.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    for star in s {
        let words = [
            star.pos_pc[0].to_bits(),
            star.pos_pc[1].to_bits(),
            star.pos_pc[2].to_bits(),
            star.abs_mag_v.to_bits(),
            star.vel_bv[0],
            star.vel_bv[1],
            star.id_flags,
        ];
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn decode(bytes: &[u8]) -> Result<Vec<Star3>, String> {
    let (header, body) = bytes.split_at_checked(HEADER_LEN).ok_or_else(|| {
        format!(
            "{} bytes is shorter than the {HEADER_LEN}-byte header",
            bytes.len()
        )
    })?;
    if header[..8] != MAGIC {
        return Err("bad magic: not a UNIATHY file".into());
    }
    let version = le_u32(header, 8);
    if version != VERSION {
        return Err(format!("version {version}; this build reads {VERSION}"));
    }
    let count = le_u32(header, 12) as usize;
    if Some(body.len()) != count.checked_mul(STAR3_SIZE) {
        return Err(format!(
            "header says {count} stars ({} B) but {} B follow it",
            count.saturating_mul(STAR3_SIZE),
            body.len()
        ));
    }
    body.chunks_exact(STAR3_SIZE)
        .enumerate()
        .map(|(index, record)| decode_record(index, record))
        .collect()
}

fn decode_record(index: usize, record: &[u8]) -> Result<Star3, String> {
    let word = |i: usize| le_u32(record, 4 * i);
    let star = Star3 {
        pos_pc: [
            f32::from_bits(word(0)),
            f32::from_bits(word(1)),
            f32::from_bits(word(2)),
        ],
        abs_mag_v: f32::from_bits(word(3)),
        vel_bv: [word(4), word(5)],
        id_flags: word(6),
    };
    let finite = glam::Vec3::from(star.pos_pc).is_finite()
        && star.abs_mag_v.is_finite()
        && star.velocity_kms().is_finite()
        && star.bv().is_finite();
    if !finite || star.id_flags & RESERVED_MASK != 0 {
        return Err(format!(
            "record {index} (AT-HYG id {}) is invalid: {star:?}",
            star.athyg_id()
        ));
    }
    Ok(star)
}

/// The little-endian `u32` at `offset`; callers guarantee `offset + 4 <= bytes.len()`.
fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(std::array::from_fn(|i| bytes[offset + i]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A per-process scratch path (no tempfile crate).
    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-athyg-{}-{}", std::process::id(), name))
    }

    fn sample_stars() -> [Star3; 3] {
        [
            // Sirius-like: AT-HYG id 586693, named.
            Star3::new(
                [-0.4944, 2.4768, -0.7584],
                1.434,
                [8.59, -8.14, -13.06],
                0.0,
                586_693,
                FLAG_NAMED,
            ),
            // No velocity.
            Star3::new(
                [506.86, 0.003, -48.75],
                0.705,
                [0.0; 3],
                0.94,
                2,
                FLAG_NO_VELOCITY,
            ),
            // Largest id, tangential motion only, exoplanet host.
            Star3::new(
                [1.0, 2.0, 3.0],
                13.47,
                [-5.82, 117.51, 80.47],
                1.62,
                (1 << 22) - 1,
                FLAG_NO_RV | FLAG_EXO_HOST,
            ),
        ]
    }

    #[test]
    fn star3_is_28_bytes() {
        assert_eq!(std::mem::size_of::<Star3>(), STAR3_SIZE);
        assert_eq!(std::mem::align_of::<Star3>(), 4);
    }

    #[test]
    fn star3_packs_and_unpacks() {
        let [sirius, no_velocity, last] = sample_stars();
        assert_eq!(sirius.athyg_id(), 586_693);
        assert_eq!(sirius.flags(), FLAG_NAMED);
        assert!(sirius.is_named() && !sirius.is_exo_host());
        assert_eq!(sirius.bv(), 0.0);
        // f16 keeps 11 significant bits: 0.05 % of each component.
        let v = sirius.velocity_kms();
        for (got, want) in [(v.x, 8.59_f32), (v.y, -8.14), (v.z, -13.06)] {
            assert!(
                (got - want).abs() <= want.abs() * 2f32.powi(-11),
                "{got} vs {want}"
            );
        }
        assert_eq!(no_velocity.velocity_kms(), glam::Vec3::ZERO);
        assert_eq!(no_velocity.flags(), FLAG_NO_VELOCITY);
        assert_eq!(last.athyg_id(), (1 << 22) - 1);
        assert_eq!(last.flags(), FLAG_NO_RV | FLAG_EXO_HOST);
        assert!(last.is_exo_host() && !last.is_named());
        assert!((last.bv() - 1.62).abs() < 1e-3);
    }

    #[test]
    fn star3_velocity_layout_matches_unpack2x16float() {
        // WGSL unpack2x16float(e) returns (f16 of bits 0..16, f16 of bits 16..32): vx and vz low.
        let star = Star3::new([0.0; 3], 0.0, [1.0, -2.0, 0.5], 1.5, 7, 0);
        assert_eq!(star.vel_bv, [0xC000_3C00, 0x3E00_3800]);
    }

    #[test]
    fn athyg_roundtrip() {
        let path = temp_path("roundtrip.bin");
        write_athyg(&path, &sample_stars()).expect("write");
        let len = std::fs::metadata(&path).expect("metadata").len();
        let loaded = load_athyg(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(len, 16 + 3 * 28);
        assert_eq!(loaded.expect("load"), sample_stars());
    }

    #[test]
    fn athyg_empty_roundtrip() {
        let path = temp_path("empty.bin");
        write_athyg(&path, &[]).expect("write");
        let loaded = load_athyg(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.expect("load"), Vec::new());
    }

    #[test]
    fn load_missing_is_not_found() {
        // Review Focus 3: a missing baked file is reported, not a panic.
        assert!(matches!(
            load_athyg(Path::new("/nonexistent/athyg_m10.bin")),
            Err(DataError::NotFound(_))
        ));
    }

    #[test]
    fn load_athyg_truncated_is_bad_format() {
        // Review Focus 3: the header says 10 stars but only 3 records follow.
        let path = temp_path("truncated.bin");
        write_athyg(&path, &sample_stars()).expect("write");
        let mut bytes = std::fs::read(&path).expect("read");
        bytes[12..16].copy_from_slice(&10u32.to_le_bytes());
        std::fs::write(&path, &bytes).expect("rewrite");
        let loaded = load_athyg(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }

    #[test]
    fn load_bad_magic_or_version_is_bad_format() {
        let path = temp_path("bad_header.bin");
        write_athyg(&path, &sample_stars()).expect("write");
        let good = std::fs::read(&path).expect("read");
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        let mut bad_version = good.clone();
        bad_version[8..12].copy_from_slice(&2u32.to_le_bytes());
        let mut short = good;
        short.truncate(10);
        for (what, bytes) in [
            ("magic", bad_magic),
            ("version", bad_version),
            ("short", short),
        ] {
            std::fs::write(&path, &bytes).expect("rewrite");
            let loaded = load_athyg(&path);
            assert!(
                matches!(loaded, Err(DataError::BadFormat(_))),
                "{what}: {loaded:?}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_non_finite_value_is_bad_format() {
        // Review Focus 3: NULL data must never reach the renderer as NaN.
        let mut stars = sample_stars();
        stars[1].abs_mag_v = f32::NAN;
        let mut bad_velocity = sample_stars();
        bad_velocity[0].vel_bv[0] = 0x7E00_3C00; // vy = f16 NaN
        let mut bad_position = sample_stars();
        bad_position[2].pos_pc[1] = f32::INFINITY;
        let path = temp_path("non_finite.bin");
        for stars in [stars, bad_velocity, bad_position] {
            write_athyg(&path, &stars).expect("write");
            let loaded = load_athyg(&path);
            assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn committed_bake_loads() {
        // data/baked/athyg_m10.bin is committed (M1 Task A7): the 332 178 subset rows minus
        // 2317 NULL distances, 92 HYG placeholder distances and the Sun.
        let path = crate::data_dir_from(None).join("baked/athyg_m10.bin");
        let stars = load_athyg(&path).expect("committed athyg_m10.bin");
        assert_eq!(stars.len(), 329_768);
        let apparent_v =
            |s: &Star3| s.abs_mag_v + 5.0 * glam::Vec3::from(s.pos_pc).length().log10() - 5.0;
        let brightest = stars
            .iter()
            .min_by(|a, b| apparent_v(a).total_cmp(&apparent_v(b)))
            .expect("non-empty");
        assert_eq!(brightest.athyg_id(), 586_693, "Sirius");
        assert!((apparent_v(brightest) - -1.46).abs() < 1e-3);
        assert!(brightest.bv().abs() < 0.02 && brightest.is_named());
        let count = |flag: u32| stars.iter().filter(|s| s.flags() & flag != 0).count();
        assert_eq!(count(FLAG_NAMED), 3505);
        assert_eq!(count(FLAG_EXO_HOST), 1018);
        assert_eq!(count(FLAG_NO_VELOCITY), 25);
        assert_eq!(count(FLAG_NO_RV), 41_943);
        assert_eq!(
            count(FLAG_FROM_ARCHIVE),
            0,
            "archive hosts are never in the baked file"
        );
        assert!(
            stars.iter().all(|s| s.velocity_kms().length() < 4000.0),
            "placeholder distances are gone"
        );
    }

    #[test]
    fn load_reserved_flag_bits_is_bad_format() {
        let mut stars = sample_stars();
        stars[0].id_flags |= 1 << 31;
        let path = temp_path("reserved_bits.bin");
        write_athyg(&path, &stars).expect("write");
        let loaded = load_athyg(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }
}
