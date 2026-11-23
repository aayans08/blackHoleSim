//! Star sectors (Task B4): 16-pc cubes, each with an f64 centre, so the GPU works with f32
//! offsets from a nearby origin and stays precise far from the SSB (design §5.2; Review Focus 5).
//!
//! [`SectorTable::build`] sorts the stars by sector at load, so each sector's stars are one
//! contiguous run of instance slots. Each star's GPU record holds its f32 offset from its
//! sector's centre. The GPU gets every occupied sector's centre relative to an *anchor* near the
//! camera (f64 difference, then f32; [`SectorTable::centres_rel`]), plus the camera − anchor
//! offset (f64 difference, then f32) in a uniform. The star pass re-uploads the centres only
//! when the camera has moved more than `stars3d::REANCHOR_PC` from the anchor: the baked AT-HYG
//! file occupies 166 011 sectors (2.66 MB of vec4s), too much to upload every frame while the
//! camera moves inside the Solar System.

use glam::{DVec3, IVec3};

/// Edge of a star sector [pc] (design §5.2).
pub const SECTOR_SIZE_PC: f64 = 16.0;

/// The sector holding the ICRS position `pos_pc` [pc, origin SSB]: floor(pos / 16 pc) per axis.
/// Casts saturate, and a NaN axis gives 0.
pub fn sector_of(pos_pc: DVec3) -> IVec3 {
    (pos_pc / SECTOR_SIZE_PC).floor().as_ivec3()
}

/// Centre of sector `k` [pc, ICRS, origin SSB]: (k + ½)·16 pc. A star's offset from its sector's
/// centre is at most 8 pc on each axis, half what an offset from the sector's corner would be.
pub fn sector_centre_pc(k: IVec3) -> DVec3 {
    (k.as_dvec3() + 0.5) * SECTOR_SIZE_PC
}

/// Stars grouped by sector: the instance order of the star pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SectorTable {
    /// The occupied sectors, in sort order (by x, then y, then z).
    pub sectors: Vec<IVec3>,
    /// `order[slot]`: index (into the slice the keys came from) of the star in instance slot
    /// `slot`. The stars of one sector are contiguous, in their input order.
    pub order: Vec<u32>,
    /// `sector_index[slot]`: index into `sectors` of the star in slot `slot`.
    pub sector_index: Vec<u32>,
}

impl SectorTable {
    /// Sorts stars with sector `keys[i]` by sector (a stable sort). The caller guarantees that
    /// `keys.len()` fits a `u32`.
    pub fn build(keys: &[IVec3]) -> Self {
        let mut order: Vec<u32> = (0..keys.len() as u32).collect();
        // `sort_by_key` is stable, so each sector keeps its stars in input order.
        order.sort_by_key(|&i| {
            let k = keys[i as usize];
            (k.x, k.y, k.z)
        });
        let mut sectors: Vec<IVec3> = Vec::new();
        let sector_index = order
            .iter()
            .map(|&i| {
                let k = keys[i as usize];
                if sectors.last() != Some(&k) {
                    sectors.push(k);
                }
                (sectors.len() - 1) as u32
            })
            .collect();
        Self {
            sectors,
            order,
            sector_index,
        }
    }

    /// Instance slot of every input star: the inverse of `order`.
    pub fn slots(&self) -> Vec<u32> {
        let mut slots = vec![0; self.order.len()];
        for (slot, &i) in self.order.iter().enumerate() {
            slots[i as usize] = slot as u32;
        }
        slots
    }

    /// Each sector's centre relative to `anchor_pc` [pc] (f64 difference, then f32), as the
    /// vec4s the star pass uploads (w = 0).
    pub fn centres_rel(&self, anchor_pc: DVec3) -> Vec<[f32; 4]> {
        centres_rel(&self.sectors, anchor_pc)
    }
}

/// The centres of `sectors` relative to `anchor_pc` [pc] (f64 difference, then f32), as the
/// vec4s the star pass uploads (w = 0).
pub(crate) fn centres_rel(sectors: &[IVec3], anchor_pc: DVec3) -> Vec<[f32; 4]> {
    sectors
        .iter()
        .map(|&k| {
            (sector_centre_pc(k) - anchor_pc)
                .as_vec3()
                .extend(0.0)
                .to_array()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stars3d::{REANCHOR_PC, pack_stars, star_rel_pos_pc};
    use glam::Vec3;
    use uni_core::splitmix64;
    use uni_data::Star3;

    #[test]
    fn sector_of_floors_every_axis() {
        assert_eq!(sector_of(DVec3::ZERO), IVec3::ZERO);
        assert_eq!(
            sector_of(DVec3::new(15.999, 16.0, -0.001)),
            IVec3::new(0, 1, -1)
        );
        assert_eq!(
            sector_of(DVec3::new(-16.0, -16.001, 33.0)),
            IVec3::new(-1, -2, 2)
        );
        assert_eq!(sector_of(DVec3::new(f64::NAN, 1e300, -1e300)).x, 0);
        assert_eq!(
            sector_centre_pc(IVec3::new(0, -1, 2)),
            DVec3::new(8.0, -8.0, 40.0)
        );
        // Every position is within 8 pc of its sector's centre on each axis.
        for p in [DVec3::new(3.0, -7.5, 1234.5), DVec3::new(-1e4, 0.25, 15.9)] {
            let off = p - sector_centre_pc(sector_of(p));
            assert!(off.abs().max_element() <= 8.0, "{p} → {off}");
        }
    }

    #[test]
    fn sector_table_groups_stars_contiguously() {
        let keys = [
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 5, 2),
            IVec3::new(1, 0, 0),
            IVec3::new(0, 0, 0),
            IVec3::new(-1, 5, 2),
            IVec3::new(1, 0, 0),
        ];
        let t = SectorTable::build(&keys);
        assert_eq!(
            t.sectors,
            vec![IVec3::new(-1, 5, 2), IVec3::ZERO, IVec3::new(1, 0, 0)]
        );
        // Stable: equal keys keep their input order.
        assert_eq!(t.order, vec![1, 4, 3, 0, 2, 5]);
        assert_eq!(t.sector_index, vec![0, 0, 1, 2, 2, 2]);
        for (slot, &i) in t.order.iter().enumerate() {
            assert_eq!(t.sectors[t.sector_index[slot] as usize], keys[i as usize]);
        }
        let slots = t.slots();
        for (i, &slot) in slots.iter().enumerate() {
            assert_eq!(t.order[slot as usize], i as u32);
        }
        let rel = t.centres_rel(DVec3::new(8.0, 8.0, 8.0));
        assert_eq!(rel.len(), 3);
        assert_eq!(rel[1], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(rel[2], [16.0, 0.0, 0.0, 0.0]);
        assert_eq!(SectorTable::build(&[]), SectorTable::default());
    }

    /// Uniform in [lo, hi) from a splitmix64 stream.
    fn uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
        *state = state.wrapping_add(1);
        let bits = splitmix64(*state) >> 11;
        lo + (hi - lo) * (bits as f64 / (1u64 << 53) as f64)
    }

    /// Review Focus 5: a camera 1 kpc from the SSB and a star 1 pc from the camera. The f32 path
    /// the GPU takes (anchor-relative sector centre − camera offset + the star's f32 offset) must
    /// point within 0.05 px of the exact f64 direction at 1080 lines and fov 60°. The anchor is
    /// the camera itself or up to `REANCHOR_PC` away (the worst case before a re-upload).
    #[test]
    fn sector_offsets_precision_at_1kpc() {
        let px_rad = 2.0 * (30f64.to_radians()).tan() / 1080.0;
        let limit = 0.05 * px_rad;
        let mut state = 0x5EC7_0B5E_u64;
        let mut worst = 0.0f64;
        for _ in 0..2000 {
            let unit = |s: &mut u64| {
                DVec3::new(
                    uniform(s, -1.0, 1.0),
                    uniform(s, -1.0, 1.0),
                    uniform(s, -1.0, 1.0),
                )
                .normalize()
            };
            let camera = unit(&mut state) * 1000.0;
            let anchor = camera + unit(&mut state) * uniform(&mut state, 0.0, REANCHOR_PC);
            // The catalogue stores positions as f32; the exact direction starts from that.
            let star_f32 = (camera + unit(&mut state)).as_vec3();
            let star = Star3::new(star_f32.to_array(), 1.0, [0.0; 3], 0.6, 1, 0);
            let packed = pack_stars(&[star]);
            let centre_rel = Vec3::from_slice(&packed.table.centres_rel(anchor)[0][..3]);
            let camera_rel = (camera - anchor).as_vec3();
            let p = star_rel_pos_pc(&packed.gpu[0], centre_rel - camera_rel, 0.0);
            let exact = star_f32.as_dvec3() - camera;
            let err = p.as_dvec3().angle_between(exact);
            worst = worst.max(err);
        }
        assert!(worst < limit, "worst {worst:.3e} rad vs {limit:.3e} rad");
        println!(
            "worst direction error {worst:.3e} rad = {:.2e} px",
            worst / px_rad
        );
    }

    /// The committed AT-HYG bake: the measured sector count the module docs quote, and every
    /// offset within 8 pc of its sector's centre.
    #[test]
    fn baked_catalogue_sectors() {
        let path = uni_data::data_dir().join("baked/athyg_m10.bin");
        let stars = uni_data::load_athyg(&path).expect("committed athyg_m10.bin");
        let packed = pack_stars(&stars);
        assert_eq!(packed.table.sectors.len(), 166_011);
        assert!(
            packed
                .gpu
                .iter()
                .all(|g| Vec3::from_slice(&g.offset_absmag[..3]).abs().max_element() <= 8.0)
        );
    }
}
