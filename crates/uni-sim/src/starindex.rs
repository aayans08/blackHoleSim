//! Nearest-star queries over the catalogue (M1 Task A4).
//!
//! Stars move linearly from their catalogue positions (`star_pos_pc`), so a grid built at the
//! catalogue epoch goes stale as time runs. Queries widen their search by the farthest any
//! gridded star can have moved (its speed × |years|); stars faster than
//! `GRID_MAX_SPEED_KMS` stay out of the grid and are always checked, so that one halo star
//! (or one bad parallax: AT-HYG has 5 stars above 1 000 km/s) does not widen every query.
//! Once that drift exceeds `MAX_DRIFT_PC` (2 × 10⁵ yr at 300 km/s), or a query has not
//! settled within `MAX_RINGS` rings of cells, the index scans every star instead: slower,
//! still exact.

use std::collections::HashMap;

use glam::{DVec3, IVec3};
use uni_core::{DAY, PC};
use uni_data::{FLAG_NO_VELOCITY, Star3};

/// Grid cell edge [pc].
const CELL_PC: f64 = 4.0;
/// Stars faster than this [km/s] are kept out of the grid and checked by every query.
/// 300 km/s is above the disk population and most of the halo near the Sun (the local escape
/// speed is ≈ 530 km/s; Piffl et al. 2014, A&A 562, A91), so few stars are left out.
const GRID_MAX_SPEED_KMS: f64 = 300.0;
/// Beyond this drift [pc] a query scans every star.
const MAX_DRIFT_PC: f64 = 64.0;
/// Beyond this many rings of cells around the query's cell a query scans every star.
const MAX_RINGS: i32 = 12;

/// 1 km/s over one Julian year (365.25 d of 86 400 s; IAU 1976) in parsecs (IAU 2012 B2):
/// 3.155 76 × 10¹⁰ m / 3.085 678 × 10¹⁶ m = 1.022 712 × 10⁻⁶ pc.
pub(crate) const PC_PER_YR_PER_KMS: f64 = 1.0e3 * 365.25 * DAY / PC;

/// A uniform grid of 4 pc cells over the catalogue stars.
pub struct StarIndex {
    /// Cell → (start, length) in `order`.
    cells: HashMap<IVec3, (u32, u32)>,
    /// Gridded star indices, sorted by cell.
    order: Vec<u32>,
    /// Stars checked by every query: faster than [`GRID_MAX_SPEED_KMS`], or with a position
    /// no cell can hold.
    loose: Vec<u32>,
    /// Largest speed among the gridded stars [pc/yr].
    grid_speed_pc_per_yr: f64,
}

impl StarIndex {
    pub fn new(stars: &[Star3]) -> Self {
        let mut keyed = Vec::with_capacity(stars.len());
        let mut loose = Vec::new();
        let mut grid_speed_pc_per_yr = 0.0_f64;
        let max_speed = GRID_MAX_SPEED_KMS * PC_PER_YR_PER_KMS;
        for (i, star) in stars.iter().enumerate() {
            let speed = velocity_pc_per_yr(star).length();
            let cell = cell_of(star_pos_pc(star, 0.0));
            match cell {
                Some(cell) if speed <= max_speed => {
                    grid_speed_pc_per_yr = grid_speed_pc_per_yr.max(speed);
                    keyed.push((cell.to_array(), i as u32));
                }
                _ => loose.push(i as u32),
            }
        }
        keyed.sort_unstable();
        let mut cells = HashMap::new();
        let mut start = 0;
        while start < keyed.len() {
            let cell = keyed[start].0;
            let len = keyed[start..].iter().take_while(|k| k.0 == cell).count();
            cells.insert(IVec3::from_array(cell), (start as u32, len as u32));
            start += len;
        }
        StarIndex {
            cells,
            order: keyed.into_iter().map(|(_, i)| i).collect(),
            loose,
            grid_speed_pc_per_yr,
        }
    }

    /// The `k` stars nearest `p_pc` [pc, ICRS] at `years` after the catalogue epoch, as
    /// (index, distance [pc]), nearest first. `stars` is the slice the index was built from.
    pub fn nearest(&self, stars: &[Star3], p_pc: DVec3, years: f64, k: usize) -> Vec<(u32, f64)> {
        self.search(stars, p_pc, years, k, f64::INFINITY)
    }

    /// Like [`StarIndex::nearest`], but only stars within `max_pc`.
    pub(crate) fn within(
        &self,
        stars: &[Star3],
        p_pc: DVec3,
        years: f64,
        max_pc: f64,
        k: usize,
    ) -> Vec<(u32, f64)> {
        self.search(stars, p_pc, years, k, max_pc)
    }

    fn search(
        &self,
        stars: &[Star3],
        p: DVec3,
        years: f64,
        k: usize,
        max_pc: f64,
    ) -> Vec<(u32, f64)> {
        let mut best = Best::new(k, max_pc);
        if k == 0 || stars.is_empty() {
            return best.into_vec();
        }
        let offer = |best: &mut Best, i: u32| {
            if let Some(star) = stars.get(i as usize) {
                best.offer(i, star_pos_pc(star, years).distance(p));
            }
        };
        let drift = self.grid_speed_pc_per_yr * years.abs();
        let centre = if drift <= MAX_DRIFT_PC { cell_of(p) } else { None };
        if let Some(centre) = centre {
            for &i in &self.loose {
                offer(&mut best, i);
            }
            for r in 0..=MAX_RINGS {
                for_each_ring_cell(centre, r, |cell| {
                    if let Some(&(start, len)) = self.cells.get(&cell) {
                        for &i in &self.order[start as usize..(start + len) as usize] {
                            offer(&mut best, i);
                        }
                    }
                });
                // A star outside rings 0..=r sat ≥ r cells from p at the catalogue epoch
                // (p is inside the centre cell) and has moved at most `drift` since.
                if f64::from(r) * CELL_PC - drift >= best.cutoff() {
                    return best.into_vec();
                }
            }
            best = Best::new(k, max_pc);
        }
        for i in 0..stars.len() as u32 {
            offer(&mut best, i);
        }
        best.into_vec()
    }
}

/// The `k` best (index, distance) pairs seen so far, nearest first.
struct Best {
    k: usize,
    max: f64,
    items: Vec<(u32, f64)>,
}

impl Best {
    fn new(k: usize, max: f64) -> Self {
        Best {
            k,
            max,
            items: Vec::with_capacity(k.min(64)),
        }
    }

    fn offer(&mut self, i: u32, d: f64) {
        if !(d <= self.max) || (self.items.len() == self.k && d >= self.cutoff()) {
            return;
        }
        let at = self.items.partition_point(|&(_, e)| e <= d);
        self.items.insert(at, (i, d));
        self.items.truncate(self.k);
    }

    /// No star at or beyond this distance can enter.
    fn cutoff(&self) -> f64 {
        if self.items.len() == self.k {
            self.items.last().map_or(self.max, |&(_, d)| d.min(self.max))
        } else {
            self.max
        }
    }

    fn into_vec(self) -> Vec<(u32, f64)> {
        self.items
    }
}

/// The grid cell holding `p` [pc], `None` for a position no `i32` cell can hold.
fn cell_of(p: DVec3) -> Option<IVec3> {
    let c = (p / CELL_PC).floor();
    let limit = f64::from(i32::MAX);
    (c.is_finite() && c.abs().max_element() < limit).then(|| c.as_ivec3())
}

/// Calls `f` for every cell at Chebyshev distance exactly `r` from `centre`.
fn for_each_ring_cell(centre: IVec3, r: i32, mut f: impl FnMut(IVec3)) {
    for dx in -r..=r {
        for dy in -r..=r {
            if dx.abs() == r || dy.abs() == r {
                for dz in -r..=r {
                    f(centre + IVec3::new(dx, dy, dz));
                }
            } else if r > 0 {
                f(centre + IVec3::new(dx, dy, -r));
                f(centre + IVec3::new(dx, dy, r));
            } else {
                f(centre);
            }
        }
    }
}

/// A star's catalogue velocity [pc/yr], ICRS; zero for a star flagged "no velocity" (A7).
pub(crate) fn velocity_pc_per_yr(s: &Star3) -> DVec3 {
    if s.flags() & FLAG_NO_VELOCITY != 0 {
        return DVec3::ZERO;
    }
    s.velocity_kms().as_dvec3() * PC_PER_YR_PER_KMS
}

/// A star's position [pc, ICRS] `years` after the catalogue epoch: linear motion with its
/// catalogue velocity, in f64. A star flagged "no velocity" stays at its catalogue position; a
/// star without a radial velocity moves tangentially only, because AT-HYG computed its velocity
/// with no radial part (A7).
pub(crate) fn star_pos_pc(s: &Star3, years: f64) -> DVec3 {
    DVec3::from_array(s.pos_pc.map(f64::from)) + velocity_pc_per_yr(s) * years
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_core::splitmix64;

    /// Uniform in [lo, hi) from a splitmix64 stream.
    fn uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let bits = splitmix64(*state) >> 11;
        lo + (hi - lo) * (bits as f64 / (1u64 << 53) as f64)
    }

    /// 1000 stars in a 60 pc box, most at disk speeds (≤ 80 km/s), every 50th a halo-fast
    /// one (≤ 1500 km/s), a few with no velocity.
    fn random_stars(seed: u64) -> Vec<Star3> {
        let mut s = seed;
        (0..1000)
            .map(|i| {
                let pos = [0; 3].map(|_| uniform(&mut s, -30.0, 30.0) as f32);
                let vmax = if i % 50 == 0 { 1500.0 } else { 80.0 };
                let vel = if i % 97 == 0 {
                    [0.0; 3]
                } else {
                    [0; 3].map(|_| uniform(&mut s, -vmax, vmax) as f32)
                };
                Star3::new(pos, 5.0, vel, 0.65, i, 0)
            })
            .collect()
    }

    fn brute_force(stars: &[Star3], p: DVec3, years: f64, k: usize) -> Vec<(u32, f64)> {
        let mut all: Vec<(u32, f64)> = stars
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, star_pos_pc(s, years).distance(p)))
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        all.truncate(k);
        all
    }

    #[test]
    fn star_index_nearest() {
        let stars = random_stars(7);
        let index = StarIndex::new(&stars);
        let mut s = 99;
        // Years span the grid's easy case (0), drifts of a few cells, and a drift so large that
        // the index falls back to a brute-force scan.
        let years = [0.0, 1.0e4, -3.0e4, 2.0e5, 5.0e6];
        for q in 0..50 {
            let p = DVec3::new(
                uniform(&mut s, -40.0, 40.0),
                uniform(&mut s, -40.0, 40.0),
                uniform(&mut s, -40.0, 40.0),
            );
            let y = years[q % years.len()];
            let k = 1 + q % 9;
            let got = index.nearest(&stars, p, y, k);
            let want = brute_force(&stars, p, y, k);
            assert_eq!(got.len(), want.len(), "query {q}");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.0, w.0, "query {q} at {p} after {y} yr: {got:?} vs {want:?}");
                assert!((g.1 - w.1).abs() < 1e-12, "query {q}");
            }
        }
    }

    #[test]
    fn nearest_handles_empty_and_oversized_requests() {
        let empty = StarIndex::new(&[]);
        assert!(empty.nearest(&[], DVec3::ZERO, 0.0, 3).is_empty());
        let stars = random_stars(3);
        let index = StarIndex::new(&stars);
        assert!(index.nearest(&stars, DVec3::ZERO, 0.0, 0).is_empty());
        let all = index.nearest(&stars, DVec3::splat(1.0e4), 0.0, 5000);
        assert_eq!(all.len(), stars.len(), "k larger than the catalogue returns every star");
        assert!(all.windows(2).all(|w| w[0].1 <= w[1].1), "nearest first");
    }

    #[test]
    fn star_moves_linearly_with_its_catalogue_velocity() {
        // 1 km/s for one Julian year: 31 557 600 km = 1.022 712e-6 pc (IAU 2012 B2 parsec).
        let star = Star3::new([1.0, 2.0, 3.0], 4.0, [1.0, 0.0, -2.0], 0.6, 0, 0);
        let p = star_pos_pc(&star, 1.0e6);
        let per_year = 365.25 * 86_400.0 * 1.0e3 / uni_core::PC;
        assert!((p.x - (1.0 + 1.0e6 * per_year)).abs() < 1e-9, "{p}");
        assert!((p.z - (3.0 - 2.0e6 * per_year)).abs() < 1e-9, "{p}");
        let fixed = Star3::new(
            [1.0, 2.0, 3.0],
            4.0,
            [5.0, 5.0, 5.0],
            0.6,
            0,
            uni_data::FLAG_NO_VELOCITY,
        );
        assert_eq!(star_pos_pc(&fixed, 1.0e6), DVec3::new(1.0, 2.0, 3.0));
    }
}
