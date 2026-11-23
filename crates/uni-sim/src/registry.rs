//! Every named object the world knows (M1 Task A4).

use std::collections::HashMap;

use glam::DVec3;
use uni_core::FrameId;
use uni_physics::photometry::Brdf;

/// Index of an object in the world's registry.
pub type ObjectId = u32;

/// What kind of object an entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    Sun,
    Planet,
    DwarfPlanet,
    Moon,
    Asteroid,
    Comet,
    Star,
    ExoHost,
    BlackHole,
}

/// One registry entry.
#[derive(Clone, Debug)]
pub struct ObjectInfo {
    pub id: ObjectId,
    /// Unique case-insensitively.
    pub name: String,
    pub kind: ObjectKind,
    /// NAIF id, for Solar System bodies.
    pub naif: Option<i32>,
    /// Index into the star data, for stars and exoplanet hosts.
    pub catalog_index: Option<u32>,
    pub parent: Option<ObjectId>,
    /// Triaxial radii [m], a ≥ b ≥ c.
    pub radii_m: DVec3,
    /// GM [m³/s²].
    pub gm_m3s2: f64,
    /// The object's own frame, if it has one.
    pub frame: Option<FrameId>,
    /// Hill radius about its parent [m].
    pub hill_radius_m: f64,
    /// Radius of the top of the atmosphere the renderer draws [m], from the body's centre (not
    /// an altitude): uni-atmo's `AtmosphereParams::top_radius_m` for Venus, Earth, Mars and
    /// Titan (Task C0; uni-app's `atmosphere_top_matches_atmo_presets` keeps the two equal).
    /// `None` for every other object. Cruise drops out on crossing it and goto stands off
    /// outside it (C3).
    pub atmosphere_top_m: Option<f64>,
}

/// Where an object's state comes from (parallel to the registry).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    /// The body source (the ephemeris): the Sun, planets, Pluto, moons.
    Ephem,
    /// Index into the world's comet orbits.
    Comet(usize),
    /// Index into the world's asteroid orbits.
    Asteroid(usize),
    /// A catalogue star or exoplanet host: `ObjectInfo::catalog_index` into the star data.
    Star,
    /// Index into the world's lenses.
    BlackHole(usize),
}

/// What the snapshot needs to draw an object (parallel to the registry).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Physical {
    pub source: Source,
    /// Depth of the object's frame below the root (Sun 1, planets 2, moons 3); 0 without one.
    pub depth: u32,
    /// Reflectance model of the surface: the one the bodies pass uses (plan B1).
    pub brdf: Brdf,
    /// V geometric albedo p_V.
    pub geometric_albedo: f64,
    /// Volumetric mean radius (a·b·c)^⅓ [m]: the size behind a body's disk-integrated flux.
    pub mean_radius_m: f64,
    /// An atmosphere the renderer models (uni-atmo's presets: Venus, Earth, Mars, Titan).
    pub has_atmosphere: bool,
}

/// NAIF ids of the bodies with an atmosphere preset in uni-atmo (plan B3: `preset_for` 299,
/// 399, 499, 606).
const ATMOSPHERE_BODIES: [i32; 4] = [299, 399, 499, 606];
/// NAIF ids of the giant planets.
const GIANTS: [i32; 4] = [599, 699, 799, 899];

/// Top of each modelled atmosphere [m from the body's centre], by NAIF id (Task C0). uni-sim
/// cannot depend on uni-atmo (it would pull in wgpu), so these copy its presets
/// (`uni_atmo::params`), and uni-app's test `atmosphere_top_matches_atmo_presets` checks that
/// they still equal `uni_atmo::preset_for(naif).top_radius_m`:
/// - Venus: the cloud tops, 6 051.8 km (WGCCRE 2015 mean radius) + 70 km (`VENUS_CLOUD_TOP_M`),
///   plus the preset's 60 km shell;
/// - Earth: 6 371 km (mean radius) + 100 km (Bruneton 2017 / Hillaire 2020 defaults);
/// - Mars: 3 389.5 km (WGCCRE 2015 mean radius) + 80 km;
/// - Titan: 2 574.73 km (WGCCRE 2015 mean radius) + 600 km, which holds the detached haze layer
///   at 500 km (Seignovert et al. 2017).
const ATMOSPHERE_TOP_M: [(i32, f64); 4] = [
    (299, 6_051_800.0 + 70_000.0 + 60_000.0),
    (399, 6_371_000.0 + 100_000.0),
    (499, 3_389_500.0 + 80_000.0),
    (606, 2_574_730.0 + 600_000.0),
];

/// Whether the renderer draws an atmosphere for body `naif`.
pub(crate) fn has_atmosphere(naif: i32) -> bool {
    ATMOSPHERE_BODIES.contains(&naif)
}

/// [`ObjectInfo::atmosphere_top_m`] of body `naif`: the top of its modelled atmosphere [m from
/// the centre], `None` without one.
pub(crate) fn atmosphere_top_m(naif: i32) -> Option<f64> {
    ATMOSPHERE_TOP_M
        .iter()
        .find(|(n, _)| *n == naif)
        .map(|&(_, top)| top)
}

/// The BRDF the bodies pass uses for `naif` (plan B1): Lambert for bodies with atmospheres and
/// the giant planets, lunar-Lambert with L = 0.9 for airless bodies (McEwen 1991, Icarus 92,
/// 298). `None` (comets, asteroids) is airless.
pub(crate) fn brdf_for(naif: Option<i32>) -> Brdf {
    match naif {
        Some(n) if has_atmosphere(n) || GIANTS.contains(&n) => Brdf::Lambert,
        _ => Brdf::LunarLambert { l: 0.9 },
    }
}

/// Case-insensitively unique names, plus aliases that `find` also accepts.
#[derive(Default)]
pub(crate) struct NameTable {
    by_key: HashMap<String, ObjectId>,
}

fn key(name: &str) -> String {
    name.trim().to_lowercase()
}

impl NameTable {
    /// The first candidate no object holds yet, else the first candidate with " [id]" added.
    /// Empty candidates are skipped; with none left the name is "Object <id>".
    pub fn unique(&self, candidates: &[&str], id: ObjectId) -> String {
        let candidates: Vec<&str> = candidates
            .iter()
            .map(|c| c.trim())
            .filter(|c| !c.is_empty())
            .collect();
        if let Some(free) = candidates
            .iter()
            .find(|c| !self.by_key.contains_key(&key(c)))
        {
            return (*free).to_owned();
        }
        match candidates.first() {
            Some(first) => format!("{first} [{id}]"),
            None => format!("Object {id}"),
        }
    }

    /// Registers `name` for `id` unless another object holds it already; returns whether it
    /// did.
    pub fn insert(&mut self, name: &str, id: ObjectId) -> bool {
        let k = key(name);
        if k.is_empty() || self.by_key.contains_key(&k) {
            return false;
        }
        self.by_key.insert(k, id);
        true
    }

    pub fn get(&self, name: &str) -> Option<ObjectId> {
        self.by_key.get(&key(name)).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_stay_unique_ignoring_case() {
        let mut names = NameTable::default();
        assert_eq!(names.unique(&["Vesta", "4 Vesta (A807 FA)"], 0), "Vesta");
        assert!(names.insert("Vesta", 0));
        assert_eq!(
            names.unique(&["VESTA", "4 Vesta (A807 FA)"], 1),
            "4 Vesta (A807 FA)"
        );
        assert_eq!(names.unique(&["vesta"], 2), "vesta [2]");
        assert_eq!(names.unique(&["", "  "], 3), "Object 3");
        assert!(
            !names.insert("vESTA", 1),
            "an alias never takes a held name"
        );
        assert_eq!(names.get(" vesta "), Some(0));
        assert_eq!(names.get("Ceres"), None);
    }

    #[test]
    fn brdf_follows_the_bodies_pass() {
        assert_eq!(brdf_for(Some(399)), Brdf::Lambert);
        assert_eq!(brdf_for(Some(699)), Brdf::Lambert);
        assert_eq!(brdf_for(Some(606)), Brdf::Lambert);
        assert_eq!(brdf_for(Some(301)), Brdf::LunarLambert { l: 0.9 });
        assert_eq!(brdf_for(Some(199)), Brdf::LunarLambert { l: 0.9 });
        assert_eq!(brdf_for(None), Brdf::LunarLambert { l: 0.9 });
        assert!(has_atmosphere(299) && !has_atmosphere(301));
    }

    /// Every body with an atmosphere has its top, above its solid radius; no other body has one.
    #[test]
    fn atmosphere_tops_cover_the_atmosphere_bodies() {
        for naif in ATMOSPHERE_BODIES {
            let top = atmosphere_top_m(naif).expect("a top for every atmosphere body");
            let solid = uni_physics::bodies::by_naif(naif).map_or(0.0, |b| b.radii_m.x);
            assert!(top > solid, "{naif}: top {top} m, radius {solid} m");
        }
        assert_eq!(ATMOSPHERE_TOP_M.len(), ATMOSPHERE_BODIES.len());
        assert_eq!(atmosphere_top_m(301), None);
        assert_eq!(atmosphere_top_m(10), None);
    }
}
