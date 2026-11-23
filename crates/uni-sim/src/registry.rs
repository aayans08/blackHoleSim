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

/// Whether the renderer draws an atmosphere for body `naif`.
pub(crate) fn has_atmosphere(naif: i32) -> bool {
    ATMOSPHERE_BODIES.contains(&naif)
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
        if let Some(free) = candidates.iter().find(|c| !self.by_key.contains_key(&key(c))) {
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
        assert_eq!(names.unique(&["VESTA", "4 Vesta (A807 FA)"], 1), "4 Vesta (A807 FA)");
        assert_eq!(names.unique(&["vesta"], 2), "vesta [2]");
        assert_eq!(names.unique(&["", "  "], 3), "Object 3");
        assert!(!names.insert("vESTA", 1), "an alias never takes a held name");
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
}
