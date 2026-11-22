//! Star names for the AT-HYG stars (`athyg_names.ron`, CC BY-SA 4.0) (M1 Task A7).
//!
//! The file is a RON list of [`StarName`], one entry per star of `athyg_m10.bin` that has a
//! proper name (IAU list, via HYG) or a Bayer or Flamsteed designation, in increasing `index`
//! order. `uni-bake athyg` writes it alongside the star file.

use std::path::Path;

use crate::DataError;

/// Names of one star.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StarName {
    /// Index of the star in the baked `athyg_m10.bin`.
    pub index: u32,
    pub proper: Option<String>,
    /// Display-ready designation with the constellation: the Bayer letter in Greek (with a
    /// superscript component number, e.g. "α¹ Cen") when the star has one, otherwise the
    /// Flamsteed number ("61 Cyg").
    pub bayer_flamsteed: Option<String>,
    /// Constellation abbreviation.
    pub con: Option<String>,
    /// Hipparcos number.
    pub hip: Option<u32>,
    /// Spectral type.
    pub spect: Option<String>,
}

/// Every named star.
#[derive(Clone, Debug, Default)]
pub struct StarNames {
    pub entries: Vec<StarName>,
}

impl StarNames {
    /// The names of the star at `index` in `athyg_m10.bin`, if it has any.
    pub fn get(&self, index: u32) -> Option<&StarName> {
        let at = self
            .entries
            .binary_search_by_key(&index, |name| name.index)
            .ok()?;
        self.entries.get(at)
    }
}

/// Reads `athyg_names.ron`. Missing → [`DataError::NotFound`]; unparsable, or indices not
/// strictly increasing → [`DataError::BadFormat`].
pub fn load_star_names(path: &Path) -> Result<StarNames, DataError> {
    let text = std::fs::read_to_string(path).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => DataError::NotFound(path.to_path_buf()),
        _ => DataError::Io(err),
    })?;
    let entries: Vec<StarName> = ron::from_str(&text)
        .map_err(|err| DataError::BadFormat(format!("{}: {err}", path.display())))?;
    if let Some(pair) = entries
        .windows(2)
        .find(|pair| pair[0].index >= pair[1].index)
    {
        return Err(DataError::BadFormat(format!(
            "{}: star index {} follows {}; indices must be strictly increasing",
            path.display(),
            pair[1].index,
            pair[0].index
        )));
    }
    Ok(StarNames { entries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-names-{}-{}", std::process::id(), name))
    }

    fn name(index: u32, proper: Option<&str>, bayer_flamsteed: Option<&str>) -> StarName {
        StarName {
            index,
            proper: proper.map(str::to_owned),
            bayer_flamsteed: bayer_flamsteed.map(str::to_owned),
            con: Some("CMa".to_owned()),
            hip: Some(32349),
            spect: Some("A0m...".to_owned()),
        }
    }

    fn write_ron(path: &Path, entries: &[StarName]) {
        let text = ron::ser::to_string_pretty(
            entries,
            ron::ser::PrettyConfig::new().compact_structs(true),
        )
        .expect("serialise");
        std::fs::write(path, text).expect("write");
    }

    #[test]
    fn names_roundtrip_and_lookup() {
        let entries = vec![
            name(3, None, Some("61 Cyg")),
            name(17, Some("Sirius"), Some("α CMa")),
            name(40, Some("Proxima Centauri"), None),
        ];
        let path = temp_path("roundtrip.ron");
        write_ron(&path, &entries);
        let loaded = load_star_names(&path);
        let _ = std::fs::remove_file(&path);
        let names = loaded.expect("load");
        assert_eq!(names.entries.len(), 3);
        let sirius = names.get(17).expect("index 17 is named");
        assert_eq!(sirius.proper.as_deref(), Some("Sirius"));
        assert_eq!(sirius.bayer_flamsteed.as_deref(), Some("α CMa"));
        assert_eq!(sirius.hip, Some(32349));
        assert_eq!(
            names.get(3).and_then(|n| n.bayer_flamsteed.as_deref()),
            Some("61 Cyg")
        );
        assert!(names.get(4).is_none());
        assert!(names.get(41).is_none());
        assert!(StarNames::default().get(0).is_none());
    }

    #[test]
    fn committed_names_match_named_flags() {
        // data/baked/athyg_names.ron and athyg_m10.bin are committed together (M1 Task A7).
        let dir = crate::data_dir_from(None).join("baked");
        let names =
            load_star_names(&dir.join("athyg_names.ron")).expect("committed athyg_names.ron");
        let stars = crate::load_athyg(&dir.join("athyg_m10.bin")).expect("committed athyg_m10.bin");
        assert_eq!(names.entries.len(), 3505);
        for (index, star) in stars.iter().enumerate() {
            assert_eq!(
                names.get(index as u32).is_some(),
                star.is_named(),
                "star {index}"
            );
        }
        let vega = names
            .entries
            .iter()
            .find(|n| n.proper.as_deref() == Some("Vega"))
            .expect("Vega");
        assert_eq!(vega.bayer_flamsteed.as_deref(), Some("α Lyr"));
        assert_eq!(vega.hip, Some(91262));
    }

    #[test]
    fn load_missing_names_is_not_found() {
        assert!(matches!(
            load_star_names(Path::new("/nonexistent/athyg_names.ron")),
            Err(DataError::NotFound(_))
        ));
    }

    #[test]
    fn load_garbage_names_is_bad_format() {
        let path = temp_path("garbage.ron");
        std::fs::write(&path, "[(index: 1, proper: Some(\"Vega\")").expect("write");
        let loaded = load_star_names(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(loaded, Err(DataError::BadFormat(_))), "{loaded:?}");
    }

    #[test]
    fn load_unsorted_names_is_bad_format() {
        // `get` binary-searches, so the file must list each index once, in increasing order.
        for indices in [[5, 2], [5, 5]] {
            let path = temp_path("unsorted.ron");
            write_ron(
                &path,
                &[
                    name(indices[0], Some("A"), None),
                    name(indices[1], Some("B"), None),
                ],
            );
            let loaded = load_star_names(&path);
            let _ = std::fs::remove_file(&path);
            assert!(
                matches!(loaded, Err(DataError::BadFormat(_))),
                "{indices:?}: {loaded:?}"
            );
        }
    }
}
