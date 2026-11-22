//! Named camera views saved to and loaded from RON files (M0 Task 6).
//!
//! A bookmark file is a RON list of [`Bookmark`]s. Hand-written files (such as the capture
//! sheet `bookmarks/m0.ron`) may carry `//` comments and leave out the optional fields; a
//! present one is written `Some(x)` unless the file starts with `#![enable(implicit_some)]`.

use std::path::Path;

use anyhow::Context;
use ron::ser::PrettyConfig;
use uni_core::CameraView;

/// A named camera view plus the optional settings to restore with it; `None` keeps the
/// current value.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Bookmark {
    pub name: String,
    pub view: CameraView,
    /// Simulation epoch, `"YYYY-MM-DDThh:mm:ss[.fff]Z"` (see `uni_core::Epoch::parse_iso_utc`).
    pub epoch_utc: Option<String>,
    /// Scene exposure [EV100].
    pub scene_ev100: Option<f32>,
    /// Star-layer exposure [EV100].
    pub star_ev100: Option<f32>,
    /// Gravitational-lens scale (the HUD's lens-scale slider).
    pub lens_scale: Option<f32>,
}

/// Reads a RON list of bookmarks. Errors name the file.
pub fn load_bookmarks(path: &Path) -> anyhow::Result<Vec<Bookmark>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading bookmarks from {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing bookmarks in {}", path.display()))
}

/// Writes `b` as a pretty-printed RON list, replacing the file. Floats are written in their
/// shortest round-trip form, so loading gives back the same bits.
pub fn save_bookmarks(path: &Path, b: &[Bookmark]) -> anyhow::Result<()> {
    let text = ron::ser::to_string_pretty(b, PrettyConfig::new())
        .context("serialising bookmarks to RON")?;
    std::fs::write(path, text).with_context(|| format!("writing bookmarks to {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, DVec3, I64Vec3};
    use std::path::PathBuf;
    use uni_core::{PC, UniversePos};

    /// A file under the system temp dir, removed when dropped (even if the test panics).
    struct TempFile(PathBuf);

    impl TempFile {
        fn new(test_name: &str) -> Self {
            let name = format!("uni-nav-{}-{}", std::process::id(), test_name);
            TempFile(std::env::temp_dir().join(name))
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn sample_bookmarks() -> Vec<Bookmark> {
        vec![
            Bookmark {
                name: "start".to_owned(),
                view: CameraView {
                    pos: UniversePos {
                        cell: I64Vec3::new(8, -1, 0),
                        local: DVec3::new(1234.5678, -0.1, 3.0e12),
                    },
                    orient: DQuat::from_rotation_y(0.3) * DQuat::from_rotation_x(-0.2),
                    fov_y_rad: 60_f64.to_radians(),
                    velocity_mps: DVec3::ZERO,
                },
                epoch_utc: Some("2026-11-20T00:00:00Z".to_owned()),
                scene_ev100: Some(3.0),
                star_ev100: Some(-2.5),
                lens_scale: Some(1.0),
            },
            Bookmark {
                name: "far away".to_owned(),
                view: CameraView {
                    pos: UniversePos::from_meters(DVec3::new(1e9 * PC, -PC / 3.0, 0.1)),
                    orient: DQuat::from_axis_angle(DVec3::new(1.0, 2.0, 3.0).normalize(), 2.0),
                    fov_y_rad: 2_f64.to_radians(),
                    velocity_mps: DVec3::new(0.0, -3.0e4, 1.25),
                },
                epoch_utc: None,
                scene_ev100: None,
                star_ev100: None,
                lens_scale: None,
            },
        ]
    }

    #[test]
    fn bookmark_ron_roundtrip() {
        let file = TempFile::new("bookmark_ron_roundtrip");
        let saved = sample_bookmarks();
        save_bookmarks(&file.0, &saved).unwrap();
        let loaded = load_bookmarks(&file.0).unwrap();
        assert_eq!(loaded.len(), saved.len());
        for (l, s) in loaded.iter().zip(&saved) {
            assert_eq!(l.name, s.name);
            assert_eq!(l.view, s.view);
        }
        // Every field survives bit-exactly, not just name and view.
        assert_eq!(loaded, saved);
    }

    #[test]
    fn optional_fields_may_be_omitted() {
        // Hand-written capture sheets (bookmarks/m0.ron) list only what they need, with comments.
        let file = TempFile::new("optional_fields_may_be_omitted");
        let text = r#"[
            // At the SSB, looking along −Z.
            (
                name: "origin",
                view: (
                    pos: (cell: (0, 0, 0), local: (0.0, 0.0, 0.0)),
                    orient: (0.0, 0.0, 0.0, 1.0),
                    fov_y_rad: 1.0,
                ),
                lens_scale: Some(1.0),
            ),
        ]"#;
        std::fs::write(&file.0, text).unwrap();
        let loaded = load_bookmarks(&file.0).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "origin");
        assert_eq!(loaded[0].view.orient, DQuat::IDENTITY);
        assert_eq!(loaded[0].epoch_utc, None);
        assert_eq!(loaded[0].scene_ev100, None);
        assert_eq!(loaded[0].lens_scale, Some(1.0));
    }

    #[test]
    fn load_errors_name_the_file() {
        let missing = TempFile::new("load_errors_name_the_file-missing");
        let err = load_bookmarks(&missing.0).unwrap_err();
        assert!(
            format!("{err:#}").contains(&*missing.0.to_string_lossy()),
            "{err:#}"
        );

        let corrupt = TempFile::new("load_errors_name_the_file-corrupt");
        std::fs::write(&corrupt.0, "[(name: \"x\", view: oops)]").unwrap();
        let err = load_bookmarks(&corrupt.0).unwrap_err();
        assert!(
            format!("{err:#}").contains(&*corrupt.0.to_string_lossy()),
            "{err:#}"
        );
    }
}
