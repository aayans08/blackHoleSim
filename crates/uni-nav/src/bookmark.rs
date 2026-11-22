//! Named camera views saved to and loaded from RON files (M0 Task 6).
//!
//! A bookmark file is a RON list of [`Bookmark`]s. Hand-written files (such as the capture
//! sheet `bookmarks/m0.ron`) may carry `//` comments and leave out the optional fields; a
//! present one is written `Some(x)` unless the file starts with `#![enable(implicit_some)]`.
//!
//! Bookmark v2 (M1): the fields after `lens_scale` are optional too, so every M0 file loads
//! unchanged. A0b added them; Task A6 resolves them (`anchor`, `look_at`, `up`) in
//! [`crate::resolve_bookmark`], and the app applies the rest. [`save_bookmarks`] leaves unset
//! fields out, so a saved file (F12) lists only what it sets, like the hand-written ones.

use std::path::Path;

use anyhow::Context;
use glam::DVec3;
use ron::ser::PrettyConfig;
use uni_core::CameraView;

/// A named camera view plus the optional settings to restore with it; `None` keeps the
/// current value.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Bookmark {
    pub name: String,
    pub view: CameraView,
    /// Simulation epoch, `"YYYY-MM-DDThh:mm:ss[.fff]Z"` (see `uni_core::Epoch::parse_iso_utc`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch_utc: Option<String>,
    /// Scene exposure [EV100].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_ev100: Option<f32>,
    /// Star-layer exposure [EV100].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub star_ev100: Option<f32>,
    /// Gravitational-lens scale (the HUD's lens-scale slider).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lens_scale: Option<f32>,
    /// Object whose centre `view.pos` is measured from (cell 0, metres, root axes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    /// Object to face; overrides `view.orient` (−Z toward it, +Y as close as possible to `up`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look_at: Option<String>,
    /// Up hint for `look_at`, root axes; `None` → ICRS +Z.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up: Option<DVec3>,
    /// Simulation time rate [sim s per real s].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_rate: Option<f64>,
    /// Auto or manual exposure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<ExposureChoice>,
    /// Whether eye-adapted (dual-exposure) stars are on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eye_adapted: Option<bool>,
    /// Object selected on load (its info card shows with `--hud`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Environment seen through a lens; `None` → [`LensEnvChoice::Band`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lens_env: Option<LensEnvChoice>,
    /// Whether cruise mode is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cruise: Option<bool>,
    /// Whether the "Highlight asteroid belt" toggle is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlight_belt: Option<bool>,
}

/// How the scene exposure is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExposureChoice {
    /// The histogram meter sets it (B6).
    Auto,
    /// The bookmark's or the HUD's EV100.
    Manual,
}

/// What a traced lens shows behind the hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LensEnvChoice {
    /// The Milky Way band environment.
    Band,
    /// The M0 debug grid (labelled; ruling 14).
    DebugGrid,
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
                anchor: None,
                look_at: None,
                up: None,
                time_rate: None,
                exposure: None,
                eye_adapted: None,
                target: None,
                lens_env: None,
                cruise: None,
                highlight_belt: None,
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
                anchor: None,
                look_at: None,
                up: None,
                time_rate: None,
                exposure: None,
                eye_adapted: None,
                target: None,
                lens_env: None,
                cruise: None,
                highlight_belt: None,
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
        // Bookmark v2 fields (M1) are all optional too, so M0 files load unchanged.
        let b = &loaded[0];
        assert_eq!(
            (&b.anchor, &b.look_at, b.up, b.time_rate),
            (&None, &None, None, None)
        );
        assert_eq!((b.exposure, b.eye_adapted, &b.target), (None, None, &None));
        assert_eq!((b.lens_env, b.cruise, b.highlight_belt), (None, None, None));
    }

    /// A file in the workspace's `bookmarks/` directory.
    fn workspace_bookmarks(file: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bookmarks")
            .join(file)
    }

    #[test]
    fn m0_bookmarks_still_parse() {
        let marks = load_bookmarks(&workspace_bookmarks("m0.ron")).unwrap();
        let names: Vec<&str> = marks.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "start",
                "sky-orion",
                "sky-galactic-center",
                "sky-zoom-sirius",
                "bh-shadow-30M",
                "bh-oblique-15M"
            ]
        );
        for b in &marks {
            assert_eq!(
                (&b.anchor, &b.look_at, b.up, b.time_rate),
                (&None, &None, None, None),
                "{}",
                b.name
            );
            assert_eq!(
                (b.exposure, b.eye_adapted, &b.target),
                (None, None, &None),
                "{}",
                b.name
            );
            assert_eq!(
                (b.lens_env, b.cruise, b.highlight_belt),
                (None, None, None),
                "{}",
                b.name
            );
        }
    }

    #[test]
    fn bookmark_v2_ron_roundtrip() {
        // Every v2 field set, each enum variant used once.
        let file = TempFile::new("bookmark_v2_ron_roundtrip");
        let mut saved = sample_bookmarks();
        let mars = &mut saved[0];
        mars.anchor = Some("Mars".to_owned());
        mars.look_at = Some("Phobos".to_owned());
        mars.up = Some(DVec3::new(0.1, -0.2, 0.97));
        mars.time_rate = Some(3600.0);
        mars.exposure = Some(ExposureChoice::Manual);
        mars.eye_adapted = Some(false);
        mars.target = Some("Deimos".to_owned());
        mars.lens_env = Some(LensEnvChoice::DebugGrid);
        mars.cruise = Some(true);
        mars.highlight_belt = Some(false);
        let far = &mut saved[1];
        far.anchor = Some("Sun".to_owned());
        far.look_at = Some("TestBH-1".to_owned());
        far.up = Some(DVec3::Y);
        far.time_rate = Some(-1e12);
        far.exposure = Some(ExposureChoice::Auto);
        far.eye_adapted = Some(true);
        far.target = Some("TestBH-1".to_owned());
        far.lens_env = Some(LensEnvChoice::Band);
        far.cruise = Some(false);
        far.highlight_belt = Some(true);

        save_bookmarks(&file.0, &saved).unwrap();
        assert_eq!(load_bookmarks(&file.0).unwrap(), saved);
    }

    #[test]
    fn saved_file_omits_unset_fields() {
        // F12 leaves `lens_scale` out (review m2); a saved file lists only the fields that are
        // set, so it reads like the hand-written capture sheets.
        let file = TempFile::new("saved_file_omits_unset_fields");
        let saved = vec![sample_bookmarks().remove(1)];
        save_bookmarks(&file.0, &saved).unwrap();
        let text = std::fs::read_to_string(&file.0).unwrap();
        assert!(!text.contains("None"), "{text}");
        assert!(!text.contains("lens_scale"), "{text}");
        assert!(!text.contains("anchor"), "{text}");
        assert_eq!(load_bookmarks(&file.0).unwrap(), saved);
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
