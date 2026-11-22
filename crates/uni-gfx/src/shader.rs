//! WGSL library with `#import "path"` expansion, plus naga validation.
//!
//! Sources are registered under a path such as `"common/camera.wgsl"`. A line of the form
//! `#import "path"` pulls in another registered file. [`ShaderLib::compose`] expands
//! imports recursively and depth-first: every file appears at most once, each import's
//! text is placed before the text of the file that imports it, and the result is the same
//! on every call. Each file's text is preceded by a `// ---- path ----` marker so naga
//! errors in composed source can be traced back to a file.
//!
//! WGSL global directives must come before every declaration, but an import's text is placed
//! before its importer's. So every `enable …;` line of every composed file is moved to the top
//! of the output, once each (review I7).

use std::collections::{HashMap, HashSet};

/// Errors from composing or validating WGSL.
#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    #[error("missing import {0}")]
    MissingImport(String),
    #[error("naga: {0}")]
    Naga(String),
}

/// A set of WGSL sources keyed by path (for example `"common/camera.wgsl"`).
pub struct ShaderLib {
    sources: HashMap<&'static str, &'static str>,
}

impl ShaderLib {
    /// A library with every `common/*.wgsl` file registered.
    pub fn with_common() -> Self {
        let mut lib = Self {
            sources: HashMap::new(),
        };
        lib.add(
            "common/constants.wgsl",
            include_str!("../shaders/common/constants.wgsl"),
        );
        lib.add(
            "common/hash.wgsl",
            include_str!("../shaders/common/hash.wgsl"),
        );
        lib.add(
            "common/color.wgsl",
            include_str!("../shaders/common/color.wgsl"),
        );
        lib.add(
            "common/exposure.wgsl",
            include_str!("../shaders/common/exposure.wgsl"),
        );
        lib.add(
            "common/camera.wgsl",
            include_str!("../shaders/common/camera.wgsl"),
        );
        lib.add(
            "common/cube.wgsl",
            include_str!("../shaders/common/cube.wgsl"),
        );
        lib
    }

    /// Registers `src` under `path`.
    pub fn add(&mut self, path: &'static str, src: &'static str) {
        self.sources.insert(path, src);
    }

    /// Expands `#import "path"` recursively, including each file at most once, with every
    /// `enable` directive hoisted to the top (once each, in first-seen order).
    pub fn compose(&self, entry: &str) -> Result<String, ShaderError> {
        let mut seen = HashSet::new();
        let mut directives = Vec::new();
        let mut body = String::new();
        self.expand(entry, &mut seen, &mut directives, &mut body)?;
        let mut out = String::with_capacity(body.len());
        for d in directives {
            out.push_str(d);
            out.push('\n');
        }
        out.push_str(&body);
        Ok(out)
    }

    /// Appends `path`'s imports (depth-first) and then its own lines to `out`, unless
    /// `path` is already in `seen`. Marking before recursing also ends import cycles. Its
    /// `enable` lines go to `directives` instead, unless already there.
    fn expand(
        &self,
        path: &str,
        seen: &mut HashSet<&'static str>,
        directives: &mut Vec<&'static str>,
        out: &mut String,
    ) -> Result<(), ShaderError> {
        let (&key, &src) = self
            .sources
            .get_key_value(path)
            .ok_or_else(|| ShaderError::MissingImport(path.to_owned()))?;
        if !seen.insert(key) {
            return Ok(());
        }
        let mut body = String::with_capacity(src.len());
        for line in src.lines() {
            if let Some(dep) = import_path(line) {
                self.expand(dep, seen, directives, out)?;
            } else if let Some(directive) = enable_directive(line) {
                if !directives.contains(&directive) {
                    directives.push(directive);
                }
            } else {
                body.push_str(line);
                body.push('\n');
            }
        }
        out.push_str("// ---- ");
        out.push_str(key);
        out.push_str(" ----\n");
        out.push_str(&body);
        Ok(())
    }

    /// Composes `entry` and creates a shader module from it.
    ///
    /// Validation errors from the device (naga, with the device's real capabilities) are
    /// caught with an error scope and returned as [`ShaderError::Naga`] instead of reaching
    /// the device's uncaptured-error handler.
    pub fn module(
        &self,
        device: &wgpu::Device,
        entry: &str,
    ) -> Result<wgpu::ShaderModule, ShaderError> {
        let src = self.compose(entry)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        match pollster::block_on(scope.pop()) {
            Some(err) => Err(ShaderError::Naga(format!("{entry}: {err}"))),
            None => Ok(module),
        }
    }
}

/// The path named by an `#import "path"` line, or `None` for any other line.
/// A bare `#import` yields `""`, which [`ShaderLib::compose`] reports as a missing import.
fn import_path(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#import")?.trim_start();
    let path = match rest.strip_prefix('"') {
        Some(quoted) => quoted.split('"').next(),
        None => rest.split_whitespace().next(),
    };
    Some(path.unwrap_or(""))
}

/// The trimmed line if it is a WGSL `enable` directive (`enable f16;`), else `None`.
fn enable_directive(line: &'static str) -> Option<&'static str> {
    let t = line.trim();
    let rest = t.strip_prefix("enable")?;
    rest.starts_with(char::is_whitespace).then_some(t)
}

/// Parses and validates WGSL with naga.
///
/// Capabilities are naga's defaults plus `SHADER_FLOAT16`, matching the device on the
/// target M3 (which enables `SHADER_F16`; see docs/research/wgpu30-skeleton/m3-probe-output.txt).
/// A shader that needs more (dual-source blending) uses [`validate_wgsl_with`].
pub fn validate_wgsl(src: &str) -> Result<naga::Module, ShaderError> {
    validate_wgsl_with(
        src,
        naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16,
    )
}

/// Parses and validates WGSL with naga under `caps`, which gate both `enable` directives and
/// the validator (for example `DUAL_SOURCE_BLENDING` for `@blend_src`, review I7).
pub fn validate_wgsl_with(
    src: &str,
    caps: naga::valid::Capabilities,
) -> Result<naga::Module, ShaderError> {
    let mut options = naga::front::wgsl::Options::new();
    options.capabilities = caps;
    let module = naga::front::wgsl::Frontend::new_with_options(options)
        .parse(src)
        .map_err(|e| ShaderError::Naga(e.emit_to_string(src)))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
        .validate(&module)
        .map_err(|e| ShaderError::Naga(e.emit_to_string(src)))?;
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMON: [&str; 6] = [
        "common/constants.wgsl",
        "common/hash.wgsl",
        "common/color.wgsl",
        "common/exposure.wgsl",
        "common/camera.wgsl",
        "common/cube.wgsl",
    ];

    #[test]
    fn every_common_shader_validates() {
        let lib = ShaderLib::with_common();
        for p in COMMON {
            let src = lib.compose(p).unwrap();
            if let Err(e) = validate_wgsl(&src) {
                panic!("{p}: {e}");
            }
        }
    }

    #[test]
    fn common_shaders_export_contract_names() {
        // Names other crates rely on (Task 4 brief, "Interfaces").
        let lib = ShaderLib::with_common();
        let has_fn = |path: &str, name: &str| {
            let m = validate_wgsl(&lib.compose(path).unwrap()).unwrap();
            m.functions
                .iter()
                .any(|(_, f)| f.name.as_deref() == Some(name))
        };
        assert!(has_fn("common/camera.wgsl", "pixel_dir"));
        assert!(has_fn("common/camera.wgsl", "exposure_scale"));
        assert!(has_fn("common/camera.wgsl", "scene_pre"));
        assert!(has_fn("common/camera.wgsl", "star_pre"));
        assert!(has_fn("common/camera.wgsl", "sim_time_s"));
        assert!(has_fn("common/exposure.wgsl", "exposure_scale"));
        assert!(has_fn("common/cube.wgsl", "cube_dir"));
        assert!(has_fn("common/hash.wgsl", "pcg3d"));
        assert!(has_fn("common/color.wgsl", "luminance"));
        assert!(has_fn("common/color.wgsl", "linear_to_srgb"));
        let consts = validate_wgsl(&lib.compose("common/constants.wgsl").unwrap()).unwrap();
        for name in ["PI", "TAU"] {
            assert!(
                consts
                    .constants
                    .iter()
                    .any(|(_, c)| c.name.as_deref() == Some(name)),
                "missing const {name}"
            );
        }
    }

    #[test]
    fn common_shaders_compose_into_one_entry_point() {
        // A feature shader importing every common file must not hit duplicate names.
        let mut lib = ShaderLib::with_common();
        lib.add(
            "test/feature.wgsl",
            r#"#import "common/camera.wgsl"
#import "common/hash.wgsl"
#import "common/color.wgsl"
#import "common/constants.wgsl"
#import "common/exposure.wgsl"
#import "common/cube.wgsl"
@group(1) @binding(0) var out_tex: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dir = pixel_dir(vec2<f32>(gid.xy)) + cube_dir(gid.z, vec2<f32>(0.5)) * sim_time_s()
        + lights.lights[0].pos_radius.xyz * f32(lights.info.x) + camera.cam_delta.xyz;
    let h = pcg3d(vec3<u32>(gid.xy, 7u));
    let n = f32(h.x) / 4294967296.0 * TAU / PI;
    let c = linear_to_srgb(abs(dir) * scene_pre() * star_pre() * n);
    textureStore(out_tex, gid.xy, vec4<f32>(c, luminance(c)));
}
"#,
        );
        let src = lib.compose("test/feature.wgsl").unwrap();
        if let Err(e) = validate_wgsl(&src) {
            panic!("{e}\n{src}");
        }
    }

    const A: &str = concat!(
        "#import \"test/b.wgsl\"\n",
        "#import \"test/c.wgsl\"\n",
        "fn a() -> f32 { return b() + c(); }\n",
    );
    const B: &str = "#import \"test/c.wgsl\"\nfn b() -> f32 { return 2.0 * c(); }\n";
    const C: &str = "fn c() -> f32 { return 1.0; }\n";

    fn abc_lib() -> ShaderLib {
        let mut lib = ShaderLib::with_common();
        lib.add("test/a.wgsl", A);
        lib.add("test/b.wgsl", B);
        lib.add("test/c.wgsl", C);
        lib
    }

    #[test]
    fn compose_expands_import_once() {
        let out = abc_lib().compose("test/a.wgsl").unwrap();
        assert_eq!(out.matches("fn c()").count(), 1, "{out}");
        assert_eq!(out.matches("fn b()").count(), 1, "{out}");
        assert_eq!(out.matches("fn a()").count(), 1, "{out}");
        assert!(!out.contains("#import"), "{out}");
        validate_wgsl(&out).unwrap();
    }

    #[test]
    fn compose_puts_imports_before_importer() {
        let out = abc_lib().compose("test/a.wgsl").unwrap();
        let pos = |s: &str| out.find(s).unwrap();
        assert!(pos("fn c()") < pos("fn b()"), "{out}");
        assert!(pos("fn b()") < pos("fn a()"), "{out}");
        // Stable: composing twice gives identical text.
        assert_eq!(out, abc_lib().compose("test/a.wgsl").unwrap());
    }

    #[test]
    fn compose_terminates_on_import_cycle() {
        let mut lib = ShaderLib::with_common();
        lib.add(
            "test/x.wgsl",
            "#import \"test/y.wgsl\"\nfn x() -> f32 { return 1.0; }\n",
        );
        lib.add(
            "test/y.wgsl",
            "#import \"test/x.wgsl\"\nfn y() -> f32 { return 2.0; }\n",
        );
        let out = lib.compose("test/x.wgsl").unwrap();
        assert_eq!(out.matches("fn x()").count(), 1, "{out}");
        assert_eq!(out.matches("fn y()").count(), 1, "{out}");
    }

    #[test]
    fn compose_missing_import_errors() {
        let mut lib = ShaderLib::with_common();
        lib.add("test/needs.wgsl", "#import \"test/nope.wgsl\"\nfn f() {}\n");
        let err = lib.compose("test/needs.wgsl").unwrap_err();
        assert!(
            matches!(&err, ShaderError::MissingImport(p) if p == "test/nope.wgsl"),
            "{err}"
        );
        let err = lib.compose("test/not-registered.wgsl").unwrap_err();
        assert!(
            matches!(&err, ShaderError::MissingImport(p) if p == "test/not-registered.wgsl"),
            "{err}"
        );
    }

    /// Fragment output with two blend sources (WGSL `dual_source_blending` extension).
    const DUAL_SOURCE: &str = r#"enable dual_source_blending;
struct BlendOut {
    @location(0) @blend_src(0) color: vec4<f32>,
    @location(0) @blend_src(1) transmittance: vec4<f32>,
}
@fragment
fn fs_main() -> BlendOut {
    return BlendOut(vec4<f32>(1.0), vec4<f32>(0.5));
}
"#;

    #[test]
    fn validate_wgsl_with_dual_source() {
        use naga::valid::Capabilities as Caps;
        let caps = Caps::default() | Caps::SHADER_FLOAT16;
        validate_wgsl_with(DUAL_SOURCE, caps | Caps::DUAL_SOURCE_BLENDING).unwrap();
        let err = validate_wgsl_with(DUAL_SOURCE, caps).unwrap_err();
        assert!(matches!(err, ShaderError::Naga(_)), "{err}");
        // validate_wgsl keeps m0.1's capabilities, which do not include dual-source blending.
        assert!(validate_wgsl(DUAL_SOURCE).is_err());
        validate_wgsl_with("enable f16;\nconst H: f16 = 1.5h;\n", caps).unwrap();
    }

    #[test]
    fn compose_hoists_enable_directives() {
        let mut lib = ShaderLib::with_common();
        lib.add(
            "test/blend.wgsl",
            concat!(
                "enable dual_source_blending;\n",
                "#import \"test/half.wgsl\"\n",
                "#import \"common/camera.wgsl\"\n",
                "struct BlendOut {\n",
                "    @location(0) @blend_src(0) color: vec4<f32>,\n",
                "    @location(0) @blend_src(1) transmittance: vec4<f32>,\n",
                "}\n",
                "@fragment\n",
                "fn fs_main() -> BlendOut {\n",
                "    return BlendOut(vec4<f32>(f32(half()) * scene_pre()), vec4<f32>(0.5));\n",
                "}\n",
            ),
        );
        // An import with its own directives, one of them repeated.
        lib.add(
            "test/half.wgsl",
            "enable f16;\nenable dual_source_blending;\nfn half() -> f16 { return 0.5h; }\n",
        );
        let out = lib.compose("test/blend.wgsl").unwrap();
        assert!(
            out.starts_with("enable dual_source_blending;\nenable f16;\n"),
            "{out}"
        );
        assert_eq!(
            out.matches("enable dual_source_blending;").count(),
            1,
            "{out}"
        );
        assert_eq!(out.matches("enable f16;").count(), 1, "{out}");
        use naga::valid::Capabilities as Caps;
        validate_wgsl_with(
            &out,
            Caps::default() | Caps::SHADER_FLOAT16 | Caps::DUAL_SOURCE_BLENDING,
        )
        .unwrap();
        // A file with no directives composes exactly as before (no blank first line).
        assert!(
            lib.compose("common/camera.wgsl")
                .unwrap()
                .starts_with("// ---- common/exposure.wgsl ----\n")
        );
    }

    #[test]
    fn validate_wgsl_reports_naga_errors() {
        let parse = validate_wgsl("fn broken( {").unwrap_err();
        assert!(matches!(parse, ShaderError::Naga(_)), "{parse}");
        let types = validate_wgsl("fn f() -> f32 { return 1u; }").unwrap_err();
        assert!(matches!(types, ShaderError::Naga(_)), "{types}");
    }
}
