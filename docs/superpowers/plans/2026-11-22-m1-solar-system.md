# M1 Real Solar System and Real Sky Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the M0 foundation into a flyable, real Solar System at the real date: planets, moons, the Sun, Saturn's rings, four atmospheres, comets and asteroids on real orbits with time warp, under a real 3-D sky of about 330 000 catalogue stars and a Milky Way band generated from the galaxy model, with bloom, auto-exposure, target selection, goto, cruise, info cards and the approved original HUD, at 60 FPS or better at 1080p internal on the M3 (sustained, last 10 s of a 60 s run).

**Architecture:**
- **CPU (f64):** `uni-physics` evaluates every orbit and rotation analytically (Standish elements, Meeus lunar, satellite and Pluto theories, JPL mean elements, IAU WGCCRE rotation, two-body conics for small bodies). `uni-sim` owns the world: a time controller, the object registry, the frame tree (Sun → planets → moons) and a per-frame `Snapshot` of everything the renderers and the HUD need, all camera-relative in f64. The camera is parented to the dominant body frame, so it rides along under time warp.
- **GPU (f32, camera-relative):** a fixed pass list (the "M1 frame contract" below). Bodies are analytic ray-ellipsoid shaders drawn on a camera-facing proxy mesh that only supplies coverage and depth; atmospheres are Hillaire 2020 LUTs applied in a depth-reading pass; stars and other point sources are PSF splats exposed separately from world light; a cubemap rendered from the galaxy model is both the band and the lens's environment.
- **Verification:** every WGSL routine that matters has a CPU twin and a `--selftest` probe; ephemerides are tested against JPL Horizons spot values; every bookmark has a numeric check (PNG-based, or HDR-based through the `--hdr` capture sidecar); every gate captures the M1 bookmarks, measures them with `uni-verify`, benches with `universe --bench` (sustained mode for the FPS gate), and is looked at by the lead.

**Tech Stack:** Rust 1.100 (edition 2024), wgpu 30.0.2 (Metal), winit 0.30.13, egui 0.37.0, glam 0.34.3, naga 30.0.2, serde + ron + serde_json, image 0.25 (PNG; JPEG/TIFF in `tools/bake` only), flate2. New crates, all ruled and all in `tools/bake` only: `intel_tex_2 = "=0.5.0"` (with `ispc_rt`, `libc`, `num_cpus`) and `rayon = "=1.12.0"` (see "Dependency changes").

**Spec:** `docs/superpowers/specs/2026-11-20-universe-explorer-design.md`
- Read §2 (anti-goals), §4 (constraints), §5.1–5.6, §6, §7, §8, §9 (M1) and §10 before any task.
- Research behind each task: `docs/research/2026-11-20-{physics,rendering,assets,wgpu,synthesis,procgen}.md`. Section references below are into those files.
- wgpu/winit/egui API usage: copy from `docs/research/wgpu30-skeleton/` and from the m0.1 code, not from memory. Adapter features on this M3: `docs/research/wgpu30-skeleton/m3-probe-output.txt`.
- Approved HUD design (Task C4/C5 input): https://claude.ai/artifact/2HExiDvXNRMPruxHppDE5B (four artboards: "In flight — minimal HUD", "Target card — Mars", "Black hole card — Cygnus X-1", "Cruise, time warp and wormhole jump"). Its measured values are summarised in Task C4.
- Plan review: `.superpowers/sdd/2026-11-22-m1-solar-system/plan-review.md`. This revision resolves C1–C4 and I1–I17; "Review findings not applied" at the end lists what it leaves. Round 2 (after `plan-rereview-r1.md`) fixes C2's sign (sky cameras now sit at Sun **+** 100 AU × the view direction), NP1 (`alpha-cen-sun` geometry), NP2 (disk fits on a black sky; no disk fit on Mars) and the six listed minor items. Round 3 (after `plan-rereview-r2.md`) defines B8's `flux` (per-pixel Ω, footprint radius, masked annulus background), boosts registry asteroids under the belt highlight and corrects the belt and Mars-haze arithmetic; `plan-rereview-r3.md` is all clear, and its four minor items are applied as lead rulings (sidecar before bloom, belt check on the mean, the flux fixture, this line).

## Global Constraints

Verbatim from the M0 plan (they bind every task), then the M1 additions.

**Toolchain and dependencies**
- rustc 1.100, `edition = "2024"`, workspace `resolver = "3"`.
- Pinned versions:
  - `wgpu = "=30.0.2"` (default-features = false; features `std`, `parking_lot`, `metal`, `wgsl`)
  - `winit = "=0.30.13"`
  - `egui`, `egui-wgpu`, `egui-winit` = `"=0.37.0"` (feature sets as in the skeleton's `Cargo.toml`)
  - `glam = { version = "=0.34.3", features = ["bytemuck", "serde"] }`
  - `bytemuck = { version = "=1.25.2", features = ["derive"] }`
  - `pollster = "=1.0.3"`
  - `naga = { version = "=30.0.2", features = ["wgsl-in"] }`
  - `image = { version = "=0.25.10", default-features = false, features = ["png"] }`
  - `serde = { version = "=1.0.229", features = ["derive"] }`
  - `ron = "=0.12.3"`
  - `anyhow = "=1.0.104"`, `thiserror = "=2.0.21"`
  - `flate2 = "=1.1.11"`, `sha2 = "=0.11.1"`
  - `log = "=0.4.35"`, `env_logger = "=0.11.11"`
- Declare every dependency once in `[workspace.dependencies]`. Crates use `x.workspace = true`.
- No other crates without the lead's approval.

**GPU**
- Metal backend only.
- Request `adapter.limits()` explicitly.
- Required features = `adapter.features() ∩ {SHADER_F16, FLOAT32_FILTERABLE, TIMESTAMP_QUERY}`.
- The app must run when `TIMESTAMP_QUERY` is absent.

**Units and coordinates**
- CPU: SI units (m, s, kg) in f64.
- GPU: f32, camera-relative.
- Black-hole maths: units of M = GM/c², hole at the origin.
- World axes: **ICRS/J2000 equatorial, right-handed, origin at the Solar System barycentre (SSB)**, +x toward RA 0h Dec 0°, +z toward the north celestial pole.
- Camera space: looks along −Z, up is +Y.

**Depth:** reversed-Z, infinite far plane, `Depth32Float`, `CompareFunction::Greater`.

**Uniform layout rule (this exact bug blanked the old renderer)**
- GPU uniform and storage structs contain only `vec4`/`mat4`-sized members, or `f32` groups padded to 16 bytes.
- No `vec3<f32>` followed by a scalar.
- Every uniform struct has a test comparing `size_of::<RustStruct>()` and field offsets with the size and offsets naga computes for the WGSL struct.

**Code rules**
- Every WGSL file is validated by a naga test in the crate that owns it.
- No `unwrap()` or `expect()` on I/O, parsing or GPU-creation results in library crates. Return `Result`. `uni-app` may `expect` at top level, with a message.
- Every physical constant or formula has a comment naming its source (paper, catalogue or IAU resolution).

**Process (design §10)**
- Agents edit only the files their task lists.
- Agents run only `cargo check -p <crate>` and `cargo test -p <crate> --lib` (or `--bins` for binary crates).
- Agents never commit, never run the app, never download, except Task 7, which downloads.
- Stop and report if `df -h /` shows less than 1.5 GB free.
- The lead commits after each wave.

**Honesty**
- Anything approximate or non-physical is labelled on screen. In M0 that means the lens background ("M0 debug grid") and speeds faster than c ("FTL · non-physical").

**Standing commit rule** (CLAUDE.md §3, renewed at tag `m0`; it overrides "Agents never commit" above). When a task's last step says commit, the implementer commits only its own paths with a pathspec commit (`git add -- <own paths> && git commit -m "..." -- <own paths>`), retrying on `index.lock`. Never `git add -A`, `git commit -a`, `git stash` or `git reset`, and never push.

### M1 additions (bind every M1 task)
- **GPU features (ruling 1):** the requested intersection becomes `adapter.features() ∩ {SHADER_F16, FLOAT32_FILTERABLE, TIMESTAMP_QUERY, TEXTURE_COMPRESSION_BC, DUAL_SOURCE_BLENDING}` (A0a). The app must still run without any of the last three: no timestamps → "n/a"; no BC → the 2k uncompressed map copies; no dual-source blending → the grey-transmittance atmosphere variant.
- **Downloads:** in M1 the Data tasks (A7, A8, A9) are the only ones that download, into `data/cache/<id>/` with `curl -L --fail`. Every downloaded file gets a `data/manifest.toml` entry (id, url, sha256 from `shasum -a 256`, license, credit, raw_path, baked_path), an `ATTRIBUTION.md` line and a measured size in its task report. Raw files are deleted after baking. Baked files ≥ 10 MB are named `*.big` and git-ignored; smaller ones are committed. AT-HYG-derived files stay CC BY-SA 4.0 and say so in `ATTRIBUTION.md` (ShareAlike).
- **Web research** (reading papers, API docs, READMEs) is allowed for every agent; only Data downloads data files.
- **Code reuse and licences (review m24):** Hillaire's sample code is MIT; if any of it is adapted, its MIT notice goes into `ATTRIBUTION.md` (B3's section). LearnOpenGL's code is CC BY-NC 4.0: cite it as a reference only and never copy it.
- **The M1 frame contract** (pass order, pre-exposure split, `lens_mask` channels, shared inputs; section below) binds every render feature.
- **Every WGSL maths routine that matters has a CPU twin and a probe.** Each render crate exports `pub fn gpu_probes(gpu: &uni_gfx::GpuContext, lib: &mut uni_gfx::ShaderLib) -> Vec<uni_gfx::probe::ProbeResult>`; the app's `--selftest` runs all of them. Agents write the probes; the lead runs them (GPU queue).
- **Honesty labels** are exact strings, defined once in `uni_sim::honesty` (world state) or `uni_ui::theme` (UI state), listed in the section "Honesty labels in M1". Approximate-but-physical models are labelled in neutral grey; non-physical or hypothetical modes in amber.
- **Bookmark-driven verification:** a feature is not done until a bookmark in `bookmarks/m1.ron` shows it and the gate's numeric check for that bookmark passes. Every bookmark has at least one numeric check (table "M1 capture bookmarks"), and every bookmark's exposure is worked out on paper (table "Exposure on paper").
- **"verify" markers (ruling 17):** every parameter or value this plan marks "verify" is checked against its cited source by the implementing agent before the task is done; no "verify" remains in code comments at the M1 gate.

## Review Focus

These failure modes are implied by the spec but no feature test covers them. Each has a pinned test in the owning task.

1. **Extreme time warp and dates far from the ephemerides' validity** (rate 10¹², year −10⁶, ±10⁷ or +10⁶, pause and resume). Expected: every position stays finite and plausible (orbital elements frozen at the validity edge, no polynomial blow-up, every e < 1), body poles stay near their J2000 orientation, the HUD shows the extrapolation labels, and the camera stays with its anchor body. *Tests:* `standish_far_dates_finite_and_flagged` (A1), `pluto_outside_window_is_two_body_and_flagged` (A1), `theories_fall_back_far_from_epoch`, `poles_frozen_outside_window` (A2), `max_warp_ten_thousand_frames_finite` (A4).
2. **The camera very close to, on, or inside a body, the Sun or an atmosphere** (1 m altitude, inside an atmosphere shell, inside the solid body or the photosphere, exactly at its centre; cruise or goto aimed at a surface). Expected: no NaN, no depth or hit-distance jitter from f32 cancellation, the body never vanishes because the camera crossed its proxy mesh, the sky is drawn from inside an atmosphere, a camera inside a star draws nothing and warns, and travel modes stop outside. *Tests:* `cap_mesh_sagitta_below_altitude` (B0), `ray_ellipsoid_from_inside_and_centre`, `ray_ellipsoid_f32_near_surface` (B1), `camera_inside_star_draws_nothing` (B2), `segment_from_inside_shell` (B3), `goto_stops_outside_atmosphere`, `cruise_drops_out_at_atmosphere_top` (C3).
3. **Missing, truncated or partial baked data** (no maps, no AT-HYG, no small bodies, a truncated `.big`, AT-HYG rows with NULL distance or velocity). Expected: the app starts with labelled fallbacks (uniform albedo, the BSC5 sky, no small bodies) and HUD warnings; NULL rows are skipped or flagged, never NaN; `--capture` exits 2 only for files the bookmark set needs. *Tests:* `load_athyg_truncated_is_bad_format`, `null_distance_rows_skipped_and_counted` (A7), `load_tex_truncated_is_bad_format` (A8), `world_data_missing_files_warns` (A4), `missing_map_falls_back_to_uniform_albedo` (B1).
4. **Point ↔ disk level-of-detail handover** (a planet shrinking to a dot, a star approached until resolved, the Sun seen from Neptune or α Cen). Expected: no brightness pop, never drawn twice, never missing. *Tests:* `disk_flux_matches_point_flux` (A3), `body_is_point_or_resolved_never_both` (A4), `lod_exposure_mix_endpoints_monotonic` (B0), `point_disk_handover_flux_continuous` (B1), `star_surface_flux_matches_catalogue_magnitude` (B2).
5. **Precision far from the SSB and at large distances** (camera at α Cen, 1 kpc out, at Neptune; a body 30 AU away; an asteroid evaluated in f32). Expected: stars and bodies stable to well under a pixel, no jitter while flying; f32 GPU orbits stay within their stated bound. *Tests:* `sector_offsets_precision_at_1kpc` (B4), `reanchor_preserves_root_position` (A6), `body_position_f32_error_subpixel_at_30au` (B1), asteroid probe ≤ 100 km (C2).
6. **Auto-exposure on small, mixed or extreme scenes** (a planet filling 0.1 % of the frame, a crescent, the Sun's disk, a sky with no world light, a camera in daylight). Expected: no body blows out to flat white once it is metered, the night view stays dark-adapted, the meter converges from clamped pixels, and stars are not shown on a daylit sky. *Tests:* `small_disk_not_blown_out`, `meter_ramp_between_thresholds`, `start_ev_stable_when_lit_area_changes`, `settle_converges_from_clamped_sun`, `daylight_disables_eye_adapted` (B6).

---

## The M1 frame contract

Task A0a writes this into `crates/uni-gfx/src/targets.rs` (replacing m0.1's "background / lens / star contract" doc comment) and into CLAUDE.md §2's render-contract block.

**Exposure split (pre-exposed HDR, as at m0.1):**
- *World light* (bodies, Sun disk, rings, atmospheres, comet nuclei, comae and tails) is written as luminance [cd/m²] × `scene_pre()`.
- *Sky light and point sources* (catalogue stars, the band, planets/Sun/asteroids drawn as points) are written as luminance × `star_pre()`. With "Eye-adapted stars" in effect, `star_pre` comes from a fixed dark-adapted EV; otherwise `star_pre == scene_pre`. This is the design §7 item 4 dual exposure, extended to every unresolved source so a planet and a star of equal magnitude look equally bright (ruling 5).
- "Eye-adapted stars" is in effect when `ExposureSettings.eye_adapted && !ExposureSettings.daylight`; `daylight` is set by the app while the camera is inside an atmosphere shell with the Sun above its local horizon, because a real daytime sky hides the stars (review m12). The HUD's amber label follows exactly this condition (review m13), not `FrameCtx.exposure`.
- A resolved body whose projected radius is between `POINT_LOD_PX` (1.5 px) and `BLEND_LOD_PX` (4 px) blends its exposure factor from `star_pre` to `scene_pre` (CPU twin `uni_planet::lod_exposure_mix`, implemented in B0), so the point → disk handover never pops. Labelled as part of "Eye-adapted stars".
- Auto-exposure meters **world light only**: the meter (pass 9) runs before any sky light or point source is written. Its rule is in Task B6; its consequence for every bookmark is in "Exposure on paper".
- Star brightness against the band depends on the field of view (review m26), deliberately: stars are written as E/Ω_pixel, the band as a surface brightness, so at fov 20° a star's pixels are (tan 30°/tan 10°)² = 10.7× (+3.42 EV) brighter relative to the band than at 60°. That is what a camera does when it zooms (point sources gain contrast over extended light), so it stays.

**`lens_mask` channels** (A0a widens `LENS_MASK_FORMAT` from m0.1's `R32Float` to `Rgba16Float`; `Renderer::render` still clears it to 0 every frame):
- `r`: fraction of the pixel's background captured by a black hole (m0.1 meaning; lens pass only).
- `g`: sky occlusion = 1 − transmittance toward the background, written with `BlendOperation::Max` on the green channel only. Bodies and comet nuclei write their coverage, rings their opacity, atmospheres `1 − luminance(T)`. The lens pass writes `g = 0` on the traced pixels it repaints.
- `b`: 1 where the lens pass wrote the lensed environment.
- `a`: reserved for M3 star magnification.
- Sky features multiply what they add by `(1 − r)(1 − g)`; the band also by `(1 − b)`; point sprites and asteroids by `(1 − r)` (review m6; they are depth-tested against bodies, so they do not use `g`; a point behind an atmosphere is not dimmed by it in M1, an accepted approximation).
- The lens pass reads the depth texture and leaves every pixel whose depth is not 0 (a body in front) untouched, so foreground objects are not lensed (spec §5.3, review m7).

**Shared inputs.** Features get per-frame data from the app through `uni_gfx::SharedInput<T>` (A0a). The app sets every `SharedInput` before `Renderer::render`; features read them only in `prepare` (review §7 item 4).

**Pass order** (push order in `uni-app/src/scene.rs`):

| # | Feature (name) | Crate | Writes | Depth |
|---|---|---|---|---|
| 1 | `exposure` (resolve; own bind group, not group 0, review m14) | uni-post | exposure buffer (storage) | — |
| 2 | `band-update` (one cubemap face per frame when needed) | uni-sky | env cubemap | — |
| 3 | `background` (M0 debug grid; only with `LensEnv::DebugGrid` and a traced lens; display-referred, so the app runs Manual exposure whenever `DebugGrid` is selected) | uni-rel | hdr | — |
| 4 | `sun` (star surfaces) | uni-planet | hdr (world), g | write, Greater |
| 5 | `bodies` | uni-planet | hdr (world), g | write, Greater |
| 6 | `rings` | uni-planet | hdr (world), g | test only |
| 7 | `comets` (Wave C) | uni-planet | hdr (world), g (nucleus) | nucleus write; coma/tails test only |
| 8 | `atmosphere` (reads the depth texture; dual-source variant: two pipelines, grey variant: one) | uni-atmo | hdr, g | none attached |
| 9 | `histogram` (meter) | uni-post | meter state | — |
| 10 | `lens` (reads depth; skips pixels with depth ≠ 0) | uni-rel | hdr (sky, star exposure), r, g = 0, b | — |
| 11 | `band` | uni-sky | hdr (sky) × (1−r)(1−g)(1−b) | test GreaterEqual at depth 0 (ruling 4) |
| 12 | `stars` | uni-sky | hdr (sky) × (1−r)(1−g) | test GreaterEqual at depth 0 |
| 13 | `points` (bodies/Sun/near stars as points) | uni-sky | hdr (sky) × (1−r) | test Greater at true depth |
| 14 | `asteroids` (Wave C) | uni-sky | hdr (sky) × (1−r) | test Greater at true depth |
| 15 | `bloom` (input clamped to `BLOOM_INPUT_MAX`) | uni-post | hdr | — |
| 16 | `tonemap` (upscale + AgX + dither) | uni-post | output | — |
| — | HUD + overlays (egui) | uni-ui | output | — |

Rationale for drawing the band after the opaque passes (ruling 4, a deliberate deviation from spec §5.3, where the background comes first): the band is sky light at star exposure, so it must not be metered, and drawing it after bodies with `(1 − g)` composites it correctly behind partly transparent atmospheres and rings. It draws only where there is no body (depth at the far plane) and honours `lens_mask` as above. A0a records this reason in the `targets.rs` doc comment and in CLAUDE.md.

**Deliberate deviations from the spec** (recorded like ruling 4; review m23):
- Pass order: the band after the opaque passes and the meter (ruling 4).
- Maps: our own `UNITEX1` container with BC7 albedo and BC4 single-channel maps (A8), not KTX2. The texel formats match spec §5.4; only the container differs (no KTX2 crate is approved).
- Atmospheres: a per-pixel march through Hillaire's transmittance and multiple-scattering LUTs (B3), not the sky-view and aerial-perspective LUTs of spec §5.1. Step 3 of the atmosphere's fallback ladder (half-resolution march) is the planned route if it misses its budget; the LUTs proper are an M2 upgrade.

---

## Built on m0.1

The M0 final-review fix wave landed as tag `m0.1` (commits `0d08201`…`6fb97fa`). This plan uses its interfaces by their real names:
- `RenderFeature { name, prepare(gpu, ctx, targets), encode(gpu, enc, frame: &FrameBindings<'_>, targets, output, timer), resize }`. `FrameBindings { camera: &wgpu::BindGroup }` is group 0, layout `Renderer::camera_bind_group_layout()`: binding 0 `CameraUniform` (240 B at m0.1, 336 B after A0a), binding 1 `ExposureUniform { pre: [f32; 4], ev100: [f32; 4] }` (32 B; `.x` scene, `.y` star). `Renderer::exposure_buffer()` has usage UNIFORM | STORAGE | COPY_DST | COPY_SRC. The `Renderer` writes both buffers from `FrameCtx` every frame; B6's resolve pass then overwrites the exposure buffer on the GPU.
- WGSL `common/camera.wgsl`: vars `camera`, `exposure`; `pixel_dir`, `exposure_scale`, `scene_pre()`, `star_pre()`, `const HDR_MAX = 60000.0`. Rust: `uni_gfx::{HDR_MAX, exposure_scale, Exposure { scene_ev100, star_ev100 }, Exposure::eye_adapted_stars()}`.
- `FrameCtx { frame_index, epoch, dt_real_s, camera, view_rot, proj, viewport /* internal size */, output_size, exposure, lenses }`; `Renderer::new(gpu, internal_size)`; `uni_app::scene::{internal_size, INTERNAL_LINES = 1080, EV100_LIMITS = -30.0..=40.0, exposure(), frame_ctx(), hud_data()}`; `uni_post::TonemapFeature` reads `hdr` 1:1 or upscales bilinearly.
- `targets.rs`: background → opaque → lens → stars → post; the lens writes `hdr` and `lens_mask` only on its traced rectangle; `LENS_MASK_FORMAT = R32Float` (`r` = captured fraction); `hdr` usage STORAGE_BINDING | TEXTURE_BINDING | RENDER_ATTACHMENT (no COPY_SRC yet).
- uni-rel: `LensScale`, `LensTracer::new(gpu, lib, camera_bgl, LensScale)`, `DebugSkyBackground::new(gpu, lib, camera_bgl, LensScale)`, `debug_background_draws(LensCoverage)` (true only for `Traced`), `LensCoverage::{Unlensed, Traced, InsideHorizon, BeyondReach}`, `lens_coverage(ctx)`, `BEYOND_REACH_WARNING`.
- Timings: `Renderer::{gpu_ms() -> Vec<(&'static str, Option<f32>)>, gpu_frame_ms() -> Option<f32>, gpu_timings() -> Option<&GpuTimings>, poll_timings(gpu)}`; `GpuTimings { seq, passes, frame_ms }`; private `MAX_TIMED_PASSES = 32`. `None` is an invalid measurement: "n/a" on screen, `null` in JSON, never 0.
- `GpuContext::{take_errors(), has_timestamps()}`; `context.rs` `WANTED_FEATURES` = SHADER_F16 | FLOAT32_FILTERABLE | TIMESTAMP_QUERY; `shader::validate_wgsl(src)` (naga default capabilities | SHADER_FLOAT16); `ShaderLib::{with_common, add, compose, module}`.
- uni-ui: `EYE_ADAPTED_STARS_LABEL` = "Eye-adapted stars (dual exposure, non-physical)"; `HudData { fps, cpu_ms, gpu_ms, gpu_frame_ms, gpu_timing_available, render_size, output_size, dist_from_sun_m, nearest_name, nearest_dist_m, speed_mps, epoch, time_rate, star_count, warnings, inside_horizon, lens_on_screen, eye_adapted_stars }`; the M0 lens labels are private (`hud.rs` `HONESTY_LABELS`). uni-sky: `DEFAULT_STAR_EV100 = −4.0`, `psf::{MIN_RADIUS_PX = 3.0, MAX_RADIUS_PX = 64.0, VISIBILITY_CUTOFF = 3e-5}`.
- CLI: `universe --capture <ron> --out <dir> [--size WxH] [--frames N] [--hud]` (JSON: name, width, height, gpu_ms, gpu_frame_ms, cpu_ms, star_count, lens_scale, epoch_utc; exit 2 on missing data); `universe --bench <ron> [--size WxH] [--frames N]` prints two text lines per bookmark (8 warm-up frames, then frame / GPU frame / CPU mean and p95 and per-pass means) — **no JSON and no sustained mode yet; B9 adds both**; `uni-verify {sheet, contact, shadow}`. Bookmark fields: `name, view {pos, orient, fov_y_rad}, epoch_utc, scene_ev100, star_ev100, lens_scale`.

A0a lists every change to these interfaces; no task assumes anything else.

## Dependency changes (decided)

The rulings decided every row. "Default stands" means no new crate.

| Change | Decision | Where |
|---|---|---|
| Device features `TEXTURE_COMPRESSION_BC`, `DUAL_SOURCE_BLENDING` | Approved (ruling 1); fallbacks stay implemented | A0a requests them; B1 uses the 2k uncompressed map copies without BC; B3 has a grey variant without dual-source |
| `image` features `jpeg`, `tiff` | Approved, `tools/bake` only (ruling 2) | A0b adds them to `tools/bake/Cargo.toml` only |
| `intel_tex_2 = "=0.5.0"` (MIT OR Apache-2.0; with `ispc_rt`, `libc`, `num_cpus`) | Approved, `tools/bake` only (round 2) | A8 encodes BC7 albedo and BC4 single-channel maps; if it fails to link, A8 falls back to its own BC1/BC4 encoder and reports |
| `rayon = "=1.12.0"` | Approved, `tools/bake` only (ruling 3) | Runtime crates use `std::thread::scope` or stay single-threaded within CPU ≤ 4 ms |
| `half` | Default stands | A7 writes `f32_to_f16` / `f16_to_f32` with an exhaustive test; WGSL uses `unpack2x16float` |
| An ephemeris crate (`anise`) with DE440s | Default stands | Horizons spot values (DE440/441) baked by A9 into `data/golden/horizons_spots.ron` |
| Internal deps only: `serde`, `ron`, `bytemuck` into `uni-data`; `serde`, `ron` into `uni-sim`/`uni-nav`; `glam` into `uni-physics`, `uni-sim` and `uni-procgen` (their stubs use `glam::DVec3`/`DQuat`; their manifests have no `[dependencies]` today) and into `uni-ui` at C0 (overlay types use `glam::Vec2`/`DVec3`); `naga` and `bytemuck` into `uni-procgen` (its WGSL layout test); `uni-data` as a dev-dependency of `uni-physics` (Horizons goldens in tests); `uni-*` path deps per the tasks | Wiring | A0b, B0 and C0 add every one, so no wave agent edits a manifest (review I8) |

## Downloads (Data tasks only)

Sizes from `docs/research/2026-11-20-assets.md` (Content-Length measured 2026-11-20) unless marked "confirm"; the Data task confirms each with a GET before keeping it and stops if a file is more than 25% larger than listed. 14 GB was free at planning time; the 1.5 GB watchdog still applies.

| id | Source / URL | Licence / credit | Raw size | Baked output (size, committed?) | Task |
|---|---|---|---|---|---|
| `athyg-m10` | AT-HYG v4.0 `data/subsets/athyg_40_reduced_m10.csv.gz`, https://codeberg.org/astronexus/athyg (Git LFS; resolve the LFS object URL) | CC BY-SA 4.0, "AT-HYG database, D. Nash (astronexus)" + ESA/Gaia/DPAC | 27.6 MB | `athyg_m10.bin` ≈ 9.2 MB (28 B × valid rows), committed; `athyg_names.ron` ≈ 0.5 MB, committed | A7 |
| `exoplanets` | NASA Exoplanet Archive TAP `pscomppars`, selected columns (incl. `sy_vmag`), CSV | public; "This research has made use of the NASA Exoplanet Archive…" | 1–3 MB (confirm) | `exoplanets.ron` ≈ 1.5 MB, committed | A7 |
| `sss-8k` | Solar System Scope: `8k_earth_daymap.jpg` 4.57 MB, `8k_earth_clouds.jpg` 11.62, `8k_moon.jpg` 15.03, `8k_mars.jpg` 8.40, `8k_mercury.jpg` 15.04, `4k_venus_atmosphere.jpg` 1.55, `8k_jupiter.jpg` 3.08, `8k_saturn.jpg` 1.10, `8k_saturn_ring_alpha.png` 0.065, `2k_uranus.jpg` 0.078, `2k_neptune.jpg` 0.24 | CC BY 4.0, "Solar System Scope" | 60.8 MB | `data/baked/maps/*.big` ≈ 411 MB on disk if the six "8k" files decode at 8192 × 4096 (BC7 albedo 44.7 MB each with mips, BC4 clouds 22.4 MB, Venus 4k 11.2 MB, Uranus/Neptune 2k 2.8 MB each, plus a 2k uncompressed copy of every map, 11.2 MB RGBA / 2.8 MB R8: 268 + 22.4 + 11.2 + 5.6 + 101 + 2.8 = 411 MB on disk; on the GPU only the BC mip chains load, 268 + 22.4 + 11.2 + 5.6 = 307 MB, or the 2k copies, 9 × 11.2 + 2.8 = 104 MB, when BC is absent), git-ignored; `maps/index.ron`, committed; `saturn_rings.ron` ≈ 20 KB, committed | A8 |
| `maps-tier-b` | Galilean moons, Pluto/Charon, Ceres global mosaics (NASA/USGS Astrogeology, public domain) — only versions ≤ 60 MB each, ≤ 300 MB raw in total (ruling 12; Titan is not in the ruled list, and its surface is hidden by haze anyway) | public domain, mission credit | confirm (≤ 60 MB each) | as above | A8 |
| `sbdb-comets` | JPL SBDB Query API, `sb-kind=c` numbered periodic + the named long-period list in A9 | public (JPL/NASA) | ≈ 0.2 MB (confirm) | `comets.ron` ≈ 0.1 MB, committed | A9 |
| `sbdb-asteroids` | JPL SBDB Query API, `sb-kind=a`, `H ≤ 14` (count confirmed first with `limit=1`; ruling 11) | public (JPL/NASA) | 5–15 MB (confirm) | `asteroids.bin` (72 B/object; ≈ 3.6 MB at 50 k), committed if < 10 MB | A9 |
| `jpl-sats` | https://ssd.jpl.nasa.gov/sats/elem/ and `/sats/phys_par/` (HTML tables) | public (JPL SSD) | ≈ 0.3 MB | `sat_elements.ron` ≈ 60 KB, committed | A9 |
| `horizons` | JPL Horizons API (`https://ssd.jpl.nasa.gov/api/horizons.api`), vectors and observer tables listed in A9 | public (JPL SSD); DE440/441 | ≈ 0.7 MB over ~170 calls | `data/golden/horizons_spots.ron`, `sky_spots.ron` ≈ 90 KB, committed | A9 |
| `facts` | NASA NSSDCA planetary and satellite fact sheets (web pages) | public (NASA) | ≈ 0.2 MB | `body_facts.ron` ≈ 30 KB, committed | A9 |
| `fonts` | Inter (rsms/inter release) Light/Regular/Medium/SemiBold TTF; IBM Plex Mono (IBM/plex release) Light/Regular/Medium TTF | SIL OFL 1.1 (licence text committed beside them) | ≤ 15 MB zips (confirm) | `data/fonts/*.ttf` ≈ 1.6 MB + `OFL-*.txt`, committed | A9 |

**Required raw total ≈ 120 MB; with tier B ≤ 420 MB; baked git-ignored ≈ 410–650 MB.** Not downloaded in M1: BMNG/GEBCO/LOLA/MOLA elevation (M2), Black Marble (M2 night lights), the full AT-HYG (M4), DE440s. HDR capture sidecars (`--hdr`, 33.2 MB each at 1920×1080) are written to `captures/` (git-ignored) only during gate runs and deleted after the checks.

## Honesty labels in M1

Exact strings. World-state labels live in `uni_sim::honesty::label`; UI-state labels in `uni_ui::theme`.

| Condition | Text | Colour | Owner |
|---|---|---|---|
| eye-adapted stars in effect (`eye_adapted && !daylight`) | `uni_ui::EYE_ADAPTED_STARS_LABEL` = `Eye-adapted stars (dual exposure, non-physical)` (m0.1) | amber | C4 |
| free flight (M1 has no ship gravity) | `FREE FLIGHT · GRAVITY NOT SIMULATED` (ruling 6) | amber | C4 |
| cruise or goto above c, or cruise at any speed | `CRUISE · FTL · NON-PHYSICAL · GRAVITY IGNORED` | amber | C4 |
| epoch outside 3000 BC – AD 3000 | `Planet positions extrapolated outside 3000 BC – AD 3000 (approximate)` | amber | A4 |
| epoch outside 1885–2099 while Pluto or Charon is on screen or selected | `Pluto position extrapolated outside 1885–2099 (approximate)` | amber | A4 |
| \|epoch − catalogue epoch\| > 100 000 yr | `Star positions: linear motion beyond ±100 000 yr (approximate)` | amber | A4 |
| band on screen | `Milky Way band: galaxy model v0 (real dust map in M4)` | grey | A4 |
| procedural belt drawn and visible (highlight on, or camera within 6 AU of the Sun) | `Faint asteroids: procedural, from catalogue statistics` | grey | A4 (string), C6 (condition) |
| "Highlight asteroid belt" toggle on | `Asteroid belt brightness exaggerated (non-physical)` (round 2) | amber | C4 |
| debug lens environment | `Lens background: M0 debug grid` (m0.1 string) | amber | B7/C4 |
| a lens traced on screen | `Stars near the hole are not yet lensed (M3)` (C4 replaces m0.1's "(M0)" wording) | amber | C4 |
| info cards | `Orbit: JPL mean elements (approximate)` · `Orbit: two-body from epoch YYYY-MM-DD (approximate far from epoch)` · `No surface map: uniform albedo` · `Tails: Finson–Probstein dust model; brightness approximate` · `Nucleus shape: procedural` · `Orbit orientation on the sky unknown: drawn with Ω = 0°` · `Inclination unknown: drawn edge-on (i = 90°)` · `Granulation: procedural (measured cell size and lifetime)` · `Atmosphere: simplified haze model (approximate)` (Venus, Titan) · `Rings: Solar System Scope texture; phase function approximate` (Saturn) · `Clouds: static map (no weather)` (Earth) · `Cloud tops: static map, super-rotating (approximate)` (Venus) · `Star motion: tangential only (no radial velocity)` · `Host star from NASA Exoplanet Archive (not in AT-HYG)` | grey | C4 |

---

## Execution waves

| Wave | Tasks | Who | Parallel? |
|---|---|---|---|
| A-0 | A0a (frozen-crate interfaces + M0 regression), then A0b (Wave A stubs, manifests, shared-file sections); "A0" in the rulings means both | Lead | sequential (review m27) |
| A-i | A1, A5, A6, A7, A8, A9 | 6 agents | yes; files disjoint |
| A-ii | A2, A3, A4 (A2/A3 consume A1's `kepler`/`frames`/`pluto`; A3 also A9's sky spots and facts; A4 consumes A2's `Ephemeris` and A3's `photometry`) | 3 agents | yes, after A1 is reviewed; the lead reviews and merges A3, then A2, then A4 |
| A-gate | A10 | Lead | — |
| B-0 | B0 (Wave B stubs + the shared CPU/WGSL twins `cap_mesh`, `sun_visible_fraction`, `lod_exposure_mix`) | Lead | — |
| B | B1, B2, B3, B4, B5, B6, B7, B8 | 8 agents | yes; files disjoint |
| B-gate | B9 (integration + gate) | Lead | — |
| C-0 | C0 (Wave C stubs + uni-sim additions incl. `World::details`) | Lead | — |
| C | C1, C2, C3, C4, C5 | 5 agents | yes; files disjoint |
| M1 gate | C6 (integration + M1 gate, tag `m1`) | Lead | — |

- **Task counts:** Wave A 9 agent tasks + 3 lead (A0a, A0b, A10); Wave B 8 + 2; Wave C 5 + 2. **Peak parallel agents: 8** (Wave B; Wave A peaks at 6, Wave C at 5). Distinct agent roles over M1: about 14.
- A4's tests that call A2/A3 functions panic on the stubs until those tasks merge; its other tests use a fake `BodySource` and pass on their own.
- **Ownership:** one agent per module directory; at most 3 agents per crate in a wave (uni-physics A1–A3, Data A7–A9 share `uni-data` and `tools/bake` by disjoint files; uni-planet B1+B2; uni-sky B4+B5; uni-ui C4+C5).
- **Shared files are pre-split by the lead's stub task (review I8):** A0b writes every Wave A manifest change (workspace pins, `tools/bake/Cargo.toml`, the uni-data, uni-physics, uni-sim, uni-procgen and uni-nav manifests), every `lib.rs` module list and re-export line, every `tools/bake/src/main.rs` subcommand arm, and the marked sections of `data/manifest.toml` and `ATTRIBUTION.md`. B0 and C0 do the same for their waves (including the probe hooks in files another agent owns). Wave agents edit only their own module files and their own marked sections.
- **Hidden in-wave dependencies are removed (review I9):** shared twins are implemented by the lead (`cube_dir` in A0a's `uni_gfx::env`; `cap_mesh`, `sun_visible_fraction`, `lod_exposure_mix` in B0; `World::details` in C0); A1's `Conic::from_state` is in the A0b stub list; A9 implements `uni_data::golden` and writes `horizons_spots.ron` and `sky_spots.ron` in its Step 1, before anything else.
- Each agent task gets its own reviewer before the lead integrates it (spec compliance against this plan, then code quality).
- Wave tasks depend only on the stubs from the wave's lead task. They must not change a stub signature; if one is wrong they report to the lead.

## Performance budget (ruling, review I15)

GPU budget per pass at 1920×1080 internal, sustained (the M3 is fanless: spec §4 expects 60–70 % of peak). Not every pass runs in every scene; the sum is the worst case with everything on screen. The window's HUD is included because the FPS gate is a window number.

| Pass | Budget [ms] | Basis | Fallback ladder, applied in this order until the pass fits |
|---|---|---|---|
| exposure (resolve) | 0.05 | one workgroup | — |
| band-update | 0 steady (≤ 0.5 on a refresh frame) | never refreshes in the Solar System (B5) | refresh one face every second frame |
| sun | 0.5 | `sun-closeup`, `sun-granulation` (full frame) | 1. one granulation layer instead of two; 2. cap mesh 48×96 → 32×64 |
| bodies | 1.5 | `mars-reference-angle` (23 % of the frame), bicubic magnification | 1. cap mesh 48×96 → 32×64; 2. bicubic only above 2 px per texel (from 1.5) |
| rings | 0.4 | `saturn-ring-plane` | 256 → 128 annulus segments |
| comets | 0.5 GPU, ≤ 1.0 CPU | `comet-neowise-2020` | 1. grains 2048 → 1024 per comet; 2. tailed comets 8 → 4 |
| atmosphere | 3.0 | `earth-map-closeup`, `earth-limb-dawn` | 1. `g` from the transmittance-LUT ratio instead of a second march; 2. 32 → 16 steps (48 → 24 on limb paths); 3. half-resolution march with a depth-aware upsample |
| histogram | 0.3 | full frame | one sample per 2×2 px |
| lens | not budgeted | M0 brute-force ground truth, only near TestBH-1 (M3 replaces it) | — |
| band | 0.3 | full-frame cube sample | — |
| stars | 2.5 | 332 k instances; m0.1 measured 0.07–0.27 ms for 9 096 BSC5 stars, so 2.5–9.9 ms if linear | 1. compute cull + indirect draw (only stars above `VISIBILITY_CUTOFF` reach the rasteriser); 2. CPU sector frustum culling; 3. drawn-star limit V 10 → 9 with the band's `v_lim` and `faint_light_fraction` at the same limit |
| points | 0.2 | ≤ a few hundred sprites | — |
| asteroids | 1.0 | 50 k real + 250 k procedural | belt 250 k → 100 k → 50 k (C6; review m18) |
| bloom | 1.0 | 6 levels | 6 → 5 → 4 levels |
| tonemap | 0.6 | m0.1 measured 0.26–0.52 ms | — |
| HUD (egui, window only) | 0.5 | — | — |
| **Sum** | **12.35** | ≤ 13 ms (ruling), 4.3 ms under 16.7 ms | |

CPU budget per frame (ruling 3: ≤ 4 ms): snapshot ≤ 2.0 ms, comet tails ≤ 1.0 ms (round robin), asteroid mean anomalies ≤ 0.5 ms, everything else ≤ 0.5 ms; `--bench` reports `snapshot_cpu_ms` and `comets_cpu_ms`.

**Gate rule.** B9 and C6 run `universe --bench bookmarks/m1-bench.ron --sustained 60 --json …` (B9 adds the mode): each bench bookmark renders continuously for 60 s with up to two frames in flight, and the report gives the first and the last 10 s separately. The FPS gate uses the last 10 s: per Solar System bookmark, p95 GPU frame ≤ 16.0 ms, p95 CPU ≤ 4.0 ms and p95 frame interval ≤ 16.7 ms (≥ 60 FPS). A pass whose last-10-s mean exceeds 1.25 × its budget gets its ladder applied as a perf-fix task even when the frame passes.

---

## Wave A — foundations

### Task A0a: Frozen-crate interfaces and the M0 regression (lead)

**Files**
- Modify (frozen crates, lead only): `crates/uni-core/src/{camera,epoch,units,color,lib}.rs`; `crates/uni-gfx/src/{frame,renderer,targets,context,shader,capture,lib}.rs`; `crates/uni-gfx/shaders/common/camera.wgsl`.
- Create: `crates/uni-gfx/src/{shared,env,probe}.rs`; `crates/uni-gfx/shaders/common/{exposure,cube}.wgsl`.
- Mechanical follow-ups so the workspace compiles: `crates/uni-rel/shaders/lens_trace.wgsl` (storage format `rgba16float`, writes `vec4(captured, 0, 1, 0)`), `crates/uni-rel/src/tracer.rs` (format constant only), `crates/uni-sky/shaders/stars.wgsl` (reads `.r`, unchanged semantics), every `FrameCtx { .. }` / `CameraView { .. }` literal in `uni-app`, `uni-nav`, `uni-gfx`, `uni-sky`, `uni-rel` tests.
- `CLAUDE.md` §2 render-contract block: the M1 frame contract and the band-order deviation with its reason (ruling 4).

**Interfaces.** Frozen M1 additions to m0.1 ("Built on m0.1" lists what exists); Wave A codes against these.

`uni-core`:
```rust
// camera.rs
pub struct CameraView {
    pub pos: UniversePos, pub orient: glam::DQuat, pub fov_y_rad: f64,
    #[serde(default)] pub velocity_mps: glam::DVec3, // root axes; 0 unless a travel mode sets it (HUD, M3 aberration)
}
// epoch.rs
impl Epoch {
    pub fn days_since_j2000(self) -> f64;             // day as f64 + sec/86400
    pub fn from_days_since_j2000(d: f64) -> Self;     // non-finite d → J2000 (logged)
    pub fn jd_tdb(self) -> f64;                       // 2451545.0 + days_since_j2000
}
// units.rs
pub const R_SUN_M: f64 = 6.957e8;        // IAU 2015 Resolution B3 nominal solar radius
pub const L_SUN_W: f64 = 3.828e26;       // IAU 2015 B3 nominal luminosity
pub const TEFF_SUN_K: f64 = 5772.0;      // IAU 2015 B3 nominal effective temperature
pub const M_V_SUN: f64 = 4.83;           // from m_V = −26.74 (Cox 2000, Allen's Astrophysical Quantities), consistent with star_illuminance_lux
// color.rs: bv_to_teff now clamps B−V to [−0.4, 2.0] and maps non-finite input to 5800 K (review T3 deferral)
```

`uni-gfx`:
```rust
// frame.rs
pub const MAX_LIGHTS: usize = 4;
pub const SIM_TIME_WRAP_S: f64 = 4_194_304.0;   // 2^22 s ≈ 48.5 d: f32 sim time keeps 0.25 s resolution
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightView {
    pub rel_pos_m: glam::DVec3,          // light centre − camera, root axes
    pub radius_m: f64,                   // photosphere radius
    pub luminous_intensity_cd: f64,      // illuminance at distance d is I/d² lux
    pub color: glam::Vec3,               // linear Rec.709, luminance 1
}
pub struct FrameCtx { /* m0.1 fields: frame_index, epoch, dt_real_s, camera, view_rot, proj, viewport, output_size, exposure, lenses */
    pub lights: Vec<LightView>, pub sim_dt_s: f64, pub jitter_px: glam::Vec2 /* M2 TAA; 0 in M1 */ }
#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {               // 336 B (m0.1: 240 B, 6 members)
    pub view_rot: [[f32;4];4], pub proj: [[f32;4];4], pub inv_view_proj: [[f32;4];4],
    pub prev_view_proj: [[f32;4];4],     // previous frame's proj·view_rot (= current on the first frame); M2 TAA
    pub viewport: [f32;4], pub output: [f32;4],
    pub time: [f32;4],                   // frame index (wrapped), dt_real_s, fov_y, aspect (as m0.1)
    pub sim: [f32;4],                    // TDB s since J2000 mod SIM_TIME_WRAP_S, sim dt [s], jitter x, y [px]
    pub cam_delta: [f32;4],              // camera translation since the previous frame [m], root axes; w 0
}
#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightGpu { pub pos_radius: [f32;4], pub color_intensity: [f32;4] }   // camera-relative m, radius m; rgb, cd
#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform { pub info: [u32;4] /* x = count ≤ MAX_LIGHTS */, pub lights: [LightGpu; MAX_LIGHTS] } // 144 B; group 0 binding 2
impl LightUniform { pub fn from_ctx(ctx: &FrameCtx) -> Self; } // first MAX_LIGHTS lights; non-finite lights dropped
// renderer.rs: group-0 layout gains binding 2 (uniform, VERTEX|FRAGMENT|COMPUTE); the Renderer keeps the previous
//              view_proj and camera UniversePos to fill prev_view_proj and cam_delta (f64 difference, then f32).
//              MAX_TIMED_PASSES 32 → 64 (bloom alone times up to 12 passes).
// targets.rs
pub const LENS_MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float; // channels: M1 frame contract
//   hdr usage gains COPY_SRC (the --hdr capture sidecar). The doc comment becomes the M1 frame contract.
// capture.rs
pub fn read_hdr(gpu: &GpuContext, targets: &FrameTargets) -> anyhow::Result<Vec<f32>>; // blocking; RGBA f32, rows top-down, f16 → f32 on the CPU
// context.rs: WANTED_FEATURES gains TEXTURE_COMPRESSION_BC | DUAL_SOURCE_BLENDING (ruling 1)
impl GpuContext { pub fn has_bc_compression(&self) -> bool; pub fn has_dual_source_blending(&self) -> bool; }
// shader.rs (review I7)
pub fn validate_wgsl_with(src: &str, caps: naga::valid::Capabilities) -> Result<naga::Module, ShaderError>;
//   validate_wgsl(src) = validate_wgsl_with(src, Capabilities::default() | SHADER_FLOAT16), unchanged.
//   ShaderLib::compose hoists every `enable …;` line of every composed file to the top of the output, once each,
//   because WGSL directives must precede declarations and imports are placed before the importer's text.
// shared.rs
pub struct SharedInput<T>(std::sync::Arc<std::sync::Mutex<T>>);   // Clone; poisoned lock → into_inner
impl<T: Clone> SharedInput<T> { pub fn new(v: T) -> Self; pub fn set(&self, v: T); pub fn get(&self) -> T;
                                 pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R; }
// env.rs (review m10)
pub const ENV_UNIT_CD_M2: f32 = 1.0e-4;  // env cubemap texel unit: 1e-6 cd/m² → 0.01 (a normal f16); 65504 units = 6.55 cd/m² headroom; writers clamp to 60000 units
pub struct EnvMap { pub texture: wgpu::Texture, pub view: wgpu::TextureView /* Cube, Rgba16Float */, pub sampler: wgpu::Sampler }
pub fn cube_dir(face: u32, uv: glam::Vec2) -> glam::Vec3;   // WebGPU cube-face convention; WGSL twin common/cube.wgsl (B5 and B7 share it; review I9)
// probe.rs
#[derive(Clone, Debug)] pub struct ProbeResult { pub name: String, pub pass: bool, pub detail: String }
pub fn run_compute(gpu: &GpuContext, wgsl: &str, entry: &str, input: &[u8], output_bytes: u64, workgroups: [u32; 3])
    -> anyhow::Result<Vec<u8>>;   // binding 0 read-only storage input, binding 1 storage output; blocking readback
```
WGSL:
- `common/exposure.wgsl` (new, **no bindings**): `struct ExposureUniform`, `const HDR_MAX`, `fn exposure_scale`. B6's resolve pass imports only this file, so it never binds group 0 while it writes the exposure buffer as storage (a uniform + storage use of one buffer in one dispatch is a usage-scope conflict; review m14).
- `common/cube.wgsl` (new): `fn cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32>`, the twin of `uni_gfx::env::cube_dir`.
- `common/camera.wgsl` imports `common/exposure.wgsl`, keeps vars `camera` (binding 0), `exposure` (binding 1), `pixel_dir`, `scene_pre()`, `star_pre()`, and gains the `CameraUniform` v2 members, `struct LightGpu`, `struct LightUniform`, `@group(0) @binding(2) var<uniform> lights: LightUniform;` and `fn sim_time_s() -> f32 { return camera.sim.x; }`.

**Steps**
- [ ] **Step 1: Write the failing tests** (in `uni-gfx` and `uni-core`; m0.1's tests are updated where their sizes change):
  ```rust
  #[test] fn camera_uniform_layout_matches_wgsl() { /* updated m0.1 test: naga size 336 == size_of; all 9 member offsets == offset_of! */ }
  #[test] fn light_uniform_layout_matches_wgsl() { /* size 144; info at 0, lights at 16, stride 32; var lights at group 0 binding 2 */ }
  #[test] fn light_uniform_drops_non_finite_and_caps_count() { /* 6 lights, one NaN → info[0] == 4, NaN one absent */ }
  #[test] fn sim_time_wraps_and_keeps_quarter_second() { /* epoch J2000 + 1e9 s → sim.x == (1e9 mod 2^22) within 0.25 */ }
  #[test] fn cam_delta_is_f64_difference() { /* two frames, camera at 1 Gpc then +1.5 m → cam_delta ≈ (1.5,0,0) ±1e-3 */ }
  #[test] fn shared_input_set_get_and_poison() { /* set/get round trip; a panicked holder does not wedge get() */ }
  #[test] fn compose_hoists_enable_directives() { /* a.wgsl = "enable dual_source_blending;\n#import \"b\"\n…" → output starts with the enable line, exactly once */ }
  #[test] fn validate_wgsl_with_dual_source() { /* a @blend_src(0)/@blend_src(1) fragment output validates with DUAL_SOURCE_BLENDING, fails without */ }
  #[test] fn exposure_wgsl_has_no_bindings() { /* common/exposure.wgsl composes alone: no global var has a binding; HDR_MAX and exposure_scale present */ }
  #[test] fn hdr_target_has_copy_src() { /* FrameTargets usage flags: hdr contains COPY_SRC; lens_mask format Rgba16Float */ }
  #[test] fn wanted_features_include_bc_and_dual_source() {}
  #[test] fn cube_face_dirs_match_webgpu() { /* face centres: +X,−X,+Y,−Y,+Z,−Z; corner (0,0) of +X == normalize(1, 1, 1)·sign table from the WebGPU spec; WGSL twin validates */ }
  #[test] fn camera_view_ron_without_velocity_parses() { /* M0 bookmark RON → velocity_mps == ZERO */ }
  #[test] fn days_since_j2000_roundtrip() { /* parse 2026-11-20T00:00:00Z → days 9819.5008 ±1e-6 (exact: 9819.500 800 74); from_days(..) round trip ±1e-6 s */ }
  #[test] fn bv_to_teff_clamps() { assert_eq!(bv_to_teff(5.0), bv_to_teff(2.0)); assert_eq!(bv_to_teff(f64::NAN), 5800.0); }
  ```
- [ ] **Step 2: Confirm they fail**, then implement the interfaces above. `cargo check --workspace` must pass.
- [ ] **Step 3: Confirm they pass:** `cargo test -p uni-core --lib && cargo test -p uni-gfx --lib`, then `cargo test --workspace --lib --bins` (m0.1's 272 tests stay green, with the size-dependent ones updated).
- [ ] **Step 4: Lead GPU check:** `universe --selftest` all PASS and `universe --capture bookmarks/m0.ron --out captures/a0` with the shadow at 158.745 ± 1 px (the mask widening, the device features and the group-0 change must not regress M0); `take_errors()` empty.
- [ ] **Step 5: Commit** `feat(uni-gfx,uni-core): M1 interfaces (lights, camera v2, lens-mask channels, shared inputs, probes, exposure.wgsl, BC + dual-source features)`.

**Done when:** the workspace compiles, M0 tests, selftest and the M0 shadow check pass, `targets.rs` and CLAUDE.md document the M1 frame contract, and the device is created with BC and dual-source blending on this M3.

---

### Task A0b: Wave A stubs, manifests and shared-file sections (lead)

**Files**
- Scaffold (stubs with `todo!()` bodies, real signatures): `crates/uni-physics/src/{lib,bodies,kepler,frames,standish,pluto,ssb,moon,galilean,saturn_moons,sat_elements,rotation,ephemeris,photometry,smallbody,dust,calc}.rs`; `crates/uni-sim/src/{lib,time,world,registry,snapshot,honesty,starindex}.rs`; `crates/uni-procgen/src/{lib,galaxy,lf}.rs` + `crates/uni-procgen/shaders/galaxy.wgsl` (valid empty functions); `crates/uni-data/src/{athyg,f16,names,exo,texture,maps,rings,smallbody,sats,facts,golden}.rs` and their module lines and re-exports in `crates/uni-data/src/lib.rs`; `tools/bake/src/{athyg,exo,map,bc,rings,sbdb,sats,facts,horizons}.rs` and their subcommand arms in `tools/bake/src/main.rs`; `crates/uni-nav/src/anchor.rs` and the Bookmark v2 fields in `crates/uni-nav/src/bookmark.rs` (`#[serde(default)]`, so A6 starts from a compiling struct).
- Manifests (review I8; no wave agent edits a manifest):
  - workspace `Cargo.toml` `[workspace.dependencies]`: `intel_tex_2 = "=0.5.0"`, `rayon = "=1.12.0"` (round 2; bake only).
  - `tools/bake/Cargo.toml`: `serde`, `serde_json`, `ron`, `bytemuck`, `glam`, `rayon`, `intel_tex_2`, and `image = { workspace = true, features = ["png", "jpeg", "tiff"] }` (bake only; no runtime crate depends on them directly).
  - `crates/uni-data/Cargo.toml`: `serde`, `ron`, `bytemuck`.
  - `crates/{uni-physics,uni-sim,uni-procgen,uni-nav}/Cargo.toml`: the internal deps in "Dependency changes", including `glam.workspace = true` for uni-physics, uni-sim and uni-procgen (uni-nav and uni-data already have it).
- Shared-file sections: marker comments `# --- A7 ---`, `# --- A8 ---`, `# --- A9 ---` in `data/manifest.toml` and matching `<!-- A7 -->`… headings in `ATTRIBUTION.md`; `.gitignore` gains `data/baked/**/*.big`; create `data/golden/.gitkeep`, `data/fonts/.gitkeep`, `data/baked/maps/.gitkeep`.

**Stub contract.** The signatures below are frozen for A1–A9 (bodies `todo!()`):
```rust
// uni-physics (A1)
pub type NaifId = i32;
pub mod bodies { pub struct BodyConst { pub name: &'static str, pub naif: NaifId, pub parent: Option<NaifId>,
                     pub gm_m3s2: f64, pub radii_m: glam::DVec3 /* a ≥ b ≥ c: equatorial, equatorial, polar */ }
                 pub fn major_bodies() -> &'static [BodyConst]; pub fn by_naif(id: NaifId) -> Option<&'static BodyConst>; }
pub mod kepler { #[derive(Clone, Copy, Debug, PartialEq)]
                 pub struct Conic { pub q_m: f64, pub e: f64, pub i_rad: f64, pub node_rad: f64, pub argp_rad: f64,
                                    pub tp: uni_core::Epoch, pub mu_m3s2: f64 }   // angles in the conic's reference frame
                 impl Conic { pub fn from_mean_anomaly(a_m: f64, e: f64, i: f64, node: f64, argp: f64, m0: f64, epoch: uni_core::Epoch, mu: f64) -> Self;
                              pub fn from_state(r: glam::DVec3, v: glam::DVec3, mu: f64, t: uni_core::Epoch) -> Self; // osculating; review I9 (C5, A4, A1 Pluto, C1)
                              pub fn state(&self, t: uni_core::Epoch) -> (glam::DVec3, glam::DVec3);  // reference frame, m and m/s, any e ≥ 0
                              pub fn period_s(&self) -> Option<f64>; }
                 pub fn solve_kepler(m_rad: f64, e: f64) -> f64;             // e < 1
                 pub fn solve_kepler_hyperbolic(m: f64, e: f64) -> f64;     // e > 1
                 pub fn propagate_universal(r0: glam::DVec3, v0: glam::DVec3, mu: f64, dt_s: f64) -> (glam::DVec3, glam::DVec3);
                 pub fn perifocal_to_reference(i: f64, node: f64, argp: f64) -> glam::DMat3; }
pub mod frames { pub const OBLIQUITY_J2000_RAD: f64;
                 pub fn ecliptic_j2000_to_icrs(v: glam::DVec3) -> glam::DVec3;
                 pub fn precession_to_j2000(t: uni_core::Epoch) -> glam::DMat3;        // mean equator/equinox of date → J2000
                 pub fn ecliptic_of_date_to_icrs(lon: f64, lat: f64, dist_m: f64, t: uni_core::Epoch) -> glam::DVec3; }
pub mod standish { #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
                   pub enum Planet { Mercury, Venus, EmBary, Mars, Jupiter, Saturn, Uranus, Neptune }   // Pluto: see `pluto` (review I1)
                   impl Planet { pub fn barycentre_id(self) -> super::NaifId; }                        // 1..=8, Horizons targets (review I2)
                   #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Validity { Table1, Table2, Extrapolated }
                   pub const TABLE2_T_MIN: f64 = -50.0; pub const TABLE2_T_MAX: f64 = 10.0;              // Julian centuries from J2000: 3000 BC – AD 3000
                   pub fn validity(t: uni_core::Epoch) -> Validity;
                   pub fn heliocentric_state_icrs(p: Planet, t: uni_core::Epoch) -> (glam::DVec3, glam::DVec3); }
pub mod pluto { pub const VALID_JD_TDB: (f64, f64) = (2409542.5, 2488068.5);                         // 1885-01-01 – 2099-12-31 (Meeus ch. 37)
                pub fn heliocentric_ecliptic_j2000(t: uni_core::Epoch) -> Option<(f64, f64, f64)>;   // l, b [rad], r [m]; None outside the window
                pub fn heliocentric_state_icrs(t: uni_core::Epoch) -> (glam::DVec3, glam::DVec3);   // two-body outside the window
                pub fn in_window(t: uni_core::Epoch) -> bool; }
pub mod ssb { pub fn sun_from_ssb_icrs(t: uni_core::Epoch) -> glam::DVec3; }
// uni-physics (A2)
pub mod moon { pub fn ecliptic_of_date(t: uni_core::Epoch) -> (f64, f64, f64);    // λ, β [rad], Δ [m], mean equinox of date
               pub fn geocentric_icrs(t: uni_core::Epoch) -> glam::DVec3; }
pub mod galilean { #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Galilean { Io, Europa, Ganymede, Callisto }
                   pub fn jovicentric_icrs(m: Galilean, t: uni_core::Epoch) -> glam::DVec3; }
pub mod saturn_moons { #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum SaturnMoon { Mimas, Enceladus, Tethys, Dione, Rhea, Titan, Hyperion, Iapetus }
                       pub fn saturnicentric_icrs(m: SaturnMoon, t: uni_core::Epoch) -> glam::DVec3; }
pub mod sat_elements { #[derive(Clone, Debug, PartialEq)]
                       pub struct SatElements { pub naif: super::NaifId, pub name: String, pub planet: super::NaifId,
                           pub a_m: f64, pub e: f64, pub argp_deg: f64, pub m_deg: f64, pub i_deg: f64, pub node_deg: f64,
                           pub n_deg_per_day: f64, pub p_apsides_yr: f64, pub p_node_yr: f64, pub epoch: uni_core::Epoch,
                           pub laplace_ra_deg: f64, pub laplace_dec_deg: f64 }
                       pub fn planetocentric_icrs(el: &SatElements, gm_planet: f64, t: uni_core::Epoch) -> glam::DVec3; }
pub mod rotation { pub fn body_to_icrs(naif: super::NaifId, t: uni_core::Epoch) -> Option<glam::DQuat>;
                   pub fn prime_meridian_deg(naif: super::NaifId, t: uni_core::Epoch) -> Option<f64>;
                   pub fn pole_icrs(naif: super::NaifId, t: uni_core::Epoch) -> Option<glam::DVec3>; }
pub mod ephemeris { #[derive(Clone, Copy, Debug, PartialEq, Eq)]
                    pub enum Theory { SunSsb, Standish, MeeusPluto, EarthFromEmb, MeeusMoon, MeeusGalilean, MeeusSaturn, JplMeanElements }
                    #[derive(Clone, Debug)] pub struct EphemBody { pub naif: super::NaifId, pub name: String, pub parent: Option<super::NaifId>, pub theory: Theory }
                    #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum EphemFlag { PlanetsExtrapolated, PlutoExtrapolated, MoonTheoryFallback }
                    pub struct Ephemeris { /* private */ }
                    impl Ephemeris { pub fn new(sats: Vec<super::sat_elements::SatElements>) -> Self;
                                     pub fn bodies(&self) -> &[EphemBody];
                                     pub fn position_rel_parent(&self, naif: super::NaifId, t: uni_core::Epoch) -> Option<glam::DVec3>;  // ICRS m; parent of planets = Sun (10)
                                     pub fn ssb_position(&self, naif: super::NaifId, t: uni_core::Epoch) -> Option<glam::DVec3>;
                                     pub fn flags(&self, t: uni_core::Epoch) -> Vec<EphemFlag>; } }
// uni-physics (A3)
pub mod photometry { #[derive(Clone, Copy, Debug, PartialEq)] pub enum Brdf { Lambert, LunarLambert { l: f64 } }
                     pub fn hg_phase(alpha_rad: f64, g: f64) -> f64;
                     pub fn asteroid_v(h: f64, g: f64, r_au: f64, delta_au: f64, alpha_rad: f64) -> f64;
                     pub fn comet_total_v(m1: f64, k1: f64, r_au: f64, delta_au: f64) -> f64;
                     pub fn reflectance(brdf: Brdf, mu0: f64, mu: f64) -> f64;      // ρ-normalised reflectance factor; CPU twin of body.wgsl
                     pub fn rho_from_geometric_albedo(brdf: Brdf, p_v: f64) -> f64;  // Lambert ρ = 1.5 p; lunar-Lambert p = ρ[L + 2(1 − L)/3]; capped at 1 (review I4)
                     pub fn lambert_phase(alpha_rad: f64) -> f64;
                     pub fn disk_illuminance_lux(brdf: Brdf, albedo: f64, radius_m: f64, sun_lux: f64, phase_rad: f64, dist_m: f64) -> f64;
                     pub fn bolometric_correction_v(teff_k: f64) -> f64;
                     pub fn star_radius_m(abs_mag_v: f64, teff_k: f64) -> f64; }
pub mod smallbody { pub fn conic_from_perihelion(q_au: f64, e: f64, i_deg: f64, node_deg: f64, argp_deg: f64, tp_jd_tdb: f64) -> super::kepler::Conic;
                    pub fn conic_from_mean_anomaly(a_au: f64, e: f64, i_deg: f64, node_deg: f64, argp_deg: f64, m0_deg: f64, epoch_jd_tdb: f64) -> super::kepler::Conic;
                    pub fn heliocentric_state_icrs(c: &super::kepler::Conic, t: uni_core::Epoch) -> (glam::DVec3, glam::DVec3); } // elements are J2000 ecliptic
pub mod dust { #[derive(Clone, Copy, Debug)] pub struct DustParams { pub beta_min: f64, pub beta_max: f64, pub n_beta: u32, pub max_age_days: f64, pub n_age: u32 }
               #[derive(Clone, Copy, Debug)] pub struct DustGrain { pub pos_rel_nucleus_m: glam::DVec3, pub beta: f64, pub age_s: f64, pub weight: f64 }
               pub fn finson_probstein(comet: &super::kepler::Conic, t_obs: uni_core::Epoch, p: &DustParams, production: &dyn Fn(f64) -> f64) -> Vec<DustGrain>;
               pub fn beta_for_grain(radius_m: f64, density_kg_m3: f64, q_pr: f64) -> f64; }
pub mod calc { pub fn local_g(gm: f64, r_m: f64) -> f64; pub fn escape_speed(gm: f64, r_m: f64) -> f64;
               pub fn circular_speed(gm: f64, r_m: f64) -> f64; pub fn hill_radius(a_m: f64, e: f64, m: f64, m_central: f64) -> f64;
               pub fn light_time_s(d_m: f64) -> f64; }
```
```rust
// uni-sim (A4)
pub mod time { pub const WARP_PRESETS: [f64; 5] = [1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12]; pub const MAX_RATE: f64 = 1.0e12;
               pub struct TimeController { /* epoch, rate, paused */ }
               impl TimeController { pub fn new(epoch: uni_core::Epoch) -> Self; pub fn epoch(&self) -> uni_core::Epoch; pub fn set_epoch(&mut self, e: uni_core::Epoch);
                                     pub fn rate(&self) -> f64; pub fn set_rate(&mut self, r: f64); pub fn step_up(&mut self); pub fn step_down(&mut self);
                                     pub fn toggle_pause(&mut self); pub fn is_paused(&self) -> bool;
                                     pub fn advance(&mut self, dt_real_s: f64) -> f64; } }  // returns sim dt [s]
pub mod registry { pub type ObjectId = u32;
                   #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub enum ObjectKind { Sun, Planet, DwarfPlanet, Moon, Asteroid, Comet, Star, ExoHost, BlackHole }
                   #[derive(Clone, Debug)] pub struct ObjectInfo { pub id: ObjectId, pub name: String, pub kind: ObjectKind, pub naif: Option<i32>,
                       pub catalog_index: Option<u32>, pub parent: Option<ObjectId>, pub radii_m: glam::DVec3, pub gm_m3s2: f64,
                       pub frame: Option<uni_core::FrameId>, pub hill_radius_m: f64 } }
pub mod snapshot { pub const POINT_LOD_PX: f64 = 1.5; pub const BLEND_LOD_PX: f64 = 4.0;
    pub fn proj_radius_px(radius_m: f64, dist_m: f64, fov_y_rad: f64, height_px: f64) -> f64; // (h/2)·tan(asin(R/d))/tan(fov/2); R ≥ d → +∞ (review m5)
    #[derive(Clone, Debug)] pub struct SunLight { pub rel_pos_m: glam::DVec3, pub radius_m: f64, pub intensity_cd: f64, pub color: glam::Vec3 }
    #[derive(Clone, Copy, Debug)] pub struct Occluder { pub rel_pos_m: glam::DVec3, pub radius_m: f64 }
    #[derive(Clone, Copy, Debug)] pub struct RingView { pub inner_m: f64, pub outer_m: f64, pub normal: glam::DVec3 }
    #[derive(Clone, Debug)] pub struct BodyView { pub id: super::registry::ObjectId, pub naif: Option<i32>, pub rel_pos_m: glam::DVec3, pub orient: glam::DQuat,
        pub radii_m: glam::DVec3, pub proj_radius_px: f64, pub sun_rel_m: glam::DVec3, pub sun_illuminance_lux: f64,
        pub occluders: Vec<Occluder>, pub ring: Option<RingView>, pub has_atmosphere: bool }
    #[derive(Clone, Debug)] pub struct StarSurfaceView { pub id: super::registry::ObjectId, pub catalog_index: Option<u32>, pub rel_pos_m: glam::DVec3,
        pub orient: glam::DQuat, pub radius_m: f64, pub teff_k: f64, pub luminous_intensity_cd: f64, pub proj_radius_px: f64 }
    #[derive(Clone, Debug)] pub struct PointView { pub id: Option<super::registry::ObjectId>, pub rel_pos_m: glam::DVec3, pub illuminance_lux: f64, pub color: glam::Vec3 }
    #[derive(Clone, Copy, Debug)] pub struct PickItem { pub id: super::registry::ObjectId, pub screen_px: glam::Vec2, pub dist_m: f64, pub priority: u8 }
    #[derive(Clone, Debug, Default)] pub struct Snapshot { pub sun: Option<SunLight>, pub bodies: Vec<BodyView>, pub star_surfaces: Vec<StarSurfaceView>,
        pub points: Vec<PointView>, pub pick: Vec<PickItem>, pub suppressed_stars: Vec<u32>, pub nearest_surface: Option<(super::registry::ObjectId, f64)>,
        pub dominant_frame: uni_core::FrameId } }
pub mod honesty { #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
                  pub enum HonestyFlag { PlanetsExtrapolated, PlutoExtrapolated, StarsExtrapolated, BandModel, ProceduralBelt }
                  pub fn label(f: HonestyFlag) -> &'static str; }
pub mod world { pub struct WorldData { pub stars: Vec<uni_data::Star3>, pub star_names: Option<uni_data::StarNames>, pub sats: Vec<uni_data::SatRecord>,
                    pub comets: Vec<uni_data::Comet>, pub asteroids: Vec<uni_data::AsteroidRecord>, pub asteroid_names: Vec<String>,
                    pub exo: Vec<uni_data::ExoSystem>, pub facts: Vec<uni_data::BodyFacts>, pub warnings: Vec<String> }
                pub fn load_world_data(dir: &std::path::Path) -> WorldData;   // never panics; each missing file → one warning
                pub struct World { pub time: super::time::TimeController, /* private */ }
                impl World { pub fn new(data: WorldData, epoch: uni_core::Epoch) -> Self;
                             pub fn tick(&mut self, dt_real_s: f64);                       // advances time, refreshes cached states
                             pub fn epoch(&self) -> uni_core::Epoch; pub fn frames(&self) -> &uni_core::FrameTree;
                             pub fn objects(&self) -> &[super::registry::ObjectInfo]; pub fn find(&self, name: &str) -> Option<super::registry::ObjectId>;
                             pub fn position(&self, id: super::registry::ObjectId) -> uni_core::UniversePos;
                             pub fn velocity(&self, id: super::registry::ObjectId) -> glam::DVec3;
                             pub fn dominant_frame(&self, pos: &uni_core::UniversePos) -> uni_core::FrameId;
                             pub fn nearest_surface(&self, pos: &uni_core::UniversePos) -> Option<(super::registry::ObjectId, f64)>;
                             pub fn snapshot(&self, view: &uni_core::CameraView, viewport: glam::UVec2) -> super::snapshot::Snapshot;
                             pub fn honesty(&self, view: &uni_core::CameraView) -> Vec<super::honesty::HonestyFlag>;
                             pub fn lenses(&self) -> &[uni_core::Lens];
                             pub fn star_data(&self) -> &[uni_data::Star3];   // AT-HYG stars, then archive-only exoplanet hosts (I11)
                             pub fn exo_systems(&self) -> &[uni_data::ExoSystem]; } }
pub mod starindex { pub struct StarIndex { /* uniform grid, 4 pc cells */ }
                    impl StarIndex { pub fn new(stars: &[uni_data::Star3]) -> Self;
                                     pub fn nearest(&self, stars: &[uni_data::Star3], p_pc: glam::DVec3, years: f64, k: usize) -> Vec<(u32, f64)>; } }
// uni-procgen (A5)
pub mod galaxy { pub struct GalaxyModel { /* parameters */ }
    #[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)] pub struct GalaxyGpu { /* vec4-only; layout test against galaxy.wgsl */ }
    pub const GALAXY_WGSL_PATH: &str = "procgen/galaxy.wgsl"; pub const GALAXY_WGSL_SRC: &str;
    impl GalaxyModel { pub fn milky_way() -> Self; pub fn sun_galactocentric_pc(&self) -> glam::DVec3;
        pub fn icrs_pc_to_galactocentric(&self, p_icrs_pc: glam::DVec3) -> glam::DVec3;
        pub fn star_density(&self, p_gc_pc: glam::DVec3) -> f64;               // stars / pc³
        pub fn v_luminosity_density(&self, p_gc_pc: glam::DVec3) -> f64;      // L☉,V / pc³
        pub fn extinction_v_per_pc(&self, p_gc_pc: glam::DVec3) -> f64;        // A_V mag / pc
        pub fn band_radiance(&self, cam_gc_pc: glam::DVec3, dir_gc: glam::DVec3, v_lim: f64) -> glam::DVec3; // linear Rec.709 cd/m²; CPU twin
        pub fn gpu(&self) -> GalaxyGpu; } }
pub mod lf { pub fn faint_light_fraction(dist_pc: f64, v_lim: f64) -> f64; }   // fraction of V light from stars fainter than v_lim at dist
// uni-nav (A6)
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)] pub enum ExposureChoice { Auto, Manual }
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)] pub enum LensEnvChoice { Band, DebugGrid }
pub struct Bookmark { /* m0.1 fields (name, view, epoch_utc, scene_ev100, star_ev100, lens_scale), then (all #[serde(default)]) */
    pub anchor: Option<String>, pub look_at: Option<String>, pub up: Option<glam::DVec3>, pub time_rate: Option<f64>,
    pub exposure: Option<ExposureChoice>, pub eye_adapted: Option<bool>, pub target: Option<String>,
    pub lens_env: Option<LensEnvChoice> /* None → Band */, pub cruise: Option<bool>, pub highlight_belt: Option<bool> }
pub trait ObjectLookup { fn find_object(&self, name: &str) -> Option<(uni_core::UniversePos, uni_core::FrameId)>; }
pub fn resolve_bookmark(b: &Bookmark, world: &dyn ObjectLookup) -> anyhow::Result<(uni_core::CameraView, uni_core::FrameId)>;
pub struct FlyCamera { pub view: uni_core::CameraView, pub speed_log10: f64, pub look_sensitivity: f64, pub anchor: uni_core::FrameId, pub rel_m: glam::DVec3 }
impl FlyCamera { pub fn reanchor(&mut self, frames: &uni_core::FrameTree, t: uni_core::Epoch, new_anchor: uni_core::FrameId);
                 pub fn sync(&mut self, frames: &uni_core::FrameTree, t: uni_core::Epoch); }   // view.pos = origin(anchor) + rel_m
// uni-data (A7–A9): types and loaders as listed in A7, A8, A9 "Produces".
```

**Steps**
- [ ] **Step 1:** Write the stubs, the manifests and the shared-file sections. `cargo check --workspace` passes; `cargo build -p uni-bake` links (it pulls `intel_tex_2`'s packaged kernels; if the link fails, remove `intel_tex_2` from the bake manifest, record the error, and A8 uses its own encoder).
- [ ] **Step 2:** `cargo test --workspace --lib --bins` stays green (stub crates have no tests yet).
- [ ] **Step 3: Commit** `build: Wave A stubs, manifests and shared-file sections`.

**Done when:** the workspace compiles with every Wave A stub, the bake links, and every file a Wave A agent would otherwise share is pre-split.

---

### Task A1: uni-physics — Kepler, frames, Standish planets, Pluto, SSB, body constants

**Files**
- Modify: `crates/uni-physics/src/{bodies,kepler,frames,standish,pluto,ssb}.rs`
- Test: unit tests in each module; the Horizons tests read `data/golden/horizons_spots.ron` through `uni_data::load_ephem_spots` (A9 implements the loader and writes the file in its Step 1)

**Interfaces**
- Consumes: `uni_core::{Epoch, AU, GM_SUN, DAY}`, `uni_data::{EphemSpot, load_ephem_spots}` (tests only).
- Produces: the A0b `bodies`, `kepler` (including `Conic::from_state`, used by A2, A4, C1 and C5), `frames`, `standish`, `pluto`, `ssb` signatures, implemented.

**Physics (cite in code)**
- Standish (JPL "Approximate Positions of the Planets", https://ssd.jpl.nasa.gov/planets/approx_pos.html, Standish & Williams): physics research §6 steps 1–8. Table 1 for 1800–2050; Tables 2a + 2b (with the b, c, s, f terms for **Jupiter–Neptune**) for 3000 BC – AD 3000. Copy the table values from the JPL page into the code with the URL and access date. The page no longer lists Pluto (review I1).
- **Outside 3000 BC – AD 3000** (T outside [`TABLE2_T_MIN`, `TABLE2_T_MAX`] = [−50, +10] Julian centuries; review C1): a, e, I, ϖ and Ω are frozen at the nearer window edge, T_e = clamp(T, −50, +10); only the mean longitude L keeps advancing at its Table 2a rate; the Table 2b terms (b T², c cos fT, s sin fT) are evaluated at T_e; the mean anomaly is reduced mod 360° before solving; `Validity::Extrapolated`. Evaluating the linear rates instead pushes e past 1 (Saturn at −2.95 × 10⁵ yr, Jupiter at +5.28 × 10⁵ yr, Mars at +9.91 × 10⁵ yr) and I to tens of degrees (review C1, `checks.py`). At the window edges (Table 2a) every e stays in [0.0062, 0.206] (Venus at +10 to Mercury at +10) and every |I| at or below 7.31° (Mercury at −50); e.g. Saturn e = 0.05551 + 0.00032044 × 50 = 0.0715 at T = −50.
- **Pluto** (ruling round 2, review I1): Meeus, *Astronomical Algorithms* 2nd ed., ch. 37 (periodic terms, valid 1885–2099), heliocentric ecliptic J2000 l, b, r, rotated to ICRS. These positions are Pluto's system barycentre (Horizons target 9). Outside 1885–2099: a two-body conic, `Conic::from_state` of the ch. 37 state at the nearer window edge with μ = GM☉ + GM(Pluto system) (DE440), propagated with `Conic::state`; `pluto::in_window` false raises `EphemFlag::PlutoExtrapolated` and the amber Pluto label (A4).
- Ecliptic J2000 → ICRS: rotate about +x by ε₀ = 84381.448″ (IAU 1976, Lieske et al. 1977). The 23 mas frame bias between J2000 mean equator and ICRS is below Standish's error and is ignored (say so in a comment).
- Precession: IAU 1976 ζ_A, z_A, θ_A polynomials (Lieske et al. 1977; Meeus ch. 21); `precession_to_j2000` is the inverse rotation (date → J2000). Its doc comment states the validity (a fit for a few centuries around J2000; the error grows with T³), that A2 uses it only inside the Moon window, and that Vondrák et al. 2011 long-term precession is the upgrade path (review m21).
- SSB: `sun_from_ssb_icrs = −Σᵢ (GMᵢ / (GM☉ + Σ GMⱼ)) · r_i,helio` over the Standish bodies Mercury–Neptune (EMB carries Earth + Moon). Pluto's pull on the SSB, GM_Pluto-sys/GM☉ × 39.5 AU = 43 km, is omitted (comment). GM values from DE440 (Park et al. 2021, AJ 161, 105).
- Bodies: `radii_m` from IAU WGCCRE 2015 (Archinal et al. 2018, CMDA 130:22) — Sun, eight planets, Pluto, Moon, Galileans, Saturn's eight major moons, Phobos, Deimos, Triton, Titania, Oberon, Charon. GM from DE440 / JPL satellite physical parameters. Velocities: analytic derivative of the conic.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn kepler_meeus_30a() { assert!((solve_kepler(5f64.to_radians(), 0.1).to_degrees() - 5.554589).abs() < 1e-6); } // Meeus Ex. 30.a
  #[test] fn kepler_residual_all_e() { /* e ∈ {0, 0.1, 0.5, 0.9, 0.99, 0.999}, 1000 M values: |E − e sinE − M| < 1e-13 */ }
  #[test] fn hyperbolic_residual() { /* e ∈ {1.001, 1.5, 5}: |e sinh H − H − M| < 1e-12·max(1,|M|) */ }
  #[test] fn universal_vallado_2_4() { /* r0=(1131.340,−2282.343,6672.423) km, v0=(−5.64305,4.30333,2.42879) km/s, μ=398600.4418, Δt=40 min
                                         → r=(−4219.7527,4363.0292,−3958.7666) km ±1e-3, v=(3.689866,−1.916735,−6.112511) km/s ±1e-6 (Vallado 2013 Ex. 2-4) */ }
  #[test] fn universal_conserves_energy_and_h() { /* e ∈ {0.3, 0.9999, 1.0, 1.7}: specific energy and |r×v| rel. error < 1e-10 after ±100 periods or 1e9 s */ }
  #[test] fn conic_state_matches_universal() { /* Conic::state(t) vs propagate_universal from perihelion state: < 1 mm for 1 AU orbits */ }
  #[test] fn conic_from_state_roundtrip() { /* e ∈ {0, 0.2, 0.97, 1.0, 1.5}: from_state(state(t0)).state(t1) == state(t1) < 1 mm at 1 AU; e = 0 and i = 0 handled (node/argp convention documented) */ }
  #[test] fn ecliptic_pole_in_icrs() { /* ecliptic_j2000_to_icrs(Z) → RA 270.0°, Dec 66.560 709° ±1e-6° */ }
  #[test] fn precession_pole_one_century() { /* angle(precession_to_j2000(J2100)·Z_date, Z) == θ_A(T=1) = 2003.8423″ ±0.01″ */ }
  #[test] fn standish_within_error_table() { /* every Horizons spot of targets 1..=8 (barycentres) @10 in 1800–2050 (review I2; 399 is not EMB and is excluded):
        heliocentric λ, φ and ρ errors ≤ the JPL 1800–2050 table:
        Mercury 15″ 1″ 1 000 km · Venus 20″ 1″ 4 000 km · EMB 20″ 8″ 6 000 km · Mars 40″ 2″ 25 000 km ·
        Jupiter 400″ 10″ 600 000 km · Saturn 600″ 25″ 1.5e6 km · Uranus 50″ 2″ 1e6 km · Neptune 10″ 1″ 2e5 km; print the measured table */ }
  #[test] fn standish_table2_window() { /* spots at JD 2086307.5 (AD 1000-01-01, Julian calendar: Horizons reads pre-1582 dates as Julian) and
        JD 2634166.5 (AD 2500-01-01) within the JPL page's 3000 BC – AD 3000 error table (record every row from the page; e.g. Jupiter 600″, Saturn 1 000″) */ }
  #[test] fn standish_far_dates_finite_and_flagged() { /* t = J2000 ± 1e6 yr and ± 1e7 yr: every planet finite, 0 ≤ e < 1, |I| < 20°, 0.2 AU < |r| < 60 AU,
        a, e, I, ϖ, Ω equal their values at the nearer window edge, validity == Extrapolated */ } // Review Focus 1, review C1
  #[test] fn sun_ssb_offset_vs_horizons() { /* spot 10@0 at 1950, 2000, 2026-11-20: |Δ| < 3000 km. Budget (comment): Standish λ errors
        Jupiter 400″ → 1 440 km and Saturn 600″ → 1 180 km of SSB shift, ≈ 1 900 km RSS, ≈ 2 600 km if aligned (review m1) */ }
  #[test] fn emb_distance_annual_range() { /* over 2026, |EMB| ∈ [0.9832, 1.0168] AU */ }
  #[test] fn pluto_meeus_37a() { /* Meeus Ex. 37.a (record the book's date and l, b, R in the test; verify): ±1e-5° and ±1e-6 AU */ }
  #[test] fn pluto_vs_horizons() { /* target 9 @10 at JD 2415020.5, 2433282.5, 2451545.0, 2461364.5008, 2469807.5, 2484417.5 (2090-01-01):
        within the accuracy Meeus states for ch. 37 (record it; verify); print the measured errors */ }
  #[test] fn pluto_outside_window_is_two_body_and_flagged() { /* 1800 and 2200: finite, 29 AU < r < 50 AU, in_window false;
        state just inside vs just outside each window edge differs by < 1 km */ } // Review Focus 1
  #[test] fn radii_order_and_flattening() { /* Saturn b/a = 54364/60268 ±1e-5; Jupiter 66854/71492; every a ≥ b ≥ c > 0 */ }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-physics --lib` (the Horizons tests also fail until A9 lands `horizons_spots.ron`; they load the file through `uni_data::data_dir()` and fail with "golden file missing: …", never skip).
- [ ] **Step 3: Implement.** `propagate_universal`: Stumpff C(z), S(z) with series for |z| < 1e-6; Newton on χ with a Laguerre fallback; for e < 1 reduce dt modulo the period first. Every public function returns finite values for finite input (NaN in → NaN out is allowed only for NaN input; document it).
- [ ] **Step 4: Confirm they pass:** `cargo test -p uni-physics --lib` (Horizons tests once A9's file exists).
- [ ] **Step 5: Commit** (pathspec, own files).

**Done when:** all tests pass, including the error-table test printed in the report, and the measured worst λ, φ and ρ error per planet and for Pluto is listed next to the source's table.

---

### Task A2: uni-physics — Moon, satellites, IAU rotation, Ephemeris (wave A-ii)

**Files**
- Modify: `crates/uni-physics/src/{moon,galilean,saturn_moons,sat_elements,rotation,ephemeris}.rs`

**Interfaces**
- Consumes: A1 (`kepler`, `frames`, `standish`, `pluto`, `ssb`, `bodies`), `uni_data::{SatRecord, load_ephem_spots}` (tests), `uni_core::Epoch`.
- Produces: the A0b `moon`, `galilean`, `saturn_moons`, `sat_elements`, `rotation`, `ephemeris` signatures. `Ephemeris::new` converts nothing itself: uni-sim maps `uni_data::SatRecord` → `SatElements`.

**Physics (cite in code)**
- Moon: Meeus ch. 47 (truncated ELP-2000/82, Chapront-Touzé & Chapront), geometric λ, β, Δ of the mean equinox of date; then obliquity of date (Meeus 22.2) and `frames::precession_to_j2000` → ICRS. Earth = EMB − (GM_Moon/(GM_Earth+GM_Moon))·r_Moon.
- Galilean moons: Meeus ch. 44, the higher-accuracy method (Lieske's E5 theory as given by Meeus), rotated from Jupiter's equatorial frame to ICRS with the pole from `rotation`.
- Saturn's eight major moons: Meeus ch. 46 (Harper & Taylor 1993 theory).
- Every other moon in `sat_elements.ron`: JPL SSD mean elements referred to the local Laplace plane (pole `laplace_ra/dec`), with apsidal and nodal precession periods (https://ssd.jpl.nasa.gov/sats/elem/). JPL states these are not for precise ephemerides; the info card labels them.
- **Phobos** (ruling round 2, review I10): its tidal secular acceleration is added as a quadratic term in mean longitude from Jacobson 2010 (AJ 139, 668; check whether the paper's value, ≈ 1.27 × 10⁻³ °/yr², is ṅ or ½ṅ and cite the equation). Without it Phobos drifts ≥ 1.6° along track at ±50 yr.
- **Pluto system:** Pluto (999) = Pluto barycentre (A1) − (GM_Charon/(GM_Pluto + GM_Charon))·r_Charon, Charon from its mean elements relative to Pluto (the offset is ≈ 2 100 km, larger than Pluto's radius).
- Validity fall-back: outside ±3000 yr from J2000 the Meeus series (polynomials in T up to T⁴) are replaced by the same moon's mean-element orbit, and `EphemFlag::MoonTheoryFallback` is raised.
- Rotation: IAU WGCCRE 2015 (Archinal et al. 2018 + 2019 erratum): α₀, δ₀, W with the periodic terms for the Moon (E1–E13), Mars, Jupiter, Saturn, Neptune and the listed moons. Body-fixed → ICRS: R = R_z(α₀ + 90°) · R_x(90° − δ₀) · R_z(W) (inverse of the report's ICRF → body form; the Earth test pins the sign).
- **Poles outside 3000 BC – AD 3000** (review C1): α₀, δ₀ and their periodic terms are evaluated with T (and d) clamped to [`standish::TABLE2_T_MIN`, `TABLE2_T_MAX`]; only W's linear term advances with the unclamped d (its periodic terms are bounded sinusoids). A quadratic secular term in W is not bounded, so it is evaluated at the clamped d (re-review r1 minor 3): the Moon's −1.4 × 10⁻¹² d² (WGCCRE 2015) is −1.4 × 10⁻¹² × (−50 × 36 525)² = −4.67° at the 3000 BC edge and −1.4 × 10⁻¹² × (10 × 36 525)² = −0.19° at the AD 3000 edge, where unclamped at 10⁷ yr (d = 3.65 × 10⁹) it would be −1.87 × 10⁷ °, finite but meaningless; A2 lists any other quadratic W term in the report and clamps it the same way. The linear pole terms otherwise move Earth's pole 55.7° from J2000 after 10 kyr (δ₀ = 90° − 0.557°·T), where the real pole precesses on a 23.4° circle; clamped, it stays within 0.557° × 50 = 27.85° (3000 BC edge) and 0.557° × 10 = 5.57° (AD 3000 edge).

**Steps**
- [ ] **Step 1: Write the failing tests.** Ruling 10: the Horizons comparisons print the measured worst error per moon (km and as a fraction of a) and fail only above a ceiling; the measured table goes in the report, and B9's intent tests check the projected error is < 1 px at every bookmark that shows the moon.
  ```rust
  #[test] fn moon_meeus_47a() { /* 1992-04-12 0h TD: λ = 133.162655° ±1e-5, β = −3.229126° ±1e-5, Δ = 368409.7 km ±0.5 (Meeus Ex. 47.a) */ }
  #[test] fn moon_vs_horizons() { /* 301@399 spots 1950–2050: print measured; ceiling 60″ and 60 km */ }
  #[test] fn galileans_vs_horizons() { /* 501–504@599: print measured; ceiling 1 000 km */ }
  #[test] fn saturn_moons_vs_horizons() { /* 601–608@699: print measured; ceiling Titan 4 000 km, others 2 % of a */ }
  #[test] fn mean_element_moons_vs_horizons() { /* 401 (with the secular term), 402, 801, 703, 704, 901@999 at 1950–2050: print measured; ceiling 5 % of a */ }
  #[test] fn earth_rotation_rate() { /* prime_meridian_deg(399, J2000) == 190.147 ±1e-9; rate over 1 day == 360.9856235 ±1e-7 */ }
  #[test] fn moon_rotation_rate() { /* mean W rate 13.17635815 °/d ±1e-6 */ }
  #[test] fn earth_pole_at_j2000_is_icrs_z() { /* pole_icrs(399, J2000) · Z > cos(1e-6) */ }
  #[test] fn subsolar_point_matches_independent_formula() { /* 2026-11-20T12:00Z: subsolar lon/lat from Ephemeris + rotation vs Meeus ch. 25 low-accuracy Sun
        + GMST (IAU 1982), implemented in the test: ≤ 0.3° (review computed 0.087°) */ }
  #[test] fn theories_fall_back_far_from_epoch() { /* t = J2000 ± 1e6 yr: Moon 356 000 < Δ < 407 000 km; Io 4.1e5 < r < 4.3e5 km; finite; MoonTheoryFallback flagged */ } // Review Focus 1
  #[test] fn poles_frozen_outside_window() { /* t = J2000 ± 1e6 and ± 1e7 yr: Earth's pole within 30° of J2000's and equal to its value at the nearer window edge;
        every body's pole finite and unit; W still advances (Earth's rate 360.9856235 °/d ±1e-6) */ } // Review Focus 1, review C1
  #[test] fn moon_w_quadratic_term_clamped() { /* t = J2000 ± 1e7 yr: the Moon's W d² term equals its value at the nearer window edge
        (−4.67° at T = −50, −0.19° at T = +10); W finite; mean rate still 13.17635815 °/d ±1e-6 */ } // re-review r1 minor 3
  #[test] fn ephemeris_parent_chain() { /* ssb_position(399) == ssb(Sun) + rel(399); Moon's parent is 399; Charon's parent is 999; every body's chain reaches Sun */ }
  #[test] fn pluto_body_offset_from_barycentre() { /* |ssb(999) − ssb(barycentre)| ≈ 2 100 km ±10 %, opposite Charon */ }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-physics --lib`.
- [ ] **Step 3: Implement.** Term tables as `const` arrays with the source and page in a comment. `Ephemeris` caches nothing across epochs (uni-sim caches per tick).
- [ ] **Step 4: Confirm they pass.** Same command.
- [ ] **Step 5: Commit** (pathspec).

**Done when:** all tests pass and the report lists the measured worst error per moon against Horizons, with the ceilings, as ruling 10 requires.

---

### Task A3: uni-physics — photometry, small bodies, Finson–Probstein dust, calculators (wave A-ii)

**Files**
- Modify: `crates/uni-physics/src/{photometry,smallbody,dust,calc}.rs`

**Interfaces**
- Consumes: A1 `kepler`, `frames`; `uni_core::{AU, GM_SUN, M_V_SUN, TEFF_SUN_K, R_SUN_M, star_illuminance_lux}`; tests only: A9's `data/golden/sky_spots.ron` (`SkySpot.apmag`, `r_au`, `phase_deg`) and `data/baked/body_facts.ron` (`geometric_albedo`).
- Produces: the A0b `photometry`, `smallbody`, `dust`, `calc` signatures.

**Physics (cite in code)**
- H, G system: Bowell et al. 1989 (Asteroids II, 524): Φᵢ = exp(−Aᵢ tan^Bᵢ(α/2)), A = (3.33, 1.87), B = (0.63, 1.22); V = H + 5 log₁₀(rΔ) − 2.5 log₁₀((1 − G)Φ₁ + GΦ₂).
- Comet total magnitude: m = M1 + 5 log₁₀Δ + K1 log₁₀ r (JPL SBDB convention).
- Lunar-Lambert reflectance (McEwen 1991, Icarus 92, 298): ρ·[2L μ₀/(μ₀ + μ) + (1 − L) μ₀]; L = 1 is Lommel–Seeliger, L = 0 Lambert.
- **Albedo calibration (review I4):** visual brightness follows the V-band geometric albedo through the BRDF, not the Bond albedo. At phase 0 the reflectance factor is ρ[L + (1 − L)μ] and the projected disk's mean μ is 2/3, so p = ρ[L + 2(1 − L)/3]: Lambert ρ = 1.5 p, lunar-Lambert (L = 0.9) ρ = p/0.9667. ρ is capped at 1 (Venus: p_V = 0.689 > 2/3 gives ρ = 1.03; the cap costs 2.5 log₁₀(0.689/0.667) = 0.036 mag). Calibrating to the Bond albedo instead makes Jupiter 0.93 mag, Saturn 0.85, Uranus 0.97 and Earth 0.64 mag too faint.
- Lambert sphere phase function: Φ(α) = [sin α + (π − α) cos α]/π.
- `disk_illuminance_lux`: numerical integral of the reflectance over the visible lit disk (≥ 256² samples), the CPU twin of what `body.wgsl` puts on screen.
- Bolometric correction: Torres 2010 (AJ 140, 1158) polynomial in log T_eff (Flower 1996 coefficients as corrected), with M_bol☉ = 4.74 (IAU 2015 B2). `star_radius_m` from L = 10^(−0.4(M_V + BC − 4.74)) L☉ and Stefan–Boltzmann.
- Dust: Finson & Probstein 1968 (ApJ 154, 327): grains leave the nucleus with zero relative velocity at t_e = t_obs − age and move on two-body orbits with μ(1 − β): `Conic::from_state(r_nucleus(t_e), v_nucleus(t_e), μ(1 − β), t_e)` then `Conic::state(t_obs)` (an elliptic or hyperbolic Kepler solve per grain, cheaper than universal variables; review I13). β ∈ [β_min, β_max] with β_max < 1 (repulsive β ≥ 1 is excluded; comment why). `weight` = production(r_h(t_e)) × a size-distribution weight (dn/ds ∝ s^−3.5, Dohnanyi 1969, mapped to β).
- β: Burns, Lamy & Soter 1979 (Icarus 40, 1): β = 5.7×10⁻⁵ Q_pr / (ρ[g cm⁻³] · s[cm]).

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn hg_zero_phase() { assert!((asteroid_v(3.33, 0.12, 2.0, 1.0, 0.0) - (3.33 + 5.0 * 2f64.log10())).abs() < 1e-12); }
  #[test] fn lambert_phase_values() { assert!((lambert_phase(0.0) - 1.0).abs() < 1e-12); assert!((lambert_phase(FRAC_PI_2) - 1.0 / PI).abs() < 1e-12); }
  #[test] fn lunar_lambert_limits() { /* L=0 → μ0; L=1, μ0=μ → 1 (per unit ρ) */ }
  #[test] fn rho_from_geometric_albedo_cases() { /* Lambert p 0.538 → 0.807; lunar-Lambert L 0.9, p 0.12 → 0.1241; Lambert p 0.689 → 1.0 (cap);
        disk_illuminance_lux at α = 0 with that ρ == p·E☉·(R/d)² ±1e-3 for both BRDFs */ }
  #[test] fn disk_flux_matches_point_flux() { /* Lambert sphere: disk_illuminance_lux vs analytic (2/3)ρ·E☉·(R/d)²·Φ(α) for α ∈ {0, 30°, 90°, 150°}: rel. error < 1e-3 */ } // Review Focus 4
  #[test] fn planet_magnitudes_vs_horizons() { /* A9's observer spots at 2026-11-20 00:00 UTC for 199, 299, 499, 599, 699, 799, 899 (Horizons quantities 9, 19, 20, 24):
        ρ = rho_from_geometric_albedo(brdf, p_V from body_facts), brdf as B1 uses it (Lambert for bodies with atmospheres and giants; lunar-Lambert L 0.9 for Mercury);
        m = −2.5 log10(E / star_illuminance_lux(0)) from disk_illuminance_lux at Horizons' r, Δ and phase: |m − APmag| ≤ 0.3,
        ≤ 0.5 for Mercury, Venus and Saturn (non-Lambertian phase curves; Saturn's rings); print the table */ } // review I4
  #[test] fn sun_radius_from_magnitude() { assert!((star_radius_m(M_V_SUN, TEFF_SUN_K) / R_SUN_M - 1.0).abs() < 0.03); }
  #[test] fn beta_half_micron() { assert!((beta_for_grain(0.5e-6, 1000.0, 1.0) - 1.14).abs() < 0.01); } // s = 0.5 µm radius → 5.7e-5/(1·5e-5 cm)
  #[test] fn beta_zero_grain_stays_on_nucleus() { /* β = 0, ages up to 30 d: |pos_rel| < 1 m */ }
  #[test] fn small_beta_drifts_antisunward() { /* β = 0.1, age 1 d at 1 AU: displacement · r̂_sun ≈ ½β(GM☉/r²)t² = 2.21e6 m ±5% */ }
  #[test] fn hyperbolic_comet_conic_finite() { /* e = 1.0002 (near-parabolic) and 1.5: state finite ±10 yr from perihelion */ }
  #[test] fn halley_1986_vs_horizons() { /* SBDB 1P elements → heliocentric position at 1986-03-13 within 0.5% of r (two-body from its epoch; record measured) */ }
  #[test] fn escape_speed_earth() { assert!((escape_speed(3.986004418e14, 6.3781e6) - 11_180.0).abs() < 5.0); }
  #[test] fn hill_radius_earth() { assert!((hill_radius(AU, 0.0, 5.972e24, 1.989e30) / 1.497e9 - 1.0).abs() < 0.01); }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-physics --lib`.
- [ ] **Step 3: Implement** as above.
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** all tests pass; the Halley test's measured error and the planet-magnitude table are reported.

---

### Task A4: uni-sim — time warp, world registry, frame tree, snapshot (wave A-ii)

**Files**
- Modify: `crates/uni-sim/src/{lib,time,world,registry,snapshot,honesty,starindex}.rs`

**Interfaces**
- Consumes: `uni_physics::{ephemeris::Ephemeris, rotation, bodies, smallbody, photometry, calc, kepler::Conic}`, `uni_data` loaders (A7–A9), `uni_core::{FrameTree, FrameMotion, UniversePos, CameraView, Epoch, Lens}`.
- Produces: the A0b `uni-sim` signatures. Tests use a fake ephemeris so they do not depend on A2's numbers: `World` is generic over `trait BodySource { fn bodies(&self) -> Vec<(i32, Option<i32>)>; fn position_rel_parent(&self, naif: i32, t: Epoch) -> Option<DVec3>; fn orient(&self, naif: i32, t: Epoch) -> Option<DQuat>; }` with `impl BodySource for Ephemeris` (private generic; the public `World::new` uses `Ephemeris`).

**Rules**
- **Frames:** ROOT (SSB, ICRS) → `Sun` (`FrameMotion::Custom`, `sun_from_ssb`) → each planet (heliocentric) → each moon. All frames translate only (ICRS axes); body orientation comes from `rotation`. `TestBH-1` (M0: 10 M☉ at SSB + 1000 AU) stays as an object with a `Lens`.
- **Dominant frame:** the deepest body whose Hill sphere (`calc::hill_radius` about its parent) contains the position; the Sun's sphere is 2.3×10⁵ AU (Galactic tidal radius, Chebotarev 1965); outside it, ROOT.
- **Data mapping:** `uni_data::SatRecord` → `uni_physics::sat_elements::SatElements` (km → m, JD → `Epoch`); comets and asteroids → `uni_physics::smallbody` conics (heliocentric, J2000 ecliptic → ICRS); catalogue stars → positions with proper motion `pos + v·Δt` (f64), luminous intensity from `M_V`, photosphere radius from `photometry::star_radius_m(M_V, bv_to_teff(B−V))`. Stars flagged "no velocity" (A7) stay fixed; "no radial velocity" stars move tangentially only (the star card says so).
- **Archive-only exoplanet hosts (review I11, round 2):** for every `ExoSystem` with `star_index == None` and a finite `dist_pc` and `v_mag`, `World::new` appends a `Star3` built from the archive's RA, Dec, distance, V and T_eff (zero velocity, flags "exoplanet host", "from archive", "no velocity"), so the host is drawn as a star and its card says `Host star from NASA Exoplanet Archive (not in AT-HYG)`. `star_data()` returns the AT-HYG stars followed by these.
- **Registry:** Sun, planets, dwarf planets, moons (from the ephemeris), comets and asteroids (one object each; asteroids as objects only when H ≤ 10 or named in `body_facts.ron`, the rest are points owned by `uni-sky`), exoplanet hosts (one object per `ExoSystem`, positioned from its `Star3`), named stars (BSC5/AT-HYG proper names), TestBH-1. Names are case-insensitive unique; `find("mars")` works.
- **Snapshot** (per frame, from the camera view and internal viewport): `proj_radius_px = (h/2)·tan(asin(R/d))/tan(fov_y/2)` for the largest radius (a sphere subtends asin(R/d) and the projection is rectilinear; the m0-era atan form overstated it by ~10 % at fov 60°; review m5), the same formula as the WGSL; a body is in `bodies` iff `proj_radius_px ≥ POINT_LOD_PX`, else in `points` with `illuminance_lux = photometry::disk_illuminance_lux(brdf, rho_from_geometric_albedo(brdf, p_V), …)` — never both. The Sun and catalogue stars nearer than 0.1 pc go to `star_surfaces` when ≥ `POINT_LOD_PX`, else `points`; their catalogue indices go to `suppressed_stars` (≤ 8). `occluders`: parent and up to 3 siblings that can eclipse the body. `ring`: Saturn's ring radii from A8's `RingProfile` (inner/outer). `pick`: every body, comet, named asteroid, the 2000 brightest named stars on screen, exoplanet hosts and TestBH-1, projected to internal pixels. CPU budget: ≤ 2 ms per snapshot (measured in the gate).
- **Honesty:** `PlanetsExtrapolated` outside 3000 BC – AD 3000; `PlutoExtrapolated` outside 1885–2099 only while Pluto or Charon is in the view frustum or picked; `StarsExtrapolated` beyond ±100 000 yr; `BandModel` while the band is enabled; `ProceduralBelt` exists here as a label, the app raises it (C6).
- **Time:** `advance` multiplies by the rate (paused → 0), clamps |rate| ≤ `MAX_RATE`, ignores NaN rates; `step_up`/`step_down` move between `WARP_PRESETS`.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn warp_presets_and_pause() { /* new → 1×; step_up ×4 → 1e12; step_up again stays; toggle_pause → advance returns 0 */ }
  #[test] fn max_warp_ten_thousand_frames_finite() { /* rate 1e12, 10 000 ticks of 1/60 s (1.67e14 s = 5.28 Myr) with the real Ephemeris: every position finite;
        every planet's osculating e (Conic::from_state about the Sun) < 1; honesty contains PlanetsExtrapolated */ } // Review Focus 1, review C1
  #[test] fn world_data_missing_files_warns() { /* load_world_data(empty temp dir) → empty vectors, one warning per missing file, no panic */ }
  #[test] fn find_is_case_insensitive() { /* find("MARS") == find("mars") */ }
  #[test] fn dominant_frame_nesting() { /* fake bodies: point 1e5 km from a moon → moon's frame; 1e7 km from planet → planet; 5 AU from everything → Sun; 1e6 AU → ROOT */ }
  #[test] fn proj_radius_uses_asin_tan() { /* R/d = 0.5, fov 60°, h 1080: (540·tan 30°/tan 30°) = 540 px exactly; R ≥ d → +∞ */ }
  #[test] fn body_is_point_or_resolved_never_both() { /* sweep camera distance 1e5..1e12 m from a 1e6 m body: id appears in exactly one of bodies/points */ } // Review Focus 4
  #[test] fn near_star_goes_to_star_surfaces_and_is_suppressed() { /* camera 1 AU from a catalogue star → star_surfaces has it, suppressed_stars contains its index */ }
  #[test] fn sun_point_from_alpha_cen() { /* camera at α Cen A's catalogue position: the Sun is a PointView with illuminance == star_illuminance_lux(M_V_SUN + 5 log10(d/10 pc)) ±0.5% */ }
  #[test] fn archive_hosts_appended_and_flagged() { /* an ExoSystem without star_index (V 18.8 at 12.47 pc) → one extra Star3 with the archive flag, M_V = 18.32 ±0.01;
        from 0.2 AU it is in star_surfaces or points */ } // review I11
  #[test] fn snapshot_rel_positions_are_f64_camera_relative() { /* camera at Neptune: |rel_pos(Neptune) − (pos(Neptune) − cam)| < 1e-3 m */ }
  #[test] fn honesty_flags() { /* epoch +5000 yr → PlanetsExtrapolated; +2e5 yr → StarsExtrapolated; 2200 with Pluto in view → PlutoExtrapolated, without → not;
        labels match the exact strings in this plan */ }
  #[test] fn star_index_nearest() { /* 1000 random stars: nearest() equals brute force for 50 queries */ }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-sim --lib`.
- [ ] **Step 3: Implement.** Cache all body states once per `tick` (one ephemeris evaluation per body per frame).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** all tests pass with the fake source and, once A2 is merged, with the real `Ephemeris`.

---

### Task A5: uni-procgen — galaxy model v0 with a WGSL twin

**Files**
- Modify: `crates/uni-procgen/src/{lib,galaxy,lf}.rs`, `crates/uni-procgen/shaders/galaxy.wgsl`

**Interfaces**
- Consumes: `uni_core::{icrs_to_galactic, PC, blackbody_linear_srgb, star_illuminance_lux}`; `naga` (test only).
- Produces: the A0b `galaxy` and `lf` signatures. WGSL in `galaxy.wgsl` (path `procgen/galaxy.wgsl`): `struct GalaxyGpu` (twin), `fn gal_density(p_gc_pc: vec3<f32>) -> f32`, `fn gal_luminosity(p_gc_pc: vec3<f32>) -> f32`, `fn gal_extinction(p_gc_pc: vec3<f32>) -> f32`, `fn gal_band_radiance(cam_gc_pc: vec3<f32>, dir: vec3<f32>, v_lim: f32) -> vec3<f32>`. B5 binds `GalaxyGpu` at its own group and imports this file.

**Model (procgen research §1.1–1.2; cite each parameter, mark "verify" ones from the brief and check them against the papers):**
- Galactocentric frame: origin at Sgr A*, +x from the Sun's projection toward the centre, +z toward the north Galactic pole; Sun at (−R₀, 0, z☉), R₀ = 8.277 kpc (GRAVITY 2022), z☉ = 25 pc (Jurić et al. 2008). Directions via `uni_core::icrs_to_galactic` (the small tilt from z☉ is ignored; say so).
- Thin + thick disk + halo: Jurić et al. 2008 (L₁ 2.6 kpc, H₁ 300 pc; L₂ 3.6 kpc, H₂ 900 pc, f 0.12; halo f_h 0.0051, q 0.64, n_h 2.77), normalised to ≈ 0.1 stars/pc³ at the Sun (McKee et al. 2015).
- Bar: triaxial exponential, half-length 5 kpc, angle 27° to the Sun–GC line, mass ≈ 1.5×10¹⁰ M☉ (Bland-Hawthorn & Gerhard 2016).
- Arms: four log spirals (Reid et al. 2019 pitch angles and reference radii), Gaussian cross-section 0.4 kpc, 15% contrast for the light, 3× for dust.
- Dust: exponential layer h_z = 120 pc following the arms, normalised so A_V ≈ 1 mag/kpc in the plane near the Sun (Drimmel & Spergel 2001; verify).
- Light: `v_luminosity_density` = density × the mean V luminosity per star from a cited local V-band luminosity function (`lf.rs`; e.g. Just & Jahreiß 2010; record the table and its source). Component colour: a blackbody colour temperature per component (thin disk 5000 K, thick disk/halo 4500 K, bar 4200 K; v0 approximations, commented).
- Band radiance (the CPU twin B5's shader must match): ∫₀^{30 kpc} j(s)·f_faint(s)·e^{−τ(s)} ds along the ray with 128 log-spaced steps from 1 pc, j in cd/m² per pc via `star_illuminance_lux` units, `f_faint = lf::faint_light_fraction(s, v_lim)` so only stars fainter than the catalogue limit (v_lim = 10) make the band — the catalogue stars are drawn individually.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn sun_position() { let g = GalaxyModel::milky_way(); assert!((g.sun_galactocentric_pc() - DVec3::new(-8277.0, 0.0, 25.0)).length() < 1e-9); }
  #[test] fn local_density_near_0p1() { /* star_density(sun) ∈ [0.09, 0.11] */ }
  #[test] fn local_luminosity_density_matches_source() { /* within 10% of the value cited in lf.rs */ }
  #[test] fn extinction_sanity() { /* A_V to infinity toward the NGP ∈ [0.03, 0.15]; along l=90°, b=0 to 1 kpc ∈ [0.5, 2.0] */ }
  #[test] fn band_brighter_toward_centre() { /* radiance luminance at (l,b)=(0,0) ... use (l=0,b=−5°) to avoid the heaviest dust: > 2× (l=180°,b=0) > 2× NGP */ }
  #[test] fn faint_fraction_limits() { /* → 1 as dist → 1e5 pc for v_lim 10; → small at 1 pc; monotonic in dist */ }
  #[test] fn galaxy_gpu_layout_matches_wgsl() { /* naga size + offsets */ }
  #[test] fn galaxy_wgsl_validates() {}
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-procgen --lib`.
- [ ] **Step 3: Implement**, plus `pub fn gpu_probes(...)` is **not** here (uni-procgen has no wgpu); B5 owns the GPU probe of `gal_band_radiance`.
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** all tests pass and every parameter has a citation comment.

---

### Task A6: uni-nav — camera parenting and bookmarks v2

**Files**
- Modify: `crates/uni-nav/src/{lib,bookmark,camera}.rs`; create `crates/uni-nav/src/anchor.rs` (stubbed by A0b; A0b also adds the Bookmark v2 fields with `#[serde(default)]`)

**Interfaces**
- Consumes: `uni_core::{FrameTree, FrameId, ROOT_FRAME, Epoch, CameraView, UniversePos}`.
- Produces: the A0b `uni-nav` signatures (Bookmark v2, `ObjectLookup`, `resolve_bookmark`, `FlyCamera.{anchor, rel_m}`, `reanchor`, `sync`).

**Rules**
- `FlyCamera::update` moves `rel_m` (the offset from the anchor frame's origin, root axes); `sync` recomputes `view.pos = frames.to_root(anchor, rel_m, t)`. Time warp therefore carries the camera with its body.
- `reanchor` keeps the root position exactly (recompute `rel_m` from the current root position in f64 via `UniversePos::delta_from`).
- Bookmark v2: `anchor` makes `view.pos` an offset from that object's centre (cell 0, metres); `look_at` overrides `orient` (−Z toward the object, +Y as close as possible to `up`, default ICRS +Z, falling back to +Y when parallel). Every M0 bookmark parses unchanged. F12 saves `anchor` = the current anchor body's name and an anchor-relative position, and omits `lens_scale` (review m2).

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn m0_bookmarks_still_parse() { /* load_bookmarks("bookmarks/m0.ron") → 6 entries, all new fields None */ }
  #[test] fn anchor_and_look_at_resolve() { /* fake lookup: Mars at 1.5 AU x; anchor Mars, pos (0,0,1e7) m, look_at Mars → pos == Mars + (0,0,1e7); forward == −Z root ±1e-12 */ }
  #[test] fn look_at_up_fallback() { /* looking along +Z with up +Z → +Y used, orient finite and unit */ }
  #[test] fn unknown_anchor_is_error() { /* resolve_bookmark with anchor "Vulcan" → Err mentioning "Vulcan" */ }
  #[test] fn reanchor_preserves_root_position() { /* anchors 1 AU apart; camera 1 Gpc from origin in a far frame too: |pos_after.delta_from(pos_before)| < 1e-3 m */ } // Review Focus 5
  #[test] fn anchored_camera_follows_moving_frame() { /* Custom frame moving 30 km/s; after 1e6 s of sync with no input, rel_m unchanged and root pos moved 3e10 m */ }
  #[test] fn bookmark_v2_ron_roundtrip() { /* every field set → save → load → equal */ }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-nav --lib`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** all tests pass and `bookmarks/m0.ron` resolves unchanged.

---

### Task A7: Data — AT-HYG m ≤ 10 stars, names, exoplanets (downloads)

**Files**
- Create/modify: `crates/uni-data/src/{athyg,f16,names,exo}.rs`; `tools/bake/src/{athyg,exo}.rs`
- Baked: `data/baked/athyg_m10.bin`, `data/baked/athyg_names.ron`, `data/baked/exoplanets.ron`
- Modify (own sections only): `data/manifest.toml` `# --- A7 ---`, `ATTRIBUTION.md` `<!-- A7 -->`
- Download to: `data/cache/athyg-m10/`, `data/cache/exoplanets/`

**Interfaces**
- Produces (uni-data):
  ```rust
  #[repr(C)] #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
  pub struct Star3 { pub pos_pc: [f32; 3],      // ICRS, origin SSB, at STAR3_EPOCH (disk format; B4 builds its own vec4-only GPU record)
                     pub abs_mag_v: f32,        // Johnson V (Tycho rows converted, see Rules)
                     pub vel_bv: [u32; 2],      // f16 pairs: (vx, vy), (vz, B−V); km/s, ICRS
                     pub id_flags: u32 }        // low 22 bits AT-HYG id (or ExoSystem index for archive hosts); flag bits below
  pub const STAR3_SIZE: usize = 28;
  pub const STAR3_EPOCH_JD_TDB: f64 = 2451545.0;   // AT-HYG: "epoch + equinox 2000.0" (version-info.md)
  pub const FLAG_NAMED: u32 = 1 << 22; pub const FLAG_EXO_HOST: u32 = 1 << 23;
  pub const FLAG_NO_VELOCITY: u32 = 1 << 24;      // AT-HYG computed no Cartesian velocity (missing distance or proper motion): stored as 0
  pub const FLAG_NO_RV: u32 = 1 << 25;            // radial velocity missing: tangential motion only
  pub const FLAG_FROM_ARCHIVE: u32 = 1 << 26;     // built by uni-sim from the NASA Exoplanet Archive (I11); never in the baked file
  impl Star3 { pub fn velocity_kms(&self) -> glam::Vec3; pub fn bv(&self) -> f32; pub fn athyg_id(&self) -> u32; pub fn flags(&self) -> u32;
               pub fn is_named(&self) -> bool; pub fn is_exo_host(&self) -> bool; }
  pub fn f32_to_f16(x: f32) -> u16; pub fn f16_to_f32(h: u16) -> f32;
  pub fn load_athyg(path: &Path) -> Result<Vec<Star3>, DataError>; pub fn write_athyg(path: &Path, s: &[Star3]) -> Result<(), DataError>;
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct StarName { pub index: u32, pub proper: Option<String>, pub bayer_flamsteed: Option<String>, pub con: Option<String>, pub hip: Option<u32>, pub spect: Option<String> }
  #[derive(Clone, Debug, Default)] pub struct StarNames { pub entries: Vec<StarName> }
  pub fn load_star_names(path: &Path) -> Result<StarNames, DataError>;
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct ExoPlanet { pub name: String, pub a_au: Option<f64>, pub e: Option<f64>, pub i_deg: Option<f64>, pub argp_deg: Option<f64>,
      pub period_d: Option<f64>, pub t_transit_jd: Option<f64>, pub mass_mjup: Option<f64>, pub radius_rjup: Option<f64>,
      pub teq_k: Option<f64>, pub method: String, pub year: u16 }
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct ExoSystem { pub host: String, pub star_index: Option<u32>, pub ra_deg: f64, pub dec_deg: f64, pub dist_pc: f64,
      pub v_mag: Option<f64> /* sy_vmag */, pub teff_k: Option<f64>, pub radius_rsun: Option<f64>, pub planets: Vec<ExoPlanet> }
  pub fn load_exoplanets(path: &Path) -> Result<Vec<ExoSystem>, DataError>;
  ```
- Bake CLI: `uni-bake athyg --input data/cache/athyg-m10/athyg_40_reduced_m10.csv.gz --out data/baked/athyg_m10.bin --names data/baked/athyg_names.ron`; `uni-bake exo --input data/cache/exoplanets/pscomppars.csv --stars data/baked/athyg_m10.bin --names data/baked/athyg_names.ron --out data/baked/exoplanets.ron`.
- Binary format: magic `b"UNIATHY\0"`, `u32` version 1, `u32` count, then `count` × 28-byte records, little-endian.

**Rules (AT-HYG semantics, verified by the review in AT-HYG `README.md` and `version-info.md`; review I3)**
- Columns: `x0, y0, z0` [pc] (equatorial, x toward RA 0, z toward the NCP), `vx, vy, vz` [km/s], `pmra`/`pmdec` [mas/yr] (renamed in v4.0), `rv` [km/s], `mag`, `absmag`, `ci`, `mag_src`. Use the README's names, not memory.
- **Tycho photometry:** for `mag_src = T`, `mag` is VT and `ci` is BT−VT. Convert to Johnson (ESA 1997, *The Hipparcos and Tycho Catalogues* vol. 1 §1.3): V = VT − 0.090 (BT−VT), B−V = 0.850 (BT−VT); recompute M_V = absmag + (V − mag). Check the validity range of the transformation in the source (stated for moderate colours; clamp BT−VT to it and say so). Without this, colours come out too red and V is off by 0.09 (BT−VT).
- **NULL distance** (≈ 5 700 rows had bad parallaxes set to empty): skip, count, report. The valid-row count, not a fixed window, is the expected count.
- **NULL velocity** (not computed when distance or proper motion is missing): store 0 and set `FLAG_NO_VELOCITY`. **Missing RV** (velocity computed "assuming no radial component", ≈ 20 % of Tycho stars): set `FLAG_NO_RV`; the star card shows `Star motion: tangential only (no radial velocity)`.
- The Sun's row (id 0, distance 0) is skipped: uni-planet and uni-sky draw the Sun.

**Steps**
- [ ] **Step 1: Download.** Read the AT-HYG README and version-info first. Record sha256s. Exoplanets: one TAP query of `pscomppars` with `pl_name, hostname, hip_name, hd_name, gaia_dr3_id (or the current Gaia id column; check the TAP schema), ra, dec, sy_dist, sy_vmag, st_teff, st_rad, pl_orbsmax, pl_orbeccen, pl_orbincl, pl_orblper, pl_orbper, pl_tranmid, pl_bmassj, pl_radj, pl_eqt, discoverymethod, disc_year`, `format=csv`. Record the archive's column definition of `pl_orblper` (C5 needs to know whether it is ω of the planet or of the star).
- [ ] **Step 2: Write the failing tests.**
  ```rust
  #[test] fn f16_roundtrip_exhaustive() { /* every finite u16 h: f32_to_f16(f16_to_f32(h)) == h (NaN payloads excepted); 1.0 → 0x3C00; 65504 → 0x7BFF; 1e-8 → subnormal */ }
  #[test] fn star3_is_28_bytes() { assert_eq!(std::mem::size_of::<Star3>(), STAR3_SIZE); }
  #[test] fn athyg_roundtrip() { /* write 3 → load → equal; file len 16 + 3·28 */ }
  #[test] fn parse_sirius_row() { /* the real Sirius row (const in the test): pos ≈ 2.64 pc toward RA 101.287°, Dec −16.716° (0.01°); abs_mag 1.4 ±0.1; B−V 0.0 ±0.02 */ }
  #[test] fn tycho_row_converted_to_johnson() { /* a real Tycho-sourced row (const in the test, star named in a comment with its catalogued Johnson B−V and V):
        B−V within 0.05 and V within 0.05 of the catalogued values */ } // review I3
  #[test] fn null_distance_rows_skipped_and_counted() { /* synthetic CSV with one empty distance and one empty velocity: 1 skipped, 1 stored with FLAG_NO_VELOCITY and zero velocity */ } // Review Focus 3
  #[test] fn missing_rv_flagged() { /* row with empty rv → FLAG_NO_RV */ }
  #[test] fn sol_is_excluded() { /* the bake skips the Sun's row (id 0 / dist 0) */ }
  #[test] fn load_missing_is_not_found() {} #[test] fn load_athyg_truncated_is_bad_format() { /* header says 10, file has 3 → BadFormat */ } // Review Focus 3
  #[test] fn host_matching_order() { /* match by Gaia id, then HIP, then HD, then position ≤ 2″ and |Δdist| ≤ 10%; unmatched → star_index None with v_mag kept */ }
  ```
- [ ] **Step 3: Confirm they fail:** `cargo test -p uni-data --lib && cargo test -p uni-bake --bins`.
- [ ] **Step 4: Implement**, then bake. Checks: written count + NULL-distance skips + the Sun == the subset's row count (the subset README documents 332 178 rows); file size == 16 + 28 × count; report the NULL-distance, no-velocity and no-RV counts, the Tycho-converted count, and both files' sizes. Exoplanets: ≥ 95 % of hosts with V ≤ 10 within 100 pc matched to a star (review I11); report every match rate (all hosts, V ≤ 10, within 100 pc) and the list of unmatched hosts within 30 pc (TRAPPIST-1, V 18.8 at 12.5 pc, is expected among them).
- [ ] **Step 5: Manifest and attribution** in the A7 sections; ATTRIBUTION states that `athyg_m10.bin` and `athyg_names.ron` are CC BY-SA 4.0 derived data. Delete `data/cache/athyg-m10/` and `data/cache/exoplanets/` after baking.
- [ ] **Step 6: Confirm tests pass and commit** (pathspec: own source files, baked files, own manifest/attribution sections).

**Done when:** tests pass; the baked files load; counts, flags, sizes and match rates are in the report.

---

### Task A8: Data — surface maps, texture baker, ring profile (downloads)

**Files**
- Create/modify: `crates/uni-data/src/{texture,maps,rings}.rs`; `tools/bake/src/{map,bc,rings}.rs`
- Baked: `data/baked/maps/index.ron` (committed), `data/baked/maps/<body>_<kind>.big` and `<body>_<kind>.rgba2k.big` / `.r2k.big` (git-ignored), `data/baked/saturn_rings.ron` (committed)
- Modify (own sections): `data/manifest.toml` `# --- A8 ---`, `ATTRIBUTION.md` `<!-- A8 -->`
- Download to: `data/cache/sss-8k/`, `data/cache/maps-tier-b/` (ruling 12: Galilean moons, Pluto/Charon, Ceres)

**Interfaces**
- Produces (uni-data):
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
  pub enum TexFormat { Bc7RgbaUnormSrgb, Bc1RgbaUnormSrgb, Bc4RUnorm, Rgba8UnormSrgb, R8Unorm }   // BC1 only if intel_tex_2 fails to link
  #[derive(Clone, Debug, PartialEq)] pub struct TexFile { pub format: TexFormat, pub width: u32, pub height: u32, pub mips: Vec<Vec<u8>> }
  pub fn load_tex(path: &Path) -> Result<TexFile, DataError>;   // magic b"UNITEX1\0"
  pub fn write_tex(path: &Path, t: &TexFile) -> Result<(), DataError>;
  #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)] pub enum MapKind { Albedo, Clouds, CloudTops }
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct MapInfo { pub body: String, pub naif: i32, pub kind: MapKind, pub file: String /* BC */, pub fallback_file: String /* 2k uncompressed */,
      pub width: u32, pub height: u32,
      pub lon0_deg: f64,        // planetocentric east longitude at u = 0 (left edge)
      pub albedo_scale: f32,    // multiplies decoded linear albedo so the map's disk-averaged reflectance gives the body's V geometric albedo (review I4)
      pub source: String }
  pub fn load_map_index(path: &Path) -> Result<Vec<MapInfo>, DataError>;
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct RingProfile { pub inner_m: f64, pub outer_m: f64, pub samples: Vec<[f32; 4]> } // linear rgb reflectance, opacity; uniform in radius
  pub fn load_ring_profile(path: &Path) -> Result<RingProfile, DataError>;
  ```
  There is no runtime BC decoder (round 2, review I17): without `TEXTURE_COMPRESSION_BC` the renderer loads the `fallback_file`.
- Bake CLI: `uni-bake map --input <jpg|png|tif> --body Earth --naif 399 --kind albedo --lon0 -180 --geometric-albedo 0.367 [--brdf lambert|lunar-lambert] [--clouds <map>] --out data/baked/maps/earth_albedo.big`; `uni-bake rings --input data/cache/sss-8k/8k_saturn_ring_alpha.png --inner-km <r> --outer-km <r> --out data/baked/saturn_rings.ron`.

**Rules**
- Decode: JPEG/PNG/TIFF through `image` (approved in the bake, ruling 2).
- **Verify every "8k" claim**: record the decoded pixel size in `index.ron` and the report; bake at the decoded size (the repo's history has mislabelled "4k" files).
- Mips: box filter in linear light down to 4×4; equirect seam-safe (wrap in u, clamp in v). The 2k fallback is exactly the box-filtered mip whose width is 2048 (or mip 0 when the source is ≤ 2k), stored uncompressed (`Rgba8UnormSrgb` / `R8Unorm`), so B1's probe can compare the GPU-decoded BC mip against it.
- **Compression (round 2):** albedo and cloud-top maps → BC7 (`intel_tex_2`, sRGB); single-channel maps (Earth clouds) → BC4 (`intel_tex_2`). `tools/bake/src/bc.rs` holds a BC4 decoder for the PSNR test. If `intel_tex_2` fails to link (A0b Step 1 or this task's Step 1), `bc.rs` instead implements BC1 (albedo, sRGB) and BC4 encoders (PCA endpoint fit + one least-squares refinement) and decoders, and the report says so.
- `lon0_deg`: determine each map's longitude origin from its source (and check a feature: Greenwich for Earth, Olympus Mons at 18.65°N 226.2°E for Mars, Tycho at 43.3°S 348.8°E for the Moon) and record it.
- **`albedo_scale` from the V geometric albedo** (NASA fact sheets, cite per body; review I4): the scale makes the map's disk-averaged reflectance factor equal `photometry::rho_from_geometric_albedo(brdf, p_V)` for the BRDF B1 uses on that body (Lambert for bodies with atmospheres and the giants, lunar-Lambert L = 0.9 for airless bodies). For Earth, the ground and cloud maps are scaled together: the cloud-composited disk average (clouds drawn over the ground as in B1) matches p_V; the report gives both scales.
- Ring profile: take one radial row of the ring PNG (colour = reflectance, alpha = opacity), resample to 1024 samples; find the PNG's inner/outer radii from its source documentation; if undocumented, fit the Cassini division (Huygens gap at 117 680 km) and A-ring edge (136 775 km) and report the fit.

**Steps**
- [ ] **Step 1: One-map BC7 bake first** (round 2): download `8k_moon.jpg`, bake it with `intel_tex_2`, record the time and size. If it does not link or run, switch to the own-encoder path above and report.
- [ ] **Step 2: Download** the rest of `sss-8k` and the ruled tier-B mosaics (GET each, sha256, size; stop on > 25 % over the listed size or > 60 MB per file).
- [ ] **Step 3: Write the failing tests.**
  ```rust
  #[test] fn bc4_roundtrip_psnr() { /* ≥ 42 dB on a smooth single-channel image (bake-side decoder) */ }
  #[test] fn bc1_roundtrip_psnr() { /* only on the own-encoder path: synthetic 256×256 gradients + noise ≥ 38 dB; solid colours exact */ }
  #[test] fn bc7_block_layout() { /* encoded size == ceil(w/4)·ceil(h/4)·16 B per mip; mip sizes halve down to 4×4 */ }
  #[test] fn tex_roundtrip() { /* write → load → equal mips */ }
  #[test] fn fallback_is_the_2k_mip() { /* the .rgba2k texels equal the box-filtered mip of width 2048 computed in the test from a synthetic 8192×4096 source */ }
  #[test] fn load_tex_missing_is_not_found() {} #[test] fn load_tex_truncated_is_bad_format() {} // Review Focus 3
  #[test] fn mip_wraps_in_u() { /* a 1-texel-wide stripe at u = 0 appears in mip 2 at both edges' average */ }
  #[test] fn albedo_scale_hits_geometric_albedo() { /* a synthetic map baked with --geometric-albedo 0.538 --brdf lambert: disk-averaged ρ·scale == 0.807 ±1e-3 */ }
  #[test] fn map_index_ron_roundtrip() {}
  #[test] fn ring_profile_cassini_division() { /* baked profile: mean opacity in 118 000–121 000 km < ½ of the mean in 100 000–110 000 km */ }
  ```
- [ ] **Step 4: Confirm they fail:** `cargo test -p uni-data --lib && cargo test -p uni-bake --bins`.
- [ ] **Step 5: Implement and bake** every map: Earth albedo (8k BC7), Earth clouds (8k BC4), Moon, Mars, Mercury (8k each), Jupiter, Saturn (decoded size), Venus cloud tops (`CloudTops`, 4k), Uranus, Neptune (2k), tier B, each with its 2k fallback. Use `rayon` across maps. Report each output's size, the total on disk (≈ 411 MB if the six "8k" maps decode at 8192×4096: BC 307 + 2k copies 104) and the GPU memory with BC (the BC mip chains only: 6 × 44.7 + 22.4 + 11.2 + 2 × 2.8 = 307 MB; the 2k copies, 9 × 11.2 + 2.8 = 104 MB, are uploaded only when BC is absent; re-review r1 minor 5).
- [ ] **Step 6: Manifest and attribution**, then delete `data/cache/sss-8k/` and `data/cache/maps-tier-b/`.
- [ ] **Step 7: Confirm tests pass and commit** (pathspec; `.big` files are ignored and not committed).

**Done when:** tests pass, every map in `index.ron` loads with its recorded size, `lon0_deg` and fallback, and the ring-profile test passes.

---

### Task A9: Data — small bodies, satellite elements, body facts, Horizons goldens, fonts (downloads)

**Files**
- Create/modify: `crates/uni-data/src/{smallbody,sats,facts,golden}.rs`; `tools/bake/src/{sbdb,sats,facts,horizons}.rs`
- Baked: `data/baked/{comets.ron, asteroids.bin, asteroid_names.txt, sat_elements.ron, body_facts.ron}`; `data/golden/{horizons_spots.ron, sky_spots.ron}`; `data/fonts/{Inter-Light,Inter-Regular,Inter-Medium,Inter-SemiBold,IBMPlexMono-Light,IBMPlexMono-Regular,IBMPlexMono-Medium}.ttf`, `data/fonts/OFL-Inter.txt`, `data/fonts/OFL-IBMPlexMono.txt`
- Modify (own sections): `data/manifest.toml` `# --- A9 ---`, `ATTRIBUTION.md` `<!-- A9 -->`

**Interfaces**
- Produces (uni-data):
  ```rust
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct Comet { pub name: String, pub spkid: i32, pub q_au: f64, pub e: f64, pub i_deg: f64, pub node_deg: f64, pub argp_deg: f64,
      pub tp_jd_tdb: f64, pub epoch_jd_tdb: f64, pub m1: Option<f64>, pub k1: Option<f64>, pub diameter_km: Option<f64> }
  pub fn load_comets(path: &Path) -> Result<Vec<Comet>, DataError>;
  #[repr(C)] #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
  pub struct AsteroidRecord { pub a_au: f64, pub e: f64, pub i_deg: f64, pub node_deg: f64, pub argp_deg: f64, pub m0_deg: f64, pub epoch_jd_tdb: f64,
      pub h: f32, pub g: f32, pub diameter_km: f32 /* NaN unknown */, pub albedo: f32 /* NaN unknown */, pub spkid: u32, pub name_index: u32 } // 72 B
  pub fn load_asteroids(bin: &Path, names: &Path) -> Result<(Vec<AsteroidRecord>, Vec<String>), DataError>;   // magic b"UNIAST1\0"
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct SatRecord { pub naif: i32, pub name: String, pub planet_naif: i32, pub a_km: f64, pub e: f64, pub argp_deg: f64, pub m_deg: f64,
      pub i_deg: f64, pub node_deg: f64, pub n_deg_per_day: f64, pub p_apsides_yr: f64, pub p_node_yr: f64, pub epoch_jd_tdb: f64,
      pub laplace_ra_deg: f64, pub laplace_dec_deg: f64, pub radius_km: Option<f64>, pub gm_km3_s2: Option<f64> }
  pub fn load_sat_elements(path: &Path) -> Result<Vec<SatRecord>, DataError>;
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct BodyFacts { pub naif: i32, pub name: String, pub kind: String, pub mean_radius_km: f64, pub mass_kg: f64, pub surface_gravity_ms2: Option<f64>,
      pub escape_velocity_kms: Option<f64>, pub sidereal_rotation_h: Option<f64>, pub orbital_period_d: Option<f64>, pub semi_major_au: Option<f64>,
      pub axial_tilt_deg: Option<f64>, pub surface_pressure_kpa: Option<f64>, pub atmosphere: Option<String>, pub mean_temp_k: Option<f64>,
      pub bond_albedo: Option<f64>, pub geometric_albedo: Option<f64>, pub moons: Vec<String>, pub source: String }
  pub fn load_body_facts(path: &Path) -> Result<Vec<BodyFacts>, DataError>;
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct EphemSpot { pub target: i32, pub center: i32, pub jd_tdb: f64, pub pos_km: [f64; 3], pub vel_km_s: [f64; 3] } // ICRF, geometric
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct SkySpot { pub target: String, pub observer: i32, pub jd_utc: f64, pub ra_deg: f64, pub dec_deg: f64, pub delta_au: f64,  // astrometric
      pub apmag: Option<f64>, pub r_au: Option<f64>, pub phase_deg: Option<f64> }                                                   // quantities 9, 19, 24
  pub fn load_ephem_spots(path: &Path) -> Result<Vec<EphemSpot>, DataError>;
  pub fn load_sky_spots(path: &Path) -> Result<Vec<SkySpot>, DataError>;
  ```

**Steps**
- [ ] **Step 1: Goldens first** (A1, A2 and A3 wait on them; review I9). Implement `golden.rs` (types, loaders, RON round-trip test) before any other step, then fetch. Vectors: `EPHEM_TYPE=VECTORS`, `REF_SYSTEM=ICRF`, `REF_PLANE=FRAME`, `VEC_CORR=NONE` (geometric), `OUT_UNITS=KM-S`, `TIME_TYPE=TDB`, `TLIST` given as JDs (TDB), never calendar dates (Horizons reads dates before 1582-10-15 as Julian; review I2):
  - Standish spots, **targets 1–8 (planetary-system barycentres)** @10 (review I2: Standish elements describe barycentres, and the 599–999 body-centre solutions do not reach AD 1000): JD 2378496.5 (1800-01-01), 2415020.5 (1900), 2433282.5 (1950), 2451545.0, 2461364.5008 (2026-11-20 00:00 UTC), 2469807.5 (2050-01-01), plus 2086307.5 (AD 1000-01-01, Julian calendar) and 2634166.5 (AD 2500-01-01) for the Table 2 test.
  - Pluto: target 9 @10 at JD 2415020.5, 2433282.5, 2451545.0, 2461364.5008, 2469807.5, 2484417.5 (2090-01-01).
  - Sun 10@0 at 1950, 2000 and 2026-11-20.
  - Moons @their planet's body centre at 1950, 2000, 2026-11-20, 2050: 301@399, 401, 402@499, 501–504@599, 601–608@699, 703, 704@799, 801@899, 901@999.
  - Small bodies @10: 1P/Halley at JD 2446502.5 (1986-03-13) with `COMMAND='DES=1P;CAP<1987;'`; Ceres at 2026-11-20 with `COMMAND='DES=2000001;'`.
  - Observer tables (`EPHEM_TYPE=OBSERVER`, `CENTER=500@399`): C/2020 F3 (NEOWISE) at 2020-07-15 03:00 UTC with `COMMAND='DES=C/2020 F3;CAP;'` and `QUANTITIES='1,20'`; Jupiter at 2026-11-20 00:00 UTC (`'1,20'`); the planets 199, 299, 499, 599, 699, 799, 899 at 2026-11-20 00:00 UTC with `QUANTITIES='1,9,19,20,24'` (APmag, r, Δ, phase angle; A3's magnitude test, review I4).
  Write them with `uni-bake horizons --plan <built-in list> --out data/golden/horizons_spots.ron --sky data/golden/sky_spots.ron`, which fetches with `curl` through `std::process::Command` and parses the `$$SOE…$$EOE` block. Commit these two files as soon as they exist (pathspec) so A1 can run its tests.
- [ ] **Step 2: SBDB.** Asteroids: first `limit=1` to read the count for `H ≤ 14`, `H ≤ 14.5`, `H ≤ 15`; report the counts; `H ≤ 14` stands unless that count exceeds about 60 k, in which case report and wait for the lead (ruling 11). Fields `spkid,full_name,a,e,i,om,w,ma,epoch,H,G,diameter,albedo`. Comets: numbered periodic (`sb-kind=c`, `sb-ns=n`) plus these long-period ones: C/1995 O1 (Hale–Bopp), C/2020 F3 (NEOWISE), C/1996 B2 (Hyakutake), C/2006 P1 (McNaught), C/2023 A3 (Tsuchinshan–ATLAS), C/2011 L4 (PANSTARRS); fields `spkid,full_name,e,q,i,om,w,tp,epoch,M1,K1,diameter`. Read the API docs (https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html) for the exact parameter syntax; responses are JSON (`serde_json`).
- [ ] **Step 3: Satellites and facts.** Parse JPL `sats/elem` and `sats/phys_par` (HTML tables) into `sat_elements.ron` (every moon listed, including the Moon; Laplace-plane pole per moon). Hand-curate `body_facts.ron` from the NSSDCA fact sheets for the Sun, planets, Pluto, the Moon and the major moons, with `geometric_albedo` (V) wherever the sheet gives it (A3, A8) and every value's sheet URL in `source`.
- [ ] **Step 4: Fonts.** Download the Inter and IBM Plex Mono release archives, keep only the listed TTFs and their OFL texts.
- [ ] **Step 5: Write the failing tests.**
  ```rust
  #[test] fn asteroid_record_is_72_bytes() { assert_eq!(std::mem::size_of::<AsteroidRecord>(), 72); }
  #[test] fn asteroid_roundtrip() {}  #[test] fn comet_ron_roundtrip() {}  #[test] fn golden_ron_roundtrip() {}
  #[test] fn ceres_row() { /* baked file: Ceres a ≈ 2.77 AU ±0.01, H ≈ 3.3 ±0.2 */ }
  #[test] fn halley_row() { /* 1P q ≈ 0.575 AU ±0.01, e ≈ 0.967 ±0.002 */ }
  #[test] fn golden_spots_cover_plan() { /* every (target, center, jd) listed in Step 1 present exactly once; planet sky spots carry apmag, r_au, phase_deg */ }
  #[test] fn facts_cover_major_bodies() { /* Sun, 8 planets, Pluto, Moon, Galileans, Titan present; radii agree with uni_physics::bodies within 0.5%; planets have geometric_albedo */ }
  #[test] fn fonts_present_and_licensed() { /* 7 TTFs start with 0x00010000 or "true"; both OFL files contain "SIL OPEN FONT LICENSE" */ }
  ```
- [ ] **Step 6: Confirm they fail, implement, confirm they pass:** `cargo test -p uni-data --lib && cargo test -p uni-bake --bins`.
- [ ] **Step 7: Manifest and attribution**, delete raw caches, **commit** (pathspec).

**Done when:** tests pass, `horizons_spots.ron` and `sky_spots.ron` were committed before A1/A2/A3 needed them, and counts and sizes are in the report.

---

### Task A10: Wave A gate (lead)

- [ ] **Step 1:** `cargo check --workspace && cargo test --workspace --lib --bins` all green; record the shared target size and free disk.
- [ ] **Step 2:** Read the A1/A2/A3 reports: every planet within Standish's 1800–2050 table (λ, φ, ρ), Pluto within Meeus's stated accuracy, every moon's measured error documented and under its ceiling, every planet's magnitude within A3's tolerance. Any miss becomes a fix task, not a loosened tolerance.
- [ ] **Step 3:** `universe --selftest` all PASS; `universe --capture bookmarks/m0.ron --out captures/wave-a` and `uni-verify shadow` 158.745 ± 1 px (M0 regression; the app still draws M0's world, so m0.ron's sky views are unchanged here).
- [ ] **Step 4:** Check `data/manifest.toml` and `ATTRIBUTION.md`: one entry per file, every licence and credit present, CC BY-SA noted for AT-HYG data; `git status` shows no `.big` file staged; raw caches deleted.
- [ ] **Step 5:** Commit the wave (one commit per task where agents have not) and update CLAUDE.md §2 "Current real layout".

---

## Wave B — rendering

### Task B0: Wave B interface stubs and shared twins (lead)

**Files**
- Manifests: `crates/{uni-planet,uni-atmo}/Cargo.toml` (uni-core, uni-gfx, uni-sim, uni-data, uni-physics, uni-atmo for uni-planet; wgpu, glam, bytemuck, anyhow, pollster); `crates/uni-sky/Cargo.toml` gains uni-procgen, uni-sim, uni-physics.
- **Implemented here, with tests** (shared by two Wave B agents each; review I9): `crates/uni-planet/src/{capmesh,eclipse,lod}.rs` and `crates/uni-planet/shaders/eclipse.wgsl` (`cap_mesh`, `sun_visible_fraction` and its WGSL twin, `lod_exposure_mix`). (`cube_dir`, shared by B5 and B7, is `uni_gfx::env::cube_dir` from A0a, so uni-rel needs no uni-sky dependency.)
- Stubs: `crates/uni-planet/src/{lib,body,material,star_surface,rings,probes}.rs` (the `probes.rs` stub already calls B2's `star_probes`/`ring_probes`), `crates/uni-planet/shaders/{body,star_surface,rings}.wgsl`; `crates/uni-atmo/src/{lib,params,lut,ground,feature,probes}.rs`, `crates/uni-atmo/shaders/{ground_lighting,transmittance_lut,multiscatter_lut,irradiance_lut,atmosphere_common,atmosphere_dual,atmosphere_grey,atmosphere_mask}.wgsl`; `crates/uni-sky/src/{stars3d,sectors,points,band,probes}.rs` + `lib.rs` module lines (the `probes.rs` stub already calls B5's band probes), `crates/uni-sky/shaders/{stars3d,points,band_update,band_draw}.wgsl`; `crates/uni-post/src/{exposure,histogram,bloom,probes}.rs` + `lib.rs` lines, `crates/uni-post/shaders/{exposure,histogram,bloom}.wgsl`; `crates/uni-rel/src/{lib,probes}.rs` (`LensEnv`); the new `debug_background_draws(coverage, env)` signature in `crates/uni-rel/src/background.rs` with the uni-app call site passing `LensEnv::DebugGrid` (M0 behaviour).
- `ATTRIBUTION.md`: a `<!-- B3 -->` section for an MIT notice if B3 adapts Hillaire's sample code.
- Every stub WGSL file is valid (functions return neutral values) so cross-crate imports compile from day one; `atmosphere_dual.wgsl` starts with `enable dual_source_blending;` and `#import "atmo/atmosphere_common.wgsl"` (A0a's `compose` hoists the directive).

**Interfaces** (frozen for Wave B):
```rust
// every render crate
pub fn register_shaders(lib: &mut uni_gfx::ShaderLib);
pub fn gpu_probes(gpu: &uni_gfx::GpuContext, lib: &mut uni_gfx::ShaderLib) -> Vec<uni_gfx::probe::ProbeResult>;

// uni-atmo
#[derive(Clone, Debug, PartialEq)]
pub struct AtmosphereParams { pub bottom_radius_m: f64, pub top_radius_m: f64,
    pub rayleigh_scattering: glam::Vec3, pub rayleigh_scale_height_m: f32,          // 1/m at (680, 550, 440) nm
    pub mie_scattering: glam::Vec3, pub mie_extinction: glam::Vec3, pub mie_scale_height_m: f32,
    pub mie_g: glam::Vec3,                                                         // per channel (review I6: Mars's blue aureole needs a stronger blue forward lobe)
    pub layer2_scattering: glam::Vec3, pub layer2_extinction: glam::Vec3,           // ozone (scattering 0) or a haze layer
    pub layer2_center_m: f32, pub layer2_width_m: f32, pub layer2_g: glam::Vec3,   // tent profile (Hillaire 2020); per-channel g
    pub ground_albedo: glam::Vec3, pub source: &'static str }
pub fn earth() -> AtmosphereParams; pub fn mars() -> AtmosphereParams; pub fn venus() -> AtmosphereParams; pub fn titan() -> AtmosphereParams;
pub fn preset_for(naif: i32) -> Option<AtmosphereParams>;                           // 399, 499, 299, 606
pub const VENUS_CLOUD_TOP_M: f64 = 70_000.0;      // altitude of the visible cloud tops (Titov et al. 2018); B1 draws Venus's map at R + this
pub struct GroundLighting { pub bgl: wgpu::BindGroupLayout, /* private groups */ }
impl GroundLighting { pub fn group(&self, naif: Option<i32>) -> &wgpu::BindGroup; }  // airless/unknown → neutral group
pub struct AtmosphereSet { /* LUTs per atmosphere, built once */ }
impl AtmosphereSet { pub fn new(gpu: &uni_gfx::GpuContext, lib: &mut uni_gfx::ShaderLib, presets: &[(i32, AtmosphereParams)]) -> anyhow::Result<Self>;
                     pub fn ground_lighting(&self) -> std::sync::Arc<GroundLighting>; }
#[derive(Clone, Debug, Default)] pub struct AtmoScene { pub atmos: Vec<AtmoDraw> }
#[derive(Clone, Debug)] pub struct AtmoDraw { pub naif: i32, pub rel_pos_m: glam::DVec3, pub orient: glam::DQuat,
    pub sun_rel_m: glam::DVec3 /* Sun − body */, pub sun_illuminance_lux: f64 }
pub struct AtmosphereFeature;  // RenderFeature "atmosphere"
impl AtmosphereFeature { pub fn new(gpu: &uni_gfx::GpuContext, lib: &mut uni_gfx::ShaderLib, camera_bgl: &wgpu::BindGroupLayout,
    set: &AtmosphereSet, input: uni_gfx::SharedInput<AtmoScene>, dual_source: bool /* GpuContext::has_dual_source_blending() */) -> anyhow::Result<Self>; }
pub fn camera_in_daylight(params: &AtmosphereParams, cam_rel_body_m: glam::DVec3, sun_dir: glam::DVec3) -> bool; // inside the shell and the Sun above the local horizon (B6's `daylight`)
// WGSL "atmo/ground_lighting.wgsl" (bound at group 2 of the body pipeline):
//   struct AtmoGround { radii: vec4<f32> }   // bottom_m, top_m, has_atmosphere (0|1), 0
//   @group(2) @binding(0) var<uniform> atmo_ground: AtmoGround;
//   @group(2) @binding(1) var atmo_transmittance_lut: texture_2d<f32>;
//   @group(2) @binding(2) var atmo_irradiance_lut: texture_2d<f32>;
//   @group(2) @binding(3) var atmo_sampler: sampler;
//   fn atmo_sun_transmittance(r_m: f32, mu_s: f32) -> vec3<f32>;   // 1 without atmosphere; 0 below the geometric horizon
//   fn atmo_sky_irradiance(r_m: f32, mu_s: f32) -> vec3<f32>;      // ground irradiance per unit sun illuminance; 0 without atmosphere

// uni-planet
pub fn cap_mesh(cam_unit: glam::DVec3, n_rings: u32, n_seg: u32) -> Vec<glam::DVec3>;   // implemented in B0: unit-sphere space, circumscribed
pub fn sun_visible_fraction(sun_ang_r: f32, occ_ang_r: f32, sep: f32) -> f32;            // implemented in B0: circle–circle overlap; WGSL twin "planet/eclipse.wgsl"
pub fn lod_exposure_mix(proj_radius_px: f64) -> f32;  // implemented in B0: 0 at ≤ POINT_LOD_PX (star_pre) → 1 at ≥ BLEND_LOD_PX (scene_pre), smoothstep in ln(radius)
#[derive(Clone, Debug, Default)] pub struct BodyScene { pub bodies: Vec<uni_sim::snapshot::BodyView>,
    pub stars: Vec<uni_sim::snapshot::StarSurfaceView>, pub eye_adapted: bool }
pub struct BodyRenderer;         // "bodies"
impl BodyRenderer { pub fn new(gpu: &uni_gfx::GpuContext, lib: &mut uni_gfx::ShaderLib, camera_bgl: &wgpu::BindGroupLayout,
    maps: &[uni_data::MapInfo], map_dir: &std::path::Path, facts: &[uni_data::BodyFacts], ring: Option<&uni_data::RingProfile>,
    ground: std::sync::Arc<uni_atmo::GroundLighting>, input: uni_gfx::SharedInput<BodyScene>) -> anyhow::Result<Self>;
    pub fn warnings(&self) -> Vec<String>;  // e.g. "No surface map: uniform albedo (Titania)", "Inside Earth" }
pub struct StarSurfaceRenderer;  // "sun"
impl StarSurfaceRenderer { pub fn new(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, input: uni_gfx::SharedInput<BodyScene>) -> anyhow::Result<Self>;
    pub fn warnings(&self) -> Vec<String>; }   // "Inside Sun" (review C2)
pub struct RingRenderer;         // "rings"
impl RingRenderer { pub fn new(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, profile: &uni_data::RingProfile, input: uni_gfx::SharedInput<BodyScene>) -> anyhow::Result<Self>; }

// uni-sky
#[derive(Clone, Debug, Default)] pub struct StarInput { pub years_since_epoch: f64, pub camera_pc: glam::DVec3 /* camera − SSB, ICRS, f64 */, pub suppressed: Vec<u32> }
pub struct StarField3;           // "stars"
impl StarField3 { pub fn new(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, stars: &[uni_data::Star3], input: uni_gfx::SharedInput<StarInput>) -> anyhow::Result<Self>;
                  pub fn count(&self) -> u32; }
#[derive(Clone, Copy, Debug)] pub struct PointSprite { pub rel_pos_m: glam::DVec3, pub illuminance_lux: f64, pub color: glam::Vec3 }
pub struct PointSprites;         // "points"
impl PointSprites { pub fn new(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, input: uni_gfx::SharedInput<Vec<PointSprite>>) -> anyhow::Result<Self>; }
#[derive(Clone, Copy, Debug, Default)] pub struct BandInput { pub camera_icrs_pc: glam::DVec3, pub enabled: bool, pub v_lim: f32 }
pub struct BandCubemap;          // "band-update"
impl BandCubemap { pub fn new(gpu, lib, model: &uni_procgen::galaxy::GalaxyModel, input: uni_gfx::SharedInput<BandInput>) -> anyhow::Result<Self>;
                   pub fn env_map(&self) -> std::sync::Arc<uni_gfx::EnvMap>; }
pub struct BandDraw;             // "band"
impl BandDraw { pub fn new(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, env: std::sync::Arc<uni_gfx::EnvMap>, input: uni_gfx::SharedInput<BandInput>) -> anyhow::Result<Self>; }

// uni-post
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum ExposureMode { Auto, Manual }
#[derive(Clone, Copy, Debug, PartialEq)] pub struct ExposureSettings { pub mode: ExposureMode, pub manual: uni_gfx::Exposure /* the one manual source (review m13) */,
    pub compensation_ev: f32, pub eye_adapted: bool, pub eye_star_ev100: f32, pub daylight: bool /* app: camera_in_daylight */,
    pub adapt_tau_s: [f32; 2] /* brightening, darkening */, pub settle: bool }
impl ExposureSettings { pub fn eye_adapted_in_effect(&self) -> bool; }   // eye_adapted && !daylight: drives the resolve and the HUD label
pub const NIGHT_EV100: f32 = -4.0;   // dark-adapted target with no metered world light; equals uni_sky::DEFAULT_STAR_EV100 (app test)
pub const EV100_MIN: f32 = -30.0; pub const EV100_MAX: f32 = 40.0;   // equal uni_app::scene::EV100_LIMITS (app test)
pub const METER_MIN_CD_M2: f32 = 1.0e-3;                             // a world-light pixel below this is not "lit"
pub const METER_MIN_LIT_PX: f32 = 50.0; pub const METER_FULL_LIT_PX: f32 = 400.0; // lit pixels, normalised to 1920×1080
pub const HIGHLIGHT_PERCENTILE: f32 = 0.995; pub const HIGHLIGHT_PRE: f32 = 4.0;  // 2.03 stops under AgX white (16.29)
#[derive(Clone, Copy, Debug, PartialEq)] pub struct Meter { pub lit_px: f32, pub ev_avg: f32, pub ev_hi: f32 }
pub fn meter_histogram(bins: &[u32; 256], total_px: u32) -> Option<Meter>;   // None when nothing is lit; CPU twin of histogram.wgsl's reduction
pub fn target_ev(m: Option<Meter>, compensation_ev: f32) -> f32;             // the B6 rule; clamped to [EV100_MIN, EV100_MAX]
pub fn adapt(ev_prev: f32, ev_target: f32, dt_s: f32, tau_s: [f32; 2]) -> f32;
pub fn resolve_exposure(s: &ExposureSettings, adapted_scene_ev: Option<f32>) -> uni_gfx::ExposureUniform;   // CPU twin of exposure.wgsl
pub struct ExposureResolve;          // "exposure"
pub struct ExposureMeter;            // "histogram"
#[derive(Clone)] pub struct ExposureReadback { /* shared ring */ }
impl ExposureReadback { pub fn last_ev100(&self) -> Option<[f32; 2]>; }
pub fn auto_exposure(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, exposure_buffer: &wgpu::Buffer /* Renderer::exposure_buffer() */,
    settings: uni_gfx::SharedInput<ExposureSettings>) -> anyhow::Result<(ExposureResolve, ExposureMeter, ExposureReadback)>;
pub const DEFAULT_BLOOM_STRENGTH: f32 = 0.04;
pub const BLOOM_INPUT_MAX: f32 = 64.0;       // first-downsample input clamp, pre-exposed (2 stops over AgX white)
pub struct Bloom;                    // "bloom"
impl Bloom { pub fn new(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, strength: uni_gfx::SharedInput<f32>) -> anyhow::Result<Self>; }

// uni-rel
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum LensEnv { Band, DebugGrid }
pub fn debug_background_draws(coverage: LensCoverage, env: LensEnv) -> bool;   // m0.1: (coverage) only
pub fn lens_labels(coverage: LensCoverage, env: LensEnv) -> (bool /* debug-grid label */, bool /* unlensed-stars label */);
// LensTracer::new(gpu, lib, camera_bgl, scale: LensScale, env: Arc<EnvMap>, mode: SharedInput<LensEnv>)   // m0.1 + env, mode
// DebugSkyBackground::new(gpu, lib, camera_bgl, scale: LensScale, mode: SharedInput<LensEnv>)            // draws only in DebugGrid and Traced
```

**Steps**
- [ ] **Step 1: Write the failing tests for the shared twins** (moved here from B1 so B2 never waits on B1):
  ```rust
  #[test] fn cap_mesh_sagitta_below_altitude() { /* Earth radius, altitudes {1 m, 100 m, 10 km, 1e6 m, 1e9 m}: max mesh–sphere gap < min(0.1·altitude, 0.25 px at 1080p/60°);
        every vertex outside the sphere; camera outside the hull */ } // Review Focus 2
  #[test] fn cap_mesh_covers_silhouette() { /* 1024 limb points project inside the mesh outline */ }
  #[test] fn eclipse_fraction_cases() { /* disjoint → 1; occluder covers → 0; centred r_o = r_s/2 → 0.75 ±1e-6; partial vs numeric integral ±1e-4 */ }
  #[test] fn lod_exposure_mix_endpoints_monotonic() { /* 0.0 at ≤ 1.5 px, 1.0 at ≥ 4 px, non-decreasing on a 1000-step sweep */ } // Review Focus 4
  #[test] fn shared_twin_shaders_validate() { /* planet/eclipse.wgsl */ }
  ```
- [ ] **Step 2:** Implement the twins, write the stubs; `cargo check --workspace` passes; `cargo test -p uni-planet --lib` passes; the M0 app still runs (B0 does not change the push order yet).
- [ ] **Step 3: Commit** `build: Wave B interface stubs and shared twins`.

---

### Task B1: uni-planet — analytic ellipsoid bodies with maps

**Files**
- Modify: `crates/uni-planet/src/{body,material,probes}.rs`, `crates/uni-planet/shaders/body.wgsl`

**Interfaces**
- Consumes: B0's `cap_mesh`, `sun_visible_fraction` (+ `planet/eclipse.wgsl`), `lod_exposure_mix`; `uni_sim::snapshot::{BodyView, POINT_LOD_PX, BLEND_LOD_PX}`, `uni_data::{MapInfo, TexFormat, load_tex, BodyFacts, RingProfile}`, `uni_atmo::{GroundLighting, GROUND_LIGHTING_WGSL, VENUS_CLOUD_TOP_M}` (B3's stub WGSL is valid and neutral until B3 merges), `uni_physics::photometry::{Brdf, reflectance, rho_from_geometric_albedo, disk_illuminance_lux}`, `uni_gfx::{FrameBindings, SharedInput, HDR_FORMAT, LENS_MASK_FORMAT, DEPTH_FORMAT, HDR_MAX, GpuContext::has_bc_compression}`.
- Produces: `BodyRenderer`, and these CPU twins (each mirrored in `body.wgsl`):
  ```rust
  pub fn ray_ellipsoid(o_unit: glam::Vec3, d_unit: glam::Vec3, c: f32) -> Option<(f32, f32)>;  // unit-sphere space (scaled by 1/radii); c = |o_unit|² − 1 computed in f64 on the CPU
  pub fn equirect_uv(lon_rad: f32, lat_rad: f32, lon0_deg: f32) -> glam::Vec2;
  pub fn ring_transmittance(p_body: glam::Vec3, sun_dir: glam::Vec3, inner: f32, outer: f32, opacity_at: impl Fn(f32) -> f32) -> f32;
  pub fn venus_cloud_longitude_offset_deg(t: uni_core::Epoch) -> f64;   // super-rotation of the cloud-top map
  ```

**Rendering rules**
- **Proxy mesh, analytic shading.** Per resolved body per frame the CPU builds B0's camera-facing cap in the body's unit-sphere space (`N_RINGS = 48` × `N_SEG = 96`, circumscribed). Vertices are computed in f64, made camera-relative, then cast to f32. The fragment shader re-intersects the exact ellipsoid; the mesh only supplies coverage and depth (vertex depth, never `frag_depth`). Camera inside the body → the body draws nothing and `warnings()` reports "Inside <name>".
- **Near-surface precision (review I5).** In body-fixed f32 at Earth's radius the camera offset is quantised to 0.5 m and c = |o|² − R² is computed from 4.06 × 10¹³ m² (ulp 4.2 × 10⁶ m²) against 2Rh = 1.27 × 10⁷ m² at h = 1 m: a 33 % error that can flip the sign of c. So the CPU passes c = |o_unit|² − 1 per body per frame from f64 (at h = 1 m, c = 3.1 × 10⁻⁷, exact to f32's 6 × 10⁻⁸ relative), and the shader uses the numerically stable quadratic (Haines et al. 2019, *Ray Tracing Gems* ch. 7: q = −(b + sign(b)√disc), t₀ = q/|d|², t₁ = c/q). No surface-anchored local origin is needed in M1: the remaining f32 hit-point error, 6 × 10⁻⁸ × R = 0.4 m at Earth, is 10⁻⁴ of an 8k texel (4.9 km); M2's terrain adds one.
- **Silhouette AA:** coverage = clamp(0.5 − d_px, 0, 1) from the angular distance between the pixel ray and the analytic limb; discard below 1/255; premultiplied blend into `hdr`; `lens_mask.g` gets the coverage with `Max`. (Edge pixels write depth, so the band and stars, depth-tested at the far plane per ruling 4, are missing their (1 − coverage) share on the 1-px rim; new decision N1.)
- **Shading:** albedo from the map (`albedo_scale` applied, sRGB decoded by the format; ρ from `rho_from_geometric_albedo` for unmapped bodies, review I4), Lambert for bodies with atmospheres and the giant planets, lunar-Lambert (L = 0.9, McEwen 1991) for airless bodies; sunlight E☉(d) × `sun_visible_fraction` over the snapshot's occluders × `ring_transmittance` (Saturn) × `atmo_sun_transmittance`; plus `atmo_sky_irradiance` × albedo for atmosphere bodies. Earth: the BC4 cloud map lit with the same BRDF and blended over the ground (static map; card note `Clouds: static map (no weather)`; dynamic clouds are M2). Venus: the `CloudTops` map on a sphere of R + `VENUS_CLOUD_TOP_M`, rotating retrograde at the cloud-top super-rotation period, 4.0 d (≈ 100 m/s at the equator; Sánchez-Lavega et al. 2017, Space Sci. Rev. 212, 1541; verify), not the 243-d surface rate (review m22); card note `Cloud tops: static map, super-rotating (approximate)`.
- **Exposure:** `mix(star_pre(), scene_pre(), lod_exposure_mix(radius_px))` when eye-adapted stars are in effect, else `scene_pre()`; clamp to `HDR_MAX`.
- **Sampling:** `textureSampleGrad` with seam-safe gradients (choose the smaller of the u and wrapped-u derivatives, Tarini 2012); when one texel covers more than 1.5 px, a 4-tap bicubic B-spline (Sigg & Hadwiger 2005, GPU Gems 2 ch. 20) so magnified maps never show texel diamonds.
- **Maps:** the BC file (BC7 albedo, BC4 clouds) when `has_bc_compression()`, else the `fallback_file` (2k uncompressed); no CPU decoding at runtime (round 2). A missing map gives uniform albedo from `BodyFacts` and a warning.
- Uniforms (vec4-only) with naga layout tests; pipeline: depth write on, compare Greater; two colour targets (hdr, lens_mask with write mask GREEN and Max).

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn ray_ellipsoid_from_inside_and_centre() { /* from the centre: exit t = radius along axes; from inside: entry < 0 < exit; no NaN for any unit dir */ } // Review Focus 2
  #[test] fn ray_ellipsoid_f32_near_surface() { /* Earth, altitudes 1 m, 10 m, 100 m, 10 km, rays at nadir and 45°: f32 hit distance (c from f64) within 1 % of the f64 result;
        c never changes sign; the naive f32 form fails at 1 m (documents why) */ } // Review Focus 2, review I5
  #[test] fn equirect_uv_lon0() { /* lon 0, lat 0 with lon0 −180 → u 0.5, v 0.5; lon 179.9 and −179.9 land at u ≈ 1 and ≈ 0 */ }
  #[test] fn saturn_ring_shadow_on_globe() { /* sun 20° above the ring plane: a southern point under the B ring gets transmittance < 0.5; a northern point gets 1 */ }
  #[test] fn point_disk_handover_flux_continuous() { /* at 1.5 px: point energy (illuminance × star_pre) vs resolved energy (disk_illuminance_lux × mixed pre) within 5% */ } // Review Focus 4
  #[test] fn body_position_f32_error_subpixel_at_30au() { /* camera at Neptune + 1e6 km, body 30 AU away: f32 centre error < 0.01 px */ } // Review Focus 5
  #[test] fn missing_map_falls_back_to_uniform_albedo() { /* MapInfo list without Titania → warning text exact, material uniform with ρ from p_V */ } // Review Focus 3
  #[test] fn venus_clouds_super_rotate() { /* offset advances −360° per 4.0 d (retrograde) ±1e-9 */ }
  #[test] fn body_uniforms_layout_match_wgsl() {} #[test] fn body_shader_validates() {}
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-planet --lib`.
- [ ] **Step 3: Implement**, plus probes in `probes.rs`: `ray_ellipsoid` (256 rays, including 1 m altitude), `equirect_uv` (256), `sun_visible_fraction` (256, B0's WGSL twin), `reflectance` (64), `ring_transmittance` (64; review §4), `bodies_disk_flux` (renders an unmapped Lambert test sphere into a 256² offscreen target and compares Σ L·Ω_pixel with `disk_illuminance_lux` within 2%), and `bc_matches_fallback` (decodes the Earth albedo BC mip of width 2048 on the GPU with `textureLoad` over a 256×256 window and compares it with the `.rgba2k.big` texels: PSNR ≥ 38 dB; reports "skipped: no BC" without the feature). B2's `star_probes`/`ring_probes` are called from here (B0 stub).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass and the probes compile; B9's captures of `start`, `earth-map-closeup`, `moon-full`, `mars-reference-angle`, `jupiter-*` and `saturn-*` show smooth limbs, no seams, no texel diamonds, and their numeric checks pass.

---

### Task B2: uni-planet — star surfaces (Sun) and Saturn's rings

**Files**
- Modify: `crates/uni-planet/src/{star_surface,rings}.rs`, `crates/uni-planet/shaders/{star_surface,rings}.wgsl`. `probes.rs` is B1's file, so B2 exposes `pub(crate) fn star_probes(..)` and `ring_probes(..)` from its own modules and B1's `gpu_probes` calls them (the B0 stub already lists the calls).

**Interfaces**
- Consumes: `uni_sim::snapshot::{StarSurfaceView, BodyView, RingView}`, `uni_data::RingProfile`, `uni_core::{blackbody_linear_srgb, pcg3d, star_illuminance_lux, R_SUN_M, TEFF_SUN_K}`, B0's `cap_mesh`, `sun_visible_fraction` and `lod_exposure_mix` (implemented, so B2 does not wait on B1).
- Produces: `StarSurfaceRenderer`, `RingRenderer`, CPU twins:
  ```rust
  pub fn limb_darkening(mu: f32) -> glam::Vec3;                     // I(μ)/I(1) at (650, 550, 450) nm
  pub fn limb_darkening_mean() -> glam::Vec3;                        // 2∫ I(μ) μ dμ, for flux normalisation
  pub fn granulation(p_unit: glam::Vec3, cell_m: f32, radius_m: f32, t_s: f32) -> f32;   // multiplicative intensity, mean 1
  pub fn ring_phase(cos_alpha: f32, lit_face: bool) -> f32;          // normalised over the sphere
  pub fn planet_shadow_on_ring(p_ring: glam::DVec3, sun_dir: glam::DVec3, radii: glam::DVec3) -> f32;
  ```

**Physics (cite in code)**
- Limb darkening: Neckel & Labs 1994 (Solar Physics 153, 91) fifth-order polynomials at the three wavelengths nearest 650, 550 and 450 nm (record the coefficients and table numbers).
- Disk normalisation: the surface luminance L(μ) = I/(π R²)·LD(μ)/⟨LD⟩, so the disk-integrated illuminance at distance d equals the view's `luminous_intensity_cd / d²`; colour `blackbody_linear_srgb(teff_k)`.
- Granulation (Nordlund, Stein & Asplund 2009, Living Rev. Solar Phys. 6, 2): cells ≈ 1.3 Mm, lifetimes ≈ 8 min, RMS intensity contrast ≈ 13% at 550 nm (verify), animated cellular noise on integer-lattice cube-face coordinates hashed with `pcg3d`, two layers phase-offset by half a lifetime and cross-faded (Neyret 2003); contrast fades out as a cell shrinks below 2 px (footprint filter, no aliasing). Other stars: cell size × (T_eff/5772)·(R/R☉), labelled as an approximation in a comment. At `sun-closeup` the cells are 1.14 px and fade out (limb darkening is measured there); `sun-granulation` resolves them at 6.7–8.7 px (review I14).
- **Camera inside a photosphere** (distance < radius; review C2): the star draws nothing and `warnings()` reports `Inside <name>`.
- Rings: annulus mesh in the ring plane (256 segments; exact plane depth), depth test without depth write, premultiplied blend, `lens_mask.g` = opacity with `Max`. Lit face: reflectance × two-lobe Henyey–Greenstein (w = 0.8, g₁ = −0.3 backscattering icy particles; g₂ = +0.6) — verify against Cuzzi et al. 2009 (Saturn from Cassini–Huygens, ch. 15); unlit face: diffuse transmission reflectance × opacity × (1 − opacity) × HG(g = +0.6). Sunlight × `planet_shadow_on_ring` (ray to the Sun vs Saturn's ellipsoid, penumbra by `sun_visible_fraction`). Card note `Rings: Solar System Scope texture; phase function approximate` (review m11).
- Exposure: world light, with `lod_exposure_mix` for small star disks.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn limb_darkening_normalised() { /* LD(1) == (1,1,1); decreasing in μ on every channel */ }
  #[test] fn limb_darkening_550_half() { /* LD_550(0.5) ∈ [0.68, 0.80] and equals the paper value recorded in the code ±1e-3 */ }
  #[test] fn limb_redder_than_centre() { /* LD_450(0.3) < LD_650(0.3) */ }
  #[test] fn star_surface_flux_matches_catalogue_magnitude() { /* Sun at 1 AU: disk integral == star_illuminance_lux(−26.74) ±1%; a star with M_V 4.38, T 5790 K at 1.34 pc: == star_illuminance_lux(m) ±1% */ } // Review Focus 4
  #[test] fn camera_inside_star_draws_nothing() { /* camera at 0.5 R☉ and at the centre: no draw call recorded, warning "Inside Sun", no NaN */ } // Review Focus 2, review C2
  #[test] fn granulation_statistics() { /* CPU twin over 4096 samples: mean cell spacing ∈ [0.9, 1.6] Mm, autocorrelation time ∈ [4, 12] min, RMS ∈ [0.08, 0.20], mean 1 ±0.01 */ }
  #[test] fn ring_phase_normalised() { /* ∫ ring_phase dΩ/4π == 1 ±1e-3 for both faces */ }
  #[test] fn planet_shadow_on_ring_cases() { /* point directly anti-sun behind Saturn → 0; point 90° from the anti-sun line → 1 */ }
  #[test] fn star_ring_uniforms_layout_match_wgsl() {} #[test] fn star_ring_shaders_validate() {}
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-planet --lib`.
- [ ] **Step 3: Implement**, with probes for `limb_darkening` (64 μ), `granulation` (256 points, bit-identical lattice hashing, value ±1e-4), `planet_shadow_on_ring` (64) and `ring_phase` (64; review §4).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; B9's `sun-closeup` limb-darkening ratio, `sun-granulation` contrast and the `saturn-*` ring checks pass.

---

### Task B3: uni-atmo — Hillaire 2020 atmospheres for Earth, Mars, Venus, Titan

**Files**
- Modify: `crates/uni-atmo/src/{lib,params,lut,ground,feature,probes}.rs`, `crates/uni-atmo/shaders/{ground_lighting,transmittance_lut,multiscatter_lut,irradiance_lut,atmosphere_common,atmosphere_dual,atmosphere_grey,atmosphere_mask}.wgsl`; `ATTRIBUTION.md` `<!-- B3 -->` section only if Hillaire's sample code is adapted (MIT notice).

**Interfaces**
- Consumes: `uni_gfx::{FrameBindings, SharedInput, DEPTH_FORMAT, LENS_MASK_FORMAT, HDR_MAX, validate_wgsl_with, GpuContext::has_dual_source_blending}`, camera group 0 (`lights` for the Sun's colour).
- Produces: the B0 `uni-atmo` API (including `camera_in_daylight`), and CPU twins `transmittance(params, r, mu) -> Vec3`, `multiscatter(params, r, mu_s) -> Vec3`, `sky_irradiance(params, r, mu_s) -> Vec3` (B9's shading samples need it), `segment_radiance(params, ro, rd, t_max, sun_dir) -> (Vec3 inscatter_per_lux, Vec3 transmittance)`.

**Model (cite in code)**
- Hillaire 2020 (CGF 39(4), "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"; sample code MIT) with Bruneton 2017's (r, μ) LUT parameterisation: transmittance 256×64, multiple-scattering 32×32, ground irradiance 64×16 (all Rgba16Float), computed once per atmosphere at startup.
- View: per pixel, intersect the shell, clip by the scene distance reconstructed from the depth texture (reversed-Z infinite: view z = near/ndc_z; along the ray ÷ cos θ to the view axis) and by the planet's own ground, march 32 steps (up to 48 for long limb paths) with single scattering through the transmittance LUT plus Hillaire's multiple-scattering term; phases: Rayleigh and Cornette–Shanks, evaluated per channel with `mie_g` and `layer2_g` (review I6). This per-pixel march instead of the sky-view/aerial-perspective LUTs is a recorded spec deviation (frame contract).
- Draw: a screen quad bounding the projected shell, or a fullscreen triangle when the camera is inside the shell (sky from inside works).
- **Two shader variants (review I7, round 2):** `atmosphere_common.wgsl` holds the march. With `dual_source`: `atmosphere_dual.wgsl` (`enable dual_source_blending;` + `@blend_src(0)` inscatter × E☉ × `scene_pre()`, `@blend_src(1)` transmittance; blend `dst·T + src`) as pipeline A, and `atmosphere_mask.wgsl` as pipeline B writing `lens_mask.g = 1 − luminance(T)` with `Max` (dual-source blending allows one colour target only — wgpu `Features::DUAL_SOURCE_BLENDING` docs). Without it: `atmosphere_grey.wgsl`, one pipeline with two targets: hdr with grey transmittance `dst·(1 − α) + L`, α = 1 − luminance(T), and `lens_mask.g` with `Max`.
- Presets (every coefficient cited; "verify" values checked against the papers before merging):
  - Earth: Hillaire 2020 / Bruneton 2017 defaults (Rayleigh (5.802, 13.558, 33.1)×10⁻⁶ m⁻¹, H_R 8 km; Mie scattering 3.996×10⁻⁶, extinction 4.440×10⁻⁶, H_M 1.2 km, g (0.8, 0.8, 0.8); ozone tent 25 km centre, 30 km width, absorption (0.650, 1.881, 0.085)×10⁻⁶ m⁻¹), with bottom radius 6 371 km and top + 100 km.
  - Mars: CO₂ Rayleigh plus dust (Collienne et al. 2013; dust single-scattering albedo **and per-wavelength asymmetry g** from Wolff et al. 2009, JGR 114, E00D04, Table 2, or Ehlers et al. 2014; verify; e.g. HG g 0.73 against 0.65 gives P(0) 23.7 against 13.5, the forward-lobe ratio the blue aureole needs; τ ≈ 0.5; H ≈ 11 km), top + 80 km.
  - Venus: bottom at the cloud tops (R + `VENUS_CLOUD_TOP_M`), CO₂ Rayleigh above and an upper-haze layer (Titov et al. 2018, Space Sci. Rev. 214:126), top + 60 km.
  - Titan: N₂ Rayleigh plus orange haze with a detached layer near 500 km as layer 2 (Tomasko et al. 2008, PSS 56, 669; per-channel g), top + 600 km.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn rayleigh_only_vertical_transmittance() { /* Earth Rayleigh only, r = bottom, μ = 1: T == exp(−β·H·(1 − e^{−100 km/H})) ±1e-4 per channel */ }
  #[test] fn earth_zenith_transmittance_550() { /* full Earth preset: T_550(bottom, μ=1) ∈ [0.80, 0.90] (review computed 0.868) */ }
  #[test] fn earth_sky_blue_at_noon() { /* zenith radiance, sun at 60° elevation: B > G > R */ }
  #[test] fn earth_sunset_reddens() { /* sun transmittance at μ_s = 0.02: R/B > 5 (review computed ≈ 186) */ }
  #[test] fn mars_blue_sunset() { /* with the per-channel dust g: sky radiance within 5° of a setting sun: B/R > 1; noon zenith: R/B > 1 */ } // review I6
  #[test] fn segment_from_inside_shell() { /* camera 10 km up: upward ray finite, t_start 0; downward ray stops at the ground; horizontal ray ok */ } // Review Focus 2
  #[test] fn camera_in_daylight_cases() { /* 10 km up, Sun 30° up → true; Sun 5° below the horizon → false; 1 000 km up (outside the shell) → false */ }
  #[test] fn lut_param_roundtrip() { /* uv ↔ (r, μ) inverse ±1e-5 */ }
  #[test] fn airless_ground_lighting_is_neutral() { /* airless group: transmittance 1, irradiance 0 (CPU twin of the stub WGSL) */ }
  #[test] fn atmo_uniforms_layout_match_wgsl() {}
  #[test] fn atmo_shaders_validate() { /* dual and mask variants with validate_wgsl_with(.., default | SHADER_FLOAT16 | DUAL_SOURCE_BLENDING); grey and LUT shaders with validate_wgsl */ } // review I7
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-atmo --lib`.
- [ ] **Step 3: Implement**, with probes: transmittance LUT vs CPU (64 samples ±1e-3), multi-scattering LUT (16 ±2%), irradiance LUT vs `sky_irradiance` (16 ±2%), `atmo_sun_transmittance` and `atmo_sky_irradiance` through the ground-lighting group (32 each; review §4), `segment_radiance` (32 rays ±2%).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; B9's `earth-limb-dawn`, `venus-crescent`, `titan-backlit` and `mars-reference-angle` checks pass (sunrise-arc colours, Venus haze ring, Titan's orange haze, the shading samples).

---

### Task B4: uni-sky — AT-HYG stars in 3-D and point sprites

**Files**
- Modify: `crates/uni-sky/src/{stars3d,sectors,points,probes}.rs`, `crates/uni-sky/shaders/{stars3d,points}.wgsl`. M0's `stars.rs`/`psf.rs`/`stars.wgsl` (`StarField`, the BSC5 fallback) stay unchanged apart from A0a's mask read.

**Interfaces**
- Consumes: `uni_data::{Star3, STAR3_EPOCH_JD_TDB}`, M0's PSF functions (`psf.rs`, read-only) and Planck LUT, `uni_core::{star_illuminance_lux, bv_to_teff, PC}`, frame contract (`star_pre()`, `lens_mask`).
- Produces: `StarField3`, `PointSprites`, CPU twins `star_apparent_mag(abs_mag: f32, d_pc: f32) -> f32`, `sector_of(pos_pc) -> IVec3`, `visible_star_count(stars, eye_star_ev100, fov_y, height_px) -> u32` (for the lead's exposure sweep).

**Rules**
- **GPU record (review m15):** `Star3` is a 28-byte disk format and not a valid WGSL storage struct (vec3 + f32 + vec2<u32> + u32 has stride 32 and breaks the "no vec3 followed by a scalar" rule). B4 defines its own vec4-only `StarGpu { offset_absmag: [f32; 4] /* offset from the sector origin [pc], M_V */, vel_bv_flags: [u32; 4] /* f16 (vx,vy), f16 (vz,B−V), flags, sector */ }` (32 B) with a naga layout test.
- **Sectors** (design §5.2): 16 pc cubes; stars sorted by sector at load; f32 offsets from each sector's f64 origin; each frame, sector origins relative to the camera (f64 → f32) are uploaded (only sectors that hold stars).
- **Per star on the GPU:** p = sector_rel + offset + v·t (km/s → pc/yr = 1.0227×10⁻⁶); d = |p|; m = M_V + 5 log₁₀ d − 5; E = `star_illuminance_lux(m)`; colour from B−V through the Planck LUT; M0's energy-normalised erf PSF + Moffat halo; footprint radius from `footprint_radius_px` with the minimum `STARS3D_MIN_RADIUS_PX = 2.0` defined in `stars3d.rs` (M0's `psf::MIN_RADIUS_PX = 3.0` is untouched; review m16); cull below `VISIBILITY_CUTOFF`; indices in `suppressed` (≤ 8, near stars drawn as surfaces or points) emit degenerate quads.
- **Output:** E/Ω × `star_pre()` × (1 − r)(1 − g); depth test `GreaterEqual` at depth 0 (M0).
- **Points:** `PointSprites` uses the same PSF, its own instance buffer from `Vec<PointSprite>` (rebuilt per frame), `star_pre()` × (1 − r) (review m6), vertex depth from the true distance, compare `Greater` (bodies in front hide them).
- Review m3 / m26: stars stay E/Ω_pixel (fov-dependent against the band, recorded in the frame contract); the exposure sweep at B9 sets the default star EV.
- Budget: `stars` ≤ 2.5 ms GPU in `sky-orion-3d` at 1080p (measured at B9; ladder in "Performance budget").

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn apparent_mag_from_ssb_matches_catalogue() { /* for 100 baked stars: star_apparent_mag(abs, |pos|) vs abs + 5log10 d − 5 ±1e-4 */ }
  #[test] fn barnard_proper_motion_100yr() { /* Barnard's star (find by name) moves 10.39″/yr × 100 ±2% on the sky */ }
  #[test] fn sector_offsets_precision_at_1kpc() { /* camera 1 kpc from the SSB, a star 1 pc away: direction error of the f32 path < 0.05 px at 1080p/60° */ } // Review Focus 5
  #[test] fn suppressed_star_is_degenerate() { /* CPU twin of the vertex stage: an index in `suppressed` yields a zero-area quad; its neighbours do not */ }
  #[test] fn point_sprite_energy_matches_star_path() { /* a PointSprite and a Star3 with equal illuminance and direction give equal display energy ±1e-6 */ }
  #[test] fn points_respect_capture_mask() { /* CPU model: lens_mask.r = 1 → a point adds nothing; r = 0.25 → 0.75 of its energy */ } // review m6
  #[test] fn footprint_minimum_two_px() { /* the faintest visible star's quad radius == 2 px; a V = −1.46 star's radius > 2 px */ }
  #[test] fn stars3d_points_layouts_match_wgsl() { /* StarGpu 32 B, vec4-only */ } #[test] fn stars3d_points_shaders_validate() {}
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-sky --lib`.
- [ ] **Step 3: Implement**, with probes: apparent magnitude and direction for 64 stars (±1e-3 mag, ±1e-6 rad) and the PSF (erf coverage, halo weight, footprint radius; closes review m8). B5's band probes are called from `probes.rs` (B0 stub).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; B9's `sky-orion-3d` points and flux checks pass and `alpha-cen-sun` places the Sun correctly.

---

### Task B5: uni-sky — Milky Way band v0 and the environment cubemap

**Files**
- Modify: `crates/uni-sky/src/band.rs`, `crates/uni-sky/shaders/{band_update,band_draw}.wgsl`; band probes exposed from `band.rs` and called by B4's `probes.rs` (B0 stub lists the call).

**Interfaces**
- Consumes: `uni_procgen::galaxy::{GalaxyModel, GalaxyGpu, GALAXY_WGSL_PATH, GALAXY_WGSL_SRC}`, `uni_gfx::{EnvMap, ENV_UNIT_CD_M2, SharedInput, cube_dir}` and `common/cube.wgsl` (A0a).
- Produces: `BandCubemap`, `BandDraw`, CPU twin `band_texel(model, cam_icrs_pc, face, uv, v_lim) -> Vec3`.

**Rules**
- Cubemap 6 × 256², Rgba16Float, 5 mips (box), values = radiance / `ENV_UNIT_CD_M2` (1e-4 cd/m², A0a), clamped to 60000 units on write. Compute pass per face: `gal_band_radiance(cam_gc, dir_gc, v_lim)` from the A5 WGSL twin.
- Refresh: all six faces on the first frame (captures stay deterministic); later, when the camera has moved more than max(1 pc, 1% of its distance to the Galactic centre) since the last build, one face per frame. In the Solar System the cubemap never refreshes.
- Draw: fullscreen triangle, `pixel_dir` → sample mip 0 → × `ENV_UNIT_CD_M2 × star_pre()` × (1 − r)(1 − g)(1 − b), additive, depth test `GreaterEqual` at depth 0 (ruling 4), clamped to `HDR_MAX`. `enabled = false` skips both passes.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn env_unit_keeps_faint_sky_normal_in_f16() { /* 1e-6 cd/m² → 0.01 units: f16 relative error < 1e-3; 1 cd/m² → 1e4 units, finite; 10 cd/m² clamps to 60000 */ } // review m10
  #[test] fn refresh_policy() { /* static camera: 6 faces in frame 0, none after; moved 2 pc: one face per frame for 6 frames */ }
  #[test] fn band_texel_matches_model() { /* band_texel at the GC direction == model.band_radiance ±1e-6 relative */ }
  #[test] fn band_layouts_match_wgsl() {} #[test] fn band_shaders_validate() {}
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-sky --lib`.
- [ ] **Step 3: Implement**, with probes: `gal_band_radiance` GPU vs CPU for 64 directions (±1%), `cube_dir` WGSL twin (24 face/uv pairs ±1e-6).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; B9's `sky-galactic-center` band-ratio check passes and `bh-lens-band` shows the same band through TestBH-1 with no seam.

---

### Task B6: uni-post — histogram auto-exposure, dual star exposure, bloom

**Files**
- Modify: `crates/uni-post/src/{exposure,histogram,bloom,probes,lib}.rs`, `crates/uni-post/shaders/{exposure,histogram,bloom}.wgsl`; `crates/uni-post/shaders/tonemap.wgsl` (dither amplitude only).

**Interfaces**
- Consumes: `Renderer::exposure_buffer()` (m0.1), `uni_gfx::{ExposureUniform, exposure_scale, Exposure, HDR_MAX}`, `common/exposure.wgsl` (A0a; the resolve pass imports only this file).
- Produces: the B0 `uni-post` API.

**Rules (cite in code)**
- **Meter (pass 9, after atmospheres, before sky light; review C3, ruling round 2):** world luminance (Rec. 709 Y) of `hdr` ÷ `scene_pre`, in cd/m²; pixels below `METER_MIN_CD_M2` = 10⁻³ are not lit; 256 log₂ bins over [−10, 32] (0.164 EV per bin; values outside go to the end bins; a pixel clamped at `HDR_MAX` counts as the lower bound HDR_MAX/scene_pre); workgroup-shared atomics then one reduction (bruop 2019, "Automatic Exposure Using a Luminance Histogram"). `lit_px` is normalised to 1920×1080 (× 2 073 600 / internal pixel count).
- **Target:**
  - ev_avg = log₂(L_avg · S/K), L_avg = 2^(mean log₂ L over the 50th–95th percentile of lit pixels), S = 100, K = 12.5 (Lagarde & de Rousiers 2014 §4.1.1, after ISO 2720). A pixel at L_avg is pre-exposed to 1/(1.2·8) = 0.104.
  - ev_hi = log₂(L_p99.5 / (1.2 · `HIGHLIGHT_PRE`)): the 99.5th-percentile lit pixel lands at 4.0, 2.03 stops under AgX white (2^4.026 = 16.29), so a small bright subject next to a large dim one is not clipped.
  - `lit_px` ≥ `METER_FULL_LIT_PX` (400 px, 0.019 % of the frame, a disk of radius 11.3 px): target = max(ev_avg, ev_hi) + compensation. `lit_px` < `METER_MIN_LIT_PX` (50 px, radius 4.0 px, where `lod_exposure_mix` has finished): `NIGHT_EV100`. Between: smoothstep in ln(lit_px) from `NIGHT_EV100` to the metered target, so a growing disk fades from its star-like look to correct exposure without a pop. Clamp to [`EV100_MIN`, `EV100_MAX`].
  - Why: the earlier "< 1 % lit → NIGHT" rule exposed every small-planet bookmark at EV −4, which puts Jupiter's ≈ 1 200 cd/m² at 1.6 × 10⁴ pre-exposed, 10 stops over white (review C3). "Exposure on paper" works the new rule through every bookmark.
- **Adaptation:** ev += (target − ev)(1 − e^{−dt/τ}), τ = 0.6 s brightening, 1.2 s darkening (record the source); `settle` → τ = 0 (captures). From clamped pixels the loop still converges: starting at EV −4, the Sun's disk (mean 1.85 × 10⁹ cd/m² = `star_illuminance_lux(−26.74)` / Ω☉ = 1.259 × 10⁵ lux / 6.794 × 10⁻⁵ sr, Ω☉ = 2π(1 − cos asin(R☉/AU)); re-review r1 minor 2) clamps at 6 × 10⁴ and reads as 4 500 cd/m² → target 15.1; at 15.1 the disk is 1.853 × 10⁹ / (1.2 × 2^15.1) = 4.4 × 10⁴ pre-exposed (unclamped, < 6 × 10⁴) → target 33.8; settled on the third frame.
- **Resolve** (pass 1): scene EV = adapted (Auto) or `manual.scene_ev100` (Manual); star EV = `eye_star_ev100` when `eye_adapted_in_effect()`, else the scene EV; writes `ExposureUniform` through a storage binding of the exposure buffer **in its own bind group**, importing only `common/exposure.wgsl`, never group 0 (a uniform and a storage use of one buffer in one dispatch is a usage-scope conflict; review m14). The CPU write the Renderer makes earlier in the submission is overwritten by this pass. `ExposureSettings.manual` is the one manual source; the app copies it into `FrameCtx.exposure` too (review m13).
- **Daylight (review m12):** `daylight = true` (the app sets it from `uni_atmo::camera_in_daylight`) turns the eye-adapted star exposure off, because a daylit sky hides the stars; the HUD label is off with it.
- **Bloom:** Jimenez 2014 (SIGGRAPH "Next Generation Post Processing in Call of Duty: Advanced Warfare"): 6-level 13-tap downsample with a Karis average on the first level, 3×3 tent upsample; `hdr = mix(hdr, bloom, strength)` (the LearnOpenGL "Physically Based Bloom" article is a reference only; its code is CC BY-NC and is not copied). No threshold. **The first downsample clamps its input to `BLOOM_INPUT_MAX` = 64** (2 stops over AgX white). Arithmetic: from 6 AU (`asteroid-belt-above`) the Sun is a V = −22.8 point whose glare saturates a 64-px-radius disk; modelling the chain as six equal-weight Gaussians (σ = 3·2^(k−1) px, k = 1..6), its bloom at the belt's inner edge (2.1 AU, 270 px: rectilinear, 540 × (2.1/6)/tan 35°) is ≈ 1.8 × 10⁻³ pre-exposed with the clamp (strength 0.04), a tenth to a twenty-third of the highlighted belt's 1.9–4.2 × 10⁻² (C2; re-review r2 minor 2), but ≈ 1.7 without it, which would wash the belt out; at the comparison region 3.6–4.6 AU (463–591 px) it is < 10⁻⁶ (8.4 × 10⁻⁷ at 463 px). Planets (≈ 0.1–0.3), the Sun's disk at its own exposure (≈ 0.13) and Sirius (35) are below the clamp and unaffected.
- **Dither:** retune the M0 TPDF from ±0.5 to ±1 LSB (review deferral T5) now that faint gradients (band, atmospheres) exist.
- **Readback:** a 3-buffer map ring like `PassTimer`; `last_ev100()` lags ≥ 1 frame.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn ev_from_uniform_luminance() { /* all pixels 1000 cd/m² → ev_avg = log2(1000·100/12.5) = 12.97 ±0.05 */ }
  #[test] fn small_disk_not_blown_out() { /* 40-px-radius Lambert disk with 1 000 cd/m² at its centre, on black (5 027 lit px): after settling, no pixel above AgX white; centre ≈ 0.12 (EV 12.7) */ } // Review Focus 6, review C3
  #[test] fn meter_ramp_between_thresholds() { /* lit 25 px → NIGHT_EV100; 400 px and more → metered; between, monotonic in lit_px; continuous at both ends */ } // Review Focus 6
  #[test] fn start_ev_stable_when_lit_area_changes() { /* `start`'s Earth, phase 60° → 90° (lit area 1.13 % → 0.75 %): |ΔEV| < 0.5 (the rule gives 0.32, from the lit part's
        shifting radiance distribution; the old 1 % rule jumped 21 EV, from 17.1 to −4) */ } // review C3 (it asked ≤ 1 stop)
  #[test] fn highlight_rule_caps_bright_minority() { /* 99 % of lit px at 10 cd/m², 1 % at 1e5: target == ev_hi; the 99.5th-percentile pixel lands at 4.0 ±2 % */ }
  #[test] fn settle_converges_from_clamped_sun() { /* start EV −4, a full-frame 1.853e9 cd/m² disk with the HDR_MAX clamp: within 0.05 EV of 33.8 in ≤ 4 settle frames */ } // Review Focus 6
  #[test] fn empty_histogram_targets_night_ev() { /* nothing lit → None → target NIGHT_EV100 */ }
  #[test] fn adapt_converges_and_settles() { /* τ 0.6 s: 95% after 1.8 s; settle → target in one step */ }
  #[test] fn resolve_eye_adapted_vs_strict() { /* eye: pre.y == exposure_scale(eye_star_ev); strict: pre.y == pre.x */ }
  #[test] fn daylight_disables_eye_adapted() { /* eye_adapted true, daylight true: eye_adapted_in_effect() false and pre.y == pre.x */ } // Review Focus 6, review m12
  #[test] fn ev_clamped_to_limits() { /* manual EV 100 → 40 and −100 → −30; a Meter with ev_avg 50 → 40; 1e12 cd/m² lands in bin 255 (2^32 is the top edge), not out of range */ } // review m3
  #[test] fn bloom_impulse_energy_conserved() { /* Karis average disabled: CPU twin of the down/up chain on a delta: Σ out == Σ in ±1% */ } // review m4
  #[test] fn karis_average_suppresses_firefly() { /* one 60000 texel among 1.0 texels: first-mip value < 2× the plain-average value / 10 */ }
  #[test] fn bloom_input_clamped() { /* a 60000 texel enters the chain as BLOOM_INPUT_MAX */ }
  #[test] fn post_layouts_match_wgsl() {}
  #[test] fn post_shaders_validate() { /* and exposure.wgsl's resolve entry point binds no group 0 resource */ } // review m14
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-post --lib`.
- [ ] **Step 3: Implement**, with probes: histogram → `meter_histogram` + `target_ev` on a synthetic 512² HDR with a known log-normal distribution (GPU vs CPU ±0.05 EV), the resolve pass vs `resolve_exposure` for 16 settings (±1e-6 relative; review §4), bloom on an impulse (±1%).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; B9's captures settle in ≤ 16 frames, every metered bookmark matches its "Exposure on paper" row within 0.5 EV, and the stars stay visible next to sunlit planets with the eye-adapted label shown.

---

### Task B7: uni-rel — lens environment from the band, mask channels

**Files**
- Modify: `crates/uni-rel/src/{tracer,background,lib,probes}.rs`, `crates/uni-rel/shaders/{lens_trace,debug_background}.wgsl`

**Interfaces**
- Consumes: `uni_gfx::{EnvMap, ENV_UNIT_CD_M2, SharedInput, cube_dir}` (A0a), `LensEnv` (B0), the depth texture (`FrameTargets::depth_view`).
- Produces: `LensTracer::new(gpu, lib, camera_bgl, scale, env, mode)`, `DebugSkyBackground::new(gpu, lib, camera_bgl, scale, mode)`, `debug_background_draws(coverage, env)`, `lens_labels(coverage, env)`.

**Rules**
- Escaped rays: sample the env cubemap at the exit direction (mip from the screen-space spread of exit directions, Jacobian by finite differences of neighbouring samples; physics research §1.4) × `ENV_UNIT_CD_M2 × star_pre()`; captured → black. Traced pixels write `lens_mask = (captured, 0, 1, 0)`.
- **Foreground bodies are not lensed (review m7, spec §5.3):** the lens pass reads the depth texture and leaves every pixel with depth ≠ 0 untouched (colour and mask).
- `LensEnv::DebugGrid` keeps the m0.1 behaviour exactly (debug grid, its labels) so the shadow regression bookmark still measures 158.745 px.
- The debug background draws only in `DebugGrid` and `Traced`.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn debug_background_only_in_debug_mode() { /* (Traced, Band) → false; (Traced, DebugGrid) → true; (Unlensed, DebugGrid) → false */ }
  #[test] fn lens_labels_by_mode() { /* debug label only in DebugGrid+Traced; unlensed-stars label whenever Traced */ }
  #[test] fn resolve_writes_mask_channels() { /* CPU model of cs_resolve: captured fraction in r, g = 0, b = 1 inside the rect; untouched outside */ }
  #[test] fn lens_skips_foreground_pixels() { /* CPU model: a traced pixel with depth 0.3 keeps its colour and mask */ } // review m7
  #[test] fn lens_shaders_validate() {} #[test] fn lens_params_layout_unchanged() {}
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-rel --lib`.
- [ ] **Step 3: Implement**; keep the M0 Binet probe; add an env-sampling probe (exit direction → cube texel for 32 rays matches a CPU lookup through `cube_dir`'s convention).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; B9's `bh-lens-band` measures 158.745 ± 1 px on the HDR with no seam at the lens rectangle, and `bh-shadow-30M-grid` still measures 158.745 ± 1 px.

---

### Task B8: tools/verify — disk, point, HDR and bench checks; sheet v2

**Files**
- Modify: `tools/verify/src/{main,sheet,shadow}.rs`; create `tools/verify/src/{disk,points,bench,hdr}.rs`

**Interfaces**
- Consumes: capture PNG + JSON v2 and the optional HDR sidecar (schema in B9 Step 2: `projected`, `ev100`, `metrics`, `honesty`, `hdr`, `regions`, `samples`, `features`, `flux`, `hud`), and the bench JSON (schema in B9 Step 3). Write fixtures from those schemas; B9 produces the real files.
- Produces the CLI (every subcommand prints `… PASS|FAIL` with numbers and exits 0 on PASS):
  - `uni-verify disk --png <f> --center <x>,<y> [--threshold 0.008] --expected-a <px> [--expected-b <px>] --tol <px> [--ratio <b/a> --ratio-tol <t>] [--exclude-angle <deg> --exclude-width <deg>]`: edge points along 720 rays from the bright region's centroid (first drop below the luma threshold), direct least-squares ellipse fit (Fitzgibbon, Pilu & Fisher 1999); the exclusion removes the wedge and its opposite (angle and angle + 180°). The default 0.008 luma (≈ 2/255) is a pre-exposed 4.49 × 10⁻⁴ through AgX, so the sky behind the limb must be darker than that: under eye-adapted stars the band alone is 9.1 × 10⁻⁴ to 1.44 × 10⁻² (3.7–33/255), so disk bookmarks run with `eye_adapted: false` (re-review r1 NP2). If fewer than half the unexcluded rays find a drop before leaving the frame, the tool FAILs with "no limb edge: background above threshold" instead of fitting the sky.
  - `uni-verify points --png <f> --json <f> --names <a,b,…> --tol <px>`: for each name in the JSON's `projected` list, the background-subtracted brightness centroid in a square window of side 2·⌈2·radius_px + 3⌉ + 1 (at least 7) around the expected pixel, so a resolved Moon is not truncated (review m25); PASS if every error ≤ tol.
  - `uni-verify bench --json <f> --max-gpu-ms <ms> --max-cpu-ms <ms> --max-interval-ms <ms> [--window last10s|first10s|all] [--budgets]`: per bookmark, p95 GPU frame, p95 CPU and p95 frame interval against the limits; `--budgets` also flags passes whose mean exceeds 1.25 × the plan's budget table (compiled in); `n/a` GPU timing is a FAIL with the reason.
  - HDR checks (`hdr.rs`; read `<name>.hdr.bin` through the JSON's `hdr` header, divide by `pre[0]` for world light or `pre[1]` for sky light as each check states):
    - `ratio --hdr-json <f> --num <region> --den <region> [--channels r/b|b/r|g|luma] [--stat mean|median] [--min x] [--max x] [--expect x --rel-tol t]`;
    - `stats --hdr-json <f> --region <name> [--rms-min x --rms-max x] [--mean-pre-min x]` (relative RMS of luminance; mean of pre-exposed values);
    - `samples --hdr-json <f> --tol 0.10` (each `samples` entry: HDR luminance ÷ `pre[0]` vs `expected_cd_m2`);
    - `profile --hdr-json <f> --feature <name> --tol <t>` (log–log slope of the radial mean between the feature's radii vs its `expected_slope`);
    - `angle --hdr-json <f> --feature <name> --tol-deg <d>` (brightness-weighted direction of the pixels in the feature's annulus that pass its colour test, vs `angle_deg`);
    - `flux --hdr-json <f> --name <n> --tol <rel>` (re-review r2 NPA): E = Σ_{ρ_i < R} (Y_i − Y_bg)·Ω_i ÷ `pre[1]` vs the entry's `expected_lux`, printing E, the expectation, Y_bg and the background's share of the sum. Here:
      - Y_i is the pixel's Rec. 709 luminance 0.2126 R + 0.7152 G + 0.0722 B (star colours have Y = 1), and ρ_i the distance of its centre (i + 0.5) from the entry's (x, y);
      - Ω_i = Ω_c·cos³θ_i, the `uni_sky::stars` convention (stars are written as E·w/Ω(θ)): Ω_c = s², s = 2 tan(fov_y/2)/H, from the header's `fov_y_rad` and `height`; cos θ_i = 1/√(1 + (s·d_i)²), d_i the pixel centre's distance from the image centre (W/2, H/2). The shader takes θ at the star's centre, not per pixel; over Rigel's footprint the two differ by 2 × 10⁻⁶ (the first-order terms cancel; computed);
      - R is the entry's `radius_px`, the footprint radius the star shader used: the PSF puts all its energy inside it (the halo is renormalised to unit energy inside R, `psf::halo_weight`; the σ = 0.65 px core leaves exp(−3²/(2·0.65²)) = 2.4 × 10⁻⁵ outside 3 px);
      - Y_bg is the median of Y over the annulus R ≤ ρ < 1.5 R, skipping pixels inside any of the entry's `mask` disks (other stars' footprints), so the band behind the star is subtracted;
      - why all three matter (`sky-orion-3d`, Rigel V 0.13 at θ = 8.69°, fov 60°; `alpha-cen-sun`, the Sun V 0.40 on the axis, fov 20°; star EV −4, `pre[1]` = 1/(1.2 × 2⁻⁴) = 13.33):
        - Ω_c alone would read Rigel 1/cos³ 8.69° = 1.0352, i.e. +3.5 %;
        - Rigel's display energy is E·pre/(Ω_c cos³θ) = 2.249 × 10⁻⁶ × 13.33/(1.1431 × 10⁻⁶ × 0.9660) = 27.16, its halo peak 0.02 × 27.16 × 0.02984 = 0.01621, so R = 4·√((0.01621/3 × 10⁻⁵)^0.4 − 1) = 4·√11.39 = 13.5 px; the Sun's display energy is 1.754 × 10⁻⁶ × 13.33/1.0662 × 10⁻⁷ = 219.3 and R = 21.0 px;
        - unsubtracted, band light b·πR²·Ω/pre (b = 9.1 × 10⁻⁴, 2.3 × 10⁻³, 5.7 × 10⁻³ pre-exposed at 23, 22, 21 mag/arcsec²) adds 1.9 %, 4.8 %, 12 % to Rigel and 0.6 %, 1.4 %, 3.6 % to the Sun; summed out to `MAX_RADIUS_PX` = 64 px it adds 43–272 % (Rigel) and 5–34 % (Sun).
  - `uni-verify shadow` gains `--hdr-json <f>` (threshold at half the median background ÷ `pre[1]`), for a shadow against the dim band.
  - `uni-verify sheet` gains a "Checks" table (from `<captures>/checks.json`: bookmark, check, measured, expected, pass) and a "Performance" table (from the bench JSON: first and last 10 s side by side).

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn ellipse_fit_synthetic() { /* antialiased ellipse a 425, b 380, 20° → a, b ±0.3 px, angle ±0.2° */ }
  #[test] fn ellipse_fit_small_agx_limb() { /* a = 60 px, b/a 0.90, Lambert limb profile through the AgX twin (uni_post::agx copied as a test fixture): record the ratio error; must be ≤ 0.004 */ } // review C4
  #[test] fn ellipse_fit_lambert_limb_black_sky() { /* a = 421.3 px, b/a 0.9351, Lambert limb (peak 0.123 pre-exposed) through the AgX twin on a 4e-7 sky,
        default threshold: a ±0.1 px, b/a ±0.001 (expected limb bias 421.3·(1 − √(1 − (4.49e-4/0.123)²)) = 0.003 px) */ } // re-review r1 NP2
  #[test] fn disk_fails_when_sky_above_threshold() { /* same disk on a 1.44e-2 sky (band at 20 mag/arcsec², star EV −4): FAIL "no limb edge", no fit */ } // re-review r1 NP2
  #[test] fn disk_exclusion_wedge() { /* ellipse plus a bright line through it at 0°: with --exclude-angle 0 --exclude-width 10, ratio ±0.003 */ }
  #[test] fn disk_half_arc() { /* only the left half lit (terminator at the centre line): --exclude-angle 0 --exclude-width 180 fits a ±0.5 px */ }
  #[test] fn disk_ignores_unconnected_bright() { /* a second bright disk in a corner leaves a, b unchanged ±0.1 px */ }
  #[test] fn points_centroid_subpixel() { /* Gaussian blobs at known sub-pixel positions → ±0.1 px; a 4.2-px-radius disk → ±0.2 px (window from radius_px) */ } // review m25
  #[test] fn bench_threshold_and_na() { /* fixture JSON: last10s GPU p95 15.9 PASS, 16.4 FAIL, n/a FAIL; first10s ignored with --window last10s */ }
  #[test] fn hdr_ratio_stats_samples() { /* synthetic f32 RGBA + JSON: each check reproduces hand-computed values ±1e-6 */ }
  #[test] fn hdr_profile_and_angle() { /* synthetic 1/ρ coma and a straight blue streak at 37°: slope −1 ±0.02, angle 37° ±0.2° */ }
  #[test] fn hdr_flux() { /* a PSF splat of known energy on black, on the axis (uni_sky::psf copied as a test fixture, R from footprint_radius_px) → flux ±1% */ }
  #[test] fn hdr_flux_sky_and_off_axis() { /* fov 60°, 1920×1080, uniform 1.4e-2 sky (20 mag/arcsec² at star EV −4): the splat written as E·w/(Ω_c·cos³θ) at θ = 0 and
        at θ = 9° (148.1 px from the centre), plus a second star in the annulus listed in `mask`, placed so its own footprint stays outside R (re-review r3 minor 3) → flux ±1% in both cases. Guards: Ω_c in place of Ω_c·cos³θ at 9°
        reads 1/cos³ 9° = 1.0379 high; no subtraction reads +29.5% (1.4e-2 × π × 13.5² / 27.2, a Rigel-like splat) */ } // re-review r2 NPA
  #[test] fn shadow_on_hdr() { /* black disk of radius 158.745 on a 1e-3 background: ±0.2 px */ }
  #[test] fn sheet_includes_checks_and_perf() { /* fixture dir → HTML contains every check row and every bench bookmark */ }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-verify --bins`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass and each subcommand prints PASS/FAIL with numbers.

---

### Task B9: Wave B integration and gate (lead)

**Files:** `crates/uni-app/src/{main,app,scene,capture,bench,selftest,world}.rs` (`world.rs` becomes a thin adapter over `uni_sim::World`), `bookmarks/m1.ron` (entries 1–19), `bookmarks/m1-bench.ron`, `CLAUDE.md`.

- [ ] **Step 1: Wire** the frame-contract order; build `World` from `load_world_data`, camera anchoring (`FlyCamera::sync`, `reanchor` to `world.dominant_frame` each frame), `Snapshot` → `SharedInput`s (`BodyScene`, `AtmoScene`, `StarInput`, `Vec<PointSprite>`, `BandInput`, `LensEnv`, `ExposureSettings` with `daylight` from `uni_atmo::camera_in_daylight`) and `FrameCtx.lights` from `snapshot.sun`. Every `SharedInput` is set before `Renderer::render`. The HUD's eye-adapted label follows `ExposureSettings::eye_adapted_in_effect()`. Selecting `LensEnv::DebugGrid` switches `ExposureSettings.mode` to Manual (the grid is display-referred and must not be metered). AT-HYG missing → M0 `StarField` (BSC5) plus a warning. App tests: `NIGHT_EV100 == DEFAULT_STAR_EV100`, `EV100_MIN..=EV100_MAX == EV100_LIMITS`.
- [ ] **Step 2: Capture JSON v2 and the HDR sidecar** (review I16, round 2). Auto-exposure captures run with `settle` until |ΔEV| < 0.05 or 16 frames, then render their last two frames in Manual at the settled EVs, so the written exposure is exactly the captured frame's. The `.hdr.bin` sidecar is copied from `hdr` just before bloom (pass 15), so `flux` and `ratio` read unbloomed energy; the PNG is the same frame after bloom (re-review r3 minor 1: bloom otherwise moves 3–4 % of a point's energy outside R). JSON gains:
  - `projected: [{name, x, y, radius_px}]` for every pick item on screen; `ev100: [scene, star]`; `honesty: [labels]`;
  - `metrics: {star_count_drawn, sun_v_mag_from_camera, band_gc_over_anticentre (CPU twin), snapshot_cpu_ms, comets_cpu_ms, lit_px}`;
  - `hdr: {file, width, height, fov_y_rad, pre: [scene, star], ev100: [scene, star], layout: "f32 LE RGBA, rows top-down"}` when `--hdr` is given: `<name>.hdr.bin` is `uni_gfx::read_hdr` of the pre-exposed HDR target (33.2 MB at 1920×1080); `width`, `height` are the HDR target's (internal) size and `fov_y_rad` the capture camera's vertical field of view, which B8's `flux` needs for Ω_c (re-review r2 NPA);
  - `regions: {name: [[x, y], …]}` (≤ 256 pixel samples each), `samples: [{x, y, expected_cd_m2: [r, g, b], what}]`, `features: {name: {x, y, r_min, r_max, angle_deg?, expected_slope?}}`, `flux: [{name, x, y, radius_px, expected_lux, mask: [[x, y, r], …]}]`, `hud: {card_rows: [labels], panels: [names]}` — each computed by the app from the World and the CPU twins (B1 reflectance and map texels from the 2k fallback, B2 limb darkening, B3 `transmittance`/`sky_irradiance`/`segment_radiance`, C1 `ion_tail_dir`, A5 galactic frame), as the bookmark table specifies.
  - `flux` entries (re-review r2 NPA): (x, y) is the star's projected HDR pixel; `radius_px` is the footprint its shader used, `psf::footprint_radius_px(star_display_energy_off_axis(E, fov_y, H, star_ev100, cos θ), halo_fraction(E))` with B4's 2-px minimum; `expected_lux` is `star_illuminance_lux(V)` of that star alone; `mask` lists every other catalogue star or point sprite whose footprint reaches the annulus R ≤ ρ < 1.5 R (its pixel and its own footprint radius).
- [ ] **Step 3: Bench v2** (the m0.1 bench prints text only). `universe --bench <ron> [--size WxH] [--frames N] [--sustained <s>] [--json <out>]`: `--sustained` renders each bookmark continuously for `s` seconds with up to two frames in flight (GPU timings read without blocking, as in the window) and reports `all`, `first10s` and `last10s` windows. JSON: `{size, frames, sustained_s, bookmarks: [{name, lens_scale, windows: {all|first10s|last10s: {frame_ms, interval_ms, gpu_frame_ms, cpu_ms: {mean, p95, count} | null, passes: {pass: {mean, valid, runs}}, snapshot_cpu_ms, comets_cpu_ms}}}]}`.
- [ ] **Step 4: Selftest** calls every crate's `gpu_probes`; prints `PASS name` / `FAIL name: detail`.
- [ ] **Step 5: Bookmarks 1–19** of the table "M1 capture bookmarks", each with a `uni-app` intent test checking its stated geometry from the World (distance, phase angle, apparent radius, ring-plane elevation, Sun angle) to ±1 %, the lit-pixel count against the meter thresholds and its "Exposure on paper" row (±0.5 EV, with the CPU twins), and, for every moon in frame other than the anchor, the angle between its theory position and its Horizons spot as seen from the bookmark camera < 1 px (ruling 10; bookmarks showing moons use the 2026-11-20 00:00 UTC spot epoch).
- [ ] **Step 6: Gate.**
  1. `cargo check --workspace && cargo test --workspace --lib --bins`.
  2. `universe --selftest`: all PASS.
  3. `universe --capture bookmarks/m1.ron --out captures/wave-b --size 1920x1080 --hdr` (release); also `bookmarks/m0.ron --out captures/wave-b-m0`: it must render with no GPU error, but from here on its SSB sky views look into the Sun's photosphere 66 000 km away (review C2) and its `bh-shadow-30M` uses the default `Band` environment, so the M0 shadow regression is bookmark 18 (`DebugGrid`).
  4. Numeric checks for bookmarks 1–19 (commands in the bookmark table), written to `captures/wave-b/checks.json`; then delete the `.hdr.bin` files (≈ 0.6 GB).
  5. **Look at every PNG** with this checklist (each item has its numbers in "Exposure on paper"): no texel diamonds, block artefacts or seams on any body (the Earth close-up for BC7 blocks, ruling 2); limbs smooth; Earth's sunrise arc blue above and red below; Mars limb thin and dusty; Venus haze ring; Titan orange haze; Saturn oblate with the Cassini division and shadows both ways; Sun limb-darkened; granulation resolved in `sun-granulation`; Orion and the Sagittarius region recognisable; the band smooth and brightest toward the centre; stars visible beside sunlit bodies with the eye-adapted label; no banding in the band or atmospheres; no body clipped to flat white.
  6. Exposure sweep (ruling 9): capture `sky-orion-3d` and `sky-galactic-center` at eye-adapted star EV −3, −4, −5; V = 6.5 must sit ≥ about 7 stops above AgX black (at −4 it is 2.315 × 10⁻² = 7.04 stops); pick the default and record why.
  7. `universe --bench bookmarks/m1-bench.ron --sustained 60 --json captures/wave-b/bench.json` (Wave B subset), then `uni-verify bench --json captures/wave-b/bench.json --window last10s --max-gpu-ms 16.0 --max-cpu-ms 4.0 --max-interval-ms 16.7 --budgets`: any Solar System bookmark over the limits, or any pass over 1.25 × its budget, becomes a perf-fix task (its ladder, in order) before Wave C.
  8. `uni-verify sheet`; publish the Wave B sheet as a private Artifact (PNGs as assets); commit the wave; update CLAUDE.md §2.

---

## Wave C — small bodies, exoplanets, navigation, HUD

### Task C0: Wave C interface stubs and uni-sim additions (lead)

**Files**
- `crates/uni-sim/src/{snapshot,registry,world}.rs` (the lead implements these additions with tests; A4's owner is done):
  ```rust
  // registry
  pub struct ObjectInfo { /* A4 fields */, pub atmosphere_top_m: Option<f64> }   // equals uni-atmo's top radius (app test)
  // snapshot
  #[derive(Clone, Debug)] pub struct CometView { pub id: ObjectId, pub name: String, pub nucleus_rel_m: glam::DVec3, pub nucleus_radius_m: f64,
      pub radius_assumed: bool, pub spkid: i32, pub sun_rel_m: glam::DVec3 /* Sun − nucleus */, pub helio_vel_mps: glam::DVec3,
      pub r_au: f64, pub delta_au: f64, pub total_v: Option<f64> /* comet_total_v from the camera */, pub m1: Option<f64>, pub k1: Option<f64>,
      pub conic: uni_physics::kepler::Conic /* heliocentric ICRS */ }
  pub struct Snapshot { /* A4 fields */, pub comets: Vec<CometView>, pub sun_rel_m: glam::DVec3 /* Sun − camera */,
      pub suppressed_asteroids: Vec<u32> /* asteroid record indices drawn as registry objects (bodies or points); review I12 */ }
  // world (review I9: cards need more than objects())
  #[derive(Clone, Debug)] pub enum ObjectDetails {
      Body { facts: Option<uni_data::BodyFacts>, sat: Option<uni_data::SatRecord> },
      Star { star: uni_data::Star3, name: Option<uni_data::StarName>, archive_host: bool },
      Comet(uni_data::Comet), Asteroid { record: uni_data::AsteroidRecord, name: Option<String> },
      ExoHost { system: uni_data::ExoSystem, star: Option<uni_data::Star3> }, BlackHole { mass_kg: f64, spin: f64 } }
  impl World { pub fn details(&self, id: ObjectId) -> ObjectDetails; }
  ```
  `comets` holds every comet with r_h < 5 AU whose nucleus or coma (radius 10⁶ km) can be on screen.
- Stubs: `crates/uni-nav/src/{nav,pick,cruise,goto}.rs` + `lib.rs` lines (`input.rs` actions are C3's); `crates/uni-ui/src/{theme,panels,cards}.rs` and `crates/uni-ui/src/overlay/{mod,markers,orbits,bracket}.rs` + `lib.rs` module lines (`theme.rs` is written with the mockup's constant values, so C5 draws in the right colours before C4 merges; C4 owns it afterwards); `crates/uni-planet/src/comet/{mod,nucleus,coma,tails}.rs` + `shaders/comet.wgsl` + the `lib.rs` line and the call in `probes.rs` (B1 is done); `crates/uni-sky/src/{asteroids,belt}.rs` + `shaders/asteroids.wgsl` + the `lib.rs` lines and the call in `probes.rs`. Manifests: `uni-nav` and `uni-ui` gain `uni-sim`, `uni-physics`, `uni-data`; `uni-ui` also gains `glam` (`Bracket`, `Marker`, `OrbitLine` and `exo_orbit_points` use it; uni-nav has it already).

**Interfaces** (frozen for Wave C):
```rust
// uni-nav
#[derive(Clone, Copy, Debug, PartialEq)] pub enum FlightMode { Free, Cruise, Goto { target: uni_sim::registry::ObjectId } }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum DropOut { Atmosphere, NearStar, BlackHole, Arrived }
#[derive(Clone, Debug)] pub struct NavStatus { pub mode: FlightMode, pub speed_mps: f64, pub ftl: bool, pub dropout: Option<DropOut>,
    pub eta_s: Option<f64>, pub selected: Option<uni_sim::registry::ObjectId> }
pub struct Navigator { pub selected: Option<uni_sim::registry::ObjectId>, pub mode: FlightMode, pub cruise_log10: f64 }
pub const CRUISE_LOG10_RANGE: (f64, f64) = (-3.0, 1.0);
pub const GOTO_DURATION_S: f64 = 6.0;          // total, turn included (review m20)
pub const GOTO_TURN_RATE_DEG_S: f64 = 90.0;
impl Navigator { pub fn new() -> Self;
    pub fn pick(&mut self, snap: &uni_sim::snapshot::Snapshot, cursor_internal_px: glam::Vec2, radius_px: f32) -> Option<uni_sim::registry::ObjectId>;
    pub fn toggle_cruise(&mut self); pub fn goto_selected(&mut self) -> bool; pub fn deselect(&mut self);
    pub fn update(&mut self, cam: &mut FlyCamera, world: &uni_sim::world::World, input: &mut InputState, dt_real_s: f64) -> NavStatus; }
pub fn handle_time_keys(input: &mut InputState, time: &mut uni_sim::time::TimeController);
pub fn standoff_distance(info: &uni_sim::registry::ObjectInfo, world: &uni_sim::world::World) -> f64;
pub fn cruise_speed(d_surface_m: f64, cruise_log10: f64) -> f64;
// Action gains: Select (left click), Goto (F), Cruise (C), TimePanel (T), TimePause (P), TimeFaster (.), TimeSlower (,), Deselect (Esc)

// uni-ui
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum LabelTone { Neutral, Nav, NonPhysical, Danger }
#[derive(Clone, Debug)] pub struct CardRow { pub label: String, pub value: String, pub mono: bool, pub tone: LabelTone }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum CardAction { Goto, Moons, Close }
#[derive(Clone, Debug)] pub struct TargetCard { pub kicker: String /* "TARGET", "TARGET · BLACK HOLE" */, pub title: String, pub subtitle: String,
    pub live: Vec<CardRow>, pub facts: Vec<CardRow>, pub notes: Vec<String>, pub sources: String, pub actions: Vec<CardAction>, pub moons: Vec<String> }
pub fn build_card(world: &uni_sim::world::World, id: uni_sim::registry::ObjectId, view: &uni_core::CameraView) -> TargetCard;   // from World::details
#[derive(Clone, Debug)] pub struct LocationPanel { pub region: String, pub from_sun_m: f64, pub light_time_s: f64, pub frame: String }
#[derive(Clone, Debug)] pub struct TimePanel { pub epoch: uni_core::Epoch, pub rate: f64, pub paused: bool, pub show_warp: bool }
#[derive(Clone, Debug)] pub struct GravityPanel { pub body: String, pub local_g: f64, pub escape_mps: f64, pub circular_mps: f64 }
#[derive(Clone, Debug)] pub struct SpeedPanel { pub header: String, pub tone: LabelTone, pub speed_mps: f64, pub note: Option<String> }
pub struct HudData { /* m0.1 fields kept (fps, cpu_ms, gpu_ms, gpu_frame_ms, …, eye_adapted_stars) */ pub location: LocationPanel, pub time: TimePanel,
    pub gravity: Option<GravityPanel>, pub speed: SpeedPanel, pub target: Option<TargetCard>, pub notes: Vec<(String, LabelTone)> }
pub struct HudOutput { pub actions: Vec<CardAction>, pub selected_moon: Option<String>, pub warp_preset: Option<usize>, pub highlight_belt: Option<bool> }
// Hud::draw(..., data: &HudData, overlay: &Overlay, controls: &mut HudControls) -> (Vec<wgpu::CommandBuffer>, HudOutput)
#[derive(Clone, Debug, Default)] pub struct Overlay { pub bracket: Option<Bracket>, pub markers: Vec<Marker>, pub orbits: Vec<OrbitLine> }
#[derive(Clone, Debug)] pub struct Bracket { pub center_pt: glam::Vec2, pub half_size_pt: f32, pub name: String, pub sub: String }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum MarkerKind { Planet, Moon, Comet, Asteroid, ExoHost, Star, BlackHole }
#[derive(Clone, Debug)] pub struct Marker { pub pt: glam::Vec2, pub label: Option<String>, pub kind: MarkerKind, pub selected: bool }
#[derive(Clone, Debug)] pub struct OrbitLine { pub pts: Vec<glam::Vec2>, pub dashed: bool, pub tone: LabelTone }
#[derive(Clone, Copy, Debug)] pub struct OverlaySettings { pub planet_orbits: bool, pub labels: bool, pub exo_orbits: bool }
pub fn build_overlay(world: &uni_sim::world::World, snap: &uni_sim::snapshot::Snapshot, view: &uni_core::CameraView,
    internal_px: glam::UVec2, output_pt: glam::Vec2, selected: Option<uni_sim::registry::ObjectId>, s: &OverlaySettings) -> Overlay;

// uni-planet comets
pub const MAX_TAILED_COMETS: usize = 8;        // ruling round 2 (review I13)
pub const TAIL_REFRESH_WALL_S: f64 = 0.25;     // ruling round 2: at most every 0.25 s of wall time under time warp
#[derive(Clone, Debug, Default)] pub struct CometScene { pub comets: Vec<uni_sim::snapshot::CometView>, pub epoch: uni_core::Epoch }
pub struct CometRenderer;   // "comets"
impl CometRenderer { pub fn new(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, input: uni_gfx::SharedInput<CometScene>) -> anyhow::Result<Self>;
    pub fn cpu_ms(&self) -> f32; }   // last frame's tail CPU time (bench `comets_cpu_ms`)

// uni-sky asteroids
pub const BELT_HIGHLIGHT_GAIN: f64 = 1.0e6;   // 15 mag; amber label (round 2, review I12); arithmetic in C2
#[derive(Clone, Debug)] pub struct BeltModel { /* sampled procedural elements */ }
impl BeltModel { pub fn from_catalogue(real: &[uni_data::AsteroidRecord], n_main: u32, n_kuiper: u32, h_cut: f32, seed: u64) -> Self; pub fn len(&self) -> usize; }
#[derive(Clone, Debug, Default)] pub struct AsteroidInput { pub epoch: uni_core::Epoch, pub sun_rel_m: glam::DVec3, pub show_belt: bool,
    pub highlight: bool, pub suppressed: Vec<u32> /* Snapshot::suppressed_asteroids */ }
pub struct AsteroidField;   // "asteroids"
impl AsteroidField { pub fn new(gpu, lib, camera_bgl: &wgpu::BindGroupLayout, real: &[uni_data::AsteroidRecord], belt: &BeltModel,
    input: uni_gfx::SharedInput<AsteroidInput>) -> anyhow::Result<Self>; pub fn counts(&self) -> (u32, u32); }
```

**Steps**
- [ ] **Step 1:** Implement the uni-sim additions with tests (`comets_in_snapshot_only_when_active`, `details_cover_every_kind` (one object of each kind returns the matching variant with its record), `suppressed_asteroids_match_registry`, and `atmosphere_top_matches_atmo_presets` in uni-app); write the stubs; `cargo check --workspace`.
- [ ] **Step 2: Commit** `build: Wave C interface stubs and uni-sim details`.

---

### Task C1: uni-planet — comets (nucleus, coma, dust and ion tails)

**Files**
- Modify: `crates/uni-planet/src/comet/{mod,nucleus,coma,tails}.rs`, `crates/uni-planet/shaders/comet.wgsl`

**Interfaces**
- Consumes: `uni_sim::snapshot::CometView`, `uni_physics::{dust::{finson_probstein, DustParams, DustGrain}, kepler::Conic, photometry::{comet_total_v, reflectance, Brdf}}`, `uni_core::{pcg3d, splitmix64, star_illuminance_lux}`. The nucleus is a real mesh (not B1's cap mesh). Probes are exposed from `comet/mod.rs`; C0's stub adds the call in `probes.rs`.
- Produces: `CometRenderer` and CPU twins `nucleus_mesh(seed: u64, radius_m: f64, subdiv: u32) -> (Vec<Vec3>, Vec<u32>)`, `coma_surface_brightness(rho_m: f64, intensity_cd: f64, scale_m: f64) -> f64`, `ion_tail_dir(sun_rel: DVec3, helio_vel: DVec3) -> DVec3`, `tail_schedule(comets: &[CometView], last_refresh: &[f64], wall_s: f64) -> Vec<usize>`.

**Model (cite in code)**
- **Nucleus:** icosphere (subdivision 5, ≈ 20 k triangles, smooth normals) displaced by integer-lattice value noise seeded from `spkid` (two octaves, ±25% radius) with a two-lobe blend for seeds whose low bit is set; mean radius from `diameter_km`, else 2 km with "size assumed" in the card; card note `Nucleus shape: procedural` (review m11); lunar-Lambert, albedo 0.04 (Lamy et al. 2004, Comets II). Opaque, depth write, `lens_mask.g` = coverage. Below `POINT_LOD_PX` uni-sim sends it to points instead.
- **Coma:** total luminous intensity of the coma from the SBDB magnitude law: I = E·Δ² with E = `star_illuminance_lux(comet_total_v(M1, K1, r, Δ))`; surface brightness S(ρ) = K/ρ · e^{−ρ/ℓ} with ℓ = 10⁵ km · r_h² (photodissociation scale, Haser 1957; scaling from Combi et al. 2004, Comets II), normalised so ∫S dA = I (K = I/(2πℓ)). Colour: 50/50 mix of C₂ Swan-band green (516 nm chromaticity) and sunlight (approximate; card note). Camera-facing quad, additive, world exposure, depth test.
- **Dust tail:** `finson_probstein` with β ∈ [0.001, 0.9] (32 log steps) × age ≤ 60 d (64 steps), production ∝ r_h^(−K1/2.5) (the magnitude law); each grain from `Conic::from_state` at its emission time then `Conic::state` (an elliptic or hyperbolic Kepler solve, not universal variables; review I13); grains drawn as soft Gaussian sprites (size from neighbour spacing), brightness ∝ weight × E☉(r_h) × HG(g = 0.6) (Marcus 2007, Icarus 191, verify), total tail light = 0.5 × coma light (v0 ratio; card note).
- **Which comets get tails, and how often (ruling round 2, review I13):** only the `MAX_TAILED_COMETS` = 8 most prominent comets on screen (brightest predicted total V from the camera) get tails; the rest draw nucleus and coma only. Tails are recomputed when the epoch has moved > 1 h of sim time since the comet's last refresh, but never more often than every `TAIL_REFRESH_WALL_S` = 0.25 s of wall time, round-robin within ≤ 1 ms of CPU per frame (8 × 2 048 grains is ≈ 16 k Kepler solves, ≈ 8 ms per full refresh at ≈ 0.5 µs each, spread over 15 frames). `cpu_ms()` reports it for `--bench`.
- **Ion tail:** straight, along the aberrated anti-sunward direction û = normalize(400 km/s · r̂_h − v_comet) (Brandt 1968; solar wind 400 km/s; ruling 8), length 0.3 AU / r_h (capped 1 AU), exponential fall-off over its length, CO⁺ blue (427 nm chromaticity desaturated 50%), width 10⁵ km, brightness 0.2 × coma (v0; card note).
- Only comets with r_h < 5 AU get a coma and tails.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn nucleus_mesh_deterministic_and_closed() { /* same seed → identical; Euler characteristic 2; no zero-area triangles */ }
  #[test] fn nucleus_mean_radius_matches_diameter() { /* ±5% */ }
  #[test] fn coma_profile_integrates_to_intensity() { /* numeric ∫ S dA to 20ℓ == I ±1% */ }
  #[test] fn coma_follows_k1_law() { /* r_h 1 → 2 AU at fixed Δ: intensity ratio == 10^(−0.4·K1·log10 2) ±1e-9 */ }
  #[test] fn ion_tail_aberration() { /* v_comet ⊥ r̂ of 40 km/s → angle from anti-sun == atan(40/400) = 5.711° ±0.01°; v = 0 → exactly anti-sun */ }
  #[test] fn dust_tail_lags_motion() { /* mean grain offset · v_comet < 0 and · r̂_h > 0 */ }
  #[test] fn inactive_beyond_5au() { /* r_h 5.1 AU → no coma or tail draws; 4.9 AU → both */ }
  #[test] fn tails_capped_at_eight() { /* 20 active comets on screen: tail_schedule picks only the 8 brightest by total_v */ } // review I13
  #[test] fn tail_refresh_rate_limited_under_warp() { /* warp 1e9 for 2 s of wall time at 60 fps: each tailed comet refreshed ≤ 9 times; per-frame grain work ≤ the 1 ms share */ } // review I13
  #[test] fn comet_layouts_match_wgsl() {} #[test] fn comet_shader_validates() {}
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-planet --lib`.
- [ ] **Step 3: Implement**, with a probe for `coma_surface_brightness` (64 radii ±1e-3).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; C6's `comet-neowise-2020` ion-tail angle and `comet-halley-1986` coma-profile checks pass, and the curved dust tail is visible.

---

### Task C2: uni-sky — real asteroids and the procedural belt

**Files**
- Modify: `crates/uni-sky/src/{asteroids,belt}.rs`, `crates/uni-sky/shaders/asteroids.wgsl` (probes exposed from `asteroids.rs`, called from `probes.rs` per the C0 stub)

**Interfaces**
- Consumes: `uni_data::AsteroidRecord`, `uni_physics::{kepler::solve_kepler, frames::ecliptic_j2000_to_icrs, photometry::asteroid_v}`, `uni_core::{splitmix64, hash_u64s, star_illuminance_lux}`, the PSF and point path from B4.
- Produces: `AsteroidField`, `BeltModel`, CPU twins `kepler_f32(m: f32, e: f32) -> f32` (the shader's solver), `split_f64(x: f64) -> (f32, f32)` (double-single), `belt_highlight_v(h: f64, r_au: f64, delta_au: f64, alpha_rad: f64, highlight: bool) -> f64`.

**Rules**
- **Real objects:** each frame the CPU computes the mean anomaly in f64 (wrapped to [0, 2π)) for every record and uploads it as f32 (≤ 0.5 ms for 50 k; measured). The GPU solves Kepler in f32 (Danby starter + 3 Newton steps), builds the heliocentric ecliptic position, rotates to ICRS, and subtracts the camera's heliocentric position passed as a double-single pair (Cesium "GPU RTE"). **Precision (review I12):** the f32 orbit itself carries a few ulps of a, 2⁻²⁴ × 3 AU = 27 km per ulp, so a GPU-drawn asteroid is good to ≈ 100 km, i.e. ≤ 6 × 10⁻⁴ px at 1 AU at 1080p/60°: invisible as a point, not "metres". **Registry asteroids** (H ≤ 10 or named; uni-sim draws them from f64 CPU states as bodies or points) are in `suppressed` and emit degenerate quads here, so none is drawn twice; the highlight reaches them through C6, which multiplies their `PointSprite` illuminance by the same gain (re-review r2 minor 1). Brightness from H, G (`asteroid_v`) at the true phase angle; PSF splat at `star_pre()` × (1 − r), depth `Greater` at true depth.
- **Procedural belt** ("drawn from the catalogue's statistical distributions", design §5.4): `BeltModel::from_catalogue` builds 3-D histograms of (a, e, i) for the main belt (1.8–5.5 AU) and the Kuiper belt (30–50 AU) from the real records, samples `n_main` = 200 000 and `n_kuiper` = 50 000 objects (seeded with `splitmix64`), Jupiter Trojans with mean longitude λ − λ_J drawn from the real Trojans' distribution around ±60°, and H > `h_cut` from N(<H) ∝ 10^(αH) with α fitted to the catalogue's own counts in [h_cut − 2, h_cut] (Jedicke & Metcalfe 1998). The procedural set never contains a registry object (H > h_cut ≥ 10). M = M₀ + n·t is evaluated in f32 on the GPU (statistical objects; no targeting). The app raises `HonestyFlag::ProceduralBelt` (C6).
- **Visibility (ruling round 2, review I12):** real photometry by default, so from afar the belt is invisible, as it really is. Arithmetic for `asteroid-belt-above` (camera 6 AU above the ecliptic): an object at r = 2.8 AU is at Δ = 6.62 AU and phase 65°, so V = H + 5 log₁₀(rΔ) + HG phase term = H + 6.34 + 2.30 = H + 8.64; with N(H < 14) = 5 × 10⁴ catalogue objects and 2 × 10⁵ procedural ones over H 14–16.0 (α = 0.35), spread over the annulus (2.1–3.3 AU → 270–424 px from the Sun at fov 70°, rectilinear: 540 × (r/6)/tan 35° = 128.5 px/AU; 3.36 × 10⁵ px), the mean glow at star EV −4 is 1.9–4.2 × 10⁻⁸ × gain pre-exposed per px. The range is the total illuminance ΣE = ∫ (dN/dH)·E(H + 8.64) dH with dN/dH = 5 × 10⁴ × 0.35 ln 10 × 10^(0.35(H − 14)) and E(V) = 10^(−0.4(V + 13.99)) lux: over H 10–16, the objects the asteroid pass draws (re-review r2 minor 2's model), ΣE = 6.17 × 10⁻¹⁰ lux; with the same law run down to H 6 (this plan's round-2 figure, which also counts the registry asteroids H ≤ 10 that C6 now boosts), 1.34 × 10⁻⁹ lux. The real number of H < 10 objects lies below that law, so the truth is in between. Glow = ΣE × `star_pre` 13.33 / Ω_annulus 0.4250 sr = 1.93 × 10⁻⁸ and 4.20 × 10⁻⁸ (computed). Point sprites write E/Ω(θ) with Ω(θ) = Ω_c cos³θ (m0.1 `uni_sky::stars`), so the mean over the annulus is ΣE · star_pre / Ω_annulus, set by its solid angle, not its pixel count: Ω_annulus = 2π(cos atan(2.1/6) − cos atan(3.3/6)) = 2π(0.94386 − 0.87623) = 0.4250 sr = 2.53 × 10⁵ on-axis pixels (Ω_c = (2 tan 35°/1080)² = 1.681 × 10⁻⁶ sr); the first plan's angle-linear 2.61 × 10⁵ px gave 4.1 × 10⁻⁸ (re-review r1 minor 1; the re-review's 3.2 × 10⁻⁸ × gain divided by the rectilinear pixel count without the cos³θ). Unboosted that is 1.9–4.2 × 10⁻⁸, 13.2–12.0 stops below AgX black (1.76 × 10⁻⁴), and H = 14 is V 22.6, far under the V ≈ 11.8 visibility limit. The opt-in **"Highlight asteroid belt"** multiplies every asteroid's illuminance by `BELT_HIGHLIGHT_GAIN` = 10⁶ (15 mag), in the asteroid pass and, for the registry asteroids, in C6's point sprites: H = 14 → V 7.6, H = 16 → V 9.6, Ceres (H 3.34, G 0.12; its HG phase term at 65° is 2.37) V = 3.34 + 6.34 + 2.37 − 15 = −2.95 at the same geometry, mean annulus glow 1.9–4.2 × 10⁻² (6.8–7.9 stops above AgX black). 10⁵ would not do: its 1.9–4.2 × 10⁻³ glow is comparable to the band behind it toward the south ecliptic pole (≈ 2.3 × 10⁻³ at 22 mag/arcsec²) and to the Sun's bloom at the belt's inner edge (1.8 × 10⁻³ at 270 px, B6), so the annulus would barely stand out. The HUD shows the amber `Asteroid belt brightness exaggerated (non-physical)`.
- Budget: `asteroids` ≤ 1.0 ms GPU in `asteroid-belt-above` at 1080p (ladder: belt 250 k → 100 k → 50 k, applied by C6).

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn kepler_f32_vs_f64() { /* e < 0.5: position rel. error < 1e-6; e < 0.9: < 1e-5 */ }
  #[test] fn double_single_split() { /* hi + lo reproduces 4.5e12 m to < 1 mm */ }
  #[test] fn belt_matches_catalogue_a_distribution() { /* KS distance on a < 0.02 */ }
  #[test] fn belt_shows_kirkwood_gap() { /* density in [2.49, 2.51] AU < 0.3 × the mean of [2.40, 2.45] and [2.55, 2.60] */ }
  #[test] fn trojans_cluster_at_l4_l5() { /* procedural Trojans' λ − λ_J ∈ ±(60 ± 25)° */ }
  #[test] fn procedural_h_above_cut() { /* every procedural H > h_cut: no duplicates of catalogue objects */ }
  #[test] fn suppressed_records_not_drawn() { /* CPU twin of the vertex stage: a suppressed index yields a zero-area quad */ } // review I12
  #[test] fn highlight_gain_applied_only_when_on() { /* belt_highlight_v: H 14 at r 2.8, Δ 6.62, α 65°: 22.64 off, 7.64 on (±0.01) */ } // review I12
  #[test] fn belt_deterministic() { /* same seed → identical buffers */ }
  #[test] fn asteroid_layouts_match_wgsl() {} #[test] fn asteroid_shader_validates() {}
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-sky --lib`.
- [ ] **Step 3: Implement**, with a probe comparing GPU positions of 256 real asteroids with `uni_physics::smallbody` f64 states: ≤ 100 km (the stated f32 bound).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; C6's `asteroid-belt-above` (highlight on) shows the belt annulus and the two Trojan clouds within budget and its checks pass.

---

### Task C3: uni-nav — selection, goto, cruise, time-warp keys

**Files**
- Modify: `crates/uni-nav/src/{nav,pick,cruise,goto,input,lib}.rs`

**Interfaces**
- Consumes: `uni_sim::{world::World, snapshot::Snapshot, registry::{ObjectInfo, ObjectKind}, time::TimeController}`, A6's `FlyCamera`.
- Produces: the C0 `uni-nav` API.

**Rules**
- **Pick:** nearest `PickItem` within `radius_px` (12 px at 1080 lines, scaled to the internal height); ties by priority (bodies and the Sun > comets > named stars and exoplanet hosts > asteroids > TestBH-1 is a body), then distance.
- **Cruise** (design §5.6, labelled `CRUISE · FTL · NON-PHYSICAL · GRAVITY IGNORED`): v = 10^cruise_log10 s⁻¹ × d_surface (≥ 1 m/s), along the camera's −Z; the wheel moves `cruise_log10` within `CRUISE_LOG10_RANGE`. Drop-outs to Free at 1 km/s: crossing an `atmosphere_top_m`; within 10 photosphere radii of the Sun or a catalogue star; within 100 r_s of a black hole. The step is limited so one frame never crosses a drop-out boundary.
- **Goto (F):** reanchor to the target's frame; turn to face it at ≤ `GOTO_TURN_RATE_DEG_S` (≤ 2 s for 180°); then approach with d − d_stop shrinking exponentially over `GOTO_DURATION_S` − the turn's duration, so the whole goto takes 6 s from any distance and any starting direction (review m20); arrival when d − d_stop < 10⁻³ d_stop → Free, `DropOut::Arrived`. `standoff_distance`: bodies 4 × max radius, or 1.5 × the atmosphere top if larger; the Sun and stars 1 AU × √(L/L☉) but ≥ 10 R★; comets 10⁵ km; asteroids 10 × radius (≥ 50 km); exoplanet hosts as stars; TestBH-1 100 GM/c² (the mockup's exit point). `velocity_mps` is set for the HUD's closing speed; `ftl` when speed > c.
- **Time keys:** P pauses, `.`/`,` step the warp presets, T toggles the time-warp panel (UI state). Space stays "up" (ruling 7).

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn pick_nearest_with_priority() { /* a planet 8 px and a named star 5 px from the cursor → the planet; two stars → the nearer */ }
  #[test] fn pick_none_beyond_radius() { /* nearest item 13 px away at 1080 lines → None */ }
  #[test] fn cruise_speed_scales_with_distance() { assert!((cruise_speed(1e6, 0.0) - 1e6).abs() < 1e-6); }
  #[test] fn cruise_drops_out_at_atmosphere_top() { /* from 1e5 km toward Earth: final altitude ≥ atmosphere top − 1 m; dropout Atmosphere */ } // Review Focus 2
  #[test] fn cruise_drops_out_near_sun_and_black_hole() { /* toward the Sun: stops at ≥ 10 R☉, NearStar; toward TestBH-1: stops at ≥ 100 r_s, BlackHole */ }
  #[test] fn goto_stops_outside_atmosphere() { /* target Earth: final distance == standoff ±0.1%, > atmosphere top */ } // Review Focus 2
  #[test] fn goto_duration_independent_of_distance() { /* from 1 AU and from 1 pc, facing the target and facing away (180° turn): 6.0 ± 0.25 s including the turn */ } // review m20
  #[test] fn goto_follows_moving_target() { /* target frame at 30 km/s: final distance within 1% of standoff */ }
  #[test] fn ftl_flag_above_c() { /* cruise speed 2.9e8 → ftl false; 3.1e8 → true */ }
  #[test] fn time_keys() { /* P toggles pause; '.' 1× → 1e3; ',' back to 1× */ }
  #[test] fn new_actions_mapped() { /* F, C, T, P, '.', ',', Esc → actions; Space still Up */ }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-nav --lib`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; in C6's window run, clicking Mars and pressing F arrives in about 6 s, and cruise drops out at Earth's atmosphere.

---

### Task C4: uni-ui — the approved HUD: panels, cards, theme, fonts

**Files**
- Modify: `crates/uni-ui/src/{hud,theme,panels,cards,format,lib}.rs`

**Interfaces**
- Consumes: the mockup (https://claude.ai/artifact/2HExiDvXNRMPruxHppDE5B), `data/fonts/*.ttf` (A9, embedded with `include_bytes!`), `uni_sim::world::{World, ObjectDetails}` (cards, via `World::details`), `uni_physics::calc`, the m0.1 HUD fields.
- Produces: the C0 `uni-ui` panel and card API, `Hud::draw` v2, and the UI-state labels in `theme` (free flight, cruise, belt highlight, the M1 lens wording).

**The design (from the mockup's four artboards, 1440 × 900 pt; egui points equal CSS px)**
- **Type:** Inter (300/400/500/600) for text; IBM Plex Mono (300/400/500) for every number. Section kickers 11 pt, weight 500, letter-spacing 0.12 em, uppercase, `#8D97A6`. Body rows 13 pt; labels `#9AA4B2`; values `#E6EAF0`, mono, right-aligned.
- **Colour carries meaning only:** cyan `#7DD3FC` = selected target / navigation; amber `#F5B04C` = non-physical or hypothetical; red `#F87171` = danger; everything else neutral grey on translucent glass. Panels: fill `rgba(10,14,22,0.55)`, 1 pt border `rgba(255,255,255,0.10)`, radius 10, padding 14×16. Target card: fill 0.62, border `rgba(125,211,252,0.35)`, radius 12, padding 20×22. egui has no backdrop blur; translucency only.
- **In flight:** Location (top-left, 24 pt margins, 300 wide: kicker LOCATION, region 18 pt, rows From Sun / Light time / Frame); Time (top-right, 280: kicker TIME, epoch mono 15 pt, Rate row "× 1 000"); Gravity (bottom-left, 260: kicker `GRAVITY · <BODY> · NOT APPLIED`, Local g, Escape v, Circular v — calculators; ruling 6); Speed (bottom-centre, 520: header in its tone — amber `FREE FLIGHT · GRAVITY NOT SIMULATED` in free flight (ruling 6) — speed mono 300 weight 40 pt, sub-line as a fraction of c, a log bar from 0.1 m/s to 10¹⁸ c with an amber tick at c); perf line bottom-right (mono 11 pt, `{fps} fps · GPU {ms} ms · CPU {ms} ms`, "n/a" for invalid timings, from m0.1's `gpu_frame_ms`) and hints "Tab hide HUD · F1 controls".
- **Notes stack** (honesty and warnings) under the Location panel, 12 pt, in their tone: the world labels from `World::honesty`, the eye-adapted label exactly when `ExposureSettings::eye_adapted_in_effect()` (the app passes it as `HudData.eye_adapted_stars`; review m13), the belt-highlight label while the toggle is on, the M1 lens labels (`Lens background: M0 debug grid`, `Stars near the hole are not yet lensed (M3)`, replacing m0.1's "(M0)" wording), and renderer warnings ("Inside Sun", "No surface map: …").
- **Target card** (right, 420): kicker TARGET, title 30 pt light, subtitle; live rows (Distance, Closing speed, Light time) between hairlines; facts rows; buttons ≥ 44 pt high: "Go to · F" (cyan), "Moons" (lists moons; clicking one selects it), "Close · Esc"; sources line 11 pt; notes (the info-card honesty labels). "Hold orbit · O" is not drawn until M3. While a card is open the Time panel collapses to one mono line at top-centre (ruling 15). Card contents per `ObjectDetails` kind: body (mockup's Mars rows; Earth adds `Clouds: static map (no weather)`, Venus `Cloud tops: static map, super-rotating (approximate)`, Saturn `Rings: Solar System Scope texture; phase function approximate`), star (distance, apparent V from here, M_V, spectral type, T_eff, luminosity, radius "estimated (Torres 2010 BC)", radial velocity, proper motion; `Star motion: tangential only (no radial velocity)` for `FLAG_NO_RV`; `Host star from NASA Exoplanet Archive (not in AT-HYG)` for archive hosts), comet (q, e, period, next perihelion, r_h, total magnitude from here, nucleus diameter, orbit-epoch, tail-model and `Nucleus shape: procedural` notes), asteroid (a, e, i, H, diameter, albedo, period), exoplanet host (star rows + one row per planet: a, P, mass or radius, method, year; orientation notes), TestBH-1 (mass, r_s, distance; "Full black-hole card in M3").
- **Time warp** (T): the panel grows to 380 with six buttons Pause / 1× / 10³ / 10⁶ / 10⁹ / 10¹² (≥ 44 pt, selected one cyan) and the mockup's explanatory line per rate ("One second is about 12 days. Moons race around planets." etc.).
- **Cruise:** Speed panel border `rgba(245,176,76,0.40)`, header amber `CRUISE · FTL · NON-PHYSICAL · GRAVITY IGNORED`, sub-line "auto drop-out near stars, atmospheres and black holes".
- **Settings toggle** "Highlight asteroid belt" (F1 controls list and the settings panel; `HudOutput::highlight_belt`).
- Number formatting per the mockup (ruling 16): grouped thousands with a space ("14 200 km", "3 389.5 km"), "6.417 × 10²³ kg", "24 h 37 m 23 s", "686.98 d", "12 min 40 s". m0.1's `format_distance` tests change accordingly (C4 owns `format.rs`).

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn theme_matches_mockup() { /* every colour/alpha/radius constant equals the values above */ }
  #[test] fn grouped_numbers() { assert_eq!(format_distance(1.42e7), "14 200 km"); /* "3 389.5 km", "6.417 × 10²³ kg", "24 h 37 m 23 s", "12 min 40 s" */ }
  #[test] fn speed_bar_c_tick() { /* tick at (log10 c + 1)/(log10(1e18 c) + 1) = 0.3449 ±1e-6 */ }
  #[test] fn mars_card_rows_in_mockup_order() { /* fake world with Mars facts: labels in order: Mean radius … Moons */ }
  #[test] fn card_notes_from_details() { /* Earth → clouds note; a FLAG_NO_RV star → tangential note; an archive host → archive note; a comet → nucleus note; exact strings */ } // review m11
  #[test] fn exo_card_notes_when_orientation_unknown() { /* planet without Ω and i → both orientation notes, exact strings */ }
  #[test] fn cruise_header_is_amber() { /* headless egui pass: the cruise header shape has the amber colour */ }
  #[test] fn belt_highlight_label_follows_toggle() { /* toggle on → amber label in notes; off → absent */ }
  #[test] fn perf_line_na() { /* gpu_frame_ms None → "GPU n/a" */ }
  #[test] fn hidden_hud_draws_nothing() { /* visible = false → the headless pass emits no shapes */ }
  #[test] fn fonts_load() { /* FontDefinitions contain "Inter" and "IBM Plex Mono" families */ }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-ui --lib`.
- [ ] **Step 3: Implement**, following the skeleton's egui 0.37 patterns (`run_ui`, drained texture deltas).
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; C6's `hud-target-mars` and `hud-cruise` captures match the mockup's layout, type and colours (looked at side by side) and their JSON checks pass.

---

### Task C5: uni-ui — overlays: target brackets, markers, orbits, exoplanet hosts

**Files**
- Modify: `crates/uni-ui/src/overlay/{mod,markers,orbits,bracket}.rs`

**Interfaces**
- Consumes: `uni_sim::{world::World, snapshot::Snapshot}`, `uni_physics::{kepler::Conic /* from_state */, standish}`, `uni_data::ExoSystem`, C4's `theme` (values written by C0).
- Produces: `build_overlay`, `draw_overlay(ui: &egui::Ui, o: &Overlay)`, and `pub fn exo_orbit_points(sys: &uni_data::ExoSystem, planet: usize, host_icrs_dir: glam::DVec3, n: usize) -> (Vec<glam::DVec3>, bool /* orientation assumed */)`.

**Rules**
- **Bracket** (mockup): four cyan corner brackets, 18 pt arms, 1.5 pt stroke, half-size = max(projected radius + 12 pt, 20 pt); below it the name (12 pt, 600, letter-spacing 0.14 em, cyan) and "distance · closing speed" (mono 13 pt).
- **Markers:** 6 pt grey rings at 70% alpha with labels for planets and dwarf planets always, moons when their parent is the dominant frame, comets and H ≤ 10 asteroids within 5 AU, exoplanet hosts (ring + centre dot) within 100 pc; no marker over a disk larger than 20 pt (label only); selected marker cyan.
- **Orbits:** 256-point polylines of the current osculating conic (`Conic::from_state` of the body's state; planets about the Sun, moons around the selected planet; comets and asteroids when selected), projected in f64 from the camera-relative positions, split at the near plane so no segment crosses the screen, antialiased 1 pt grey at 35% alpha (selected: cyan).
- **Exoplanet orbits** (when the host is selected or within 0.05 pc; ruling 13; review m9): sky-plane geometry in the convention of Winn 2010 ("Transits and Occultations", eqs. 1–2): **+Z points toward the observer** (from the host toward the Sun), X along the node line at position angle Ω from celestial north, Y completing the frame; z = r sin(ω + f) sin i, so the planet transits (in front of the star) at true anomaly f = 90° − ω with ω the planet's argument of periastron. A7 recorded the archive's definition of `pl_orblper`; when it is the star's ω★ (the usual radial-velocity convention), ω = ω★ + 180°; the code says which. Unknown Ω → 0°, unknown i → 90°, drawn dashed with the card notes; a planet dot where `t_transit_jd` and the period place it.
- Internal-pixel positions are scaled to output points (`output_pt / internal_px`).

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn orbit_polyline_matches_conic() { /* closes for e < 1; every point at the conic's radius ±1e-9 relative */ }
  #[test] fn near_plane_split() { /* orbit passing behind the camera → no segment longer than the screen diagonal */ }
  #[test] fn exo_sky_plane_geometry() { /* Ω 0, i 90°: orbit plane contains the line of sight and north; i 0: plane ⟂ line of sight */ }
  #[test] fn exo_transit_in_front_of_star() { /* at t_transit the planet's Z toward the observer is > 0 (between star and Sun); half a period later < 0 */ } // review m9
  #[test] fn exo_dashed_when_assumed() { /* Ω missing → dashed; Ω and i known → solid */ }
  #[test] fn bracket_size_from_radius() { /* radius 3 pt → half-size 20; radius 100 pt → 112 */ }
  #[test] fn markers_suppressed_over_large_disks() { /* body disk 25 pt → label, no ring; 15 pt → ring and label */ }
  #[test] fn overlay_scales_internal_to_output() { /* internal 1662×1080, output 1470×956 pt: internal (831, 540) → (735, 478) */ }
  ```
- [ ] **Step 2: Confirm they fail:** `cargo test -p uni-ui --lib`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm they pass.**
- [ ] **Step 5: Commit** (pathspec).

**Done when:** tests pass; C6's `exo-trappist1` shows seven orbits around the drawn host, dashed where assumed.

---

### Task C6: Wave C integration and the M1 gate (lead)

**Files:** `crates/uni-app/src/*`, `bookmarks/m1.ron` (entries 20–26), `bookmarks/m1-bench.ron`, `CLAUDE.md`.

- [ ] **Step 1: Wire** `comets` (pass 7) and `asteroids` (pass 14, `AsteroidInput.suppressed` from the snapshot, `highlight` from the HUD toggle or the bookmark's `highlight_belt`); while the highlight is on, the app also multiplies by `BELT_HIGHLIGHT_GAIN` the `illuminance_lux` of every `PointView` whose `id` is a registry object made from an `AsteroidRecord` (the objects behind `Snapshot::suppressed_asteroids`, Ceres included whatever its `ObjectKind`) before it becomes a `PointSprite`, so the registry asteroids (H ≤ 10 or named) brighten with the belt the asteroid pass draws (re-review r2 minor 1; app test `belt_highlight_boosts_registry_points`: Ceres's sprite ×10⁶ with the highlight on and unchanged off, planets' and stars' sprites unchanged). A registry asteroid resolved as a body (≥ `POINT_LOD_PX`) is not boosted, so it dims by the gain as it crosses into a disk with the highlight on; the amber label already says the brightness is exaggerated. Raise `ProceduralBelt` while the belt is drawn and either the highlight is on or the camera is within 6 AU of the Sun; `Navigator` between input and the camera (click → `pick`, F, C, Esc), `handle_time_keys`, HUD v2 data (`location`, `time`, `gravity`, `speed`, `target` via `build_card`, `notes` from `world.honesty` and UI state), `build_overlay`; `HudOutput` actions (Go to, Moons, warp presets, highlight toggle). Captures with `--hud` honour the bookmark's `target`, `cruise` and `highlight_belt`. The capture JSON gains `hud.card_rows`, `metrics.asteroid_counts`, `metrics.belt_radial_hist` and `metrics.overlay_orbits`; bench JSON gains `comets_cpu_ms`.
- [ ] **Step 2: Bookmarks 20–26**, each with its intent test (as in B9 Step 5).
- [ ] **Step 3: M1 gate.**
  1. `cargo check --workspace && cargo test --workspace --lib --bins` green; record target size and free disk.
  2. `universe --selftest`: every probe PASS.
  3. **Standish:** re-run `cargo test -p uni-physics --lib standish_within_error_table -- --nocapture`; paste the measured table into the sheet: every planet within Standish's 1800–2050 λ, φ, ρ errors; Pluto within Meeus's stated accuracy.
  4. `universe --capture bookmarks/m1.ron --out captures/m1 --size 1920x1080 --hdr` (release) and `bookmarks/m0.ron --out captures/m1-m0` (renders without GPU errors; the shadow regression is bookmark 18); every numeric check in the bookmark table PASS, written to `checks.json`; delete the `.hdr.bin` files (≈ 0.86 GB).
  5. **Look at every PNG** (the B9 checklist plus): comets' curved dust tail and straight blue ion tail; the highlighted belt annulus and Trojan clouds with the amber label; exoplanet orbits around TRAPPIST-1; HUD captures against the mockup; the amber labels present exactly where the honesty table says.
  6. **Constellations recognisable:** `sky-orion-3d`, `sky-galactic-center`, `sky-orion-50kyr` (distorted but traceable).
  7. **Performance:** `universe --bench bookmarks/m1-bench.ron --size 1920x1080 --sustained 60 --json captures/m1/bench.json`, then `uni-verify bench --json captures/m1/bench.json --window last10s --max-gpu-ms 16.0 --max-cpu-ms 4.0 --max-interval-ms 16.7 --budgets`: PASS for every Solar System bookmark (≥ 60 FPS at 1080p internal, sustained). A FAIL, or a pass over 1.25 × its budget, becomes a perf-fix task with that pass's ladder; the belt step (250 k → 100 k → 50 k) is applied here (ruling 11 says B9, but the belt is built in Wave C; review m18). M1 is not tagged until it passes.
  8. **Window run:** release build, 60 s with `RUST_LOG=warn`: no GPU errors (`take_errors` empty); the HUD FPS line read in `start` and `saturn-ring-plane` (window at native Retina size, 1080 internal lines); fly Earth → Mars by goto, cruise to Saturn, warp 10⁶; note anything odd as fix tasks.
  9. `uni-verify sheet` with checks and performance (first and last 10 s); publish the M1 sheet as a private Artifact; update CLAUDE.md (§2 layout, measured numbers, honesty labels in §3); commit; `git tag m1`.

---

## M1 capture bookmarks (`bookmarks/m1.ron`)

1920×1080 captures; fov_y 60° unless stated; Auto exposure with eye-adapted stars unless stated; `lens_env` omitted means `Band`; positions use `anchor` + `look_at` (A6). The lead computes the offsets numerically from the World at each epoch, and a `uni-app` intent test checks each stated geometry (±1 %), the lit-pixel count and the "Exposure on paper" row (±0.5 EV), and, for every moon in frame other than the anchor, the theory-vs-Horizons error < 1 px (ruling 10). "disk", "points", "shadow", "bench" are PNG/JSON checks; "HDR …" checks read the `--hdr` sidecar (`ratio`, `stats`, `samples`, `profile`, `angle`, `flux`; B8), with the regions, samples and features written into the capture JSON by the app (B9 Step 2). Geometry numbers below were computed for this plan (projected radii by (h/2)·tan(asin(R/d))/tan(fov/2)).

| # | Name | Where / when | Purpose | Numeric checks | Wave |
|---|---|---|---|---|---|
| 1 | `start` | Earth from 60 000 km (radius 99.9 px), phase ≈ 60°, Moon in frame (≈ 4.2 px); 2026-11-20 00:00 UTC | window start; Earth atmosphere, terminator, Moon | points: Moon ±2 px; intent: Moon vs Horizons < 1 px | B |
| 2 | `earth-limb-dawn` | 420 km altitude, 22° of arc into the night side from the terminator, looking horizontally toward the Sun's azimuth: the limb lies 20.26° below horizontal at 2 351 km, and the Sun is ≈ 1.8° below it (its ray from the camera passes 74 km under the surface), so the limb shows the sunrise arc | limb glow, reddened low layers, stars above the limb | HDR ratio `limb_arc_high` (tangent heights 30–60 km) B/R ≥ 1.5 and `limb_arc_low` (0–12 km) R/B ≥ 1.5; atmosphere pass ms recorded | B |
| 3 | `earth-map-closeup` | 1 500 km above the Strait of Gibraltar (36.0°N, 5.6°W), 30° off nadir toward the coast, fov 30°; 2026-11-20 12:00 UTC (Sun ≈ 34° up: 90° − 36° − 19.7°); one 8k texel (4.9 km) ≈ 6.6 px, the magnified path | BC7 block artefacts (ruling 2), bicubic magnification, aerial perspective (review m17) | HDR samples: 16 points in low-gradient map areas vs the CPU twins (map texel × BRDF × sunlight × transmittances + inscatter) ±10 % | B |
| 4 | `moon-full` | Moon at phase ≤ 2° (terminator sliver 0.16 px), 20 000 km from its centre, fov 20° (a = 267.0 px); `eye_adapted: false` (strict: the band's brightest 1.44 × 10⁻² at star EV −4 becomes 1.44 × 10⁻² × 2^(−4 − 15.3) = 2.2 × 10⁻⁸, 2 × 10⁴ below the disk threshold 4.49 × 10⁻⁴; re-review r1 NP2); intent: not eclipsed | airless BRDF, map, sharp limb | disk: a = 267.0 ± 1 px, b/a = 1.000 ± 0.003 (± 0.8 px); HDR samples 16 points ±10 % | B |
| 5 | `mars-reference-angle` | Mars a = 425 px (850 px across) at fov 60°, i.e. 8 210 km from its centre; phase ≈ 45°, lit limb left, north up | the reference's viewing angle, on rendering qualities only (spec §8) | HDR samples 16 points ±10 %; HDR ratio `limb_haze` R/B ≥ 1.0; intent: projected radius 425 px from the World ±1 %. No `disk` check (re-review r1 NP2): B3's dust haze (τ 0.5, H 11 km, g 0.7 at 135°) crosses the 4.49 × 10⁻⁴ threshold at a tangent height of ≈ 73 km, and near this limb 1 km of tangent height is ≈ 0.15 px (the derivative in h of 540·tan(asin((R+h)/d))/tan 30°, with R = 3 396.2 km, the equatorial radius that also gives the 425 px (424.97 px), and d = 8 210 km: (935.31/8 210)·(1 − 0.41367²)^(−3/2) = 0.1510 px/km; 73 km → 11.09 px exactly; re-review r2 minor 3), so the fitted edge would sit ≈ 11 px outside the solid limb (≈ 7 px with ten times fainter haze: 73 − 11 ln 10 = 47.7 km → 7.2 px; an estimate) | B |
| 6 | `jupiter-galileans` | Jupiter from 5.0×10⁶ km (28.8 px), phase < 10°, in its equatorial plane ±2°, pole up, fov 30°; 2026-11-20 00:00 UTC (the spot epoch); intent: all four moons inside the frame (Callisto ≤ 20.6° from Jupiter, half-width 25.5°; review C4) | moons on their theories | points: every moon the intent test finds visible (not occulted, not in Jupiter's shadow; at least three) ±2 px; intent: each moon vs Horizons < 1 px | B |
| 7 | `jupiter-disk` | Jupiter from 3.0×10⁶ km on the Jupiter–Sun line (phase ≤ 1°), fov 3.5° (a = 421.3 px); `eye_adapted: false` (strict, re-review r1 NP2: the band's 1.44 × 10⁻² becomes 1.44 × 10⁻² × 2^(−4 − 12.9) = 1.2 × 10⁻⁷, 3 800× below the threshold, and the +8.48 EV point gain of fov 3.5° leaves only stars brighter than V 4.4 above it, as isolated points the fit ignores; Lambert limb bias at the threshold 421.3·(1 − √(1 − (4.49 × 10⁻⁴/0.123)²)) = 0.003 px); intent: no moon and no moon shadow on or within 5 px of the disk (a shadow stops a dozen rays inside the limb and the direct fit is not robust to that; re-review r1 minor 4), else the lead moves the epoch by whole hours until none is | oblateness (review C4: at 3×10⁶ km and fov 30° Jupiter was 48 px, so ±0.004 was ±0.19 px) | disk: a = 421.3 ± 1 px, b/a = √((b/a)²cos²β + sin²β) for the sub-camera latitude β (0.9351 at β = 0, 0.9353 at 3.1°) ± 0.004 (± 1.7 px) | B |
| 8 | `saturn-ring-plane` | 1.5×10⁶ km, 4° above the lit ring face (the south face: B′ ≈ 8.5° S on the date), phase < 10°, fov 20° (Saturn a = 123.1 px; Cassini division 240–249 px from the centre); 2026-11-20 | Cassini division, ring shadow on the globe | HDR ratio `ring_cassini` / `ring_b` (luminance, same azimuth) ≤ 0.5; HDR ratio `globe_ring_shadow` / `globe_lit_ref` ≤ 0.5 (globe points the app picks where the CPU twin's ring transmittance < 0.3 and the line of sight misses the rings; the B-ring shadow lies at ≈ 4.5–8.3° N) | B |
| 9 | `saturn-edge-on` | in the ring plane at Saturn's 2025 equinox (the epoch where the World's solar ring-plane elevation B′ crosses 0, ≈ 2025-05-06; intent \|B′\| < 0.5°, so the Sun is in the ring plane too and the phase can be ≤ 1°), 2.0×10⁶ km, fov 4.3° (a = 433.6 px); `eye_adapted: false` (strict, re-review r1 NP2: band 1.44 × 10⁻² × 2^(−4 − 11.1) = 4.1 × 10⁻⁷, 1 100× below the threshold; stars above it only brighter than V 5.3; limb bias 0.003 px); intent: no moon and no moon shadow on or within 5 px of the disk (moon shadow transits are frequent at equinox; \|B′\| < 0.5° leaves about ±30 days to choose from) | ellipsoid rendering (review C4: at fov 60° Saturn was 28.2 px, so ±0.004 was ±0.11 px; on 2026-11-20 an edge-on camera has phase ≥ 8.5° and a 4.8-px polar terminator sliver) | disk: a = 433.6 ± 1 px, b/a = 0.9020 ± 0.004 (± 1.7 px) with `--exclude-angle 0 --exclude-width 10` (the ring line and its opposite) | B |
| 10 | `saturn-night-side` | phase 150°, 25° above the unlit (north) face, 1.5×10⁶ km, fov 20° | planet shadow on rings, forward-scattered ring light | HDR ratio `ring_in_planet_shadow` / `ring_lit_same_radius` ≤ 0.2 | B |
| 11 | `sun-closeup` | 0.02 AU from the Sun's centre looking at it (disk radius 481.8 px, whole disk in frame), fov 30° | limb darkening | HDR ratio `mu_0p5` / `disk_centre` (G channel) = LD₅₅₀(0.5) from B2's twin ± 3 %; JSON ev100 within limits, no NaN | B |
| 12 | `sun-granulation` | 3.0×10⁸ m above the photosphere looking at the Sun's centre, fov 30°: cells 8.7 px at the frame centre, 6.7 px (foreshortened) in the corners; the limb (44.3° off-axis) is out of frame | resolved granulation (review I14: at `sun-closeup` cells are 1.14 px and fade out) | HDR stats `granulation_patch` (central 256×256 px) relative RMS ∈ [0.08, 0.20] | B |
| 13 | `venus-crescent` | phase 155°, 150 000 km, fov 20° (cloud-top radius 125.1 px) | haze ring around the night limb | HDR stats `night_limb_haze` mean pre-exposed ≥ 1.4 × 10⁻³ (3 stops above AgX black) | B |
| 14 | `titan-backlit` | phase 160°, 60 000 km, fov 20° (131.6 px) | orange haze, detached layer | HDR ratio `haze_ring` R/B ≥ 1.2; HDR stats `haze_ring` mean pre-exposed ≥ 1.4 × 10⁻³ | B |
| 15 | `sky-orion-3d` | anchor `Sun`, offset **+100 AU × the view direction v** (camera = Sun + 100 AU·v, looking along +v, so the Sun is directly behind it; re-review r1 C2: the round-1 text's −100 AU put the Sun at the frame centre), looking at RA 83.8°, Dec −1.2°, north up; 2026-11-20 | AT-HYG sky; constellation recognisable | points: Betelgeuse, Rigel, Bellatrix, Alnilam ±1 px; HDR flux Rigel ±5 % (B8 `flux`: Ω_c·cos³θ per pixel for Rigel's 8.69° off-axis, R = its 13.5-px footprint, annulus-median band subtracted; re-review r2 NPA); intent: the other catalogue stars inside Rigel's flux radius sum to < 1 % of its illuminance (13.5 px × 0.0613°/px on the axis ≈ 0.8°), the Sun ≥ 90° from the axis (here 180°) and ≥ 10 AU away (here 100 AU), and TestBH-1 (SSB + (1000, 0, 0) AU) outside the 49.66° half-diagonal: cos θ = (1000 x̂·v − 100)/\|1000 x̂ − 100 v\| with x̂·v = cos 1.2° cos 83.8° = 0.1080 gives (108.0 − 100)/994.2 → θ = 89.5° (review C2) | B |
| 16 | `sky-galactic-center` | as 15, toward RA 266.405°, Dec −28.936° (from the SSB on this date the Sun's centre was 10.0° from this direction and its photosphere 66 000 km away, filling a 65.9° cone; review C2) | band v0, dust lane, star density | HDR ratio (÷ `pre[1]`, median) `band_plane` (b ≈ −5°, \|l\| < 10°) / `band_high_lat` (\|b\| ≈ 25°) ≥ 2; JSON `band_gc_over_anticentre` ≥ 2 (CPU twin); intent as 15 (Sun 180°; TestBH-1: x̂·v = cos 28.936° cos 266.405° = −0.0549, (−54.9 − 100)/1010.4 → 98.8°) | B |
| 17 | `alpha-cen-sun` | 0.05 pc from α Cen A **toward the Sun**, on the Sun–A line, looking at the Sun: RA 39.902°, Dec +60.834° (α Cen's antipode); fov 20°. α Cen A (V −0.01 + 5 log₁₀(0.05 pc/1.325–1.348 pc) ≈ −7.1 from there) and B (≈ −5.8) are 180° off-axis, behind the camera; the Sun is V = 4.83 + 5 log₁₀(1.2975/10) = 0.40 at 1.2975 pc (if the catalogue puts A at 1.3475 pc; the app uses its own value). Re-review r1 NP1: the round-1 placement (beyond A, looking back) put A, 10^(0.4 × (0.56 + 7.13)) ≈ 1 180× brighter than the Sun, on the Sun's pixel | 3-D star positions, the Sun as a star | points: Sun ±1 px; JSON `sun_v_mag` = M_V☉ + 5 log₁₀(d/10 pc) ± 0.01; HDR flux Sun = `star_illuminance_lux(sun_v_mag)` ± 5 % (B8 `flux`: R = the Sun's 21.0-px footprint, annulus-median band subtracted; re-review r2 NPA); intent: α Cen A and B ≥ 90° off-axis, no catalogue star brighter than the Sun within 64 px (`MAX_RADIUS_PX`, the largest footprint any star can have; 64 × 0.0187°/px = 1.20° at fov 20°) of it, and the other catalogue stars inside the flux radius (the 21.0-px footprint, 21.0 × 0.0187° = 0.39°; not the 64 px) sum to < 1 % of the Sun's illuminance (one star 5 mag fainter alone is 1 %) | B |
| 18 | `bh-shadow-30M-grid` | m0.1's `bh-shadow-30M` with `lens_env: DebugGrid`, manual exposure | M0 regression | `uni-verify shadow --expected-px 158.745 --tol 1.0` | B |
| 19 | `bh-lens-band` | static, 30 M from TestBH-1 on the side away from the Galactic centre, looking through the hole toward the GC; the Sun is 86.9° off-axis, outside the 49.7° half-diagonal; `lens_env: Band`, auto exposure | the band lensed with no seam (B7; review m17) | `uni-verify shadow --hdr-json … --expected-px 158.745 --tol 1.0`; HDR ratio (median) `lens_rect_inner_edge` / `lens_rect_outer_edge` = 1 ± 0.1 | B |
| 20 | `comet-neowise-2020` | anchor Earth, 7 000 km from its centre toward the Sun (629 km altitude, outside the atmosphere); 2020-07-15 03:00 UTC; looking at C/2020 F3; fov 20° | curved dust tail, straight blue ion tail | points: nucleus at the Horizons RA/Dec (`sky_spots.ron`) projected, ±5 px (= 0.094° at fov 20°; review m2); HDR angle `ion_tail` ±1° of `ion_tail_dir` projected | C |
| 21 | `comet-halley-1986` | 1P from 2×10⁶ km, side view; 1986-03-13; fov 20° (653 km/px) | nucleus, coma, tails | HDR profile `coma`: log–log slope between 8 and 40 px within ±0.1 of the model's −1 − ⟨ρ/ℓ⟩ (ℓ = 10⁵ km · r_h²), written by the app | C |
| 22 | `asteroid-belt-above` | 6 AU above the ecliptic, looking at the Sun; fov 70°; `highlight_belt: true` | belt, Trojan clouds, budget | JSON counts (real, procedural); JSON `belt_radial_hist` (projected CPU positions) shows the 2.50 AU Kirkwood dip and L4/L5 peaks; HDR ratio (÷ `pre[1]`, `--stat mean`) `belt_annulus` (2.1–3.3 AU, 270–424 px from the Sun, rectilinear 128.5 px/AU) / `belt_outer` (3.6–4.6 AU, 463–591 px; partly beyond the 540-px half-height, so the app samples only on-screen pixels) ≥ 3 (expected from means (1.9–4.2 × 10⁻² + 2.3 × 10⁻³)/2.7 × 10⁻³ ≈ 8–16: the highlighted belt's 1.9–4.2 × 10⁻² (C2's range) plus ≈ 2.3 × 10⁻³ of band, against ≈ 2.7 × 10⁻³ of band and sparse Hildas; the Sun's bloom is < 10⁻⁶ there, B6. The check reads the mean, not the median: for a field of point sprites (0.74 per px, σ 0.65 px) the median is 0.59 × the mean, which would put the median ratio near 5 and below 3 if the catalogue's N(H < 14) is under 2.5 × 10⁴, while the mean ratio stays ≈ 4.4 even then. Before judging, C6 recomputes the expected mean ratio from A9's measured N(H < 14) and the band CPU twin and logs it next to the measured one; re-review r2 minor 2, r3 minor 2); amber label in JSON honesty; bench within budget | C |
| 23 | `exo-trappist1` | 0.2 AU from TRAPPIST-1, `--hud`, target TRAPPIST-1 | host drawn (archive star, I11), 7 orbits, card | JSON overlay orbit count == 7; intent: TRAPPIST-1 in `star_surfaces` or `points` | C |
| 24 | `hud-target-mars` | 5 with `--hud`, target Mars | target card per mockup | JSON `hud.card_rows` equals the mockup's Mars order; visual against the mockup | C |
| 25 | `hud-cruise` | between Earth and Mars, `cruise: true`, `--hud` | amber cruise labelling | JSON honesty contains the cruise label | C |
| 26 | `sky-orion-50kyr` | 15 at 2026 + 50 000 yr, `--hud` | proper motion over 50 kyr; extrapolation label | points: Betelgeuse, Rigel, Bellatrix, Alnilam ±1 px (their +50 kyr positions); JSON honesty contains the planets-extrapolated label and not the stars one | C |

`bookmarks/m1-bench.ron`: `start`, `earth-limb-dawn`, `earth-map-closeup`, `mars-reference-angle`, `jupiter-galileans`, `saturn-ring-plane`, `sun-closeup`, `sky-orion-3d` (the Wave B gate uses these), plus `comet-neowise-2020` and `asteroid-belt-above` (Wave C).

`bookmarks/m0.ron` is still captured at each gate for GPU errors, but from Wave B on its SSB sky views include the Sun and its `bh-shadow-30M` uses the `Band` environment; the M0 shadow regression is bookmark 18.

### Exposure on paper (ruling round 2, review C3)

Every metered bookmark worked through the B6 rule: Lambert or lunar-Lambert disks with ρ from p_V (A3), sunlight at the World's distances on the bookmark date (Venus 0.721 AU, Earth 0.988, Mars 1.614, Jupiter 5.321, Saturn 9.42; Saturn 9.588 at the 2025 equinox), the radiance distribution sampled over the lit disk, then ev_avg, ev_hi and the target. Atmospheres are left out except where noted; they add light, and the highlight rule caps what they add. A pixel's pre-exposed value is L/(1.2·2^EV); AgX white is 2^4.026 = 16.29, AgX black 2^−12.474 = 1.76 × 10⁻⁴, and "visible" means ≥ 1.4 × 10⁻³ (3 stops above black). B9's intent tests re-derive each row with the CPU twins and must agree within 0.5 EV.

| Bookmark | Lit px (of 2 073 600) | Metered scene EV | Brightest lit pixel → pre-exposed | 5th-percentile lit pixel → pre-exposed | Notes |
|---|---|---|---|---|---|
| `start` | 23 400 (1.13 %) | 17.1 (ev_hi 12.2) | Earth sub-solar ≈ 0.14 | 9.1 × 10⁻³ | the Moon (4.2 px, scene exposure) ≈ 0.025; the old 1 % rule left this 0.13 % above its cliff |
| `earth-limb-dawn` | ≈ 26 000 (arc ≈ 26 px tall × ≈ 1 000 px) | set by the arc | ≤ 4.0 (the forward-scattering arc near the hidden Sun is capped by ev_hi) | — | the ground below the limb is night |
| `earth-map-closeup` | full frame | ≈ 15 (ground ρ 0.06–0.8 under a 34° Sun plus inscatter) | clouds ≈ 0.5 | ocean ≈ 0.09 | |
| `moon-full` | 224 000 (10.8 %) | 15.3 | 0.15 | 0.095 | |
| `mars-reference-angle` | 480 000 (23.2 %) | 14.6 | 0.13 | 0.012 | |
| `jupiter-galileans` | 2 440 (0.12 %) | 12.9 | 0.12 | 0.027 | the moons are points at star exposure (ruling 5; new decision N2) |
| `jupiter-disk` | 521 000 (25.1 %) | 12.9 | 0.12 | 0.028 | |
| `saturn-ring-plane` | 54 900 incl. the lit rings (2.6 %) | 11.1 | 0.13 | 0.028 | ring I/F ≈ 0.35 |
| `saturn-edge-on` | 533 000 (25.7 %) | 11.1 | 0.12 | 0.028 | rings edge-on to Sun and camera |
| `saturn-night-side` | 75 600, mostly the rings' unlit face (3.6 %) | 10.3 | 0.11 | 2.2 × 10⁻³ | unlit face ≈ 0.4 × 0.25 × HG(0.6, 30°) = 0.35 of E/π |
| `venus-crescent` | 2 280 crescent (0.11 %) | 16.7 | 0.24 | 6.1 × 10⁻³ | the haze ring is visible iff its radiance ≥ 1.4 × 10⁻³ × 1.2 × 2^16.7 = 183 cd/m² (B3's twin decides) |
| `titan-backlit` | ≥ 1 680 surface (0.08 %); the forward-scattering haze adds | 7.3 surface only; the haze raises it | ≤ 4.0 (ev_hi) | 5.6 × 10⁻³ | haze visible iff ≥ 0.27 cd/m² at EV 7.3, scaling as 2^EV |
| `sun-closeup` | 728 000 (disk radius 481.8 px) | 33.8 (mean disk 1.85 × 10⁹ cd/m² = 1.259 × 10⁵ lux / 6.794 × 10⁻⁵ sr, +log₂(1.853/1.6) = +0.21 EV over the round-1 figure; ev_hi 28.8) | centre (≈ 2.3 × 10⁹) → 2.3 × 10⁹/(1.2 × 2^33.8) ≈ 0.13 | limb at ≈ 0.35 of centre ≈ 0.046 | settles from EV −4 in 3 frames (B6) |
| `sun-granulation` | full frame | 33.8 | 0.13 ± 13 % granulation | — | |
| `comet-*` (20, 21) | coma and tail pixels ≥ 10⁻³ cd/m² | ≈ −2 ± 2, set by ev_hi on the coma core | ≤ 4.0 | a 10⁻³ cd/m² tail ≈ 4 × 10⁻³ at EV −2.3 | |
| 15–17, 19, 22, 23, 25, 26 | < 50 (no world light on screen) | `NIGHT_EV100` −4 | — | — | sky light only; 18 runs Manual (debug grid) |

Sky light at star EV −4 (every bookmark with eye-adapted stars; ruling 9): a V = 6.5 star peaks at 2.315 × 10⁻², 7.04 stops above AgX black; V = 0 at 9.2 (a white core); Sirius at 35.4. The band at 20 / 21.5 / 23 mag/arcsec² (1.08 × 10⁻³ / 2.7 × 10⁻⁴ / 6.8 × 10⁻⁵ cd/m²) is 1.4 × 10⁻² / 3.6 × 10⁻³ / 9.1 × 10⁻⁴ pre-exposed (6.4 / 4.4 / 2.4 stops above black). At fov 20° stars gain +3.42 EV against the band (frame contract).

Disk-fit bookmarks (4, 7, 9) run with `eye_adapted: false` (re-review r1 NP2): the eye-adapted band above (9.1 × 10⁻⁴ to 1.4 × 10⁻²) is brighter than `uni-verify disk`'s 4.49 × 10⁻⁴ threshold everywhere, so the limb would never be found, and raising the threshold to 0.004–0.014 cuts Jupiter's and Saturn's Lambert limbs (last pixel 6–8.5 × 10⁻³) up to 3 px inside. Strict, the sky follows the scene EV: band ≤ 1.44 × 10⁻² × 2^(−4 − EV) = 2.2 × 10⁻⁸ (Moon, EV 15.3), 1.2 × 10⁻⁷ (Jupiter, 12.9), 4.1 × 10⁻⁷ (Saturn, 11.1). A V = 6.5 star at fov f and scene EV peaks at 2.315 × 10⁻² × (tan 30°/tan(f/2))² × 2^(−4 − EV): 3.9 × 10⁻⁷ (Moon, fov 20°), 6.8 × 10⁻⁵ (Jupiter, fov 3.5°, ×357), 1.6 × 10⁻⁴ (Saturn, fov 4.3°, ×237), so only stars brighter than V −1.2 / 4.4 / 5.3 cross the threshold, as isolated points the fit ignores (`disk_ignores_unconnected_bright`). Mars (5) keeps eye-adapted stars and has no disk check.

---

## Open decisions for the lead

All seventeen original decisions are decided by the rulings below; the tasks follow them.

1. **Decided (ruling 1):** `TEXTURE_COMPRESSION_BC` and `DUAL_SOURCE_BLENDING` requested (A0a); both fallbacks kept (2k uncompressed maps, grey-transmittance atmosphere).
2. **Decided (ruling 2 + round 2):** `image` `jpeg`/`tiff` in `tools/bake` only; `intel_tex_2 = "=0.5.0"` for BC7/BC4 in the bake only, with A8's own BC1/BC4 encoder as the link-failure fallback; 2k uncompressed copies instead of a BC7 CPU decoder; the Earth close-up (bookmark 3) checks for block artefacts.
3. **Decided (ruling 3 + round 2):** `rayon = "=1.12.0"` in `tools/bake` only; runtime crates use `std::thread::scope` or one thread within CPU ≤ 4 ms.
4. **Decided (ruling 4):** the band after the opaque passes and the meter, at the far plane, honouring `lens_mask`; recorded in `targets.rs` and CLAUDE.md (A0a). See N1.
5. **Decided (ruling 5):** point-like planets, asteroids and the distant Sun at star exposure with the 1.5–4 px blend, under the "Eye-adapted stars" label. See N2.
6. **Decided (ruling 6):** amber `FREE FLIGHT · GRAVITY NOT SIMULATED` and `GRAVITY · <BODY> · NOT APPLIED` (C4).
7. **Decided (ruling 7):** Space = up, P = pause (C3).
8. **Decided (ruling 8):** ion tail aberrated ≈ 5° from anti-sunward (C1).
9. **Decided at the B9 gate (ruling 9):** the sweep at −3, −4, −5 with V = 6.5 ≥ ~7 stops above AgX black (B9 Step 6.6).
10. **Decided (ruling 10 + round 2):** measured, documented moon errors with ceilings in A2, < 1 px at every bookmark showing the moon (B9 intent tests); Phobos with its secular acceleration.
11. **Decided (ruling 11):** `H ≤ 14` unless the count exceeds ~60 k; belt 200 k + 50 k, lowered at C6 if over budget (the belt is built in Wave C).
12. **Decided (ruling 12):** tier B = Galilean moons, Pluto/Charon, Ceres (Titan removed from the download list to match).
13. **Decided (ruling 13):** Ω = 0°, i = 90° when unknown, dashed and labelled (C5).
14. **Decided (ruling 14):** the debug grid stays as an opt-in lens environment (bookmark 18).
15. **Decided (ruling 15):** one-line mono time at top centre while a card is open (C4).
16. **Decided (ruling 16):** grouped numbers; C4 updates `format_distance` and its tests.
17. **Decided (ruling 17):** labelled approximations accepted; every "verify" checked against its source before its task is done.

**New decisions this revision had to defer** (the plan proceeds with the stated default until the lead rules):

- **N1. Far-plane depth test for the band and stars (review m8).** Body edge pixels write depth (coverage ≥ 1/255), so with ruling 4's far-plane test the band and stars miss their (1 − coverage) share on a 1-px rim: an aliased silhouette against the band and stars vanishing one pixel early. `(1 − g)` already equals the uncovered fraction, so dropping the depth test for the band and stars fixes it exactly. Default: ruling 4 as written. Recommendation: drop the test and rely on `(1 − g)`.
- **N2. Points beside a metered body (ruling 5's consequence).** In `jupiter-galileans` Jupiter is metered at EV 12.9 while the moons (0.6–1.1 px, points) use star EV −4, so per pixel they show ≈ 17 stops brighter relative to Jupiter than they are (Io, V ≈ −5.5 from there, saturates with a wide glare). This is the labelled eye-adapted look ruling 5 approved. Alternative: points within a few degrees of a metered resolved body use the scene exposure. Default: ruling 5 as written; the bookmark's checks pass either way.
- **N3. Daylight turns eye-adapted stars off (review m12).** Applied by default in B6 (a daylit sky hides stars). Alternative: keep them on with a daylight note.
- **N4. FPS-gate thresholds.** The plan reads the performance ruling as: last 10 s of a 60 s sustained run, p95 GPU frame ≤ 16.0 ms, p95 CPU ≤ 4.0 ms, p95 frame interval ≤ 16.7 ms, and any pass above 1.25 × its budget gets its ladder applied. Confirm or change the 1.25 factor.

## Lead rulings (2026-11-22)

These rulings answer the open decisions above. Tasks follow them, not the "default if not approved" columns, except where a ruling says the default stands.

1. **Approved.** `TEXTURE_COMPRESSION_BC` and `DUAL_SOURCE_BLENDING` join the requested intersection (Global Constraints amended). The fallbacks in the dependency table stay implemented, because the app must still run without either feature.
2. **`image` `jpeg`/`tiff`: approved, in `tools/bake` only.**
   - **BC encoder:** `intel_tex_2`, for BC7 albedo, is approved only if the plan review verifies all of these:
     - the exact current version;
     - a permissive licence;
     - prebuilt `aarch64-apple-darwin` kernels;
     - no build-time ISPC or C toolchain.
   - Otherwise the default stands: our own BC1/BC4 encoder with the PSNR test.
   - Either way, the Wave B gate includes a close-up capture of the Earth map to check for block artefacts.
3. **`rayon`: approved for `tools/bake` only, at a version the plan review verifies exists.** Runtime crates use `std::thread::scope` or stay single-threaded within CPU ≤ 4 ms in `--bench`.
4. **Approved.** The band is drawn after the opaque passes and the exposure meter. It is a deliberate deviation from spec §5.3. A0 records it in the frame-contract doc comment and in CLAUDE.md, with the reason. The band draws only where there is no body (depth at the far plane), and it honours `lens_mask` exactly as the frame contract states.
5. **Approved.** Point-like planets, asteroids and the distant Sun use star exposure, with the 1.5–4 px blend. The "Eye-adapted stars" label covers them.
6. **Approved.** The HUD shows amber `FREE FLIGHT · GRAVITY NOT SIMULATED` and `GRAVITY · <BODY> · NOT APPLIED`. Honesty wins over the mockup.
7. **Approved.** Space = up (spec §5.6), P = pause.
8. **Approved.** The ion tail is aberrated about 5° from anti-sunward (Brandt 1968). It is physical and cited, and it corrects the spec's wording.
9. **Decided at the B9 gate** by the same rule as M0 Task 8c: V = 6.5 must sit at least about 7 stops above the real AgX black. The lead runs a sweep at −3, −4 and −5, looks at sky-orion and sky-GC, and picks.
10. **Accept A2's measured moon errors** if each is documented and is under 1 px at every m1 bookmark that shows that moon. Otherwise A2 escalates to the lead with the numbers.
11. **Asteroids:** `H ≤ 14` stands unless the count exceeds about 60 k, in which case A9 reports and the lead picks. **Belt:** 200 k main + 50 k Kuiper is the starting point. B9 lowers it if the belt pass breaks its per-pass budget.
12. **Tier-B maps approved:** Galilean moons, Pluto/Charon and Ceres. Only public-domain NASA/USGS mosaics, ≤ 60 MB each, ≤ 300 MB raw in total, deleted after baking. Uniform albedo with "No surface map" remains the fallback for any body without a map.
13. **Approved:** Ω = 0° and i = 90° when unknown, drawn dashed and labelled "orientation unknown".
14. **Approved:** the M0 debug grid stays as an opt-in lens environment for the shadow regression bookmark, with its honesty label.
15. **Approved:** a one-line mono time display at top centre while a card is open.
16. **Approved:** grouped numbers. Update `format_distance` and its tests in the owning task.
17. **Accepted as labelled approximations**, on one condition: every parameter marked "verify" must be checked against its cited source by the implementing agent before its task is done. No "verify" may remain in code comments at the M1 gate.

**Licences:** AT-HYG ShareAlike is accepted. ATTRIBUTION.md states that the derived star files are CC BY-SA 4.0.

**Round 2, after the plan review** (`.superpowers/sdd/2026-11-22-m1-solar-system/plan-review.md`):

- **Crates.** The review verified every condition, so these are approved for `tools/bake` only:
  - `intel_tex_2 = "=0.5.0"`. Its transitive `ispc_rt`, `libc` and `num_cpus` are accepted.
  - `rayon = "=1.12.0"`.
  - Neither is a dependency of any runtime crate.
  - A8's first step is to build a one-map BC7 bake on this Mac. If `intel_tex_2` fails to link, A8 falls back to its own BC1/BC4 encoder and reports.
  - No-BC fallback (I17): the bake also writes a 2k `Rgba8UnormSrgb` copy of each albedo map (`*.rgba2k.big`), so there is no BC7 CPU decoder.
- **Dual-source blending (I7).** `uni-gfx` shader validation gains a naga capabilities parameter. This is a lead-owned interface change in A0. Two shader variants exist, with and without dual-source blending.
- **HDR capture sidecar (I16).** Approved, with no new crate. `--capture --hdr` also writes `<name>.hdr.bin`: little-endian f32 RGBA of the pre-exposed HDR, plus a JSON header with the size and the exposure. `uni-verify` reads it for photometric checks. It is off by default to save disk. Every bookmark gets a numeric check, PNG-based or HDR-based.
- **Asteroid belt (I12).** Real photometry by default, so the belt is invisible from afar, as it really is. An opt-in "Highlight asteroid belt" toggle shows an amber label "Asteroid belt brightness exaggerated (non-physical)". Registry asteroids are excluded from the procedural belt. Precision claims are limited to what f32 camera-relative maths gives, with the bound stated.
- **Exoplanet hosts not in AT-HYG (I11).** They are added as extra stars from the archive's own RA, Dec, distance and V, flagged as from the archive.
- **Comet tails (I13).** Tails render for at most the 8 most prominent comets on screen. Under time warp, tail geometry is recomputed at most every 0.25 s of wall time.
- **Performance (I15).** Every pass gets a GPU budget at 1080p internal, summing to ≤ 13 ms and leaving headroom under 16.7 ms. There is a concrete fallback ladder, in order, for each expensive pass. `--bench` gains a sustained mode that renders for ≥ 60 s and reports the first and last 10 s separately, to expose thermal throttling. The M1 FPS gate uses the last-10 s numbers.
- **Auto-exposure (C3).** Planets must be correctly exposed when they fill only a small part of the frame. The meter design is up to the plan fix, but B9's visual checks must be passable on paper for every bookmark, with the arithmetic shown.
- **Moons (I10).** Ruling 10 stands: measured, documented errors under 1 px at the bookmarks. A2's fixed tolerances become the "measured and documented" form. Phobos either includes its secular acceleration (cite the source) or states its validity window.
- **Pluto (I1).** Use a cited source that still covers it, valid over the stated window, such as Meeus's periodic terms (ch. 37, 1885–2099). Pluto beyond that window is flagged as approximate on screen.

**Performance (riskiest part 1):** if `--bench` shows less than 60 FPS at 1080p internal in any Solar System bookmark at the Wave B gate, the lead dispatches perf-fix tasks before Wave C starts.

## Riskiest parts

1. **60 FPS on the fanless M3, sustained.** ~332 k star splats + up to 300 k asteroid points + four atmospheres + bloom; m0.1 measured the M0 star pass at 0.07–0.27 ms for 9 096 stars, which scales to 2.5–9.9 ms if linear, and the same lens view at 101–186 ms across one session as the M3 heated. Mitigation: the per-pass budget table (12.35 ms ≤ 13 ms), a written fallback ladder per expensive pass, the 60-s sustained bench with the gate on its last 10 s, perf-fix tasks before Wave C and before tagging.
2. **The compositing contract across six crates** (exposure split, `lens_mask` channels, depth-reading atmosphere, pass order, depth-skipping lens). Mitigation: the contract lives in one doc comment, every composite has an HDR check (ring division, ring and planet shadows, haze colours, band ratio, lens seam), B9 runs before Wave C.
3. **Ephemeris composition:** Meeus "of date" → J2000 precession, map `lon0`, WGCCRE rotation sign, Earth = EMB − Moon, Pluto = barycentre − Charon's share, barycentre vs body-centre Horizons targets. A sign slip moves terminators and features. Mitigation: Horizons goldens with barycentre targets and JD lists, the subsolar-point test, feature-location checks for `lon0`, the shading samples, points checks in captures.
4. **Auto-exposure on 26 different views.** A rule that looks reasonable can still blow out half the gate (review C3). Mitigation: the meter is designed against the bookmarks ("Exposure on paper"), every intent test re-derives its row, and Review Focus 6 pins the edge cases.
5. **Near-surface precision and the proxy-mesh/depth scheme** (Review Focus 2 and 5): covered by sagitta, f32-cancellation and f32-error tests, but only captures and the window run prove there is no flicker.
6. **Wave A scheduling:** A2/A3 wait on A1, A1/A2/A3 wait on A9's goldens, three Data agents share `uni-data` and `tools/bake` by file. Mitigation: A9 implements `golden.rs` and commits the two golden files first; A0b pre-splits every shared manifest and module list.
7. **Data semantics and licences:** AT-HYG Tycho photometry, NULL rows and RV flags; SSS map conventions and real resolutions; geometric-albedo calibration; CC BY-SA ShareAlike for derived star files; SBDB/Horizons API syntax; `intel_tex_2`'s packaged kernels linking on this Mac. Mitigation: Data reads each README/API doc first and records it in the manifest; A0b and A8 build the bake before committing to BC7.

## Review findings not applied

- **m8 (silhouette depth vs the band):** the fix conflicts with ruling 4's far-plane depth test; deferred as N1.
- **m6, partly:** points and asteroids now multiply by `(1 − r)`, but not by `(1 − g)`: they are depth-tested at their true distance, so `g` (which does not know depth) would wrongly dim a point in front of an atmosphere. A point behind an atmosphere is not dimmed in M1; the frame contract records this.
- **I5, partly:** c from f64 and the stable quadratic are in; the surface-anchored local origin below 10 km is not, because the remaining f32 hit-point error (0.4 m at Earth) is 10⁻⁴ of an 8k texel. M2's terrain needs it.
- **m18:** ruling 11's text ("B9 lowers it") is left as the lead wrote it; C6 applies the belt step, since the belt is built in Wave C.
- **m21, partly:** `precession_to_j2000` documents its validity and is used only inside the Moon window; Vondrák et al. 2011 long-term precession is not implemented in M1.
- **Superseded by the round-2 rulings** (the plan follows the ruling, not the review's alternative): I12's visibility options (opt-in highlight with an amber label), I13's "freeze grain ages above warp 10⁴" (tails for ≤ 8 comets, refreshed at most every 0.25 s of wall time), I15's "p95 ≤ 13 ms sustained" gate (budgets sum ≤ 13 ms; the FPS gate uses the last 10 s at 16 ms), I16's `.hdr.f16` sidecar (`.hdr.bin`, f32 RGBA plus a JSON header), I17's alternatives (BC7 with 2k uncompressed fallbacks).
- **Re-review r1, not changed in round 2:** minor 6 (disks of 4–7 px radius over-expose in the meter ramp; deliberate, the continuous fade from the star look, and no bookmark sits there) and C4's note that Callisto's true maximum separation is asin(a/d) = 22.12°, not 20.63° (still inside the 25.47° half-width). Re-review minor 1's 3.2 × 10⁻² belt glow was not adopted: the glow follows the annulus's solid angle (C2's visibility paragraph). Re-review r2 confirmed that convention; its lower total illuminance (H 10–16 only) is adopted as the low end, so the highlighted glow is now stated as 1.9–4.2 × 10⁻² and `asteroid-belt-above`'s expected ratio as ≈ 8–16 in means, ≈ 5 in the median the check reads (still ≥ 3).
- **C2's alternative placement** (Earth's night side) was not used; the sky bookmarks sit 100 AU from the Sun, which also keeps Earth's atmosphere out of the sky views.
