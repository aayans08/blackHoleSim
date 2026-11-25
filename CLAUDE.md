# CLAUDE.md: Universe Explorer (Rust + wgpu)

Loaded into every agent's context. It must describe the repository as it actually is; anything not built yet
is marked as a target. The lead updates it after each wave.

## 1. Goal (design §1)

One original program in which the owner can fly anywhere in the universe: from a planet's surface to its star
system, through interstellar space, past black holes, out to the Milky Way and on to other galaxies. Real sizes
and distances, real physics, no loading screens.
- **Look:** the quality of the owner's reference screenshot (`docs/reference/owner-reference.png`, local only and
  git-ignored because it comes from a commercial product). Smooth and detailed at every distance, no visible
  polygons or texels, cinematic but physically based lighting.
- **Physics:** as real as current knowledge allows: light bending, accretion disks, gravity, time dilation,
  Hawking radiation. Anything non-physical is labelled on screen.
- **Original engine:** no code, assets, UI or signature look from any existing product. Openly licensed
  scientific data and maps only, credited in `ATTRIBUTION.md`.
- **Machine:** Apple M3 MacBook Air (16 GB RAM, 8-core GPU, fanless, 2560×1664), macOS 14.6, Metal.

Anti-goals (design §2): claiming something works without seeing it; fake physics presented as real; the
pixelated look; copying a commercial product; docs that lie.

## 2. Current real layout

> **Updated by the lead after each wave.** Last update: the M1 gate, steps 1–5 (2026-11-25, HEAD `335b309`; the M1
> final fix wave is in progress; ledger `.superpowers/sdd/2026-11-22-m1-solar-system/progress.md`, gate report
> `task-M1-gate-report.md`). On the M3: `cargo test --workspace --lib --bins` 1 082 passed, 0 failed, 7 ignored;
> `universe --selftest` 55/55 PASS (every render crate's GPU probes plus `depth_writers_write_g`); `--capture
> bookmarks/m1.ron --hdr` renders 27/27 with no GPU error (the `--hud` set 27/27, `bookmarks/m0.ron` 6/6); the plan's
> numeric checks 42/42 PASS (row 27's lens seam ratio 1.043); the M0 shadow regression (bookmark `bh-shadow-30M-grid`)
> measures 158.745 px. Not yet run at the gate (target): step 6, the 60-s bench, and step 7, the window run. The last
> 60-s bench (`--bench bookmarks/m1-bench.ron --sustained 60`, C6, 2026-11-24, before C1L's comet changes, machine
> busy): 10/10 views within GPU p95 ≤ 16 ms, CPU p95 ≤ 4 ms, interval p95 ≤ 16.7 ms over the last 10 s (132–228 fps;
> 3 views passed only in a re-run). Per-pass budgets re-baselined by ruling: band 0.6 ms, histogram 0.6 ms (≈0.4 ms
> per pass boundary on this GPU). The app window runs the full M1 stack with the Navigator, HUD v2 and overlay (C6:
> `start` at 60 fps, CPU 1.9–2.4 ms, no GPU error). M1 capture sheet: target (after the gate). Wave B capture sheet:
> https://claude.ai/artifact/CxNGdYe4roydB4AvvuhTh7 ; engine sky before/after: https://claude.ai/artifact/58pKRE6QDzjqQLM9CrtkLN
> Older sheets (M0): https://claude.ai/artifact/826GQeJr5fJL542uRbm9sx (the earlier Ld8A… sheet has corrupted timings).

What exists now:
```
Cargo.toml                   workspace root: members, [workspace.dependencies] pins (Global Constraints + serde_json;
                             bake-only intel_tex_2, rayon)
.cargo/config.toml           dev/release/dist profiles only; builds go to Cargo's default, the git-ignored target/
crates/
  uni-core/     FROZEN API, implemented: units, pos (UniversePos), epoch (TDB, leap seconds), frame (FrameTree),
                camera, hash (pcg3d, splitmix64), color (CIE blackbody, B−V→Teff, star lux), galactic
  uni-gfx/      FROZEN API, implemented: context (GpuContext, headless + windowed, GPU error log), targets (pass
                contract, HDR_MAX), frame (FrameCtx, CameraUniform 336 B, ExposureUniform 32 B, LightUniform
                144 B, exposure_scale), timer (PassTimer: Option ms per pass, each pass's end stamp
                GpuTimings::pass_end_ms / PassTimer::last_end_ms, whole-frame ms; Renderer::gpu_timings()), renderer
                (RenderFeature, Renderer, FrameBindings), shared (SharedInput<T>), env (cube_dir twin), probe
                (compute probes for --selftest), shader (ShaderLib #import + naga validate), capture (read_hdr),
                proj (reversed-Z infinite); shaders/common/*.wgsl
  uni-post/     histogram auto-exposure (meter samples 1 px per 2×2 block, decodes lens_mask.a star share with
                pre_mix), dual star exposure, bloom (6 levels), upscale + AgX tonemap + TPDF dither; CPU twins
                [M1 B6, B9h]. The default star EV −6.5 is the app's (`uni_app::scene::DEFAULT_STAR_EV100`);
                `uni_sky::DEFAULT_STAR_EV100` (−4) is M0's, kept for the debug-grid bookmark
  uni-nav/      quaternion 6-DoF FlyCamera, log speed [10⁻¹, 10²⁶] m/s, winit input → Action, RON bookmarks, anchor
                (camera parented to a body frame, resolve_bookmark) [A6]; Navigator [C3]: aim ring on the destination
                nearest the view centre (fills, then flies), click pick (within 12 px of an object's projected
                centre at 1080 lines) + F to fly to the selection, animated goto (turn, then 4.5 s
                accelerate–cruise–decelerate to a stand-off; frameless targets followed by position + velocity),
                cruise (speed from the distance to the nearest surface, ≤ 0.05 pc per frame, labelled FTL),
                time-warp keys, Esc chain aim → goto → card → selection
  uni-data/     loaders: BSC5, AT-HYG 3-D stars + names, maps (BC7/BC4 .big + 2k fallbacks, index.ron),
                saturn ring profile, JPL satellite elements, SBDB comets and asteroids, exoplanet hosts, body
                facts, Horizons goldens; data_dir() [M1 A7–A9.1]
  uni-physics/  planets: Standish 1992 Table 1 (1800–2050) / Table 2 with 20-yr blend bands 1780–1800 and
                2050–2070, Meeus Pluto, SSB; moons: Meeus 47 Moon, E5 Galileans, Meeus 46 Saturn moons
                (approximate, labelled), JPL mean elements; WGCCRE 2015 rotation; Ephemeris; photometry
                (Mallama & Hilton 2018, phase correction C ≤ 10), small bodies (two-body + planet-perturbed
                RK4, osculate_at), Finson–Probstein dust, calculators [M1 A1–A3]
  uni-sim/      time warp, registry (17 821 objects incl. 8 973 H ≤ 10 asteroids, grows at runtime),
                cached f64 states (threaded refresh, 0.42 ms per tick release), snapshot (CPU mean 0.26–0.97 ms in
                C6's sustained bench, 1.95 ms on a busy machine), honesty flags, near-star index; comets start from
                data/baked/comets_2026.ron and catch up after date jumps under the "catching up" label [M1 A4, A4r,
                A4p]. Wave C [C0, C6]: active comets in the snapshot, destinations and marker candidates (the Sun,
                planets, the dominant planet's major moons, ≤ 3 other bright ones), World::details for cards,
                set_selected / set_goto_target, COMET_VISUAL_LAWS (NEOWISE calibrated to COBS V 2.1; other comets
                SBDB's law, noted on the card)
  uni-procgen/  galaxy model v0 (+ galaxy.wgsl), faint-star luminosity function [M1 A5]
  uni-sky/      StarField3 (AT-HYG 333 294 stars in 3-D sectors, eye-glare halo fitted to CIE glare, vivid colours
                labelled, Young 1967 twinkle under an atmosphere), points (reflected bodies at the blended exposure),
                band v0 + environment cubemap (procedural dust/star-cloud detail + 12 named clouds, labelled), M0
                BSC5 StarField fallback; band and catalogue stars have no depth test (lens_mask only) [M1 B4, B5,
                B4L, B9c]; asteroids [C2]: the 33 678 SBDB H ≤ 14 records solved on the GPU (f32 Kepler with a
                double-single mean anomaly, ≤ ~100 km in the main belt; uni-sim's registry asteroids are skipped
                here), a procedural belt from the catalogue's (a, e, i) histograms (app ladder 80 k main + 20 k
                Kuiper, Trojans at L4/L5, labelled procedural), opt-in belt highlight (amber, non-physical)
  uni-rel/      binet CPU tracer (golden-tested); LensTracer (environment = the band cubemap, or the M0 debug grid with
                LensEnv::DebugGrid; skips pixels with a body in front); DebugSkyBackground; LensScale [M1 B7]
  uni-planet/   analytic ellipsoid bodies with BC7 maps (proxy cap mesh, exact limb), star surfaces (limb darkening,
                granulation), Saturn's rings (Salo & French 2010 layer), eclipse/LOD twins [M1 B0–B2, B10, B11];
                comets [C1, C1L] (comet/{mod,nucleus,coma,tails}.rs, comet.wgsl): nucleus mesh in the LOD band like a
                body, coma (SBDB total minus the nucleus), Finson–Probstein dust tail (1 024 grains per tail, EWA
                splats, Biver et al. 2022 dust law where known), CO⁺ ion tail anchored on Aravind et al. 2025; ≤ 8
                tailed comets; grey note when a tail is partly drawn
  uni-atmo/     Hillaire 2020 atmospheres for Earth, Mars, Venus, Titan (dual-source and grey variants; half-res
                march, 16–24 steps), ground lighting, scintillation [M1 B3, B9p]
  uni-ui/       egui HUD v2 [C4, C4w]: collapsed by default (place line + "i", one time line, honesty line, perf
                line); expanded Location, Time/warp, Speed, Gravity ("not applied") panels and the target card (it
                uses the height down to the block below; its rows scroll only when they do not fit, below about
                1320 pt wide); honesty notes and renderer warnings pinned, one line each, never scrolled; Tab hides
                the whole HUD; Inter / IBM Plex Mono; window least size 960 × 740 pt. Overlay [C5]: markers,
                brackets, orbit lines only for the selection or goto target (exoplanet orbits dashed), kept off the
                HUD panels
  uni-app/      binary `universe [--start <name>]`: the window at a bookmarks/m1.ron view (default "start") with the
                full M1 stack, Navigator, HUD and overlay; an occluded window stops presenting and retries at 4 Hz
                [C6w]; `--capture <ron> --out <dir> [--size] [--frames] [--hud] [--hdr]` (capture JSON v2: settle
                then Manual, honesty and HUD blocks, pre-bloom HDR sidecar from the pre-bloom tap, per-pass
                durations from pass_end_ms, projected/regions/samples/flux); `--bench <ron> [--sustained s] [--json
                out]`; `--selftest` (55 checks: pcg3d, agx, binet, every render crate's GPU probes, depth writers
                write g). `--sky v1` is accepted and ignored
tools/bake/     bin `uni-bake`: bsc5, athyg, exo, map, rings, sbdb, sats, facts, horizons, comets-epoch
                (raw inputs in data/cache/, outputs in data/baked/)
tools/verify/   bin `uni-verify`: `sheet --captures <dir> --out <html> [--bench <json>]`, `contact --captures <dir>
                --out <png> [--width 2400]` (one labelled grid PNG), `shadow`, `disk` (limb ellipse fit), `points`
                (projected positions vs brightness centroids), `bench` (p95 gates on a bench JSON); HDR checks on
                the `--hdr` sidecar: `ratio`, `stats`, `samples`, `profile`, `angle`, `flux`, `limb` (also as
                `hdr <check>`)
bookmarks/m0.ron             the six M0 capture bookmarks (RON, implicit_some)
bookmarks/m1.ron, m1-bench.ron  the 27 M1 capture bookmarks (intent-tested) and the 10 bench views; sky-v1.ron
data/baked/bsc5.bin          Yale Bright Star Catalogue, 9096 stars (CDS V/50), committed
data/manifest.toml           one entry per data file: id, url, sha256, license, credit, paths (44 entries)
data/baked/                  small baked outputs, committed: bsc5.bin, athyg_m10.bin (329 768 stars, CC BY-SA),
                             athyg_names.ron, asteroids.bin (33 678) + names, comets.ron (512, SBDB) and
                             comets_2026.ron (derived), sat_elements.ron, body_facts.ron, exoplanets.ron,
                             saturn_rings.ron, maps/index.ron; maps/*.big are git-ignored (rebuild with uni-bake map)
data/fonts/                  Inter 4.1 and IBM Plex Mono 2.5.0 TTFs + their OFL texts, committed; the HUD embeds the
                             fonts with include_bytes! (uni-ui theme.rs)
data/golden/                 horizons_spots.ron, sky_spots.ron: JPL Horizons spot values, test goldens only
docs/superpowers/specs/      the design (binding)
docs/superpowers/plans/      milestone plans
docs/research/               research reports, golden values, wgpu30-skeleton/
.superpowers/sdd/            workflow ledger, task briefs and reports (git-ignored)
```
Target responsibilities per crate are in design §5.1. Every render crate implements `uni_gfx::RenderFeature`
(prepare + encode). The `uni-core` and `uni-gfx` interfaces are frozen. After M0 only the lead changes them.
Git-ignored outputs: `target/` (builds), `data/cache/` (raw downloads), `captures/` (capture PNG + JSON).
`archive/` (the legacy Python/GLSL tarball and the old wgpu 0.19 renderer) and the unused `textures/*.jpg` were
removed from the tree on 2026-11-25 (owner decision); history keeps them.

**Render contract (uni-gfx: the M1 frame contract, set by Task A0a; full text in the `crates/uni-gfx/src/targets.rs`
doc comment; every feature must follow it)**
- *Shared group 0.* `RenderFeature::encode(gpu, enc, frame: &FrameBindings, targets, output, timer)`. Bind
  `frame.camera` at group 0 (layout `Renderer::camera_bind_group_layout()`): binding 0 `CameraUniform` (336 B:
  view_rot, proj, inv_view_proj, prev_view_proj, viewport, output, time, sim, cam_delta), binding 1
  `ExposureUniform` (32 B), binding 2 `LightUniform` (144 B: the first `MAX_LIGHTS` = 4 finite
  `FrameCtx::lights`). The `Renderer` is the only CPU writer of all three; it keeps the previous frame's
  view_proj and camera `UniversePos` for `prev_view_proj` and `cam_delta` (f64 difference, then f32). `sim.x` is
  TDB seconds since J2000 mod 2^22 s (`sim_time_s()`, ≤ 0.25 s steps). Features keep no camera copies and put
  their own bindings in groups ≥ 1. WGSL: `common/camera.wgsl` (group 0, `pixel_dir`, `scene_pre()`,
  `star_pre()`, `sim_time_s()`); `common/exposure.wgsl` (`ExposureUniform`, `HDR_MAX`, `exposure_scale`, no
  bindings: a pass that writes the exposure buffer as storage imports only it, never group 0); `common/cube.wgsl`
  (`cube_dir`, the WebGPU cube-face convention, twin `uni_gfx::env::cube_dir`).
- *Shared inputs.* Per-frame app → feature data goes through `uni_gfx::SharedInput<T>`. The app sets each one
  before `Renderer::render`; features read them only in `prepare`.
- *Pre-exposed HDR.* `hdr` (Rgba16Float) holds light × exposure, and the tonemap applies AgX with no exposure
  of its own. World light (bodies, the Sun's disk, rings, atmospheres, comets) writes luminance [cd/m²] ×
  `scene_pre()`. Sky light and point sources (catalogue stars, the band, planets/Sun/asteroids drawn as points)
  write luminance × `star_pre()` (ruling 5). `star_pre == scene_pre` unless "Eye-adapted stars" is in effect:
  `ExposureSettings.eye_adapted && !ExposureSettings.daylight`, where the app sets `daylight` while the camera is
  inside an atmosphere shell with the Sun above its local horizon (a daytime sky hides the stars, review m12). The
  HUD's amber label follows exactly this condition, not `FrameCtx.exposure` (review m13). A resolved body whose
  projected radius is between `POINT_LOD_PX` (1.5 px) and `BLEND_LOD_PX` (4 px) blends its exposure factor from
  `star_pre` to `scene_pre` in log space, `pre_mix(star_pre, scene_pre, w)` = exp2(mix(log2 a, log2 b, w))
  (`common/exposure.wgsl`, CPU twin `uni_gfx::pre_mix`, B11; w from `uni_planet::lod_exposure_mix`, B0), so the
  point → disk handover never pops; the blend is labelled as part of "Eye-adapted stars". Each factor is `exposure_scale(EV) =
  1/(1.2·2^EV)`. Clamp what you write to `HDR_MAX` (60000 < f16's 65504). The Sun's disk (~1.6e9 cd/m²) is ~4.1e4
  at scene EV 15. The exposure buffer is also `STORAGE`, so the auto-exposure resolve can write it on the GPU;
  the meter reads world light only, before any sky light or point source is written. Environment cubemap texels
  hold absolute luminance in units of `ENV_UNIT_CD_M2` (1e-4 cd/m²), not pre-exposed.
- *Field-of-view dependence (deliberate, review m26).* Star brightness against the band depends on the field of
  view: stars are written as E/Ω_pixel, the band as a surface brightness, so at fov 20° a star's pixels are
  (tan 30°/tan 10°)² = 10.7× (+3.42 EV) brighter relative to the band than at 60°. That is what a camera does
  when it zooms (point sources gain contrast over extended light), so it stays; it is not a bug to fix.
- *Lens mask.* `lens_mask` is Rgba16Float (since A0a; m0.1 had R32Float), cleared to 0 every frame. `r`: the
  captured fraction of a traced pixel (1 = shadow). `g`: sky occlusion, 1 − transmittance toward the
  background, written with `BlendOperation::Max` (bodies and comet nuclei their coverage, rings
  their opacity, atmospheres 1 − luminance(T)); the lens pass writes `g = 0` on pixels it repaints. `b`: 1 where
  the lens pass wrote the lensed environment. `a` (since B10, lead ruling on B6 I1): the star share s = 1 − w of a
  world writer whose exposure is `pre_mix(star_pre, scene_pre, w)` (bodies, rings, star surfaces and atmospheres in the
  1.5–4 px LOD band; w snapped to 1 when 1 − w < 2⁻¹⁴), Max-composited, 0 elsewhere; the histogram meter divides by
  `pre_mix(scene_pre, star_pre, s)`. M3's star magnification will need a new channel. Sky features multiply what they
  add by (1 − r)(1 − g), the band also by (1 − b); point sprites and asteroids by (1 − r) only, and by (1 − g) too
  when the camera is inside an atmosphere shell (B4 fix). The lens pass writes (captured, 0, 1, 0) on its traced
  pixels.
- *Pass order* (push order in `uni-app/src/scene.rs`; full table with writes and depth in `targets.rs`). Built
  (M1), by owning crate: exposure resolve (uni-post; own bind group, never group 0) → band-update (uni-sky) →
  background (uni-rel: the M0 debug grid, only with `LensEnv::DebugGrid` and a traced lens; it is
  display-referred, so the app runs Manual exposure whenever `DebugGrid` is selected) → sun → bodies → rings →
  comets (uni-planet) → atmosphere (uni-atmo; dual-source variant: two pipelines, grey-transmittance variant:
  one) → histogram meter (uni-post) → lens (uni-rel) → band → stars → points → asteroids (uni-sky) → bloom
  (input clamped to `BLOOM_INPUT_MAX`) → tonemap (uni-post), then the HUD and overlays (uni-ui). uni-app adds
  its own diagnostic passes (`tap.rs`): the clear mark and the bench's frame tag first (B9t), and three capture
  taps, depth + lens mask after comets, world light after the meter, pre-bloom `hdr` before bloom (a tap encodes
  nothing unless a capture arms it). The lens pass writes `hdr` and `lens_mask` only on the pixels it traces
  (its screen rectangle, or the whole frame from inside a horizon), and its escaped rays show the same
  environment as the sky drawn around them (M0: both draw the debug grid). It reads depth and leaves pixels
  with depth ≠ 0 (a body in front) untouched (B7).
- *Band-order deviation (ruling 4).* The band is drawn after the opaque passes and the meter, not first as the
  design (§5.3) puts the background. The band is sky light at star exposure, so it must not be metered, and
  drawing it after the bodies with (1 − g) composites it correctly behind partly transparent atmospheres and
  rings. Since B9c (lead ruling) the band and the catalogue stars (3-D and the BSC5 fallback) have NO depth test
  and no depth attachment: their only occlusion is the `lens_mask` factors, so an anti-aliased body edge (g =
  coverage) keeps its (1 − coverage) share of sky. Every opaque writer must therefore write `g`; the selftest
  check `depth_writers_write_g` (C6) enforces it on the GPU. Reflected-body point sprites keep their true-depth
  `Greater` test.
- *Internal vs output size.* `FrameCtx.viewport` is the internal render size, and `Renderer::render` keeps the
  targets at it. `FrameCtx.output_size` is the swapchain or capture size. The tonemap reads `hdr` 1:1 when the
  two are equal (captures, `--bench`) and bilinearly upscales otherwise (the window: about 1080 lines at the
  window's aspect, `scene::internal_size`). The lens rectangle and `pixel_dir` both use the internal size.
- *Timing.* `Renderer::gpu_ms() -> Vec<(&str, Option<f32>)>`, `gpu_frame_ms() -> Option<f32>` and
  `gpu_timings()`, up to 64 timed passes per frame. Per-pass durations are differences of the passes' end stamps
  (`GpuTimings::pass_end_ms`, each end tick minus the frame bracket's begin; `scene::pass_durations`, B9t): Metal
  samples a render pass's begin tick early, so the raw spans do not difference cleanly. In `--sustained` runs
  (two frames in flight) Metal often has not written the last render pass's (the tonemap's) end tick when the
  frame resolves, so the tonemap reads "n/a" in most frames (C6: 0.1–25 % valid); serial runs are valid.
  `None` means no valid measurement: show "n/a" (HUD, terminal, sheet) or write `null` (capture JSON), never 0.
- *GPU errors and features.* `GpuContext` logs uncaptured errors and device loss instead of panicking. The
  window shows the latest on the HUD; `--capture`, `--bench` and `--selftest` fail on them. The device is
  created with `adapter.features() ∩ {SHADER_F16, FLOAT32_FILTERABLE, TIMESTAMP_QUERY, TEXTURE_COMPRESSION_BC,
  DUAL_SOURCE_BLENDING}` (ruling 1; the M3 has all five). The app must run without any of the last three
  (`has_timestamps()`, `has_bc_compression()`, `has_dual_source_blending()`): no timestamps → "n/a"; no BC → the
  2k uncompressed map copies; no dual-source blending → the grey-transmittance atmosphere. WGSL using
  `@blend_src` validates with `validate_wgsl_with(src, caps)`; `ShaderLib::compose` hoists every `enable …;`
  line to the top, once each.
- *Probes and HDR read-back.* `uni_gfx::probe::run_compute` runs a compute probe (read-only storage input at
  binding 0, storage output at binding 1 of one group) and reads it back. `universe --selftest` runs uni-app's
  pcg3d, agx and binet probes, every render crate's `gpu_probes(gpu, lib) -> Vec<ProbeResult>` (uni-gfx,
  uni-planet, uni-atmo, uni-sky, uni-post, uni-rel; a crate that returns no probe fails) and
  `depth_writers_write_g`: 55 checks. `uni_gfx::read_hdr` reads the pre-exposed `hdr` back as RGBA f32 (`hdr` has
  `COPY_SRC`); render probes use it. The `--hdr` capture sidecar comes from uni-app's pre-bloom tap instead
  (`tap.rs`, a copy of `hdr` just before bloom).

## 3. Global Constraints

Verbatim from the M0 plan. They bind every task.

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
- (Superseded in M1, ruling 1: the set also includes `TEXTURE_COMPRESSION_BC` and `DUAL_SOURCE_BLENDING`, and
  the app must run without either; see §2 "GPU errors and features".)

**Units and coordinates**
- CPU: SI units (m, s, kg) in f64.
- GPU: f32, camera-relative.
- Black-hole maths: units of M = GM/c², hole at the origin.
- World axes: **ICRS/J2000 equatorial, right-handed, origin at the Solar System barycentre (SSB)**, +x toward RA 0h Dec 0°, +z toward the north celestial pole.
- Camera space: looks along −Z, up is +Y.

**Depth:** reversed-Z, infinite far plane, `Depth32Float`, `CompareFunction::Greater`.
(Note added after M0: things drawn exactly on the far plane at depth 0, such as stars, use `GreaterEqual`.)

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
- (Added after M0, review I4.) Dual exposure: whenever star EV ≠ scene EV (the default since B4L: star EV −6.5,
  scene EV metered by auto-exposure; M0's was star EV −4, scene EV 0), the HUD shows the amber line "Eye-adapted
  stars (dual exposure, non-physical)" (design §2, §7.4).
  (Refined in M1, review m12/m13: the label follows `ExposureSettings.eye_adapted && !daylight`, as §2 states.)

**Standing commit rule** (renewed at tag `m0`; it overrides "Agents never commit" above, which is kept verbatim
from the M0 plan). When a workflow task tells an implementer to commit, it commits only its own paths with a
pathspec commit
(`git add -- <own paths> && git commit -m "..." -- <own paths>`), retrying on `index.lock`. Never
`git add -A`, `git commit -a`, `git stash` or `git reset`, and never push. Commit messages carry no
`Co-Authored-By` line (owner decision 2026-11-25).

## 4. Build, disk and GPU rules (design §4, §10)

**Build**
- One target dir: the repo's git-ignored `target/`, Cargo's default (`.cargo/config.toml` sets profiles only, so a
  clone builds on any machine; since 2026-11-25). Never override it with `--target-dir` or `CARGO_TARGET_DIR`.
  Parallel agents wait on Cargo's lock; that is expected.
- Profiles: dev has `debug = "line-tables-only"`, `incremental = false`, dependencies at `opt-level = 2` with
  `debug = false`. Release has `lto = "thin"`. The fat-LTO `dist` profile is built only at the very end.
- Run `cargo clean` for a profile before building another heavy one.
- No worktrees. Everyone edits this one checkout and touches only the files their task owns.
- After M0, only the lead edits the `uni-core` and `uni-gfx` interfaces. Others ask the lead.
- Lead's wave cycle: `cargo check --workspace` + tests → build the app → run the captures and look at them →
  send failures back to their owners → once green, commit the wave on `universe-explorer`, tag milestones (`m0`, …).

**Disk**
- Stop builds and downloads below 1.5 GB free (`df -h /`). Budget about 3 GB for builds plus data (not a ceiling).
- Raw downloads go to `data/cache/` (git-ignored) via `curl` and are deleted after baking. Only the Data team
  downloads (in M0, Task 7).
- Small baked outputs in `data/baked/` are committed. Large baked files are named `*.big` and git-ignored.
- Every data file gets an entry in `data/manifest.toml` and a credit in `ATTRIBUTION.md`.

**GPU (Metal via wgpu 30)**
- No f64 in shaders: the M3 has no FP64. Double-precision work stays on the CPU.
- GPU timestamps only at pass boundaries (`TIMESTAMP_QUERY_INSIDE_PASSES` is false on this M3).
- Measured on this M3 (M0 fix wave). Metal writes pass timestamps into the counter buffer late. A query resolve
  in the same command buffer read stale values, so `PassTimer` resolves only after the frame has finished.
  Metal also orders encoders only through shared tracked resources (wgpu-hal barriers are no-ops), so a pass
  that touches nothing can run alongside others. The frame brackets therefore write a texel of `hdr`.
- 32 storage textures per shader stage. BC and ASTC texture compression are available.
- `MTLCreateSystemDefaultDevice()` returns nil from a windowless shell, but `MTLCopyAllDevices()` sees the M3,
  and wgpu enumerates adapters the same way.
- Tests that need a GPU run through the app's `--selftest` mode, not through `cargo test`.
- App runs and captures go through one serialized queue (Verify or the lead): one GPU job at a time.
- Copy wgpu/winit/egui API usage from `docs/research/wgpu30-skeleton/`, which compiled and ran on this machine
  (adapter features and limits: `m3-probe-output.txt`), not from memory.

**Git:** branch `universe-explorer`; the legacy tree is snapshot commit `075272e`. Nothing is pushed to GitHub
without asking the owner.

## 5. Verification rule (design §8)

**Nothing is done without captures plus measured numbers.**
- "Done", "achieved", "works" and ✅ never appear without evidence: capture PNGs from fixed bookmarks, measured
  GPU and CPU ms, and passing test output.
- Headless capture (exists): `universe --capture <bookmarks.ron> --out <dir>` renders each bookmark `--frames`
  times (default 4) and saves the last as PNG. It writes JSON with GPU ms per pass (`null` = not validly
  measured), whole-frame GPU ms, CPU ms, star count, lens scale and epoch (v2 adds EVs, honesty, measurements).
  Auto-exposure bookmarks render until the scene and star EV each move < 0.05 EV per frame (at most 16 frames),
  then render in Manual at the settled EVs; the JSON's `settled` records whether they settled. No TAA yet (target).
- Frame rate: `universe --bench <bookmarks.ron> [--size 1920x1080] [--frames 120]` prints mean and p95 frame ms
  (serial: encode + submit + GPU wait, an upper bound on the window's frame time), GPU frame ms and GPU ms per
  pass at that internal resolution. `--sustained <s> [--json <out>]` runs s seconds per view with two frames in
  flight (windows all / first10s / last10s); `uni-verify bench` gates its JSON. The window's HUD shows fps, frame
  ms and the render size.
- Cheap visual check: `uni-verify contact --captures <dir> --out <png>` tiles every capture into one PNG.
- Each milestone ends with a capture sheet (an HTML or Markdown page of the PNGs plus numbers). An agent or
  report may say "done" only with a link to that sheet and the test output.
- The lead opens and looks at every capture PNG and checks the numbers before the next wave starts. The owner
  reviews the sheets without blocking; a rejection becomes fix tasks.
- Physics is tested against golden values: the full list is in `docs/research/2026-11-20-physics.md` §9, and
  M0's numbers come from `docs/research/m0-golden-values.py`.
- Every WGSL maths routine that matters has a CPU twin; `--selftest` checks that they agree on the GPU.
- Reports describe the code as it is. Say plainly what is approximate, untested or unmeasured.

## 6. Precision and physics conventions

**Precision (design §5.2, frozen in M0)**
- Absolute position: `UniversePos { cell: I64Vec3, local: DVec3 }`, cell size 2⁴⁴ m (≈117 AU): sub-millimetre
  locally, range about 10³² m.
- Every origin is made camera-relative in f64 on the CPU, then cast to f32. Star batches use a per-sector
  f64→f32 origin plus f32 offsets (built, B4: 16-pc sectors whose centres go to the GPU relative to an anchor
  near the camera, re-uploaded once the camera is more than 1 pc from it; M0's BSC5 fallback stars are unit
  directions at infinity).
- Frames nest Galaxy → system barycentre → body (`FrameTree`). The camera joins the nearest dominant frame
  (built: A6 anchor, the app's per-frame reanchor to `dominant_frame`, a goto's target frame).
- Time: `Epoch { day: i64, sec: f64 }`, TDB since J2000.
- Determinism: noise takes integer-lattice + fraction inputs; one integer hash in Rust and WGSL, tested to agree.
- No depth writes from fragment shaders (Apple GPUs are tile-based).

**Schwarzschild metric** (Schwarzschild 1916), r_s = 2GM/c². In black-hole code G = c = 1 and lengths are in
M = GM/c², so r_s = 2M:

    ds² = −(1 − r_s/r) c² dt² + (1 − r_s/r)⁻¹ dr² + r² (dθ² + sin²θ dφ²)

**Light paths: exact Binet form** (physics research §1.1). The orbit equation is u'' + u = 3M·u², with u = 1/r
and ' = d/dφ. Move the ray as a particle in flat Cartesian 3-space (Schwarzschild r, θ, φ read as Euclidean):

    d²x/dλ² = −(3/2)·r_s·h²·x/r⁵ = −3M·h²·x/r⁵,    h = |x × dx/dλ| (conserved)

- The acceleration points toward the hole. The legacy reference code had this sign wrong.
- λ is the affine parameter. With h = L, the radial equation r'' = h²/r³ − 3M·h²/r⁴ and the integral
  r'² = E² − (1 − 2M/r)·h²/r² match the Schwarzschild null geodesic exactly. Never renormalise dx/dλ: that
  changes h. Take redshifts from the conserved quantities (b = h/E).
- Static camera at r_c: b = r_c·sin δ / √(1 − 2M/r_c), δ = angle between the ray and the inward radial direction.
- Captured when r < 2M(1 + ε). Escaped when r > R_esc (about 1000M) and moving outward. RK4 or leapfrog, step ∝ r.

**Key radii (non-spinning hole)**

| Quantity | Value |
|---|---|
| Event horizon | r = 2M (= r_s) |
| Photon sphere | r = 3M |
| ISCO | r = 6M |
| Critical impact parameter (shadow edge) | b_c = 3√3 M ≈ 5.196 M |
| Shadow angular radius, static observer at r | sin α = (3√3 M/r)·√(1 − 2M/r) (Synge 1966) |
| Weak-field deflection | α ≈ 4M/b + (15π/4)(M/b)²; Sun limb 1.75″ |

**Accretion disk: Novikov–Thorne** (Novikov & Thorne 1973; flux from Page & Thorne 1974), a thin disk with zero
torque at the ISCO:
- F(r) = −(Ṁ/(4π√−g))·Ω_{,r}/(E − ΩL)²·∫_{r_isco}^{r} (E − ΩL)·L_{,r} dr, with √−g = r on the equator, and
  T_eff(r) = (F/σ_SB)^{1/4}. F(r_isco) = 0. Compute F once on the CPU as a 1D LUT.
- Newtonian check (Shakura & Sunyaev 1973): σT⁴ = (3GMṀ/(8πr³))·(1 − √(r_in/r)), peaking at r = (49/36)·r_in.
- Redshift g = (p·u_cam)/(p·u_em). Schwarzschild, observer at infinity: g = √(1 − 3M/r)/(1 − Ωλ), with
  Ω = √(M/r³) and λ = L_z/E of the photon travelling emitter → camera. Binet tracer: λ = b·(ĥ·ẑ), with
  ĥ = −normalize(x × v) of the BACKWARD-traced ray (camera → emitter) and ẑ the disk's angular-momentum axis.
  Getting this sign wrong puts the bright side on the receding half, which is exactly the old renderer's bug.
  For a static camera at r_c, divide by √(1 − 2M/r_c).
- Observed colour is Planck at T_obs = g·T; bolometric intensity ∝ g⁴ (I_ν/ν³ is invariant).
- Real disks are blue-white in the visible band, and the approaching side is the brighter one. A film-style
  look (no beaming, no colour shift) is allowed only as a labelled "Cinematic" mode.

**Hawking temperature** (Hawking 1975): T_H = ħc³/(8πGMk_B) ≈ 6.17×10⁻⁸ K × (M☉/M). For any astrophysical hole
it is far colder than the CMB: HUD numbers only, never a visible glow.

## 7. Where to read more

- Design (binding): `docs/superpowers/specs/2026-11-20-universe-explorer-design.md`. Read §2, §4, §5.1–5.3, §8
  and §10 before any task.
- Current plan: `docs/superpowers/plans/2026-11-22-m1-solar-system.md` (M0's: `2026-11-20-m0-foundation.md`).
  Read only your task's section. Workflow briefs and reports live in `.superpowers/sdd/2026-11-22-m1-solar-system/`.
- Research (equations, data sources, citations, measured numbers): `docs/research/2026-11-20-*.md` (physics,
  rendering, wgpu, assets, procgen, rust, python, docs, synthesis).
- wgpu 30 / winit 0.30 / egui 0.37 skeleton that compiled and ran here: `docs/research/wgpu30-skeleton/`.
- Legacy code and docs: snapshot commit `075272e`, and `archive/` in the commits before its removal (2026-11-25).
  History only: do not copy physics or claims from them (`docs/research/2026-11-20-docs.md` lists what was wrong).
- Licences and credits: `ATTRIBUTION.md` and `data/manifest.toml`.
