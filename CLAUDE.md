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

> **Updated by the lead after each wave.** Last update: the M0 final-review fix wave (after tag `m0`; report:
> `.superpowers/sdd/2026-11-20-m0-foundation/final-fix-report.md`).
> 272 unit tests pass (workspace lib + bins; 241 at tag `m0`). On the M3, `universe --selftest` passes 3/3
> (pcg3d bit-exact, AgX 1.5e-5, Binet 1.27e-4 rad), `--capture bookmarks/m0.ron` renders 6/6 bookmarks (each
> within 1 code value of its `m0` capture), and the shadow measures 158.745 px (GR: 158.745). Every GPU pass time
> is now a valid measurement. `--bench` at 1920×1080 measured: sky views 0.9–1.3 ms GPU per frame; `start`
> (lens scale 0.5) 38.7 ms GPU, 25.6 fps serial; `bh-shadow-30M` (lens scale 1.0) 157 ms. At the window's
> 1728×1080 internal size, `start` runs at 27 fps. The brute-force lens pass is ground truth only; M3 replaces it.
> The window itself was not run in this wave, so its FPS is still unmeasured; the HUD now shows it.
> Capture sheet after this wave (lead re-run at `caee1f3`, with the bench table): https://claude.ai/artifact/826GQeJr5fJL542uRbm9sx
> The older M0 sheet (https://claude.ai/artifact/Ld8A6nHFkfFfYHESX3CBxE) shows timings corrupted by a Metal query-resolve
> race (tonemap 0.00 ms, stars 1.1–4.7 ms); don't quote them.

What exists now:
```
Cargo.toml                   workspace root: members, [workspace.dependencies] pins (Global Constraints + serde_json)
.cargo/config.toml           shared target dir /Users/aayan/.cargo/shared-target/blackhole + dev/release/dist profiles
crates/
  uni-core/     FROZEN API, implemented: units, pos (UniversePos), epoch (TDB, leap seconds), frame (FrameTree),
                camera, hash (pcg3d, splitmix64), color (CIE blackbody, B−V→Teff, star lux), galactic
  uni-gfx/      FROZEN API, implemented: context (GpuContext, headless + windowed, GPU error log), targets (pass
                contract, HDR_MAX), frame (FrameCtx, CameraUniform 240 B, ExposureUniform 32 B, exposure_scale),
                timer (PassTimer: Option ms per pass + whole-frame ms), renderer (RenderFeature, Renderer,
                FrameBindings), shader (ShaderLib #import + naga validate), capture, proj (reversed-Z
                infinite); shaders/common/*.wgsl
  uni-post/     upscale (internal → output) + AgX tonemap + TPDF dither pass on pre-exposed HDR; CPU twin
                uni_post::agx
  uni-nav/      quaternion 6-DoF FlyCamera, log speed [10⁻¹, 10²⁶] m/s, winit input → Action, RON bookmarks
  uni-data/     BSC5 loader (binary format v1); data_dir()
  uni-sky/      StarField: energy-conserving erf-integrated PSF splats, Moffat glare, Planck LUT
  uni-rel/      binet CPU tracer (golden-tested); LensTracer compute pass (writes only its traced rectangle,
                lens_mask); DebugSkyBackground (the M0 debug grid, drawn only while the lens is traced); LensScale
  uni-ui/       egui HUD v0 (windowed + headless) and unit formatting
  uni-app/      binary `universe`: windowed app (internal ~1080 lines, upscaled to the window);
                `--capture <bookmarks.ron> --out <dir> [--size WxH] [--frames N] [--hud]`;
                `--bench <bookmarks.ron> [--size WxH] [--frames N]`; `--selftest`. World (TestBH-1: 10 M☉ at
                SSB + 1000 AU), scene wiring, F12 saves a view bookmark
  uni-planet/, uni-atmo/, uni-physics/, uni-procgen/, uni-sim/   doc comment only (later milestones)
tools/bake/     bin `uni-bake bsc5`: data/cache/bsc5/catalog.gz → data/baked/bsc5.bin (9096 stars)
tools/verify/   bin `uni-verify`: `sheet --captures <dir> --out <html>`, `contact --captures <dir> --out <png>
                [--width 2400]` (one labelled grid PNG), `shadow --png <f> --expected-px <r> --tol <t>`
bookmarks/m0.ron             the six M0 capture bookmarks (RON, implicit_some)
data/baked/bsc5.bin          Yale Bright Star Catalogue, 9096 stars (CDS V/50), committed
archive/legacy-2025.tar.xz   legacy Python, GLSL, run scripts, old docs and old CLAUDE.md (reference only)
archive/old-rust/            the old wgpu 0.19 renderer and shaders (every frame one flat colour; reference only)
textures/*.jpg               7 Solar System Scope 2k maps (CC BY 4.0, see ATTRIBUTION.md);
                             stars_milky_way.jpg is never the sky: no photo skyboxes (design §7.6)
data/manifest.toml           one entry per data file: id, url, sha256, license, credit, paths (BSC5 so far)
data/baked/                  small baked outputs, committed (bsc5.bin so far)
docs/superpowers/specs/      the design (binding)
docs/superpowers/plans/      milestone plans
docs/research/               research reports, golden values, wgpu30-skeleton/
.superpowers/sdd/            workflow ledger, task briefs and reports (git-ignored)
```
Target responsibilities per crate are in design §5.1. Every render crate implements `uni_gfx::RenderFeature`
(prepare + encode). The `uni-core` and `uni-gfx` interfaces are frozen. After M0 only the lead changes them.
Git-ignored outputs: `data/cache/` (raw downloads), `captures/` (capture PNG + JSON).

**Render contract (uni-gfx: the M1 frame contract, set by Task A0a; full text in the `crates/uni-gfx/src/targets.rs`
doc comment; every feature follows it)**
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
  write luminance × `star_pre()` (ruling 5). `star_pre == scene_pre` unless "Eye-adapted stars" is in effect
  (`eye_adapted && !daylight`, which the HUD label follows). A body 1.5–4 px in radius blends from `star_pre` to
  `scene_pre`. Each factor is `exposure_scale(EV) = 1/(1.2·2^EV)`. Clamp what you write to `HDR_MAX` (60000 <
  f16's 65504). The Sun's disk (~1.6e9 cd/m²) is ~4.1e4 at scene EV 15. The exposure buffer is also `STORAGE`,
  so the auto-exposure resolve can write it on the GPU; the meter reads world light only, before any sky light
  or point source is written. Environment cubemap texels hold absolute luminance in units of `ENV_UNIT_CD_M2`
  (1e-4 cd/m²), not pre-exposed.
- *Lens mask.* `lens_mask` is Rgba16Float (since A0a; m0.1 had R32Float), cleared to 0 every frame. `r`: the
  captured fraction of a traced pixel (1 = shadow). `g`: sky occlusion, 1 − transmittance toward the
  background, written with `BlendOperation::Max` on green only (bodies and comet nuclei their coverage, rings
  their opacity, atmospheres 1 − luminance(T)); the lens pass writes `g = 0` on pixels it repaints. `b`: 1 where
  the lens pass wrote the lensed environment. `a`: reserved for M3's star magnification. Sky features multiply
  what they add by (1 − r)(1 − g), the band also by (1 − b); point sprites and asteroids by (1 − r) only. Built
  at A0a: the lens pass writes (captured, 0, 1, 0) on its traced pixels and the star pass multiplies by (1 − r);
  nothing writes `g` yet.
- *Pass order.* M1 target: exposure resolve → band-update → background (M0 debug grid, only with the debug lens
  environment and a traced lens) → sun → bodies → rings → comets → atmosphere → histogram (meter) → lens → band
  → stars → points → asteroids → bloom → tonemap, then the HUD. Built at A0a: background → lens → stars →
  tonemap. The lens pass writes `hdr` and `lens_mask` only on the pixels it traces (its screen rectangle, or the
  whole frame from inside a horizon), and its escaped rays show the same environment as the sky drawn around
  them (M0: both draw the debug grid). Target (B7): it reads depth and leaves pixels with depth ≠ 0 (a body in
  front) untouched.
- *Band-order deviation (ruling 4).* The band is drawn after the opaque passes and the meter, not first as the
  design (§5.3) puts the background. The band is sky light at star exposure, so it must not be metered, and
  drawing it after the bodies with (1 − g) composites it correctly behind partly transparent atmospheres and
  rings. It draws only where there is no body (depth test `GreaterEqual` at depth 0) and honours `lens_mask` as
  above.
- *Internal vs output size.* `FrameCtx.viewport` is the internal render size, and `Renderer::render` keeps the
  targets at it. `FrameCtx.output_size` is the swapchain or capture size. The tonemap reads `hdr` 1:1 when the
  two are equal (captures, `--bench`) and bilinearly upscales otherwise (the window: about 1080 lines at the
  window's aspect, `scene::internal_size`). The lens rectangle and `pixel_dir` both use the internal size.
- *Timing.* `Renderer::gpu_ms() -> Vec<(&str, Option<f32>)>` and `gpu_frame_ms() -> Option<f32>`, up to 64 timed
  passes per frame. `None` means no valid measurement: show "n/a" (HUD, terminal, sheet) or write `null`
  (capture JSON), never 0.
- *GPU errors and features.* `GpuContext` logs uncaptured errors and device loss instead of panicking. The
  window shows the latest on the HUD; `--capture`, `--bench` and `--selftest` fail on them. The device is
  created with `adapter.features() ∩ {SHADER_F16, FLOAT32_FILTERABLE, TIMESTAMP_QUERY, TEXTURE_COMPRESSION_BC,
  DUAL_SOURCE_BLENDING}` (ruling 1; the M3 has all five). The app must run without any of the last three
  (`has_timestamps()`, `has_bc_compression()`, `has_dual_source_blending()`): no timestamps → "n/a"; no BC → the
  2k uncompressed map copies; no dual-source blending → the grey-transmittance atmosphere. WGSL using
  `@blend_src` validates with `validate_wgsl_with(src, caps)`; `ShaderLib::compose` hoists every `enable …;`
  line to the top, once each.
- *Probes and HDR read-back.* `uni_gfx::probe::run_compute` runs a compute probe (read-only storage input at
  binding 0, storage output at binding 1 of one group) and reads it back; each render crate's
  `gpu_probes(gpu, lib) -> Vec<ProbeResult>` runs under `universe --selftest` (so far uni-gfx's `cube_dir`).
  `uni_gfx::read_hdr` reads the pre-exposed `hdr` back as RGBA f32 (`hdr` has `COPY_SRC`), for the `--hdr`
  capture sidecar (target).

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
- (Added after M0, review I4.) Dual exposure: whenever star EV ≠ scene EV (the default: star EV −4, scene EV 0),
  the HUD shows the amber line "Eye-adapted stars (dual exposure, non-physical)" (design §2, §7.4).

**Standing commit rule** (renewed at tag `m0`; it overrides "Agents never commit" above, which is kept verbatim
from the M0 plan). When a workflow task tells an implementer to commit, it commits only its own paths with a
pathspec commit
(`git add -- <own paths> && git commit -m "..." -- <own paths>`), retrying on `index.lock`. Never
`git add -A`, `git commit -a`, `git stash` or `git reset`, and never push.

## 4. Build, disk and GPU rules (design §4, §10)

**Build**
- One shared target dir, `/Users/aayan/.cargo/shared-target/blackhole`, set in `.cargo/config.toml`. Never
  override it with `--target-dir` or `CARGO_TARGET_DIR`. Parallel agents wait on Cargo's lock; that is expected.
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
  measured), whole-frame GPU ms, CPU ms, star count, lens scale and epoch. M0 has no TAA or auto-exposure
  (target: once they exist, captures must render until they settle).
- Frame rate: `universe --bench <bookmarks.ron> [--size 1920x1080] [--frames 120]` prints mean and p95 frame ms
  (serial: encode + submit + GPU wait, an upper bound on the window's frame time), GPU frame ms and GPU ms per
  pass at that internal resolution. The window's HUD shows fps, frame ms and the render size.
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
- Every origin is made camera-relative in f64 on the CPU, then cast to f32. Target (not built yet): star
  batches use a per-sector f64→f32 origin plus f32 offsets (M0's BSC5 stars are unit directions at infinity).
- Frames nest Galaxy → system barycentre → body (`FrameTree` exists). Target (not built yet): the camera joins
  the nearest dominant frame (M0's camera lives in the root frame).
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
- Current plan: `docs/superpowers/plans/2026-11-20-m0-foundation.md`. Read only your task's section. Workflow
  briefs and reports live in `.superpowers/sdd/2026-11-20-m0-foundation/`.
- Research (equations, data sources, citations, measured numbers): `docs/research/2026-11-20-*.md` (physics,
  rendering, wgpu, assets, procgen, rust, python, docs, synthesis).
- wgpu 30 / winit 0.30 / egui 0.37 skeleton that compiled and ran here: `docs/research/wgpu30-skeleton/`.
- Legacy code and docs: `archive/legacy-2025.tar.xz` and commit `075272e`. History only: do not copy physics or
  claims from them (`docs/research/2026-11-20-docs.md` lists what was wrong).
- Licences and credits: `ATTRIBUTION.md` and `data/manifest.toml`.
