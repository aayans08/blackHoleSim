# M0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken single-file renderer with a new wgpu 30 Cargo workspace. You can fly it in 6-DoF with f64 precision through the real bright-star sky (point-spread-function splats, no texels), see a physically correct Schwarzschild black hole, and capture any bookmark headlessly to PNG with measured GPU times.

**Architecture:**
- A Cargo workspace of `uni-*` crates (design §5.1).
- Positions are `UniversePos` (i64 cell + f64 local), uploaded camera-relative in f32 (§5.2).
- A fixed pass list of `RenderFeature`s draws into an HDR target:
  1. lens tracer (compute)
  2. star splats
  3. AgX tonemap
  4. egui HUD
- Every WGSL maths routine that matters has a CPU twin, and `--selftest` checks the two agree on the GPU.

**Tech Stack:** Rust 1.100 (edition 2024), wgpu 30.0.2 (Metal), winit 0.30.13, egui 0.37.0, glam 0.34.3, naga 30.0.2, serde + ron, image 0.25 (PNG only).

**Spec:** `docs/superpowers/specs/2026-11-20-universe-explorer-design.md`
- Read §2 (anti-goals), §4 (constraints), §5.1–5.3 (architecture), §8 (verification) and §10 (process) before any task.
- The research behind each task is in `docs/research/2026-11-20-*.md`. A wgpu 30 skeleton that compiles and has run on this machine is at `docs/research/wgpu30-skeleton/`: copy API usage from it, not from memory.

## Global Constraints

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

## Review Focus

These failure modes are implied by the spec but no feature test covers them. Each has a pinned test in the owning task.

1. **The window is minimised or resized to 0×0, or moved between displays with different Retina scale factors.** Expected: no panic and no wgpu validation error; rendering resumes. *Test in Task 4:* `targets_clamp_zero_size`.
2. **The camera goes inside or exactly onto the event horizon, or is at the hole's centre.** Expected: the image is black, the HUD says "inside event horizon", and no NaN reaches the GPU. *Test in Task 9:* `static_observer_ray_inside_horizon_is_none`, `trace_from_inside_is_captured`.
3. **The baked data file is missing or corrupt.** Expected: the app starts with an empty sky and a HUD warning; `--capture` exits non-zero with a clear message. *Test in Task 7:* `load_missing_file_is_not_found`, `load_bad_magic_is_bad_format`.
4. **The GPU has no timestamp queries.** Expected: everything renders and the HUD shows "GPU ms n/a". *Test in Task 4:* `disabled_timer_yields_none`.
5. **Extreme distances and speeds** (1 Gpc from the origin, speed 10²⁵ m/s). Expected: positions and directions stay finite and mm-precise near the camera. *Tests:* `offset_precision_far_from_origin` in Task 3, `extreme_speed_stays_finite` in Task 6.

---

## Execution waves

| Wave | Tasks | Who | Runs in parallel? |
|---|---|---|---|
| 0 | 1, 2 | Lead | no (sequential) |
| 1 | 3, 4, 5, 6, 7, 8, 9, 10 (plus lead Task 14, the HUD mockup) | 8 implementer agents | yes; files are disjoint |
| 2 | 11, 12 | 2 agents | yes |
| 3 | 13 (gate) | Lead | no |

Each wave-1 and wave-2 task gets its own reviewer before the lead integrates it. The reviewer checks the task against this plan and the spec, plus code quality.

Tasks in wave 1 depend only on the **interface stubs from Task 2**. They must not change any signature in `uni-core` or `uni-gfx`. If a signature is wrong, report it to the lead instead of editing it.

---

### Task 1: Repository housekeeping (lead)

**Files**
- Create:
  - `archive/legacy-2025.tar.xz`
  - `CLAUDE.md` (full rewrite)
  - `ATTRIBUTION.md`
  - `data/manifest.toml`
  - `data/baked/.gitkeep`
- Delete:
  - every `*.py` at the repo root
  - `shaders/*.vert`, `shaders/*.frag`
  - `black_hole_shader.glsl`, `run.sh`, `RUN_PANDA3D.sh`, `requirements.txt`
  - the legacy docs: `README.md`, `README_REALISTIC.md`, `RENDERING.md`, `IMPLEMENTATION_REPORT.md`, `PANDA3D_GUIDE.md`, `BUILD_REPORT.md`, `QUICKSTART.md`, `QUICKSTART.txt`, `IMPROVEMENTS.md`, `RUST_MIGRATION.md`, `RUST_RENDERER.md`, `FINAL_SOLUTION.md`
  - `__pycache__/`, `target/`
  - the duplicate textures: `textures/planets/*`, `textures/stars/*`, `textures/space/*`
  - `textures/hdri/`, all `.DS_Store`
- Modify: `.gitignore`
- Do not touch yet: `src/main.rs`, `shaders/compute.wgsl`, `shaders/display.wgsl`. Task 2 moves them.

**Steps**
- [ ] **Step 1: Build the archive.**
  - Command: `tar -cJf archive/legacy-2025.tar.xz <files listed above> CLAUDE.md`. Include the old CLAUDE.md before rewriting it.
  - Verify: `tar -tJf archive/legacy-2025.tar.xz | wc -l` is at least 33, and the archive is under 400 KB.
- [ ] **Step 2: Delete the files listed above.** `cargo clean` removes `target/`.
- [ ] **Step 3: Append `data/baked/*.big` and `captures/` to `.gitignore`.** The first is the pattern for large baked files.
- [ ] **Step 4: Write `CLAUDE.md`, about 150–250 lines.** Sections:
  1. Goal (from design §1)
  2. Current real layout. Update it whenever the layout changes.
  3. The "Global Constraints" above, verbatim.
  4. Build, disk and GPU rules (design §10)
  5. The verification rule: "nothing is done without captures plus measured numbers" (design §8)
  6. Correct physics conventions:
     - the Schwarzschild metric
     - the Binet geodesic form d²x/dλ² = −(3/2)·r_s·h²·x/r⁵ = −3M·h²·x/r⁵
     - key radii: horizon 2M, photon sphere 3M, ISCO 6M, b_c = 3√3 M
     - the Novikov–Thorne temperature law
  7. Pointers to the design doc, this plan and `docs/research/`

  It must contain **no** claims about performance or features that are not yet measured.
- [ ] **Step 5: Write `ATTRIBUTION.md`** with sections for Data, Textures and Algorithms. For now it lists the 7 kept Solar System Scope 2k maps (CC BY 4.0, "Solar System Scope", solarsystemscope.com/textures).
- [ ] **Step 6: Write `data/manifest.toml`** as a list of entries, each with `id`, `url`, `sha256`, `license`, `credit`, `raw_path` and `baked_path`. Leave it empty; Task 7 adds the first entry.
- [ ] **Step 7: Verify.** `git status` shows only the intended deletions and additions. `du -sh .` is under 15 MB.
- [ ] **Step 8: Commit.** `git commit -m "chore: archive legacy Python/GLSL/docs, rewrite CLAUDE.md"`

---

### Task 2: Workspace scaffold and frozen interface stubs (lead)

**Files**
- Create:
  - `Cargo.toml` (workspace root; replaces the package manifest)
  - `.cargo/config.toml`
  - `crates/uni-{app,core,gfx,sky,planet,atmo,rel,post,physics,data,procgen,sim,nav,ui}/Cargo.toml` and `src/lib.rs` (`uni-app`: `src/main.rs`)
  - `tools/bake/` (bin crate `uni-bake`)
  - `tools/verify/` (bin crate `uni-verify`)
  - `crates/uni-gfx/shaders/common/{constants,hash,color,camera}.wgsl` (stubs)
- Move (`git mv`): `src/main.rs`, `shaders/compute.wgsl`, `shaders/display.wgsl` → `archive/old-rust/`. These are reference only and not compiled. Then remove the empty `src/` and `shaders/`.

**`.cargo/config.toml`**
```toml
[build]
target-dir = "/Users/aayan/.cargo/shared-target/blackhole"
[profile.dev]
debug = "line-tables-only"
incremental = false
[profile.dev.package."*"]
opt-level = 2
debug = false
[profile.release]
lto = "thin"
codegen-units = 16
incremental = false
[profile.dist]
inherits = "release"
lto = "fat"
codegen-units = 1
strip = true
```
Profiles go in the workspace `Cargo.toml` if Cargo refuses them in config.

**Interfaces.** These are the frozen M0 contract. Write them as real Rust with `todo!()` bodies so the workspace compiles. Wave 1 fills in the bodies.

`uni-core` (no wgpu dependency):
```rust
// units.rs: constants (CODATA 2018 / IAU 2012 B2 / IAU 2015 B3)
pub const C: f64 = 299_792_458.0;
pub const G: f64 = 6.674_30e-11;
pub const GM_SUN: f64 = 1.327_124_400_41e20;
pub const M_SUN: f64 = GM_SUN / G;
pub const AU: f64 = 1.495_978_707e11;
pub const PC: f64 = 3.085_677_581_491_367_3e16;
pub const LY: f64 = 9.460_730_472_580_8e15;
pub const DAY: f64 = 86_400.0;
pub fn schwarzschild_radius(mass_kg: f64) -> f64;   // 2GM/c²
pub fn gravitational_radius(mass_kg: f64) -> f64;   // GM/c²

// pos.rs
pub const CELL_SIZE_M: f64 = 17_592_186_044_416.0; // 2^44 m
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct UniversePos { pub cell: glam::I64Vec3, pub local: glam::DVec3 }
impl UniversePos {
    pub fn from_meters(m: glam::DVec3) -> Self;
    pub fn normalized(self) -> Self;               // each |local| component ≤ CELL_SIZE_M/2
    pub fn offset(self, d: glam::DVec3) -> Self;   // returns normalized
    pub fn delta_from(self, origin: UniversePos) -> glam::DVec3; // (self − origin) in metres
}

// epoch.rs: TDB seconds since J2000.0 = 2000-01-01T12:00:00 TT
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Epoch { pub day: i64, pub sec: f64 } // 0 ≤ sec < 86400
impl Epoch {
    pub const J2000: Epoch = Epoch { day: 0, sec: 0.0 };
    pub fn from_tdb_seconds(s: f64) -> Self;
    pub fn add_seconds(self, s: f64) -> Self;
    pub fn seconds_since(self, other: Epoch) -> f64;
    pub fn from_unix_utc(unix_s: f64) -> Self;          // IERS leap-second table; TDB≈TT
    pub fn to_unix_utc(self) -> f64;                    // inverse of from_unix_utc
    pub fn parse_iso_utc(s: &str) -> Option<Self>;      // "YYYY-MM-DDThh:mm:ss[.fff]Z"
    pub fn julian_centuries(self) -> f64;
}

// frame.rs
pub type FrameId = u32;
pub const ROOT_FRAME: FrameId = 0; // ICRS axes, origin SSB
pub type MotionFn = Box<dyn Fn(Epoch) -> (UniversePos, glam::DQuat) + Send + Sync>;
pub enum FrameMotion {
    Fixed { origin: UniversePos, rotation: glam::DQuat }, // relative to parent
    Custom(MotionFn),                                     // relative to parent; used by uni-physics later
}
pub struct FrameTree { /* private */ }
impl FrameTree {
    pub fn new() -> Self;
    pub fn add(&mut self, parent: FrameId, name: &str, motion: FrameMotion) -> FrameId;
    pub fn find(&self, name: &str) -> Option<FrameId>;
    pub fn origin_in_root(&self, id: FrameId, t: Epoch) -> UniversePos;
    pub fn rotation_to_root(&self, id: FrameId, t: Epoch) -> glam::DQuat;
    pub fn to_root(&self, id: FrameId, local_m: glam::DVec3, t: Epoch) -> UniversePos;
}

// camera.rs
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CameraView {
    pub pos: UniversePos,      // in root frame
    pub orient: glam::DQuat,   // camera→root rotation; camera looks −Z, up +Y
    pub fov_y_rad: f64,
}
#[derive(Clone, Debug)]
pub struct Lens { pub name: String, pub pos: UniversePos, pub mass_kg: f64, pub spin: f64 }

// hash.rs: Jarzynski & Olano 2020, "Hash Functions for GPU Rendering"; twin of shaders/common/hash.wgsl
pub fn pcg3d(v: glam::UVec3) -> glam::UVec3;
pub fn splitmix64(x: u64) -> u64;        // Steele et al. 2014; mixes x + 0x9E3779B97F4A7C15
pub fn hash_u64s(parts: &[u64]) -> u64;  // fold with splitmix64

// color.rs
pub fn blackbody_xyz(temp_k: f64) -> glam::DVec3;          // CIE 1931 2°, Wyman–Sloan–Shirley 2013 fit; Y = 1
pub fn xyz_to_linear_srgb(xyz: glam::DVec3) -> glam::DVec3; // IEC 61966-2-1 D65 matrix
pub fn blackbody_linear_srgb(temp_k: f64) -> glam::Vec3;    // Y = 1, negatives clamped to 0
pub fn bv_to_teff(bv: f64) -> f64;                          // Ballesteros 2012
pub fn star_illuminance_lux(v_mag: f64) -> f64;             // 10^(−0.4(m + 13.99))

// galactic.rs: Hipparcos ICRS→Galactic (ESA 1997, Vol. 1 §1.5.3); rows = galactic x, y, z axes in ICRS
pub const ICRS_TO_GALACTIC: [[f64; 3]; 3] = [
    [-0.054_875_560_416_215_4, -0.873_437_090_234_885_0, -0.483_835_015_548_713_2],
    [ 0.494_109_427_875_583_7, -0.444_829_629_960_011_2,  0.746_982_244_497_218_9],
    [-0.867_666_149_019_004_7, -0.198_076_373_431_201_5,  0.455_983_776_175_066_9],
];
pub fn icrs_to_galactic(v: glam::DVec3) -> glam::DVec3;
pub fn radec_to_icrs(ra_rad: f64, dec_rad: f64) -> glam::DVec3;
```

`uni-gfx`:
```rust
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const LENS_MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct SurfaceState { pub surface: wgpu::Surface<'static>, pub config: wgpu::SurfaceConfiguration }
// config.format = Bgra8Unorm, view_formats = [Bgra8UnormSrgb]
pub struct GpuContext {
    pub instance: wgpu::Instance, pub adapter: wgpu::Adapter,
    pub device: wgpu::Device, pub queue: wgpu::Queue,
    pub surface: Option<SurfaceState>,
}
impl GpuContext {
    pub async fn new_windowed(instance: wgpu::Instance, window: std::sync::Arc<winit::window::Window>) -> anyhow::Result<Self>;
    pub async fn new_headless() -> anyhow::Result<Self>;
    pub fn resize_surface(&mut self, w: u32, h: u32); // no-op if w or h is 0
    pub fn has_timestamps(&self) -> bool;
}

pub struct FrameTargets {
    pub size: glam::UVec2,                       // never 0 in either axis
    pub hdr: wgpu::Texture, pub hdr_view: wgpu::TextureView,             // STORAGE|TEXTURE|RENDER_ATTACHMENT
    pub depth: wgpu::Texture, pub depth_view: wgpu::TextureView,
    pub lens_mask: wgpu::Texture, pub lens_mask_view: wgpu::TextureView, // 1.0 = ray captured
}
impl FrameTargets { pub fn new(device: &wgpu::Device, size: glam::UVec2) -> Self; }

#[derive(Clone, Copy, Debug)]
pub struct Exposure { pub scene_ev100: f32, pub star_ev100: f32 }
#[derive(Clone, Debug)]
pub struct LensView { pub rel_pos_m: glam::DVec3, pub mass_kg: f64, pub spin: f64 } // lens − camera, root axes
pub struct FrameCtx {
    pub frame_index: u64, pub epoch: uni_core::Epoch, pub dt_real_s: f64,
    pub camera: uni_core::CameraView,
    pub view_rot: glam::Mat4,   // root→camera rotation only (camera at origin)
    pub proj: glam::Mat4,       // reversed-Z infinite (reversed_z_infinite_proj)
    pub viewport: glam::UVec2,
    pub exposure: Exposure,
    pub lenses: Vec<LensView>,
}

// camera.wgsl twin; group(0) binding(0) for every feature
#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_rot: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4], // inverse(proj * view_rot): pixel → root-axes direction
    pub viewport: [f32; 4],           // w, h, 1/w, 1/h
    pub exposure: [f32; 4],           // scene_ev100, star_ev100, 0, 0
    pub time: [f32; 4],               // frame_index as f32, dt_real_s, fov_y, aspect
}
impl CameraUniform { pub fn from_ctx(ctx: &FrameCtx) -> Self; }

pub struct PassTimer { /* private */ }
impl PassTimer {
    pub fn new(gpu: &GpuContext, max_passes: u32) -> Self;
    pub fn disabled() -> Self;
    pub fn compute_writes(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>>;
    pub fn render_writes(&mut self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>>;
    pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder);
    pub fn read_back(&mut self, gpu: &GpuContext);        // non-blocking; results lag by ≥1 frame
    pub fn last_ms(&self) -> Vec<(&'static str, f32)>;
}

pub trait RenderFeature {
    fn name(&self) -> &'static str;
    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets);
    fn encode(&mut self, gpu: &GpuContext, enc: &mut wgpu::CommandEncoder,
              targets: &FrameTargets, output: &wgpu::TextureView, timer: &mut PassTimer);
    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {}
}

pub struct Renderer { /* features in push order, targets, camera buffer + bind group, timer */ }
impl Renderer {
    pub fn new(gpu: &GpuContext, size: glam::UVec2) -> Self;
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout;
    pub fn camera_bind_group(&self) -> &wgpu::BindGroup;
    pub fn push(&mut self, f: Box<dyn RenderFeature>);
    pub fn targets(&self) -> &FrameTargets;
    pub fn resize(&mut self, gpu: &GpuContext, size: glam::UVec2); // clamps 0 → 1
    pub fn render(&mut self, gpu: &GpuContext, ctx: &FrameCtx, output: &wgpu::TextureView) -> wgpu::CommandBuffer;
    pub fn gpu_ms(&self) -> Vec<(&'static str, f32)>;
}

pub fn reversed_z_infinite_proj(fov_y: f32, aspect: f32, near: f32) -> glam::Mat4;

pub struct ShaderLib { /* path → source */ }
#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    #[error("missing import {0}")] MissingImport(String),
    #[error("naga: {0}")] Naga(String),
}
impl ShaderLib {
    pub fn with_common() -> Self;                       // registers "common/*.wgsl"
    pub fn add(&mut self, path: &'static str, src: &'static str);
    pub fn compose(&self, entry: &str) -> Result<String, ShaderError>; // expands `#import "path"` once each, recursively
    pub fn module(&self, device: &wgpu::Device, entry: &str) -> Result<wgpu::ShaderModule, ShaderError>;
}
pub fn validate_wgsl(src: &str) -> Result<naga::Module, ShaderError>;

pub struct CaptureTarget { pub texture: wgpu::Texture, pub view: wgpu::TextureView, pub size: glam::UVec2 }
impl CaptureTarget {
    pub fn new(device: &wgpu::Device, size: glam::UVec2) -> Self;    // CAPTURE_FORMAT, RENDER_ATTACHMENT|COPY_SRC
    pub fn read_rgba8(&self, gpu: &GpuContext) -> anyhow::Result<Vec<u8>>; // blocking; strips row padding
    pub fn save_png(&self, gpu: &GpuContext, path: &std::path::Path) -> anyhow::Result<()>;
}
```

**Steps**
- [ ] **Step 1: Write the workspace `Cargo.toml`.**
  - `[workspace]` members: the 14 crates plus the 2 tools.
  - `[workspace.dependencies]` with every pin from Global Constraints.
  - `[workspace.package] edition = "2024"`, `rust-version = "1.100"`.
- [ ] **Step 2: Write the crate manifests, `lib.rs` files and the stub interfaces above.**
  - Write them exactly as above.
  - Bodies are `todo!()`.
  - Crates not used in M0 (`uni-planet`, `uni-atmo`, `uni-physics`, `uni-procgen`, `uni-sim`) get only a `//!` doc comment naming their purpose and milestone.
- [ ] **Step 3: Verify.** `cargo check --workspace` exits 0 with no errors. Record the time and the size of the shared target dir (`du -sh ~/.cargo/shared-target/blackhole`).
- [ ] **Step 4: Commit.** `git commit -m "build: wgpu 30 workspace with frozen M0 interfaces"`

---

### Task 3: uni-core implementation

**Files**
- Modify: `crates/uni-core/src/{units,pos,epoch,frame,camera,hash,color,galactic}.rs`
- Test: unit tests inside each module (`#[cfg(test)]`)

**Interfaces**
- Produces: everything in the Task 2 `uni-core` block, implemented.
- Consumes: nothing.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn schwarzschild_radius_of_sun() { assert!((schwarzschild_radius(M_SUN) - 2953.25).abs() < 0.1); }
  #[test] fn from_meters_roundtrip_1pc() { let p = UniversePos::from_meters(DVec3::new(PC, -PC, 0.5)); assert!((p.delta_from(UniversePos::default()) - DVec3::new(PC, -PC, 0.5)).length() < 1e-3); }
  #[test] fn offset_precision_far_from_origin() { // Review Focus 5
      let far = UniversePos::from_meters(DVec3::new(1e9 * PC, 0.0, 0.0)); // 1 Gpc
      let near = far.offset(DVec3::new(0.001, 0.0, 0.0));
      assert!((near.delta_from(far).x - 0.001).abs() < 1e-6); }
  #[test] fn normalized_local_within_half_cell() { /* offset by 3.7 * CELL_SIZE_M on every axis; every |local| ≤ CELL_SIZE_M/2 */ }
  #[test] fn j2000_from_unix() { let e = Epoch::from_unix_utc(946_727_935.816); assert!(e.seconds_since(Epoch::J2000).abs() < 1e-3); }
  #[test] fn parse_2026_11_20() { let e = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap(); assert_eq!(e.day, 9819); assert!((e.sec - 43_269.184).abs() < 1e-3); }
  #[test] fn epoch_add_crosses_days() { let e = Epoch::J2000.add_seconds(-1.0); assert_eq!(e.day, -1); assert!((e.sec - 86_399.0).abs() < 1e-9); }
  #[test] fn frame_tree_fixed_child() { /* child at +1 AU x, rotated 90° about z; to_root(child, (1,0,0)) == (AU, 1, 0) within 1e-6 */ }
  #[test] fn pcg3d_golden() {
      assert_eq!(pcg3d(UVec3::new(0, 0, 0)), UVec3::new(0x9bafd7c6, 0xa8e88a6b, 0x3f15482c));
      assert_eq!(pcg3d(UVec3::new(1, 2, 3)), UVec3::new(0xfa9f79a6, 0x48f2f44c, 0x596f5ab1));
      assert_eq!(pcg3d(UVec3::new(0xFFFF_FFFF, 7, 42)), UVec3::new(0xe387d2d4, 0x47aa5f4a, 0xd51ae662)); }
  #[test] fn splitmix64_golden() { assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF); }
  #[test] fn blackbody_6504_is_white() { let c = blackbody_linear_srgb(6504.0); assert!((c.x / c.y - 1.0).abs() < 0.05 && (c.z / c.y - 1.0).abs() < 0.05); }
  #[test] fn blackbody_ordering() { let a = blackbody_linear_srgb(3000.0); assert!(a.x > a.y && a.y > a.z); let b = blackbody_linear_srgb(20000.0); assert!(b.z > b.x); }
  #[test] fn sun_teff_from_bv() { assert!((bv_to_teff(0.65) - 5778.4).abs() < 1.0); }
  #[test] fn star_lux_golden() { assert!((star_illuminance_lux(0.0) / 2.5351e-6 - 1.0).abs() < 1e-3); assert!((star_illuminance_lux(-26.74) / 1.2589e5 - 1.0).abs() < 1e-3); }
  #[test] fn galactic_center_direction() { /* galactic +x axis in ICRS has RA 266.4050° ± 1e-3, Dec −28.9362° ± 1e-3 */ }
  ```
  The golden values come from `docs/research/m0-golden-values.py`.
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-core --lib`. Expected: failures (`todo!()` panics).
- [ ] **Step 3: Implement.**
  - `normalized` moves whole cells out of `local` using `(local / CELL).round()` per axis.
  - `delta_from` computes `(cell − origin.cell) as f64 * CELL + (local − origin.local)`.
  - Leap-second table: the 28 IERS entries from 1972-01-01 (TAI−UTC = 10 s) through 2017-01-01 (37 s). TT = TAI + 32.184 s.
  - `blackbody_xyz` integrates Planck × the multi-lobe CMF fit from 380 to 780 nm in 1 nm steps, then normalises to Y = 1.
- [ ] **Step 4: Confirm they pass.** Run `cargo test -p uni-core --lib`. Expected: all pass.

---

### Task 4: uni-gfx implementation

**Files**
- Modify:
  - `crates/uni-gfx/src/{context,targets,frame,timer,renderer,shader,capture,proj}.rs`
  - `crates/uni-gfx/shaders/common/{constants,hash,color,camera}.wgsl`
- Test: unit tests in each module.

**Interfaces**
- Consumes: `uni_core::{Epoch, CameraView}`.
- Produces: the Task 2 `uni-gfx` block. WGSL-side names other crates rely on:
  - `camera.wgsl`:
    - `struct CameraUniform` (same layout as the Rust struct)
    - `@group(0) @binding(0) var<uniform> camera: CameraUniform;`
    - `fn pixel_dir(pix: vec2<f32>) -> vec3<f32>`, which returns the unit direction in root axes for a pixel centre
    - `fn exposure_scale(ev100: f32) -> f32 { return 1.0 / (1.2 * exp2(ev100)); }`
  - `hash.wgsl`: `fn pcg3d(v: vec3<u32>) -> vec3<u32>`, the twin of `uni_core::pcg3d`
  - `color.wgsl`: `fn luminance(c: vec3<f32>) -> f32` (Rec. 709 weights), `fn linear_to_srgb(c: vec3<f32>) -> vec3<f32>`
  - `constants.wgsl`: `const PI: f32`, `const TAU: f32`

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn every_common_shader_validates() { let lib = ShaderLib::with_common(); for p in ["common/constants.wgsl","common/hash.wgsl","common/color.wgsl","common/camera.wgsl"] { validate_wgsl(&lib.compose(p).unwrap()).unwrap(); } }
  #[test] fn compose_expands_import_once() { /* a.wgsl imports b and c; b imports c; composed text contains c's body exactly once */ }
  #[test] fn compose_missing_import_errors() { /* matches!(err, ShaderError::MissingImport(_)) */ }
  #[test] fn camera_uniform_layout_matches_wgsl() { /* naga Layouter on the composed camera.wgsl: size == size_of::<CameraUniform>() (=240) and each member offset == offset_of! */ }
  #[test] fn reversed_z_maps_near_to_one_far_to_zero() { let p = reversed_z_infinite_proj(1.0, 1.5, 0.1); /* (0,0,-0.1,1) → ndc z ≈ 1; (0,0,-1e12,1) → ndc z ∈ (0, 1e-9) */ }
  #[test] fn targets_clamp_zero_size() { /* pure size logic: FrameTargets::clamp_size(UVec2::ZERO) == UVec2::ONE (add this pub(crate) helper) */ } // Review Focus 1
  #[test] fn disabled_timer_yields_none() { let mut t = PassTimer::disabled(); assert!(t.compute_writes("x").is_none()); assert!(t.last_ms().is_empty()); } // Review Focus 4
  ```
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-gfx --lib`. Expected: failures.
- [ ] **Step 3: Implement, following `docs/research/wgpu30-skeleton/src/main.rs` for every wgpu call.**
  - `new_headless` uses `InstanceDescriptor::new_without_display_handle()`, Metal backend and `compatible_surface: None`.
  - The surface uses `Bgra8Unorm` with `view_formats [Bgra8UnormSrgb]`. Tonemapping writes through the sRGB view; egui uses the plain view.
  - Device features and limits as in Global Constraints.
  - `PassTimer` uses one `QuerySet` of `2 × max_passes` timestamps, resolved into a buffer, copied to a ring of 3 map-read buffers, converted with `queue.get_timestamp_period()`.
  - `Renderer::render` writes `CameraUniform::from_ctx(ctx)`, clears `hdr` to 0 and `lens_mask` to 0, calls each feature's `prepare` then `encode` in push order, resolves the timer and returns the finished buffer.
  - `CaptureTarget::read_rgba8` handles `COPY_BYTES_PER_ROW_ALIGNMENT` (256) padding.
- [ ] **Step 4: Confirm they pass.** Run `cargo test -p uni-gfx --lib`. Expected: all pass.

---

### Task 5: uni-post — exposure, AgX tonemap, dither

**Files**
- Create:
  - `crates/uni-post/src/{lib,agx,tonemap}.rs`
  - `crates/uni-post/shaders/tonemap.wgsl`
- Test: in `agx.rs` and `tonemap.rs`

**Interfaces**
- Consumes: `uni_gfx::{RenderFeature, GpuContext, FrameTargets, PassTimer, ShaderLib, Renderer::camera_bind_group_layout}`, plus `camera.wgsl`.
- Produces:
  - `pub fn agx(rgb_linear: glam::Vec3) -> glam::Vec3`: the CPU twin. Input is scene-linear Rec.709; output is **display-linear** (the sRGB view does the encoding).
  - `pub struct TonemapFeature`, which implements `RenderFeature` with `name() == "tonemap"`:
    - `pub fn new(gpu: &GpuContext, lib: &mut ShaderLib, camera_bgl: &wgpu::BindGroupLayout, output_format: wgpu::TextureFormat) -> anyhow::Result<Self>`
  - WGSL `fn agx(c: vec3<f32>) -> vec3<f32>` in `tonemap.wgsl` (registered as `"post/tonemap.wgsl"`). `--selftest` compares it with the CPU twin.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn agx_black_is_black() { assert!(agx(Vec3::ZERO).max_element() < 1e-3); }
  #[test] fn agx_monotonic_grey() { let mut last = -1.0; for i in 0..200 { let x = 1e-4 * 1.08f32.powi(i); let y = agx(Vec3::splat(x)).y; assert!(y >= last - 1e-6); last = y; } }
  #[test] fn agx_bounded() { assert!(agx(Vec3::splat(1e6)).max_element() <= 1.0 + 1e-4); }
  #[test] fn agx_mid_grey_reasonable() { let y = agx(Vec3::splat(0.18)).y; assert!(y > 0.10 && y < 0.30); } // display-linear
  #[test] fn tonemap_shader_validates() { /* compose "post/tonemap.wgsl" with common, validate_wgsl ok */ }
  ```
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-post --lib`.
- [ ] **Step 3: Implement.**
  - AgX follows Sobotka's AgX "base" with the default contrast look: inset matrix, log2 encoding over [−12.47393, 4.026069] EV, the 6th-order polynomial sigmoid fit, outset matrix, then decode to display-linear. Cite the source in comments.
  - Fullscreen-triangle pass: `hdr × exposure_scale(scene_ev100)` → `agx`, plus triangular-PDF interleaved-gradient-noise dither of ±0.5/255 in sRGB space (convert, add, convert back), → `output`.
- [ ] **Step 4: Confirm they pass.** Run `cargo test -p uni-post --lib`.

---

### Task 6: uni-nav — 6-DoF camera, input, bookmarks

**Files**
- Create: `crates/uni-nav/src/{lib,camera,input,bookmark}.rs`
- Test: in each module

**Interfaces**
- Consumes: `uni_core::{CameraView, UniversePos, C}`, plus winit events.
- Produces:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
  pub enum Action { Forward, Back, Left, Right, Up, Down, RollLeft, RollRight, SpeedUp, SpeedDown, ToggleHud, Screenshot, Help }
  pub struct InputState { /* pressed set, mouse delta, wheel */ }
  impl InputState {
      pub fn new() -> Self;
      pub fn on_window_event(&mut self, e: &winit::event::WindowEvent);  // keys, wheel, mouse buttons
      pub fn on_device_event(&mut self, e: &winit::event::DeviceEvent);  // raw mouse motion (only while look is held)
      pub fn is_down(&self, a: Action) -> bool;
      pub fn take_mouse_delta(&mut self) -> glam::DVec2;
      pub fn take_wheel(&mut self) -> f64;
      pub fn take_pressed_once(&mut self, a: Action) -> bool;
  }
  pub struct FlyCamera { pub view: uni_core::CameraView, pub speed_log10: f64, pub look_sensitivity: f64 }
  impl FlyCamera {
      pub fn new(view: uni_core::CameraView) -> Self;  // speed_log10 = 3.0 (1 km/s)
      pub fn speed_mps(&self) -> f64;                   // 10^speed_log10
      pub fn is_ftl(&self) -> bool;                     // speed_mps() > C
      pub fn update(&mut self, input: &mut InputState, dt_s: f64);
  }
  #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
  pub struct Bookmark { pub name: String, pub view: uni_core::CameraView, pub epoch_utc: Option<String>,
                        pub scene_ev100: Option<f32>, pub star_ev100: Option<f32>, pub lens_scale: Option<f32> }
  pub fn load_bookmarks(path: &std::path::Path) -> anyhow::Result<Vec<Bookmark>>;
  pub fn save_bookmarks(path: &std::path::Path, b: &[Bookmark]) -> anyhow::Result<()>;
  ```
- **Key map:**
  - W/S/A/D: forward/back/left/right
  - Space / Left Ctrl: up/down
  - Q/E: roll
  - Mouse wheel or `=`/`-`: speed by ±0.1 in log10 per notch
  - Right mouse button held: look (raw deltas)
  - Tab: toggle HUD; F12: screenshot; F1: help
- **Speed range:** `speed_log10` clamped to [−1, 26].
- **Rotations** apply in camera-local axes (yaw about local +Y, pitch about local +X, roll about local −Z). The quaternion is renormalised every update.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn forward_moves_along_minus_z() { /* identity orient, speed 1e3, Forward held, dt 1.0 → delta_from(start) ≈ (0,0,−1000) ±1e-6 */ }
  #[test] fn orientation_stays_unit() { /* 1e5 updates with mouse delta (3,−2) and roll: |orient.length() − 1| < 1e-9 */ }
  #[test] fn speed_clamped() { /* 400 SpeedUp presses → speed_log10 == 26.0; 400 SpeedDown → −1.0 */ }
  #[test] fn extreme_speed_stays_finite() { /* speed_log10 = 26, Forward, dt 0.016, 1000 updates → pos.local and pos.cell finite and no overflow */ } // Review Focus 5
  #[test] fn bookmark_ron_roundtrip() { /* save 2 bookmarks to a temp dir, load, assert_eq! on name and view */ }
  #[test] fn ftl_flag() { /* speed_log10 = 9 → is_ftl() true; 8 → false */ }
  ```
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-nav --lib`.
- [ ] **Step 3: Implement.** Movement integrates `view.pos = view.pos.offset(orient * dir_local * speed * dt)`.
- [ ] **Step 4: Confirm they pass.** Run `cargo test -p uni-nav --lib`.

---

### Task 7: uni-data and `uni-bake` — Yale Bright Star Catalogue (BSC5)

**Files**
- Create:
  - `crates/uni-data/src/{lib,bsc5}.rs`
  - `tools/bake/src/{main,bsc5}.rs`
  - `data/baked/bsc5.bin` (committed, about 220 KB)
- Modify: `data/manifest.toml`, `ATTRIBUTION.md`
- Download to: `data/cache/bsc5/` (git-ignored)

**Interfaces**
- Consumes: `uni_core::{radec_to_icrs, bv_to_teff}`.
- Produces (in `uni-data`):
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub struct CatalogStar { pub dir_icrs: glam::Vec3, pub v_mag: f32, pub teff_k: f32, pub hr: u32 }
  #[derive(Debug, thiserror::Error)]
  pub enum DataError {
      #[error("not found: {0}")] NotFound(std::path::PathBuf),
      #[error("bad format: {0}")] BadFormat(String),
      #[error(transparent)] Io(#[from] std::io::Error),
  }
  pub fn data_dir() -> std::path::PathBuf;   // $UNI_DATA_DIR, else the workspace `data/` (via CARGO_MANIFEST_DIR at compile time)
  pub fn load_bsc5(path: &std::path::Path) -> Result<Vec<CatalogStar>, DataError>;
  pub fn write_bsc5(path: &std::path::Path, stars: &[CatalogStar]) -> Result<(), DataError>;
  ```
- **Binary format v1** (little-endian):
  - magic `b"UNIBSC5\0"`, then `u32` version = 1, then `u32` count
  - then `count` records of 24 B: `f32×3` direction, `f32` V mag, `f32` Teff, `u32` HR
- **Bake CLI:** `uni-bake bsc5 --input data/cache/bsc5/catalog.gz --out data/baked/bsc5.bin`

**Steps**
- [ ] **Step 1: Download.**
  - `curl -L --fail -o data/cache/bsc5/catalog.gz https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz`
  - `curl -L --fail -o data/cache/bsc5/ReadMe https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe`
  - Read the ReadMe's byte-by-byte description and use **its** column positions: HR, RAh/RAm/RAs, DE-/DEd/DEm/DEs (J2000), Vmag, B-V.
  - Record the sha256 in `data/manifest.toml`: `id = "bsc5"`, licence "public (CDS V/50)", credit "Hoffleit & Warren 1991, Yale Bright Star Catalogue 5th ed. via CDS".
- [ ] **Step 2: Write the failing tests.**
  ```rust
  #[test] fn parse_sirius_line() { /* the HR 2491 line from the real file, embedded as a const: dir ≈ radec(101.2872°, −16.7161°) within 0.01°; v_mag == −1.46; teff from B−V 0.00 */ }
  #[test] fn record_is_24_bytes() { /* write 3 stars to a temp file; file length == 16 + 3*24 */ }
  #[test] fn roundtrip() { /* write_bsc5 then load_bsc5 → equal */ }
  #[test] fn load_missing_file_is_not_found() { assert!(matches!(load_bsc5(Path::new("/nonexistent/x.bin")), Err(DataError::NotFound(_)))); } // Review Focus 3
  #[test] fn load_bad_magic_is_bad_format() { /* 32 bytes of zeros → Err(BadFormat) */ }
  ```
- [ ] **Step 3: Confirm they fail.** Run `cargo test -p uni-data --lib && cargo test -p uni-bake --bins`.
- [ ] **Step 4: Implement.**
  - Gzip via flate2.
  - Skip records with blank coordinates; there are about 14.
  - A missing B−V gets Teff 5800 K and a log warning.
- [ ] **Step 5: Confirm they pass.** Same command as Step 3.
- [ ] **Step 6: Bake and check.**
  - Run `cargo run -p uni-bake -- bsc5 --input data/cache/bsc5/catalog.gz --out data/baked/bsc5.bin`.
  - It must print a count between 9090 and 9110.
  - The file size must equal 16 + 24 × count.
  - Add an `ATTRIBUTION.md` entry.

---

### Task 8: uni-sky — star point-spread-function splats

**Files**
- Create:
  - `crates/uni-sky/src/{lib,stars,psf}.rs`
  - `crates/uni-sky/shaders/stars.wgsl`
- Test: in `psf.rs`, `stars.rs`

**Interfaces**
- Consumes:
  - `uni_core::{blackbody_linear_srgb, star_illuminance_lux}`
  - `uni_data::CatalogStar`
  - `uni_gfx::{RenderFeature, ShaderLib, HDR_FORMAT, FrameTargets::lens_mask_view}`
  - the camera bind group
- Produces:
  ```rust
  #[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
  pub struct StarInstance { pub dir: [f32; 4] /* xyz dir (root axes), w = illuminance lux */, pub color: [f32; 4] /* linear sRGB, Y = 1; w unused */ }
  pub fn star_instance(s: &uni_data::CatalogStar) -> StarInstance;
  pub fn psf_weights(sub_px: glam::Vec2, sigma_px: f32, radius_px: i32) -> Vec<f32>; // CPU twin of the shader's per-pixel weights
  pub struct StarField { /* instance buffer, pipeline */ }
  impl StarField {
      pub fn new(gpu: &GpuContext, lib: &mut ShaderLib, camera_bgl: &wgpu::BindGroupLayout, stars: &[uni_data::CatalogStar]) -> anyhow::Result<Self>;
      pub fn count(&self) -> u32;
  }
  impl RenderFeature for StarField { /* name() == "stars" */ }
  ```

**Rendering rules (the "not pixelated" requirement)**
- Each star is an instanced, screen-aligned quad centred on its exact sub-pixel projected position, with no snapping.
- The fragment shader evaluates an energy-normalised Gaussian core (σ = 0.65 px) analytically per pixel, using the pixel's area integral (erf differences). That way a sum over the footprint equals the star's total energy wherever its centre lies.
- A wide, faint glare halo (Moffat-like, β = 2.5, at about 2% of the energy) is added only for stars with illuminance above 1e-6 lux (about V < 1.0).
- The quad radius grows with brightness, with a minimum of 3 px.
- Blending is additive into `hdr`. Pixel luminance = E / Ω_pixel × exp2(scene_ev100 − star_ev100) × (1 − lens_mask).
- Stars behind the camera, and stars with a zero-area projection, are culled in the vertex shader.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn psf_conserves_energy_any_subpixel() { for (x, y) in [(0.0, 0.0), (0.5, 0.5), (0.13, 0.91), (0.99, 0.01)] { let s: f32 = psf_weights(Vec2::new(x, y), 0.65, 3).iter().sum(); assert!((s - 1.0).abs() < 2e-3); } }
  #[test] fn psf_peak_moves_with_subpixel() { /* weight of the centre pixel at offset 0.5 > weight at offset 0.9 */ }
  #[test] fn sirius_instance() { /* CatalogStar{v_mag: −1.46, teff_k: 9940} → dir.w ≈ star_illuminance_lux(−1.46); color.z > color.x */ }
  #[test] fn stars_shader_validates() {}
  #[test] fn star_instance_size_32() { assert_eq!(std::mem::size_of::<StarInstance>(), 32); }
  ```
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-sky --lib`.
- [ ] **Step 3: Implement** by following the rules above.
- [ ] **Step 4: Confirm they pass.** Run `cargo test -p uni-sky --lib`.

---

### Task 9: uni-rel — Schwarzschild ground-truth tracer

**Files**
- Create:
  - `crates/uni-rel/src/{lib,binet,tracer}.rs`
  - `crates/uni-rel/shaders/{binet,debug_sky,lens_trace}.wgsl`
- Test: in `binet.rs`, `tracer.rs`

**Interfaces**
- Consumes:
  - `uni_core::{gravitational_radius, Lens}`
  - `uni_gfx::{RenderFeature, LensView, FrameTargets, ShaderLib}`
  - the camera bind group
- Produces:
  ```rust
  pub mod binet { // Units: M = 1, hole at origin. Binet/Cartesian form: a = −3·h²·x/r⁵, h = |x × v| (exact for Schwarzschild null geodesics)
      #[derive(Clone, Copy, Debug, PartialEq)]
      pub enum RayEnd { Captured, Escaped { dir: glam::DVec3 }, MaxSteps }
      pub fn trace(x0: glam::DVec3, v0: glam::DVec3, r_far: f64, max_steps: u32) -> RayEnd; // RK4, dλ = clamp(0.02·r, 1e-3, 5.0)
      pub fn deflection_for_impact(b: f64) -> Option<f64>;  // ray from x = −1e4 with impact b; None if captured
      pub fn static_observer_ray(cam: glam::DVec3, local_dir: glam::DVec3) -> Option<glam::DVec3>; // None if r ≤ 2; tangential part ÷ √(1 − 2/r)
      pub fn shadow_angular_radius(r_cam: f64) -> f64;     // asin(3√3/r · √(1 − 2/r)) (Synge 1966)
  }
  pub struct LensTracer { /* pipeline, params buffer */ }
  impl LensTracer {
      pub fn new(gpu: &GpuContext, lib: &mut ShaderLib, camera_bgl: &wgpu::BindGroupLayout) -> anyhow::Result<Self>;
      pub fn set_resolution_scale(&mut self, s: f32); // 0.25..=1.0; capture default 1.0, interactive default 0.5
  }
  impl RenderFeature for LensTracer { /* name() == "lens" */ }
  ```

**GPU rules**
- For each `LensView` in `ctx.lenses` (only the first in M0), the tracer computes a screen rectangle: the projected circle of angular radius θ_max, where the weak-field deflection 2r_s/b drops to half a pixel. It is clamped to the screen.
- For each pixel in the rectangle (traced at the resolution scale, then upsampled bilinearly):
  1. Build the camera-relative ray in M units from `pixel_dir`.
  2. Apply `static_observer_ray`.
  3. Integrate `binet.wgsl` (the same RK4 and step rule; max 2000 steps).
  4. If captured: write black to `hdr` and 1.0 to `lens_mask`.
  5. If escaped: write `debug_sky(exit_dir)` to `hdr`.
- `debug_sky` draws a celestial grid every 15° in ICRS, with a base luminance of 0.05 × white and lines at 0.4 × white, both at scene EV. **It is never pure black**, so the shadow is the only black region.
- The HUD labels it "Lens background: M0 debug grid".

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn deflection_b10m() { assert!((binet::deflection_for_impact(10.0).unwrap() - 0.59040).abs() < 2e-3); }   // b = 5 r_s
  #[test] fn deflection_b20m() { assert!((binet::deflection_for_impact(20.0).unwrap() - 0.23614).abs() < 2e-3); }   // b = 10 r_s
  #[test] fn deflection_b100m() { assert!((binet::deflection_for_impact(100.0).unwrap() - 0.04122).abs() < 1e-3); } // b = 50 r_s
  #[test] fn critical_impact() { assert!(binet::deflection_for_impact(5.19).is_none()); assert!(binet::deflection_for_impact(5.21).is_some()); } // b_c = 3√3 = 5.196152
  #[test] fn shadow_at_30m() { assert!((binet::shadow_angular_radius(30.0).to_degrees() - 9.6327).abs() < 1e-3); }
  #[test] fn static_observer_ray_inside_horizon_is_none() { assert!(binet::static_observer_ray(DVec3::new(1.5, 0.0, 0.0), DVec3::X).is_none()); } // Review Focus 2
  #[test] fn trace_from_inside_is_captured() { assert_eq!(binet::trace(DVec3::new(1.0, 0.0, 0.0), DVec3::Y, 1e3, 100), binet::RayEnd::Captured); } // Review Focus 2
  #[test] fn lens_shaders_validate() {}
  ```
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-rel --lib`.
- [ ] **Step 3: Implement.**
  - The CPU `trace` and the WGSL integrator must use the same step rule.
  - Write the GPU parameters as a uniform struct of vec4s only, with a layout test.
- [ ] **Step 4: Confirm they pass.** Run `cargo test -p uni-rel --lib`.

---

### Task 10: uni-ui — HUD v0 and unit formatting

**Files**
- Create: `crates/uni-ui/src/{lib,format,hud}.rs`
- Test: in `format.rs`

**Interfaces**
- Consumes: `uni_core::{Epoch, AU, PC, LY, C}`, egui 0.37 (copy the integration from `docs/research/wgpu30-skeleton/src/ui.rs`).
- Produces:
  ```rust
  pub fn format_distance(m: f64) -> String;
  pub fn format_speed(mps: f64) -> String;
  pub fn format_epoch(e: uni_core::Epoch) -> String;   // "2026-11-20 00:00:00 UTC"
  pub struct HudData {
      pub fps: f32, pub cpu_ms: f32, pub gpu_ms: Vec<(&'static str, f32)>, pub gpu_timing_available: bool,
      pub dist_from_sun_m: f64, pub nearest_name: String, pub nearest_dist_m: f64,
      pub speed_mps: f64, pub epoch: uni_core::Epoch, pub time_rate: f64,
      pub star_count: u32, pub warnings: Vec<String>, pub inside_horizon: bool,
  }
  pub struct HudControls { pub scene_ev100: f32, pub star_ev100: f32, pub lens_scale: f32, pub visible: bool, pub show_help: bool }
  pub struct Hud { /* egui ctx, state, renderer */ }
  impl Hud {
      pub fn new(window: &winit::window::Window, gpu: &uni_gfx::GpuContext, plain_format: wgpu::TextureFormat) -> Self;
      pub fn on_window_event(&mut self, window: &winit::window::Window, e: &winit::event::WindowEvent) -> bool; // consumed
      pub fn draw(&mut self, window: &winit::window::Window, gpu: &uni_gfx::GpuContext, enc: &mut wgpu::CommandEncoder,
                  target: &wgpu::TextureView, size_px: [u32; 2], data: &HudData, controls: &mut HudControls) -> Vec<wgpu::CommandBuffer>;
  }
  ```
- **Formatting rules (exact):**
  - Distance:
    - d < 1e3 m: `"{:.1} m"`
    - d < 0.01 AU: `"{:.0} km"`
    - d < 0.1 pc: `"{:.3} AU"`
    - d < 1e3 pc: `"{:.2} pc ({:.1} ly)"`
    - d < 1e6 pc: `"{:.2} kpc"`
    - otherwise: `"{:.2} Mpc"`
  - Speed:
    - < 1e3 m/s: `"{:.1} m/s"`
    - < 0.01 c: `"{:.2} km/s"`
    - ≤ 1 c: `"{:.3} c"`
    - above c: `"{:.2e} c · FTL (non-physical)"`
- **HUD v0 layout:** a single translucent panel, top-left.
  - It shows FPS, CPU ms, GPU ms per pass ("GPU ms n/a" if timestamps are unavailable), distance from the Sun, the nearest named object and its distance, speed, epoch and rate, and star count.
  - Warnings are shown in amber. "Inside event horizon" is shown in red when set.
  - Exposure sliders (scene EV, star EV), a lens-scale slider, and the label "Lens background: M0 debug grid".
  - F1 shows a controls list.
  - The final visual design comes from the M1 mockup (Task 14); this is functional only.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn distances() {
      assert_eq!(format_distance(500.0), "500.0 m");
      assert_eq!(format_distance(1.5e6), "1500 km");
      assert_eq!(format_distance(AU), "1.000 AU");
      assert_eq!(format_distance(1.3384 * PC), "1.34 pc (4.37 ly)");
      assert_eq!(format_distance(8200.0 * PC), "8.20 kpc");
      assert_eq!(format_distance(16.8e6 * PC), "16.80 Mpc"); }
  #[test] fn speeds() {
      assert_eq!(format_speed(12.0), "12.0 m/s");
      assert_eq!(format_speed(29_780.0), "29.78 km/s");
      assert_eq!(format_speed(0.5 * C), "0.500 c");
      assert_eq!(format_speed(1000.0 * C), "1.00e3 c · FTL (non-physical)"); }
  #[test] fn epoch_format() { assert_eq!(format_epoch(Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap()), "2026-11-20 00:00:00 UTC"); }
  ```
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-ui --lib`.
- [ ] **Step 3: Implement.**
  - `format_epoch` uses `Epoch::to_unix_utc` (Task 3).
- [ ] **Step 4: Confirm they pass.** Run `cargo test -p uni-ui --lib`.

---

### Task 11: uni-app — wiring, CLI, capture and selftest (wave 2)

**Files**
- Create:
  - `crates/uni-app/src/{main,app,world,capture,selftest}.rs`
  - `bookmarks/m0.ron`
- Test: `crates/uni-app/src/world.rs` unit tests

**Interfaces**
- Consumes: every crate above.
- Produces the `universe` binary:
  - `universe`: windowed mode, start bookmark `"start"` from `bookmarks/m0.ron`.
  - `universe --capture <bookmarks.ron> --out <dir> [--size 1920x1080] [--frames 4] [--hud]`
    - Headless.
    - For each bookmark it writes `<dir>/<name>.png` and `<dir>/<name>.json`. The JSON is `{name, width, height, gpu_ms: {pass: ms}, cpu_ms, star_count, lens_scale, epoch_utc}`.
    - Exit code is 0 on success and 2 if a data file is missing.
  - `universe --selftest`
    - Headless.
    - Runs the GPU twins and compares them with the CPU twins:
      - `pcg3d`: 4096 inputs must be bit-exact.
      - `agx`: 256 inputs within 1e-3.
      - Binet: 64 rays, exit direction within 2e-3 rad, with matching capture flags.
    - Prints `PASS name` or `FAIL name: detail` per check. Exit code is 0 only if everything passes.
- `World` (in `world.rs`):
  - `epoch`, `time_rate`, `frames: FrameTree`, `lenses: Vec<Lens>`
  - `TestBH-1`: mass 10 M☉ at SSB + (1000 AU, 0, 0), spin 0
  - `fn nearest(&self, cam: &UniversePos) -> (String, f64)`, which considers the Sun at the SSB in M0 and the lenses
  - `fn load_sky(path: &Path) -> (Vec<uni_data::CatalogStar>, Vec<String>)`, which returns stars plus warning strings and never panics
- Feature order: `LensTracer`, `StarField`, `TonemapFeature`. The HUD is drawn after them, windowed only, or with `--hud`.

**`bookmarks/m0.ron` entries.** fov_y is 60° unless stated. Positions are given relative to the SSB or to TestBH-1. The implementer computes the quaternions; the comments in the file describe the intent.
1. `start`: 60 M from TestBH-1, looking at it, with the hole 15° right of centre.
2. `sky-orion`: at the SSB, looking at RA 83.8°, Dec −1.2°.
3. `sky-galactic-center`: at the SSB, looking at RA 266.405°, Dec −28.936°.
4. `sky-zoom-sirius`: at the SSB, fov_y 2°, looking at Sirius (RA 101.2872°, Dec −16.7161°).
5. `bh-shadow-30M`: a static observer at r = 30 M from TestBH-1, looking straight at it, lens_scale 1.0. Expected shadow radius at 1920×1080: **158.745 px**.
6. `bh-oblique-15M`: r = 15 M, with the hole 20° off-axis.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn nearest_is_test_bh_at_start() { /* camera 60 M from TestBH-1 → nearest().0 == "TestBH-1", dist ≈ 60·GM/c² (±1 m) */ }
  #[test] fn bookmarks_file_parses() { /* load_bookmarks("bookmarks/m0.ron") has the 6 names above */ }
  #[test] fn missing_catalog_warns_not_panics() { /* World::load_sky(Path::new("/nonexistent")) → empty Vec + one warning string */ } // Review Focus 3
  ```
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-app --bins`.
- [ ] **Step 3: Implement.**
  - Follow the skeleton's `ApplicationHandler`.
  - Skip rendering while minimised (size 0).
  - Windowed mode uses `ControlFlow::Poll` and Fifo present mode.
  - CPU ms is measured around update plus encode.
- [ ] **Step 4: Confirm they pass.** Run `cargo test -p uni-app --bins`.
- [ ] **Step 5: Lead only: run the binary unsandboxed.**
  - `cargo run -p uni-app -- --selftest` must print all PASS.
  - `cargo run -p uni-app -- --capture bookmarks/m0.ron --out captures/m0` must write 6 PNG/JSON pairs.

---

### Task 12: tools/verify — capture sheet and shadow measurement (wave 2)

**Files**
- Create: `tools/verify/src/{main,sheet,shadow}.rs`
- Test: in `shadow.rs`

**Interfaces**
- Consumes: the PNG/JSON files written by Task 11, and `image`.
- Produces the `uni-verify` binary:
  - `uni-verify sheet --captures <dir> --out <dir>/index.html`
    - A self-contained HTML page with every PNG (relative links), its JSON metrics as a table, and the git commit hash and date.
  - `uni-verify shadow --png <file> --expected-px <f64> --tol <f64>`
    - Finds the connected region of pixels with luma < 2/255 that contains the image centre.
    - Radius = √(area/π).
    - Prints `measured=<r> expected=<e> PASS|FAIL`. Exit code is 0 on PASS.

**Steps**
- [ ] **Step 1: Write the failing tests.**
  ```rust
  #[test] fn synthetic_disc_radius() { /* 1920×1080 grey image with a black disc of radius 158.745 at the centre (antialiased) → measured within 0.5 px */ }
  #[test] fn ignores_black_outside_centre_region() { /* add an unconnected black square in a corner → radius unchanged */ }
  ```
- [ ] **Step 2: Confirm they fail.** Run `cargo test -p uni-verify --bins`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm they pass.** Same command.

---

### Task 13: M0 gate (lead)

- [ ] **Step 1: Check the workspace.** `cargo check --workspace && cargo test --workspace --lib --bins` must be all green. Record the shared target size and free disk.
- [ ] **Step 2: Run the selftest unsandboxed.** `cargo run -p uni-app -- --selftest` must print all PASS.
- [ ] **Step 3: Capture.** `cargo run --release -p uni-app -- --capture bookmarks/m0.ron --out captures/m0 --size 1920x1080`. `captures/` is git-ignored (Task 1).
- [ ] **Step 4: Measure the shadow.** `cargo run -p uni-verify -- shadow --png captures/m0/bh-shadow-30M.png --expected-px 158.745 --tol 1.0` must print PASS.
- [ ] **Step 5: Look at every PNG with the Read tool.**
  - `sky-zoom-sirius`: round, smooth stars with no square texels.
  - `sky-orion`: Orion recognisable.
  - `sky-galactic-center`: the Sagittarius/Scorpius region recognisable.
  - The BH captures show a black shadow, a distorted grid and an Einstein-ring-like compression near the edge.
  - Any failure becomes a fix task for the owning crate, then repeat from Step 1.
- [ ] **Step 6: Check performance.** Open the JSON metrics. Record GPU ms per pass at 1920×1080 for every bookmark in the sheet. M0 has no FPS gate, but numbers must be present, or "n/a" with a reason.
- [ ] **Step 7: Build the sheet.**
  - `cargo run -p uni-verify -- sheet --captures captures/m0 --out captures/m0/index.html`.
  - Publish it to the owner as a private Artifact page, with the PNGs uploaded as assets.
- [ ] **Step 8: Commit and tag.**
  - `git commit` the wave.
  - `git tag m0`.
  - Update the "Current real layout" section of CLAUDE.md to match reality.

---

### Task 14: HUD visual design mockup (lead, parallel with wave 1)

- [ ] **Step 1: Quickstart.** Run Artifact `quickstart` with intent `design`, then create an original HUD mockup.
  - Layouts: in flight (minimal), target info card, black-hole info card (radii, time dilation, Hawking T), and time-warp/jump controls.
  - Style: translucent, instrument-like, thin type, SI units, colour used only for meaning, and a clear "non-physical" badge style.
  - Inspiration comes from general online references for space UI and scientific instruments. Nothing is copied from a specific product.
- [ ] **Step 2: Share it.** Send the link to the owner for feedback. The approved design becomes the input to M1's `uni-ui` task.
