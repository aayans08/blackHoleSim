## 1. Current state

- **The only Rust program has never been seen working.** It is `src/main.rs` (927 lines, no modules) plus `shaders/compute.wgsl` and `shaders/display.wgsl`, on wgpu 0.19.4 / winit 0.29.15. `cargo check` finishes with 0 errors and 0 warnings, and both shaders pass naga validation. Nobody has looked at its output. Working from the naga field offsets and evaluating the shader by hand, every frame comes out as one flat slate-blue colour.
- **Three separate bugs each blank the image on their own:**
  - The WGSL `vec3` alignment rule means the shader reads `schwarzschild_radius = 0` and `fov = 0`.
  - Each ray travels at most 150 × 0.1 km = 15 km, but the camera is 301 km away, so no ray reaches the hole or escapes.
  - The starting camera points about 88° away from the hole.

  The image is also flipped vertically and sRGB-encoded twice.
- **All the physics is a heuristic.**
  - Lensing gives a shadow of 4.03 Rs (correct value 2.598 Rs) and a weak-field deflection of 3Rs/b (correct value 2Rs/b), integrated with forward Euler and renormalised every step.
  - The disk has its Doppler sign reversed, uses Newtonian speeds, blends temperature linearly and uses step-function colours. Clamps make it nearly invisible and it is never orange.
  - There is no CPU physics at all.
- **The sky is what makes it look "pixelated".**
  - A square 2048² texture is mapped as a 2:1 equirect.
  - The RNG returns values in [0, 2), so about 75% of the stars are dropped.
  - Stars are single white texels with no mipmaps.
  - The v axis uses Repeat instead of ClampToEdge.
  - The background is stored as linear values, so space renders blue-grey.
- **Basic foundations are missing.**
  - There is no raster pipeline or depth buffer, so planets are impossible.
  - Positions are f32 kilometres, which gives 16 km resolution at 1 AU.
  - The camera is yaw/pitch at a fixed 50 km/s.
  - There is no HUD, FPS counter or GPU timing, and only `SurfaceError::Lost` is handled.
- **The legacy Python/GLSL code (about 190 KB) is all dead or fake.**
  - It contains 7 ad-hoc lensing variants. One of them bends rays away from the hole on the way out.
  - The Barnes–Hut code is never called and has a Velocity-Verlet bug.
  - There is no real astronomy data, and every position is arbitrary.
  - Nothing in Rust depends on it, so it is safe to archive.
- **The docs are actively harmful.**
  - CLAUDE.md (40 KB, loaded into every agent) describes files that don't exist. It claims RK4, self-shadowing, bloom, adaptive quality and "60 FPS ✅ Achieved", none of which exist. Its reference `compute_acceleration` has gravity pointing the wrong way and would not compile.
  - Each era ended with a report declaring the work finished, backed by benchmarks that were never measured. The "4K" textures are 2K, and the Panda3D "Barnes-Hut integration" is a function that only does `pass`.
- **Nothing is under version control.** Only `.gitattributes` is committed, so one bad `rm` loses everything. The history is 4 framework switches and at least 17 renderers in about 3.5 weeks.
- **Assets:**
  - 7 unique 2K Solar System Scope maps (CC BY 4.0) plus 5 byte-identical duplicates.
  - The HDRI is gone.
  - `target/` holds a stale 385 MB wgpu 0.19 release build.
- **The new stack is proven on this machine.** A wgpu 30.0.2 / winit 0.30.13 / egui 0.37 skeleton ran on this M3 (16 GB) at about 60 FPS with no validation errors. Metal facts measured here:
  - no f64 on the GPU
  - timestamps only at pass boundaries
  - 32 storage textures per stage
  - BC and ASTC compression available
  - `adapter.limits()` must be requested explicitly, or you get WebGPU defaults
- **Environment, checked now:**
  - The toolchain is rustc/cargo 1.100.0. The Rust report saw 1.89 because it was updated part-way through the research; 1.100 meets egui 0.37's minimum of 1.95.
  - `df` shows 14 GiB free, not about 3.4 GB. Free space swung between 1.2 and 14 GB during the research, so this plan still budgets for 3.4 GB.

## 2. Keep, port, archive, delete

**Order matters. Do these before any agent starts:**
1. Make one snapshot commit of the whole tree, excluding `target/`, `textures/` and `__pycache__/`.
2. Archive the legacy files.
3. Delete what is listed below.
4. Replace CLAUDE.md.

| Action | Items |
|---|---|
| **Port the pattern to wgpu 30 / winit 0.30** (rewrite the code against the verified skeleton) | Device, adapter and surface setup with Metal and HighPerformance. HDR compute texture → fullscreen-triangle tonemap. Resize and bind-group rebuild, as a helper. bytemuck `Pod` uniforms (switch to vec4 packing). The key-mapping idea. The shape of the emission–absorption loop. ACES as an optional tonemapper (default to AgX). |
| **Rewrite completely** | All geodesic and disk shader code. Uniform structs on both sides. The starfield. Camera and units. Post-processing. The monolithic `main.rs`. |
| **Carry forward as specs, not code** | BH constants and radii (re-expressed via GM, plus b_c = 3√3/2 Rs). The Barnes–Hut algorithm, with the Verlet bug fixed, softening, and an acceleration API. The layered planet material recipe (InterstellarRenderer.py:333-439). The atmosphere coefficients and Henyey–Greenstein phase (shaders/atmosphere.frag:19-36); **not** shaders.py:122, which has the colour order reversed. The HDR post design. Multi-crossing disk accumulation. Galaxy morphology recipes, placed in a galactic frame. Orbit-camera mode and multiplicative zoom. Shader maths in Rs/M units. Flamm's-paraboloid overlay. Also: FINAL_SOLUTION.md:72-97 ("what makes it look real") and IMPROVEMENTS.md:261-292 (reference list) go into the new design doc. |
| **Archive** into `archive/legacy-2025.tar.zst` after the snapshot commit | All `*.py`, `shaders/*.vert`/`*.frag`, `black_hole_shader.glsl`, `run.sh`, `RUN_PANDA3D.sh`, `requirements.txt`. All legacy docs: README*, RENDERING, IMPLEMENTATION_REPORT, PANDA3D_GUIDE, BUILD_REPORT, QUICKSTART.*, IMPROVEMENTS, RUST_*, FINAL_SOLUTION. The old CLAUDE.md. |
| **Delete** | `target/` (385 MB, wgpu 0.19). `__pycache__/` (stale `.pyc` for deleted modules). The 5 duplicate textures under `textures/{planets,stars,space}/`. `.DS_Store`. The empty `textures/hdri/`. |
| **Keep as dev placeholders only** | The 7 unique 2K SSS maps, with CC BY attribution recorded. Replace them with NASA public-domain sources or procedural detail before release (see open question 2). |
| **Replace first** | **CLAUDE.md**, at about 150 lines: the real layout, crate ownership, the units and precision rules, build and disk rules, and a verification rule ("no claim of done without a capture plus measured numbers"). Keep the old file's correct metric, Christoffel symbols and radii (about :180-232) and the correct Binet geodesic form. Expand `.gitignore` to cover `__pycache__/`, `.DS_Store` and `data/cache/`. |

## 3. Biggest technical risks, ranked

1. **Progress that can't be checked.** Every past attempt was declared done without anyone looking at the output, and agents cannot see the window. *Mitigation:*
   - Build headless capture early (`--capture bookmarks.ron` → PNG plus per-pass GPU ms as JSON).
   - Add physics unit tests and golden metrics.
   - Have the owner review a capture sheet at every milestone gate.
2. **The precision foundation.** It covers about 30 orders of magnitude with no f64 on the GPU, and retrofitting it later touches everything. *Mitigation:* freeze `UniversePos`, the frame tree, `Epoch` and the camera-relative upload contract in M0, before any feature work.
3. **Planet quality on a fanless M3 Air.** Reaching the reference image needs quadtree terrain, high-resolution normal/albedo atlases, Hillaire atmosphere, cloud shells and TAA within about 9 ms at 1080p internal. Sustained throughput is about 60–70% of peak, and every budget figure so far is an estimate. *Mitigation:* GPU timestamps per pass from M0, an adaptive-quality controller, and TAAU to reach native panel resolution.
4. **Lensing composited with the raster scene.** Full-screen per-pixel geodesic tracing does not fit the budget (about 15–25 RK steps per pixel against the 60–150 a ray needs). Using lookup tables plus remapping the already-rendered scene introduces parallax errors for geometry close to the lens, and Kerr needs 4D tables or analytic elliptic functions. *Mitigation:*
   - Schwarzschild lookup tables first, validated against a brute-force tracer.
   - Lens point stars exactly per star.
   - Flag the approximation for lensed scene geometry.
   - Kerr as phase 2.
5. **Crisp stars and HDR.** The image needs 10⁵–5×10⁵ point-spread-function splats that are excluded from TAA, deflected and magnified near lenses, and shown next to sunlit planets despite a 10¹³ brightness ratio. That requires a deliberate, documented dual-exposure choice.
6. **Procedural universe determinism and streaming.**
   - CPU and GPU noise must match exactly, using integer hashes written twice.
   - The luminosity-stratified octree must stay within a fixed cost.
   - Catalogue stars and procedural stars must merge without duplicates.
   - Re-entering a sector must reproduce it exactly at any time-warp value.
7. **Physics under time warp.** Each object needs the right motion tier (rails, potential integration, n-body, ship geodesic) with no visible handoff glitches, and non-physical modes (cruise, jumps) must be labelled honestly.
8. **Agent concurrency on one disk and one GPU.**
   - Builds wait on a shared cargo lock, and so do GPU captures.
   - One broken shared crate stalls everyone.
   - Disk headroom swings by gigabytes because of other activity on the machine.
9. **Licensing and originality.** SSS textures are another product's processed assets (CC BY). AT-HYG and OpenNGC are CC BY-SA, which applies to any redistributed derived data. The reference screenshot comes from a commercial product, so match its quality, not its look.
10. **Dependency churn.** wgpu releases a new major version about every quarter, and egui pins winit ^0.30.13 (the 0.31 beta can't be used). Pin exact versions and upgrade only as a deliberate milestone task.

## 4. Candidate engine architectures

### A. Hybrid raster + compute on a CPU frame tree (recommended)
- **Rendering:**
  - Raster with reversed-Z for anything with geometry.
  - A background "at infinity" layer drawn as directions only.
  - Compute for lookup tables, lensing and post-processing.
- **Lensing:** a screen-region pass driven by lookup tables.
- **CPU:** all f64 and integer work.

### B. One unified compute ray tracer
This extends the current code. Everything is ray-marched per pixel: analytic spheres, SDF or height-field terrain, volumetric atmosphere, and geodesic bending applied uniformly.

### C. Bevy + big_space
An off-the-shelf ECS with its render graph, egui and integer-grid precision.

| | A. Hybrid | B. Unified tracer | C. Bevy |
|---|---|---|---|
| Planet detail at the reference bar | High: meshes plus texture atlases, cheap on a tile-based GPU | Poor: height-field marching blows the ~15k flop/pixel budget | High |
| Lensing generality | Exact for stars, disk and shadow; approximate for scene geometry near the lens | Exact for everything | Like A, but fighting the engine's pipeline |
| M3 Air performance | Best | Worst; no hardware ray tracing on Metal via wgpu | Good, with less control over tile-memory practices |
| Parallel agent teams | One crate per feature | One giant shader, so constant conflicts | Plugins parallelise well |
| Build size and disk | About 0.6 GB measured for the core stack | Same | Larger (not measured), plus version churn |
| Fits "build our own engine" | Yes | Yes | No |
| Reuses current code | Patterns only | Most of the current shape (but that code is wrong) | None |

**Recommendation: A.** Inside each lensing region it borrows B's per-pixel idea: a lookup-table tracer only where deflection exceeds about 1 px. Details follow.

**Module layout.** A Cargo workspace. Crate names need a prefix because `core` collides with Rust's built-in crate.
```
crates/
  uni-app/         binary: winit ApplicationHandler, main loop, wiring (lead)
  uni-core/        UniversePos, FrameTree, Epoch, units/constants, splitmix/PCG hash + RNG, Planck→XYZ→sRGB LUT
  uni-gfx/         device/surface, transient targets, pass scheduler, WGSL #import preprocessor + naga test,
                   GPU timestamps, headless capture; shaders/common/*.wgsl (hash, color, camera, constants)
  uni-sky/         star PSF splats, galaxy/Milky Way cubemap, clusters, nebula/galaxy impostors
  uni-planet/      cube-sphere quadtree, CPU f64 tile gen + GPU detail, materials, ocean, rings, night lights, star surfaces
  uni-atmo/        Hillaire transmittance/multi-scatter/sky-view/aerial LUTs, 2D cloud shells
  uni-rel/         BH/wormhole/NS deflection + disk-crossing LUT builders, lens pass, disk shading
  uni-post/        histogram exposure, bloom pyramid, TAA/TAAU, AgX/PBR-Neutral, blue-noise dither
  uni-physics/     Kepler/Standish/Meeus, IAU rotation, epicycles, MW potential, leapfrog/WH/DP45, ship SR+GR, Hawking calcs
  uni-data/        runtime loaders for baked catalogs, known-objects tables (BHs, HVS, Sgr A* S-stars)
  uni-procgen/     galaxy model, sector octree, IMF/SSE templates, binaries, clusters, remnants, systems, planet params
  uni-sim/         world state, time controller, streaming/LOD scheduler, rayon job system
  uni-nav/         camera/ship controller, flight modes, autopilot, jump sequencing, bookmarks
  uni-ui/          egui HUD, search, info cards, overlays
tools/bake/        offline: CSV/TIF/IMG/JPG → packed .bin + KTX2 (BC7) tiles
tools/verify/      headless bookmark captures, golden metrics, physics validation, perf report
data/manifest.toml (URL, SHA-256, licence, credit) · data/baked/ (small, committed) · data/cache/ (ignored)
```
Each render crate implements one trait: `RenderFeature { prepare(&FrameCtx), encode(&mut Encoder, &FrameTargets) }`. `FrameCtx` provides camera-relative transforms, jitter, exposure, time and lens list. A change to an interface in `uni-core` or `uni-gfx` goes through the lead.

**Render graph.** A fixed, hand-written pass list with typed resource handles, not a general graph library.
1. **CPU.** Sim tick (epoch, rails, ship). Query the visible set (octree culling, magnitude LOD). Upload instances with camera-relative origins.
2. **Amortised compute.**
   - Atmosphere lookup tables, only when the sun or planet changes.
   - One galaxy-cubemap face per frame.
   - Lens tables, rebuilt when the camera radius changes beyond a tolerance; CPU rayon or compute.
   - Terrain tile detail.
3. **Background pass, no depth.** Galaxy cubemap and band, plus distant-galaxy impostors.
4. **Opaque pass.** Reversed-Z `Depth32Float` with an infinite far plane: terrain tiles, oceans, rings, star spheres. Optional 4× MSAA as a transient attachment.
5. **Atmosphere and clouds.** Applied as a fullscreen pass with depth, including aerial perspective.
6. **Lens pass, compute, only inside each lens's screen region.** For each pixel:
   - Look up its exit direction.
   - If the ray is captured: black.
   - If it crosses the disk: shade analytically with g-factor and Planck at gT.
   - Otherwise remap the rendered scene color, using depth so objects in front of the lens are not lensed, or fall back to a camera-centred environment cubemap.
   - Also output the per-pixel magnification for the star pass.
7. **TAA/TAAU resolve.** Stars are excluded.
8. **Star pass at native resolution.** Point-spread-function splats; point-lens equation in the vertex shader for the far field and the lookup table in the near field; magnification μ; Doppler D. Stars use their own exposure curve.
9. **Bloom** (13-tap down / tent up), then exposure from the previous frame's histogram, then AgX, then blue-noise dither, then the swapchain.
10. **egui HUD.**

**Coordinates and precision.**
- **Absolute positions:** `UniversePos { cell: I64Vec3, local: DVec3 }` with cell size 2⁴⁴ m (about 117 AU). That gives sub-millimetre precision locally and a range of about 1.6×10³² m.
- **Motion:** a `FrameTree` of galaxy → system barycentre → body frames (inertial and rotating). The camera and ship are always parented to the nearest frame.
- **GPU uploads:**
  - Every object or tile origin is computed relative to the camera in f64, then cast to f32.
  - Star batches use a per-sector origin (f64 → f32 camera-relative) plus f32 local offsets, so double-single emulation is not needed.
- **Depth:** reversed-Z, infinite far plane, `CompareFunction::Greater`, and no fragment depth writes.
- **Time:** `Epoch { day: i64, sec: f64 }` in TDB since J2000.
- **Noise:** integer-lattice-plus-fraction inputs, with the same integer hash in Rust and WGSL and a test that cross-checks them.

**Universe database and procedural generation.**
- **Baked real data** (tier 1 is about 342 MB raw, deleted after baking):
  - AT-HYG m10 (332k stars with 3D velocities), packed at about 32 B/star ≈ 11 MB
  - BSC5 for star names
  - OpenNGC, Harris globulars, the NASA Exoplanet Archive
  - Standish elements, Meeus lunar terms, IAU rotation
  - a hand-cited table of known black holes and hypervelocity stars
- **Procedural generation:** follows the procgen brief's chain, with every stage seeded only from its own key and earlier outputs:
  - a galaxy density/potential/dust model (one model drives both the band and the resolved stars)
  - a luminosity-stratified sparse octree with Poisson counts per cell
  - startup IMF × star-formation-history × SSE template tables
  - binaries, clusters, remnants, HVS/runaways
  - planetary systems and planet parameters generated on demand
- **Catalogue merge:** procedural stars are suppressed inside each luminosity bin's completeness distance.
- **Caching:** RAM LRU only; nothing generated is written to disk.
- **IDs:** 64-bit, with catalogue and procedural IDs in separate namespaces.

**Physics and time tiers.**
1. **Analytic rails:**
   - Kepler orbits for planets, moons and binaries
   - Standish elements for the major planets
   - epicycles for disk stars
   - linear proper motion for catalogue stars within about 10⁵ yr
2. **Orbits integrated in MWPotential2014 (rescaled):** halo stars, hypervelocity stars and runaways, stored as precomputed orbit tables.
3. **Sandbox n-body:** a Wisdom–Holman-style integrator for visited procedural systems, and direct GPU summation or Barnes–Hut for clusters you get close to.
4. **The ship:**
   - an f64 adaptive Dormand–Prince restricted n-body integrator, plus 1PN correction near the Sun
   - Kerr–Schild timelike geodesics inside about 100 rs of a black hole
   - proper velocity for special relativity

Hawking radiation, time dilation and tidal effects are HUD calculators, not visuals.

**Travel.** Flight modes:
- **Inertial:** real gravity, no assist.
- **Station-keep:** the engine cancels gravity and shows the thrust needed.
- **Orbit-hold.**
- **Cruise:** logarithmic speed tied to distance to the nearest surface; gravity is ignored, and the mode is labelled as such.
- **Relativistic:** aberration and Doppler on.
- **Jump:** a wormhole mouth pair is spawned ahead of the ship and at the destination. Rendering uses the Ellis/Morris–Thorne lookup table plus a destination cubemap rendered progressively from the destination frame. When ℓ changes sign, the camera is re-parented to the destination frame. The mode is labelled hypothetical.

## 5. Phased roadmap

**Standing teams and ownership.** One agent per module directory. At most about 3 agents per crate, split by module subdirectory.

| Team | Agents | Owns |
|---|---|---|
| Lead / integrator | 1 | workspace `Cargo.toml`, `uni-app`, CLAUDE.md, interface changes to `uni-core`/`uni-gfx`, merge queue |
| Core & sim | 2–3 | `uni-core`, `uni-sim`, `uni-physics` |
| GFX platform | 2–3 | `uni-gfx` (incl. `shaders/common`), `uni-post` |
| Sky | 2–3 | `uni-sky` |
| Planet & atmosphere | 3–5 | `uni-planet`, `uni-atmo` |
| Relativity | 2–3 | `uni-rel` |
| Data | 1–2 | `uni-data`, `tools/bake`, `data/manifest.toml`, ATTRIBUTION |
| Procgen | 3–5 | `uni-procgen/src/{galaxy,stars,systems,surfaces}` |
| Nav / UI | 1–2 | `uni-nav`, `uni-ui` |
| Verify / perf | 1–2 | `tools/verify`, golden sets, all release builds |

**Build, disk and GPU rules.** These go into the new CLAUDE.md.
- **One shared target directory.** Set `[build] target-dir = "/Users/aayan/.cargo/shared-target"` in `~/.cargo/config.toml`, with the measured dev profile (`debug = "line-tables-only"`; dependencies at `opt-level = 2` with `debug = false`) and release with `lto = "thin"`.
- **Worktrees are allowed only as source-only checkouts on that shared target**, never with a target directory of their own. `data/` is reached through an absolute path in `UNI_DATA_DIR`.
- **What agents run:** `cargo check -p <crate>` and `cargo test -p <crate> --lib` only. No `--all-targets`.
- **Full app runs and captures** go through one serialized queue owned by Verify. That covers the cargo lock, the single GPU and thermal throttling.
- **Release builds** are done only by Verify, followed by `cargo clean -r`.
- **Downloads** are done only by Data, into `data/cache/`, and raw files are deleted after baking.
- **Disk watchdog:** stop builds and downloads when free space drops below 1.5 GB.

| Disk item (new usage) | Size |
|---|---|
| Delete the old `target/` | −385 MB |
| Shared dev target (core stack, measured) | 611 MB |
| image/rayon/noise extras (estimate) | +100–250 MB |
| rust-analyzer check data | +270 MB |
| Release build, only during perf runs | +~650 MB |
| Raw tier-1 data, only until baked | ≤350 MB |
| Baked data (estimate) | 100–200 MB |
| **Peak** | **about 2.3 GB, fits the 3.4 GB budget with about 1 GB to spare** |

**Milestones.** Each one ends with something the owner can fly. Each gate is a capture sheet plus measured numbers.

Order: M0 → M1 → {M2, M3, M4 in parallel; peak about 25–30 agents} → M5 → M6, with M7 running continuously from M2 onward.

- **M0, foundation (lead + about 6 agents).**
  - *Flyable:* the new-stack app with 6-degree-of-freedom quaternion flight, f64 positions and logarithmic speed, through a real bright-star sky. The sky is BSC5's 9,110 stars as point-spread-function sprites with Planck colours. A HUD shows FPS, GPU ms per pass, position in AU/pc and speed. A correct Schwarzschild black hole sits at a test location, rendered by a brute-force half-resolution Binet RK4 tracer, which also becomes the ground truth for M3.
  - *Work:*
    - snapshot, archive and cleanup; new CLAUDE.md; workspace and crate stubs with frozen traits
    - `uni-core` types and hash twins
    - `uni-gfx` port, AgX and dither, timestamps, headless capture, WGSL preprocessor
    - `uni-nav` camera
    - BSC5 bake
    - the tracer
  - *Gate:*
    - no visible texels on stars
    - shadow radius in the capture equals 3√3 M within 1 px
    - unit tests: b_c = 2.598 Rs, deflection 0.236 rad at b = 10 Rs
- **M1, the real Solar System and real sky (10–14 agents).**
  - *Flyable:* real scale from Earth to Neptune, then out to Alpha Centauri.
    - Planets and moons move on Standish/Meeus orbits with IAU rotation, at real dates with time warp.
    - Bodies are analytic ray-sphere impostors with placeholder maps.
    - The Sun has limb darkening and granulation; Saturn's rings cast shadows both ways.
    - Hillaire atmospheres from space for Earth, Mars, Venus and Titan.
    - AT-HYG m10 in 3D with proper motion.
    - Milky Way band v0 from the galaxy model.
    - Bloom, auto-exposure and dual star exposure.
    - Target selection, goto, and info cards.
  - *Gate:*
    - positions within the Standish error table when checked against DE440s spot values
    - constellations recognisable in captures
    - 60 FPS or better at 1080p internal in Solar System scenes
- **M2, planet surfaces (8–10 agents, `uni-planet`/`uni-atmo`).**
  - *Flyable:* descend from orbit to the surfaces of Earth, the Moon and Mars.
    - Cube-sphere CDLOD terrain from GEBCO, LOLA and MOLA elevation, plus procedural detail.
    - Normal and albedo atlases; craters.
    - Ocean glint.
    - Cloud shells with cyclones and cloud shadows.
    - Night lights, aerial perspective, TAA and MSAA.
  - *Gate:* a Mars-like orbital capture side by side with the reference (soft terminator, limb glow, cloud detail, no aliasing), at 9 ms GPU or less at 1080p internal.
- **M3, black holes and relativity (5–8 agents, `uni-rel` plus ship code in `uni-physics`).**
  - *Flyable:* fly to and orbit the real black holes, plus a procedural test hole with a bright thin disk:
    - Cyg X-1 with its companion and a thin disk
    - Gaia BH1–3 as dark lenses beside their companions
    - Sgr A* as a dim radiatively inefficient flow, with S2 and the other S-stars orbiting
  - *Includes:*
    - lookup-table lensing
    - Novikov–Thorne disk with g-factor and g³/g⁴ beaming
    - lensed and magnified point stars
    - special-relativistic camera mode
    - ship geodesic integrator
    - HUD readouts for time dilation, tidal stretch and Hawking radiation
  - *Gate:*
    - lookup-table result within 1 px of the M0 tracer
    - approaching side of the disk is the bright side
    - S2 precesses about 12′ per orbit
- **M4, procedural Milky Way (8–12 agents, `uni-procgen` + `uni-sky`).**
  - *Flyable:* leave the catalogue bubble and fly anywhere in the Galaxy.
    - Octree-streamed stars with IMF/SSE properties, epicycle motion and binaries.
    - Young blue open clusters like the one in the reference.
    - Remnants, mostly dark.
    - S5-HVS1 and procedural Hills ejections and runaways that move correctly under time warp.
    - The band is generated from the same model as the stars and changes as you fly; the galaxy can be seen from outside.
    - HII regions and dust v1; Harris globulars; catalogue merge without duplicates.
  - *Gate:*
    - local density about 0.1 stars/pc³
    - binary fractions by stellar mass
    - hypervelocity stars unbound
    - golden sector hashes
    - star count on screen and CPU time per frame stay bounded
- **M5, procedural planetary systems (8–10 agents).**
  - *Flyable:* fly to any star's planets.
    - Systems from SAG13/Cumming occurrence with metallicity scaling.
    - Real exoplanet-archive systems filled in procedurally.
    - Physics-driven atmosphere class, colour, ice, oceans and biomes.
    - Sudarsky/Rhines gas-giant banding, moons, rings and rogue planets, all reusing the M2 pipelines.
  - *Gate:* occurrence statistics, plus a variety sheet that includes a procedural world matching the reference quality.
- **M6, travel and the deep universe (6–8 agents).**
  - *Flyable:* wormhole jumps to anywhere.
    - Search and bookmarks.
    - Local Group and nearby-galaxy (UNGC) galaxies as volumes or impostors; Andromeda can be entered through the same pipeline with different parameters.
    - Large-scale structure from 2MRS.
    - Kerr black holes (Kerr–Schild integration inside lens regions), neutron stars and pulsars, and a cluster n-body sandbox.
- **M7, hardening (continuous from M2).**
  - adaptive quality via a PI controller, TAAU to native panel resolution, a 30 FPS battery mode
  - volumetric clouds near the surface, an erosion filter
  - an analytic Kerr path

## 6. Open questions for the owner

1. **Machine and target:** 13" or 15" Air, and 8- or 10-core GPU? Which matters more: 1080p internal at 60 FPS, or 1440p at about 30?
2. **Assets:** is it acceptable to ship Solar System Scope CC BY maps with attribution, or should we use only NASA public-domain sources plus procedural detail? Are CC BY-SA star and deep-sky data files acceptable?
3. **Disk:** plan against 3.4 GB (as in the brief) or the 14 GiB free now? How much can go to real data (tier 1 is about 342 MB raw, temporarily)?
4. **Realism versus looks defaults:**
   - Physically correct exposure would make stars invisible next to a sunlit planet. Should we use a labelled dual exposure instead?
   - Real disks are blue-white. Do you want a film-style orange option?
   - Sgr A* is honestly dim. Is that acceptable?
5. **Non-physical travel:** jump anywhere via a labelled hypothetical wormhole, or only at placed wormholes? Is a labelled faster-than-light cruise acceptable?
6. **Time warp range:** thousands of years (enough to watch stars move) or millions and more (stellar evolution, galactic orbits)?
7. **Scope for v1:** Milky Way only, or other galaxies you can fly into? Pure explorer, or ship simulation (fuel, thrust limits)?
8. **Repo:** may we make the snapshot commit and push it? Is the GitHub repo public, and what licence should it have (none exists)? Also, `.claude/settings.local.json` currently allows `Edit(/**)` and unrestricted WebFetch. Please confirm that scope is what you want before 10–30 agents run.