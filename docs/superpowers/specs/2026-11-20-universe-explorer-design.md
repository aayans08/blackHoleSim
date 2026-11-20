# Universe Explorer: Design

**Date:** 2026-11-20
**Status:** Draft, awaiting owner approval. A builder check by 4 reviewers has been folded in.
**Repo:** `/Users/aayan/Documents/GitHub/blackHoleSim`
**Research behind this doc:** `docs/research/2026-11-20-*.md` (nine reports: rust, python, docs, assets, rendering, physics, procgen, wgpu, synthesis). Builders should read the ones relevant to their crate. They hold the equations, data sources, measured numbers and citations that this doc only summarises.

---

## 1. Intent

The owner wants one program in which they can **fly anywhere in the universe**, from a planet's surface to its star system, through interstellar space, past black holes, out to the whole Milky Way and on to other galaxies. It should look as good as the owner's reference screenshot and run on **real physics**.

In the owner's words: "a full universe system using high tier graphics and real physics", "be able to go through the universe", "real feel like Space Engine", "nothing too pixilated", "all real physics or as real as you can get via current physics knowledge, like hawking radiation, accretion disk, light bending, gravitational pull".

The reference screenshot (a Mars-like planet) sets the visual bar:
- a planet with a thin warm atmospheric rim glow, cloud wisps and a storm vortex
- varied terrain and albedo, and a soft-but-sharp terminator
- a dense field of colour-varied point stars
- a blue-white open star cluster
- a faint dust-laned Milky Way band

**This is an original engine.** The genre is "universe explorer", but no code, assets, UI or signature look are taken from any existing product. We match the reference's *quality*, not its *look*.

**Why a rewrite.** The repo has gone through about 17 renderers in 4 frameworks in roughly 3.5 weeks.
- Each one was declared finished in a report, with no one looking at its output.
- The current Rust renderer compiles, but every frame comes out as one flat colour (uniform-layout bug, 15 km rays, camera aimed away).
- Its physics is heuristic: the shadow is 55% too large and the Doppler sign is reversed.
- The textures are 2k, and the "4k" files are mislabelled.
- CLAUDE.md describes files and results that don't exist.

Details are in `docs/research/2026-11-20-rust.md` and `2026-11-20-docs.md`. The most important lesson for this design: **nothing is done until someone has looked at it and measured it.**

## 2. Goals, non-goals, anti-goals

### Goals (v1)
1. **Seamless 6-DoF free flight** with no loading screens. Speed ranges from walking pace to arbitrarily fast. Distances and sizes are real.
2. **Real objects in their real places, at the real date and time:**
   - the Solar System, planets and moons on real orbits
   - about 2.5M catalogued stars at their real 3-D positions with real motions
   - known deep-sky objects
   - known black holes (Sgr A\*, Cygnus X-1, Gaia BH1–3, …)
   - nearby galaxies
3. **Procedural worlds and stars that are physically consistent** wherever the catalogue runs out. Star masses, ages, colours, binaries, clusters, remnants and planetary systems follow published astrophysics. Special objects behave correctly: rogue planets drift, hypervelocity stars are unbound from the Galaxy, and runaway stars move fast.
4. **Real relativistic and gravitational physics:**
   - correct light bending, shadows and Einstein rings around black holes (Schwarzschild first, Kerr later)
   - Novikov–Thorne accretion disks with correct Doppler beaming and gravitational redshift
   - gravity acting on the player's ship
   - time dilation, tidal forces and Hawking-radiation readouts
5. **Visual quality at the reference bar:**
   - physically based atmospheres, cloud layers, high-resolution terrain
   - crisp point stars with no visible texels
   - a Milky Way band generated from the same model as the stars
   - HDR with bloom and auto-exposure
   - temporal anti-aliasing
6. **Wormhole jumps** for long distances, rendered from the real wormhole metric (Ellis / Morris–Thorne, as in James et al. 2015).
7. **Runs well on the owner's M3 MacBook Air** (§4).

### Non-goals (v1)
- Ship-simulation gameplay: fuel, resources, damage, missions.
- Walking or driving on surfaces. You can fly down to the ground, but there is no character controller.
- Multiplayer, VR, Windows or Linux tuning. wgpu keeps those possible later; we just don't test them.
- Offline-quality physics: GRMHD disks, full spectral radiative transfer, DNGR beam tracing, n-body simulation of whole galaxies.
- Cosmological evolution or animated stellar evolution. Stars have ages and evolutionary states, but time warp does not evolve them.
- Audio.

### Anti-goals (things that make this a failure even if it "works")
1. **Claiming something works without seeing it.** Every milestone ends with headless captures (PNG) from fixed bookmarks, measured GPU times and passing physics tests, and the owner looks at the captures. The words "done", "achieved" and "✅" must never appear without that evidence.
2. **Fake physics presented as real.** Anything non-physical must be labelled in the UI as non-physical: faster-than-light cruise, wormhole jumps (wormholes are hypothetical; they need exotic matter), dual exposure for stars, cinematic colour modes.
3. **The pixelated look:** texel-sized stars, aliasing on edges, visible LOD pops, banding.
4. **Copying another commercial product's code, assets, UI or signature look.** The owner's reference screenshot comes from such a product, so we match its quality, not its look. Openly licensed scientific data and maps are fine with attribution in `ATTRIBUTION.md`. That includes CC BY (for example the Solar System Scope maps), CC BY-SA (AT-HYG, OpenNGC) and public domain (NASA, USGS, JPL).
5. **Docs that lie.** CLAUDE.md and the reports must describe the code as it actually is.

## 3. What the owner decided

- **Travel:** free travel however you like, not an animation. Any speed. Huge jumps through a "wormhole", as in Star Trek.
- **Scale:** real sizes and distances.
- **Graphics:** use both CPU and GPU so it looks good, with nothing pixelated; "a real feel like Space Engine".
- **Content:** real objects from known astronomy, plus generated worlds. Generated worlds must be physically realistic in how they formed and how they behave. A rogue star or hypervelocity star must act like one.
- **Physics:** all real, as far as current knowledge goes: Hawking radiation, accretion disks, light bending, gravitational pull, etc.
- **Language and stack:** build on the existing Rust code. Compress the Python experiments to save space.
- **Agents:** use as many as needed (30 or more is fine), with a manager agent checking that everything works.
- **Toolchain:** Rust updated to 1.100 (done).
- **GPU runs:** the app may run outside the sandbox for GPU tests and screenshots (allowed).
- **Disk:** `venv/` deleted and the Spotify cache cleared; about 14 GB free now. Keep pip, Google and Homebrew caches.
- **Permissions:** "allow everything". The project's `.claude/settings.local.json` allows edits inside the repo, web fetches, cargo/rustup and read-only shell commands.
- **Git:** local commits are approved. Work lives on branch `universe-explorer`; the snapshot of the legacy tree is commit `075272e`. **Nothing is pushed to GitHub without asking the owner.**

## 4. Constraints

**Hardware**
- Apple M3 MacBook Air: 16 GB RAM, 8-core GPU, fanless, built-in display 2560×1664, macOS 14.6.
- Expect about 60–70% of peak GPU throughput under sustained load.

**GPU (Metal via wgpu)**
- No f64 in shaders.
- Timestamps only at pass boundaries.
- 32 storage textures per stage.
- BC and ASTC texture compression are available.
- Request `adapter.limits()` explicitly; otherwise you get the WebGPU defaults.

**Stack (pinned, verified on this machine at about 60 FPS)**
- rustc 1.100, edition 2024
- wgpu 30.0.2, winit 0.30.13
- egui / egui-wgpu / egui-winit 0.37.0
- glam 0.34.3, bytemuck 1.25, pollster 1.0
- rand 0.10 + rand_pcg, fastnoise-lite 1.1, rayon 1.12, half 2.7, image 0.25 (trimmed features)
- Pin exact versions. Upgrades are deliberate tasks.

**Disk**
- Free space swings by gigabytes because of other activity on the machine.
- Aim for about 3 GB for builds plus data. **This is not a hard ceiling.** The rule that matters: stop builds and downloads if free space drops below 1.5 GB.
- Run `cargo clean` for a profile before building another heavy one, and build the fat-LTO `dist` profile only at the very end.

**GPU access from agent shells**
- `MTLCreateSystemDefaultDevice()` returns nil from a windowless shell, but `MTLCopyAllDevices()` sees the Apple M3. wgpu 30 enumerates adapters the same way and has already run on this machine at about 60 FPS.
- App runs and captures may run unattended. The owner approved GPU use, and captures may run outside the sandbox where needed.
- Tests that need a GPU (the WGSL side of the hash twins, LUT-versus-tracer comparisons) run through the binary's `--selftest` mode, not through `cargo test`.
- One GPU job at a time; the capture queue is serialized (§10).

**Network**
- The Data team downloads into `data/cache/` with `curl`. That is the only team that touches the network apart from web research.

**Nothing in git yet.** Only `.gitattributes` is committed. See §12 for the snapshot.

## 5. Architecture (my call: hybrid raster + compute on a CPU f64 frame tree)

Three options were compared in `docs/research/2026-11-20-synthesis.md` §4:
- **(A) Hybrid raster + compute.** Chosen.
- **(B) One per-pixel compute ray tracer.** Rejected: planets cost too much, and one giant shader means constant merge conflicts.
- **(C) Bevy.** Rejected: it is not "our own engine", and it churns its versions.

### 5.1 Workspace layout

Cargo workspace. The `uni-` prefix avoids a clash with Rust's `core` crate.

```
crates/
  uni-app/      binary `universe`: winit ApplicationHandler, main loop, wiring
  uni-core/     UniversePos, FrameTree, Epoch, units/constants, integer hash + RNG (Rust/WGSL twins), Planck→XYZ→sRGB
  uni-gfx/      device/surface, render targets, fixed pass scheduler, WGSL #import preprocessor + naga tests,
                GPU timestamps, headless capture; shaders/common/*.wgsl
  uni-sky/      star point-spread-function splats, galaxy/Milky-Way cubemap, clusters, nebula and galaxy impostors
  uni-planet/   cube-sphere quadtree terrain, CPU f64 tile generation + GPU detail, materials, ocean, rings, night lights, star surfaces
  uni-atmo/     Hillaire 2020 atmosphere LUTs (transmittance, multi-scatter, sky-view, aerial perspective), cloud shells
  uni-rel/      black hole / wormhole / neutron star deflection LUTs, lens pass, accretion-disk shading
  uni-post/     histogram auto-exposure, bloom pyramid, TAA/TAAU, AgX tonemap, blue-noise dither
  uni-physics/  Kepler/Standish/Meeus orbits, IAU rotation, epicycles, Galaxy potential, integrators (leapfrog, Wisdom–Holman, DP45),
                ship special + general relativity, Hawking/tidal/time-dilation calculators
  uni-data/     runtime loaders for baked catalogues; known-object tables (black holes, hypervelocity stars, S-stars)
  uni-procgen/  galaxy model, sector octree, IMF/stellar-evolution templates, binaries, clusters, remnants, planetary systems, planet parameters
  uni-sim/      world state, time controller, streaming/LOD scheduler, rayon job system
  uni-nav/      camera/ship controller, flight modes, autopilot, jump sequencing, bookmarks
  uni-ui/       egui HUD, search, info cards, overlays, settings
tools/bake/     offline: CSV/TIF/IMG/JPG → packed .bin + KTX2 (BC7) tiles
tools/verify/   headless bookmark captures, golden physics metrics, perf reports
data/manifest.toml   URL, SHA-256, licence and credit for every data file
data/baked/          small baked outputs, committed
data/cache/          raw downloads, git-ignored, deleted after baking
```

Every render crate implements one trait:

```rust
trait RenderFeature {
    fn prepare(&mut self, ctx: &FrameCtx);
    fn encode(&self, enc: &mut wgpu::CommandEncoder, targets: &FrameTargets);
}
```

`FrameCtx` carries camera-relative transforms, TAA jitter, exposure, epoch, and the list of lenses on screen. **`uni-core` and `uni-gfx` interfaces are frozen at the end of M0.** After that, changes to them go through the lead/integrator.

### 5.2 Precision and coordinates (frozen in M0; everything depends on this)

**Positions**
- An absolute position is `UniversePos { cell: I64Vec3, local: DVec3 }`, with cell size 2⁴⁴ m (≈117 AU).
- That gives sub-millimetre precision locally and a range of about 10³² m.

**Frame tree**
- Frames nest: Galaxy → system barycentre → body (inertial and body-fixed rotating).
- The camera and ship are always parented to the nearest dominant frame.

**GPU rule**
- Every origin is computed relative to the camera in f64 on the CPU, then cast to f32.
- Star batches use a per-sector f64→f32 origin plus f32 offsets.
- Black-hole maths runs in hole-centred coordinates, in units of M = GM/c².

**Depth**
- Reversed-Z `Depth32Float` with an infinite far plane and `CompareFunction::Greater`.
- No depth writes from fragment shaders. This matters on Apple's tile-based GPUs.

**Time**
- `Epoch { day: i64, sec: f64 }` in TDB since J2000.

**Determinism**
- Procedural noise takes integer-lattice + fraction inputs.
- The same integer hash is written in Rust and in WGSL, with a test that checks they agree.

### 5.3 Frame (fixed pass list, no general render graph)

1. **CPU.** Sim tick (epoch, rails, ship). Visible-set query (octree culling, magnitude LOD). Camera-relative instance upload.
2. **Amortised compute.** Each item refreshes only when needed:
   - atmosphere LUTs, when the sun or planet changes
   - one galaxy-cubemap face per frame
   - lens LUTs, when the camera radius changes beyond a tolerance
   - terrain tile detail
3. **Background (no depth).** Galaxy cubemap and band; distant-galaxy impostors.
4. **Opaque.** Terrain tiles, oceans, rings, star spheres. MSAA attachments are transient.
5. **Atmosphere and clouds.** A fullscreen pass using depth, with aerial perspective.
6. **Lens pass (compute, only inside each lens's screen rectangle).** Per pixel:
   - look up the exit direction from the LUT
   - captured → black
   - crosses the disk → analytic Novikov–Thorne shading, with Planck colour at g·T
   - otherwise remap the rendered scene colour, using depth so foreground objects are not lensed, or fall back to an environment cubemap
   - output per-pixel magnification for the star pass
7. **TAA/TAAU resolve.** Stars are excluded.
8. **Star pass at native resolution.**
   - Stars are point-spread-function splats.
   - Lensing uses the point-lens equation in the far field and the LUT in the near field.
   - Brightness includes magnification μ and Doppler factor D.
   - Stars use their own exposure (§7.4).
9. **Post.** Bloom (13-tap down / tent up), then histogram exposure, then AgX tonemap, then blue-noise dither, then the swapchain.
10. **egui HUD.**

### 5.4 Universe content

**Real data** (bake to compact binaries; delete raw files; record every licence in `data/manifest.toml` and `ATTRIBUTION.md`)
- **Stars**
  - AT-HYG (Gaia DR3 + Tycho-2 + Hipparcos; CC BY-SA 4.0), m ≤ 10 tier first (about 330k stars), with 3-D positions and velocities.
  - Full ~2.5M stars as a later tier.
  - BSC5 for bootstrap and names.
- **Deep sky**
  - OpenNGC (CC BY-SA)
  - Harris globular clusters
  - an open-cluster list
- **Galaxies**
  - Updated Nearby Galaxy Catalog (about 870 galaxies within 11 Mpc)
  - 2MRS or Cosmicflows-4 for large-scale structure (later)
- **Solar System**
  - JPL Standish Keplerian elements (KB) first; DE440s (31 MB) as an optional upgrade
  - Meeus lunar terms
  - JPL satellite elements
  - IAU WGCCRE rotation
- **Surfaces**
  - Solar System Scope 8k set (CC BY 4.0)
  - NASA public-domain sources where better: Blue Marble NG + GEBCO elevation, Black Marble, LRO colour + LOLA elevation, MOLA elevation
  - Transcode to KTX2 BC7.
- **Exoplanets:** NASA Exoplanet Archive PSCompPars.
- **Black holes and special stars:** a hand-curated table with citations. Sgr A\* and the S-stars, M87\*, Cyg X-1, Gaia BH1/2/3, S5-HVS1 and other known hypervelocity stars.

**Total real-data budget:** about 0.5 GB peak, about 0.35 GB steady (`docs/research/2026-11-20-assets.md` §4.6).

**What gets committed**
- Baked files under 10 MB are committed in `data/baked/`.
- Anything larger is regenerated from `data/manifest.toml` by `tools/bake` and git-ignored. That keeps the repo small and works whether or not it is ever pushed or made public.

**The real Milky Way seen from inside**
- Real dust features (the Great Rift, the Coalsack) should be visible, since the owner prefers real data where it exists.
- Within a few kpc of the Sun, the band blends the galaxy model with a real 3-D dust map (Edenhofer et al. 2024 or Lallement et al. 2022), downsampled and baked to about 50 MB or less.
- Farther out, the model takes over.
- The Data team confirms sizes before downloading. M1 ships band v0 from the model; M4 adds the real dust map.

**Procedural generation** (`docs/research/2026-11-20-procgen.md` has the equations and distributions)
- One Milky Way model drives everything:
  - thin and thick disk, bar/bulge, halo
  - Reid 2019 spiral arms
  - dust
  - a rescaled MWPotential2014 potential
  
  The same model produces both the resolved stars and the sky band, so the band changes correctly as you fly.
- A luminosity-stratified sparse octree with Poisson counts per cell, seeded only from cell keys.
- Star population: Kroupa IMF × star-formation history × stellar-evolution templates, giving mass, age, metallicity, luminosity, radius, temperature and phase.
- Also generated:
  - binaries and multiples, by primary mass
  - open and globular clusters
  - remnants: white dwarfs, neutron stars and pulsars, stellar black holes, most of them dark
  - runaway stars, and hypervelocity stars from Hills-mechanism ejections at the Galactic Centre
- Planetary systems:
  - occurrence from SAG13 / Cumming with metallicity scaling
  - Hill-stable spacing, mass–radius relations
  - equilibrium temperature and greenhouse effect
  - Jeans-escape atmosphere retention, tidal locking
- Surface classes driven by those physics: ice, oceans, biomes, Sudarsky gas-giant classes, craters scaled by age, simplified tectonics.
- Rogue planets.
- **Catalogue merge:** procedural stars are suppressed inside each luminosity bin's catalogue completeness distance, so there are no duplicates.
- Everything generated is deterministic and lazy. It is cached in RAM only and never written to disk.

### 5.5 Physics tiers (`docs/research/2026-11-20-physics.md`)

**1. Rails (analytic).**
- Kepler orbits: planets via Standish, moons, binaries.
- IAU rotation.
- Epicycles for disk stars.
- Linear proper motion for catalogue stars over ±10⁵ yr.

**2. Integrated in the Galactic potential.**
- Halo stars, hypervelocity stars and runaways.
- Precomputed orbit tables, so they stay correct under any time warp.

**3. Sandbox n-body.**
- Wisdom–Holman for a visited procedural system.
- Direct summation or Barnes–Hut for clusters you get close to.

**4. The ship.**
- Restricted n-body with an f64 adaptive Dormand–Prince integrator: bodies stay on rails, plus J2 oblateness and 1PN near the Sun.
- Kerr–Schild timelike geodesics within about 100 r_s of a black hole.
- Proper velocity for special relativity.

**5. Light.**
- Schwarzschild via the exact Binet form d²x/dλ² = −(3/2) r_s h² x/r⁵, using Bruneton-style precomputed LUTs.
- **Kerr in M3.** Real holes spin: Cygnus X-1 has a\* > 0.99. Spinning holes use per-pixel Kerr–Schild Hamiltonian integration, only inside the lens's screen rectangle, at reduced resolution with upsampling, so the physics is exact and only the speed suffers. Faster Kerr methods (LUTs, analytic paths) come in M6/M7.
- Disk: Novikov–Thorne T(r), with redshift g including Doppler and gravity. Observed colour is Planck at g·T, and intensity ∝ g⁴.

**6. Calculators shown on the HUD.**
- Hawking temperature, power and evaporation time.
- Time dilation: static, orbiting and general Kerr.
- Tidal acceleration.
- Escape and circular speed.

**Hawking radiation, honestly** (my call):
- For any astrophysical black hole it is far colder than the CMB, so it is invisible and is shown only as HUD numbers.
- A labelled **sandbox tool** lets you spawn a hypothetical micro black hole of any mass. Its Hawking temperature, luminosity and evaporation countdown are computed correctly. Because the emission is mostly gamma rays, it is shown through a labelled "invisible-spectrum" false-colour view, not as fake visible glow.

### 5.6 Travel (my call; non-physical modes are labelled in the HUD)

| Mode | Physics | Notes |
|---|---|---|
| **Inertial** | Real gravity, real thrust | Thrust in m/s², Δv shown |
| **Station-keep / Orbit-hold** | Real gravity, autopilot spends thrust | Shows the thrust it needs |
| **Cruise** | Gravity ignored; labelled **FTL (non-physical)** | Logarithmic speed scaled to the distance to the nearest surface; automatic drop-out at atmospheres, near stars, and at 10–100 r_s from black holes |
| **Jump** | Wormhole (Ellis / Morris–Thorne LUT); labelled **hypothetical** | Spawns a mouth pair (ahead of the ship and at the destination), renders the destination cubemap progressively, re-parents the camera when ℓ changes sign. Works to any bookmark, search result or coordinate |
| **Relativistic view** (toggle) | Special relativity: aberration, Doppler colour, searchlight effect | Available in Inertial mode at high speed |

**Controls (default):**
- WASD + Space/Ctrl: translate
- Mouse: look. Q/E: roll.
- Mouse wheel and number keys: speed
- F: focus/goto
- J: jump
- T: time-warp controls
- `/`: search
- Tab: HUD
- F12: screenshot

**Time warp:** pause, then from real time to ×10¹², so you can watch stars move over millions of years. Time warp never animates stellar evolution in v1.

## 6. Performance targets (my call)

- **Target:** 60 FPS at 1080p internal resolution, upscaled to native 2560×1664 with TAAU. Floor of 30 FPS in the heaviest scenes, such as low over terrain with clouds and a lens on screen.
- **GPU budget per frame (estimate, to be measured):** about 6–9 ms total.
  - background: about 1 ms
  - terrain: 2–3 ms
  - atmosphere: 0.5–1 ms
  - clouds: 0.5 ms
  - lens: 0.5 ms
  - post: 1–1.5 ms
- **Adaptive quality:** a PI controller on smoothed GPU time (per-pass timestamps from day one). It reduces, in order:
  1. render scale
  2. step counts
  3. LUT and cubemap refresh rates
  4. cloud and terrain detail
- **Battery mode:** 30 FPS.

## 7. Visual decisions (my call; all toggleable in settings)

1. **Real accretion-disk colours by default:** blue-white and lopsided, with the approaching side much brighter. A labelled "Cinematic (film-style)" toggle removes beaming and colour shift, as the film Interstellar did.
2. **Sgr A\* is honestly dim.** It has a radiatively inefficient flow, not a bright disk. A procedural "showcase" hole near the start bookmarks has a bright thin disk.
3. **Dormant black holes (Gaia BH1–3) have no disk.** They appear only as lenses.
4. **Dual exposure for stars, on by default, labelled "Eye-adapted stars".** The reference screenshot shows stars next to a sunlit planet, which a physical exposure would hide. A toggle switches to strict photometric exposure.
5. **Tonemapping:** AgX by default, with ACES and PBR-Neutral as options.
6. **The sky is rendered from data and models, never from a photo skybox.**

## 8. Verification (the core anti-failure rule)

**Headless capture**
- `universe --capture <bookmarks.ron> --out <dir>` renders each bookmark to PNG after the TAA and exposure settle.
- It writes JSON with GPU ms per pass, CPU ms and object counts.
- It must exist by the end of M0. Every gate uses it.

**The reference screenshot**
- It is kept locally at `docs/reference/owner-reference.png`. That folder is git-ignored, because the image comes from a commercial product.
- It shows a terraformed, cloudy, partly green world. Real Mars cannot honestly look like that.
- So the **M2 gate compares real Earth** (clouds, a cyclone, oceans, limb glow) **and real Mars** (MOLA terrain, thin dusty limb, polar caps, real water-ice clouds) on rendering qualities only.
- The like-for-like comparison with the reference is the **M5 gate**: a procedural world with an atmosphere, clouds and life. If a "terraformed Mars" preset is ever added, it is labelled hypothetical.

**Bookmarks**
- A fixed, versioned set, for example:
  - Earth from low orbit at dawn
  - Earth over a cyclone
  - real Mars at the reference's viewing angle
  - Mars at 1 km altitude
  - Saturn ring-plane
  - the Sun close up
  - Sgr A\* at 50 r_s
  - the showcase black hole edge-on
  - Gaia BH1 lensing its companion
  - inside a young open cluster
  - the galactic plane from 1 kpc above
  - outside the Milky Way
  - a wormhole mid-jump
  - a procedural ocean world

**Physics tests** (`cargo test`, golden values; full list in `docs/research/2026-11-20-physics.md` §9)
- Schwarzschild shadow b_c = 3√3 M within 1 px in captures.
- Deflection 0.236 rad at b = 10 r_s.
- Sun-limb deflection 1.75″.
- ISCO and photon orbits for a* ∈ {0, 0.5, 0.998}.
- Novikov–Thorne F(r_isco) = 0.
- T_H(M☉) = 6.17×10⁻⁸ K.
- Planet positions within the Standish error table against DE440s spot values.
- Earth rotation 360.9856°/day.
- Leapfrog energy bounded over 10⁴ orbits.
- S2 precession ≈ 12′ per orbit.
- Local stellar density ≈ 0.1 stars/pc³.
- Hypervelocity stars unbound.
- Rust/WGSL hash twins agree.

**Gates**
- Each milestone ends with a capture sheet: an HTML or markdown page of the PNGs plus numbers.
- An agent or report may say "done" only with a link to that sheet and the test output.
- **The lead (manager) checks every gate.** It opens and looks at every capture PNG, checks the numbers, and only then starts the next wave.
- **The owner reviews the sheets but does not block.** The owner is shown each milestone's sheet and can reject things at any time. A rejection becomes fix tasks in the next wave, and work does not stop to wait for review.

## 9. Milestones

Each milestone ends with something the owner can fly.

### M0: Foundation (lead + about 6–8 agents)

**Prep**
- Snapshot commit (§12).
- Archive the legacy Python, GLSL and old docs to `archive/legacy-2025.tar.xz`.
- Delete `target/`, `__pycache__/`, duplicate textures, `.DS_Store` and the empty `hdri/`.
- New CLAUDE.md. New `.gitignore`.

**Build**
- Workspace with crate stubs and frozen core traits.
- `uni-core` types and hash twins.
- `uni-gfx`: wgpu 30 setup, pass scheduler, timestamps, headless capture, WGSL preprocessor.
- `uni-post` v0: AgX and dither.
- `uni-nav` v0: quaternion 6-DoF camera, logarithmic speed.
- BSC5 bake.
- `uni-sky` v0: PSF star splats with Planck colours.
- egui HUD v0: FPS, GPU ms per pass, position in AU/pc, speed.
- A brute-force half-resolution Binet RK4 black-hole tracer at a test location. It also serves as ground truth for M3.

**Gate**
- No visible texels on stars.
- Shadow radius in the capture equals 3√3 M within 1 px.
- Physics unit tests pass.

### M1: Real Solar System and real sky (10–14 agents)

**Content**
- Planets and moons on real orbits at the real date, with time warp.
- Analytic ray-sphere bodies with 8k maps.
- The Sun with limb darkening and granulation.
- Saturn's rings, with shadows both ways.
- Hillaire atmospheres for Earth, Mars, Venus and Titan.
- AT-HYG m ≤ 10 stars in 3-D with proper motion.
- Milky Way band v0 from the galaxy model.
- Bloom, auto-exposure, dual star exposure.
- Target selection, goto, info cards.
- Cruise mode.

**Gate**
- Standish positions within the error table.
- Constellations recognisable.
- 60 FPS or better at 1080p internal in Solar System scenes.

### Parallel phase: M2, M3 and M4 together (peak about 25–30 agents)

**M2, planet surfaces (8–10 agents)**
- *Content:*
  - cube-sphere CDLOD terrain from GEBCO, LOLA and MOLA elevation plus procedural detail
  - normal and albedo atlases, craters
  - ocean glint
  - cloud shells with cyclones and cloud shadows
  - night lights, aerial perspective, TAA and MSAA
- *Gate:*
  - Earth captures show clouds with a cyclone, ocean glint and limb glow.
  - Mars captures show MOLA terrain, a thin dusty limb and polar caps.
  - Both have a soft terminator and no aliasing, and look good from orbit down to about 1 km altitude. Quality closer to the ground is M7.
  - 9 ms of GPU time or less at 1080p internal.

**M3, black holes and relativity (5–8 agents)**
- *Content:*
  - Schwarzschild lens LUTs, plus per-pixel Kerr–Schild tracing for spinning holes (§5.5)
  - Novikov–Thorne disk with g-factor and g⁴ beaming
  - lensed and magnified stars
  - the real black holes
  - Sgr A\* with S-stars on real orbits
  - special-relativistic view
  - ship geodesics near holes
  - Hawking, time-dilation and tidal HUD readouts, plus the micro black hole sandbox
- *Gate:*
  - LUT within 1 px of the M0 tracer
  - the approaching side of the disk is the brighter one
  - S2 precession test passes

**M4, procedural Milky Way (8–12 agents)**
- *Content:*
  - octree-streamed stars with IMF and stellar-evolution properties, epicycle motion and binaries
  - young blue open clusters
  - remnants
  - S5-HVS1 and procedural hypervelocity and runaway stars that move correctly under time warp
  - the band generated from the same model as the stars
  - the Galaxy viewable from outside
  - HII regions and dust v1
  - globular clusters
  - catalogue merge without duplicates
- *Gate:*
  - density and binary-fraction statistics
  - hypervelocity stars unbound
  - golden sector hashes
  - bounded star count and CPU time

### M5: Procedural planetary systems (8–10 agents)
- *Content:*
  - fly to any star's planets
  - real exoplanet systems completed procedurally
  - physics-driven atmosphere, colour, ice, oceans and biomes
  - gas-giant banding
  - moons, rings, rogue planets
- *Gate:* occurrence statistics, plus a variety sheet. The sheet includes a procedural world with an atmosphere, clouds, storms and life, compared side by side with the owner's reference at the same quality bar.

### M6: Travel and the deep universe (6–8 agents)
- *Content:*
  - wormhole jumps to anywhere
  - search and bookmarks
  - Local Group and nearby galaxies as volumes or impostors; Andromeda enterable with the same pipeline and different parameters
  - large-scale structure
  - fast Kerr (LUTs or analytic paths)
  - neutron stars and pulsars
  - a cluster n-body sandbox

### M7: Hardening (continuous from M2)
- Adaptive quality controller.
- TAAU to native resolution.
- Battery mode.
- Volumetric clouds near surfaces.
- Erosion.
- Analytic Kerr path.
- Performance passes.

## 10. Team structure and process (my call)

**Ownership:** one agent per module directory, at most about 3 agents per crate, each with clear file ownership.

| Team | Owns |
|---|---|
| **Lead / integrator** (the "manager") | Workspace `Cargo.toml`, `uni-app`, CLAUDE.md, interface changes, merge order. Runs the full build and capture after every wave, looks at the captures, and sends regressions back to their owners. |
| Core & sim | `uni-core`, `uni-sim`, `uni-physics` |
| GFX platform | `uni-gfx`, `uni-post` |
| Sky | `uni-sky` |
| Planet & atmosphere | `uni-planet`, `uni-atmo` |
| Relativity | `uni-rel` |
| Data | `uni-data`, `tools/bake`, the manifest, ATTRIBUTION.md. The only team that downloads. |
| Procgen | `uni-procgen` |
| Nav / UI | `uni-nav`, `uni-ui` |
| Verify / perf | `tools/verify`, golden sets, release builds, the capture queue |

**Build and disk rules** (go into CLAUDE.md)
- **One shared target directory**, an absolute path such as `~/.cargo/shared-target`, set in `.cargo/config.toml`.
- **Slim dev profile:** `debug = "line-tables-only"`, `incremental = false`, dependencies at `opt-level = 2` with `debug = false`.
- **Release profile:** `lto = "thin"`. A separate `dist` profile uses fat LTO.
- **No worktrees.** All agents edit one checkout, each touching only the files it owns.
- **After M0, only the lead edits `uni-core` and `uni-gfx` interfaces.** Others request changes, so a half-finished core edit can't break everyone's `cargo check`.
- **What agents run:** `cargo check -p <crate>` and `cargo test -p <crate> --lib`.
- **Agents do not commit.** Parallel commits would collide on git's index lock and pick up each other's files.
- **Wave cycle.** At the end of each wave the lead:
  1. runs `cargo check --workspace` and the tests;
  2. builds the app;
  3. runs the captures and looks at them;
  4. sends failures back to the owning agents;
  5. once green, commits the wave on `universe-explorer` (one commit per crate or task) and tags milestones (`m0`, `m1`, …).
- **App runs and captures** go through one serialized queue (Verify or the lead). That covers the cargo lock, the single GPU and heat.
- **Disk watchdog:** stop builds and downloads below 1.5 GB free.

## 11. Left to the builder

- Exact LUT resolutions, step tolerances, noise functions and shader structure.
- The internal API of each crate, beyond the frozen core types and the `RenderFeature` trait.
- HUD layout and styling. It must be clean and readable, label units, and mark non-physical modes.
- Bookmark coordinates and the exact capture-sheet format.
- Choices among equivalent published models, for example the IMF form or the mass–radius fit, as long as each is cited in a code comment.

## 12. Repository housekeeping

1. **Snapshot commit** of the current tree as it is, before anything is archived or deleted. Exclude `target/`, `__pycache__/` and `.DS_Store`.
2. **`archive/legacy-2025.tar.xz`** containing:
   - all `*.py`
   - `shaders/*.vert` and `shaders/*.frag`
   - `black_hole_shader.glsl`
   - `run.sh`, `RUN_PANDA3D.sh`, `requirements.txt`
   - the legacy docs: README\*, RENDERING, IMPLEMENTATION_REPORT, PANDA3D_GUIDE, BUILD_REPORT, QUICKSTART.\*, IMPROVEMENTS, RUST_\*, FINAL_SOLUTION
   - the old CLAUDE.md
   
   Then delete the originals. They stay in git history as well.
3. **New CLAUDE.md** of about 150–250 lines covering:
   - the goal
   - the real layout
   - the precision rules
   - the physics conventions
   - the build, disk and GPU rules
   - the verification rule
   - pointers to this doc and to `docs/research/`
4. **`.gitignore`** gains `venv/`, `__pycache__/`, `.DS_Store`, `data/cache/`, large baked files and `docs/reference/`. The first four are already done.
5. **Commits:** the lead commits each wave on `universe-explorer` (§10). No pushing without the owner's OK.
