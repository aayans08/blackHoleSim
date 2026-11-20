# Technical brief: free-flight universe explorer (Rust + wgpu on an M3 MacBook Air)

Each technique is tagged **[ESSENTIAL]**, **[WORTH IT]** or **[OVERKILL]**. "Overkill" means not worth doing in an agent-built project for now.

**Starting point.** The repo was only read, not changed. It uses `wgpu = "0.19"` and `winit = "0.29"`, with one 927-line `src/main.rs` and `shaders/compute.wgsl`. The current shader's "geodesic" is not real physics: it uses a `post_newtonian_correction` term and an arbitrary `*0.1` step. `df` reports 15 GiB available on `/`.

**What the reference screenshot shows** (`/private/tmp/claude-501/-Users-aayan-Documents-GitHub-blackHoleSim/a5dea4dc-6c0c-4041-adc0-ee08923628b5/images/1.png`). It looks like an existing commercial product. Match its *quality*, not its style.
- **Planet:** about 850 px across. Most of its detail is in albedo and normal maps, not in mesh geometry.
- **Lighting:** a soft terminator from multiple scattering, and a warm glow on the lit limb.
- **Surface:** 2D-looking cloud swirls, including a cyclone, plus basins and a canyon.
- **Sky:** a dense star field of colored, sub-pixel-crisp stars. Stars are visible next to a sunlit planet, which needs separate star exposure (see §6). There is also a small blue young cluster and a faint brown dust band.

---

## 1. Precision at astronomical scale

**[ESSENTIAL] Hierarchical f64 positions on the CPU, camera-relative f32 on the GPU.**
- A single f64 world frame in meters is not enough.
  - At galactic scale (1e21 m), f64 resolution is about 1e21 × 1.1e-16 ≈ 110 km.
  - At the scale of the observable universe (4e26 m) it is about 0.3 AU.
- **Option A:** store `UniversePos { cell: [i64;3], local: DVec3 }` with a cell size of about 1e13 m.
  - The local resolution is then about 1 mm.
  - The range is about 9e31 m, which is much larger than the observable universe.
- **Option B:** use a tree of reference frames: galaxy → star-system barycenter → planet, with the planet frame rotating. This fits orbital mechanics better.
- **Recommended:** a frame tree whose nodes hold (i64 cell, f64 local). This is the same idea as Bevy's `big_space`, which uses integer grid cells, nested grids that can rotate with a planet and orbit a star, and a floating origin ([big_space](https://github.com/aevyrie/big_space)). Implement it yourself on raw wgpu; don't depend on Bevy.
- **Per frame:**
  1. On the CPU, compute every object's position relative to the camera in f64.
  2. Cast that to f32 and use it as the object's model translation.
  3. Vertex positions stay local to the object or tile.
- This is the standard pattern:
  - [Frozen Fractal](https://frozenfractal.com/blog/2024/4/11/around-the-world-14-floating-the-origin/).
  - KSP2 builds planet meshes relative to the floating origin, not the planet center ([KSP dev insights](https://www.kerbalspaceprogram.com:443/news/dev-diaries-developer-insights-12-planet-tech)).
  - [ralith.com](https://ralith.com/blog/planetary-terrain/) builds matrices in f64 and rounds the combined matrix to f32. f32 resolution at one Earth radius is about 0.5 m.

**f64 is not available on the GPU.** wgpu's `SHADER_F64` is Vulkan-only, and Apple GPUs and Metal have no FP64 ([wgpu Features docs](https://docs.rs/wgpu/latest/wgpu/struct.Features.html), [Apple forums](https://developer.apple.com/forums/thread/709663)).
- **[WORTH IT, as a fallback]** GPU "relative to eye" (RTE) with a high/low float split: each position becomes two f32s, and the subtraction is done in the shader. Use it only for large geometry batches in a shared frame ([Cesium GPU RTE](https://cesium.com/blog/2015/05/26/graphics-tech-in-cesium-stack/), [Godot double emulation](https://godotengine.org/article/emulating-double-precision-gpu-render-large-worlds)).

**[ESSENTIAL] Precision of procedural noise inputs.** f32 noise evaluated at planet-scale coordinates goes blocky at fine octaves.
- Generate deep terrain levels on the CPU in f64. This also meets the owner's "use CPU and GPU together" goal.
- Or feed the GPU (integer tile lattice + local fraction) inputs.
- Use **integer hashes** (PCG or splitmix), never `fract(sin())` hashes, so results match across GPUs and between CPU and GPU ([Godot forum](https://forum.godotengine.org/t/help-with-procedural-planets/139951), [gamedev.net](https://ehe.gamedev.net/forums/topic/709083-how-to-sample-noise-beyond-data-type-size)).
- Write the noise once in Rust and once in WGSL with identical integer math, so CPU-built and GPU-built tiles meet without seams.

**[ESSENTIAL] Reversed-Z with `Depth32Float`, an infinite far plane and `CompareFunction::Greater`.**
- Reversed float Z spreads precision roughly logarithmically.
- It removes depth errors and sensitivity to where the far plane is placed ([Reed / NVIDIA "Depth Precision Visualized"](https://developer.nvidia.com/content/depth-precision-visualized)).
- With a near plane of 0.01 m, depth = n/z stays a normal float out to about 1e36 m.
- Relative depth error is about 6e-8, so at 1e7 m the error is about 0.6 m. That is enough to separate ocean, terrain and cloud layers.

**[OVERKILL / AVOID] Logarithmic depth.**
- It needs fragment-shader depth writes. Apple says changing depth in the fragment shader costs performance because the depth test must wait for the shader to run ([Apple](https://developer.apple.com/tutorials/data/documentation/metal/calculating-primitive-visibility-using-depth-testing.md)). That works against tile-based GPUs.
- Outerra's author, who invented the approach, later noted that a reversed float buffer gives similar benefits ([Outerra](https://outerra.blogspot.com/2009/08/logarithmic-z-buffer.html)).

**[WORTH IT] Separate passes by layer, not by depth range:**
1. A "background at infinity" layer drawn as directions only, with no depth: galaxies, the Milky Way cubemap, stars.
2. Scene objects with reversed-Z.
3. Volumes and transparent objects.

**[OVERKILL] Multi-frustum or "scaled space" depth partitioning** (Cesium, KSP). With reversed float Z plus camera-relative coordinates it isn't needed.

**[ESSENTIAL] Time representation.**
- f64 seconds since J2000 degrade to about 4 s resolution at 1e9 years.
- If the time warp reaches geological timescales, use `(i64 days, f64 seconds)` or i128 nanoseconds.

---

## 2. Universe structure and level of detail across ~30 orders of magnitude

**[ESSENTIAL] Deterministic sector generation.**
- Seed every node with `hash64(universe_seed, galaxy_id, level, ix, iy, iz)` using splitmix64 or xxhash. Generate content purely from the seed, so any region regenerates the same way.
- **Chain:**
  1. **Galaxy model.** Density ρ(x) from an exponential thin and thick disk, a bulge or bar, log-spiral arm modulation and a dust layer.
  2. **Star count per sector.** Integrate ρ over the sector's volume and draw a Poisson-distributed count.
  3. **Star masses.** Sample from the Kroupa 2001 initial mass function, a broken power law with slopes 0.3, 1.3 and 2.3 and breaks at 0.08 and 0.5 M☉ ([IMF package paper](https://arxiv.org/pdf/2607.26007)).
  4. **Age and metallicity** follow from where the sector is in the galaxy.
  5. **Luminosity, radius and temperature** come from age, mass and metallicity via the Hurley–Pols–Tout (2000) single-star-evolution (SSE) formulas, which are within about 5% of detailed stellar models from the main sequence through the AGB ([arXiv astro-ph/0001295](https://arxiv.org/abs/astro-ph/0001295)).
  6. **Remnants.** The same formulas give white dwarfs, neutron stars and black holes.
- Elite's Stellar Forge used a similar galaxy-scale mass and age distribution, seeded from one top-down galaxy map ([80.lv](https://80.lv/articles/generating-the-universe-in-elite-dangerous/)). Use only the concept, not their data or look.

**[ESSENTIAL] A magnitude-space level-of-detail octree** (Gaia Sky's MS-LOD: bright stars in high octree levels, faint stars deeper, nodes culled by solid angle) ([Sagristà et al. 2019](https://vcg.iwr.uni-heidelberg.de/static/publications/Sagrista2019gaiaSky.pdf), [docs](https://gaia.ari.uni-heidelberg.de/gaiasky/docs/3.5.3/LOD-catalogs.html)).
- Concretely: store the brightest absolute magnitude M_min in each node.
- Descend into a node only if `M_min + 5·log10(d_min/10pc)` is brighter than the current limiting magnitude.
- This bounds the number of stars on screen at about 1e5–5e5 at any location.
- Generate and stream sectors on worker threads (rayon), and upload them with `queue.write_buffer`.

**Merging real catalog stars with procedural ones [ESSENTIAL].**
- Use AT-HYG: Tycho-2 + Gaia DR3 + HYG, over 2.5M stars with Gaia distances and velocities, licensed **CC BY-SA 4.0** ([Codeberg](https://codeberg.org/astronexus/athyg)). It has smaller subsets, including a "HYGlike" drop-in.
- Convert it to a packed binary of about 22 bytes per star:
  - position as f32×3 in parsecs from the Sun,
  - V magnitude as f16,
  - BP−RP color as f16,
  - velocity as f16×3.
- That is about 55 MB for 2.5M stars, or about 3 MB for a ~120k bright subset. Both fit the disk limit.
- Inside the region where the catalog is complete, reject any procedural star that would have been bright enough, as seen from Sol, to appear in the catalog. This avoids duplicates.

**Other real objects:**
- [OpenNGC](https://github.com/mattiaverga/OpenNGC) (CC-BY-SA-4.0) for NGC/IC galaxies, nebulae and clusters.
- JPL approximate Keplerian elements for the planets (small).
- Known black holes such as Sgr A*, M87* and Gaia BH1–3, entered as a hand-curated table.

**[ESSENTIAL] Objects must move like real ones.**
- Stars carry 3D velocities and are integrated in a Milky Way potential. MWPotential2014 uses a power-law bulge cut off at 1.9 kpc, a Miyamoto–Nagai disk (a = 3 kpc, b = 0.28 kpc, 6.8e10 M☉) and an NFW halo with r_s = 16 kpc ([parameters as quoted in later papers](https://arxiv.org/pdf/1804.04995); verify against galpy's source).
- **Hypervelocity stars:** use the Hills mechanism, where a binary is disrupted by Sgr A* and one star is ejected. Example: S5-HVS1, ejected at about 1800 km/s about 4.8 Myr ago ([Koposov et al.](https://arxiv.org/abs/1907.11725)).
- **Planets and moons:** Kepler orbits "on rails", which are exact and cheap.
- **N-body integration [WORTH IT]:** use a symplectic integrator only for interacting procedural systems and for the ship.

**Level of detail by scale:**

| Scale | Representation |
|---|---|
| Galaxies at cosmic scale | Impostors |
| A galaxy | Volume model plus star octree |
| Nearby stars | Point sprites |
| Planets smaller than ~2 px | Points whose brightness comes from albedo × Lambert phase function |
| Planets ~2–64 px | Analytic ray-sphere with atmosphere, no mesh |
| Planets larger than that | Quadtree terrain |

**[WORTH IT] Impostors for distant galaxies.** Render each procedural galaxy once into a 64–128 px atlas tile. Draw it as an oriented quad in the galaxy's disk plane, which is 3D-correct for thin disks. Ellipticals become Sérsic-profile sprites.

**[OVERKILL] A cosmic-web simulation** (for example the Zel'dovich approximation). A noise-based filament/void density field is enough.

**Star point sprites [ESSENTIAL]:**
- **Brightness:** illuminance E = 10^(−0.4(m + 14.18)) lux. For the Sun (m = −26.74) this gives about 1e5 lux.
- **Primitive:** WebGPU points are always 1 px, so draw instanced quads with additive blending into the HDR target.
- **Shape and size:** the point-spread function (PSF) is a Gaussian, σ ≈ 0.6–0.8 px, integrated over each pixel with `erf`. Clamp the minimum size and keep total flux constant. This stops sub-pixel stars from flickering or looking pixelated in motion.
- **Bright stars [WORTH IT]:** add glare halos based on the physically based glare PSF ([Spencer et al. 1995](https://w1.graphics.cornell.edu/pubs/1995/SSZG95.pdf)) and the star treatment in the night-sky model ([Jensen et al. 2001](https://graphics.ucsd.edu/~henrik/papers/nightsky/)).
- **[WORTH IT, later]** Splat stars in a compute pass with u32 fixed-point atomics, if instanced quads become a bottleneck.

---

## 3. Planets at the quality of the reference screenshot

**[ESSENTIAL] Cube-sphere quadtree terrain.**
- Six faces. Use a warped (approximately equal-area) cube-to-sphere mapping ([Zucker & Higashi 2018, JCGT](https://jcgt.org/published/0007/02/01/paper.pdf)) so tiles stay even in size.
- Tiles are 33² or 65² vertices.
- Use CDLOD distance-based geomorphing between levels ([Strugar](https://github.com/huiwy/CDLOD)), or skirts to hide cracks.
- **Per tile:** a f64 origin on the CPU; positions stored as f32 relative to the tile.
- **Key to the screenshot look:** give each tile a 129²–257² normal map and albedo texture at higher resolution than its mesh, in an atlas. At orbital distance, nearly all the visible detail is texture.

**Terrain detail:**
- **[ESSENTIAL]** fBm, ridged multifractal noise and domain warping, plus biome and albedo chosen from latitude, altitude, temperature and moisture.
- **[ESSENTIAL]** Craters sampled from a power-law size-frequency distribution by inverse-CDF sampling ([A&A 2022](https://www.aanda.org/articles/aa/full_html/2022/03/aa40709-21/aa40709-21.html)). Place them in per-cell jittered positions; a cube-sphere Poisson sampler works well ([Zucker & Higashi](https://jcgt.org/published/0007/02/01/paper.pdf)).
- **[WORTH IT]** A noise-based erosion filter evaluated per point ([runevision, Mar 2026](https://blog.runevision.com/2026/03/)), which gives branching gullies with no simulation. Its code is MPL-2.0, so implement it from the description to respect the owner's no-copy rule.
- **[OVERKILL]** Hydraulic erosion simulation.

**[ESSENTIAL] Atmosphere: Hillaire 2020 lookup tables (LUTs)** ([paper](https://diglib.eg.org/handle/10.1111/cgf14050), [MIT sample](https://github.com/sebh/UnrealEngineSkyAtmosphere)).
- **Transmittance LUT.**
- **Multiple-scattering LUT:** a 2D table indexed by altitude and sun angle. It approximates infinitely many scattering orders.
- **Sky-view LUT:** latitude/longitude layout, with a non-linear latitude mapping so the horizon keeps its detail. It can hold several suns.
- **Aerial-perspective volume (froxels).**
- **Space views:** turn on the per-pixel transmittance option. This gives the terminator and limb glow seen in the reference image.
- **Cost:** on a GTX 1080 at 720p, the sky plus atmosphere costs 0.14 ms and the total is 0.31 ms (from the SIGGRAPH 2020 slides). A third-party implementation measured 0.036 ms for a 64×64 sky-view LUT on a GTX 1060 ([AirGuanZ](https://www.Github.com/AirGuanZ/AtmosphereRenderer)).
- **LUT sizes (unverified, paper defaults from memory):** transmittance about 256×64, multiple scattering 32×32, sky-view about 192×108, aerial perspective 32³.
- Rebuild the LUTs only for the nearest one or two planets.
- **[WORTH IT, as a reference only]** Bruneton's precomputed scattering ([new implementation, BSD](https://ebruneton.github.io/precomputed_atmospheric_scattering)). It is heavier to rebuild per planet.

**Clouds:**
- **[ESSENTIAL] A 2D cloud shell for the orbital view:**
  - procedural coverage animated by flow or curl noise, with vortex warps for cyclones;
  - lit through the transmittance LUT;
  - shadows on the ground sampled along the sun direction;
  - 2–3 layers for parallax.
  - This matches the reference image.
- **[WORTH IT, later]** Volumetric clouds near the surface in the Nubis style, about 2 ms on PS4 ([Schneider 2015](https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn)).
- **[OVERKILL]** Fluid-simulated clouds ([Nubis Cubed](https://www.guerrilla-games.com/read/nubis-cubed)).

**Oceans:**
- **[ESSENTIAL]** A sea-level sphere with Fresnel and Cox–Munk sun glint. Slope variance grows linearly with wind speed ([Cox–Munk 1954](https://data-ww3.ifremer.fr/BIB/Cox_Munk_JOSA1954.pdf)). Clamp the glint near the horizon.
- **[WORTH IT]** Level-of-detail transition from geometry to a normal distribution to a BRDF ([Bruneton, Neyret, Holzschuch 2010](https://hal.archives-ouvertes.fr/inria-00443630v3)).
- **[WORTH IT, later]** FFT waves (Tessendorf) near the surface.

**[ESSENTIAL] Rings.** Use an analytic ray-plane intersection with a 1D radial density and color texture.
- Shadows both ways: ring onto planet (ray-plane test) and planet onto ring (ray-sphere test).
- A forward-scattering Henyey–Greenstein phase function.

**[ESSENTIAL] Night lights.** An emissive map multiplied by the night factor and by atmospheric transmittance.
- Earth: NASA Black Marble (public domain), 2–4k BC7.
- Other worlds: procedural, driven by a habitability field.

---

## 4. Stars (seen up close)

- **[ESSENTIAL] Blackbody color to sRGB.**
  1. Integrate Planck × CIE 1931 color-matching functions to get XYZ. The analytic fits by [Wyman, Sloan & Shirley 2013](https://research.nvidia.com/labs/rtr/publication/wyman2013simple/) take about 10 lines of code.
  2. Convert to linear Rec.709/sRGB (D65).
  3. Store chromaticity in a 1D LUT over 1,000–50,000 K. Mitchell Charity's table is a cross-check ([Planckian locus refs](https://dweb.link/ipfs/bafybeiemxf5abjwjbikoz4mc3a3dla6ual3jsgpdr4cjr3oz3evfyavhwq/wiki/Planckian_locus.html)).
  4. Compute absolute luminance separately from L = 4πR²σT⁴.
- **[ESSENTIAL] Limb darkening:** quadratic law I(μ)/I(1) = 1 − a(1−μ) − b(1−μ)².
  - Take coefficients from Claret or MPS tables by temperature.
  - The Sun reference is the 5th-order polynomial of Neckel & Labs ([refs](https://ar5iv.labs.arxiv.org/html/2206.06641)).
  - A linear coefficient of about 0.6 in V for Sun-like stars is approximate and unverified.
- **[WORTH IT] Granulation:** animated cellular (Voronoi) noise with cells about 1,000 km across and lifetimes of minutes, plus sunspots and faculae.
- **Corona and glare:**
  - **[ESSENTIAL]** Glare comes from the bloom pyramid plus a per-star glare sprite whose energy scales with flux. The real corona is about 1e-6 of disk brightness, so what viewers see is mostly the PSF.
  - **[OVERKILL]** FFT convolution with a PSF, and spectral rendering.

---

## 5. Galaxies, nebulae and clusters

**[ESSENTIAL] One galaxy model, used from both inside and outside.** The sky band must change as you fly, so a photo of the Milky Way won't work.
- **Outside:** ray-march emission (stars plus HII regions, H-alpha) and dust absorption at quarter resolution with temporal accumulation. The alternative is the Gaia Sky approach: generate millions of particles per channel (stars, HII, gas, dust, bulge) and render the gas and dust channels at lower resolution ([Gaia Sky procedural galaxies](https://gaia.ari.uni-heidelberg.de/gaiasky/docs/master/Procedural-galaxies.html), [preview](https://gaiasky.space/news/2025/galaxy-generation/)). Particles are cheap on M3 because they blend additively.
- **Ray-marching dust and H-alpha:** the VIRUP paper covers GPU ray marching of simulated galaxy dust and H-alpha emission ([VIRUP](https://arxiv.org/pdf/2110.04308)).
- **Inside:** render the same density model into a cubemap of about 6×256² (diffuse band plus dust extinction) from the camera's position.
  - Re-render it progressively, one face per frame, when the camera moves more than a threshold, for example about 1% of the distance to the nearest density feature.
  - Draw catalog and procedural point stars on top.
- **[ESSENTIAL] Star clusters.**
  - **Profiles:** Plummer or King spatial profiles.
  - **Star properties:** initial-mass-function sampling plus a single age, run through the Hurley formulas.
  - **Look:** young open clusters come out blue, like the one in the reference image. Globular clusters get red giants and blue horizontal-branch stars.
- **[WORTH IT] Nebulae:** low-resolution 3D noise volumes, ray-marched with emission (H-alpha, OIII) and absorption only when nearby. Far away, use impostors.

---

## 6. HDR pipeline

**[ESSENTIAL] Render targets and exposure.** Use `Rgba16Float` targets in physical units, with exposure in EV100.

**[ESSENTIAL] Separate star exposure, and why.**
- Sunlight is about 1e5 lux. A magnitude-6 star is about 8e-9 lux. That is a 1e13 ratio.
- Physically correct exposure would make stars invisible next to a lit planet, but the reference image shows both.
- So map stars through their own limiting-magnitude curve, tied to exposure but clamped, and composite them before tone mapping. This is an artistic choice and should be documented as one.

**[ESSENTIAL] Bloom.** Use an energy-conserving pyramid: a 13-tap downsample and 3×3 tent upsample ([Jimenez 2014](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/)). Mix it in with a small lerp instead of a brightness threshold ([LearnOpenGL physically based bloom](https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom)).

**[ESSENTIAL] Auto-exposure.** Build a log-luminance histogram in a compute shader, ignore the darkest and brightest percentiles, and adapt over time ([bruop](https://bruop.github.io/exposure)).

**Tone mapping:**
- **[ESSENTIAL] AgX by default.** It desaturates very bright suns and stars gracefully. Implement it from the reference definition ([sobotka/AgX via AgXc notes](https://www.github.com/MrLixm/AgXc)).
- **[WORTH IT] Khronos PBR Neutral as an option.** It maps colors 1:1 up to a threshold and compresses only the highlights, so planet albedo stays faithful. The spec is about 13 lines of GLSL ([gamedev.net](https://gamedev.net/news/khronos-pbr-neutral-tone-mapper-released-for-true-to-life-color-rendering-of-3d-objects-r2733)).
- Bevy ships AgX and PBR Neutral, which is useful as a behavioral reference ([Bevy example](https://bevy.org/examples/3d-rendering/tonemapping/)).
- ACES has hue shifts and is not recommended as the default.

**[ESSENTIAL] Dithering.** Add blue-noise dither before the 8-bit swapchain. Dark space gradients and the faint dust band band visibly without it, which reads as "pixelated".

**Anti-aliasing:**
- **[ESSENTIAL] TAA.** Use Halton jitter and motion vectors computed from f64 camera deltas. Reject stale history with YCoCg clipping or variance clipping ([Karis 2014 / Salvi 2016 summary](https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf)). TAA also makes jittered atmosphere and cloud ray marching converge.
- **Stars must skip TAA.** Composite them after TAA at native resolution; otherwise TAA smears and ghosts them.
- **[WORTH IT] 4× MSAA for meshes.** It is cheap on Apple tile-based GPUs because the resolve happens in tile memory with no extra bandwidth ([Apple](https://developer.apple.com/tutorials/data/documentation/metal/improving-edge-rendering-quality-with-multisample-antialiasing-msaa.md)).
- **[WORTH IT] TAAU** (temporal upsampling) for dynamic resolution.
- **[OVERKILL]** MetalFX, which would need wgpu-hal interop with raw Metal. FSR2 is open source, but write your own TAAU to respect the no-copy rule.

---

## 7. Wormhole travel, and black holes (same technique)

**Wormhole metric** (Dneg/Interstellar, [James et al. 2015](https://arxiv.org/abs/1502.03809)):
- ds² = −dt² + dℓ² + r(ℓ)²dΩ²
- r = ρ for |ℓ| ≤ a.
- Outside, r = ρ + 𝓜[x·atan x − ½ln(1+x²)], with x = 2(|ℓ|−a)/(π𝓜).
- Lensing width 𝓦 ≈ 1.43 𝓜.
- **Ray equations:**
  - dℓ/dt = p_ℓ, dθ/dt = p_θ/r², dφ/dt = b/(r²sin²θ)
  - dp_ℓ/dt = B²(dr/dℓ)/r³, dp_θ/dt = (b²/r²)·cosθ/sin³θ
  - b and B² are conserved.
- **[ESSENTIAL] A 1D or 2D lookup table instead of per-pixel integration.**
  - The geometry is spherically symmetric, so each ray stays in one plane, and its fate depends only on (ℓ_camera, ψ), where ψ is the angle to the radial direction and B = r(ℓ_c)·sin ψ.
  - Each frame, integrate about 2,048 rays in f64 on the CPU with rayon. Store, for each ray, which side it exits on and the angle swept. This takes well under 1 ms.
  - Or precompute a (ℓ_c, ψ) table of about 256×1024 once per wormhole.
  - Per pixel: find the ray's plane → look up the table → rotate to the final direction → sample this side's sky or the far side's sky.
- **[ESSENTIAL] A seamless jump.**
  - Render the destination's environment (stars, galaxy cubemap, nearby bodies) into a cubemap from the far mouth. Refresh it progressively, at a resolution that scales with how big the mouth is on screen.
  - When ℓ_c changes sign, re-parent the camera to the destination mouth's frame. With the frame tree this is trivial.
  - The table is symmetric in ℓ, so the image stays continuous.
  - For arbitrary long jumps, spawn a pair of mouths: one ahead of the ship and one at the destination.
- **Choosing the shape.** The paper reports that short wormholes look like a zoom with no parallax, and long ones look like generic tunnels. Nolan used 2a = 0.01ρ and 𝓦 = 0.05ρ. Choose a moderate a/ρ and 𝓦, and add parallax cues inside the throat, such as particles.
- **Physics honesty.** Label wormholes in the UI as hypothetical: they need exotic matter.
- **[OVERKILL]** The ray-bundle and geodesic-deviation filtering used by DNGR.

**Black holes [ESSENTIAL]: replace the current fake shader with Bruneton's precomputed beam tracing** ([arXiv 2010.08735](https://arxiv.org/pdf/2010.08735), BSD-3).
- Two small tables (512×512 deflection and 64×32 disk crossings, both RG32F) give intersections in constant time per pixel. They precompute in about 11 s on a CPU.
- About 150 fps at 1080p on a GTX 960.
- For comparison, ray marching needs about 1,000 steps for the same accuracy (about 45 fps), and at 25 steps stars are degrees off.
- Stars stay point-like with a hand-filtered cubemap that stores one star per texel.
- Doppler and beaming: I_ν/ν³ is invariant, so a blackbody shifts to T_obs = g·T.
- **Accretion disk:** thin-disk profile T ∝ r^(−3/4)(1−√(r_in/r))^(1/4), with r_in at the ISCO.
- **For a quick check of an integrator:** the exact Schwarzschild photon path follows from Binet's equation u″+u = 1.5 r_s u². That is equivalent to a Cartesian acceleration a = −1.5 r_s h² x/|x|⁵, with h = |x×v|.
- **[WORTH IT] Hawking radiation as a data overlay, not a visual.**
  - T = 6e-8 K × (M☉/M), so every astrophysical black hole is colder than the CMB and appears dark.
  - Evaporation takes about 2e67 years × (M/M☉)³ ([refs](https://ar5iv.labs.arxiv.org/html/1904.02010)).
  - Render it visibly only for hypothetical primordial black holes.
- **[OVERKILL] Kerr (spinning) black holes** need 4D tables.

---

## 8. Performance on an M3 MacBook Air with wgpu/Metal

**Hardware.**
- The 10-core M3 GPU peaks at about 3.55–3.6 TFLOPS FP32 ([Notebookcheck](https://www.notebookcheck.net/Apple-M3-10-Core-GPU-Benchmarks-and-Specs.765636.0.html)). The 8-core version is about 20% lower.
- The Air has no fan and throttles under sustained load ([Wccftech](https://wccftech.com/m3-macbook-air-severely-throttles-under-load/)). Plan on about 60–70% of peak sustained.
- Unified memory bandwidth is about 100 GB/s. This figure is from memory and unverified.

**Arithmetic.**
- At 1440p (3.7 MP) and 60 fps, peak compute is about 16k FLOPs per pixel per frame, and realistically about 5–8k.
- So one full-screen loop of about 100 cheap steps is fine. Nested ray marches (for example a light march inside a view march) are not; they must go through LUTs.

**Estimated GPU budget at 1080p internal resolution (estimates, not measured):**

| Item | Estimate |
|---|---|
| Background: galaxy cubemap plus 1e5–5e5 stars | ~1 ms |
| Terrain, 0.5–1.5M triangles | 2–3 ms |
| Atmosphere LUTs and application, scaled from 0.31 ms on a GTX 1080 | ~0.5–1 ms |
| 2D cloud layer | ~0.5 ms |
| Black hole or wormhole lensing via LUT | ~0.5 ms |
| Bloom, exposure, tone mapping and TAA | ~1–1.5 ms |
| **Total** | **~6–9 ms** |

- **60 fps at 1080p internal resolution is realistic.**
- **1440p internal is about 30–45 fps in heavy scenes.**
- The native panel resolution (2560×1664 on the 13", 2880×1864 on the 15") should be reached by TAAU upscaling, not by rendering natively.

**[ESSENTIAL] Adaptive quality.**
- **Timing:** measure GPU time per pass with `TIMESTAMP_QUERY`, which works on Metal at pass boundaries. Timestamps *inside* passes are not available on Apple GPUs ([wgpu Features](https://docs.rs/wgpu/latest/wgpu/struct.Features.html)).
- **Controller:** a PI controller on a smoothed frame time.
- **What to scale down, in order:**
  1. render scale, 0.5–1.0 with TAAU;
  2. ray-march step counts;
  3. LUT and cubemap refresh rates (spread across frames);
  4. cloud and terrain detail.
- Track sustained thermal state. Use `Fifo` vsync, and offer a 30 fps "battery" mode.

**Apple tile-based GPU practices.**
- **[ESSENTIAL]** Never write depth from the fragment shader.
- **[ESSENTIAL]** Use `LoadOp::Clear` and `StoreOp::Discard` for attachments that won't be read later.
- **[WORTH IT]** Mark MSAA and depth attachments as transient (memoryless). wgpu has `TextureUsages::TRANSIENT`; check how your version maps it to Metal ([Dawn notes](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/dawn/features/transient_attachments.md)).
- **[WORTH IT]** `SHADER_F16` (supported on Metal) for post-processing and color math. Never use it for positions.

**[ESSENTIAL] Upgrade early.** Move from wgpu 0.19 to the current major version (context7 lists v30.0.1) and from winit 0.29 to the current release, before writing more code. Ray queries are still Vulkan-only experimental in wgpu, so **[OVERKILL]** hardware ray tracing on Metal.

**Disk budget [ESSENTIAL].**
- Cargo settings:
  - `[profile.dev] debug = "line-tables-only"`
  - `[profile.dev.package."*"] opt-level = 3`
  - one shared `CARGO_TARGET_DIR`
  - `CARGO_INCREMENTAL=0` when disk is tight
  - `cargo clean` between big upgrades
- Assets: generate procedurally by default; store the catalog as about 3–55 MB of binary; keep the few real textures at 2–4k BC7 (BC is supported on Apple Silicon Macs).

**Use the CPU and GPU together (M3: 4 performance + 4 efficiency cores).**
- **CPU (rayon jobs):**
  - f64 orbits and frame transforms
  - octree sector generation
  - deep-level terrain tiles in f64
  - wormhole and black-hole tables
  - catalog streaming and culling
- **GPU:**
  - all rendering
  - near-camera noise detail
  - atmosphere LUTs
  - the galaxy cubemap