# Attribution

Every third-party data file, texture and published algorithm used by this project is credited here.
Downloaded data files are also recorded, with URL and SHA-256, in `data/manifest.toml`.

<!-- M1 Wave A sections (pre-split by Task A0b, review I8): each Data task writes only between its own
opening and closing comment markers, named after the task (A7 and A9 under Data, A8 under Textures; A9 may
open a Fonts heading inside its own section). -->

## Data

**Yale Bright Star Catalogue, 5th Revised Ed.** (`data/baked/bsc5.bin`). Hoffleit, D. & Warren Jr., W. H. 1991,
*The Bright Star Catalogue, 5th Revised Ed. (Preliminary Version)*, Astronomical Data Center, NSSDC/ADC; CDS
catalogue [V/50](https://cdsarc.cds.unistra.fr/viz-bin/cat/V/50), obtained via CDS (VizieR mirror at the CfA). Licence:
public (CDS V/50). The baked file keeps, for 9096 of the 9110 entries, the J2000 direction, V magnitude and HR number,
plus an effective temperature derived from B−V with Ballesteros (2012, EPL 97, 34008). The 14 entries without a
position (novae and non-stellar objects) are omitted.

<!-- A7 -->
**AT-HYG v4.0** (`data/baked/athyg_m10.bin`, `data/baked/athyg_names.ron`). The Augmented Tycho-HYG star catalogue, version
4.0, subset `athyg_40_reduced_m10` (stars to V/VT 10 plus all stars within 100 light years), by David Nash
(astronexus), [codeberg.org/astronexus/athyg](https://codeberg.org/astronexus/athyg). Licence: [Creative Commons
Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
**`athyg_m10.bin` and `athyg_names.ron` are derived data and are themselves licensed CC BY-SA 4.0** (ShareAlike): anyone
who redistributes them, or data derived from them, must do so under the same licence with this credit. Changes made
by `uni-bake athyg`: rows without a usable distance and the Sun are dropped; positions are recomputed from RA, Dec and
distance; Tycho magnitudes are converted to Johnson V and B−V (red dwarfs take the B−V of the main-sequence dwarf
sequence at their absolute magnitude); 23 of the brightest stars take their Johnson V and B−V
from the Yale Bright Star Catalogue (credited above); stars without a colour index get a B−V from their spectral type;
velocities and colours are stored as 16-bit floats; names are reformatted (Greek Bayer letters).

AT-HYG is built from these catalogues, which are credited as its sources:
- ESA Gaia DR3: this work has made use of data from the European Space Agency (ESA) mission
  *Gaia* ([cosmos.esa.int/gaia](https://www.cosmos.esa.int/gaia)), processed by the *Gaia* Data Processing and Analysis
  Consortium (DPAC, [cosmos.esa.int/web/gaia/dpac/consortium](https://www.cosmos.esa.int/web/gaia/dpac/consortium)).
  Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the
  *Gaia* Multilateral Agreement.
- Tycho-2: Høg, E. et al. 2000, A&A 355, L27 (CDS I/259). Hipparcos and Tycho: ESA 1997, *The Hipparcos and Tycho
  Catalogues*, ESA SP-1200.
- The Yale Bright Star Catalogue, the Gliese-Jahreiss catalogue of nearby stars, and the HYG database
  ([codeberg.org/astronexus/hyg](https://codeberg.org/astronexus/hyg), CC BY-SA 4.0), whose proper names follow the IAU
  Working Group on Star Names.

**NASA Exoplanet Archive** (`data/baked/exoplanets.ron`), Planetary Systems Composite Parameters table (`pscomppars`),
queried 2026-11-22. This research has made use of the NASA Exoplanet Archive, which is operated by the California
Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet
Exploration Program. Archive paper: Christiansen et al. (2025), *The Planetary Science Journal* (the NASA Exoplanet
Archive's overview paper, which replaces Akeson et al. 2013). Each planet's parameters come from the publication the
archive cites for it.

Published values used by the star bake (`tools/bake/src/athyg.rs`):
- Tycho → Johnson photometry, ESA 1997, *The Hipparcos and Tycho Catalogues*, ESA SP-1200 vol. 1, §1.3 Appendix 4
  (M. Grenon), Table 1.3.4, interpolated between its nodes and extrapolated with its end values as ESA prescribes
  (the simple form V = VT − 0.090 (BT−VT), B−V = 0.850 (BT−VT) is eq. 1.3.20 and §2.2 eqs. 2.2.1–2.2.2).
- Mean main-sequence B−V by spectral type, and B−V against absolute magnitude for the red dwarfs that ESA's
  transformation does not cover: Pecaut, M. J. & Mamajek, E. E. 2013, ApJS 208, 9, Table 5, in E. Mamajek's
  version 2022.04.16 ([EEM_dwarf_UBVIJHK_colors_Teff.txt](https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)).

Used to check the red-dwarf colours, not read by the bake: Gliese, W. & Jahreiss, H. 1991, *Preliminary Version of
the Third Catalogue of Nearby Stars* (CNS3), CDS catalogue V/70A.
<!-- /A7 -->

<!-- A9 -->
**JPL Horizons spot values** (`data/golden/horizons_spots.ron`, `data/golden/sky_spots.ron`; test goldens only,
never drawn). Geometric state vectors and geocentric observer tables, retrieved through the Horizons API on
2026-11-22. Citation, as the Horizons manual asks: Giorgini, JD and JPL Solar System Dynamics Group, NASA/JPL
Horizons On-Line Ephemeris System, <https://ssd.jpl.nasa.gov/horizons/>, data retrieved 2026-Nov-22 (system
described in Giorgini et al. 1996, "JPL's On-Line Solar System Data Service", BAAS 28(3), 1158). Developed by the
Jet Propulsion Laboratory (Solar System Dynamics Group), California Institute of Technology, under contract with
NASA. The underlying ephemerides, as each answer names them: DE441 for the Sun, the planets, Pluto and the Moon
(Park, R. S., Folkner, W. M., Williams, J. G. & Boggs, D. H. 2021, "The JPL Planetary and Lunar Ephemerides DE440
and DE441", AJ 161, 105); the JPL satellite ephemerides mar099, jup365, sat441, ura184, nep098 and plu060 for the
other moons; the JPL small-body orbit solutions JPL#75 (1P/Halley), JPL#48 (1 Ceres) and JPL#31 (C/2020 F3).
Licence: public JPL Solar System Dynamics data; the Horizons manual states no terms beyond the citation.

**JPL Small-Body Database** (`data/baked/asteroids.bin`, `data/baked/asteroid_names.txt`: every asteroid with
H ≤ 14 whose orbit is closed and complete, 33 678 objects; `data/baked/comets.ron`: 506 numbered periodic comets
without fragments, plus C/1995 O1, C/2020 F3, C/1996 B2, C/2006 P1, C/2023 A3 and C/2011 L4). Osculating
orbital elements and physical parameters retrieved through the SBDB Query API
(<https://ssd-api.jpl.nasa.gov/sbdb_query.api>) on 2026-11-22. Credit: JPL Small-Body Database, Solar System
Dynamics Group, Jet Propulsion Laboratory, California Institute of Technology, under contract with NASA
(<https://ssd.jpl.nasa.gov/tools/sbdb_query.html>). The orbit solutions are JPL's and the Minor Planet Center's, as
SBDB records per object. Licence: public JPL Solar System Dynamics data (no stated terms).
`data/baked/comets_2026.ron` is derived from `comets.ron`: its orbital elements are re-osculated to 2026-11-20 TDB by this project's planet-perturbed integrator (`uni-bake comets-epoch`), so they are ours, not JPL's; names, ids and magnitudes are SBDB's.

**JPL planetary-satellite mean elements and physical parameters** (`data/baked/sat_elements.ron`, 460 moons).
"Planetary Satellite Mean Elements" (<https://ssd.jpl.nasa.gov/sats/elem/>) and "Planetary Satellite Physical
Parameters" (<https://ssd.jpl.nasa.gov/sats/phys_par/>), JPL Solar System Dynamics Group, retrieved 2026-11-22.
They are fitted to the JPL satellite ephemerides that each row names (MAR099, JUP365, JUP347–349, SAT415–480,
URA117/182/184, NEP098/104/105, PLU060; DE405/LE405 for the Moon) and cite, among others, Jacobson (2009, AJ 137,
4322), Brozović & Jacobson (2009, AJ 137, 3834; 2024, AJ 167, 256), Jacobson et al. (2012, AJ 144, 132), Nimmo et
al. (2017, Icarus 287, 12) and Archinal et al. (2018, CMDA 130:22). The reference-plane poles that JPL does not
tabulate are the IAU WGCCRE 2015 planet poles (Archinal et al. 2018, via NAIF `pck00011.tpc`) and the J2000
ecliptic pole with the IAU 1976 obliquity 84381.448″ (Lieske et al. 1977, A&A 58, 1). Licence: public JPL Solar
System Dynamics data (no stated terms).

**NASA NSSDCA fact sheets** (`data/baked/body_facts.ron`: the Sun, the planets, Pluto, the Moon and 21 moons,
hand-curated). Williams, D. R., Planetary Fact Sheets, NASA Space Science Data Coordinated Archive (NSSDCA), NASA
Goddard Space Flight Center, <https://nssdc.gsfc.nasa.gov/planetary/factsheet/>, retrieved 2026-11-22: the
planetary fact table, the Small Worlds table and the sheets for the Sun, each planet, the Moon, Pluto and the
Jovian, Saturnian, Uranian and Neptunian satellites. Each entry's `source` lists its sheets. Licence: public NASA
data (US Government work).

### Fonts

**Inter 4.1** (`data/fonts/Inter-Light.ttf`, `Inter-Regular.ttf`, `Inter-Medium.ttf`, `Inter-SemiBold.ttf`),
copyright 2016 The Inter Project Authors (<https://github.com/rsms/inter>), from the release archive
`Inter-4.1.zip` (`extras/ttf/`), unmodified. Licence: SIL Open Font License 1.1, full text in
`data/fonts/OFL-Inter.txt`.

**IBM Plex Mono 2.5.0** (`data/fonts/IBMPlexMono-Light.ttf`, `IBMPlexMono-Regular.ttf`, `IBMPlexMono-Medium.ttf`),
copyright 2017 IBM Corp., with Reserved Font Name "Plex" (<https://github.com/IBM/plex>), from the release archive
`ibm-plex-mono.zip` of `@ibm/plex-mono@2.5.0` (`fonts/complete/ttf/`), unmodified. Licence: SIL Open Font License
1.1, full text in `data/fonts/OFL-IBMPlexMono.txt`.
<!-- /A9 -->

## Textures

Solar System Scope, 2k texture maps. Licence: [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Credit: "Solar System Scope"
([solarsystemscope.com/textures](https://www.solarsystemscope.com/textures/)). The source page says the maps
are based on NASA elevation and imagery data.

The files are unmodified 2048×1024 originals, renamed on download. Origin was checked by matching each file's MD5
against a fresh download of the original (`docs/research/2026-11-20-assets.md` §2).

| File | Original file name | SHA-256 |
|---|---|---|
| `textures/earth_day.jpg` | `2k_earth_daymap.jpg` | `767ee1dc6eb3802699bfccf6f264880f8acd0b80de3191cd24984fe279b07b7c` |
| `textures/earth_night.jpg` | `2k_earth_nightmap.jpg` | `c16fd1bc096ab91a5c5265c6ff9847c43f489f2e2ee790ccdbcbd03251cf3a5a` |
| `textures/earth_clouds.jpg` | `2k_earth_clouds.jpg` | `fffd7f68d41b37274822150e54a6ef605af1d3ec35624d9f628c3b896bfa42ed` |
| `textures/mars.jpg` | `2k_mars.jpg` | `2d187f3e77a98eaa8cea5f4cc722f633c122ef170b9e94ace6b5fb6cbc3f8e01` |
| `textures/moon.jpg` | `2k_moon.jpg` | `2764ba6535ea0481a062846ee033cc7a909dae05b31a8fd13f3e98f3a7fd92bd` |
| `textures/sun.jpg` | `2k_sun.jpg` | `ff0f076ba65e03b5ab518451bc96699325be38e3ccbdd5869ee1c00f3a0c8816` |
| `textures/stars_milky_way.jpg` | `2k_stars_milky_way.jpg` | `ec2240e6f3784d962ab81609b228bfd470e87c86d1fed5e99134d4cd3de24239` |

<!-- A8 -->
### Baked surface maps and Saturn's rings (M1)

**Solar System Scope, 8k/4k/2k texture maps** → `data/baked/maps/*.big` (git-ignored, rebuilt by `uni-bake map`;
the committed `data/baked/maps/index.ron` lists them) and `data/baked/saturn_rings.ron`. Licence:
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: "Solar System Scope"
([solarsystemscope.com/textures](https://www.solarsystemscope.com/textures/)), based on NASA elevation and imagery
data. Changes made: decoded; brightness scaled to each body's geometric albedo with every channel capped at
reflectance 1 (the index's `source` string gives the scale), except Earth's day map, scaled so its ocean has the
sea surface's measured reflectance 0.06 (Payne 1972, J. Atmos. Sci. 29, 959), and Venus's cloud-top map, reduced to
a neutral grey pattern of lower contrast with disk-averaged reflectance 0.95; box-filtered into mip chains in linear light,
block-compressed (BC7/BC4) plus 2k uncompressed copies; the ring PNG reduced to one radial row of 1024 samples with
fitted radii. Files used:
`8k_earth_daymap.jpg`, `8k_earth_clouds.jpg`, `8k_moon.jpg`, `8k_mars.jpg`, `8k_mercury.jpg`,
`4k_venus_atmosphere.jpg`, `8k_jupiter.jpg` (decodes at 4096×2048), `8k_saturn.jpg` (decodes at 4096×2048),
`8k_saturn_ring_alpha.png`, `2k_uranus.jpg`, `2k_neptune.jpg`. URLs and SHA-256 are in `data/manifest.toml` (A8).

**NASA/USGS global mosaics (public domain)**, baked the same way (URLs, SHA-256 and catalogue pages in
`data/manifest.toml`), from the USGS Astrogeology Science Center
([astrogeology.usgs.gov](https://astrogeology.usgs.gov/)):
- Io: Io Voyager–Galileo SSI Global Mosaic 1 km (1024-px browse). NASA/JPL Galileo SSI and Voyager 1; USGS.
- Europa: Europa Voyager–Galileo SSI Global Mosaic 500 m (1024-px browse). NASA/JPL Galileo and Voyager; USGS.
- Ganymede: Ganymede Voyager/Galileo Simple Cylindrical, 2 km/px. NASA/JPL Galileo SSI and Voyager 1 and 2; USGS.
- Callisto: Callisto Voyager–Galileo SSI Global Mosaic 1 km (1024-px browse). NASA/JPL Galileo and Voyager; USGS.
- Pluto and Charon: New Horizons LORRI–MVIC Global Mosaics 300 m, July 2017 (1024-px browses).
  NASA/Johns Hopkins University Applied Physics Laboratory/Southwest Research Institute; USGS.
- Ceres: Ceres Dawn FC Global Mosaic 400 m (20 px/deg, October 2015). NASA/JPL-Caltech/UCLA/MPS/DLR/IDA; mosaic by
  DLR (T. Roatsch, Dawn Team); distributed by USGS.

Unimaged areas of these mosaics (Pluto's and Charon's southern polar night, polar gaps elsewhere) are filled with the
mean albedo of the imaged area; the index's `source` string says how much. Ganymede (8271×4135) and Ceres
(7383×3691) are area-resampled to 8192×4096, the nearest 2:1 power of two; brightness is scaled as above.

**Geometric albedos** that set each map's `albedo_scale`: NASA NSSDCA planetary fact sheets
([nssdc.gsfc.nasa.gov/planetary/factsheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/): Mercury, Venus, Earth,
Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto and Charon) and the Jovian satellite fact sheet (Io, Europa,
Ganymede, Callisto); Ceres from the JPL Small-Body Database (Li et al. 2006, Icarus 182, 143). **Ring radii** for the
fit and its checks: NSSDCA Saturnian Rings Fact Sheet.

**Algorithms in the bake:** Intel ISPC Texture Compressor through the `intel_tex_2` crate (MIT OR Apache-2.0) for
BC7 and BC4; sRGB transfer function per IEC 61966-2-1:1999; BC4 block layout per Microsoft's Direct3D 11
block-compression reference (bake-side test decoder); lunar-Lambert photometric function (McEwen 1991, Icarus 92,
298).
<!-- /A8 -->

## Algorithms

None implemented yet. Each published model or algorithm is credited here when its code lands, in addition to the
source comment required next to the code.

<!-- M1 Wave B section (pre-split by Task B0, review I8): Task B3 writes only between its own markers, and only if
it adapts any of Hillaire's sample code (MIT): then the MIT notice goes here (M1 global constraints, review m24). -->
<!-- B3 -->
- **Atmospheres (`crates/uni-atmo`, Task B3)** follow Sébastien Hillaire, "A Scalable and Production Ready Sky
  and Atmosphere Rendering Technique", Computer Graphics Forum 39(4), 2020, and its sample code
  [UnrealEngineSkyAtmosphere](https://github.com/sebh/UnrealEngineSkyAtmosphere) (MIT). Adapted from the sample
  (rewritten in WGSL and Rust, not copied): the Earth constants of `SetupEarthAtmosphere`, and the structure and
  sample counts of the multiple-scattering pass (`NewMultiScattCS`: 8 × 8 directions, 20 steps, f_ms and
  Ψ_ms = L₂/(1 − f_ms)) and of the per-step integration in `IntegrateScatteredLuminance`. Its licence:

  > MIT License
  >
  > Copyright (c) 2020 Epic Games, Inc.
  >
  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  > documentation files (the "Software"), to deal in the Software without restriction, including without
  > limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
  > Software, and to permit persons to whom the Software is furnished to do so, subject to the following
  > conditions:
  >
  > The above copyright notice and this permission notice shall be included in all copies or substantial
  > portions of the Software.
  >
  > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
  > TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  > THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
  > CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  > DEALINGS IN THE SOFTWARE.
- The LUT parameterisations and the optical-depth quadrature follow the formulas of Eric Bruneton, "Precomputed
  Atmospheric Scattering: a New Implementation" (2017, [ebruneton/precomputed_atmospheric_scattering](https://github.com/ebruneton/precomputed_atmospheric_scattering),
  BSD-3-Clause), reimplemented from its documentation; no code copied.
<!-- /B3 -->
