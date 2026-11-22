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
<!-- /A8 -->

## Algorithms

None implemented yet. Each published model or algorithm is credited here when its code lands, in addition to the
source comment required next to the code.
