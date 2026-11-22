# Attribution

Every third-party data file, texture and published algorithm used by this project is credited here.
Downloaded data files are also recorded, with URL and SHA-256, in `data/manifest.toml`.

<!-- M1 Wave A sections (pre-split by Task A0b, review I8): each Data task writes only between its own
`<!-- A7 -->` … `<!-- /A7 -->` markers (A7 and A9 under Data, A8 under Textures; A9 may open a Fonts
heading inside its section). -->

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
