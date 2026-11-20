// rel/debug_sky.wgsl: the M0 lens background, a celestial grid every 15° in ICRS.
//
// NOT a real sky: the HUD labels it "Lens background: M0 debug grid". The base is
// 0.05 × white and the lines 0.4 × white, both after exposure at the scene EV, so the sky is
// never pure black and the black-hole shadow is the only black region.
//
// Directions are in root axes = ICRS/J2000 equatorial: +x toward RA 0h Dec 0°, +z toward the
// north celestial pole (IAU 1997 Resolution B2 / Global Constraints).

#import "common/camera.wgsl"

// Grid spacing: 15° (one hour of RA) in radians.
const SKY_GRID_SPACING: f32 = 0.2617993877991494;
// Base and line levels as fractions of white after exposure.
const SKY_BASE: f32 = 0.05;
const SKY_LINE: f32 = 0.4;
// A line is one footprint wide, but never wider than this fraction of the spacing, so a
// heavily minified region (near the photon ring) averages to a 10% grid instead of white.
const SKY_MAX_LINE_FRACTION: f32 = 0.1;

// Fraction of a box footprint of width `f`, centred at `u`, covered by lines of width `w`
// centred on every integer (all in units of the spacing). Exact box filter of the pulse train:
// the line measure in [0, x) is floor(x)·w + min(fract(x), w) for lines on [k, k + w)
// (I. Quílez, "Filterable procedurals", iquilezles.org/articles/filterableprocedurals).
fn grid_line_coverage(u: f32, w: f32, f: f32) -> f32 {
    let ww = clamp(w, 0.0, 1.0);
    let ff = max(f, 1e-6);
    let a = u + 0.5 * ww + 0.5 * ff;
    let b = u + 0.5 * ww - 0.5 * ff;
    let ia = floor(a) * ww + min(fract(a), ww);
    let ib = floor(b) * ww + min(fract(b), ww);
    return clamp((ia - ib) / ff, 0.0, 1.0);
}

// Scene-linear colour of the debug sky in direction `dir` (root axes, any length), for a pixel
// whose footprint on the sky is `footprint` radians across, at scene exposure `ev100`.
fn debug_sky(dir: vec3<f32>, footprint: f32, ev100: f32) -> vec3<f32> {
    let d = normalize(dir);
    let dec = asin(clamp(d.z, -1.0, 1.0));
    let ra = atan2(d.y, d.x);
    // Along a parallel, RA angles shrink by cos(dec); floored so the poles stay finite.
    let cos_dec = max(sqrt(max(1.0 - d.z * d.z, 0.0)), 1e-4);
    let fp = max(footprint, 1e-7);
    let width = min(fp, SKY_MAX_LINE_FRACTION * SKY_GRID_SPACING);
    // Parallels (constant Dec) and meridians (constant RA), each every 15°.
    let c_dec = grid_line_coverage(dec / SKY_GRID_SPACING, width / SKY_GRID_SPACING, fp / SKY_GRID_SPACING);
    let ra_unit = SKY_GRID_SPACING * cos_dec;
    let c_ra = grid_line_coverage(ra / SKY_GRID_SPACING, width / ra_unit, fp / ra_unit);
    let c = 1.0 - (1.0 - c_dec) * (1.0 - c_ra);
    // Divide by the exposure scale so the tonemapper's × exposure_scale(ev100) yields 0.05 / 0.4.
    // Capped at the hdr target's largest finite value (IEEE 754 binary16: 65504), which the
    // lines would pass above scene EV ≈ 17.
    return vec3<f32>(min(mix(SKY_BASE, SKY_LINE, c) / exposure_scale(ev100), 65504.0));
}
