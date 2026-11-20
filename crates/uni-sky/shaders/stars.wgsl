// sky/stars.wgsl: catalogue stars as instanced point-spread-function (PSF) splats (M0 Task 8).
//
// Each star is one instance of a screen-aligned quad (two triangles, six vertices) centred on
// its exact sub-pixel projected position. Nothing is snapped to the pixel grid. The fragment
// shader integrates the PSF core over its own pixel square (erf differences), so the sum over a
// star's footprint equals the star's energy wherever its centre falls. That is what stops stars
// from crawling or looking pixelated in motion.
//
//   PSF = (1 − h)·Gaussian core (σ = CORE_SIGMA_PX) + h·Moffat glare halo,
//   h = HALO_ENERGY_FRACTION above HALO_MIN_LUX, else 0.
//
//   pixel luminance [cd/m²] = E / Ω_pixel × exp2(scene_ev100 − star_ev100) × (1 − lens_mask)
//                             × PSF weight
//
// E is the star's illuminance at the observer [lux], and Ω_pixel = (2·tan(fov_y/2) / H)² is the
// pixel solid angle [sr] (small-angle). The tonemap then multiplies by exposure_scale(scene_ev100),
// so on screen a star is exposed at star_ev100 whatever the scene exposure is. That is the
// "Eye-adapted stars" dual exposure of design §7 item 4.
//
// Blending is additive into `hdr`. Depth is tested (GreaterEqual, no writes) at the far plane.
// The CPU twins are in crates/uni-sky/src/psf.rs and stars.rs. `shader_constants_match_cpu`
// checks that the constants below match them, and `star_instance_layout_matches_wgsl` checks
// the struct layout.

#import "common/camera.wgsl"
#import "common/constants.wgsl"

// Twin of `uni_sky::StarInstance`: 32 bytes, two vec4s.
struct StarInstance {
    // xyz: unit direction to the star in root (ICRS) axes. w: illuminance E at the observer
    // [lux], finite and ≥ 0 (`star_instance` guarantees it).
    dir: vec4<f32>,
    // rgb: linear sRGB colour normalised to luminance Y = 1. w: unused.
    color: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> stars: array<StarInstance>;
// R32Float, 1.0 where a black hole captured the ray. R32Float is not filterable without
// FLOAT32_FILTERABLE, so it is read with textureLoad at the fragment's integer pixel. It is the
// same size as `hdr` (uni_gfx::FrameTargets), so the vertex stage also reads the attachment size
// from it.
@group(1) @binding(1) var lens_mask: texture_2d<f32>;

// PSF constants; see the Rust twins in psf.rs for their sources.
const CORE_SIGMA_PX: f32 = 0.65;
const HALO_ENERGY_FRACTION: f32 = 0.02;
const HALO_MIN_LUX: f32 = 1.0e-6;
const HALO_ALPHA_PX: f32 = 4.0;
const HALO_BETA: f32 = 2.5;
const MIN_RADIUS_PX: f32 = 3.0;
const MAX_RADIUS_PX: f32 = 64.0;
const VISIBILITY_CUTOFF: f32 = 3.0e-5;
// Stars whose clip-space w is at most this are behind the camera or on its plane. Directions
// are unit vectors, so on-screen stars have w ≥ cos(fov/2) ≫ this.
const MIN_CLIP_W: f32 = 1.0e-6;
// Upper bound for one splat's colour, below the largest finite f16 (65504), so a splat can never
// write infinity into the Rgba16Float target.
const HDR_MAX: f32 = 60000.0;
// Clip position for culled vertices. All six vertices coincide (zero area), and it lies outside
// the clip volume (z > w), so nothing is rasterised.
const CULLED: vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0);

// erf(x) by Abramowitz & Stegun 1964, Handbook of Mathematical Functions, eq. 7.1.26
// (after Hastings 1955), |error| ≤ 1.5e-7. Twin of `uni_sky::psf::erf`.
fn psf_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * exp(-a * a);
    return select(-y, y, x >= 0.0);
}

// Fraction of a unit-energy 1-D Gaussian (σ = CORE_SIGMA_PX, centred at 0) inside
// [edge, edge + 1]. Neighbouring pixels share edges, so a row's sum telescopes and the energy is
// conserved at any sub-pixel position. Twin of `uni_sky::psf::pixel_coverage`.
fn pixel_coverage(edge: f32) -> f32 {
    let k = 1.0 / (sqrt(2.0) * CORE_SIGMA_PX);
    return max(0.5 * (psf_erf((edge + 1.0) * k) - psf_erf(edge * k)), 0.0);
}

// Moffat 1969 (A&A 3, 455) profile with unit energy over the plane [1/px²]. Twin of
// `uni_sky::psf::moffat`.
fn moffat(r: f32) -> f32 {
    let a2 = HALO_ALPHA_PX * HALO_ALPHA_PX;
    return (HALO_BETA - 1.0) / (PI * a2) * pow(1.0 + r * r / a2, -HALO_BETA);
}

// Halo weight of a pixel whose centre is r px from the star: the Moffat profile lowered by its
// value at the footprint radius (so it reaches 0 continuously there) and renormalised to unit
// energy inside it. Twin of `uni_sky::psf::halo_weight`.
fn halo_weight(r: f32, radius: f32) -> f32 {
    if (!(r < radius)) {
        return 0.0;
    }
    let a2 = HALO_ALPHA_PX * HALO_ALPHA_PX;
    let edge = moffat(radius);
    let inside = 1.0 - pow(1.0 + radius * radius / a2, 1.0 - HALO_BETA)
        - PI * radius * radius * edge;
    return (moffat(r) - edge) / inside;
}

// Footprint radius [px]: where both the core and the halo of a star with display-normalised
// energy `energy` have fallen below VISIBILITY_CUTOFF, clamped to [MIN_RADIUS_PX, MAX_RADIUS_PX].
// Twin of `uni_sky::psf::footprint_radius_px`.
fn footprint_radius_px(energy: f32, halo_fraction: f32) -> f32 {
    var r = MIN_RADIUS_PX;
    let core_peak = (1.0 - halo_fraction) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
    if (core_peak > VISIBILITY_CUTOFF) {
        r = max(r, CORE_SIGMA_PX * sqrt(2.0 * log(core_peak / VISIBILITY_CUTOFF)));
    }
    let halo_peak = halo_fraction * energy * moffat(0.0);
    if (halo_peak > VISIBILITY_CUTOFF) {
        let q = pow(halo_peak / VISIBILITY_CUTOFF, 1.0 / HALO_BETA);
        r = max(r, HALO_ALPHA_PX * sqrt(q - 1.0));
    }
    return min(r, MAX_RADIUS_PX);
}

// Solid angle of one pixel [sr]: (2·tan(fov_y/2) / H)², the small-angle value at the image
// centre. Twin of `uni_sky::pixel_solid_angle`.
fn pixel_solid_angle(fov_y: f32, height_px: f32) -> f32 {
    let side = 2.0 * tan(0.5 * fov_y) / height_px;
    return side * side;
}

// Corner (±1, ±1) of the quad for vertex `vi` in 0..6, as two triangles (−−, +−, −+) and
// (−+, +−, ++). Bit `vi` of 0x32 gives the x corner and bit `vi` of 0x2C the y corner.
fn quad_corner(vi: u32) -> vec2<f32> {
    let bits = vec2<u32>((0x32u >> vi) & 1u, (0x2Cu >> vi) & 1u);
    return vec2<f32>(bits) * 2.0 - 1.0;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    // Star centre in framebuffer pixels: x right, y down, and pixel k spans [k, k + 1].
    @location(0) @interpolate(flat) centre_px: vec2<f32>,
    // Pixel luminance [cd/m²] per unit PSF weight, before the lens mask:
    // colour · E/Ω_pixel · 2^(scene_ev100 − star_ev100).
    @location(1) @interpolate(flat) radiance: vec3<f32>,
    // x: halo share h; y: footprint radius R [px].
    @location(2) @interpolate(flat) halo: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var out: VsOut;
    out.pos = CULLED;
    out.centre_px = vec2<f32>(0.0);
    out.radiance = vec3<f32>(0.0);
    out.halo = vec2<f32>(0.0);

    let star = stars[ii];
    let lux = star.dir.w;
    // Stars are at infinity, so the direction goes in with w = 0. The reversed-Z infinite
    // projection maps view-space (x, y, z, 0) to clip (f·x/aspect, f·y, 0, −z). Depth is
    // exactly 0, the far plane, and clip.w = −z is positive only in front of the camera
    // (which looks along −Z).
    let clip = camera.proj * camera.view_rot * vec4<f32>(star.dir.xyz, 0.0);
    // Behind the camera, on its plane, without light, or NaN: no quad.
    if (!(clip.w > MIN_CLIP_W) || !(lux > 0.0)) {
        return out;
    }

    // The rasteriser maps NDC onto the attachment, so use the attachment's size, not the
    // camera's viewport (they are equal unless a resize is in flight).
    let size = vec2<f32>(textureDimensions(lens_mask));
    let ndc = clip.xy / clip.w;
    let centre = vec2<f32>(0.5 * (ndc.x + 1.0) * size.x, 0.5 * (1.0 - ndc.y) * size.y);

    // lux → pixel luminance [cd/m²] per unit PSF weight: 2^(scene_ev100 − star_ev100) / Ω_pixel.
    let omega = pixel_solid_angle(camera.time.z, size.y);
    let to_luminance = exp2(camera.exposure.x - camera.exposure.y) / omega;
    // Display-normalised energy (the footprint's sum after the tonemap's exposure). It sizes the
    // quad: E/(Ω·1.2·2^star_ev100), whatever the scene exposure is.
    let energy = lux * to_luminance * exposure_scale(camera.exposure.x);
    let h = select(0.0, HALO_ENERGY_FRACTION, lux > HALO_MIN_LUX);
    let radius = footprint_radius_px(energy, h);

    // Off screen by more than its footprint, or too far out to place a quad: no quad. A centre
    // that projects to a huge or NaN pixel position (w barely above MIN_CLIP_W) would give a quad
    // with zero area in f32, so it is culled here too.
    let reach = vec2<f32>(radius + 1.0);
    if (!(all(centre > -reach) && all(centre < size + reach))) {
        return out;
    }

    // Half-size R + 0.5 px, so every pixel that overlaps the footprint is shaded. Pixels → NDC
    // scales by 2/size. The y flip only reverses the winding of a symmetric quad, and culling is
    // off.
    out.pos = vec4<f32>(ndc + quad_corner(vi) * (radius + 0.5) * 2.0 / size, 0.0, 1.0);
    out.centre_px = centre;
    out.radiance = star.color.rgb * (lux * to_luminance);
    out.halo = vec2<f32>(h, radius);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // @builtin(position).xy is this pixel's centre (k + 0.5), so floor gives the pixel index k.
    let pix = floor(in.pos.xy);
    let captured = clamp(textureLoad(lens_mask, vec2<i32>(pix), 0).r, 0.0, 1.0);
    // Core: the Gaussian integrated over this pixel's square [pix, pix + 1]², relative to the
    // star's centre.
    let edge = pix - in.centre_px;
    var weight = (1.0 - in.halo.x) * pixel_coverage(edge.x) * pixel_coverage(edge.y);
    if (in.halo.x > 0.0) {
        weight += in.halo.x * halo_weight(length(pix + vec2<f32>(0.5) - in.centre_px), in.halo.y);
    }
    let rgb = in.radiance * (weight * (1.0 - captured));
    return vec4<f32>(min(rgb, vec3<f32>(HDR_MAX)), 0.0);
}
