// sky/band_update.wgsl: the band-update compute pass (Task B5). It renders the Milky Way band of
// uni-procgen's galaxy model v0 into the environment cubemap, then box-filters the faces it wrote
// into the 4 lower mips. CPU twin of one texel: `uni_sky::band::band_texel` (in cd/m²; the pass
// stores it ÷ ENV_UNIT_CD_M2, clamped).
//
// Texels hold absolute luminance [cd/m²] in units of uni_gfx::ENV_UNIT_CD_M2 (1e-4 cd/m², review
// m10), not pre-exposed (uni_gfx::env). Faces follow common/cube.wgsl's convention (WebGPU layer
// order, root = ICRS axes); the galaxy model works in galactocentric axes (galaxy.wgsl), so each
// direction is rotated ICRS → Galactic first.
//
// Per galaxy.wgsl's contract the importer declares GalaxyGpu at its own group and copies it into
// `galaxy` before calling any gal_* function. Group 0 is unused (the pass needs no camera).
// Two entry points with separate bind group layouts in Rust (band.rs):
// - cs_band_update: bindings 0, 1, 2 (galaxy, params, mip 0 as storage);
// - cs_band_downsample: bindings 1, 3, 4 (params, mip n − 1 read, mip n as storage).

#import "common/cube.wgsl"
#import "procgen/galaxy.wgsl"

// Twin of `uni_sky::band::BandUpdateParams` (band_layouts_match_wgsl): vec4-sized members only
// (CLAUDE.md §3 uniform layout rule).
struct BandUpdateParams {
    // xyz: camera position, galactocentric [pc]; w: v_lim, the faintest V magnitude drawn as
    // single stars (fainter starlight is the band).
    cam_gc_v_lim: vec4<f32>,
    // Rows of the ICRS → Galactic rotation (uni_core::ICRS_TO_GALACTIC, ESA 1997 Vol. 1
    // §1.5.3: the galactic x, y, z axes in ICRS); w = 0.
    icrs_to_gal: array<vec4<f32>, 3>,
    // x: first cube face of this batch; y: number of faces (dispatch z); zw: 0.
    faces: vec4<u32>,
}

// Luminance [cd/m²] of one texel unit. Twin of uni_gfx::ENV_UNIT_CD_M2.
const BAND_ENV_UNIT_CD_M2: f32 = 1.0e-4;
// Largest texel value written [units], 6 cd/m², below f16's largest finite value (65504).
// Twin of uni_sky::band::ENV_MAX_UNITS (= uni_gfx::HDR_MAX).
const BAND_ENV_MAX_UNITS: f32 = 60000.0;

@group(1) @binding(0) var<uniform> band_galaxy: GalaxyGpu;
@group(1) @binding(1) var<uniform> band_params: BandUpdateParams;
// Mip 0 of every face (a 2D-array view of the cube).
@group(1) @binding(2) var band_out: texture_storage_2d_array<rgba16float, write>;
// The downsample's source (mip n − 1) and destination (mip n), 2D-array views of the cube.
@group(1) @binding(3) var band_mip_src: texture_2d_array<f32>;
@group(1) @binding(4) var band_mip_dst: texture_storage_2d_array<rgba16float, write>;

// `dir` (ICRS axes) in Galactic axes (uni_core::icrs_to_galactic).
fn band_icrs_to_gal(dir: vec3<f32>) -> vec3<f32> {
    let m = band_params.icrs_to_gal;
    return vec3<f32>(dot(m[0].xyz, dir), dot(m[1].xyz, dir), dot(m[2].xyz, dir));
}

// Luminance [cd/m²] → texel units, never negative, clamped to BAND_ENV_MAX_UNITS (twin of
// band.rs `env_units`).
fn band_env_units(cd_m2: vec3<f32>) -> vec3<f32> {
    return clamp(cd_m2 / BAND_ENV_UNIT_CD_M2, vec3<f32>(0.0), vec3<f32>(BAND_ENV_MAX_UNITS));
}

// One texel of mip 0 per invocation: gid.xy the texel, gid.z the face within the batch. The
// direction goes through the texel's centre.
@compute @workgroup_size(8, 8, 1)
fn cs_band_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(band_out);
    if (gid.x >= size.x || gid.y >= size.y || gid.z >= band_params.faces.y) {
        return;
    }
    galaxy = band_galaxy;
    let face = band_params.faces.x + gid.z;
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let dir_gc = band_icrs_to_gal(cube_dir(face, uv));
    let cam = band_params.cam_gc_v_lim;
    let rgb = gal_band_radiance(cam.xyz, dir_gc, cam.w);
    textureStore(band_out, gid.xy, face, vec4<f32>(band_env_units(rgb), 1.0));
}

// One texel of mip n per invocation: the mean of its 2 × 2 children in mip n − 1 (a box filter;
// the mean radiance, so a coarser mip keeps the band's surface brightness).
@compute @workgroup_size(8, 8, 1)
fn cs_band_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(band_mip_dst);
    if (gid.x >= size.x || gid.y >= size.y || gid.z >= band_params.faces.y) {
        return;
    }
    let face = band_params.faces.x + gid.z;
    let p = vec2<i32>(gid.xy) * 2;
    let sum = textureLoad(band_mip_src, p, face, 0)
        + textureLoad(band_mip_src, p + vec2<i32>(1, 0), face, 0)
        + textureLoad(band_mip_src, p + vec2<i32>(0, 1), face, 0)
        + textureLoad(band_mip_src, p + vec2<i32>(1, 1), face, 0);
    textureStore(band_mip_dst, gid.xy, face, 0.25 * sum);
}
