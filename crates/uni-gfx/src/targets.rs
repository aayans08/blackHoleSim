//! Per-frame render targets: HDR colour, reversed-Z depth and the lens mask, all at the internal
//! render resolution.
//!
//! # The M1 frame contract
//!
//! It binds every M1 render feature; CLAUDE.md §2 lists which parts are built so far and which
//! are still targets. `Renderer::render` clears `hdr` and `lens_mask` to 0 and `depth` to 0.0
//! (infinitely far) every frame, then runs the features in push order.
//!
//! ## Exposure split (pre-exposed HDR)
//!
//! `hdr` holds light × exposure (`shaders/common/exposure.wgsl`); the tonemap adds none.
//! - *World light* (bodies, the Sun's disk, rings, atmospheres, comet nuclei, comae and tails):
//!   luminance [cd/m²] × `scene_pre()`.
//! - *Sky light and point sources* (catalogue stars, the Milky Way band, planets, the Sun and
//!   asteroids drawn as points): luminance × `star_pre()`. With "Eye-adapted stars" in effect
//!   `star_pre` comes from a fixed dark-adapted EV, otherwise `star_pre == scene_pre`: design §7
//!   item 4's dual exposure, extended to every unresolved source so a planet and a star of equal
//!   magnitude look equally bright (ruling 5).
//! - "Eye-adapted stars" is in effect when `ExposureSettings.eye_adapted &&
//!   !ExposureSettings.daylight`. The app sets `daylight` while the camera is inside an
//!   atmosphere shell with the Sun above its local horizon, because a real daytime sky hides the
//!   stars (review m12). The HUD's amber label follows exactly this condition, not
//!   `FrameCtx.exposure` (review m13).
//! - A resolved body whose projected radius is between `POINT_LOD_PX` (1.5 px) and
//!   `BLEND_LOD_PX` (4 px) blends its exposure factor from `star_pre` to `scene_pre` with the
//!   weight w (CPU twin `uni_planet::lod_exposure_mix`, B0), so the point → disk handover never
//!   pops. The blend is in log space (task B11): pre = `pre_mix(star_pre, scene_pre, w)` =
//!   star_pre^(1 − w)·scene_pre^w (`common/exposure.wgsl`, CPU twin `uni_gfx::frame::pre_mix`),
//!   so the exposure moves by equal stops per step of w across the band. (A linear mix stayed
//!   within one stop of `star_pre` until w = ½ and dropped the rest of a 19-stop gap just below
//!   4 px.) The blend is labelled as part of "Eye-adapted stars". Its rings and its atmosphere
//!   use the body's own weight, and every such writer stores its star share in `lens_mask.a`
//!   (below).
//! - Auto-exposure meters world light only: the meter (pass 9) runs before any sky light or
//!   point source is written. It reads each pixel at the pre it was written with,
//!   `pre_mix(scene_pre, star_pre, lens_mask.a)`, so a body in the LOD band is metered at its
//!   true luminance (lead ruling B6 I1).
//! - Star brightness against the band depends on the field of view, deliberately (review m26):
//!   stars are written as E/Ω_pixel and the band as a surface brightness, so at fov 20° a star's
//!   pixels are (tan 30°/tan 10°)² = 10.7× (+3.42 EV) brighter relative to the band than at
//!   60°. That is what a camera does when it zooms (point sources gain contrast over extended
//!   light), so it stays; it is not a bug to fix.
//! - Every feature clamps what it writes to [`HDR_MAX`].
//!
//! ## `lens_mask` channels ([`LENS_MASK_FORMAT`])
//!
//! - `r`: fraction of the pixel's background captured by a black hole (lens pass only).
//! - `g`: sky occlusion, 1 − transmittance toward the background, written with
//!   `BlendOperation::Max` on the green channel only. Bodies and comet nuclei write their
//!   coverage, rings their opacity, atmospheres 1 − luminance(T). The lens pass writes `g = 0` on
//!   the traced pixels it repaints.
//! - `b`: 1 where the lens pass wrote the lensed environment.
//! - `a`: the **star share** s = 1 − w of the world writers in the LOD band (lead ruling B6 I1),
//!   w being the exposure weight of `pre = pre_mix(star_pre, scene_pre, w)`: bodies, rings and
//!   star surfaces (uni-planet) and a body's atmosphere (uni-atmo, with its body's w) write
//!   `select(0, 1 − w, coverage > 0)` (an atmosphere: on every fragment its march does not
//!   discard), composited with `BlendOperation::Max` like `g`. It is 0 everywhere else: the
//!   clear value 0 means scene exposure, and plain world light (w = 1, or eye-adapted stars not
//!   in effect) stores 0. w is snapped to 1 when 1 − w < 2⁻¹⁴, f16's smallest normal, because
//!   Metal may flush f16 subnormals on store; the share is then 0 or an f16 normal and reads
//!   back as the writer's pre. With the log blend the snap moves pre by less than
//!   ΔEV·2⁻¹⁴ stops (ΔEV = scene EV − star EV; 0.0012 stop at 19 stops), and a share that was
//!   flushed would misread its pixel by as little, so the snap leaves no visible step (with B10's
//!   linear blend it was a step of 1 + 2⁻¹⁴·(2^ΔEV − 1), 33× at 19 stops). The meter (pass 9)
//!   divides each pixel by `pre_mix(scene_pre, star_pre, s)`, which is the writer's pre; with
//!   Max compositing no reading grows with the scene EV, so the exposure loop cannot run away
//!   (uni-post `histogram.rs`). The lens pass, after the meter, writes `a = 0` on the pixels it
//!   repaints. M3's star magnification will use a new channel or target, not `a`.
//!
//! Sky features multiply what they add by (1 − r)(1 − g); the band also by (1 − b). Point
//! sprites and asteroids multiply by (1 − r) only: they are depth-tested against bodies, so a
//! point behind an atmosphere is not dimmed by it (an accepted M1 approximation).
//!
//! The lens pass reads `depth` and leaves every pixel whose depth is not 0 (a body in front)
//! untouched, so foreground objects are not lensed (design §5.3). It writes `hdr` and
//! `lens_mask` only on the pixels it traces (its screen rectangle; the whole frame from inside a
//! horizon), and its escaped rays show the same environment the band draws.
//!
//! ## Shared inputs
//!
//! Features get per-frame data from the app through [`crate::SharedInput`]. The app sets every
//! one before `Renderer::render`; features read them only in `prepare`.
//!
//! ## Pass order (push order in `uni-app/src/scene.rs`)
//!
//! | # | Feature | Crate | Writes | Depth |
//! |---|---|---|---|---|
//! | 1 | `exposure` (auto-exposure resolve; own bind group: imports `common/exposure.wgsl`, never group 0) | uni-post | exposure buffer (storage) | — |
//! | 2 | `band-update` (one environment cube face per frame, when needed) | uni-sky | environment cubemap | — |
//! | 3 | `background` (the M0 debug grid; only with `LensEnv::DebugGrid` and a traced lens; display-referred, so the app runs Manual exposure whenever `DebugGrid` is selected) | uni-rel | hdr | — |
//! | 4 | `sun` (star surfaces) | uni-planet | hdr (world), g, a | write, `Greater` |
//! | 5 | `bodies` | uni-planet | hdr (world), g, a | write, `Greater` |
//! | 6 | `rings` | uni-planet | hdr (world), g, a | test only |
//! | 7 | `comets` (Wave C) | uni-planet | hdr (world), g (nucleus) | nucleus write; coma and tails test only |
//! | 8 | `atmosphere` (reads `depth`; dual-source variant: two pipelines, grey-transmittance variant: one) | uni-atmo | hdr, g, a | none attached |
//! | 9 | `histogram` (the exposure meter; reads `hdr` and `lens_mask.a`) | uni-post | meter state | — |
//! | 10 | `lens` (reads `depth`; skips pixels with depth ≠ 0) | uni-rel | hdr (sky, star exposure), r, g = 0, b | — |
//! | 11 | `band` | uni-sky | hdr (sky) × (1 − r)(1 − g)(1 − b) | none: no depth attachment (B9c ruling) |
//! | 12 | `stars` | uni-sky | hdr (sky) × (1 − r)(1 − g) | none: no depth attachment (B9c ruling) |
//! | 13 | `points` (bodies, the Sun and near stars as points) | uni-sky | hdr (sky) × (1 − r) | test `Greater` at true depth |
//! | 14 | `asteroids` (Wave C) | uni-sky | hdr (sky) × (1 − r) | test `Greater` at true depth |
//! | 15 | `bloom` (input clamped to `BLOOM_INPUT_MAX`) | uni-post | hdr | — |
//! | 16 | `tonemap` (upscale + AgX + dither) | uni-post | output view | — |
//! | — | HUD and overlays (egui) | uni-ui | output view | — |
//!
//! **The band is drawn after the opaque passes and the meter** (ruling 4, a deliberate deviation
//! from design §5.3, where the background comes first). The band is sky light at star exposure,
//! so it must not be metered; and drawing it after the bodies with (1 − g) composites it
//! correctly behind partly transparent atmospheres and rings. Since B9c (lead ruling) the band and
//! the catalogue stars have no depth test: `lens_mask` alone occludes them, so an anti-aliased body
//! edge (g = coverage) keeps its (1 − coverage) share of sky. Every opaque writer must write `g`.

use glam::UVec2;

/// Pre-exposed scene-linear HDR colour (Rec. 709 primaries): light × exposure, see
/// `shaders/common/camera.wgsl`. The tonemap applies AgX to it with no further exposure.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Reversed-Z depth (near = 1, far = 0; compare `Greater`, or `GreaterEqual` for things drawn
/// exactly on the far plane, such as stars).
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Lens mask: `r` captured fraction, `g` sky occlusion, `b` lensed environment written, `a` the
/// world writers' star share 1 − w (see "lens_mask channels" above). Cleared to 0 every frame.
/// Read with `textureLoad` at integer pixels; blendable, so `g` and `a` can use
/// `BlendOperation::Max`.
pub const LENS_MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Largest value any feature writes into `hdr` (below f16's largest finite value, 65504).
/// Twin of `HDR_MAX` in `shaders/common/exposure.wgsl`.
pub const HDR_MAX: f32 = 60000.0;

/// Written by compute (storage) and raster (attachment) passes, sampled by later ones.
const SHARED_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::STORAGE_BINDING
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::RENDER_ATTACHMENT);
/// `hdr` is also copied out for the `--hdr` capture sidecar ([`crate::capture::read_hdr`]).
const HDR_USAGE: wgpu::TextureUsages = SHARED_USAGE.union(wgpu::TextureUsages::COPY_SRC);
const LENS_MASK_USAGE: wgpu::TextureUsages = SHARED_USAGE;

/// The textures every feature renders into, at the internal render resolution.
pub struct FrameTargets {
    /// Internal render resolution. Never 0 in either axis.
    pub size: UVec2,
    /// Pre-exposed HDR colour.
    /// `STORAGE_BINDING | TEXTURE_BINDING | RENDER_ATTACHMENT | COPY_SRC`.
    pub hdr: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    /// `RENDER_ATTACHMENT | TEXTURE_BINDING`; cleared to 0.0 (infinitely far) every frame.
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// See [`LENS_MASK_FORMAT`]. `STORAGE_BINDING | TEXTURE_BINDING | RENDER_ATTACHMENT`.
    pub lens_mask: wgpu::Texture,
    pub lens_mask_view: wgpu::TextureView,
}

impl FrameTargets {
    /// Allocates all targets at `size`. Each axis is clamped to at least 1 and at most the
    /// device's `max_texture_dimension_2d`, so a minimised window or an oversized Retina
    /// surface never produces an invalid texture.
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        let size = Self::fit(device, size);
        let target = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (hdr, hdr_view) = target("hdr", HDR_FORMAT, HDR_USAGE);
        let (depth, depth_view) = target(
            "depth",
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let (lens_mask, lens_mask_view) = target("lens_mask", LENS_MASK_FORMAT, LENS_MASK_USAGE);
        Self {
            size,
            hdr,
            hdr_view,
            depth,
            depth_view,
            lens_mask,
            lens_mask_view,
        }
    }
}

impl FrameTargets {
    /// The size [`FrameTargets::new`] allocates for a request of `size` on `device`: each axis
    /// clamped to 1..=`max_texture_dimension_2d`.
    pub fn fit(device: &wgpu::Device, size: UVec2) -> UVec2 {
        let max = device.limits().max_texture_dimension_2d.max(1);
        Self::clamp_size(size).min(UVec2::splat(max))
    }

    /// Clamps each axis to at least 1 texel, so a minimised (0×0) window still gets valid targets.
    pub(crate) fn clamp_size(size: UVec2) -> UVec2 {
        size.max(UVec2::ONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdr_target_has_copy_src() {
        // The `--hdr` capture sidecar copies hdr to a buffer (`capture::read_hdr`).
        assert!(HDR_USAGE.contains(wgpu::TextureUsages::COPY_SRC));
        assert!(HDR_USAGE.contains(
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
        ));
        // Four lens-mask channels (r capture, g sky occlusion, b lensed environment, a M3).
        assert_eq!(LENS_MASK_FORMAT, wgpu::TextureFormat::Rgba16Float);
        assert!(LENS_MASK_USAGE.contains(
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
        ));
    }

    // Review Focus 1: a minimised or 0×0 window must never create a zero-sized texture.
    #[test]
    fn targets_clamp_zero_size() {
        assert_eq!(FrameTargets::clamp_size(UVec2::ZERO), UVec2::ONE);
        assert_eq!(
            FrameTargets::clamp_size(UVec2::new(0, 720)),
            UVec2::new(1, 720)
        );
        assert_eq!(
            FrameTargets::clamp_size(UVec2::new(1280, 0)),
            UVec2::new(1280, 1)
        );
        assert_eq!(
            FrameTargets::clamp_size(UVec2::new(1920, 1080)),
            UVec2::new(1920, 1080)
        );
    }
}
