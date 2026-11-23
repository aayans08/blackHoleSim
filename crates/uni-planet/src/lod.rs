//! The point → disk exposure blend (Task B0, shared by B1 and B2).

use uni_sim::snapshot::{BLEND_LOD_PX, POINT_LOD_PX};

/// Weight of the scene exposure for a resolved body or star of projected radius
/// `proj_radius_px` [px] while "Eye-adapted stars" is in effect: the pass writes its light with
/// `pre_mix(star_pre, scene_pre, lod_exposure_mix(radius))` (M1 frame contract, "Exposure split";
/// ruling 5), the log-space blend star_pre^(1 − w)·scene_pre^w of uni-gfx
/// (`common/exposure.wgsl`, CPU twin `uni_gfx::frame::pre_mix`; task B11), so the exposure in
/// stops follows this weight: (EV_scene − EV_star)·w stops below the star exposure.
///
/// - 0 at and below [`POINT_LOD_PX`] (1.5 px), where uni-sim still hands the body over as a point
///   drawn at star exposure, so the resolved disk starts at exactly the point's exposure;
/// - 1 at and above [`BLEND_LOD_PX`] (4 px), world light at scene exposure;
/// - in between, smoothstep in ln(radius): t = ln(r / 1.5) / ln(4 / 1.5), weight t²(3 − 2t)
///   (the cubic Hermite smoothstep, WGSL `smoothstep`), so the factor and its slope are
///   continuous at both ends and the blend takes equal steps per factor of size.
///
/// +∞ (a camera inside the body, `uni_sim::snapshot::proj_radius_px`) gives 1, and so does NaN:
/// scene exposure, the physical one. The CPU computes it per body per frame and uploads it in the
/// pass's per-body uniform, so it has no WGSL twin (ruling B.5). Labelled on screen as part of
/// "Eye-adapted stars".
pub fn lod_exposure_mix(proj_radius_px: f64) -> f32 {
    if proj_radius_px.is_nan() || proj_radius_px >= BLEND_LOD_PX {
        return 1.0;
    }
    if proj_radius_px <= POINT_LOD_PX {
        return 0.0;
    }
    let t = (proj_radius_px / POINT_LOD_PX).ln() / (BLEND_LOD_PX / POINT_LOD_PX).ln();
    (t * t * (3.0 - 2.0 * t)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review Focus 4: 0 at and below POINT_LOD_PX (star exposure, where the body is still a
    /// point), 1 at and above BLEND_LOD_PX (scene exposure), never decreasing in between, so a
    /// body growing past 1.5 px never pops.
    #[test]
    fn lod_exposure_mix_endpoints_monotonic() {
        for r in [0.0, 0.3, 1.0, 1.499_999, POINT_LOD_PX] {
            assert_eq!(lod_exposure_mix(r), 0.0, "r = {r}");
        }
        for r in [BLEND_LOD_PX, 4.000_001, 10.0, 1.0e6, f64::INFINITY] {
            assert_eq!(lod_exposure_mix(r), 1.0, "r = {r}");
        }
        // Non-decreasing and in [0, 1] on a 1000-step sweep across the band and past both ends.
        let mut prev = 0.0_f32;
        for i in 0..=1000 {
            let r = 1.0 + 4.0 * f64::from(i) / 1000.0;
            let m = lod_exposure_mix(r);
            assert!((0.0..=1.0).contains(&m), "r = {r}: {m}");
            assert!(m >= prev, "r = {r}: {m} < {prev}");
            prev = m;
        }
        // Smoothstep in ln(radius): half way at the geometric mean, and flat at both ends, so
        // the exposure is continuous with a continuous slope where the blend starts and stops.
        let mid = (POINT_LOD_PX * BLEND_LOD_PX).sqrt();
        assert!((lod_exposure_mix(mid) - 0.5).abs() < 1e-6);
        assert!(lod_exposure_mix(POINT_LOD_PX * (1.0 + 1e-4)) < 1e-6);
        assert!(lod_exposure_mix(BLEND_LOD_PX * (1.0 - 1e-4)) > 1.0 - 1e-6);
        // A camera inside a body has radius +∞ (uni-sim); NaN never reaches the GPU.
        assert_eq!(lod_exposure_mix(f64::NAN), 1.0);
    }
}
