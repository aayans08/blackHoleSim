//! The capture JSON v2's measurement geometry (M1 Task B9 Step 2): what the app writes beside a
//! capture so `uni-verify` (B8) can check it, computed from the World, the frame's camera and
//! the CPU twins.
//!
//! - `projected`: every pick item on screen ({name, x, y, radius_px}; radius 0 for a point).
//! - `metrics`: `star_count_drawn` (catalogue stars in frame above the star pass's visibility
//!   cut, `uni_sky::stars3d::star_is_visible`), `sun_v_mag_from_camera` (M_V☉ + 5 log₁₀(d/10 pc)),
//!   `band_gc_over_anticentre` (the galaxy model's CPU band radiance toward l = 0° over l = 180°,
//!   b = 0°, luminance), `snapshot_cpu_ms`, `comets_cpu_ms` (`null` before Wave C),
//!   `comets_settle_ms` (the capture's per-bookmark settle), `lit_px` (pixels of world light at or
//!   above the meter's 10⁻³ cd/m², read from the HDR tap right after the meter, so before any
//!   sky light: `null` if the tap is missing) and `earth_flux` (ruling B3 I4: Earth's rendered
//!   disk-integrated illuminance from the same tap against the p_V point flux of A3/A4's point
//!   path at this phase, the Mallama & Hilton 2018 phase law; recorded, no pass/fail) and
//!   `paper` (the view's "Exposure on paper" with the CPU twins, `crate::paper`).
//! - `regions`, `samples`, `features`, `flux`: per bookmark, as the plan's table "M1 capture
//!   bookmarks" names them (see [`measure`]).
//!
//! Pixel coordinates are framebuffer pixels (x right, y down, pixel k spanning [k, k + 1]),
//! the convention of `uni_sim::snapshot` and B8.

use std::collections::BTreeMap;

use glam::{DQuat, DVec2, DVec3, Vec3};
use uni_core::{PC, UniversePos};
use uni_sim::snapshot::{BodyView, Snapshot};

use crate::scene::{Frame, luminance};
use crate::tap::HdrImage;
use crate::world::World;

/// Most pixel samples per region (brief: ≤ 256).
pub const MAX_REGION_SAMPLES: usize = 256;
/// The meter's lit threshold [cd/m²] (uni-post `METER_MIN_CD_M2`).
const LIT_CD_M2: f32 = uni_post::METER_MIN_CD_M2;

/// One `projected` entry.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Projected {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub radius_px: f64,
}

/// One `samples` entry: a pixel and the luminance the CPU twins expect there [cd/m²].
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Sample {
    pub x: f64,
    pub y: f64,
    pub expected_cd_m2: [f64; 3],
    pub what: String,
}

/// One `features` entry (comet checks, Wave C).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Feature {
    pub x: f64,
    pub y: f64,
    pub r_min: f64,
    pub r_max: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_slope: Option<f64>,
}

/// One `flux` entry (re-review r2 NPA).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FluxEntry {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub radius_px: f64,
    pub expected_lux: f64,
    pub mask: Vec<[f64; 3]>,
    /// Illuminance of every other catalogue star and point sprite whose centre lies inside R
    /// [lx]: light B8's `flux` sums with the star's own (B4L's halo widened the footprints, so
    /// this is no longer under the plan's 1 %; the gate reads it beside the check).
    pub neighbour_lux: f64,
}

/// `metrics.earth_flux`: Earth's rendered flux against its point flux (ruling B3 I4).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct EarthFlux {
    pub phase_deg: f64,
    pub proj_radius_px: f64,
    /// Σ (Y ÷ pre_scene)·Ω over Earth's disk and atmosphere in the world tap [lx].
    pub rendered_lux: f64,
    /// The p_V point flux (Lambert, Mallama & Hilton 2018 phase correction) [lx].
    pub point_lux: f64,
    /// −2.5 log₁₀(rendered / point) [mag]: positive when the disk is fainter.
    pub gap_mag: f64,
}

/// `metrics`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct Metrics {
    pub star_count_drawn: u32,
    pub sun_v_mag_from_camera: f64,
    pub band_gc_over_anticentre: f64,
    pub snapshot_cpu_ms: f32,
    pub comets_cpu_ms: Option<f32>,
    pub comets_settle_ms: f32,
    pub lit_px: Option<u32>,
    pub earth_flux: Option<EarthFlux>,
    /// The view's "Exposure on paper" with the CPU twins (`crate::paper`): the EV and lit
    /// pixels the meter should find, for comparing with `ev100` and `lit_px`.
    pub paper: Option<crate::paper::Paper>,
}

/// Everything [`measure`] writes, flattened into the capture JSON.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct Measurements {
    pub projected: Vec<Projected>,
    pub metrics: Metrics,
    pub regions: BTreeMap<String, Vec<[f64; 2]>>,
    pub samples: Vec<Sample>,
    pub features: BTreeMap<String, Feature>,
    pub flux: Vec<FluxEntry>,
}

/// What [`measure`] reads.
pub struct Input<'a> {
    pub name: &'a str,
    pub world: &'a World,
    pub frame: &'a Frame,
    /// [scene, star] EV100 of the captured frame.
    pub ev100: [f32; 2],
    /// The HDR tap right after the meter (world light only).
    pub world_hdr: Option<&'a HdrImage>,
    /// The band's faint-star limit [V].
    pub v_lim: f32,
    pub comets_settle_ms: f32,
    /// Saturn's ring profile, for the paper exposure.
    pub rings: Option<&'a uni_data::RingProfile>,
    /// The map index and directory, for the map samples' CPU twin.
    pub maps: &'a [uni_data::MapInfo],
    pub map_dir: &'a std::path::Path,
}

/// The capture camera: projection to framebuffer pixels and back.
pub struct Cam {
    pub pos: UniversePos,
    to_cam: DQuat,
    to_root: DQuat,
    tan_half_y: f64,
    aspect: f64,
    pub size: DVec2,
    pub fov_y_rad: f64,
}

impl Cam {
    pub fn new(view: &uni_core::CameraView, size: glam::UVec2) -> Self {
        let size = size.max(glam::UVec2::ONE).as_dvec2();
        let orient = view.orient.normalize();
        Self {
            pos: view.pos,
            to_cam: orient.conjugate(),
            to_root: orient,
            tan_half_y: (0.5 * view.fov_y_rad).tan(),
            aspect: size.x / size.y,
            size,
            fov_y_rad: view.fov_y_rad,
        }
    }

    /// Where a point at `rel` [m] from the camera lands [px]; `None` behind the camera
    /// (`uni_sim::snapshot`'s projector).
    pub fn project(&self, rel: DVec3) -> Option<DVec2> {
        let c = self.to_cam * rel;
        if !(c.z < 0.0) {
            return None;
        }
        let ndc = DVec2::new(c.x / self.aspect, c.y) / (-c.z * self.tan_half_y);
        Some(DVec2::new(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * self.size)
    }

    /// Unit direction [root axes] through framebuffer point `p` (a pixel's centre is k + 0.5).
    pub fn ray(&self, p: DVec2) -> DVec3 {
        let ndc = DVec2::new(2.0 * p.x / self.size.x - 1.0, 1.0 - 2.0 * p.y / self.size.y);
        let c = DVec3::new(
            ndc.x * self.aspect * self.tan_half_y,
            ndc.y * self.tan_half_y,
            -1.0,
        );
        (self.to_root * c).normalize()
    }

    /// The camera's forward axis [root axes].
    pub fn forward(&self) -> DVec3 {
        self.to_root * DVec3::NEG_Z
    }

    /// Projected radius [px] of a sphere of `radius_m` at `dist_m` (the snapshot's rule).
    pub fn radius_px(&self, radius_m: f64, dist_m: f64) -> f64 {
        uni_sim::snapshot::proj_radius_px(radius_m, dist_m, self.fov_y_rad, self.size.y)
    }

    /// Angle [rad] one pixel subtends on the axis.
    pub fn pixel_angle(&self) -> f64 {
        2.0 * self.tan_half_y / self.size.y
    }

    /// Solid angle [sr] of pixel centre `p`: Ω_c·cos³θ (`uni_sky::stars` convention).
    pub fn pixel_solid_angle(&self, p: DVec2) -> f64 {
        let cos = self.ray(p).dot(self.forward());
        let side = self.pixel_angle();
        side * side * cos * cos * cos
    }

    /// The centres of every pixel, row by row.
    pub fn pixels(&self) -> impl Iterator<Item = DVec2> + '_ {
        let (w, h) = (self.size.x as u32, self.size.y as u32);
        (0..h).flat_map(move |y| {
            (0..w).map(move |x| DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5))
        })
    }
}

/// Ray–ellipsoid intersection: the nearest positive distance [m] along unit `dir` from the
/// camera to the body `b` (its triaxial radii, in its body axes); `None` on a miss.
pub fn hit_body(b: &BodyView, dir: DVec3) -> Option<f64> {
    hit_ellipsoid(-b.rel_pos_m, dir, b.orient, b.radii_m)
}

/// Ray–ellipsoid intersection from `origin` (relative to the centre, root axes) along unit
/// `dir`, for radii `radii` in the body axes `orient` (body → root).
pub fn hit_ellipsoid(origin: DVec3, dir: DVec3, orient: DQuat, radii: DVec3) -> Option<f64> {
    let inv = orient.conjugate();
    let o = (inv * origin) / radii;
    let d = (inv * dir) / radii;
    let a = d.length_squared();
    let bq = o.dot(d);
    let c = o.length_squared() - 1.0;
    let disc = bq * bq - a * c;
    if !(disc >= 0.0) {
        return None;
    }
    let s = disc.sqrt();
    let t0 = (-bq - s) / a;
    let t1 = (-bq + s) / a;
    if t0 > 0.0 {
        Some(t0)
    } else if t1 > 0.0 {
        Some(t1)
    } else {
        None
    }
}

/// The ellipsoid's radius [m] along unit body-relative direction `u` (root axes).
pub fn ellipsoid_radius(u: DVec3, orient: DQuat, radii: DVec3) -> f64 {
    1.0 / ((orient.conjugate() * u) / radii).length()
}

/// Tangent altitude [m] of the ray along unit `dir` from the camera past body `b`: the closest
/// approach to the centre minus the ellipsoid's radius toward that point, and the distance to
/// that point; `None` if the closest approach is behind the camera.
pub fn tangent_altitude(b: &BodyView, dir: DVec3) -> Option<(f64, f64)> {
    let t = b.rel_pos_m.dot(dir);
    if !(t > 0.0) {
        return None;
    }
    let p = dir * t - b.rel_pos_m; // closest point, relative to the centre
    let r = p.length();
    Some((r - ellipsoid_radius(p / r, b.orient, b.radii_m), t))
}

/// Phase angle [rad] of body `b` seen from the camera (Sun–body–camera).
pub fn phase_rad(b: &BodyView) -> f64 {
    angle(b.sun_rel_m, -b.rel_pos_m)
}

/// Angle between two vectors [rad], accurate near 0 and π.
pub fn angle(a: DVec3, b: DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b))
}

/// Up to [`MAX_REGION_SAMPLES`] of `pts`, evenly spread over the list.
pub fn thin(pts: Vec<DVec2>) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n <= MAX_REGION_SAMPLES {
        return pts.into_iter().map(|p| [p.x, p.y]).collect();
    }
    (0..MAX_REGION_SAMPLES)
        .map(|i| pts[i * n / MAX_REGION_SAMPLES])
        .map(|p| [p.x, p.y])
        .collect()
}

/// The body named `name` in the snapshot.
fn body<'a>(world: &World, snap: &'a Snapshot, name: &str) -> Option<&'a BodyView> {
    let id = world.sim.find(name)?;
    snap.bodies.iter().find(|b| b.id == id)
}

/// Computes the measurements of one capture (module docs).
pub fn measure(input: &Input<'_>) -> Measurements {
    let frame = input.frame;
    let world = input.world;
    let snap = &frame.snapshot;
    let cam = Cam::new(&frame.ctx.camera, frame.ctx.viewport);
    let mut m = Measurements {
        projected: projected(world, frame, &cam, input.ev100[1]),
        ..Measurements::default()
    };
    m.metrics = metrics(input, &cam);
    let regions = &mut m.regions;
    let mut add = |name: &str, pts: Vec<DVec2>| {
        regions.insert(name.to_owned(), thin(pts));
    };
    match input.name {
        "earth-limb-dawn" => {
            if let Some(e) = body(world, snap, "Earth") {
                let (high, low) = limb_arcs(&cam, e, input.world, snap);
                add("limb_arc_high", high);
                add("limb_arc_low", low);
            }
        }
        "mars-reference-angle" => {
            if let Some(b) = body(world, snap, "Mars") {
                add("limb_haze", haze_band(&cam, b, 10.0e3, 40.0e3, true));
            }
        }
        "venus-crescent" => {
            if let Some(b) = body(world, snap, "Venus") {
                // The drawn Venus is its cloud deck, 70 km above the solid radius (B1, B3):
                // the haze above the deck's night limb.
                let mut deck = b.clone();
                deck.radii_m += DVec3::splat(uni_atmo::VENUS_CLOUD_TOP_M);
                add(
                    "night_limb_haze",
                    haze_band(&cam, &deck, 2.0e3, 30.0e3, false),
                );
            }
        }
        "titan-backlit" => {
            if let Some(b) = body(world, snap, "Titan") {
                add("haze_ring", haze_band(&cam, b, 100.0e3, 500.0e3, false));
            }
        }
        "sun-closeup" => {
            if let Some(s) = snap.star_surfaces.first() {
                add("disk_centre", sun_mu_region(&cam, s, 0.99, 1.0));
                add("mu_0p5", sun_mu_region(&cam, s, 0.49, 0.51));
            }
        }
        "sun-granulation" => {
            let c = cam.size / 2.0;
            let pts = (0..16)
                .flat_map(|j| (0..16).map(move |i| (i, j)))
                .map(|(i, j)| {
                    DVec2::new(
                        c.x - 128.0 + 8.0 + 16.0 * f64::from(i),
                        c.y - 128.0 + 8.0 + 16.0 * f64::from(j),
                    )
                    .floor()
                        + DVec2::splat(0.5)
                })
                .collect();
            add("granulation_patch", pts);
        }
        "sky-galactic-center" => {
            let (plane, high) = band_regions(&cam);
            add("band_plane", plane);
            add("band_high_lat", high);
        }
        "saturn-ring-plane" | "saturn-night-side" => {
            if let Some(s) = body(world, snap, "Saturn") {
                let r = crate::measure::saturn_regions(&cam, s, input.name == "saturn-ring-plane");
                for (k, v) in r {
                    add(k, v);
                }
            }
        }
        _ => {}
    }
    m.samples = match input.name {
        "earth-map-closeup" => map_samples(input, &cam, "Earth"),
        "moon-full" => map_samples(input, &cam, "Moon"),
        "mars-reference-angle" => map_samples(input, &cam, "Mars"),
        _ => Vec::new(),
    };
    m.flux = match input.name {
        "sky-orion-3d" => flux_entries(input, &cam, &["Rigel"]),
        "alpha-cen-sun" => flux_entries(input, &cam, &["Sun"]),
        _ => Vec::new(),
    };
    m
}

/// Every pick item on screen with a radius: a resolved body's or star surface's projected
/// radius; for a point or catalogue star, the radius of its bright core ([`bright_radius_px`]),
/// so B8's centroid window covers a glare blob whole.
fn projected(world: &World, frame: &Frame, cam: &Cam, star_ev100: f32) -> Vec<Projected> {
    let snap = &frame.snapshot;
    let years = frame.ctx.epoch.seconds_since(uni_core::Epoch::J2000) / crate::scene::JULIAN_YEAR_S;
    let cam_pc = crate::scene::camera_icrs_pc(&cam.pos);
    snap.pick
        .iter()
        .map(|p| {
            let resolved = snap
                .bodies
                .iter()
                .find(|b| b.id == p.id)
                .map(|b| b.proj_radius_px)
                .or_else(|| {
                    snap.star_surfaces
                        .iter()
                        .find(|s| s.id == p.id)
                        .map(|s| s.proj_radius_px)
                });
            let radius_px = resolved.unwrap_or_else(|| {
                let reflected = world
                    .sim
                    .objects()
                    .get(p.id as usize)
                    .is_some_and(|o| crate::scene::is_reflected(o.kind));
                let scale = if reflected {
                    frame.reflected_scale
                } else {
                    1.0
                };
                let point = snap
                    .points
                    .iter()
                    .find(|q| q.id == Some(p.id))
                    .map(|q| (q.rel_pos_m, q.illuminance_lux * scale));
                let star = || {
                    let i = world.sim.objects().get(p.id as usize)?.catalog_index?;
                    let s = world.sim.star_data().get(i as usize)?;
                    let rel = (star_pos_pc(s, years) - cam_pc) * PC;
                    let d_pc = rel.length() / PC;
                    let v = f64::from(s.abs_mag_v) + 5.0 * (d_pc / 10.0).log10();
                    Some((rel, uni_core::star_illuminance_lux(v)))
                };
                point.or_else(star).map_or(0.0, |(rel, lux)| {
                    bright_radius_px(cam, rel, lux, star_ev100)
                })
            });
            Projected {
                name: world.name(p.id).to_owned(),
                x: f64::from(p.screen_px.x),
                y: f64::from(p.screen_px.y),
                radius_px,
            }
        })
        .collect()
}

/// Radius [px] within which a point source of illuminance `lux` at `rel` is brighter than 1
/// pre-exposed (its saturated core in the PNG): the larger of where its Gaussian core and its
/// glare halo (B4L's Moffat, without the window) fall to 1, at most its footprint; 0 for a
/// source that never reaches 1.
pub fn bright_radius_px(cam: &Cam, rel: DVec3, lux: f64, star_ev100: f32) -> f64 {
    use uni_sky::stars3d::{STARS3D_HALO_ALPHA_PX, STARS3D_HALO_BETA, STARS3D_HALO_FRACTION};
    let cos = rel.normalize_or_zero().dot(cam.forward()) as f32;
    let e = uni_sky::stars::star_display_energy_off_axis(
        lux as f32,
        cam.fov_y_rad as f32,
        cam.size.y as f32,
        star_ev100,
        cos,
    );
    let h = STARS3D_HALO_FRACTION;
    let sigma = uni_sky::psf::CORE_SIGMA_PX;
    let core_peak = (1.0 - h) * e / (std::f32::consts::TAU * sigma * sigma);
    let mut r = 0.0_f32;
    if core_peak > 1.0 {
        r = r.max(sigma * (2.0 * core_peak.ln()).sqrt());
    }
    let halo_peak = h * e * uni_sky::stars3d::stars3d_moffat(0.0);
    if halo_peak > 1.0 {
        r = r.max(STARS3D_HALO_ALPHA_PX * (halo_peak.powf(1.0 / STARS3D_HALO_BETA) - 1.0).sqrt());
    }
    f64::from(r.min(uni_sky::stars3d::footprint_radius_px(e, h)))
}

/// The `metrics` block.
fn metrics(input: &Input<'_>, cam: &Cam) -> Metrics {
    let world = input.world;
    let frame = input.frame;
    let sun_d = world
        .position_of("Sun")
        .map_or(f64::NAN, |s| s.delta_from(cam.pos).length());
    let cam_pc = crate::scene::camera_icrs_pc(&cam.pos);
    let model = uni_procgen::galaxy::GalaxyModel::milky_way();
    let cam_gc = model.icrs_pc_to_galactocentric(cam_pc);
    let band = |dir: DVec3| {
        let c = model.band_radiance(cam_gc, dir, f64::from(input.v_lim), 1.0_f64.to_radians());
        0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
    };
    let ratio = band(DVec3::X) / band(DVec3::NEG_X);
    Metrics {
        star_count_drawn: stars_drawn(world, frame, cam, input.ev100[1]),
        sun_v_mag_from_camera: uni_core::M_V_SUN + 5.0 * (sun_d / (10.0 * PC)).log10(),
        band_gc_over_anticentre: ratio,
        snapshot_cpu_ms: frame.snapshot_cpu_ms,
        comets_cpu_ms: None,
        comets_settle_ms: input.comets_settle_ms,
        lit_px: input.world_hdr.map(|img| lit_px(img, input.ev100)),
        earth_flux: input
            .world_hdr
            .and_then(|img| earth_flux(world, frame, cam, img, input.ev100[0])),
        paper: Some(crate::paper::on_paper(
            world,
            &frame.ctx.camera,
            frame.ctx.viewport,
            input.rings,
        )),
    }
}

/// Pixels of world light at or above the meter's threshold in the world tap: luminance ÷
/// scene pre ≥ 10⁻³ cd/m². (A body in the 1.5–4 px LOD band was written nearer star pre and
/// is read here at scene pre: at most a few dozen pixels.)
pub fn lit_px(img: &HdrImage, ev100: [f32; 2]) -> u32 {
    let pre = uni_gfx::exposure_scale(ev100[0]);
    let n = img
        .pixels
        .iter()
        .filter(|p| luminance(Vec3::new(p[0], p[1], p[2])) / pre >= LIT_CD_M2)
        .count();
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Catalogue stars the star pass draws in this frame: in frame (with their footprint), not
/// suppressed, above the visibility cut at star EV `star_ev100` (the vertex stage's rule, no
/// twinkle). Positions include proper motion to the frame's epoch.
fn stars_drawn(world: &World, frame: &Frame, cam: &Cam, star_ev100: f32) -> u32 {
    let stars = world.sim.star_data();
    let years = frame.ctx.epoch.seconds_since(uni_core::Epoch::J2000) / crate::scene::JULIAN_YEAR_S;
    let cam_pc = crate::scene::camera_icrs_pc(&cam.pos);
    let fwd = cam.forward();
    let h = cam.size.y as f32;
    let fov = cam.fov_y_rad as f32;
    let suppressed = &frame.snapshot.suppressed_stars;
    let mut n = 0_u32;
    for (i, s) in stars.iter().enumerate() {
        let rel = star_pos_pc(s, years) - cam_pc;
        let d = rel.length();
        if !(d > 0.0) || suppressed.contains(&(i as u32)) {
            continue;
        }
        let cos = rel.dot(fwd) / d;
        if cos < 0.1 {
            continue;
        }
        let Some(p) = cam.project(rel) else { continue };
        let v = f64::from(s.abs_mag_v) + 5.0 * (d / 10.0).log10();
        let lux = uni_core::star_illuminance_lux(v) as f32;
        let e = uni_sky::stars::star_display_energy_off_axis(lux, fov, h, star_ev100, cos as f32);
        if !uni_sky::stars3d::star_is_visible(e, uni_sky::stars3d::STARS3D_HALO_FRACTION) {
            continue;
        }
        let r = f64::from(uni_sky::stars3d::footprint_radius_px(
            e,
            uni_sky::stars3d::STARS3D_HALO_FRACTION,
        ));
        if p.x > -r && p.y > -r && p.x < cam.size.x + r && p.y < cam.size.y + r {
            n += 1;
        }
    }
    n
}

/// A catalogue star's position [pc, ICRS] `years` after J2000: linear motion with its
/// catalogue velocity (uni-sim's `star_pos_pc`, which is crate-private there).
pub fn star_pos_pc(s: &uni_data::Star3, years: f64) -> DVec3 {
    let v = if s.flags() & uni_data::FLAG_NO_VELOCITY != 0 {
        DVec3::ZERO
    } else {
        s.velocity_kms().as_dvec3() * f64::from(uni_sky::stars3d::PC_PER_YR_PER_KMS)
    };
    DVec3::from_array(s.pos_pc.map(f64::from)) + v * years
}

/// Earth's rendered flux against its point flux, when its whole disk and atmosphere are in
/// frame and the camera is outside the atmosphere (ruling B3 I4).
fn earth_flux(
    world: &World,
    frame: &Frame,
    cam: &Cam,
    img: &HdrImage,
    scene_ev100: f32,
) -> Option<EarthFlux> {
    let b = body(world, &frame.snapshot, "Earth")?;
    let d = b.rel_pos_m.length();
    let top = b.radii_m.x + 100.0e3; // the Kármán line, above B3's 100 km shell top
    if d <= top {
        return None;
    }
    let c = cam.project(b.rel_pos_m)?;
    let r = cam.radius_px(top, d) + 2.0;
    if c.x - r < 0.0 || c.y - r < 0.0 || c.x + r > cam.size.x || c.y + r > cam.size.y {
        return None;
    }
    let pre = f64::from(uni_gfx::exposure_scale(scene_ev100));
    let mut lux = 0.0;
    let (x0, x1) = ((c.x - r).floor() as i64, (c.x + r).ceil() as i64);
    let (y0, y1) = ((c.y - r).floor() as i64, (c.y + r).ceil() as i64);
    for y in y0..y1 {
        for x in x0..x1 {
            let p = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
            if (p - c).length() > r {
                continue;
            }
            if let Some(t) = img.at(x, y) {
                let l = f64::from(luminance(Vec3::new(t[0], t[1], t[2]))) / pre;
                lux += l * cam.pixel_solid_angle(p);
            }
        }
    }
    let phase = phase_rad(b);
    let brdf = b.brdf;
    let weight = match brdf {
        uni_physics::photometry::Brdf::Lambert => 2.0 / 3.0,
        uni_physics::photometry::Brdf::LunarLambert { l } => l + 2.0 * (1.0 - l) / 3.0,
    };
    let rho = uni_physics::photometry::rho_from_geometric_albedo(brdf, b.geometric_albedo);
    let corr = uni_physics::photometry::planet_phase_correction(399, brdf, phase)
        .min(uni_sim::snapshot::MAX_PHASE_CORRECTION);
    let ratio = b.mean_radius_m / d;
    let point = rho
        * weight
        * b.sun_illuminance_lux
        * ratio
        * ratio
        * uni_physics::photometry::disk_phase_function(brdf, phase)
        * corr;
    Some(EarthFlux {
        phase_deg: phase.to_degrees(),
        proj_radius_px: b.proj_radius_px,
        rendered_lux: lux,
        point_lux: point,
        gap_mag: -2.5 * (lux / point).log10(),
    })
}

/// `earth-limb-dawn`'s two limb arcs: sky pixels (rays that miss the ground) whose tangent
/// altitude is 30–60 km (`high`) or 0–12 km (`low`), within ±15° of azimuth of the hidden Sun as
/// seen from the camera (the sunrise arc).
fn limb_arcs(cam: &Cam, e: &BodyView, world: &World, snap: &Snapshot) -> (Vec<DVec2>, Vec<DVec2>) {
    let _ = world;
    let up = (-e.rel_pos_m).normalize(); // local vertical at the camera
    let sun = snap
        .sun
        .as_ref()
        .map_or(DVec3::X, |s| s.rel_pos_m.normalize());
    let horiz = |v: DVec3| (v - up * v.dot(up)).normalize_or_zero();
    let sun_h = horiz(sun);
    let (mut high, mut low) = (Vec::new(), Vec::new());
    for p in cam.pixels() {
        let dir = cam.ray(p);
        if horiz(dir).dot(sun_h) < 15_f64.to_radians().cos() {
            continue;
        }
        if hit_body(e, dir).is_some() {
            continue;
        }
        let Some((alt, _)) = tangent_altitude(e, dir) else {
            continue;
        };
        if (30.0e3..=60.0e3).contains(&alt) {
            high.push(p);
        } else if (0.0..=12.0e3).contains(&alt) {
            low.push(p);
        }
    }
    (high, low)
}

/// Sky pixels just outside body `b`'s ellipsoid limb with a tangent altitude in [`lo`, `hi`] m,
/// on its lit half (`lit`) or its night half (the side of the image away from the Sun).
fn haze_band(cam: &Cam, b: &BodyView, lo: f64, hi: f64, lit: bool) -> Vec<DVec2> {
    let Some(c) = cam.project(b.rel_pos_m) else {
        return Vec::new();
    };
    let d = b.rel_pos_m.length();
    let r = cam.radius_px(b.radii_m.x + hi, d) + 2.0;
    let sun_img = cam
        .project(b.rel_pos_m + b.sun_rel_m.normalize() * b.radii_m.x)
        .map_or(DVec2::X, |s| (s - c).normalize_or_zero());
    let mut out = Vec::new();
    let (x0, x1) = (
        (c.x - r).floor().max(0.0) as i64,
        (c.x + r).ceil().min(cam.size.x) as i64,
    );
    let (y0, y1) = (
        (c.y - r).floor().max(0.0) as i64,
        (c.y + r).ceil().min(cam.size.y) as i64,
    );
    for y in y0..y1 {
        for x in x0..x1 {
            let p = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
            let side = (p - c).normalize_or_zero().dot(sun_img);
            if (lit && side < 0.5) || (!lit && side > -0.5) {
                continue;
            }
            let dir = cam.ray(p);
            if hit_body(b, dir).is_some() {
                continue;
            }
            if let Some((alt, _)) = tangent_altitude(b, dir)
                && (lo..=hi).contains(&alt)
            {
                out.push(p);
            }
        }
    }
    out
}

/// Sun-disk pixels whose μ (cosine of the angle between the surface normal and the line of
/// sight) is in [`lo`, `hi`].
fn sun_mu_region(
    cam: &Cam,
    s: &uni_sim::snapshot::StarSurfaceView,
    lo: f64,
    hi: f64,
) -> Vec<DVec2> {
    let mut out = Vec::new();
    for p in cam.pixels() {
        let dir = cam.ray(p);
        let Some(t) = hit_ellipsoid(-s.rel_pos_m, dir, DQuat::IDENTITY, DVec3::splat(s.radius_m))
        else {
            continue;
        };
        let n = (dir * t - s.rel_pos_m).normalize();
        let mu = -n.dot(dir);
        if (lo..=hi).contains(&mu) {
            out.push(p);
        }
    }
    out
}

/// `sky-galactic-center`'s regions: pixels whose direction has Galactic latitude b ∈ [−6°, −4°]
/// with |l| < 10° (`band_plane`), and |b| ∈ [24°, 26°] (`band_high_lat`).
fn band_regions(cam: &Cam) -> (Vec<DVec2>, Vec<DVec2>) {
    let (mut plane, mut high) = (Vec::new(), Vec::new());
    for p in cam.pixels() {
        let g = uni_core::icrs_to_galactic(cam.ray(p));
        let b = g.z.asin().to_degrees();
        let l = g.y.atan2(g.x).to_degrees();
        if (-6.0..=-4.0).contains(&b) && l.abs() < 10.0 {
            plane.push(p);
        } else if (24.0..=26.0).contains(&b.abs()) {
            high.push(p);
        }
    }
    (plane, high)
}

/// Saturn's regions (see `saturn_regions`'s callers): with `lit_face` (`saturn-ring-plane`)
/// `ring_cassini` / `ring_b` on one azimuth band and `globe_ring_shadow` / `globe_lit_ref`;
/// otherwise (`saturn-night-side`) `ring_in_planet_shadow` / `ring_lit_same_radius`.
pub fn saturn_regions(cam: &Cam, s: &BodyView, lit_face: bool) -> Vec<(&'static str, Vec<DVec2>)> {
    let Some(ring) = s.ring else {
        return Vec::new();
    };
    let n = ring.normal;
    let sun = s.sun_rel_m.normalize();
    // Ring radii [m]: Cassini division 117 580–122 170 km, B ring 92 000–117 580 km, C ring
    // 74 658–92 000 km (Cuzzi et al. 2009, "Ring particle composition and size distribution",
    // in *Saturn from Cassini-Huygens*, table 15.1).
    let in_cassini = |r: f64| (118.5e6..=121.5e6).contains(&r);
    let in_b = |r: f64| (100.0e6..=112.0e6).contains(&r);
    let in_c = |r: f64| (78.0e6..=88.0e6).contains(&r);
    let mut cassini = Vec::new();
    let mut b_ring = Vec::new();
    let mut shadow = Vec::new();
    let mut lit_c = Vec::new();
    let mut globe_shadow = Vec::new();
    let mut globe_ref: Vec<(DVec2, f64)> = Vec::new();
    let mut shadow_mu0 = Vec::new();
    let c_img = cam.project(s.rel_pos_m).unwrap_or(cam.size / 2.0);
    for p in cam.pixels() {
        let dir = cam.ray(p);
        let globe = hit_body(s, dir);
        // Ring-plane crossing.
        let denom = dir.dot(n);
        let ring_hit = (denom.abs() > 1e-9)
            .then(|| s.rel_pos_m.dot(n) / denom)
            .filter(|t| *t > 0.0)
            .and_then(|t| {
                let q = dir * t - s.rel_pos_m;
                let r = q.length();
                (r >= ring.inner_m && r <= ring.outer_m).then_some((t, q, r))
            });
        if let Some((t, q, r)) = ring_hit {
            let in_front = globe.is_none_or(|g| t < g);
            if in_front {
                let inv = s.orient.conjugate();
                let shadowed =
                    uni_planet::planet_shadow_on_ring(inv * q, inv * sun, s.radii_m) < 0.5;
                // Same azimuth band: within ±3° of the image direction from the centre to the
                // ring's ansa nearest the camera's right.
                let az = (p - c_img).normalize_or_zero();
                if lit_face {
                    if !shadowed && az.x.abs() > 0.995 {
                        if in_cassini(r) {
                            cassini.push(p);
                        } else if in_b(r) {
                            b_ring.push(p);
                        }
                    }
                } else if in_c(r) {
                    if shadowed {
                        shadow.push(p);
                    } else {
                        lit_c.push(p);
                    }
                }
            }
        }
        if lit_face && let Some(g) = globe {
            let blocked = ring_hit.is_some_and(|(t, _, _)| t < g);
            if blocked {
                continue;
            }
            let x = dir * g - s.rel_pos_m;
            let normal =
                (s.orient * ((s.orient.conjugate() * x) / (s.radii_m * s.radii_m))).normalize();
            let mu0 = normal.dot(sun);
            if mu0 < 0.2 {
                continue;
            }
            // The rings' shadow on the globe: where the ray toward the Sun crosses the ring
            // plane inside the rings (opaque B ring and inner A ring).
            let ds = sun.dot(n);
            let crossing = (ds.abs() > 1e-9)
                .then(|| -x.dot(n) / ds)
                .filter(|t| *t > 0.0)
                .map(|t| (x + sun * t).length());
            match crossing {
                Some(r) if (95.0e6..=115.0e6).contains(&r) => {
                    globe_shadow.push(p);
                    shadow_mu0.push(mu0);
                }
                Some(r) if r >= ring.inner_m && r <= ring.outer_m => {}
                _ => globe_ref.push((p, mu0)),
            }
        }
    }
    if lit_face {
        shadow_mu0.sort_by(f64::total_cmp);
        let median = shadow_mu0.get(shadow_mu0.len() / 2).copied().unwrap_or(1.0);
        let refs = globe_ref
            .into_iter()
            .filter(|(_, mu0)| (mu0 - median).abs() < 0.05)
            .map(|(p, _)| p)
            .collect();
        vec![
            ("ring_cassini", cassini),
            ("ring_b", b_ring),
            ("globe_ring_shadow", globe_shadow),
            ("globe_lit_ref", refs),
        ]
    } else {
        vec![
            ("ring_in_planet_shadow", shadow),
            ("ring_lit_same_radius", lit_c),
        ]
    }
}

/// How many map samples a bookmark gets (plan: 16 points).
const MAP_SAMPLES: usize = 16;
/// A sample's map neighbourhood (5 × 5 texels of the 2k copy) must vary by less than this
/// fraction of its mean; of those, the 16 flattest are kept ("low-gradient map areas"), so the
/// GPU's filtering of the 8k map cannot move them much. (The Moon's map has its craters'
/// shading baked in: its flattest 5 × 5 patches still vary by 15–20 %.)
const MAP_FLAT_MAX: f32 = 0.30;

/// One map of the 2k uncompressed copies: its texels (RGBA8 sRGB or R8) and calibration.
struct CpuMap {
    tex: uni_data::TexFile,
    info: uni_data::MapInfo,
}

impl CpuMap {
    fn load(input: &Input<'_>, naif: i32, kind: uni_data::MapKind) -> Option<Self> {
        let info = input
            .maps
            .iter()
            .find(|m| m.naif == naif && m.kind == kind)?
            .clone();
        let tex = uni_data::load_tex(&input.map_dir.join(&info.fallback_file)).ok()?;
        Some(Self { tex, info })
    }

    /// The mean linear value (RGB, or R in x) of the 3 × 3 texels around `uv` (about one pixel's
    /// footprint at these bookmarks) and the spread of the 5 × 5 around it (max − min of the
    /// luminance, over their mean).
    fn patch(&self, uv: glam::Vec2) -> (Vec3, f32) {
        let (w, h) = (self.tex.width as i64, self.tex.height as i64);
        let bpp = match self.tex.format {
            uni_data::TexFormat::R8Unorm => 1,
            _ => 4,
        };
        let cx = (f64::from(uv.x) * w as f64).floor() as i64;
        let cy = (f64::from(uv.y) * h as f64).floor() as i64;
        let (mut sum, mut sum5) = (Vec3::ZERO, 0.0_f32);
        let (mut lo, mut hi) = (f32::INFINITY, 0.0_f32);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let x = (cx + dx).rem_euclid(w);
                let y = (cy + dy).clamp(0, h - 1);
                let i = ((y * w + x) * bpp) as usize;
                let px = &self.tex.mips[0][i..i + bpp as usize];
                let v = if bpp == 1 {
                    Vec3::splat(f32::from(px[0]) / 255.0)
                } else {
                    Vec3::new(srgb(px[0]), srgb(px[1]), srgb(px[2]))
                };
                let y_lum = luminance(v);
                if dx.abs() <= 1 && dy.abs() <= 1 {
                    sum += v;
                }
                sum5 += y_lum;
                lo = lo.min(y_lum);
                hi = hi.max(y_lum);
            }
        }
        let m = sum5 / 25.0;
        (
            sum / 9.0,
            if m > 0.0 {
                (hi - lo) / m
            } else {
                f32::INFINITY
            },
        )
    }
}

/// sRGB code value → linear (IEC 61966-2-1 §5.2).
fn srgb(code: u8) -> f32 {
    let c = f64::from(code) / 255.0;
    (if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }) as f32
}

/// `samples` for a mapped body (plan rows 3–5): 16 pixels in low-gradient map areas, lit and
/// seen at μ₀, μ > 0.25, with the luminance the CPU twins expect there [cd/m²]: B1's
/// reflectance (the 2k map copy × `albedo_scale`, Earth's clouds as areal cover, W-B2), A3's
/// BRDF and phase correction, the sunlight at the body, and for a body with an atmosphere B3's
/// ground lighting (sun transmittance and skylight, `ground_lighting`) and the view path
/// (`segment_radiance`: transmittance and in-scatter; a sphere of the preset's radii). Eclipses
/// are not modelled (none at these bookmarks).
fn map_samples(input: &Input<'_>, cam: &Cam, name: &str) -> Vec<Sample> {
    let world = input.world;
    let Some(b) = body(world, &input.frame.snapshot, name) else {
        return Vec::new();
    };
    let Some(naif) = b.naif else {
        return Vec::new();
    };
    let Some(albedo) = CpuMap::load(input, naif, uni_data::MapKind::Albedo) else {
        return Vec::new();
    };
    let clouds = CpuMap::load(input, naif, uni_data::MapKind::Clouds);
    let preset = uni_atmo::preset_for(naif);
    let sun_color = {
        let c = uni_core::blackbody_linear_srgb(uni_core::TEFF_SUN_K);
        c / luminance(c)
    };
    let phase = phase_rad(b);
    let corr = uni_physics::photometry::planet_phase_correction(naif, b.brdf, phase)
        .min(uni_sim::snapshot::MAX_PHASE_CORRECTION);
    let e = b.sun_illuminance_lux;
    let inv = b.orient.conjugate();
    let mut found = Vec::new();
    let step = 24.0;
    let (w, h) = (cam.size.x, cam.size.y);
    let mut y = step / 2.0;
    while y < h {
        let mut x = step / 2.0;
        while x < w {
            let p = DVec2::new(x.floor() + 0.5, y.floor() + 0.5);
            x += step;
            let dir = cam.ray(p);
            let Some(t) = hit_body(b, dir) else { continue };
            let xr = dir * t - b.rel_pos_m;
            let pb = inv * xr;
            let lon = pb.y.atan2(pb.x) as f32;
            let lat = pb.z.atan2(pb.truncate().length()) as f32;
            let normal = (b.orient * (pb / (b.radii_m * b.radii_m))).normalize();
            let s = (b.sun_rel_m - xr).normalize();
            let (mu0, mu) = (normal.dot(s), normal.dot(-dir));
            if mu0 < 0.25 || mu < 0.25 {
                continue;
            }
            let uv = uni_planet::equirect_uv(lon, lat, albedo.info.lon0_deg as f32);
            let (texel, spread) = albedo.patch(uv);
            if spread > MAP_FLAT_MAX {
                continue;
            }
            let mut rho = texel * albedo.info.albedo_scale;
            if let Some(cl) = &clouds {
                let uvc = uni_planet::equirect_uv(lon, lat, cl.info.lon0_deg as f32);
                let (cover, cspread) = cl.patch(uvc);
                if cspread > MAP_FLAT_MAX && cover.x > 0.02 {
                    continue;
                }
                let c = cover.x;
                rho = (1.0 - c) * rho + Vec3::splat(c * cl.info.albedo_scale);
            }
            let refl = uni_physics::photometry::reflectance(b.brdf, mu0, mu.max(1e-4)) as f32;
            let k = (e * corr) as f32 / std::f32::consts::PI;
            let expected = match &preset {
                Some(params) => {
                    let (sun_t, sky) = uni_atmo::ground::ground_lighting(
                        Some(params),
                        params.bottom_radius_m,
                        mu0,
                    );
                    let ground = rho * sun_color * k * (sun_t * refl + sky);
                    let (insc, trans) = uni_atmo::segment_radiance(params, -b.rel_pos_m, dir, t, s);
                    trans * ground + insc * sun_color * e as f32
                }
                None => rho * sun_color * k * refl,
            };
            found.push((
                spread,
                Sample {
                    x: p.x,
                    y: p.y,
                    expected_cd_m2: expected.as_dvec3().to_array(),
                    what: format!(
                        "{name} lon {:.2}° lat {:.2}° (map spread {:.1} %)",
                        lon.to_degrees(),
                        lat.to_degrees(),
                        100.0 * spread
                    ),
                },
            ));
        }
        y += step;
    }
    // The flattest 16 (the plan's "low-gradient map areas").
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    found
        .into_iter()
        .take(MAP_SAMPLES)
        .map(|(_, s)| s)
        .collect()
}

/// `flux` entries for the named sources (re-review r2 NPA): a catalogue star (its position at
/// the epoch, its V from the camera) or the Sun as a point sprite. `radius_px` is the footprint
/// the shader used (`stars3d::footprint_radius_px` of its off-axis display energy, the 2-px
/// minimum included, B4L's halo share); `mask` lists every other catalogue star or point whose
/// footprint reaches the annulus R ≤ ρ < 1.5 R.
fn flux_entries(input: &Input<'_>, cam: &Cam, names: &[&str]) -> Vec<FluxEntry> {
    let world = input.world;
    let frame = input.frame;
    let star_ev = input.ev100[1];
    let fov = cam.fov_y_rad as f32;
    let h = cam.size.y as f32;
    let halo = uni_sky::stars3d::STARS3D_HALO_FRACTION;
    let years = frame.ctx.epoch.seconds_since(uni_core::Epoch::J2000) / crate::scene::JULIAN_YEAR_S;
    let cam_pc = crate::scene::camera_icrs_pc(&cam.pos);
    let fwd = cam.forward();
    // (pixel, footprint radius, lux, catalogue index) of every catalogue star in or near frame.
    let footprint = |rel: DVec3, lux: f64| -> Option<(DVec2, f64)> {
        let p = cam.project(rel)?;
        let cos = rel.normalize().dot(fwd) as f32;
        let e = uni_sky::stars::star_display_energy_off_axis(lux as f32, fov, h, star_ev, cos);
        uni_sky::stars3d::star_is_visible(e, halo)
            .then(|| (p, f64::from(uni_sky::stars3d::footprint_radius_px(e, halo))))
    };
    let mut others: Vec<(DVec2, f64, Option<u32>, f64)> = Vec::new();
    for (i, s) in world.sim.star_data().iter().enumerate() {
        if frame.snapshot.suppressed_stars.contains(&(i as u32)) {
            continue;
        }
        let rel = (star_pos_pc(s, years) - cam_pc) * PC;
        if rel.normalize_or_zero().dot(fwd) < 0.1 {
            continue;
        }
        let d_pc = rel.length() / PC;
        let lux =
            uni_core::star_illuminance_lux(f64::from(s.abs_mag_v) + 5.0 * (d_pc / 10.0).log10());
        if let Some((p, r)) = footprint(rel, lux) {
            others.push((p, r, Some(i as u32), lux));
        }
    }
    for pt in &frame.snapshot.points {
        let reflected = pt
            .id
            .and_then(|id| world.sim.objects().get(id as usize))
            .is_some_and(|o| crate::scene::is_reflected(o.kind));
        let lux = pt.illuminance_lux
            * if reflected {
                frame.reflected_scale
            } else {
                1.0
            };
        if let Some((p, r)) = footprint(pt.rel_pos_m, lux) {
            others.push((p, r, None, pt.illuminance_lux));
        }
    }
    let mut out = Vec::new();
    for &name in names {
        let Some(id) = world.sim.find(name) else {
            continue;
        };
        let info = &world.sim.objects()[id as usize];
        let (rel, lux, index) = match info.catalog_index {
            Some(i) => {
                let s = &world.sim.star_data()[i as usize];
                let rel = (star_pos_pc(s, years) - cam_pc) * PC;
                let d_pc = rel.length() / PC;
                let v = f64::from(s.abs_mag_v) + 5.0 * (d_pc / 10.0).log10();
                (rel, uni_core::star_illuminance_lux(v), Some(i))
            }
            None => {
                let Some(pt) = frame.snapshot.points.iter().find(|p| p.id == Some(id)) else {
                    continue;
                };
                (pt.rel_pos_m, pt.illuminance_lux, None)
            }
        };
        let Some((p, r)) = footprint(rel, lux) else {
            continue;
        };
        let r = r.max(f64::from(uni_sky::STARS3D_MIN_RADIUS_PX));
        let rest = || {
            others.iter().filter(move |(q, _, i, _)| {
                !(index.is_some() && *i == index) && (*q - p).length() > 0.5
            })
        };
        let mask = rest()
            .filter(|(q, rq, _, _)| (*q - p).length() - rq < 1.5 * r)
            .map(|(q, rq, _, _)| [q.x, q.y, *rq])
            .collect();
        let neighbour_lux = rest()
            .filter(|(q, _, _, _)| (*q - p).length() < r)
            .map(|(_, _, _, l)| l)
            .sum();
        out.push(FluxEntry {
            name: name.to_owned(),
            x: p.x,
            y: p.y,
            radius_px: r,
            expected_lux: lux,
            mask,
            neighbour_lux,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::UVec2;
    use uni_core::CameraView;

    fn cam(orient: DQuat, fov: f64) -> Cam {
        Cam::new(
            &CameraView {
                pos: UniversePos::default(),
                orient,
                fov_y_rad: fov,
                velocity_mps: DVec3::ZERO,
            },
            UVec2::new(1920, 1080),
        )
    }

    #[test]
    fn projection_and_rays_round_trip() {
        let c = cam(
            DQuat::from_rotation_z(0.4) * DQuat::from_rotation_x(0.3),
            60_f64.to_radians(),
        );
        for p in [
            DVec2::new(960.0, 540.0),
            DVec2::new(0.5, 0.5),
            DVec2::new(1919.5, 1079.5),
            DVec2::new(100.25, 900.75),
        ] {
            let q = c.project(c.ray(p) * 1e9).unwrap();
            assert!((q - p).length() < 1e-6, "{p} -> {q}");
        }
        assert!(
            c.ray(DVec2::new(960.0, 540.0))
                .abs_diff_eq(c.forward(), 1e-12)
        );
        assert!(c.project(-c.forward()).is_none());
        // Ω_c = (2 tan 30° / 1080)², and cos³θ off axis.
        let side = 2.0 * 30_f64.to_radians().tan() / 1080.0;
        assert!((c.pixel_solid_angle(DVec2::new(960.0, 540.0)) - side * side).abs() < 1e-18);
        assert!(c.pixel_solid_angle(DVec2::new(0.5, 0.5)) < side * side * 0.7);
    }

    #[test]
    fn ellipsoid_hits_and_tangent_altitudes() {
        let radii = DVec3::new(2.0, 2.0, 1.0);
        let q = DQuat::IDENTITY;
        // From x = 10 toward the centre: the near surface at x = 2.
        assert_eq!(
            hit_ellipsoid(DVec3::new(10.0, 0.0, 0.0), DVec3::NEG_X, q, radii),
            Some(8.0)
        );
        assert_eq!(
            hit_ellipsoid(DVec3::new(10.0, 0.0, 0.0), DVec3::X, q, radii),
            None
        );
        // From inside: the exit.
        assert_eq!(hit_ellipsoid(DVec3::ZERO, DVec3::Z, q, radii), Some(1.0));
        assert!((ellipsoid_radius(DVec3::Z, q, radii) - 1.0).abs() < 1e-12);
        assert!((ellipsoid_radius(DVec3::Y, q, radii) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn thin_keeps_at_most_256_evenly() {
        let pts: Vec<DVec2> = (0..1000).map(|i| DVec2::new(f64::from(i), 0.0)).collect();
        let t = thin(pts);
        assert_eq!(t.len(), MAX_REGION_SAMPLES);
        assert_eq!(t[0], [0.0, 0.0]);
        assert!(t[255][0] > 990.0);
        assert_eq!(thin(vec![DVec2::ONE]).len(), 1);
    }

    #[test]
    fn lit_px_counts_world_light_over_the_meter_threshold() {
        let ev = [10.0, -6.5];
        let pre = uni_gfx::exposure_scale(10.0);
        let img = HdrImage {
            size: UVec2::new(3, 1),
            pixels: vec![
                [2e-3 * pre; 4],
                [0.5e-3 * pre, 0.5e-3 * pre, 0.5e-3 * pre, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        };
        assert_eq!(lit_px(&img, ev), 2);
    }
}
