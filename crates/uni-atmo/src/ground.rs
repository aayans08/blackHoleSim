//! The ground-lighting group the bodies pass binds at group 2 (Task B3).
//!
//! `shaders/ground_lighting.wgsl` declares the group and two functions the body shader (B1)
//! calls per pixel: `atmo_sun_transmittance(r_m, mu_s)` (the Sun's light left after the path
//! through the atmosphere) and `atmo_sky_irradiance(r_m, mu_s)` (skylight on the ground per unit
//! sun illuminance). Airless and unknown bodies get the neutral group: transmittance 1,
//! irradiance 0. [`ground_lighting`] is the CPU twin of both functions.
//!
//! [`GroundLighting::neutral`] builds the layout and the neutral group with 1 × 1 stand-in LUTs;
//! [`crate::AtmosphereSet::new`] adds one group per atmosphere over its real transmittance and
//! irradiance LUTs.
//!
//! **r_m for a surface point** is its radius in the atmosphere's frame
//! ([`crate::feature::atmosphere_frame`]): the atmosphere's bottom radius (`atmo_ground.radii.x`)
//! plus the point's height above the body's reference ellipsoid, not its distance from the
//! centre. The atmosphere follows the ellipsoid, so a point on the ground of an oblate Earth is
//! at the bottom whatever its latitude. mu_s is the cosine of the Sun's zenith angle about the
//! surface normal.
//!
//! B0's stub described r_m only as a "radius". This reading is B3's and needs the lead's ruling
//! (B3 review I2). A caller that passes `length(p_body)` instead lights Earth's equatorial
//! ground as if 7.1 km up and Mars's as if 6.7 km up. The B3 report quotes the effect from
//! `report_numbers`.

use glam::Vec3;
use uni_gfx::GpuContext;

use crate::params::AtmosphereParams;

/// Twin of `struct AtmoGround` in `shaders/ground_lighting.wgsl` (one vec4;
/// `atmo_ground_layout_matches_wgsl` checks it with naga).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmoGround {
    /// bottom radius [m], top radius [m], has_atmosphere (0 or 1), 0.
    pub radii: [f32; 4],
}

/// Format of the transmittance and irradiance LUTs (B3 brief: all LUTs Rgba16Float, which is
/// filterable, so the body shader samples them bilinearly).
pub const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Group 2 of the body pipeline: binding 0 `AtmoGround`, 1 the transmittance LUT, 2 the
/// irradiance LUT, 3 a filtering sampler; fragment and compute stages (B1's probes sample it in
/// compute).
fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
    let visibility = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    let lut = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<AtmoGround>() as u64),
            },
            count: None,
        },
        lut(1),
        lut(2),
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ]
}

/// The bind groups of the ground-lighting contract, one per atmosphere plus the neutral one.
///
/// Built once by [`crate::AtmosphereSet::new`] and shared with the bodies pass through
/// [`crate::AtmosphereSet::ground_lighting`]. B1 puts [`GroundLighting::bgl`] at group 2 of its
/// pipeline layout and binds [`GroundLighting::group`] per body.
pub struct GroundLighting {
    /// Layout of group 2 (bindings 0–3, see `ground_lighting.wgsl`).
    pub bgl: wgpu::BindGroupLayout,
    /// The neutral group: `has_atmosphere` 0, so transmittance 1 and irradiance 0.
    neutral: wgpu::BindGroup,
    /// Linear clamp-to-edge, shared by every group.
    sampler: wgpu::Sampler,
    /// One group per atmosphere, by NAIF id.
    groups: Vec<(i32, wgpu::BindGroup)>,
}

impl GroundLighting {
    /// The group for the body with NAIF id `naif`: its atmosphere's group for the bodies the
    /// [`crate::AtmosphereSet`] was built with (299, 399, 499 and 606 in M1); airless and
    /// unknown bodies (and `None`) get the neutral group.
    pub fn group(&self, naif: Option<i32>) -> &wgpu::BindGroup {
        naif.and_then(|id| self.groups.iter().find(|(n, _)| *n == id))
            .map_or(&self.neutral, |(_, g)| g)
    }

    /// Adds the group of body `naif` over its LUT views (`AtmoGround` = bottom, top, 1, 0).
    pub(crate) fn add(
        &mut self,
        gpu: &GpuContext,
        naif: i32,
        params: &AtmosphereParams,
        transmittance: &wgpu::TextureView,
        irradiance: &wgpu::TextureView,
    ) -> anyhow::Result<()> {
        let device = &gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmo ground"),
            size: size_of::<AtmoGround>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &uniform,
            0,
            bytemuck::bytes_of(&AtmoGround::from_params(params)),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmo ground bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(transmittance),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(irradiance),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the ground-lighting group of NAIF {naif}: {err}");
        }
        self.groups.retain(|(n, _)| *n != naif);
        self.groups.push((naif, group));
        Ok(())
    }

    /// The layout and the neutral group: `AtmoGround` all zero, 1 × 1 stand-in LUTs (wgpu
    /// zero-initialises them) and a linear clamp-to-edge sampler. Validation errors come back as
    /// an `Err`, not a device error.
    pub(crate) fn neutral(gpu: &GpuContext) -> anyhow::Result<Self> {
        let device = &gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo ground bgl"),
            entries: &layout_entries(),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmo ground neutral"),
            size: size_of::<AtmoGround>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &uniform,
            0,
            bytemuck::bytes_of(&AtmoGround { radii: [0.0; 4] }),
        );
        let stand_in = |label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: LUT_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let transmittance = stand_in("atmo neutral transmittance");
        let irradiance = stand_in("atmo neutral irradiance");
        let sampler = crate::lut::lut_sampler(device);
        let neutral = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmo ground neutral bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&transmittance),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&irradiance),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the neutral ground-lighting group: {err}");
        }
        Ok(Self {
            bgl,
            neutral,
            sampler,
            groups: Vec::new(),
        })
    }
}

impl AtmoGround {
    /// `radii` = (bottom, top, 1, 0) for an atmosphere.
    pub fn from_params(params: &AtmosphereParams) -> Self {
        Self {
            radii: [
                params.bottom_radius_m as f32,
                params.top_radius_m as f32,
                1.0,
                0.0,
            ],
        }
    }
}

/// CPU twin of `atmo_sun_transmittance` and `atmo_sky_irradiance`: (fraction of the Sun's
/// light reaching radius `r_m` with the Sun at zenith cosine `mu_s`, skylight irradiance per
/// unit sun illuminance), per channel. `None` (an airless body) gives (1, 0). With an atmosphere,
/// r is clamped to [bottom, top] as the shader does, the transmittance is 0 below the geometric
/// horizon ([`crate::transmittance`]), and the irradiance is [`crate::sky_irradiance`].
/// Non-finite input with an atmosphere gives (0, 0).
pub fn ground_lighting(params: Option<&AtmosphereParams>, r_m: f64, mu_s: f64) -> (Vec3, Vec3) {
    let Some(p) = params else {
        return (Vec3::ONE, Vec3::ZERO);
    };
    if !(r_m.is_finite() && mu_s.is_finite()) {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let r = r_m.clamp(p.bottom_radius_m, p.top_radius_m);
    (
        crate::lut::transmittance(p, r, mu_s),
        crate::lut::sky_irradiance(p, r, mu_s),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;
    use uni_gfx::{ShaderLib, validate_wgsl};
    use wgpu::naga;

    #[test]
    fn atmo_ground_layout_matches_wgsl() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let m = validate_wgsl(&lib.compose(crate::GROUND_LIGHTING_WGSL).unwrap()).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("AtmoGround"))
            .expect("struct AtmoGround");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("AtmoGround is not a struct");
        };
        assert_eq!(layouter[handle].size as usize, size_of::<AtmoGround>());
        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        assert_eq!(wgsl, [("radii".to_owned(), offset_of!(AtmoGround, radii))]);
        assert_eq!(layout_entries().map(|e| e.binding), [0, 1, 2, 3]);
    }

    /// The airless group: transmittance 1, irradiance 0 (CPU twin of the neutral WGSL path);
    /// with an atmosphere, 0 below the geometric horizon and the LUT twin above it.
    #[test]
    fn airless_ground_lighting_is_neutral() {
        for (r, mu) in [(1.0e6, 1.0), (6.4e6, -0.3), (3.0e6, 0.0), (0.0, 0.5)] {
            let (t, e) = ground_lighting(None, r, mu);
            assert_eq!((t, e), (glam::Vec3::ONE, glam::Vec3::ZERO));
        }
        let p = crate::params::earth();
        let (t, e) = ground_lighting(Some(&p), p.bottom_radius_m, -0.2);
        assert_eq!(t, glam::Vec3::ZERO);
        assert!(e.max_element() >= 0.0 && e.is_finite());
        let (t, e) = ground_lighting(Some(&p), p.bottom_radius_m, 1.0);
        assert!(
            (t - crate::lut::transmittance(&p, p.bottom_radius_m, 1.0))
                .abs()
                .max_element()
                < 1e-6
        );
        assert!(
            (e - crate::lut::sky_irradiance(&p, p.bottom_radius_m, 1.0))
                .abs()
                .max_element()
                < 1e-6
        );
        assert!(e.min_element() > 0.0);
    }
}
