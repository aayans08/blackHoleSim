//! Surface materials of the bodies pass (Task B1; B0 stub, empty).
//!
//! B1 puts here what the bodies pass needs per body: the surface map (the BC mip chain when
//! `GpuContext::has_bc_compression()`, else the 2k uncompressed `fallback_file`; no CPU decoding
//! at runtime), the reflectance scale (`MapInfo::albedo_scale`), Earth's cloud layer composited
//! as ρ = (1 − c)·ρ_ground + c (areal coverage of opaque clouds, constraints §4a W-B2), Venus's
//! cloud-top map, and the uniform-albedo fallback ρ = `rho_from_geometric_albedo(brdf, p_V)` for
//! a body without a map. Materials are keyed by NAIF id, never sized per `ObjectId`.
