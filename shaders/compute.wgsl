// Black Hole Ray Tracer - Compute Shader
// Implements Schwarzschild geodesic integration with volumetric accretion disk

struct Camera {
    position: vec3<f32>,
    forward: vec3<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    fov: f32,
    aspect: f32,
    _padding: vec2<f32>,
}

struct BlackHole {
    position: vec3<f32>,
    schwarzschild_radius: f32,  // Rs = 29.54 km for 10 solar masses
    mass: f32,
    _padding: vec2<f32>,
}

struct DiskParams {
    inner_radius: f32,  // ISCO at 3 Rs
    outer_radius: f32,  // 12 Rs
    thickness: f32,     // Vertical scale height
    density: f32,       // Base density
    temperature_inner: f32,  // ~10^7 K
    temperature_outer: f32,  // ~10^4 K
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> black_hole: BlackHole;
@group(0) @binding(2) var<uniform> disk: DiskParams;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var environment_map: texture_2d<f32>;
@group(0) @binding(5) var env_sampler: sampler;

// Physical constants
const C: f32 = 299792.458;  // Speed of light in km/s
const G: f32 = 6.67430e-11; // Gravitational constant
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Schwarzschild metric and geodesic integration
struct GeodesicState {
    position: vec3<f32>,
    velocity: vec3<f32>,
    energy: f32,
    angular_momentum: f32,
}

// Calculate Schwarzschild metric components
fn schwarzschild_metric(r: f32, rs: f32) -> vec4<f32> {
    let f = 1.0 - rs / r;
    return vec4<f32>(
        -f,           // g_tt
        1.0 / f,      // g_rr
        r * r,        // g_θθ
        r * r         // g_φφ (simplified for equatorial)
    );
}

// Compute gravitational acceleration for geodesic (simplified)
fn geodesic_acceleration(pos: vec3<f32>, vel: vec3<f32>, rs: f32) -> vec3<f32> {
    let r = length(pos);
    if (r < rs * 1.01) {
        return vec3<f32>(0.0);
    }

    let r2 = r * r;
    let r3 = r2 * r;

    // Deflection toward black hole
    let to_bh = -pos / r;

    // Post-Newtonian correction for light deflection
    let deflection_strength = (1.5 * rs) / r2;

    // Perpendicular component of velocity
    let vr = dot(vel, to_bh);
    let v_perp = vel - vr * to_bh;

    return deflection_strength * (to_bh + v_perp / r);
}

// Integrate geodesic using simple Euler (faster for M3)
fn integrate_geodesic(state: ptr<function, GeodesicState>, dt: f32, rs: f32) {
    let accel = geodesic_acceleration((*state).position, (*state).velocity, rs);

    (*state).velocity += accel * dt;
    (*state).velocity = normalize((*state).velocity);  // Keep unit speed for light
    (*state).position += (*state).velocity * dt;
}

// Volumetric accretion disk density
fn disk_density(pos: vec3<f32>, bh_pos: vec3<f32>) -> f32 {
    let rel_pos = pos - bh_pos;
    let r_cylindrical = length(rel_pos.xz);
    let height = abs(rel_pos.y);

    // Check if within disk radial bounds
    if (r_cylindrical < disk.inner_radius || r_cylindrical > disk.outer_radius) {
        return 0.0;
    }

    // Vertical density profile (Gaussian)
    let scale_height = disk.thickness * (r_cylindrical / disk.inner_radius);
    let vertical_density = exp(-height * height / (2.0 * scale_height * scale_height));

    // Radial density profile (power law)
    let radial_density = pow(disk.inner_radius / r_cylindrical, 2.75);

    return disk.density * radial_density * vertical_density;
}

// Temperature at position in disk (power law)
fn disk_temperature(pos: vec3<f32>, bh_pos: vec3<f32>) -> f32 {
    let r = length((pos - bh_pos).xz);
    if (r < disk.inner_radius || r > disk.outer_radius) {
        return 0.0;
    }

    // Temperature power law: T ∝ r^(-0.75)
    let t = mix(
        disk.temperature_inner,
        disk.temperature_outer,
        pow((r - disk.inner_radius) / (disk.outer_radius - disk.inner_radius), 0.75)
    );

    return t;
}

// Blackbody radiation (Planck function simplified)
fn blackbody_emission(temperature: f32) -> vec3<f32> {
    if (temperature < 1000.0) {
        return vec3<f32>(0.0);
    }

    // Approximate blackbody color using Wien's law
    let t_normalized = temperature / 10000.0;

    var color: vec3<f32>;
    if (t_normalized < 0.5) {
        // Red to orange
        color = vec3<f32>(1.0, t_normalized * 2.0, 0.0);
    } else if (t_normalized < 1.0) {
        // Orange to yellow
        color = vec3<f32>(1.0, 1.0, (t_normalized - 0.5) * 2.0);
    } else {
        // White to blue
        let blue_shift = min((t_normalized - 1.0) * 0.5, 1.0);
        color = vec3<f32>(1.0 - blue_shift * 0.3, 1.0 - blue_shift * 0.2, 1.0);
    }

    // VERY conservative intensity to prevent blowout
    // Use logarithmic scale for extreme temperatures
    let temp_log = log(temperature / 1000.0) / log(10.0);  // log10(T/1000)
    let intensity = clamp(temp_log * 0.05, 0.001, 0.5);  // Max 0.5 per sample

    return color * intensity;
}

// Doppler shift factor
fn doppler_shift(pos: vec3<f32>, vel_obs: vec3<f32>, bh_pos: vec3<f32>) -> f32 {
    let r = length((pos - bh_pos).xz);
    if (r < disk.inner_radius || r > disk.outer_radius) {
        return 1.0;
    }

    // Keplerian velocity (circular orbit)
    let v_kepler = sqrt(black_hole.schwarzschild_radius * C * C / (2.0 * r));

    // Tangent direction (counter-clockwise in xz plane)
    let rel_pos = pos - bh_pos;
    let tangent = normalize(vec3<f32>(-rel_pos.z, 0.0, rel_pos.x));
    let vel_disk = tangent * v_kepler;

    // Doppler factor: 1 + v·n/c
    let doppler = 1.0 + dot(vel_disk, normalize(vel_obs)) / C;

    return doppler;
}

// Gravitational redshift
fn gravitational_redshift(r: f32, rs: f32) -> f32 {
    return sqrt(1.0 - rs / r);
}

// Sample environment map (stars)
fn sample_environment(direction: vec3<f32>) -> vec3<f32> {
    // Convert direction to spherical coordinates
    let theta = acos(direction.y);
    let phi = atan2(direction.z, direction.x);

    let uv = vec2<f32>(
        phi / TWO_PI + 0.5,
        theta / PI
    );

    return textureSampleLevel(environment_map, env_sampler, uv, 0.0).rgb;
}

// Main ray tracing function
fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>) -> vec3<f32> {
    var state = GeodesicState(
        ray_origin,
        normalize(ray_direction),
        0.0,
        0.0
    );

    let rs = black_hole.schwarzschild_radius;
    let event_horizon = rs * 1.01;  // Slightly outside actual horizon

    // Ray marching parameters
    let max_steps = 150;
    let dt = 0.1;  // Time step in geometric units
    let max_distance = 2000.0;  // km

    var accumulated_color = vec3<f32>(0.0);
    var transmittance = 1.0;

    for (var i = 0; i < max_steps; i++) {
        let r = length(state.position - black_hole.position);

        // Check if fell into black hole
        if (r < event_horizon) {
            return vec3<f32>(0.0);  // Black
        }

        // Check if escaped to infinity
        if (r > max_distance) {
            let env_color = sample_environment(normalize(state.velocity));
            return accumulated_color + env_color * transmittance;
        }

        // Integrate geodesic
        integrate_geodesic(&state, dt, rs);

        // Sample disk at current position
        let density = disk_density(state.position, black_hole.position);

        if (density > 0.0001) {
            let temperature = disk_temperature(state.position, black_hole.position);
            let emission = blackbody_emission(temperature);

            // Doppler shift - clamp to reasonable range
            let doppler = doppler_shift(state.position, state.velocity, black_hole.position);
            let doppler_clamped = clamp(doppler, 0.5, 2.0);  // Prevent extreme values
            let doppler_boosted = emission * doppler_clamped;  // Linear, not squared

            // Gravitational redshift
            let redshift = gravitational_redshift(r, rs);

            // Apply VERY conservative scaling (0.01 = 1% brightness)
            let final_emission = doppler_boosted * redshift * 0.01;

            // Safety clamp per-step to prevent NaN/infinity
            let safe_emission = min(final_emission, vec3<f32>(0.1));

            // Absorption (Beer's law)
            let absorption = density * dt * 0.3;
            let local_transmittance = exp(-absorption);

            // Emission-absorption integration
            accumulated_color += safe_emission * transmittance * (1.0 - local_transmittance);

            // Clamp accumulated color each step
            accumulated_color = min(accumulated_color, vec3<f32>(5.0));

            transmittance *= local_transmittance;

            // Early termination if opaque
            if (transmittance < 0.01) {
                return accumulated_color;
            }
        }
    }

    // Max steps reached, sample environment
    let env_color = sample_environment(normalize(state.velocity));
    return accumulated_color + env_color * transmittance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    let coords = vec2<i32>(global_id.xy);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Generate ray
    let uv = (vec2<f32>(coords) + 0.5) / vec2<f32>(dims);
    let ndc = uv * 2.0 - 1.0;

    let aspect_ratio = camera.aspect;
    let tan_half_fov = tan(camera.fov * 0.5);

    let ray_direction = normalize(
        camera.forward +
        camera.right * ndc.x * aspect_ratio * tan_half_fov +
        camera.up * ndc.y * tan_half_fov
    );

    // Trace ray
    let color = trace_ray(camera.position, ray_direction);

    // Write to output texture
    textureStore(output_texture, coords, vec4<f32>(color, 1.0));
}
