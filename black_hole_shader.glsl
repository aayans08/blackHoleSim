#version 330

// Vertex Shader
#ifdef VERTEX_SHADER
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 texcoord;
uniform mat4 p3d_ModelViewProjectionMatrix;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    texcoord = p3d_MultiTexCoord0;
}
#endif

// Fragment Shader
#ifdef FRAGMENT_SHADER
in vec2 texcoord;
out vec4 fragColor;

uniform vec3 camera_pos;
uniform vec3 black_hole_pos;
uniform float schwarzschild_radius;
uniform sampler2D environment_map;
uniform float time;

const int MAX_STEPS = 100;
const float STEP_SIZE = 0.1;
const float PI = 3.14159265359;

// Ray trace through curved spacetime
vec3 trace_ray(vec3 origin, vec3 direction) {
    vec3 pos = origin;
    vec3 dir = normalize(direction);

    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 to_bh = black_hole_pos - pos;
        float r = length(to_bh);

        // Event horizon check
        if(r < schwarzschild_radius * 1.01) {
            return vec3(0.0, 0.0, 0.0); // Pure black
        }

        // Gravitational deflection (simplified Schwarzschild metric)
        float deflection_strength = schwarzschild_radius / (r * r);
        vec3 deflection = normalize(to_bh) * deflection_strength;

        // Update direction (light bending)
        dir = normalize(dir + deflection * STEP_SIZE);

        // Move along ray
        pos += dir * STEP_SIZE;

        // Check if escaped to infinity
        if(r > schwarzschild_radius * 100.0) {
            // Sample environment map
            float theta = atan(dir.y, dir.x);
            float phi = acos(dir.z);
            vec2 uv = vec2(theta / (2.0 * PI) + 0.5, phi / PI);
            return texture(environment_map, uv).rgb;
        }
    }

    return vec3(0.01, 0.01, 0.02); // Deep space fallback
}

// Accretion disk rendering
vec3 accretion_disk_color(vec3 pos, vec3 dir) {
    // Check intersection with disk plane (z = 0)
    float t = -pos.z / dir.z;

    if(t > 0.0) {
        vec3 hit = pos + dir * t;
        vec3 to_bh = black_hole_pos - hit;
        float r = length(to_bh.xy); // Distance in xy plane

        // Disk boundaries
        float inner = schwarzschild_radius * 3.0; // ISCO
        float outer = schwarzschild_radius * 10.0;

        if(r > inner && r < outer) {
            // Temperature-based color (hotter at inner edge)
            float temp = 1.0 - (r - inner) / (outer - inner);
            temp = pow(temp, 0.5);

            // Blackbody radiation colors
            vec3 color;
            if(temp > 0.8) {
                color = vec3(0.8, 0.9, 1.0); // Blue-white (hot)
            } else if(temp > 0.5) {
                color = vec3(1.0, 1.0, 0.8); // White
            } else if(temp > 0.3) {
                color = vec3(1.0, 0.8, 0.4); // Yellow-orange
            } else {
                color = vec3(1.0, 0.3, 0.1); // Orange-red (cooler)
            }

            // Add some rotation-based brightness variation
            float angle = atan(hit.y, hit.x) + time * 0.5;
            float brightness = 0.7 + 0.3 * sin(angle * 5.0 + r);

            // Doppler effect (approaching side brighter)
            float doppler = 1.0 + 0.3 * sin(angle);

            return color * temp * brightness * doppler * 2.0;
        }
    }

    return vec3(0.0);
}

void main() {
    // Convert screen coords to ray direction
    vec2 uv = texcoord * 2.0 - 1.0;

    // Camera ray
    vec3 ray_origin = camera_pos;
    vec3 ray_dir = normalize(vec3(uv.x, uv.y, -1.0));

    // Trace through curved spacetime
    vec3 color = trace_ray(ray_origin, ray_dir);

    // Add accretion disk
    vec3 disk_color = accretion_disk_color(ray_origin, ray_dir);
    color += disk_color;

    // HDR tone mapping
    color = color / (color + vec3(1.0));

    // Gamma correction
    color = pow(color, vec3(1.0/2.2));

    fragColor = vec4(color, 1.0);
}
#endif
