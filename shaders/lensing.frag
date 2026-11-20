#version 330

// Simplified gravitational lensing shader for M3
// Based on Schwarzschild metric

uniform vec3 blackHolePosition;
uniform float schwarzschildRadius;
uniform sampler2D environmentMap;
uniform vec3 cameraPosition;

in vec2 texcoord;
in vec3 worldPos;

out vec4 fragColor;

const int MAX_STEPS = 30;  // Reduced for M3 performance
const float STEP_SIZE = 0.1;

vec3 calculateLensing(vec3 rayOrigin, vec3 rayDir) {
    vec3 pos = rayOrigin;
    vec3 dir = rayDir;

    // Ray marching through curved spacetime
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 toBH = blackHolePosition - pos;
        float dist = length(toBH);

        // Event horizon check
        if (dist < schwarzschildRadius * 1.5) {
            return vec3(0.0, 0.0, 0.0);  // Black
        }

        // Simplified deflection calculation
        float deflection = schwarzschildRadius / (dist * dist);
        dir += normalize(toBH) * deflection * STEP_SIZE;
        dir = normalize(dir);

        // Step forward
        pos += dir * STEP_SIZE;

        // Escape check
        if (dist > 1000.0) {
            break;
        }
    }

    // Sample environment with final ray direction
    // Simple spherical mapping
    float theta = atan(dir.z, dir.x);
    float phi = asin(dir.y);
    vec2 uv = vec2(theta / (2.0 * 3.14159) + 0.5, phi / 3.14159 + 0.5);

    return texture(environmentMap, uv).rgb;
}

void main() {
    vec3 rayOrigin = cameraPosition;
    vec3 rayDir = normalize(worldPos - cameraPosition);

    vec3 color = calculateLensing(rayOrigin, rayDir);

    fragColor = vec4(color, 1.0);
}
