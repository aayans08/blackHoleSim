#version 330

// Atmospheric scattering fragment shader
// Simplified Rayleigh + Mie scattering for M3 optimization

uniform sampler2D p3d_Texture0;
uniform vec3 sunDirection;
uniform vec3 planetCenter;
uniform float atmosphereRadius;
uniform float planetRadius;

in vec2 texcoord;
in vec3 v_position;
in vec3 v_normal;
in vec3 v_view_dir;

out vec4 fragColor;

// Rayleigh scattering coefficient (wavelength dependent)
const vec3 betaR = vec3(5.8e-6, 13.5e-6, 33.1e-6);

// Mie scattering coefficient
const float betaM = 2.0e-5;

// Scattering scale heights
const float HR = 8000.0;  // Rayleigh
const float HM = 1200.0;  // Mie

float rayleighPhase(float cosTheta) {
    return 3.0 / (16.0 * 3.14159) * (1.0 + cosTheta * cosTheta);
}

float miePhase(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

void main() {
    // Get base planet texture
    vec4 baseColor = texture(p3d_Texture0, texcoord);

    // Calculate lighting
    float NdotL = max(dot(v_normal, sunDirection), 0.0);

    // Simple atmospheric rim lighting effect
    float rim = 1.0 - max(dot(v_view_dir, v_normal), 0.0);
    rim = pow(rim, 3.0);

    // Blue atmospheric glow
    vec3 atmosphereColor = vec3(0.4, 0.6, 1.0) * rim;

    // Combine
    vec3 finalColor = baseColor.rgb * NdotL + atmosphereColor * 0.5;

    fragColor = vec4(finalColor, baseColor.a);
}
