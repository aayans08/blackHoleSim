#version 330

// Simplified PBR fragment shader

uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;

uniform vec3 lightDirection;
uniform vec3 lightColor;
uniform vec3 cameraPosition;

in vec2 texcoord;
in vec3 worldPos;
in vec3 worldNormal;
in vec3 viewPos;

out vec4 fragColor;

const float PI = 3.14159265359;

// Fresnel (Schlick approximation)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Normal Distribution Function (GGX/Trowbridge-Reitz)
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return a2 / denom;
}

// Geometry function (Smith's method)
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float ggx1 = NdotV / (NdotV * (1.0 - k) + k);
    float ggx2 = NdotL / (NdotL * (1.0 - k) + k);

    return ggx1 * ggx2;
}

void main() {
    // Sample textures
    vec3 albedo = texture(albedoMap, texcoord).rgb;
    vec3 normal = normalize(worldNormal);  // Could use normalMap
    float metallic = 0.0;  // Could use metallicMap
    float roughness = 0.5;  // Could use roughnessMap

    // Vectors
    vec3 N = normal;
    vec3 V = normalize(cameraPosition - worldPos);
    vec3 L = normalize(lightDirection);
    vec3 H = normalize(V + L);

    // Base reflectivity
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // Cook-Torrance BRDF
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
    vec3 specular = numerator / denominator;

    // Diffuse
    vec3 kD = vec3(1.0) - F;
    kD *= 1.0 - metallic;

    float NdotL = max(dot(N, L), 0.0);
    vec3 radiance = lightColor;

    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Ambient
    vec3 ambient = vec3(0.03) * albedo;
    vec3 color = ambient + Lo;

    // Gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    fragColor = vec4(color, 1.0);
}
