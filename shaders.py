"""
Shader definitions for ultra-realistic black hole rendering.
Organized shader library with HDR, bloom, PBR, and advanced effects.
"""

class Shaders:
    """Collection of GLSL shaders for realistic rendering pipeline."""

    # Basic vertex shader for fullscreen quad
    FULLSCREEN_VERTEX = '''
    #version 330 core
    in vec2 in_position;
    in vec2 in_texcoord;
    out vec2 v_texcoord;

    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        v_texcoord = in_texcoord;
    }
    '''

    # Main scene rendering shader with HDR output
    MAIN_SCENE_FRAGMENT = '''
    #version 330 core

    uniform vec2 u_resolution;
    uniform float u_time;
    uniform float u_schwarzschild_radius;
    uniform vec3 u_camera_rotation;
    uniform float u_zoom;
    uniform sampler2D u_planet_texture;
    uniform sampler2D u_planet_normal;
    uniform sampler2D u_planet_specular;
    uniform bool u_use_textures;

    in vec2 v_texcoord;
    layout(location = 0) out vec4 fragColor;      // HDR color
    layout(location = 1) out vec4 brightColor;    // Bright areas for bloom

    const float PI = 3.14159265359;
    const int MAX_STEPS = 500;
    const float MAX_DIST = 150.0;

    // Rotation matrices
    mat3 rotationMatrixX(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat3(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
    }

    mat3 rotationMatrixY(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat3(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
    }

    // Schwarzschild metric for gravitational effects
    float schwarzschildMetric(float r) {
        if (r <= u_schwarzschild_radius * 1.0001) return 0.0;
        return 1.0 - u_schwarzschild_radius / r;
    }

    // High quality noise function
    float hash(vec3 p) {
        return fract(sin(dot(p, vec3(12.9898, 78.233, 54.321))) * 43758.5453);
    }

    float noise(vec3 p) {
        vec3 i = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);

        float n = i.x + i.y * 57.0 + 113.0 * i.z;
        return mix(
            mix(mix(hash(vec3(n)), hash(vec3(n + 1.0)), f.x),
                mix(hash(vec3(n + 57.0)), hash(vec3(n + 58.0)), f.x), f.y),
            mix(mix(hash(vec3(n + 113.0)), hash(vec3(n + 114.0)), f.x),
                mix(hash(vec3(n + 170.0)), hash(vec3(n + 171.0)), f.x), f.y), f.z);
    }

    // Fresnel-Schlick approximation for realistic reflections
    vec3 fresnelSchlick(float cosTheta, vec3 F0) {
        return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
    }

    // Cook-Torrance BRDF for physically-based rendering
    float distributionGGX(vec3 N, vec3 H, float roughness) {
        float a = roughness * roughness;
        float a2 = a * a;
        float NdotH = max(dot(N, H), 0.0);
        float NdotH2 = NdotH * NdotH;

        float num = a2;
        float denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = PI * denom * denom;

        return num / denom;
    }

    float geometrySchlickGGX(float NdotV, float roughness) {
        float r = (roughness + 1.0);
        float k = (r * r) / 8.0;

        float num = NdotV;
        float denom = NdotV * (1.0 - k) + k;

        return num / denom;
    }

    float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float ggx2 = geometrySchlickGGX(NdotV, roughness);
        float ggx1 = geometrySchlickGGX(NdotL, roughness);

        return ggx1 * ggx2;
    }

    // Rayleigh scattering for realistic atmospheres
    vec3 rayleighScattering(float cosTheta, float height) {
        // Wavelength-dependent scattering (blue scatters more)
        vec3 scatterCoeff = vec3(0.58, 0.42, 0.35) * 0.0025;

        // Phase function
        float phase = 0.75 * (1.0 + cosTheta * cosTheta);

        // Density falloff with height
        float density = exp(-height * 8.0);

        return scatterCoeff * phase * density;
    }

    // Mie scattering for haze/dust
    vec3 mieScattering(float cosTheta, float height) {
        float g = 0.76; // Anisotropy factor
        float g2 = g * g;

        // Henyey-Greenstein phase function
        float num = 1.0 - g2;
        float denom = pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
        float phase = (num / denom) / (4.0 * PI);

        float density = exp(-height * 1.2);

        return vec3(0.4, 0.4, 0.4) * phase * density * 0.003;
    }

    // Enhanced planet rendering with PBR and textures
    vec3 renderPlanet(vec3 rayDir, vec3 planetCenter, float planetRadius,
                      vec3 sunDir, vec3 viewPos) {
        vec3 toPlanet = planetCenter;
        float dotPlanet = dot(normalize(rayDir), normalize(toPlanet));

        if (dotPlanet > 0.85) {
            float angularDist = acos(dotPlanet);

            if (angularDist < planetRadius) {
                float normalizedDist = angularDist / planetRadius;
                vec3 surfaceNormal = normalize(rayDir - normalize(toPlanet) * dotPlanet);

                // Spherical UV mapping
                float lon = atan(surfaceNormal.x, surfaceNormal.z);
                float lat = asin(surfaceNormal.y);
                vec2 uv = vec2(lon / PI * 0.5 + 0.5, lat / (PI * 0.5) * 0.5 + 0.5);

                // Get surface properties
                vec3 albedo;
                vec3 normal = surfaceNormal;
                float roughness = 0.8;
                float metallic = 0.0;
                float specular = 0.0;

                if (u_use_textures) {
                    // Sample textures
                    albedo = texture(u_planet_texture, uv).rgb;

                    // Normal mapping (simplified)
                    vec3 normalMap = texture(u_planet_normal, uv).rgb * 2.0 - 1.0;
                    // Transform to world space (simplified)
                    normal = normalize(surfaceNormal + normalMap * 0.3);

                    // Specular map (water = shiny)
                    specular = texture(u_planet_specular, uv).r;
                    roughness = 1.0 - specular * 0.7;
                } else {
                    // Procedural surface
                    float continents = noise(vec3(lon * 2.0, lat * 2.0, 0.0));
                    continents += noise(vec3(lon * 5.0, lat * 5.0, 10.0)) * 0.5;
                    continents = smoothstep(0.45, 0.55, continents);

                    vec3 ocean = vec3(0.02, 0.08, 0.2);
                    vec3 land = vec3(0.2, 0.45, 0.15);
                    albedo = mix(ocean, land, continents);

                    specular = (1.0 - continents) * 0.8;
                    roughness = mix(0.1, 0.9, continents);
                }

                // Cloud layer
                float clouds = noise(vec3(lon * 8.0 + u_time * 0.005, lat * 8.0, 80.0));
                clouds = smoothstep(0.5, 0.6, clouds);
                albedo = mix(albedo, vec3(1.0), clouds * 0.8);

                // PBR Lighting calculation
                vec3 N = normal;
                vec3 V = normalize(viewPos - planetCenter);
                vec3 L = normalize(sunDir);
                vec3 H = normalize(V + L);

                // Light intensity
                float NdotL = max(dot(N, L), 0.0);

                // Cook-Torrance BRDF
                vec3 F0 = vec3(0.04);
                F0 = mix(F0, albedo, metallic);

                vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                float NDF = distributionGGX(N, H, roughness);
                float G = geometrySmith(N, V, L, roughness);

                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
                vec3 specularBRDF = numerator / denominator;

                // Energy conservation
                vec3 kS = F;
                vec3 kD = vec3(1.0) - kS;
                kD *= 1.0 - metallic;

                // Final lighting
                vec3 Lo = (kD * albedo / PI + specularBRDF) * vec3(3.0) * NdotL;

                // Ambient
                vec3 ambient = vec3(0.03) * albedo;

                // Atmospheric scattering
                float atmosphere = pow(normalizedDist, 0.5);
                float height = normalizedDist;

                vec3 rayleigh = rayleighScattering(dot(V, L), height);
                vec3 mie = mieScattering(dot(V, L), height);
                vec3 atmosColor = (rayleigh + mie) * vec3(3.0);

                vec3 finalColor = Lo + ambient;
                finalColor = mix(finalColor, atmosColor, atmosphere * 0.3);

                // Edge fade
                float edgeFade = 1.0 - smoothstep(0.85, 1.0, normalizedDist);

                return finalColor * edgeFade;
            }
        }

        return vec3(0.0);
    }

    // Star field with proper circular stars
    vec3 renderStars(vec3 dir) {
        vec3 stars = vec3(0.0);

        // Spherical coordinates for undistorted stars
        float theta = atan(dir.y, dir.x);
        float phi = acos(dir.z);
        vec2 sphereCoord = vec2(theta, phi) * 10.0;

        // Multiple layers of stars
        // Bright foreground stars
        vec2 brightGrid = floor(sphereCoord * 3.0);
        vec2 brightPos = fract(sphereCoord * 3.0) - 0.5;
        float brightHash = hash(vec3(brightGrid, 0.0));

        if (brightHash > 0.985) {
            float dist = length(brightPos);
            if (dist < 0.15) {
                float star = exp(-dist * dist * 100.0) * 3.0;
                float glow = exp(-dist * 20.0);
                vec3 starColor = mix(vec3(1.0, 0.95, 0.8), vec3(0.8, 0.9, 1.0), brightHash);
                stars += starColor * (star + glow * 0.5);
            }
        }

        // Medium stars
        vec2 medGrid = floor(sphereCoord * 8.0);
        vec2 medPos = fract(sphereCoord * 8.0) - 0.5;
        float medHash = hash(vec3(medGrid + 100.0, 0.0));

        if (medHash > 0.97) {
            float dist = length(medPos);
            if (dist < 0.1) {
                float star = exp(-dist * dist * 200.0) * 2.0;
                stars += vec3(1.0, 1.0, 0.95) * star;
            }
        }

        // Background stars
        vec2 bgGrid = floor(sphereCoord * 20.0);
        vec2 bgPos = fract(sphereCoord * 20.0) - 0.5;
        float bgHash = hash(vec3(bgGrid + 200.0, 0.0));

        if (bgHash > 0.92) {
            float dist = length(bgPos);
            if (dist < 0.05) {
                float star = exp(-dist * dist * 500.0) * bgHash;
                stars += vec3(0.95, 0.95, 1.0) * star;
            }
        }

        return stars;
    }

    // Main ray tracing function with HDR output
    vec4 traceRay(vec3 ro, vec3 rd) {
        vec3 color = vec3(0.0);
        vec3 pos = ro;
        vec3 dir = rd;

        float minDistToHole = 1000.0;

        // Ray marching through curved spacetime
        for (int i = 0; i < MAX_STEPS; i++) {
            float r = length(pos);
            minDistToHole = min(minDistToHole, r);

            // Event horizon
            if (r < u_schwarzschild_radius * 1.05) {
                if (r > u_schwarzschild_radius * 1.0 && r < u_schwarzschild_radius * 1.02) {
                    float edgeGlow = 1.0 - (r - u_schwarzschild_radius) / (u_schwarzschild_radius * 0.02);
                    color += vec3(0.05, 0.05, 0.05) * edgeGlow * 0.1;
                }
                return vec4(color, 1.0);
            }

            if (r > MAX_DIST) break;

            // Gravitational lensing
            if (r < u_schwarzschild_radius * 20.0) {
                float deflection = u_schwarzschild_radius / (r * r);
                vec3 toCenter = -normalize(pos);
                vec3 perpComponent = dir - dot(dir, toCenter) * toCenter;

                if (length(perpComponent) > 0.001) {
                    vec3 deflectDir = cross(toCenter, cross(toCenter, dir));
                    if (length(deflectDir) > 0.001) {
                        deflectDir = normalize(deflectDir);
                        float bendStrength = deflection * 0.2;
                        if (r < u_schwarzschild_radius * 3.0) {
                            bendStrength *= 2.0;
                        }
                        dir = normalize(dir + deflectDir * bendStrength);
                    }
                }
            }

            // Step forward
            float metric = schwarzschildMetric(r);
            float stepSize = max(0.005, min(0.3, r * 0.008)) * metric;
            pos += dir * stepSize;
        }

        // Photon sphere glow
        if (minDistToHole < u_schwarzschild_radius * 1.8 && minDistToHole > u_schwarzschild_radius) {
            float glowStrength = 1.0 - (minDistToHole - u_schwarzschild_radius) / (u_schwarzschild_radius * 0.8);
            color += vec3(0.15, 0.12, 0.1) * glowStrength * 0.15;
        }

        // Background rendering
        vec3 bgColor = vec3(0.0);

        // Planet
        vec3 planetCenter = vec3(0.5, -0.3, 0.8);
        vec3 sunDir = normalize(vec3(1.0, 0.3, 0.5));
        bgColor += renderPlanet(dir, planetCenter, 0.35, sunDir, ro);

        // Stars
        bgColor += renderStars(dir);

        // Nebulae (subtle)
        float nebula = noise(dir * 3.0);
        nebula = pow(max(0.0, nebula - 0.5), 2.0);
        bgColor += vec3(0.02, 0.04, 0.06) * nebula;

        color += bgColor;

        return vec4(color, 1.0);
    }

    void main() {
        vec2 uv = (v_texcoord - 0.5) * 2.0;
        uv.x *= u_resolution.x / u_resolution.y;

        // Camera setup
        vec3 ro = vec3(0.0, 0.5, -35.0) * u_zoom;
        vec3 rd = normalize(vec3(uv.x * 0.4, uv.y * 0.4, 1.0));

        // Apply rotation
        mat3 rotX = rotationMatrixX(u_camera_rotation.x);
        mat3 rotY = rotationMatrixY(u_camera_rotation.y);
        mat3 rotation = rotY * rotX;

        ro = rotation * ro;
        rd = rotation * rd;

        // Trace ray (HDR output)
        vec4 col = traceRay(ro, rd);

        fragColor = col;

        // Extract bright areas for bloom
        float brightness = dot(col.rgb, vec3(0.2126, 0.7152, 0.0722));
        if (brightness > 1.0) {
            brightColor = vec4(col.rgb, 1.0);
        } else {
            brightColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
    '''

    # Gaussian blur shader (separable, horizontal pass)
    BLUR_HORIZONTAL_FRAGMENT = '''
    #version 330 core

    uniform sampler2D u_texture;
    uniform vec2 u_resolution;

    in vec2 v_texcoord;
    out vec4 fragColor;

    // 9-tap Gaussian blur
    const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    void main() {
        vec2 texelSize = 1.0 / u_resolution;
        vec3 result = texture(u_texture, v_texcoord).rgb * weights[0];

        for (int i = 1; i < 5; i++) {
            result += texture(u_texture, v_texcoord + vec2(texelSize.x * i, 0.0)).rgb * weights[i];
            result += texture(u_texture, v_texcoord - vec2(texelSize.x * i, 0.0)).rgb * weights[i];
        }

        fragColor = vec4(result, 1.0);
    }
    '''

    # Gaussian blur shader (vertical pass)
    BLUR_VERTICAL_FRAGMENT = '''
    #version 330 core

    uniform sampler2D u_texture;
    uniform vec2 u_resolution;

    in vec2 v_texcoord;
    out vec4 fragColor;

    const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    void main() {
        vec2 texelSize = 1.0 / u_resolution;
        vec3 result = texture(u_texture, v_texcoord).rgb * weights[0];

        for (int i = 1; i < 5; i++) {
            result += texture(u_texture, v_texcoord + vec2(0.0, texelSize.y * i)).rgb * weights[i];
            result += texture(u_texture, v_texcoord - vec2(0.0, texelSize.y * i)).rgb * weights[i];
        }

        fragColor = vec4(result, 1.0);
    }
    '''

    # Final composition with ACES tone mapping and bloom
    TONEMAP_FRAGMENT = '''
    #version 330 core

    uniform sampler2D u_scene;      // HDR scene
    uniform sampler2D u_bloom;      // Blurred bloom
    uniform float u_exposure;
    uniform float u_bloom_strength;

    in vec2 v_texcoord;
    out vec4 fragColor;

    // ACES Filmic Tone Mapping
    vec3 acesFilmic(vec3 x) {
        float a = 2.51;
        float b = 0.03;
        float c = 2.43;
        float d = 0.59;
        float e = 0.14;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }

    // Reinhard tone mapping (alternative)
    vec3 reinhard(vec3 color) {
        return color / (color + vec3(1.0));
    }

    void main() {
        vec3 hdrColor = texture(u_scene, v_texcoord).rgb;
        vec3 bloomColor = texture(u_bloom, v_texcoord).rgb;

        // Add bloom
        hdrColor += bloomColor * u_bloom_strength;

        // Exposure
        vec3 mapped = hdrColor * u_exposure;

        // Tone mapping
        mapped = acesFilmic(mapped);

        // Gamma correction
        mapped = pow(mapped, vec3(1.0 / 2.2));

        // Subtle vignette
        vec2 uv = v_texcoord * 2.0 - 1.0;
        float vignette = 1.0 - length(uv) * 0.2;
        mapped *= vignette;

        fragColor = vec4(mapped, 1.0);
    }
    '''
