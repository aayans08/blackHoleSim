import moderngl
import numpy as np
import pygame
from pygame.locals import *
import struct
import math

class InterstellarRenderer:
    def __init__(self, engine, width=1280, height=800):
        self.engine = engine
        self.width = width
        self.height = height

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Black Hole - Interstellar Style")
        self.clock = pygame.time.Clock()

        self.ctx = moderngl.create_context()

        self.time = 0.0
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.zoom = 1.0

        self.setup_shaders()
        self.setup_buffers()

    def setup_shaders(self):
        vertex_shader = '''
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 v_texcoord;

        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            v_texcoord = in_texcoord;
        }
        '''

        # Enhanced Interstellar-style fragment shader
        fragment_shader = '''
        #version 330

        uniform vec2 u_resolution;
        uniform float u_time;
        uniform float u_schwarzschild_radius;
        uniform vec3 u_camera_rotation;
        uniform float u_zoom;

        in vec2 v_texcoord;
        out vec4 fragColor;

        const float PI = 3.14159265359;
        const int MAX_STEPS = 500;
        const float MAX_DIST = 150.0;
        const float EPSILON = 0.0001;

        mat3 rotationMatrixX(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return mat3(
                1.0, 0.0, 0.0,
                0.0, c, -s,
                0.0, s, c
            );
        }

        mat3 rotationMatrixY(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return mat3(
                c, 0.0, s,
                0.0, 1.0, 0.0,
                -s, 0.0, c
            );
        }

        float schwarzschildMetric(float r) {
            if (r <= u_schwarzschild_radius * 1.0001) return 0.0;
            return 1.0 - u_schwarzschild_radius / r;
        }

        // Hash functions for pseudo-random numbers
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
        }

        float hash3(vec3 p) {
            return fract(sin(dot(p, vec3(12.9898, 78.233, 54.321))) * 43758.5453);
        }

        // Noise function
        float noise(vec3 p) {
            vec3 i = floor(p);
            vec3 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);

            float n = i.x + i.y * 57.0 + 113.0 * i.z;
            return mix(mix(mix(hash3(vec3(n)), hash3(vec3(n + 1.0)), f.x),
                      mix(hash3(vec3(n + 57.0)), hash3(vec3(n + 58.0)), f.x), f.y),
                  mix(mix(hash3(vec3(n + 113.0)), hash3(vec3(n + 114.0)), f.x),
                      mix(hash3(vec3(n + 170.0)), hash3(vec3(n + 171.0)), f.x), f.y), f.z);
        }

        // Thin accretion disk check
        bool diskIntersection(vec3 pos, vec3 dir, out float t, out vec3 hitPoint) {
            // Disk in y=0 plane
            if (abs(dir.y) < 0.0001) return false;

            t = -pos.y / dir.y;
            if (t < 0.0 || t > 2.0) return false;

            hitPoint = pos + dir * t;
            float r = length(hitPoint.xz);

            // Innermost stable circular orbit to outer edge
            return r > u_schwarzschild_radius * 3.0 && r < u_schwarzschild_radius * 25.0;
        }

        // Get disk properties at a point
        vec3 getDiskProperties(vec3 pos) {
            float r = length(pos.xz);

            // Normalized radius for temperature
            float normalizedR = (r - u_schwarzschild_radius * 3.0) / (u_schwarzschild_radius * 22.0);
            normalizedR = clamp(normalizedR, 0.0, 1.0);

            // Temperature falloff (hotter inside)
            float temp = pow(1.0 - normalizedR, 2.5);

            // Orbital velocity for Doppler shift - with rotation
            float angle = atan(pos.z, pos.x);
            float orbitalSpeed = sqrt(u_schwarzschild_radius / r) * 1.5;
            vec3 velocity = vec3(-sin(angle), 0.0, cos(angle)) * orbitalSpeed;

            // Animated rotation and turbulence
            float rotationAngle = angle - u_time * orbitalSpeed * 0.5;  // Disk rotation

            // Moving hot spots and turbulence
            float swirl1 = sin(rotationAngle * 6.0 - r * 0.15 + u_time * 0.8 / sqrt(r));
            float swirl2 = cos(rotationAngle * 10.0 + r * 0.2 - u_time * 0.6 / sqrt(r));
            float turbulence = 0.85 + 0.1 * swirl1 + 0.05 * swirl2;

            // Add moving bright spots
            float hotSpot = sin(rotationAngle * 3.0 - u_time * 2.0 / sqrt(r));
            if (hotSpot > 0.7) {
                turbulence += (hotSpot - 0.7) * 0.5;
            }

            return vec3(temp, length(velocity), turbulence);
        }

        // Blackbody radiation color
        vec3 blackbodyColor(float temp) {
            vec3 color;

            if (temp > 0.9) {
                // Blue-white (hottest)
                color = vec3(0.7, 0.8, 1.0);
            } else if (temp > 0.75) {
                // White
                color = vec3(1.0, 1.0, 1.0);
            } else if (temp > 0.6) {
                // Yellow-white
                color = vec3(1.0, 1.0, 0.85);
            } else if (temp > 0.45) {
                // Yellow
                color = vec3(1.0, 0.95, 0.6);
            } else if (temp > 0.3) {
                // Orange
                color = vec3(1.0, 0.7, 0.3);
            } else if (temp > 0.15) {
                // Red-orange
                color = vec3(1.0, 0.4, 0.1);
            } else {
                // Deep red
                color = vec3(0.8, 0.2, 0.05);
            }

            return color;
        }

        // Main ray marching function
        vec4 traceRay(vec3 ro, vec3 rd) {
            vec3 color = vec3(0.0);
            vec3 pos = ro;
            vec3 dir = rd;

            float totalBrightness = 0.0;
            int diskCrossings = 0;
            float minDistToHole = 1000.0;

            // Track multiple disk crossings for lensing
            for (int i = 0; i < MAX_STEPS; i++) {
                float r = length(pos);
                minDistToHole = min(minDistToHole, r);

                // Event horizon with visible edge
                if (r < u_schwarzschild_radius * 1.05) {
                    // Add a very subtle edge definition (no purple)
                    if (r > u_schwarzschild_radius * 1.0 && r < u_schwarzschild_radius * 1.02) {
                        float edgeGlow = 1.0 - (r - u_schwarzschild_radius) / (u_schwarzschild_radius * 0.02);
                        color += vec3(0.05, 0.05, 0.05) * edgeGlow * 0.1;  // Very subtle gray
                    }

                    // Black hole shadow
                    if (diskCrossings == 0) {
                        return vec4(color, 1.0);
                    }
                    break;
                }

                if (r > MAX_DIST) break;

                // Strong gravitational lensing near black hole
                if (r < u_schwarzschild_radius * 20.0) {
                    float deflection = u_schwarzschild_radius / (r * r);
                    vec3 toCenter = -normalize(pos);

                    // Calculate perpendicular deflection
                    vec3 perpComponent = dir - dot(dir, toCenter) * toCenter;
                    if (length(perpComponent) > 0.001) {
                        perpComponent = normalize(perpComponent);
                        vec3 deflectDir = cross(toCenter, cross(toCenter, dir));
                        if (length(deflectDir) > 0.001) {
                            deflectDir = normalize(deflectDir);

                            // Stronger deflection closer to black hole
                            float bendStrength = deflection * 0.2;
                            if (r < u_schwarzschild_radius * 3.0) {
                                bendStrength *= 2.0;
                            }

                            dir = normalize(dir + deflectDir * bendStrength);
                        }
                    }
                }

                // ACCRETION DISK DISABLED - Black hole is not actively feeding
                /*
                // Check for disk intersection
                float t;
                vec3 hitPoint;
                if (diskIntersection(pos, dir, t, hitPoint)) {
                    vec3 props = getDiskProperties(hitPoint);
                    float temp = props.x;
                    float velocity = props.y;
                    float turbulence = props.z;

                    // Enhanced Doppler shift for rotation
                    vec3 viewDir = normalize(hitPoint - ro);
                    vec3 diskMotion = normalize(vec3(-hitPoint.z, 0, hitPoint.x));
                    float velocityProjection = velocity * dot(diskMotion, viewDir);
                    float doppler = 1.0 + velocityProjection * 0.4;  // Stronger Doppler effect

                    // Get color from temperature
                    vec3 diskColor = blackbodyColor(temp);

                    // Apply effects
                    diskColor *= turbulence;
                    diskColor *= doppler;

                    // Gravitational redshift
                    float metric = schwarzschildMetric(length(hitPoint.xz));
                    diskColor *= pow(metric, 0.4);

                    // Intensity based on temperature and distance
                    float intensity = temp * temp * turbulence;

                    // Add contribution with proper blending
                    float weight = intensity * exp(-totalBrightness * 0.3);
                    color += diskColor * weight * 1.5;
                    totalBrightness += intensity;

                    diskCrossings++;

                    // Limit crossings but allow multiple for lensing
                    if (diskCrossings >= 4 || totalBrightness > 3.0) break;
                }
                */

                // Step forward
                float metric = schwarzschildMetric(r);
                float stepSize = max(0.005, min(0.3, r * 0.008)) * metric;
                pos += dir * stepSize;
            }

            // Add subtle photon sphere glow if ray passed close (no purple)
            if (minDistToHole < u_schwarzschild_radius * 1.8 && minDistToHole > u_schwarzschild_radius) {
                float glowStrength = 1.0 - (minDistToHole - u_schwarzschild_radius) / (u_schwarzschild_radius * 0.8);
                color += vec3(0.15, 0.12, 0.1) * glowStrength * 0.15;  // Subtle warm glow
            }

            // SPACE BACKGROUND
            if (totalBrightness < 2.5) {
                vec3 bgColor = vec3(0.0);

                // Subtle nebula background (no purple)
                float nebula1 = noise(dir * 3.0);
                float nebula2 = noise(dir * 7.0);
                float nebula = pow(max(0.0, nebula1 * 0.5 + nebula2 * 0.3 - 0.3), 2.0);

                // Dark space with subtle blue/teal hints
                vec3 nebulaColor = vec3(0.02, 0.04, 0.06) * nebula;  // Very dark blue
                bgColor += nebulaColor;

                // DRAMATIC DEPTH STAR FIELD - from very close to very far

                // LARGE DETAILED PLANET WITH REALISTIC TEXTURE
                vec3 planetCenter = vec3(0.5, -0.3, 0.8);
                vec3 toPlanet = planetCenter;
                float dotPlanet = dot(normalize(dir), normalize(toPlanet));

                if (dotPlanet > 0.85) {  // Planet is in view
                    float planetRadius = 0.35;
                    vec3 rayDir = normalize(dir);
                    float angularDist = acos(dotPlanet);

                    if (angularDist < planetRadius) {
                        float normalizedDist = angularDist / planetRadius;
                        vec3 surfaceNormal = normalize(rayDir - normalize(toPlanet) * dotPlanet);

                        // Spherical coordinates for texture mapping
                        float lon = atan(surfaceNormal.x, surfaceNormal.z) / 3.14159;
                        float lat = asin(surfaceNormal.y) / 1.5708;

                        // DETAILED SURFACE FEATURES
                        // Continental plates (large scale)
                        float continents = noise(vec3(lon * 2.0, lat * 2.0, 0.0));
                        continents += noise(vec3(lon * 5.0, lat * 5.0, 10.0)) * 0.5;
                        continents += noise(vec3(lon * 12.0, lat * 12.0, 20.0)) * 0.25;

                        // Coastline details
                        float coastDetail = noise(vec3(lon * 30.0, lat * 30.0, 30.0)) * 0.1;
                        continents = smoothstep(0.48 - coastDetail, 0.52 + coastDetail, continents);

                        // Mountain ranges and elevation
                        float mountains = 0.0;
                        if (continents > 0.5) {
                            mountains = noise(vec3(lon * 40.0, lat * 40.0, 50.0));
                            mountains = pow(max(0.0, mountains - 0.4), 2.0) * 3.0;
                        }

                        // Rivers and lakes
                        float water = noise(vec3(lon * 50.0, lat * 50.0, 60.0));
                        water = smoothstep(0.7, 0.72, water) * (1.0 - continents);

                        // Vegetation zones
                        float vegetation = noise(vec3(lon * 20.0, lat * 20.0, 70.0));
                        vegetation *= continents * (1.0 - mountains);
                        vegetation = smoothstep(0.3, 0.7, vegetation);

                        // Desert zones (low vegetation, high temp)
                        float desert = 1.0 - abs(lat) * 2.0;  // Near equator
                        desert *= (1.0 - vegetation) * continents;

                        // Ice caps
                        float ice = smoothstep(0.6, 0.8, abs(lat));

                        // Cloud layer with movement
                        float clouds = noise(vec3(lon * 8.0 + u_time * 0.005, lat * 8.0, 80.0));
                        clouds += noise(vec3(lon * 15.0 - u_time * 0.003, lat * 15.0, 90.0)) * 0.5;
                        clouds = smoothstep(0.5, 0.6, clouds);

                        // Storm systems
                        float storms = noise(vec3(lon * 5.0 + u_time * 0.01, lat * 5.0, 100.0));
                        storms = pow(max(0.0, storms - 0.6), 3.0) * 2.0;

                        // SURFACE COLORS
                        vec3 deepOcean = vec3(0.02, 0.08, 0.2);
                        vec3 shallowOcean = vec3(0.05, 0.2, 0.4);
                        vec3 beachSand = vec3(0.76, 0.7, 0.5);
                        vec3 grassland = vec3(0.2, 0.45, 0.15);
                        vec3 forest = vec3(0.1, 0.3, 0.08);
                        vec3 desertSand = vec3(0.76, 0.64, 0.38);
                        vec3 rockColor = vec3(0.5, 0.45, 0.4);
                        vec3 snowColor = vec3(0.95, 0.95, 1.0);
                        vec3 cloudColor = vec3(1.0, 1.0, 1.0);

                        // Mix surface colors based on features
                        vec3 surfaceColor = deepOcean;

                        // Ocean depth variation
                        float oceanDepth = 1.0 - continents * 2.0;
                        surfaceColor = mix(shallowOcean, deepOcean, smoothstep(0.0, 0.5, oceanDepth));

                        // Land colors
                        if (continents > 0.01) {
                            vec3 landBase = mix(beachSand, grassland, smoothstep(0.0, 0.1, continents));
                            landBase = mix(landBase, forest, vegetation);
                            landBase = mix(landBase, desertSand, desert);
                            landBase = mix(landBase, rockColor, mountains);
                            surfaceColor = mix(surfaceColor, landBase, continents);
                        }

                        // Ice and snow
                        surfaceColor = mix(surfaceColor, snowColor, ice);
                        surfaceColor = mix(surfaceColor, snowColor * 0.9, mountains * (1.0 - desert));

                        // Clouds
                        surfaceColor = mix(surfaceColor, cloudColor, clouds * 0.8);
                        surfaceColor = mix(surfaceColor, vec3(0.3, 0.3, 0.35), storms * clouds);

                        // Atmospheric scattering
                        float atmosphere = pow(normalizedDist, 0.5);
                        vec3 atmosColor = vec3(0.4, 0.6, 1.0);
                        surfaceColor = mix(surfaceColor, atmosColor, atmosphere * 0.15);

                        // Day/night terminator with proper shading
                        vec3 sunDir = normalize(vec3(1.0, 0.3, 0.5));
                        float sunAngle = dot(surfaceNormal, sunDir);
                        float daylight = smoothstep(-0.1, 0.1, sunAngle);
                        float directLight = max(0.0, sunAngle);

                        // Night side city lights
                        float cityLights = 0.0;
                        if (daylight < 0.5 && continents > 0.5) {
                            cityLights = noise(vec3(lon * 100.0, lat * 100.0, 110.0));
                            cityLights = pow(max(0.0, cityLights - 0.7), 3.0) * (1.0 - clouds);
                        }

                        // Apply lighting
                        vec3 nightColor = surfaceColor * 0.1 + vec3(1.0, 0.8, 0.5) * cityLights * 0.3;
                        vec3 dayColor = surfaceColor * (0.3 + directLight * 0.7);
                        surfaceColor = mix(nightColor, dayColor, daylight);

                        // Specular reflection on water
                        if (continents < 0.3 && clouds < 0.3) {
                            vec3 viewDir = normalize(-rayDir);
                            vec3 reflectDir = reflect(-sunDir, surfaceNormal);
                            float specular = pow(max(0.0, dot(viewDir, reflectDir)), 20.0);
                            surfaceColor += vec3(1.0, 0.95, 0.8) * specular * 0.5;
                        }

                        // Edge fade for spherical appearance
                        float edgeFade = 1.0 - smoothstep(0.85, 1.0, normalizedDist);
                        bgColor = mix(bgColor, surfaceColor, edgeFade);
                    }
                }

                // MOVING ASTEROIDS
                for (float i = 0.0; i < 3.0; i++) {
                    float asteroidTime = u_time * 0.05 + i * 2.345;
                    vec3 asteroidPos = vec3(
                        sin(asteroidTime) * 0.6 + 0.2,
                        cos(asteroidTime * 0.8) * 0.4,
                        sin(asteroidTime * 0.6) * 0.5 + 0.5
                    );

                    float asteroidDot = dot(normalize(dir), normalize(asteroidPos));
                    if (asteroidDot > 0.998) {
                        float asteroidDist = acos(asteroidDot);
                        float asteroidSize = 0.003 + i * 0.001;

                        if (asteroidDist < asteroidSize) {
                            float brightness = 1.0 - asteroidDist / asteroidSize;
                            vec3 asteroidColor = vec3(0.5, 0.45, 0.4);
                            bgColor += asteroidColor * brightness;
                        }
                    }
                }

                // BRIGHT, VARIED STARS
                // CIRCULAR STARS (using spherical coordinates)
                // Convert direction to spherical coordinates for non-distorted mapping
                float theta = atan(dir.y, dir.x);
                float phi = acos(dir.z);
                vec2 sphereCoord = vec2(theta, phi) * 10.0;

                // Bright foreground stars
                vec2 brightGrid = floor(sphereCoord * 3.0);
                vec2 brightPos = fract(sphereCoord * 3.0) - 0.5;
                float brightHash = hash(brightGrid);

                if (brightHash > 0.985) {
                    float dist = length(brightPos);
                    if (dist < 0.15) {
                        float star = exp(-dist * dist * 100.0) * 3.0;
                        float glow = exp(-dist * 20.0);
                        vec3 starColor = mix(vec3(1.0, 0.95, 0.8), vec3(0.8, 0.9, 1.0), brightHash);
                        bgColor += starColor * (star + glow * 0.5);
                    }
                }

                // Medium stars
                vec2 medGrid = floor(sphereCoord * 8.0);
                vec2 medPos = fract(sphereCoord * 8.0) - 0.5;
                float medHash = hash(medGrid + vec2(100.0));

                if (medHash > 0.97) {
                    float dist = length(medPos);
                    if (dist < 0.1) {
                        float star = exp(-dist * dist * 200.0) * 2.0;
                        bgColor += vec3(1.0, 1.0, 0.95) * star;
                    }
                }

                // Background star field
                vec2 bgGrid = floor(sphereCoord * 20.0);
                vec2 bgPos = fract(sphereCoord * 20.0) - 0.5;
                float bgHash = hash(bgGrid + vec2(200.0));

                if (bgHash > 0.92) {
                    float dist = length(bgPos);
                    if (dist < 0.05) {
                        float star = exp(-dist * dist * 500.0) * bgHash;
                        bgColor += vec3(0.95, 0.95, 1.0) * star;
                    }
                }

                // MEDIUM BRIGHT STARS
                vec2 mediumCoord = dir.xy * 30.0;
                vec2 mediumCell = floor(mediumCoord);
                float mediumHash = fract(sin(dot(mediumCell, vec2(45.123, 89.456))) * 23456.789);

                if (mediumHash > 0.99) {
                    vec2 starPos = fract(mediumCoord) - 0.5;
                    float starDist = length(starPos);

                    float star = exp(-starDist * starDist * 200.0) * 2.0;
                    float glow = exp(-starDist * 5.0) * 0.5;

                    vec3 starColor = mix(vec3(1.0, 0.9, 0.7), vec3(0.7, 0.8, 1.0), mediumHash);
                    bgColor += starColor * (star + glow);
                }

                // MIDGROUND LAYER 2: Medium distance stars
                vec2 mgCoord = dir.xy * 80.0 + vec2(17.3, 31.2);
                vec2 mgCellId = floor(mgCoord);
                float mgHash = fract(sin(dot(mgCellId, vec2(45.233, 93.865))) * 34758.5453);

                if (mgHash > 0.98) {  // Higher density
                    vec2 starPos = fract(mgCoord) - 0.5;
                    float starDist = length(starPos);

                    // Medium sized stars
                    float starSize = 0.025 + mgHash * 0.035;
                    float star = 1.0 - smoothstep(0.0, starSize, starDist);
                    star = pow(star, 1.8);

                    // Simpler glow
                    float glow = exp(-starDist * 8.0) * 0.5;

                    float colorVar = fract(mgHash * 4321.98);
                    vec3 starColor = mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.95, 0.8), colorVar);

                    // Apply depth fade based on coordinate variation
                    float depthOffset = sin(dir.x * 2.0 + dir.y * 1.5) * 0.3;
                    float depthFade = 1.0 - depthOffset * 0.2;
                    bgColor += starColor * (star * 1.5 + glow) * depthFade;
                }

                // BACKGROUND LAYER 3: Distant star clusters
                vec2 bgCoord = dir.xy * 200.0 - vec2(51.7, 83.2);
                vec2 bgCellId = floor(bgCoord);
                float bgHash2 = fract(sin(dot(bgCellId, vec2(73.156, 41.926))) * 52879.5453);

                if (bgHash2 > 0.96) {  // Much higher density
                    vec2 starPos = fract(bgCoord) - 0.5;
                    float starDist = length(starPos);

                    // Small, simple stars
                    float starSize = 0.01 + bgHash2 * 0.015;
                    float star = 1.0 - smoothstep(0.0, starSize, starDist);

                    // Clustered appearance
                    float cluster = sin(bgCoord.x * 0.5) * cos(bgCoord.y * 0.5) * 0.3 + 0.7;

                    vec3 starColor = vec3(0.8, 0.85, 0.95);
                    bgColor += starColor * star * 0.6 * cluster;
                }

                // FAR BACKGROUND LAYER 4: Very distant star fields
                vec2 farCoord = dir.xz * 500.0 + vec2(123.4, 234.5);
                float farHash = fract(sin(dot(floor(farCoord), vec2(127.1, 311.7))) * 43758.5453);

                if (farHash > 0.96) {  // Dense but faint
                    vec2 starPos = fract(farCoord) - 0.5;
                    float star = 1.0 - smoothstep(0.0, 0.008, length(starPos));

                    // Add fog/dust attenuation for distance
                    float dustAtten = 0.7 + 0.3 * sin(farCoord.x * 0.1) * cos(farCoord.y * 0.1);

                    vec3 starColor = vec3(0.65, 0.7, 0.75);
                    bgColor += starColor * star * 0.25 * dustAtten;
                }

                // ULTRA-FAR LAYER 5: Background star haze
                vec2 ultraFarCoord = dir.yz * 1000.0 - vec2(432.1, 765.3);
                float ultraFarHash = fract(sin(dot(floor(ultraFarCoord), vec2(543.21, 876.54))) * 65432.1);

                if (ultraFarHash > 0.92) {  // Very dense, very faint
                    vec2 starPos = fract(ultraFarCoord) - 0.5;
                    float star = 1.0 - smoothstep(0.0, 0.005, length(starPos));

                    // Atmospheric scattering effect
                    vec3 starColor = vec3(0.5, 0.52, 0.6);
                    bgColor += starColor * star * 0.15;
                }

                // Add subtle star field glow for depth
                float fieldGlow = 0.0;
                for (float i = 0.0; i < 3.0; i++) {
                    vec2 fieldCoord = dir.xy * (800.0 + i * 200.0);
                    float fieldNoise = fract(sin(dot(floor(fieldCoord), vec2(32.34 + i, 76.21))) * 52343.32);
                    if (fieldNoise > 0.98) {
                        fieldGlow += 0.02 * (1.0 - i * 0.3);
                    }
                }
                bgColor += vec3(0.4, 0.45, 0.55) * fieldGlow;

                // REALISTIC NEBULAE - Based on actual cosmic colors

                // Large emission nebula (H-alpha red/pink like Orion)
                vec3 nebDir0 = normalize(dir - vec3(0.2, 0.1, 0.5));
                float nebDist0 = 1.0 - dot(nebDir0, normalize(vec3(0.2, 0.1, 0.5)));
                if (nebDist0 < 0.7) {
                    float nebIntensity = pow(1.0 - nebDist0 / 0.7, 2.0);

                    float nebPattern = 0.0;
                    nebPattern += sin(dir.x * 4.0 + dir.y * 3.0) * 0.4;
                    nebPattern += cos(dir.y * 6.0 - dir.z * 4.0) * 0.3;
                    nebPattern += sin(dir.x * 12.0) * cos(dir.z * 8.0) * 0.2;
                    nebPattern = smoothstep(-0.3, 0.8, nebPattern);

                    // Realistic H-alpha emission colors
                    vec3 nebColor = mix(
                        vec3(0.4, 0.05, 0.05),  // Deep red
                        vec3(0.9, 0.25, 0.15),  // Bright H-alpha red
                        nebPattern
                    );

                    // Add OIII emission (blue-green)
                    float oiiiEmission = sin(dir.x * 20.0) * cos(dir.y * 18.0);
                    if (oiiiEmission > 0.65) {
                        nebColor = mix(nebColor, vec3(0.1, 0.6, 0.7), (oiiiEmission - 0.65) * 1.5);
                    }

                    bgColor += nebColor * nebIntensity * nebPattern * 0.35;
                }

                // Massive pink-red emission nebula (like Carina)
                vec3 nebDir1 = normalize(dir + vec3(0.4, -0.2, 0.3));
                float nebAngle1 = acos(dot(nebDir1, vec3(0.4, -0.2, 0.3)));
                if (nebAngle1 < 0.6) {  // Much larger
                    float nebIntensity = exp(-nebAngle1 * 3.0);  // Slower falloff

                    // Complex multi-scale structure
                    float nebPattern = 0.0;
                    nebPattern += sin(nebDir1.x * 4.0 + nebDir1.y * 3.0) * 0.4;
                    nebPattern += cos(nebDir1.y * 7.0 - nebDir1.z * 5.0) * 0.3;
                    nebPattern += sin(nebDir1.x * 12.0) * cos(nebDir1.z * 10.0) * 0.2;
                    nebPattern += sin(nebDir1.x * 25.0 + nebDir1.y * 20.0) * 0.1;
                    nebPattern = smoothstep(-0.4, 0.8, nebPattern);

                    vec3 nebColor = mix(
                        vec3(0.6, 0.1, 0.2),  // Deep red
                        vec3(1.0, 0.3, 0.5),  // Bright pink-red
                        nebPattern
                    );

                    // Add bright hot regions
                    float hotSpots = sin(nebDir1.x * 30.0) * cos(nebDir1.y * 25.0);
                    if (hotSpots > 0.7) {
                        nebColor = mix(nebColor, vec3(1.0, 0.6, 0.7), (hotSpots - 0.7) * 3.0);
                    }

                    // Blue reflection areas
                    float blueArea = cos(nebDir1.x * 18.0) * sin(nebDir1.y * 15.0);
                    if (blueArea > 0.6) {
                        nebColor = mix(nebColor, vec3(0.4, 0.6, 1.0), (blueArea - 0.6));
                    }

                    bgColor += nebColor * nebIntensity * nebPattern * 0.5;
                }

                // Bright cyan-blue reflection nebula
                vec3 nebDir2 = normalize(dir - vec3(-0.6, 0.3, -0.2));
                float nebAngle2 = acos(dot(nebDir2, vec3(-0.6, 0.3, -0.2)));
                if (nebAngle2 < 0.5) {  // Larger
                    float nebIntensity = exp(-nebAngle2 * 4.0);

                    float nebPattern = 0.0;
                    nebPattern += sin(nebDir2.x * 6.0) * cos(nebDir2.y * 5.0) * 0.5;
                    nebPattern += sin(nebDir2.z * 8.0 + nebDir2.x * 7.0) * 0.3;
                    nebPattern += cos(nebDir2.x * 15.0 - nebDir2.y * 12.0) * 0.2;
                    nebPattern = smoothstep(-0.3, 0.7, nebPattern);

                    vec3 nebColor = mix(
                        vec3(0.1, 0.3, 0.5),  // Deep blue
                        vec3(0.3, 0.7, 1.0),  // Bright cyan
                        nebPattern * 1.2
                    );

                    // Add bright stars within nebula
                    float starField = sin(nebDir2.x * 50.0) * cos(nebDir2.y * 45.0);
                    if (starField > 0.8) {
                        nebColor += vec3(0.8, 0.9, 1.0) * (starField - 0.8) * 5.0;
                    }

                    bgColor += nebColor * nebIntensity * nebPattern * 0.4;
                }

                // Golden-orange nebula
                vec3 nebDir3 = normalize(dir + vec3(0.3, 0.5, -0.4));
                float nebAngle3 = acos(dot(nebDir3, vec3(0.3, 0.5, -0.4)));
                if (nebAngle3 < 0.45) {
                    float nebIntensity = exp(-nebAngle3 * 5.0);

                    float nebPattern = sin(nebDir3.x * 7.0) * cos(nebDir3.y * 9.0) +
                                      sin(nebDir3.z * 5.0) * 0.5;
                    nebPattern = smoothstep(-0.3, 0.8, nebPattern);

                    vec3 nebColor = mix(
                        vec3(0.5, 0.3, 0.1),  // Deep orange
                        vec3(1.0, 0.7, 0.3),  // Bright golden
                        nebPattern
                    );

                    bgColor += nebColor * nebIntensity * nebPattern * 0.35;
                }

                // Reflection nebula (realistic blue like Pleiades)
                vec3 nebDir4 = normalize(dir - vec3(0.7, -0.4, 0.5));
                float nebAngle4 = acos(dot(nebDir4, vec3(0.7, -0.4, 0.5)));
                if (nebAngle4 < 0.3) {
                    float nebIntensity = exp(-nebAngle4 * 8.0);

                    float nebPattern = cos(nebDir4.x * 10.0 + nebDir4.y * 8.0) *
                                      sin(nebDir4.z * 9.0);
                    nebPattern = smoothstep(-0.3, 0.5, nebPattern);

                    // Realistic reflection nebula blue
                    vec3 nebColor = mix(
                        vec3(0.05, 0.1, 0.2),  // Deep space blue
                        vec3(0.15, 0.3, 0.5),  // Reflection blue
                        nebPattern
                    );

                    bgColor += nebColor * nebIntensity * nebPattern * 0.25;
                }

                // Subtle dark dust lanes (less aggressive)
                vec3 dustDir = normalize(dir - vec3(0.1, -0.1, 0.7));
                float dustAngle = acos(dot(dustDir, vec3(0.1, -0.1, 0.7)));
                if (dustAngle < 0.2) {
                    float dustIntensity = exp(-dustAngle * 10.0);
                    float dustPattern = sin(dustDir.x * 25.0) * cos(dustDir.y * 20.0);
                    dustPattern = smoothstep(0.4, 0.8, abs(dustPattern));
                    bgColor *= 1.0 - dustIntensity * dustPattern * 0.3;  // Less darkening
                }

                // RICH GALAXY FIELD with multiple depths and types

                // Add many more galaxies at different distances
                for (float layer = 0.0; layer < 5.0; layer++) {
                    float layerScale = 20.0 + layer * 30.0;
                    vec2 galaxyGrid = dir.xy * layerScale + vec2(layer * 13.7, layer * 17.3);
                    vec2 gridCell = floor(galaxyGrid);
                    float galaxyHash = fract(sin(dot(gridCell, vec2(127.1 + layer, 311.7))) * 43758.5453);

                    // More galaxies per layer
                    if (galaxyHash > 0.85 - layer * 0.05) {
                        vec2 galPos = fract(galaxyGrid) - 0.5;
                        float galDist = length(galPos);

                        // Galaxy type based on hash
                        float typeHash = fract(galaxyHash * 6789.12);
                        float sizeVar = 0.15 - layer * 0.02;

                        if (galDist < sizeVar) {
                            // Distance-based fading
                            float depthFade = 1.0 / (1.0 + layer * 0.5);
                            float edgeFade = smoothstep(sizeVar, sizeVar * 0.7, galDist);

                            vec3 galColor;
                            float brightness = 0.0;

                            if (typeHash > 0.7) {
                                // Spiral galaxy - simplified for clarity
                                float angle = atan(galPos.y, galPos.x);
                                float r = galDist * 15.0;

                                // Simple two-arm spiral
                                float spiral = 0.0;
                                for (float arm = 0.0; arm < 2.0; arm++) {
                                    float armAngle = angle - r * 0.3 - arm * 3.14159;
                                    spiral += exp(-pow(sin(armAngle), 2.0) * 3.0) * exp(-r * 0.2);
                                }

                                // Bright core
                                float core = exp(-galDist * 100.0) * 2.0;

                                // Clear colors
                                if (galDist < 0.05) {
                                    galColor = vec3(1.0, 0.9, 0.7);  // Yellow core
                                } else {
                                    galColor = vec3(0.6, 0.7, 0.95);  // Blue arms
                                }

                                brightness = (spiral + core) * 1.5;

                            } else if (typeHash > 0.4) {
                                // Elliptical galaxy
                                float ellipse = exp(-galDist * 40.0) * 1.5;
                                float halo = exp(-galDist * 15.0) * 0.3;
                                galColor = vec3(1.0, 0.85, 0.65);
                                brightness = ellipse + halo;

                            } else if (typeHash > 0.2) {
                                // Edge-on galaxy
                                float edgeOn = exp(-abs(galPos.y) * 100.0) * exp(-abs(galPos.x) * 20.0);
                                float dustLane = 1.0 - exp(-abs(galPos.y) * 500.0) * 0.5;
                                galColor = vec3(0.8, 0.7, 0.5) * dustLane;
                                brightness = edgeOn;

                            } else {
                                // Irregular galaxy - more chaotic structure
                                float angle = atan(galPos.y, galPos.x);
                                float r = galDist * 8.0;

                                // Multiple irregular clumps
                                float clumps = 0.0;
                                for (float i = 0.0; i < 4.0; i++) {
                                    float clumpAngle = angle + i * 1.57;
                                    float clumpR = r - sin(i * 2.3) * 0.3;
                                    clumps += exp(-clumpR * clumpR * 3.0) * (0.7 + sin(clumpAngle * 3.0) * 0.3);
                                }

                                // Asymmetric star-forming regions
                                float starBurst = max(0.0, sin(angle * 4.0 + r * 2.0)) * exp(-r * 0.5);

                                // Irregular shape modulation
                                float shape = 1.0 + sin(angle * 5.0 - r) * 0.4 + cos(angle * 3.0) * 0.3;

                                // Blue-shifted due to active star formation
                                vec3 baseColor = vec3(0.5, 0.6, 0.85);
                                vec3 burstColor = vec3(0.7, 0.8, 1.0);

                                galColor = mix(baseColor, burstColor, starBurst);
                                brightness = (clumps * 0.5 + starBurst * 0.3) * shape * exp(-galDist * 20.0);
                            }

                            bgColor += galColor * brightness * depthFade * edgeFade * 0.25;  // Reduced overall brightness
                        }
                    }
                }

                // LARGE, DETAILED GALAXIES
                // Galaxy 1 - Large prominent spiral
                vec2 gal1Pos = (dir.xy - vec2(0.4, 0.15)) * 4.0;  // Much larger
                float gal1Dist = length(gal1Pos);
                if (gal1Dist < 1.5) {
                    float angle = atan(gal1Pos.y, gal1Pos.x);
                    float r = gal1Dist * 8.0;

                    // Multiple spiral arms
                    float spiral = 0.0;
                    for (float arm = 0.0; arm < 3.0; arm++) {
                        float armAngle = angle - log(r + 0.1) * 0.3 - arm * 2.094;
                        float armWidth = 0.4 - r * 0.02;
                        spiral += exp(-pow(sin(armAngle), 2.0) * 3.0 / (armWidth * armWidth)) * exp(-r * 0.1);
                    }

                    // Star forming regions
                    float starRegions = sin(angle * 8.0 - r * 0.5) * 0.3 + 0.7;
                    spiral *= starRegions;

                    // Bright galactic core with bulge
                    float core = exp(-gal1Dist * gal1Dist * 5.0) * 3.0;
                    float bulge = exp(-gal1Dist * 3.0) * 0.8;

                    // Dust lanes
                    float dust = 1.0 - sin(angle * 4.0 + r * 0.2) * 0.2 * exp(-r * 0.3);

                    vec3 coreColor = vec3(1.0, 0.95, 0.7);
                    vec3 armColor = vec3(0.5, 0.7, 1.0);
                    vec3 galColor = mix(coreColor, armColor, smoothstep(0.0, 0.4, gal1Dist));
                    galColor *= dust;

                    bgColor += galColor * (spiral * 0.5 + core + bulge * 0.3);
                }

                // Galaxy 2 - Large edge-on galaxy
                vec2 gal2Pos = (dir.xy - vec2(-0.5, -0.2)) * 5.0;
                float gal2X = abs(gal2Pos.x);
                float gal2Y = abs(gal2Pos.y);

                if (gal2X < 1.2 && gal2Y < 0.15) {
                    // Disk profile
                    float disk = exp(-gal2Y * 50.0) * exp(-gal2X * 2.0);

                    // Central bulge
                    float bulge = exp(-gal2X * 5.0 - gal2Y * 100.0) * 2.0;

                    // Dust lane
                    float dustLane = 1.0 - exp(-gal2Y * 200.0) * 0.5 * (1.0 - gal2X);

                    vec3 diskColor = vec3(0.9, 0.85, 0.7);
                    vec3 bulgeColor = vec3(1.0, 0.95, 0.8);
                    vec3 galColor = mix(diskColor, bulgeColor, bulge / (disk + bulge + 0.1));
                    galColor *= dustLane;

                    bgColor += galColor * (disk + bulge) * 0.7;
                }

                // Galaxy 3 - Smaller but bright spiral
                vec2 gal3Pos = (dir.xy - vec2(0.1, -0.5)) * 6.0;
                float gal3Dist = length(gal3Pos);
                if (gal3Dist < 1.0) {
                    float angle = atan(gal3Pos.y, gal3Pos.x);
                    float spiral = sin(angle * 2.0 - gal3Dist * 6.0) * 0.5 + 0.5;
                    spiral *= exp(-gal3Dist * 2.5);
                    float core = exp(-gal3Dist * gal3Dist * 8.0) * 2.0;

                    vec3 galColor = mix(vec3(1.0, 0.9, 0.6), vec3(0.6, 0.7, 0.95), gal3Dist);
                    bgColor += galColor * (spiral * 0.4 + core);
                }


                // Subtle dust clouds (not purple)
                float dust = pow(max(0.0, noise(dir * 1.2) - 0.5), 2.0);
                bgColor += vec3(0.05, 0.04, 0.03) * dust;  // Brown dust

                // Small distant nebula patch
                float distantNebula = noise(dir * 5.0 + vec3(50.0));
                distantNebula = pow(max(0.0, distantNebula - 0.6), 2.0);
                bgColor += vec3(0.05, 0.08, 0.12) * distantNebula * 0.3;

                // BACKGROUND: Small spiral galaxy (far) - Face-on view
                vec3 farGalDir = normalize(dir + vec3(0.3, -0.6, 0.4));
                float farGalDot = dot(farGalDir, normalize(vec3(0.3, -0.6, 0.4)));

                if (farGalDot > 0.96) {
                    float galDist = (1.0 - farGalDot) * 250.0;
                    float fadeFactor = smoothstep(0.96, 0.965, farGalDot);

                    // Small but structured spiral
                    float phi = atan(farGalDir.y, farGalDir.x);
                    float r = galDist;

                    // Two main spiral arms for distant galaxy
                    float spiral = 0.0;
                    for (float arm = 0.0; arm < 2.0; arm++) {
                        float armAngle = phi - r * 0.15 - arm * 3.142;
                        float armWidth = 0.5;
                        float armDist = mod(armAngle + 3.142, 6.283) - 3.142;
                        spiral += exp(-armDist * armDist / (armWidth * armWidth)) * exp(-r * r * 0.08);
                    }

                    // Small bright core
                    float core = exp(-r * r * 0.5);
                    float halo = exp(-r * r * 0.15) * 0.2;

                    // Muted colors for distance with slight blue shift in arms
                    vec3 coreColor = vec3(0.65, 0.6, 0.5);
                    vec3 armColor = vec3(0.45, 0.5, 0.6);
                    vec3 galColor = mix(coreColor, armColor, spiral / (spiral + core + 0.1));

                    bgColor += galColor * (spiral * 0.2 + core * 0.4 + halo) * fadeFactor * 0.6;
                }

                // FAR BACKGROUND: Distant galaxy field
                vec2 distantField = dir.xz * 200.0;
                float distantHash = fract(sin(dot(floor(distantField), vec2(127.1, 311.7))) * 43758.5453);

                if (distantHash > 0.992) {  // Slightly more galaxies
                    vec2 galPos = fract(distantField) - 0.5;
                    float galSize = length(galPos);

                    if (galSize < 0.03) {  // Very small
                        // Galaxy types at distance
                        float galType = fract(distantHash * 8765.43);
                        vec3 distantGalColor;
                        float shape = 1.0;

                        if (galType > 0.7) {
                            // Elliptical - round fuzzy blob
                            distantGalColor = vec3(0.35, 0.3, 0.25);
                            shape = exp(-galSize * 60.0);
                        } else if (galType > 0.4) {
                            // Spiral - slight elongation
                            distantGalColor = vec3(0.25, 0.28, 0.35);
                            float elongation = abs(galPos.x) * 0.5 + abs(galPos.y);
                            shape = exp(-elongation * 50.0);
                        } else {
                            // Irregular - asymmetric
                            distantGalColor = vec3(0.3, 0.25, 0.3);
                            float irregular = sin(atan(galPos.y, galPos.x) * 3.0) * 0.2 + 0.8;
                            shape = exp(-galSize * 55.0) * irregular;
                        }

                        bgColor += distantGalColor * shape * 0.12;
                    }
                }

                // ULTRA-FAR: Galaxy clusters (very faint, grouped)
                vec2 clusterField = dir.xy * 100.0 + vec2(50.0, 30.0);
                float clusterHash = fract(sin(dot(floor(clusterField), vec2(543.21, 876.54))) * 65432.1);

                if (clusterHash > 0.996) {
                    // Create cluster of faint galaxies
                    vec2 clusterCenter = fract(clusterField) - 0.5;
                    float clusterRadius = length(clusterCenter);

                    if (clusterRadius < 0.15) {
                        // Multiple galaxies in cluster
                        for (float i = 0.0; i < 5.0; i++) {
                            vec2 galOffset = vec2(sin(i * 1.26), cos(i * 1.26)) * 0.05;
                            float galDist = length(clusterCenter + galOffset);
                            float gal = exp(-galDist * 30.0);
                            bgColor += vec3(0.2, 0.18, 0.22) * gal * 0.05;
                        }
                    }
                }

                // Milky way band (our galaxy edge-on view)
                float milkyWay = abs(dir.y);
                if (milkyWay < 0.25) {
                    float mwIntensity = exp(-milkyWay * 12.0);

                    // Complex structure
                    float mwPattern = 0.0;
                    mwPattern += sin(dir.x * 60.0) * cos(dir.z * 50.0) * 0.5;
                    mwPattern += sin(dir.x * 150.0 + dir.z * 120.0) * 0.3;
                    mwPattern += cos(dir.x * 30.0 - dir.z * 40.0) * 0.2;
                    mwPattern = smoothstep(-0.5, 0.8, mwPattern);

                    // Dark rifts
                    float darkRift = sin(dir.x * 20.0 + dir.z * 15.0);
                    if (darkRift > 0.6) {
                        mwPattern *= 0.3;
                    }

                    vec3 mwColor = vec3(0.18, 0.15, 0.12) * mwPattern;
                    bgColor += mwColor * mwIntensity * 0.15;
                }

                color += bgColor * (1.0 - totalBrightness * 0.4);
            }

            return vec4(color, 1.0);
        }

        void main() {
            vec2 uv = (v_texcoord - 0.5) * 2.0;
            uv.x *= u_resolution.x / u_resolution.y;

            // Camera setup for Interstellar-like view
            vec3 ro = vec3(0.0, 0.5, -35.0) * u_zoom;
            vec3 rd = normalize(vec3(uv.x * 0.4, uv.y * 0.4, 1.0));

            // Apply camera rotation
            mat3 rotX = rotationMatrixX(u_camera_rotation.x);
            mat3 rotY = rotationMatrixY(u_camera_rotation.y);
            mat3 rotation = rotY * rotX;

            ro = rotation * ro;
            rd = rotation * rd;

            // Ray trace
            vec4 col = traceRay(ro, rd);

            // Post-processing
            // Tone mapping
            col.rgb = col.rgb / (col.rgb + vec3(1.0));

            // Gamma correction
            col.rgb = pow(col.rgb, vec3(0.85));

            // Slight vignette
            float vignette = 1.0 - length(uv) * 0.2;
            col.rgb *= vignette;

            // Bloom-like glow for bright areas
            float brightness = dot(col.rgb, vec3(0.299, 0.587, 0.114));
            if (brightness > 0.7) {
                col.rgb += col.rgb * (brightness - 0.7) * 0.5;
            }

            fragColor = vec4(col.rgb, 1.0);
        }
        '''

        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        # Set uniforms
        rs_km = self.engine.calculate_schwarzschild_radius() / 1000
        self.program['u_schwarzschild_radius'] = rs_km / 100.0
        self.program['u_resolution'] = (self.width, self.height)

    def setup_buffers(self):
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ], dtype='f4')

        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')

        vbo = self.ctx.buffer(vertices)
        ibo = self.ctx.buffer(indices)

        self.vao = self.ctx.vertex_array(
            self.program,
            [(vbo, '2f 2f', 'in_position', 'in_texcoord')],
            ibo
        )

    def handle_input(self):
        keys = pygame.key.get_pressed()

        # Camera rotation
        rotation_speed = 0.02
        if keys[pygame.K_LEFT]:
            self.camera_angle_y -= rotation_speed
        if keys[pygame.K_RIGHT]:
            self.camera_angle_y += rotation_speed
        if keys[pygame.K_UP]:
            self.camera_angle_x -= rotation_speed
        if keys[pygame.K_DOWN]:
            self.camera_angle_x += rotation_speed

        # Zoom
        if keys[pygame.K_z]:
            self.zoom *= 0.98
        if keys[pygame.K_x]:
            self.zoom *= 1.02

        # Reset view
        if keys[pygame.K_r]:
            self.camera_angle_x = 0.0
            self.camera_angle_y = 0.0
            self.zoom = 1.0

        # Clamp values
        self.camera_angle_x = max(-1.5, min(1.5, self.camera_angle_x))
        self.zoom = max(0.3, min(3.0, self.zoom))

    def draw_info(self):
        # Create overlay surface for info
        info_surface = pygame.Surface((400, 120), pygame.SRCALPHA)
        info_surface.fill((0, 0, 0, 100))

        font = pygame.font.Font(None, 20)
        texts = [
            f"Interstellar-Style Black Hole Renderer",
            f"Mass: {self.engine.mass / 1.989e30:.1f} Solar Masses",
            f"Controls: Arrow Keys (rotate), Z/X (zoom), R (reset)",
            f"FPS: {self.clock.get_fps():.1f}"
        ]

        y_offset = 10
        for text in texts:
            rendered = font.render(text, True, (200, 200, 220))
            info_surface.blit(rendered, (10, y_offset))
            y_offset += 25

        # This won't work directly with OpenGL context, but kept for reference
        # In production, you'd render this text using OpenGL

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            self.handle_input()

            # Update time
            self.time += 0.016

            # Update shader uniforms
            self.program['u_time'] = self.time
            self.program['u_camera_rotation'] = (self.camera_angle_x, self.camera_angle_y, 0.0)
            self.program['u_zoom'] = self.zoom

            # Clear and render
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.vao.render(moderngl.TRIANGLES)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()