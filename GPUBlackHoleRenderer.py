import moderngl
import numpy as np
import pygame
from pygame.locals import *
import struct

class GPUBlackHoleRenderer:
    def __init__(self, engine, width=1280, height=800):  # MacBook Air friendly resolution
        self.engine = engine
        self.width = width
        self.height = height

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Black Hole - GPU Ray Tracer")
        self.clock = pygame.time.Clock()

        self.ctx = moderngl.create_context()

        self.time = 0.0
        self.camera_angle_x = -0.1  # Slight tilt to see both sides of disk
        self.camera_angle_y = 0.0
        self.zoom = 1.2  # Slightly zoomed out for better view

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
        const int MAX_STEPS = 300;
        const float MAX_DIST = 100.0;
        const float EPSILON = 0.001;

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
            if (r <= u_schwarzschild_radius * 1.001) return 0.0;
            return 1.0 - u_schwarzschild_radius / r;
        }

        // More accurate light bending using deflection angle
        vec3 bendLight(vec3 pos, vec3 dir) {
            float r = length(pos);
            if (r < u_schwarzschild_radius * 10.0 && r > u_schwarzschild_radius) {
                // Einstein deflection angle approximation
                float b = r; // impact parameter
                float deflection = 4.0 * u_schwarzschild_radius / b;

                vec3 toCenter = -normalize(pos);
                vec3 perpDir = normalize(cross(cross(dir, toCenter), dir));

                // Apply deflection
                dir = normalize(dir + perpDir * deflection * 0.1);
            }
            return dir;
        }

        // Check if ray intersects the thin accretion disk
        bool diskIntersection(vec3 pos, vec3 dir, out float t) {
            // Disk is in the y=0 plane
            if (abs(dir.y) < 0.0001) return false;

            t = -pos.y / dir.y;
            if (t < 0.0) return false;

            vec3 hitPoint = pos + dir * t;
            float r = length(hitPoint.xz);

            // Check if within disk radii
            return r > u_schwarzschild_radius * 3.0 && r < u_schwarzschild_radius * 15.0;
        }

        float getDiskIntensity(vec3 pos) {
            float r = length(pos.xz);
            if (r < u_schwarzschild_radius * 3.0 || r > u_schwarzschild_radius * 15.0) {
                return 0.0;
            }

            // Intensity falls off with radius
            float intensity = exp(-(r - u_schwarzschild_radius * 4.0) * 0.08);

            // Add swirl pattern
            float angle = atan(pos.z, pos.x) + u_time * 0.5 / sqrt(r / u_schwarzschild_radius);
            intensity *= 0.85 + 0.15 * sin(angle * 8.0 - r * 0.3);

            return intensity;
        }

        vec3 accretionDiskColor(vec3 pos, float intensity) {
            float r = length(pos.xz);
            float temp = 1.0 - (r - u_schwarzschild_radius * 3.0) / (u_schwarzschild_radius * 12.0);
            temp = clamp(temp, 0.0, 1.0);

            // Doppler shift from orbital motion
            float angle = atan(pos.z, pos.x);
            vec3 velocity = vec3(-sin(angle), 0.0, cos(angle)) * sqrt(u_schwarzschild_radius / r);
            vec3 viewDir = normalize(pos - vec3(0.0, 3.0, -30.0));
            float doppler = 1.0 + dot(velocity, viewDir) * 0.3;

            // Color based on temperature
            vec3 color;
            if (temp > 0.85) {
                // Innermost: white-blue
                color = vec3(0.8, 0.9, 1.0);
            } else if (temp > 0.6) {
                // Inner: white-hot
                color = vec3(1.0, 1.0, 0.95);
            } else if (temp > 0.4) {
                // Middle: yellow-white
                color = vec3(1.0, 0.95, 0.7);
            } else if (temp > 0.25) {
                // Outer-middle: orange
                color = vec3(1.0, 0.7, 0.3);
            } else {
                // Outer: red-orange
                color = vec3(0.9, 0.4, 0.1);
            }

            // Apply Doppler shift
            color *= doppler;

            // Gravitational redshift
            float metric = schwarzschildMetric(r);
            color *= pow(metric, 0.3);

            return color * intensity * 2.5;
        }

        vec4 rayMarch(vec3 ro, vec3 rd) {
            vec3 color = vec3(0.0);
            vec3 pos = ro;
            vec3 dir = rd;

            float totalIntensity = 0.0;
            int diskHits = 0;

            for (int i = 0; i < MAX_STEPS; i++) {
                float r = length(pos);

                // Event horizon check
                if (r < u_schwarzschild_radius * 1.01) {
                    if (diskHits == 0) {
                        return vec4(0.0, 0.0, 0.0, 1.0);  // Black hole
                    }
                    break;
                }

                if (r > MAX_DIST) break;

                // Gravitational lensing - bend the ray
                float strength = u_schwarzschild_radius / (r * r);
                vec3 toCenter = -normalize(pos);

                // Perpendicular component for bending
                vec3 perpComponent = dir - dot(dir, toCenter) * toCenter;
                if (length(perpComponent) > 0.001) {
                    vec3 bendDirection = normalize(cross(cross(dir, toCenter), dir));
                    dir = normalize(dir + bendDirection * strength * 0.15);
                }

                // Check disk intersection
                float t;
                if (diskIntersection(pos, dir, t) && t < 1.0) {
                    vec3 hitPoint = pos + dir * t;
                    float intensity = getDiskIntensity(hitPoint);

                    if (intensity > 0.01) {
                        vec3 diskCol = accretionDiskColor(hitPoint, intensity);

                        // Add this contribution
                        float weight = intensity * exp(-totalIntensity * 0.5);
                        color += diskCol * weight;
                        totalIntensity += intensity;
                        diskHits++;

                        // Continue tracing to capture multiple images
                        if (diskHits >= 3 || totalIntensity > 2.0) break;
                    }
                }

                // Step forward
                float metric = schwarzschildMetric(r);
                float stepSize = max(0.01, min(0.5, r * 0.01)) * metric;
                pos += dir * stepSize;
            }

            // Add background if we didn't collect enough light
            if (totalIntensity < 1.0) {
                vec3 bgColor = vec3(0.01, 0.01, 0.02);

                // Stars
                float starHash = fract(sin(dot(floor(dir.xy * 100.0), vec2(12.9898, 78.233))) * 43758.5453);
                if (starHash > 0.995) {
                    bgColor += vec3(0.8);
                }

                color += bgColor * (1.0 - totalIntensity);
            }

            return vec4(color, 1.0);
        }

        void main() {
            vec2 uv = (v_texcoord - 0.5) * 2.0;
            uv.x *= u_resolution.x / u_resolution.y;

            // Camera positioned to center black hole
            vec3 ro = vec3(0.0, 3.0, -30.0) * u_zoom;
            vec3 rd = normalize(vec3(uv.x * 0.5, uv.y * 0.5, 1.0));

            mat3 rotX = rotationMatrixX(u_camera_rotation.x);
            mat3 rotY = rotationMatrixY(u_camera_rotation.y);
            mat3 rotation = rotY * rotX;

            ro = rotation * ro;
            rd = rotation * rd;

            vec4 col = rayMarch(ro, rd);

            col.rgb = pow(col.rgb, vec3(0.85));

            col.rgb = col.rgb / (col.rgb + vec3(1.0));

            fragColor = col;
        }
        '''

        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

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

        if keys[pygame.K_LEFT]:
            self.camera_angle_y -= 0.02
        if keys[pygame.K_RIGHT]:
            self.camera_angle_y += 0.02
        if keys[pygame.K_UP]:
            self.camera_angle_x -= 0.02
        if keys[pygame.K_DOWN]:
            self.camera_angle_x += 0.02

        if keys[pygame.K_z]:
            self.zoom *= 0.98
        if keys[pygame.K_x]:
            self.zoom *= 1.02

        if keys[pygame.K_r]:
            self.camera_angle_x = 0.3
            self.camera_angle_y = 0.0
            self.zoom = 1.0

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

            self.time += 0.016

            self.program['u_time'] = self.time
            self.program['u_camera_rotation'] = (self.camera_angle_x, self.camera_angle_y, 0.0)
            self.program['u_zoom'] = self.zoom

            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.vao.render(moderngl.TRIANGLES)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()