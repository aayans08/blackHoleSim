"""
Ultra-Realistic Black Hole Renderer with HDR + Bloom
Built on top of InterstellarRenderer with enhanced post-processing
"""

import moderngl
import numpy as np
import pygame
from pygame.locals import *
from InterstellarRenderer import InterstellarRenderer


class RealisticRenderer(InterstellarRenderer):
    """
    Enhanced renderer with HDR rendering and bloom post-processing.
    Inherits from InterstellarRenderer and adds multi-pass rendering.
    """

    def __init__(self, engine, width=1280, height=800):
        """Initialize with HDR framebuffers"""
        self.engine = engine
        self.width = width
        self.height = height

        # HDR parameters
        self.exposure = 1.0
        self.bloom_strength = 0.3
        self.bloom_threshold = 0.8

        # Initialize parent (sets up pygame, context, basic shaders)
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Black Hole - Ultra-Realistic (HDR + Bloom)")
        self.clock = pygame.time.Clock()

        self.ctx = moderngl.create_context()

        self.time = 0.0
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.zoom = 1.0

        # Setup parent shaders first
        self.setup_shaders()
        self.setup_buffers()

        # Then add HDR/bloom pipeline
        self.setup_hdr_pipeline()

        print("\n" + "="*70)
        print("ULTRA-REALISTIC RENDERER INITIALIZED")
        print("="*70)
        print("✓ HDR rendering with 16-bit floating point")
        print("✓ Two-pass Gaussian bloom")
        print("✓ ACES filmic tone mapping")
        print("✓ All InterstellarRenderer features included")
        print("\nExtra Controls:")
        print("  E/D - Adjust exposure")
        print("  B/N - Adjust bloom strength")
        print("="*70 + "\n")

    def setup_hdr_pipeline(self):
        """Setup HDR framebuffers and bloom shaders"""
        # Main HDR framebuffer (16-bit float)
        self.hdr_color = self.ctx.texture((self.width, self.height), 4, dtype='f2')
        self.hdr_depth = self.ctx.depth_texture((self.width, self.height))
        self.hdr_fbo = self.ctx.framebuffer(
            color_attachments=[self.hdr_color],
            depth_attachment=self.hdr_depth
        )

        # Bloom buffers (half resolution for performance)
        bloom_w = self.width // 2
        bloom_h = self.height // 2

        self.bloom_h_tex = self.ctx.texture((bloom_w, bloom_h), 4, dtype='f2')
        self.bloom_h_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_h_tex])

        self.bloom_v_tex = self.ctx.texture((bloom_w, bloom_h), 4, dtype='f2')
        self.bloom_v_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_v_tex])

        # Bloom extract shader (horizontal blur)
        bloom_h_shader = '''
        #version 330
        uniform sampler2D u_texture;
        uniform float u_threshold;
        uniform vec2 u_direction;
        in vec2 v_texcoord;
        out vec4 fragColor;

        void main() {
            vec3 color = vec3(0.0);
            vec2 off = vec2(1.3333333) * u_direction;

            // 9-tap Gaussian blur
            color += texture(u_texture, v_texcoord).rgb * 0.2270270270;
            color += texture(u_texture, v_texcoord + off * 1.0).rgb * 0.3162162162;
            color += texture(u_texture, v_texcoord - off * 1.0).rgb * 0.3162162162;
            color += texture(u_texture, v_texcoord + off * 2.0).rgb * 0.0702702703;
            color += texture(u_texture, v_texcoord - off * 2.0).rgb * 0.0702702703;

            // Extract bright areas
            float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
            if(brightness > u_threshold)
                fragColor = vec4(color, 1.0);
            else
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
        '''

        # Vertical blur shader
        bloom_v_shader = '''
        #version 330
        uniform sampler2D u_texture;
        uniform vec2 u_direction;
        in vec2 v_texcoord;
        out vec4 fragColor;

        void main() {
            vec3 color = vec3(0.0);
            vec2 off = vec2(1.3333333) * u_direction;

            color += texture(u_texture, v_texcoord).rgb * 0.2270270270;
            color += texture(u_texture, v_texcoord + off * 1.0).rgb * 0.3162162162;
            color += texture(u_texture, v_texcoord - off * 1.0).rgb * 0.3162162162;
            color += texture(u_texture, v_texcoord + off * 2.0).rgb * 0.0702702703;
            color += texture(u_texture, v_texcoord - off * 2.0).rgb * 0.0702702703;

            fragColor = vec4(color, 1.0);
        }
        '''

        # Tone mapping shader (ACES + bloom combine)
        tonemap_shader = '''
        #version 330
        uniform sampler2D u_scene;
        uniform sampler2D u_bloom;
        uniform float u_exposure;
        uniform float u_bloom_strength;
        in vec2 v_texcoord;
        out vec4 fragColor;

        // ACES Filmic Tone Mapping
        vec3 aces(vec3 x) {
            const float a = 2.51;
            const float b = 0.03;
            const float c = 2.43;
            const float d = 0.59;
            const float e = 0.14;
            return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
        }

        void main() {
            vec3 hdrColor = texture(u_scene, v_texcoord).rgb;
            vec3 bloomColor = texture(u_bloom, v_texcoord).rgb;

            // Add bloom
            hdrColor += bloomColor * u_bloom_strength;

            // Apply exposure
            hdrColor *= u_exposure;

            // Tone map
            vec3 mapped = aces(hdrColor);

            // Gamma correction
            mapped = pow(mapped, vec3(1.0/2.2));

            fragColor = vec4(mapped, 1.0);
        }
        '''

        # Compile bloom shaders
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

        self.bloom_h_prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=bloom_h_shader
        )

        self.bloom_v_prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=bloom_v_shader
        )

        self.tonemap_prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=tonemap_shader
        )

        # Setup uniforms
        self.bloom_h_prog['u_threshold'] = self.bloom_threshold
        self.bloom_h_prog['u_direction'] = (1.0 / (self.width / 2), 0.0)
        self.bloom_v_prog['u_direction'] = (0.0, 1.0 / (self.height / 2))
        self.tonemap_prog['u_exposure'] = self.exposure
        self.tonemap_prog['u_bloom_strength'] = self.bloom_strength

        # Bind texture units
        self.hdr_color.use(0)
        self.bloom_v_tex.use(1)

    def handle_input(self):
        """Extended input handling with HDR controls"""
        super().handle_input()

        keys = pygame.key.get_pressed()

        # Exposure control
        if keys[pygame.K_e]:
            self.exposure *= 1.02
            self.tonemap_prog['u_exposure'] = self.exposure
        if keys[pygame.K_d]:
            self.exposure *= 0.98
            self.tonemap_prog['u_exposure'] = self.exposure

        # Bloom strength
        if keys[pygame.K_b]:
            self.bloom_strength = min(1.0, self.bloom_strength * 1.02)
            self.tonemap_prog['u_bloom_strength'] = self.bloom_strength
        if keys[pygame.K_n]:
            self.bloom_strength = max(0.0, self.bloom_strength * 0.98)
            self.tonemap_prog['u_bloom_strength'] = self.bloom_strength

    def render_scene(self):
        """Render main scene to HDR framebuffer"""
        self.hdr_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Update uniforms
        self.program['u_time'] = self.time
        self.program['u_camera_rotation'] = (self.camera_angle_x, self.camera_angle_y, 0.0)
        self.program['u_zoom'] = self.zoom

        # Render
        self.vao.render(moderngl.TRIANGLES)

    def apply_bloom(self):
        """Two-pass Gaussian blur bloom"""
        # Horizontal pass (extract bright + blur)
        self.bloom_h_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.hdr_color.use(0)
        self.bloom_h_prog['u_texture'] = 0
        self.vao.render(moderngl.TRIANGLES)

        # Vertical pass
        self.bloom_v_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bloom_h_tex.use(0)
        self.bloom_v_prog['u_texture'] = 0
        self.vao.render(moderngl.TRIANGLES)

    def tonemap_and_display(self):
        """Apply tone mapping and display final image"""
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.hdr_color.use(0)
        self.bloom_v_tex.use(1)
        self.tonemap_prog['u_scene'] = 0
        self.tonemap_prog['u_bloom'] = 1

        self.vao.render(moderngl.TRIANGLES)

    def run(self):
        """Main rendering loop with HDR pipeline"""
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

            # Multi-pass rendering
            self.render_scene()      # Scene -> HDR buffer
            self.apply_bloom()        # HDR -> bloom buffers
            self.tonemap_and_display() # Combine and tonemap -> screen

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
