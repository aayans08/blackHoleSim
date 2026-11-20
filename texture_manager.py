"""
Texture management system for realistic planet rendering.
Handles loading, caching, and GPU upload of textures.
"""

import os
import numpy as np
from PIL import Image
import moderngl


class TextureManager:
    """Manages texture loading, caching, and GPU memory."""

    def __init__(self, ctx):
        """
        Initialize texture manager.

        Args:
            ctx: ModernGL context
        """
        self.ctx = ctx
        self.cache = {}  # Texture cache
        self.texture_dir = os.path.join(os.path.dirname(__file__), 'textures')

    def load_texture(self, filename, generate_mipmaps=True, anisotropy=4.0):
        """
        Load texture from file with caching.

        Args:
            filename: Texture filename (will look in textures/ directory)
            generate_mipmaps: Generate mipmaps for better quality at distance
            anisotropy: Anisotropic filtering level (4.0 or 8.0 recommended)

        Returns:
            ModernGL Texture object
        """
        # Check cache first
        if filename in self.cache:
            return self.cache[filename]

        filepath = os.path.join(self.texture_dir, filename)

        try:
            # Load image with PIL
            img = Image.open(filepath)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Get image data
            img_data = np.array(img, dtype=np.uint8)

            # Flip vertically (OpenGL expects bottom-left origin)
            img_data = np.flipud(img_data)

            # Create texture
            texture = self.ctx.texture(
                size=img.size,
                components=3,
                data=img_data.tobytes()
            )

            # Generate mipmaps for better quality
            if generate_mipmaps:
                texture.build_mipmaps()

            # Set filtering
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

            # Anisotropic filtering (reduces blurriness at angles)
            if hasattr(texture, 'anisotropy'):
                texture.anisotropy = anisotropy

            # Cache the texture
            self.cache[filename] = texture

            print(f"Loaded texture: {filename} ({img.size[0]}x{img.size[1]})")

            return texture

        except FileNotFoundError:
            print(f"Warning: Texture not found: {filepath}")
            # Return a simple procedural texture
            return self.create_procedural_texture()

    def create_procedural_texture(self, size=512):
        """
        Create a simple procedural texture as fallback.

        Args:
            size: Texture resolution (square)

        Returns:
            ModernGL Texture object
        """
        # Create simple noise-based procedural texture
        np.random.seed(42)
        data = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

        texture = self.ctx.texture(
            size=(size, size),
            components=3,
            data=data.tobytes()
        )

        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        return texture

    def create_earth_like_texture(self, size=1024):
        """
        Create procedural Earth-like planet texture.

        Args:
            size: Texture resolution (square)

        Returns:
            ModernGL Texture object with diffuse, normal, and specular maps
        """
        # Diffuse map
        diffuse = self._generate_earth_diffuse(size)

        # Normal map (for surface detail)
        normal = self._generate_earth_normal(size)

        # Specular map (ocean = shiny, land = not)
        specular = self._generate_earth_specular(size)

        # Create textures
        diffuse_tex = self.ctx.texture(
            size=(size, size),
            components=3,
            data=diffuse.tobytes()
        )
        diffuse_tex.build_mipmaps()
        diffuse_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        normal_tex = self.ctx.texture(
            size=(size, size),
            components=3,
            data=normal.tobytes()
        )
        normal_tex.build_mipmaps()
        normal_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        specular_tex = self.ctx.texture(
            size=(size, size),
            components=3,
            data=specular.tobytes()
        )
        specular_tex.build_mipmaps()
        specular_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        return diffuse_tex, normal_tex, specular_tex

    def _generate_earth_diffuse(self, size):
        """Generate Earth-like diffuse color map."""
        data = np.zeros((size, size, 3), dtype=np.uint8)

        # Simple noise-based continents
        for y in range(size):
            for x in range(size):
                # Use simple hash-based noise
                noise_val = self._hash2d(x / 100.0, y / 100.0)

                if noise_val > 0.5:
                    # Land (green/brown)
                    data[y, x] = [51, 115, 38]  # Green
                else:
                    # Ocean (blue)
                    data[y, x] = [5, 20, 51]  # Dark blue

        return data

    def _generate_earth_normal(self, size):
        """Generate normal map for surface detail."""
        # Flat normal map (pointing up in tangent space)
        data = np.full((size, size, 3), [128, 128, 255], dtype=np.uint8)
        return data

    def _generate_earth_specular(self, size):
        """Generate specular map (ocean = white, land = black)."""
        data = np.zeros((size, size, 3), dtype=np.uint8)

        for y in range(size):
            for x in range(size):
                noise_val = self._hash2d(x / 100.0, y / 100.0)

                if noise_val <= 0.5:
                    # Ocean is reflective
                    data[y, x] = [200, 200, 200]

        return data

    def _hash2d(self, x, y):
        """Simple 2D hash function for procedural noise."""
        import math
        return math.fract(math.sin(x * 12.9898 + y * 78.233) * 43758.5453)

    def create_blank_texture(self, size=64, color=(128, 128, 128)):
        """
        Create a solid color texture.

        Args:
            size: Texture size
            color: RGB color tuple

        Returns:
            ModernGL Texture
        """
        data = np.full((size, size, 3), color, dtype=np.uint8)

        texture = self.ctx.texture(
            size=(size, size),
            components=3,
            data=data.tobytes()
        )

        return texture

    def clear_cache(self):
        """Release all cached textures."""
        for texture in self.cache.values():
            texture.release()
        self.cache.clear()


# Standalone math helper since math.fract doesn't exist
import math as _math

def _fract(x):
    """Return fractional part of x."""
    return x - _math.floor(x)

# Monkey-patch for the hash function
_math.fract = _fract
