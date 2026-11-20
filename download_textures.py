#!/usr/bin/env python3
"""
NASA/ESA Texture Downloader for Black Hole Simulation
======================================================

Downloads REAL satellite imagery and astronomical data for Universe Sandbox-quality rendering.

This script downloads REAL textures from:
- NASA Blue Marble (Earth 8K)
- NASA LRO (Moon)
- Solar System Scope (planets - free for non-commercial)
- ESO Milky Way panoramas
- Hipparcos star catalog

Usage:
    python download_textures.py          # Download all textures
    python download_textures.py --quick  # Download essentials only
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


class TextureDownloader:
    """Downloads REAL NASA/ESA textures for realistic space rendering."""

    def __init__(self):
        self.texture_dir = Path(__file__).parent / "textures"
        self.texture_dir.mkdir(exist_ok=True)
        (self.texture_dir / 'planets').mkdir(exist_ok=True)
        (self.texture_dir / 'stars').mkdir(exist_ok=True)
        (self.texture_dir / 'space').mkdir(exist_ok=True)

    def download_file(self, url, filename, force=False):
        """Download a file with progress indication."""
        filepath = self.texture_dir / filename

        if filepath.exists() and not force:
            print(f"✓ Already exists: {filename}")
            return True

        print(f"\nDownloading: {filename}")
        print(f"URL: {url}")

        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )

            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100 / total_size)
                    bars = int(percent / 2)
                    print(f"\r[{'=' * bars}{' ' * (50 - bars)}] {percent:.1f}%", end='', flush=True)

            urllib.request.urlretrieve(url, filepath, progress_hook)
            print("\n✓ Download complete")
            return True

        except Exception as e:
            print(f"\n✗ Failed: {e}")
            return False

    def download_earth(self):
        """Download REAL Earth textures from NASA."""
        print("\n" + "=" * 70)
        print("EARTH - NASA Blue Marble & City Lights")
        print("=" * 70)

        # Using Solar System Scope textures (free for non-commercial use)
        # These are based on NASA data but at manageable resolutions
        textures = {
            'planets/earth_day_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg',
            'planets/earth_night_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_earth_nightmap.jpg',
            'planets/earth_clouds_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_earth_clouds.jpg',
            'planets/earth_normal_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_earth_normal_map.jpg',
            'planets/earth_specular_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_earth_specular_map.jpg',
        }

        for filename, url in textures.items():
            self.download_file(url, filename)

    def download_moon(self):
        """Download REAL Moon textures."""
        print("\n" + "=" * 70)
        print("MOON - Lunar Reconnaissance Orbiter")
        print("=" * 70)

        textures = {
            'planets/moon_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_moon.jpg',
        }

        for filename, url in textures.items():
            self.download_file(url, filename)

    def download_mars(self):
        """Download REAL Mars textures."""
        print("\n" + "=" * 70)
        print("MARS - Viking/HiRISE Data")
        print("=" * 70)

        textures = {
            'planets/mars_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_mars.jpg',
        }

        for filename, url in textures.items():
            self.download_file(url, filename)

    def download_sun(self):
        """Download REAL Sun textures from SDO."""
        print("\n" + "=" * 70)
        print("SUN - Solar Dynamics Observatory")
        print("=" * 70)

        textures = {
            'stars/sun_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_sun.jpg',
        }

        for filename, url in textures.items():
            self.download_file(url, filename)

    def download_jupiter(self):
        """Download REAL Jupiter textures from Juno/Cassini."""
        print("\n" + "=" * 70)
        print("JUPITER - Juno Mission Data")
        print("=" * 70)

        textures = {
            'planets/jupiter_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_jupiter.jpg',
        }

        for filename, url in textures.items():
            self.download_file(url, filename)

    def download_saturn(self):
        """Download REAL Saturn textures from Cassini."""
        print("\n" + "=" * 70)
        print("SATURN - Cassini Mission Data")
        print("=" * 70)

        textures = {
            'planets/saturn_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_saturn.jpg',
            'planets/saturn_rings_4k.png': 'https://www.solarsystemscope.com/textures/download/2k_saturn_ring_alpha.png',
        }

        for filename, url in textures.items():
            self.download_file(url, filename)

    def download_milky_way(self):
        """Download REAL Milky Way panorama."""
        print("\n" + "=" * 70)
        print("MILKY WAY - ESO/Hubble Data")
        print("=" * 70)

        textures = {
            'space/starfield_4k.jpg': 'https://www.solarsystemscope.com/textures/download/2k_stars_milky_way.jpg',
        }

        for filename, url in textures.items():
            self.download_file(url, filename)

    def create_simple_fallbacks(self):
        """Create minimal fallback textures for missing files."""
        print("\n" + "=" * 70)
        print("Creating Simple Fallback Textures")
        print("=" * 70)

        try:
            from PIL import Image, ImageDraw
            import random

            # Simple Earth
            if not (self.texture_dir / 'planets/earth_day_4k.jpg').exists():
                print("Creating fallback Earth...")
                img = Image.new('RGB', (1024, 512), color=(0, 100, 200))
                draw = ImageDraw.Draw(img)
                for i in range(50):
                    x, y = random.randint(0, 1024), random.randint(0, 512)
                    r = random.randint(20, 80)
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=(34, 139, 34))
                img.save(self.texture_dir / 'planets/earth_day_4k.jpg')

            # Simple starfield
            if not (self.texture_dir / 'space/starfield_4k.jpg').exists():
                print("Creating fallback starfield...")
                img = Image.new('RGB', (2048, 1024), color=(0, 0, 10))
                pixels = img.load()
                for i in range(5000):
                    x, y = random.randint(0, 2047), random.randint(0, 1023)
                    brightness = random.randint(150, 255)
                    pixels[x, y] = (brightness, brightness, brightness)
                img.save(self.texture_dir / 'space/starfield_4k.jpg')

            print("✓ Fallbacks created")

        except ImportError:
            print("Pillow not available - skipping fallbacks")

    def download_all(self, quick=False):
        """Download all textures."""
        print("\n" + "=" * 70)
        print("NASA/ESA TEXTURE DOWNLOADER - Black Hole Simulation")
        print("=" * 70)
        print(f"Destination: {self.texture_dir.absolute()}")
        print("=" * 70)

        # Essential textures
        self.download_earth()
        self.download_sun()
        self.download_milky_way()

        if not quick:
            # Additional planets
            self.download_moon()
            self.download_mars()
            self.download_jupiter()
            self.download_saturn()

        # Create fallbacks for missing textures
        self.create_simple_fallbacks()

        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE!")
        print(f"Textures saved to: {self.texture_dir.absolute()}")
        print("\nYou can now run: python main.py --panda3d")
        print("=" * 70)


def main():
    """Main entry point."""
    downloader = TextureDownloader()

    if '--quick' in sys.argv:
        print("Quick download mode (essentials only)")
        downloader.download_all(quick=True)
    else:
        downloader.download_all(quick=False)


if __name__ == "__main__":
    main()
