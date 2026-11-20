import sys
import pygame
from blackHoleEngine import BlackHoleEngine
from PygameRenderer import PygameRenderer

def main():
    """Black hole visualization with multiple renderer options."""

    # Parse command line arguments
    use_ultra = '--ultra' in sys.argv or '-u' in sys.argv
    use_realistic = '--realistic' in sys.argv or '-r' in sys.argv
    use_interstellar = '--interstellar' in sys.argv or '-i' in sys.argv
    use_gpu = '--gpu' in sys.argv or '-g' in sys.argv
    use_panda3d = '--panda3d' in sys.argv or '-p' in sys.argv

    # Create black hole at origin with 10 solar masses (3D for Panda3D)
    if use_panda3d:
        engine = BlackHoleEngine(10, [0, 0, 0], [0, 0, 0])
    else:
        engine = BlackHoleEngine(10, [0, 0], [0, 0])

    if use_ultra:
        print("Starting ULTRA-REALISTIC Black Hole Renderer...")
        print("Proper Schwarzschild geodesic integration with real physics!")
        print("\nFeatures:")
        print("  - REAL gravitational lensing (not fake bending)")
        print("  - Accretion disk with Doppler shift & gravitational redshift")
        print("  - Photon sphere visualization (Einstein rings)")
        print("  - HDRI environment mapping (real Milky Way)")
        print("  - Optimized for M3 GPU performance")
        print("\nControls:")
        print("  Arrow Keys: Rotate camera")
        print("  Z/X: Zoom in/out")
        print("  W/S: Move closer/farther")
        print("  R: Reset view")
        print("  ESC: Exit")
        try:
            from RealisticBlackHoleRenderer import RealisticBlackHoleRenderer
            renderer = RealisticBlackHoleRenderer(engine)
            renderer.run()
        except ImportError as e:
            print(f"Error loading ultra-realistic renderer: {e}")
            print("Make sure moderngl is installed: pip install moderngl")
            sys.exit(1)
    elif use_panda3d:
        print("Starting Panda3D Universe Renderer...")
        print("Professional black hole simulation with REAL NASA textures!")
        print("\nFeatures:")
        print("  - Real NASA/ESA satellite imagery")
        print("  - Physically-based rendering (PBR)")
        print("  - Gravitational lensing effects")
        print("  - Barnes-Hut N-body physics")
        print("  - Universe Sandbox quality visuals")
        print("\nControls:")
        print("  W/A/S/D: Move camera")
        print("  Arrow Keys: Rotate camera")
        print("  Q/E: Zoom in/out")
        print("  R: Reset view")
        print("  ESC: Exit")
        print("\nNOTE: Run 'python download_textures.py' first for best quality!")
        try:
            from PandaUniverse import PandaUniverseRenderer
            renderer = PandaUniverseRenderer(engine)
            renderer.run()
        except ImportError as e:
            print(f"Error loading Panda3D renderer: {e}")
            print("Make sure panda3d is installed:")
            print("  pip install panda3d")
            sys.exit(1)
    elif use_realistic:
        print("Starting Ultra-Realistic Black Hole Renderer...")
        print("HDR pipeline with bloom, PBR, and texture support!")
        print("\nControls:")
        print("  Arrow Keys: Rotate camera")
        print("  Z/X: Zoom in/out")
        print("  E/D: Adjust exposure")
        print("  B/N: Adjust bloom strength")
        print("  R: Reset view")
        print("  ESC: Exit")
        try:
            from RealisticRenderer import RealisticRenderer
            renderer = RealisticRenderer(engine)
            renderer.run()
        except ImportError as e:
            print(f"Error loading Realistic renderer: {e}")
            print("Make sure moderngl and Pillow are installed:")
            print("  pip install moderngl Pillow")
            print("Falling back to Interstellar renderer...")
            from InterstellarRenderer import InterstellarRenderer
            renderer = InterstellarRenderer(engine)
            renderer.run()
    elif use_interstellar:
        print("Starting Interstellar-style Black Hole Renderer...")
        print("Photorealistic gravitational lensing with stunning visuals!")
        print("\nControls:")
        print("  Arrow Keys: Rotate camera")
        print("  Z/X: Zoom in/out")
        print("  R: Reset view")
        print("  ESC: Exit")
        try:
            from InterstellarRenderer import InterstellarRenderer
            renderer = InterstellarRenderer(engine)
            renderer.run()
        except ImportError as e:
            print(f"Error loading Interstellar renderer: {e}")
            print("Make sure moderngl is installed: pip install moderngl")
            print("Falling back to GPU renderer...")
            from GPUBlackHoleRenderer import GPUBlackHoleRenderer
            renderer = GPUBlackHoleRenderer(engine)
            renderer.run()
    elif use_gpu:
        print("Starting GPU-accelerated renderer...")
        print("Smooth gravitational lensing on GPU!")
        print("\nControls:")
        print("  Arrow Keys: Rotate camera")
        print("  Z/X: Zoom in/out")
        print("  R: Reset view")
        print("  ESC: Exit")
        try:
            from GPUBlackHoleRenderer import GPUBlackHoleRenderer
            renderer = GPUBlackHoleRenderer(engine)
            renderer.run()
        except ImportError as e:
            print(f"Error loading GPU renderer: {e}")
            print("Make sure moderngl is installed: pip install moderngl")
            print("Falling back to standard renderer...")
            renderer = PygameRenderer(engine)
            renderer.run()
    else:
        print("Black Hole Visualization Engine")
        print("=" * 50)
        print("\nAvailable renderers:")
        print("  python main.py --ultra         - ULTRA-REALISTIC (RECOMMENDED!)")
        print("                                   Real Schwarzschild physics, HDRI, proper lensing")
        print("  python main.py --panda3d       - Panda3D Universe Renderer")
        print("                                   Real NASA textures, 3D, Universe Sandbox quality")
        print("  python main.py --realistic     - Ultra-Realistic HDR/PBR")
        print("  python main.py --interstellar  - Interstellar-style rendering")
        print("  python main.py --gpu           - GPU-accelerated lensing")
        print("  python main.py                 - Standard pygame renderer")
        print("\nFeatures:")
        print("  - Accurate Schwarzschild metric gravitational lensing")
        print("  - HDR rendering with bloom post-processing")
        print("  - Physically-based rendering (PBR)")
        print("  - Barnes-Hut N-body physics simulation")
        print("  - Photorealistic space environment")
        print("  - Real-time interactive controls")
        print("\nStarting with standard renderer...")
        renderer = PygameRenderer(engine)
        renderer.run()

if __name__ == "__main__":
    main()