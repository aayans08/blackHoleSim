# Panda3D Universe Renderer - User Guide

## Overview

The **Panda3D Universe Renderer** is a professional-quality black hole and universe simulation system that rivals **Universe Sandbox** and **Space Engine** in visual quality. It uses:

- **REAL NASA/ESA textures** from satellite missions (Blue Marble, LRO, Cassini, SDO, etc.)
- **Physically-based rendering (PBR)** for realistic material properties
- **Gravitational lensing effects** based on the Schwarzschild metric
- **Barnes-Hut N-body physics** for accurate multi-body simulations
- **Atmospheric scattering** for realistic planet atmospheres

## Quick Start

### 1. Install Dependencies

```bash
pip install panda3d numpy Pillow
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

### 2. Download Real NASA Textures

Download REAL satellite imagery textures:

```bash
# Full download (all planets)
python download_textures.py

# Quick download (essentials: Earth, Sun, Milky Way)
python download_textures.py --quick
```

This downloads:
- **Earth**: NASA Blue Marble day/night/clouds textures (2K-4K)
- **Moon**: Lunar Reconnaissance Orbiter texture
- **Mars**: Viking/HiRISE surface data
- **Sun**: Solar Dynamics Observatory imagery
- **Jupiter**: Juno mission data
- **Saturn**: Cassini mission data + rings
- **Milky Way**: ESO panorama background

**Note**: Textures are ~50-100MB total. The renderer works without them (uses procedural fallbacks), but REAL textures look dramatically better!

### 3. Run the Renderer

```bash
python main.py --panda3d
```

## Controls

### Camera Movement
- **Arrow Keys**: Rotate camera around black hole
  - Left/Right: Rotate horizontally
  - Up/Down: Rotate vertically
- **Q**: Zoom out
- **E**: Zoom in
- **R**: Reset camera to default position

### System
- **ESC**: Exit the renderer

## Features

### 1. Black Hole Visualization
The renderer accurately depicts:
- **Event Horizon**: The point of no return (Schwarzschild radius Rs = 2GM/c²)
- **Photon Sphere**: 1.5 × Rs where light orbits the black hole
- **ISCO**: Innermost Stable Circular Orbit (3 × Rs)
- **Accretion Disk**: Temperature-gradient colored particles with realistic physics

### 2. Real Celestial Bodies
All planets/stars use REAL NASA textures:
- **Earth**:
  - Day texture from Blue Marble
  - Night texture showing city lights
  - Cloud layer
  - Atmospheric rim lighting
- **Sun**:
  - Real solar surface from SDO
  - Self-illumination and point light
- **Other planets**: Mars, Moon, Jupiter, Saturn available

### 3. Space Environment
- **Milky Way Background**: Real ESO gigapixel panorama
- **Star Field**: 5000+ procedural stars with realistic colors (or real star catalog)
- **Realistic Lighting**: PBR materials with HDR

### 4. Physics Integration
- Connected to Barnes-Hut N-body simulation engine
- Real-time gravitational calculations
- Support for multiple bodies orbiting the black hole

## Performance Optimization (M3 MacBook Air)

The renderer is specifically optimized for Apple Silicon (M3) with 16GB RAM:

### Graphics Settings
- **Anti-aliasing**: 4x MSAA (adjustable)
- **Texture Resolution**: 2K-4K (optimized for ~8GB RAM usage)
- **Frame Rate**: Target 60 FPS for single black hole + planets
- **Anisotropic Filtering**: 16x for sharp textures

### Quality Modes
You can adjust quality in the code (`PandaUniverse.py`):

```python
# In configure_panda3d():
multisamples 2    # Lower AA for better performance
multisamples 8    # Higher AA for better quality
```

### Memory Management
- Compressed textures (automatic)
- Mipmapping enabled for distant objects
- LOD (Level of Detail) system for planets
- Efficient geometry caching

### Expected Performance
| Scene | FPS Target | Memory Usage |
|-------|-----------|--------------|
| Black hole only | 120+ | ~2GB |
| Black hole + Earth | 90+ | ~3GB |
| Full solar system | 60+ | ~5GB |
| Complex scenario | 30+ | ~7GB |

## Advanced Usage

### Customizing the Scene

Edit `PandaUniverse.py` to customize:

```python
# Change black hole mass (in solar masses)
engine = BlackHoleEngine(mass=10, position=[0,0,0], velocity=[0,0,0])

# Add more planets
mars = self.create_planet(
    name="Mars",
    radius=3.389,  # km
    position=(150, 50, 0),
    texture_day='mars',
    has_atmosphere=False
)

# Adjust camera distance
self.camera_distance = 100.0  # Units
```

### Using Custom Textures

Place your own textures in the `textures/` directory:
- `textures/planets/` - Planet textures
- `textures/stars/` - Star textures
- `textures/space/` - Space backgrounds

Format: JPG or PNG, power-of-two dimensions recommended (1024, 2048, 4096)

### Physics Simulation

The renderer integrates with `blackHoleEngine.py`:

```python
# Add orbiting bodies
engine.add_body(
    mass=5.972e24,  # Earth mass in kg
    position=[100000, 0, 0],  # meters
    velocity=[0, 30000, 0]  # m/s
)
```

Bodies will be automatically rendered and updated in real-time!

## Shaders

The renderer includes professional GLSL shaders:

### Available Shaders
1. **atmosphere.vert/frag**: Rayleigh + Mie atmospheric scattering
2. **lensing.vert/frag**: Gravitational lensing (Schwarzschild metric)
3. **pbr.vert/frag**: Physically-based rendering for realistic materials

### Custom Shaders
You can add custom shaders in the `shaders/` directory and apply them:

```python
shader = Shader.load(Shader.SL_GLSL,
    vertex='shaders/custom.vert',
    fragment='shaders/custom.frag'
)
planet.setShader(shader)
```

## Troubleshooting

### "No textures loaded" Warning
**Solution**: Run `python download_textures.py` to download NASA textures.
The renderer will work with procedural fallbacks but won't look as realistic.

### Low Frame Rate
**Solutions**:
1. Reduce anti-aliasing: Change `multisamples` in `configure_panda3d()`
2. Lower texture resolution: Download only 2K textures instead of 4K
3. Reduce number of celestial bodies
4. Close other applications to free RAM

### Black Screen
**Solutions**:
1. Check that Panda3D is installed: `pip install panda3d`
2. Verify graphics drivers are up to date
3. Try running in windowed mode (default)

### Import Errors
**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

## Comparison to Other Renderers

| Feature | Panda3D | Interstellar | GPU | Pygame |
|---------|---------|-------------|-----|--------|
| Dimension | 3D | 2D | 2D | 2D |
| Textures | REAL NASA | Procedural | Procedural | Simple |
| Physics | Full 3D | 2D | 2D | 2D |
| Lensing | Yes (3D) | Yes (2D) | Yes (2D) | No |
| PBR | Yes | Partial | No | No |
| Performance | 60+ FPS | 60+ FPS | 90+ FPS | 120+ FPS |
| Quality | Universe Sandbox | Interstellar Movie | High | Basic |

**Recommendation**: Use `--panda3d` for the best visual quality and realism!

## Technical Details

### Coordinate System
- Origin: Black hole center
- Units: Kilometers (for visualization)
- Axes: Right-handed (X: right, Y: forward, Z: up)

### Schwarzschild Metric Implementation
The gravitational lensing shader uses a simplified Schwarzschild metric:

```glsl
float deflection = schwarzschildRadius / (distance * distance);
rayDirection += normalize(toBlackHole) * deflection * stepSize;
```

For full accuracy, see the academic papers in `TECHNICAL_REFERENCE.md`.

### Barnes-Hut Physics
The N-body simulation uses an octree with θ = 0.5 for O(N log N) performance.
See `blackHoleEngine.py` for implementation details.

## Credits

### Textures
- NASA Blue Marble (Earth day/night)
- NASA LRO (Moon)
- Solar System Scope (planets - free for non-commercial use)
- ESO (Milky Way panorama)

### Frameworks
- Panda3D: Disney/Carnegie Mellon game engine
- NumPy: Numerical computing

### Physics
- Schwarzschild metric for gravitational lensing
- Barnes-Hut algorithm for N-body simulation

## License

This project is for educational and non-commercial use.
NASA imagery is public domain. Solar System Scope textures are free for non-commercial use.

## Support

For issues or questions:
1. Check this guide
2. Review `CLAUDE.md` for technical architecture
3. Check `TECHNICAL_REFERENCE.md` for physics details
4. Review the source code in `PandaUniverse.py`

## Future Enhancements

Potential improvements:
- [ ] Full Kerr metric for rotating black holes
- [ ] Volumetric nebulae with ray marching
- [ ] Real-time accretion disk physics
- [ ] VR support
- [ ] Multi-black hole systems
- [ ] Wormhole visualization
- [ ] Gravitational wave visualization

---

**Enjoy exploring the universe!** 🌌
