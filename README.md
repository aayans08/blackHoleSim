# Black Hole Visualization Engine

**Quality over quantity** - A streamlined, production-quality black hole visualization with accurate physics and photorealistic rendering.

## Features

### Physics
- **Barnes-Hut N-body simulation** (O(N log N) gravitational calculations)
- **Schwarzschild metric** gravitational lensing
- Accurate black hole radii:
  - Event horizon (Rs = 2GM/c²)
  - Photon sphere (1.5 Rs)
  - ISCO (3 Rs)
- Velocity Verlet integration for orbital mechanics

### Rendering
- **InterstellarRenderer** (recommended): Movie-quality GLSL shaders
  - Real-time gravitational lensing via ray marching
  - Procedural planets with atmospheres
  - Multi-layer star fields with depth
  - Realistic nebulae (emission, reflection, dark)
  - Spiral, elliptical, and irregular galaxies
  - Post-processing (HDR, tone mapping, vignette)
- **GPUBlackHoleRenderer**: Lightweight GPU fallback
- **PygameRenderer**: Simple CPU visualization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run best quality renderer (recommended)
python main.py --interstellar

# Or GPU fallback
python main.py --gpu

# Or basic CPU renderer
python main.py
```

## Controls

- **Arrow Keys**: Rotate camera
- **Z/X**: Zoom in/out
- **R**: Reset view
- **ESC**: Exit

## System Requirements

- Python 3.8+
- Modern GPU with OpenGL 3.3+ (for --interstellar and --gpu modes)
- ~100MB RAM for rendering

## Documentation

See [CLAUDE.md](CLAUDE.md) for complete technical reference including:
- Physics engine API
- Barnes-Hut algorithm details
- Shader customization guide
- Performance optimization tips
- Development guidelines

## Architecture

```
blackHoleSim/
├── blackHoleEngine.py         # Physics + Barnes-Hut N-body
├── InterstellarRenderer.py    # Premium GLSL renderer
├── GPUBlackHoleRenderer.py    # GPU fallback
├── PygameRenderer.py          # CPU fallback
└── main.py                    # Entry point
```

## What Changed (Refactor)

**Removed:**
- 10+ redundant/low-quality renderers
- Unused dependencies (scipy, numba, einsteinpy, matplotlib, ursina)
- Texture download scripts
- Duplicate physics implementations

**Added:**
- Barnes-Hut algorithm for efficient N-body physics
- Complete physics engine API (ISCO, photon sphere calculations)
- Comprehensive technical documentation

**Result:** Clean, maintainable codebase focused on the best implementations.

## Contributing

See [CLAUDE.md](CLAUDE.md) Development Guidelines section.

## License

See LICENSE file.