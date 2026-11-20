# Rust Black Hole Renderer

A photorealistic black hole renderer implemented in Rust using wgpu compute shaders, featuring real Schwarzschild geodesic integration and volumetric accretion disk rendering.

## Features

### Physics
- **Schwarzschild Geodesic Integration**: Real general relativity calculations using RK4 integration
- **Gravitational Lensing**: Light paths bent by spacetime curvature
- **Gravitational Redshift**: Frequency shifts due to gravitational potential
- **Doppler Beaming**: Relativistic Doppler effects from disk rotation

### Rendering
- **Volumetric Accretion Disk**: 3D disk with proper density falloff and temperature gradient
- **Temperature-Based Emission**: Blackbody radiation from 10^4 K (red) to 10^7 K (blue-white)
- **Self-Shadowing**: Disk occludes itself for realistic depth
- **HDR Tone Mapping**: ACES tone mapping for photorealistic output
- **Environment Map**: Procedural starfield background

### Performance
- **GPU Compute Shaders**: Ray tracing runs entirely on GPU (Metal backend on M3)
- **Optimized for M3 MacBook Air**: Targets 30-60 FPS at 1280x800
- **Adaptive Quality**: Configurable ray marching steps for performance tuning

## Build and Run

### Prerequisites
- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- macOS with Metal support (tested on M3 MacBook Air)

### Build
```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

### Run
```bash
# Run in release mode (recommended for performance)
cargo run --release

# Run with debug output
RUST_LOG=debug cargo run --release
```

## Controls

- **W/A/S/D** - Move camera forward/left/backward/right
- **Space** - Move camera up
- **Left Shift** - Move camera down
- **Left Mouse + Drag** - Rotate camera view
- **ESC** - Exit

## Physics Parameters

### Black Hole
- **Mass**: 10 solar masses (2×10^31 kg)
- **Schwarzschild Radius**: 29.54 km
- **Event Horizon**: 1.01 × Rs (rendering cutoff)

### Accretion Disk
- **Inner Radius**: 3 Rs (ISCO - Innermost Stable Circular Orbit)
- **Outer Radius**: 12 Rs
- **Vertical Thickness**: 0.5 Rs (scale height increases with radius)
- **Temperature**: 10^7 K (inner) to 10^4 K (outer)
- **Orbital Velocity**: Keplerian (v = √(GM/r))

### Camera
- **Initial Position**: 15 Rs from black hole
- **Movement Speed**: 50 km/s
- **Field of View**: 60 degrees

## Technical Details

### Architecture
```
main.rs (850 lines)
├── Camera System
│   ├── FPS-style controls (WASD + mouse)
│   └── Uniform buffer updates
├── Renderer
│   ├── wgpu device/queue setup
│   ├── Compute pipeline (ray tracing)
│   └── Display pipeline (tone mapping)
└── Input Handling

shaders/compute.wgsl (450 lines)
├── Schwarzschild Geodesic Integration
│   ├── RK4 integrator (4th order accuracy)
│   └── Geodesic acceleration calculation
├── Volumetric Disk Sampling
│   ├── 3D density profile (radial + vertical)
│   └── Temperature gradient
├── Emission-Absorption Integration
│   ├── Blackbody radiation
│   ├── Beer's law absorption
│   ├── Doppler beaming (4th power)
│   └── Gravitational redshift
└── Environment Map Sampling

shaders/display.wgsl (90 lines)
├── ACES Tone Mapping
├── Gamma Correction (sRGB)
└── Full-screen quad rendering
```

### Performance Optimization

#### M3-Specific
- **Metal Backend**: Native Apple Silicon GPU support
- **8×8 Workgroups**: Optimized for Apple GPU architecture
- **16-bit Float Textures**: HDR output with reduced memory bandwidth
- **VSync**: Locked to 60 FPS to prevent thermal throttling

#### Ray Marching
- **256 Max Steps**: Balance between quality and performance
- **Adaptive Step Size**: 0.5 geometric units
- **Early Termination**: Stops at event horizon or when fully opaque
- **Distance Culling**: Escapes to environment map at 1000 km

#### Memory Usage
- **Compute Texture**: ~8 MB (1280×800 RGBA16F)
- **Environment Map**: ~16 MB (2048×2048 RGBA8)
- **Uniform Buffers**: <1 KB
- **Total GPU Memory**: ~30-50 MB

### Shader Mathematics

#### Schwarzschild Metric
```
ds² = -(1 - Rs/r) c²dt² + (1 - Rs/r)⁻¹ dr² + r²(dθ² + sin²θ dφ²)

where Rs = 2GM/c² (Schwarzschild radius)
```

#### Geodesic Equation (Cartesian)
```
d²x/dτ² = -1.5 Rs/r² [(dr/dτ)² - (1/3)|v_tangential|²] r̂
          - 1.5 Rs/r² (2 dr/dτ) v_tangential/r
```

#### Disk Density Profile
```
ρ(r, z) = ρ₀ (r_in/r)^2.75 exp(-z²/(2h²))

where h(r) = h₀ (r/r_in) (scale height)
```

#### Blackbody Emission
```
B(T) ∝ T⁴ (Stefan-Boltzmann law)
Color mapped via Wien's displacement law
```

#### Doppler Beaming
```
I_obs = I_rest × (1 + v·n/c)⁴

where v = √(GM/r) θ̂ (Keplerian velocity)
```

## Customization

### Modify Black Hole Mass
Edit `src/main.rs`:
```rust
let schwarzschild_radius = 29.54; // km
// Change to: 2.95 for 1 solar mass
//            295.4 for 100 solar masses
```

### Adjust Disk Parameters
Edit `src/main.rs`:
```rust
let disk_params_uniform = DiskParamsUniform {
    inner_radius: schwarzschild_radius * 3.0,  // ISCO
    outer_radius: schwarzschild_radius * 12.0, // Outer edge
    thickness: schwarzschild_radius * 0.5,     // Vertical scale
    density: 1.0,                              // Brightness
    temperature_inner: 10000000.0,             // Hot (blue)
    temperature_outer: 10000.0,                // Cool (red)
    _padding: [0.0; 2],
};
```

### Tune Ray Marching Quality
Edit `shaders/compute.wgsl`:
```wgsl
let max_steps = 256;  // Increase for better quality (slower)
let dt = 0.5;         // Decrease for finer integration (slower)
```

## Troubleshooting

### Low Frame Rate
1. Lower resolution: Edit window size in `src/main.rs`
2. Reduce ray marching steps in `shaders/compute.wgsl`
3. Decrease workgroup count (affects resolution)

### Black Screen
- Check Metal backend is available (macOS only)
- Verify GPU supports compute shaders
- Check console for shader compilation errors

### Crashes on Startup
- Update Rust: `rustup update`
- Update dependencies: `cargo update`
- Check macOS version (requires 10.13+)

## Future Enhancements

- [ ] Multiple black holes (binary systems)
- [ ] Kerr metric (rotating black holes)
- [ ] Photon sphere visualization
- [ ] Adjustable time dilation effects
- [ ] Save rendered frames to disk
- [ ] Real-time parameter adjustment UI
- [ ] Neutron star rendering
- [ ] Wormhole visualization

## References

1. **General Relativity**: Misner, Thorne, Wheeler - "Gravitation"
2. **Numerical Methods**: Press et al. - "Numerical Recipes"
3. **Accretion Disks**: Shakura & Sunyaev (1973)
4. **Rendering**: Interstellar (2014) - Kip Thorne's visualizations

## License

MIT License - See LICENSE file

## Acknowledgments

- Inspired by the visual effects work on "Interstellar" (2014)
- Physics calculations based on Kip Thorne's research
- Built with Rust's wgpu graphics library
- Optimized for Apple Silicon (M1/M2/M3)
