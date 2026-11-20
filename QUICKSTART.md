# Black Hole Renderer - Quick Start

## Running the Renderer

The Rust black hole renderer is **ready to run**! Simply execute:

```bash
cargo run --release
```

A window will open showing a photorealistic black hole with an accretion disk.

## What You'll See

### Visual Features
- **Black sphere** in the center (event horizon)
- **Glowing accretion disk** orbiting the black hole
  - Blue-white inner regions (10 million K)
  - Orange-red outer regions (10,000 K)
- **Gravitational lensing** - stars appear distorted near the black hole
- **Einstein ring** - background stars lensed into a ring
- **Doppler beaming** - one side of the disk brighter than the other
- **Starfield background** - procedurally generated stars

### Camera Controls
```
W/S - Forward/Backward
A/D - Left/Right
Space - Up
Shift - Down
Left Mouse + Drag - Rotate view
ESC - Exit
```

## Performance

**Expected FPS on M3 MacBook Air:**
- 30-60 FPS at 1280×800 resolution
- Real-time physics simulation
- GPU-accelerated ray tracing

## Project Files

### Source Code
- `src/main.rs` - Rust application (850 lines)
  - Window/event handling
  - wgpu setup (Metal backend)
  - Camera controls
  - Render loop

### Shaders (WGSL)
- `shaders/compute.wgsl` - Ray tracer (450 lines)
  - Schwarzschild geodesic integration
  - Volumetric disk rendering
  - Doppler/redshift calculations

- `shaders/display.wgsl` - Display shader (90 lines)
  - HDR tone mapping (ACES)
  - Gamma correction
  - Final output

## Quick Reference

| Action | Command |
|--------|---------|
| **Run** | `cargo run --release` |
| **Build only** | `cargo build --release` |
| **Clean build** | `cargo clean && cargo build --release` |
| **Enable logs** | `RUST_LOG=debug cargo run --release` |
| **Binary location** | `target/release/black_hole_renderer` |

---

**Enjoy exploring the black hole!**

For detailed technical documentation, see `RUST_RENDERER.md`.
