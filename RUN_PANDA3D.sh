#!/bin/bash
#
# Quick-start script for Panda3D Universe Renderer
# Run this to launch the black hole simulation with real NASA textures!
#

echo "======================================================================"
echo "  Panda3D Universe Renderer - Black Hole Simulation"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found!"
    echo "Please install Python 3.8+ first."
    exit 1
fi

# Check if panda3d is installed
if ! python -c "import panda3d" 2>/dev/null; then
    echo "⚠️  Panda3D not found. Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        exit 1
    fi
    echo "✅ Dependencies installed!"
    echo ""
fi

# Check if textures exist
if [ ! -f "textures/planets/earth_day_4k.jpg" ]; then
    echo "⚠️  NASA textures not found."
    echo "📥 Downloading real NASA/ESA satellite imagery..."
    echo ""
    python download_textures.py --quick
    if [ $? -ne 0 ]; then
        echo "⚠️  Texture download failed, but renderer will use fallbacks."
    else
        echo ""
        echo "✅ Textures downloaded successfully!"
    fi
    echo ""
fi

# Launch the renderer
echo "======================================================================"
echo "  Launching Panda3D Universe Renderer..."
echo "======================================================================"
echo ""
echo "Controls:"
echo "  Arrow Keys: Rotate camera around black hole"
echo "  Q/E: Zoom in/out"
echo "  R: Reset camera"
echo "  ESC: Exit"
echo ""
echo "Features:"
echo "  ✨ Real NASA Blue Marble Earth textures"
echo "  ✨ Gravitational lensing effects"
echo "  ✨ Physically-based rendering (PBR)"
echo "  ✨ Universe Sandbox quality visuals"
echo ""
echo "======================================================================"
echo ""

python main.py --panda3d

# Exit
echo ""
echo "Thanks for using Panda3D Universe Renderer!"
