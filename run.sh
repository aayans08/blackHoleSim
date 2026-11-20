#!/bin/bash

# Black Hole Visualization Launcher

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q pygame numpy scipy moderngl numba 2>/dev/null

# Run with GPU by default, fall back to improved CPU if fails
echo "Starting Black Hole Visualization..."
echo "==============================="
echo ""

# Try GPU first
python main.py --gpu 2>/dev/null

# If GPU fails, try improved CPU
if [ $? -ne 0 ]; then
    echo "GPU mode failed, trying improved CPU renderer..."
    python main.py --improved --quality medium
fi