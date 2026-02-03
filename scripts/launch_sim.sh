#!/bin/bash
# Launch PX4 SITL with Gazebo Harmonic
# Usage: ./scripts/launch_sim.sh

set -e

PX4_DIR="${PX4_AUTOPILOT_DIR:-$HOME/PX4-Autopilot}"

echo "=========================================="
echo "AI Grand Prix - Launching Simulator"
echo "=========================================="

# Check if PX4 directory exists
if [ ! -d "$PX4_DIR" ]; then
    echo "Error: PX4 Autopilot not found at $PX4_DIR"
    echo "Set PX4_AUTOPILOT_DIR environment variable or install PX4"
    exit 1
fi

echo "PX4 Directory: $PX4_DIR"
echo "Starting PX4 SITL with Gazebo x500 drone..."
echo ""

cd "$PX4_DIR"
make px4_sitl gz_x500
