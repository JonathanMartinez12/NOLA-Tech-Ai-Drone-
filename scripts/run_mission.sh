#!/bin/bash
# Run the waypoint mission
# Usage: ./scripts/run_mission.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "AI Grand Prix - Waypoint Mission"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "$PROJECT_DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Run the mission
cd "$PROJECT_DIR"
python main.py "$@"
