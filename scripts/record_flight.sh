#!/bin/bash
# Record flight telemetry
# Usage: ./scripts/record_flight.sh [output_name]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_NAME="${1:-flight_$(date +%Y%m%d_%H%M%S)}"

echo "=========================================="
echo "AI Grand Prix - Flight Recording"
echo "=========================================="
echo "Recording: $OUTPUT_NAME"

# Activate virtual environment if it exists
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Run with telemetry recording enabled
cd "$PROJECT_DIR"
DRONE_LOGGING__RECORD_TELEMETRY=true python main.py -v "$@"

echo ""
echo "Recording saved to telemetry_data/"
