# AI Grand Prix - Autonomous Drone Racing

Autonomous drone racing software for the **Anduril AI Grand Prix 2026** competition.

> **Prize:** $500,000 and a job at Anduril
> **Season 1:** Spring 2026, culminating in AI Grand Prix Ohio

## Overview

This is a complete autonomy stack for drone racing, designed to run on Neros.tech racing drones with PX4 flight controller. The system implements the full autonomy pipeline:

```
PERCEIVE → LOCALIZE → PLAN → ACT
```

### Current Status: Phase 1 Complete

- [x] **Phase 1: Foundation** - Waypoint navigation with smooth velocity control
- [ ] Phase 2: Perception - Gate detection and tracking
- [ ] Phase 3: Autonomous Navigation - Fly through detected gates
- [ ] Phase 4: Racing Optimization - Minimum-time trajectories
- [ ] Phase 5: Competition Ready - Robustness and reliability

## Quick Start

### Prerequisites

1. **Ubuntu 24.04** (native or WSL2)
2. **PX4 Autopilot** with SITL support
3. **Gazebo Harmonic** simulator
4. **Python 3.10+**

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NOLA-Tech-Ai-Drone-.git
cd NOLA-Tech-Ai-Drone-

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

1. **Start PX4 SITL with Gazebo:**
   ```bash
   cd ~/PX4-Autopilot
   make px4_sitl gz_x500
   ```

2. **In another terminal, run the waypoint mission:**
   ```bash
   cd ~/NOLA-Tech-Ai-Drone-
   source venv/bin/activate
   python main.py
   ```

3. **For a smaller test pattern:**
   ```bash
   python main.py --test
   ```

### Command Line Options

```
python main.py [options]

Options:
  -c, --connection    MAVSDK connection string (default: udp://:14540)
  -s, --speed         Flight speed in m/s (default: 2.0)
  -t, --test          Run small test pattern instead of full demo
  -v, --verbose       Enable verbose (DEBUG) logging
```

## Project Structure

```
drone-racing/
├── main.py                     # Entry point - runs waypoint mission
├── config/
│   ├── settings.py             # All tunable parameters
│   └── drone_specs.py          # Physical drone specifications
├── perception/                 # Vision and sensing (Phase 2+)
│   ├── camera.py               # Camera interface
│   ├── gate_detector.py        # Gate detection (HSV + YOLO)
│   ├── obstacle_detector.py    # Obstacle detection
│   └── depth_estimator.py      # Distance estimation
├── state_estimation/           # Localization and tracking
│   ├── drone_state.py          # Telemetry subscription
│   ├── kalman_filter.py        # Sensor fusion (EKF)
│   └── world_model.py          # Track gates and obstacles
├── planning/                   # Path and trajectory planning
│   ├── path_planner.py         # High-level path planning
│   ├── trajectory_generator.py # Smooth trajectory generation
│   ├── racing_line.py          # Racing line optimization
│   └── collision_avoidance.py  # Obstacle avoidance
├── control/                    # Flight control
│   ├── flight_controller.py    # High-level commands
│   ├── offboard_controller.py  # Velocity/attitude control
│   └── pid_controller.py       # PID implementations
├── utils/                      # Utilities
│   ├── logger.py               # Structured logging
│   ├── math_helpers.py         # Vector math, transforms
│   ├── telemetry_recorder.py   # Flight data recording
│   └── visualization.py        # Debug visualization
├── tests/                      # Test suite
├── scripts/                    # Helper scripts
├── requirements.txt
└── README.md
```

## Architecture

### Main Control Loop

The system runs at 50Hz, executing:

1. **PERCEIVE**: Capture camera frames, detect gates and obstacles
2. **LOCALIZE**: Fuse sensor data, estimate current state
3. **PLAN**: Calculate optimal path to next gate
4. **ACT**: Send velocity commands to flight controller

### Coordinate Frames

- **NED (North-East-Down)**: Global reference frame used by PX4
- **Body Frame**: Drone-relative (forward, right, down)
- **Camera Frame**: Image coordinates for perception

### Key Components

| Component | Description |
|-----------|-------------|
| `DroneState` | Real-time telemetry tracking via MAVSDK |
| `FlightController` | High-level commands (takeoff, goto, land) |
| `OffboardController` | Direct velocity/position control |
| `GateDetector` | Visual gate detection (HSV, YOLO) |
| `PathPlanner` | Gate-to-gate path planning |
| `TrajectoryGenerator` | Smooth trajectory generation |

## Configuration

All tunable parameters are in `config/settings.py`:

```python
# Control parameters
max_velocity = 5.0              # m/s
max_vertical_velocity = 2.0     # m/s
position_threshold = 0.5        # meters (waypoint acceptance)

# Perception parameters
gate_color_lower = (5, 100, 100)    # HSV lower bound
gate_color_upper = (25, 255, 255)   # HSV upper bound

# Safety limits
min_altitude = 1.0              # meters
max_altitude = 50.0             # meters
geofence_radius = 100.0         # meters
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_math_helpers.py -v
```

### Code Style

```bash
# Format code
black .

# Type checking
mypy .

# Lint
ruff check .
```

### Adding New Features

1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request

## Phase 2: Perception (Next Steps)

To implement gate detection:

1. **Enable camera in Gazebo** - Add camera sensor to drone model
2. **Subscribe to camera topic** - Via ROS2 or Gazebo transport
3. **Implement HSV detection** - `perception/gate_detector.py`
4. **Add visual servoing** - Fly toward detected gates

## Telemetry and Debugging

### Viewing Logs

Logs are written to `logs/` with timestamps:
```bash
tail -f logs/drone_racing_*.log
```

### Telemetry Recording

Flight data is saved to `telemetry_data/`:
```bash
# View recorded data
python -c "import pandas as pd; print(pd.read_csv('telemetry_data/telemetry_*.csv'))"
```

### Visualization

```python
from utils.visualization import plot_flight_path_2d

positions = [...]  # From telemetry
plot_flight_path_2d(positions, output_path="flight.png")
```

## Safety

The system includes multiple safety features:

- **Geofencing**: Configurable flight boundary
- **Altitude limits**: Min/max altitude enforcement
- **Velocity limits**: Hardware-respecting speed caps
- **Emergency stop**: Immediate halt capability
- **Graceful shutdown**: Signal handling for clean exit

## Competition Notes

### Neros.tech Drone Specs (Estimated)

| Parameter | Value |
|-----------|-------|
| Max Speed | 20 m/s |
| Max Acceleration | 10 m/s² |
| Weight | ~1.5 kg |
| Flight Time | ~8 minutes |

### Race Strategy

1. **Reliability first** - Complete every lap
2. **Smooth trajectories** - Minimize energy loss
3. **Aggressive corners** - Optimal racing lines
4. **Consistent performance** - Predictable lap times

## Resources

- [PX4 Autopilot](https://px4.io/)
- [MAVSDK Python](https://mavsdk.mavlink.io/main/en/python/)
- [Gazebo Harmonic](https://gazebosim.org/)
- [OpenCV](https://opencv.org/)
- [YOLOv8](https://docs.ultralytics.com/)

## License

MIT License - See LICENSE file

## Author

AI Grand Prix Competitor
LSU Computer Science
Competing for $500k and a job at Anduril

---

**Let's race!**
