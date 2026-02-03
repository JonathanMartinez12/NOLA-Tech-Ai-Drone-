"""
State estimation module for AI Grand Prix Drone Racing.

This module handles:
- Drone state tracking (position, velocity, attitude)
- Sensor fusion with Kalman filtering
- World model for tracking gates and obstacles
"""

from state_estimation.drone_state import DroneState

__all__ = ["DroneState"]
