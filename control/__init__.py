"""
Control module for AI Grand Prix Drone Racing.

This module provides flight control interfaces:
- FlightController: High-level commands (takeoff, land, goto)
- OffboardController: Low-level velocity/attitude control
- PID controllers for position and velocity tracking
"""

from control.flight_controller import FlightController
from control.offboard_controller import OffboardController

__all__ = ["FlightController", "OffboardController"]
