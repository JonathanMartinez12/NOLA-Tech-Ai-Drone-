"""
Planning module for AI Grand Prix Drone Racing.

This module handles path planning and trajectory generation:
- Path planning (A*, RRT)
- Trajectory optimization
- Racing line calculation
- Collision avoidance

Phase 3+ implementation - stubs provided for structure.
"""

from planning.path_planner import PathPlanner
from planning.trajectory_generator import TrajectoryGenerator

__all__ = ["PathPlanner", "TrajectoryGenerator"]
