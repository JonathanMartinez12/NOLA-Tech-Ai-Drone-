"""
Utility modules for AI Grand Prix Drone Racing.

This package contains helper functions and utilities used throughout
the drone racing system, including logging, math operations, and
visualization tools.
"""

from utils.logger import setup_logger, get_logger
from utils.math_helpers import (
    normalize_angle,
    angle_difference,
    clamp,
    vector_magnitude,
    vector_normalize,
    ned_to_enu,
    enu_to_ned,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "normalize_angle",
    "angle_difference",
    "clamp",
    "vector_magnitude",
    "vector_normalize",
    "ned_to_enu",
    "enu_to_ned",
]
