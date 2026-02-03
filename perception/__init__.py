"""
Perception module for AI Grand Prix Drone Racing.

This module handles visual perception:
- Camera interface for capturing frames
- Gate detection (HSV color-based and ML-based)
- Obstacle detection
- Distance estimation

Phase 2+ implementation - stubs provided for structure.
"""

from perception.gate_detector import GateDetector, GateDetection

__all__ = ["GateDetector", "GateDetection"]
