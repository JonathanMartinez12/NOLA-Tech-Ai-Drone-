"""
Obstacle detection for collision avoidance.

Detects other drones and obstacles in the racing environment.
Phase 3+ implementation - stub provided for structure.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ObstacleDetection:
    """Result of obstacle detection."""

    detected: bool = False
    center_pixel: Optional[Tuple[int, int]] = None
    distance_m: float = 0.0
    size_m: float = 0.0
    confidence: float = 0.0
    obstacle_type: str = "unknown"  # "drone", "structure", "unknown"


@dataclass
class ObstacleDetector:
    """
    Obstacle detector for collision avoidance.

    Phase 3+ - will detect:
    - Other competing drones
    - Course structures
    - Unexpected obstacles

    Currently a stub for project structure.
    """

    def detect(self, image: NDArray[np.uint8]) -> List[ObstacleDetection]:
        """
        Detect obstacles in an image.

        Args:
            image: BGR image from camera

        Returns:
            List of detected obstacles
        """
        # TODO Phase 3: Implement obstacle detection
        # Options:
        # 1. YOLOv8 trained on drone/obstacle classes
        # 2. Depth-based detection with stereo/depth camera
        # 3. Motion detection for moving obstacles

        return []

    def get_nearest_obstacle(
        self,
        obstacles: List[ObstacleDetection]
    ) -> Optional[ObstacleDetection]:
        """Get the nearest detected obstacle."""
        if not obstacles:
            return None

        return min(obstacles, key=lambda o: o.distance_m)

    def is_path_clear(
        self,
        obstacles: List[ObstacleDetection],
        safety_distance: float = 2.0,
    ) -> bool:
        """
        Check if the forward path is clear of obstacles.

        Args:
            obstacles: List of detected obstacles
            safety_distance: Minimum safe distance in meters

        Returns:
            True if path is clear
        """
        for obs in obstacles:
            if obs.distance_m < safety_distance:
                return False
        return True
