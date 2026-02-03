"""
Depth estimation from visual data.

Estimates distances to objects using monocular depth estimation
or stereo vision if available.

Phase 3+ implementation - stub provided for structure.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DepthEstimator:
    """
    Depth estimation from camera images.

    Phase 3+ implementation options:
    1. Monocular depth estimation (MiDaS, DPT)
    2. Stereo vision (if stereo camera available)
    3. Structure from motion (SfM)

    Currently a stub for project structure.
    """

    def estimate_depth(self, image: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        """
        Estimate depth map from a single image.

        Args:
            image: BGR image from camera

        Returns:
            Depth map (same resolution as input) in meters, or None
        """
        # TODO Phase 3: Implement depth estimation
        # Options:
        # 1. MiDaS/DPT neural network for monocular depth
        # 2. Stereo matching if we have stereo cameras
        # 3. LiDAR integration if available

        return None

    def get_depth_at_point(
        self,
        depth_map: NDArray[np.float32],
        x: int,
        y: int,
        window_size: int = 5,
    ) -> float:
        """
        Get depth at a specific pixel location.

        Uses a small window average for robustness.

        Args:
            depth_map: Depth map from estimate_depth
            x: X pixel coordinate
            y: Y pixel coordinate
            window_size: Averaging window size

        Returns:
            Depth in meters
        """
        if depth_map is None:
            return 0.0

        h, w = depth_map.shape[:2]
        half = window_size // 2

        # Clamp to valid range
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)

        # Average depth in window
        window = depth_map[y1:y2, x1:x2]
        valid_depths = window[window > 0]

        if len(valid_depths) == 0:
            return 0.0

        return float(np.median(valid_depths))
