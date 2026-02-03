"""
Camera interface for capturing frames from the drone.

Provides interface to the simulated camera in Gazebo and will
support real camera hardware when available.

Phase 2 implementation - stub provided for structure.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


@dataclass
class CameraFrame:
    """Container for a camera frame with metadata."""

    image: NDArray[np.uint8]  # BGR image
    timestamp: float  # Capture time
    frame_id: int  # Sequential frame number
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if self.image is not None:
            self.height, self.width = self.image.shape[:2]


@dataclass
class Camera:
    """
    Camera interface for drone vision.

    In simulation, captures from Gazebo camera topic.
    For real hardware, will interface with actual camera.

    This is a stub for Phase 2 - will be implemented with ROS2
    integration or direct Gazebo transport.

    Example:
        camera = Camera()
        await camera.start()

        frame = camera.get_frame()
        if frame is not None:
            process_image(frame.image)

        await camera.stop()
    """

    _is_running: bool = field(default=False, init=False)
    _current_frame: Optional[CameraFrame] = field(default=None, init=False)
    _frame_count: int = field(default=0, init=False)

    def __post_init__(self):
        self.settings = get_settings()

    async def start(self) -> bool:
        """
        Start the camera capture.

        Returns:
            True if camera started successfully
        """
        logger.info("Starting camera (stub - not implemented)")
        self._is_running = True

        # TODO Phase 2: Subscribe to Gazebo camera topic
        # Options:
        # 1. ROS2 subscription to /camera/image_raw
        # 2. Direct Gazebo transport subscription
        # 3. MAVSDK camera plugin (if available)

        return True

    async def stop(self) -> None:
        """Stop the camera capture."""
        logger.info("Stopping camera")
        self._is_running = False

    def get_frame(self) -> Optional[CameraFrame]:
        """
        Get the most recent camera frame.

        Returns:
            CameraFrame if available, None otherwise
        """
        # TODO Phase 2: Return actual camera frame
        return self._current_frame

    def get_frame_sync(self) -> Optional[CameraFrame]:
        """Synchronous version of get_frame."""
        return self._current_frame

    @property
    def is_running(self) -> bool:
        """Whether camera capture is active."""
        return self._is_running

    @property
    def frame_rate(self) -> float:
        """Current frame rate in Hz."""
        return self.settings.perception.camera_fps

    @property
    def resolution(self) -> tuple:
        """Camera resolution (width, height)."""
        return self.settings.perception.camera_resolution
