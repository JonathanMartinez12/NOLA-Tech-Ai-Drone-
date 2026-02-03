"""
Kalman Filter implementation for sensor fusion.

Provides Extended Kalman Filter (EKF) for fusing GPS, IMU, and visual
position estimates into a robust state estimate.

This is a simplified implementation for Phase 1. Can be extended with:
- More sophisticated process models
- Additional sensor inputs
- Adaptive noise estimation
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


@dataclass
class KalmanFilter:
    """
    Extended Kalman Filter for drone state estimation.

    State vector: [x, y, z, vx, vy, vz]
    - Position (x, y, z) in NED frame
    - Velocity (vx, vy, vz) in NED frame

    This is a constant-velocity model with process noise.
    """

    # State dimension
    n_states: int = 6

    # State vector [x, y, z, vx, vy, vz]
    state: NDArray[np.float64] = field(default_factory=lambda: np.zeros(6))

    # State covariance matrix
    P: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(6) * 10.0  # Initial uncertainty
    )

    # Process noise covariance
    Q: NDArray[np.float64] = field(default_factory=lambda: np.eye(6))

    # Measurement noise covariance (for GPS position)
    R_gps: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(3) * 0.5  # GPS noise
    )

    # Time of last update
    last_update_time: float = 0.0

    # Initialization flag
    is_initialized: bool = False

    def __post_init__(self):
        """Initialize noise matrices from settings."""
        settings = get_settings()

        # Process noise
        pos_noise = settings.state_estimation.position_process_noise
        vel_noise = settings.state_estimation.velocity_process_noise
        self.Q = np.diag([pos_noise, pos_noise, pos_noise,
                         vel_noise, vel_noise, vel_noise])

        # Measurement noise
        gps_noise = settings.state_estimation.gps_position_noise
        self.R_gps = np.eye(3) * (gps_noise ** 2)

    def initialize(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        timestamp: float = 0.0,
    ) -> None:
        """
        Initialize the filter with known state.

        Args:
            position: Initial position (x, y, z) in NED frame
            velocity: Initial velocity (vx, vy, vz)
            timestamp: Current timestamp in seconds
        """
        self.state = np.array([
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2]
        ])

        # Reset covariance to initial uncertainty
        self.P = np.eye(6) * 10.0

        self.last_update_time = timestamp
        self.is_initialized = True

        logger.debug(
            "Kalman filter initialized",
            position=position,
            velocity=velocity
        )

    def predict(self, dt: float) -> None:
        """
        Predict step: propagate state forward in time.

        Uses constant-velocity model:
        x_new = x + vx * dt
        vx_new = vx

        Args:
            dt: Time step in seconds
        """
        if not self.is_initialized:
            logger.warning("Kalman filter not initialized, skipping predict")
            return

        if dt <= 0:
            return

        # State transition matrix (constant velocity model)
        F = np.eye(6)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        # Scale process noise by dt
        Q_scaled = self.Q * dt
        self.P = F @ self.P @ F.T + Q_scaled

    def update_position(
        self,
        measured_position: Tuple[float, float, float],
        noise_scale: float = 1.0,
    ) -> None:
        """
        Update step: incorporate position measurement.

        Args:
            measured_position: Measured position (x, y, z) in NED
            noise_scale: Scale factor for measurement noise (>1 = less trust)
        """
        if not self.is_initialized:
            # Initialize with first measurement
            self.initialize(measured_position)
            return

        # Measurement vector
        z = np.array(measured_position)

        # Measurement matrix (we observe position directly)
        H = np.zeros((3, 6))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z

        # Scaled measurement noise
        R = self.R_gps * noise_scale

        # Innovation (measurement residual)
        y = z - H @ self.state

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def update_velocity(
        self,
        measured_velocity: Tuple[float, float, float],
        noise_scale: float = 1.0,
    ) -> None:
        """
        Update step: incorporate velocity measurement.

        Args:
            measured_velocity: Measured velocity (vx, vy, vz)
            noise_scale: Scale factor for measurement noise
        """
        if not self.is_initialized:
            return

        # Measurement vector
        z = np.array(measured_velocity)

        # Measurement matrix (we observe velocity)
        H = np.zeros((3, 6))
        H[0, 3] = 1  # vx
        H[1, 4] = 1  # vy
        H[2, 5] = 1  # vz

        # Velocity measurement noise (use similar to position)
        R = np.eye(3) * 0.3 * noise_scale

        # Innovation
        y = z - H @ self.state

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def get_position(self) -> Tuple[float, float, float]:
        """Get current estimated position."""
        return (self.state[0], self.state[1], self.state[2])

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current estimated velocity."""
        return (self.state[3], self.state[4], self.state[5])

    def get_position_uncertainty(self) -> Tuple[float, float, float]:
        """Get 1-sigma position uncertainty."""
        return (
            np.sqrt(self.P[0, 0]),
            np.sqrt(self.P[1, 1]),
            np.sqrt(self.P[2, 2])
        )

    def get_velocity_uncertainty(self) -> Tuple[float, float, float]:
        """Get 1-sigma velocity uncertainty."""
        return (
            np.sqrt(self.P[3, 3]),
            np.sqrt(self.P[4, 4]),
            np.sqrt(self.P[5, 5])
        )
