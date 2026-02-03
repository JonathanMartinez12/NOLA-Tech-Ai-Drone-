"""
Low-level offboard controller for direct velocity and attitude control.

Provides direct interface to MAVSDK offboard mode with velocity and
position setpoint generation. Used by higher-level controllers and
for advanced maneuvers.

Usage:
    offboard = OffboardController(drone, state)
    await offboard.start()

    # Send velocity commands
    await offboard.set_velocity(2.0, 0.0, 0.0, yaw=90.0)

    # Or use position control
    await offboard.set_position(10.0, 5.0, -5.0)

    await offboard.stop()
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from mavsdk import System
from mavsdk.offboard import (
    OffboardError,
    VelocityNedYaw,
    PositionNedYaw,
    VelocityBodyYawspeed,
)

from config.settings import get_settings
from state_estimation.drone_state import DroneState
from utils.logger import get_logger
from utils.math_helpers import clamp, clamp_vector, vector_magnitude
import numpy as np

logger = get_logger(__name__)


@dataclass
class OffboardController:
    """
    Low-level offboard controller for direct setpoint control.

    Provides velocity and position control interfaces with built-in
    safety limits and smooth transitions.

    Attributes:
        drone: MAVSDK System instance
        state: DroneState for telemetry access
    """

    drone: System
    state: DroneState

    is_active: bool = field(default=False, init=False)
    _setpoint_task: Optional[asyncio.Task] = field(default=None, init=False)

    # Current setpoint (for keepalive)
    _current_velocity: Tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0), init=False
    )
    _current_yaw: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.settings = get_settings()

    # ==========================================================================
    # Lifecycle Management
    # ==========================================================================

    async def start(self) -> bool:
        """
        Start offboard mode.

        Must be called before sending any setpoints. Starts a background
        task to maintain offboard mode with keepalive setpoints.

        Returns:
            True if offboard mode started successfully
        """
        if self.is_active:
            logger.debug("Offboard already active")
            return True

        try:
            # Initialize with zero velocity at current yaw
            self._current_velocity = (0.0, 0.0, 0.0)
            self._current_yaw = self.state.yaw_deg

            # Send initial setpoint (required before starting)
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, self._current_yaw)
            )

            # Start offboard mode
            await self.drone.offboard.start()
            self.is_active = True

            # Start keepalive task
            self._setpoint_task = asyncio.create_task(self._keepalive_loop())

            logger.info("Offboard controller started")
            return True

        except OffboardError as e:
            logger.error("Failed to start offboard", error=str(e))
            return False

    async def stop(self) -> bool:
        """
        Stop offboard mode and return to hold.

        Returns:
            True if stopped successfully
        """
        if not self.is_active:
            return True

        try:
            # Cancel keepalive task
            if self._setpoint_task:
                self._setpoint_task.cancel()
                try:
                    await self._setpoint_task
                except asyncio.CancelledError:
                    pass
                self._setpoint_task = None

            # Stop offboard
            await self.drone.offboard.stop()
            self.is_active = False

            logger.info("Offboard controller stopped")
            return True

        except OffboardError as e:
            logger.error("Failed to stop offboard", error=str(e))
            return False

    async def _keepalive_loop(self) -> None:
        """
        Background task to maintain offboard mode.

        PX4 requires continuous setpoint updates or it will exit offboard mode.
        """
        rate = self.settings.control.offboard_setpoint_rate
        interval = 1.0 / rate

        try:
            while self.is_active:
                await self.drone.offboard.set_velocity_ned(
                    VelocityNedYaw(
                        self._current_velocity[0],
                        self._current_velocity[1],
                        self._current_velocity[2],
                        self._current_yaw,
                    )
                )
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Keepalive loop error", error=str(e))
            self.is_active = False

    # ==========================================================================
    # Velocity Control
    # ==========================================================================

    async def set_velocity_ned(
        self,
        vel_north: float,
        vel_east: float,
        vel_down: float,
        yaw_deg: Optional[float] = None,
    ) -> None:
        """
        Set velocity setpoint in NED frame.

        Args:
            vel_north: North velocity (m/s)
            vel_east: East velocity (m/s)
            vel_down: Down velocity (m/s, positive = descend)
            yaw_deg: Target yaw in degrees (None = maintain current)
        """
        if not self.is_active:
            logger.warning("Offboard not active, ignoring velocity command")
            return

        # Apply velocity limits
        vel_north = clamp(
            vel_north,
            -self.settings.control.max_horizontal_velocity,
            self.settings.control.max_horizontal_velocity,
        )
        vel_east = clamp(
            vel_east,
            -self.settings.control.max_horizontal_velocity,
            self.settings.control.max_horizontal_velocity,
        )
        vel_down = clamp(
            vel_down,
            -self.settings.control.max_vertical_velocity,
            self.settings.control.max_vertical_velocity,
        )

        # Update stored setpoint
        self._current_velocity = (vel_north, vel_east, vel_down)
        if yaw_deg is not None:
            self._current_yaw = yaw_deg

        # Send immediately (don't wait for keepalive)
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(vel_north, vel_east, vel_down, self._current_yaw)
        )

    async def set_velocity_body(
        self,
        vel_forward: float,
        vel_right: float,
        vel_down: float,
        yaw_rate: float = 0.0,
    ) -> None:
        """
        Set velocity setpoint in body frame.

        Useful for maneuvers relative to drone's current heading.

        Args:
            vel_forward: Forward velocity (m/s)
            vel_right: Right velocity (m/s)
            vel_down: Down velocity (m/s)
            yaw_rate: Yaw rate in degrees/second
        """
        if not self.is_active:
            logger.warning("Offboard not active, ignoring velocity command")
            return

        # Apply limits
        vel_forward = clamp(
            vel_forward,
            -self.settings.control.max_horizontal_velocity,
            self.settings.control.max_horizontal_velocity,
        )
        vel_right = clamp(
            vel_right,
            -self.settings.control.max_horizontal_velocity,
            self.settings.control.max_horizontal_velocity,
        )
        vel_down = clamp(
            vel_down,
            -self.settings.control.max_vertical_velocity,
            self.settings.control.max_vertical_velocity,
        )
        yaw_rate = clamp(
            yaw_rate,
            -self.settings.control.max_yaw_rate,
            self.settings.control.max_yaw_rate,
        )

        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vel_forward, vel_right, vel_down, yaw_rate)
        )

    async def set_velocity_toward(
        self,
        target: Tuple[float, float, float],
        speed: float,
        face_target: bool = True,
    ) -> None:
        """
        Set velocity toward a target position.

        Calculates direction to target and sets velocity along that vector.

        Args:
            target: Target position (N, E, D)
            speed: Desired speed magnitude (m/s)
            face_target: Whether to yaw toward target
        """
        pos = self.state.position_ned
        error = np.array([
            target[0] - pos[0],
            target[1] - pos[1],
            target[2] - pos[2],
        ])

        distance = vector_magnitude(error)

        if distance < 0.1:
            # Already at target
            await self.set_velocity_ned(0, 0, 0)
            return

        # Normalize and scale by speed
        direction = error / distance
        velocity = direction * speed

        # Clamp velocity
        velocity = clamp_vector(velocity, self.settings.control.max_velocity)

        # Calculate yaw to face target
        yaw = None
        if face_target and (abs(error[0]) > 0.1 or abs(error[1]) > 0.1):
            yaw = math.degrees(math.atan2(error[1], error[0]))

        await self.set_velocity_ned(velocity[0], velocity[1], velocity[2], yaw)

    # ==========================================================================
    # Position Control
    # ==========================================================================

    async def set_position_ned(
        self,
        pos_north: float,
        pos_east: float,
        pos_down: float,
        yaw_deg: Optional[float] = None,
    ) -> None:
        """
        Set position setpoint in NED frame.

        PX4 will handle the trajectory to reach this position.

        Args:
            pos_north: North position (m)
            pos_east: East position (m)
            pos_down: Down position (m, negative = up)
            yaw_deg: Target yaw in degrees
        """
        if not self.is_active:
            logger.warning("Offboard not active, ignoring position command")
            return

        yaw = yaw_deg if yaw_deg is not None else self.state.yaw_deg

        await self.drone.offboard.set_position_ned(
            PositionNedYaw(pos_north, pos_east, pos_down, yaw)
        )

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    async def hold(self) -> None:
        """Hold current position (zero velocity)."""
        await self.set_velocity_ned(0, 0, 0)

    async def stop_movement(self) -> None:
        """Immediately stop all movement."""
        self._current_velocity = (0.0, 0.0, 0.0)
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(0, 0, 0, self.state.yaw_deg)
        )

    def get_current_setpoint(self) -> Tuple[Tuple[float, float, float], float]:
        """
        Get current velocity setpoint.

        Returns:
            Tuple of (velocity_ned, yaw_deg)
        """
        return (self._current_velocity, self._current_yaw)
