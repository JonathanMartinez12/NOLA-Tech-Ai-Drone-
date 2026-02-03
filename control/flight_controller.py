"""
High-level flight controller for drone racing.

Provides intuitive commands like takeoff, land, and goto_waypoint.
Handles the complexity of MAVSDK interactions and provides smooth,
safe flight behavior.

Usage:
    controller = FlightController(drone, state)

    await controller.arm()
    await controller.takeoff(altitude=5.0)
    await controller.goto((10, 0, -5), speed=2.0)
    await controller.land()
"""

import asyncio
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, VelocityNedYaw

from config.settings import get_settings
from state_estimation.drone_state import DroneState
from utils.logger import get_logger
from utils.math_helpers import (
    distance_3d,
    clamp,
    normalize_angle_deg,
    angle_difference_deg,
    calculate_bearing_deg,
)

logger = get_logger(__name__)


@dataclass
class FlightController:
    """
    High-level flight controller providing intuitive flight commands.

    Wraps MAVSDK actions and offboard control into easy-to-use methods.
    All methods are async and include proper error handling and safety checks.

    Attributes:
        drone: MAVSDK System instance
        state: DroneState for accessing current telemetry

    Example:
        controller = FlightController(drone, state)
        await controller.arm()
        await controller.takeoff(5.0)
        await controller.goto((10, 5, -5), speed=2.0)
        await controller.land()
    """

    drone: System
    state: DroneState
    _offboard_active: bool = False

    def __post_init__(self):
        self.settings = get_settings()

    # ==========================================================================
    # Basic Flight Commands
    # ==========================================================================

    async def arm(self) -> bool:
        """
        Arm the drone motors.

        Returns:
            True if successfully armed, False otherwise

        Raises:
            ActionError: If arming fails due to safety checks
        """
        if self.state.is_armed:
            logger.info("Drone already armed")
            return True

        logger.info("Arming drone...")
        try:
            await self.drone.action.arm()
            # Wait for armed state to propagate
            await asyncio.sleep(0.5)

            if self.state.is_armed:
                logger.info("Drone armed successfully")
                return True
            else:
                logger.warning("Arm command sent but drone not reporting armed")
                return False

        except ActionError as e:
            logger.error("Failed to arm", error=str(e))
            return False

    async def disarm(self) -> bool:
        """
        Disarm the drone motors.

        Only works if the drone is landed.

        Returns:
            True if successfully disarmed
        """
        if not self.state.is_armed:
            logger.info("Drone already disarmed")
            return True

        if self.state.is_in_air:
            logger.error("Cannot disarm while in air!")
            return False

        logger.info("Disarming drone...")
        try:
            await self.drone.action.disarm()
            await asyncio.sleep(0.5)
            logger.info("Drone disarmed")
            return True

        except ActionError as e:
            logger.error("Failed to disarm", error=str(e))
            return False

    async def takeoff(self, altitude: Optional[float] = None) -> bool:
        """
        Take off to specified altitude.

        Uses smooth velocity-based ascent for controlled takeoff.

        Args:
            altitude: Target altitude in meters (positive up).
                     Uses default from settings if not specified.

        Returns:
            True if takeoff successful
        """
        target_alt = altitude or self.settings.takeoff_land.default_takeoff_altitude

        logger.info("Starting takeoff", target_altitude=target_alt)

        # Ensure armed
        if not self.state.is_armed:
            if not await self.arm():
                return False

        # Start offboard mode for controlled ascent
        if not await self._start_offboard():
            # Fallback to action takeoff
            logger.info("Using action takeoff as fallback")
            try:
                await self.drone.action.set_takeoff_altitude(target_alt)
                await self.drone.action.takeoff()
                await self._wait_for_altitude(target_alt, timeout=30.0)
                return True
            except ActionError as e:
                logger.error("Takeoff failed", error=str(e))
                return False

        # Velocity-based ascent
        ascent_velocity = self.settings.takeoff_land.takeoff_velocity

        logger.info("Ascending", velocity=ascent_velocity)

        while self.state.altitude < target_alt - 0.3:
            # Calculate remaining distance
            remaining = target_alt - self.state.altitude

            # Reduce velocity near target for smooth stop
            if remaining < 1.0:
                vel = ascent_velocity * (remaining / 1.0)
                vel = max(vel, 0.2)  # Minimum velocity
            else:
                vel = ascent_velocity

            # Send velocity command (negative D = up)
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, -vel, self.state.yaw_deg)
            )
            await asyncio.sleep(0.05)

        # Hold position briefly to stabilize
        logger.info("Stabilizing at altitude", altitude=self.state.altitude)
        await self._hold_position(self.settings.takeoff_land.hover_stabilization_time)

        logger.info("Takeoff complete", final_altitude=self.state.altitude)
        return True

    async def land(self) -> bool:
        """
        Land the drone safely.

        Uses controlled descent followed by disarm.

        Returns:
            True if landing successful
        """
        logger.info("Starting landing sequence", current_altitude=self.state.altitude)

        if not self.state.is_in_air:
            logger.info("Drone already on ground")
            return True

        # Use offboard for controlled descent
        if self._offboard_active:
            descent_velocity = self.settings.takeoff_land.land_velocity
            land_threshold = self.settings.takeoff_land.land_detection_altitude

            logger.info("Controlled descent", velocity=descent_velocity)

            while self.state.altitude > land_threshold:
                # Reduce velocity near ground
                if self.state.altitude < 1.0:
                    vel = descent_velocity * 0.5
                else:
                    vel = descent_velocity

                await self.drone.offboard.set_velocity_ned(
                    VelocityNedYaw(0.0, 0.0, vel, self.state.yaw_deg)  # Positive D = down
                )
                await asyncio.sleep(0.05)

            # Stop offboard and use action land for final touchdown
            await self._stop_offboard()

        # Use action land for final touchdown
        logger.info("Final touchdown")
        try:
            await self.drone.action.land()

            # Wait for landing to complete
            timeout = 30.0
            start_time = asyncio.get_event_loop().time()
            while self.state.is_in_air:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    logger.warning("Landing timeout, forcing disarm")
                    break
                await asyncio.sleep(0.1)

            await asyncio.sleep(1.0)  # Let it settle
            await self.disarm()

            logger.info("Landing complete")
            return True

        except ActionError as e:
            logger.error("Landing failed", error=str(e))
            return False

    # ==========================================================================
    # Navigation Commands
    # ==========================================================================

    async def goto(
        self,
        target: Tuple[float, float, float],
        speed: float = 2.0,
        yaw: Optional[float] = None,
        acceptance_radius: Optional[float] = None,
    ) -> bool:
        """
        Fly to a target position using smooth velocity control.

        Uses proportional control with velocity ramping for smooth motion.

        Args:
            target: Target position (North, East, Down) in NED frame
            speed: Desired flight speed in m/s
            yaw: Target yaw in degrees (None = point toward target)
            acceptance_radius: Distance to consider arrived (meters)

        Returns:
            True if waypoint reached successfully
        """
        target_n, target_e, target_d = target
        radius = acceptance_radius or self.settings.control.position_threshold

        logger.info(
            "Flying to waypoint",
            target=target,
            speed=speed,
            current_pos=self.state.position_ned
        )

        # Ensure offboard mode is active
        if not self._offboard_active:
            if not await self._start_offboard():
                logger.error("Failed to start offboard mode")
                return False

        # Main control loop
        while True:
            # Get current state
            curr_n, curr_e, curr_d = self.state.position_ned

            # Calculate error vector
            error_n = target_n - curr_n
            error_e = target_e - curr_e
            error_d = target_d - curr_d

            # Calculate distance to target
            distance = math.sqrt(error_n**2 + error_e**2 + error_d**2)

            # Check if arrived
            if distance < radius:
                logger.info("Waypoint reached", distance=distance)
                break

            # Calculate direction (unit vector)
            if distance > 0.01:
                dir_n = error_n / distance
                dir_e = error_e / distance
                dir_d = error_d / distance
            else:
                dir_n = dir_e = dir_d = 0.0

            # Calculate velocity magnitude with smooth ramping
            # Slow down as we approach the target
            if distance < 2.0:
                # Proportional slowdown near target
                vel_mag = speed * (distance / 2.0)
                vel_mag = max(vel_mag, 0.3)  # Minimum velocity
            else:
                vel_mag = speed

            # Clamp to max velocity
            vel_mag = min(vel_mag, self.settings.control.max_velocity)

            # Calculate velocity components
            vel_n = dir_n * vel_mag
            vel_e = dir_e * vel_mag
            vel_d = dir_d * vel_mag

            # Clamp vertical velocity
            vel_d = clamp(
                vel_d,
                -self.settings.control.max_vertical_velocity,
                self.settings.control.max_vertical_velocity
            )

            # Calculate yaw (face direction of travel or specified yaw)
            if yaw is not None:
                target_yaw = yaw
            else:
                # Face direction of horizontal travel
                if abs(error_n) > 0.1 or abs(error_e) > 0.1:
                    target_yaw = calculate_bearing_deg((0, 0), (error_n, error_e))
                else:
                    target_yaw = self.state.yaw_deg

            # Send velocity command
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(vel_n, vel_e, vel_d, target_yaw)
            )

            await asyncio.sleep(0.02)  # 50 Hz control loop

        return True

    async def goto_sequence(
        self,
        waypoints: List[Tuple[float, float, float]],
        speed: float = 2.0,
        acceptance_radius: Optional[float] = None,
    ) -> bool:
        """
        Fly through a sequence of waypoints.

        Args:
            waypoints: List of (N, E, D) positions
            speed: Flight speed in m/s
            acceptance_radius: Distance to consider waypoint reached

        Returns:
            True if all waypoints reached
        """
        logger.info("Starting waypoint sequence", num_waypoints=len(waypoints))

        for i, waypoint in enumerate(waypoints):
            logger.info(f"Flying to waypoint {i+1}/{len(waypoints)}", target=waypoint)

            if not await self.goto(waypoint, speed=speed, acceptance_radius=acceptance_radius):
                logger.error(f"Failed to reach waypoint {i+1}")
                return False

            # Brief pause at waypoint
            await self._hold_position(0.5)

        logger.info("Waypoint sequence complete")
        return True

    async def return_to_home(self, speed: float = 2.0) -> bool:
        """
        Return to home position (0, 0) at current altitude.

        Returns:
            True if returned home successfully
        """
        current_alt = self.state.position_ned[2]  # Keep current altitude
        logger.info("Returning to home", current_altitude=-current_alt)

        return await self.goto((0.0, 0.0, current_alt), speed=speed)

    # ==========================================================================
    # Offboard Mode Management
    # ==========================================================================

    async def _start_offboard(self) -> bool:
        """
        Start offboard mode for direct velocity control.

        Offboard mode allows us to send velocity/position setpoints directly.
        Must send setpoints before starting, and continue sending while active.

        Returns:
            True if offboard mode started successfully
        """
        if self._offboard_active:
            return True

        logger.debug("Starting offboard mode...")

        try:
            # Must send setpoint before starting offboard
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, self.state.yaw_deg)
            )

            await self.drone.offboard.start()
            self._offboard_active = True

            logger.info("Offboard mode started")
            return True

        except OffboardError as e:
            logger.error("Failed to start offboard mode", error=str(e))
            return False

    async def _stop_offboard(self) -> bool:
        """
        Stop offboard mode and return to hold.

        Returns:
            True if stopped successfully
        """
        if not self._offboard_active:
            return True

        logger.debug("Stopping offboard mode...")

        try:
            await self.drone.offboard.stop()
            self._offboard_active = False
            logger.info("Offboard mode stopped")
            return True

        except OffboardError as e:
            logger.error("Failed to stop offboard mode", error=str(e))
            return False

    async def _hold_position(self, duration: float) -> None:
        """
        Hold current position for specified duration.

        Args:
            duration: Time to hold in seconds
        """
        if not self._offboard_active:
            await asyncio.sleep(duration)
            return

        end_time = asyncio.get_event_loop().time() + duration

        while asyncio.get_event_loop().time() < end_time:
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, self.state.yaw_deg)
            )
            await asyncio.sleep(0.05)

    async def _wait_for_altitude(
        self,
        target_altitude: float,
        timeout: float = 30.0,
        tolerance: float = 0.5,
    ) -> bool:
        """
        Wait until drone reaches target altitude.

        Args:
            target_altitude: Target altitude in meters
            timeout: Maximum wait time
            tolerance: Acceptable altitude error

        Returns:
            True if altitude reached within timeout
        """
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if abs(self.state.altitude - target_altitude) < tolerance:
                return True
            await asyncio.sleep(0.1)

        logger.warning(
            "Altitude timeout",
            target=target_altitude,
            current=self.state.altitude
        )
        return False

    # ==========================================================================
    # Safety Methods
    # ==========================================================================

    async def emergency_stop(self) -> None:
        """
        Emergency stop - immediately hold position.

        Use only in emergency situations.
        """
        logger.warning("EMERGENCY STOP TRIGGERED")

        if self._offboard_active:
            # Zero velocity to stop
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, self.state.yaw_deg)
            )

    async def check_geofence(self) -> bool:
        """
        Check if drone is within allowed flight area.

        Returns:
            True if within geofence, False if outside
        """
        pos = self.state.position_ned
        distance_from_home = math.sqrt(pos[0]**2 + pos[1]**2)

        max_radius = self.settings.control.geofence_radius
        max_alt = self.settings.control.max_altitude
        min_alt = self.settings.control.min_altitude

        if distance_from_home > max_radius:
            logger.warning("Geofence breach - distance", distance=distance_from_home)
            return False

        if self.state.altitude > max_alt:
            logger.warning("Geofence breach - too high", altitude=self.state.altitude)
            return False

        if self.state.altitude < min_alt and self.state.is_in_air:
            logger.warning("Geofence breach - too low", altitude=self.state.altitude)
            return False

        return True
