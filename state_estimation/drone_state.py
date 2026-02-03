"""
Drone state tracking and telemetry subscription.

Provides real-time access to drone state by subscribing to MAVSDK
telemetry streams. Maintains the latest known state and provides
convenient access methods.

Usage:
    drone = System()
    await drone.connect(...)

    state = DroneState(drone)
    await state.start_tracking()

    # Access current state
    pos = state.position_ned
    vel = state.velocity_ned
    yaw = state.yaw_deg

    # Stop when done
    await state.stop_tracking()
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from mavsdk import System
from mavsdk.telemetry import FlightMode, LandedState

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DroneState:
    """
    Tracks and provides access to current drone state.

    Subscribes to MAVSDK telemetry streams and maintains the latest
    state information. All state values are thread-safe to read.

    Attributes:
        drone: MAVSDK System instance

    State Properties:
        position_ned: Current position (North, East, Down) in meters
        velocity_ned: Current velocity (North, East, Down) in m/s
        attitude_euler: Current attitude (roll, pitch, yaw) in degrees
        is_armed: Whether the drone is armed
        flight_mode: Current flight mode
        is_connected: Whether we have telemetry connection

    Example:
        state = DroneState(drone)
        await state.start_tracking()

        while True:
            print(f"Position: {state.position_ned}")
            print(f"Altitude: {state.altitude}")
            await asyncio.sleep(0.1)
    """

    drone: System

    # Position state (NED frame, relative to home)
    _position_n: float = field(default=0.0, repr=False)
    _position_e: float = field(default=0.0, repr=False)
    _position_d: float = field(default=0.0, repr=False)

    # Velocity state (NED frame)
    _velocity_n: float = field(default=0.0, repr=False)
    _velocity_e: float = field(default=0.0, repr=False)
    _velocity_d: float = field(default=0.0, repr=False)

    # Attitude state (degrees)
    _roll_deg: float = field(default=0.0, repr=False)
    _pitch_deg: float = field(default=0.0, repr=False)
    _yaw_deg: float = field(default=0.0, repr=False)

    # System state
    _is_armed: bool = field(default=False, repr=False)
    _flight_mode: FlightMode = field(default=FlightMode.UNKNOWN, repr=False)
    _landed_state: LandedState = field(default=LandedState.UNKNOWN, repr=False)

    # GPS state
    _latitude_deg: float = field(default=0.0, repr=False)
    _longitude_deg: float = field(default=0.0, repr=False)
    _absolute_altitude_m: float = field(default=0.0, repr=False)

    # Home position (for relative calculations)
    _home_position: Optional[Tuple[float, float, float]] = field(default=None, repr=False)

    # Battery
    _battery_voltage: float = field(default=0.0, repr=False)
    _battery_remaining: float = field(default=100.0, repr=False)

    # Tracking tasks
    _tracking_tasks: list = field(default_factory=list, repr=False)
    _is_tracking: bool = field(default=False, repr=False)

    # Connection state
    _last_update_time: float = field(default=0.0, repr=False)
    _connected: bool = field(default=False, repr=False)

    # ==========================================================================
    # Properties for accessing current state
    # ==========================================================================

    @property
    def position_ned(self) -> Tuple[float, float, float]:
        """Current position in NED frame (North, East, Down) meters."""
        return (self._position_n, self._position_e, self._position_d)

    @property
    def velocity_ned(self) -> Tuple[float, float, float]:
        """Current velocity in NED frame (North, East, Down) m/s."""
        return (self._velocity_n, self._velocity_e, self._velocity_d)

    @property
    def attitude_euler(self) -> Tuple[float, float, float]:
        """Current attitude (roll, pitch, yaw) in degrees."""
        return (self._roll_deg, self._pitch_deg, self._yaw_deg)

    @property
    def altitude(self) -> float:
        """Current altitude above home in meters (positive up)."""
        return -self._position_d

    @property
    def yaw_deg(self) -> float:
        """Current yaw heading in degrees (0=North, 90=East)."""
        return self._yaw_deg

    @property
    def yaw_rad(self) -> float:
        """Current yaw heading in radians."""
        return math.radians(self._yaw_deg)

    @property
    def speed(self) -> float:
        """Current speed magnitude in m/s."""
        return math.sqrt(
            self._velocity_n ** 2 +
            self._velocity_e ** 2 +
            self._velocity_d ** 2
        )

    @property
    def horizontal_speed(self) -> float:
        """Current horizontal speed magnitude in m/s."""
        return math.sqrt(self._velocity_n ** 2 + self._velocity_e ** 2)

    @property
    def vertical_speed(self) -> float:
        """Current vertical speed in m/s (positive = climbing)."""
        return -self._velocity_d

    @property
    def is_armed(self) -> bool:
        """Whether the drone is armed."""
        return self._is_armed

    @property
    def flight_mode(self) -> FlightMode:
        """Current flight mode."""
        return self._flight_mode

    @property
    def flight_mode_str(self) -> str:
        """Current flight mode as string."""
        return self._flight_mode.name

    @property
    def is_in_air(self) -> bool:
        """Whether the drone is in the air."""
        return self._landed_state == LandedState.IN_AIR

    @property
    def is_landed(self) -> bool:
        """Whether the drone is on the ground."""
        return self._landed_state in (LandedState.ON_GROUND, LandedState.UNKNOWN)

    @property
    def battery_voltage(self) -> float:
        """Current battery voltage."""
        return self._battery_voltage

    @property
    def battery_percent(self) -> float:
        """Battery remaining percentage (0-100)."""
        return self._battery_remaining

    @property
    def gps_position(self) -> Tuple[float, float, float]:
        """GPS position (latitude, longitude, altitude) in degrees/meters."""
        return (self._latitude_deg, self._longitude_deg, self._absolute_altitude_m)

    @property
    def is_connected(self) -> bool:
        """Whether we have active telemetry connection."""
        return self._connected

    # ==========================================================================
    # Telemetry subscription methods
    # ==========================================================================

    async def start_tracking(self) -> None:
        """
        Start subscribing to all telemetry streams.

        Creates async tasks for each telemetry type. Must be called
        before accessing state properties.
        """
        if self._is_tracking:
            logger.warning("State tracking already started")
            return

        logger.info("Starting drone state tracking...")
        self._is_tracking = True

        # Start all telemetry subscription tasks
        self._tracking_tasks = [
            asyncio.create_task(self._track_position_velocity()),
            asyncio.create_task(self._track_attitude()),
            asyncio.create_task(self._track_armed()),
            asyncio.create_task(self._track_flight_mode()),
            asyncio.create_task(self._track_landed_state()),
            asyncio.create_task(self._track_gps()),
            asyncio.create_task(self._track_battery()),
        ]

        # Wait briefly for initial data
        await asyncio.sleep(0.5)
        logger.info(
            "State tracking started",
            position=self.position_ned,
            armed=self.is_armed,
            mode=self.flight_mode_str
        )

    async def stop_tracking(self) -> None:
        """Stop all telemetry subscriptions."""
        if not self._is_tracking:
            return

        logger.info("Stopping state tracking...")
        self._is_tracking = False

        # Cancel all tracking tasks
        for task in self._tracking_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tracking_tasks, return_exceptions=True)
        self._tracking_tasks = []
        logger.info("State tracking stopped")

    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait until drone telemetry is available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected, False if timeout
        """
        logger.info("Waiting for drone telemetry connection...")
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if self._connected:
                logger.info("Telemetry connection established")
                return True
            await asyncio.sleep(0.1)

        logger.error("Timeout waiting for telemetry connection", timeout=timeout)
        return False

    async def set_home_position(self) -> None:
        """Set current position as home (origin for NED calculations)."""
        self._home_position = (self._latitude_deg, self._longitude_deg, self._absolute_altitude_m)
        logger.info(
            "Home position set",
            lat=self._latitude_deg,
            lon=self._longitude_deg,
            alt=self._absolute_altitude_m
        )

    # ==========================================================================
    # Private telemetry tracking methods
    # ==========================================================================

    async def _track_position_velocity(self) -> None:
        """Subscribe to position and velocity telemetry."""
        try:
            async for pos_vel in self.drone.telemetry.position_velocity_ned():
                self._position_n = pos_vel.position.north_m
                self._position_e = pos_vel.position.east_m
                self._position_d = pos_vel.position.down_m

                self._velocity_n = pos_vel.velocity.north_m_s
                self._velocity_e = pos_vel.velocity.east_m_s
                self._velocity_d = pos_vel.velocity.down_m_s

                self._connected = True
                self._last_update_time = asyncio.get_event_loop().time()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Position/velocity tracking error", error=str(e))

    async def _track_attitude(self) -> None:
        """Subscribe to attitude telemetry."""
        try:
            async for attitude in self.drone.telemetry.attitude_euler():
                self._roll_deg = attitude.roll_deg
                self._pitch_deg = attitude.pitch_deg
                self._yaw_deg = attitude.yaw_deg

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Attitude tracking error", error=str(e))

    async def _track_armed(self) -> None:
        """Subscribe to armed state."""
        try:
            async for armed in self.drone.telemetry.armed():
                self._is_armed = armed

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Armed state tracking error", error=str(e))

    async def _track_flight_mode(self) -> None:
        """Subscribe to flight mode."""
        try:
            async for mode in self.drone.telemetry.flight_mode():
                if mode != self._flight_mode:
                    logger.debug("Flight mode changed", old=self._flight_mode.name, new=mode.name)
                self._flight_mode = mode

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Flight mode tracking error", error=str(e))

    async def _track_landed_state(self) -> None:
        """Subscribe to landed state."""
        try:
            async for landed in self.drone.telemetry.landed_state():
                self._landed_state = landed

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Landed state tracking error", error=str(e))

    async def _track_gps(self) -> None:
        """Subscribe to GPS position."""
        try:
            async for position in self.drone.telemetry.position():
                self._latitude_deg = position.latitude_deg
                self._longitude_deg = position.longitude_deg
                self._absolute_altitude_m = position.absolute_altitude_m

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("GPS tracking error", error=str(e))

    async def _track_battery(self) -> None:
        """Subscribe to battery state."""
        try:
            async for battery in self.drone.telemetry.battery():
                self._battery_voltage = battery.voltage_v
                self._battery_remaining = battery.remaining_percent * 100

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Battery tracking error", error=str(e))

    # ==========================================================================
    # Utility methods
    # ==========================================================================

    def get_state_dict(self) -> dict:
        """
        Get current state as a dictionary.

        Useful for logging and debugging.

        Returns:
            Dictionary with all current state values
        """
        return {
            "position_ned": self.position_ned,
            "velocity_ned": self.velocity_ned,
            "attitude": self.attitude_euler,
            "altitude": self.altitude,
            "speed": self.speed,
            "is_armed": self.is_armed,
            "flight_mode": self.flight_mode_str,
            "is_in_air": self.is_in_air,
            "battery_percent": self.battery_percent,
        }

    def __repr__(self) -> str:
        return (
            f"DroneState("
            f"pos=({self._position_n:.1f}, {self._position_e:.1f}, {self._position_d:.1f}), "
            f"alt={self.altitude:.1f}m, "
            f"speed={self.speed:.1f}m/s, "
            f"armed={self._is_armed}, "
            f"mode={self._flight_mode.name})"
        )
