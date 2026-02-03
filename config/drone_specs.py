"""
Physical drone specifications and performance limits.

These values represent the actual physical capabilities and constraints
of the Neros.tech drone used in the AI Grand Prix competition.

Note: Some values are estimates based on typical racing drone specs.
Update these when official specifications become available.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


@dataclass(frozen=True)
class DroneSpecs:
    """
    Physical specifications of the racing drone.

    All values are based on the Neros.tech competition drone.
    These are hardware limits that should never be exceeded.

    Attributes:
        name: Drone model identifier
        mass_kg: Total mass including battery
        max_thrust_n: Maximum thrust from all motors combined
        thrust_to_weight: Thrust-to-weight ratio (>1 means can hover)

    Note:
        Values marked with (estimated) should be updated when
        official specifications become available.
    """

    # Identification
    name: str = "Neros.tech Racing Drone"

    # Physical properties
    mass_kg: float = 1.5  # (estimated) Total mass with battery
    arm_length_m: float = 0.25  # (estimated) Motor-to-center distance

    # Propulsion limits
    max_thrust_n: float = 40.0  # (estimated) Maximum total thrust
    thrust_to_weight: float = 2.7  # (estimated) Thrust-to-weight ratio

    # Velocity limits (physical maximums)
    max_horizontal_speed_ms: float = 20.0  # (estimated) Max forward/lateral speed
    max_vertical_speed_ms: float = 10.0  # (estimated) Max climb/descent rate
    max_speed_ms: float = 25.0  # (estimated) Maximum speed magnitude

    # Acceleration limits (physical maximums)
    max_horizontal_accel_ms2: float = 10.0  # (estimated) Max horizontal acceleration
    max_vertical_accel_ms2: float = 15.0  # (estimated) Max vertical acceleration (up)
    max_decel_ms2: float = 12.0  # (estimated) Max braking deceleration

    # Attitude limits
    max_roll_deg: float = 60.0  # (estimated) Maximum roll angle
    max_pitch_deg: float = 60.0  # (estimated) Maximum pitch angle
    max_yaw_rate_dps: float = 200.0  # (estimated) Maximum yaw rate degrees/sec

    # Attitude rate limits
    max_roll_rate_dps: float = 300.0  # (estimated) Roll rate limit
    max_pitch_rate_dps: float = 300.0  # (estimated) Pitch rate limit

    # Battery
    battery_capacity_mah: int = 1500  # (estimated) Battery capacity
    battery_voltage_nominal: float = 14.8  # (estimated) 4S LiPo nominal
    flight_time_minutes: float = 8.0  # (estimated) Typical flight time

    # Sensors
    has_gps: bool = True
    has_optical_flow: bool = False  # May be added
    has_lidar: bool = False  # May be added
    has_depth_camera: bool = False  # May be added

    # Camera specifications (for perception)
    camera_fov_horizontal_deg: float = 90.0  # (estimated) Horizontal FOV
    camera_fov_vertical_deg: float = 70.0  # (estimated) Vertical FOV
    camera_resolution: Tuple[int, int] = (1280, 720)  # (estimated) Resolution
    camera_fps: int = 60  # (estimated) Frame rate

    # Physical dimensions (for collision checking)
    width_m: float = 0.5  # (estimated) Tip-to-tip width
    length_m: float = 0.5  # (estimated) Front-to-back length
    height_m: float = 0.15  # (estimated) Total height
    collision_radius_m: float = 0.35  # (estimated) Simplified collision sphere

    def can_achieve_velocity(self, velocity_ms: float) -> bool:
        """Check if a given velocity is physically achievable."""
        return abs(velocity_ms) <= self.max_speed_ms

    def can_achieve_acceleration(self, accel_ms2: float) -> bool:
        """Check if a given acceleration is physically achievable."""
        return abs(accel_ms2) <= self.max_horizontal_accel_ms2

    def hover_thrust_ratio(self) -> float:
        """
        Calculate the thrust ratio needed to hover.

        Returns:
            Float between 0-1 representing throttle percentage for hover.
        """
        return 1.0 / self.thrust_to_weight

    def max_climb_rate(self) -> float:
        """Calculate maximum sustainable climb rate based on thrust."""
        # Simplified: excess thrust beyond hover enables climb
        excess_thrust_ratio = 1.0 - self.hover_thrust_ratio()
        return excess_thrust_ratio * self.max_vertical_speed_ms

    def stopping_distance(self, current_velocity_ms: float) -> float:
        """
        Calculate minimum stopping distance from current velocity.

        Args:
            current_velocity_ms: Current velocity magnitude in m/s

        Returns:
            Minimum distance needed to stop (meters)
        """
        # Using kinematic equation: d = v^2 / (2*a)
        return (current_velocity_ms ** 2) / (2 * self.max_decel_ms2)


@lru_cache()
def get_drone_specs() -> DroneSpecs:
    """
    Get the drone specifications instance (cached singleton).

    Returns:
        DroneSpecs: The drone physical specifications.

    Example:
        specs = get_drone_specs()
        max_speed = specs.max_horizontal_speed_ms
    """
    return DroneSpecs()


# Convenience constants for commonly used values
# These can be imported directly: from config.drone_specs import MAX_SPEED
MAX_SPEED = get_drone_specs().max_speed_ms
MAX_HORIZONTAL_SPEED = get_drone_specs().max_horizontal_speed_ms
MAX_VERTICAL_SPEED = get_drone_specs().max_vertical_speed_ms
COLLISION_RADIUS = get_drone_specs().collision_radius_m
