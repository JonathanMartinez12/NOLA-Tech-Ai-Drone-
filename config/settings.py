"""
Centralized settings and tunable parameters for the drone racing system.

All magic numbers and configuration values should be defined here,
making it easy to tune the system without modifying code logic.

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.control.max_velocity)
"""

from functools import lru_cache
from typing import Tuple
from pydantic import Field
from pydantic_settings import BaseSettings


class ConnectionSettings(BaseSettings):
    """Settings for drone connection."""

    # MAVSDK connection string for PX4 SITL
    system_address: str = Field(
        default="udp://:14540",
        description="MAVSDK connection address (UDP for SITL, serial for real)"
    )

    # Connection timeout in seconds
    connection_timeout: float = Field(
        default=30.0,
        description="Timeout for initial drone connection"
    )

    # Heartbeat timeout - how long to wait for heartbeat before assuming disconnected
    heartbeat_timeout: float = Field(
        default=5.0,
        description="Timeout for heartbeat before assuming disconnection"
    )


class ControlSettings(BaseSettings):
    """Settings for flight control."""

    # Velocity limits (m/s)
    max_velocity: float = Field(
        default=5.0,
        description="Maximum allowed velocity magnitude"
    )
    max_horizontal_velocity: float = Field(
        default=5.0,
        description="Maximum horizontal velocity (North/East)"
    )
    max_vertical_velocity: float = Field(
        default=2.0,
        description="Maximum vertical velocity (Up/Down)"
    )

    # Acceleration limits (m/s^2)
    max_acceleration: float = Field(
        default=3.0,
        description="Maximum acceleration for velocity ramping"
    )

    # Position control
    position_threshold: float = Field(
        default=0.5,
        description="Distance threshold to consider waypoint reached (meters)"
    )
    velocity_threshold: float = Field(
        default=0.3,
        description="Velocity threshold to consider drone stopped (m/s)"
    )

    # Yaw control
    max_yaw_rate: float = Field(
        default=60.0,
        description="Maximum yaw rate in degrees per second"
    )
    yaw_threshold: float = Field(
        default=5.0,
        description="Yaw error threshold to consider aligned (degrees)"
    )

    # Control loop timing
    control_loop_rate: float = Field(
        default=50.0,
        description="Control loop frequency in Hz"
    )
    offboard_setpoint_rate: float = Field(
        default=20.0,
        description="Rate to send offboard setpoints (Hz)"
    )

    # Safety
    min_altitude: float = Field(
        default=1.0,
        description="Minimum allowed altitude above ground (meters)"
    )
    max_altitude: float = Field(
        default=50.0,
        description="Maximum allowed altitude (meters)"
    )
    geofence_radius: float = Field(
        default=100.0,
        description="Maximum distance from home position (meters)"
    )


class TakeoffLandSettings(BaseSettings):
    """Settings for takeoff and landing operations."""

    default_takeoff_altitude: float = Field(
        default=5.0,
        description="Default altitude for takeoff (meters)"
    )
    takeoff_velocity: float = Field(
        default=1.0,
        description="Vertical velocity during takeoff (m/s)"
    )
    land_velocity: float = Field(
        default=0.5,
        description="Vertical velocity during landing (m/s)"
    )
    land_detection_altitude: float = Field(
        default=0.3,
        description="Altitude below which landing is considered complete"
    )
    hover_stabilization_time: float = Field(
        default=2.0,
        description="Time to hover and stabilize after takeoff (seconds)"
    )


class PerceptionSettings(BaseSettings):
    """Settings for perception and gate detection."""

    # Camera settings
    camera_fps: int = Field(
        default=30,
        description="Camera frame rate"
    )
    camera_resolution: Tuple[int, int] = Field(
        default=(640, 480),
        description="Camera resolution (width, height)"
    )

    # Gate detection - HSV color ranges for orange gates
    # These values work for typical orange racing gates in simulation
    gate_color_lower: Tuple[int, int, int] = Field(
        default=(5, 100, 100),
        description="Lower HSV bound for gate color detection"
    )
    gate_color_upper: Tuple[int, int, int] = Field(
        default=(25, 255, 255),
        description="Upper HSV bound for gate color detection"
    )

    # Gate detection thresholds
    min_gate_area: int = Field(
        default=1000,
        description="Minimum contour area to consider as gate (pixels)"
    )
    max_gate_area: int = Field(
        default=500000,
        description="Maximum contour area to consider as gate (pixels)"
    )
    gate_aspect_ratio_min: float = Field(
        default=0.5,
        description="Minimum aspect ratio for gate detection"
    )
    gate_aspect_ratio_max: float = Field(
        default=2.0,
        description="Maximum aspect ratio for gate detection"
    )

    # Detection confidence
    min_detection_confidence: float = Field(
        default=0.6,
        description="Minimum confidence to consider detection valid"
    )
    detection_history_size: int = Field(
        default=5,
        description="Number of frames to average for stable detection"
    )

    # Distance estimation
    gate_actual_width: float = Field(
        default=2.0,
        description="Actual gate width in meters (for distance estimation)"
    )
    camera_focal_length: float = Field(
        default=500.0,
        description="Camera focal length in pixels (calibration dependent)"
    )


class StateEstimationSettings(BaseSettings):
    """Settings for state estimation and sensor fusion."""

    # Kalman filter process noise
    position_process_noise: float = Field(
        default=0.1,
        description="Process noise for position estimation"
    )
    velocity_process_noise: float = Field(
        default=0.5,
        description="Process noise for velocity estimation"
    )

    # Measurement noise
    gps_position_noise: float = Field(
        default=0.5,
        description="GPS position measurement noise (meters)"
    )
    imu_acceleration_noise: float = Field(
        default=0.1,
        description="IMU acceleration measurement noise (m/s^2)"
    )
    visual_position_noise: float = Field(
        default=1.0,
        description="Visual position estimation noise (meters)"
    )

    # Update rates
    state_update_rate: float = Field(
        default=100.0,
        description="State estimation update rate (Hz)"
    )

    # Outlier rejection
    max_position_jump: float = Field(
        default=5.0,
        description="Maximum position change per update to accept (meters)"
    )
    max_velocity_jump: float = Field(
        default=10.0,
        description="Maximum velocity change per update to accept (m/s)"
    )


class PlanningSettings(BaseSettings):
    """Settings for path planning and trajectory generation."""

    # Waypoint following
    waypoint_acceptance_radius: float = Field(
        default=1.0,
        description="Radius to consider waypoint reached (meters)"
    )
    lookahead_distance: float = Field(
        default=3.0,
        description="Distance to look ahead for path following"
    )

    # Trajectory generation
    trajectory_time_step: float = Field(
        default=0.1,
        description="Time step for trajectory discretization (seconds)"
    )
    min_trajectory_duration: float = Field(
        default=1.0,
        description="Minimum trajectory segment duration (seconds)"
    )

    # Racing line
    corner_cutting_factor: float = Field(
        default=0.3,
        description="How aggressively to cut corners (0=none, 1=maximum)"
    )

    # Collision avoidance
    safety_margin: float = Field(
        default=1.5,
        description="Safety margin around obstacles (meters)"
    )
    collision_check_distance: float = Field(
        default=10.0,
        description="Distance to check ahead for collisions (meters)"
    )


class LoggingSettings(BaseSettings):
    """Settings for logging and telemetry."""

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_to_file: bool = Field(
        default=True,
        description="Whether to log to file"
    )
    log_directory: str = Field(
        default="logs",
        description="Directory for log files"
    )

    # Telemetry recording
    record_telemetry: bool = Field(
        default=True,
        description="Whether to record telemetry data"
    )
    telemetry_directory: str = Field(
        default="telemetry_data",
        description="Directory for telemetry recordings"
    )
    telemetry_rate: float = Field(
        default=10.0,
        description="Rate to record telemetry (Hz)"
    )


class Settings(BaseSettings):
    """
    Main settings class that aggregates all configuration sections.

    This provides a single point of access for all system configuration.
    """

    connection: ConnectionSettings = Field(default_factory=ConnectionSettings)
    control: ControlSettings = Field(default_factory=ControlSettings)
    takeoff_land: TakeoffLandSettings = Field(default_factory=TakeoffLandSettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    state_estimation: StateEstimationSettings = Field(default_factory=StateEstimationSettings)
    planning: PlanningSettings = Field(default_factory=PlanningSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    class Config:
        env_prefix = "DRONE_"  # Environment variables like DRONE_CONTROL_MAX_VELOCITY
        env_nested_delimiter = "__"


@lru_cache()
def get_settings() -> Settings:
    """
    Get the global settings instance (cached singleton).

    Returns:
        Settings: The global settings object with all configuration.

    Example:
        settings = get_settings()
        max_vel = settings.control.max_velocity
    """
    return Settings()
