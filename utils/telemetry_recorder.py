"""
Telemetry recording for flight data analysis.

Records drone state and control inputs during flight for post-flight
analysis, debugging, and performance optimization.

Usage:
    recorder = TelemetryRecorder()
    await recorder.start()

    # During flight loop
    recorder.record(timestamp, position, velocity, attitude, command)

    await recorder.stop()
    recorder.save("flight_001.csv")
"""

import asyncio
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TelemetryFrame:
    """Single frame of telemetry data."""

    timestamp: float  # Seconds since recording start

    # Position (NED frame, meters)
    position_n: float
    position_e: float
    position_d: float

    # Velocity (NED frame, m/s)
    velocity_n: float
    velocity_e: float
    velocity_d: float

    # Attitude (radians)
    roll: float
    pitch: float
    yaw: float

    # Control commands (what we commanded)
    cmd_velocity_n: float = 0.0
    cmd_velocity_e: float = 0.0
    cmd_velocity_d: float = 0.0
    cmd_yaw: float = 0.0

    # Additional state
    battery_voltage: float = 0.0
    armed: bool = False
    flight_mode: str = ""


@dataclass
class TelemetryRecorder:
    """
    Records telemetry data during flight.

    Stores data in memory during flight, then saves to CSV for analysis.
    Designed to be low-overhead for real-time recording.

    Attributes:
        frames: List of recorded telemetry frames
        is_recording: Whether recording is active
        start_time: Timestamp when recording started
    """

    frames: List[TelemetryFrame] = field(default_factory=list)
    is_recording: bool = False
    start_time: Optional[float] = None
    _record_interval: float = field(default=0.1)  # 10 Hz default
    _last_record_time: float = field(default=0.0)

    def __post_init__(self):
        settings = get_settings()
        self._record_interval = 1.0 / settings.logging.telemetry_rate

    def start(self) -> None:
        """Start recording telemetry."""
        self.frames = []
        self.is_recording = True
        self.start_time = asyncio.get_event_loop().time()
        self._last_record_time = 0.0
        logger.info("Telemetry recording started")

    def stop(self) -> None:
        """Stop recording telemetry."""
        self.is_recording = False
        duration = len(self.frames) * self._record_interval
        logger.info(
            "Telemetry recording stopped",
            frames=len(self.frames),
            duration_seconds=round(duration, 2)
        )

    def record(
        self,
        position: tuple,
        velocity: tuple,
        attitude: tuple,
        command: Optional[tuple] = None,
        battery_voltage: float = 0.0,
        armed: bool = False,
        flight_mode: str = "",
    ) -> bool:
        """
        Record a telemetry frame if enough time has passed.

        Uses rate limiting to avoid recording too frequently.

        Args:
            position: (N, E, D) position in meters
            velocity: (N, E, D) velocity in m/s
            attitude: (roll, pitch, yaw) in radians
            command: (vn, ve, vd, yaw) commanded values
            battery_voltage: Current battery voltage
            armed: Whether drone is armed
            flight_mode: Current flight mode string

        Returns:
            True if frame was recorded, False if skipped (rate limiting)
        """
        if not self.is_recording or self.start_time is None:
            return False

        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self.start_time

        # Rate limiting
        if elapsed - self._last_record_time < self._record_interval:
            return False

        self._last_record_time = elapsed

        # Create frame
        frame = TelemetryFrame(
            timestamp=elapsed,
            position_n=position[0],
            position_e=position[1],
            position_d=position[2],
            velocity_n=velocity[0],
            velocity_e=velocity[1],
            velocity_d=velocity[2],
            roll=attitude[0],
            pitch=attitude[1],
            yaw=attitude[2],
            cmd_velocity_n=command[0] if command else 0.0,
            cmd_velocity_e=command[1] if command else 0.0,
            cmd_velocity_d=command[2] if command else 0.0,
            cmd_yaw=command[3] if command else 0.0,
            battery_voltage=battery_voltage,
            armed=armed,
            flight_mode=flight_mode,
        )

        self.frames.append(frame)
        return True

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save recorded telemetry to CSV file.

        Args:
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        settings = get_settings()

        # Create output directory
        output_dir = Path(settings.logging.telemetry_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"telemetry_{timestamp}.csv"

        filepath = output_dir / filename

        # Write CSV
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "timestamp",
                "position_n", "position_e", "position_d",
                "velocity_n", "velocity_e", "velocity_d",
                "roll", "pitch", "yaw",
                "cmd_velocity_n", "cmd_velocity_e", "cmd_velocity_d", "cmd_yaw",
                "battery_voltage", "armed", "flight_mode"
            ])

            # Data rows
            for frame in self.frames:
                writer.writerow([
                    frame.timestamp,
                    frame.position_n, frame.position_e, frame.position_d,
                    frame.velocity_n, frame.velocity_e, frame.velocity_d,
                    frame.roll, frame.pitch, frame.yaw,
                    frame.cmd_velocity_n, frame.cmd_velocity_e,
                    frame.cmd_velocity_d, frame.cmd_yaw,
                    frame.battery_voltage, frame.armed, frame.flight_mode
                ])

        logger.info("Telemetry saved", filepath=str(filepath), frames=len(self.frames))
        return filepath

    def get_statistics(self) -> dict:
        """
        Calculate statistics from recorded telemetry.

        Returns:
            Dictionary with flight statistics
        """
        if not self.frames:
            return {"error": "No telemetry data recorded"}

        positions = np.array([
            (f.position_n, f.position_e, f.position_d) for f in self.frames
        ])
        velocities = np.array([
            (f.velocity_n, f.velocity_e, f.velocity_d) for f in self.frames
        ])

        # Calculate velocity magnitudes
        vel_magnitudes = np.linalg.norm(velocities, axis=1)

        # Calculate total distance traveled
        position_diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(position_diffs, axis=1)
        total_distance = np.sum(distances)

        return {
            "duration_seconds": self.frames[-1].timestamp,
            "total_frames": len(self.frames),
            "total_distance_m": round(total_distance, 2),
            "max_velocity_ms": round(np.max(vel_magnitudes), 2),
            "avg_velocity_ms": round(np.mean(vel_magnitudes), 2),
            "max_altitude_m": round(-np.min(positions[:, 2]), 2),  # -D = altitude
            "min_altitude_m": round(-np.max(positions[:, 2]), 2),
        }
