"""
Trajectory generation for smooth, time-optimal flight.

Generates smooth trajectories that respect drone dynamics while
minimizing time through the course.

Phase 3+ implementation - currently provides basic polynomial trajectories.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from config.drone_specs import get_drone_specs
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrajectoryPoint:
    """A single point on a trajectory."""

    time: float  # Time from trajectory start (seconds)
    position: Tuple[float, float, float]  # (N, E, D)
    velocity: Tuple[float, float, float]  # (vN, vE, vD)
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0  # Radians


@dataclass
class Trajectory:
    """
    A complete trajectory with timing information.

    Contains a series of trajectory points that define position,
    velocity, and acceleration over time.
    """

    points: List[TrajectoryPoint] = field(default_factory=list)
    total_time: float = 0.0

    def get_state_at_time(self, t: float) -> Optional[TrajectoryPoint]:
        """
        Interpolate trajectory state at a given time.

        Args:
            t: Time from trajectory start

        Returns:
            Interpolated TrajectoryPoint or None if t is out of range
        """
        if not self.points or t < 0:
            return None

        if t >= self.total_time:
            return self.points[-1]

        # Find surrounding points
        for i in range(len(self.points) - 1):
            if self.points[i].time <= t < self.points[i + 1].time:
                # Linear interpolation between points
                p1 = self.points[i]
                p2 = self.points[i + 1]
                dt = p2.time - p1.time
                alpha = (t - p1.time) / dt if dt > 0 else 0

                pos = tuple(
                    p1.position[j] + alpha * (p2.position[j] - p1.position[j])
                    for j in range(3)
                )
                vel = tuple(
                    p1.velocity[j] + alpha * (p2.velocity[j] - p1.velocity[j])
                    for j in range(3)
                )

                return TrajectoryPoint(
                    time=t,
                    position=pos,
                    velocity=vel,
                    yaw=p1.yaw + alpha * (p2.yaw - p1.yaw),
                )

        return self.points[-1]


@dataclass
class TrajectoryGenerator:
    """
    Generates smooth trajectories between waypoints.

    Phase 3+ will implement:
    - Minimum-snap polynomial trajectories
    - Time-optimal trajectories
    - Dynamics-constrained optimization

    Currently provides simple linear interpolation.
    """

    def __post_init__(self):
        self.settings = get_settings()
        self.specs = get_drone_specs()

    def generate_trajectory(
        self,
        waypoints: List[Tuple[float, float, float]],
        speed: float = 2.0,
    ) -> Trajectory:
        """
        Generate a trajectory through waypoints.

        Args:
            waypoints: List of (N, E, D) positions
            speed: Desired average speed in m/s

        Returns:
            Trajectory object with timed points
        """
        if len(waypoints) < 2:
            logger.warning("Need at least 2 waypoints for trajectory")
            return Trajectory()

        # Clamp speed to drone limits
        speed = min(speed, self.specs.max_horizontal_speed_ms)

        points = []
        current_time = 0.0

        for i in range(len(waypoints)):
            wp = waypoints[i]

            if i == 0:
                # First waypoint - start at rest
                points.append(TrajectoryPoint(
                    time=0.0,
                    position=wp,
                    velocity=(0.0, 0.0, 0.0),
                ))
            else:
                # Calculate segment
                prev_wp = waypoints[i - 1]
                segment = np.array(wp) - np.array(prev_wp)
                distance = float(np.linalg.norm(segment))

                if distance < 0.01:
                    continue

                # Time for segment
                segment_time = distance / speed
                current_time += segment_time

                # Direction for velocity
                direction = segment / distance
                velocity = tuple(direction * speed)

                points.append(TrajectoryPoint(
                    time=current_time,
                    position=wp,
                    velocity=velocity if i < len(waypoints) - 1 else (0.0, 0.0, 0.0),
                ))

        trajectory = Trajectory(
            points=points,
            total_time=current_time,
        )

        logger.debug(
            "Generated trajectory",
            waypoints=len(waypoints),
            points=len(points),
            duration=current_time,
        )

        return trajectory

    def generate_minimum_time_trajectory(
        self,
        waypoints: List[Tuple[float, float, float]],
    ) -> Trajectory:
        """
        Generate a minimum-time trajectory (Phase 4+).

        This will optimize for minimum lap time while respecting
        dynamics constraints.

        Args:
            waypoints: List of positions to pass through

        Returns:
            Time-optimized trajectory
        """
        # TODO Phase 4: Implement minimum-time optimization
        # - Use polynomial trajectory primitives
        # - Optimize timing between waypoints
        # - Account for acceleration limits

        # For now, use maximum safe speed
        max_speed = min(
            self.specs.max_horizontal_speed_ms * 0.8,
            self.settings.control.max_velocity,
        )
        return self.generate_trajectory(waypoints, speed=max_speed)


def generate_polynomial_trajectory(
    start_pos: NDArray[np.float64],
    end_pos: NDArray[np.float64],
    start_vel: NDArray[np.float64],
    end_vel: NDArray[np.float64],
    duration: float,
    num_points: int = 20,
) -> List[TrajectoryPoint]:
    """
    Generate a minimum-jerk polynomial trajectory between two states.

    Uses 5th order polynomial for smooth motion with specified
    boundary conditions.

    Args:
        start_pos: Starting position [x, y, z]
        end_pos: Ending position [x, y, z]
        start_vel: Starting velocity [vx, vy, vz]
        end_vel: Ending velocity [vx, vy, vz]
        duration: Trajectory duration in seconds
        num_points: Number of points to generate

    Returns:
        List of TrajectoryPoints
    """
    points = []
    times = np.linspace(0, duration, num_points)

    for t in times:
        # Normalized time
        tau = t / duration if duration > 0 else 0

        # 5th order polynomial coefficients for minimum jerk
        # p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # With constraints: p(0)=p0, p(1)=p1, p'(0)=v0*T, p'(1)=v1*T
        #                   p''(0)=0, p''(1)=0

        # Position polynomial
        h00 = 1 - 10*tau**3 + 15*tau**4 - 6*tau**5
        h10 = tau - 6*tau**3 + 8*tau**4 - 3*tau**5
        h01 = 10*tau**3 - 15*tau**4 + 6*tau**5
        h11 = -4*tau**3 + 7*tau**4 - 3*tau**5

        pos = (
            h00 * start_pos +
            h10 * duration * start_vel +
            h01 * end_pos +
            h11 * duration * end_vel
        )

        # Velocity polynomial (derivative)
        dh00 = -30*tau**2 + 60*tau**3 - 30*tau**4
        dh10 = 1 - 18*tau**2 + 32*tau**3 - 15*tau**4
        dh01 = 30*tau**2 - 60*tau**3 + 30*tau**4
        dh11 = -12*tau**2 + 28*tau**3 - 15*tau**4

        vel = (
            dh00 * start_pos / duration +
            dh10 * start_vel +
            dh01 * end_pos / duration +
            dh11 * end_vel
        ) if duration > 0 else np.zeros(3)

        points.append(TrajectoryPoint(
            time=t,
            position=tuple(pos),
            velocity=tuple(vel),
        ))

    return points
