"""
Collision avoidance for safe racing.

Provides real-time collision detection and avoidance for:
- Other competing drones
- Course obstacles
- Race boundaries

Phase 3+ implementation - stub provided for structure.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from state_estimation.world_model import Obstacle
from config.settings import get_settings
from config.drone_specs import get_drone_specs
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CollisionRisk:
    """Assessment of collision risk with an obstacle."""

    obstacle_id: int
    distance: float  # Current distance in meters
    time_to_collision: float  # Estimated time to collision (-1 if diverging)
    risk_level: str  # "none", "low", "medium", "high", "critical"
    avoidance_direction: Optional[Tuple[float, float, float]] = None  # Suggested avoidance


@dataclass
class CollisionAvoidance:
    """
    Collision avoidance system for drone racing.

    Monitors obstacles and provides avoidance commands when necessary.

    Phase 3+ will implement:
    - Predictive collision detection
    - Velocity obstacle method
    - Emergency maneuvers
    - Multi-drone coordination

    Currently provides basic distance-based avoidance.
    """

    _last_check_time: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.settings = get_settings()
        self.specs = get_drone_specs()

    def check_collisions(
        self,
        current_pos: Tuple[float, float, float],
        current_vel: Tuple[float, float, float],
        obstacles: List[Obstacle],
    ) -> List[CollisionRisk]:
        """
        Check for potential collisions with obstacles.

        Args:
            current_pos: Current drone position (N, E, D)
            current_vel: Current drone velocity (vN, vE, vD)
            obstacles: List of known obstacles

        Returns:
            List of collision risks, sorted by severity
        """
        risks = []

        for obs in obstacles:
            risk = self._assess_collision_risk(current_pos, current_vel, obs)
            if risk.risk_level != "none":
                risks.append(risk)

        # Sort by risk level (critical first)
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
        risks.sort(key=lambda r: risk_order.get(r.risk_level, 4))

        return risks

    def get_avoidance_velocity(
        self,
        current_pos: Tuple[float, float, float],
        desired_vel: Tuple[float, float, float],
        obstacles: List[Obstacle],
    ) -> Tuple[float, float, float]:
        """
        Modify velocity to avoid collisions.

        Uses velocity obstacles to compute a collision-free velocity
        that stays as close as possible to the desired velocity.

        Args:
            current_pos: Current position
            desired_vel: Desired velocity without avoidance
            obstacles: List of obstacles

        Returns:
            Collision-free velocity
        """
        risks = self.check_collisions(current_pos, desired_vel, obstacles)

        if not risks:
            return desired_vel

        # Find highest risk
        highest_risk = risks[0]

        if highest_risk.risk_level in ("critical", "high"):
            # Emergency avoidance
            if highest_risk.avoidance_direction:
                # Blend avoidance direction with desired velocity
                avoidance = np.array(highest_risk.avoidance_direction)
                desired = np.array(desired_vel)

                # Strength of avoidance based on risk
                if highest_risk.risk_level == "critical":
                    blend = 0.9  # Mostly avoidance
                else:
                    blend = 0.6  # Significant avoidance

                result = (1 - blend) * desired + blend * avoidance
                return tuple(result)

        return desired_vel

    def is_path_safe(
        self,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        obstacles: List[Obstacle],
    ) -> bool:
        """
        Check if a straight-line path is free of obstacles.

        Args:
            start_pos: Path start position
            end_pos: Path end position
            obstacles: List of obstacles

        Returns:
            True if path is safe
        """
        safety_margin = self.settings.planning.safety_margin + self.specs.collision_radius_m

        # Check distance from path to each obstacle
        path = np.array(end_pos) - np.array(start_pos)
        path_length = np.linalg.norm(path)

        if path_length < 0.01:
            return True

        path_dir = path / path_length

        for obs in obstacles:
            # Vector from start to obstacle
            to_obs = np.array(obs.position) - np.array(start_pos)

            # Project onto path
            proj_length = np.dot(to_obs, path_dir)

            # Check if obstacle is along the path segment
            if proj_length < 0 or proj_length > path_length:
                continue

            # Distance from path to obstacle
            proj_point = np.array(start_pos) + proj_length * path_dir
            distance = np.linalg.norm(np.array(obs.position) - proj_point)

            if distance < safety_margin + obs.radius:
                return False

        return True

    def _assess_collision_risk(
        self,
        current_pos: Tuple[float, float, float],
        current_vel: Tuple[float, float, float],
        obstacle: Obstacle,
    ) -> CollisionRisk:
        """Assess collision risk with a single obstacle."""
        # Calculate relative position and velocity
        rel_pos = np.array(obstacle.position) - np.array(current_pos)
        rel_vel = np.array(obstacle.velocity) - np.array(current_vel)

        distance = float(np.linalg.norm(rel_pos))

        # Time to closest approach
        vel_squared = np.dot(rel_vel, rel_vel)
        if vel_squared > 0.01:
            time_to_closest = -np.dot(rel_pos, rel_vel) / vel_squared
        else:
            time_to_closest = float("inf")

        # Distance at closest approach
        if time_to_closest > 0 and time_to_closest < float("inf"):
            closest_pos = rel_pos + rel_vel * time_to_closest
            closest_distance = float(np.linalg.norm(closest_pos))
        else:
            closest_distance = distance

        # Determine risk level
        safety_margin = self.specs.collision_radius_m + obstacle.radius + 1.0

        if closest_distance < safety_margin and time_to_closest > 0:
            if time_to_closest < 1.0:
                risk_level = "critical"
            elif time_to_closest < 2.0:
                risk_level = "high"
            elif time_to_closest < 5.0:
                risk_level = "medium"
            else:
                risk_level = "low"
        elif distance < safety_margin:
            risk_level = "high"
        else:
            risk_level = "none"

        # Calculate avoidance direction (perpendicular to relative position)
        avoidance_direction = None
        if risk_level != "none":
            # Avoid perpendicular to obstacle direction
            if distance > 0.1:
                avoid = np.cross(rel_pos / distance, [0, 0, 1])
                avoid = avoid / np.linalg.norm(avoid) if np.linalg.norm(avoid) > 0.1 else np.array([1, 0, 0])
                avoidance_direction = tuple(avoid)

        return CollisionRisk(
            obstacle_id=obstacle.id,
            distance=distance,
            time_to_collision=time_to_closest if time_to_closest > 0 else -1,
            risk_level=risk_level,
            avoidance_direction=avoidance_direction,
        )
