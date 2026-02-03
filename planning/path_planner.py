"""
Path planning for drone racing.

Generates paths through gates using graph-based or sampling-based methods.

Phase 3+ implementation - currently provides basic waypoint sequencing.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from state_estimation.world_model import WorldModel, Gate
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PathPlanner:
    """
    Path planner for drone racing navigation.

    Currently provides basic gate-to-gate waypoint generation.
    Phase 3+ will add:
    - A* path planning with obstacle avoidance
    - RRT/RRT* for complex environments
    - Racing line optimization

    Example:
        planner = PathPlanner(world_model)
        waypoints = planner.plan_path(current_position)
    """

    world_model: WorldModel
    _current_path: List[Tuple[float, float, float]] = field(default_factory=list, init=False)

    def plan_path(
        self,
        current_position: Tuple[float, float, float],
        lookahead_gates: int = 3,
    ) -> List[Tuple[float, float, float]]:
        """
        Plan a path through upcoming gates.

        Args:
            current_position: Current drone position (N, E, D)
            lookahead_gates: Number of gates to plan ahead

        Returns:
            List of waypoints (N, E, D) to fly through
        """
        waypoints = []

        # Get current gate and upcoming gates
        for i in range(lookahead_gates):
            gate_idx = self.world_model.current_gate_index + i
            if gate_idx >= len(self.world_model.gate_sequence):
                break

            gate_id = self.world_model.gate_sequence[gate_idx]
            gate = self.world_model.gates.get(gate_id)

            if gate is None:
                continue

            # For first gate, add approach point
            if i == 0:
                approach = gate.get_approach_point(distance=3.0)
                waypoints.append(approach)

            # Add gate center as waypoint
            waypoints.append(gate.get_center())

            # Add exit point for smooth transition to next gate
            exit_point = gate.get_exit_point(distance=2.0)
            waypoints.append(exit_point)

        self._current_path = waypoints
        return waypoints

    def get_next_waypoint(
        self,
        current_position: Tuple[float, float, float],
        acceptance_radius: float = 1.0,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get the next waypoint to fly to.

        Automatically advances through the path as waypoints are reached.

        Args:
            current_position: Current drone position
            acceptance_radius: Distance to consider waypoint reached

        Returns:
            Next waypoint or None if path complete
        """
        if not self._current_path:
            return None

        # Check if we've reached current waypoint
        next_wp = self._current_path[0]
        distance = np.linalg.norm(
            np.array(current_position) - np.array(next_wp)
        )

        if distance < acceptance_radius:
            # Remove reached waypoint
            self._current_path.pop(0)

            if not self._current_path:
                return None

            next_wp = self._current_path[0]

        return next_wp

    def replan(
        self,
        current_position: Tuple[float, float, float],
        reason: str = "unknown",
    ) -> List[Tuple[float, float, float]]:
        """
        Replan the path from current position.

        Called when path needs to be updated due to:
        - New gate detection
        - Obstacle appeared
        - Position error accumulated

        Args:
            current_position: Current drone position
            reason: Reason for replanning (for logging)

        Returns:
            New path waypoints
        """
        logger.info("Replanning path", reason=reason, position=current_position)
        return self.plan_path(current_position)

    def clear_path(self) -> None:
        """Clear the current planned path."""
        self._current_path = []


def plan_simple_path(
    start: Tuple[float, float, float],
    gates: List[Gate],
) -> List[Tuple[float, float, float]]:
    """
    Simple path planning through a list of gates.

    Utility function for basic path generation.

    Args:
        start: Starting position
        gates: Ordered list of gates to fly through

    Returns:
        List of waypoints
    """
    waypoints = []

    for i, gate in enumerate(gates):
        # Add approach point for first gate
        if i == 0:
            waypoints.append(gate.get_approach_point(distance=3.0))

        # Add gate center
        waypoints.append(gate.get_center())

        # Add exit point
        waypoints.append(gate.get_exit_point(distance=2.0))

    return waypoints
