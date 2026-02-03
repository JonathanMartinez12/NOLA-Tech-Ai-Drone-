"""
Racing line optimization for minimum lap time.

Calculates the optimal path through gates that minimizes lap time
by optimizing entry/exit angles and corner cutting.

Phase 4+ implementation - stub provided for structure.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from state_estimation.world_model import Gate
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RacingLine:
    """
    Optimized racing line through a course.

    Contains the optimal path through all gates that minimizes
    total lap time while respecting gate passage requirements.

    Attributes:
        waypoints: Optimized waypoint positions
        gate_entry_angles: Optimal entry angle for each gate (radians)
        gate_exit_angles: Optimal exit angle for each gate (radians)
        estimated_lap_time: Predicted lap time in seconds
    """

    waypoints: List[Tuple[float, float, float]]
    gate_entry_angles: List[float]
    gate_exit_angles: List[float]
    estimated_lap_time: float = 0.0


@dataclass
class RacingLineOptimizer:
    """
    Optimizes the racing line through a course.

    Phase 4+ will implement:
    - Apex optimization for corners
    - Entry/exit angle optimization
    - Speed profiling for minimum time
    - Iterative path refinement

    Currently provides basic racing line estimation.
    """

    def __post_init__(self):
        self.settings = get_settings()

    def optimize(
        self,
        gates: List[Gate],
        max_iterations: int = 100,
    ) -> RacingLine:
        """
        Optimize racing line through gates.

        Args:
            gates: Ordered list of gates to pass through
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized RacingLine
        """
        if len(gates) < 2:
            return RacingLine(
                waypoints=[g.get_center() for g in gates],
                gate_entry_angles=[0.0] * len(gates),
                gate_exit_angles=[0.0] * len(gates),
            )

        # Basic racing line: connect gate centers with corner cutting
        waypoints = []
        entry_angles = []
        exit_angles = []

        cutting_factor = self.settings.planning.corner_cutting_factor

        for i, gate in enumerate(gates):
            # Get previous and next gate for angle calculation
            prev_gate = gates[i - 1] if i > 0 else None
            next_gate = gates[i + 1] if i < len(gates) - 1 else None

            # Calculate entry angle
            if prev_gate:
                delta = np.array(gate.get_center()) - np.array(prev_gate.get_center())
                entry_angle = np.arctan2(delta[1], delta[0])
            else:
                entry_angle = gate.orientation

            # Calculate exit angle
            if next_gate:
                delta = np.array(next_gate.get_center()) - np.array(gate.get_center())
                exit_angle = np.arctan2(delta[1], delta[0])
            else:
                exit_angle = gate.orientation

            entry_angles.append(entry_angle)
            exit_angles.append(exit_angle)

            # Add approach waypoint (cut corner if applicable)
            if prev_gate and next_gate:
                # Calculate cutting point
                gate_center = np.array(gate.get_center())

                # Direction from previous to next gate
                through_dir = np.array(next_gate.get_center()) - np.array(prev_gate.get_center())
                through_dir = through_dir / np.linalg.norm(through_dir)

                # Offset toward the inside of the turn
                turn_dir = np.cross([0, 0, 1], np.append(through_dir[:2], 0))[:2]
                offset = turn_dir * cutting_factor

                # Apply offset (only horizontal)
                cut_point = gate_center.copy()
                cut_point[0] += offset[0]
                cut_point[1] += offset[1]

                waypoints.append(tuple(cut_point))
            else:
                waypoints.append(gate.get_center())

        # Estimate lap time (simplified)
        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            segment = np.array(waypoints[i + 1]) - np.array(waypoints[i])
            total_distance += np.linalg.norm(segment)

        avg_speed = 5.0  # Conservative estimate
        estimated_time = total_distance / avg_speed

        return RacingLine(
            waypoints=waypoints,
            gate_entry_angles=entry_angles,
            gate_exit_angles=exit_angles,
            estimated_lap_time=estimated_time,
        )

    def calculate_apex(
        self,
        entry_point: Tuple[float, float, float],
        gate: Gate,
        exit_point: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Calculate the optimal apex point for a turn through a gate.

        The apex is the point where the drone is closest to the
        inside of the turn while still passing through the gate.

        Args:
            entry_point: Position before gate
            gate: The gate to pass through
            exit_point: Position after gate

        Returns:
            Optimal apex position
        """
        # TODO Phase 4: Implement proper apex calculation
        # For now, just return gate center
        return gate.get_center()
