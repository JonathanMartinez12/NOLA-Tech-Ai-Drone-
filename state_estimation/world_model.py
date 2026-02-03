"""
World model for tracking gates, obstacles, and race progress.

Maintains a representation of the known world state:
- Gate positions and states (detected, passed, etc.)
- Obstacle positions
- Race progress tracking
- Map updates from perception

This is a stub for Phase 1. Will be expanded with perception integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class GateState(Enum):
    """State of a gate in the race."""
    UNKNOWN = "unknown"       # Gate exists but not yet seen
    DETECTED = "detected"     # Gate has been visually detected
    APPROACHING = "approaching"  # Currently flying toward this gate
    PASSED = "passed"         # Successfully passed through
    MISSED = "missed"         # Failed to pass through


@dataclass
class Gate:
    """
    Represents a racing gate in the world.

    Attributes:
        id: Unique identifier for the gate
        position: (N, E, D) position in NED frame
        orientation: Yaw angle the gate is facing (radians)
        width: Gate width in meters
        height: Gate height in meters
        state: Current state in race progression
        confidence: Detection confidence (0-1)
        last_seen: Timestamp when last detected
    """

    id: int
    position: Tuple[float, float, float]  # NED frame
    orientation: float = 0.0  # Radians, 0 = facing North
    width: float = 2.0
    height: float = 2.0
    state: GateState = GateState.UNKNOWN
    confidence: float = 0.0
    last_seen: float = 0.0

    def get_center(self) -> Tuple[float, float, float]:
        """Get gate center position."""
        return self.position

    def get_approach_point(self, distance: float = 3.0) -> Tuple[float, float, float]:
        """
        Get a point in front of the gate to aim for.

        Args:
            distance: Distance in front of gate

        Returns:
            Position (N, E, D) to aim for when approaching
        """
        # Calculate point in front of gate based on orientation
        n = self.position[0] - distance * np.cos(self.orientation)
        e = self.position[1] - distance * np.sin(self.orientation)
        d = self.position[2]
        return (n, e, d)

    def get_exit_point(self, distance: float = 3.0) -> Tuple[float, float, float]:
        """
        Get a point behind the gate to aim for after passing.

        Args:
            distance: Distance behind gate

        Returns:
            Position (N, E, D) after passing through
        """
        n = self.position[0] + distance * np.cos(self.orientation)
        e = self.position[1] + distance * np.sin(self.orientation)
        d = self.position[2]
        return (n, e, d)


@dataclass
class Obstacle:
    """
    Represents an obstacle in the world (other drones, structures, etc.).

    Attributes:
        id: Unique identifier
        position: Current position in NED frame
        velocity: Estimated velocity (for moving obstacles)
        radius: Collision radius in meters
        last_seen: Timestamp when last detected
    """

    id: int
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    last_seen: float = 0.0

    def predict_position(self, dt: float) -> Tuple[float, float, float]:
        """Predict position after dt seconds using constant velocity."""
        return (
            self.position[0] + self.velocity[0] * dt,
            self.position[1] + self.velocity[1] * dt,
            self.position[2] + self.velocity[2] * dt,
        )


@dataclass
class WorldModel:
    """
    Maintains the world state for race navigation.

    Tracks all known gates, obstacles, and race progress.
    Updated continuously by perception system.

    Example:
        world = WorldModel()

        # Add gates from course definition
        world.add_gate(Gate(id=0, position=(10, 0, -5)))
        world.add_gate(Gate(id=1, position=(20, 10, -5)))

        # Update from perception
        world.update_gate(0, position=(10.1, 0.2, -5.1), confidence=0.9)

        # Track progress
        world.mark_gate_passed(0)
        next_gate = world.get_next_gate()
    """

    gates: Dict[int, Gate] = field(default_factory=dict)
    obstacles: Dict[int, Obstacle] = field(default_factory=dict)

    # Race progress
    current_gate_index: int = 0
    gates_passed: int = 0
    race_started: bool = False
    race_finished: bool = False

    # Configuration
    gate_sequence: List[int] = field(default_factory=list)  # Order of gates to pass

    def add_gate(self, gate: Gate) -> None:
        """Add a gate to the world model."""
        self.gates[gate.id] = gate
        if gate.id not in self.gate_sequence:
            self.gate_sequence.append(gate.id)
        logger.debug("Gate added", gate_id=gate.id, position=gate.position)

    def remove_gate(self, gate_id: int) -> None:
        """Remove a gate from the world model."""
        if gate_id in self.gates:
            del self.gates[gate_id]
            if gate_id in self.gate_sequence:
                self.gate_sequence.remove(gate_id)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the world model."""
        self.obstacles[obstacle.id] = obstacle

    def update_gate(
        self,
        gate_id: int,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Update gate information from perception.

        Args:
            gate_id: ID of gate to update
            position: New position estimate
            orientation: New orientation estimate
            confidence: Detection confidence
        """
        if gate_id not in self.gates:
            # Create new gate if not exists
            self.gates[gate_id] = Gate(
                id=gate_id,
                position=position or (0, 0, 0),
            )

        gate = self.gates[gate_id]

        if position is not None:
            # Could do filtering/smoothing here
            gate.position = position

        if orientation is not None:
            gate.orientation = orientation

        if confidence is not None:
            gate.confidence = confidence

        gate.last_seen = time.time()
        gate.state = GateState.DETECTED

    def update_obstacle(
        self,
        obstacle_id: int,
        position: Tuple[float, float, float],
        velocity: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Update obstacle position from perception."""
        if obstacle_id not in self.obstacles:
            self.obstacles[obstacle_id] = Obstacle(id=obstacle_id, position=position)

        obs = self.obstacles[obstacle_id]
        obs.position = position
        if velocity is not None:
            obs.velocity = velocity
        obs.last_seen = time.time()

    def get_next_gate(self) -> Optional[Gate]:
        """Get the next gate to fly through."""
        if self.current_gate_index >= len(self.gate_sequence):
            return None

        gate_id = self.gate_sequence[self.current_gate_index]
        return self.gates.get(gate_id)

    def mark_gate_passed(self, gate_id: int) -> None:
        """Mark a gate as successfully passed."""
        if gate_id in self.gates:
            self.gates[gate_id].state = GateState.PASSED
            self.gates_passed += 1

            # Advance to next gate
            if self.gate_sequence and gate_id == self.gate_sequence[self.current_gate_index]:
                self.current_gate_index += 1

            logger.info(
                "Gate passed",
                gate_id=gate_id,
                gates_passed=self.gates_passed,
                gates_remaining=len(self.gate_sequence) - self.current_gate_index
            )

            # Check if race finished
            if self.current_gate_index >= len(self.gate_sequence):
                self.race_finished = True
                logger.info("Race finished!", total_gates=self.gates_passed)

    def get_nearby_obstacles(
        self,
        position: Tuple[float, float, float],
        radius: float = 10.0,
    ) -> List[Obstacle]:
        """Get obstacles within radius of position."""
        nearby = []
        for obs in self.obstacles.values():
            distance = np.linalg.norm(
                np.array(obs.position) - np.array(position)
            )
            if distance <= radius:
                nearby.append(obs)
        return nearby

    def clear_stale_detections(self, max_age: float = 5.0) -> None:
        """Remove detections that haven't been seen recently."""
        current_time = time.time()

        # Clear stale obstacles
        stale_obstacles = [
            obs_id for obs_id, obs in self.obstacles.items()
            if current_time - obs.last_seen > max_age
        ]
        for obs_id in stale_obstacles:
            del self.obstacles[obs_id]

    def reset(self) -> None:
        """Reset race progress (gates remain, progress cleared)."""
        self.current_gate_index = 0
        self.gates_passed = 0
        self.race_started = False
        self.race_finished = False

        for gate in self.gates.values():
            gate.state = GateState.UNKNOWN
            gate.confidence = 0.0

        self.obstacles.clear()
        logger.info("World model reset")
