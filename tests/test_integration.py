"""
Integration tests for the drone racing system.

These tests verify that components work together correctly.
For full integration testing, PX4 SITL must be running.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Import modules to test
from config.settings import get_settings, Settings
from config.drone_specs import get_drone_specs, DroneSpecs
from state_estimation.drone_state import DroneState
from state_estimation.world_model import WorldModel, Gate, GateState
from planning.path_planner import PathPlanner
from utils.telemetry_recorder import TelemetryRecorder


class TestConfigurationLoading:
    """Test configuration modules load correctly."""

    def test_settings_loads(self):
        """Test that settings can be loaded."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.control.max_velocity > 0

    def test_drone_specs_loads(self):
        """Test that drone specs can be loaded."""
        specs = get_drone_specs()
        assert isinstance(specs, DroneSpecs)
        assert specs.max_speed_ms > 0

    def test_settings_has_all_sections(self):
        """Test that settings has all expected sections."""
        settings = get_settings()

        assert hasattr(settings, "connection")
        assert hasattr(settings, "control")
        assert hasattr(settings, "takeoff_land")
        assert hasattr(settings, "perception")
        assert hasattr(settings, "state_estimation")
        assert hasattr(settings, "planning")
        assert hasattr(settings, "logging")


class TestWorldModel:
    """Test the world model for tracking race state."""

    @pytest.fixture
    def world_model(self):
        """Create a world model with gates."""
        world = WorldModel()

        # Add some test gates
        world.add_gate(Gate(id=0, position=(10, 0, -5)))
        world.add_gate(Gate(id=1, position=(20, 10, -5)))
        world.add_gate(Gate(id=2, position=(10, 20, -5)))
        world.add_gate(Gate(id=3, position=(0, 10, -5)))

        return world

    def test_add_gates(self, world_model):
        """Test adding gates to world model."""
        assert len(world_model.gates) == 4
        assert len(world_model.gate_sequence) == 4

    def test_get_next_gate(self, world_model):
        """Test getting the next gate to fly through."""
        next_gate = world_model.get_next_gate()

        assert next_gate is not None
        assert next_gate.id == 0

    def test_mark_gate_passed(self, world_model):
        """Test marking a gate as passed."""
        world_model.mark_gate_passed(0)

        assert world_model.gates[0].state == GateState.PASSED
        assert world_model.gates_passed == 1
        assert world_model.current_gate_index == 1

        # Next gate should now be gate 1
        next_gate = world_model.get_next_gate()
        assert next_gate.id == 1

    def test_race_completion(self, world_model):
        """Test race completion detection."""
        # Pass all gates
        for gate_id in world_model.gate_sequence:
            world_model.mark_gate_passed(gate_id)

        assert world_model.race_finished is True
        assert world_model.gates_passed == 4

    def test_reset(self, world_model):
        """Test world model reset."""
        # Pass some gates
        world_model.mark_gate_passed(0)
        world_model.mark_gate_passed(1)

        # Reset
        world_model.reset()

        assert world_model.current_gate_index == 0
        assert world_model.gates_passed == 0
        assert world_model.race_finished is False
        assert world_model.gates[0].state == GateState.UNKNOWN


class TestPathPlanner:
    """Test path planning functionality."""

    @pytest.fixture
    def world_with_gates(self):
        """Create world model with gates for planning."""
        world = WorldModel()
        world.add_gate(Gate(id=0, position=(10, 0, -5), orientation=0))
        world.add_gate(Gate(id=1, position=(20, 10, -5), orientation=1.57))
        return world

    def test_plan_path(self, world_with_gates):
        """Test basic path planning."""
        planner = PathPlanner(world_with_gates)
        current_pos = (0, 0, -5)

        waypoints = planner.plan_path(current_pos, lookahead_gates=2)

        assert len(waypoints) > 0
        # Should have approach, center, and exit points

    def test_get_next_waypoint(self, world_with_gates):
        """Test getting next waypoint."""
        planner = PathPlanner(world_with_gates)
        planner.plan_path((0, 0, -5))

        # Get first waypoint
        wp = planner.get_next_waypoint((0, 0, -5))
        assert wp is not None

    def test_waypoint_advancement(self, world_with_gates):
        """Test waypoint advancement when reached."""
        planner = PathPlanner(world_with_gates)
        planner.plan_path((0, 0, -5))

        # Get first waypoint
        wp1 = planner.get_next_waypoint((0, 0, -5))

        # "Reach" the waypoint (get next from its location)
        wp2 = planner.get_next_waypoint(wp1, acceptance_radius=1.0)

        # Should have advanced to next waypoint (or same if only one)
        assert wp2 is not None


class TestTelemetryRecorder:
    """Test telemetry recording functionality."""

    def test_start_stop(self):
        """Test starting and stopping recording."""
        recorder = TelemetryRecorder()

        recorder.start()
        assert recorder.is_recording is True

        recorder.stop()
        assert recorder.is_recording is False

    def test_record_frame(self):
        """Test recording telemetry frames."""
        recorder = TelemetryRecorder()
        recorder.start()

        # Record some data
        recorded = recorder.record(
            position=(0, 0, -5),
            velocity=(1, 0, 0),
            attitude=(0, 0, 0),
        )

        recorder.stop()

        assert len(recorder.frames) > 0

    def test_statistics(self):
        """Test statistics calculation."""
        recorder = TelemetryRecorder()
        recorder._record_interval = 0  # Disable rate limiting for test
        recorder.start()

        # Record multiple frames with movement
        for i in range(10):
            recorder._last_record_time = 0  # Force recording
            recorder.record(
                position=(i, 0, -5),
                velocity=(1, 0, 0),
                attitude=(0, 0, 0),
            )

        recorder.stop()

        stats = recorder.get_statistics()
        assert "total_frames" in stats
        assert "total_distance_m" in stats


class TestDroneState:
    """Test drone state tracking with mock drone."""

    @pytest.mark.asyncio
    async def test_state_initialization(self, mock_drone):
        """Test DroneState initialization."""
        state = DroneState(mock_drone)

        assert state.is_armed is False
        assert state.altitude == 0.0

    @pytest.mark.asyncio
    async def test_position_properties(self, mock_drone):
        """Test position property access."""
        state = DroneState(mock_drone)

        pos = state.position_ned
        assert len(pos) == 3

        vel = state.velocity_ned
        assert len(vel) == 3

    @pytest.mark.asyncio
    async def test_get_state_dict(self, mock_drone):
        """Test state dictionary generation."""
        state = DroneState(mock_drone)

        state_dict = state.get_state_dict()

        assert "position_ned" in state_dict
        assert "velocity_ned" in state_dict
        assert "attitude" in state_dict
        assert "is_armed" in state_dict


class TestGateIntegration:
    """Test gate detection and navigation integration."""

    def test_gate_approach_point(self):
        """Test gate approach point calculation."""
        gate = Gate(id=0, position=(10, 0, -5), orientation=0)

        approach = gate.get_approach_point(distance=3.0)

        # Approach should be 3m in front of gate (facing North)
        assert approach[0] < gate.position[0]  # South of gate
        assert approach[1] == pytest.approx(gate.position[1])

    def test_gate_exit_point(self):
        """Test gate exit point calculation."""
        gate = Gate(id=0, position=(10, 0, -5), orientation=0)

        exit_pt = gate.get_exit_point(distance=3.0)

        # Exit should be 3m behind gate (past it to the North)
        assert exit_pt[0] > gate.position[0]  # North of gate
        assert exit_pt[1] == pytest.approx(gate.position[1])


# Skip these tests if SITL is not available
@pytest.mark.skipif(True, reason="Requires PX4 SITL running")
class TestSITLIntegration:
    """Integration tests requiring PX4 SITL."""

    @pytest.mark.asyncio
    async def test_connection_to_sitl(self):
        """Test connecting to PX4 SITL."""
        from mavsdk import System

        drone = System()
        await drone.connect(system_address="udp://:14540")

        async for state in drone.core.connection_state():
            if state.is_connected:
                assert True
                return

        pytest.fail("Could not connect to SITL")

    @pytest.mark.asyncio
    async def test_telemetry_reception(self):
        """Test receiving telemetry from SITL."""
        from mavsdk import System

        drone = System()
        await drone.connect(system_address="udp://:14540")

        state = DroneState(drone)
        await state.start_tracking()

        await asyncio.sleep(1.0)

        assert state.is_connected
        await state.stop_tracking()
