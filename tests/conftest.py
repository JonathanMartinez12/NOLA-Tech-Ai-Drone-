"""
Pytest configuration and shared fixtures for drone racing tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_drone():
    """
    Create a mock MAVSDK drone for testing without hardware.

    Returns a MagicMock that simulates MAVSDK System behavior.
    """
    drone = MagicMock()

    # Mock telemetry streams
    async def mock_position_velocity():
        """Simulate position/velocity telemetry."""
        while True:
            pos_vel = MagicMock()
            pos_vel.position.north_m = 0.0
            pos_vel.position.east_m = 0.0
            pos_vel.position.down_m = -5.0
            pos_vel.velocity.north_m_s = 0.0
            pos_vel.velocity.east_m_s = 0.0
            pos_vel.velocity.down_m_s = 0.0
            yield pos_vel
            await asyncio.sleep(0.02)

    async def mock_attitude():
        """Simulate attitude telemetry."""
        while True:
            attitude = MagicMock()
            attitude.roll_deg = 0.0
            attitude.pitch_deg = 0.0
            attitude.yaw_deg = 0.0
            yield attitude
            await asyncio.sleep(0.02)

    async def mock_armed():
        """Simulate armed state."""
        yield False
        while True:
            yield True
            await asyncio.sleep(0.1)

    async def mock_flight_mode():
        """Simulate flight mode."""
        from mavsdk.telemetry import FlightMode
        while True:
            yield FlightMode.HOLD
            await asyncio.sleep(0.1)

    async def mock_landed_state():
        """Simulate landed state."""
        from mavsdk.telemetry import LandedState
        while True:
            yield LandedState.ON_GROUND
            await asyncio.sleep(0.1)

    async def mock_health():
        """Simulate health telemetry."""
        while True:
            health = MagicMock()
            health.is_global_position_ok = True
            health.is_home_position_ok = True
            yield health
            await asyncio.sleep(0.1)

    async def mock_connection_state():
        """Simulate connection state."""
        state = MagicMock()
        state.is_connected = True
        yield state

    async def mock_identification():
        """Simulate identification."""
        info = MagicMock()
        info.hardware_uid = "test-drone-001"
        yield info

    # Set up telemetry mocks
    drone.telemetry.position_velocity_ned = mock_position_velocity
    drone.telemetry.attitude_euler = mock_attitude
    drone.telemetry.armed = mock_armed
    drone.telemetry.flight_mode = mock_flight_mode
    drone.telemetry.landed_state = mock_landed_state
    drone.telemetry.health = mock_health
    drone.core.connection_state = mock_connection_state
    drone.info.get_identification = mock_identification

    # Mock actions
    drone.action.arm = AsyncMock()
    drone.action.disarm = AsyncMock()
    drone.action.takeoff = AsyncMock()
    drone.action.land = AsyncMock()
    drone.action.set_takeoff_altitude = AsyncMock()

    # Mock offboard
    drone.offboard.start = AsyncMock()
    drone.offboard.stop = AsyncMock()
    drone.offboard.set_velocity_ned = AsyncMock()
    drone.offboard.set_position_ned = AsyncMock()
    drone.offboard.set_velocity_body = AsyncMock()

    return drone


@pytest.fixture
def sample_image():
    """Create a sample test image with an orange gate."""
    # Create a black 640x480 image
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw an orange rectangle (simulating a gate)
    # Orange in BGR: (0, 128, 255)
    orange = (0, 128, 255)
    image[150:330, 200:440] = orange

    # Make it look more like a gate frame (hollow)
    image[180:300, 230:410] = (0, 0, 0)

    return image


@pytest.fixture
def sample_waypoints():
    """Sample waypoints for testing navigation."""
    return [
        (10.0, 0.0, -5.0),   # 10m North
        (10.0, 10.0, -5.0),  # 10m North, 10m East
        (0.0, 10.0, -5.0),   # 10m East
        (0.0, 0.0, -5.0),    # Back to start
    ]
