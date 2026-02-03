"""
Tests for control modules.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from control.pid_controller import PIDController, PIDController3D, PositionController


class TestPIDController:
    """Tests for the PIDController class."""

    def test_proportional_only(self):
        """Test P-only controller."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)

        # Error of 10 should give output of 10
        output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)
        assert output == pytest.approx(10.0)

    def test_proportional_gain(self):
        """Test proportional gain effect."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)

        # Error of 10 with gain of 2 should give output of 20
        output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)
        assert output == pytest.approx(20.0)

    def test_integral_accumulation(self):
        """Test integral term accumulates over time."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)

        # Multiple updates with constant error should accumulate
        for _ in range(10):
            output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)

        # After 10 updates of 0.1s with error of 10:
        # integral = 10 * 10 * 0.1 = 10
        assert output == pytest.approx(10.0, rel=0.1)

    def test_integral_anti_windup(self):
        """Test integral anti-windup limits."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, integral_max=5.0)

        # Many updates should be limited by anti-windup
        for _ in range(100):
            output = pid.update(setpoint=100.0, measurement=0.0, dt=0.1)

        # Output should be clamped
        assert output <= 5.0

    def test_output_clamping(self):
        """Test output saturation limits."""
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0, output_min=-5.0, output_max=5.0)

        # Large error should be clamped
        output = pid.update(setpoint=100.0, measurement=0.0, dt=0.1)
        assert output == 5.0

        output = pid.update(setpoint=-100.0, measurement=0.0, dt=0.1)
        assert output == -5.0

    def test_reset(self):
        """Test controller reset."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)

        # Accumulate some state
        for _ in range(10):
            pid.update(setpoint=10.0, measurement=0.0, dt=0.1)

        # Reset
        pid.reset()

        # After reset, integral should be zero
        # First update should only have P term
        output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)
        assert output == pytest.approx(10.0, rel=0.1)  # Only P term

    def test_zero_dt_handling(self):
        """Test handling of zero time step."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)

        # Should not crash with dt=0
        output = pid.update(setpoint=10.0, measurement=0.0, dt=0.0)
        assert not float("inf") == output


class TestPIDController3D:
    """Tests for the 3D PID controller."""

    def test_independent_axes(self):
        """Test that axes are controlled independently."""
        pid = PIDController3D(
            kp=(1.0, 2.0, 3.0),
            ki=(0.0, 0.0, 0.0),
            kd=(0.0, 0.0, 0.0),
        )

        setpoint = (10.0, 10.0, 10.0)
        measurement = (0.0, 0.0, 0.0)

        output = pid.update(setpoint, measurement, dt=0.1)

        # Each axis should have different output due to different gains
        assert output[0] == pytest.approx(10.0)
        assert output[1] == pytest.approx(20.0)
        assert output[2] == pytest.approx(30.0)

    def test_reset(self):
        """Test 3D controller reset."""
        pid = PIDController3D()

        # Accumulate some state
        for _ in range(10):
            pid.update((10.0, 10.0, 10.0), (0.0, 0.0, 0.0), dt=0.1)

        # Reset
        pid.reset()

        # Verify reset worked (no assertion error)
        output = pid.update((10.0, 10.0, 10.0), (0.0, 0.0, 0.0), dt=0.1)
        assert len(output) == 3


class TestPositionController:
    """Tests for the position controller."""

    def test_zero_error(self):
        """Test zero output when at target."""
        controller = PositionController()

        target = (10.0, 10.0, -5.0)
        current = (10.0, 10.0, -5.0)

        vel = controller.update(target, current, dt=0.1)

        assert vel[0] == pytest.approx(0.0)
        assert vel[1] == pytest.approx(0.0)
        assert vel[2] == pytest.approx(0.0)

    def test_position_error_generates_velocity(self):
        """Test that position error generates velocity command."""
        controller = PositionController(kp_horizontal=1.0, kp_vertical=0.5)

        target = (10.0, 0.0, -5.0)
        current = (0.0, 0.0, -5.0)

        vel = controller.update(target, current, dt=0.1)

        # Should command velocity toward target
        assert vel[0] > 0  # Positive North velocity
        assert vel[1] == pytest.approx(0.0, abs=0.1)  # No East velocity
        assert vel[2] == pytest.approx(0.0, abs=0.1)  # No vertical velocity

    def test_velocity_limits(self):
        """Test velocity output limits."""
        controller = PositionController(
            kp_horizontal=10.0,  # High gain
            max_horizontal_velocity=2.0,
        )

        target = (100.0, 0.0, -5.0)  # Far target
        current = (0.0, 0.0, -5.0)

        vel = controller.update(target, current, dt=0.1)

        # Should be clamped to max
        assert abs(vel[0]) <= 2.0
        assert abs(vel[1]) <= 2.0

    def test_feed_forward(self):
        """Test velocity feed-forward term."""
        controller = PositionController(kp_horizontal=1.0)

        target = (0.0, 0.0, -5.0)
        current = (0.0, 0.0, -5.0)
        feed_forward = (1.0, 0.0, 0.0)

        vel = controller.update(target, current, dt=0.1, feed_forward=feed_forward)

        # With zero position error, output should be feed-forward
        assert vel[0] == pytest.approx(1.0)
