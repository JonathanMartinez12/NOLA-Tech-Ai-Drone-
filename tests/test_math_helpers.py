"""
Tests for math helper functions.
"""

import math
import numpy as np
import pytest

from utils.math_helpers import (
    normalize_angle,
    normalize_angle_deg,
    angle_difference,
    angle_difference_deg,
    clamp,
    clamp_vector,
    vector_magnitude,
    vector_normalize,
    distance_3d,
    distance_2d,
    ned_to_enu,
    enu_to_ned,
    body_to_ned,
    ned_to_body,
    quaternion_to_euler,
    euler_to_quaternion,
    lerp,
    lerp_vector,
    smooth_step,
    calculate_bearing,
    calculate_bearing_deg,
)


class TestAngleNormalization:
    """Tests for angle normalization functions."""

    def test_normalize_angle_positive(self):
        """Test normalizing angles > pi."""
        assert abs(normalize_angle(3 * math.pi) - (-math.pi)) < 0.001
        assert abs(normalize_angle(2.5 * math.pi) - (0.5 * math.pi)) < 0.001

    def test_normalize_angle_negative(self):
        """Test normalizing angles < -pi."""
        assert abs(normalize_angle(-3 * math.pi) - (math.pi)) < 0.001
        assert abs(normalize_angle(-2.5 * math.pi) - (-0.5 * math.pi)) < 0.001

    def test_normalize_angle_in_range(self):
        """Test angles already in valid range."""
        assert normalize_angle(0.5) == pytest.approx(0.5)
        assert normalize_angle(-0.5) == pytest.approx(-0.5)
        assert normalize_angle(math.pi) == pytest.approx(math.pi)

    def test_normalize_angle_deg(self):
        """Test degree normalization."""
        assert normalize_angle_deg(270) == pytest.approx(-90)
        assert normalize_angle_deg(-270) == pytest.approx(90)
        assert normalize_angle_deg(180) == pytest.approx(180)


class TestAngleDifference:
    """Tests for angle difference calculations."""

    def test_angle_difference_positive(self):
        """Test positive angle difference."""
        diff = angle_difference(0, math.pi / 2)
        assert diff == pytest.approx(math.pi / 2)

    def test_angle_difference_negative(self):
        """Test negative angle difference."""
        diff = angle_difference(0, -math.pi / 2)
        assert diff == pytest.approx(-math.pi / 2)

    def test_angle_difference_wrap_around(self):
        """Test angle difference across the wrap-around point."""
        # Going from 170° to -170° should be a +20° turn, not -340°
        diff = angle_difference_deg(170, -170)
        assert diff == pytest.approx(20)

        # Going from -170° to 170° should be a -20° turn
        diff = angle_difference_deg(-170, 170)
        assert diff == pytest.approx(-20)


class TestClamp:
    """Tests for clamping functions."""

    def test_clamp_in_range(self):
        """Test clamping values within range."""
        assert clamp(5, 0, 10) == 5
        assert clamp(0, 0, 10) == 0
        assert clamp(10, 0, 10) == 10

    def test_clamp_below_range(self):
        """Test clamping values below range."""
        assert clamp(-5, 0, 10) == 0

    def test_clamp_above_range(self):
        """Test clamping values above range."""
        assert clamp(15, 0, 10) == 10

    def test_clamp_vector(self):
        """Test vector magnitude clamping."""
        vec = np.array([3.0, 4.0, 0.0])  # Magnitude = 5
        clamped = clamp_vector(vec, 2.5)
        assert vector_magnitude(clamped) == pytest.approx(2.5)

        # Direction should be preserved
        expected_dir = vec / 5.0
        actual_dir = clamped / 2.5
        np.testing.assert_array_almost_equal(expected_dir, actual_dir)


class TestVectorOperations:
    """Tests for vector operations."""

    def test_vector_magnitude(self):
        """Test magnitude calculation."""
        assert vector_magnitude(np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)
        assert vector_magnitude(np.array([1.0, 1.0, 1.0])) == pytest.approx(math.sqrt(3))

    def test_vector_normalize(self):
        """Test vector normalization."""
        vec = np.array([3.0, 4.0, 0.0])
        normalized = vector_normalize(vec)
        np.testing.assert_array_almost_equal(normalized, [0.6, 0.8, 0.0])

    def test_vector_normalize_zero(self):
        """Test normalizing zero vector."""
        vec = np.array([0.0, 0.0, 0.0])
        normalized = vector_normalize(vec)
        np.testing.assert_array_equal(normalized, [0.0, 0.0, 0.0])

    def test_distance_3d(self):
        """Test 3D distance calculation."""
        p1 = (0, 0, 0)
        p2 = (3, 4, 0)
        assert distance_3d(p1, p2) == pytest.approx(5.0)

    def test_distance_2d(self):
        """Test 2D distance calculation (ignoring z)."""
        p1 = (0, 0, 100)
        p2 = (3, 4, 200)
        assert distance_2d(p1, p2) == pytest.approx(5.0)


class TestCoordinateTransforms:
    """Tests for coordinate frame transformations."""

    def test_ned_to_enu_round_trip(self):
        """Test NED -> ENU -> NED round trip."""
        original = np.array([10.0, 5.0, -3.0])
        enu = ned_to_enu(original)
        back = enu_to_ned(enu)
        np.testing.assert_array_almost_equal(original, back)

    def test_ned_to_enu_values(self):
        """Test specific NED to ENU conversion."""
        ned = np.array([10.0, 5.0, -3.0])  # 10m North, 5m East, 3m up
        enu = ned_to_enu(ned)
        # ENU should be: 5m East, 10m North, 3m Up
        np.testing.assert_array_almost_equal(enu, [5.0, 10.0, 3.0])

    def test_body_to_ned_facing_north(self):
        """Test body to NED transform when facing North."""
        body = np.array([1.0, 0.0, 0.0])  # 1m forward
        ned = body_to_ned(body, yaw_rad=0.0)
        np.testing.assert_array_almost_equal(ned, [1.0, 0.0, 0.0])  # 1m North

    def test_body_to_ned_facing_east(self):
        """Test body to NED transform when facing East."""
        body = np.array([1.0, 0.0, 0.0])  # 1m forward
        ned = body_to_ned(body, yaw_rad=math.pi / 2)
        np.testing.assert_array_almost_equal(ned, [0.0, 1.0, 0.0], decimal=5)  # 1m East


class TestQuaternionOperations:
    """Tests for quaternion operations."""

    def test_quaternion_euler_round_trip(self):
        """Test quaternion -> euler -> quaternion round trip."""
        # Create a quaternion from euler angles
        roll, pitch, yaw = 0.1, 0.2, 0.3
        quat = euler_to_quaternion(roll, pitch, yaw)

        # Convert back
        r, p, y = quaternion_to_euler(quat)

        assert r == pytest.approx(roll, abs=0.001)
        assert p == pytest.approx(pitch, abs=0.001)
        assert y == pytest.approx(yaw, abs=0.001)

    def test_identity_quaternion(self):
        """Test identity quaternion (no rotation)."""
        quat = euler_to_quaternion(0, 0, 0)
        np.testing.assert_array_almost_equal(quat, [1.0, 0.0, 0.0, 0.0])


class TestInterpolation:
    """Tests for interpolation functions."""

    def test_lerp(self):
        """Test linear interpolation."""
        assert lerp(0, 10, 0.0) == 0
        assert lerp(0, 10, 1.0) == 10
        assert lerp(0, 10, 0.5) == 5

    def test_lerp_clamped(self):
        """Test lerp is clamped to [0, 1]."""
        assert lerp(0, 10, -0.5) == 0
        assert lerp(0, 10, 1.5) == 10

    def test_lerp_vector(self):
        """Test vector interpolation."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 10.0, 10.0])
        result = lerp_vector(a, b, 0.5)
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_smooth_step(self):
        """Test smooth step function."""
        assert smooth_step(0.0) == 0.0
        assert smooth_step(1.0) == 1.0
        assert smooth_step(0.5) == pytest.approx(0.5)  # S-curve passes through center


class TestBearing:
    """Tests for bearing calculations."""

    def test_bearing_north(self):
        """Test bearing to point due North."""
        bearing = calculate_bearing((0, 0), (10, 0))
        assert bearing == pytest.approx(0.0)

    def test_bearing_east(self):
        """Test bearing to point due East."""
        bearing = calculate_bearing((0, 0), (0, 10))
        assert bearing == pytest.approx(math.pi / 2)

    def test_bearing_deg(self):
        """Test bearing in degrees."""
        bearing = calculate_bearing_deg((0, 0), (0, 10))
        assert bearing == pytest.approx(90.0)
