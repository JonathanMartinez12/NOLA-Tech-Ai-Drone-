"""
Mathematical helper functions for drone racing.

Provides common mathematical operations used throughout the system:
- Angle normalization and difference calculations
- Vector operations (magnitude, normalization)
- Coordinate frame transformations (NED <-> ENU)
- Quaternion operations
- Clamping and interpolation

All functions use numpy arrays for vector operations.
"""

import math
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray


# Type aliases for clarity
Vector3 = NDArray[np.float64]  # 3D vector [x, y, z]
Quaternion = NDArray[np.float64]  # [w, x, y, z]


def normalize_angle(angle_rad: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].

    Args:
        angle_rad: Angle in radians

    Returns:
        Normalized angle in radians within [-pi, pi]

    Example:
        >>> normalize_angle(3 * math.pi)  # 540 degrees
        -3.141592...  # -180 degrees
    """
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad


def normalize_angle_deg(angle_deg: float) -> float:
    """
    Normalize an angle to the range [-180, 180] degrees.

    Args:
        angle_deg: Angle in degrees

    Returns:
        Normalized angle in degrees within [-180, 180]
    """
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


def angle_difference(angle1_rad: float, angle2_rad: float) -> float:
    """
    Calculate the shortest angular difference between two angles.

    The result is the angle you need to rotate from angle1 to reach angle2,
    taking the shortest path (which might be negative).

    Args:
        angle1_rad: First angle in radians
        angle2_rad: Second angle in radians

    Returns:
        Shortest angular difference in radians, range [-pi, pi]
        Positive = counterclockwise, Negative = clockwise

    Example:
        >>> angle_difference(0, math.pi/2)  # 0 to 90 degrees
        1.5707...  # +90 degrees (turn left)
        >>> angle_difference(0, -math.pi/2)  # 0 to -90 degrees
        -1.5707...  # -90 degrees (turn right)
    """
    diff = angle2_rad - angle1_rad
    return normalize_angle(diff)


def angle_difference_deg(angle1_deg: float, angle2_deg: float) -> float:
    """
    Calculate the shortest angular difference in degrees.

    Args:
        angle1_deg: First angle in degrees
        angle2_deg: Second angle in degrees

    Returns:
        Shortest angular difference in degrees, range [-180, 180]
    """
    diff = angle2_deg - angle1_deg
    return normalize_angle_deg(diff)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value to a specified range.

    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Value clamped to [min_value, max_value]

    Example:
        >>> clamp(15, 0, 10)
        10
        >>> clamp(-5, 0, 10)
        0
    """
    return max(min_value, min(value, max_value))


def clamp_vector(
    vector: Vector3,
    max_magnitude: float
) -> Vector3:
    """
    Clamp a vector's magnitude while preserving direction.

    Args:
        vector: Input vector
        max_magnitude: Maximum allowed magnitude

    Returns:
        Vector with magnitude <= max_magnitude, same direction
    """
    mag = vector_magnitude(vector)
    if mag > max_magnitude and mag > 0:
        return vector * (max_magnitude / mag)
    return vector


def vector_magnitude(vector: Union[Vector3, Tuple[float, ...]]) -> float:
    """
    Calculate the magnitude (length) of a vector.

    Args:
        vector: Input vector (any dimension)

    Returns:
        Magnitude of the vector

    Example:
        >>> vector_magnitude(np.array([3.0, 4.0, 0.0]))
        5.0
    """
    return float(np.linalg.norm(vector))


def vector_normalize(vector: Vector3) -> Vector3:
    """
    Normalize a vector to unit length.

    Args:
        vector: Input vector

    Returns:
        Unit vector in same direction, or zero vector if input is zero

    Example:
        >>> vector_normalize(np.array([3.0, 4.0, 0.0]))
        array([0.6, 0.8, 0.0])
    """
    mag = vector_magnitude(vector)
    if mag < 1e-10:  # Avoid division by zero
        return np.zeros_like(vector)
    return vector / mag


def distance_3d(
    point1: Union[Vector3, Tuple[float, float, float]],
    point2: Union[Vector3, Tuple[float, float, float]]
) -> float:
    """
    Calculate 3D Euclidean distance between two points.

    Args:
        point1: First point (x, y, z)
        point2: Second point (x, y, z)

    Returns:
        Distance between points

    Example:
        >>> distance_3d((0, 0, 0), (3, 4, 0))
        5.0
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return float(np.linalg.norm(p2 - p1))


def distance_2d(
    point1: Union[Tuple[float, float], Vector3],
    point2: Union[Tuple[float, float], Vector3]
) -> float:
    """
    Calculate 2D Euclidean distance (ignoring z/altitude).

    Args:
        point1: First point (x, y) or (x, y, z)
        point2: Second point (x, y) or (x, y, z)

    Returns:
        2D distance between points
    """
    p1 = np.array(point1)[:2]
    p2 = np.array(point2)[:2]
    return float(np.linalg.norm(p2 - p1))


# =============================================================================
# Coordinate Frame Transformations
# =============================================================================
# NED (North-East-Down): Used by PX4 and most flight controllers
# ENU (East-North-Up): Used by ROS and some robotics systems
# Body Frame: Relative to drone (forward, right, down)


def ned_to_enu(ned: Vector3) -> Vector3:
    """
    Convert coordinates from NED to ENU frame.

    NED (North-East-Down) -> ENU (East-North-Up)
    - North -> Y (becomes North in ENU)
    - East -> X (becomes East in ENU)
    - Down -> -Z (becomes Up in ENU)

    Args:
        ned: Position or velocity in NED frame [N, E, D]

    Returns:
        Position or velocity in ENU frame [E, N, U]
    """
    return np.array([ned[1], ned[0], -ned[2]])


def enu_to_ned(enu: Vector3) -> Vector3:
    """
    Convert coordinates from ENU to NED frame.

    ENU (East-North-Up) -> NED (North-East-Down)
    - East -> Y (becomes East in NED)
    - North -> X (becomes North in NED)
    - Up -> -Z (becomes Down in NED)

    Args:
        enu: Position or velocity in ENU frame [E, N, U]

    Returns:
        Position or velocity in NED frame [N, E, D]
    """
    return np.array([enu[1], enu[0], -enu[2]])


def body_to_ned(
    body_vector: Vector3,
    yaw_rad: float
) -> Vector3:
    """
    Transform a vector from body frame to NED frame.

    Body frame: X=forward, Y=right, Z=down
    Assumes roll and pitch are zero (horizontal flight).

    Args:
        body_vector: Vector in body frame [forward, right, down]
        yaw_rad: Current yaw angle in radians (0 = North)

    Returns:
        Vector in NED frame [N, E, D]
    """
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # Rotation matrix for yaw only (simplified for horizontal flight)
    north = body_vector[0] * cos_yaw - body_vector[1] * sin_yaw
    east = body_vector[0] * sin_yaw + body_vector[1] * cos_yaw
    down = body_vector[2]

    return np.array([north, east, down])


def ned_to_body(
    ned_vector: Vector3,
    yaw_rad: float
) -> Vector3:
    """
    Transform a vector from NED frame to body frame.

    Args:
        ned_vector: Vector in NED frame [N, E, D]
        yaw_rad: Current yaw angle in radians (0 = North)

    Returns:
        Vector in body frame [forward, right, down]
    """
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # Inverse rotation (transpose of yaw rotation matrix)
    forward = ned_vector[0] * cos_yaw + ned_vector[1] * sin_yaw
    right = -ned_vector[0] * sin_yaw + ned_vector[1] * cos_yaw
    down = ned_vector[2]

    return np.array([forward, right, down])


# =============================================================================
# Quaternion Operations
# =============================================================================


def quaternion_to_euler(quat: Quaternion) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def euler_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float
) -> Quaternion:
    """
    Convert Euler angles to quaternion.

    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians

    Returns:
        Quaternion [w, x, y, z]
    """
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


# =============================================================================
# Interpolation
# =============================================================================


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.

    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0 = a, 1 = b)

    Returns:
        Interpolated value
    """
    return a + (b - a) * clamp(t, 0.0, 1.0)


def lerp_vector(a: Vector3, b: Vector3, t: float) -> Vector3:
    """
    Linear interpolation between two vectors.

    Args:
        a: Start vector
        b: End vector
        t: Interpolation factor (0 = a, 1 = b)

    Returns:
        Interpolated vector
    """
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t


def smooth_step(t: float) -> float:
    """
    Smooth step function for easing (S-curve).

    Provides smooth acceleration and deceleration.

    Args:
        t: Input value in range [0, 1]

    Returns:
        Smoothed value in range [0, 1]
    """
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# =============================================================================
# Bearing and Heading
# =============================================================================


def calculate_bearing(
    from_pos: Union[Tuple[float, float], Vector3],
    to_pos: Union[Tuple[float, float], Vector3]
) -> float:
    """
    Calculate bearing from one position to another.

    Args:
        from_pos: Starting position (N, E) or (N, E, D)
        to_pos: Target position (N, E) or (N, E, D)

    Returns:
        Bearing in radians, 0 = North, positive = clockwise
    """
    dn = to_pos[0] - from_pos[0]  # North difference
    de = to_pos[1] - from_pos[1]  # East difference

    return math.atan2(de, dn)


def calculate_bearing_deg(
    from_pos: Union[Tuple[float, float], Vector3],
    to_pos: Union[Tuple[float, float], Vector3]
) -> float:
    """
    Calculate bearing in degrees.

    Args:
        from_pos: Starting position (N, E) or (N, E, D)
        to_pos: Target position (N, E) or (N, E, D)

    Returns:
        Bearing in degrees, 0 = North, positive = clockwise
    """
    return math.degrees(calculate_bearing(from_pos, to_pos))
