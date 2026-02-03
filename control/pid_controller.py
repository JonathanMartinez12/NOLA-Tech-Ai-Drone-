"""
PID controller implementations for drone control.

Provides tunable PID controllers for:
- Position control (position -> velocity)
- Velocity control (velocity -> acceleration/thrust)
- Attitude control (angle -> rate)

Each controller includes anti-windup and output limiting.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from utils.math_helpers import clamp
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PIDController:
    """
    Generic PID controller with anti-windup.

    Implements a standard PID controller with:
    - Proportional, Integral, and Derivative terms
    - Integral windup prevention
    - Output saturation
    - Derivative on measurement (reduces derivative kick)

    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        output_min: Minimum output value
        output_max: Maximum output value
        integral_max: Maximum integral accumulator (anti-windup)

    Example:
        pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        output = pid.update(setpoint=10.0, measurement=8.0, dt=0.02)
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0

    output_min: float = -float("inf")
    output_max: float = float("inf")
    integral_max: float = 10.0

    # Internal state
    _integral: float = field(default=0.0, repr=False)
    _last_error: Optional[float] = field(default=None, repr=False)
    _last_measurement: Optional[float] = field(default=None, repr=False)
    _last_time: Optional[float] = field(default=None, repr=False)

    def update(
        self,
        setpoint: float,
        measurement: float,
        dt: Optional[float] = None,
    ) -> float:
        """
        Calculate PID output for current error.

        Args:
            setpoint: Desired value
            measurement: Current measured value
            dt: Time step (seconds). If None, uses wall clock.

        Returns:
            Controller output (clamped to output limits)
        """
        # Calculate time step if not provided
        current_time = time.time()
        if dt is None:
            if self._last_time is None:
                dt = 0.02  # Default 50Hz
            else:
                dt = current_time - self._last_time
        self._last_time = current_time

        # Avoid division by zero
        if dt <= 0:
            dt = 0.001

        # Calculate error
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        self._integral = clamp(self._integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self._integral

        # Derivative term (on measurement to avoid derivative kick)
        if self._last_measurement is not None:
            # Derivative of measurement (negative because d(error)/dt = -d(meas)/dt when setpoint constant)
            d_measurement = (measurement - self._last_measurement) / dt
            d_term = -self.kd * d_measurement
        else:
            d_term = 0.0

        self._last_measurement = measurement
        self._last_error = error

        # Calculate total output
        output = p_term + i_term + d_term

        # Clamp output
        output = clamp(output, self.output_min, self.output_max)

        return output

    def reset(self) -> None:
        """Reset controller state (integral and derivative history)."""
        self._integral = 0.0
        self._last_error = None
        self._last_measurement = None
        self._last_time = None

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """Update PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd


@dataclass
class PIDController3D:
    """
    3D PID controller for position/velocity control in 3D space.

    Wraps three PID controllers for X, Y, Z axes with coordinated
    output limiting.

    Example:
        pid_3d = PIDController3D(
            kp=(1.0, 1.0, 0.5),  # Different gains per axis
            ki=(0.1, 0.1, 0.05),
            kd=(0.05, 0.05, 0.02),
        )
        vel_cmd = pid_3d.update(target_pos, current_pos, dt)
    """

    # Gains per axis (x, y, z)
    kp: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ki: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    kd: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    output_max: Tuple[float, float, float] = (5.0, 5.0, 2.0)

    # Individual axis controllers
    _pid_x: PIDController = field(init=False)
    _pid_y: PIDController = field(init=False)
    _pid_z: PIDController = field(init=False)

    def __post_init__(self):
        self._pid_x = PIDController(
            kp=self.kp[0], ki=self.ki[0], kd=self.kd[0],
            output_min=-self.output_max[0], output_max=self.output_max[0]
        )
        self._pid_y = PIDController(
            kp=self.kp[1], ki=self.ki[1], kd=self.kd[1],
            output_min=-self.output_max[1], output_max=self.output_max[1]
        )
        self._pid_z = PIDController(
            kp=self.kp[2], ki=self.ki[2], kd=self.kd[2],
            output_min=-self.output_max[2], output_max=self.output_max[2]
        )

    def update(
        self,
        setpoint: Tuple[float, float, float],
        measurement: Tuple[float, float, float],
        dt: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate 3D PID output.

        Args:
            setpoint: Target position/velocity (x, y, z)
            measurement: Current position/velocity (x, y, z)
            dt: Time step in seconds

        Returns:
            Output tuple (x, y, z)
        """
        out_x = self._pid_x.update(setpoint[0], measurement[0], dt)
        out_y = self._pid_y.update(setpoint[1], measurement[1], dt)
        out_z = self._pid_z.update(setpoint[2], measurement[2], dt)

        return (out_x, out_y, out_z)

    def reset(self) -> None:
        """Reset all axis controllers."""
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_z.reset()


@dataclass
class PositionController:
    """
    Position controller that outputs velocity commands.

    Takes position error and outputs velocity setpoints.
    Includes feed-forward for known velocity targets.

    Designed for drone waypoint following.
    """

    # Position gains
    kp_horizontal: float = 1.0
    kp_vertical: float = 0.5

    # Velocity limits
    max_horizontal_velocity: float = 5.0
    max_vertical_velocity: float = 2.0

    # Integral term for steady-state error
    ki_horizontal: float = 0.05
    ki_vertical: float = 0.02

    _pid_n: PIDController = field(init=False)
    _pid_e: PIDController = field(init=False)
    _pid_d: PIDController = field(init=False)

    def __post_init__(self):
        self._pid_n = PIDController(
            kp=self.kp_horizontal,
            ki=self.ki_horizontal,
            kd=0.0,
            output_min=-self.max_horizontal_velocity,
            output_max=self.max_horizontal_velocity,
        )
        self._pid_e = PIDController(
            kp=self.kp_horizontal,
            ki=self.ki_horizontal,
            kd=0.0,
            output_min=-self.max_horizontal_velocity,
            output_max=self.max_horizontal_velocity,
        )
        self._pid_d = PIDController(
            kp=self.kp_vertical,
            ki=self.ki_vertical,
            kd=0.0,
            output_min=-self.max_vertical_velocity,
            output_max=self.max_vertical_velocity,
        )

    def update(
        self,
        target_pos: Tuple[float, float, float],
        current_pos: Tuple[float, float, float],
        dt: float,
        feed_forward: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate velocity command from position error.

        Args:
            target_pos: Target position (N, E, D)
            current_pos: Current position (N, E, D)
            dt: Time step
            feed_forward: Optional velocity feed-forward (N, E, D)

        Returns:
            Velocity command (N, E, D) in m/s
        """
        vel_n = self._pid_n.update(target_pos[0], current_pos[0], dt)
        vel_e = self._pid_e.update(target_pos[1], current_pos[1], dt)
        vel_d = self._pid_d.update(target_pos[2], current_pos[2], dt)

        # Add feed-forward if provided
        if feed_forward is not None:
            vel_n += feed_forward[0]
            vel_e += feed_forward[1]
            vel_d += feed_forward[2]

            # Re-clamp after feed-forward
            vel_n = clamp(vel_n, -self.max_horizontal_velocity, self.max_horizontal_velocity)
            vel_e = clamp(vel_e, -self.max_horizontal_velocity, self.max_horizontal_velocity)
            vel_d = clamp(vel_d, -self.max_vertical_velocity, self.max_vertical_velocity)

        return (vel_n, vel_e, vel_d)

    def reset(self) -> None:
        """Reset controller state."""
        self._pid_n.reset()
        self._pid_e.reset()
        self._pid_d.reset()
