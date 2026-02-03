"""
Debug visualization utilities for drone racing.

Provides functions for visualizing:
- Detected gates and obstacles
- Flight paths and trajectories
- State estimation data
- Control commands

Note: These are debugging tools, not meant for real-time display
during competition flights.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from utils.logger import get_logger

logger = get_logger(__name__)


def draw_gate_detection(
    frame: NDArray[np.uint8],
    gate_center: Optional[Tuple[int, int]] = None,
    gate_corners: Optional[List[Tuple[int, int]]] = None,
    confidence: float = 0.0,
    distance: float = 0.0,
) -> NDArray[np.uint8]:
    """
    Draw gate detection visualization on a camera frame.

    Args:
        frame: Input image (BGR format)
        gate_center: Center point of detected gate (x, y) in pixels
        gate_corners: List of corner points [(x,y), ...] if available
        confidence: Detection confidence 0-1
        distance: Estimated distance to gate in meters

    Returns:
        Annotated image with gate visualization
    """
    output = frame.copy()

    if gate_center is None:
        # No detection - show "NO GATE" indicator
        cv2.putText(
            output, "NO GATE DETECTED",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )
        return output

    # Draw center crosshair
    cx, cy = gate_center
    color = _confidence_to_color(confidence)

    # Crosshair
    cv2.drawMarker(
        output, (cx, cy),
        color, cv2.MARKER_CROSS, 30, 2
    )

    # Draw corners if available
    if gate_corners and len(gate_corners) >= 4:
        corners = np.array(gate_corners, dtype=np.int32)
        cv2.polylines(output, [corners], True, color, 2)

    # Draw info text
    info_text = f"Conf: {confidence:.2f} | Dist: {distance:.1f}m"
    cv2.putText(
        output, info_text,
        (cx - 80, cy - 40), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2
    )

    # Draw aim point (center of frame for reference)
    h, w = frame.shape[:2]
    cv2.circle(output, (w // 2, h // 2), 5, (255, 255, 255), 1)

    return output


def draw_trajectory(
    frame: NDArray[np.uint8],
    trajectory_points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """
    Draw a trajectory path on a frame.

    Args:
        frame: Input image
        trajectory_points: List of (x, y) points in pixel coordinates
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with trajectory drawn
    """
    output = frame.copy()

    if len(trajectory_points) < 2:
        return output

    points = np.array(trajectory_points, dtype=np.int32)
    cv2.polylines(output, [points], False, color, thickness)

    # Draw points
    for i, point in enumerate(trajectory_points):
        # Fade color based on position (older = dimmer)
        alpha = (i + 1) / len(trajectory_points)
        point_color = tuple(int(c * alpha) for c in color)
        cv2.circle(output, point, 3, point_color, -1)

    return output


def draw_velocity_vector(
    frame: NDArray[np.uint8],
    origin: Tuple[int, int],
    velocity: Tuple[float, float],
    scale: float = 10.0,
    color: Tuple[int, int, int] = (255, 255, 0),
) -> NDArray[np.uint8]:
    """
    Draw a velocity vector arrow on the frame.

    Args:
        frame: Input image
        origin: Starting point (x, y) in pixels
        velocity: Velocity vector (vx, vy) in m/s
        scale: Pixels per m/s for visualization
        color: BGR color tuple

    Returns:
        Image with velocity arrow drawn
    """
    output = frame.copy()

    vx, vy = velocity
    end_x = int(origin[0] + vx * scale)
    end_y = int(origin[1] + vy * scale)

    cv2.arrowedLine(output, origin, (end_x, end_y), color, 2, tipLength=0.3)

    return output


def create_hud_overlay(
    frame: NDArray[np.uint8],
    altitude: float,
    velocity: float,
    heading: float,
    battery: float,
    mode: str,
) -> NDArray[np.uint8]:
    """
    Create a heads-up display overlay for flight data.

    Args:
        frame: Input image
        altitude: Current altitude in meters
        velocity: Current speed in m/s
        heading: Current heading in degrees
        battery: Battery percentage
        mode: Current flight mode string

    Returns:
        Image with HUD overlay
    """
    output = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent overlay background
    overlay = output.copy()
    cv2.rectangle(overlay, (5, 5), (200, 130), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

    # Text color
    text_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    # Draw telemetry data
    y_offset = 25
    line_height = 22

    cv2.putText(output, f"ALT: {altitude:.1f} m", (10, y_offset),
                font, font_scale, text_color, 1)
    y_offset += line_height

    cv2.putText(output, f"VEL: {velocity:.1f} m/s", (10, y_offset),
                font, font_scale, text_color, 1)
    y_offset += line_height

    cv2.putText(output, f"HDG: {heading:.0f} deg", (10, y_offset),
                font, font_scale, text_color, 1)
    y_offset += line_height

    # Battery with color coding
    batt_color = (0, 255, 0) if battery > 30 else (0, 165, 255) if battery > 15 else (0, 0, 255)
    cv2.putText(output, f"BAT: {battery:.0f}%", (10, y_offset),
                font, font_scale, batt_color, 1)
    y_offset += line_height

    cv2.putText(output, f"MODE: {mode}", (10, y_offset),
                font, font_scale, text_color, 1)

    return output


def plot_flight_path_2d(
    positions: List[Tuple[float, float, float]],
    waypoints: Optional[List[Tuple[float, float, float]]] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Create a 2D plot of the flight path.

    Args:
        positions: List of (N, E, D) positions from flight
        waypoints: Optional list of target waypoints
        output_path: Path to save figure (displays if None)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot flight path
    if positions:
        n_coords = [p[0] for p in positions]
        e_coords = [p[1] for p in positions]
        ax.plot(e_coords, n_coords, 'b-', linewidth=1, label='Flight Path')
        ax.scatter(e_coords[0], n_coords[0], c='g', s=100, marker='o', label='Start')
        ax.scatter(e_coords[-1], n_coords[-1], c='r', s=100, marker='x', label='End')

    # Plot waypoints
    if waypoints:
        wp_n = [w[0] for w in waypoints]
        wp_e = [w[1] for w in waypoints]
        ax.scatter(wp_e, wp_n, c='orange', s=150, marker='s', label='Waypoints')
        for i, (n, e, _) in enumerate(waypoints):
            ax.annotate(f'WP{i}', (e, n), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Flight Path (Top-Down View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info("Flight path plot saved", path=output_path)
    else:
        plt.show()

    plt.close()


def _confidence_to_color(confidence: float) -> Tuple[int, int, int]:
    """Convert confidence value (0-1) to BGR color (red->yellow->green)."""
    if confidence < 0.5:
        # Red to yellow
        r = 255
        g = int(255 * (confidence / 0.5))
        b = 0
    else:
        # Yellow to green
        r = int(255 * ((1 - confidence) / 0.5))
        g = 255
        b = 0

    return (b, g, r)  # BGR format for OpenCV
