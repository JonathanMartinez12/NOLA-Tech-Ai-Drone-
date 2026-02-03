"""
Gate detection for drone racing.

Provides gate detection using:
1. HSV color-based detection (Phase 2 - simple, fast)
2. YOLOv8 ML-based detection (Phase 3+ - robust, accurate)

The detector identifies racing gates in camera frames and estimates
their position relative to the drone.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GateDetection:
    """
    Result of gate detection in a single frame.

    Attributes:
        detected: Whether a gate was detected
        center_pixel: Gate center in image coordinates (x, y)
        corners: Four corner points in image coordinates
        confidence: Detection confidence (0-1)
        distance_m: Estimated distance to gate in meters
        bearing_rad: Bearing to gate relative to camera center
        area_pixels: Gate area in pixels (for distance estimation)
    """

    detected: bool = False
    center_pixel: Optional[Tuple[int, int]] = None
    corners: Optional[List[Tuple[int, int]]] = None
    confidence: float = 0.0
    distance_m: float = 0.0
    bearing_rad: float = 0.0
    area_pixels: int = 0

    def get_aim_error(self, image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Calculate aim error (how far gate center is from image center).

        Returns:
            (horizontal_error, vertical_error) normalized to [-1, 1]
        """
        if not self.detected or self.center_pixel is None:
            return (0.0, 0.0)

        cx, cy = self.center_pixel
        img_cx = image_width / 2
        img_cy = image_height / 2

        # Normalize to [-1, 1] range
        h_error = (cx - img_cx) / img_cx
        v_error = (cy - img_cy) / img_cy

        return (h_error, v_error)


@dataclass
class GateDetector:
    """
    Gate detector using color-based computer vision.

    Uses HSV color thresholding to detect orange racing gates.
    Works well in simulation with consistent lighting.

    Phase 2 implementation with HSV detection.
    Phase 3+ will add YOLOv8 ML-based detection.

    Example:
        detector = GateDetector()

        # Process a frame
        frame = camera.get_frame()
        detection = detector.detect(frame.image)

        if detection.detected:
            print(f"Gate at distance {detection.distance_m}m")
    """

    _detection_history: List[GateDetection] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.settings = get_settings()

    def detect(self, image: NDArray[np.uint8]) -> GateDetection:
        """
        Detect gates in an image using HSV color thresholding.

        Args:
            image: BGR image from camera

        Returns:
            GateDetection with results
        """
        if image is None:
            return GateDetection()

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Get color thresholds from settings
        lower = np.array(self.settings.perception.gate_color_lower)
        upper = np.array(self.settings.perception.gate_color_upper)

        # Create mask for gate color (orange)
        mask = cv2.inRange(hsv, lower, upper)

        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return GateDetection()

        # Find the best gate candidate (largest valid contour)
        best_detection = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.settings.perception.min_gate_area:
                continue
            if area > self.settings.perception.max_gate_area:
                continue

            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]

            if width == 0 or height == 0:
                continue

            # Check aspect ratio (gates are roughly square)
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < self.settings.perception.gate_aspect_ratio_min:
                continue
            if aspect_ratio > self.settings.perception.gate_aspect_ratio_max:
                continue

            # This is a valid gate candidate
            if area > max_area:
                max_area = area
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Get center
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = int(rect[0][0]), int(rect[0][1])

                # Estimate distance based on apparent size
                distance = self._estimate_distance(max(width, height))

                # Calculate bearing
                img_center_x = image.shape[1] / 2
                bearing = math.atan2(cx - img_center_x, self.settings.perception.camera_focal_length)

                # Calculate confidence based on area and shape regularity
                confidence = self._calculate_confidence(area, aspect_ratio, contour)

                best_detection = GateDetection(
                    detected=True,
                    center_pixel=(cx, cy),
                    corners=[(int(p[0]), int(p[1])) for p in box],
                    confidence=confidence,
                    distance_m=distance,
                    bearing_rad=bearing,
                    area_pixels=int(area),
                )

        if best_detection is None:
            return GateDetection()

        # Add to history for smoothing
        self._add_to_history(best_detection)

        return best_detection

    def detect_with_visualization(
        self,
        image: NDArray[np.uint8]
    ) -> Tuple[GateDetection, NDArray[np.uint8]]:
        """
        Detect gates and return annotated image for debugging.

        Args:
            image: BGR image from camera

        Returns:
            Tuple of (detection result, annotated image)
        """
        detection = self.detect(image)
        viz_image = image.copy()

        if detection.detected:
            # Draw gate outline
            if detection.corners:
                corners = np.array(detection.corners)
                cv2.drawContours(viz_image, [corners], 0, (0, 255, 0), 2)

            # Draw center crosshair
            if detection.center_pixel:
                cx, cy = detection.center_pixel
                cv2.drawMarker(
                    viz_image, (cx, cy),
                    (0, 255, 0), cv2.MARKER_CROSS, 20, 2
                )

            # Add text overlay
            text = f"Dist: {detection.distance_m:.1f}m  Conf: {detection.confidence:.2f}"
            cv2.putText(
                viz_image, text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                viz_image, "No gate detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        return detection, viz_image

    def get_smoothed_detection(self) -> Optional[GateDetection]:
        """
        Get detection smoothed over recent history.

        Useful for reducing noise in gate position estimates.

        Returns:
            Smoothed detection or None if no recent detections
        """
        valid_detections = [d for d in self._detection_history if d.detected]

        if not valid_detections:
            return None

        # Average center position
        avg_cx = sum(d.center_pixel[0] for d in valid_detections) / len(valid_detections)
        avg_cy = sum(d.center_pixel[1] for d in valid_detections) / len(valid_detections)

        # Average distance
        avg_dist = sum(d.distance_m for d in valid_detections) / len(valid_detections)

        # Average confidence
        avg_conf = sum(d.confidence for d in valid_detections) / len(valid_detections)

        return GateDetection(
            detected=True,
            center_pixel=(int(avg_cx), int(avg_cy)),
            confidence=avg_conf,
            distance_m=avg_dist,
        )

    def _estimate_distance(self, apparent_width_pixels: float) -> float:
        """
        Estimate distance to gate based on apparent size.

        Uses pinhole camera model: distance = (real_width * focal_length) / apparent_width

        Args:
            apparent_width_pixels: Width of gate in pixels

        Returns:
            Estimated distance in meters
        """
        if apparent_width_pixels <= 0:
            return 0.0

        real_width = self.settings.perception.gate_actual_width
        focal_length = self.settings.perception.camera_focal_length

        distance = (real_width * focal_length) / apparent_width_pixels
        return distance

    def _calculate_confidence(
        self,
        area: float,
        aspect_ratio: float,
        contour: NDArray,
    ) -> float:
        """Calculate detection confidence based on shape properties."""
        # Start with base confidence
        confidence = 0.5

        # Reward larger gates (closer = more certain)
        if area > 10000:
            confidence += 0.2
        elif area > 5000:
            confidence += 0.1

        # Reward good aspect ratio (closer to 1.0 = more gate-like)
        if 0.8 <= aspect_ratio <= 1.2:
            confidence += 0.2
        elif 0.6 <= aspect_ratio <= 1.5:
            confidence += 0.1

        # Reward convex shapes (gates should be roughly convex)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity > 0.8:
                confidence += 0.1

        return min(confidence, 1.0)

    def _add_to_history(self, detection: GateDetection) -> None:
        """Add detection to history buffer."""
        self._detection_history.append(detection)

        # Keep only recent history
        max_history = self.settings.perception.detection_history_size
        if len(self._detection_history) > max_history:
            self._detection_history = self._detection_history[-max_history:]

    def clear_history(self) -> None:
        """Clear detection history."""
        self._detection_history = []
