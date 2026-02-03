"""
Tests for gate detection.
"""

import numpy as np
import pytest
import cv2

from perception.gate_detector import GateDetector, GateDetection


class TestGateDetection:
    """Tests for the GateDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a gate detector instance."""
        return GateDetector()

    @pytest.fixture
    def orange_gate_image(self):
        """Create a test image with an orange gate."""
        # Create a black 640x480 image
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw an orange gate frame
        # Orange in BGR: approximately (0, 128, 255)
        orange = (0, 140, 255)

        # Outer rectangle
        cv2.rectangle(image, (150, 100), (490, 380), orange, -1)

        # Inner rectangle (black - makes it a frame)
        cv2.rectangle(image, (200, 150), (440, 330), (0, 0, 0), -1)

        return image

    @pytest.fixture
    def empty_image(self):
        """Create an empty test image with no gate."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_detect_orange_gate(self, detector, orange_gate_image):
        """Test that detector finds an orange gate."""
        detection = detector.detect(orange_gate_image)

        assert detection.detected is True
        assert detection.confidence > 0.5
        assert detection.center_pixel is not None

        # Check center is approximately in middle of image
        cx, cy = detection.center_pixel
        assert 200 < cx < 440
        assert 100 < cy < 380

    def test_no_detection_on_empty_image(self, detector, empty_image):
        """Test that detector returns no detection for empty image."""
        detection = detector.detect(empty_image)

        assert detection.detected is False
        assert detection.center_pixel is None
        assert detection.confidence == 0.0

    def test_detection_has_distance_estimate(self, detector, orange_gate_image):
        """Test that detection includes distance estimate."""
        detection = detector.detect(orange_gate_image)

        assert detection.detected is True
        assert detection.distance_m > 0

    def test_detection_has_corners(self, detector, orange_gate_image):
        """Test that detection includes corner points."""
        detection = detector.detect(orange_gate_image)

        assert detection.detected is True
        assert detection.corners is not None
        assert len(detection.corners) == 4

    def test_get_aim_error(self, detector, orange_gate_image):
        """Test aim error calculation."""
        detection = detector.detect(orange_gate_image)

        assert detection.detected is True

        h_error, v_error = detection.get_aim_error(640, 480)

        # Errors should be normalized to [-1, 1]
        assert -1 <= h_error <= 1
        assert -1 <= v_error <= 1

    def test_detect_with_visualization(self, detector, orange_gate_image):
        """Test visualization output."""
        detection, viz_image = detector.detect_with_visualization(orange_gate_image)

        assert detection.detected is True
        assert viz_image is not None
        assert viz_image.shape == orange_gate_image.shape

        # Visualization should have some green pixels (from drawing)
        green_channel = viz_image[:, :, 1]
        assert np.max(green_channel) > 200

    def test_history_accumulation(self, detector, orange_gate_image):
        """Test that detection history accumulates."""
        # Detect multiple times
        for _ in range(5):
            detector.detect(orange_gate_image)

        # Get smoothed detection
        smoothed = detector.get_smoothed_detection()

        assert smoothed is not None
        assert smoothed.detected is True

    def test_clear_history(self, detector, orange_gate_image):
        """Test history clearing."""
        # Add some detections
        for _ in range(3):
            detector.detect(orange_gate_image)

        # Clear history
        detector.clear_history()

        # Should have no smoothed detection
        smoothed = detector.get_smoothed_detection()
        assert smoothed is None


class TestGateDetectionDataclass:
    """Tests for the GateDetection dataclass."""

    def test_default_values(self):
        """Test default values for GateDetection."""
        detection = GateDetection()

        assert detection.detected is False
        assert detection.center_pixel is None
        assert detection.corners is None
        assert detection.confidence == 0.0
        assert detection.distance_m == 0.0

    def test_aim_error_no_detection(self):
        """Test aim error returns zero when no detection."""
        detection = GateDetection()
        h_error, v_error = detection.get_aim_error(640, 480)

        assert h_error == 0.0
        assert v_error == 0.0

    def test_aim_error_centered(self):
        """Test aim error is zero when gate is centered."""
        detection = GateDetection(
            detected=True,
            center_pixel=(320, 240),  # Center of 640x480
        )
        h_error, v_error = detection.get_aim_error(640, 480)

        assert h_error == pytest.approx(0.0)
        assert v_error == pytest.approx(0.0)

    def test_aim_error_offset(self):
        """Test aim error for offset gate."""
        detection = GateDetection(
            detected=True,
            center_pixel=(480, 240),  # Right of center
        )
        h_error, v_error = detection.get_aim_error(640, 480)

        assert h_error == pytest.approx(0.5)  # 50% right
        assert v_error == pytest.approx(0.0)  # Vertically centered
