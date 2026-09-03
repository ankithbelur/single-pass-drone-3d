from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np


class FrameQualityAssessor:
    """
    Evaluates the visual quality of drone video frames.

    Metrics:
    - Sharpness
    - Brightness
    - Contrast
    - Overall quality score
    """

    def __init__(
        self,
        target_brightness: float = 128.0,
        max_brightness_distance: float = 128.0,
    ):
        self.target_brightness = target_brightness
        self.max_brightness_distance = max_brightness_distance

    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """Calculate sharpness using variance of the Laplacian."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def calculate_brightness(frame: np.ndarray) -> float:
        """Calculate average brightness."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def calculate_contrast(frame: np.ndarray) -> float:
        """Calculate contrast using grayscale standard deviation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def brightness_score(self, brightness: float) -> float:
        """
        Score brightness from 0 to 1.
        Values closer to target brightness receive higher scores.
        """
        distance = abs(brightness - self.target_brightness)
        score = 1.0 - (distance / self.max_brightness_distance)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def sharpness_score(sharpness: float) -> float:
        """
        Normalize sharpness into a 0 to 1 range.

        The scale is intentionally capped to reduce the effect
        of extremely sharp frames.
        """
        return float(np.clip(sharpness / 500.0, 0.0, 1.0))

    @staticmethod
    def contrast_score(contrast: float) -> float:
        """Normalize contrast into a 0 to 1 range."""
        return float(np.clip(contrast / 64.0, 0.0, 1.0))

    def assess_frame(
        self,
        frame_or_path: Union[np.ndarray, str, Path],
    ) -> Dict[str, float]:
        """
        Assess a frame and return individual metrics and
        an overall quality score.
        """

        if isinstance(frame_or_path, (str, Path)):
            frame = cv2.imread(str(frame_or_path))

            if frame is None:
                raise ValueError(
                    f"Could not load frame: {frame_or_path}"
                )
        else:
            frame = frame_or_path

        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided.")

        sharpness = self.calculate_sharpness(frame)
        brightness = self.calculate_brightness(frame)
        contrast = self.calculate_contrast(frame)

        sharpness_normalized = self.sharpness_score(sharpness)
        brightness_normalized = self.brightness_score(brightness)
        contrast_normalized = self.contrast_score(contrast)

        overall_score = (
            0.50 * sharpness_normalized
            + 0.25 * brightness_normalized
            + 0.25 * contrast_normalized
        )

        return {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "sharpness_score": round(sharpness_normalized, 4),
            "brightness_score": round(brightness_normalized, 4),
            "contrast_score": round(contrast_normalized, 4),
            "overall_quality_score": round(overall_score, 4),
        }