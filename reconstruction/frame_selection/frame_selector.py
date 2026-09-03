from pathlib import Path

import cv2
import numpy as np


class FrameSelector:
    """Selects high-quality and sufficiently different frames."""

    def __init__(
        self,
        min_sharpness: float = 100.0,
        min_difference: float = 10.0,
        frame_interval: int = 10,
    ):
        self.min_sharpness = min_sharpness
        self.min_difference = min_difference
        self.frame_interval = frame_interval

    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """Calculate sharpness using variance of the Laplacian."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

    @staticmethod
    def calculate_difference(
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> float:
        """Calculate average visual difference between two frames."""

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        difference = cv2.absdiff(gray1, gray2)

        return float(np.mean(difference))

    def select_frames(
        self,
        frame_paths: list[Path],
    ) -> list[Path]:
        """Select useful frames from an ordered sequence."""

        selected_frames = []
        previous_selected_frame = None

        for index, frame_path in enumerate(frame_paths):

            # Apply temporal spacing
            if index % self.frame_interval != 0:
                continue

            frame = cv2.imread(str(frame_path))

            if frame is None:
                continue

            sharpness = self.calculate_sharpness(frame)

            # Reject blurry frames
            if sharpness < self.min_sharpness:
                continue

            # Always accept the first valid frame
            if previous_selected_frame is None:
                selected_frames.append(frame_path)
                previous_selected_frame = frame
                continue

            difference = self.calculate_difference(
                previous_selected_frame,
                frame,
            )

            # Accept frames with sufficient viewpoint/content change
            if difference >= self.min_difference:
                selected_frames.append(frame_path)
                previous_selected_frame = frame

        return selected_frames 
    