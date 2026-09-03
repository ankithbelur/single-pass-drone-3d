from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from reconstruction.quality_assessment.frame_quality import (
    FrameQualityAssessor,
)


class ConfidenceAwareFrameSelector:
    """
    Selects frames using a combination of:

    - Visual quality
    - Scene change
    - Temporal spacing

    The resulting confidence score estimates how useful
    a frame may be for downstream 3D reconstruction.
    """

    def __init__(
        self,
        min_quality_score: float = 0.55,
        min_scene_difference: float = 10.0,
        min_frame_gap: int = 5,
    ):
        self.min_quality_score = min_quality_score
        self.min_scene_difference = min_scene_difference
        self.min_frame_gap = min_frame_gap

        self.quality_assessor = FrameQualityAssessor()

    @staticmethod
    def calculate_scene_difference(
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
    ) -> float:
        """
        Calculate average visual difference between two frames.
        """

        previous_gray = cv2.cvtColor(
            previous_frame,
            cv2.COLOR_BGR2GRAY,
        )

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY,
        )

        difference = cv2.absdiff(previous_gray, current_gray)

        return float(np.mean(difference))

    def calculate_confidence(
        self,
        quality_score: float,
        scene_difference: float,
    ) -> float:
        """
        Combine frame quality and scene information.

        Scene difference is normalized and capped so that
        extremely large differences do not dominate.
        """

        scene_score = min(
            scene_difference / 30.0,
            1.0,
        )

        confidence = (
            0.70 * quality_score
            + 0.30 * scene_score
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def select_frames(
        self,
        sequence_path: Path,
    ) -> List[Dict]:
        """
        Select useful frames from an image sequence.
        """

        frame_paths = sorted(sequence_path.glob("*.jpg"))

        if not frame_paths:
            raise ValueError(
                f"No JPG frames found in {sequence_path}"
            )

        selected_frames = []

        previous_frame = None
        last_selected_index = -self.min_frame_gap

        for index, frame_path in enumerate(frame_paths):

            frame = cv2.imread(str(frame_path))

            if frame is None:
                continue

            quality_result = self.quality_assessor.assess_frame(
                frame
            )

            quality_score = quality_result[
                "overall_quality_score"
            ]

            if previous_frame is None:
                scene_difference = 0.0
            else:
                scene_difference = (
                    self.calculate_scene_difference(
                        previous_frame,
                        frame,
                    )
                )

            confidence = self.calculate_confidence(
                quality_score,
                scene_difference,
            )

            meets_quality = (
                quality_score >= self.min_quality_score
            )

            meets_scene_change = (
                scene_difference
                >= self.min_scene_difference
            )

            meets_spacing = (
                index - last_selected_index
                >= self.min_frame_gap
            )

            # Always retain the first frame if its quality
            # is acceptable.
            should_select = (
                (index == 0 and meets_quality)
                or (
                    meets_quality
                    and meets_scene_change
                    and meets_spacing
                )
            )

            if should_select:
                selected_frames.append(
                    {
                        "frame_name": frame_path.name,
                        "frame_index": index,
                        "quality_score": quality_score,
                        "scene_difference": round(
                            scene_difference,
                            2,
                        ),
                        "confidence_score": round(
                            confidence,
                            4,
                        ),
                    }
                )

                last_selected_index = index

            previous_frame = frame

        return selected_frames
    