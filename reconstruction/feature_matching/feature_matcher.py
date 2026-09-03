from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np


class FeatureMatcher:
    """
    Detects and matches ORB features between two frames.

    Pipeline:
    1. Detect keypoints
    2. Compute descriptors
    3. Match descriptors
    4. Filter matches
    """

    def __init__(
        self,
        max_features: int = 3000,
        match_ratio: float = 0.75,
    ):
        self.max_features = max_features
        self.match_ratio = match_ratio

        self.detector = cv2.ORB_create(
            nfeatures=self.max_features
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

    @staticmethod
    def load_image(
        image_or_path: Union[np.ndarray, str, Path],
    ) -> np.ndarray:
        """Load an image if a path is provided."""

        if isinstance(image_or_path, (str, Path)):
            image = cv2.imread(str(image_or_path))

            if image is None:
                raise ValueError(
                    f"Could not load image: {image_or_path}"
                )

            return image

        if image_or_path is None or image_or_path.size == 0:
            raise ValueError("Invalid image provided.")

        return image_or_path

    def detect_and_describe(
        self,
        image_or_path: Union[np.ndarray, str, Path],
    ):
        """Detect ORB keypoints and compute descriptors."""

        image = self.load_image(image_or_path)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        keypoints, descriptors = (
            self.detector.detectAndCompute(
                gray,
                None,
            )
        )

        return keypoints, descriptors

    def match_images(
        self,
        image1: Union[np.ndarray, str, Path],
        image2: Union[np.ndarray, str, Path],
    ) -> Dict:
        """Match features between two images using Lowe-style ratio filtering."""

        keypoints1, descriptors1 = (
            self.detect_and_describe(image1)
        )

        keypoints2, descriptors2 = (
            self.detect_and_describe(image2)
        )

        if descriptors1 is None or descriptors2 is None:
            return {
                "keypoints1": keypoints1,
                "keypoints2": keypoints2,
                "matches": [],
                "raw_match_count": 0,
                "good_match_count": 0,
            }

        raw_matches = self.matcher.knnMatch(
            descriptors1,
            descriptors2,
            k=2,
        )

        good_matches = []

        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue

            best_match, second_match = match_pair

            if (
                best_match.distance
                < self.match_ratio * second_match.distance
            ):
                good_matches.append(best_match)

        good_matches = sorted(
            good_matches,
            key=lambda match: match.distance,
        )

        return {
            "keypoints1": keypoints1,
            "keypoints2": keypoints2,
            "matches": good_matches,
            "raw_match_count": len(raw_matches),
            "good_match_count": len(good_matches),
        }