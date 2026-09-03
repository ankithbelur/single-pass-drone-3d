import cv2
import numpy as np


class GeometricVerifier:
    """
    Verifies feature matches using RANSAC and the Fundamental Matrix.
    """

    def __init__(
        self,
        ransac_threshold: float = 1.0,
        confidence: float = 0.99,
    ):
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence

    def verify_matches(self, keypoints1, keypoints2, matches):
        """
        Removes geometrically inconsistent matches using RANSAC.

        Returns:
            fundamental_matrix
            inlier_matches
            inlier_mask
        """

        if len(matches) < 8:
            raise ValueError(
                "At least 8 matches are required to estimate "
                "the Fundamental Matrix."
            )

        points1 = np.float32([
            keypoints1[match.queryIdx].pt
            for match in matches
        ])

        points2 = np.float32([
            keypoints2[match.trainIdx].pt
            for match in matches
        ])

        fundamental_matrix, mask = cv2.findFundamentalMat(
            points1,
            points2,
            cv2.FM_RANSAC,
            self.ransac_threshold,
            self.confidence,
        )

        if fundamental_matrix is None or mask is None:
            raise RuntimeError(
                "Fundamental Matrix estimation failed."
            )

        mask = mask.ravel().astype(bool)

        inlier_matches = [
            match
            for match, is_inlier in zip(matches, mask)
            if is_inlier
        ]

        return (
            fundamental_matrix,
            inlier_matches,
            mask,
        )