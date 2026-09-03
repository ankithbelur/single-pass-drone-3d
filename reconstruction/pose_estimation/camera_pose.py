from typing import Dict

import cv2
import numpy as np


class CameraPoseEstimator:
    """
    Estimates relative camera pose between two images.
    """

    def __init__(
        self,
        focal_length: float,
        principal_point: tuple,
    ):
        self.camera_matrix = np.array(
            [
                [focal_length, 0, principal_point[0]],
                [0, focal_length, principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def estimate_pose(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
    ) -> Dict:

        if len(points1) < 8:
            raise ValueError(
                "At least 8 matched points are required."
            )

        essential_matrix, essential_mask = cv2.findEssentialMat(
            points1,
            points2,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=2.0,
        )

        if essential_matrix is None:
            raise RuntimeError(
                "Essential Matrix estimation failed."
            )

        inlier_count, rotation, translation, pose_mask = (
            cv2.recoverPose(
                essential_matrix,
                points1,
                points2,
                self.camera_matrix,
            )
        )

        essential_inliers = int(
            np.count_nonzero(essential_mask)
        )

        return {
            "essential_matrix": essential_matrix,
            "rotation": rotation,
            "translation": translation,
            "essential_mask": essential_mask,
            "pose_mask": pose_mask,
            "essential_inliers": essential_inliers,
            "pose_inliers": int(inlier_count),
        }