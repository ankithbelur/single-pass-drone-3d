from typing import Dict

import cv2
import numpy as np


class PointTriangulator:
    """
    Triangulates matched 2D feature points from two camera views
    into 3D points.
    """

    def __init__(self, camera_matrix: np.ndarray):
        self.camera_matrix = np.asarray(
            camera_matrix,
            dtype=np.float64,
        )

    def create_projection_matrices(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
    ):
        """
        Create projection matrices for the two cameras.

        Camera 1:
            P1 = K [I | 0]

        Camera 2:
            P2 = K [R | t]
        """

        identity = np.eye(3, dtype=np.float64)

        zero_translation = np.zeros(
            (3, 1),
            dtype=np.float64,
        )

        extrinsic1 = np.hstack(
            (identity, zero_translation)
        )

        extrinsic2 = np.hstack(
            (rotation, translation)
        )

        projection1 = self.camera_matrix @ extrinsic1
        projection2 = self.camera_matrix @ extrinsic2

        return projection1, projection2

    def triangulate_points(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> Dict:
        """
        Triangulate corresponding points from two images.

        Returns:
            3D points and projection matrices.
        """

        if len(points1) == 0 or len(points2) == 0:
            raise ValueError(
                "Point arrays cannot be empty."
            )

        if len(points1) != len(points2):
            raise ValueError(
                "points1 and points2 must have "
                "the same number of points."
            )

        projection1, projection2 = (
            self.create_projection_matrices(
                rotation,
                translation,
            )
        )

        points1 = np.asarray(
            points1,
            dtype=np.float64,
        ).T

        points2 = np.asarray(
            points2,
            dtype=np.float64,
        ).T

        homogeneous_points = cv2.triangulatePoints(
            projection1,
            projection2,
            points1,
            points2,
        )

        points_3d = (
            homogeneous_points[:3]
            / homogeneous_points[3]
        ).T

        valid_mask = np.isfinite(
            points_3d
        ).all(axis=1)

        points_3d = points_3d[valid_mask]

        return {
            "points_3d": points_3d,
            "projection1": projection1,
            "projection2": projection2,
            "valid_point_count": len(points_3d),
        }