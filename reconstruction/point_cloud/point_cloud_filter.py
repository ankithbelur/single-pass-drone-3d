import numpy as np


class PointCloudFilter:
    """
    Filters noisy and invalid points from a 3D point cloud.
    """

    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 200.0,
        outlier_std_ratio: float = 2.5,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.outlier_std_ratio = outlier_std_ratio

    def filter_points(
        self,
        points_3d: np.ndarray,
    ) -> dict:
        """
        Filter points based on depth and statistical distance.
        """

        if points_3d is None or len(points_3d) == 0:
            raise ValueError("Point cloud is empty.")

        points_3d = np.asarray(points_3d)

        # Remove invalid numerical values
        valid_mask = np.isfinite(points_3d).all(axis=1)

        valid_points = points_3d[valid_mask]

        # Depth filtering
        depth_mask = (
            (valid_points[:, 2] > self.min_depth)
            & (valid_points[:, 2] < self.max_depth)
        )

        depth_filtered = valid_points[depth_mask]

        if len(depth_filtered) == 0:
            return {
                "points_3d": np.empty((0, 3)),
                "input_point_count": len(points_3d),
                "valid_point_count": 0,
                "removed_point_count": len(points_3d),
            }

        # Statistical outlier filtering
        centroid = np.mean(depth_filtered, axis=0)

        distances = np.linalg.norm(
            depth_filtered - centroid,
            axis=1,
        )

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        threshold = (
            mean_distance
            + self.outlier_std_ratio * std_distance
        )

        outlier_mask = distances <= threshold

        filtered_points = depth_filtered[outlier_mask]

        return {
            "points_3d": filtered_points,
            "input_point_count": len(points_3d),
            "valid_point_count": len(filtered_points),
            "removed_point_count": (
                len(points_3d) - len(filtered_points)
            ),
        }