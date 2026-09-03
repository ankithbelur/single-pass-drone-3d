from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from reconstruction.feature_matching.feature_matcher import (
    FeatureMatcher,
)
from reconstruction.feature_matching.geometric_verification import (
    GeometricVerifier,
)
from reconstruction.pose_estimation.camera_pose import (
    CameraPoseEstimator,
)
from reconstruction.triangulation.point_triangulator import (
    PointTriangulator,
)
from reconstruction.point_cloud.point_cloud_filter import (
    PointCloudFilter,
)


def main():
    sequence_path = Path(
        "datasets/raw/UAV123_10fps/data_seq/UAV123_10fps/bike1"
    )

    image1 = sequence_path / "000001.jpg"
    image2 = sequence_path / "000006.jpg"

    # Load image dimensions
    img1 = cv2.imread(str(image1))

    if img1 is None:
        raise ValueError("Could not load test image.")

    height, width = img1.shape[:2]

    # Feature matching
    matcher = FeatureMatcher()

    match_results = matcher.match_images(
        image1,
        image2,
    )

    # Geometric verification
    verifier = GeometricVerifier()

    (
        _,
        inlier_matches,
        _,
    ) = verifier.verify_matches(
        match_results["keypoints1"],
        match_results["keypoints2"],
        match_results["matches"],
    )

    # Extract corresponding points
    points1 = np.float32([
        match_results["keypoints1"][match.queryIdx].pt
        for match in inlier_matches
    ])

    points2 = np.float32([
        match_results["keypoints2"][match.trainIdx].pt
        for match in inlier_matches
    ])

    # Approximate camera intrinsics
    focal_length = float(max(width, height))

    principal_point = (
        width / 2,
        height / 2,
    )

    # Camera pose estimation
    pose_estimator = CameraPoseEstimator(
        focal_length=focal_length,
        principal_point=principal_point,
    )

    pose_results = pose_estimator.estimate_pose(
        points1,
        points2,
    )

    # Triangulation
    triangulator = PointTriangulator(
        pose_estimator.camera_matrix
    )

    triangulation_results = (
        triangulator.triangulate_points(
            points1,
            points2,
            pose_results["rotation"],
            pose_results["translation"],
        )
    )

    points_3d = triangulation_results["points_3d"]

    # Point cloud filtering
    point_filter = PointCloudFilter()

    filter_results = point_filter.filter_points(
        points_3d
    )

    filtered_points = filter_results["points_3d"]

    print("\nPoint Cloud Filtering Results")
    print("-" * 60)

    print(
        f"\nInput 3D points: "
        f"{filter_results['input_point_count']}"
    )

    print(
        f"Filtered 3D points: "
        f"{filter_results['valid_point_count']}"
    )

    print(
        f"Removed points: "
        f"{filter_results['removed_point_count']}"
    )

    if len(filtered_points) > 0:
        print("\nFirst 10 Filtered Points:")

        for point in filtered_points[:10]:
            print(
                f"X: {point[0]:.4f}, "
                f"Y: {point[1]:.4f}, "
                f"Z: {point[2]:.4f}"
            )


if __name__ == "__main__":
    main()