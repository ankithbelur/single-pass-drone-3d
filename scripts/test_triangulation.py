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


def main():
    sequence_path = Path(
        "datasets/raw/UAV123_10fps/data_seq/UAV123_10fps/bike1"
    )

    image1 = sequence_path / "000001.jpg"
    image2 = sequence_path / "000006.jpg"

    # Load image to obtain dimensions
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
        fundamental_matrix,
        inlier_matches,
        inlier_mask,
    ) = verifier.verify_matches(
        match_results["keypoints1"],
        match_results["keypoints2"],
        match_results["matches"],
    )

    # Extract geometrically verified points
    points1 = np.float32([
        match_results["keypoints1"][match.queryIdx].pt
        for match in inlier_matches
    ])

    points2 = np.float32([
        match_results["keypoints2"][match.trainIdx].pt
        for match in inlier_matches
    ])

    # Approximate camera parameters
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

    triangulation_results = triangulator.triangulate_points(
        points1,
        points2,
        pose_results["rotation"],
        pose_results["translation"],
    )

    points_3d = triangulation_results["points_3d"]

    print("\nTriangulation Results")
    print("-" * 60)

    print(f"\nInput matched points: {len(points1)}")

    print(
        f"Valid 3D points: "
        f"{triangulation_results['valid_point_count']}"
    )

    print("\nFirst 10 3D Points:")

    for point in points_3d[:10]:
        print(
            f"X: {point[0]:.4f}, "
            f"Y: {point[1]:.4f}, "
            f"Z: {point[2]:.4f}"
        )


if __name__ == "__main__":
    main()