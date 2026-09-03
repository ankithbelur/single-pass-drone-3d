from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from reconstruction.feature_matching.feature_matcher import FeatureMatcher
from reconstruction.feature_matching.geometric_verification import (
    GeometricVerifier,
)
from reconstruction.pose_estimation.camera_pose import (
    CameraPoseEstimator,
)


def main():
    sequence_path = Path(
        "datasets/raw/UAV123_10fps/data_seq/UAV123_10fps/bike1"
    )

    image1 = sequence_path / "000001.jpg"
    image2 = sequence_path / "000006.jpg"

    # Load image
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

    # Extract matched point coordinates
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
    estimator = CameraPoseEstimator(
        focal_length=focal_length,
        principal_point=principal_point,
    )

    pose_results = estimator.estimate_pose(
        points1,
        points2,
    )

    print("\nCamera Pose Estimation Results")
    print("-" * 60)

    print(
        f"\nGeometric inlier matches: "
        f"{len(inlier_matches)}"
    )

    print(
        f"Matched points used: {len(points1)}"
    )

    print(
        f"Essential Matrix inliers: "
        f"{pose_results['essential_inliers']}"
    )

    print(
        f"Pose inliers: "
        f"{pose_results['pose_inliers']}"
    )

    print("\nCamera Matrix:")
    print(estimator.camera_matrix)

    print("\nRotation Matrix (R):")
    print(pose_results["rotation"])

    print("\nTranslation Vector (t):")
    print(pose_results["translation"])


if __name__ == "__main__":
    main()