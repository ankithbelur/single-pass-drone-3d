import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from reconstruction.feature_matching.feature_matcher import FeatureMatcher
from reconstruction.feature_matching.geometric_verification import GeometricVerifier


def main():
    image1_path = (
        "datasets/raw/UAV123_10fps/data_seq/"
        "UAV123_10fps/bike1/000001.jpg"
    )

    image2_path = (
        "datasets/raw/UAV123_10fps/data_seq/"
        "UAV123_10fps/bike1/000006.jpg"
    )

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        raise FileNotFoundError(
            "Could not load one or both images."
        )

    # Feature matching
    matcher = FeatureMatcher()

    match_results = matcher.match_images(
        image1,
        image2,
    )

    keypoints1 = match_results["keypoints1"]
    keypoints2 = match_results["keypoints2"]
    good_matches = match_results["matches"]

    # Geometric verification
    verifier = GeometricVerifier()

    fundamental_matrix, inlier_matches, inlier_mask = (
        verifier.verify_matches(
            keypoints1,
            keypoints2,
            good_matches,
        )
    )

    print("\nGeometric Verification Results")
    print("-" * 60)

    print(f"\nRaw matches: {match_results['raw_match_count']}")
    print(f"Good matches: {len(good_matches)}")
    print(f"Inlier matches: {len(inlier_matches)}")

    inlier_ratio = len(inlier_matches) / len(good_matches)

    print(f"Inlier ratio: {inlier_ratio:.4f}")

    print("\nFundamental Matrix:")
    print(fundamental_matrix)

    # Draw verified matches
    output_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    output_path = (
        "outputs/geometric_matches_"
        "000001_000006.jpg"
    )

    cv2.imwrite(output_path, output_image)

    print("\nVerified match visualization saved to:")
    print(output_path)


if __name__ == "__main__":
    main()