from pathlib import Path
import sys

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from reconstruction.feature_matching.feature_matcher import (
    FeatureMatcher,
)


FRAME_DIRECTORY = Path(
    "datasets/raw/UAV123_10fps/data_seq/UAV123_10fps/bike1"
)


def main():
    matcher = FeatureMatcher(
        max_features=3000,
        match_ratio=0.75,
    )

    frame1 = FRAME_DIRECTORY / "000001.jpg"
    frame2 = FRAME_DIRECTORY / "000006.jpg"

    result = matcher.match_images(
        frame1,
        frame2,
    )

    print("\nFeature Matching Results")
    print("-" * 60)

    print(f"Image 1: {frame1.name}")
    print(f"Image 2: {frame2.name}")

    print(
        f"\nKeypoints in image 1: "
        f"{len(result['keypoints1'])}"
    )

    print(
        f"Keypoints in image 2: "
        f"{len(result['keypoints2'])}"
    )

    print(
        f"Raw matches: "
        f"{result['raw_match_count']}"
    )

    print(
        f"Good matches: "
        f"{result['good_match_count']}"
    )

    # Visualize the first 50 good matches
    visualization = cv2.drawMatches(
        cv2.imread(str(frame1)),
        result["keypoints1"],
        cv2.imread(str(frame2)),
        result["keypoints2"],
        result["matches"][:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    output_path = Path(
        "outputs/feature_matches_000001_000006.jpg"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        visualization,
    )

    print(f"\nMatch visualization saved to:")
    print(output_path)


if __name__ == "__main__":
    main()