from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from reconstruction.frame_selection.confidence_selector import (
    ConfidenceAwareFrameSelector,
)


SEQUENCE_PATH = Path(
    "datasets/raw/UAV123_10fps/data_seq/UAV123_10fps/bike1"
)


def main():
    selector = ConfidenceAwareFrameSelector(
        min_quality_score=0.55,
        min_scene_difference=10.0,
        min_frame_gap=5,
    )

    selected_frames = selector.select_frames(
        SEQUENCE_PATH
    )

    print("\nConfidence-Aware Frame Selection")
    print("-" * 70)

    print(f"Total selected frames: {len(selected_frames)}")

    print("\nFirst 15 selected frames:")
    print("-" * 70)

    for frame in selected_frames[:15]:
        print(
            f"{frame['frame_name']} | "
            f"Quality: {frame['quality_score']:.4f} | "
            f"Scene Diff: {frame['scene_difference']:.2f} | "
            f"Confidence: {frame['confidence_score']:.4f}"
        )


if __name__ == "__main__":
    main()