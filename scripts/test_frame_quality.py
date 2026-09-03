from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from reconstruction.quality_assessment.frame_quality import (
    FrameQualityAssessor,
)


FRAME_DIRECTORY = Path(
    "datasets/raw/UAV123_10fps/data_seq/UAV123_10fps/bike1"
)


def main():
    assessor = FrameQualityAssessor()

    frame_files = sorted(FRAME_DIRECTORY.glob("*.jpg"))

    # Test a few frames across the sequence
    test_indices = [0, 100, 300, 500, 700, 900]

    print("\nFrame Quality Assessment")
    print("-" * 70)

    for index in test_indices:
        if index >= len(frame_files):
            continue

        frame_path = frame_files[index]

        result = assessor.assess_frame(frame_path)

        print(f"\nFrame: {frame_path.name}")

        for metric, value in result.items():
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main() 