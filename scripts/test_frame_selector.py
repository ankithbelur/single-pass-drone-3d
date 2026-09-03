import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from reconstruction.preprocessing.frame_sequence_reader import (
    FrameSequenceReader,
)
from reconstruction.frame_selection.frame_selector import (
    FrameSelector,
)


def main():
    sequence_path = (
        "datasets/raw/UAV123_10fps/"
        "data_seq/UAV123_10fps/bike1"
    )

    reader = FrameSequenceReader(sequence_path)

    selector = FrameSelector(
        min_sharpness=100.0,
        min_difference=10.0,
        frame_interval=10,
    )

    selected_frames = selector.select_frames(
        reader.frame_paths
    )

    print("\nFrame Selection Results")
    print("-" * 40)
    print(f"Total frames: {len(reader.frame_paths)}")
    print(f"Selected frames: {len(selected_frames)}")

    if selected_frames:
        print("\nSelected Frame Analysis")
        print("-" * 40)

        previous_frame = None

        for frame_path in selected_frames[:10]:
            frame = cv2.imread(str(frame_path))

            sharpness = selector.calculate_sharpness(frame)

            if previous_frame is None:
                difference = 0.0
            else:
                difference = selector.calculate_difference(
                    previous_frame,
                    frame,
                )

            print(
                f"{frame_path.name} | "
                f"Sharpness: {sharpness:.2f} | "
                f"Difference: {difference:.2f}"
            )

            previous_frame = frame


if __name__ == "__main__":
    main()