import sys
from pathlib import Path

# Allow importing modules from the project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from reconstruction.preprocessing.frame_sequence_reader import FrameSequenceReader


def main():
    sequence_path = (
        "datasets/raw/UAV123_10fps/"
        "data_seq/UAV123_10fps/bike1"
    )

    reader = FrameSequenceReader(sequence_path)

    metadata = reader.get_metadata()

    print("\nFrame Sequence Metadata:")
    print("-" * 30)

    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Test loading the first frame
    frame = reader.get_frame(0)

    print("\nFirst frame loaded successfully.")
    print(f"Frame shape: {frame.shape}")


if __name__ == "__main__":
    main()