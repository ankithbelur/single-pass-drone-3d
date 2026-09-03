import sys
from pathlib import Path

# Allow importing modules from the project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from reconstruction.preprocessing.video_reader import VideoReader


def main():
    video_path = input("Enter the path to a video file: ").strip()

    reader = VideoReader(video_path)
    metadata = reader.get_metadata()

    print("\nVideo Metadata:")
    print("-" * 30)

    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()