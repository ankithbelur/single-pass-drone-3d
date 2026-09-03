from pathlib import Path
import cv2


class VideoReader:
    """Reads and extracts metadata from a drone video."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}"
            )

    def get_metadata(self) -> dict:
        """Extract basic metadata from the video."""

        capture = cv2.VideoCapture(str(self.video_path))

        if not capture.isOpened():
            raise ValueError(
                f"Unable to open video: {self.video_path}"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = (
            frame_count / fps
            if fps > 0
            else 0
        )

        capture.release()

        return {
            "video_path": str(self.video_path),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }
    