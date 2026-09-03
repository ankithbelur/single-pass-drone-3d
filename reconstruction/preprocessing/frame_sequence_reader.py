from pathlib import Path
import cv2


class FrameSequenceReader:
    """Reads metadata and frames from an ordered image sequence."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, sequence_path: str):
        self.sequence_path = Path(sequence_path)

        if not self.sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence folder not found: {self.sequence_path}"
            )

        if not self.sequence_path.is_dir():
            raise ValueError(
                f"Expected a directory: {self.sequence_path}"
            )

        self.frame_paths = self._get_frame_paths()

        if not self.frame_paths:
            raise ValueError(
                f"No supported image frames found in: {self.sequence_path}"
            )

    def _get_frame_paths(self) -> list[Path]:
        """Return frame paths in sorted order."""

        return sorted(
            [
                path
                for path in self.sequence_path.iterdir()
                if path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        )

    def get_metadata(self) -> dict:
        """Extract metadata from the frame sequence."""

        first_frame = cv2.imread(str(self.frame_paths[0]))

        if first_frame is None:
            raise ValueError(
                f"Unable to read frame: {self.frame_paths[0]}"
            )

        height, width = first_frame.shape[:2]

        return {
            "sequence_path": str(self.sequence_path),
            "frame_count": len(self.frame_paths),
            "width": width,
            "height": height,
            "first_frame": self.frame_paths[0].name,
            "last_frame": self.frame_paths[-1].name,
        }

    def get_frame(self, index: int):
        """Load and return a frame by index."""

        if index < 0 or index >= len(self.frame_paths):
            raise IndexError(
                f"Frame index {index} is out of range"
            )

        frame = cv2.imread(str(self.frame_paths[index]))

        if frame is None:
            raise ValueError(
                f"Unable to read frame: {self.frame_paths[index]}"
            )

        return frame
    