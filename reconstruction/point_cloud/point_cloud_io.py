from pathlib import Path
import numpy as np


class PointCloudIO:
    """Utilities for saving and loading 3D point clouds."""

    @staticmethod
    def save_ply(
        points: np.ndarray,
        output_path,
    ) -> Path:
        """
        Save Nx3 points to an ASCII PLY file.
        """

        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Points must have shape (N, 3)."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(
                f"element vertex {len(points)}\n"
            )
            file.write(
                "property float x\n"
            )
            file.write(
                "property float y\n"
            )
            file.write(
                "property float z\n"
            )
            file.write("end_header\n")

            for x, y, z in points:
                file.write(
                    f"{x:.6f} {y:.6f} {z:.6f}\n"
                )

        return output_path