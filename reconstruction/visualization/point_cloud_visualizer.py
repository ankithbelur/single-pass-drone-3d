from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt


class PointCloudVisualizer:
    """
    Visualizes 3D point clouds.
    """

    @staticmethod
    def visualize(
        points: np.ndarray,
        output_path: Union[str, Path] = None,
        show: bool = False,
    ):
        """
        Visualize a 3D point cloud.

        Args:
            points: NumPy array of shape (N, 3)
            output_path: Optional path to save the visualization
            show: Whether to display the plot
        """

        points = np.asarray(points)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Points must have shape (N, 3)."
            )

        if len(points) == 0:
            raise ValueError(
                "Point cloud is empty."
            )

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(
            111,
            projection="3d",
        )

        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            cmap="viridis",
            s=5,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(
            f"3D Point Cloud ({len(points)} points)"
        )

        fig.colorbar(
            scatter,
            ax=ax,
            label="Depth (Z)",
        )

        if output_path is not None:

            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close(fig)