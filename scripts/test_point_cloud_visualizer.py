from pathlib import Path
import sys

import numpy as np

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from reconstruction.visualization.point_cloud_visualizer import (
    PointCloudVisualizer,
)


def main():

    np.random.seed(42)

    points = np.random.randn(
        1000,
        3,
    )

    # Create some depth variation
    points[:, 2] = (
        points[:, 2] * 10 + 50
    )

    output_path = (
        "outputs/point_cloud_visualization.png"
    )

    PointCloudVisualizer.visualize(
        points,
        output_path=output_path,
        show=False,
    )

    output_file = Path(output_path)

    print("\nPoint Cloud Visualization Results")

    print("-" * 60)

    print(
        f"\nPoints visualized: {len(points)}"
    )

    print(
        f"Output file: {output_path}"
    )

    print(
        f"File exists: {output_file.exists()}"
    )


if __name__ == "__main__":
    main()