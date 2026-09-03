from pathlib import Path
import sys

import numpy as np

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from reconstruction.point_cloud.point_cloud_io import (
    PointCloudIO,
)


def main():

    points = np.array(
        [
            [-2.8578, -6.8786, 65.2446],
            [-1.9964, -6.8917, 67.3488],
            [27.1882, -14.8669, 70.7147],
            [-2.5501, -6.9736, 65.1839],
            [24.4816, -9.5889, 66.7264],
            [-13.0381, -12.0852, 61.2300],
            [-20.0316, -7.3230, 59.1467],
            [18.9657, -15.7817, 69.8069],
            [-2.7372, -6.9367, 66.5202],
            [28.2549, 6.0585, 68.2747],
        ],
        dtype=np.float64,
    )

    output_path = Path(
        "outputs/point_cloud_test.ply"
    )

    saved_path = PointCloudIO.save_ply(
        points,
        output_path,
    )

    print("\nPoint Cloud Export Results")
    print("-" * 60)

    print(f"\nPoints exported: {len(points)}")

    print(f"Output file: {saved_path}")

    print(
        f"File exists: {saved_path.exists()}"
    )


if __name__ == "__main__":
    main()