#!/bin/bash

set -e

echo "============================================================"
echo " SINGLE-PASS DRONE VIDEO TO 3D RECONSTRUCTION PIPELINE"
echo "============================================================"

echo ""
echo "[1/11] Frame Sequence Reader"
python scripts/test_frame_sequence_reader.py

echo ""
echo "[2/11] Frame Selection"
python scripts/test_frame_selector.py

echo ""
echo "[3/11] Frame Quality Assessment"
python scripts/test_frame_quality.py

echo ""
echo "[4/11] Confidence-Aware Frame Selection"
python scripts/test_confidence_selector.py

echo ""
echo "[5/11] Feature Matching"
python scripts/test_feature_matcher.py

echo ""
echo "[6/11] Geometric Verification"
python scripts/test_geometric_verification.py

echo ""
echo "[7/11] Camera Pose Estimation"
python scripts/test_camera_pose.py

echo ""
echo "[8/11] 3D Point Triangulation"
python scripts/test_triangulation.py

echo ""
echo "[9/11] Point Cloud Filtering"
python scripts/test_point_cloud_filter.py

echo ""
echo "[10/11] Point Cloud Export"
python scripts/test_point_cloud_io.py

echo ""
echo "[11/11] Point Cloud Visualization"
python scripts/test_point_cloud_visualizer.py

echo ""
echo "============================================================"
echo " PIPELINE COMPLETED SUCCESSFULLY!"
echo "============================================================"