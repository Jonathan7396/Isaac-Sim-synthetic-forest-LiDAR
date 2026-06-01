"""
Visualize an Isaac Sim LiDAR semantic point cloud using Open3D.

This script loads XYZ points and semantic labels saved from the RTX LiDAR
capture pipeline, assigns colors based on semantic labels, optionally downsamples
the point cloud, and visualizes it in an Open3D window.

Expected files:
    lidar_output_manual_VLS128_asset_semantic/y_up_xyz.npy
    lidar_output_manual_VLS128_asset_semantic/y_up_semantic_labels.npy
"""

from pathlib import Path

import numpy as np
import open3d as o3d


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("lidar_output_manual_VLS128_asset_semantic")

XYZ_PATH = OUTPUT_DIR / "y_up_xyz.npy"
LABELS_PATH = OUTPUT_DIR / "y_up_semantic_labels.npy"

WINDOW_NAME = "Semantic LiDAR Point Cloud"

# Set to None if you do not want downsampling.
VOXEL_SIZE = 0.2


# -----------------------------------------------------------------------------
# Label colors
# -----------------------------------------------------------------------------
# RGB values must be between 0.0 and 1.0.
#
# Add your own labels here as needed.
# Example:
#     "[EUCAM_MID1]": [0.36, 0.25, 0.20],
#
# Make sure the label text matches exactly what appears in y_up_semantic_labels.npy.

LABEL_COLOR_MAP = {
    "[EUCAM_MID1]": [0.36, 0.25, 0.20],  # example tree label
    "unmapped": [0.0, 1.0, 0.0],         # green
    "unlabeled": [1.0, 1.0, 1.0],        # white
}

DEFAULT_COLOR = [0.5, 0.5, 0.5]          # grey for labels not listed above


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_point_cloud_data(xyz_path: Path, labels_path: Path):
    """Load XYZ points and semantic labels from NumPy files."""
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    xyz = np.load(xyz_path)
    labels = np.load(labels_path, allow_pickle=True)

    if len(xyz) != len(labels):
        raise ValueError(
            f"XYZ and label count mismatch: {len(xyz)} XYZ points, "
            f"{len(labels)} labels"
        )

    return xyz, labels


def print_label_summary(labels):
    """Print point count for each semantic label."""
    print("\n==============================")
    print("SEMANTIC LABEL SUMMARY")
    print("==============================")
    print(f"Total points: {len(labels)}")

    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in sorted(zip(unique_labels, counts), key=lambda x: x[1], reverse=True):
        percentage = 100.0 * count / len(labels)
        print(f"{str(label):30s} {count:10d} points   ({percentage:.2f}%)")


def assign_colors(labels):
    """Assign RGB colors to each point based on its semantic label."""
    colors = []

    for label in labels:
        label = str(label)
        color = LABEL_COLOR_MAP.get(label, DEFAULT_COLOR)
        colors.append(color)

    return np.array(colors, dtype=float)


def create_open3d_point_cloud(xyz, colors):
    """Create an Open3D point cloud from XYZ coordinates and RGB colors."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main():
    xyz, labels = load_point_cloud_data(XYZ_PATH, LABELS_PATH)

    print_label_summary(labels)

    colors = assign_colors(labels)
    point_cloud = create_open3d_point_cloud(xyz, colors)

    if VOXEL_SIZE is not None and VOXEL_SIZE > 0:
        print(f"\nDownsampling point cloud with voxel size: {VOXEL_SIZE}")
        point_cloud = point_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)

    print("\nOpening Open3D visualizer...")
    o3d.visualization.draw_geometries(
        [point_cloud],
        window_name=WINDOW_NAME,
    )


if __name__ == "__main__":
    main()