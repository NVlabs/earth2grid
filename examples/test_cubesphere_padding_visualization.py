#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test script to visualize cubesphere padding with E3SM pg2 lat/lon data.

This script loads lat/lon coordinates from an E3SM ne1024pg2 grid,
converts to XY format, applies padding, and visualizes the results.
"""

import matplotlib.pyplot as plt
import torch
import xarray as xr
from matplotlib.patches import Rectangle

from earth2grid import cubesphere


def main():
    # Load the lat/lon data
    # This data contains E3SM lat/lon coordinates in ne1024pg2 grid
    data_path = "./examples/latlon_ne1024pg2.nc"
    print(f"Loading data from {data_path}")

    try:
        ds0 = xr.open_dataset(data_path)
    except FileNotFoundError:
        print(f"\nError: Data file not found at {data_path}")
        print("Please download the test data from:")
        print("  pbss:zeyuanhu/scream-runs/latlon_ne1024pg2.nc")
        print(f"\nAnd place it at: {data_path}")
        return

    # Grid parameters
    ne = 1024
    npg = 2
    pad_width = 1024
    face_size = ne * npg  # 2048

    # Define orderings
    e3sm = cubesphere.E3SMpgOrder(ne=ne, npg=npg)
    xy = cubesphere.XY(face_size=face_size)

    # Load lon and lat as torch tensors
    lon_1d = torch.from_numpy(ds0["lon"].values)
    lat_1d = torch.from_numpy(ds0["lat"].values)

    print(f"Loaded data shapes: lon={lon_1d.shape}, lat={lat_1d.shape}")
    print(f"Expected total points: {e3sm.total_pts}")

    # Stack into single tensor: (2, ncol)
    stacked = torch.stack([lon_1d, lat_1d], dim=0)
    print(f"Stacked shape: {stacked.shape}")

    # Convert from E3SM ordering to XY (6-face 2D format)
    # Shape: (2, ncol) -> (2, 6, face_size, face_size)
    faces = cubesphere.reorder(stacked, src=e3sm, dest=xy)
    print(f"Faces shape after reorder: {faces.shape}")

    # Apply padding to all faces at once
    # Shape: (2, 6, face_size, face_size) -> (2, 6, face_size + 2*pad_width, face_size + 2*pad_width)
    padded = cubesphere.pad(faces, padding=pad_width)
    print(f"Padded shape: {padded.shape}")

    # Extract lon and lat
    padded_lon_all = padded[0]
    padded_lat_all = padded[1]

    # Visualize each face
    for face_id in range(6):
        padded_lon = padded_lon_all[face_id].numpy()
        padded_lat = padded_lat_all[face_id].numpy()

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Longitude
        im0 = axes[0].imshow(padded_lon, origin="lower", cmap="viridis")
        axes[0].set_title(f"Padded Longitude (Face {face_id})")
        plt.colorbar(im0, ax=axes[0], label="Longitude")

        # Latitude
        im1 = axes[1].imshow(padded_lat, origin="lower", cmap="plasma")
        axes[1].set_title(f"Padded Latitude (Face {face_id})")
        plt.colorbar(im1, ax=axes[1], label="Latitude")

        # Add boundary boxes to both to show original face region
        for ax in axes:
            rect_center = Rectangle(
                (pad_width - 0.5, pad_width - 0.5),
                face_size,
                face_size,
                linewidth=2,
                edgecolor="white",
                facecolor="none",
            )
            ax.add_patch(rect_center)

        plt.tight_layout()
        plt.savefig(f"cubesphere_padded_face_{face_id}.png", dpi=150, bbox_inches="tight")
        print(f"Saved cubesphere_padded_face_{face_id}.png")
        plt.close()

    print("Done! All face visualizations saved.")


if __name__ == "__main__":
    main()
