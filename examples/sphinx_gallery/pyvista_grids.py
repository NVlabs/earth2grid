# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Plot grids with PyVista
-----------------------

"""
import pyvista as pv

# %%
import earth2grid


def label(mesh, plotter, text):
    """
    Add a label above a mesh in a PyVista plot.

    Parameters:
    - mesh: The mesh to label.
    - plotter: A PyVista plotter instance.
    - text: The label text.
    - color: The color of the text label. Default is 'white'.
    """
    # Calculate the center of the mesh and the top Z-coordinate plus an offset
    center = mesh.center
    label_pos = [center[0], center[1], mesh.bounds[5] + 0.5]  # Offset to place label above the mesh

    # Add the label using point labels for precise 3D positioning
    plotter.add_point_labels(
        [label_pos], [text], point_size=0, render_points_as_spheres=False, shape_opacity=0, font_size=20
    )


grid = earth2grid.healpix.Grid(level=4)
hpx = grid.to_pyvista()
latlon = earth2grid.latlon.equiangular_lat_lon_grid(32, 64, includes_south_pole=False).to_pyvista()


pl = pv.Plotter()
mesh = hpx.translate([0, 2.5, 0])
pl.add_mesh(mesh, show_edges=True)
label(mesh, pl, "HealPix")

pl.add_mesh(latlon, show_edges=True)
label(latlon, pl, "Equiangular Lat/Lon")

pl.camera.position = (5, 0, 5)
pl.show()
