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
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from earth2grid import get_regridder, healpix, healpix_bare
from earth2grid.healpix.core import _rotate, local2local, local2xy, ring2double
from earth2grid.healpix.visualization import _to_mesh


def test_ring2xy():
    nside = 4
    p = torch.arange(12 * nside * nside)
    xy = healpix.ring2xy(nside, p)
    hpd = healpix_bare.ring2hpd(nside, p)
    xy_from_c = hpd @ torch.tensor([nside * nside, nside, 1])
    assert torch.allclose(xy.long(), xy_from_c)


@pytest.mark.xfail
def test_grid_visualize():
    grid = healpix.Grid(level=4, pixel_order=healpix.XY())
    z = np.cos(10 * np.deg2rad(grid.lat))
    grid.visualize(z)
    plt.savefig("test_grid_visualize.png")


def test__to_mesh(regtest):
    z = torch.arange(12 * 2 * 2)
    xx, yy, out = _to_mesh(z)

    assert out.shape == (10, 10)
    assert xx.shape == (11, 11)
    assert yy.shape == (11, 11)
    np.savetxt(regtest, out, fmt="%.0f", delimiter="\t")


def test_pcolormesh():
    z = np.random.randn(12 * 16 * 16)
    healpix.pcolormesh(z)


@pytest.mark.parametrize("origin", list(healpix.Compass))
def test_grid_healpix_orientations(tmp_path, origin):
    nest_grid = healpix.Grid(level=4, pixel_order=healpix.PixelOrder.NEST)
    grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=origin))

    nest_lat = nest_grid.lat.reshape([12, -1])
    lat = grid.lat.reshape([12, -1])

    for i in range(12):
        assert set(nest_lat[i]) == set(lat[i])


@pytest.mark.parametrize("rot", range(4))
def test_xy2xy_index_same_values(tmp_path, rot):
    n = 8
    i = torch.arange(12 * n * n)
    S = healpix.XY()
    E = healpix.XY(origin=healpix.Compass.E)
    i_rot = healpix.xy2xy(n, S, E, i)
    i = i.reshape(12, -1)
    i_rot = i_rot.reshape(12, -1)
    for f in range(12):
        assert set(i[f].numpy()) == set(i_rot[f].numpy())


@pytest.mark.parametrize("rot", range(4))
def test_rotate(rot):
    n = 32
    x, y = torch.tensor([0, 0])
    xr, yr = _rotate(n, rot, x, y)
    xb, yb = _rotate(n, 4 - rot, xr, yr)
    np.testing.assert_array_equal(xb, x)
    np.testing.assert_array_equal(yb, y)


@pytest.mark.parametrize("origin", list(healpix.Compass))
@pytest.mark.parametrize("clockwise", [True, False])
def test_Grid_reorder(tmp_path, origin, clockwise):
    src_grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=origin, clockwise=clockwise))
    dest_grid = healpix.Grid(level=4, pixel_order=healpix.PixelOrder.NEST)

    z = np.cos(np.deg2rad(src_grid.lat)) * np.cos(np.deg2rad(src_grid.lon))
    z = torch.from_numpy(z)
    z_reorder = src_grid.reorder(dest_grid.pixel_order, z)
    z_roundtrip = dest_grid.reorder(src_grid.pixel_order, z_reorder)
    np.testing.assert_array_equal(z, z_roundtrip)


def get_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices += [torch.device("cuda")]
    return devices


@pytest.mark.parametrize("origin", list(healpix.Compass))
@pytest.mark.parametrize("clockwise", [True, False])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("device", get_devices())
def test_grid_healpix_pad(tmp_path, origin, clockwise, padding, device):
    grid = healpix.Grid(level=4, pixel_order=healpix.XY(origin=origin, clockwise=clockwise))
    hpx_pad_grid = healpix.Grid(level=4, pixel_order=healpix.HEALPIX_PAD_XY)
    z = np.cos(np.deg2rad(grid.lat)) * np.cos(np.deg2rad(grid.lon))
    z = torch.from_numpy(z)
    regrid = get_regridder(grid, hpx_pad_grid)
    z_hpx_pad = regrid(z)

    n = grid._nside()
    z = z.view(-1, 12, n, n)
    z_hpx_pad = z_hpx_pad.view(-1, 12, n, n)

    padded = healpix.pad(z_hpx_pad.to(device), padding).cpu()

    def grad_abs(z):
        fx, fy = np.gradient(z, axis=(-1, -2))
        return np.mean(np.abs(fx)) + np.mean(np.abs(fy))

    # the padded dtile should not vary much more than the non-padded tile
    sigma_padded = grad_abs(padded)
    sigma = grad_abs(z)

    if sigma_padded > sigma * 1.1:
        fig, axs = plt.subplots(3, 4)
        axs = axs.ravel()
        for i in range(12):
            ax = axs[i]
            ax.pcolormesh(padded[0, i])
        output_path = tmp_path / "output.png"
        fig.savefig(output_path.as_posix())

        raise ValueError(
            f"The gradient of the padded data {sigma_padded} is too large. "
            f"Examine the padding in the image at {output_path}."
        )


def test_to_image():
    grid = healpix.Grid(level=4)
    lat = torch.tensor(grid.lat)
    lat_img = grid.to_image(lat)
    n = 2**grid.level
    assert lat_img.shape == (5 * n, 5 * n)


def test_conv2d():
    f = 12
    nside = 16
    npix = f * nside * nside
    cin = 3
    cout = 4
    n = 1

    x = torch.ones(n, cin, 1, npix)
    weight = torch.zeros(cout, cin, 3, 3)
    out = healpix.conv2d(x, weight, padding=(1, 1))
    assert out.shape == (n, cout, 1, npix)


@pytest.mark.parametrize("nside", [16])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("src_pixel_order", [healpix.HEALPIX_PAD_XY, healpix.PixelOrder.RING, healpix.PixelOrder.NEST])
@pytest.mark.parametrize("dest_pixel_order", [healpix.HEALPIX_PAD_XY, healpix.PixelOrder.RING, healpix.PixelOrder.NEST])
def test_reorder(nside, src_pixel_order, dest_pixel_order, device):
    # Generate some test data
    if device == "cuda" and torch.cuda.device_count() == 0:
        pytest.skip("no cuda devices available")

    data = torch.randn(1, 2, 12 * nside * nside, device=device)
    out = healpix.reorder(data, src_pixel_order, dest_pixel_order)
    out = healpix.reorder(out, dest_pixel_order, src_pixel_order)
    assert torch.all(data == out), data - out


def test_latlon_cuda_set_device_regression():
    """See https://github.com/NVlabs/earth2grid/issues/6"""

    if torch.cuda.device_count() == 0:
        pytest.skip()

    default = torch.get_default_device()
    try:
        torch.set_default_device("cuda")
        grid = healpix.Grid(4)
        grid.lat
    finally:
        torch.set_default_device(default)


@pytest.mark.parametrize("device,do_torch", [("cpu", True), ("cuda", True), ("cpu", False)])
def test_zonal_average(device, do_torch):

    if device == "cuda" and torch.cuda.device_count() == 0:
        pytest.skip("no cuda devices available")

    # hpx 2 in ring order
    x = np.array(
        [
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
        ]
    )
    x = x[None]
    if do_torch:
        x = torch.from_numpy(x).to(device)
    zonal = healpix.zonal_average(x)
    if do_torch:
        zonal = zonal.cpu().numpy()
    assert zonal.shape == (1, 7)
    assert np.all(zonal == np.arange(7))


def test_to_double_pixelization(regtest):
    n = 2
    x = np.arange(12 * n * n)
    x = healpix.to_double_pixelization(x)
    assert x.dtype == x.dtype
    np.savetxt(regtest, x, fmt="%d")


def test_to_double_pixelization_cuda(device="cuda"):
    if not torch.cuda.is_available():
        pytest.skip()

    n = 2
    x = np.arange(12 * n * n)
    xnp = healpix.to_double_pixelization(x)

    x = torch.arange(12 * n * n, device=device)
    x = healpix.to_double_pixelization(x)

    np.testing.assert_array_equal(xnp, x.cpu().numpy())


def test_local2xy():
    x, y, f = local2xy(1, torch.tensor([1]), torch.tensor([0]), torch.tensor([0]))
    assert f.item() == 1

    x, y, f = local2xy(1, torch.tensor([-1]), torch.tensor([0]), torch.tensor([0]))
    assert f.item() == 4

    x, y, f = local2xy(4, torch.tensor([-1]), torch.tensor([0]), torch.tensor([0]))
    assert (x.item(), y.item(), f.item()) == (3, 0, 4)

    x, y, f = local2xy(4, torch.tensor([4]), torch.tensor([0]), torch.tensor([0]))
    assert (x.item(), y.item(), f.item()) == (0, 3, 1)


def test_ring2double_preserves_dtype():
    p = torch.tensor([0])
    i, j = ring2double(1024, p)
    assert i.dtype == p.dtype
    assert j.dtype == p.dtype


def test_local2local_round_trip():
    nside = 4
    f = torch.arange(12)
    i = torch.arange(nside)
    j = torch.arange(nside)

    f, j, i = torch.meshgrid(f, j, i, indexing="ij")
    i1, j1 = local2local(nside, healpix.XY(), healpix.HEALPIX_PAD_XY, i, j)
    i2, j2 = local2local(nside, healpix.HEALPIX_PAD_XY, healpix.XY(), i1, j1)

    assert torch.all(i2 == i)
    assert torch.all(j2 == j)


def test_local2local_S_to_E():
    nside = 4
    i = torch.tensor([0])
    j = torch.tensor([0])
    S = healpix.XY()
    E = healpix.XY(origin=healpix.Compass.E)

    i1, j1 = local2local(nside, src=E, dest=S, x=i, y=j)

    assert (i1.item(), j1.item()) == (nside - 1, 0)

    pix = torch.tensor([0])
    pix = healpix.xy2xy(nside, src=E, dest=S, i=pix)
    assert pix.item() == nside - 1
