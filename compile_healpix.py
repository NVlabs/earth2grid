import torch

from earth2grid.healpix.padding import pad

nside = 64
p = torch.arange(12 * nside * nside).reshape(1, 1, 12, nside, nside).float()

pad = torch.compile(pad)
out = pad(p, pad=16)
