# %%
import contextlib
import pprint
from typing import Callable

import diffusers
import einops
import healpy
import matplotlib.pyplot as plt
import torch

from earth2grid import healpix

calls = []


def wrap_torch_functional(f):
    def func(*args, **kwargs):
        calls.append(f)
        return f(*args, **kwargs)

    return func


# wrap all functions in nn functional to see what is called
for item in dir(torch.nn.functional):
    func = getattr(torch.nn.functional, item)
    if isinstance(func, Callable):  # type: ignore
        setattr(torch.nn.functional, item, wrap_torch_functional(func))


torch_conv2d = torch.nn.functional.conv2d
torch_group_norm = torch.nn.functional.group_norm
torch_attention = torch.nn.functional.scaled_dot_product_attention


in_c = 5
out_c = 4
h = w = 256
f = 12
n = 1
npix = h * w * f

device = "cuda"

model = diffusers.AutoencoderKL(in_channels=in_c, out_channels=out_c)
model.to(device)
x = torch.zeros(n, in_c, 1, npix).to(device)
out = model(x)
print("output keys", out.keys())
print("functions in torch.nn.functional called by diffusers.AutoencoderKL")
pprint.pprint(set(calls))


# %%


def disabled_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    """Computes scaled dot product attention on query, key and value tensors,
    using an optional attention mask if passed, and applying dropout if a
    probability greater than 0.0 is specified.

    Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
        attn_mask (optional Tensor): Attention mask; shape :math:`(N, ..., L, S)`. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element *should* take part in attention.
            A float mask of the same type as query, key, value that is added to the attention score.
    Shape legend:
    - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
    - :math:`S: \text{Source sequence length}`
    - :math:`L: \text{Target sequence length}`
    - :math:`E: \text{Embedding dimension of the query and key}`
    - :math:`Ev: \text{Embedding dimension of the value}`
    """
    return value


@contextlib.contextmanager
def healpix_ops(no_attention: bool = False):
    """Override the parts of torch.nn.functional used by the diffusers autoencoder

    Args:
        no_attention: if true, than have attention just return the value with no
            spatial mixing
    """
    torch.nn.functional.conv2d = healpix.conv2d
    if no_attention:
        torch.nn.functional.scaled_dot_product_attention = no_attention
    # TODO scaled-dot-product attention
    yield
    torch.nn.functional.conv2d = torch_conv2d


with healpix_ops():
    decoder_out = model(x)

assert decoder_out.sample.shape == (n, out_c, 1, npix)

# %%

# the pixel_order is important for compatibility with healpix.pad
hpx_grid = healpix.Grid(level=8, pixel_order=healpix.HEALPIX_PAD_XY)
lat = torch.from_numpy(hpx_grid.lat).cuda()
lon = torch.from_numpy(hpx_grid.lon).cuda()
z = torch.cos(3 * torch.deg2rad(lat)) * torch.cos(3 * torch.deg2rad(lon))

# needs to be in ring order for healpy interop
z_view = hpx_grid.reorder(healpix.PixelOrder.RING, z)


def visualize(z, **kwargs):
    z_view = hpx_grid.reorder(healpix.PixelOrder.RING, z)
    healpy.mollview(z_view.cpu(), **kwargs)


# try the network

in_c = 1
out_c = 1
nside = 2**hpx_grid.level
n = 12

device = "cuda"
input = einops.rearrange(z, "(f x y) -> () () () (f x y)", f=12, x=nside, y=nside)
input = input.float()

with torch.no_grad():
    model = diffusers.AutoencoderKL(in_channels=in_c, out_channels=out_c)
    model.to(device)

    with healpix_ops():
        decoder_out = model(input)
    output = decoder_out.sample
    output_hpx_nn = einops.rearrange(output, "f c x y -> c (f x y)")

    decoder_out = model(input)
    output = decoder_out.sample
    output_orig_nn = einops.rearrange(output, "f c x y -> c (f x y)")

plt.figure(figsize=(12, 4))
visualize(z.cpu(), flip='geo', sub=131, title="Input")
visualize(output_hpx_nn[0].detach().cpu(), flip='geo', sub=133, title="NN output - with hpx overrides")
visualize(output_orig_nn[0].detach().cpu(), flip='geo', sub=132, title="NN output - with default")
plt.savefig("plot.png")
