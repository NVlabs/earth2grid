import torch

from earth2grid.spatial import barycentric_coords_with_origin, select_simplex


def test_bary_origin():
    x, y, z = torch.eye(3)
    simplices = [
        [x, y, z],
        [-x, y, z],
        [x, -y, z],
        [-x, -y, z],
        [x, y, -z],
        [-x, y, -z],
        [x, -y, -z],
        [-x, -y, -z],
    ]
    simplices = torch.stack([torch.stack(pts) for pts in simplices])
    simplices = simplices.float()

    points = torch.tensor([[1, 1, 1], x]).float()
    points = points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    simplices = torch.tensor(simplices)
    coords = barycentric_coords_with_origin(points.double(), simplices.double())

    # should all be 0.5774
    exp = coords[0][0][0]
    assert torch.allclose(coords[0][0], exp)

    # [1,1,1] should only be in first quadrant
    index = select_simplex(coords)
    assert index[0] == 0

    # [x] should be in one of the quadrants containing x
    index = select_simplex(coords)
    assert index[0] in [0, 2, 4, 6]
