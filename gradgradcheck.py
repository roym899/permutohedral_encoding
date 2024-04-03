import numpy as np
import torch

import permutohedral_encoding
from permutohedral_encoding.funcs import PermutoEncodingFunc

if __name__ == "__main__":
    # create encoding
    pos_dim = 3
    capacity = pow(2, 8)
    nr_levels = 24
    nr_feat_per_level = 2
    coarsest_scale = 0.1
    finest_scale = 0.001
    scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
    dtype = torch.float64

    encoding = permutohedral_encoding.PermutoEncoding(
        pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=dtype
    )
    with torch.no_grad():
        encoding.features[:] = 0.0

    x = torch.rand(10, pos_dim, dtype=dtype, requires_grad=True).cuda()

    outs = encoding(x)
    scalar = outs.sum()

    grads = torch.autograd.grad(scalar, x, create_graph=True)
    scalar2 = grads[0].sum()

    scalar2.backward()
    breakpoint()
