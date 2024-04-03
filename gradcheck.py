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
    x = torch.rand(10, pos_dim, dtype=dtype).cuda()
    out_grads = torch.zeros(1, 48, dtype=dtype).cuda()
    x.requires_grad = True
    single_grad_correct = torch.autograd.gradcheck(
        lambda lat, pos: PermutoEncodingFunc.apply(encoding.lattice, lat, pos, encoding.anneal_window, True, True),
        (
            encoding.features,
            x,
        ),
    )
    if single_grad_correct:
        print("Single grad check passed")
    double_grad_correct = torch.autograd.gradgradcheck(
        lambda pos, lat: PermutoEncodingFunc.apply(encoding.lattice, lat, pos, encoding.anneal_window, True, True),
        (
            x,
            encoding.features,
        ),
    )
    if double_grad_correct:
        print("Double grad check passed")
