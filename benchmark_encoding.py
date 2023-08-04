import time

import numpy as np
import torch

import permutohedral_encoding

# torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)


def mm_f(dtype):
    res = matrix[dtype] @ matrix[dtype]


def enc_f(dtype):
    with torch.no_grad():
        encoding[dtype](points[dtype])


def enc_fnb_all(dtype):
    points[dtype].requires_grad = True
    res = encoding[dtype](points[dtype])
    loss = res.sum()
    loss.backward()


def enc_fnb_lat(dtype):
    points[dtype].requires_grad = False
    res = encoding[dtype](points[dtype])
    loss = res.sum()
    loss.backward()


def ind_add(dtype):
    res = torch.index_add(acc[dtype], 0, ind, points[dtype].flatten())


def benchmark(f, name, trials, skip_first_n=3, **kwargs):
    global mlps, queries
    t_min = 1000
    t_max = -1000
    total = 0

    for _ in range(skip_first_n):
        f(**kwargs)

    for _ in range(trials):
        torch.cuda.synchronize()
        t1 = time.time()

        f(**kwargs)

        torch.cuda.synchronize()
        t2 = time.time()
        t_min = min(t_min, t2 - t1)
        t_max = max(t_max, t2 - t1)
        total += t2 - t1

    avg = total / trials
    print(f"{name:30s}{trials=}\t{t_min=:.4f}\t{t_max=:.4f}\t{avg=:.4f}")
    return avg


if __name__ == "__main__":
    # create encoding
    pos_dim = 3
    capacity = pow(2, 12)
    nr_levels = 24
    nr_feat_per_level = 2
    coarsest_scale = 0.1
    finest_scale = 0.001
    scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
    dtypes = (torch.float16, torch.float32, torch.float64)
    # dtypes = (torch.float16,)
    # funcs = (mm_f, enc_f, enc_fnb_allgrads, enc_fnb_latgrads)
    funcs = (enc_f, enc_fnb_all, enc_fnb_lat)
    funcs = (enc_f, enc_fnb_lat)

    encoding = {
        dtype: permutohedral_encoding.PermutoEncoding(
            pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=dtype
        )
        for dtype in dtypes
    }

    num_points = 10000000
    points = {
        dtype: torch.rand(num_points, pos_dim, dtype=dtype, device="cuda")
        for dtype in dtypes
    }

    matrix = {
        dtype: torch.rand(5000, 5000, dtype=dtype, device="cuda") for dtype in dtypes
    }
    acc = {
        dtype: torch.zeros(num_points, dtype=dtype, device="cuda") for dtype in dtypes
    }
    ind = torch.randint(num_points, size=(3 * num_points,), device="cuda")

    for func in funcs:
        for dtype in dtypes:
            benchmark(lambda: func(dtype), f"{dtype} {func.__name__}", 10)
