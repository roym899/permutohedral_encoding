import time

import numpy as np
import torch

import permutohedral_encoding

torch.backends.cudnn.benchmark = True


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
    print(f"{name:40s}{trials=}\t{t_min=:.4f}\t{t_max=:.4f}\t{avg=:.4f}")
    return avg


# create encoding
pos_dim = 2
capacity = pow(2, 10)
nr_levels = 24
nr_feat_per_level = 2
coarsest_scale = 1
finest_scale = 0.001
scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)

encoding_f32 = permutohedral_encoding.PermutoEncoding(
    pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=torch.float32
)
encoding_f16 = permutohedral_encoding.PermutoEncoding(
    pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=torch.float16
)
encoding_f64 = permutohedral_encoding.PermutoEncoding(
    pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=torch.float64
)

num_points = 3000000
points_f16 = torch.rand(num_points, 2, dtype=torch.float16).to("cuda")
points_f32 = torch.rand(num_points, 2, dtype=torch.float32).to("cuda")
points_f64 = torch.rand(num_points, 2, dtype=torch.float64).to("cuda")
a16 = torch.rand(3000, 3000, device="cuda", dtype=torch.float16)
a32 = torch.rand(3000, 3000, device="cuda", dtype=torch.float32)
a64 = torch.rand(3000, 3000, device="cuda", dtype=torch.float64)


def aa16():
    res = a16 @ a16


def aa32():
    res = a32 @ a32


def aa64():
    res = a64 @ a64


def f64():
    encoding_f64(points_f64)


def f16():
    encoding_f16(points_f16)


def f32():
    encoding_f32(points_f32)


def f64():
    encoding_f64(points_f64)


f16(), print("f16 works")
f32(), print("f32 works")
f64(), print("f64 works")

benchmark(f16, "f16 forward", 10)
benchmark(f32, "f32 forward", 10)
benchmark(f64, "f64 forward", 10)

benchmark(aa16, "aa16 forward", 10)
benchmark(aa32, "aa32 forward", 10)
benchmark(aa64, "aa64 forward", 10)
