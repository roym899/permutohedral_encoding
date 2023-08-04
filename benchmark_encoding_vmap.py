import time

import numpy as np
import torch
import torch.nn as nn

import permutohedral_encoding
from benchmark_encoding import benchmark


class SimpleMLP(nn.Module):
    def __init__(
        self, num_layers=2, num_mlp_channels=64, encoding=True, dtype=torch.float32
    ):
        super().__init__()
        self._num_layers = num_layers
        self._layers = nn.Sequential()
        if encoding:
            self._encoding = permutohedral_encoding.PermutoEncoding(
                dim_in, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=dtype
            )
            self._layers.append(
                nn.Linear(self._encoding.output_dims(), num_mlp_channels, dtype=dtype)
            )
        else:
            self._encoding = None
            self._layers.append(nn.Linear(3, num_mlp_channels, dtype=dtype))
        for _ in range(self._num_layers):
            self._layers.append(
                nn.Linear(num_mlp_channels, num_mlp_channels, dtype=dtype)
            )
        self._layers.append(nn.Linear(num_mlp_channels, dim_out, dtype=dtype))

    def forward(self, x):
        if self._encoding is not None:
            x = self._encoding(x)

        for layer in self._layers:
            x = torch.relu(layer(x))
        return x


def create_wrapper(dtype):
    def f(params, buffers, queries):
        return torch.func.functional_call(wrapper_model[dtype], (params, buffers), queries)
    return f


def single_model(dtype, set_to_none) -> None:
    outs = mlps[dtype][0](queries[dtype])
    if with_grads:
        loss = outs.flatten().sum()
        loss.backward()
        mlps[dtype][0].zero_grad(set_to_none=set_to_none)


def for_loop(dtype, set_to_none) -> None:
    outs = torch.empty(num_queries, dim_out, dtype=dtype)

    for i, (mlp, mask) in enumerate(zip(unique_mlps[dtype], masks)):
        query = queries[dtype][mask]
        outs[mask] = mlp(query)

    if with_grads:
        loss = outs.flatten().sum()
        loss.backward()
        for mlp in mlps[dtype]:
            mlp.zero_grad(set_to_none=set_to_none)


def vmap(dtype, set_to_none) -> None:
    ins = queries[dtype].view(num_mlps, -1, 3)
    outs = torch.vmap(wrappers[dtype])(params[dtype], buffers[dtype], ins)
    if with_grads:
        loss = outs.flatten().sum()
        loss.backward()
        for mlp in mlps[dtype]:
            mlp.zero_grad(set_to_none=set_to_none)


if __name__ == "__main__":
    torch.set_default_device("cuda")

    dim_in = 3
    capacity = pow(2, 12)
    nr_levels = 24
    nr_feat_per_level = 2
    coarsest_scale = 0.1
    finest_scale = 0.001
    dim_out = 2
    num_rays_per_mlp = 10
    num_samples_per_ray = 128
    num_trials = 20
    num_mlps = 100
    num_queries = num_mlps * num_samples_per_ray * num_rays_per_mlp
    encoding = True
    with_grads = True
    set_to_none = True
    dtypes = (torch.float16, torch.float32, torch.float64)
    funcs = [single_model, for_loop, vmap]

    # init nets and prepare data
    scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
    assignments = torch.randint(high=num_mlps, size=(num_queries,))
    unique_assignments = assignments.unique()
    masks = unique_assignments[:, None] == assignments
    mlps = {}
    queries = {}
    params = {}
    buffers = {}
    unique_mlps = {}
    wrapper_model = {}
    wrappers = {}
    for dtype in dtypes:
        wrapper_model[dtype] = SimpleMLP(dtype=dtype)
        wrappers[dtype] = create_wrapper(dtype)
        mlps[dtype] = [
            SimpleMLP(dtype=dtype, encoding=encoding) for _ in range(num_mlps)
        ]
        queries[dtype] = torch.randn(num_queries, 3, dtype=dtype)
        params[dtype], buffers[dtype] = torch.func.stack_module_state(mlps[dtype])
        unique_mlps[dtype] = [mlps[dtype][i.item()] for i in unique_assignments]

    for func in funcs:
        for dtype in dtypes:
            benchmark(lambda: func(dtype, set_to_none), f"{dtype} {func.__name__}", 10)

    exit()
    avg_vmap = benchmark(
        vmap,
        "vmap (set_to_none==True)",
        num_trials,
        set_to_none=True,
    )

    vmap_speedup = avg_for / avg_vmap
    vmap_slowdown = avg_vmap / avg_ideal
    print(f"{vmap_speedup=} {vmap_slowdown=}")
