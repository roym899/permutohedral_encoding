import numpy as np
import torch
import time
from torch.autograd.gradcheck import GradcheckError
from typing import Literal

import permutohedral_encoding
from permutohedral_encoding.funcs import PermutoEncodingFunc, PermutoEncodingFuncBack


def repeat_gradcheck(check_func, label=None, num_runs=10, **kwargs) -> None:
    start = time.time()
    if label is not None:
        print(label, end=": ", flush=True)
    for _ in range(num_runs):
        good = check_func(**kwargs)
        if good:
            print("✓", end="", flush=True)
        else:
            print("✗", end="", flush=True)
    end = time.time()
    avg_time = (end - start) / num_runs
    print(f" ({avg_time} s)")


def run_first_order_check():
    encoding = permutohedral_encoding.PermutoEncoding(
        pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=dtype
    )
    grad_outs = torch.rand(24, 2, 1, dtype=dtype).cuda()
    grad_outs.requires_grad = True
    x = torch.rand(1, pos_dim, dtype=dtype).cuda()
    x.requires_grad = True

    return torch.autograd.gradcheck(
        lambda features, pos: PermutoEncodingFunc.apply(
            encoding.lattice, features, pos, encoding.anneal_window, True, True
        ),
        (
            encoding.features,
            x,
        ),
        raise_exception=False,
    )


def run_full_second_order_check():
    encoding = permutohedral_encoding.PermutoEncoding(
        pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=dtype
    )
    grad_outs = torch.rand(24, 2, 1, dtype=dtype).cuda()
    grad_outs.requires_grad = True
    x = torch.rand(1, pos_dim, dtype=dtype).cuda()
    x.requires_grad = True

    return torch.autograd.gradgradcheck(
        lambda pos, features: PermutoEncodingFunc.apply(
            encoding.lattice, features, pos, encoding.anneal_window, True, True
        ),
        (
            x,
            encoding.features,
        ),
        raise_exception=False,
    )


def run_partial_second_order_check(
    input_name: Literal["features", "positions", "grad_outs"],
    output_name: Literal["grad_features", "grad_positions"],
):
    encoding = permutohedral_encoding.PermutoEncoding(
        pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=dtype
    )
    grad_outs = torch.rand(24, 2, 1, dtype=dtype).cuda()
    grad_outs.requires_grad = True
    x = torch.rand(1, pos_dim, dtype=dtype).cuda()
    x.requires_grad = True

    if output_name == "grad_features":
        out_index = 1
    elif output_name == "grad_positions":
        out_index = 2

    if input_name == "features":
        inputs = encoding.features
        func = lambda features: PermutoEncodingFuncBack.apply(
            encoding.lattice, features, x, encoding.anneal_window, True, True, grad_outs
        )[out_index]
    elif input_name == "positions":
        inputs = (x,)
        func = lambda pos: PermutoEncodingFuncBack.apply(
            encoding.lattice,
            encoding.features,
            pos,
            encoding.anneal_window,
            True,
            True,
            grad_outs,
        )[out_index]
    elif input_name == "grad_outs":
        inputs = (grad_outs,)
        func = lambda grad_outs: PermutoEncodingFuncBack.apply(
            encoding.lattice,
            encoding.features,
            x,
            encoding.anneal_window,
            True,
            True,
            grad_outs,
        )[out_index]

    return torch.autograd.gradcheck(
        func,
        inputs,
        raise_exception=False,
    )


if __name__ == "__main__":
    # create encoding
    num_runs = 5
    pos_dim = 3
    capacity = pow(2, 8)
    nr_levels = 24
    nr_feat_per_level = 2
    coarsest_scale = 0.1
    finest_scale = 0.001
    scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
    dtype = torch.float64

    repeat_gradcheck(run_first_order_check, "Singleback", num_runs=num_runs)

    repeat_gradcheck(run_full_second_order_check, "Doubleback", num_runs=num_runs)

    for input_name in ["features", "positions", "grad_outs"]:
        for output_name in ["grad_features", "grad_positions"]:
            repeat_gradcheck(
                lambda: run_partial_second_order_check(
                    input_name=input_name, output_name=output_name
                ),
                f"Partial doubleback in: {input_name}, outs: {output_name}",
                num_runs=num_runs,
            )
