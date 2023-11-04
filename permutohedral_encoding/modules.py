import math
from typing import Optional

import torch

from permutohedral_encoding import find_cpp_package, funcs, utils

_C = find_cpp_package.find_package()


class PermutoEncoding(torch.nn.Module):
    def __init__(
        self,
        pos_dim,
        capacity,
        nr_levels,
        nr_feat_per_level,
        scale_per_level,
        apply_random_shift_per_level: bool = True,
        concat_points: bool = False,
        concat_points_scaling: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        init_scale: float = 1e-5
    ):
        super(PermutoEncoding, self).__init__()
        self.pos_dim = pos_dim
        self.capacity = capacity
        self.nr_levels = nr_levels
        self.nr_feat_per_level = nr_feat_per_level
        self.scale_per_level = scale_per_level
        self.apply_random_shift_per_level = apply_random_shift_per_level
        self.concat_points = concat_points
        self.concat_points_scaling = concat_points_scaling
        self.dtype = dtype

        # create hashmap values
        lattice_values = (
            torch.randn(capacity, nr_levels, nr_feat_per_level, dtype=dtype) * init_scale
        )
        # make it nr_levels x capacity x nr_feat
        lattice_values = lattice_values.permute(1, 0, 2).contiguous()
        self.lattice_values = torch.nn.Parameter(lattice_values.cuda())

        # each levels of the hashamp can be randomly shifted so that we minimize
        #  collisions
        if apply_random_shift_per_level:
            random_shift_per_level = torch.randn(nr_levels, pos_dim, dtype=dtype) * 10
            self.random_shift_per_level = torch.nn.Parameter(
                random_shift_per_level.cuda()
            )  # we make it a parameter just so it gets saved when we checkpoint
        else:
            self.random_shift_per_level = torch.nn.Parameter(
                torch.empty((1), dtype=dtype).cuda()
            )
            raise NotImplementedError("No random shift is not implemented.")

        # make a anneal window of all ones
        self.anneal_window = torch.ones((nr_levels), dtype=dtype).cuda()

        # make the lattice wrapper
        self.fixed_params, self.lattice = self._make_lattice_wrapper()

    def _make_lattice_wrapper(self):
        fixed_params = _C.EncodingFixedParams(
            self.pos_dim,
            self.capacity,
            self.nr_levels,
            self.nr_feat_per_level,
            self.scale_per_level,
            self.random_shift_per_level,
            self.concat_points,
            self.concat_points_scaling,
        )
        lattice = _C.EncodingWrapper.create(
            self.pos_dim, self.nr_feat_per_level, fixed_params
        )
        return fixed_params, lattice

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Avoid pickling native objects
        del state["fixed_params"]
        del state["lattice"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct native entries
        self.fixed_params, self.lattice = self._make_lattice_wrapper()

    def forward(self, positions, anneal_window=None):
        if anneal_window is None:
            anneal_window = self.anneal_window
        else:
            anneal_window = anneal_window.cuda()

        require_lattice_values_grad = (
            self.lattice_values.requires_grad and torch.is_grad_enabled()
        )
        require_positions_grad = positions.requires_grad and torch.is_grad_enabled()

        sliced_values = funcs.PermutoEncodingFunc.apply(
            self.lattice,
            self.lattice_values,
            positions,
            anneal_window,
            require_lattice_values_grad,
            require_positions_grad,
        )

        if sliced_values.dim() == 4:
            batch_size = sliced_values.shape[0]
            sliced_values = sliced_values.permute(0, 3, 1, 2).reshape(
                batch_size, len(positions), -1
            )  # from lvl, val, nr_positions to nr_positions x lvl x val
        elif sliced_values.dim() == 3:
            sliced_values = sliced_values.permute(2, 0, 1).reshape(
                len(positions), -1
            )  # from lvl, val, nr_positions to nr_positions x lvl x val

        return sliced_values

    def output_dims(self):
        # if we concat also the points, we add a series of extra resolutions to contain
        #  those points
        nr_resolutions_extra = 0
        if self.concat_points:
            nr_resolutions_extra = math.ceil(
                float(self.pos_dim) / self.nr_feat_per_level
            )

        out_dims = self.nr_feat_per_level * (self.nr_levels + nr_resolutions_extra)

        return out_dims


# coarse2fine  which slowly anneals the weights of a vector of size nr_values. t is
#  between 0 and 1
class Coarse2Fine(torch.nn.Module):
    def __init__(self, nr_values):
        super(Coarse2Fine, self).__init__()

        self.nr_values = nr_values
        self.last_t = 0

    def forward(self, t):
        alpha = (
            t * self.nr_values
        )  # because cosine_easing_window except the alpha to be in range 0, nr_values
        window = utils.cosine_easing_window(self.nr_values, alpha)

        self.last_t = t
        assert t <= 1.0, "t cannot be larger than 1.0"

        return window

    def get_last_t(self):
        return self.last_t
