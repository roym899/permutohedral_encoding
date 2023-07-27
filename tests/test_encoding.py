"""Module to test permutohedral encoding.

Run this module as a script to generate new test files when sure the output is
correct necessary.

Currently only forward and single backward (both with float32) is tested.
"""
import pathlib

import numpy as np
import torch

import permutohedral_encoding as permuto_enc


def create_encoding() -> permuto_enc.PermutoEncoding:
    pos_dim = 3
    capacity = pow(2, 12)
    nr_levels = 24
    nr_feat_per_level = 2
    coarsest_scale = 1.0
    finest_scale = 0.0001
    scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
    return permuto_enc.PermutoEncoding(
        pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list
    )


@torch.no_grad()
def test_forward() -> None:
    """Test forward."""
    test_files_dir = pathlib.Path(__file__).parents[0] / "test_files"
    state_dict = torch.load(test_files_dir / "encoding_state_dict.pt")
    in_points = torch.load(test_files_dir / "in_points.pt")
    correct_out_features = torch.load(test_files_dir / "out_features.pt")

    encoding = create_encoding()
    encoding.load_state_dict(state_dict)
    out_features = encoding(in_points)
    assert torch.allclose(out_features, correct_out_features)


def test_lattice_grad() -> None:
    """Test lattice gradients."""
    test_files_dir = pathlib.Path(__file__).parents[0] / "test_files"
    state_dict = torch.load(test_files_dir / "encoding_state_dict.pt")
    in_points = torch.load(test_files_dir / "in_points.pt")
    correct_lattice_grad = torch.load(test_files_dir / "encoding_lattice_grad.pt")

    encoding = create_encoding()
    encoding.load_state_dict(state_dict)
    out_features = encoding(in_points)
    loss = out_features.sum()
    loss.backward()
    assert torch.allclose(
        encoding.lattice_values.grad, correct_lattice_grad
    )


def test_in_points_grad() -> None:
    """Test input gradients."""
    test_files_dir = pathlib.Path(__file__).parents[0] / "test_files"
    state_dict = torch.load(test_files_dir / "encoding_state_dict.pt")
    in_points = torch.load(test_files_dir / "in_points.pt")
    correct_in_points_grad = torch.load(test_files_dir / "in_points_grad.pt")

    in_points.requires_grad = True
    encoding = create_encoding()
    encoding.load_state_dict(state_dict)
    out_features = encoding(in_points)
    loss = out_features.sum()
    loss.backward()

    assert torch.allclose(in_points.grad, correct_in_points_grad)


if __name__ == "__main__":
    nr_points = 1000
    encoding = create_encoding()
    torch.save(encoding.state_dict(), "encoding_state_dict.pt")
    in_points = torch.rand(nr_points, 3).cuda()
    in_points.requires_grad = True
    out_features = encoding(in_points)
    loss = out_features.sum()
    loss.backward()
    torch.save(out_features, "out_features.pt")
    torch.save(in_points, "in_points.pt")
    torch.save(in_points.grad, "in_points_grad.pt")
    torch.save(encoding.lattice_values.grad, "encoding_lattice_grad.pt")
