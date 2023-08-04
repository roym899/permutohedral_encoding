import torch

from permutohedral_encoding import find_cpp_package

_C = find_cpp_package.find_package()


class PermutoEncodingFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        lattice,
        lattice_values,
        positions,
        anneal_window,
        require_lattice_values_grad,
        require_positions_grad,
    ):
        added_batch_dim = False

        # add batch dimension if not already there
        if positions.dim() == 2:
            added_batch_dim = True
            positions = positions.unsqueeze(0)
        if lattice_values.dim() == 3:
            lattice_values = lattice_values.unsqueeze(0)

        # forward
        input_struct = _C.EncodingInput(
            lattice_values,
            positions,
            anneal_window,
            require_lattice_values_grad,
            require_positions_grad,
        )
        sliced_values = lattice.forward(input_struct)

        # remove batch dimension again if it was added
        if added_batch_dim:
            sliced_values = sliced_values[0]

        return sliced_values

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            lattice,
            lattice_values,
            positions,
            anneal_window,
            require_lattice_values_grad,
            require_positions_grad,
        ) = inputs

        input_struct = _C.EncodingInput(
            lattice_values,
            positions,
            anneal_window,
            require_lattice_values_grad,
            require_positions_grad,
        )

        sliced_values = output

        # save for back
        ctx.lattice = lattice
        ctx.input_struct = input_struct
        ctx.save_for_backward(sliced_values)

    @staticmethod
    def vmap(
        info,
        in_dims,
        lattice,
        lattice_values,
        positions,
        anneal_window,
        require_lattice_values_grad,
        require_positions_grad,
    ):
        # ensure in_dims is as expected
        assert in_dims[0] is None
        assert in_dims[3] is None
        assert in_dims[4] is None
        assert in_dims[5] is None
        assert in_dims[1] == 0, "vmapping is only supported over first dimension."
        assert in_dims[2] == 0, "vmapping is only supported over first dimension."

        # call kernel
        sliced_values = PermutoEncodingFunc.apply(
            lattice,
            lattice_values,
            positions,
            anneal_window,
            lattice_values.requires_grad and torch.is_grad_enabled(),
            positions.requires_grad and torch.is_grad_enabled(),
        )
        out_dims = 0
        return sliced_values, out_dims

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_sliced_values_monolithic):
        # restore from ctx
        lattice = ctx.lattice
        input_struct = ctx.input_struct
        (sliced_values,) = ctx.saved_tensors

        assert (
            input_struct.m_require_lattice_values_grad
            or input_struct.m_require_positions_grad
        ), (
            "We cannot perform the backward function on the slicing because we did not "
            "precompute the required tensors in the forward pass. To enable this, set "
            "the model.train(), set torch.set_grad_enabled(True) and make "
            "lattice_values have required_grad=True"
        )

        # for now do not support double backward
        grad_sliced_values_monolithic = grad_sliced_values_monolithic.contiguous()

        ctx.save_for_backward(grad_sliced_values_monolithic)
        ctx.lattice = lattice
        ctx.input_struct = input_struct

        lattice_values_grad, positions_grad = lattice.backward(
            input_struct, grad_sliced_values_monolithic
        )

        return None, lattice_values_grad, positions_grad, None, None, None

        # NOTE we pass the tensors of lattice_values and positiosn explicitly and not
        #  throught the input struct so that we can compute gradients from them for the
        #  double backward pass
        return PermutoEncodingFuncBack.apply(
            lattice,
            input_struct,
            grad_sliced_values_monolithic,
            input_struct.m_lattice_values,
            input_struct.m_positions_raw,
            sliced_values,
        )


# NOTE in order to enable a double backward like in
#  https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
class PermutoEncodingFuncBack(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        lattice,
        input_struct,
        grad_sliced_values_monolithic,
        lattice_values,
        positions,
        sliced_values_hom,
    ):
        lattice_values_grad = None
        positions_grad = None

        if (
            input_struct.m_require_lattice_values_grad
            or input_struct.m_require_positions_grad
        ):
            grad_sliced_values_monolithic = grad_sliced_values_monolithic.contiguous()

            ctx.save_for_backward(grad_sliced_values_monolithic)
            ctx.lattice = lattice
            ctx.input_struct = input_struct

            lattice_values_grad, positions_grad = lattice.backward(
                input_struct, grad_sliced_values_monolithic
            )

        return None, lattice_values_grad, positions_grad, None, None, None

    @staticmethod
    def backward(
        ctx,
        dummy1,
        double_lattice_values_grad,
        double_positions_grad,
        dummy5,
        dummy6,
        dummy7,
    ):
        # NOTE in the forward pass of this module we do
        #  lattice_values_grad, positions_grad = slice_back(
        #     lattice_values_monolithic, grad_sliced_values_monolithic, positions)
        #  now in the backward pass we have the upstream gradient which is
        #  double_lattice_values_grad, double_positions_grad
        #  we want to propagate the double_positions_grad into lattice_values_monolithic
        #  and grad_sliced_values_monolithic

        (grad_sliced_values_monolithic,) = ctx.saved_tensors
        lattice = ctx.lattice
        input_struct = ctx.input_struct

        (
            grad_lattice_values_monolithic,
            grad_grad_sliced_values_monolithic,
        ) = lattice.double_backward_from_positions(
            input_struct, double_positions_grad, grad_sliced_values_monolithic
        )

        return (
            None,
            None,
            grad_grad_sliced_values_monolithic,
            grad_lattice_values_monolithic,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
