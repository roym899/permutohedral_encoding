import torch

from permutohedral_encoding import find_cpp_package

_C = find_cpp_package.find_package()


class PermutoEncodingFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        lattice,
        features,
        positions,
        anneal_window,
        require_features_grad,
        require_positions_grad,
    ):
        # add batch dimension if not already there
        added_batch_dim = False
        if positions.dim() == 2:
            added_batch_dim = True
            positions = positions.unsqueeze(0)
            features = features.unsqueeze(0)

        assert positions.dim() == 3
        assert features.dim() == 4

        # print("forward features", features.shape)
        # print("forward positions", positions.shape)

        # forward
        input_struct = _C.EncodingInput(
            features,
            positions,
            anneal_window,
            require_features_grad,
            require_positions_grad,
        )
        outs = lattice.forward(input_struct)

        # remove batch dimension again if it was added
        if added_batch_dim:
            outs = outs[0]

        return outs

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        """Setup the context from inputs and outputs.

        Args:
            ctx: Context object of this function. Passed as first argument to backward.
            inputs: Tuple of inputs to the forward function.
            output: Tuple of outputs of the forward function.
        """
        (
            lattice,
            features,
            positions,
            anneal_window,
            require_features_grad,
            require_positions_grad,
        ) = inputs

        outs = output

        # save for back
        ctx.lattice = lattice
        ctx.require_features_grad = require_features_grad
        ctx.require_positions_grad = require_positions_grad
        ctx.save_for_backward(positions, anneal_window, features, outs)

    @staticmethod
    def vmap(
        info,
        in_dims: int,
        lattice,
        features: torch.Tensor,
        positions: torch.Tensor,
        anneal_window: torch.Tensor,
        require_features_grad: bool,
        require_positions_grad: bool,
    ):
        # ensure in_dims is as expected
        assert in_dims[0] is None
        assert in_dims[3] is None
        assert in_dims[4] is None
        assert in_dims[5] is None
        assert in_dims[1] == 0, "vmapping is only supported over first dimension."
        assert in_dims[2] == 0, "vmapping is only supported over first dimension."

        # call kernel
        outs = PermutoEncodingFunc.apply(
            lattice,
            features,
            positions,
            anneal_window,
            features.requires_grad and torch.is_grad_enabled(),
            positions.requires_grad and torch.is_grad_enabled(),
        )
        out_dims = 0
        return outs, out_dims

    @staticmethod
    def backward(ctx, grad_outs):
        # Unpack saved_tensors
        positions, anneal_window, features, _ = ctx.saved_tensors

        # This is required to support double backward
        # (see https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)
        return PermutoEncodingFuncBack.apply(
            ctx.lattice,
            features,
            positions,
            anneal_window,
            ctx.require_features_grad,
            ctx.require_positions_grad,
            grad_outs,
        )


# NOTE in order to enable a double backward like in
#  https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
class PermutoEncodingFuncBack(torch.autograd.Function):
    @staticmethod
    def forward(
        lattice,
        features: torch.Tensor,
        positions: torch.Tensor,
        anneal_window: torch.Tensor,
        require_features_grad: bool,
        require_positions_grad: bool,
        grad_outs: torch.Tensor,
    ):
        """Return the gradient of each input of PermutoEncodingFunc.

        Args:
            forward_ctx: Context object of PermutoEncodingFunc.
            grad_outs: Gradients of the output of the forward function.

        Returns:
            Tuple of gradients of the inputs of PermutoEncodingFunc.
            None for non-differentiable inputs / non-tensor inputs.
        """
        # add batch dimension if not already there
        added_batch_dim = False
        if positions.dim() == 2:
            added_batch_dim = True
            positions = positions.unsqueeze(0)
            features = features.unsqueeze(0)
            grad_outs = grad_outs.unsqueeze(0)

        assert positions.dim() == 3
        assert features.dim() == 4
        assert grad_outs.dim() == 4

        input_struct = _C.EncodingInput(
            features,
            positions,
            anneal_window,
            require_features_grad,
            require_positions_grad,
        )

        assert (
            input_struct.m_require_features_grad
            or input_struct.m_require_positions_grad
        ), (
            "We cannot perform the backward function on the slicing because we did not "
            "precompute the required tensors in the forward pass. To enable this, set "
            "the model.train(), set torch.set_grad_enabled(True) and make "
            "features have required_grad=True"
        )

        # print("back features", features.shape)
        # print("back positions", positions.shape)
        # print("back grad_outs", grad_outs.shape)

        grad_outs = grad_outs.contiguous()
        grad_features, grad_positions = lattice.backward(input_struct, grad_outs)

        # remove batch dimension again if it was added
        if added_batch_dim:
            grad_features = grad_features[0]
            grad_positions = grad_positions[0]

        # One output for each input of PermutoEncodingFunc.forward
        return (
            None,  # lattice
            grad_features,  # features
            grad_positions,  # positions
            None,  # anneal_window
            None,  # require_features_grad
            None,  # require_positions_grad
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Setup the context from inputs and outputs.


        Args:
            ctx: Context object of this function. Passed as first argument to backward.
            inputs: Tuple of inputs to the forward function.
            output: Tuple of outputs of the forward function.
        """
        (
            lattice,
            features,
            positions,
            anneal_window,
            require_features_grad,
            require_positions_grad,
            grad_outs,
        ) = inputs

        _, grad_features, grad_positions, _, _, _ = output

        # save for back
        ctx.lattice = lattice
        ctx.require_features_grad = require_features_grad
        ctx.require_positions_grad = require_positions_grad
        ctx.save_for_backward(positions, anneal_window, features, grad_outs)

    @staticmethod
    def backward(
        ctx,
        dummy1,  # lattice
        grad_grad_features,  # features
        grad_grad_positions,  # positions
        dummy2,  # anneal_window
        dummy3,  # require_features_grad
        dummy4,  # require_positions_grad
    ):
        """Compute the derivative of PermutoEncodingFuncBack wrt to its inputs.

        Args:
            ctx: Context object of PermutoEncodingFuncBack.
            dummy1: Unused.
            grad_grad_features: dl2/d(dl1/df).
            grad_grad_positions: dl2/d(dl1/dp).
            dummy2: Unused.
            dummy3: Unused.
            dummy4: Unused.

        Returns:
            dummy1: Unused.
            grad_features: dl2/df.
            grad_positions: dl2/dp.
            dummy2: Unused.
            dummy3: Unused.
            dummy4: Unused.
            grad_grad_outs: dl2/d(dl1/do).
        """

        # unpack context
        (
            positions,
            anneal_window,
            features,
            grad_outs,
        ) = ctx.saved_tensors
        lattice = ctx.lattice
        require_features_grad = ctx.require_features_grad
        require_positions_grad = ctx.require_positions_grad

        # add batch dimension if not already there
        added_batch_dim = False
        if grad_grad_positions.dim() == 2:
            added_batch_dim = True
            grad_grad_features = grad_grad_features.unsqueeze(0)
            grad_grad_positions = grad_grad_positions.unsqueeze(0)
            positions = positions.unsqueeze(0)
            features = features.unsqueeze(0)
            grad_outs = grad_outs.unsqueeze(0)

        input_struct = _C.EncodingInput(
            features,
            positions,
            anneal_window,
            require_features_grad,
            require_positions_grad,
        )

        # print("double back features", features.shape)
        # print("double back positions", positions.shape)
        # print("double back grad_grad_features", grad_grad_features.shape)
        # print("double back grad_grad_positions", grad_grad_positions.shape)
        # print("double back grad_outs", grad_outs.shape)

        grad_outs = grad_outs.contiguous()

        (
            grad_features,
            grad_positions,
            grad_grad_outs,
        ) = lattice.double_backward(
            input_struct, grad_grad_positions, grad_grad_features, grad_outs
        )

        grad_features = grad_features.contiguous()

        if added_batch_dim:
            grad_features = grad_features[0]
            grad_positions = grad_positions[0]
            grad_grad_outs = grad_grad_outs[0]

        # One output for each input of PermutoEncodingFuncBack.forward
        return (
            None,  # lattice
            grad_features,  # features
            grad_positions,  # positions
            None,  # anneal_window
            None,  # require_features_grad
            None,  # require_positions_grad
            grad_grad_outs,  # grad_outs
        )
