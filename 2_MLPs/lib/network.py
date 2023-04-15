"""Basic network modules."""

import numpy as np
from typing import List

from lib.network_base import Module, Parameter


class Linear(Module):
    """Linear layer module.

    Args:
        in_features: Number of input channels
        out_features: Number of output channels
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        w_data = 0.01 * np.random.randn(in_features, out_features)
        self.W = Parameter(w_data, name="W")

        b_data = 0.01 * np.ones(out_features)
        self.b = Parameter(b_data, name="b")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass calculation for the linear module.

        Args:
            x: Input data with shape (batch_size, in_features)

        Returns:
            Output data with shape (batch_size, out_features)
        """
        assert len(x.shape) == 2, (
            "x.shape should be (batch_size, input_size)"
            " but is {}.".format(x.shape))
        self.input_cache = x
        # START TODO #################
        # Remember: Access weight data through self.W.data
        z = x @ self.W.data + self.b.data
        # END TODO ##################
        return z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the Linear module.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        # we do not need the backward pass yet
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        """Return module parameters.

        Returns:
            List of all learnable parameters of the linear module.
        """
        # START TODO #################
        # Return all parameters of Linear
        return [self.W, self.b]
        # END TODO ##################


class Sequential(Module):
    """A sequential container to stack modules.

    Modules will be added to it in the order they are passed to the
    constructor.

    Example network with one hidden layer:
    model = Sequential(
                  Linear(5,10),
                  ReLU(),
                  Linear(10,10),
                )

    Args:
        *args: Arbitrary number of parameters, one module each.
    """

    def __init__(self, *args: Module):
        super().__init__()
        self.modules = args

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Calculate the forward pass of the sequential container.

        Args:
            Input data, shape depends on first module in the container.

        Returns:
            Output data, shape depends on all modules in the container.
        """
        # START TODO #################
        # Remember: module(x) is equivalent to module.forward(x)
        for module in self.modules:
            x = module(x)
        # END TODO ##################
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the sequential container.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        # we do not need the backward pass yet
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        """Return the module parameters.

        Returns:
            List of module Parameters.
        """
        # iterate over modules and retrieve their parameters, iterate over
        # parameters to flatten the list
        return [param for module in self.modules
                for param in module.parameters()]
