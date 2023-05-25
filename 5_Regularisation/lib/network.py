"""Basic network modules."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

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
        # Access weight data through self.W.data
        z = x @ self.W.data + self.b.data
        return z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the Linear module.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        x = self.input_cache
        assert self.W.grad is not None and self.b.grad is not None, "Gradients are none. Forgot to use zero_grad?"
        self.W.grad += x.T @ grad
        self.b.grad += np.sum(grad, axis=0)
        return grad @ self.W.data.T

    def parameters(self) -> Tuple[Parameter, Parameter]:
        """Return module parameters.

        Returns:
            All learnable parameters of the linear module.
        """
        # Return all parameters of Linear
        return self.W, self.b


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
    """

    def __init__(self, *args: Module):
        super().__init__()
        self.modules = args

    def forward(self, x: np.ndarray) -> np.ndarray:
        for module in self.modules:
            x = module(x)  # equivalent to module.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self) -> List[Parameter]:
        # iterate over modules and retrieve their parameters, iterate over
        # parameters to flatten the list
        return [param for module in self.modules
                for param in module.parameters()]

    def train(self, mode: bool = True) -> Sequential:
        """Set the train mode of the Sequential module and it's sub-modules.

        This only affects some modules, e.g., Dropout.

        Returns:
            self.
        """
        for module in self.modules:
            module.train(mode)
        return self
