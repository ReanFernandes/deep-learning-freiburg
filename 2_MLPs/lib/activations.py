"""Activation modules."""

import numpy as np
import lib
from lib.network_base import Module


class Sigmoid(Module):
    """Sigmoid function module."""

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Apply logistic sigmoid activation function to all elements of a matrix.

        Args:
            z: The input matrix with arbitrary shape.

        Returns:
            Matrix with the activation function applied to all elements of the input matrix z.
        """
        # START TODO #################
        return 1 / (1 +np.exp(-z))
        # END TODO###################

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Apply sigmoid and cache the activation for later.

        Args:
            z: The input matrix with arbitrary shape.

        Returns:
            A np.ndarray with the activation function applied to all elements of the input matrix z.
        """
        h = self._sigmoid(z)
        # here it's useful to store the activation
        #  instead of the input
        self.input_cache = h

        return h

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the Sigmoid.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        # we do not need the backward pass yet
        raise NotImplementedError


class ReLU(Module):
    """ReLU function module."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Apply ReLU activation function to all elements of a matrix.

        Args:
            z: The input matrix with arbitrary shape.

        Returns:
            Matrix with the activation function applied to all elements of the input matrix z.
        """
        self.input_cache = z
        # START TODO #################
        # Use e.g. np.maximum to implement this function efficiently.
        return np.maximum(0,z)
        # END TODO###################

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of this module.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        # we do not need the backward pass yet
        raise NotImplementedError


class Softmax(Module):
    """Softmax module."""

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Apply the softmax function to convert the input logits to probabilities.

        Args:
            z: Input logits (raw output of a module) with shape (batch_size, num_classes).

        Returns:
            Matrix with shape (batch_size, num_classes), transformed such that the probabilities for each element
                in the batch sum to one.
        """
        # For correct results, the sum in the denominator of the softmax function must sum over all axes except the
        # batch axis. To achieve that, pass the keyword "axis=reduction_axes" to the numpy functions you use here.
        # To make the dimensions work out, you will either have to add "keepdims=True"
        # or use np.expand_dims after reduction operations.
        reduction_axes = tuple(range(1, len(z.shape)))
        # START TODO #################
        # First, shift the input for numerical stability by subtracting the maximum value of the input from all inputs.
        # This will not change the solution (since softmax(x) = softmax(x + c) for all scalars c),
        # but make the calculation numerically stable.
        
        z -= np.max(z, axis=reduction_axes, keepdims=True)

        h = np.exp(z) / np.sum(np.exp(z), axis=reduction_axes, keepdims=True)

        # END TODO# #################
        return h

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Apply the softmax function.

        Args:
            z: Input logits (raw output of a module) with shape (batch_size, num_classes).

        Returns:
            Matrix with shape (batch_size, num_classes), transformed such that the probabilities for each element
                in the batch sum to one.
        """
        h = self._softmax(z)
        return h

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of this module.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        # we do not need the backward pass yet
        raise NotImplementedError
