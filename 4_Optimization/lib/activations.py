"""Activation modules."""

import numpy as np

from lib.network_base import Module


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
        return np.maximum(0, z)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of this module.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.
        """
        z = self.input_cache
        return grad.reshape(z.shape) * np.where(z > 0, 1, 0)


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
        # don't reduce (sum) over batch axis
        reduction_axes = tuple(range(1, len(z.shape)))

        shift_z = z - np.max(z, axis=reduction_axes, keepdims=True)
        exps = np.exp(shift_z)
        h = exps / np.sum(exps, axis=reduction_axes, keepdims=True)
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
        error_msg = ("Softmax doesn't need to implement a gradient here, as it's"
                     "only needed in CrossEntropyLoss, where we can simplify"
                     "the gradient for the combined expression.")
        raise NotImplementedError(error_msg)
