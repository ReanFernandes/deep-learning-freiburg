"""Activation modules."""

import numpy as np

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
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Apply sigmoid and cache the activation for later.

        Args:
            z: The input matrix with arbitrary shape.

        Returns:
            A np.ndarray with the activation function applied to all elements of the input matrix z.

        """
        h = self._sigmoid(z)
        # here it's useful to store the activation
        # instead of the input
        self.input_cache = h

        return h

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the Sigmoid.

        Args:
            grad: The gradient of the following layer.

        Returns:
            The gradient of this module.

        """
        h = self.input_cache
        return grad * h * (1 - h)


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
        # START TODO ################
        gradient = grad * np.maximum( 0, z/abs(z))
        # END TODO###################
        return gradient


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
        shift_z = shift_z.astype(np.float128)
        exps = np.exp(shift_z).astype(np.float128)
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

    def backward(self, grad: np.ndarray) -> None:
        """Calculate the backward pass of this module.

        Args:
            grad: The gradient of the following layer.

        Returns:
            None

        """
        error_msg = ("Softmax doesn't need to implement a gradient here, as it's"
                     "only needed in CrossEntropyLoss, where we can simplify"
                     "the gradient for the combined expression.")
        raise NotImplementedError(error_msg)
