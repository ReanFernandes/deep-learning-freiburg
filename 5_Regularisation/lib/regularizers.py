"""Regularization modules."""

import numpy as np

from typing import List
from lib.losses import CrossEntropyLoss
from lib.network_base import Module, Parameter


class Dropout(Module):
    """Set input elements to zero during training with probability p_delete.

    Args:
        p_delete: Probability of an element to be zeroed. Default: 0.5.
        fix_seed: If true, we always use the same seed in the forward pass.
            This is only needed for gradient checking and should only be
            set True for gradient checking.

    Notes:
         In the slides and the deep learning book, p describes the probability of an input being retained,
         while in this class (and PyTorch), p_delete describes the probability of an input being zeroed.
         The relationship is p_delete = 1 - p.
    """

    def __init__(self, p_delete: float = 0.5, fix_seed=False):
        """

        """
        super().__init__()
        self.p_delete = p_delete
        # Scale values up during training to compensate for the fact that a higher number of neurons is expected to
        # be active during validation and testing.
        self.scale = 1 / (1 - self.p_delete)
        self.fix_seed = fix_seed
        # Mask to denote which neurons are active and which not.
        # To be populated in the forward pass and used in the backward pass.
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout during training.

        Set values to zero with probability p during training
        and scale them by 1/(1-p). Returns identity during
        evaluation mode (--> self.training = False).

        Note: This layer should work with all kinds of input shapes.
        """
        if not self.training:
            return x
        if self.fix_seed:  # we need this for gradient and output checking
            np.random.seed(0)
        # START TODO ################
        # use numpy binomial function to populate mask
        # docs: (https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
        self.mask = (np.random.binomial(1, 1 - self.p_delete, x.shape)) * self.scale
        return self.mask * x
        # END TODO ################

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate backward pass of dropout.

        Args:
            grad: Gradient of the next layer.

        Returns:
            The gradient of this module.
        """
        if not self.training:
            raise ValueError("Model is set to evaluation mode.")
        # START TODO ################
        return self.mask * grad
        # END TODO ################


class L1Regularization(Module):
    """Module to calculate L1 Regularization loss.

    Args:
        lambd: Regularization strength.
        parameters: Model parameters to regularize.
    """

    def __init__(self, lambd: float, parameters: List[Parameter]):
        super().__init__()
        self.lambd = lambd
        self.params = parameters

    def forward(self) -> float:
        """Calculate forward pass of L1 regularization loss.

        Returns:
            Loss
        """
        # START TODO ################
        l_reg = 0
        for param in self.params:
            l_reg = self.lambd * np.sum(np.abs(param.data))
        return l_reg

        # END TODO ################

    def backward(self, _=None) -> None:
        """Calculate the backward pass of L1 regularization loss.
        Forward has no inputs, so update the gradient of the parameters (self.params) directly.

        Args:
            _: Unused gradient, we introduce the argument to have a unified interface with
                other Module objects. This simplifies code for gradient checking.
                We don't need this arg since there will not be a layer after the loss layer.

        Returns:
            None.
        """
        # START TODO ################
        for param in self.params:
            param.grad[param.data > 0] = self.lambd
            param.grad[param.data < 0] = -self.lambd

        # END TODO ################

    def parameters(self) -> List[Parameter]:
        """Return the module parameters.

        Returns:
            List of module Parameters
        """
        return [p for p in self.params]


class L2Regularization(Module):
    """Module to calculate L2 Regularization loss.

    Args:
        lambd: Regularization strength.
        parameters: Model parameters to regularize.
    """

    def __init__(self, lambd: float, parameters: List[Parameter]):
        super().__init__()
        self.lambd = lambd
        self.params = parameters

    def forward(self) -> np.ndarray:
        """Calculate forward pass of L2 regularization loss.

        Returns:
            Loss
        """
        # START TODO ################
        l2_reg = 0
        for param in self.params:
            sum_data = 0.5 * (np.sum(param.data ** 2))
            l2_reg += sum_data
        return l2_reg * self.lambd

        # END TODO ################

    def backward(self, _=None) -> None:
        """Calculate the backward pass of L2 regularization loss.
        Forward has no inputs, so update the gradient of the parameters (self.params) directly.

        Args:
            _: Unused gradient, we introduce the argument to have a unified interface with
                other Module objects. This simplifies code for gradient checking.
                We don't need this arg since there will not be a layer after the loss layer.

        Returns:
            None
        """
        # START TODO ################
        for param in self.params:
            param.grad = self.lambd * param.data
        # END TODO ################

    def parameters(self) -> List[Parameter]:
        """Return the module parameters.

        Returns:
            List of module Parameters
        """
        return [p for p in self.params]


class RegularizedCrossEntropy(Module):
    """Combines cross-entropy loss and regularization loss by summing them.

    Args:
        regularization_loss: Regularization loss module.
    """

    def __init__(self, regularization_loss: Module):
        super().__init__()
        self.reg_loss = regularization_loss
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate forward pass of regularized cross-entropy.

        Args:
            a: Prediction logits.
            y: True labels.

        Returns:
            Sum of cross-entropy loss and regularization loss.
        """
        return self.cross_entropy(a, y) + self.reg_loss()

    def backward(self, _=None) -> np.ndarray:
        """Calculate backward pass of regularized cross-entropy.

        Args:
            _: Unused gradient, we introduce the argument to have a unified interface with
                other Module objects. This simplifies code for gradient checking.
                We don't need this arg since there will not be a layer after the loss layer.

        Returns:
            The gradient of this module.
        """
        self.reg_loss.backward(_)  # this updates parameter gradients, no grad w.r.t input
        return self.cross_entropy.backward()
