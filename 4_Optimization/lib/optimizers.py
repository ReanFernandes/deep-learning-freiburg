"""Optimizer classes."""

from typing import Optional, List, Dict, Tuple

import numpy as np

from lib.network_base import Parameter


class Optimizer:
    """The base class for optimizers.

    All optimizers must implement a step() method that updates the parameters.
    The general optimization loop then looks like this:

    for inputs, targets in dataset:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    `zero_grad` initializes the gradients of the parameters to zero. This
    allows to accumulate gradients (instead of replacing it) during
    backpropagation, which is e.g. useful for skip connections.

    Args:
        params: The parameters to be optimized.
        lr: Optimizer learning rate.
    """

    def __init__(self, params: List[Parameter], lr: float = 1.0):
        self._params = params
        self.lr = lr

    def step(self) -> None:
        """Update the parameters."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Clear the gradients of all optimized parameters."""
        for param in self._params:
            assert isinstance(param, Parameter)
            param.grad = np.zeros_like(param.data)


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer with optional Momentum.

    Args:
            params: List of parameters of model to optimize.
            lr: Learning rate.
            momentum: Momentum factor to optionally use with SGD.
    """

    def __init__(self, params: List[Parameter], lr: float = 1.0,
                 momentum: Optional[float] = None):
        super().__init__(params, lr=lr)
        self.momentum = momentum
        if self.momentum:
            for param in self._params:
                param.state_dict["momentum"] = np.zeros_like(param.data)

    def step(self):
        for p in self._params:
            if self.momentum:
                # update the momentum
                p.state_dict["momentum"] *= self.momentum
                p.state_dict["momentum"] -= self.lr * p.grad
                # update the parameter
                p.data += p.state_dict["momentum"]
            else:
                p.data -= self.lr * p.grad


class Adam(Optimizer):
    """Adam Optimizer.

        Args:
            params: List of parameters of model to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages of gradient and its square.
            eps: Term added to the denominator to improve numerical stability
    """

    def __init__(self, params: List[Parameter], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08):
        super().__init__(params, lr=lr)
        # we stick to the pytorch API, the variable names corresponding
        # to the DL book are given in the comments
        # learning_rate is called epsilon in the DL book and alpha in the lecture
        self.betas = betas  # betas are called rho (decay rates) in the DL book and lecture
        self.eps = eps  # eps is called delta in the DL book and lecture
        self.t = 0
        for param in self._params:
            # first order moment variables, called m in the paper
            param.state_dict["s"] = np.zeros_like(param.data)
            # second order moment variables, called v in the paper
            param.state_dict["r"] = np.zeros_like(param.data)

    def step(self) -> None:
        """Update the parameters and decaying averages of past gradients."""
        # START TODO ################

        eps, learning_rate, t = self.eps, self.lr, self.t
        beta1, beta2 = self.betas[0], self.betas[1]
        t = t + 1
        for p in self._params:
            m, v = p.state_dict["s"], p.state_dict["r"]
            m = beta1 * m + (1 - beta1) * p.grad  # momentum
            mt = m / (1 - beta1 ** t)  # bias correction
            v = beta2 * v + (1 - beta2) * (p.grad * p.grad)  # RMS prop
            vt = v / (1 - beta2 ** t)  # bias correction
            p.data -= learning_rate * mt / (np.sqrt(vt) + eps)
            # update values
            p.state_dict["s"], p.state_dict["r"], self.t = m, v, t

    # END TODO###################


def create_optimizer(name: str, params: List[Parameter], hyperparams: Dict[str, float]) -> Optimizer:
    """Helper function to create optimizers.

    Args:
        name: Name of the optimizer (adam or sgd).
        params: Model parameters to optimize.
        hyperparams: Hyperparameters for the optimizer (lr, momentum etc.) as a Dictionary.

    Returns:
        Optimizer for the model.
    """
    # Double star expression is used to convert the dictionary to keyword arguments for the class constructors.
    if name.lower() == "sgd":
        return SGD(params, **hyperparams)
    elif name.lower() == "adam":
        return Adam(params, **hyperparams)
    else:
        raise ValueError(f"Optimizer name {name} unknown.")
