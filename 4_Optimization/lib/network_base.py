"""Network base classes to inherit from."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import scipy.optimize


class Parameter:
    """A trainable parameter.

    This class not only stores the value of the parameter (self.data) but also
    tensors / properties associated with it, such as the gradient (self.grad)
    of the current backward pass.

    Args:
        data: Parameter value with arbitrary shape
        grad: Parameter gradient, same shape as data.
        name: Parameter name
    """

    def __init__(self, data: np.ndarray, grad: Optional[np.ndarray] = None, name: Optional[str] = None):
        self.data = data
        self.grad = grad if grad is not None else np.zeros_like(data)
        self.name = name
        self.state_dict = dict()  # dict to store additional, optional information


class Module:
    """The base class all network modules must inherit from."""

    def __init__(self):
        # Cache of the input of the forward pass.
        # We need it during the backward pass in most layers,
        #  e.g., to compute the gradient w.r.t to the weights.
        self.input_cache = None
        self.training = True

    def __call__(self, *args) -> np.ndarray:
        """Alias for forward, convenience function.

        Args:
           args: The inputs, e.g., the output of the previous layer.
        """
        return self.forward(*args)

    def forward(self, *args) -> np.ndarray:
        """Compute the forward pass through the module.

        Args:
           args: The inputs, e.g., the output of the previous layer.

        Returns:
            Layer output.
        """
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute the backward pass through the module.

        This method computes the gradients with respect to the trainable parameters and with respect to the first input.
        If the module has trainable parameters, this method needs to update the respective parameter.grad property.

        Args:
            grad: The gradient of the following layer. For the final layers of the network (e.g. loss layers)
                this gradient is not needed.

        Returns:
            The gradient with respect to the first input argument. In general
            it might be useful to return the gradients w.r.t. to all inputs, we
            omit this here to keep things simple.
        """
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        """Return the module parameters.

        Returns:
            List of module Parameters
        """
        return []  # default to empty list

    def train(self, mode: bool = True) -> Module:
        """Set the module to training mode.

        This only affects some Modules, such as Dropout.

        Returns:
            self.
        """
        self.training = mode
        return self

    def eval(self) -> Module:
        """Set the module to evaluation mode.

        This only affects some Modules, such as Dropout.

        Returns:
            self.
        """
        return self.train(False)

    def check_gradients(self, input_args: Tuple[np.ndarray]) -> None:
        """Verify the implementation of the gradients.

        This includes the gradient with respect to the input as well as the
        gradients w.r.t. the parameters if the module contains any.

        As the scipy grad check only works on scalar functions, we compute
        the sum over the output to obtain a scalar.
        """
        assert isinstance(input_args, tuple), (
            "input_args must be a tuple but is {}".format(type(input_args)))
        TOLERANCE = 1e-6
        self.check_gradients_wrt_input(input_args, TOLERANCE)
        self.check_gradients_wrt_params(input_args, TOLERANCE)

    def _zero_grad(self) -> None:
        """(Re-) intialize the param's grads to 0. Helper for grad checking."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def check_gradients_wrt_input(self, input_args: Tuple[np.ndarray], tolerance: float) -> None:
        """Verify the implementation of the module's gradient w.r.t. input."""

        def output_given_input(x: np.ndarray):
            """Wrap self.forward for scipy.optimize.check_grad."""
            # we only compute the gradient w.r.t. to the first input arg.
            args = (x.reshape(input_args[0].shape),) + input_args[1:]
            return np.sum(self.forward(*args))

        def grad_given_input(x: np.ndarray):
            """Wrap self.backward for scipy.optimize.check_grad."""
            self._zero_grad()
            # run self.forward to store the new input
            args = (x.reshape(input_args[0].shape),) + input_args[1:]
            out = self.forward(*args)
            # compute the gradient w.r.t. to the input
            return np.ravel(self.backward(np.ones_like(out)))

        error = scipy.optimize.check_grad(
            output_given_input, grad_given_input, np.ravel(input_args[0]))
        num_outputs = np.prod(self.forward(*input_args).shape)
        if np.squeeze(error) / num_outputs > tolerance:
            raise RuntimeError("Check of gradient w.r.t. to input for {} failed."
                               "Error {:.4E} > {:.4E}."
                               .format(self, np.squeeze(error), tolerance))

    def check_gradients_wrt_params(self, input_args: Tuple[np.ndarray], tolerance: float) -> None:
        """Verify the implementation of the module's gradient w.r.t. params."""
        for param in self.parameters():
            def output_given_params(new_param: np.ndarray):
                """Wrap self.forward, change the parameters to new_param."""
                param.data = new_param.reshape(param.data.shape)
                return np.sum(self.forward(*input_args))

            def grad_given_params(new_param: np.ndarray):
                self._zero_grad()
                param.data = new_param.reshape(param.data.shape)
                out = self.forward(*input_args)
                # compute the gradient w.r.t. to param
                self.backward(np.ones_like(out))
                return np.ravel(param.grad)

            # check if parameter and gradient have the same shape
            assert param.data.shape == param.grad.shape, (
                f"Gradient shape {param.grad.shape} does not fit parameter shape {param.data.shape}.")

            # flatten the param as scipy can only handle 1D params
            param_init = np.ravel(np.copy(param.data))
            error = scipy.optimize.check_grad(output_given_params,
                                              grad_given_params,
                                              param_init)
            num_outputs = np.prod(self.forward(*input_args).shape)
            if np.squeeze(error) / num_outputs > tolerance:
                raise RuntimeError(
                    f"Check of gradient w.r.t. to param '{param.name}' for {self} failed. "
                    f"Error {error:.4e} > {tolerance:.4e}.")
