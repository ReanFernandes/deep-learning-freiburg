"""Functions for gradient checking and zeroing."""

from typing import Tuple

import numpy as np

from lib.activations import ReLU, Sigmoid
from lib.losses import CrossEntropyLoss
from lib.network import Linear, Sequential
from lib.network_base import Parameter
from lib.utilities import one_hot_encoding


def check_gradients() -> None:
    """Check the gradients of the individual modules and sequential network.

       This includes the gradient with respect to the input as well as the
        gradients w.r.t. the parameters if the module contains any.
    """
    input_vector = np.random.uniform(-1., 1., size=(2, 10))
    input_args = (input_vector,)

    # layers + activations
    ReLU().check_gradients(input_args)
    Sigmoid().check_gradients(input_args)
    Linear(10, 30).check_gradients(input_args)

    # START TODO ################
    # Instantiate a Sequential network with layers: linear, sigmoid, linear and
    # perform the gradient check on it.
    model = Sequential(Linear(2,10),
                        Sigmoid(),
                        Linear(10,2))
    # END TODO ##################

    # losses
    input_args_losses = tuple([one_hot_encoding(np.array([1, 2]), 3),  # a
                               one_hot_encoding(np.array([1, 1]), 3)])  # y (ground truth)
    CrossEntropyLoss().check_gradients(input_args_losses)


def zero_grad(params: Tuple[Parameter, Parameter]) -> None:
    """Clear the gradients of all optimized parameters."""
    # START TODO ################
    params[0].grad = np.zeros_like(params[0].grad)
    params[1].grad = np.zeros_like(params[1].grad)
    # END TODO ##################
