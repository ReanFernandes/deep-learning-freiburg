"""Model definitions."""

import numpy as np
from typing import Tuple

from lib.activations import ReLU, Softmax
from lib.dataset import X, y
from lib.losses import CrossEntropyLoss
from lib.network import Sequential, Linear
from lib.network_base import Module
from lib.utilities import one_hot_encoding


def create_2unit_net() -> Module:
    """Create a two-layer MLP (1 hidden layer, 1 output layer) with 2 hidden units as described in the exercise.

    Returns:
        2-layer MLP module with 2 hidden units.
    """
    # START TODO #################
    # Define the model here
    model = Sequential(Linear(X.shape[1], 2),
                       ReLU(),
                       Linear(2, 2))
    # END TODO ##################

    # START TODO #################
    # Change the model weights according to the assignment sheet
    # Hint: You can access the parameters using model.parameters() which contains a list of Parameter objects
    # e.g. model.parameters()[0].data contains the weights of the first layer
    model.parameters()[0].data = np.array([[3.21, -2.34],[3.21, -2.34]])    # First layer weight
    model.parameters()[1].data = np.array([-3.21, 2.34])                    # First layer bias
    model.parameters()[2].data = np.array([[3.19, -2.68],[4.64, -3.44]])    # Second layer weight
    model.parameters()[3].data = np.array([-4.08, 4.42])                    # Second Layer bias
    # END TODO ##################

    return model


def create_3unit_net() -> Module:
    """Create a two-layer MLP (1 hidden layer, 1 output layer) with 3 hidden units as described in the exercise.

    Returns:
        2-layer MLP module with 3 hidden units.
    """
    # START TODO #################
    model = Sequential(Linear(X.shape[1], 3),
                       ReLU(),
                       Linear(3, 2))
    # END TODO ##################

    # START TODO #################
    # Change the model weights according to the assignment sheet
    # Hint: You can access the parameters using model.parameters() which contains a list of Parameter objects
    # e.g. model.parameters()[0].data contains the weights of the first layer
    model.parameters()[0].data = np.array([[3.21, -2.34],[3.21, -2.34]])    # First layer weight
    model.parameters()[1].data = np.array([-3.21, 2.34])                    # First layer bias
    model.parameters()[2].data = np.array([[3.19, -2.68],[4.64, -3.44]])    # Second layer weight
    model.parameters()[3].data = np.array([-4.08, 4.42])                    # Second Layer bias
    # END TODO ##################

    return model

    


def run_model_on_xor(model: Module, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """Run the XOR dataset through the model and compute the loss.

    Args:
        model: MLP to use for prediction
        verbose: Whether to print the outputs.

    Returns:
        Tuple containing:
            Class predictions after softmax with shape (batch_size, num_classes)
            Cross-Entropy loss given the model outputs and the true labels

    """
    # Here we test if our prediction works. We first get the so-called "logits" (the MLP output before the softmax),
    # then run them through the softmax function. We have to transform the prediction into one-hot format,
    # and finally we can check whether our MLP predicts the correct values.

    # START TODO #################
    # propagate the input data (stored in the imported variable X) through the model.
    prediction = model(X)
    # END TODO ##################

    if verbose:
        print("Raw prediction logits:")
        print(prediction)
        print()
    softmax_function = Softmax()
    pred_softmax = softmax_function(prediction)
    if verbose:
        print("Prediction after softmax:")
        print(pred_softmax)
        print()

    # START TODO #################
    # Use the one_hot_encoding function (imported from file lib/utilites.py) on the labels to convert them to
    # one-hot encoding. The labels are stored in the imported variable y.
    Y_onehot = one_hot_encoding(y,len(np.unique(y)))
    # END TODO ##################

    if verbose:
        print("True labels, one-hot encoded:")
        print(Y_onehot)
        print()

    # Pass prediction and ground truth to the generalized Cross-Entropy Loss.
    # Hint: Since the loss has a softmax already implemented inside of it, you need to pass the raw logits of the
    # prediction. The loss expects one-hot encoded labels of shape (batchsize, num_classes)
    loss_fn = CrossEntropyLoss()

    # START TODO #################
    # given the true labels Y and the predictions,
    # compute the cross entropy loss defined above
    loss = loss_fn(prediction, Y_onehot)
    # END TODO ##################

    if verbose:
        print("Loss:", loss)

    # return predictions and loss for testing
    return pred_softmax, loss


def run_test_model(model: Module) -> None:
    """Helper function to test if the model predicts the correct classes.

    Args:
        model: Module to predict the classes.

    Returns:
        None
    """
    pred_softmax, loss = run_model_on_xor(model, verbose=False)
    Y_onehot = one_hot_encoding(y, 2)
    np.testing.assert_allclose(
        pred_softmax, Y_onehot, atol=1e-3,
        err_msg=f"The model predicts the wrong classes. Ground-truth: {Y_onehot}, predictions: {pred_softmax}")
    assert np.abs(loss) < 1e-3, f"Loss is too high: {loss}"
