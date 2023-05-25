"""Helper functions for model evaluation."""

from typing import Tuple

import numpy as np

from lib.network import Module


def accuracy(y: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate accuracy given the ground truths and predictions.

    Args:
        y: Target labels, not one-hot encoded.
        predictions: Predictions of the model.

    Returns:
        Accuracy.

    """
    y_predicted = np.argmax(predictions, axis=-1)
    y = np.argmax(y, axis=-1)
    return np.sum(np.equal(y_predicted, y)) / len(y)


def evaluate(x_val: np.ndarray, y_val: np.ndarray, model: Module, loss_fn: Module,
             batch_size: int) -> Tuple[float, float]:
    """Evaluate a model

    Args:
        x_val: Validation data.
        y_val: Validation labels.
        model: Model to evaluate.
        loss_fn: Loss function to use for evaluation.
        batch_size: Batch size.

    Returns:
        A 2-tuple of (evaluation accuracy, evaluation cost)

    """
    predictions = []
    eval_cost = 0.

    # calculate how many minibatches we get out of the validation dataset given the batch size
    num_val_batches = len(x_val) // batch_size
    assert len(x_val) % batch_size == 0, (
        f"Training dataset size of {len(x_val)} is not divisible by batch size {batch_size}.")

    for batch_num in range(num_val_batches):
        # get the minibatch data given the current minibatch number
        minibatch_start = batch_num * batch_size
        minibatch_end = (batch_num + 1) * batch_size
        x_batch = x_val[minibatch_start:minibatch_end]
        y_batch = y_val[minibatch_start:minibatch_end]

        # note that when using cross entropy loss, the softmax is included in the
        # loss and we'd need to apply it manually here to obtain the output as probabilities.
        # However, softmax only rescales the outputs and doesn't change the argmax,
        # so we'll skip this here, as we're only interested in the class prediction.
        y_predicted = model(x_batch)
        predictions.append(y_predicted)
        eval_cost += loss_fn(y_predicted, y_batch)

    # concatenate the minibatch validation predictions back together and calculate the accuracy
    predictions = np.concatenate(predictions, axis=0)
    eval_accuracy = accuracy(y_val, predictions)

    # return accuracy and loss for this epoch
    return eval_accuracy, eval_cost
