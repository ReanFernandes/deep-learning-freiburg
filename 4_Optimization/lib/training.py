"""Helper functions for model training."""

from typing import Tuple, List, Dict

import numpy as np

from lib.activations import ReLU
from lib.dataset_mnist import load_mnist_data
from lib.losses import CrossEntropyLoss
from lib.model_evaluation import accuracy, evaluate
from lib.network import Linear, Sequential
from lib.network_base import Module
from lib.optimizers import create_optimizer, Optimizer
from lib.utilities import save_result


def train(model: Module, loss_fn: Module, optimizer: Optimizer, x_train: np.ndarray, y_train: np.ndarray,
          x_val: np.ndarray, y_val: np.ndarray, num_epochs: int, batch_size: int, scheduler=None
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Train a model which can be arbitrary in its architecture and optimizer.

    Args:
        model: Model to train.
        loss_fn: Loss function to use.
        optimizer: Optimizer to use.
        x_train: Train data.
        y_train: Train labels.
        x_val: Validation data.
        y_val: Validation labels.
        num_epochs: Number of epochs.
        batch_size: Batch size.
        scheduler: Learning rate scheduler to use.

    Returns:
        4-tuple of (train costs, train accuracies, evaluation costs, evaluation accuracies), each of which is
        is an array with shape (num_epochs,).

    """
    # initialize arrays to store losses and accuracies
    train_costs, train_accuracies = np.zeros(num_epochs), np.zeros(num_epochs)
    eval_costs, eval_accuracies = np.zeros(num_epochs), np.zeros(num_epochs)

    # calculate how many minibatches we get out of the training dataset given
    # the batch size
    num_train_batches = len(x_train) // batch_size
    assert len(x_train) % batch_size == 0, (
        f"Training dataset size of {len(x_train)} is not divisible by batch size {batch_size}.")

    # create indices of the training data to shuffle it later
    train_idx = np.arange(len(x_train))

    for epoch in range(num_epochs):
        print("Epoch {} / {}:".format(epoch + 1, num_epochs))
        training_predictions = []

        # shuffle training data order
        np.random.shuffle(train_idx)
        x_train_shuffled = x_train[train_idx]
        y_train_shuffled = y_train[train_idx]

        # train for one epoch
        model.train()
        for batch_num in range(num_train_batches):
            # get the minibatch data given the current minibatch number
            minibatch_start = batch_num * batch_size
            minibatch_end = (batch_num + 1) * batch_size
            x_batch = x_train_shuffled[minibatch_start:minibatch_end]
            y_batch = y_train_shuffled[minibatch_start:minibatch_end]

            # zero gradients
            optimizer.zero_grad()

            # do the forward pass, remember the predictions and calculate the
            # loss
            y_batch_predicted = model(x_batch)
            training_predictions.append(y_batch_predicted)
            loss = loss_fn(y_batch_predicted, y_batch)

            # do the backward pass
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()

            # aggregate loss for this epoch
            train_costs[epoch] += loss

        # normalize loss over number of steps done
        train_costs[epoch] /= num_train_batches

        # step the LR scheduler if needed
        if scheduler:
            scheduler.step()

        # set model to evaluation mode
        model.eval()

        # concatenate the minibatch training predictions back together and
        # calculate the accuracy
        training_predictions = np.concatenate(training_predictions, axis=0)
        train_accuracies[epoch] = accuracy(
            y_train_shuffled, training_predictions)
        print("  Training Accuracy: {:.4f}".format(train_accuracies[epoch]))
        print("  Training Cost: {:.4f}".format(train_costs[epoch]))

        # evaluate
        eval_accuracies[epoch], eval_costs[epoch] = evaluate(
            x_val, y_val, model, loss_fn, batch_size)
        print("  Eval Accuracy: {:.4f}".format(eval_accuracies[epoch]))
    return train_costs, train_accuracies, eval_costs, eval_accuracies


def create_and_train_model(opt_name: str, opt_hyperparams: Dict, x_train: np.ndarray, y_train: np.ndarray,
                           x_val: np.ndarray, y_val: np.ndarray, batch_size: int = 50, num_epochs: int = 10
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Define a Sequential model.

    Args:
        opt_name: Name of the optimizer (adam or sgd).
        opt_hyperparams: Hyperparameters for the optimizer (lr, momentum etc.) as a Dictionary.
        x_train: Train dataset.
        y_train: Train set labels.
        x_val: Validation dataset.
        y_val: Validation set labels.
        batch_size: Batch size to use in training.
        num_epochs: Number of epochs to train.

    Returns:
        A 4-tuple (train costs, train accuracies, evaluation costs, evaluation accuracies), each of which
        is an array with shape (num_epochs,).

    """

    hidden_units = 30

    # START TODO ################
    # Create a model with one hidden layer, comprising of 30 units and a ReLU
    # Create the loss and optimizer objects
    # Use CrossEntropyLoss for loss
    # Use the create_optimizer function to create the optimizer
    # Use train function from above to train the model and return the 4-tuple
    # Please note, the mnist dataset consists of 28x28 images for 10 different classes (0,...,9)
    # Images have therefore been flattened to a 784 vector
    model = Sequential(Linear(28 * 28, hidden_units),
                       ReLU(),
                       Linear(hidden_units, 10))
    loss = CrossEntropyLoss()
    optimizer = create_optimizer(opt_name, model.parameters(), opt_hyperparams)
    return train(model=model,
                 loss_fn=loss,
                 optimizer=optimizer,
                 x_train=x_train, y_train=y_train,
                 x_val=x_val, y_val=y_val,
                 num_epochs=num_epochs,
                 batch_size=batch_size)

    # END TODO ###################


def compare_optimizers() -> Dict[str, Tuple[List[float], List[float], List[float], List[float]]]:
    """Train models with the different optimizers and compile the results into a dictionary.

    Returns:
        Dictionary with the costs and accuracies during training and evaluation using different optimizers.

        The dict should have keys 'sgd', 'momentum' and 'adam' and as values
        4-tuples (train costs, train accuracies, evaluation costs, evaluation accuracies), each of which
        is an array with shape (num_epochs,).
    """
    sgd_learning_rate = 0.1
    momentum = 0.9
    adam_learning_rate = 0.01

    x_train, x_val, x_test, y_train, y_val, y_test = load_mnist_data()

    # START TODO ################
    # Compile the results of training a model with different
    # optimizers into a dictionary with keys 'sgd', 'sgd_momentum' and 'adam'
    # Use create_and_train_model function above to train individual models
    # optim_results = {
    #    "sgd": create_and_train_model("sgd", {"lr": sgd_learning_rate}, ... ),
    #    "sgd_momentum": ...,
    #    "adam": ...,
    # }
    optim_results = {
        "sgd": create_and_train_model("sgd", {"lr": sgd_learning_rate}, x_train, y_train, x_val, y_val),
        "sgd_momentum": create_and_train_model("sgd", {"lr": sgd_learning_rate, "momentum": momentum},
                                               x_train, y_train, x_val, y_val),
        "adam": create_and_train_model("adam", {"lr": adam_learning_rate}, x_train, y_train, x_val, y_val)}
    # END TODO ###################
    save_result('optimizers_comparison', optim_results)
    return optim_results
