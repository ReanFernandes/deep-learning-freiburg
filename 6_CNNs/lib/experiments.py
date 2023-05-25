from typing import Tuple

import numpy as np
import os
from pathlib import Path
import pickle

from lib.activations import ReLU
from lib.convolutions import Conv2d, Flatten
from lib.dataset_mnist import load_mnist_data
from lib.losses import CrossEntropyLoss
from lib.model_training import train
from lib.network import Sequential, Linear
from lib.optimizers import SGD
from lib.network_base import Module
from lib.model_evaluation import evaluate
from lib.plot_images import plot_data

# define path to store the trained conv model
RESULTS_DIR = Path("results")
RESULTS_FILE = "conv_model.pkl"


def create_conv_model() -> Module:
    """Create a 3-layer convolutional model which will later be trained on the MNIST dataset.

    Returns:
        Convolutional model.
    """
    # setup model hyperparameters
    kernel_size = (4, 4)
    stride = (2, 2)
    padding = (1, 1)
    n_filters_conv1 = 5
    n_filters_conv2 = 10

    # START TODO ################
    # Create a 3-layer convolutional model.
    # model = Sequential(
    #     Conv2d( ... ),
    #     ReLU(),
    #     Conv2d( ... ),
    #     ReLU(),
    #     Flatten()
    #     Linear( ... ))
    # The model must run for the hard-coded hyperparameters given above
    conv_model = Sequential(
        Conv2d(in_channels=1, out_channels=n_filters_conv1,
               kernel_size=kernel_size, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=n_filters_conv1, out_channels=n_filters_conv2,
               kernel_size=kernel_size, stride=stride,
               padding=padding),
        ReLU(),
        Flatten(),
        Linear(490, 10))
    # END TODO ################
    return conv_model


def load_mnist_data_subset(max_datapoints: int):
    """Load MNIST data and create a smaller subset for faster training.

    Args:
        max_datapoints: How many datapoints to keep.

    Returns:
        6-Tuple of training, test, validation data and labels.
    """
    # load mnist data
    x_train, x_val, x_test, y_train, y_val, y_test = load_mnist_data()

    # reduce training dataset size
    print(f"Training set size will be reduced to {max_datapoints}")
    x_train, x_val, x_test, y_train, y_val, y_test = (
        data[:max_datapoints] for data in (x_train, x_val, x_test, y_train,
                                           y_val, y_test))
    return x_train, x_val, x_test, y_train, y_val, y_test


def run_conv_experiment(
        max_datapoints: int = 10000) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Module]:
    """Create a convolutional model and run MNIST on it.

    Args:
        max_datapoints: How many datapoints to keep.

    Returns:
        4-tuple of (train costs, train accuracies, evaluation costs, evaluation accuracies), each of which is
        a list with num_epochs entries.

    """
    # setup loss
    loss_fn = CrossEntropyLoss()

    # setup training hyperparameters
    num_epochs = 10
    batch_size = 50
    learning_rate = 0.01
    momentum = 0.9

    # create model
    conv_model = create_conv_model()

    # create optimizer for the model
    optimizer = SGD(conv_model.parameters(), lr=learning_rate,
                    momentum=momentum)

    # load training data
    x_train, x_val, x_test, y_train, y_val, y_test = load_mnist_data_subset(max_datapoints)

    # train the model
    train_results = train(conv_model, loss_fn, optimizer, x_train, y_train,
                          x_val, y_val, num_epochs=num_epochs,
                          batch_size=batch_size)

    # store weights
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with (RESULTS_DIR / RESULTS_FILE).open("wb") as fh:
        pickle.dump(conv_model, fh)

    # return training results
    return train_results


def run_shifted_conv_experiment(max_datapoints: int = 10000):
    loss_fn = CrossEntropyLoss()

    # load training data
    x_train, x_val, x_test, y_train, y_val, y_test = load_mnist_data_subset(max_datapoints)

    # shift validation data
    shift = 2
    # START TODO ################
    # shift the validation data by 'shift' pixels to the right and to the bottom
    # hint: you can use np.pad and slicing
    # x_val_shifted = ...
    x_val_sliced = x_val[:, :, :-shift, :-shift]
    x_val_shifted = np.pad(x_val_sliced, ((0, 0), (0, 0), (shift, 0), (shift, 0)))
    # END TODO ################

    # As always, visualizing your data is important
    # plot the shifted and non shifted data using the function you defined above
    plot_data(np.concatenate((x_val[:4], x_val_shifted[:4]), axis=0), rows=2, cols=4, plot_border=True,
              title="Validation and shifted validation data.")

    # setup training hyperparameters for the MLP
    num_epochs = 10
    batch_size = 50
    learning_rate = 0.05
    momentum = 0.9
    linear_units = 30
    # START TODO ################
    # Train a 2-layer MLP with the given hyper parameters. Remember to use a Relu activation for the hidden layer.
    # Use the non-shifted validation data for evaluation during training, to verify that the model trains well.
    # Note that you need to reshape the training, validation and shifted validation data to train an MLP.
    # In short: Create the model, reshape the data, create the optimizer (SGD) and train the model.
    mlp_model = Sequential(Linear(28 * 28, linear_units),
                           ReLU(),
                           Linear(linear_units, 10))
    x_train_mlp = x_train.reshape(x_train.shape[0], -1)
    x_val_mlp = x_val.reshape(x_val.shape[0], -1)
    x_val_shifted_mlp = x_val_shifted.reshape(x_val_shifted.shape[0], -1)
    # create optimizer for the model
    optimizer = SGD(mlp_model.parameters(), lr=learning_rate, momentum=momentum)
    # train the model
    train_results = train(mlp_model, loss_fn, optimizer, x_train_mlp, y_train, x_val_mlp, y_val,
                          num_epochs=num_epochs,
                          batch_size=batch_size)
    # END TODO ################

    # run the first convolution experiment again to get the trained convolution model.
    print("Reload convolutional model from disk.")
    try:
        with (RESULTS_DIR / RESULTS_FILE).open("rb") as fh:
            conv_model = pickle.load(fh)
    except FileNotFoundError:
        raise RuntimeError("Run the first experiment before you run the second experiment.")

    print("Accuracy on data using an MLP: ",
          evaluate(x_val_mlp, y_val, mlp_model, loss_fn, batch_size)[0])
    print("Accuracy on shifted data using an MLP: ",
          evaluate(x_val_shifted_mlp, y_val, mlp_model, loss_fn, batch_size)[0])
    print("Accuracy on data using convolutional model: ",
          evaluate(x_val, y_val, conv_model, loss_fn, batch_size)[0])
    print("Accuracy on shifted data using convolutional model: ",
          evaluate(x_val_shifted, y_val, conv_model, loss_fn, batch_size)[0])
