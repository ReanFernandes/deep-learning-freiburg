"""Experiments with optimizers."""

from typing import Tuple, Dict

import numpy as np

from lib.activations import ReLU
from lib.dataset_mnist import load_mnist_data
from lib.losses import CrossEntropyLoss
from lib.model_training import train
from lib.network import Linear, Sequential
from lib.network_base import Module
from lib.optimizers import SGD
from lib.regularizers import Dropout, L1Regularization, L2Regularization, RegularizedCrossEntropy
from lib.utilities import save_result


def build_model(linear_units: int, input_units: int = 784, num_classes: int = 10) -> Module:
    """Build a Sequential model with one hidden layer.

    Args:
        linear_units: Number of units in the hidden layer.
        input_units: Number of units in the input layer, must be 784 for the 28x28 pixel MNIST input.
        num_classes: Number of classes, must be 10 for MNIST.
    Returns:
        Sequential model with `linear_units` units in the hidden layer and ReLU activation after it.

    """
    # START TODO ################
    model = Sequential(Linear(input_units,
                              linear_units),
                       ReLU(),
                       Linear(linear_units, num_classes))
    return model
    # END TODO ################


def build_model_dropout(linear_units: int, p_delete_input: float = 0.1, p_delete_hidden: float = 0.2,
                        input_units: int = 784, num_classes: int = 10) -> Module:
    """Build a Sequential model with one hidden layer.
    The model should use Dropout on the input and the hidden layer.
    Dropout strength is given by the function's arguments.

    Args:
        linear_units: Number of units in the hidden layer.
        p_delete_input: Dropout parameter for input units.
        p_delete_hidden: Dropout parameter for hidden units.
        input_units: Number of units in the input layer, must be 784 for the 28x28 pixel MNIST input.
        num_classes: Number of classes, must be 10 for MNIST.
    Returns:
        Sequential model with `linear_units` units in the hidden layer, ReLU activation after it,
        and Dropout layers before each linear layer.

    """
    # START TODO ################
    model = Sequential(Dropout(p_delete=p_delete_input),
                       Linear(input_units, linear_units),
                       ReLU(),
                       Dropout(p_delete=p_delete_hidden),
                       Linear(linear_units, num_classes)
                       )
    return model
    # END TODO ################


def train_models() -> Tuple[Dict[str, Module], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Train four models.

    Train four model as follows: Without regularization, with L1 regularization, with L2 regularization,
    and using Dropout.

    Returns:
        2-tuple:
            1st Dictionary with keys 'before_training, 'no_reg', 'l1', 'l2' and 'dropout'
            and the respective trained models as values.
            2nd dictionary has the same keys except for 'before_training'
            and the results of function train(...) for the respective models as values.

    """
    np.random.seed(42)
    num_epochs = 20
    batch_size = 50
    learning_rate = 0.05
    momentum = 0.9
    linear_units = 30
    lambda_l1 = 0.0001
    lambda_l2 = 0.0001

    # Load MNIST data
    x_train, x_val, _, y_train, y_val, _ = load_mnist_data()

    models = {}  # dict to store the trained models
    results = {}  # dict to store the training results
    # let's save a model, which we won't train, to get the initial parameter distribution
    models["before_training"] = build_model(linear_units)

    # ---------- No regularization ----------
    print("No regularization.")
    # build model
    models["no_reg"] = build_model(linear_units)
    # start all model experiments from the same weight initialization
    _copy_parameters(models["before_training"], models["no_reg"])
    # create loss, optimizer and train the model
    cross_entropy = CrossEntropyLoss()
    optimizer = SGD(models["no_reg"].parameters(), lr=learning_rate, momentum=momentum)
    training_result = train(models["no_reg"], cross_entropy, optimizer, x_train, y_train,
                            x_val, y_val, num_epochs=num_epochs, batch_size=batch_size)
    # write back result
    results['no_reg'] = training_result

    # ---------- L2 regularization ----------
    print("L2 regularization.")
    # build model
    models["l2"] = build_model(linear_units)
    # start all model experiments from the same weight initialization
    _copy_parameters(models["before_training"], models["l2"])
    # get weight parameters for regularization
    params = [p for p in models["l2"].parameters() if "W" in p.name]
    # START TODO ################
    # create loss, optimizer and train the model
    loss = RegularizedCrossEntropy(L2Regularization(lambda_l2, params))
    optimizer = SGD(params=params, lr=learning_rate, momentum=momentum)
    training_result = train(models["l2"], loss, optimizer, x_train, y_train,
                            x_val, y_val, num_epochs, batch_size)
    # END TODO ################
    # write back result
    results['l2'] = training_result

    # ---------- L1 regularization ----------
    print("L1 regularization.")
    # build model
    models["l1"] = build_model(linear_units)
    # start all model experiments from the same weight initialization
    _copy_parameters(models["before_training"], models["l1"])
    # get weight parameters for regularization
    params = [p for p in models["l1"].parameters() if "W" in p.name]
    # START TODO ################
    # create loss, optimizer and train the model
    loss = RegularizedCrossEntropy(L1Regularization(lambda_l1, params))
    optimizer = SGD(params=params, lr=learning_rate, momentum=momentum)
    training_result = train(models["l1"], loss, optimizer, x_train, y_train,
                            x_val, y_val, num_epochs, batch_size)
    # END TODO ################
    # write back result
    results['l1'] = training_result

    # ---------- Dropout ----------
    print("Dropout.")
    # build dropout model
    models["dropout"] = build_model_dropout(linear_units)
    # start all model experiments from the same weight initialization
    _copy_parameters(models["before_training"], models["dropout"])
    # START TODO ################
    loss = CrossEntropyLoss()
    optimizer = SGD(models["dropout"].parameters(), learning_rate, momentum)
    training_result = train(models["dropout"], loss, optimizer, x_train, y_train,
                            x_val, y_val, num_epochs, batch_size)
    # END TODO ################
    # write back result
    results['dropout'] = training_result

    save_result('trained_models', models)
    save_result('trained_model_results', results)

    return models, results


def _copy_parameters(model1: Module, model2: Module) -> None:
    """Copy parameter data from model1 to model2. This is used in the exercise to make sure experiments are more
    comparable.

    Args:
        model1:
        model2:

    Returns:
        None
    """
    params1, params2 = model1.parameters(), model2.parameters()
    assert len(params1) == len(params2), "Models for parameter copying don't match."
    for p1, p2 in zip(params1, params2):
        assert p1.name == p2.name, "Models for parameter copying don't match."
        p2.data = np.copy(p1.data)
