"""Create some random input for tests."""


import numpy as np

from typing import Tuple
from lib.utilities import one_hot_encoding


def create_dummy_data(train_batch_size: int = 1000, val_batch_size: int = 200, input_dim: int = 784,
                      output_dim: int = 10
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates dummy data with a fixed seed.

    Args:
        train_batch_size: Batch size of training data.
        val_batch_size: Batch size of validation data.
        input_dim: Input dimensions of data.
        output_dim: Dimension of output (number of classes).

    Returns:
        4-tuple of (train data, validation data, train labels, validation labels)

    """
    np.random.seed(42)
    x_train = np.random.rand(train_batch_size, input_dim)
    x_val = np.random.rand(val_batch_size, input_dim)
    y_train = one_hot_encoding(np.random.randint(0, output_dim, size=train_batch_size), output_dim)
    y_val = one_hot_encoding(np.random.randint(0, output_dim, size=val_batch_size), output_dim)

    return x_train, x_val, y_train, y_val
