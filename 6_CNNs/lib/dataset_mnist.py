import gzip
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from lib.utilities import one_hot_encoding

MNIST_URL = "https://github.com/automl-classroom/dl2020-data/blob/master/mnist.pkl.gz?raw=true"


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST data from file, download from the internet if not available.

    Returns:
        A 6-tuple of (train data, validation data, test data, train labels, validation labels, test labels).
    """
    data_path = Path("../data/mnist.pkl.gz")

    # download file if necessary
    if not data_path.is_file():
        print(f"File {data_path} not found, downloading...")
        import urllib.request as request
        with request.urlopen(MNIST_URL) as req:
            data = req.read()

        os.makedirs("../data", exist_ok=True)
        with open(data_path, 'wb') as f:
            f.write(data)
        print("Downloaded")

    # load data from downloaded file
    with gzip.open(data_path, 'rb') as f:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = pickle.load(f, encoding='latin1')

    # setup labels
    num_classes = 10
    y_train = one_hot_encoding(y_train, num_classes)
    y_val = one_hot_encoding(y_val, num_classes)
    y_test = one_hot_encoding(y_test, num_classes)

    # reshape input data as images
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_val = x_val.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    # print data shape
    print("Train:\tX {}\ty {}\nVal:\tX {}\ty {}\nTest:\tX {}\ty {}".format(
        x_train.shape, y_train.shape, x_val.shape, y_val.shape,
        x_test.shape, y_test.shape))

    return x_train, x_val, x_test, y_train, y_val, y_test
