import numpy as np

from lib.losses import CrossEntropyLoss
from lib.utilities import one_hot_encoding


def test_gradient_celoss():
    """Test gradient of Cross-Entropy loss."""
    pred = one_hot_encoding(np.array([1, 2]), 3)
    ground_truth = one_hot_encoding(np.array([1, 1]), 3)
    input_args = tuple([pred, ground_truth])
    CrossEntropyLoss().check_gradients(input_args)


if __name__ == '__main__':
    test_gradient_celoss()
    print("Test complete.")
