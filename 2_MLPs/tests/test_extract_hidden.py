import numpy as np

from lib.activations import ReLU
from lib.models import Linear, Sequential
from lib.model_utilities import extract_hidden


def test_extract_hidden():
    """Test the extract_hidden function with a 2-layer MLP"""
    np.random.seed(42)  # fix seed so the output is consistent
    linear_units = 2
    model = Sequential(Linear(2, linear_units), ReLU(), Linear(linear_units, 2))
    x = np.array([[1, 2], [-3, 12], [1, 2], [-3, 12]])
    h = extract_hidden(model, x)
    np.testing.assert_equal(h, [[3, 2], [9, 8], [3, 2],
                                [9, 8]], err_msg="Extract_hidden function is not implemented correctly.")


if __name__ == '__main__':
    test_extract_hidden()
    print("Test complete.")
