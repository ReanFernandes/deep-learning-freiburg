import numpy as np

from lib.activations import ReLU


def test_relu():
    """Test the ReLU function on an example."""
    x = np.linspace(-4, +4, 9)
    relu = ReLU()
    y = relu(x)
    np.testing.assert_allclose(y, [0., 0., 0., 0., 0., 1., 2., 3., 4.], err_msg="ReLU is not implemented correctly.")


if __name__ == '__main__':
    test_relu()
    print("Test complete.")
