import numpy as np

from lib.network import Linear


def test_gradient_linear():
    """Test gradients of linear layer."""
    input_vector = np.random.uniform(-1., 1., size=(2, 10))
    Linear(10, 30).check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(4, 20))
    Linear(20, 40).check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(6, 40))
    Linear(40, 60).check_gradients((input_vector,))


if __name__ == '__main__':
    test_gradient_linear()
    print("Test complete.")
