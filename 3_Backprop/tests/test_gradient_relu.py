import numpy as np

from lib.activations import ReLU


def test_gradient_relu():
    """Test ReLU gradient."""
    input_vector = np.random.uniform(-1., 1., size=(2, 10))
    ReLU().check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(4, 20))
    ReLU().check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(6, 40))
    ReLU().check_gradients((input_vector,))


if __name__ == '__main__':
    test_gradient_relu()
    print("Test complete.")
