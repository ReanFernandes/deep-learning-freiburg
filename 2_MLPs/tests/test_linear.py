import numpy as np

from lib.network import Linear


def test_linear_forward():
    """Create a linear model and check the forward pass."""
    np.random.seed(42)  # fix seed so the output is consistent
    model = Linear(2, 3)
    x = np.array([[1, 2], [-3, 12], [7, -5], [0, 0]])
    y = model(x)
    y_truth = np.array([[0.045428, 0.003934, 0.011794], [0.177862, -0.01395, -0.037527],
                        [-0.031382, 0.012029, 0.067045], [0.01, 0.01, 0.01]])
    np.testing.assert_allclose(y, y_truth, rtol=1e-4, err_msg="Linear model is not implemented correctly.")


def test_linear_parameters():
    """Create a linear module and check parameters."""
    np.random.seed(42)  # fix seed so the output is consistent
    params = Linear(2, 3).parameters()

    err_msg = "parameters function is not implemented correctly."
    for param in params:
        if param.name == 'W':
            np.testing.assert_allclose(
                    param.data,
                    [[0.00496714, -0.00138264, 0.00647689],
                     [0.0152303, -0.00234153, -0.00234137]],
                    rtol=1e-5, err_msg=err_msg)
        else:
            np.testing.assert_allclose(
                    param.data, [0.01, 0.01, 0.01],
                    rtol=1e-5, err_msg=err_msg)


if __name__ == '__main__':
    test_linear_forward()
    test_linear_parameters()
    print("Test complete.")
