import numpy as np

from lib.network import Sequential, Linear


def test_sequential():
    """Create a sequential model and check the output for an example."""
    np.random.seed(42)  # fix seed so the output is consistent
    model = Sequential(Linear(2, 3), Linear(3, 1))
    x = np.array([[1, 2], [-3, 12]])
    y = model(x)
    np.testing.assert_allclose(
        y, [[0.01069222], [0.01287794]], rtol=1e-5, err_msg="Linear layer is not implemented correctly.")


if __name__ == '__main__':
    test_sequential()
    print("Test complete.")
