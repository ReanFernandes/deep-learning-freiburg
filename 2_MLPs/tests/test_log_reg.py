import numpy as np

from lib.log_reg import logistic_regression


def test_log_reg():
    """Test Logistic Regression given some input and labels"""
    np.random.seed(42)  # fix seed so the output is consistent

    # create a solvable problem with a linear decision boundary
    x = np.array([[-1, -3], [-3, -4], [-12, -5], [-1, -2], [2, 1], [5, 2], [7, 6], [2, 3]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # solve it with linear regression and check the solution
    p, s = logistic_regression(x, y)
    err_msg = "Logistic regression is not implemented correctly."
    np.testing.assert_allclose(p, y, err_msg=err_msg)
    assert isinstance(s, float), f"Score return by logistic regression is not a float, but a {type(s)}"
    np.testing.assert_allclose(s, 1.0, err_msg=err_msg)


if __name__ == '__main__':
    test_log_reg()
    print("Test complete.")
