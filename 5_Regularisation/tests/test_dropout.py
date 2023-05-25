import numpy as np

from lib.regularizers import Dropout


def test_forward_pass():
    """Test Dropout module forward pass."""
    err_msg = "Dropout is not implemented correctly."

    # create non-zero inputs
    x = np.random.rand(10) + 2

    # check if dropout can delete all inputs
    eps = 1e-16
    y = Dropout(p_delete=1 - eps)(x)
    np.testing.assert_almost_equal(y, np.zeros_like(y), err_msg=err_msg)

    # check if dropout can retain all inputs
    y = Dropout(p_delete=0)(x)
    np.testing.assert_almost_equal(y, x, err_msg=err_msg)

    # check scaling
    x = np.ones(10) * 2
    y = Dropout(0.363)(x)
    np.testing.assert_almost_equal(y[y > 0], 3.13971743, err_msg=err_msg)

    # check for a hardcoded array and p_delete = 0.4
    x = np.array([0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898])
    y = Dropout(0.4, fix_seed=True)(x)
    y_expected = np.array([0.44092602, 0., 0., 0.94738991, 0.03131633])
    np.testing.assert_almost_equal(y, y_expected, err_msg=err_msg)


def test_gradient():
    """Test the gradient of the Dropout module."""
    x = np.random.rand(1, 1, 4, 4)
    Dropout(fix_seed=True).check_gradients((x,))


if __name__ == '__main__':
    test_forward_pass()
    test_gradient()
    print('Test complete.')
