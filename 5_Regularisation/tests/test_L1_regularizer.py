import numpy as np

from lib.network_base import Parameter
from lib.regularizers import L1Regularization


def test_forward_pass():
    """Test the forward pass of L1 regularization."""
    np.random.seed(0)
    param_data = np.random.randn(50, 1) * 0.1
    param_data[6] = 0
    params = [Parameter(param_data)]
    l1 = L1Regularization(0.1, params)
    loss = l1.forward()

    err_msg = 'L1Regularization forward pass not implemented correctly'
    np.testing.assert_almost_equal(loss, 0.44749749800368427, decimal=5, err_msg=err_msg)


def test_gradient():
    """Test the gradient of L1 Regularization."""
    np.random.seed(0)
    param_data = np.random.randn(50, 1) * 0.1
    params = [Parameter(param_data)]
    L1Regularization(0.1, params).check_gradients_wrt_params((), 1e-6)


if __name__ == '__main__':
    test_forward_pass()
    test_gradient()
    print('Test complete.')
