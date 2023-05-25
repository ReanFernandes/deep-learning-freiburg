import numpy as np

from lib.lr_schedulers import PiecewiseConstantLR
from lib.network import Linear
from lib.optimizers import SGD


def test_piecewise_lr():
    """Test PiecewiseConstantLR Module."""
    model = Linear(100, 20)
    optimizer = SGD(model.parameters(), lr=0.2)

    epochs = [1, 3, 5, 9]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    piecewise_scheduler = PiecewiseConstantLR(optimizer, epochs=epochs, learning_rates=learning_rates)
    optimizer_lrs = []

    for i in range(10):
        piecewise_scheduler.step()
        optimizer_lrs.append(piecewise_scheduler.optimizer.lr)

    err_msg = 'PiecewiseConstantLR is not implemented correctly'

    np.testing.assert_allclose(np.array(optimizer_lrs),
                               np.array([0.2, 0.1, 0.1, 0.01, 0.01, 0.001,
                                         0.001, 0.001, 0.001, 0.0001]),
                               rtol=1e-5,
                               err_msg=err_msg)


if __name__ == '__main__':
    test_piecewise_lr()
    print('Test complete.')
