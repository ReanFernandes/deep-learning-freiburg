import numpy as np

from lib.lr_schedulers import CosineAnnealingLR
from lib.network import Linear
from lib.optimizers import SGD


def test_cosine_lr():
    """Test CosineAnnealingLR Module."""
    model = Linear(100, 20)
    optimizer = SGD(model.parameters(), lr=0.2)

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=10)
    optimizer_lrs = []

    for i in range(10):
        cosine_scheduler.step()
        optimizer_lrs.append(cosine_scheduler.optimizer.lr)

    assert np.isclose(np.array(optimizer_lrs), np.array([0.1951056, 0.1809016, 0.1587785, 0.1309016, 0.1,
                                                         0.0690983, 0.0412214, 0.0190983, 0.0048943, 0.0])).all()

    err_msg = 'CosineAnnealingLR is not implemented correctly'
    np.testing.assert_allclose(np.array(optimizer_lrs),
                               np.array([0.1951056, 0.1809016, 0.1587785, 0.1309016, 0.1,
                                         0.0690983, 0.0412214, 0.0190983, 0.0048943, 0.0]),
                               rtol=1e-5,
                               err_msg=err_msg)


if __name__ == '__main__':
    test_cosine_lr()
    print('Test complete.')
