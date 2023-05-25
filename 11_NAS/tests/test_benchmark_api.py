""" Test for benchmark_api.py """
import numpy as np
from lib.benchmark_api import NB201BenchmarkAPI, Metric, Dataset
import os

arch_str = '|nor_conv_1x1~0|+|none~0|nor_conv_1x1~1|+|none~0|nor_conv_1x1~1|nor_conv_1x1~2|'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_cifar10_benchmark_api():
    api = NB201BenchmarkAPI(os.path.join(ROOT_DIR, 'test_data/nb201_test_data.pickle'), Dataset.CIFAR10)
    train_accs = api.query(arch_str, Metric.TRAIN_ACCURACY, last_epoch_only=False)

    assert len(train_accs) == 200
    expected_train_accs = [
        92.54799997558594,
        92.63599999023438,
        92.49599998291016,
        92.35999998291015,
        92.54400001708984]
    np.testing.assert_allclose(train_accs[-5:], expected_train_accs, rtol=1e-5)

    train_losses = api.query(arch_str, Metric.TRAIN_LOSS, last_epoch_only=False)

    assert len(train_losses) == 200
    expected_train_losses = [
        0.225907910490036,
        0.2237298686504364,
        0.22357741302490233,
        0.2252167712020874,
        0.2237737162733078]
    np.testing.assert_allclose(train_losses[-5:], expected_train_losses, rtol=1e-5)

    eval_accs = api.query(arch_str, Metric.VAL_ACCURACY, last_epoch_only=False)

    assert len(eval_accs) == 201
    expected_eval_accs = [83.52800001220703, 83.54000001464844, 83.6040000048828, 83.56000000732422, 83.68]
    np.testing.assert_allclose(eval_accs[-5:], expected_eval_accs, rtol=1e-5)

    eval_losses = api.query(arch_str, Metric.VAL_LOSS, last_epoch_only=False)

    assert len(eval_losses) == 201
    expected_eval_losses = [
        0.5172317409133911,
        0.5182065695762634,
        0.5172186483573914,
        0.518911274394989,
        0.522079783821106]
    np.testing.assert_allclose(eval_losses[-5:], expected_eval_losses, rtol=1e-5)

    train_acc = api.query(arch_str, Metric.TRAIN_ACCURACY, last_epoch_only=True)
    np.testing.assert_almost_equal(train_acc, 92.54400001708984, decimal=6)

    train_loss = api.query(arch_str, Metric.TRAIN_LOSS, last_epoch_only=True)
    np.testing.assert_almost_equal(train_loss, 0.2237737162733078, decimal=6)

    eval_acc = api.query(arch_str, Metric.VAL_ACCURACY, last_epoch_only=True)
    np.testing.assert_almost_equal(eval_acc, 83.68, decimal=6)

    eval_loss = api.query(arch_str, Metric.VAL_LOSS, last_epoch_only=True)
    np.testing.assert_almost_equal(eval_loss, 0.522079783821106, decimal=6)

    params = api.query(arch_str, Metric.PARAMS)
    np.testing.assert_almost_equal(params, 0.185306, decimal=6)

    train_time = api.query(arch_str, Metric.TRAIN_TIME)
    np.testing.assert_almost_equal(train_time, 10.446996112664541, decimal=6)


if __name__ == '__main__':
    test_cifar10_benchmark_api()
    print('Test complete.')
