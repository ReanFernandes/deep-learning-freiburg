import numpy as np

from lib.convolution_warmup import *


def test_pad():
    err_msg = 'pad_array() not implemented correctly'

    X = np.arange(18).reshape(1, 2, 3, 3)
    pad_output = pad_array(X, (2, 1))

    X_expected = np.array(
        [[[[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 2, 0],
           [0, 3, 4, 5, 0],
           [0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]],

          [[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 9, 10, 11, 0],
           [0, 12, 13, 14, 0],
           [0, 15, 16, 17, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]]]
    )

    np.testing.assert_equal(pad_output, X_expected, err_msg=err_msg)
    print('test_pad() passed.')


def test_dot_product_single_filter():
    X, _, f, _ = create_data_and_filter()
    err_msg = 'dot_product_single_filter() not implemented correctly'

    result = dot_product_single_filter(X, f, (2, 1))
    np.testing.assert_equal(result, 5718.0)
    print('test_dot_product_single_filter() passed.')


def test_dot_product_single_filter_simple():
    X, _, f, _ = create_simple_data_and_filter()
    err_msg = 'dot_product_single_filter() not implemented correctly'

    result = dot_product_single_filter(X, f, (2, 1))
    np.testing.assert_equal(result, 212.0)
    print('test_dot_product_single_filter_simple() passed.')


def test_dot_product_single_filter_with_batches():
    _, X2, f, _ = create_data_and_filter()
    err_msg = 'dot_product_single_filter_with_batches() not implemented correctly'

    result = dot_product_single_filter_with_batches(X2, f, (2, 1))
    np.testing.assert_equal(result, np.array([5718, 13368]))
    print('test_dot_product_single_filter_with_batches() passed.')


def test_dot_product_single_filter_with_batches_simple():
    _, X2, f, _ = create_simple_data_and_filter()
    err_msg = 'dot_product_single_filter_with_batches() not implemented correctly'

    result = dot_product_single_filter_with_batches(X2, f, (2, 1))
    np.testing.assert_equal(result, np.array([212., 612.]))
    print('test_dot_product_single_filter_with_batches_simple() passed.')


def test_dot_product_multiple_filters_with_batches():
    _, X2, _, f2 = create_data_and_filter()
    err_msg = 'dot_product_single_filter_with_batches() not implemented correctly'

    result = dot_product_multiple_filters_with_batches(X2, f2, (2, 1))
    np.testing.assert_equal(result, np.array([[5718, 15276, 24834],
                                              [13368, 39126, 64884]]))
    print('test_dot_product_multiple_filters_with_batches() passed.')


def test_dot_product_multiple_filters_with_batches_simple():
    _, X2, _, f2 = create_simple_data_and_filter()
    err_msg = 'dot_product_single_filter_with_batches() not implemented correctly'

    result = dot_product_multiple_filters_with_batches(X2, f2, (2, 1))
    np.testing.assert_equal(result, np.array([[212., 424., 636.],
                                              [612., 1224., 1836.]]))
    print('test_dot_product_multiple_filters_with_batches_simple() passed.')


def create_simple_data_and_filter():
    X = np.arange(50).reshape(2, 5, 5)
    X2 = np.arange(100).reshape(2, 2, 5, 5)
    f = np.ones((2, 2, 2))
    f2 = np.ones((3, 2, 2, 2))
    f2[1] *= 2
    f2[2] *= 3

    return X, X2, f, f2


def create_data_and_filter():
    X = np.arange(50).reshape(2, 5, 5)
    X2 = np.arange(100).reshape(2, 2, 5, 5)
    f = np.arange(18).reshape(2, 3, 3)
    f2 = np.arange(18 * 3).reshape(3, 2, 3, 3)

    return X, X2, f, f2


if __name__ == '__main__':
    test_pad()

    # The next three tests use trivial values, so it should help you solve it with pen and paper
    # to better understand the expected results
    test_dot_product_single_filter_simple()
    test_dot_product_single_filter_with_batches_simple()
    test_dot_product_multiple_filters_with_batches_simple()
    #
    # # Final tests
    test_dot_product_single_filter()
    test_dot_product_single_filter_with_batches()
    test_dot_product_multiple_filters_with_batches()

    print('Tests complete.')
