import numpy as np
import os
from lib.convolutions import Conv2d


def torch_validate_conv2d(conv, x, conv_name):
    y_torch = np.load(os.path.join(os.path.dirname(__file__), conv_name + '.npy'))
    try:
        y_pred = conv(x)
        np.testing.assert_allclose(y_pred, y_torch, rtol=10e-4,
                                   err_msg="Forward pass implementation is not correct")
        print("Done")
    except Exception as e:
        raise ValueError("forward pass not OK, error", e)


def test_conv_output():
    """Test convolution layer."""
    np.random.seed(9001)
    x = np.random.rand(2, 3, 11, 11)
    # standard conv
    conv = Conv2d(3, 10, (3, 3))
    # conv with strides
    conv_stride = Conv2d(3, 10, (3, 3), stride=(2, 2))
    # conv with padding
    conv_pad = Conv2d(3, 10, (3, 3), padding=(1, 1))
    # conv with padding and strides
    conv_stride_pad = Conv2d(3, 10, (3, 3), stride=(2, 2), padding=(1, 1))

    convs = [conv, conv_stride, conv_pad, conv_stride_pad]
    convs_names = ['conv', 'conv_stride', 'conv_pad', 'conv_stride_pad']
    for conv, conv_name in zip(convs, convs_names):
        print("Check implementation for {} ... ".format(conv))
        torch_validate_conv2d(conv, x, conv_name)
    print("All implementations checked.")


def test_x_cached():
    x = np.arange(16).reshape(1, 1, 4, 4)
    conv = Conv2d(1, 1, (3, 3))
    conv(x)
    assert (conv.input_cache == x).all()
    print('test_x_cached() passed.')


def test_x_cached_padded():
    x = np.arange(16).reshape(1, 1, 4, 4)
    conv_pad = Conv2d(1, 1, (3, 3), padding=(1, 1))
    conv_pad(x)
    x_padded = np.array([[[[0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 2, 3, 0],
                           [0, 4, 5, 6, 7, 0],
                           [0, 8, 9, 10, 11, 0],
                           [0, 12, 13, 14, 15, 0],
                           [0, 0, 0, 0, 0, 0]]]])

    assert (conv_pad.input_cache == x_padded).all()
    print('test_x_cached_padded() passed.')


if __name__ == '__main__':
    test_conv_output()
    test_x_cached()
    test_x_cached_padded()
    print('Tests complete.')
