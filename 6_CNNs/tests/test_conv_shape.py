import numpy as np

from lib.convolutions import calculate_conv_output


def test_conv_output():
    err_msg = "Convolution output shape calculation is not implemented correctly."
    type_err_msg = "Elements of out of calculate_conv_output must be of type int."

    output = calculate_conv_output(input_shape=[11, 11], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0])
    np.testing.assert_equal(isint(output), True, err_msg=type_err_msg)
    np.testing.assert_equal(output, (9, 9), err_msg=err_msg)

    output = calculate_conv_output(input_shape=[11, 11], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
    np.testing.assert_equal(isint(output), True, err_msg=type_err_msg)
    np.testing.assert_equal(output, (11, 11), err_msg=err_msg)

    output = calculate_conv_output(input_shape=[11, 11], kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
    np.testing.assert_equal(isint(output), True, err_msg=type_err_msg)
    np.testing.assert_equal(output, (6, 6), err_msg=err_msg)


def isint(x):
    for item in x:
        if not isinstance(item, int):
            return False
    return True


if __name__ == '__main__':
    test_conv_output()
    print('Test complete.')
