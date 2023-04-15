import numpy as np

from lib.activations import Softmax


def test_softmax():
    """Test the Softmax function on an example."""
    softmax = Softmax()

    # test if the function was implemented correctly
    x = np.array([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
                  [-7.0, 12.0, 5.0, 1.0, 1.0, 5.0, 3.0]])
    np.testing.assert_allclose(
        softmax(x), [[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813],
                     [0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813],
                     [5.59172157e-09, 9.98023332e-01, 9.10079478e-04, 1.66686871e-05,
                      1.66686871e-05, 9.10079478e-04, 1.23165864e-04]],
        rtol=1e-5, err_msg="Softmax is not correct implemented.")

    # test if the shifting was implemented correctly
    x = np.array([[1.0, 2.0, 9999.0], [1.0, 2.0, 9999.0]])
    assert not np.any(np.isnan(softmax(x))), \
        "Your softmax is numerically unstable. Subtract the maximum value of the input from the input before " \
        "calculating the softmax. This will not change the solution (softmax(x) = softmax(x + c) for all scalars c) " \
        "but make the calculation numerically stable."


if __name__ == '__main__':
    test_softmax()
    print("Test complete.")
