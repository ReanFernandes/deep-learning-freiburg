import numpy as np

from lib.eigendecomp import get_euclidean_norm


def test_euclidean_norm():
    """Test euclidean norm."""
    test_vectors = [np.array([-7, 5]), np.array([0, 0, 0]), np.array([4, -4, 5, -8]), np.array([6]),
                    np.arange(100)]
    test_values = [8.60232527, 0, 11, 6, 573.0183243143276]
    for vec, val in zip(test_vectors, test_values):
        norm = get_euclidean_norm(vec)
        assert isinstance(norm, float), f"Norm should be a float but is {type(norm)}"
        np.testing.assert_allclose(norm, val, err_msg="Norm is not correctly implemented.")


if __name__ == '__main__':
    test_euclidean_norm()
    print("Test complete.")
