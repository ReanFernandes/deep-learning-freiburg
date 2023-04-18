from lib.gradient_utilities import check_gradients


def test_gradients():
    """Test gradients of various modules."""
    try:
        check_gradients()
    except TypeError:
        raise TypeError("Backward function did not return a gradient, instead returned None") from None


if __name__ == '__main__':
    test_gradients()
    print("Test complete.")
