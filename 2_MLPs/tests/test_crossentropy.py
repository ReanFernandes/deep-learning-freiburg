import numpy as np

from lib.losses import CrossEntropyLoss


def test_crossentropy():
    """Test cross-entropy loss on an example."""
    preds = np.array([[1, 2], [3, 4]])
    labels = np.array([[0, 1], [1, 0]])
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(preds, labels)
    assert isinstance(loss, float), f"Returned Cross-Entropy-Loss is not a float, but a {type(loss)}"
    np.testing.assert_allclose(loss, 0.8132616875182228, err_msg="Cross-Entropy-Loss is not implemented correctly.")


if __name__ == '__main__':
    test_crossentropy()
    print("Test complete.")
