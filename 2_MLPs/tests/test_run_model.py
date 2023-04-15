import numpy as np

from lib.models import Linear, run_model_on_xor


def test_run_model():
    """Test the run_model function with a linear model."""
    np.random.seed(42)  # fix seed so the output is consistent
    model = Linear(2, 2)
    pred_softmax, loss = run_model_on_xor(model, verbose=False)
    np.testing.assert_allclose(
        pred_softmax, [[0.5, 0.5], [0.49781166, 0.50218834], [0.50158744, 0.49841256], [0.49939909, 0.50060091]],
        err_msg="Incorrect softmax predictions returned from run_model function.")
    np.testing.assert_allclose(loss, 0.6931510155325383, err_msg="Incorrect loss returned from run_model function.")


if __name__ == '__main__':
    test_run_model()
    print("Test complete.")
