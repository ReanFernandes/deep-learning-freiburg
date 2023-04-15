import numpy as np

from lib.models import run_model_on_xor, create_2unit_net, create_3unit_net


def test_2unit_3unit_model_output():
    """Test whether the output of the 2 unit model and 3 unit model is the same"""
    model_2unit = create_2unit_net()
    model_3unit = create_3unit_net()
    pred_softmax_2unit, loss_2unit = run_model_on_xor(model_2unit, verbose=False)
    pred_softmax_3unit, loss_3unit = run_model_on_xor(model_3unit, verbose=False)
    np.testing.assert_allclose(pred_softmax_2unit, pred_softmax_3unit, err_msg="Output of the two models should be the"
                                                                               "same")


if __name__ == '__main__':
    test_2unit_3unit_model_output()
    print("Test complete.")
