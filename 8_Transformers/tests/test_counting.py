import numpy as np
import torch

from lib.counting import CountingModel


def test_counting_model_forward():
    torch.manual_seed(0)

    batch_size = 2
    d_in = 2
    model = CountingModel(d_in, d_in - 1, 2)
    x = torch.rand(batch_size, 1, d_in)
    out, attention_weights = model(x)

    expected_out = np.array(
        [[[0.4014813, 0.7585268],
          [0.11432427, 0.41835368]],

         [[0.40151885, 0.758571],
          [0.38857868, 0.74324495]]]
    )
    expected_weights = np.array(
        [[[0.589777], [0.51559067]],
         [[0.5600901], [0.51033545]]])

    err_msg = "Attention forward pass not implemented correctly"
    np.testing.assert_allclose(
        out.detach().numpy(), expected_out, err_msg=err_msg, rtol=1e-5
    )
    np.testing.assert_allclose(
        attention_weights.detach().numpy(), expected_weights, err_msg=err_msg, rtol=1e-5
    )


def test_counting_model_backward():
    torch.manual_seed(0)
    torch.set_printoptions(precision=6)
    batch_size = 2
    d_in = 2
    model = CountingModel(d_in, d_in - 1, 2)
    x = torch.rand(batch_size, 1, d_in)
    out, attention_weights = model(x)
    loss_function = torch.nn.MSELoss()
    expected_out = torch.tensor(
        [
            [[0.40, 0.75], [0.11, 0.42]],
            [[0.40, 0.75], [0.39, 0.74]],
        ]
    )
    torch.set_printoptions(precision=9)
    loss = loss_function(out, expected_out)
    model.zero_grad()
    loss.backward()

    expected_grads = np.array(
        [
            [[[-4.0360110e-06, 3.9807510e-06],
              [-1.0555357e-03, 1.0500017e-03]]],
            [0.00277475, -0.00193849],
            [-0.00182812, -0.0008131],
            [[-4.6097844e-07, - 6.8154151e-07],
             [-6.8884947e-07, - 1.0190042e-06]],
            [-1.1103156e-06, -1.6592525e-06],
            [[-0.00022636, - 0.00031209],
             [0.00022636, 0.00031209]],
            [-0.00054178, 0.00054178],
            [[0.00063819, - 0.00063819],
             [-0.00541609, 0.00541609]],
            [0.00147578, 0.00467411]
        ], dtype=object)

    err_msg = "Attention forward pass not implemented correctly"

    for i, param in enumerate(model.parameters()):
        np.testing.assert_allclose(
            param.grad.detach().numpy(), expected_grads[i], rtol=1e-2, err_msg=err_msg
        )


if __name__ == "__main__":
    test_counting_model_forward()
    test_counting_model_backward()
    print("Test complete.")
