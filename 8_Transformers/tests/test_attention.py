import numpy as np
import torch

from lib.attention import attention_function


def test_attention_function():
    torch.manual_seed(0)
    torch.set_printoptions(precision=6)
    dk = 2
    q = torch.rand(1, 3, dk)
    k = torch.rand(1, 3, dk)
    v = torch.rand(1, 3, dk)
    attention, weights = attention_function(q, k, v, dk)

    expected_attention = np.array(
        [[[0.60555935, 0.9019348],
          [0.5243514, 0.7717216],
          [0.5818891, 0.8646183]]],
    )

    expected_weights = np.array(
        [[[0.659022, 0.6232489, 0.58434784],
          [0.52855724, 0.5218704, 0.5148287],
          [0.6244473, 0.59448713, 0.56364226]]],
    )

    err_msg = "attention_function not implemented correctly"
    np.testing.assert_allclose(
        attention.detach().numpy(), expected_attention, err_msg=err_msg, rtol=1e-5
    )
    np.testing.assert_allclose(
        weights.detach().numpy(), expected_weights, err_msg=err_msg, rtol=1e-5
    )


if __name__ == "__main__":
    test_attention_function()
    print("Test complete.")
