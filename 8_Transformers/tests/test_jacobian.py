import numpy as np
import torch

from lib.utils import calculate_jacobian


def test_jacobian():
    torch.manual_seed(0)
    torch.set_printoptions(precision=8)
    input = torch.rand(1, 1, 3, 3)
    target = torch.rand(1, 9)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(9, 9))
    criterion = torch.nn.functional.mse_loss
    gradient, output = calculate_jacobian(input, target, model, criterion)
    expected_grad = np.array(
        [
            [
                [
                    [-0.02169365, -0.07151975, -0.05031663],
                    [-0.10872032, 0.04322950, 0.13368227],
                    [-0.00335179, 0.08614062, 0.20584446],
                ]
            ]
        ]
    )
    expected_output = np.array(
        [
            [
                -0.08845613,
                -0.50036943,
                -0.08141938,
                0.36171785,
                -0.22846931,
                -0.10322702,
                -0.14116710,
                -0.03239115,
                -0.24230444,
            ]
        ]
    )

    err_msg = "calculate_jacobian not implemented correctly"
    np.testing.assert_allclose(
        gradient.detach().numpy(), expected_grad, err_msg=err_msg, rtol=1e-5
    )
    np.testing.assert_allclose(
        output.detach().numpy(), expected_output, err_msg=err_msg, rtol=1e-5
    )


if __name__ == "__main__":
    test_jacobian()
    print("Test complete.")
