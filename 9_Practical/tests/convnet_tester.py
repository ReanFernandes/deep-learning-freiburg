import torch
import numpy as np


def convnet_test(convnetmodel, expected_result):
    torch.manual_seed(0)
    convnet = convnetmodel()

    outputs = convnet(torch.rand(4, 3, 32, 32))

    err_msg = str(convnetmodel) + ' not implemented correctly'
    np.testing.assert_allclose(outputs.detach().numpy(), expected_result, atol=1e-5, err_msg=err_msg)
