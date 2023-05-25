import torch
import numpy as np
from lib.transfer_learning import ModifiedResNet4


def test_modified_resnet():
    torch.random.manual_seed(9001)
    x = torch.rand(1, 3, 224, 224)

    model = ModifiedResNet4()
    logits = model(x)

    err_msg = 'ModifiedResNet4 either not implemented correctly or you have swapped the order of initialization'
    expected_logits = np.array([[0.607865,  1.06168, -0.404919, -0.612327, -0.052961, -0.613137,
                                 -0.259476, -0.239556,  0.674133, -0.836252]])
    np.testing.assert_allclose(logits.detach().numpy(), expected_logits, atol=1e-4, err_msg=err_msg)


def test_modified_resnet_layers_frozen():
    model = ModifiedResNet4()
    learnable_params = sum([params.numel() for params in model.parameters() if params.requires_grad])
    assert learnable_params == 2364426


if __name__ == '__main__':
    test_modified_resnet()
    test_modified_resnet_layers_frozen()
    print('Test complete.')
