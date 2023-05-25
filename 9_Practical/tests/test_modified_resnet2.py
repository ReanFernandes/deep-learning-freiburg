import torch
import numpy as np
from lib.transfer_learning import ModifiedResNet2


def test_modified_resnet():
    torch.random.manual_seed(9001)
    x = torch.rand(1, 3, 224, 224)

    model = ModifiedResNet2()
    torch.nn.init.xavier_uniform_(model.resnet.fc.weight)
    model.resnet.fc.bias.data = torch.Tensor([0] * 10)
    logits = model(x)

    err_msg = 'ModifiedResNet2 not implemented correctly'
    expected_logits = np.array([[-2.6372, -0.5771, -1.0881, -0.4059, 2.0506,
                                 0.2592, 1.4818, 1.2425, -2.5308, 2.1344]])
    np.testing.assert_allclose(logits.detach().numpy(), expected_logits, atol=1e-4, err_msg=err_msg)


def test_modified_resnet_layers_frozen():
    model = ModifiedResNet2()
    learnable_params = sum([params.numel() for params in model.parameters() if params.requires_grad])
    assert learnable_params == 5130


if __name__ == '__main__':
    test_modified_resnet()
    test_modified_resnet_layers_frozen()
    print('Test complete.')
