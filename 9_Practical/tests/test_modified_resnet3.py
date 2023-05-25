import torch
import numpy as np
from lib.transfer_learning import ModifiedResNet3


def test_modified_resnet():
    torch.random.manual_seed(9001)
    x = torch.rand(1, 3, 224, 224)

    model = ModifiedResNet3()
    logits = model(x)

    err_msg = 'ModifiedResNet3 not implemented correctly'
    expected_logits = np.array([[0.586052,  1.088286, -0.399033, -0.616515, -0.097618, -0.622059,
                                 -0.204248, -0.271284,  0.704981, -0.809093]])
    np.testing.assert_allclose(logits.detach().numpy(), expected_logits, atol=1e-4, err_msg=err_msg)


def test_modified_resnet_layers_frozen():
    model = ModifiedResNet3()
    learnable_params = sum([params.numel() for params in model.parameters() if params.requires_grad])
    assert learnable_params == 2364426


if __name__ == '__main__':
    test_modified_resnet()
    test_modified_resnet_layers_frozen()
    print('Test complete.')
