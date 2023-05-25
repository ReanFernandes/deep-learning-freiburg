"""Test for train function in lib\training.py"""
import torch
import torch.nn as nn
import numpy as np

from lib.training import train
from lib.primitives import ReLUConvBN
from lib.utilities import set_seed


def test_train():
    set_seed(124)
    err_msg = 'Training function is not implemented correctly'
    test_model = nn.Sequential(
        ReLUConvBN(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(3*16*16, 10)
    )
    optimizer = torch.optim.SGD(test_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    results = train(test_model, 1, 1, criterion, optimizer)
    np.testing.assert_allclose(results["train_acc"], [0.2808], rtol=1e-2, err_msg=err_msg)
    np.testing.assert_allclose(results["train_loss"], [2.0239268112003805], rtol=1e-2, err_msg=err_msg)
    np.testing.assert_allclose(results["val_acc"], [0.266], rtol=1e-2, err_msg=err_msg)
    np.testing.assert_allclose(results["val_loss"], [2.1091806457936766], rtol=1e-2, err_msg=err_msg)
    np.testing.assert_allclose(results["val_error"], [0.734], rtol=1e-2, err_msg=err_msg)
    np.testing.assert_allclose(results["params"], 0.007777, rtol=1e-2, err_msg=err_msg)


if __name__ == "__main__":
    test_train()
    print("Test complete.")
