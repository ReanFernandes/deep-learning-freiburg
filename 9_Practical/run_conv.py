"""Train and evaluate the CNN models"""

import argparse
import torch
import numpy as np

from lib.models import ConvNet1, ConvNet2, ConvNet3, ConvNet4, ConvNet5
from lib.training import train_and_evaluate


def run_conv(model_n, num_epochs=1):
    models = [ConvNet1, ConvNet2, ConvNet3, ConvNet4, ConvNet5]
    torch.manual_seed(1337)  # for deterministic runs
    model = models[model_n - 1]()
    print(model)

    train_and_evaluate(model, num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=int, help='ConvNetx model to run (1-5)', default=1)
    parser.add_argument('-e', type=int, help='Number of epochs to run', default=10)
    args = parser.parse_args()

    print('Training and evaluating ConvNet{}'.format(str(args.x)))
    run_conv(args.x, args.e)
