""" Run Transfer Learning """

import argparse
from lib.transfer_learning import *
from lib.training import train_and_evaluate


def run_transfer_learning(num_epochs=10):
    model = ModifiedResNet1()
    train_and_evaluate(model, num_epochs)
    model = ModifiedResNet2()
    train_and_evaluate(model, num_epochs)
    model = ModifiedResNet3()
    train_and_evaluate(model, num_epochs)
    model = ModifiedResNet4()
    train_and_evaluate(model, num_epochs)
    model = ModifiedResNet5()
    train_and_evaluate(model, num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help='Number of epochs to run', default=10)
    args = parser.parse_args()
    run_transfer_learning(args.e)
