"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from lib.lr_schedulers import PiecewiseConstantLR, CosineAnnealingLR
from lib.optimizers import Adam, SGD
from lib.network_base import Parameter
from lib.utilities import load_result


def plot_learning_curves() -> None:
    """Plot the performance of SGD, SGD with momentum, and Adam optimizers.

    Note:
        This function requires the saved results of compare_optimizers() above, so make
        sure you run compare_optimizers() first.
    """
    optim_results = load_result('optimizers_comparison')
    # START TODO ################
    # train result are tuple(train_costs, train_accuracies, eval_costs,
    # eval_accuracies). You can access the iterable via
    # optim_results.items()
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="Accuracy")
    for it in optim_results.items():
        exp, metrics = it
        ax0.plot(range(len(metrics[0])), metrics[0], label=f'loss_{exp}')
        ax1.plot(range(len(metrics[0])), metrics[1], label=f'acc_{exp}')
    ax0.legend()
    ax1.legend()
    plt.savefig("res.png")
    # END TODO ###################


def plot_lr_schedules() -> None:
    """Plot the learning rate schedules of piecewise and cosine schedulers.

    """
    num_epochs = 80
    base_lr = 0.1

    piecewise_scheduler = PiecewiseConstantLR(Adam([], lr=base_lr), [10, 20, 40, 50], [0.1, 0.05, 0.01, 0.001])
    cosine_scheduler = CosineAnnealingLR(Adam([], lr=base_lr), num_epochs)

    # START TODO ################
    # plot piecewise lr and cosine lr
    names = ["piecewise", "cosine"]
    schedulers = [piecewise_scheduler, cosine_scheduler]

    # START TODO ################
    for scheduler, schedule_name in zip(schedulers, names):
        lr = []
        for epoch in range(num_epochs):
            scheduler.step()
            lr.append(scheduler.optimizer.lr)
        plt.plot(range(num_epochs), lr, label=schedule_name)

    plt.xlabel('epochs')
    plt.ylabel('Learning rate')
    plt.legend()
    plt.savefig("Schedulers.png")
    
    # END TODO ################
