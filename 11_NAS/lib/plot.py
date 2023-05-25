"""Plotting functions."""
import numpy as np
import matplotlib.pyplot as plt


def plot_error_curves(results) -> None:
    """Plot the validation errors over time (epochs).

    Args:
        results: Structure of metrics

    Returns:
        None
    """

    for val_errors in results["Val_errors"]:
        plt.plot(val_errors)

    plt.xlabel('epochs'), plt.ylabel('validation error')
    plt.title('Learning curves for different hyperparameters')
    plt.grid()
    plt.xticks(np.arange(0, 5), [str(i) for i in range(1, 6)])
    plt.show()


def plot_incumbents(results) -> None:
    """Plot the incumbents of a particular method
    Args:
        results: Dictionary containing the results of the runs

    Returns:
        None
    """
    plt.plot(results["Epochs"], results["Incumbents"])
    plt.xlabel("# Epochs")
    plt.ylabel("Validation Error")
    plt.title("Incumbents")
    plt.xlim((0, 51))
    plt.grid()
    plt.show()


def plot_comparison(optimizer1_results, optimizer2_results) -> None:
    """Plot the comparison of two different optimizer incumbents
    Args:
        optimizer1_results: Dictionary containing results of the first optimizer
        optimizer2_results: Dictionary containing results of the second optimizer

    Returns:
        None
    """
    plt.plot(optimizer1_results["Epochs"], optimizer1_results["Incumbents"], label="NAS")
    plt.plot(optimizer2_results["Epochs"], optimizer2_results["Incumbents"], label="NAS & HPO")
    plt.xlabel("# Epochs")
    plt.ylabel("Validation Error")
    plt.title("Incumbents")
    plt.legend()
    plt.grid()
    plt.show()
