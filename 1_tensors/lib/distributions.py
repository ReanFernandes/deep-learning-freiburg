"""
Code for the exercise on statistical distributions.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_clt(n_repetitions: int = 1024, sample_sizes: Tuple[int] = (1, 16, 64, 1024)) -> None:
    """
    The Central Limit Theorem states, that for i.i.d. samples from (almost) any distribution,
    the mean of the samples follows approximately a normal distribution.

    This function draws a number of samples and calculates the mean. This happens n_repetitions times and then
    the means are plotted in a histogram.

    Args:
        n_repetitions: Create this many sample means.
        sample_sizes: Calculate sample mean over this size.

    Returns:
        None
    """
    # Define distributions
    distributions = {
        "exponential": lambda size: np.random.exponential(1, size),
        "normal": lambda size: np.random.normal(1, 1, size)
    }

    # Setup plot
    fig, axes = plt.subplots(len(sample_sizes), len(distributions), figsize=(8, 12))
    for i, dist_name in enumerate(distributions.keys()):
        fig.axes[i].set_title(dist_name)

    for nrow, n_samples in enumerate(sample_sizes):
        for ncol, (dist_name, dist_fn) in enumerate(distributions.items()):
            # Get the axis to plot on
            axis = axes[nrow, ncol]
            # START TODO #############
            # For n_repetitions times, calculate the mean of n_samples draws from the distribution.
            # Hint: Don't use for loops.
            # samples = ...
            # means = ....
            raise NotImplementedError
            # END TODO #############

            # Draw histogram and compare it to the PDF of the mean distribution
            _, bins, _ = axis.hist(means, label=f"{n_samples} samples", density=True, bins=50)
            x = np.linspace(bins[0], bins[-1], 100)
            pdf = stats.norm.pdf(x, 1, 1 / np.sqrt(n_samples))
            axis.plot(x, pdf)

            axis.legend()
    plt.legend()
    plt.show()


def std_normal(n_samples: int = 1) -> np.ndarray:
    """
    Sample from a standard normal distribution.
    The normal distribution is approximated via a uniform distribution.
    See the assignment sheet for details on the computation.

    Args:
        n_samples: number of samples of a standard normal distribution

    Returns:
        Samples with shape (n_samples)
    """
    n_uniform_samples = 1024
    # START TODO #############
    # Sample (n_samples, n_uniform_samples) from a uniform distribution [-b, b]
    # with the scale b given by the formulae in the assignment sheet.
    # Then, average over the last axis and return.
    raise NotImplementedError
    # END TODO #############


def normal(loc: float = 0.0, scale: float = 1.0, n_samples: int = 1) -> np.ndarray:
    """
    Sample from a normal distribution.
    The normal distribution is approximated via a uniform dist.

    Args:
        loc: mean of the distribution
        scale: standard deviation spread of the distribution.
        n_samples: number of samples

    Returns:
        Samples with shape (n_samples,)
    """
    # START TODO #############
    # Hint: use the function std_normal.
    raise NotImplementedError
    # END TODO #############


def plot_normal(n_samples: int = 10000, bins: int = 40) -> None:
    """
    Compare the approximations via CLT and numpy.random.normal

    Args:
        n_samples: How many samples to draw for the approximation.
        bins: Number of bins for the histogram.
    """
    # plot histograms for standard normal with std_normal and np.random.normal
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    mean, std = 0, 1
    ax1.set_title(f"mean: {mean}, std: {std}, n_samples: {n_samples}")
    ax1.hist(std_normal(n_samples), bins, label='custom', histtype='stepfilled')
    ax1.hist(np.random.normal(mean, std, n_samples), bins, label='numpy', histtype='step')
    ax1.legend()

    # plot histograms for N(mean=-10, std=3) with normal and np.random.normal
    mean, std = -10, 3
    ax2.set_title(f"mean: {mean}, std: {std}, n_samples: {n_samples}")
    ax2.hist(normal(mean, std, n_samples), bins, label='custom', histtype='stepfilled')
    ax2.hist(np.random.normal(mean, std, n_samples), bins, label='numpy', histtype='step')
    ax2.legend()

    plt.show()
