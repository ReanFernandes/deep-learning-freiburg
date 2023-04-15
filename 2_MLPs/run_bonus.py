"""Script to run the bonus exercise."""

import matplotlib.pyplot as plt
import numpy as np

from lib.dataset import X, y
from lib.models import create_2unit_net
from lib.model_utilities import extract_hidden


def main() -> None:
    """Create the 2 unit model and ."""
    # create 2unit model
    model = create_2unit_net()

    # plot its hidden representation
    h = extract_hidden(model, X)
    plot_repr_space(X[:, 0], X[:, 1], h[:, 0], h[:, 1], y.flatten())


def plot_repr_space(x1: np.ndarray, x2: np.ndarray, h1: np.ndarray, h2: np.ndarray, y: np.ndarray) -> None:
    """
    Plot representation space.

    Args:
        x1: First input feature with shape (nr_examples,).
        x2: Second input feature with shape (nr_examples,).
        h1: First learned feature with shape (nr_examples,).
        h2: Second learned feature with shape (nr_examples,).
        y: True labels with shape (nr_examples,).
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for k, (i, j) in enumerate(zip(x1, x2)):
        ax[0].scatter(i, j, c='b', marker=r"${}$".format(y[k]), s=100)
    for k, (i, j) in enumerate(zip(h1, h2)):
        ax[1].scatter(i, j, c='b', marker=r"${}$".format(y[k]), s=100)

    ax[0].set_yticks([0, 1])
    ax[0].set_xticks([0, 1])
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_title("Original x space")

    ax[1].set_yticks([0, 1])
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xlabel('h1')
    ax[1].set_ylabel('h2')
    ax[1].set_title("Learned h space")

    plt.show()


if __name__ == '__main__':
    main()
