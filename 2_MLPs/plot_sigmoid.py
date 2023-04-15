"""Script to plot the Sigmoid."""

import matplotlib.pyplot as plt
import numpy as np

from lib.activations import Sigmoid


def main():
    # create input
    x = np.linspace(-4, +4, 101)

    # create sigmoid module and apply it to the input
    sigmoid = Sigmoid()
    y = sigmoid(x)  # equal to call of sigmoid.forward(x)

    # plot input and sigmoid output
    plt.plot(x, y)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
