"""ReLU plot."""

import matplotlib.pyplot as plt
import numpy as np

from lib.activations import ReLU


def plot_relu() -> None:
    """Plot the ReLU function in the range (-4, 4).

    Returns:
        None
    """
    # START TODO #################
    # Create input data, run through ReLU and plot.
    input  = np.linspace(-4, 4, 101)
    relu = ReLU()
    output = relu(input)
    plt.plot(input, output)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.show()
    
    # END TODO###################
