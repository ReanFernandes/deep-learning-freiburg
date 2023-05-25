import matplotlib.pyplot as plt
import numpy as np


def plot_data(data: np.ndarray, rows: int = 5, cols: int = 4, plot_border: bool = True, title: str = "") -> None:
    """Plot the given image data.

    Args:
        data: image data shaped (n_samples, channels, width, height).
        rows: number of rows in the plot .
        cols: number of columns in the plot.
        plot_border: add a border to the plot of each individual digit.
                     If True, also disable the ticks on the axes of each image.
        title: add a title to the plot.

    Returns:
        None

    Note:

    """
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1).axis('on' if plot_border else 'off')
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.imshow(data[i, 0, :, :], cmap='Greys', vmin=0, vmax=1)
    plt.suptitle(title)
    plt.show()
