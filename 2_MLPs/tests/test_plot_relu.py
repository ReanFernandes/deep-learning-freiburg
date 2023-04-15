import matplotlib.pyplot as plt

from lib.plot import plot_relu


def test_plot_relu():
    """Test if the ReLU plot runs through."""
    # enable interactive mode so plt.show() will not block the process.
    plt.ion()
    plot_relu()


if __name__ == '__main__':
    test_plot_relu()
    print("Test complete.")
