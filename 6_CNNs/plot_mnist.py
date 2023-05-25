"""Script to plot MNIST data."""
from lib.plot_images import plot_data
from lib.dataset_mnist import load_mnist_data


def main():
    print("Load MNIST data.")
    x_train, x_val, x_test, y_train, y_val, y_test = load_mnist_data()

    print("Plot training data.")
    print(x_train.shape)
    plot_data(x_train, rows=5, cols=4, plot_border=False, title="MNIST Training data")


if __name__ == '__main__':
    main()
