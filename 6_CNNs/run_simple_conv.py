"""Script to run a simple 2D convolution operation."""

import numpy as np

from lib.convolutions import Conv2d


def main():
    # create some random input, simulating 10 grayscale 5x5 images
    x = np.random.rand(10, 1, 5, 5)
    print(f"Input shape: {x.shape}")

    # define the convolution layer
    conv_model = Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
    print(f"Model: {conv_model}")

    # run the input through the convolution and look at the output
    output = conv_model(x)
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    main()
