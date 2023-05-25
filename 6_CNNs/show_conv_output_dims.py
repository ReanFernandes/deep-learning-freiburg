"""Script to show output dimensions of a convolution given input and parameters."""

from lib.convolutions import calculate_conv_output


def main():
    input_shape = [7, 7]
    kernel_size = [3, 3]
    stride = [1, 1]
    padding = [2, 2]

    print(f"Input shape: {input_shape}")
    print(f"Kernel size: {kernel_size}")
    print(f"Stride: {stride}")
    print(f"Padding: {padding}")
    print("\nCalculating output size.")
    output_shape = calculate_conv_output(input_shape, kernel_size, stride, padding)
    print(f"Output shape: {output_shape}")


if __name__ == '__main__':
    main()
