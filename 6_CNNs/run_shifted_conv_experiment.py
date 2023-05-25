"""Script to run the 2D convolution experiment on MNIST with shifted validation data."""

import argparse

from lib.experiments import run_shifted_conv_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_datapoints", type=int, default=10000,
                        help="Maximum number of datapoints. Default 10000, can be up to 50000.")
    args = parser.parse_args()
    run_shifted_conv_experiment(max_datapoints=args.max_datapoints)


if __name__ == '__main__':
    main()
