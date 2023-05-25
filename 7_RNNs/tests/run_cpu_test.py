"""Script to test PyTorch CPU installation."""
import torch
import torch.nn as nn


def main():
    print("Creating 2-layer MLP with PyTorch.")
    model = nn.Sequential(
        nn.Linear(100, 30),
        nn.ReLU(),
        nn.Linear(30, 10))

    print("Running some input through the MLP.")
    x = torch.randn(50, 1, 1, 100)
    output = model(x)

    print("Success!")
    print(f"Input shape: {x.shape}")
    print(f"Model: {model}")
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    main()
