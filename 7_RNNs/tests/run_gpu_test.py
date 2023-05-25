"""Script to test PyTorch GPU installation."""
import torch
import torch.nn as nn


def main():
    cuda_available = torch.cuda.is_available()
    print(f"GPU available: {cuda_available}")
    if not cuda_available:
        raise RuntimeError(f"No GPU detected!")

    print("Creating 2-layer MLP with PyTorch.")
    model = nn.Sequential(
        nn.Linear(100, 30),
        nn.ReLU(),
        nn.Linear(30, 10))

    print("Create some input.")
    x = torch.randn(50, 1, 1, 100)

    print("Move model and data to GPU.")
    model = model.cuda()
    x = x.cuda()

    print("Run the computation on the GPU.")
    output = model(x)

    print("Success!")
    print(f"Input shape: {x.shape}")
    print(f"Model: {model}")
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    main()
