"""Run the XOR experiment."""

from lib.experiments import backward_pass
import numpy as np

from lib.activations import ReLU
from lib.dataset import X, y
from lib.gradient_utilities import zero_grad
from lib.losses import CrossEntropyLoss
from lib.network import Linear, Sequential
from lib.utilities import one_hot_encoding


def main():
    print("Running XOR experiment...\n")

    # Define model
    linear_units = 2
    model = Sequential(
        Linear(2, linear_units), ReLU(), Linear(linear_units, 2))

    # Define learning rate and loss
    lr = 1.
    loss_fn = CrossEntropyLoss()

    # Convert labels to one-hot
    labels = one_hot_encoding(y, 2)

    # Set the gradients to zero
    zero_grad(model.parameters())

    # Perform a forward pass and show the predictions
    y_predicted = model(X)
    loss = loss_fn(y_predicted, labels)

    # Show parameters, gradients, loss and predictions before the update
    print("---------- Model, loss and predictions before the update:\n")
    show_sequential_params_and_grads(model)
    print(f"Loss: {loss:.4f}\n")
    print(f"Predictions: {np.argmax(y_predicted, axis=1)}\n")

    # Perform one step of updating the parameters
    print("Perform one update step.\n")
    backward_pass(model, loss_fn, lr)

    # Calculate the new predictions and loss
    y_predicted = model(X)
    loss = loss_fn(y_predicted, labels)

    # Show parameters, gradients, loss and predictions after the update
    print("---------- Model, loss and predictions after 1 update step:\n")
    show_sequential_params_and_grads(model)
    print(f"Loss: {loss:.4f}\n")
    print(f"Predictions: {np.argmax(y_predicted, axis=1)}\n")

    # Now perform multiple steps of updating the parameters and observe the loss
    n_epochs = 1000
    print(f"---------- Train for {n_epochs} epochs:\n")
    for i in range(n_epochs):
        # reset gradient to zero
        zero_grad(model.parameters())

        # perform forward pass
        y_predicted = model(X)
        loss = loss_fn(y_predicted, labels)

        # perform backward pass
        backward_pass(model, loss_fn, lr)

        # show progress
        if (i + 1) % 100 == 0:
            print(f"Loss after {i + 1:3d} steps: {loss:.6f}")

    # Print final predictions after multiple steps of updates
    print(f"\n---------- Done training.\n")
    y_predicted = model(X)
    print(f"Final predictions: {np.argmax(y_predicted, axis=1)}")
    print(f"Ground truth:      {y}")


def show_sequential_params_and_grads(model: Sequential) -> None:
    """Helper function to show parameters and gradients of a sequential model.

    Args:
        model: Sequential models

    Returns:
        None
    """
    print("***** Parameters:\n")
    for module in model.modules:
        for p in module.parameters():
            print(f"{type(module).__name__}.{p.name}")
            print(p.data)
            print()

    print("***** Gradients:\n")
    for module in model.modules:
        for p in module.parameters():
            print(f"{type(module).__name__}.{p.name}")
            print(p.grad)
            print()


if __name__ == '__main__':
    main()
