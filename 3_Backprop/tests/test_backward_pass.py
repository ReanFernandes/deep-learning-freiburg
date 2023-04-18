import numpy as np

from lib.experiments import backward_pass
from lib.gradient_utilities import zero_grad
from lib.losses import CrossEntropyLoss
from lib.network import Linear
from lib.utilities import one_hot_encoding


def test_backward_pass():
    """Test multiple backward steps."""

    np.random.seed(42)  # fix seed for consistent results

    # create model and loss, zero gradients
    linear_units = 2
    model = Linear(2, linear_units)
    loss_fn = CrossEntropyLoss()
    zero_grad(model.parameters())

    # create inputs and labels
    x = np.random.rand(4, 2)  # inputs shape (4, 2)
    y_classes = np.random.randint(0, 2, size=4)  # labels shape (4)
    y = one_hot_encoding(y_classes, 2)  # one-hot labels shape (4, 2)
    lr = 0.1

    # for some epochs
    for i in range(100):
        # do forward pass and calculate loss
        y_predicted = model(x)
        loss = loss_fn(y_predicted, y)
        if (i + 1) % 10 == 0:
            print(f"Epoch {i + 1:3d} Loss: {loss:.3e}")
        backward_pass(model, loss_fn, lr)

    # check parameters for correctness
    params = model.parameters()
    inputs = (params[0].data, params[1].data, params[0].grad, params[1].grad)
    ground_truths = ([[-4.41894638, 4.42253088], [-13.79277066, 13.81447784]], [-21.3430101, 21.3630101],
                     [[0.45268016, -0.45268016], [1.41033216, -1.41033216]], [2.18581226, -2.18581226])
    for inp, gt in zip(inputs, ground_truths):
        np.testing.assert_allclose(
            inp, gt, rtol=1e-5, err_msg="Linear Layer backward pass is not implemented correctly.", )


def test_backward_pass_10D():
    """Test multiple backward steps."""

    np.random.seed(42)  # fix seed for consistent results

    # create model and loss, zero gradients
    linear_units = 2
    model = Linear(10, linear_units)
    loss_fn = CrossEntropyLoss()
    zero_grad(model.parameters())

    # create inputs and labels
    x = np.random.rand(4, 10)  # inputs shape (4, 10)
    y_classes = np.random.randint(0, 2, size=4)  # labels shape (4)
    y = one_hot_encoding(y_classes, 2)  # one-hot labels shape (4, 2)
    lr = 0.1

    # for some epochs
    for i in range(101):
        # do forward pass and calculate loss
        y_predicted = model(x)
        loss = loss_fn(y_predicted, y)
        if (i + 1) % 10 == 0:
            print(f"Epoch {i + 1:3d} Loss: {loss:.3e}")
        backward_pass(model, loss_fn, lr)

    # check parameters for correctness
    params = model.parameters()
    inputs = (params[0].data, params[1].data, params[0].grad, params[1].grad)
    ground_truths = ([[2.11499972, -2.11141523],
                      [4.77612144, -4.75441425],
                      [-4.5260154, 4.5213325],
                      [-4.58617116, 4.60963763],
                      [-13.15223193, 13.15296278],
                      [2.52667465, -2.53596613],
                      [-5.42128355, 5.40457037],
                      [-6.16253603, 6.13966398],
                      [9.49525479, -9.50224062],
                      [-13.07704091, 13.05383763]], [0.43713853, -0.41713853],
                     [[-0.21347064, 0.21347064],
                      [-0.47904391, 0.47904391],
                      [0.50868798, -0.50868798],
                      [0.52625607, -0.52625607],
                      [1.45356364, -1.45356364],
                      [-0.26052414, 0.26052414],
                      [0.64406605, -0.64406605],
                      [0.71614034, -0.71614034],
                      [-0.96623017, 0.96623017],
                      [1.46438329, -1.46438329]], [0.0111689, -0.0111689]
                     )
    for inp, gt in zip(inputs, ground_truths):
        np.testing.assert_allclose(
            inp, gt, rtol=1e-5, err_msg="Linear Layer backward pass is not implemented correctly.", )


if __name__ == '__main__':
    test_backward_pass()
    print("Test complete.")
