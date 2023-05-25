import numpy as np

from lib.losses import CrossEntropyLoss
from lib.network import Linear
from lib.optimizers import Adam
from lib.training import train
from tests.test_train_model import create_dummy_data


def test_adam():
    """Test Adam optimizer."""
    # create some data (also fixes the seed)
    x_train, x_val, y_train, y_val = create_dummy_data(
        train_batch_size=10, val_batch_size=10, input_dim=10, output_dim=5)

    # create a model and train it with Adam
    model = Linear(10, 5)
    optimizer = Adam(model.parameters())
    train(model, CrossEntropyLoss(), optimizer, x_train, y_train,
          x_val, y_val, num_epochs=10, batch_size=10)

    # check updates are correct
    ground_truth_0 = np.array([[-0.00331477, -0.01023302, 0.023417, 0.00253593, -0.01957613],
                               [0.00089653, -0.00363904, 0.04059712, -0.00321162, -0.0027305],
                               [0.01313359, -0.00938841, 0.01909129, -0.01037581, -0.0292427],
                               [-0.01850472, 0.00366342, 0.02170206, 0.0062565, -0.00492424],
                               [0.03770286, -0.01029258, 0.00986239, 0.01187653, -0.00377795],
                               [0.00648497, -0.00391959, -0.00946149, 0.0133325, -0.00945779],
                               [0.00780274, -0.01473393, 0.00388962, -0.0180656, -0.0330831],
                               [0.00730834, 0.00098146, -0.01340412, 0.01632782, -0.02076647],
                               [-0.00307219, -0.01748909, -0.02041193, -0.00184271, -0.00548316],
                               [0.02777654, -0.01665042, -0.0270708, 0.01655023, -0.02433648]])
    ground_truth_1 = np.array([1.99749233e-02, 1.86637649e-05, 1.99770474e-02, 1.99722885e-02, 3.98844885e-05])

    err_msg = "Adam is not implemented correctly."
    np.testing.assert_allclose(optimizer._params[0].data, ground_truth_0, rtol=1e-3, err_msg=err_msg)
    np.testing.assert_allclose(optimizer._params[1].data, ground_truth_1, rtol=1e-3, err_msg=err_msg)


if __name__ == '__main__':
    test_adam()
    print('Test complete.')
