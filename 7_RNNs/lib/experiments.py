"""Experiments with LSTM"""
import os

import numpy as np
import torch
import torch.optim as optim

from lib.models import NoiseRemovalModel, LSTM
from lib.plot import plot_curves
from lib.utilities import NUM_FUNCTIONS_TRAIN, NUM_FUNCTIONS_VAL, sample_sine_functions,\
    prepare_sequences, percentage_noise_removed


def train_lstm_echo(num_epochs=100) -> None:
    """
    Train the LSTM to echo a value at a specific index of a sequence.

    Args:
        num_epochs: Epochs to train.
    """
    # Create 1000 training sequences of length 10
    num_samples_train = 1000
    num_samples_test = 100
    seq_length = 10
    batch_size = 5
    torch.manual_seed(42)

    # we use a hidden size larger than 1 as it makes training easier
    # since as prediction we compute the mean over the output.
    hidden_size = 6
    training_sequences = torch.rand(num_samples_train, seq_length, 1)
    test_sequences = torch.rand(num_samples_test, seq_length, 1)
    model = LSTM(1, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    def accuracy(y, label, eps=1e-2):
        assert y.shape == label.shape, (y.shape, label.shape)
        return np.sum(np.abs(y - label) < eps) / len(y)

    loss: torch.Tensor = 0
    for epoch in range(num_epochs):
        for batch_idx in range(num_samples_train // batch_size):
            optimizer.zero_grad()
            batch = training_sequences[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            output = model(batch)[1][0]
            labels = batch[:, 1]  # echo the second element
            loss = loss_fn(output, labels.repeat(1, hidden_size))
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"epoch: {epoch}, loss: {loss:.5f}", end=" ")
            batch = test_sequences
            output = model(batch)[1][0]
            labels = batch[:, 1]  # echo the second element
            acc = accuracy(np.squeeze(labels.cpu().numpy(), axis=-1),
                           np.mean(output.detach().cpu().numpy(), axis=-1))
            print(f"test accuracy: {acc}")
        scheduler.step()


def train_noise_removal_model(hidden_size=40, shift=10, lr=0.01, num_epochs=101, batch_size=10, plot=True,
                              reproduce=False) -> None:
    """
    Train a noise removal model with the given parameters.

    Args:
        hidden_size: Number of units of the LSTM hidden state size.
        shift: Number of steps the RNN is run before its output is considered ("many-to-many shifted to
                      the right").
        lr: Learning rate of the NoiseRemovalModel.
        num_epochs: Number of epochs to train the model.
        batch_size: Batch size for training the model.
        plot: Flag to indicate whether or not to plot the results during model training.
        reproduce: Boolean flag to indicate if the experiment will run with fixed seed. Defaults to False.
    """
    # If flag is given set seeds for deterministic runs for same configurations
    if reproduce:
        torch.manual_seed(42)
        np.random.seed(42)

    model = NoiseRemovalModel(hidden_size=hidden_size, shift=shift)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    train_functions = sample_sine_functions(NUM_FUNCTIONS_TRAIN)
    val_functions = sample_sine_functions(NUM_FUNCTIONS_VAL)
    train_sequences, noisy_train_sequences = prepare_sequences(train_functions)
    val_sequences, noisy_val_sequences = prepare_sequences(val_functions)

    os.makedirs("output_plots", exist_ok=True)
    loss: torch.Tensor = 0
    for epoch in range(num_epochs):
        # START TODO #############
        # training loop here
        for batch_idx in range(NUM_FUNCTIONS_TRAIN // batch_size):
            optimizer.zero_grad()
            batch = noisy_train_sequences[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            output = model(batch)
            labels = train_sequences[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
        # END TODO #############
        print(f"epoch: {epoch}, train      loss:{loss:.5f}")
        # compute the validation loss
        output = model(noisy_val_sequences)
        loss = loss_fn(val_sequences, output)
        print(f"epoch: {epoch}, validation loss:{loss:.5f}")
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            np_tensors = [a.detach().numpy() for a in (val_sequences, noisy_val_sequences, output)]
            if plot:
                output_file = f"output_plots/denoising_epoch_{epoch}.png"
                plot_curves(*np_tensors, file=output_file)
                print(f"Plotted curves to {output_file}")
            print(f"{percentage_noise_removed(*np_tensors):2.4f}% of noise removed.")
        scheduler.step()


def run_hpo() -> None:
    """
    Function to train noise removal models with different hyperparameters.
    Note: In order to have a fair comparison, please make sure the train/test data used for
    all hyperparameters is the same
    """

    # START TODO
    # To have the same results for the same hyperparameter settings for different runs, use
    # the boolean flag "reproduce"
    batch_size = range(10, 100, 10)
    for batch in batch_size:
        train_noise_removal_model(hidden_size=10, shift=10, batch_size=batch, reproduce=True)
        print('for batch size = ', batch)

    num_epochs = range(100, 400, 100)
    for num_epoch in num_epochs:
        train_noise_removal_model(hidden_size=10, shift=10, batch_size=20, num_epochs=num_epoch, reproduce=True)
        print('for num epochs = ', num_epoch)

    shifts = range(2, 30, 2)
    for shift in shifts:
        train_noise_removal_model(hidden_size=10, shift=shift, batch_size=20, num_epochs=200, reproduce=True)
        print('for shift = ', shift)
    # END TODO
