"""Code for training the models."""

import logging
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


class AverageMeter(object):
    """Class to help track averages of various values like training loss, accuracy etc."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1) -> None:
        """Update the values.

        Params:
            val : New (average) value to add.
            n   : Number of instances the new value is average of.

        Returns:
            None
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the accuracy given the predictions and the ground truth.

    Params:
        logits : Model predictions.
        labels : Ground truth.

    Returns:
        Accuracy
    """
    preds = torch.argmax(logits, dim=1)
    return torch.sum(preds == labels) / len(labels)


def eval_fn(model: nn.Module,
            criterion: nn.Module,
            loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """Evaluation method

    Args:
        model     : Model to evaluate.
        criterion : Loss criterion.
        loader    : Data loader for either training or testing set.

    Returns:
        Accuracy on the data.
    """
    score = AverageMeter()
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():  # no gradient needed
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            n = images.size(0)
            losses.update(loss.item(), n)
            score.update(acc.item(), images.size(0))

    return score.avg, losses.avg


def train_fn(model: nn.Module,
             optimizer: torch.optim.Optimizer,
             criterion: nn.Module,
             loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """Training method.

    Params:
        model     : Model to train
        optimizer : Optimization algorithm
        criterion : Loss function
        loader    : Data loader for training set

    Returns:
        (accuracy, loss) on the data.
    """
    time_begin = time.time()
    score = AverageMeter()
    losses = AverageMeter()
    model.train()
    time_train = 0

    for images, labels in loader:
        # START TODO #################
        # Perform a forward/backward pass on the model
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        # END TODO #################

        acc = accuracy(logits, labels)
        n = images.size(0)
        losses.update(loss.item(), n)
        score.update(acc.item(), n)

    time_train += time.time() - time_begin
    logging.info(f'Training time: {time_train}s')
    return score.avg, losses.avg


def train(model, num_epochs, batch_size, train_criterion, model_optimizer) -> Dict[str, float]:
    """Training loop for model.

    Params:
        num_epochs          : Number of epochs
        batch_size          : Size of the batch
        learning_rate       : Model optimizer learning rate
        train_criterion     : Which loss to use during training (torch.nn._Loss)
        model_optimizer     : Which model optimizer to use during training (torch.optim.Optimizer)
        data_augmentations  : List of data augmentations to apply such as rescaling.
                              (list[transformations], transforms.Composition[list[transformations]], None)
                              If none only ToTensor is used

    Returns:
        Dictionary of metrics obtained by training the model from scratch, of the following form:
        {
            'train_acc' : value
            'train_loss': value,
            'val_acc'   : value,
            'val_loss'  : value,
            'train_time': value,
            'params'    : value
        }
    """
    time_begin = time.time()

    logging.basicConfig(level=logging.INFO)

    data_augmentations = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    # Load the dataset
    data_dir = '../dataset'

    train_data = torchvision.datasets.CIFAR10(data_dir, train=True, transform=data_augmentations, download=True)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)

    # START TODO #################
    # Create the data loader for the validation set
    # Hint: Use the same data augmentations as for the training set, download the dataset if it is not already
    # downloaded and make sure to signify that it will not be used for training
    # Use the same batch size as for the training set, but do not shuffle the data
    val_data = torchvision.datasets.CIFAR10(data_dir, train=False, transform=data_augmentations, download=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    # END TODO #################

    train_data.data = train_data.data[: int(0.05 * len(train_data))]
    val_data.data = val_data.data[: int(0.05 * len(val_data))]

    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    val_errors = []

    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        # Get training accuracy and train loss from train_fn and append them to train_accs and train_losses
        # START TODO #################
        train_score, train_loss = train_fn(model, model_optimizer, train_criterion, train_loader)
        train_accs.append(train_score)
        train_losses.append(train_loss)
        # END TODO #################
        logging.info('Train accuracy: %f', train_score)

        # Get validation accuracy, validation loss and validation errors(1 - validation accuracy)
        # from eval_fn and append them to val_accs, val_losses and val_errors
        # START TODO #################
        val_acc, val_loss = eval_fn(model, train_criterion, val_loader)
        val_accs.append(val_acc)
        val_errors.append(1-val_acc)
        val_losses.append(val_loss)
        # END TODO #################
        logging.info('Validation accuracy: %f', val_acc)

    logging.info('Accuracy at each epoch: ' + str(val_accs))
    logging.info('Mean of accuracies across all epochs: ' + str(100 * np.mean(val_accs)) + '%')
    logging.info('Accuracy of model at final epoch: ' + str(100 * val_accs[-1]) + '%')

    time_end = time.time()
    train_time = (time_end - time_begin) / 3600

    n_params = sum(p.numel() for p in model.parameters()) / (10 ** 6)

    results = {
        'train_acc': train_accs,
        'train_loss': train_losses,
        'val_acc': val_accs,
        'val_loss': val_losses,
        'val_error': val_errors,
        'train_time': train_time,
        'params': n_params
    }

    return results
