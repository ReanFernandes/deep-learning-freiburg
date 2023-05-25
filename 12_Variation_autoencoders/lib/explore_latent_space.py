"""
Code to plot the latent space of the trained VAE model.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data

from lib.model_vae import VAE


def get_mu_logvar(model: VAE, test_loader: data.DataLoader, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the model's prediction of the mean and the logvar for all images in the test set.

    Args:
        model: Model to use for prediction.
        test_loader: Data to input to the model.
        device: Torch device ("cpu" or "cuda")

    Returns:
        Tuple of mean and logvar both with shape (num_training_samples, latent_size)
    """
    model = model.to(device)
    mus, logvars = [], []
    # START TODO ########################
    # For each image batch of the test_loader, encode the images to compute mean and logvar with the model.
    # Collect mean and logvar in the two lists.
    # Remember that the images need to be sent to the same device as the model.
    raise NotImplementedError
    # END TODO ########################

    # return the collected means and logvars as a concatenated tensor
    return torch.cat(mus), torch.cat(logvars)


@torch.no_grad()
def sample_on_grid(model: VAE, latent0_min: float, latent0_max: float, latent1_min: float, latent1_max: float,
                   grid_size: int, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a model with a two-dimensional latent space, create a 2D grid of latent variables and compute their
    reconstruction.

    Args:
        model: Model to compute the reconstructions.
        latent0_min: Minimum value of the first latent dimension.
        latent0_max: Maximum value of the first latent dimension.
        latent1_min: Minimum value of the second latent dimension.
        latent1_max: Maximum value of the second latent dimension.
        grid_size: Representations to compute per axis.
        device: Torch device ("cpu" or "cuda")

    Returns:
        Tuple of:
            Decoded images with shape (grid_size, grid_size, channels, height, width)
            Step size per dimension with shape (latent_size = 2)
    """
    # Create a grid in two directions from latent_min to latent_max. First, create the 2 axes.
    x1 = torch.linspace(latent0_min, latent0_max, grid_size)
    x2 = torch.linspace(latent1_min, latent1_max, grid_size)
    # Compute the step size (needed for correct plotting of class means)
    step_size = np.array([(latent0_max - latent0_min) / grid_size, (latent1_max - latent1_min) / grid_size])
    # Create the individual grid for each axis.
    mesh_grid_x1, mesh_grid_x2 = torch.meshgrid((x1, x2))
    # Stack those grids in the last axis. The result will be a meshgrid with shape (grid_size, grid_size, 2)
    mesh_grid = torch.stack([mesh_grid_x1, mesh_grid_x2], dim=-1)
    # Move meshgrid to device.
    mesh_grid = mesh_grid.to(device)
    # Set model to evaluation mode.
    model.eval()

    # START TODO ########################
    # Reshape the meshgrid such that it can be input to the model's decoder.
    # Then, use the reshaped meshgrid as latent variables and run them through the decoder.
    raise NotImplementedError
    # END TODO ########################

    # the decoded shape is (grid_size ** 2, 1, 28, 28)
    # reshape it to recreate the original grid structure
    _, c, h, w = decoded.shape
    decoded_reshaped = decoded.reshape(grid_size, grid_size, c, h, w).cpu().numpy()
    return decoded_reshaped, step_size


def plot_latent_space(model: VAE, test_loader: data.DataLoader, device: str = "cpu") -> None:
    """

    Args:
        model: Trained model to get the latent space from.
        test_loader: Dataloader to compute the latent space reconstructions.
        device: Torch device ("cpu" or "cuda")
    """
    # Set model to evaluation mode.
    model.eval()
    # Get means and logvars for the entire test split.
    mus, logvars = get_mu_logvar(model, test_loader, device=device)

    # Move mean and calculated standard deviation to numpy
    mus = mus.detach().cpu().numpy()
    stddev = np.exp(0.5 * logvars.detach().cpu().numpy())

    # Collect labels for the test set
    labels = torch.cat([label for _, label in test_loader]).detach().numpy()

    # Calculate average mean per class with shape (dimensions=2, num_classes=10)
    class_mu_0 = np.bincount(labels, weights=mus[:, 0]) / np.bincount(labels)
    class_mu_1 = np.bincount(labels, weights=mus[:, 1]) / np.bincount(labels)
    mean_mu = np.stack((class_mu_0, class_mu_1))
    print(f"Per class mean of estimated mean: {mean_mu.shape}\n{mean_mu}")

    # Calculate average standard deviation per class with shape (dimensions=2, num_classes=10)
    class_stddev_0 = np.bincount(labels, weights=stddev[:, 0]) / np.bincount(labels)
    class_stddev_1 = np.bincount(labels, weights=stddev[:, 1]) / np.bincount(labels)
    mean_stddev = np.stack((class_stddev_0, class_stddev_1))
    print(f"Per class mean of estimated std deviation: {mean_stddev.shape}\n{mean_stddev}")

    # Get the minimum and maximum values of the latent space means.
    # We will use this as boundaries from which we sample in the latent space.
    latent_min = np.min(mean_mu - mean_stddev * 2, axis=1)
    latent_max = np.max(mean_mu + mean_stddev * 2, axis=1)

    # Produce a 20x20 2D grid of evenly spaced values between latent_min and latent_max
    grid_size = 20
    decoded, step_size = sample_on_grid(model, latent_min[0], latent_max[0], latent_min[1], latent_max[1],
                                        grid_size=grid_size, device=device)

    # Visualize the decoded images. We create one big image out of the small 400 images.
    # Input shape is (grid_size, grid_size, num_channels, height, width).
    # We split the array in the horizontal grid axis (0) and concatenate it in the width axis (4)
    decoded = np.concatenate(np.split(decoded, grid_size, axis=0), axis=4)
    # Then, we split the array in the vertical grid axis (1) and concatenate it in the height axis (3)
    decoded = np.concatenate(np.split(decoded, grid_size, axis=1), axis=3)
    # Finally we squeeze away the grid axis 0 and 1 and the 1-dimensional channel axis 2.
    # This leaves us with an output of shape (height * grid_size, width * grid_size) which we can plot.
    decoded = decoded.squeeze(0).squeeze(0).squeeze(0)
    plt.figure(figsize=(12, 12))
    plt.imshow(decoded, cmap='Greys')
    plt.axis('off')

    # visualize the mean mu and the mean standard deviation of each class
    # scale the mean accordingly, as the plot's axes represent pixels
    scale = (28 / step_size)

    mean_mu_scaled = 28 / 2 + scale.reshape(2, 1) * (mean_mu - latent_min.reshape(2, 1))
    # we scale the stddev by 2 so the arrows are extended a bit more
    mean_stddev_scaled = mean_stddev * scale.reshape(2, 1) * 2
    # plot the std deviation
    plt.errorbar(mean_mu_scaled[0], mean_mu_scaled[1],
                 yerr=mean_stddev_scaled[1], xerr=mean_stddev_scaled[0],
                 linestyle='None')
    # plot the means
    plt.scatter(mean_mu_scaled[0], mean_mu_scaled[1])

    for i, txt in enumerate([str(i) for i in range(0, 10)]):
        plt.annotate(txt, (mean_mu_scaled[0][i] + 3, mean_mu_scaled[1][i] - 3),
                     color='blue', bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 1, 'edgecolor': 'none'})
    plt.show()
