"""
Model definition of Variational Auto-Encoder.
"""

from typing import Tuple

import torch
from torch import nn


class VAE(nn.Module):
    """
    Variational auto encoder.

    Create the VAE with the MLP encoder having one hidden layer and two output layers (mean and log of the variance).
    Input, hidden and latent size are given by the arguments.

    As an activation function for the hidden layers we will use ReLU. For the decoder output we will use sigmoid.
    We estimate the log of the variance with base e (natural logarithm).

    The decoder should be as powerful as the encoder.

    Args:
        in_channels: Number of input channels (1 for grayscale MNIST images)
        in_height: Input image height (28 for MNIST)
        in_width: Input image width (28 for MNIST)
        hidden_size: Hidden size.
        latent_size: Dimension of the latent space.
    """

    def __init__(self, in_channels: int = 1, in_height: int = 28, in_width: int = 28, hidden_size: int = 100,
                 latent_size: int = 2):
        super().__init__()

        # Calculate input size.
        input_size = in_channels * in_height * in_width

        # Create ReLU activation for the hidden layers.
        self.activation = nn.ReLU()

        # Create Sigmoid activation for the output layer.
        self.activation_output = nn.Sigmoid()

        # Create the Flatten and Unflatten operations needed to work on images with linear layers.
        self.input_flatten = nn.Flatten()
        self.output_unflatten = nn.Unflatten(1, (in_channels, in_height, in_width))

        # START TODO ########################
        # Create the linear layers needed for the VAE: Encode input to hidden size, compute mu and logvar from
        # hidden size, decode from latent size to hidden size and from hidden size to input size.
        raise NotImplementedError
        # END TODO ########################

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode x to distribution.

        Compute the hidden representation and use the activation function.
        Esimate the mean and the log of the variance from the hidden representation, i.e. use two linear layers
        each going from the hidden representation to the latent representation.
        Return the mean and the log of the variance.

        Args:
             x: Input with shape (batch_size, num_channels, height, width). For MNIST (batch_size, 1, 28, 28)

        Returns:
            Mean and Logvar both with shape (batch_size, latent_size)
        """
        # Flatten x to shape (batch_size, input_size)
        x = self.input_flatten(x)

        # START TODO ########################
        raise NotImplementedError
        # END TODO ########################
        return mu, logvar

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        """
        Sample from the estimated distribution.

        In training mode, return a sample from N(mu, sigma^2). Note that you have to use function torch.randn_like
            to sample from a standard normal distribution first and then shift and scale the result.
            The reason we need this "reparametrization trick" is that sampling from a normal distribution does not
            provide a gradient while sampling from a standard normal distribution does provide a gradient.
        In evaluation mode, just return the mean.

        Hint: You estimate the log of the variance, so you need to transform
        this to standard deviation first.
        """
        if self.training:
            # START TODO ########################
            # During training we estimate log(var) and sample from a normal distribution N(mu, var)
            # Remember that: stddev = sqrt(var)
            raise NotImplementedError
            # END TODO ########################
        else:
            # START TODO ########################
            # During validation, we return the mean
            raise NotImplementedError
            # END TODO ########################

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation.

        Compute the hidden representation from the latent space z, use ReLU as activation.
        Then reconstruct the signal from the hidden representation, using a sigmoid activation.

        Args:
            z: Latent representation with shape (batch_size, latent_size)

        Returns:
            Decoded images with shape (batch_size, num_channels, height, width). For MNIST (batch_size, 1, 28, 28)
        """
        # START TODO ########################
        # Compute the hidden representation from the latent space z, use ReLU as activation.
        # Then, compute the output representation, use Sigmoid as activation.
        raise NotImplementedError
        # END TODO ########################

        # Unflatten the output back to an image.
        return self.output_unflatten(output)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Call the encode, reparameterize and decode functions. Return decoded result, mean and logvar.

        Args:
            x: Input with shape (batch_size, num_channels, height, width). For MNIST (batch_size, 1, 28, 28)

        Returns:
            Tuple of decoded result with shape (batch_size, num_channels, height, width),
                mean with shape (batch_size, latent_size)
                and logvar with shape (batch_size, latent_size)
        """
        # START TODO ########################
        raise NotImplementedError
        # END TODO ########################
