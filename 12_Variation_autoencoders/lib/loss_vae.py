"""
Loss to train the VAE model.
"""
import torch

from torch import nn


class VAELoss(nn.Module):
    """
    Reconstruction loss and KL divergence losses summed over all elements of the batch.

    Args:
        kl_loss_weight:
    """

    def __init__(self, kl_loss_weight: float = 1.):
        """

        """
        super().__init__()
        self.input_flatten = nn.Flatten()
        self.kl_loss_weight = kl_loss_weight
        self.bce_loss = nn.BCELoss(reduction="sum")

    def forward(self, x_reconstructed, x, mu, logvar):
        """
        Forward pass of the Reconstruction + KL Loss.

        Todos:
            1. Compute Binary Crossentropy Loss between x and reconstructed x.
                Hint: As we create the loss with reduction="sum", it will automatically sum over the batch dimension.
            2. Compute KL divergence between distributions N(mu, sigma^2) and N(0, 1) as in pen-and-paper 1.2.
                Note that you need to compute DKL(P || Q) with P = N(mu, sigma^2) and Q = N(0, 1).
                Hint: We have 2-dimensional distributions here. You can compute the KL divergence for each dimension
                individually and then sum over the dimensions. As always, don't use for-loops.
            3. Sum both losses with KL divergence loss weighted by self.kl_loss_weight

        References:
            VAE Paper https://arxiv.org/abs/1312.6114

        Args:
            x_reconstructed: Reconstructed input with shape (batch_size, num_channels, height, width).
                For MNIST (batch_size, 1, 28, 28)
            x: Original input with shape (batch_size, num_channels, height, width). For MNIST (batch_size, 1, 28, 28)
            mu: Mean (batch_size, latent_size)
            logvar: Log of the variance (batch_size, latent_size)

        Returns:
            Scalar loss.
        """
        # flatten the reconstruction and input to be of shape (batch_size, input_size)
        x_reconstructed = self.input_flatten(x_reconstructed)
        x = self.input_flatten(x)

        # START TODO ########################
        # See docstring for details.
        raise NotImplementedError
        # END TODO ########################
        return rec_loss + kl_div_loss * self.kl_loss_weight
