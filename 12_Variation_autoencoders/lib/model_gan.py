"""
GAN model.

Original tutorial from PyTorch: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Author of the original tutorial: https://github.com/inkawhich
"""
from functools import partial
from typing import Union

import torch
from torch import nn


def weights_init(layer: Union[nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d]) -> None:
    """
    Custom weights initialization called on netG and netD.

    Will initialize all weights as described in the tutorial.

    Args:
        layer: Layer to initialize
    """
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


class Generator(nn.Module):
    """
    GAN Generator. The input will be a sample from the latent representation with shape (batch_size, latent_size, 1, 1).

    Create the following layers:

    - Transposed convolution with output channels: channels_multiplier * 8, kernel: 4, stride: 1, padding: 0,
        bias: False, then BatchNorm2d and activation_g.
    - 3 times the following: Transposed convolution with output channels: input channels / 2, kernel: 4, stride: 2,
        padding: 1, bias: False then BatchNorm2d and activation_g.
    - Finally a regular convolution with output channels: num_input_channels, kernel: 3, stride: 1, padding: 1,
        bias: False and a Tanh activation.

    Args:
        channels_multiplier: Convolution channel multiplier (called ngf in the GAN tutorial).
        num_input_channels: Input channels (3 for RGB images, called nc in the GAN tutorial).
        latent_size: Size of the latent space.

    """

    def __init__(self, channels_multiplier: int = 32, num_input_channels: int = 3, latent_size: int = 64):
        super().__init__()
        # Define activation function class.
        activation_g = nn.ReLU

        # START TODO ##########
        # Create the layers as described in the docstring.
        # layers = [ ... ]
        raise NotImplementedError
        # END TODO ##########
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor):
        """
        Generator forward pass.

        Args:
            z: Input latent representation with shape (batch_size, latent_size, 1, 1)

        Returns:
            Generated images with shape (batch_size, 3, 32, 32)
        """
        for layer in self.layers:
            z = layer(z)
        return z


class Discriminator(nn.Module):
    """
    GAN Discriminator. The input will be images with shape (batch_size, 3, 32, 32).

    Create the following layers:

    - Convolution with output channels: channels_multiplier, kernel: 4, stride: 2, padding: 1, bias: False
        then activation_d.
    - 2 times the following: Convolution with output channels: input channels * 2, kernel: 4, stride: 2,
        padding: 1, bias: False then BatchNorm2d and activation_d.
    - Finally a convolution with output channels: 1, kernel: 4, stride: 1, padding: 0, bias: False and
        a Sigmoid activation.

    Args:
        channels_multiplier: Convolution channel multiplicator (called ndf in the GAN tutorial).
        num_input_channels: Input channels (3 for RGB images, called nc in the GAN tutorial).

    """

    def __init__(self, channels_multiplier=64, num_input_channels=3):
        super().__init__()
        # Define activation function class.
        activation_d = partial(nn.LeakyReLU, 0.2, inplace=True)
        # START TODO ##########
        # Create the layers as described in the docstring.
        # layers = [ ... ]
        raise NotImplementedError
        # END TODO ##########
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        """
        Discriminator forward pass.

        Args:
            x: Input images with shape (batch_size, 3, 32, 32)

        Returns:
            Discriminator result with shape (batch_size, 1, 1, 1)
        """
        for layer in self.layers:
            x = layer(x)
        return x
