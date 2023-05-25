from functools import partial
from typing import Union
import torch
from torch import nn
from torch.nn import BCELoss


def update_discriminator(discriminator: nn.Module, generator: nn.Module, real_images: torch.Tensor,
                         noise: torch.Tensor, label: torch.Tensor, optimizerD: torch.optim.Optimizer):
    """
    update the discriminator D to maximize log(D(x)) + log(1 - D(G(z))), note that we keep the
    generator G fixed here

    Args:
        discriminator: discriminator network
        generator: generator network
        real_images: Input images in our original dataset with shape (batch_size, num_channels, height, width)
        noise: random noise of shape (batch_size, latent_dimension, 1, 1)
        label: Tensor of all ones of shape (batch_size,)
        optimizerD: optimizer that updates only the weights of discriminator network

    Returns:
        Tuple (errD_real, D_x, errD_fake, D_G_z1, fake_images) as denoted by the code comments
    """
    criterion = BCELoss()
    discriminator.zero_grad()
    # START TODO ########################
    # Pass the real_images batch through discriminator
    # Use bceloss to compute mean of error i.e -log(D(x)), between output
    # and labels denoted by errD_real
    raise NotImplementedError
    # END TODO ########################
    errD_real.backward()
    D_x = output.mean().item()

    fake_label = 1.0 - label
    # START TODO ########################
    # Generate fake_images batch from noise using generator
    # Pass the fake_images batch to discriminator
    # don't forget to use fake_images.detach() when passing, since generator is kept fixed
    # Use bceloss to compute mean of error i.e -log(1 - D(G(z))) between output
    # and fake_labels denoted by errD_fake
    raise NotImplementedError
    # END TODO ########################

    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
    # Update discriminator weights
    optimizerD.step()

    return (errD_real, D_x, errD_fake, D_G_z1, fake_images)


def update_generator(discriminator: nn.Module, generator: nn.Module, fake_images: torch.Tensor,
                     label: torch.Tensor, optimizerG: torch.optim.Optimizer):
    """
    update the generator G to maximize log(D(G(z))), note that we keep the discriminator D fixed here

    Args:
        discriminator: discriminator network
        generator: generator network
        fake_images: fake images batch generated from random noise using generator
                     has shape (batch_size, num_channels, height, width)
        label: Tensor of all ones of shape (batch_size,)
        optimizerG: optimizer that updates only the weights of generator network

    Returns:
        Tuple (errG, D_G_z2) as denoted by the code comments
    """
    criterion = BCELoss()
    generator.zero_grad()
    # START TODO ########################
    # Note that the G(z) is the fake_images batch
    # Pass the fake_images through discriminator
    # Use bceloss to compute mean of error i.e -log(D(G(z))) between output
    # and labels denoted by errG
    raise NotImplementedError
    # END TODO ########################
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()

    return (errG, D_G_z2)
