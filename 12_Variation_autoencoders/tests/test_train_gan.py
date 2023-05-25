import numpy as np
import torch
from torch import optim
from lib.model_gan import Discriminator, Generator
from lib.train_gan import update_discriminator, update_generator


def test_train_gan():
    # define hyperparameters
    latent_size = 8
    batch_size = 2
    num_channels = 1
    lr = 0.0002

    # set seed
    torch.manual_seed(1111)
    np.random.seed(2222)

    # create generator
    label = torch.ones(batch_size,)
    latent_noise = torch.randn((batch_size, latent_size, 1, 1))
    generator = Generator(channels_multiplier=num_channels, latent_size=latent_size)

    # create discriminator
    images = torch.randn((batch_size, 3, 32, 32))
    discriminator = Discriminator(channels_multiplier=num_channels)

    # Setup Adam optimizers for both generator and discriminator
    optimizerD = optim.SGD(discriminator.parameters(), lr=lr)
    optimizerG = optim.SGD(generator.parameters(), lr=lr)

    # testing outputs of update_generator and update_discriminator functions
    errD_real, D_x, errD_fake, D_G_z1, fake_images = update_discriminator(discriminator, generator, images,
                                                                          latent_noise, label, optimizerD)
    errG, D_G_z2 = update_generator(discriminator, generator, fake_images, label, optimizerG)

    errD_real = errD_real.item()
    errD_fake = errD_fake.item()
    errG = errG.item()
    ud_output = np.array([errD_real, D_x, errD_fake, D_G_z1])
    ug_output = np.array([errG, D_G_z2])

    # Test outputs
    true_ud_output = np.array([0.5513347387313843, 0.5765248537063599, 0.5437031388282776, 0.41870588064193726])
    true_ug_output = np.array([0.8764630556106567, 0.41724222898483276])
    err_msg_d = "update_discriminator output incorrect"
    err_msg_g = "update_generator output incorrect"
    np.testing.assert_allclose(true_ud_output, ud_output, err_msg=err_msg_d, rtol=1e-6)
    np.testing.assert_allclose(true_ug_output, ug_output, err_msg=err_msg_g, rtol=1e-6)


if __name__ == "__main__":
    test_train_gan()
    print('Test complete.')
