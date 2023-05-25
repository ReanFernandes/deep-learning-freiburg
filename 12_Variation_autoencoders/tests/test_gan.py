import numpy as np
import torch
from lib.model_gan import Discriminator, Generator


def test_gan():
    # define hyperparameters
    latent_size = 64
    batch_size = 2
    num_channels = 4

    # test generator
    latent_noise = torch.randn((batch_size, latent_size, 1, 1))
    generator = Generator(channels_multiplier=num_channels, latent_size=latent_size)
    generated_images = generator(latent_noise)
    true_shape = (batch_size, 3, 32, 32)
    assert generated_images.shape == true_shape, (
        f"Generator output shape is {generated_images.shape} but should be {true_shape}")
    # count parameters
    g_params_truth = 43748
    g_params = np.sum([np.product(param.shape) for param in generator.parameters()])
    assert g_params == g_params_truth, f"Generator should have {g_params_truth} parameters but has {g_params}"

    # test discriminator
    images = torch.randn((batch_size, 3, 32, 32))
    discriminator = Discriminator(channels_multiplier=num_channels)
    output_disc = discriminator(images)
    true_shape_disc = (batch_size, 1, 1, 1)
    assert output_disc.shape == true_shape_disc, (
        f"Discriminator output shape is {output_disc.shape} but should be {true_shape_disc}")
    # count parameters
    d_params_truth = 3056
    d_params = np.sum([np.product(param.shape) for param in discriminator.parameters()])
    assert d_params == d_params_truth, f"Discriminator should have {d_params_truth} parameters but has {d_params}"

    # testing outputs of Generator and Discriminator
    torch.manual_seed(1000)
    latent_noise = torch.randn((2, 4, 1, 1))
    images = torch.randn((2, 1, 32, 32))
    generator = Generator(channels_multiplier=1, latent_size=4, num_input_channels=1)
    discriminator = Discriminator(channels_multiplier=1, num_input_channels=1)
    generated_images = generator(latent_noise).detach().cpu().numpy()
    output_disc = discriminator(images).detach().cpu().numpy()
    pooled_generated_images = np.sum(np.transpose(np.reshape(generated_images,
                                     (2, 1, 4, 8, 4, 8)), (0, 1, 2, 4, 3, 5)), axis=(-1, -2))

    true_output_disc = np.array([[[[0.33339936]]], [[[0.47757638]]]])
    true_pooled_generated_images = np.array([[[[8.034148, 8.571291, 9.892173, 9.802911],
                                             [8.788626, 7.984515, 9.268524, 9.225212],
                                             [9.317131, 12.20800, 10.682001, 7.3813562],
                                             [7.919389, 9.032231, 8.341071, 8.030525]]],

                                             [[[8.566567, 10.598253, 11.736382, 10.6683035],
                                              [9.646444, 9.193956, 10.95442, 12.050827],
                                              [8.904692, 10.281017, 10.374569, 8.506515],
                                              [8.256763, 8.119479, 9.861954, 9.638344]]]])

    err_msg_g = "Output of Generator is incorrect"
    err_msg_d = "Output of Discriminator is incorrect"
    np.testing.assert_allclose(pooled_generated_images, true_pooled_generated_images, rtol=1e-06, err_msg=err_msg_g)
    np.testing.assert_allclose(output_disc, true_output_disc, rtol=1e-06, err_msg=err_msg_d)


if __name__ == "__main__":
    test_gan()
    print('Test complete.')
