import torch

from lib.loss_vae import VAELoss


def test_vae_loss():
    batch_size, in_channels, in_height, in_width, hidden_size, latent_size = 2, 1, 28, 28, 100, 2
    loss_fn = VAELoss()
    torch.manual_seed(420)

    image_batch = torch.rand((batch_size, in_channels, in_height, in_width))
    decoded_batch = torch.rand((batch_size, in_channels, in_height, in_width))
    mu = torch.randn(batch_size, latent_size)
    logvar = torch.randn(batch_size, latent_size)

    loss = loss_fn(decoded_batch, image_batch, mu, logvar)
    loss_truth = 1571.545166015625
    assert (loss - loss_truth) ** 2 < 1e-8, f"Loss is {loss} but should be {loss_truth}"


if __name__ == "__main__":
    test_vae_loss()
    print('Test complete.')
