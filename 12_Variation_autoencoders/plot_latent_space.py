"""
Plot the latent space of a trained VAE model.
"""
import argparse
from pathlib import Path

import torch

from lib.explore_latent_space import plot_latent_space
from lib.model_vae import VAE
from lib.train_vae import setup_mnist_dataloader


def main():
    # setup script arguments and set the --help text to this scripts docstring at the top
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kl_loss_weight", type=float, default=1., help="KL divergence loss weight.")
    args = parser.parse_args()
    output_dir = f"results_kl{args.kl_loss_weight}"

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on device: {device}")

    # setup test dataloader
    _, test_loader = setup_mnist_dataloader()

    # create model with default values and reload weights from training
    model = VAE()
    model_file = Path(output_dir) / f"model.pth"
    assert model_file.is_file(), f"File {model_file} not found. Train the model first."
    model.load_state_dict(torch.load(model_file))

    # plot latent space of the model
    plot_latent_space(model, test_loader, device=device)


if __name__ == "__main__":
    main()
