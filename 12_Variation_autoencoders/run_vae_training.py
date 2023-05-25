"""
Train a VAE model.
"""
import argparse

import torch

from lib.model_vae import VAE
from lib.train_vae import train_vae, setup_mnist_dataloader


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

    # setup dataloader
    train_loader, test_loader = setup_mnist_dataloader()

    # create model with default values
    model = VAE()
    train_vae(model, train_loader, test_loader, kl_loss_weight=args.kl_loss_weight,
              device=device, output_dir=output_dir)


if __name__ == "__main__":
    main()
