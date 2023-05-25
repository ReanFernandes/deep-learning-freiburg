"""
Bonus exercise: Train the DCGan.

Original tutorial from PyTorch: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Author of the original tutorial: https://github.com/inkawhich
"""

import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm

from lib.model_gan import Generator, Discriminator, weights_init
from lib.train_gan import update_discriminator, update_generator


def single_image_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a single image to numpy.

    Args:
        image_tensor: Image with shape (channels, height, width)

    Returns:
        Numpy image with shape (height, width, channels)
    """
    return np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))


def batch_images_to_numpy(image_batch: torch.Tensor, max_images: int = 64, padding: int = 2, normalize: bool = True
                          ) -> np.ndarray:
    """
    Convert a batch of images to numpy and return an image grid.

    Args:
        image_batch: Image batch with shape (batch_size, channels, height, width)
        max_images: Limit the grid to this number of images.
        padding: Add black border between the images.
        normalize: Normalize the images to be in the value range [0, 1].

    Returns:
        Batch of numpy images with shape (batch_size, height, width, channels)
    """
    return single_image_to_numpy(vutils.make_grid(image_batch[:max_images], padding=padding, normalize=normalize))


def get_augmentation(image_size: int) -> transforms.Compose:
    """
    Given the input image size, define data augmentations.

    Args:
        image_size: Input image size.

    Returns:
        Composition of transforms that the dataloader will do.
    """
    return transforms.Compose([
        # First resize the smaller edge to our target image size, keeping the aspect ratio
        transforms.Resize(image_size),
        # 50-50 chance to either random crop center crop.
        transforms.RandomChoice([
            transforms.RandomCrop(image_size), transforms.CenterCrop(image_size)
        ]),
        # 50% chance to flip the image horizontally
        transforms.RandomHorizontalFlip(p=0.5),
        # Prepare the input to the model
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def main():
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use this if you want new results

    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    workers = 2  # Number of workers for the dataloader
    batch_size = 128  # Batch size during training
    image_size = 32  # Image size
    nc = 3  # Number of channels in the training images. For color images this is 3
    nz = 64  # Size of z latent vector (i.e. size of generator input)
    ngf = 32  # Size of feature maps in generator
    ndf = 32  # Size of feature maps in discriminator
    num_epochs = 500  # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    checkpoint_every = 100  # How often to save images
    beginning_checkpoint_every = 10  # At the very start of training checkpoint more often

    # Create flowers dataset
    dset_name = "flowers"
    dataroot_train = "dataset_flowers_64px"
    assert Path(dataroot_train).is_dir(), f"Flowers dataset not found on path {dataroot_train}."
    dataset_train = dset.ImageFolder(root=dataroot_train, transform=get_augmentation(image_size))

    # Name the experiment
    run_name = f"{dset_name}_b{batch_size}_nz{nz}_c{ngf},{ndf}_lr{lr:.1e}"

    # Create the dataloaders
    dataloader_train = data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Create model ----------

    # Create the generator
    netG = Generator(ngf, nc, nz)
    netG = netG.to(device)
    netG.apply(weights_init)  # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02
    print(netG)

    # Create the Discriminator
    netD = Discriminator(ndf, nc).to(device)
    netD = netD.to(device)
    netD.apply(weights_init)  # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise_all = torch.randn(1024, nz, 1, 1, device=device)
    fixed_noise = fixed_noise_all[:64]

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # setup model saving
    output_dir = Path(
        f"results_gan/run_{run_name}_{str(datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]}")
    os.makedirs(output_dir, exist_ok=False)
    print(f"Saving output to {output_dir}")
    # save fixed noise
    fixed_noise_file = output_dir / f"fixed_noise.pth"
    torch.save(fixed_noise_all, fixed_noise_file)
    iters = 0

    # Plot some training images
    real_image_batch, label_batch = next(iter(dataloader_train))
    print(f"Batch shape: {len(real_image_batch)} * {real_image_batch[0].shape}")
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    output_file = output_dir / f"training_images.png"
    plt.imsave(output_file, batch_images_to_numpy(real_image_batch))
    plt.close()
    print(f"Saved training images to {output_file}")

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    print(f"Starting Training Loop on {device}")

    # For each epoch
    print(f"One epoch corresponds to {len(dataloader_train)} iters.")
    pbar = tqdm(total=num_epochs)
    start_epoch = iters // len(dataloader_train)
    for epoch in range(start_epoch, num_epochs):
        # For each batch in the dataloader
        # pbar.set_description(desc=f">T:{iters} E:{epoch}/{num_epochs} S:{len(dataloader_train)} ")
        for i, (real_images, _) in enumerate(dataloader_train, 0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # Train with all-real batch
            real_cpu = real_images.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            errD_real, D_x, errD_fake, D_G_z1, fake_images = update_discriminator(netD, netG, real_cpu,
                                                                                  noise, label, optimizerD)
            errD = errD_real + errD_fake

            # (2) Update G network: maximize log(D(G(z)))
            errG, D_G_z2 = update_generator(netD, netG, fake_images, label, optimizerG)

            # Output training stats
            if i % 50 == 0:
                pbar.set_description(desc=f"iters:{iters} E:{epoch}/{num_epochs} S:{i}/{len(dataloader_train)} ")
                err_d, err_g = errD.item(), errG.item()
                pbar.write(f"LD {err_d:.4f} LG {err_g:.4f} Dx {D_x:.4f} DGz {D_G_z1:.4f} {D_G_z2:.4f}")
                if err_d < 1e-3 or err_d > 50 or err_g < 1e-3 or err_g > 50:
                    pbar.write(f"WARNING: Generator/Discriminator loss is too high/low. Networks might not converge.")

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % checkpoint_every == 0 or (iters < checkpoint_every and iters % beginning_checkpoint_every == 0)
                    or ((epoch == num_epochs - 1) and (i == len(dataloader_train) - 1))):
                pbar.write(f"iters: {iters} E:{epoch}/{num_epochs}"
                           + " - Validation. Saving generator output, loss graphs to results_gan/")

                # generate images images
                output_file_fake = output_dir / f"output_{iters}_steps.png"
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                fake_imgs = vutils.make_grid(fake, padding=2, normalize=True)
                plt.imsave(output_file_fake, batch_images_to_numpy(fake_imgs))
                img_list.append(fake_imgs)

                # save loss graph
                plt.figure(figsize=(10, 5))
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(G_losses, label="G")
                plt.plot(D_losses, label="D")
                plt.xlabel(f"iterations ({len(dataloader_train)} iterations ~ 1 epoch)")
                plt.ylabel("Loss")
                plt.legend()
                output_file_loss = output_dir / f"losses_{iters}_steps.png"
                plt.savefig(output_file_loss)
                plt.close()

            iters += 1
        pbar.update()

    # save model
    model_file_d = output_dir / "model_d.pth"
    model_file_g = output_dir / "model_g.pth"
    opt_file_d = output_dir / "optimizer_d.pth"
    opt_file_g = output_dir / "optimizer_g.pth"
    torch.save(netD.state_dict(), model_file_d)
    torch.save(netG.state_dict(), model_file_g)
    torch.save(optimizerD.state_dict(), opt_file_d)
    torch.save(optimizerG.state_dict(), opt_file_g)


if __name__ == "__main__":
    main()
