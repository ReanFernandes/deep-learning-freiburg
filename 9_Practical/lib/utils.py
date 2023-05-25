"""Helper functions for loading data"""
from typing import Tuple

import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def create_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders

    Returns:
        Tuple of (trainloader, testloader)
    """

    os.makedirs('../data', exist_ok=True)
    batch_size = 128
    num_workers = 0  # on windows, multiple works can lead to errors

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    indices = np.arange(len(trainset))
    train_indices, test_indices = train_test_split(indices, train_size=500 * 10, stratify=trainset.targets,
                                                   random_state=1337)

    train_dataset = Subset(trainset, train_indices)
    test_dataset = Subset(trainset, test_indices[:1000])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def show_random_train_images() -> None:
    """Sample 4 random images from the training set and display them in a grid"""

    # get some random training images
    trainloader, _ = create_dataloaders()
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    img = torchvision.utils.make_grid(images[:4, :, :, :])

    # show images
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
