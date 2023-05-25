import os
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.cuda # using gpu for the training


def calculate_jacobian(
        input: torch.Tensor, target: torch.Tensor, model: torch.nn.Module, criterion: torch.nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the Jacobian, i.e., the gradient of the loss w.r.t. the input (for a 2d image)

    Args:
        input: The input tensor with shape (batch_size, channels, height, width)
        target: The target output tensor with shape (batch_size,)
        model: The neural network model (whose output is also returned by this function)
        criterion: The loss criterion

    Returns:
        Tuple of:
            A torch.Tensor of gradients with shape (batch_size, channels, height, width)
            A torch.Tensor of model outputs with shape (batch_size, num_classes)
    """
    # START TODO #############
    # make sure the subsequent backward pass calculate the gradients w.r.t. the input
    
    # passing all arguments to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    target = target.to(device)
    model = model.to(device)
    input.requires_grad = True

    # END TODO #############
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    # START TODO #############
    gradient = input.grad
    
    # END TODO #############
    return gradient.cpu(), output.cpu()


def create_dataloader(batch_size) -> DataLoader:
    """
    Helper function for loading data.
    create the dataloader from sample ImageNet images given in the sample_images folder
    Returns:
        dataloader
    """
    num_workers = 0  # on windows, multiple workers can lead to errors
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # these are the normalization settings for ImageNet

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    vis_data = torchvision.datasets.ImageFolder(
        root=os.path.join("data", "imagenet-sample-images-master"), transform=transform
    )
    vis_loader = torch.utils.data.DataLoader(
        dataset=vis_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return vis_loader
