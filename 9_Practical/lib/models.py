"""CNN models to train"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1(nn.Module):
    """
        The CNN model with 3 filters, kernel size 5, and padding 2
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # initialize required parameters / layers needed to build the network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(in_features=768, out_features=10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape [batch_size, *feature_dim] (minibatch of data)
        Returns:
            scores: Pytorch tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.conv1(x)
        # print(f"after conv layer {x.shape}")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print(f"after relu layer {x.shape}")
        x = torch.flatten(x, 1)
        # x = self.flatten(x)
        # print(f"after flat layer {x.shape}")
        x = self.fc1(x)
        # print(f"after fc layer {x.shape}")
        # END TODO #############
        return x


class ConvNet2(nn.Module):
    """
        The CNN model with 16 filters, kernel size 5, and padding 2
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(in_features=4096, out_features=10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.conv1(x)
        # print(f"after conv layer {x.shape}")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print(f"after relu layer {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"after flat layer {x.shape}")
        x = self.fc1(x)
        # print(f"after fc layer {x.shape}")
        # END TODO #############
        return x


class ConvNet3(nn.Module):
    """
        The CNN model with 16 filters, kernel size 3, and padding 1
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # Define the layers need to build the network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=4096, out_features=10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.conv1(x)
        # print(f"after conv layer {x.shape}")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print(f"after relu layer {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"after flat layer {x.shape}")
        x = self.fc1(x)
        # print(f"after fc layer {x.shape}")
        # END TODO #############
        return x


class ConvNet4(nn.Module):
    """
        The CNN model with 16 filters, kernel size 3, padding 1 and batch normalization
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # Define the layers need to build the network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.dense1_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(in_features=4096, out_features=10)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.conv1(x)
        # print(f"after conv layer {x.shape}")
        x = self.dense1_bn(x)
        # print(f"after bn layer {x.shape}")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print(f"after relu layer {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"after flat layer {x.shape}")
        x = self.fc1(x)
        # print(f"after fc layer {x.shape}")
        # END TODO #############
        return x


class ConvNet5(nn.Module):
    """ Your custom CNN """

    def __init__(self):
        super().__init__()

        # START TODO #############
        raise NotImplementedError
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        raise NotImplementedError
        # END TODO #############
        return x
