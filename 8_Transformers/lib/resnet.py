import torch.nn as nn

from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.resnet(x)
