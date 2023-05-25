from __future__ import annotations
from torch import nn
from neps.search_spaces.architecture.primitives import AbstractPrimitive


class DownSampleBlock(AbstractPrimitive):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(locals())
        self.conv_a = ReLUConvBN(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv_b = ReLUConvBN(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
        )

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        residual = self.downsample(inputs)
        return residual + basicblock


class ReLUConvBN(AbstractPrimitive):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__(locals())

        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return self.op(x)


class AvgPool(AbstractPrimitive):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.op = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.op(x)


class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super().__init__(locals())

    def forward(self, x):
        return x


class Zero(AbstractPrimitive):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, stride, **kwargs):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super().__init__(locals())
        self.stride = int(stride)

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        else:
            return x[:, :, :: self.stride, :: self.stride].mul(0.0)
