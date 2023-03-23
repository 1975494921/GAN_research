import torch
from torch import nn
import torch.nn.functional as F
import math


class EqLR_Conv2d(nn.Module):
    """
    The implementation of the equalized learning rate convolutional layer.

    References
    ----------
    The idea of code is from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/utils.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super(EqLR_Conv2d, self).__init__()
        self.stride = stride
        self.padding = padding

        # Compute the scaling factor for the weights
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        self.scale = (2 / fan_in) ** 0.5

        # Initialize the weight and bias parameters
        weight_matrix = torch.zeros(out_channels, in_channels, *kernel_size)
        self.weight = nn.Parameter(weight_matrix)

        bias_matrix = torch.zeros(out_channels)
        self.bias = nn.Parameter(bias_matrix)

        # Initialize the weight and bias parameters using normal and zeros initialization, respectively
        self.weight_init()

    def weight_init(self):
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, data):
        # Scale the weight parameter by the scaling factor and perform the convolution operation
        out = F.conv2d(data, self.weight * self.scale, self.bias, stride=self.stride, padding=self.padding)

        return out


class Minibatch_std(nn.Module):
    """
    The implementation of the minibatch standard deviation layer.

    References
    ----------
    The idea of code is from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/utils.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Get the size of the input tensor
        size = list(x.shape)

        # set the feature dimension to 1, occupying the first dimension
        size[1] = 1

        # compute the standard deviation of each feature across the mini-batch
        std = torch.std(x, dim=0)

        # create a tensor with the mean standard deviation repeated along the feature dimension
        mean_std = torch.mean(std).expand(tuple(size))

        # concatenate the input tensor with the mean standard deviation tensor along the feature dimension
        out = torch.cat([x, mean_std], dim=1)

        return out


class PixelNorm(nn.Module):
    """
    The implementation of the pixel normalization layer.

    References
    ----------
    The idea of code is from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/utils.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sigma = torch.sum(x ** 2, dim=1, keepdim=True)
        return x / torch.sqrt(sigma + 10e-8)


def depth_to_size(depth):
    """Convert the depth of the layer to the size of the image."""
    return int(2 ** (depth + 1))



def size_to_depth(size):
    """Convert the size of the image to the depth of the layer."""
    return int(math.log2(size) - 1)
