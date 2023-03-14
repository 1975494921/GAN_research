import torch
from torch import nn
import torch.nn.functional as F
import math


class EqLR_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super(EqLR_Conv2d, self).__init__()
        self.stride = stride
        self.padding = padding

        # Compute the scaling factor for the weights
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        self.scale = (2 / fan_in) ** 0.5

        # Initialize the weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Initialize the weight and bias parameters using normal and zeros initialization, respectively
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Scale the weight parameter by the scaling factor and perform the convolution operation
        out = F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)

        return out


class Minibatch_std(nn.Module):
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


# class PixelNorm(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + 10e-8)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sigma = torch.sum(x ** 2, dim=1, keepdim=True)
        return x / torch.sqrt(sigma + 10e-8)

def depth_to_size(depth):
    return int(2 ** (depth + 1))


def size_to_depth(size):
    return int(math.log2(size) - 1)
