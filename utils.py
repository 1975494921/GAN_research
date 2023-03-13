import torch
from torch import nn
import torch.nn.functional as F
import math


class EqualizedLR_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = math.sqrt(2 / (in_ch * kernel_size[0] * kernel_size[1]))

        self.weight = nn.Parameter(torch.Tensor(out_ch, in_ch, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_ch))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + 10e-8)


class Minibatch_std(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        size = list(x.size())
        size[1] = 1

        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)), dim=1)


def depth_to_size(depth):
    return int(2 ** (depth + 1))


def size_to_depth(size):
    return int(math.log2(size) - 1)
