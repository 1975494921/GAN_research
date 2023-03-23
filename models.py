from collections import OrderedDict
from torch import nn
from utils import EqLR_Conv2d, Minibatch_std, PixelNorm
from utils import depth_to_size, size_to_depth
from config import Config
import torch.nn.functional as F
import numpy as np


def print_func(*args):
    # use this function to print debug information
    if Config.debug:
        print(*args)


def weights_init(w):
    """
    Initialize the weights of the network

    Parameters
    ----------
    w: nn.Module
        the network to be initialized

    References
    ----------
    This code is written on my own
    """
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.1)
        nn.init.constant_(w.bias.data, 0)

    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(w.weight.data)
        nn.init.constant_(w.bias.data, 0)


class Noise_Net(nn.Module):
    """
    Preprocess the noise vector to make the generator more stable

    Parameters
    ----------
    in_dim: int
        the dimension of the noise vector
    out_dim: int
        the dimension of the output noise vector

    References
    ----------
    The code is written on my own
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = int(in_dim // 2)
        self.FN = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
        )
        self.FN.apply(weights_init)

    def forward(self, x):
        x = self.FN(x)
        return x


class FromRGB(nn.Module):
    """
    The layer to convert the RGB image to the feature map for the discriminator

    Parameters
    ----------
    in_channel: int
        the number of input channels
    out_channel: int
        the number of output channels

    References
    ----------
    The code is modified from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/model.py
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = EqLR_Conv2d(in_channel, out_channel, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)
        return x


class ToRGB(nn.Module):
    """
    The layer to convert the feature map to the RGB image for the generator

    Parameters
    ----------
    in_channel: int
        the number of input channels
    out_channel: int
        the number of output channels

    References
    ----------
    The code is modified from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/model.py
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = EqLR_Conv2d(in_channel, out_channel, kernel_size=(1, 1))

    def forward(self, data):
        return self.conv(data)


class Residual_Block(nn.Module):
    """
    The residual block for the generator

    Parameters
    ----------
    direct: nn.Module
        the direct path of the residual block
    in_channel: int
        the number of input channels
    out_channel: int
        the number of output channels
    kernel_size: tuple
        the size of the kernel
    stride: tuple
        the stride of the convolution
    padding: tuple
        the padding of the convolution

    References
    ----------
    The code is written on my own
    """

    def __init__(self, direct, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)):
        super().__init__()
        self.direct = direct
        self.residual = nn.Identity()
        if in_channel != out_channel:
            self.residual = EqLR_Conv2d(in_channel, out_channel,
                                        kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, data):
        out = self.direct(data)

        return out + self.residual(data)


class G_Block(nn.Module):
    """
    The basic block of the Generator

    Parameters
    ----------
    in_channel: int
        the number of input channels
    out_channel: int
        the number of output channels
    initial_block: bool
        indicate whether it is the initial block
    resnet: bool
        indicate whether it is the residual block

    References
    ----------
    The idea of the code is from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/model.py
    The parameters of the Progressive GAN are from https://arxiv.org/pdf/1710.10196.pdf

    There are serval improvements I made:
    1. The residual connection is added to the basic block
    2. The up_sample is replaced by nn.Identity() in the initial block, the code is more concise
    3. Use the OrderedDict to store the layers, the code is more concise
    """
    def __init__(self, in_channel, out_channel, initial_block=False, resnet=False):
        super().__init__()
        up_sample = nn.Identity()

        if initial_block:
            conv_layer1 = EqLR_Conv2d(in_channel, out_channel, kernel_size=(4, 4), padding=(3, 3))

        else:
            up_sample = nn.Upsample(scale_factor=2, mode='nearest')
            conv_layer1 = EqLR_Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))

        conv_layer2 = EqLR_Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
        if resnet:
            conv_layer1 = Residual_Block(conv_layer1, in_channel, out_channel)
            conv_layer2 = Residual_Block(conv_layer2, out_channel, out_channel)

        self.block = nn.Sequential(
            OrderedDict([
                ('up_sample', up_sample),
                ('conv_layer_1', conv_layer1),
                ('leaky_relu_1', nn.LeakyReLU(0.2)),
                ('pixel_norm_1', PixelNorm()),
                ('conv_layer_2', conv_layer2),
                ('leaky_relu_2', nn.LeakyReLU(0.2)),
                ('pixel_norm_2', PixelNorm())
            ])
        )

    def forward(self, data):
        return self.block(data)


class Generator(nn.Module):
    """
    The Generator class of the Progressive GAN

    Parameters
    ----------
    noise_dim: int
        the dimension of the noise
    latent_size: int
        the dimension of the latent vector
    out_res: int
        the output resolution of the generator
    noise_net: nn.Module
        indicate whether to use the noise network
    resnet: bool
        indicate whether to use the residual block

    References
    ----------
    The idea of the code is from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/model.py

    To fit my project requirements, I changed the whole structure of the code and added some features:
    1. The noise network is allowed to be used, which means the noise is not simply from the random normal distribution,
       this is a good way to improve the quality of the generated images.

    2. The residual block is allowed to be used, which means the generator is allowed to be deeper.

    3. The control of the alpha is added, which means the alpha can be changed during the training process, instead of
       just being from 0 to 1 when the training starts. The speed of the alpha change can also be controlled.

    4. In the training process, the gradient on different layers can be scaled, which means the training process can be
       more stable if new layers are added, or the previous layers are frozen.

    5. The depth setting is added, which means the depth of the generator can be changed during the training process,
         instead of just being from 1 to the max depth when the training starts. This allows the generator to be more flexible.
    """

    def __init__(self, noise_dim, latent_size, out_res, noise_net=False, resnet=False):
        super().__init__()
        self.resnet = resnet
        self._current_depth = 1
        self.alpha = 0.1
        self.delta_alpha = 1e-3
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.noise_net = nn.Identity()
        self.current_net = nn.ModuleList([])
        self.toRGBs = nn.ModuleList([])
        self.out_res = out_res
        self.max_depth = size_to_depth(out_res)
        if noise_net:
            self.noise_net = Noise_Net(noise_dim, latent_size)

        # append the initial block
        self.current_net.append(G_Block(latent_size, 512, initial_block=True, resnet=self.resnet))
        self.toRGBs.append(ToRGB(512, 3))

        for depth in range(2, self.max_depth + 1):
            if depth < 6:
                in_channel, out_channel = 512, 512
            else:
                in_channel, out_channel = 512 // np.power(2, depth - 6), 512 // np.power(2, depth - 5)

            self.current_net.append(G_Block(in_channel, out_channel, resnet=self.resnet))
            self.toRGBs.append(ToRGB(out_channel, 3))

    def forward(self, noise):
        noise = self.noise_net(noise)
        x = noise.unsqueeze(-1).unsqueeze(-1)
        for block in self.current_net[:int(self._current_depth) - 1]:
            x = block(x)

        model_out = self.current_net[self._current_depth - 1](x)
        image_out = self.toRGBs[self._current_depth - 1](model_out)

        if self._current_depth > 1 and self.alpha < 1:
            x_old = self.up_sample(x)
            old_image_out = self.toRGBs[self._current_depth - 2](x_old)
            image_out = (1 - self.alpha) * old_image_out + self.alpha * image_out

        return image_out

    def get_alpha(self):
        return self.alpha

    def increase_alpha(self, delta=None):
        if delta is None:
            delta = self.delta_alpha

        self.alpha = min(self.alpha + delta, 1)

    def growing_net(self, alpha_start=0, delta_alpha=1e-3):
        if self._current_depth < self.max_depth:
            self.delta_alpha = delta_alpha
            self.alpha = alpha_start
            self._current_depth += 1
            im_size = depth_to_size(self._current_depth)
            print("Generator depth is set to {}, image size {}x{}".
                  format(self._current_depth, im_size, im_size))
        else:
            print("Net cannot be grow, depth should be within {} to {}".format(1, self.max_depth))

    def set_depth(self, depth: int, alpha_start=0, delta_alpha=1e-3):
        if 1 <= depth <= self.max_depth:
            self._current_depth = int(depth)
            self.delta_alpha = delta_alpha
            self.alpha = alpha_start
            im_size = depth_to_size(self._current_depth)
            print("Generator depth is set to {}, image size {}x{}".
                  format(self._current_depth, im_size, im_size))
        else:
            print("Depth value invalid..., should be within {} to {}".format(1, self.max_depth))

    def scale_grad(self, factor=0.5):
        for block in self.current_net[:self._current_depth - 1]:
            for param in block.parameters():
                if param.grad is not None:
                    param.grad *= factor

    def scale_grad_rgb(self, factor=0.5):
        for param in self.toRGBs[self._current_depth - 1].parameters():
            if param.grad is not None:
                param.grad *= factor

    def scale_grad_noise(self, factor=0.5):
        for param in self.noise_net.parameters():
            if param.grad is not None:
                param.grad *= factor


class D_Block(nn.Module):
    """
    The discriminator block is a convolutional block with a minibatch standard deviation layer, which is used to
    calculate the standard deviation of the feature maps of the current layer.

    Parameters
    ----------
    in_channel: int
        The number of input channels.
    out_channel: int
        The number of output channels.
    last_block: bool
        Whether the block is the final block of the discriminator.
    resnet: bool
        Whether the block is a residual block.

    References
    ----------
    The idea of the code is from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/model.py
    The parameters of the Progressive GAN are from https://arxiv.org/pdf/1710.10196.pdf

    """
    def __init__(self, in_channel, out_channel, last_block=False, resnet=False):
        super().__init__()
        minibatchstd_layer = nn.Identity()

        conv_layer_1 = EqLR_Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
        conv_layer_2 = EqLR_Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
        out_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        if last_block:
            minibatchstd_layer = Minibatch_std()
            in_channel += 1
            conv_layer_1 = EqLR_Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
            conv_layer_2 = EqLR_Conv2d(out_channel, out_channel, kernel_size=(4, 4), padding=(0, 0))
            out_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_channel, 1)
            )

        if resnet:
            conv_layer_1 = Residual_Block(self.conv1, in_channel, out_channel)
            conv_layer_2 = Residual_Block(self.conv2, out_channel, out_channel)

        self.apply(weights_init)
        self.block = nn.Sequential(
            OrderedDict([
                ('minibatchstd_layer', minibatchstd_layer),
                ('conv_layer_1', conv_layer_1),
                ('leaky_relu_1', nn.LeakyReLU(0.2)),
                ('conv_layer_2', conv_layer_2),
                ('leaky_relu_2', nn.LeakyReLU(0.2)),
                ('out_layer', out_layer)
            ])
        )

    def forward(self, data):
        data = self.block(data)

        return data


class Discriminator(nn.Module):
    """
    The Discriminator class of the Progressive GAN

    Parameters
    ----------
    latent_size: int
        The dimension of the latent vector.
    img_size: int
        The size of the generated image.
    resnet: bool
        Whether the generator is a residual generator.

    References
    ----------
    The idea of the code is from https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/model.py

    To fit my project requirements, I changed the whole structure of the code and added some features:

    1. The residual block is allowed to be used, which means the generator is allowed to be deeper.

    2. The control of the alpha is added, which means the alpha can be changed during the training process, instead of
       just being from 0 to 1 when the training starts. The speed of the alpha change can also be controlled.

    3. In the training process, the gradient on different layers can be scaled, which means the training process can be
       more stable if new layers are added, or the previous layers are frozen.

    4. The depth setting is added, which means the depth of the generator can be changed during the training process,
         instead of just being from 1 to the max depth when the training starts. This allows the generator to be more flexible.
    """
    def __init__(self, latent_size, img_size, resnet=False):
        super().__init__()
        self._current_depth = 1
        self.alpha = 0.1
        self.delta_alpha = 1e-3
        self.down_sample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.img_size = img_size
        self._max_depth = size_to_depth(img_size)
        self.current_net = nn.ModuleList([])
        self.fromRGBs = nn.ModuleList([])
        self._internal_index = self._max_depth - self._current_depth
        self.resnet = resnet
        self.train_last_layer_only = False

        print_func("max_depth: {}".format(self._max_depth))
        dim_list = []
        for depth in range(1, size_to_depth(img_size)):
            if depth < 4:
                in_channel, out_channel = 512, 512
            else:
                in_channel, out_channel = 512 // np.power(2, depth - 3), 512 // np.power(2, depth - 4)

            dim_list.append((in_channel, out_channel))

        dim_list.reverse()
        for in_channel, out_channel in dim_list:
            print_func(in_channel, out_channel)
            self.current_net.append(D_Block(in_channel, out_channel, resnet=self.resnet))
            self.fromRGBs.append(FromRGB(3, in_channel))

        self.current_net.append(D_Block(512, 512, last_block=True, resnet=self.resnet))
        self.fromRGBs.append(FromRGB(3, 512))

    def forward(self, image_rgb):
        print_func("start forward......")
        print_func("input shape: {}".format(image_rgb.shape))
        RGBBlock_index = self._internal_index

        print_func("fromRGBs block index: {}".format(RGBBlock_index))
        data = self.fromRGBs[RGBBlock_index](image_rgb)
        print_func("fromRGB shape: {}".format(data.shape))

        data = self.current_net[self._internal_index](data)
        print_func("last_net shape: {}".format(data.shape))

        if self.alpha < 1 and self._internal_index < self._max_depth - 1:
            image_rgb = self.down_sample(image_rgb)
            x_old = self.fromRGBs[self._internal_index + 1](image_rgb)
            data = (1 - self.alpha) * x_old + self.alpha * data

        for block in self.current_net[self._internal_index + 1:]:
            print_func(data.shape)
            print_func(block)
            data = block(data)

        return data

    def set_depth(self, depth: int, alpha_start=0, delta_alpha=1e-3):
        if 1 <= depth <= self._max_depth:
            self._current_depth = int(depth)
            self._internal_index = self._max_depth - self._current_depth
            self.delta_alpha = delta_alpha
            self.alpha = alpha_start
            im_size = depth_to_size(self._current_depth)
            print("Discriminator depth is set to {}, image size {}x{}".
                  format(self._current_depth, im_size, im_size))
        else:
            print("Depth value invalid..., should be within {} to {}".format(1, self._max_depth))

    def growing_net(self, alpha_start=0, delta_alpha=1e-3):
        if self._current_depth < self._max_depth:
            self._current_depth += 1
            self._internal_index = self._max_depth - self._current_depth
            self.delta_alpha = delta_alpha
            self.alpha = alpha_start
            im_size = depth_to_size(self._current_depth)
            print("Discriminator depth is set to {}, image size {}x{}".
                  format(self._current_depth, im_size, im_size))
        else:
            print("Net cannot be grow, depth should be within {} to {}".format(1, self._max_depth))

    def scale_grad(self, factor=0.5):
        for block in self.current_net[self._internal_index + 1:]:
            for param in block.parameters():
                if param.grad is not None:
                    param.grad *= factor

    def get_alpha(self):
        return self.alpha

    def increase_alpha(self, delta=None):
        if delta is None:
            delta = self.delta_alpha

        self.alpha = min(self.alpha + delta, 1)
