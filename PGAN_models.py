import torch
from torch import nn, optim
import math
from PGAN_utils import EqualizedLR_Conv2d, PixelNorm, Minibatch_std
from PGAN_utils import depth_to_size, size_to_depth
from config import Config
import torch.nn.functional as F


def print_func(*args):
    if Config.debug:
        print(*args)


def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(w.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(w.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(w.bias.data, 0)


class FromRGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class ToRGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.conv(x)


class Noise_Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = int(in_dim / 2)
        self.FN = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
            nn.LeakyReLU(0.2),
        )
        self.FN.apply(weights_init)

    def forward(self, x):
        x = self.FN(x)
        out = x.unsqueeze(-1).unsqueeze(-1)
        return out


class ECA_Block(nn.Module):
    # Channel attention
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.normal_(self.conv.weight)

    def forward(self, x):
        # feature descriptor on the global spatial information
        attn = self.avg_pool(x)

        # Two different branches of ECA module
        attn = self.conv(attn.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        attn = torch.sigmoid(attn)

        return x * attn.expand_as(x)


class SA_Block(nn.Module):
    # Spatial Attention
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        attn = self.conv(result)
        attn = self.sigmoid(attn)

        return x * attn.expand_as(x)


class Residual_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual = None
        self.ECA = None
        self.SA = None
        if in_ch != out_ch:
            self.residual = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.ECA = ECA_Block()
            self.SA = SA_Block()

    def forward(self, x):
        if self.residual is not None:
            x = self.residual(x)
            x = self.ECA(x)
            x = self.SA(x)

        return x


class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch, initial_block=False):
        super().__init__()
        self.upsample = None
        self.residual = None
        self.conv1_ECA = ECA_Block()
        self.conv1_SA = SA_Block()
        self.conv2_ECA = ECA_Block()
        self.conv2_SA = SA_Block()

        if initial_block:
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.residual = Residual_Block(in_ch, out_ch)
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        # x = self.conv1(x*scale1)
        residual_out = self.residual(x) if self.residual is not None else None

        x = self.conv1_ECA(self.conv1(x))
        x = self.conv1_SA(x)
        x = self.relu(x)
        x = self.pixel_norm(x)
        # x = self.conv2(x*scale2)
        x = self.conv2_ECA(self.conv2(x))
        x = self.conv2_SA(x)
        x = self.relu(x)
        if residual_out is not None:
            x = x + residual_out

        x = self.pixel_norm(x)
        return x


class Generator(nn.Module):
    def __init__(self, noise_dim, latent_size, out_res):
        super().__init__()
        self._current_depth = 1  # starting from 1, 4x4
        self.alpha = 0.1
        self.delta_alpha = 1e-3
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.noise_net = Noise_Net(noise_dim, latent_size)
        self.current_net = nn.ModuleList([G_Block(latent_size, latent_size, initial_block=True)])
        self.toRGBs = nn.ModuleList([ToRGB(latent_size, 3)])
        self.out_res = out_res
        self.max_depth = size_to_depth(out_res)
        # __add_layers(out_res)
        for d in range(2, int(math.log2(out_res))):
            if d < 6:
                # low res blocks 8x8, 16x16, 32x32 with 512 channels
                in_ch, out_ch = 512, 512
            else:
                # from 64x64 (5th block), the number of channels halved for each block
                in_ch, out_ch = int(512 / 2 ** (d - 6)), int(512 / 2 ** (d - 5))

            self.current_net.append(G_Block(in_ch, out_ch))
            self.toRGBs.append(ToRGB(out_ch, 3))

    def forward(self, noise):
        x = self.noise_net(noise)
        # x = noise
        for block in self.current_net[:int(self._current_depth) - 1]:
            x = block(x)

        model_out = self.current_net[self._current_depth - 1](x)
        x_rgb = self.toRGBs[self._current_depth - 1](model_out)

        if self.alpha < 1 and self._current_depth > 1:
            x_old = self.up_sample(x)
            old_rgb = self.toRGBs[self._current_depth - 2](x_old)
            x_rgb = (1 - self.alpha) * old_rgb + self.alpha * x_rgb

            # self.alpha += self.delta_alpha
            # self.alpha = min(self.alpha, 1)

        return x_rgb

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

    def scale_grad_noise(self, factor=0.5):
        for param in self.noise_net.parameters():
            if param.grad is not None:
                param.grad *= factor


class D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, final_block=False):
        super().__init__()
        self.conv1_ECA = ECA_Block()
        self.conv1_SA = SA_Block()
        self.conv2_ECA = ECA_Block()
        self.conv2_SA = SA_Block()
        self.residual = None

        if final_block:
            self.minibatchstd = Minibatch_std()
            self.conv1 = EqualizedLR_Conv2d(in_ch + 1, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(4, 4), stride=(1, 1))
            self.outlayer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_ch, 1),
                # nn.Sigmoid()
            )
        else:
            self.minibatchstd = None
            self.residual = Residual_Block(in_ch, out_ch)
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.relu = nn.LeakyReLU(0.2)
        # nn.init.normal_(self.conv1.weight)
        # nn.init.normal_(self.conv2.weight)
        # nn.init.zeros_(self.conv1.bias)
        # nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.minibatchstd is not None:
            x = self.minibatchstd(x)

        residual_out = self.residual(x) if self.residual is not None else None
        x = self.conv1_ECA(self.conv1(x))
        x = self.conv1_SA(x)
        x = self.relu(x)
        x = self.conv2_ECA(self.conv2(x))
        x = self.conv2_SA(x)
        if residual_out is not None:
            x = x + residual_out

        x = self.relu(x)
        x = self.outlayer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_size, img_size):
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

        print_func("max_depth: {}".format(self._max_depth))
        dim_list = []
        for d in range(1, size_to_depth(img_size)):
            if d < 4:
                in_ch, out_ch = 512, 512
            else:
                in_ch, out_ch = int(512 / 2 ** (d - 3)), int(512 / 2 ** (d - 4))

            dim_list.append((in_ch, out_ch))

        dim_list.reverse()
        for in_ch, out_ch in dim_list:
            print_func(in_ch, out_ch)
            self.current_net.append(D_Block(in_ch, out_ch))
            self.fromRGBs.append(FromRGB(3, in_ch))

        self.current_net.append(D_Block(latent_size, latent_size, final_block=True))
        self.fromRGBs.append(FromRGB(3, latent_size))

    def forward(self, x_rgb):
        print_func("start forward......")
        print_func("input shape: {}".format(x_rgb.shape))
        RGBBlock_index = self._internal_index

        print_func("fromRGBs block index: {}".format(RGBBlock_index))
        x = self.fromRGBs[RGBBlock_index](x_rgb)
        print_func("fromRGB shape: {}".format(x.shape))

        x = self.current_net[self._internal_index](x)
        print_func("last_net shape: {}".format(x.shape))

        if self.alpha < 1 and self._internal_index < self._max_depth - 1:
            x_rgb = self.down_sample(x_rgb)
            x_old = self.fromRGBs[self._internal_index + 1](x_rgb)
            x = (1 - self.alpha) * x_old + self.alpha * x
            # self.alpha += self.delta_alpha
            # self.alpha = min(self.alpha, 1)

        for block in self.current_net[self._internal_index + 1:]:
            print_func(x.shape)
            print_func(block)
            x = block(x)

        return torch.sigmoid(0.5 * x)

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
