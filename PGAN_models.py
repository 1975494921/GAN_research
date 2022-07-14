import torch
import torch.nn as nn
import math
import torch.optim as optim
from PGAN_utils import EqualizedLR_Conv2d, PixelNorm, Minibatch_std
from PGAN_utils import depth_to_size, size_to_depth
from config import Config


def print_func(*args):
    if Config.debug:
        print(*args)


def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(w.weight.data)
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
    def __init__(self):
        super().__init__()
        self.FN = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 512),
            nn.Dropout(0.2)
        )
        self.FN.apply(weights_init)

    def forward(self, x):
        x = self.FN(x)
        out = x.unsqueeze(-1).unsqueeze(-1)
        return out


class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch, initial_block=False):
        super().__init__()
        if initial_block:
            self.upsample = None
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        # x = self.conv1(x*scale1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pixel_norm(x)
        # x = self.conv2(x*scale2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pixel_norm(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_size, out_res):
        super().__init__()
        self._current_depth = 1  # starting from 1, 4x4
        self.alpha = 1
        self.delta_alpha = 0
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
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

    def forward(self, x):
        for block in self.current_net[:int(self._current_depth) - 1]:
            x = block(x)

        model_out = self.current_net[self._current_depth - 1](x)

        x_rgb = self.toRGBs[self._current_depth - 1](model_out)
        if self.alpha < 1 and self._current_depth != 1:
            x_old = self.up_sample(x)
            old_rgb = self.toRGBs[self._current_depth - 2](x_old)
            x_rgb = (1 - self.alpha) * old_rgb + self.alpha * x_rgb

            self.alpha += self.delta_alpha
            self.alpha = min(self.alpha, 1)

        return x_rgb

    def growing_net(self, alpha_start=0, delta_alpha=1e-3):
        if self._current_depth < self.max_depth:
            self.delta_alpha = delta_alpha
            self.alpha = alpha_start
            self._current_depth += 1
            im_size = depth_to_size(self._current_depth)
            print("Generator depth changed to {}, image size {}x{}".
                  format(self._current_depth, im_size, im_size))
        else:
            print("Net cannot be grow, depth should be within {} to {}".format(1, self.max_depth))

    def set_depth(self, depth: int, alpha_start=0, delta_alpha=1e-3):
        if 1 <= depth <= self.max_depth:
            self._current_depth = int(depth)
            self.delta_alpha = delta_alpha
            self.alpha = alpha_start
            im_size = depth_to_size(self._current_depth)
            print("Generator depth changed to {}, image size {}x{}".
                  format(self._current_depth, im_size, im_size))
        else:
            print("Depth value invalid..., should be within {} to {}".format(1, self.max_depth))

    def scale_grad(self, factor=0.5):
        for block in self.current_net[:self._current_depth - 1]:
            for param in block.parameters():
                if param.grad is not None:
                    param.grad *= factor


class D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, final_block=False):
        super().__init__()
        self.fn = nn.Linear(in_ch, out_ch)
        if final_block:
            self.minibatchstd = Minibatch_std()
            self.conv1 = EqualizedLR_Conv2d(in_ch + 1, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(4, 4), stride=(1, 1))
            self.outlayer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_ch, 1)
            )
        else:
            self.minibatchstd = None
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

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.outlayer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_size, img_size):
        super().__init__()
        self._current_depth = 1
        self.alpha = 1
        self.delta_alpha = 0
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

        if self.alpha < 1 and self._internal_index != self._max_depth-1:
            x_rgb = self.down_sample(x_rgb)
            x_old = self.fromRGBs[self._internal_index + 1](x_rgb)
            x = (1 - self.alpha) * x_old + self.alpha * x
            self.alpha += self.delta_alpha
            self.alpha = min(self.alpha, 1)

        for block in self.current_net[self._internal_index + 1:]:
            print_func(x.shape)
            print_func(block)
            x = block(x)
        return x

    def set_depth(self, depth: int, alpha_start=0, delta_alpha=1e-3):
        if 1 <= depth <= self._max_depth:
            self._current_depth = int(depth)
            self._internal_index = self._max_depth - self._current_depth
            self.delta_alpha = delta_alpha
            self.alpha = alpha_start
            im_size = depth_to_size(self._current_depth)
            print("Discriminator depth changed to {}, image size {}x{}".
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
            print("Discriminator depth changed to {}, image size {}x{}".
                  format(self._current_depth, im_size, im_size))
        else:
            print("Net cannot be grow, depth should be within {} to {}".format(1, self._max_depth))

    def scale_grad(self, factor=0.5):
        for block in self.current_net[self._internal_index + 1:]:
            for param in block.parameters():
                if param.grad is not None:
                    param.grad *= factor


# dis = Discriminator(512, 1024).to(Config.devices[0])
# for i in range(1, size_to_depth(1024)+1):
#     dis.set_depth(i)
#     gen_image = torch.randn(1, 3, depth_to_size(i), depth_to_size(i)).to(Config.devices[0])
#     out = dis(gen_image)
#     print(out.shape)


gen = Generator(512, 1024).to(Config.devices[0])
gen = nn.DataParallel(gen, device_ids=Config.device_groups[0])
dis = Discriminator(512, 1024).to(Config.devices[1])
dis = nn.DataParallel(dis, device_ids=Config.device_groups[1])
a = Noise_Net().to(Config.devices[0])

G_optim = optim.Adam(gen.parameters(), lr=0.0003)
D_optim = optim.SGD(gen.parameters(), lr=0.0002)
loss_func = nn.MSELoss()
real_label = torch.ones((56, 1)).to(Config.devices[1])

for i in range(9, 10):
    gen.module.set_depth(i)
    dis.module.set_depth(i)
    for _ in range(5000):
        noise = torch.randn(56, 256).to(Config.devices[0])
        noise_out = a(noise)
        image = gen(noise_out)
        image = image.to(Config.devices[1])
        score = dis(image)
        G_optim.zero_grad()
        D_optim.zero_grad()
        loss = loss_func(score, real_label)
        loss.backward()
        D_optim.step()
        G_optim.step()
        print(loss)