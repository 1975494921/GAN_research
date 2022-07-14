import torch
from torch import nn
from config import Config
import math
from torch import optim


def weights_init(w):
    classname = w.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(w.weight.data)
        nn.init.constant_(w.bias.data, 0)


class Noise_Net(nn.Module):
    def __init__(self, requires_grad=True):
        super().__init__()
        self.requires_grad = requires_grad
        hidden_dim = int(Config.nz_num / 4)
        self.FN = nn.Sequential(
            nn.Linear(Config.nz_num, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, Config.nz_num),
        )
        self.m = nn.BatchNorm2d(Config.nz_dim)
        self.set_require_grad(self.requires_grad)

    def set_require_grad(self, grad):
        self.requires_grad = grad
        for param in self.FN.parameters():
            param.requires_grad = self.requires_grad

        for param in self.m.parameters():
            param.requires_grad = self.requires_grad

    def weight_init(self):
        self.FN.apply(weights_init)
        self.m.apply(weights_init)

    def get_optim(self, lr=0.0001):
        assert self.requires_grad is True, "Noise_Net do not requires grad"
        return optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.FN(x)
        x = torch.reshape(x, (Config.batch_size, Config.nz_dim, Config.nz_side, Config.nz_side))
        out = x
        return out


class Base_Net(nn.Module):
    def __init__(self, in_size, out_size, requires_grad=True):
        super().__init__()
        self.requires_grad = requires_grad
        self.layers = nn.ModuleList()
        self._make_layers(in_size, out_size)
        self.set_require_grad(self.requires_grad)

    def _make_layers(self, in_size, out_size):
        N = int(math.log2(out_size / in_size))
        inter_channel = Config.ngf * 16
        for i in range(N):
            if i == 0:
                base_layer = nn.Sequential(
                    nn.ConvTranspose2d(Config.nz_dim, inter_channel, 4, 2, 1),
                    nn.BatchNorm2d(inter_channel),
                    nn.ReLU(True)
                ).cuda()
                self.layers.append(base_layer)

            else:
                new_layer = nn.Sequential(
                    nn.ConvTranspose2d(inter_channel, int(inter_channel / 2), 4, 2, 1),
                    nn.BatchNorm2d(int(inter_channel / 2)),
                    nn.ReLU(True)
                ).cuda()
                self.layers.append(new_layer)
                inter_channel = int(inter_channel / 2)

    def set_require_grad(self, grad):
        self.requires_grad = grad
        for param in self.layers.parameters():
            param.requires_grad = self.requires_grad

    def get_optim(self, lr=0.0001):
        assert self.requires_grad is True, "Base_Net do not requires grad"
        return optim.Adam(self.parameters(), lr=lr)

    def weight_init(self):
        for layer in self.layers:
            layer.apply(weights_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


def main():
    noise_net = Noise_Net().cuda()
    noise_net_optim = noise_net.get_optim(0.001)
    noise = torch.randn((Config.batch_size, Config.nz_num)).cuda()
    base_net = Base_Net(Config.nz_side, 32).cuda()
    base_net_optim = base_net.get_optim(0.001)
    loss_func = nn.MSELoss()
    target = torch.randn((Config.batch_size, 64, 32, 32)).cuda()
    i = 0
    noise_net.weight_init()
    base_net.weight_init()
    while True:
        i += 1
        out = noise_net(noise)
        out = base_net(out)
        noise_net_optim.zero_grad()
        base_net_optim.zero_grad()
        loss = loss_func(out, target)
        loss.backward()
        noise_net_optim.step()
        base_net_optim.step()
        if i % 10 == 0:
            print(loss)


if __name__ == '__main__':
    main()
