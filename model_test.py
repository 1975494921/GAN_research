from PGAN_models import Discriminator, Generator, Noise_Net
from config import Config
from PGAN_utils import size_to_depth, depth_to_size
import torch
from torch import nn, optim


def test_generator():
    gen = Generator(256, 512, 1024).to(Config.devices[0])
    gen = nn.DataParallel(gen, device_ids=Config.device_groups[0])
    noise = torch.randn(64, 512, 1, 1).to(Config.devices[0])
    target = torch.randn(64, 3, 1024, 1024).to(Config.devices[0])
    optimizer = optim.Adam(gen.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()
    gen.module.set_depth(9)

    for i in range(1, 1000):
        out = gen(noise)
        print(out.shape)
        optimizer.zero_grad()
        loss = loss_func(out, target)
        gen.module.scale_grad(0.2)
        optimizer.step()


def test_discriminator():
    dis = Discriminator(512, 1024).to(Config.devices[0])
    dis = nn.DataParallel(dis, device_ids=Config.device_groups[0])
    target = torch.randn(64, 1).to(Config.devices[0])
    optimizer = optim.Adam(dis.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()
    for i in range(1, size_to_depth(1024) + 1):
        dis.module.set_depth(i)
        gen_image = torch.randn(64, 3, depth_to_size(i), depth_to_size(i)).to(Config.devices[0])
        for _ in range(100):
            out = dis(gen_image)
            optimizer.zero_grad()
            loss = loss_func(out, target)
            loss.backward()
            dis.module.scale_grad(0.2)
            optimizer.step()
            print(out.shape)


def test_all():
    gen = Generator(256, 512, 1024).to(Config.devices[0])
    gen = nn.DataParallel(gen, device_ids=Config.device_groups[0])
    dis = Discriminator(512, 1024).to(Config.devices[1])
    dis = nn.DataParallel(dis, device_ids=Config.device_groups[1])

    G_optim = optim.Adam(gen.parameters(), lr=0.0003)
    D_optim = optim.SGD(gen.parameters(), lr=0.0002)
    loss_func = nn.MSELoss()
    real_label = torch.ones((56, 1)).to(Config.devices[1])

    for i in range(1, 10):
        gen.module.set_depth(i)
        dis.module.set_depth(i)
        for j in range(1):
            noise = torch.randn(56, 256).to(Config.devices[0])
            image = gen(noise)
            image = image.to(Config.devices[1])
            score = dis(image)
            G_optim.zero_grad()
            D_optim.zero_grad()
            loss = loss_func(score, real_label)
            loss.backward()
            D_optim.step()
            G_optim.step()
            print(loss)

test_all()
