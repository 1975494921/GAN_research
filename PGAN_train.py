from PGAN_models import Discriminator, Generator, Noise_Net
from config import Config
from PGAN_utils import size_to_depth, depth_to_size
import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision import datasets

model_root = 'model_trains'
if not os.path.isdir(model_root):
    os.mkdir(model_root)

valid = True
train_key = "train001"
while not valid:
    train_key = input("Please input the train key folder: ")
    if train_key == "":
        print("Input cannot be empty")
    else:
        valid = True

model_dir = os.path.join(model_root, train_key)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

G_net = Generator(256, 512, 1024).to(Config.devices[0])
G_net = nn.DataParallel(G_net, device_ids=Config.device_groups[0])
D_net = Discriminator(512, 1024).to(Config.devices[1])
D_net = nn.DataParallel(D_net, device_ids=Config.device_groups[1])

pretrained_file = os.path.join(model_dir, "model_{}.pth".format(2))
model_state_dict = torch.load(pretrained_file, map_location=torch.device(Config.devices[0]))
G_net.load_state_dict(model_state_dict['G_net'])
D_net.load_state_dict(model_state_dict['D_net'])
print("Loaded model file: {}".format(pretrained_file))

data_dir = "/home/zceelil/dataset"
epos_list = [0, 200, 500, 1000, 1000, 1000, 1000, 1000, 1000, 10]
batch_list = [0, 4000, 1000, 1000, 1000, 1000, 500, 200, 80, 40]
save_internal = [0, 10, 10, 10, 5, 5, 5, 3, 1, 1]
start_depth = model_state_dict['current_depth']
end_depth = size_to_depth(256)


G_optim = optim.Adam(G_net.parameters(), lr=0.0003)
D_optim = optim.Adam(D_net.parameters(), lr=0.0002)
loss_func = nn.BCELoss()

torch.set_default_tensor_type(torch.FloatTensor)

del model_state_dict
print("start training...")
for depth in range(start_depth, end_depth + 1):
    batch_size = batch_list[depth]
    img_size = depth_to_size(depth)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    G_net.module.set_depth(depth, alpha_start=0.5)
    D_net.module.set_depth(depth, alpha_start=0.5)

    for epo in range(epos_list[depth]):
        batch_size = batch_list[depth]
        for batch_ndx, sample in enumerate(data_loader):
            sample = sample[0].to(Config.devices[1])
            real_label = torch.full((sample.shape[0], 1,), 1, dtype=torch.float).to(Config.devices[1])
            fake_label = torch.full((sample.shape[0], 1,), 0, dtype=torch.float).to(Config.devices[1])

            # update D
            D_optim.zero_grad()
            noise = torch.randn(sample.shape[0], 256).to(Config.devices[0])
            fake = G_net(noise).to(Config.devices[1])
            fake_out = D_net(fake.detach())
            real_out = D_net(sample)

            # gradient_penalty = 0
            # if False:
            #     ## Gradient Penalty
            #     eps = torch.rand(sample.shape[0], 1, 1, 1).to(Config.devices[1])
            #     eps = eps.expand_as(sample)
            #     x_hat = eps * sample + (1 - eps) * fake.detach()
            #     x_hat.requires_grad = True
            #     px_hat = D_net(x_hat)
            #     grad = torch.autograd.grad(
            #         outputs=px_hat.sum(),
            #         inputs=x_hat,
            #         create_graph=True
            #     )[0]
            #     grad_norm = grad.view(sample.shape[0], -1).norm(2, dim=1)
            #     gradient_penalty = 2 * ((grad_norm - 1) ** 2).mean()

            ###########

            D_loss_real = loss_func(real_out, real_label)
            D_loss_fake = loss_func(fake_out, fake_label)

            D_loss = D_loss_real.mean() + D_loss_fake.mean()

            D_loss.backward()
            D_net.module.scale_grad(0.6)
            D_optim.step()

            ## update G
            G_optim.zero_grad()
            fake_out = D_net(fake)

            G_loss = loss_func(fake_out, real_label)
            G_loss.backward()
            G_net.module.scale_grad(0.6)
            if depth > 1:
                G_net.module.scale_grad_noise(0.5)

            G_optim.step()

            ##############
            print("\r G_loss: {}; D_loss: {}".format(G_loss.mean().item(), D_loss.mean().item()), end="")

        if (epo+1) % save_internal[depth] == 0:
            save_dict = {'G_net': G_net.state_dict(), 'D_net': D_net.state_dict(), 'current_depth': depth,
                         'noise_size': 256, 'img_size': depth_to_size(depth), 'latent_size': 512}

            save_path = os.path.join(model_dir, "model_{}.pth".format(depth))
            torch.save(save_dict, save_path)
            print("\n Model saves at Depth:{} Epoch:{}".format(depth, epo))

    save_dict = {'G_net': G_net.state_dict(), 'D_net': D_net.state_dict(), 'current_depth': depth,
                 'noise_size': 256, 'img_size': depth_to_size(depth), 'latent_size': 512}
    save_path = os.path.join(model_dir, "model_{}.pth".format(depth))
    torch.save(save_dict, save_path)
    print("\n Model saves at Depth:{}".format(depth))